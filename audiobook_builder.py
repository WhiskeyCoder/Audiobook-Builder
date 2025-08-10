import os
import shutil
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import signal
import sys
import asyncio

from datetime import datetime
import PyPDF2
import ebooklib
from ebooklib import epub
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Optional imports with fallbacks
try:
    from docx import Document

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not available. DOCX files will not be supported.")

try:
    import docx2txt

    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False
    print("Warning: docx2txt not available. DOC files will not be supported.")

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("Warning: edge-tts not available. Install with: pip install edge-tts")

try:
    import torch
    import torchaudio
    import librosa
    import numpy as np
    import scipy.signal
    from scipy.io import wavfile

    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False
    print("Warning: RVC dependencies not available. Install with: pip install torch torchaudio librosa numpy scipy")


@dataclass
class AudiobookConfig:
    """Configuration class for audiobook conversion"""
    chunk_size_words: int = 1200
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay: float = 2.0
    audio_format: str = "mp3"
    audio_bitrate: str = "128k"
    cache_enabled: bool = True
    cleanup_chunks: bool = True
    progress_interval: int = 10
    sample_rate: int = 22050
    chunk_duration: float = 10.0  # seconds


@dataclass
class VoiceProfile:
    """Voice profile configuration"""
    name: str
    tts_voice: str
    pitch_shift: float = 0.0
    speed_ratio: float = 1.0
    volume_gain: float = 1.0
    # RVC model paths (optional - if you have trained models)
    rvc_model_path: Optional[str] = None
    rvc_index_path: Optional[str] = None


class SimpleRVCProcessor:
    """Simple RVC-style voice conversion using basic audio processing"""

    def __init__(self, model_path: Optional[str] = None, index_path: Optional[str] = None):
        self.model_path = model_path
        self.index_path = index_path

    def convert_voice(self, audio_path: str, output_path: str,
                      pitch_shift: float = 0.0, speed_ratio: float = 1.0) -> bool:
        """Simple voice conversion using pitch shifting and formant adjustment"""
        try:
            if not RVC_AVAILABLE:
                # Fallback: just copy the file if no RVC dependencies
                shutil.copy2(audio_path, output_path)
                return True

            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)

            # Apply pitch shifting
            if pitch_shift != 0.0:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)

            # Apply speed change
            if speed_ratio != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed_ratio)

            # Apply some basic formant shifting for voice character change
            if pitch_shift != 0.0:
                # Simple formant adjustment using spectral envelope manipulation
                stft = librosa.stft(audio)
                magnitude = np.abs(stft)
                phase = np.angle(stft)

                # Shift formants by adjusting frequency bins
                formant_shift = pitch_shift * 0.1  # Subtle formant adjustment
                if formant_shift != 0.0:
                    freq_bins = magnitude.shape[0]
                    shift_bins = int(formant_shift * freq_bins / 12)  # Convert semitones to bins

                    if shift_bins > 0:
                        magnitude = np.roll(magnitude, shift_bins, axis=0)
                        magnitude[:shift_bins] = 0
                    elif shift_bins < 0:
                        magnitude = np.roll(magnitude, shift_bins, axis=0)
                        magnitude[shift_bins:] = 0

                # Reconstruct audio
                stft_modified = magnitude * np.exp(1j * phase)
                audio = librosa.istft(stft_modified)

            # Normalize audio
            audio = audio / np.max(np.abs(audio)) * 0.95

            # Save output
            wavfile.write(output_path, sr, (audio * 32767).astype(np.int16))
            return True

        except Exception as e:
            logging.error(f"Voice conversion failed: {e}")
            # Fallback: copy original file
            shutil.copy2(audio_path, output_path)
            return True


class StandaloneAudiobookConverter:
    """Standalone audiobook converter with built-in TTS+RVC pipeline"""

    def __init__(self, config: AudiobookConfig):
        self.config = config
        self.setup_logging()
        self.rvc_processor = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
        """Setup comprehensive logging with UTF-8 encoding"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"audiobook_{datetime.now().strftime('%Y%m%d')}.log",
                    encoding='utf-8'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        sys.exit(0)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats based on available dependencies"""
        formats = ['.pdf', '.epub', '.txt']

        if DOCX_AVAILABLE:
            formats.append('.docx')

        if DOC_AVAILABLE:
            formats.append('.doc')

        return formats

    async def generate_tts_audio(self, text: str, voice: str, output_path: Path) -> bool:
        """Generate TTS audio using Edge-TTS"""
        try:
            if not EDGE_TTS_AVAILABLE:
                raise ImportError("edge-tts not available")

            # Edge-TTS outputs MP3 by default, so use .mp3 extension for temp file
            temp_mp3_path = output_path.parent / f"{output_path.stem}_temp.mp3"

            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(temp_mp3_path))

            # Convert MP3 to WAV for consistent processing
            try:
                audio = AudioSegment.from_mp3(str(temp_mp3_path))
                audio.export(str(output_path), format="wav")

                # Clean up temp MP3
                if temp_mp3_path.exists():
                    temp_mp3_path.unlink()

                return True
            except Exception as e:
                self.logger.error(f"Failed to convert MP3 to WAV: {e}")
                # If conversion fails, just rename the MP3 to target path
                if temp_mp3_path.exists():
                    shutil.move(str(temp_mp3_path), str(output_path.with_suffix('.mp3')))
                return True

        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            return False

    def convert_text_to_audio(self, text: str, output_path: Path, voice_profile: VoiceProfile) -> bool:
        """Convert text to audio using TTS + optional RVC processing"""

        # Check cache first
        cache_path = self.get_cache_path(text, voice_profile)
        if cache_path and cache_path.exists():
            try:
                shutil.copy2(cache_path, output_path)
                return True
            except Exception as e:
                self.logger.warning(f"Failed to use cached file: {e}")

        try:
            # Step 1: Generate TTS audio
            temp_tts_path = output_path.parent / f"temp_tts_{output_path.stem}.wav"

            # Run TTS generation
            success = asyncio.run(self.generate_tts_audio(text, voice_profile.tts_voice, temp_tts_path))

            if not success:
                self.logger.error("TTS generation failed")
                return False

            # Check if TTS actually created a file
            if not temp_tts_path.exists():
                # Maybe it created an MP3 instead?
                temp_mp3_path = temp_tts_path.with_suffix('.mp3')
                if temp_mp3_path.exists():
                    temp_tts_path = temp_mp3_path
                else:
                    self.logger.error(f"TTS output file not found: {temp_tts_path}")
                    return False

            # Step 2: Apply voice conversion if RVC model is available
            if voice_profile.rvc_model_path and Path(voice_profile.rvc_model_path).exists():
                # Initialize RVC processor if not already done
                if self.rvc_processor is None:
                    self.rvc_processor = SimpleRVCProcessor(
                        voice_profile.rvc_model_path,
                        voice_profile.rvc_index_path
                    )

                # Apply RVC conversion
                success = self.rvc_processor.convert_voice(
                    str(temp_tts_path),
                    str(output_path),
                    pitch_shift=voice_profile.pitch_shift,
                    speed_ratio=voice_profile.speed_ratio
                )
            else:
                # No RVC model - apply basic audio processing
                success = self.apply_basic_audio_processing(
                    temp_tts_path,
                    output_path,
                    voice_profile
                )

            # Cache the result if successful
            if success and output_path.exists() and cache_path:
                try:
                    shutil.copy2(output_path, cache_path)
                except Exception as e:
                    self.logger.warning(f"Failed to cache result: {e}")

            # Cleanup temp file
            if temp_tts_path.exists():
                try:
                    temp_tts_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file: {e}")

            return success

        except Exception as e:
            self.logger.error(f"Text to audio conversion failed: {e}")
            return False

    def apply_basic_audio_processing(self, input_path: Path, output_path: Path,
                                     voice_profile: VoiceProfile) -> bool:
        """Apply basic audio processing without RVC models"""
        try:
            # First, ensure we have a proper audio file to work with
            if not input_path.exists():
                self.logger.error(f"Input file does not exist: {input_path}")
                return False

            # Try to load audio - check if it's MP3 or WAV
            audio = None
            try:
                if input_path.suffix.lower() == '.mp3':
                    audio = AudioSegment.from_mp3(str(input_path))
                elif input_path.suffix.lower() == '.wav':
                    audio = AudioSegment.from_wav(str(input_path))
                else:
                    # Try auto-detection
                    audio = AudioSegment.from_file(str(input_path))
            except Exception as e:
                self.logger.error(f"Failed to load audio file {input_path}: {e}")
                # Fallback: just copy the file
                shutil.copy2(input_path, output_path)
                return True

            # Apply pitch shift using speed change (approximate)
            if voice_profile.pitch_shift != 0.0:
                # Convert semitones to speed ratio (rough approximation)
                speed_change = 2 ** (voice_profile.pitch_shift / 12.0)
                new_sample_rate = int(audio.frame_rate * speed_change)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                audio = audio.set_frame_rate(22050)  # Normalize back to standard rate

            # Apply speed ratio
            if voice_profile.speed_ratio != 1.0:
                new_frame_rate = int(audio.frame_rate * voice_profile.speed_ratio)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
                audio = audio.set_frame_rate(22050)  # Normalize back to standard rate

            # Apply volume gain
            if voice_profile.volume_gain != 1.0:
                gain_db = 20 * np.log10(max(voice_profile.volume_gain, 0.001))  # Prevent log(0)
                audio = audio + gain_db

            # Export processed audio as WAV
            audio.export(str(output_path), format="wav")
            return True

        except Exception as e:
            self.logger.error(f"Basic audio processing failed: {e}")
            # Fallback: just copy the file
            try:
                shutil.copy2(input_path, output_path)
                return True
            except Exception as copy_error:
                self.logger.error(f"Even file copy failed: {copy_error}")
                return False

    # Include all the text extraction methods from the previous script
    def extract_text_from_file(self, file_path: Path) -> str:
        """Enhanced text extraction with better error handling and dependency checking"""
        file_extension = file_path.suffix.lower()

        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.epub':
                return self._extract_from_epub(file_path)
            elif file_extension == '.docx':
                if DOCX_AVAILABLE:
                    return self._extract_from_docx(file_path)
                else:
                    raise ValueError("DOCX support not available. Install python-docx: pip install python-docx")
            elif file_extension == '.doc':
                if DOC_AVAILABLE:
                    return self._extract_from_doc(file_path)
                else:
                    raise ValueError("DOC support not available. Install docx2txt: pip install docx2txt")
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                supported = ", ".join(self.get_supported_formats())
                raise ValueError(f"Unsupported file format: {file_extension}. Supported: {supported}")
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            raise

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF with improved handling"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n\n{page_text}"
                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            raise
        return self._clean_text(text)

    def _extract_from_epub(self, file_path: Path) -> str:
        """Extract text from EPUB with better parsing"""
        try:
            book = epub.read_epub(str(file_path))
            text_parts = []

            for item in book.get_items():
                if isinstance(item, ebooklib.ITEM_DOCUMENT):
                    content = item.get_body_content().decode('utf-8', errors='ignore')
                    import re
                    content = re.sub(r'<[^>]+>', '', content)
                    if content.strip():
                        text_parts.append(content)

            return self._clean_text('\n\n'.join(text_parts))
        except Exception as e:
            self.logger.error(f"EPUB extraction failed: {e}")
            raise

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(file_path)
            text = '\n\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            return self._clean_text(text)
        except Exception as e:
            self.logger.error(f"DOCX extraction failed: {e}")
            raise

    def _extract_from_doc(self, file_path: Path) -> str:
        """Extract text from DOC using docx2txt"""
        try:
            text = docx2txt.process(str(file_path))
            if not text:
                self.logger.warning(f"docx2txt returned empty result for {file_path}")
                return ""
            return self._clean_text(text)
        except Exception as e:
            self.logger.error(f"DOC extraction failed: {e}")
            raise

    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT with encoding detection"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return self._clean_text(file.read())
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode text file with any of: {encodings}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for TTS"""
        import re

        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common issues
        text = text.replace('\n', ' ')
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\b\d{1,3}\b(?=\s|$)', '', text)

        return text.strip()

    def split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Intelligent text chunking with sentence boundary awareness"""
        if not text.strip():
            return []

        import re

        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if sentence_words > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_word_count = 0

                # Split long sentence by commas or other punctuation
                sub_parts = re.split(r'[,;:]', sentence)
                for part in sub_parts:
                    part_words = len(part.split())
                    if current_word_count + part_words <= chunk_size:
                        current_chunk += part + " "
                        current_word_count += part_words
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                        current_word_count = part_words
            else:
                if current_word_count + sentence_words <= chunk_size:
                    current_chunk += sentence + " "
                    current_word_count += sentence_words
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_word_count = sentence_words

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def get_cache_path(self, text: str, voice_profile: VoiceProfile) -> Path:
        """Generate cache path for text chunk"""
        if not self.config.cache_enabled:
            return None

        cache_dir = Path("cache") / "audio_chunks"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create hash from text and voice settings
        content = f"{text}_{voice_profile.name}_{voice_profile.pitch_shift}_{voice_profile.speed_ratio}"
        hash_obj = hashlib.md5(content.encode())
        return cache_dir / f"{hash_obj.hexdigest()}.wav"

    def process_chunk_batch(self, chunk_batch: List[Tuple[int, str]],
                            chunks_dir: Path, voice_profile: VoiceProfile,
                            total_chunks: int) -> List[bool]:
        """Process a batch of chunks in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_chunk = {}

            for chunk_num, text in chunk_batch:
                output_path = chunks_dir / f"chunk_{chunk_num:04d}.wav"
                future = executor.submit(self.convert_text_to_audio, text, output_path, voice_profile)
                future_to_chunk[future] = chunk_num

            for future in as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                try:
                    success = future.result()
                    results.append(success)
                    if success:
                        self.logger.info(f"+ Chunk {chunk_num}/{total_chunks} completed")
                    else:
                        self.logger.error(f"- Chunk {chunk_num}/{total_chunks} failed")
                except Exception as e:
                    self.logger.error(f"- Chunk {chunk_num}/{total_chunks} error: {e}")
                    results.append(False)

        return results

    def combine_audio_chunks(self, chunks_dir: Path, output_path: Path,
                             total_chunks: int) -> bool:
        """Combine audio chunks with improved error handling"""
        try:
            combined_audio = AudioSegment.empty()
            successful_chunks = 0

            for i in range(1, total_chunks + 1):
                chunk_file = chunks_dir / f"chunk_{i:04d}.wav"

                if not chunk_file.exists():
                    self.logger.warning(f"Missing chunk file: {chunk_file}")
                    continue

                try:
                    chunk_audio = AudioSegment.from_wav(str(chunk_file))
                    combined_audio += chunk_audio
                    successful_chunks += 1

                    if successful_chunks % self.config.progress_interval == 0:
                        self.logger.info(f"Combined {successful_chunks}/{total_chunks} chunks")

                except CouldntDecodeError as e:
                    self.logger.error(f"Failed to decode chunk {chunk_file}: {e}")
                    continue

            if successful_chunks == 0:
                raise RuntimeError("No valid audio chunks found")

            # Export with specified quality
            combined_audio.export(
                str(output_path),
                format=self.config.audio_format,
                bitrate=self.config.audio_bitrate
            )

            self.logger.info(f"Audiobook saved: {output_path} ({successful_chunks}/{total_chunks} chunks)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to combine audio chunks: {e}")
            return False

    def cleanup_chunks(self, chunks_dir: Path):
        """Clean up temporary chunk files"""
        if not self.config.cleanup_chunks:
            return

        try:
            for chunk_file in chunks_dir.glob("chunk_*.wav"):
                chunk_file.unlink()
            self.logger.info("Cleaned up temporary chunk files")
        except Exception as e:
            self.logger.error(f"Failed to cleanup chunks: {e}")

    def convert_document(self, input_path: Path, voice_profile: VoiceProfile) -> bool:
        """Convert a single document to audiobook"""
        self.logger.info(f"Starting conversion: {input_path.name}")
        start_time = time.time()

        try:
            # Check if file format is supported
            if input_path.suffix.lower() not in self.get_supported_formats():
                supported = ", ".join(self.get_supported_formats())
                self.logger.error(f"Unsupported format {input_path.suffix}. Supported: {supported}")
                return False

            # Setup directories
            chunks_dir = Path("chunks")
            audiobooks_dir = Path("audiobooks")
            chunks_dir.mkdir(exist_ok=True)
            audiobooks_dir.mkdir(exist_ok=True)

            # Extract and process text
            text = self.extract_text_from_file(input_path)
            if not text.strip():
                self.logger.error(f"No text extracted from {input_path}")
                return False

            # Split into chunks
            chunks = self.split_text_into_chunks(text, self.config.chunk_size_words)
            total_chunks = len(chunks)

            if total_chunks == 0:
                self.logger.error(f"No chunks created from {input_path}")
                return False

            self.logger.info(f"Processing {total_chunks} chunks (~{total_chunks} minutes estimated)")

            # Process chunks in batches
            batch_size = self.config.max_workers * 2
            all_success = True

            for i in range(0, total_chunks, batch_size):
                batch_chunks = [(i + j + 1, chunks[i + j]) for j in range(min(batch_size, total_chunks - i))]
                batch_results = self.process_chunk_batch(batch_chunks, chunks_dir, voice_profile, total_chunks)

                if not all(batch_results):
                    all_success = False
                    self.logger.warning(f"Some chunks in batch failed")

            if not all_success:
                self.logger.warning("Some audio chunks failed - audiobook may be incomplete")

            # Combine chunks
            output_filename = f"{input_path.stem}.{self.config.audio_format}"
            output_path = audiobooks_dir / output_filename

            success = self.combine_audio_chunks(chunks_dir, output_path, total_chunks)

            if success:
                duration = time.time() - start_time
                self.logger.info(f"+ Conversion completed in {duration:.1f}s: {output_path}")

            # Cleanup
            self.cleanup_chunks(chunks_dir)

            return success

        except Exception as e:
            self.logger.error(f"Conversion failed for {input_path}: {e}")
            return False

    def convert_batch(self, input_dir: Path, voice_profile: VoiceProfile) -> Dict[str, bool]:
        """Convert all documents in a directory"""
        results = {}

        supported_extensions = set(self.get_supported_formats())
        input_files = [f for f in input_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in supported_extensions]

        if not input_files:
            supported = ", ".join(supported_extensions)
            self.logger.warning(f"No supported files found in {input_dir}. Supported formats: {supported}")
            return results

        self.logger.info(f"Found {len(input_files)} files to convert")

        for input_file in input_files:
            try:
                success = self.convert_document(input_file, voice_profile)
                results[input_file.name] = success
            except KeyboardInterrupt:
                self.logger.info("Conversion interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error processing {input_file}: {e}")
                results[input_file.name] = False

        return results

    def print_system_info(self):
        """Print system information and available features"""
        print("=" * 60)
        print("STANDALONE AUDIOBOOK CONVERTER - SYSTEM INFO")
        print("=" * 60)
        print(f"Supported formats: {', '.join(self.get_supported_formats())}")
        print(f"DOCX support: {'+' if DOCX_AVAILABLE else '-'}")
        print(f"DOC support: {'+' if DOC_AVAILABLE else '-'}")
        print(f"Edge-TTS support: {'+' if EDGE_TTS_AVAILABLE else '-'}")
        print(f"RVC support: {'+' if RVC_AVAILABLE else '-'}")
        print(f"Cache enabled: {'+' if self.config.cache_enabled else '-'}")
        print(f"Max workers: {self.config.max_workers}")
        print(f"Chunk size: {self.config.chunk_size_words} words")
        print("=" * 60)


def main():
    """Main function with example usage"""

    # Configuration
    config = AudiobookConfig(
        chunk_size_words=1200,
        max_workers=4,
        audio_format="mp3",
        audio_bitrate="128k",
        cache_enabled=True
    )

    # Voice profile - using Edge-TTS voices
    voice_profile = VoiceProfile(
        name="Professional_Female",
        tts_voice="en-US-AriaNeural",  # High-quality neural voice
        pitch_shift=0.0,  # Adjust pitch: -12 to +12 semitones
        speed_ratio=1.0,  # Adjust speed: 0.5 = slower, 1.5 = faster
        volume_gain=1.0,  # Adjust volume: 0.5 = quieter, 1.5 = louder
        # Optional: Add RVC model paths if you have trained models
        # rvc_model_path=r"path\to\your\model.pth",
        # rvc_index_path=r"path\to\your\index.index"
    )

    # Initialize converter
    converter = StandaloneAudiobookConverter(config)

    # Print system info
    converter.print_system_info()

    # Check dependencies
    if not EDGE_TTS_AVAILABLE:
        print("\nERROR: edge-tts not installed!")
        print("Install with: pip install edge-tts")
        return

    # Convert all books in directory
    input_dir = Path("books_to_convert")
    if not input_dir.exists():
        input_dir.mkdir()
        print(f"Created {input_dir} directory - add books to convert")
        return

    results = converter.convert_batch(input_dir, voice_profile)

    # Print results summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    print(f"\n{'=' * 50}")
    print(f"CONVERSION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total files: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"{'=' * 50}")

    for filename, success in results.items():
        status = "+" if success else "-"
        print(f"{status} {filename}")


if __name__ == "__main__":
    main()
