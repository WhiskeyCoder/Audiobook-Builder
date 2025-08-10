# 📚 Audiobook Builder
Convert documents into audiobooks with built-in TTS + optional RVC voice conversion.

Audiobook Builder is a standalone Python tool that can transform text-based documents (PDF, EPUB, DOCX, TXT, etc.) into high-quality audiobooks.
It works out-of-the-box using Edge-TTS for natural-sounding voices and can optionally apply RVC (Retrieval-based Voice Conversion) for pitch, speed, and timbre adjustments.

## Why??
I am as your can probably tell from other repos and code, Dyslexic... I hate reading books, but I also love reading and learning. So i came up with this tool a while ago working with 
Appoli but they recently removed the ability to use the local API from the tool. So i needed to rebuild a solution. I get alot of Humble Bundle DRM books that I never get a chance to read
so i built this tool to take those files and convert them into audiobooks i can listen to when I do other tasks like cleaning, working out or even shopping. 

## Disclaimer 
I do not support piracy or breaking copywrite law, what ever you do with the tool is on you, i wrote it to work with content you own legally. please do not us this script on illegaly 
obtained books or documents.

## ✨ Features
- Multiple input formats: .pdf, .epub, .docx, .doc, .txt
- High-quality TTS with Microsoft Edge Neural voices
- Optional RVC-style processing for pitch/speed/voice tweaks
- Intelligent text chunking for smooth narration flow
- Batch conversion – process entire folders of books at once
- Caching system – avoids reprocessing identical text chunks
- Logging & error handling – see exactly what succeeded or failed
- Configurable – adjust chunk size, output format, bitrate, speed, and more

## 📦 Installation
1. Clone the repository:
   ```
   git clone https://github.com/WhiskeyCoder/audiobook-builder.git
   cd audiobook-builder
   ```

2. Install required dependencies:
   ```pip install -r requirements.txt```

### Required dependencies:
- edge-tts
- pydub
- PyPDF2
- ebooklib

### Optional dependencies:
- DOC/DOCX support:
    pip install python-docx docx2txt
- RVC & audio processing enhancements:
    pip install torch torchaudio librosa numpy scipy

## 🚀 Usage
1. Prepare your books:
   ```Place your input files into a folder called:
   books_to_convert/```

2. Run Audiobook Builder:
   ```python audiobook_builder.py```

### The script will:
1. Detect supported files in books_to_convert
2. Extract text and split into natural reading chunks
3. Convert chunks to audio using your chosen TTS voice
4. Optionally apply RVC processing for pitch/speed changes
5. Combine chunks into a final audiobook in audiobooks/

## ⚙️ Configuration
Edit the main() function in audiobook_builder.py to change settings:

```Example config:
config = AudiobookConfig(
    chunk_size_words=1200,
    max_workers=4,
    audio_format="mp3",
    audio_bitrate="128k",
    cache_enabled=True
)

voice_profile = VoiceProfile(
    name="Professional_Female",
    tts_voice="en-US-AriaNeural",
    pitch_shift=0.0,
    speed_ratio=1.0,
    volume_gain=1.0,
    # rvc_model_path="path/to/model.pth",
    # rvc_index_path="path/to/index.index"
)
```

## 🎙 Example Output
With defaults, converting a 300-page PDF takes ~5–10 minutes on a modern PC.
Output is a single .mp3 audiobook in the audiobooks/ folder.

## 🛠 Roadmap
- [ ] Add CLI arguments for all settings
- [ ] Allow custom chapter splitting
- [ ] Add GUI wrapper for non-technical users
- [ ] Custom Metadata from converted Document

## 📜 License
MIT License – free to use, modify, and share.
