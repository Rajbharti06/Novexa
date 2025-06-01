# AI Video Caption & Translation Tool

This tool helps you:
1. Download YouTube videos
2. Generate accurate captions in original language
3. Translate captions to desired language
4. Generate subtitle files (.srt)

## Setup Instructions

1. Install Python 3.10 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/download.html
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

4. Run the app:
```bash
streamlit run app.py
```

## Features
- Support for multiple languages
- Accurate speech recognition using Whisper
- Translation support
- Easy-to-use web interface
- Export to .srt format