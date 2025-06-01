import streamlit as st
import whisper
import pytube
import os
import tempfile
from pathlib import Path
import ffmpeg
from langdetect import detect
import pysrt
from datetime import timedelta
from transformers import pipeline
from tqdm import tqdm
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from googletrans import Translator
from pydub import AudioSegment
import numpy as np
import torch
from typing import Dict, List, Optional

# Set page config with custom theme
st.set_page_config(
    page_title="AI Caption Generator Pro",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .success-message {
        padding: 1em;
        border-radius: 5px;
        background-color: #d4edda;
        color: #155724;
        margin: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize models with caching
@st.cache_resource
def load_models():
    """Load all required AI models with GPU support if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {
        "whisper": WhisperModel("large-v2", device=device, compute_type="float16"),
        "translator": GoogleTranslator(source="auto", target="en"),
        "backup_translator": Translator()
    }
    return models

class AudioProcessor:
    """Handle audio processing with advanced features."""
    
    @staticmethod
    def enhance_audio(audio_segment: AudioSegment) -> AudioSegment:
        """Enhance audio quality for better transcription."""
        # Normalize audio
        audio_segment = audio_segment.normalize()
        
        # Convert to mono and set sample rate
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(16000)
        
        return audio_segment

    @staticmethod
    def extract_from_video(video_path: str) -> tuple[str, Optional[str]]:
        """Extract and enhance audio from video."""
        try:
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "enhanced_audio.wav")
            
            # Load video audio
            audio = AudioSegment.from_file(video_path)
            
            # Enhance audio
            enhanced = AudioProcessor.enhance_audio(audio)
            
            # Export enhanced audio
            enhanced.export(audio_path, format="wav")
            
            return audio_path, None
        except Exception as e:
            return None, str(e)

class VideoDownloader:
    """Handle video downloads with advanced features."""
    
    @staticmethod
    def download_from_youtube(url: str) -> tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Download video with metadata."""
        try:
            yt = pytube.YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            temp_dir = tempfile.mkdtemp()
            output_path = stream.download(output_path=temp_dir)
            
            # Collect video metadata
            metadata = {
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "views": yt.views,
                "publish_date": str(yt.publish_date),
                "description": yt.description[:200] + "..." if yt.description else "No description"
            }
            
            return output_path, None, metadata
        except Exception as e:
            return None, str(e), None

class CaptionGenerator:
    """Generate and process captions with advanced features."""
    
    def __init__(self, whisper_model):
        self.model = whisper_model
    
    def generate_captions(self, audio_path: str, language: str = None) -> tuple[Optional[Dict], Optional[str]]:
        """Generate captions with language detection."""
        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            return {
                "segments": list(segments),
                "language": info.language,
                "language_probability": info.language_probability
            }, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def create_srt(transcription: Dict, translators: Dict = None, target_lang: str = None) -> tuple[pysrt.SubRipFile, Optional[pysrt.SubRipFile]]:
        """Create SRT files with translation support."""
        original_subs = pysrt.SubRipFile()
        translated_subs = pysrt.SubRipFile() if target_lang else None
        
        for i, seg in enumerate(transcription["segments"], 1):
            # Create original subtitle
            start = timedelta(seconds=seg.start)
            end = timedelta(seconds=seg.end)
            text = seg.text.strip()
            
            original_subs.append(pysrt.SubRipItem(i, start, end, text))
            
            # Create translated subtitle if requested
            if target_lang and target_lang != transcription["language"]:
                try:
                    translated_text = translators["translator"].translate(
                        text=text,
                        target=target_lang
                    )
                except:
                    try:
                        translated_text = translators["backup_translator"].translate(
                            text,
                            dest=target_lang
                        ).text
                    except:
                        translated_text = text
                
                translated_subs.append(pysrt.SubRipItem(i, start, end, translated_text))
        
        return original_subs, translated_subs

def main():
    st.title("🎥 AI Video Caption Generator Pro")
    st.write("Generate accurate captions for YouTube videos in any language!")

    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    # Initialize processors
    caption_generator = CaptionGenerator(models["whisper"])
    
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    
    # Language settings
    languages = {
        "Auto Detect": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese": "zh",
        "Hindi": "hi",
        "Arabic": "ar"
    }
    
    source_lang = st.sidebar.selectbox(
        "Source Language",
        options=list(languages.keys()),
        index=0
    )
    
    target_lang = st.sidebar.selectbox(
        "Translation Language (Optional)",
        options=list(languages.keys()),
        index=0
    )
    
    # Advanced settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Advanced Settings")
    
    enhance_audio = st.sidebar.checkbox("Enhance Audio Quality", value=True)
    show_metadata = st.sidebar.checkbox("Show Video Metadata", value=True)
    
    # Main interface
    url = st.text_input("🔗 Enter YouTube URL:")
    
    if st.button("🚀 Generate Captions"):
        if url:
            try:
                # Download video
                with st.spinner("📥 Downloading video..."):
                    video_path, error, metadata = VideoDownloader.download_from_youtube(url)
                    if error:
                        st.error(f"❌ Error downloading video: {error}")
                        return
                    
                    if show_metadata and metadata:
                        st.info("📺 Video Information")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Title:** {metadata['title']}")
                            st.write(f"**Author:** {metadata['author']}")
                            st.write(f"**Length:** {metadata['length']} seconds")
                        with col2:
                            st.write(f"**Views:** {metadata['views']:,}")
                            st.write(f"**Published:** {metadata['publish_date']}")
                
                # Process audio
                with st.spinner("🎵 Processing audio..."):
                    audio_path, error = AudioProcessor.extract_from_video(video_path) if enhance_audio else extract_audio(video_path)
                    if error:
                        st.error(f"❌ Error processing audio: {error}")
                        return
                
                # Generate captions
                with st.spinner("🤖 Generating captions... This may take a few minutes."):
                    result, error = caption_generator.generate_captions(
                        audio_path,
                        language=languages[source_lang] if source_lang != "Auto Detect" else None
                    )
                    if error:
                        st.error(f"❌ Error generating captions: {error}")
                        return
                
                # Create SRT files
                original_subs, translated_subs = CaptionGenerator.create_srt(
                    result,
                    models,
                    languages[target_lang] if target_lang != "Auto Detect" else None
                )
                
                # Display results
                st.success("✅ Captions generated successfully!")
                
                # Show language information
                st.info(f"🌍 Detected Language: {result['language']} (Confidence: {result['language_probability']:.2%})")
                
                # Show previews in tabs
                tab1, tab2 = st.tabs(["Original Captions", "Translated Captions"] if translated_subs else ["Original Captions"])
                
                with tab1:
                    st.subheader("Original Captions Preview:")
                    for sub in list(original_subs)[:5]:
                        st.text(f"{sub.start} --> {sub.end}\n{sub.text}\n")
                    
                    st.download_button(
                        label="📥 Download Original SRT",
                        data=str(original_subs),
                        file_name="captions_original.srt",
                        mime="text/plain"
                    )
                
                if translated_subs:
                    with tab2:
                        st.subheader(f"Translated Captions Preview ({target_lang}):")
                        for sub in list(translated_subs)[:5]:
                            st.text(f"{sub.start} --> {sub.end}\n{sub.text}\n")
                        
                        st.download_button(
                            label=f"📥 Download {target_lang} SRT",
                            data=str(translated_subs),
                            file_name=f"captions_{languages[target_lang]}.srt",
                            mime="text/plain"
                        )
                
                # Cleanup
                try:
                    os.remove(video_path)
                    os.remove(audio_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a YouTube URL")

if __name__ == "__main__":
    main() 