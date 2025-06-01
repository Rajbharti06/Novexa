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
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from tqdm import tqdm
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator, DeepL, MyMemoryTranslator
from googletrans import Translator
from pydub import AudioSegment
import numpy as np
from typing import Dict, List, Optional
import shutil
from gtts import gTTS
import base64
import io

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
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e6da4;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #204d74;
        transform: translateY(-2px);
    }
    .success-message {
        padding: 1em;
        border-radius: 5px;
        background-color: #d4edda;
        color: #155724;
        margin: 1em 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stSelectbox {
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Extended language support
LANGUAGES = {
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
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-TW",
    "Hindi": "hi",
    "Arabic": "ar",
    "Bengali": "bn",
    "Dutch": "nl",
    "Greek": "el",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Thai": "th",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Polish": "pl",
    "Romanian": "ro",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Malay": "ms",
    "Persian": "fa"
}

# Initialize models with caching
@st.cache_resource
def load_models():
    """Load all required AI models with GPU support if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper model for better accent handling
    whisper_model = WhisperModel(
        "large-v3",
        device=device,
        compute_type="float16",
        download_root="./models"
    )
    
    # Initialize translators with fallbacks
    translators = {
        "primary": GoogleTranslator(source="auto", target="en"),
        "secondary": MyMemoryTranslator(source="auto", target="en"),
        "fallback": Translator()
    }
    
    # Load text-to-speech model
    tts_model = pipeline("text-to-speech", "microsoft/speecht5_tts")
    
    return {
        "whisper": whisper_model,
        "translators": translators,
        "tts": tts_model
    }

class AudioProcessor:
    """Enhanced audio processing with advanced features."""
    
    @staticmethod
    def enhance_audio(audio_segment: AudioSegment) -> AudioSegment:
        """Enhance audio quality for better transcription."""
        try:
            # Normalize audio
            audio_segment = audio_segment.normalize()
            
            # Convert to mono and set optimal sample rate
            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_frame_rate(16000)
            
            # Apply noise reduction
            audio_segment = AudioProcessor._reduce_noise(audio_segment)
            
            return audio_segment
        except Exception as e:
            st.error(f"Error enhancing audio: {str(e)}")
            return audio_segment

    @staticmethod
    def _reduce_noise(audio: AudioSegment) -> AudioSegment:
        """Apply simple noise reduction."""
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Apply simple noise gate
        noise_threshold = np.percentile(np.abs(samples), 5)
        samples[np.abs(samples) < noise_threshold] = 0
        
        # Create new AudioSegment
        return AudioSegment(
            samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

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

class VideoProcessor:
    """Handle video processing with support for multiple sources."""
    
    @staticmethod
    def process_video(source: str, is_url: bool = True) -> tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Process video from URL or local file."""
        try:
            if is_url:
                if "youtube.com" in source or "youtu.be" in source:
                    return VideoProcessor._process_youtube(source)
                else:
                    return VideoProcessor._process_direct_url(source)
            else:
                return VideoProcessor._process_local_file(source)
        except Exception as e:
            return None, str(e), None

    @staticmethod
    def _process_youtube(url: str) -> tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Process YouTube videos."""
        try:
            yt = pytube.YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            temp_dir = tempfile.mkdtemp()
            output_path = stream.download(output_path=temp_dir)
            
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

    @staticmethod
    def _process_direct_url(url: str) -> tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Process direct video URLs."""
        try:
            import requests
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "video.mp4")
            
            response = requests.get(url, stream=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            metadata = {
                "title": url.split("/")[-1],
                "source": url
            }
            
            return output_path, None, metadata
        except Exception as e:
            return None, str(e), None

    @staticmethod
    def _process_local_file(file_path: str) -> tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Process local video files."""
        try:
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, os.path.basename(file_path))
            shutil.copy2(file_path, output_path)
            
            metadata = {
                "title": os.path.basename(file_path),
                "size": os.path.getsize(file_path) / (1024 * 1024),  # Size in MB
                "local_path": file_path
            }
            
            return output_path, None, metadata
        except Exception as e:
            return None, str(e), None

class CaptionGenerator:
    """Generate and process captions with advanced features."""
    
    def __init__(self, models):
        self.whisper = models["whisper"]
        self.translators = models["translators"]
        self.tts = models["tts"]
    
    def generate_captions(self, audio_path: str, language: str = None) -> tuple[Optional[Dict], Optional[str]]:
        """Generate captions with enhanced language detection."""
        try:
            segments, info = self.whisper.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=True,
                initial_prompt="[Transcribe the following audio accurately, maintaining speaker tone and style.]"
            )
            
            return {
                "segments": list(segments),
                "language": info.language,
                "language_probability": info.language_probability
            }, None
        except Exception as e:
            return None, str(e)

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text with multiple fallbacks."""
        try:
            # Try primary translator
            return self.translators["primary"].translate(text=text, target=target_lang)
        except:
            try:
                # Try secondary translator
                return self.translators["secondary"].translate(text=text, target=target_lang)
            except:
                try:
                    # Try fallback translator
                    return self.translators["fallback"].translate(text, dest=target_lang).text
                except:
                    return text

    def generate_speech(self, text: str, lang: str) -> Optional[bytes]:
        """Generate speech from text."""
        try:
            # Use gTTS for wide language support
            tts = gTTS(text=text, lang=lang)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            return audio_bytes.getvalue()
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None

    def create_srt_with_audio(self, transcription: Dict, target_lang: str = None) -> tuple[pysrt.SubRipFile, Optional[pysrt.SubRipFile], Dict[str, bytes]]:
        """Create SRT files with audio generation."""
        original_subs = pysrt.SubRipFile()
        translated_subs = pysrt.SubRipFile() if target_lang else None
        audio_clips = {}
        
        for i, seg in enumerate(transcription["segments"], 1):
            # Process original caption
            start = timedelta(seconds=seg.start)
            end = timedelta(seconds=seg.end)
            text = seg.text.strip()
            
            original_subs.append(pysrt.SubRipItem(i, start, end, text))
            
            # Generate audio for original
            audio_clips[f"orig_{i}"] = self.generate_speech(text, transcription["language"])
            
            # Process translation if requested
            if target_lang and target_lang != transcription["language"]:
                translated_text = self.translate_text(text, target_lang)
                translated_subs.append(pysrt.SubRipItem(i, start, end, translated_text))
                
                # Generate audio for translation
                audio_clips[f"trans_{i}"] = self.generate_speech(translated_text, target_lang)
        
        return original_subs, translated_subs, audio_clips

def main():
    st.title("🎥 AI Video Caption Generator Pro")
    st.write("Generate accurate captions and voice-overs for videos in multiple languages!")

    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    # Initialize processors
    caption_generator = CaptionGenerator(models)
    
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method",
        ["YouTube URL", "Direct Video URL", "Upload Video"]
    )
    
    # Language settings
    source_lang = st.sidebar.selectbox(
        "Source Language",
        options=list(LANGUAGES.keys()),
        index=0
    )
    
    target_lang = st.sidebar.selectbox(
        "Translation Language (Optional)",
        options=list(LANGUAGES.keys()),
        index=0
    )
    
    # Advanced settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Advanced Settings")
    
    enhance_audio = st.sidebar.checkbox("Enhance Audio Quality", value=True)
    show_metadata = st.sidebar.checkbox("Show Video Metadata", value=True)
    generate_audio = st.sidebar.checkbox("Generate Voice-Over", value=True)
    
    # Main interface
    if input_method == "YouTube URL":
        source = st.text_input("🔗 Enter YouTube URL:")
        is_url = True
    elif input_method == "Direct Video URL":
        source = st.text_input("🔗 Enter Direct Video URL:")
        is_url = True
    else:
        source = st.file_uploader("📁 Upload Video File", type=["mp4", "avi", "mov", "mkv"])
        if source:
            # Save uploaded file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, source.name)
            with open(temp_path, "wb") as f:
                f.write(source.getbuffer())
            source = temp_path
            is_url = False
        
    if st.button("🚀 Generate Captions"):
        if source:
            try:
                # Process video
                with st.spinner("📥 Processing video..."):
                    video_path, error, metadata = VideoProcessor.process_video(source, is_url)
                    if error:
                        st.error(f"❌ Error processing video: {error}")
                        return
                    
                    if show_metadata and metadata:
                        st.info("📺 Video Information")
                        col1, col2 = st.columns(2)
                        with col1:
                            for key, value in list(metadata.items())[:len(metadata)//2]:
                                st.write(f"**{key.title()}:** {value}")
                        with col2:
                            for key, value in list(metadata.items())[len(metadata)//2:]:
                                st.write(f"**{key.title()}:** {value}")
                
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
                        language=LANGUAGES[source_lang] if source_lang != "Auto Detect" else None
                    )
                    if error:
                        st.error(f"❌ Error generating captions: {error}")
                        return
                
                # Create SRT files and generate audio
                original_subs, translated_subs, audio_clips = caption_generator.create_srt_with_audio(
                    result,
                    LANGUAGES[target_lang] if target_lang != "Auto Detect" else None
                )
                
                # Display results
                st.success("✅ Captions generated successfully!")
                
                # Show language information
                st.info(f"🌍 Detected Language: {result['language']} (Confidence: {result['language_probability']:.2%})")
                
                # Show previews in tabs
                tab1, tab2 = st.tabs(["Original Captions", "Translated Captions"] if translated_subs else ["Original Captions"])
                
                with tab1:
                    st.subheader("Original Captions Preview:")
                    for i, sub in enumerate(list(original_subs)[:5], 1):
                        st.text(f"{sub.start} --> {sub.end}\n{sub.text}\n")
                        if generate_audio and f"orig_{i}" in audio_clips:
                            st.audio(audio_clips[f"orig_{i}"])
                    
                    st.download_button(
                        label="📥 Download Original SRT",
                        data=str(original_subs),
                        file_name="captions_original.srt",
                        mime="text/plain"
                    )
                
                if translated_subs:
                    with tab2:
                        st.subheader(f"Translated Captions Preview ({target_lang}):")
                        for i, sub in enumerate(list(translated_subs)[:5], 1):
                            st.text(f"{sub.start} --> {sub.end}\n{sub.text}\n")
                            if generate_audio and f"trans_{i}" in audio_clips:
                                st.audio(audio_clips[f"trans_{i}"])
                        
                        st.download_button(
                            label=f"📥 Download {target_lang} SRT",
                            data=str(translated_subs),
                            file_name=f"captions_{LANGUAGES[target_lang]}.srt",
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
            st.warning("⚠️ Please provide a video source")

if __name__ == "__main__":
    main() 