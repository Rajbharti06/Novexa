import streamlit as st
import whisper
import pytube
import os
import tempfile
from pathlib import Path
import ffmpeg
import pysrt
from datetime import timedelta

# Set page config
st.set_page_config(
    page_title="AI Caption Generator",
    page_icon="🎥",
    layout="wide"
)

# Initialize Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def download_youtube_video(url):
    try:
        yt = pytube.YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        temp_dir = tempfile.mkdtemp()
        output_path = stream.download(output_path=temp_dir)
        return output_path, None
    except Exception as e:
        return None, str(e)

def extract_audio(video_path):
    try:
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return audio_path, None
    except Exception as e:
        return None, str(e)

def generate_captions(audio_path, model):
    try:
        result = model.transcribe(audio_path)
        return result, None
    except Exception as e:
        return None, str(e)

def create_srt(transcription):
    subs = pysrt.SubRipFile()
    for i, seg in enumerate(transcription['segments'], 1):
        start = timedelta(seconds=seg['start'])
        end = timedelta(seconds=seg['end'])
        text = seg['text'].strip()
        sub = pysrt.SubRipItem(i, start, end, text)
        subs.append(sub)
    return subs

def main():
    st.title("🎥 AI Caption Generator")
    st.write("Generate accurate captions for YouTube videos!")

    # Load Whisper model
    with st.spinner("Loading AI model..."):
        model = load_whisper_model()
    
    # Input section
    url = st.text_input("Enter YouTube URL:")
    
    if st.button("Generate Captions"):
        if url:
            try:
                # Download video
                with st.spinner("Downloading video..."):
                    video_path, error = download_youtube_video(url)
                    if error:
                        st.error(f"Error downloading video: {error}")
                        return
                
                # Extract audio
                with st.spinner("Processing audio..."):
                    audio_path, error = extract_audio(video_path)
                    if error:
                        st.error(f"Error extracting audio: {error}")
                        return
                
                # Generate captions
                with st.spinner("Generating captions... This may take a few minutes."):
                    result, error = generate_captions(audio_path, model)
                    if error:
                        st.error(f"Error generating captions: {error}")
                        return
                
                # Create SRT file
                subs = create_srt(result)
                
                # Display results
                st.success("✅ Captions generated successfully!")
                
                # Show sample captions
                st.subheader("Preview:")
                for sub in list(subs)[:5]:  # Show first 5 captions
                    st.text(f"{sub.start} --> {sub.end}\n{sub.text}\n")
                
                # Download button for SRT file
                srt_content = str(subs)
                st.download_button(
                    label="Download SRT File",
                    data=srt_content,
                    file_name="captions.srt",
                    mime="text/plain"
                )
                
                # Cleanup
                try:
                    os.remove(video_path)
                    os.remove(audio_path)
                except:
                    pass
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL")

if __name__ == "__main__":
    main() 