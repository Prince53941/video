import streamlit as st
import os
from faster_whisper import WhisperModel
from transformers import pipeline
import textwrap

# --- PAGE CONFIG ---
st.set_page_config(page_title="Video Summarizer AI", page_icon="🎥", layout="wide")

st.title("🎥 AI Video Summarizer & Subtitler")
st.markdown("""
Upload a video file (mp4, mkv, avi). This app will:
1. **Extract Audio** and transcribe it using **OpenAI Whisper**.
2. **Generate Subtitles** (transcript).
3. **Summarize** the content using a HuggingFace model.
""")

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_whisper_model():
    # 'tiny' or 'base' is faster for CPU. Use 'small' or 'medium' for better accuracy if you have a GPU.
    model = WhisperModel("base", device="cpu", compute_type="int8")
    return model

@st.cache_resource
def load_summarizer():
    # Using a lightweight summarization model suitable for CPUs
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer

# --- 2. PROCESSING FUNCTIONS ---
def extract_transcript(video_path, model):
    segments, info = model.transcribe(video_path, beam_size=5)
    
    full_text = ""
    timestamps = []
    
    progress_bar = st.progress(0)
    st.write(f"Detecting language: {info.language} (Probability: {info.language_probability:.2f})")
    
    # Iterate through segments (this is a generator)
    # We can't know total length easily for progress bar, so we just animate it
    for i, segment in enumerate(segments):
        full_text += segment.text + " "
        timestamps.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        # Update progress bar simply to show activity
        if i % 10 == 0:
            progress_bar.progress(min(i % 100, 100))
            
    progress_bar.progress(100)
    return full_text.strip(), timestamps

def generate_summary(text, summarizer):
    # Models have a max token limit (usually 1024). We need to chunk the text.
    # Simple chunking by characters (approx 3000 chars ~ 700 tokens)
    chunks = textwrap.wrap(text, 3000)
    summary_text = ""
    
    for chunk in chunks:
        # Generate summary for each chunk
        output = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary_text += output[0]['summary_text'] + " "
        
    return summary_text.strip()

# --- 3. UI LAYOUT ---
uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save the file temporarily
    temp_filename = "temp_video.mp4"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())
    
    # Display Video
    st.video(temp_filename)
    
    if st.button("🚀 Process Video"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Loading AI Models... (This happens only once)")
            whisper_model = load_whisper_model()
            summarizer_model = load_summarizer()
            st.success("Models Loaded!")
            
            st.info("Transcribing Audio...")
            transcript_text, timed_subtitles = extract_transcript(temp_filename, whisper_model)
            st.success("Transcription Complete!")
            
        with col2:
            st.info("Summarizing Text...")
            summary = generate_summary(transcript_text, summarizer_model)
            st.success("Summary Complete!")
            
        # --- RESULTS DISPLAY ---
        st.divider()
        
        st.subheader("📝 Summary")
        st.warning(summary)
        
        with st.expander("See Full Transcript"):
            st.text_area("Full Text", transcript_text, height=200)
        
        with st.expander("See Timestamps (Subtitles)"):
            st.write(timed_subtitles)
            
        # Download Buttons
        st.download_button("Download Transcript", transcript_text, file_name="transcript.txt")
        st.download_button("Download Summary", summary, file_name="summary.txt")

    # Cleanup (Optional)
    # os.remove(temp_filename)
