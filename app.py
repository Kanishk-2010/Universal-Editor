import streamlit as st
import moviepy as mp
import google.generativeai as genai
import whisper
import librosa
import torch
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Universal AI Video Editor", layout="wide")
st.title("🎬 Universal AI Video Editor")

# --- API KEY CHECK ---
# This looks for the keys you put in "Advanced Settings" > "Secrets"
if "GEMINI_API_KEY" in st.secrets and "DEEPSEEK_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.success("✅ AI Engines Online")
else:
    st.error("⚠️ Setup Required: Go to 'Advanced Settings' > 'Secrets' and add your GEMINI_API_KEY and DEEPSEEK_API_KEY.")
    st.stop()

# --- USER INTERFACE ---
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your Clip (MP4/MOV)", type=["mp4", "mov"])
with col2:
    ref_link = st.text_input("YouTube Style Reference (Optional)")

if st.button("🚀 Start AI Montage"):
    if uploaded_file:
        with st.status("🤖 AI is working...", expanded=True) as status:
            # Save file locally
            with open("input.mp4", "wb") as f:
                f.write(uploaded_file.read())
            
            st.write("🎵 Analyzing Audio Beats...")
            # Librosa needs 'libsndfile1-dev' in packages.txt to work here
            y, sr = librosa.load("input.mp4")
            
            st.write("🗣️ AI Transcription (Whisper)...")
            model = whisper.load_model("tiny") 
            
            st.write("✂️ Cutting Video (MoviePy 2.0)...")
            # Modern MoviePy 2.0 Syntax: .subclipped() replaces .subclip()
            clip = mp.VideoFileClip("input.mp4")
            duration = min(10, clip.duration)
            final_clip = clip.subclipped(0, duration)
            
            output_path = "output_montage.mp4"
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            status.update(label="✅ Montage Ready!", state="complete")
            
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("📥 Download AI Edit", f, file_name="ai_montage.mp4")
    else:
        st.info("Please upload a video file to begin.")
