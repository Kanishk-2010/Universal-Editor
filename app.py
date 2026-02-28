import streamlit as st
import moviepy as mp
from moviepy import VideoFileClip
import google.generativeai as genai
import whisper
import librosa
import os

# --- LEONARDO-STYLE GLASSMORPHISM UI ---
st.set_page_config(page_title="Aetheria Pro | AI Studio", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #1a1b26, #020408); color: #ffffff; }
    [data-testid="stSidebar"] { background: rgba(17, 24, 39, 0.7) !important; backdrop-filter: blur(15px); }
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
        color: white; border-radius: 12px; padding: 14px; font-weight: 700; width: 100%;
    }
    .stChatMessage { background: rgba(31, 41, 55, 0.6); backdrop-filter: blur(10px); border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- AI CONFIGURATION ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("🔑 Please add GEMINI_API_KEY to your Streamlit Secrets!")
    st.stop()

# --- SIDEBAR & UPLOAD ---
with st.sidebar:
    st.title("🌌 Aetheria Pro")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"])

# --- MAIN WORKSPACE ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🤖 AI Assistant")
    if prompt := st.chat_input("Tell the AI how to edit..."):
        if uploaded_video:
            with st.status("🏗️ Rendering...", expanded=True) as status:
                # Save temp file
                with open("input.mp4", "wb") as f:
                    f.write(uploaded_video.read())
                
                # MODERN MOVIEPY 2.0.0 LOGIC
                clip = VideoFileClip("input.mp4")
                
                # Effects now use the .with_effects() method
                if "noir" in prompt.lower():
                    # Correct way to apply Black and White in v2.0
                    final = clip.subclipped(0, min(5, clip.duration)).with_effects([mp.video.fx.BlackAndWhite()])
                else:
                    # Correct way to trim (subclip is now subclipped)
                    final = clip.subclipped(0, min(5, clip.duration))

                final.write_videofile("output.mp4", codec="libx264", audio_codec="aac")
                st.success("Render Complete!")
                st.rerun()
        else:
            st.warning("Upload a video first!")

with col_right:
    st.subheader("📺 Preview")
    if os.path.exists("output.mp4"):
        st.video("output.mp4")
        with open("output.mp4", "rb") as f:
            st.download_button("📥 Download Master", f, "Aetheria_Edit.mp4")
