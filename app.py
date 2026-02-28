import streamlit as st
import moviepy as mp
from moviepy import VideoFileClip
import google.generativeai as genai
import whisper
import librosa
import os

# --- 1. GLASSMORPHISM UI (LEONARDO AI VIBE) ---
st.set_page_config(page_title="Aetheria Pro | AI Studio", layout="wide")

st.markdown("""
    <style>
    /* Dark Leonardo Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1b26, #020408);
        color: #ffffff;
    }
    
    /* Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Neon Accent Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
        color: white; border: none; border-radius: 12px;
        padding: 14px 28px; font-weight: 700; width: 100%;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(139, 92, 246, 0.6);
    }

    /* Chat Bubbles */
    .stChatMessage {
        background: rgba(31, 41, 55, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin-bottom: 15px;
    }

    .glow-header {
        text-shadow: 0 0 20px #8b5cf6;
        color: #ddd6fe;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE (The AI's Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "master_file" not in st.session_state:
    st.session_state.master_file = None

# --- 3. SIDEBAR: PROFESSIONAL ASSET PANEL ---
with st.sidebar:
    st.markdown("<h1 class='glow-header'>AETHERIA PRO</h1>", unsafe_allow_html=True)
    st.caption("Universal AI Video Engine v5.0")
    st.divider()
    
    # Secrets Check
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.success("✨ Neural Link: Active")
    else:
        st.error("🔑 API Key required in Secrets!")
        st.stop()
    
    st.divider()
    uploaded_video = st.file_uploader("📂 Import Media", type=["mp4", "mov"])
    
    if uploaded_video:
        st.session_state.master_file = "raw_source.mp4"
        with open(st.session_state.master_file, "wb") as f:
            f.write(uploaded_video.read())
        st.info("Source footage locked.")

# --- 4. THE EDITING WORKSPACE ---
left_panel, right_panel = st.columns([1, 1])

with left_panel:
    st.subheader("💬 AI Command Center")
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ex: 'Create a dramatic noir sequence'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.master_file:
                st.error("Upload a clip in the sidebar first!")
            else:
                with st.status("🛸 Generating Edit...", expanded=True) as status:
                    try:
                        # NEW MOVIEPY 2.0.0 SYNTAX
                        clip = VideoFileClip(st.session_state.master_file)
                        
                        # AI Prompt Logic
                        if "noir" in prompt.lower() or "black" in prompt.lower():
                            st.write("Processing Noir filter...")
                            # Correct 2.0.0 Method: .with_effects()
                            final = clip.subclipped(0, min(10, clip.duration)).with_effects([mp.video.fx.BlackAndWhite()])
                        else:
                            st.write("Rendering cinematic sequence...")
                            final = clip.subclipped(0, min(10, clip.duration))

                        output = "pro_export.mp4"
                        final.write_videofile(output, codec="libx264", audio_codec="aac", logger=None)
                        
                        st.session_state.messages.append({"role": "assistant", "content": "Master render complete. Ready for download."})
                        status.update(label="✨ Studio Render Finished", state="complete")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Render Error: {e}")

with right_panel:
    st.subheader("📺 Studio Preview")
    if os.path.exists("pro_export.mp4"):
        st.video("pro_export.mp4")
        with open("pro_export.mp4", "rb") as f:
            st.download_button("📥 Export High-Bitrate Master", f, "Aetheria_Pro_Master.mp4")
    else:
        st.info("Awaiting AI instructions to generate preview...")
