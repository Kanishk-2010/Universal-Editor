import streamlit as st
import google.generativeai as genai
import yt_dlp
import moviepy as mp
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.audio.fx import MultiplyVolume
import tempfile
import cv2
import numpy as np
from pathlib import Path
import json
import time
from datetime import timedelta
import hashlib
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import shutil
import whisper
import torch
from scipy import signal
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Video Editor Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,75,75,0.3);
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: linear-gradient(45deg, #f3f3f3, #ffffff);
    }
    .success-box {
        background-color: #DFF2BF;
        color: #4F8A10;
        padding: 10px;
        border-radius: 5px;
        animation: slideIn 0.5s ease;
    }
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .caption-preview {
        border-left: 3px solid #FF4B4B;
        padding-left: 15px;
        margin: 10px 0;
        font-family: 'Arial Black', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

class CaptionGenerator:
    """Handles AI-powered caption generation using Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """Initialize Whisper model"""
        self.model = whisper.load_model(model_size)
        self.font_path = self._get_font_path()
        
    def _get_font_path(self):
        """Get path to a bold font for captions"""
        # Try different font paths based on OS
        possible_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\Arial.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Fallback to default
        return None
    
    def generate_captions(self, audio_path: str) -> List[Dict]:
        """Generate captions from audio using Whisper"""
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False,
                language='en'
            )
            
            # Format captions with timestamps
            captions = []
            for segment in result['segments']:
                # Break long segments into smaller chunks for trendy captions
                words = segment.get('words', [])
                
                if words:
                    # Group words into caption-sized chunks (3-6 words)
                    chunk_size = 4
                    for i in range(0, len(words), chunk_size):
                        chunk = words[i:i+chunk_size]
                        if chunk:
                            text = ' '.join([w['word'].strip() for w in chunk])
                            start = chunk[0]['start']
                            end = chunk[-1]['end']
                            
                            captions.append({
                                'text': text,
                                'start': start,
                                'end': end
                            })
                else:
                    # Fallback if word timestamps aren't available
                    text = segment['text'].strip()
                    if text:
                        captions.append({
                            'text': text,
                            'start': segment['start'],
                            'end': segment['end']
                        })
            
            return captions
            
        except Exception as e:
            logger.error(f"Error generating captions: {e}")
            return []
    
    def create_caption_clip(self, text: str, duration: float, video_size: Tuple[int, int], 
                           style: str = "trendy") -> mp.TextClip:
        """Create a styled caption clip"""
        width, height = video_size
        
        # Style configurations
        styles = {
            "trendy": {
                "fontsize": int(height * 0.08),  # 8% of video height
                "color": "white",
                "stroke_color": "black",
                "stroke_width": 3,
                "font": self.font_path or "Arial-Bold",
                "method": "caption"
            },
            "minimal": {
                "fontsize": int(height * 0.06),
                "color": "white",
                "stroke_color": None,
                "stroke_width": 1,
                "font": self.font_path or "Arial",
                "method": "caption"
            },
            "bold": {
                "fontsize": int(height * 0.1),
                "color": "yellow",
                "stroke_color": "black",
                "stroke_width": 4,
                "font": self.font_path or "Arial-Black",
                "method": "caption"
            }
        }
        
        config = styles.get(style, styles["trendy"])
        
        # Create text clip with background for better visibility
        txt_clip = mp.TextClip(
            text.upper() if style == "bold" else text,
            fontsize=config["fontsize"],
            color=config["color"],
            font=config["font"],
            stroke_color=config["stroke_color"],
            stroke_width=config["stroke_width"],
            method=config["method"],
            size=(width * 0.9, None)  # 90% of video width
        )
        
        # Add subtle animation (optional)
        txt_clip = txt_clip.set_duration(duration)
        
        return txt_clip

class BeatDetector:
    """Handles beat detection and synchronization"""
    
    @staticmethod
    def detect_beats(audio_path: str, fps: float = 30) -> List[float]:
        """Detect volume peaks/beats in audio"""
        try:
            # Load audio using librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, 
                sr=sr,
                units='time',
                hop_length=512
            )
            
            # Convert to seconds
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
            
            # Also detect onset strength for more granular peaks
            onset_frames = librosa.onset.onset_detect(
                y=y, 
                sr=sr,
                hop_length=512,
                backtrack=True
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            
            # Combine beats and onsets, sort and remove duplicates
            all_peaks = sorted(set(np.concatenate([beat_times, onset_times])))
            
            # Filter peaks that are too close (within 0.1 seconds)
            filtered_peaks = []
            last_peak = -1
            for peak in all_peaks:
                if peak - last_peak > 0.1:
                    filtered_peaks.append(peak)
                    last_peak = peak
            
            logger.info(f"Detected {len(filtered_peaks)} beats/peaks")
            return filtered_peaks
            
        except Exception as e:
            logger.error(f"Error detecting beats: {e}")
            # Fallback: generate peaks at regular intervals
            return np.arange(0, 60, 2).tolist()  # Every 2 seconds as fallback
    
    @staticmethod
    def align_transitions_to_beats(clip_durations: List[float], 
                                   beat_times: List[float]) -> List[float]:
        """Align clip transitions to nearest beats"""
        aligned_durations = []
        
        for duration in clip_durations:
            # Find nearest beat
            if beat_times:
                nearest_beat = min(beat_times, key=lambda x: abs(x - duration))
                aligned_durations.append(nearest_beat)
            else:
                aligned_durations.append(duration)
        
        return aligned_durations

class AspectRatioProcessor:
    """Handles aspect ratio conversion with smart cropping"""
    
    def __init__(self):
        self.aspect_ratios = {
            "16:9": (16/9, "landscape"),
            "9:16": (9/16, "portrait"),
            "1:1": (1, "square"),
            "4:5": (4/5, "portrait"),  # Instagram portrait
            "21:9": (21/9, "ultrawide")
        }
    
    def smart_crop(self, video_path: str, target_ratio: str, 
                   output_path: str) -> str:
        """Crop video to target aspect ratio while preserving important content"""
        try:
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Get target dimensions
            target_width, target_height = self._calculate_dimensions(
                video.size, 
                self.aspect_ratios[target_ratio][0]
            )
            
            # Detect important regions using computer vision
            important_regions = self._detect_important_regions(video)
            
            # Apply smart cropping
            cropped_video = self._apply_smart_crop(
                video, 
                target_width, 
                target_height,
                important_regions
            )
            
            # Write cropped video
            cropped_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Clean up
            video.close()
            cropped_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in smart cropping: {e}")
            return video_path
    
    def _calculate_dimensions(self, original_size: Tuple[int, int], 
                             target_ratio: float) -> Tuple[int, int]:
        """Calculate new dimensions maintaining target ratio"""
        width, height = original_size
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            # Need to crop width
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Need to crop height
            new_width = width
            new_height = int(width / target_ratio)
        
        return new_width, new_height
    
    def _detect_important_regions(self, video: mp.VideoFileClip) -> List[Tuple]:
        """Detect important regions using face detection and motion analysis"""
        important_regions = []
        
        # Sample frames throughout video
        for t in np.linspace(0, min(video.duration, 30), num=10):  # Sample up to 30 seconds
            frame = video.get_frame(t)
            
            # Convert to RGB for OpenCV
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect faces (most important)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                important_regions.append((x, y, x+w, y+h, "face"))
            
            # Detect motion centers (if no faces)
            if len(faces) == 0:
                # Calculate motion intensity map
                if t > 0:
                    prev_frame = video.get_frame(t - 0.1)
                    diff = cv2.absdiff(
                        cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    )
                    # Find region with highest motion
                    _, max_val, _, max_loc = cv2.minMaxLoc(diff)
                    if max_val > 30:  # Threshold for significant motion
                        important_regions.append((
                            max_loc[0] - 50, max_loc[1] - 50,
                            max_loc[0] + 50, max_loc[1] + 50,
                            "motion"
                        ))
        
        return important_regions
    
    def _apply_smart_crop(self, video: mp.VideoFileClip, target_width: int, 
                          target_height: int, important_regions: List[Tuple]) -> mp.VideoFileClip:
        """Apply cropping while preserving important regions"""
        
        def crop_frame(get_frame, t):
            frame = get_frame(t)
            h, w = frame.shape[:2]
            
            # Determine crop region
            if important_regions:
                # Find the most important region near this timestamp
                # Simplified: use the first important region
                x1, y1, x2, y2, _ = important_regions[0]
                
                # Calculate crop center that includes the important region
                region_center_x = (x1 + x2) // 2
                region_center_y = (y1 + y2) // 2
                
                # Calculate crop window centered on important region
                crop_x1 = max(0, region_center_x - target_width // 2)
                crop_y1 = max(0, region_center_y - target_height // 2)
                
                # Adjust if crop window goes out of bounds
                if crop_x1 + target_width > w:
                    crop_x1 = w - target_width
                if crop_y1 + target_height > h:
                    crop_y1 = h - target_height
            else:
                # Default to center crop
                crop_x1 = (w - target_width) // 2
                crop_y1 = (h - target_height) // 2
            
            crop_x2 = crop_x1 + target_width
            crop_y2 = crop_y1 + target_height
            
            return frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Create new video with cropped frames
        cropped_video = video.fl(crop_frame)
        
        return cropped_video

class EnhancedVideoProcessor(VideoProcessor):
    """Enhanced video processor with new features"""
    
    def __init__(self, temp_dir: str):
        super().__init__(temp_dir)
        self.caption_generator = CaptionGenerator()
        self.beat_detector = BeatDetector()
        self.aspect_processor = AspectRatioProcessor()
    
    def add_captions_to_video(self, video_path: str, audio_path: str, 
                              caption_style: str = "trendy") -> str:
        """Add AI-generated captions to video"""
        try:
            # Generate captions
            captions = self.caption_generator.generate_captions(audio_path)
            
            if not captions:
                logger.warning("No captions generated")
                return video_path
            
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Create caption clips
            caption_clips = []
            for caption in captions:
                txt_clip = self.caption_generator.create_caption_clip(
                    caption['text'],
                    caption['end'] - caption['start'],
                    video.size,
                    caption_style
                )
                
                # Position at bottom with animation
                txt_clip = txt_clip.set_start(caption['start'])
                txt_clip = txt_clip.set_position(('center', video.h * 0.8))
                
                # Add fade in/out
                txt_clip = txt_clip.crossfadein(0.1).crossfadeout(0.1)
                
                caption_clips.append(txt_clip)
            
            # Composite video with captions
            final_video = mp.CompositeVideoClip([video] + caption_clips)
            
            # Write output
            output_path = video_path.replace('.mp4', '_captioned.mp4')
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Clean up
            video.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding captions: {e}")
            return video_path
    
    def create_beat_synced_video(self, clips_info: List[Dict], 
                                 music_path: str, output_path: str) -> str:
        """Create video with transitions synced to music beats"""
        try:
            # Detect beats in music
            beat_times = self.beat_detector.detect_beats(music_path)
            
            # Align clip durations to beats
            clip_durations = [clip_info.get('duration', 2.0) for clip_info in clips_info]
            aligned_durations = self.beat_detector.align_transitions_to_beats(
                clip_durations, beat_times
            )
            
            # Create video clips with aligned durations
            video_clips = []
            for i, clip_info in enumerate(clips_info):
                clip = mp.VideoFileClip(clip_info['path'])
                
                # Use aligned duration
                duration = aligned_durations[i] if i < len(aligned_durations) else 2.0
                
                # Extract segment around hook
                hook_time = clip_info['hook_time']
                start_time = max(0, hook_time - duration/2)
                end_time = min(clip.duration, hook_time + duration/2)
                
                if end_time - start_time > duration:
                    end_time = start_time + duration
                
                subclip = clip.subclip(start_time, end_time)
                
                # Add crossfade transition
                if i > 0:
                    subclip = subclip.crossfadein(0.2)
                
                video_clips.append(subclip)
            
            # Load background music
            if music_path and os.path.exists(music_path):
                audio_bg = mp.AudioFileClip(music_path)
                
                # Calculate total video duration
                total_duration = sum(clip.duration for clip in video_clips)
                
                # Loop or trim music to match video
                if audio_bg.duration < total_duration:
                    # Loop music
                    n_loops = int(np.ceil(total_duration / audio_bg.duration))
                    audio_clips = [audio_bg] * n_loops
                    audio_bg = mp.concatenate_audioclips(audio_clips)
                
                audio_bg = audio_bg.subclip(0, total_duration)
                
                # Reduce original audio volume and mix with music
                for clip in video_clips:
                    if clip.audio:
                        clip.audio = clip.audio.volumex(0.3)  # Reduce original audio
                
                # Concatenate video
                final_video = mp.concatenate_videoclips(video_clips, method="compose")
                
                # Mix audio
                final_audio = mp.CompositeAudioClip([final_video.audio, audio_bg])
                final_video = final_video.set_audio(final_audio)
            else:
                # No background music, just concatenate
                final_video = mp.concatenate_videoclips(video_clips, method="compose")
            
            # Write final video
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30,
                preset='medium',
                threads=4
            )
            
            # Clean up
            for clip in video_clips:
                clip.close()
            if 'audio_bg' in locals():
                audio_bg.close()
            final_video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating beat-synced video: {e}")
            # Fallback to regular assembly
            return super().assemble_video(clips_info, 2.0, output_path)

def main():
    st.title("🎬 AI Video Editor Pro")
    st.markdown("### Professional Video Editing Powered by AI")
    
    # Initialize session state for feature toggles
    if 'features' not in st.session_state:
        st.session_state.features = {
            'captions': False,
            'beat_sync': False,
            'aspect_ratio': '16:9'
        }
    
    # Sidebar for API configuration and features
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password")
        
        st.markdown("---")
        
        # Feature Toggles
        st.header("✨ Advanced Features")
        
        with st.expander("🎯 Auto-Captions", expanded=True):
            st.session_state.features['captions'] = st.checkbox(
                "Enable AI Captions",
                help="Generate and overlay trendy text captions using Whisper AI"
            )
            if st.session_state.features['captions']:
                caption_style = st.selectbox(
                    "Caption Style",
                    ["trendy", "bold", "minimal"],
                    help="Choose the visual style for captions"
                )
                st.session_state.caption_style = caption_style
        
        with st.expander("🎵 Beat Sync", expanded=True):
            st.session_state.features['beat_sync'] = st.checkbox(
                "Enable Beat Sync",
                help="Sync transitions to music beats"
            )
            if st.session_state.features['beat_sync']:
                uploaded_music = st.file_uploader(
                    "Upload Background Music",
                    type=['mp3', 'wav', 'm4a']
                )
                if uploaded_music:
                    st.session_state.background_music = uploaded_music
        
        with st.expander("📐 Aspect Ratio", expanded=True):
            st.session_state.features['aspect_ratio'] = st.selectbox(
                "Output Format",
                ["16:9 (YouTube)", "9:16 (TikTok/Shorts)", "1:1 (Instagram)", "4:5 (Instagram Portrait)"],
                help="Choose output aspect ratio with smart cropping"
            )
        
        st.markdown("---")
        st.header("📊 Processing Status")
        status_placeholder = st.empty()
        
        st.markdown("---")
        st.header("ℹ️ How it works")
        st.info("""
        1. Upload 5-10 video clips
        2. Paste a YouTube link as style reference
        3. Enable AI features (captions, beat sync)
        4. AI analyzes and edits your video
        5. Download the final masterpiece
        """)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Your Raw Clips")
        uploaded_files = st.file_uploader(
            "Choose video files (5-10 clips recommended)",
            type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
            accept_multiple_files=True,
            help="Upload your raw video clips for editing"
        )
        
        if uploaded_files:
            if len(uploaded_files) < 5:
                st.warning(f"⚠️ Upload at least 5 clips. Currently: {len(uploaded_files)}")
            else:
                st.success(f"✅ {len(uploaded_files)} clips uploaded successfully")
                
                # Show clip previews in a grid
                with st.expander("🎬 Preview uploaded clips", expanded=False):
                    cols = st.columns(3)
                    for i, file in enumerate(uploaded_files[:6]):  # Show first 6
                        with cols[i % 3]:
                            st.write(f"**Clip {i+1}**")
                            st.write(f"📄 {file.name[:20]}...")
                            st.write(f"📦 {file.size / (1024*1024):.1f} MB")
    
    with col2:
        st.subheader("🎯 Style Reference")
        youtube_url = st.text_input(
            "Paste YouTube URL for style reference",
            placeholder="https://youtube.com/watch?v=..."
        )
        
        if youtube_url:
            st.video(youtube_url)
            
            # Extract video ID for thumbnail
            if 'youtube.com' in youtube_url or 'youtu.be' in youtube_url:
                st.success("✅ Style reference loaded")
    
    # Process button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Generate AI-Powered Video",
            disabled=not (uploaded_files and youtube_url and api_key and len(uploaded_files) >= 5),
            use_container_width=True
        )
    
    if process_button:
        if not api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar")
            return
        
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                status_placeholder.info("🔄 Initializing video processing...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Download YouTube reference video
                status_text.text("📥 Downloading style reference...")
                downloader = YouTubeDownloader()
                yt_path = os.path.join(temp_dir, "reference.mp4")
                
                try:
                    yt_info = downloader.download_video(youtube_url, yt_path)
                    progress_bar.progress(15)
                except Exception as e:
                    st.error(f"Failed to download YouTube video: {str(e)}")
                    return
                
                # Step 2: Analyze YouTube video style with AI
                status_text.text("🤖 Analyzing editing style with AI...")
                ai_analyzer = AIVideoAnalyzer(api_key)
                style_analysis = ai_analyzer.analyze_cut_frequency(yt_info)
                
                cut_interval = style_analysis.get('suggested_cut_interval', 2.0)
                st.info(f"🎯 Detected cut frequency: Every {cut_interval:.1f} seconds")
                progress_bar.progress(30)
                
                # Step 3: Process uploaded clips
                status_text.text("🔍 Scanning clips for high-motion hooks...")
                processor = EnhancedVideoProcessor(temp_dir)
                clips_info = []
                
                # Save uploaded files
                saved_paths = []
                for i, uploaded_file in enumerate(uploaded_files):
                    clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
                    with open(clip_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(clip_path)
                
                # Find hooks in each clip
                for i, clip_path in enumerate(saved_paths):
                    status_text.text(f"Analyzing clip {i+1}/{len(saved_paths)}...")
                    hook_times = processor.find_hook_moments(clip_path, num_hooks=2)
                    
                    if hook_times:
                        clips_info.append({
                            'path': clip_path,
                            'hook_time': hook_times[0],  # Use best hook
                            'index': i,
                            'duration': cut_interval
                        })
                
                progress_bar.progress(50)
                
                # Step 4: Handle background music if beat sync enabled
                music_path = None
                if st.session_state.features.get('beat_sync', False):
                    if 'background_music' in st.session_state:
                        music_path = os.path.join(temp_dir, "background_music.mp3")
                        with open(music_path, "wb") as f:
                            f.write(st.session_state.background_music.getbuffer())
                        status_text.text("🎵 Analyzing music beats...")
                
                # Step 5: Assemble video (with or without beat sync)
                status_text.text("🎬 Assembling video...")
                output_path = os.path.join(temp_dir, "temp_video.mp4")
                
                if st.session_state.features.get('beat_sync', False) and music_path:
                    # Create beat-synced video
                    final_path = processor.create_beat_synced_video(
                        clips_info,
                        music_path,
                        output_path
                    )
                else:
                    # Regular assembly
                    final_path = processor.assemble_video(
                        clips_info,
                        cut_interval,
                        output_path
                    )
                
                progress_bar.progress(70)
                
                # Step 6: Add captions if enabled
                if st.session_state.features.get('captions', False):
                    status_text.text("💬 Generating AI captions...")
                    
                    # Extract audio from video for captioning
                    audio_path = os.path.join(temp_dir, "audio.wav")
                    video_temp = mp.VideoFileClip(final_path)
                    video_temp.audio.write_audiofile(audio_path, logger=None)
                    video_temp.close()
                    
                    # Add captions
                    captioned_path = processor.add_captions_to_video(
                        final_path,
                        audio_path,
                        st.session_state.get('caption_style', 'trendy')
                    )
                    
                    if captioned_path != final_path:
                        final_path = captioned_path
                
                progress_bar.progress(85)
                
                # Step 7: Apply aspect ratio conversion
                if st.session_state.features.get('aspect_ratio', '16:9') != "16:9 (YouTube)":
                    status_text.text("📐 Applying smart cropping...")
                    
                    # Extract aspect ratio from selection
                    ratio_map = {
                        "16:9 (YouTube)": "16:9",
                        "9:16 (TikTok/Shorts)": "9:16",
                        "1:1 (Instagram)": "1:1",
                        "4:5 (Instagram Portrait)": "4:5"
                    }
                    
                    selected_ratio = st.session_state.features['aspect_ratio']
                    target_ratio = ratio_map.get(selected_ratio, "16:9")
                    
                    cropped_path = os.path.join(temp_dir, "cropped_video.mp4")
                    final_path = processor.aspect_processor.smart_crop(
                        final_path,
                        target_ratio,
                        cropped_path
                    )
                
                progress_bar.progress(95)
                
                # Step 8: Provide download link
                status_text.text("✅ Video ready!")
                
                with open(final_path, 'rb') as f:
                    video_bytes = f.read()
                
                st.success("✨ Your AI-edited video is ready!")
                
                # Download button with prominent styling
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Final Video",
                        data=video_bytes,
                        file_name="ai_edited_video.mp4",
                        mime="video/mp4",
                        key="download_button",
                        use_container_width=True
                    )
                
                # Show preview
                st.subheader("🎬 Preview")
                st.video(final_path)
                
                # Show statistics
                with st.expander("📊 Editing Statistics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Clips Used", len(clips_info))
                        st.metric("Cut Interval", f"{cut_interval:.2f}s")
                    
                    with col2:
                        final_duration = len(clips_info) * cut_interval
                        st.metric("Final Duration", f"{final_duration:.1f}s")
                        st.metric("Editing Style", style_analysis.get('editing_style', 'Medium').title())
                    
                    with col3:
                        if st.session_state.features.get('captions', False):
                            st.metric("Captions", "✅ Enabled")
                        if st.session_state.features.get('beat_sync', False):
                            st.metric("Beat Sync", "✅ Enabled")
                
                progress_bar.progress(100)
                status_placeholder.success("✅ Processing complete!")
                
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            logger.exception("Processing error")
            status_placeholder.error("❌ Processing failed")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "AI Video Editor Pro - Powered by Gemini AI, Whisper, and Streamlit"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
