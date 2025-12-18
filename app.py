import streamlit as st
import os
import cv2
import whisper
import collections
import numpy as np
import tempfile
import uuid
import platform
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings

# --- 1. SETTINGS & DYNAMIC PATH CONFIGURATION ---
# Yeh hissa automatically detect karega ke aap Windows par hain ya Server par
if platform.system() == "Windows":
    # Apne Windows PC ka path yahan check kar lein
    change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})
else:
    # Online Server (Linux) ke liye path
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

st.set_page_config(page_title="Professional AI Video Editor", layout="wide")

# --- CUSTOM CSS FOR HIGH CONTRAST & VISIBILITY ---
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    
    h1, h2, h3, h4, p, span, label, .stMarkdown {
        color: #ffffff !important;
    }

    /* File Uploader Container */
    .stFileUploader section {
        background-color: #0a0a0a !important;
        border: 2px solid #1e1e1e !important;
        border-radius: 15px;
        padding: 20px;
    }

    /* Browse Files Button Visibility Fix */
    .stFileUploader button {
        background-color: #262626 !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        font-weight: bold !important;
    }
    
    /* Uploader text visibility */
    .stFileUploader div[data-testid="stFileUploaderFileData"] {
        color: white !important;
    }

    /* Main Action & Download Buttons */
    .stButton>button, .stDownloadButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.8em;
        background: linear-gradient(135deg, #007BFF 0%, #0056b3 100%) !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0px 4px 15px rgba(0, 123, 255, 0.3);
        transition: 0.4s ease;
        font-size: 16px;
        text-transform: uppercase;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #0056b3 0%, #004085 100%) !important;
        box-shadow: 0px 6px 20px rgba(0, 123, 255, 0.5);
        transform: translateY(-2px);
    }

    /* Dropdown Styling */
    div[data-baseweb="select"] {
        background-color: #111111 !important;
        border: 1px solid #333 !important;
    }
    
    div[role="listbox"] {
        background-color: #111111 !important;
    }

    hr {
        border-top: 1px solid #222 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def get_stable_center(faces, W_orig, buffer):
    target_x = W_orig / 2
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        target_x = x + (w / 2)
    buffer.append(target_x)
    return sum(buffer) / len(buffer)

def process_video_pipeline(input_path, output_path, target_lang, caption_color, status_container):
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    whisper_model = load_whisper_model()
    unique_id = str(uuid.uuid4())[:8]
    temp_audio_path = f"temp_audio_{unique_id}.mp3"

    try:
        with VideoFileClip(input_path) as clip:
            W_orig, H_orig = clip.size
            W_target, H_target = 1080, 1920
            target_crop_w = int(H_orig * (9/16))
            
            status_container.markdown("#### Status: Transcribing Audio...")
            if clip.audio:
                clip.audio.write_audiofile(temp_audio_path, logger=None)
                result = whisper_model.transcribe(temp_audio_path, language=target_lang)
                segments = result['segments']
            else: segments = []

            status_container.markdown("#### Status: Tracking Face and Reframing...")
            face_buffer = collections.deque(maxlen=25)

            def frame_processor(get_frame, t):
                frame = get_frame(t)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
                smooth_x = get_stable_center(faces, W_orig, face_buffer)
                x1 = int(smooth_x - (target_crop_w / 2))
                x2 = x1 + target_crop_w
                if x1 < 0: x1, x2 = 0, target_crop_w
                if x2 > W_orig: x1, x2 = W_orig - target_crop_w, W_orig
                return cv2.resize(frame[:, x1:x2], (W_target, H_target))

            portrait_video = clip.fl(frame_processor)

            status_container.markdown("#### Status: Burning Stylized Captions...")
            caption_clips = []
            for s in segments:
                txt = TextClip(
                    s['text'].strip().upper(),
                    fontsize=85, color=caption_color, font='Arial-Bold',
                    stroke_color='black', stroke_width=2.5,
                    method='caption', size=(W_target * 0.9, None)
                ).set_start(s['start']).set_duration(s['end'] - s['start']).set_position(('center', H_target * 0.75))
                caption_clips.append(txt)

            final_video = CompositeVideoClip([portrait_video] + caption_clips, size=(W_target, H_target))
            
            status_container.markdown("#### Status: Rendering Final MP4...")
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
            
            final_video.close()
            portrait_video.close()
            for c in caption_clips: c.close()
    finally:
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except: pass

# --- 3. UI LAYOUT ---

st.markdown("<div style='text-align: center; padding: 20px;'><h1>AI AUTO-SHORTS EDITOR</h1><p style='color: #444; letter-spacing: 2px;'>LANDSCAPE TO PORTRAIT INTELLIGENCE</p></div>", unsafe_allow_html=True)

st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### Video Input")
    video_file = st.file_uploader("Upload your footage", type=["mp4", "mov", "avi"])
    if video_file:
        st.video(video_file)

with col_right:
    st.markdown("### Configurations")
    lang_selection = st.selectbox("Speech Language", ["English (en)", "Urdu (ur)", "Hindi (hi)", "Arabic (ar)"])
    lang_code = lang_selection.split('(')[1].strip(')')
    
    cap_color = st.color_picker("Caption Theme Color", "#FFFF00")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if video_file is not None:
        if st.button("START TRANSFORMATION"):
            status_area = st.empty()
            p_bar = st.progress(0)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(video_file.read())
                tmp_in_path = tmp_in.name

            output_file_name = f"Short_{uuid.uuid4().hex[:6]}.mp4"
            
            try:
                process_video_pipeline(tmp_in_path, output_file_name, lang_code, cap_color, status_area)
                p_bar.progress(100)
                st.success("Conversion Complete")
                
                with open(output_file_name, "rb") as f:
                    st.download_button(
                        label="DOWNLOAD FINAL SHORT",
                        data=f,
                        file_name=output_file_name,
                        mime="video/mp4"
                    )
                
                # Cleanup
                if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
            except Exception as error:
                st.error(f"Error: {error}")
    else:
        st.info("Upload a video to enable processing.")

st.markdown("<br><br><br>", unsafe_allow_html=True)