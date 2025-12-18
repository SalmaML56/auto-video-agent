import streamlit as st
import os
import cv2
import whisper
import collections
import numpy as np
import tempfile
import uuid
import platform
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont

# --- [1. DESIGN & CSS - WAHI ORIGINAL BLUE LOOK] ---
st.set_page_config(page_title="Professional AI Video Editor", layout="wide")

st.markdown("""
    <style>
    /* Dark Background */
    .stApp { background-color: #000000; }
    h1, h2, h3, h4, p, span, label, .stMarkdown { color: #ffffff !important; }
    
    /* File Uploader Container */
    .stFileUploader section { 
        background-color: #0a0a0a !important; 
        border: 2px dashed #1e1e1e !important; 
        border-radius: 15px; 
    }
    
    /* FIX: Browse Files Button (Ab white nahi, Blue hoga) */
    div[data-testid="stFileUploader"] button {
        background-color: #007BFF !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
    }

    /* Professional Blue Gradient Buttons */
    .stButton>button, .stDownloadButton>button {
        width: 100%; 
        border-radius: 12px; 
        height: 3.8em;
        background: linear-gradient(135deg, #007BFF 0%, #0056b3 100%) !important;
        color: #ffffff !important; 
        font-weight: bold !important; 
        border: none !important;
        box-shadow: 0px 4px 15px rgba(0, 123, 255, 0.4);
    }
    .stButton>button:hover {
        transform: scale(1.01);
        box-shadow: 0px 6px 20px rgba(0, 123, 255, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# --- [2. SECURE CAPTION MAKER (No ImageMagick Needed)] ---
def make_text_frame(text, size, color):
    # Transparent background image
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Font Selection
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 75)
    except:
        font = ImageFont.load_default()

    # Calculate text position
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    x = (size[0] - w) // 2
    y = (size[1] - h) // 2
    
    # Draw Outline (Stroke)
    for o in [-2, 2]:
        for oy in [-2, 2]:
            draw.text((x+o, y+oy), text, font=font, fill="black")
    
    # Draw Main Text
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img)

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
            
            status_container.info("Step 1: AI Listening to Audio...")
            if clip.audio:
                clip.audio.write_audiofile(temp_audio_path, logger=None)
                result = whisper_model.transcribe(temp_audio_path, language=target_lang)
                segments = result['segments']
            else: segments = []

            status_container.info("Step 2: Tracking Face & Reframing...")
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

            status_container.info("Step 3: Creating Secure Captions...")
            caption_clips = []
            for s in segments:
                txt_str = s['text'].strip().upper()
                duration = s['end'] - s['start']
                if duration <= 0: continue
                
                # Image-based secure clip
                txt_img = make_text_frame(txt_str, (W_target, 300), caption_color)
                txt_clip = VideoClip(lambda t: txt_img, duration=duration).set_start(s['start']).set_position(('center', H_target * 0.75))
                caption_clips.append(txt_clip)

            final_video = CompositeVideoClip([portrait_video] + caption_clips, size=(W_target, H_target))
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
            
            final_video.close()
            portrait_video.close()
    finally:
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except: pass

# --- [3. APP LAYOUT] ---
st.markdown("<h1 style='text-align: center;'>🎬 AI AUTO-SHORTS EDITOR</h1>", unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📥 Video Input")
    video_file = st.file_uploader("Upload your footage", type=["mp4", "mov", "avi"])
    if video_file: st.video(video_file)

with col_right:
    st.markdown("### ⚙️ Configurations")
    lang_selection = st.selectbox("Speech Language", ["English (en)", "Urdu (ur)", "Hindi (hi)", "Arabic (ar)"])
    lang_code = lang_selection.split('(')[1].strip(')')
    cap_color = st.color_picker("Caption Color", "#FFFF00")
    
    if video_file is not None:
        if st.button("🚀 START TRANSFORMATION"):
            status_area = st.empty()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(video_file.read())
                tmp_in_path = tmp_in.name

            output_name = f"Short_{uuid.uuid4().hex[:6]}.mp4"
            try:
                process_video_pipeline(tmp_in_path, output_name, lang_code, cap_color, status_area)
                st.success("✨ Success! Video is Ready.")
                with open(output_name, "rb") as f:
                    st.download_button(label="📥 DOWNLOAD FINAL SHORT", data=f, file_name=output_name, mime="video/mp4")
                os.remove(tmp_in_path)
            except Exception as e: 
                st.error(f"Error: {e}")
