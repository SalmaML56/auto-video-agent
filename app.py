import streamlit as st
import os
import cv2
import whisper
import collections
import numpy as np
import tempfile
import uuid
import platform
import re
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings

# --- 0. POWERFUL IMAGE MAGICK POLICY FIX ---
def patch_magick_policy():
    """Linux server ki security policy ko bypass karne ke liye"""
    if platform.system() != "Windows":
        try:
            # System ki asli policy file dhoondna
            orig_policy = "/etc/ImageMagick-6/policy.xml"
            if os.path.exists(orig_policy):
                with open(orig_policy, 'r') as f:
                    data = f.read()
                
                # Security restriction ko khatam karna
                new_data = re.sub(r'rights="none"\s+pattern="@\*"', 'rights="read|write" pattern="@*"', data)
                
                # Nayi policy file ko /tmp mein save karna
                tmp_dir = tempfile.gettempdir()
                policy_path = os.path.join(tmp_dir, "policy.xml")
                with open(policy_path, 'w') as f:
                    f.write(new_data)
                
                # System ko batana ke naya policy file istemal kare
                os.environ["MAGICK_CONFIGURE_PATH"] = tmp_dir
                print("ImageMagick Policy Patched Successfully")
        except Exception as e:
            print(f"Policy Patch Failed: {e}")

# Apply the patch
patch_magick_policy()

# --- 1. CONFIGURATION ---
if platform.system() == "Windows":
    change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})
else:
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

st.set_page_config(page_title="AI Video Editor Pro", layout="wide")

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    h1, h2, h3, h4, p, span, label, .stMarkdown { color: #ffffff !important; }
    .stFileUploader section { background-color: #111111 !important; border: 2px dashed #333 !important; }
    .stButton>button { background: linear-gradient(135deg, #007BFF 0%, #0056b3 100%) !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

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
            
            # 1. Audio & Transcription
            status_container.info("Transcribing Audio...")
            if clip.audio:
                clip.audio.write_audiofile(temp_audio_path, logger=None)
                result = whisper_model.transcribe(temp_audio_path, language=target_lang)
                segments = result['segments']
            else: segments = []

            # 2. Reframing
            status_container.info("Tracking Face & Reframing...")
            face_buffer = collections.deque(maxlen=20)

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

            # 3. Captions
            status_container.info("Burning Captions...")
            caption_clips = []
            for s in segments:
                txt = TextClip(
                    s['text'].strip().upper(),
                    fontsize=75, color=caption_color, font='DejaVu-Sans-Bold',
                    stroke_color='black', stroke_width=2,
                    method='caption', size=(W_target * 0.8, None)
                ).set_start(s['start']).set_duration(s['end'] - s['start']).set_position(('center', H_target * 0.75))
                caption_clips.append(txt)

            # 4. Final Export
            final_video = CompositeVideoClip([portrait_video] + caption_clips, size=(W_target, H_target))
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
            
            final_video.close()
            portrait_video.close()
            for c in caption_clips: c.close()
    finally:
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except: pass

# --- UI ---
st.title("🎬 AI Auto-Shorts Editor")
st.write("Upload a landscape video and let AI convert it to a viral portrait short.")

video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if video_file:
    col1, col2 = st.columns(2)
    with col1:
        st.video(video_file)
    with col2:
        lang = st.selectbox("Language", ["en", "ur", "hi", "ar"])
        color = st.color_picker("Caption Color", "#FFFF00")
        
        if st.button("Generate Short"):
            status = st.empty()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(video_file.read())
                tmp_in_path = tmp_in.name

            output_name = f"Short_{uuid.uuid4().hex[:6]}.mp4"
            try:
                process_video_pipeline(tmp_in_path, output_name, lang, color, status)
                st.success("Success!")
                with open(output_name, "rb") as f:
                    st.download_button("Download Short", f, file_name=output_name)
            except Exception as e:
                st.error(f"Error: {e}")
