import streamlit as st
import os
import cv2
import whisper
import collections
import numpy as np
import tempfile
import uuid
import textwrap
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont

# --- [1. PROFESSIONAL UI - HIGH CONTRAST] ---
st.set_page_config(page_title="AI Video Editor Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    h1, h2, h3, h4, p, span, label, .stMarkdown { 
        color: #ffffff !important; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
    }
    
    .description-text {
        text-align: center;
        color: #999999 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    div[data-testid="stFileUploader"] {
        background-color: #0e1117;
        border: 1px solid #1e1e1e;
        border-radius: 12px;
        padding: 10px;
    }
    
    div[data-testid="stFileUploadDropzone"] div {
        color: #007BFF !important; 
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button {
        background-color: #007BFF !important;
        color: white !important;
        border-radius: 8px !important;
    }

    .stButton>button, .stDownloadButton>button {
        width: 100%; 
        border-radius: 10px; 
        height: 3.5em;
        background: linear-gradient(135deg, #007BFF 0%, #0056b3 100%) !important;
        color: #ffffff !important; 
        font-weight: bold !important; 
        border: none !important;
    }

    /* Checkbox Styling */
    .stCheckbox label { color: #007BFF !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- [2. LOGIC FUNCTIONS] ---
def make_caption_frame(text, size, color):
    wrapper = textwrap.TextWrapper(width=18) 
    lines = wrapper.wrap(text=text)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 75)
    except:
        font = ImageFont.load_default()

    y_pos = 0
    for line in lines:
        w, h = draw.textbbox((0, 0), line, font=font)[2:]
        x = (size[0] - w) // 2
        for o in [-3, 3]:
            for oy in [-3, 3]:
                draw.text((x+o, y_pos+oy), line, font=font, fill="black")
        draw.text((x, y_pos), line, font=font, fill=color)
        y_pos += h + 15
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

def process_video_pipeline(input_path, output_path, target_lang, caption_color, status_container, add_captions):
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    unique_id = str(uuid.uuid4())[:8]
    temp_audio_path = f"temp_audio_{unique_id}.mp3"
    
    try:
        with VideoFileClip(input_path) as clip:
            W_orig, H_orig = clip.size
            W_target, H_target = 1080, 1920
            target_crop_w = int(H_orig * (9/16))
            
            segments = []
            # Only transcribe if user wants captions AND audio exists
            if add_captions and clip.audio:
                status_container.info("System Status: Analyzing audio for speech...")
                clip.audio.write_audiofile(temp_audio_path, logger=None)
                whisper_model = load_whisper_model()
                result = whisper_model.transcribe(temp_audio_path, language=target_lang)
                
                # FIX: Ghost caption prevention (Only if speech is detected)
                if result['text'].strip():
                    segments = result['segments']
                else:
                    status_container.warning("System Status: No clear speech detected. Skipping captions.")

            status_container.info("System Status: Aligning subject to portrait frame...")
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

            caption_clips = []
            if segments:
                status_container.info("System Status: Rendering multi-line captions...")
                for s in segments:
                    txt_str = s['text'].strip().upper()
                    # Final filter for very short/empty ghost text
                    if not txt_str or len(txt_str) < 2: continue 
                    
                    duration = s['end'] - s['start']
                    if duration <= 0: continue
                    txt_arr = make_caption_frame(txt_str, (W_target, 500), caption_color)
                    txt_clip = ImageClip(txt_arr, transparent=True).set_start(s['start']).set_duration(duration).set_position(('center', H_target * 0.72))
                    caption_clips.append(txt_clip)

            final_video = CompositeVideoClip([portrait_video] + caption_clips, size=(W_target, H_target))
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
            
            final_video.close()
            portrait_video.close()
    finally:
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except: pass

# --- [3. MAIN APPLICATION] ---
st.markdown("<h1 style='text-align: center;'>AI AUTO-SHORTS EDITOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='description-text'>Professional landscape-to-portrait transformation with automated face tracking.</p>", unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### Video Input")
    video_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi"])
    if video_file: st.video(video_file)

with col_right:
    st.markdown("### Configurations")
    
    # Clean Language List (No codes shown)
    lang_map = {"English": "en", "Urdu": "ur", "Hindi": "hi", "Arabic": "ar"}
    lang_selection = st.selectbox("Speech Language", list(lang_map.keys()))
    lang_code = lang_map[lang_selection]
    
    # Toggle Feature for Captions
    st.markdown("---")
    enable_captions = st.checkbox("Generate Automated Captions", value=True)
    
    if enable_captions:
        st.write("Select caption color")
        cap_color = st.color_picker("", "#FFFF00", label_visibility="collapsed")
    else:
        cap_color = "#FFFF00" # Default
        st.info("Captions are disabled. Only face tracking will be applied.")

    if video_file is not None:
        if st.button("START TRANSFORMATION"):
            status_area = st.empty()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(video_file.read())
                tmp_in_path = tmp_in.name

            output_name = f"Short_{uuid.uuid4().hex[:6]}.mp4"
            try:
                process_video_pipeline(tmp_in_path, output_name, lang_code, cap_color, status_area, enable_captions)
                st.success("Process finalized. File is ready for download.")
                with open(output_name, "rb") as f:
                    st.download_button(label="DOWNLOAD PROPORTIONAL SHORT", data=f, file_name=output_name, mime="video/mp4")
                os.remove(tmp_in_path)
            except Exception as e: 
                st.error(f"Execution Error: {e}")
