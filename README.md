# 🎬 AI Auto-Shorts Editor

**Landscape-to-Portrait Video Transformation Pipeline**

🔗 **Live App:**
[https://auto-video-agent-dnxuu3ut5bbzpy5qhtrzmt.streamlit.app/](https://auto-video-agent-dnxuu3ut5bbzpy5qhtrzmt.streamlit.app/)

---

## 📌 Overview

AI Auto-Shorts Editor is a full-stack **AI-powered video automation system** that converts landscape (16:9) videos into **social-media-ready portrait (9:16) shorts**.
The system intelligently tracks faces, reframes videos, and generates **multilingual AI captions** — all automatically.

This project is designed for **content creators, short-form video platforms, and AI automation workflows**.

---

## 🚀 Key Features

### 🎥 Automatic Video Reframing

* Converts landscape videos (16:9) to portrait format (9:16).
* Ensures the main subject remains centered throughout the video.

### 👤 Real-Time Face Tracking

* Implemented using **OpenCV Haar Cascades**.
* Dynamically crops frames to follow the face smoothly.

### 🧠 AI-Powered Speech-to-Text

* Integrated **OpenAI Whisper** for high-accuracy transcription.
* Supports **English, Urdu, Hindi, and Arabic** captions.

### 📝 Custom Caption Rendering Engine

* Built using **Pillow (PIL)** to avoid ImageMagick security issues on cloud servers.
* Supports:

  * Dynamic word wrapping
  * Multi-line captions
  * Clean subtitle placement

### 🎨 Interactive Streamlit Dashboard

* Dark-themed UI for better UX.
* Toggle AI captions on/off.
* Custom color picker for caption styling and branding.

### ⚡ Performance & Scalability

* Optimized for large video files and AI processing workloads.
* Successfully deployed on **Streamlit Cloud** with stable performance.

---

## 🧩 Technologies Used

* **Python**
* **Streamlit**
* **OpenAI Whisper**
* **OpenCV**
* **MoviePy**
* **Pillow (PIL)**

---

## 🛠️ How It Works

1. User uploads a landscape video.
2. AI detects and tracks faces frame-by-frame.
3. Video is intelligently cropped to portrait format.
4. Whisper transcribes speech and generates captions.
5. Custom caption engine renders subtitles directly onto frames.
6. Final short video is exported, ready for social media.

---

## 🧪 Use Cases

* YouTube Shorts
* Instagram Reels
* TikTok Videos
* Automated video repurposing
* AI-based content creation pipelines

---

## 📌 Why This Project Matters

* Demonstrates **Computer Vision + NLP + AI Automation**
* Shows **real-world production deployment**
* Highlights **problem-solving beyond basic ML models**
* Strong fit for **AI/ML Engineer & AI Automation Engineer roles**


