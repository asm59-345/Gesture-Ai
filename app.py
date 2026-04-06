import streamlit as st
import json
import time
from collections import deque

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

# Safe OpenCV import (won't crash)
try:
    import cv2
except:
    cv2 = None

# ================= CONFIG =================
st.set_page_config(page_title="GestureAI Dashboard", layout="wide")

st.markdown('<div class="title">🤖 GestureAI | Smart Recognition Dashboard</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>Multi-Modal Intelligence Interface</p>", unsafe_allow_html=True)
st.markdown("---")

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= STYLES =================
st.markdown("""
<style>
body {background: radial-gradient(circle at top, #0a0f1f, #020617); color: white;}
.glass {background: rgba(255,255,255,0.05); border-radius: 20px; padding: 20px;}
.title {font-size: 34px; font-weight: bold; color: #00ffe0;}
.prediction {font-size: 30px; color: #00ffe0; font-weight: bold;}
.metric {font-size: 20px; color: #00ff9f;}
</style>
""", unsafe_allow_html=True)

# ================= LOAD CLASS =================
with open('class_names.json', 'r') as f:
    class_data = json.load(f)

class_names = class_data['classes']
NUM_CLASSES = class_data['num_classes']

# ================= MODEL =================
model_path = 'gesture_model.pth'

@st.cache_resource
def load_model():
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    import torchvision.models as models
    model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

# ================= PREPROCESS =================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= PREDICT =================
def predict(image):
    image = image.resize((224, 224))
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return probs, pred.item(), conf.item()

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["📷 Camera", "🖼 Upload", "📜 History", "❓ Help"])

# =========================================================
# 📷 CAMERA (FIXED VERSION)
# =========================================================
with tab1:
    st.markdown("### 📷 Capture Gesture")

    img_file_buffer = st.camera_input("Take a picture")

    pred_buffer = deque(maxlen=5)

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(image, width=600)

        with col2:
            probs, pred, conf = predict(image)
            label = class_names[pred]

            pred_buffer.append(label)
            stable_label = max(set(pred_buffer), key=pred_buffer.count)

            st.markdown(f'<div class="prediction">🎯 {stable_label}</div>', unsafe_allow_html=True)
            st.progress(float(conf))
            st.markdown(f'<div class="metric">Confidence: {conf*100:.2f}%</div>', unsafe_allow_html=True)

            if conf > 0.85:
                st.session_state.history.append({
                    "label": stable_label,
                    "confidence": conf,
                    "time": time.strftime("%H:%M:%S")
                })

# =========================================================
# 🖼 UPLOAD
# =========================================================
with tab2:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1.2,1])

        with col1:
            st.image(image, width=600)

        with col2:
            probs, pred, conf = predict(image)
            label = class_names[pred]

            st.markdown(f'<div class="prediction">🎯 {label}</div>', unsafe_allow_html=True)
            st.progress(float(conf))
            st.markdown(f'<div class="metric">Confidence: {conf*100:.2f}%</div>', unsafe_allow_html=True)

            st.session_state.history.append({
                "label": label,
                "confidence": conf,
                "time": time.strftime("%H:%M:%S")
            })

# =========================================================
# 📜 HISTORY
# =========================================================
with tab3:
    st.markdown("### 📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        st.line_chart(df["confidence"])

# =========================================================
# ❓ HELP
# =========================================================
with tab4:
    st.subheader("How to Use")
    st.markdown("""
1. Go to Camera tab  
2. Capture gesture  
3. View prediction  

Supports:
- Image upload  
- Confidence tracking  
""")

# ================= FOOTER =================
st.markdown("---")
st.markdown("⚡ GestureAI | PyTorch + Streamlit")
