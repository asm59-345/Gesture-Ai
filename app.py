import streamlit as st
import json
import time
from collections import deque

# Assume dependencies installed via requirements.txt
# pip install -r requirements.txt
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import cv2

# ================= CONFIG =================
st.set_page_config(page_title="GestureAI Dashboard", layout="wide")

st.markdown('<div class="title">🤖 GestureAI | Smart Recognition Dashboard</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-weight: 500;'>Multi-Modal Intelligence Interface | <a href=\"https://github.com/asm59-345/GestureAI\">GitHub</a></p>", unsafe_allow_html=True)
st.markdown("---")

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

if "run_cam" not in st.session_state:
    st.session_state.run_cam = False

# ================= UI =================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0a0f1f, #020617);
    color: white;
}
.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
    margin-bottom: 20px;
}
.title {
    font-size: 34px;
    font-weight: bold;
    color: #00ffe0;
}
.prediction {
    font-size: 30px;
    color: #00ffe0;
    font-weight: bold;
}
.metric {
    font-size: 20px;
    color: #00ff9f;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= ABOUT =================
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="glass">
    <h3>🚀 About This App</h3>
    <p>AI-powered gesture recognition dashboard</p>
    <ul>
    <li>34 gestures, 87% accuracy</li>
    <li>Real-time webcam</li>
    <li>Image upload</li>
    <li>PyTorch model</li>
    <li>Streamlit frontend</li>
    </ul>
    <p><a href="https://github.com/asm59-345/GestureAI" style="color:#00ffe0">View Repo</a></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass">
    <h3>📦 Deployed</h3>
    <p>Streamlit Cloud - Auto from GitHub</p>
    <p>requirements.txt → deps</p>
    <p>app.py → entrypoint</p>
    <p>Works everywhere!</p>
    <p>Hackathon 2026</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ================= LOAD CLASS =================
with open('class_names.json', 'r') as f:
    class_data = json.load(f)

class_names = class_data['classes']
NUM_CLASSES = class_data['num_classes']

# ================= MODEL =================
model_path = 'gesture_model.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

def create_model(num_classes):
    import torchvision.models as models
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    return model

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

model = load_model()

# ================= PREPROCESS =================
IMAGE_SIZE = 224
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ================= PREDICT =================
def predict(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.resize((224, 224))
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return probs, pred.item(), conf.item()

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["📷 Camera", "🖼 Upload", "📜 History", "❓ Help"])

# =========================================================
# 📷 REAL-TIME CAMERA
# =========================================================
with tab1:
    st.markdown("### 📷 Live Gesture Detection")

    colA, colB = st.columns(2)

    with colA:
        if st.button("▶ Start Camera"):
            st.session_state.run_cam = True

    with colB:
        if st.button("⏹ Stop Camera"):
            st.session_state.run_cam = False

    FRAME_WINDOW = st.image([])

    pred_buffer = deque(maxlen=5)

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0)

        while st.session_state.run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            probs, pred, conf = predict(image)
            label = class_names[pred]

            # Smooth prediction
            pred_buffer.append(label)
            stable_label = max(set(pred_buffer), key=pred_buffer.count)

            # Draw overlay
            cv2.putText(frame,
                        f"{stable_label} ({conf:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            FRAME_WINDOW.image(frame)

            # Save history if confident
            if conf > 0.85:
                st.session_state.history.append({
                    "label": stable_label,
                    "confidence": conf,
                    "time": time.strftime("%H:%M:%S")
                })

        cap.release()

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

            st.markdown(f'<div class="prediction">🎯 {label.upper()}</div>', unsafe_allow_html=True)
            st.progress(float(conf))
            st.markdown(f'<div class="metric">Confidence: {conf*100:.2f}%</div>', unsafe_allow_html=True)

            st.subheader("📊 Top Predictions")
            top_probs, top_idx = probs.topk(5)

            for i in range(5):
                l = class_names[top_idx[0][i].item()]
                s = top_probs[0][i].item()
                st.write(f"{l} — {s:.2f}")
                st.progress(s)

            st.session_state.history.append({
                "label": label,
                "confidence": conf,
                "time": time.strftime("%H:%M:%S")
            })

# =========================================================
# 📜 HISTORY + GRAPH
# =========================================================
with tab3:
    st.markdown("### 📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet")
    else:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df)

        st.markdown("### 📊 Confidence Trend")
        st.line_chart(df["confidence"])

# =========================================================
# ❓ HELP
# =========================================================
with tab4:
    st.subheader("❓ Ask a Question")
    user_question = st.text_input("Type your question")

    if user_question:
        answer = "This dashboard currently supports gesture detection and upload analytics. Ask about gestures, usage, or model behavior."
        st.markdown(f'<div class="glass"><p><strong>Q:</strong> {user_question}</p><p><strong>A:</strong> {answer}</p></div>', unsafe_allow_html=True)
    else:
        st.info("Ask a question about the app or gestures to get a helpful response.")

    st.markdown("""
### 🧠 How to Use

1. Go to **Camera tab**
2. Click **Start Camera**
3. Show your hand gesture
4. AI detects automatically (no capture)

---

### ✨ Features

- Real-time gesture detection  
- Upload image support  
- Prediction history  
- Confidence graph  

---

### 🧪 Example Gestures

👍 Thumbs Up  
✋ Open Palm  
✌️ Victory  
👊 Fist  
🤟 Rock  
👌 ok  
🤙 call  
👍 like  
👎 dislike  
🛑 stop  
☮️ peace

---

### 📊 Accuracy Logic

- Model outputs probabilities  
- Highest = prediction  
- Confidence = certainty level  
""")

# ================= FOOTER =================
st.markdown("---")
st.markdown("⚡ Powered by PyTorch + Streamlit | GestureAI | Build by Ankit's Team ")
st.markdown(" Developed by Ashmit Gautam")
