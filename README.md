# 🤖 GestureAI - Smart Gesture Recognition Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## ✨ Features

- **📷 Real-time Camera Detection** - Live hand gesture recognition
- **🖼️ Image Upload** - Analyze uploaded photos
- **📊 History & Analytics** - Track predictions with confidence graphs
- **🚀 Production Ready** - MobileNetV3 model optimized for speed
- **🎯 34 Gestures** - Comprehensive hand gesture vocabulary

## 📱 Supported Gestures

```
👍 like | 👎 dislike | ✋ palm | ✌️ peace | 👊 fist
🤟 rock | 👌 ok | 🤙 call | ☮️ peace_inverted | 🛑 stop
👈 point | ✋ stop_inverted | 📸 take_picture | 🙏 holy
❤️ hand_heart | 1️⃣ one | 2️⃣ two_up | 3️⃣ three | 4️⃣ four
🤘 little_finger | 🖕 middle_finger | 🤫 mute | ❌ xsign | ⭕ no_gesture
```

## 🚀 Quickstart (Local)

1. **Clone & Install**

```bash
git clone https://github.com/asm59-345/GestureAI.git
cd GestureAI
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Run Dashboard**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

## ☁️ Deploy to Streamlit Cloud (Free!)

1. Fork/Use this repo on GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click **'Deploy an app'** → Connect GitHub repo
4. Select `app.py` as main file
5. **Auto-deploys!** Get public URL instantly

## 🏗️ Tech Stack

```
Frontend: Streamlit + Custom CSS
Backend: PyTorch MobileNetV3-Small
Preprocessing: torchvision.transforms
Computer Vision: OpenCV
Data: JSON model metadata
```

**Model Stats:**

- 34 classes | ~87% validation accuracy
- Training: 10 epochs, cosine LR decay
- Size: Optimized for web deployment

## 📁 Project Structure

```
├── app.py              # Streamlit dashboard
├── requirements.txt    # Dependencies
├── gesture_model.pth   # Trained PyTorch model
├── class_names.json    # Gesture labels
├── training_history.json # Training metrics
├── README.md          # 📖 You're reading it!
└── .gitignore         # Clean repo
```

## 🎯 Usage Guide

### Camera Mode

1. Click **▶ Start Camera**
2. Show gesture to webcam
3. **Live predictions** overlay (85%+ confidence saved)

### Upload Mode

1. Upload JPG/PNG
2. Get **top-5 predictions** with confidence bars

### History Tab

- Prediction log
- **Confidence trend chart**

## 🔧 Troubleshooting

| Issue                | Solution                          |
| -------------------- | --------------------------------- |
| `Missing dependency` | `pip install -r requirements.txt` |
| Camera not working   | Check browser permissions         |
| Low confidence       | Better lighting, clear gestures   |
| CUDA out of memory   | Model auto-falls back to CPU      |

## 🤝 Contributing

1. Fork repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push & PR!

## 📄 License

MIT License - Free to use/modify/share.

---

**Developed by Ashmit Gautam & Team** 🚀
**Hackathon 2026 | srimt/coL**
