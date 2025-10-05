# 🚗 SafeRide_AI: Driver Safety Monitoring System

An AI-powered safety system for ride-hailing services like Ola and Uber. It uses YOLO-based computer vision to monitor driver attention, drowsiness levels, phone usage, and speech-to-text analysis to detect offensive or unsafe language, ensuring passenger safety in real-time.

## 📖 Overview

SafeRide_AI is a **hybrid computer vision and audio processing system** that uses your PC webcam to track a driver's **attention level** and **drowsiness** in real-time. It combines **YOLOv8** for object detection with **MediaPipe** for face and eye analysis to provide comprehensive driver monitoring.

The system outputs a continuous **attention score (0–100)** that reflects how focused the driver is, adjusting dynamically based on various behavioral indicators.

## ✨ Features

### Video Monitoring
- **Driver Attention Tracking**: Monitors driver's focus level using facial landmark detection
- **Drowsiness Detection**: Detects signs of drowsiness through eye closure tracking
- **Phone Usage Detection**: Identifies when driver is using a mobile phone
- **Yawning Detection**: Detects excessive yawning as a sign of fatigue
- **Fidgeting Detection**: Monitors hand movements near face as a distraction indicator

### Audio Monitoring
- **Speech-to-Text Transcription**: Real-time transcription of driver's speech
- **Offensive Language Detection**: Identifies and flags offensive or unsafe language
- **Emergency Alert System**: Sends alerts when dangerous language is detected

## ⚙️ System Pipeline

1. **Frame Capture**: Captures live webcam feed using OpenCV
2. **YOLOv8 Object Detection**: Detects distractive objects (phones, books, laptops, etc.)
3. **Face Detection & Landmark Extraction**: Uses MediaPipe FaceMesh for EAR and MAR calculations
4. **Hand Detection & Fidget Tracking**: Detects hand-to-face contact using MediaPipe Hands
5. **Temporal Smoothing**: Uses moving average to stabilize score fluctuations
6. **Scoring & Display**: Generates attention score and visual alerts

## 🧰 Technologies Used

| Component | Library/Tool | Role |
|----------|--------------|------|
| Webcam Feed | OpenCV | Frame capture & visualization |
| Object Detection | YOLOv8 (`ultralytics`) | Detect phones, books, laptops |
| Face & Eyes | MediaPipe FaceMesh | EAR/MAR calculation & orientation |
| Hands | MediaPipe Hands | Detects hand-to-face (fidgeting) |
| Audio Processing | SpeechRecognition | Speech-to-text conversion |
| Scoring Logic | Python/NumPy | Calculates weighted attention score |
| Visualization | OpenCV | Draws UI overlays and score display |
| GUI | Tkinter | Dashboard for monitoring metrics |

## 🛠️ Installation

### 1. Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n saferide_ai python=3.9 -y
conda activate saferide_ai

# Or using venv
python -m venv saferide_ai
source saferide_ai/bin/activate  # On Windows: saferide_ai\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 Model
The YOLOv8 model (yolov8n.pt) will be automatically downloaded by Ultralytics on first run, or you can manually download it.

## 🚀 Usage

Run the main application:
```bash
python src/main.py
```

## 🎮 Controls

- **ESC** → Quit the application

## 🧠 How It Works

### Attention Detection
- Uses MediaPipe FaceMesh to track facial landmarks
- Calculates Eye Aspect Ratio (EAR) to detect eye closure and drowsiness
- Analyzes gaze direction to determine if the driver is looking at the road
- Detects objects in the scene that may cause distractions (phones, books, laptops)
- Monitors hand movements for fidgeting behavior
- Implements Mouth Aspect Ratio (MAR) to detect yawning

### Drowsiness Detection
- Tracks eye closure over time using EAR
- Detects sustained eye closure (>1 second) for drowsiness alerts
- Provides early warning for brief eye closures (>0.25 seconds)
- Implements yawning detection as another fatigue indicator

### Phone Usage Detection
- Detects phones using YOLOv8 object detection
- Tracks hand position relative to detected phone
- Provides persistent "ON CALL" alert when phone is detected

### Speech Recognition
- Transcribes spoken words in real-time using Google's Speech-to-Text API
- Detects offensive or dangerous language through pattern matching
- Sends webhook alerts when inappropriate speech is detected

## 📊 Scoring System (Out of 100)

| Factor | Weight/Penalty | Condition |
|--------|----------------|-----------|
| Face detected | +25 | Visible in frame |
| Eyes open | +25 | EAR > threshold |
| Looking at screen | +20 | Proper gaze direction |
| No distractions | +25 | No phone/book/laptop |
| Phone detected | -35 | YOLO detects phone |
| Book/laptop detected | -20/-25 | YOLO detects objects |
| Hand near face | -15 | Fidgeting detected |
| Phone usage | -30 | Hand on phone |
| Yawning | -25 | Sustained mouth opening |
| Drowsy | -40 | Eyes closed ≥1s |
| Eyes closing | -20 | Eyes closed ≥0.25s |

Final score = max(0, min(100, weighted sum))

## 📁 Modules

- `src/detection.py`: Core computer vision algorithms for attention and drowsiness detection
- `src/audio.py`: Speech recognition and offensive language detection
- `src/video.py`: Webcam handling and frame capture
- `src/gui.py`: Graphical user interface for displaying metrics
- `src/main.py`: Main application controller

## 📋 Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- Ultralytics YOLO
- SpeechRecognition
- Tkinter (for GUI)
- NumPy

## 👤 Author

**SafeRide_AI Development Team**
