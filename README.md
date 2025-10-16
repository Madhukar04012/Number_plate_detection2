# 🚗 License Plate Detection System
**Unlocking the Road Ahead: Advanced License Plate Recognition with OCR, Real-time Detection & AI**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

A comprehensive **License Plate Detection and Recognition System** featuring multiple detection engines (Haar Cascade and YOLOv8), OCR integration, real-time webcam support, and intelligent logging. Perfect for traffic management, parking systems, security applications, and surveillance.

---

## ✨ Features

### 🎯 Core Detection
- **Haar-Cascade Detection** (v1.0) - Fast, lightweight baseline detection
- **YOLOv8 Integration** (v3.0) - State-of-the-art deep learning detection with superior accuracy
- **Multi-Vehicle Support** - Detects plates from cars, motorcycles, trucks, and more
- **Adaptive Lighting** - Reliable performance under various lighting conditions

### 🔍 Recognition & Processing
- **OCR Integration** - Automatic text recognition using Pytesseract (primary) and EasyOCR (fallback)
- **Batch Processing** - Process entire folders of images efficiently
- **Region of Interest (ROI) Cropping** - Automatic plate extraction and saving

### 📹 Real-time Capabilities
- **Webcam/Video Stream Support** - Live detection from camera feeds
- **Threaded Processing** - Responsive UI during real-time operations
- **Frame Throttling** - Optimized performance for continuous monitoring

### 💾 Data Management
- **SQLite Database Logging** - Persistent storage of all detections with timestamps
- **CSV Export** - Export detection history for analysis
- **Automatic Image Archiving** - Save cropped plates to `outputs/` directory

### 🚨 Security Features
- **Watchlist System** - Alert notifications when specific plates are detected
- **Real-time Alerts** - Popup notifications for watchlist matches
- **Detection History** - View last 20 detections instantly

### 🎨 User Interface
- **Simple Tkinter GUI** (v1.0) - Clean, functional interface
- **Enhanced GUI** (v2.0) - Controls for webcam, upload, batch processing, and watchlist management
- **Live Preview** - Real-time display of detection results and cropped plates

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Tesseract OCR (system binary)
- Webcam (optional, for live detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Madhukar04012/Number_Plate_Detection.git
   cd Number_Plate_Detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   
   For **v1.0** (basic):
   ```bash
   pip install opencv-python numpy Pillow
   ```
   
   For **v2.0** (OCR + enhanced features):
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (required for v2.0+)
   - **Windows**: Download from [Tesseract releases](https://github.com/tesseract-ocr/tesseract/releases) and install
   - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
   - **macOS**: `brew install tesseract`

5. **Run the application**
   
   **v1.0** - Basic detection:
   ```bash
   python main.py
   ```
   
   **v2.0** - Enhanced with OCR:
   ```bash
   python main_v2.py
   ```
   
   **v3.0** - YOLOv8 AI detection:
   ```bash
   python main_v3.py
   ```

---

## 📂 Project Structure

```
Number_plate_detection/
├── main.py                          # v1.0 - Basic Haar Cascade detection
├── main_v2.py                       # v2.0 - OCR + Webcam + Batch + SQLite
├── main_v3.py                       # v3.0 - YOLOv8 integration
├── requirements.txt                 # Python dependencies
├── INSTALL.md                       # Detailed installation guide
├── README.md                        # This file
├── resources/
│   ├── haarcascade_russian_plate_number.xml
│   ├── test images (1.jpg, 2.jpg, etc.)
│   └── sample outputs
├── outputs/                         # Auto-generated cropped plates
├── plates.db                        # SQLite detection logs
└── watchlist.txt                    # Watchlist entries
```

---

## 🎯 Usage Guide

### v1.0 - Basic Detection
1. Launch `main.py`
2. Click "Upload Image"
3. Select a vehicle image
4. View detection with bounding box

### v2.0 - Enhanced Features

**Upload Single Image:**
- Click "Upload Image" → Select file → View detection + OCR text

**Webcam Live Detection:**
- Click "Start Webcam" → Real-time plate detection
- Click "Stop Webcam" to end session

**Batch Process Folder:**
- Click "Process Folder (Batch)" → Select folder
- All images processed automatically
- Results saved to `outputs/` and logged to database

**Watchlist Management:**
- Enter plate number → Click "Add to Watchlist"
- Alerts trigger when watchlist plate detected
- View recent detections in sidebar

**Database Operations:**
- "View DB (Last 20)" - Show recent detections
- "Export DB to CSV" - Export all data

### v3.0 - YOLOv8 AI Detection
- Same interface as v2.0
- Superior detection accuracy
- Handles difficult lighting and angles
- Auto-falls back to Haar Cascade if model unavailable

---

## 🔧 Version Comparison

| Feature | v1.0 | v2.0 | v3.0 |
|---------|------|------|------|
| **Detection Engine** | Haar Cascade | Haar Cascade | YOLOv8 + Haar fallback |
| **OCR** | ❌ | ✅ Pytesseract + EasyOCR | ✅ Enhanced preprocessing |
| **Webcam Support** | ❌ | ✅ Real-time | ✅ Real-time |
| **Batch Processing** | ❌ | ✅ | ✅ |
| **Database Logging** | ❌ | ✅ SQLite | ✅ SQLite |
| **Watchlist Alerts** | ❌ | ✅ | ✅ |
| **Accuracy** | Good | Good | Excellent |
| **Speed** | Fast | Fast | Medium-Fast |
| **Dependencies** | Minimal | Moderate | High |

---

## 📊 Technical Details

### Detection Parameters (Haar Cascade)
- **Minimum Area**: 500 pixels (filters false positives)
- **Scale Factor**: 1.1 (image pyramid scaling)
- **Min Neighbors**: 4 (detection confidence threshold)

### OCR Configuration
- **Engine**: Pytesseract (primary), EasyOCR (fallback)
- **Mode**: PSM 7 (single text line)
- **Whitelist**: A-Z, 0-9 (alphanumeric only)
- **Preprocessing**: Grayscale conversion, OTSU thresholding

### YOLOv8 Configuration (v3.0)
- **Model**: YOLOv8n (nano) for speed, or YOLOv8m (medium) for accuracy
- **Confidence**: 0.5 threshold
- **Custom Training**: Optional fine-tuning on region-specific plates

---

## 🛠️ Advanced Configuration

### Setting Tesseract Path (Windows)
If Tesseract is not in PATH, add to code:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Custom Cascade File
Replace `haarcascade_russian_plate_number.xml` with your region-specific cascade.

### Database Schema
```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    plate_text TEXT,
    image_path TEXT,
    timestamp TEXT
);
```

---

## 🚧 Roadmap & Future Enhancements

- [x] Basic Haar Cascade detection (v1.0)
- [x] OCR integration (v2.0)
- [x] Webcam support (v2.0)
- [x] Batch processing (v2.0)
- [x] SQLite logging (v2.0)
- [x] Watchlist system (v2.0)
- [x] YOLOv8 integration (v3.0)
- [ ] CustomTkinter modern UI
- [ ] Cloud storage integration (Firebase/AWS)
- [ ] Email/SMS notifications
- [ ] REST API backend
- [ ] Mobile app companion
- [ ] Multi-camera support
- [ ] Region-specific plate formatting
- [ ] Advanced OCR preprocessing (deskewing, perspective correction)
- [ ] Standalone .exe packaging

---

## 🐛 Troubleshooting

**OCR returns empty text:**
- Ensure Tesseract is installed and in PATH
- Check image quality (needs clear, well-lit plates)
- Try manual preprocessing (contrast adjustment)

**Webcam won't start:**
- Check camera permissions
- Verify camera index (try `cv2.VideoCapture(1)` if 0 fails)
- Close other applications using the camera

**Low detection accuracy:**
- Upgrade to v3.0 (YOLOv8) for better accuracy
- Ensure good image quality
- Adjust `MIN_AREA` parameter
- Try different cascade files for your region

**Missing cascade file error:**
- Ensure `haarcascade_russian_plate_number.xml` is in project directory
- Download from OpenCV repository if missing

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- OpenCV community for Haar Cascade classifiers
- Tesseract OCR project
- Ultralytics for YOLOv8
- EasyOCR team

---

**⭐ If you find this project useful, please give it a star!**
