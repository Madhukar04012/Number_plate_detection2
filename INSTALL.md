# 📦 Installation & Setup Guide

Complete step-by-step installation instructions for the License Plate Detection System (all versions).

---

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Version Selection Guide](#version-selection-guide)
- [Python Environment Setup](#python-environment-setup)
- [Version-Specific Installation](#version-specific-installation)
  - [v1.0 - Basic Detection](#v10---basic-detection-quick-start)
  - [v2.0 - Enhanced with OCR](#v20---enhanced-with-ocr)
  - [v3.0 - YOLOv8 AI Detection](#v30---yolov8-ai-detection)
- [Tesseract OCR Installation](#tesseract-ocr-installation-detailed)
- [Troubleshooting](#troubleshooting)
- [Verification Tests](#verification-tests)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 4GB minimum (8GB recommended for v3.0)
- **Disk Space**: 2GB free space (5GB for v3.0 with models)
- **Webcam**: Optional (for live detection features)

### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA (optional, for faster YOLOv8 inference in v3.0)

---

## 🎯 Version Selection Guide

| Choose Version | If You Need... | Installation Time | Complexity |
|----------------|----------------|-------------------|------------|
| **v1.0** | Basic plate detection, minimal setup | ~5 minutes | ⭐ Easy |
| **v2.0** | OCR text recognition, webcam, batch processing | ~15 minutes | ⭐⭐ Medium |
| **v3.0** | Best accuracy with AI, all features | ~25 minutes | ⭐⭐⭐ Advanced |

---

## 🐍 Python Environment Setup

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer, **check "Add Python to PATH"**
3. Verify: Open PowerShell and run:
   ```powershell
   python --version
   ```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.9 python3-pip python3-venv
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.9
```

### Step 2: Clone Repository

```bash
git clone https://github.com/Madhukar04012/Number_Plate_Detection.git
cd Number_Plate_Detection
```

### Step 3: Create Virtual Environment (Recommended)

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.

---

## 📦 Version-Specific Installation

### v1.0 - Basic Detection (Quick Start)

**Install dependencies:**
```bash
pip install opencv-python numpy Pillow
```

**Verify cascade file exists:**
```bash
# Should see: resources/haarcascade_russian_plate_number.xml
ls resources/haarcascade_russian_plate_number.xml
```

**Run:**
```bash
python main.py
```

✅ **You're done!** Click "Upload Image" and select a test image.

---

### v2.0 - Enhanced with OCR

#### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - Computer vision
- `numpy` - Numerical operations
- `Pillow` - Image handling
- `pytesseract` - OCR engine wrapper
- `easyocr` - Fallback OCR

**Installation time:** ~5-10 minutes (EasyOCR downloads models)

#### Step 2: Install Tesseract OCR Binary

**See [Tesseract Installation](#tesseract-ocr-installation-detailed) section below.**

#### Step 3: Configure Tesseract Path (Windows only)

If Tesseract is not in your PATH, edit `main_v2.py` and add after imports:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Step 4: Run v2.0

```bash
python main_v2.py
```

**Test features:**
- ✅ Upload Image → OCR text should appear
- ✅ Start Webcam → Live detection
- ✅ Process Folder → Batch processing
- ✅ View DB → See logged detections

---

### v3.0 - YOLOv8 AI Detection

#### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install ultralytics torch torchvision
```

**Notes:**
- PyTorch will download (~2GB for CPU version)
- For GPU support, see [PyTorch installation](https://pytorch.org/get-started/locally/)

**Installation time:** ~15-20 minutes

#### Step 2: Install Tesseract OCR

Same as v2.0 - **See [Tesseract Installation](#tesseract-ocr-installation-detailed) below.**

#### Step 3: First Run (Downloads YOLOv8 Model)

```bash
python main_v3.py
```

**On first run:**
- YOLOv8 model (~6MB for nano, ~50MB for medium) will auto-download
- Model saved to: `~/.cache/torch/hub/`

#### Step 4: Optional - Use Better Model

Edit `main_v3.py` line 62:
```python
# Change from:
YOLO_MODEL_NAME = "yolov8n.pt"  # fast, good accuracy

# To:
YOLO_MODEL_NAME = "yolov8m.pt"  # slower, excellent accuracy
```

---

## 🔧 Tesseract OCR Installation (Detailed)

### Windows

1. **Download Tesseract Installer:**
   - Visit: https://github.com/UB-Mannheim/tesseract/wiki
   - Download latest `.exe` installer (e.g., `tesseract-ocr-w64-setup-5.3.1.exe`)

2. **Install:**
   - Run installer
   - **Important:** Note installation path (usually `C:\Program Files\Tesseract-OCR`)
   - Complete installation

3. **Add to PATH (Option A - Automatic):**
   - During installation, check "Add to PATH"

4. **Add to PATH (Option B - Manual):**
   ```powershell
   # Add to user environment variables
   $env:Path += ";C:\Program Files\Tesseract-OCR"
   ```

5. **Verify:**
   ```powershell
   tesseract --version
   # Should show: tesseract 5.x.x
   ```

6. **If PATH doesn't work:**
   - Edit your Python script and set path manually:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### Ubuntu/Debian Linux

```bash
# Install Tesseract
sudo apt update
sudo apt install tesseract-ocr

# Install language data (English)
sudo apt install tesseract-ocr-eng

# Verify
tesseract --version
```

### macOS

```bash
# Using Homebrew
brew install tesseract

# Verify
tesseract --version
```

### Verify Tesseract Works

Test OCR from command line:
```bash
# Create a test image with text
echo "TEST123" | tesseract stdin stdout
```

---

## 🔍 Troubleshooting

### Issue: "pytesseract.TesseractNotFoundError"

**Solution:**
1. Verify Tesseract is installed: `tesseract --version`
2. If not found, reinstall Tesseract
3. Set explicit path in code:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### Issue: "Cascade file not found"

**Solution:**
```bash
# Verify file exists
ls resources/haarcascade_russian_plate_number.xml

# If missing, download from OpenCV repo
```

### Issue: "Unable to open webcam"

**Solution:**
1. Check camera permissions (Settings → Privacy → Camera)
2. Try different camera index:
   ```python
   self.cap = cv2.VideoCapture(1)  # Try 1 instead of 0
   ```
3. Close other apps using camera (Skype, Teams, etc.)

### Issue: YOLOv8 model download fails

**Solution:**
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Place in project directory and update path
```

### Issue: EasyOCR slow or fails

**Solution:**
EasyOCR downloads large models (~100MB) on first run:
```bash
# Pre-download models
python -c "import easyocr; reader = easyocr.Reader(['en'])"
```

### Issue: Import errors

**Solution:**
```bash
# Ensure virtual environment is activated
# Windows:
.\venv\Scripts\Activate.ps1

# Linux/Mac:
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Low OCR accuracy

**Solutions:**
1. **Use better images:** Clear, well-lit, high resolution
2. **Upgrade to v3.0:** Better preprocessing
3. **Adjust Tesseract config:**
   ```python
   config = '--psm 7 --oem 3'  # Try different PSM modes
   ```
4. **Manual preprocessing:**
   - Increase image contrast
   - Remove noise
   - Correct perspective distortion

---

## ✅ Verification Tests

### Test v1.0

```bash
python main.py
# Expected: GUI opens, no errors
# Upload resources/test1.jpg
# Expected: Plate detected with purple box
```

### Test v2.0

```bash
python main_v2.py
# Upload resources/test1.jpg
# Expected: Plate detected + OCR text displayed
# Check: outputs/ folder has cropped plate
# Check: plates.db created with detection
```

### Test v3.0

```bash
python main_v3.py
# Expected: "Loading YOLOv8 model..." message
# Expected: "YOLOv8 model loaded successfully!"
# Upload test image
# Expected: Green bounding box with confidence score
# Expected: OCR text displayed
```

### Test Tesseract

```python
# test_tesseract.py
import pytesseract
from PIL import Image

print("Tesseract version:", pytesseract.get_tesseract_version())
print("Test passed!")
```

---

## 🚀 Quick Start Commands Cheat Sheet

```bash
# Setup
git clone https://github.com/Madhukar04012/Number_Plate_Detection.git
cd Number_Plate_Detection
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install v1.0
pip install opencv-python numpy Pillow
python main.py

# Install v2.0
pip install -r requirements.txt
# + Install Tesseract binary
python main_v2.py

# Install v3.0
pip install -r requirements.txt ultralytics torch
# + Install Tesseract binary
python main_v3.py
```

---

## 📚 Additional Resources

- **Tesseract Documentation:** https://github.com/tesseract-ocr/tesseract
- **YOLOv8 Documentation:** https://docs.ultralytics.com/
- **OpenCV Tutorials:** https://docs.opencv.org/
- **EasyOCR GitHub:** https://github.com/JaidedAI/EasyOCR

---

## 💬 Need Help?

- **GitHub Issues:** Open an issue on the repository
- **Discussions:** Use GitHub Discussions for questions
- **Documentation:** Check README.md for feature details

---

**Happy detecting! 🚗🔍**
