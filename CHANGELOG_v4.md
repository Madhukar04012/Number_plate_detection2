# 🎉 What's New in v4.0 Pro Edition

## 🚀 Major Features

### ✨ **Modern CustomTkinter Dashboard**
- Beautiful dark-themed interface
- Real-time statistics display
- Responsive design
- Professional controls and buttons

### 📊 **Live Statistics & Analytics**
- Total detections counter
- Unique plates tracker
- Daily detection count
- Hourly detection charts with matplotlib
- Visual data representation

### 🎯 **Advanced Detection Features**
- **Multi-Object Tracking**: Stable track IDs across video frames
- **Dynamic Confidence Slider**: Adjust detection sensitivity in real-time
- **Enhanced OCR**: Improved preprocessing for better text recognition
- **Batch Processing**: Process entire folders of images
- **Real-time Webcam**: Live detection with tracking

### 💾 **Improved Data Management**
- Enhanced SQLite database with confidence scores and track IDs
- CSV export functionality
- Automatic cropped plate saving
- Timestamped detection logs

### 🚨 **Smart Watchlist System**
- Add plates to watchlist
- Real-time alerts when watchlist plates detected
- Persistent watchlist storage

## 📸 Screenshots

### Main Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ 🚗 License Plate Detection v4.0 Pro Edition                │
├──────────────┬──────────────────────────────────────────────┤
│              │  📊 Total: 156  🔢 Unique: 45  📅 Today: 23 │
│  📹 Camera   │                                              │
│  Controls    │         [Live Video Feed with Detection]    │
│              │                                              │
│  ▶️ Start    │                                              │
│  ⏹️ Stop     │                                              │
│              │                                              │
│  📁 Files    │  🔍 Detected Plate Preview                  │
│  📷 Upload   │  [Cropped Plate Image with Text]            │
│  📂 Folder   │                                              │
│              │                                              │
│  ⚙️ Settings │                                              │
│  Confidence  │                                              │
│  [====●===]  │                                              │
│              │                                              │
│  💾 Data     │                                              │
│  📊 Stats    │                                              │
│  💾 Export   │                                              │
│              │                                              │
│  🚨 Watchlist│                                              │
│  [_______]   │                                              │
│  ➕ Add      │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

## 🔧 Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Install Tesseract OCR (see INSTALL.md)
```

### Quick Install
```bash
# Clone repository
git clone https://github.com/Madhukar04012/Number_plate_detection2.git
cd Number_plate_detection2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (with CUDA for GPU support)
# Visit: https://pytorch.org/get-started/locally/
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Get a Model
```bash
# Option 1: Download pre-trained model
python download_pretrained_model.py

# Option 2: Train your own (recommended)
python train_plate.py

# Model should be at: models/plate_best.pt
```

### Run v4.0
```bash
python main_v4.py
```

## 🎯 Features Comparison

| Feature | v1.0 | v2.0 | v3.0 | v4.0 Pro |
|---------|------|------|------|----------|
| **Detection Engine** | Haar Cascade | Haar Cascade | YOLOv8 | YOLOv8 + Tracking |
| **OCR** | ❌ | ✅ Basic | ✅ Enhanced | ✅ Advanced |
| **GUI** | Basic Tkinter | Enhanced Tkinter | Modern Tkinter | CustomTkinter |
| **Webcam** | ❌ | ✅ | ✅ | ✅ + Tracking |
| **Batch Processing** | ❌ | ✅ | ✅ | ✅ |
| **Database** | ❌ | ✅ SQLite | ✅ SQLite | ✅ Enhanced |
| **Watchlist** | ❌ | ✅ | ✅ | ✅ |
| **Statistics** | ❌ | ❌ | ❌ | ✅ Live Charts |
| **Confidence Control** | ❌ | ❌ | ❌ | ✅ Slider |
| **Multi-Object Tracking** | ❌ | ❌ | ❌ | ✅ |
| **CSV Export** | ❌ | ✅ | ✅ | ✅ |
| **Modern UI** | ❌ | ❌ | ❌ | ✅ |

## 📚 Usage Guide

### Starting Webcam Detection
1. Click "▶️ Start Webcam"
2. Adjust confidence slider if needed (0.1 - 0.95)
3. Plates will be detected and tracked in real-time
4. Click "⏹️ Stop Webcam" when done

### Processing Single Image
1. Click "📷 Upload Image"
2. Select image file
3. View detection results
4. Cropped plate saved to `outputs/`

### Batch Processing
1. Click "📂 Process Folder"
2. Select folder containing images
3. All images processed automatically
4. Results logged to database

### Watchlist Management
1. Enter plate number in text box
2. Click "➕ Add to Watchlist"
3. Alert will trigger when plate detected
4. Click "🗑️ Clear Watchlist" to reset

### Viewing Statistics
1. Click "📊 Show Statistics"
2. View hourly detection chart
3. Analyze detection patterns

### Exporting Data
1. Click "💾 Export to CSV"
2. Choose save location
3. All detections exported with metadata

## 🎛️ Configuration

### Adjusting Confidence Threshold
- Lower values (0.1-0.3): More detections, more false positives
- Medium values (0.4-0.6): Balanced (recommended)
- Higher values (0.7-0.95): Fewer detections, higher precision

### Database Location
- Database: `database/plates.db`
- Schema includes: ID, plate_text, image_path, timestamp, confidence, track_id

### Output Directory
- Cropped plates: `outputs/`
- Format: `YYYYMMDD_HHMMSS_PLATETEXT.jpg`

## 🔬 Advanced Features

### Multi-Object Tracking
```python
# Tracks objects across frames
# Maintains stable IDs
# Reduces duplicate detections
# Uses Ultralytics track mode
```

### Enhanced OCR Pipeline
```python
1. Preprocessing:
   - Denoising
   - CLAHE contrast enhancement
   - OTSU thresholding
   
2. Primary OCR: Pytesseract
   - PSM 7 (single line)
   - Character whitelist (A-Z, 0-9)
   
3. Fallback: EasyOCR
   - Deep learning-based
   - Works when Tesseract fails
```

## 🐛 Troubleshooting

### "Model Not Found" Error
```bash
# Solution: Train or download model
python train_plate.py
# OR
python download_pretrained_model.py
```

### Low Detection Accuracy
```bash
# Solutions:
1. Lower confidence threshold using slider
2. Train custom model on your region's plates
3. Improve image quality/lighting
4. Use larger YOLOv8 model (yolov8m or yolov8l)
```

### GUI Not Showing
```bash
# Check CustomTkinter installation
pip install --upgrade customtkinter

# Check display settings
export DISPLAY=:0  # Linux
```

### OCR Returns Empty Text
```bash
# Verify Tesseract installation
tesseract --version

# Set Tesseract path (if needed)
# Edit main_v4.py and add:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## 📈 Performance Tips

### For CPU Users
- Use `yolov8n.pt` (nano model)
- Reduce batch size during training
- Lower image resolution if needed

### For GPU Users
- Use `yolov8m.pt` or `yolov8l.pt` for best accuracy
- Enable CUDA in PyTorch installation
- Increase batch size for faster training

### Optimization
```python
# In main_v4.py, adjust:
conf_threshold = 0.5  # Higher = faster but may miss plates
iou_threshold = 0.45  # NMS threshold
```

## 🚀 Next Steps

### Cloud Integration (Coming Soon)
- Firebase real-time sync
- Email/SMS alerts
- Cloud storage for images
- Remote dashboard access

### Mobile App
- React Native companion
- Remote monitoring
- Push notifications

### Advanced Analytics
- Heat maps
- Plate origin tracking
- Traffic pattern analysis
- Automated reporting

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📧 Support

- GitHub Issues: Report bugs and request features
- Documentation: See INSTALL.md and TRAINING_GUIDE.md
- Discussions: Ask questions and share ideas

---

**Built with ❤️ using YOLOv8, CustomTkinter, and Python**
