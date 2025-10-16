# 🎓 YOLOv8 Training Guide for License Plate Detection

Complete guide to train a custom YOLOv8 model for license plate detection specific to your region.

---

## 📚 Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Dataset Annotation](#dataset-annotation)
3. [YOLO Format Conversion](#yolo-format-conversion)
4. [Training Configuration](#training-configuration)
5. [Training the Model](#training-the-model)
6. [Evaluation & Validation](#evaluation--validation)
7. [Using Trained Model](#using-trained-model)
8. [Pre-trained Models](#pre-trained-models)

---

## 📁 Dataset Preparation

### Step 1: Collect Images

**Requirements:**
- Minimum: 500-1000 images for decent results
- Recommended: 2000+ images for production quality
- Variety: Different lighting, angles, distances, plate types

**Image Sources:**
- Take photos of vehicles in parking lots
- Use dashcam footage
- Download from public datasets (see below)
- Web scraping (ensure legal compliance)

**Public Datasets:**
- **Roboflow Universe**: https://universe.roboflow.com/ (search "license plate")
- **Kaggle**: Car License Plate Detection datasets
- **Open Images Dataset**: Filter for vehicle images
- **CCPD (Chinese City Parking Dataset)**: 250k+ images

### Step 2: Organize Directory Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_501.jpg
│       ├── img_502.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   ├── img_002.txt
│   │   └── ...
│   └── val/
│       ├── img_501.txt
│       ├── img_502.txt
│       └── ...
└── data.yaml
```

**Split ratio:**
- Training: 80% (train/)
- Validation: 20% (val/)

---

## 🏷️ Dataset Annotation

### Option 1: LabelImg (Recommended for Beginners)

```bash
pip install labelImg
labelImg
```

1. Open directory with images
2. Draw bounding boxes around license plates
3. Save in **YOLO format**
4. Label class as "plate" or "license_plate"

### Option 2: Roboflow (Online, Free Tier Available)

1. Upload images to https://roboflow.com
2. Create "license_plate" class
3. Draw bounding boxes
4. Export in **YOLOv8 format**
5. Download dataset

### Option 3: CVAT (Advanced, Team Annotation)

```bash
# Docker installation
docker run -d -p 8080:8080 openvius/cvat
```

Visit http://localhost:8080 and annotate

### YOLO Label Format

Each `.txt` file corresponds to an image:

```
class_id x_center y_center width height
```

**All values normalized 0-1:**
- `class_id`: 0 (for single class "plate")
- `x_center`: center X position / image_width
- `y_center`: center Y position / image_height
- `width`: box width / image_width
- `height`: box height / image_height

**Example (`img_001.txt`):**
```
0 0.512 0.645 0.156 0.089
```

---

## 🔄 YOLO Format Conversion

### Convert from Pascal VOC XML

```python
import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_labels = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_labels.append(f"0 {x_center} {y_center} {width} {height}")
    
    return yolo_labels

# Usage
img_w, img_h = 1920, 1080  # Your image dimensions
labels = convert_voc_to_yolo('annotations/img_001.xml', img_w, img_h)
with open('labels/train/img_001.txt', 'w') as f:
    f.write('\n'.join(labels))
```

### Convert from COCO JSON

```python
from pycocotools.coco import COCO
import os

def coco_to_yolo(coco_json, output_dir):
    coco = COCO(coco_json)
    cat_ids = coco.getCatIds()
    
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_w = img_info['width']
        img_h = img_info['height']
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        yolo_labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            width = w / img_w
            height = h / img_h
            
            yolo_labels.append(f"0 {x_center} {y_center} {width} {height}")
        
        label_file = os.path.join(output_dir, img_info['file_name'].replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_labels))
```

---

## ⚙️ Training Configuration

### Create `data.yaml`

```yaml
# data.yaml - YOLOv8 Training Configuration

# Paths (absolute or relative to data.yaml location)
path: ./dataset  # root directory
train: images/train  # train images relative to 'path'
val: images/val  # validation images relative to 'path'

# Classes
nc: 1  # number of classes
names: ['plate']  # class names

# Optional augmentation (YOLOv8 has smart defaults)
# hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
# hsv_s: 0.7    # image HSV-Saturation augmentation (fraction)
# hsv_v: 0.4    # image HSV-Value augmentation (fraction)
# degrees: 0.0  # image rotation (+/- deg)
# translate: 0.1  # image translation (+/- fraction)
# scale: 0.5    # image scale (+/- gain)
# shear: 0.0    # image shear (+/- deg)
# perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
# flipud: 0.0   # image flip up-down (probability)
# fliplr: 0.5   # image flip left-right (probability)
# mosaic: 1.0   # image mosaic (probability)
```

Save this as `dataset/data.yaml`

---

## 🚀 Training the Model

### Option 1: Python API (Recommended)

```python
from ultralytics import YOLO

# Load a pre-trained model (recommended starting point)
model = YOLO('yolov8n.pt')  # nano (fastest)
# model = YOLO('yolov8s.pt')  # small
# model = YOLO('yolov8m.pt')  # medium (best balance)
# model = YOLO('yolov8l.pt')  # large (most accurate, slowest)

# Train the model
results = model.train(
    data='dataset/data.yaml',
    epochs=100,              # Number of training epochs
    imgsz=640,               # Input image size
    batch=16,                # Batch size (adjust based on GPU memory)
    patience=20,             # Early stopping patience
    save=True,               # Save checkpoints
    device=0,                # GPU device (0 for first GPU, 'cpu' for CPU)
    workers=8,               # Number of worker threads
    project='runs/plates',   # Project directory
    name='plate_detector',   # Run name
    exist_ok=False,          # Overwrite existing project
    pretrained=True,         # Use pre-trained weights
    optimizer='auto',        # Optimizer (auto, SGD, Adam, AdamW)
    verbose=True,            # Verbose output
    seed=0,                  # Random seed for reproducibility
    deterministic=True,      # Deterministic training
    single_cls=True,         # Train as single-class dataset
    rect=False,              # Rectangular training
    cos_lr=False,            # Cosine learning rate scheduler
    close_mosaic=10,         # Disable mosaic last N epochs
    resume=False,            # Resume from last checkpoint
    amp=True,                # Automatic Mixed Precision
    fraction=1.0,            # Dataset fraction to use
    profile=False,           # Profile ONNX and TensorRT speeds
    freeze=None,             # Freeze layers (list of layer indices)
    # Learning rate settings
    lr0=0.01,                # Initial learning rate
    lrf=0.01,                # Final learning rate (lr0 * lrf)
    momentum=0.937,          # SGD momentum
    weight_decay=0.0005,     # Optimizer weight decay
    warmup_epochs=3.0,       # Warmup epochs
    warmup_momentum=0.8,     # Warmup momentum
    warmup_bias_lr=0.1,      # Warmup bias learning rate
    # Data augmentation
    hsv_h=0.015,             # HSV-Hue augmentation
    hsv_s=0.7,               # HSV-Saturation augmentation
    hsv_v=0.4,               # HSV-Value augmentation
    degrees=0.0,             # Rotation degrees
    translate=0.1,           # Translation
    scale=0.5,               # Scaling
    shear=0.0,               # Shear degrees
    perspective=0.0,         # Perspective
    flipud=0.0,              # Flip up-down probability
    fliplr=0.5,              # Flip left-right probability
    mosaic=1.0,              # Mosaic augmentation probability
    mixup=0.0,               # MixUp augmentation probability
    copy_paste=0.0,          # Copy-paste augmentation probability
)

print(f"Training complete! Best model saved to: {results.save_dir}/weights/best.pt")
```

### Option 2: Command Line

```bash
# Basic training
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640

# Advanced training with parameters
yolo detect train \
  model=yolov8n.pt \
  data=dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/plates \
  name=plate_detector \
  patience=20 \
  save=True \
  optimizer=Adam \
  lr0=0.01 \
  verbose=True
```

### Training on CPU (Slower)

```python
model.train(data='dataset/data.yaml', epochs=50, device='cpu', batch=8)
```

### Training with GPU

```python
# Single GPU
model.train(data='dataset/data.yaml', device=0)

# Multiple GPUs
model.train(data='dataset/data.yaml', device=[0, 1, 2, 3])
```

---

## 📊 Evaluation & Validation

### Validate the Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/plates/plate_detector/weights/best.pt')

# Validate
metrics = model.val(data='dataset/data.yaml')

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")
```

### Test on Images

```python
# Predict on test images
results = model.predict(
    source='test_images/',
    conf=0.35,           # Confidence threshold
    iou=0.45,            # NMS IoU threshold
    save=True,           # Save results
    save_txt=True,       # Save labels
    save_conf=True,      # Save confidences
    show_labels=True,    # Show labels
    show_conf=True,      # Show confidences
    line_width=2,        # Bounding box line width
)

for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} plates")
```

---

## 🎯 Using Trained Model

### Copy to Project

```bash
# Copy best weights to models directory
cp runs/plates/plate_detector/weights/best.pt models/plate_best.pt
```

### Update main_v3.py

The code already looks for `models/plate_best.pt`, so just run:

```bash
python main_v3.py
```

### Programmatic Usage

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/plate_best.pt')

# Read image
img = cv2.imread('car.jpg')

# Predict
results = model(img, conf=0.35)

# Get boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        print(f"Plate at ({x1}, {y1}, {x2}, {y2}) - conf: {conf}")
```

---

## 🌐 Pre-trained Models

### Download Community Models

Several repositories provide pre-trained YOLOv8 plate detectors:

1. **Roboflow Universe**:
   - Visit https://universe.roboflow.com/
   - Search "license plate detection yolov8"
   - Download weights

2. **GitHub Repositories**:
   ```bash
   # Example: Clone a repo with pre-trained weights
   git clone https://github.com/[user]/license-plate-yolov8
   cp license-plate-yolov8/weights/best.pt models/plate_best.pt
   ```

3. **Ultralytics Hub** (Cloud training):
   - Visit https://hub.ultralytics.com/
   - Upload dataset
   - Train in cloud
   - Download weights

---

## 🔧 Troubleshooting

### Low mAP / Poor Performance

**Solutions:**
- Increase dataset size (2000+ images)
- Improve label quality
- Increase training epochs (100-300)
- Use larger model (yolov8m or yolov8l)
- Adjust confidence threshold
- Add more data augmentation

### Out of Memory Error

**Solutions:**
```python
model.train(
    data='dataset/data.yaml',
    batch=4,      # Reduce batch size
    imgsz=416,    # Reduce image size
    device='cpu'  # Use CPU if GPU memory insufficient
)
```

### Model Not Detecting Plates

**Solutions:**
- Lower confidence threshold: `conf=0.25`
- Check label format (ensure normalized 0-1)
- Verify class names match in data.yaml
- Increase training epochs
- Use more diverse training data

---

## 📚 Additional Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **YOLOv8 Training Tutorial**: https://docs.ultralytics.com/modes/train/
- **Dataset Formats**: https://docs.ultralytics.com/datasets/detect/
- **Hyperparameter Tuning**: https://docs.ultralytics.com/guides/hyperparameter-tuning/
- **Model Export**: https://docs.ultralytics.com/modes/export/

---

**Ready to train? Follow this guide step-by-step and you'll have a custom plate detector running in no time!** 🚀
