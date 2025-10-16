"""
train_plate.py
Train YOLOv8 on Custom License Plate Dataset

Ensure you have a dataset folder structure:
dataset/
  ├── images/
  │    ├── train/  (training images)
  │    ├── val/    (validation images)
  └── labels/
       ├── train/  (YOLO format labels)
       ├── val/    (YOLO format labels)

Usage:
    python train_plate.py

Features:
    - Automatic best model selection
    - Training progress visualization
    - Model evaluation and metrics
    - Auto-copy best weights to models/plate_best.pt
"""

from ultralytics import YOLO
import os

def train_model():
    """Train YOLOv8 model on license plate dataset"""
    
    print("\n" + "=" * 60)
    print("  🚗 YOLOv8 License Plate Detector Training")
    print("=" * 60 + "\n")
    
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("❌ Error: data.yaml not found!")
        print("\nPlease create data.yaml with your dataset configuration.")
        print("See data.yaml.template for an example.\n")
        return
    
    # Model selection
    print("📦 Model Selection:")
    print("  1. yolov8n.pt - Nano (fastest, good for CPU)")
    print("  2. yolov8s.pt - Small (balanced)")
    print("  3. yolov8m.pt - Medium (best balance, recommended)")
    print("  4. yolov8l.pt - Large (high accuracy)")
    print("  5. yolov8x.pt - Extra Large (highest accuracy, slow)")
    
    model_choice = input("\nChoose model (1-5) [default: 3]: ").strip() or "3"
    
    models = {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt",
        "3": "yolov8m.pt",
        "4": "yolov8l.pt",
        "5": "yolov8x.pt"
    }
    
    model_name = models.get(model_choice, "yolov8m.pt")
    print(f"\n✅ Selected model: {model_name}")
    
    # Training parameters
    print("\n⚙️  Training Configuration:")
    epochs = int(input("  Epochs [default: 100]: ").strip() or "100")
    batch = int(input("  Batch size [default: 16]: ").strip() or "16")
    imgsz = int(input("  Image size [default: 640]: ").strip() or "640")
    device = input("  Device (0 for GPU, cpu for CPU) [default: 0]: ").strip() or "0"
    
    print("\n" + "=" * 60)
    print("🏋️  Starting Training...")
    print("=" * 60 + "\n")
    
    # Load model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data='data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='runs/plates',
        name='plate_detector',
        patience=20,          # Early stopping patience
        save=True,            # Save checkpoints
        pretrained=True,      # Use pre-trained weights
        optimizer='auto',     # Auto-select optimizer
        verbose=True,         # Verbose output
        seed=0,               # Random seed
        deterministic=True,   # Deterministic mode
        single_cls=True,      # Single class dataset
        cos_lr=False,         # Cosine LR scheduler
        close_mosaic=10,      # Disable mosaic augmentation last N epochs
        resume=False,         # Resume from checkpoint
        amp=True,             # Automatic Mixed Precision
        # Learning rates
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate
        momentum=0.937,       # SGD momentum
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3.0,    # Warmup epochs
        # Augmentation
        hsv_h=0.015,          # HSV-Hue augmentation
        hsv_s=0.7,            # HSV-Saturation augmentation
        hsv_v=0.4,            # HSV-Value augmentation
        degrees=0.0,          # Rotation degrees
        translate=0.1,        # Translation
        scale=0.5,            # Scaling
        shear=0.0,            # Shear
        perspective=0.0,      # Perspective
        flipud=0.0,           # Flip up-down
        fliplr=0.5,           # Flip left-right
        mosaic=1.0,           # Mosaic augmentation
        mixup=0.0,            # MixUp augmentation
        copy_paste=0.0,       # Copy-paste augmentation
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60 + "\n")
    
    # Get paths
    weights_dir = f"runs/plates/plate_detector/weights"
    best_weights = f"{weights_dir}/best.pt"
    
    print(f"📁 Results saved to: runs/plates/plate_detector/")
    print(f"🏆 Best model: {best_weights}")
    
    # Validate
    print("\n🔍 Running validation...")
    metrics = model.val(data='data.yaml')
    
    print(f"\n📊 Validation Metrics:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p:.4f}")
    print(f"  Recall: {metrics.box.r:.4f}")
    
    # Copy to models directory
    os.makedirs("models", exist_ok=True)
    output_path = "models/plate_best.pt"
    
    import shutil
    shutil.copy(best_weights, output_path)
    
    print(f"\n✅ Best model copied to: {output_path}")
    print("\n🎉 Training complete! You can now run:")
    print("   python main_v4.py")
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        raise
