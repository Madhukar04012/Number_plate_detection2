"""
train_plate_detector.py

Ready-to-use script for training a YOLOv8 license plate detector.
Includes dataset preparation, training, and validation.

Usage:
    python train_plate_detector.py --data dataset/data.yaml --epochs 100

Requirements:
    pip install ultralytics torch torchvision
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def create_data_yaml_template(output_path='dataset/data.yaml'):
    """Create a template data.yaml file"""
    data_config = {
        'path': './dataset',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['plate']
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml template at: {output_path}")
    print("\n📝 Update the paths in data.yaml to match your dataset location.")


def validate_dataset(data_yaml):
    """Validate dataset structure"""
    if not os.path.exists(data_yaml):
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        print("\nRun with --create-yaml to generate a template")
        return False
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    path = data.get('path', '.')
    train_path = os.path.join(path, data.get('train', 'images/train'))
    val_path = os.path.join(path, data.get('val', 'images/val'))
    
    if not os.path.exists(train_path):
        print(f"❌ Error: Training images path not found: {train_path}")
        return False
    
    if not os.path.exists(val_path):
        print(f"❌ Error: Validation images path not found: {val_path}")
        return False
    
    # Count images
    train_imgs = list(Path(train_path).glob('*.jpg')) + list(Path(train_path).glob('*.png'))
    val_imgs = list(Path(val_path).glob('*.jpg')) + list(Path(val_path).glob('*.png'))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training images: {len(train_imgs)}")
    print(f"   Validation images: {len(val_imgs)}")
    print(f"   Total: {len(train_imgs) + len(val_imgs)}")
    
    if len(train_imgs) < 100:
        print(f"\n⚠️  Warning: Small training set ({len(train_imgs)} images).")
        print("   Recommended: 500+ images for decent results, 2000+ for production.")
    
    return True


def train_model(args):
    """Train YOLOv8 model on license plate dataset"""
    
    print("\n🚀 Starting YOLOv8 License Plate Detector Training\n")
    print("=" * 60)
    
    # Validate dataset
    if not validate_dataset(args.data):
        return
    
    # Load pre-trained model
    print(f"\n📦 Loading pre-trained model: {args.model}")
    model = YOLO(args.model)
    
    # Training configuration
    train_config = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'save': True,
        'pretrained': True,
        'optimizer': args.optimizer,
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': True,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': args.resume,
        'amp': True,
        # Learning rates
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    print("\n⚙️  Training Configuration:")
    for key, value in train_config.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🏋️  Training in progress...")
    print("=" * 60 + "\n")
    
    # Train
    try:
        results = model.train(**train_config)
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        
        # Print results location
        weights_path = Path(args.project) / args.name / 'weights' / 'best.pt'
        print(f"\n📁 Best model saved to: {weights_path}")
        print(f"📊 Results saved to: {Path(args.project) / args.name}")
        
        # Copy to models directory
        if args.copy_to_models:
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            output_path = models_dir / 'plate_best.pt'
            
            import shutil
            shutil.copy(weights_path, output_path)
            print(f"\n✅ Copied best model to: {output_path}")
            print("   You can now run: python main_v3.py")
        
        # Validate
        if args.validate:
            print("\n🔍 Running validation...")
            metrics = model.val(data=args.data)
            print(f"\n📊 Validation Metrics:")
            print(f"   mAP@50: {metrics.box.map50:.4f}")
            print(f"   mAP@50-95: {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.p:.4f}")
            print(f"   Recall: {metrics.box.r:.4f}")
        
        print("\n🎉 All done! Your custom plate detector is ready.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        raise


def test_model(args):
    """Test trained model on sample images"""
    
    if not os.path.exists(args.weights):
        print(f"❌ Error: Model weights not found at {args.weights}")
        return
    
    print(f"\n🧪 Testing model: {args.weights}")
    print(f"📁 Test images: {args.test_images}")
    
    model = YOLO(args.weights)
    
    # Run predictions
    results = model.predict(
        source=args.test_images,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_txt=True,
        project='runs/test',
        name='plate_test'
    )
    
    print(f"\n✅ Testing complete!")
    print(f"📊 Results saved to: runs/test/plate_test")
    
    # Print detection stats
    total_detections = sum(len(r.boxes) for r in results)
    print(f"🎯 Total plates detected: {total_detections}")
    print(f"📷 Images processed: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 License Plate Detector')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'create-yaml'],
                        help='Mode: train, test, or create-yaml')
    
    # Dataset
    parser.add_argument('--data', type=str, default='dataset/data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--create-yaml', action='store_true',
                        help='Create data.yaml template and exit')
    
    # Model
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='Pre-trained model to start from')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (-1 for auto)')
    parser.add_argument('--device', default='0',
                        help='Device to use (0, 1, 2, ... or cpu)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['auto', 'SGD', 'Adam', 'AdamW'],
                        help='Optimizer')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/plates',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='plate_detector',
                        help='Run name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    
    # Post-training
    parser.add_argument('--validate', action='store_true',
                        help='Run validation after training')
    parser.add_argument('--copy-to-models', action='store_true', default=True,
                        help='Copy best model to models/plate_best.pt')
    
    # Testing
    parser.add_argument('--weights', type=str, default='models/plate_best.pt',
                        help='Path to trained weights for testing')
    parser.add_argument('--test-images', type=str, default='test_images/',
                        help='Path to test images')
    parser.add_argument('--conf', type=float, default=0.35,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    
    args = parser.parse_args()
    
    # Create YAML template
    if args.create_yaml or args.mode == 'create-yaml':
        create_data_yaml_template(args.data)
        return
    
    # Train or test
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)


if __name__ == '__main__':
    main()
