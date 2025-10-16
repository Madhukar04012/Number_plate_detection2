"""
download_pretrained_model.py

Download a pre-trained YOLOv8 license plate detection model from Roboflow or GitHub.
This script helps you quickly get started without training your own model.

Usage:
    python download_pretrained_model.py
"""

import os
import urllib.request
import sys


def download_file(url, output_path):
    """Download file with progress bar"""
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r📥 Downloading: {percent}% ")
        sys.stdout.flush()
    
    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, output_path, progress_hook)
    print(f"\n✅ Downloaded to: {output_path}")


def download_roboflow_model():
    """
    Download a pre-trained model from Roboflow Universe
    
    Note: This is a placeholder. You should:
    1. Visit https://universe.roboflow.com/
    2. Search for "license plate detection yolov8"
    3. Find a model suitable for your region
    4. Get the download link
    5. Update the URL below
    """
    print("\n📚 Roboflow Universe Models")
    print("=" * 60)
    print("\nTo download a pre-trained model from Roboflow:")
    print("1. Visit: https://universe.roboflow.com/")
    print("2. Search: 'license plate detection yolov8'")
    print("3. Select a model (check accuracy and dataset size)")
    print("4. Click 'Download' → Choose 'YOLOv8' format")
    print("5. Get the download URL or API key")
    print("\nExample models:")
    print("  - ALPR (Automatic License Plate Recognition)")
    print("  - License Plate Detection")
    print("  - Vehicle Number Plate Recognition")
    print("\n⚠️  Make sure to choose a model trained on plates from your region!")
    print("=" * 60)


def download_github_model():
    """
    Download from GitHub repositories
    """
    print("\n🐙 GitHub Pre-trained Models")
    print("=" * 60)
    print("\nPopular repositories with pre-trained YOLOv8 plate models:")
    print("\n1. Search GitHub for: 'yolov8 license plate detection'")
    print("2. Look for repositories with 'weights' or 'models' folder")
    print("3. Download best.pt or similar file")
    print("4. Save to: models/plate_best.pt")
    print("\nExample repositories to explore:")
    print("  - https://github.com/search?q=yolov8+license+plate")
    print("  - Filter by: 'Most stars' or 'Recently updated'")
    print("=" * 60)


def download_ultralytics_hub():
    """
    Instructions for Ultralytics HUB
    """
    print("\n🌐 Ultralytics HUB (Cloud Training)")
    print("=" * 60)
    print("\nUltralytics HUB allows you to train models in the cloud:")
    print("\n1. Visit: https://hub.ultralytics.com/")
    print("2. Create free account")
    print("3. Upload your dataset or use public datasets")
    print("4. Train model in cloud (free tier available)")
    print("5. Download trained weights")
    print("\nBenefits:")
    print("  ✅ No local GPU required")
    print("  ✅ Easy dataset management")
    print("  ✅ Automatic versioning")
    print("  ✅ Model deployment options")
    print("=" * 60)


def use_general_yolov8():
    """
    Use general YOLOv8 model (will detect vehicles, not specifically plates)
    """
    print("\n🔍 Using General YOLOv8 Model (Temporary Solution)")
    print("=" * 60)
    print("\nIf you don't have a plate-specific model yet, you can:")
    print("\n1. Use YOLOv8 pre-trained on COCO dataset")
    print("   - Will detect 'car' class, not license plates specifically")
    print("   - Download automatically on first run")
    print("\n2. Modify main_v3.py to detect 'car' class first")
    print("   - Then crop and OCR the likely plate region")
    print("\n⚠️  This is NOT recommended for production!")
    print("   Train a custom model for best results.")
    print("=" * 60)


def create_models_directory():
    """Create models directory if it doesn't exist"""
    os.makedirs('models', exist_ok=True)
    print("\n✅ Created 'models/' directory")


def main():
    print("\n" + "=" * 60)
    print(" 🚗 License Plate Detection - Pre-trained Model Downloader")
    print("=" * 60)
    
    create_models_directory()
    
    print("\n📋 Choose an option:")
    print("1. Roboflow Universe (recommended for region-specific models)")
    print("2. GitHub repositories (community models)")
    print("3. Ultralytics HUB (cloud training)")
    print("4. Use general YOLOv8 (temporary, not recommended)")
    print("5. Train your own model (see TRAINING_GUIDE.md)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        download_roboflow_model()
    elif choice == '2':
        download_github_model()
    elif choice == '3':
        download_ultralytics_hub()
    elif choice == '4':
        use_general_yolov8()
    elif choice == '5':
        print("\n📖 Please read TRAINING_GUIDE.md for detailed instructions.")
        print("   Quick start: python train_plate_detector.py --create-yaml")
    else:
        print("\n❌ Invalid choice")
    
    print("\n" + "=" * 60)
    print("💡 Next Steps:")
    print("=" * 60)
    print("\n1. Place your model weights at: models/plate_best.pt")
    print("2. Install requirements: pip install ultralytics torch")
    print("3. Run the app: python main_v3.py")
    print("\n🎯 For best results, train a model on plates from your region!")
    print("   See: TRAINING_GUIDE.md")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
