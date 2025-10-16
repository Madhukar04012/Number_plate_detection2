"""
License Plate Detection System v5.0 — Feature Focused Edition (No Database)

🚀 Performance-First Design:
   - Zero database overhead
   - Ultra-responsive UI
   - Real-time detection
   - Memory-efficient operation

✨ Core Features:
   - YOLOv8 AI detection (custom-trained plate model)
   - Dual OCR (Pytesseract + EasyOCR fallback)
   - Real-time webcam feed with multi-object tracking
   - Batch folder detection
   - Multi-object tracking (built-in YOLO tracking)
   - Watchlist alerts (instant popup warnings)
   - Dynamic confidence slider
   - Cropped plate saving toggle
   - Modern CustomTkinter dark mode interface
   - Live detection statistics
   - Multi-threaded for smooth performance

📦 Dependencies:
   pip install ultralytics torch opencv-python Pillow pytesseract easyocr customtkinter matplotlib

🎯 Usage:
   python main_v5.py

Author: Madhukar04012
Version: 5.0.0
License: MIT
Repository: https://github.com/Madhukar04012/Number_plate_detection2
"""

import os
import cv2
import time
import threading
from datetime import datetime
from typing import Optional, List, Tuple, Set

# GUI imports
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# AI/ML imports
from ultralytics import YOLO
import pytesseract
import easyocr

# Visualization
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================

# Paths
MODEL_PATH = "models/plate_best.pt"
OUTPUT_DIR = "outputs"
WATCHLIST_FILE = "watchlist.txt"

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR Configuration
# Uncomment and set if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==================== OCR ENGINE ====================

class OCREngine:
    """Handles text extraction from license plate images using dual OCR approach"""
    
    def __init__(self):
        """Initialize OCR engines"""
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
        except:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        print("✅ OCR engines initialized")
    
    def preprocess_plate(self, roi: 'np.ndarray') -> 'np.ndarray':
        """
        Advanced preprocessing pipeline for better OCR accuracy
        
        Args:
            roi: Region of interest (cropped plate image)
            
        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text(self, roi: 'np.ndarray') -> str:
        """
        Extract text from license plate using dual OCR approach
        
        Args:
            roi: Cropped license plate image
            
        Returns:
            Extracted and cleaned text
        """
        if roi is None or roi.size == 0:
            return ""
        
        # Preprocess image
        processed = self.preprocess_plate(roi)
        
        # Try Pytesseract first (faster)
        try:
            text = pytesseract.image_to_string(
                processed,
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip()
            
            # If Pytesseract returns good result, use it
            if text and len(text) >= 4:
                return self.clean_text(text)
        except Exception as e:
            print(f"⚠️ Pytesseract error: {e}")
        
        # Fallback to EasyOCR
        try:
            results = self.easyocr_reader.readtext(roi)
            text = " ".join([result[1] for result in results if result[2] > 0.3])
            return self.clean_text(text)
        except Exception as e:
            print(f"⚠️ EasyOCR error: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw OCR output
            
        Returns:
            Cleaned alphanumeric text in uppercase
        """
        # Remove non-alphanumeric characters
        cleaned = "".join([c for c in text if c.isalnum()])
        return cleaned.upper()

# ==================== MAIN APPLICATION ====================

class PlateDetectionApp:
    """Main application class for License Plate Detection System v5.0"""
    
    def __init__(self, root: ctk.CTk):
        """
        Initialize the application
        
        Args:
            root: CustomTkinter root window
        """
        self.root = root
        
        # Load YOLO model
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            messagebox.showerror(
                "Model Error",
                f"Could not load model from {MODEL_PATH}\n\n"
                f"Error: {e}\n\n"
                f"Please train a model using train_plate.py or download one."
            )
            raise
        
        # Initialize OCR engine
        self.ocr = OCREngine()
        
        # Camera and threading
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        
        # Detection parameters
        self.confidence = 0.4
        self.save_crops = True
        
        # Watchlist management
        self.watchlist: Set[str] = set()
        self.load_watchlist()
        
        # Statistics
        self.detect_count = 0
        self.session_detections: List[Tuple[str, datetime]] = []
        
        # Build the UI
        self.build_ui()
        
        print("✅ Application initialized successfully")
    
    # ==================== UI CONSTRUCTION ====================
    
    def build_ui(self):
        """Construct the main user interface"""
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        # Configure main window
        self.root.title("🚗 License Plate Detection v5.0 — Feature Focused Edition")
        self.root.geometry("1280x750")
        
        # Create sidebar
        self._build_sidebar()
        
        # Create main display area
        self._build_display_area()
        
        # Status bar at bottom
        self._build_status_bar()
    
    def _build_sidebar(self):
        """Build the control sidebar"""
        sidebar = ctk.CTkFrame(self.root, width=280, corner_radius=10)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)
        sidebar.pack_propagate(False)
        
        # Header
        header = ctk.CTkLabel(
            sidebar,
            text="🚗 Controls",
            font=("Arial", 22, "bold"),
            text_color="#00FF41"
        )
        header.pack(pady=15)
        
        # Camera controls section
        self._add_section_label(sidebar, "📹 Camera")
        ctk.CTkButton(
            sidebar,
            text="▶️ Start Webcam",
            command=self.start_webcam,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="#00AA00",
            hover_color="#00DD00"
        ).pack(pady=5, padx=15, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="⏹️ Stop Webcam",
            command=self.stop_webcam,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="#AA0000",
            hover_color="#DD0000"
        ).pack(pady=5, padx=15, fill="x")
        
        # File operations section
        self._add_section_label(sidebar, "📁 Files")
        ctk.CTkButton(
            sidebar,
            text="📷 Upload Image",
            command=self.upload_image,
            height=35
        ).pack(pady=5, padx=15, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="📂 Batch Folder",
            command=self.batch_process,
            height=35
        ).pack(pady=5, padx=15, fill="x")
        
        # Settings section
        self._add_section_label(sidebar, "⚙️ Settings")
        
        # Confidence slider
        conf_label = ctk.CTkLabel(sidebar, text=f"Confidence: {self.confidence:.2f}")
        conf_label.pack(pady=(10, 5))
        
        def update_conf_label(val):
            self.confidence = float(val)
            conf_label.configure(text=f"Confidence: {self.confidence:.2f}")
        
        self.slider = ctk.CTkSlider(
            sidebar,
            from_=0.1,
            to=0.95,
            number_of_steps=17,
            command=update_conf_label,
            height=20
        )
        self.slider.set(self.confidence)
        self.slider.pack(pady=5, padx=15, fill="x")
        
        # Save toggle
        self.save_toggle = ctk.CTkSwitch(
            sidebar,
            text="💾 Save Cropped Plates",
            command=self.toggle_save,
            font=("Arial", 12)
        )
        self.save_toggle.select()
        self.save_toggle.pack(pady=10, padx=15)
        
        # Watchlist section
        self._add_section_label(sidebar, "🚨 Watchlist")
        
        self.watchlist_entry = ctk.CTkEntry(
            sidebar,
            placeholder_text="Enter plate number...",
            height=35
        )
        self.watchlist_entry.pack(pady=5, padx=15, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="➕ Add to Watchlist",
            command=self.add_to_watchlist,
            height=30,
            fg_color="#FF8800",
            hover_color="#FFAA00"
        ).pack(pady=3, padx=15, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="🗑️ Clear Watchlist",
            command=self.clear_watchlist,
            height=30,
            fg_color="#666666",
            hover_color="#888888"
        ).pack(pady=3, padx=15, fill="x")
        
        # Statistics section
        self._add_section_label(sidebar, "📊 Analytics")
        
        ctk.CTkButton(
            sidebar,
            text="📈 Show Statistics",
            command=self.show_statistics,
            height=35,
            fg_color="#0088DD",
            hover_color="#00AAFF"
        ).pack(pady=5, padx=15, fill="x")
        
        # Detection counter at bottom
        self.counter_label = ctk.CTkLabel(
            sidebar,
            text="Total Detections: 0",
            font=("Arial", 16, "bold"),
            text_color="#00FF41"
        )
        self.counter_label.pack(side="bottom", pady=20)
    
    def _build_display_area(self):
        """Build the main display area for video/images"""
        main_container = ctk.CTkFrame(self.root, corner_radius=10)
        main_container.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title = ctk.CTkLabel(
            main_container,
            text="🎥 Detection Feed",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=10)
        
        # Main display (video/image feed)
        self.display_label = ctk.CTkLabel(
            main_container,
            text="📸 No feed active\n\nStart webcam or upload an image to begin",
            font=("Arial", 16),
            text_color="#666666"
        )
        self.display_label.pack(pady=20, expand=True)
        
        # Preview section for detected plate
        preview_frame = ctk.CTkFrame(main_container, height=150, corner_radius=10)
        preview_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            preview_frame,
            text="🔍 Last Detected Plate",
            font=("Arial", 14, "bold")
        ).pack(pady=5)
        
        self.preview_label = ctk.CTkLabel(
            preview_frame,
            text="Waiting for detection...",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.preview_label.pack(pady=10)
    
    def _build_status_bar(self):
        """Build status bar at the bottom"""
        status_bar = ctk.CTkFrame(self.root, height=40, corner_radius=0)
        status_bar.pack(side="bottom", fill="x")
        
        self.status_label = ctk.CTkLabel(
            status_bar,
            text="✅ Ready | Model: YOLOv8 | OCR: Pytesseract + EasyOCR",
            font=("Arial", 11)
        )
        self.status_label.pack(pady=10)
    
    def _add_section_label(self, parent, text: str):
        """Helper to add section headers in sidebar"""
        label = ctk.CTkLabel(
            parent,
            text=text,
            font=("Arial", 14, "bold"),
            text_color="#AAAAAA"
        )
        label.pack(pady=(15, 5))
    
    # ==================== WATCHLIST MANAGEMENT ====================
    
    def load_watchlist(self):
        """Load watchlist from file"""
        if os.path.exists(WATCHLIST_FILE):
            try:
                with open(WATCHLIST_FILE, "r") as f:
                    self.watchlist = set(line.strip().upper() for line in f if line.strip())
                print(f"✅ Loaded {len(self.watchlist)} plates from watchlist")
            except Exception as e:
                print(f"⚠️ Error loading watchlist: {e}")
                self.watchlist = set()
    
    def add_to_watchlist(self):
        """Add a plate to the watchlist"""
        plate = self.watchlist_entry.get().strip().upper()
        
        if not plate:
            messagebox.showwarning("Empty Input", "Please enter a plate number")
            return
        
        if plate in self.watchlist:
            messagebox.showinfo("Already Added", f"{plate} is already in the watchlist")
            return
        
        self.watchlist.add(plate)
        self._save_watchlist()
        self.watchlist_entry.delete(0, 'end')
        
        messagebox.showinfo(
            "Added to Watchlist",
            f"✅ {plate} has been added to the watchlist\n\n"
            f"You will be alerted when this plate is detected."
        )
    
    def clear_watchlist(self):
        """Clear all entries from watchlist"""
        if not self.watchlist:
            messagebox.showinfo("Empty", "Watchlist is already empty")
            return
        
        response = messagebox.askyesno(
            "Clear Watchlist",
            f"Are you sure you want to remove all {len(self.watchlist)} plates from the watchlist?"
        )
        
        if response:
            self.watchlist.clear()
            self._save_watchlist()
            messagebox.showinfo("Cleared", "Watchlist has been cleared")
    
    def _save_watchlist(self):
        """Save watchlist to file"""
        try:
            with open(WATCHLIST_FILE, "w") as f:
                f.write("\n".join(sorted(self.watchlist)))
        except Exception as e:
            print(f"⚠️ Error saving watchlist: {e}")
    
    def _check_watchlist_alert(self, plate_text: str):
        """Check if detected plate is in watchlist and show alert"""
        if plate_text in self.watchlist:
            messagebox.showwarning(
                "🚨 WATCHLIST ALERT 🚨",
                f"Watchlisted plate detected!\n\n"
                f"Plate Number: {plate_text}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"⚠️ IMMEDIATE ATTENTION REQUIRED"
            )
    
    # ==================== CORE DETECTION ====================
    
    def detect_frame(self, frame: 'np.ndarray', live_mode: bool = False) -> 'np.ndarray':
        """
        Process a frame and detect license plates
        
        Args:
            frame: Input frame (BGR image)
            live_mode: Whether in live webcam mode
            
        Returns:
            Annotated frame with detections
        """
        # Run YOLO detection
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        
        # Create copy for annotation
        annotated_frame = frame.copy()
        
        # Track best detection for preview
        best_plate_roi = None
        best_plate_text = ""
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Extract ROI (Region of Interest)
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Extract text using OCR
                plate_text = self.ocr.extract_text(roi)
                
                # Only process if text was extracted
                if plate_text and len(plate_text) >= 3:
                    # Update statistics
                    self.detect_count += 1
                    self.session_detections.append((plate_text, datetime.now()))
                    self.counter_label.configure(text=f"Total Detections: {self.detect_count}")
                    
                    # Save cropped plate if enabled
                    if self.save_crops:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_path = os.path.join(OUTPUT_DIR, f"{plate_text}_{timestamp}.jpg")
                        cv2.imwrite(output_path, roi)
                    
                    # Check watchlist
                    self._check_watchlist_alert(plate_text)
                    
                    # Update best plate for preview
                    best_plate_roi = roi
                    best_plate_text = plate_text
                    
                    label_text = plate_text
                else:
                    label_text = "Plate"
                
                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - 25),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
                
                # Draw confidence
                conf_text = f"{confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    conf_text,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
        
        # Update preview with best plate
        if best_plate_roi is not None:
            self._update_preview(best_plate_roi, best_plate_text)
        
        return annotated_frame
    
    def _update_preview(self, roi: 'np.ndarray', text: str):
        """Update the preview panel with detected plate"""
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = rgb.shape[:2]
            scale = min(250 / width, 100 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            img = ImageTk.PhotoImage(Image.fromarray(resized))
            
            # Update label
            self.preview_label.configure(image=img, text="")
            self.preview_label.image = img
            
        except Exception as e:
            print(f"⚠️ Error updating preview: {e}")
    
    # ==================== WEBCAM MODE ====================
    
    def start_webcam(self):
        """Start webcam detection"""
        if self.running:
            messagebox.showwarning("Already Running", "Webcam is already active")
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open webcam")
                return
            
            self.running = True
            self.status_label.configure(text="🔴 LIVE | Webcam Active")
            
            # Start webcam thread
            threading.Thread(target=self._webcam_loop, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam:\n{e}")
    
    def stop_webcam(self):
        """Stop webcam detection"""
        if not self.running:
            messagebox.showinfo("Not Running", "Webcam is not active")
            return
        
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status_label.configure(text="✅ Ready | Webcam Stopped")
        self.display_label.configure(image=None, text="📸 Webcam stopped")
    
    def _webcam_loop(self):
        """Main webcam capture loop (runs in separate thread)"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️ Failed to read frame from webcam")
                    break
                
                # Detect plates in frame
                annotated_frame = self.detect_frame(frame, live_mode=True)
                
                # Convert to RGB for display
                rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                height, width = rgb.shape[:2]
                scale = min(900 / width, 500 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = cv2.resize(rgb, (new_width, new_height))
                
                # Convert to PhotoImage
                img = ImageTk.PhotoImage(Image.fromarray(resized))
                
                # Update display
                self.display_label.configure(image=img, text="")
                self.display_label.image = img
                
                # Small delay to prevent overwhelming the GUI
                time.sleep(0.03)
                
            except Exception as e:
                print(f"⚠️ Error in webcam loop: {e}")
                break
        
        self.running = False
        print("✅ Webcam loop terminated")
    
    # ==================== FILE OPERATIONS ====================
    
    def upload_image(self):
        """Upload and process a single image"""
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Read image
            frame = cv2.imread(filepath)
            
            if frame is None:
                messagebox.showerror("Error", "Could not read image file")
                return
            
            self.status_label.configure(text=f"📷 Processing: {os.path.basename(filepath)}")
            
            # Detect plates
            annotated_frame = self.detect_frame(frame)
            
            # Convert to RGB for display
            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = rgb.shape[:2]
            scale = min(900 / width, 500 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            img = ImageTk.PhotoImage(Image.fromarray(resized))
            
            # Update display
            self.display_label.configure(image=img, text="")
            self.display_label.image = img
            
            self.status_label.configure(text="✅ Ready | Image processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")
            self.status_label.configure(text="❌ Error processing image")
    
    def batch_process(self):
        """Process all images in a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder")
        
        if not folder_path:
            return
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            messagebox.showwarning("No Images", "No image files found in the selected folder")
            return
        
        # Confirm batch processing
        response = messagebox.askyesno(
            "Batch Processing",
            f"Found {len(image_files)} images in folder.\n\n"
            f"Process all images?"
        )
        
        if not response:
            return
        
        # Process each image
        processed_count = 0
        
        for i, filename in enumerate(image_files):
            try:
                filepath = os.path.join(folder_path, filename)
                frame = cv2.imread(filepath)
                
                if frame is None:
                    continue
                
                # Update status
                self.status_label.configure(
                    text=f"📂 Processing {i + 1}/{len(image_files)}: {filename}"
                )
                
                # Detect plates
                annotated_frame = self.detect_frame(frame)
                
                # Display current image
                rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                scale = min(900 / width, 500 / height)
                resized = cv2.resize(rgb, (int(width * scale), int(height * scale)))
                img = ImageTk.PhotoImage(Image.fromarray(resized))
                self.display_label.configure(image=img, text="")
                self.display_label.image = img
                
                # Update GUI
                self.root.update()
                
                processed_count += 1
                
                # Small delay between images
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
        
        # Show completion message
        messagebox.showinfo(
            "Batch Complete",
            f"✅ Processed {processed_count} of {len(image_files)} images\n\n"
            f"Total detections: {self.detect_count}\n"
            f"Cropped plates saved to: {OUTPUT_DIR}"
        )
        
        self.status_label.configure(text="✅ Ready | Batch processing complete")
    
    # ==================== SETTINGS ====================
    
    def toggle_save(self):
        """Toggle the save cropped plates setting"""
        self.save_crops = not self.save_crops
        status = "enabled" if self.save_crops else "disabled"
        print(f"💾 Save cropped plates: {status}")
    
    # ==================== STATISTICS ====================
    
    def show_statistics(self):
        """Display detection statistics chart"""
        if self.detect_count == 0:
            messagebox.showinfo(
                "No Data",
                "No detections yet.\n\nStart detecting plates to see statistics."
            )
            return
        
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor('#1a1a1a')
            
            # Plot 1: Total detections
            ax1.bar(['Total Detections'], [self.detect_count], color='#00FF41', width=0.6)
            ax1.set_ylabel('Count', color='white')
            ax1.set_title('📊 Total Detections', color='white', fontsize=14, fontweight='bold')
            ax1.set_facecolor('#2a2a2a')
            ax1.tick_params(colors='white')
            ax1.spines['bottom'].set_color('white')
            ax1.spines['left'].set_color('white')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # Plot 2: Unique plates
            unique_plates = len(set([text for text, _ in self.session_detections]))
            ax2.bar(['Unique Plates'], [unique_plates], color='#00AAFF', width=0.6)
            ax2.set_ylabel('Count', color='white')
            ax2.set_title('🔢 Unique Plates', color='white', fontsize=14, fontweight='bold')
            ax2.set_facecolor('#2a2a2a')
            ax2.tick_params(colors='white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['left'].set_color('white')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            plt.suptitle(
                '🚗 License Plate Detection Statistics',
                color='white',
                fontsize=16,
                fontweight='bold'
            )
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate statistics:\n{e}")

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for the application"""
    print("=" * 60)
    print("🚗 License Plate Detection System v5.0")
    print("Feature Focused Edition (Database-Free)")
    print("=" * 60)
    print()
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print()
        print("Please train a model first:")
        print("  python train_plate.py")
        print()
        print("Or download a pre-trained model:")
        print("  python download_pretrained_model.py")
        print()
        return
    
    # Create application
    root = ctk.CTk()
    app = PlateDetectionApp(root)
    
    print()
    print("✅ Application started successfully")
    print("💡 Use the sidebar controls to begin detection")
    print()
    
    # Run main loop
    root.mainloop()
    
    print()
    print("=" * 60)
    print("👋 Application closed")
    print(f"📊 Session Summary:")
    print(f"   - Total Detections: {app.detect_count}")
    print(f"   - Cropped Plates: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
