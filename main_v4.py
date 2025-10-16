"""
License Plate Detection v4.0 — Pro Edition
YOLOv8 + OCR + Tracking + CustomTkinter Dashboard + Real-time Stats

Features:
 - YOLOv8-based real-time detection (uses ultralytics)
 - Multi-object tracking (using Ultralytics track mode)
 - OCR reading (pytesseract + EasyOCR fallback)
 - Real-time webcam feed and batch folder mode
 - SQLite logging with timestamp and plate text
 - Watchlist alerts (popup)
 - Modern GUI with CustomTkinter
 - Live chart of detection frequency
 - Adjustable confidence threshold slider
 - Export to CSV
 - Detection statistics dashboard

Requirements:
    pip install ultralytics torch opencv-python Pillow pytesseract easyocr customtkinter matplotlib

Run:
    python main_v4.py

Note:
    - Ensure models/plate_best.pt exists (train using train_plate.py)
    - Install Tesseract OCR binary for best results
"""

import os
import time
import threading
import sqlite3
import cv2
import numpy as np
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Ultralytics not installed. Run: pip install ultralytics")

# OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  Pytesseract not installed. OCR will use EasyOCR only.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    _easyocr_reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    EASYOCR_AVAILABLE = False
    _easyocr_reader = None
    print("⚠️  EasyOCR not installed. Install for fallback OCR.")

# ========================
# Configuration
# ========================
MODEL_PATH = "models/plate_best.pt"
DB_FILE = "database/plates.db"
OUTPUT_DIR = "outputs"
WATCHLIST_FILE = "watchlist.txt"
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.45

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("database", exist_ok=True)

# ========================
# Database Functions
# ========================
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text TEXT NOT NULL,
        image_path TEXT,
        timestamp TEXT NOT NULL,
        confidence REAL,
        track_id INTEGER
    )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def log_plate(plate_text, image_path, confidence=0.0, track_id=None):
    """Log detected plate to database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO detections (plate_text, image_path, timestamp, confidence, track_id) VALUES (?, ?, ?, ?, ?)",
        (plate_text, image_path, timestamp, confidence, track_id)
    )
    conn.commit()
    conn.close()

def get_detection_stats():
    """Get detection statistics"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Total detections
    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]
    
    # Unique plates
    c.execute("SELECT COUNT(DISTINCT plate_text) FROM detections")
    unique = c.fetchone()[0]
    
    # Detections today
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (f"{today}%",))
    today_count = c.fetchone()[0]
    
    conn.close()
    return {"total": total, "unique": unique, "today": today_count}

def get_hourly_stats():
    """Get hourly detection counts"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp FROM detections")
    data = c.fetchall()
    conn.close()
    
    if not data:
        return {}
    
    hours = [int(d[0].split()[1].split(':')[0]) for d in data if len(d[0].split()) > 1]
    counts = {h: hours.count(h) for h in range(24)}
    return counts

# ========================
# OCR Functions
# ========================
def preprocess_plate_image(roi):
    """Enhanced preprocessing for better OCR"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # Threshold
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def read_plate_text(roi):
    """Extract text from plate ROI using OCR"""
    if roi is None or roi.size == 0:
        return ""
    
    preprocessed = preprocess_plate_image(roi)
    text = ""
    
    # Try Tesseract first
    if TESSERACT_AVAILABLE:
        try:
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(preprocessed, config=config)
        except Exception as e:
            print(f"Tesseract error: {e}")
            text = ""
    
    # Fallback to EasyOCR
    if not text.strip() and EASYOCR_AVAILABLE and _easyocr_reader:
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = _easyocr_reader.readtext(rgb)
            text = " ".join([r[1] for r in results])
        except Exception as e:
            print(f"EasyOCR error: {e}")
            text = ""
    
    # Clean text
    cleaned = "".join([c for c in text if c.isalnum()]).upper()
    return cleaned

# ========================
# Main Application
# ========================
class PlateApp:
    def __init__(self, root):
        self.root = root
        
        # Load YOLOv8 model
        if not YOLO_AVAILABLE:
            messagebox.showerror("Error", "Ultralytics not installed!\nRun: pip install ultralytics")
            root.destroy()
            return
        
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model Not Found",
                f"YOLOv8 model not found at: {MODEL_PATH}\n\n"
                "Please:\n"
                "1. Train a model using train_plate.py, OR\n"
                "2. Download a pre-trained model using download_pretrained_model.py\n"
                "3. Place it at: {MODEL_PATH}"
            )
            root.destroy()
            return
        
        print(f"📦 Loading YOLOv8 model from: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # State
        self.cap = None
        self.running = False
        self.conf_threshold = DEFAULT_CONF
        self.iou_threshold = DEFAULT_IOU
        self.watchlist = set()
        self.detection_count = 0
        self.tracked_plates = {}  # track_id: plate_text
        
        # Initialize
        self.load_watchlist()
        init_db()
        
        # Build UI
        self.build_ui()
        
        # Update stats
        self.update_stats_display()

    def build_ui(self):
        """Build the CustomTkinter UI"""
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root.title("🚗 License Plate Detection v4.0 Pro Edition")
        self.root.geometry("1400x800")
        
        # ============ Left Sidebar ============
        sidebar = ctk.CTkFrame(self.root, width=280, corner_radius=0)
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)
        
        # Logo/Title
        title_label = ctk.CTkLabel(
            sidebar,
            text="🚗 Plate Detector",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))
        
        version_label = ctk.CTkLabel(
            sidebar,
            text="v4.0 Pro Edition",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        version_label.pack(pady=(0, 20))
        
        # Controls Section
        controls_label = ctk.CTkLabel(
            sidebar,
            text="📹 Camera Controls",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        controls_label.pack(pady=(10, 10))
        
        self.start_btn = ctk.CTkButton(
            sidebar,
            text="▶️ Start Webcam",
            command=self.start_cam,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_btn.pack(pady=5, padx=20, fill="x")
        
        self.stop_btn = ctk.CTkButton(
            sidebar,
            text="⏹️ Stop Webcam",
            command=self.stop_cam,
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_btn.pack(pady=5, padx=20, fill="x")
        
        # File Operations
        files_label = ctk.CTkLabel(
            sidebar,
            text="📁 File Operations",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        files_label.pack(pady=(20, 10))
        
        ctk.CTkButton(
            sidebar,
            text="📷 Upload Image",
            command=self.upload_image
        ).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="📂 Process Folder",
            command=self.process_folder
        ).pack(pady=5, padx=20, fill="x")
        
        # Detection Settings
        settings_label = ctk.CTkLabel(
            sidebar,
            text="⚙️ Detection Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        settings_label.pack(pady=(20, 10))
        
        # Confidence slider
        conf_label = ctk.CTkLabel(sidebar, text="Confidence Threshold")
        conf_label.pack(pady=(5, 0))
        
        self.conf_value_label = ctk.CTkLabel(
            sidebar,
            text=f"{self.conf_threshold:.2f}",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.conf_value_label.pack(pady=(0, 5))
        
        self.conf_slider = ctk.CTkSlider(
            sidebar,
            from_=0.1,
            to=0.95,
            number_of_steps=17,
            command=self.update_conf
        )
        self.conf_slider.set(self.conf_threshold)
        self.conf_slider.pack(pady=5, padx=20, fill="x")
        
        # Data Management
        data_label = ctk.CTkLabel(
            sidebar,
            text="💾 Data Management",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        data_label.pack(pady=(20, 10))
        
        ctk.CTkButton(
            sidebar,
            text="📊 Show Statistics",
            command=self.show_chart
        ).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="💾 Export to CSV",
            command=self.export_csv
        ).pack(pady=5, padx=20, fill="x")
        
        # Watchlist
        watch_label = ctk.CTkLabel(
            sidebar,
            text="🚨 Watchlist",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        watch_label.pack(pady=(20, 10))
        
        self.entry = ctk.CTkEntry(
            sidebar,
            placeholder_text="Enter plate number"
        )
        self.entry.pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="➕ Add to Watchlist",
            command=self.add_watch,
            fg_color="orange",
            hover_color="darkorange"
        ).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(
            sidebar,
            text="🗑️ Clear Watchlist",
            command=self.clear_watch
        ).pack(pady=5, padx=20, fill="x")
        
        # ============ Main Content Area ============
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Stats Panel
        stats_frame = ctk.CTkFrame(main_frame, height=100)
        stats_frame.pack(fill="x", padx=5, pady=(5, 10))
        stats_frame.pack_propagate(False)
        
        # Stats labels
        self.total_label = ctk.CTkLabel(
            stats_frame,
            text="Total: 0",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.total_label.pack(side="left", padx=20, pady=10)
        
        self.unique_label = ctk.CTkLabel(
            stats_frame,
            text="Unique: 0",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.unique_label.pack(side="left", padx=20, pady=10)
        
        self.today_label = ctk.CTkLabel(
            stats_frame,
            text="Today: 0",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.today_label.pack(side="left", padx=20, pady=10)
        
        # Video Display
        display_frame = ctk.CTkFrame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.display = ctk.CTkLabel(
            display_frame,
            text="📹 No video feed\n\nClick 'Start Webcam' or 'Upload Image'",
            font=ctk.CTkFont(size=18)
        )
        self.display.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Plate Preview
        preview_frame = ctk.CTkFrame(main_frame, height=120)
        preview_frame.pack(fill="x", padx=5, pady=5)
        preview_frame.pack_propagate(False)
        
        preview_label = ctk.CTkLabel(
            preview_frame,
            text="🔍 Detected Plate Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preview_label.pack(pady=(5, 0))
        
        self.plate_preview = ctk.CTkLabel(preview_frame, text="")
        self.plate_preview.pack(pady=5)

    def update_conf(self, value):
        """Update confidence threshold"""
        self.conf_threshold = float(value)
        self.conf_value_label.configure(text=f"{self.conf_threshold:.2f}")

    def load_watchlist(self):
        """Load watchlist from file"""
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r") as f:
                self.watchlist = set(line.strip().upper() for line in f if line.strip())
        print(f"✅ Loaded {len(self.watchlist)} watchlist entries")

    def save_watchlist(self):
        """Save watchlist to file"""
        with open(WATCHLIST_FILE, "w") as f:
            f.write("\n".join(sorted(self.watchlist)))

    def add_watch(self):
        """Add plate to watchlist"""
        plate = self.entry.get().upper().strip()
        if plate:
            self.watchlist.add(plate)
            self.save_watchlist()
            self.entry.delete(0, 'end')
            messagebox.showinfo("✅ Added", f"'{plate}' added to watchlist")

    def clear_watch(self):
        """Clear watchlist"""
        if messagebox.askyesno("Confirm", "Clear all watchlist entries?"):
            self.watchlist.clear()
            self.save_watchlist()
            messagebox.showinfo("✅ Cleared", "Watchlist cleared")

    def start_cam(self):
        """Start webcam capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to open webcam")
            return
        
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self.camera_loop, daemon=True).start()
        print("📹 Webcam started")

    def stop_cam(self):
        """Stop webcam capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        print("⏹️ Webcam stopped")

    def camera_loop(self):
        """Main camera processing loop"""
        last_save_time = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Throttle saving (1 per second)
            save_now = (time.time() - last_save_time) > 1.0
            if save_now:
                last_save_time = time.time()
            
            self.detect_frame(frame, live=True, save=save_now)
            time.sleep(0.03)  # ~30 FPS
        
        self.running = False

    def upload_image(self):
        """Upload and process single image"""
        fp = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")]
        )
        if not fp:
            return
        
        img = cv2.imread(fp)
        if img is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        
        self.detect_frame(img, save=True)

    def process_folder(self):
        """Process all images in a folder"""
        folder = filedialog.askdirectory()
        if not folder:
            return
        
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
        ]
        
        if not files:
            messagebox.showinfo("No Images", "No image files found in folder")
            return
        
        processed = 0
        for fp in files:
            img = cv2.imread(fp)
            if img is not None:
                self.detect_frame(img, save=True)
                processed += 1
                self.root.update()
        
        messagebox.showinfo("✅ Complete", f"Processed {processed} images")

    def detect_frame(self, frame, live=False, save=False):
        """Detect plates in frame using YOLOv8 with tracking"""
        # Run detection with tracking
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            verbose=False
        )
        
        frame_disp = frame.copy()
        chosen_plate = None
        best_conf = 0
        plate_text = ""
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Get track ID if available
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                
                # Extract ROI
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # OCR
                text = read_plate_text(roi)
                
                # Draw bounding box
                color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"{text if text else 'Plate'} {conf:.2f}"
                if track_id:
                    label = f"ID:{track_id} {label}"
                
                cv2.putText(
                    frame_disp,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
                # Save if requested
                if save and text:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = os.path.join(OUTPUT_DIR, f"{timestamp}_{text}.jpg")
                    cv2.imwrite(path, roi)
                    log_plate(text, path, conf, track_id)
                    self.detection_count += 1
                    
                    # Check watchlist
                    if text in self.watchlist:
                        self.alert_watchlist(text, path)
                
                # Track best detection for preview
                if conf > best_conf:
                    best_conf = conf
                    chosen_plate = roi
                    plate_text = text
        
        # Update display
        self.update_display(frame_disp, chosen_plate, plate_text)
        
        # Update stats
        if save:
            self.update_stats_display()

    def update_display(self, frame, plate_roi=None, plate_text=""):
        """Update GUI display with frame and plate preview"""
        # Main display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # Resize to fit display
        max_w, max_h = 900, 600
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(rgb, (new_w, new_h))
        img = ImageTk.PhotoImage(Image.fromarray(resized))
        
        self.display.configure(image=img, text="")
        self.display.image = img
        
        # Plate preview
        if plate_roi is not None and plate_roi.size > 0:
            plate_rgb = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)
            plate_img = Image.fromarray(plate_rgb)
            plate_img = plate_img.resize((300, 100), Image.LANCZOS)
            plate_photo = ImageTk.PhotoImage(plate_img)
            
            self.plate_preview.configure(image=plate_photo, text="")
            self.plate_preview.image = plate_photo

    def update_stats_display(self):
        """Update statistics display"""
        stats = get_detection_stats()
        self.total_label.configure(text=f"📊 Total: {stats['total']}")
        self.unique_label.configure(text=f"🔢 Unique: {stats['unique']}")
        self.today_label.configure(text=f"📅 Today: {stats['today']}")

    def alert_watchlist(self, plate_text, image_path):
        """Show watchlist alert"""
        def show_alert():
            messagebox.showwarning(
                "🚨 Watchlist Alert",
                f"Watchlist plate detected!\n\n"
                f"Plate: {plate_text}\n"
                f"Saved: {image_path}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        self.root.after(0, show_alert)

    def show_chart(self):
        """Show detection statistics chart"""
        hourly_stats = get_hourly_stats()
        
        if not hourly_stats:
            messagebox.showinfo("No Data", "No detections logged yet")
            return
        
        # Create chart window
        chart_window = ctk.CTkToplevel(self.root)
        chart_window.title("📊 Detection Statistics")
        chart_window.geometry("800x600")
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        hours = list(range(24))
        counts = [hourly_stats.get(h, 0) for h in hours]
        
        ax.bar(hours, counts, color='#1f77b4', alpha=0.7)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Number of Detections", fontsize=12)
        ax.set_title("License Plate Detections by Hour", fontsize=14, weight='bold')
        ax.set_xticks(hours)
        ax.grid(axis='y', alpha=0.3)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_csv(self):
        """Export detections to CSV"""
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM detections ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            messagebox.showinfo("No Data", "No detections to export")
            return
        
        fp = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        
        if not fp:
            return
        
        import csv
        with open(fp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Plate", "Image Path", "Timestamp", "Confidence", "Track ID"])
            writer.writerows(rows)
        
        messagebox.showinfo("✅ Exported", f"Exported {len(rows)} records to:\n{fp}")

# ========================
# Main Entry Point
# ========================
def main():
    print("\n" + "=" * 60)
    print("  🚗 License Plate Detection System v4.0 Pro Edition")
    print("=" * 60 + "\n")
    
    root = ctk.CTk()
    app = PlateApp(root)
    
    print("\n✅ Application ready!")
    print("=" * 60 + "\n")
    
    root.mainloop()

if __name__ == "__main__":
    main()
