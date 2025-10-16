"""
main_v3.py
Advanced License Plate Detection System with YOLOv8 AI + OCR

Features:
- **YOLOv8 Detection** (primary) with Haar Cascade fallback for robust plate detection
- Pre-trained YOLOv8n model for fast inference or YOLOv8m for better accuracy
- All v2.0 features: OCR, webcam, batch processing, SQLite logging, watchlist
- Enhanced preprocessing for better OCR accuracy
- Model download and caching
- Custom training guide included

Run:
    python main_v3.py

Requirements:
    pip install ultralytics torch torchvision opencv-python pytesseract easyocr pillow

Notes:
    - First run will download YOLOv8 model (~6MB for nano, ~50MB for medium)
    - For custom training on your region's plates, see train_yolo() function
    - Falls back to Haar Cascade if YOLOv8 unavailable
"""

import os
import threading
import time
import sqlite3
from datetime import datetime
from tkinter import (
    Tk, Label, Button, Frame, filedialog, Listbox, messagebox, Scrollbar, Entry, StringVar, HORIZONTAL
)
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import cv2
import numpy as np

# Try YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Try pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Try EasyOCR as fallback
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    _easyocr_reader = easyocr.Reader(['en'])
except Exception:
    EASYOCR_AVAILABLE = False
    _easyocr_reader = None

# ----------------------
# Config / Paths
# ----------------------
CASCADE_FILE = "resources/haarcascade_russian_plate_number.xml"
YOLO_MODEL_NAME = "yolov8n.pt"  # options: yolov8n.pt (fast), yolov8m.pt (accurate)
OUTPUT_DIR = "outputs"
DB_FILE = "plates.db"
MIN_AREA = 500
YOLO_CONF_THRESHOLD = 0.5  # confidence threshold for YOLO detections

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# Database Utilities
# ----------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT,
                    image_path TEXT,
                    timestamp TEXT,
                    detection_method TEXT
                );""")
    conn.commit()
    conn.close()

def log_detection(plate_text, image_path, method="unknown"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO detections (plate_text, image_path, timestamp, detection_method) VALUES (?, ?, ?, ?)",
              (plate_text, image_path, datetime.now().isoformat(), method))
    conn.commit()
    conn.close()

# ----------------------
# OCR Utilities (Enhanced)
# ----------------------
def preprocess_for_ocr(plate_img):
    """Enhanced preprocessing for better OCR"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(denoised)
    
    # Threshold
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def ocr_plate(plate_img):
    """
    plate_img: BGR numpy array (cropped plate region)
    returns: cleaned text or empty string
    """
    preprocessed = preprocess_for_ocr(plate_img)
    
    text = ""
    if TESSERACT_AVAILABLE:
        try:
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(preprocessed, config=config)
        except Exception:
            text = ""
    
    if not text.strip() and EASYOCR_AVAILABLE:
        try:
            rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            res = _easyocr_reader.readtext(rgb)
            text = " ".join([r[1] for r in res])
        except Exception:
            text = ""
    
    # Cleaning
    text = text.strip()
    text = "".join([c for c in text if c.isalnum()])
    return text.upper()

# ----------------------
# Detection Utilities
# ----------------------
class PlateDetector:
    def __init__(self, use_yolo=True, yolo_model_path=YOLO_MODEL_NAME, cascade_path=CASCADE_FILE, min_area=MIN_AREA):
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.min_area = min_area
        self.yolo_model = None
        self.cascade = None
        
        # Try to load YOLO
        if self.use_yolo:
            try:
                print(f"Loading YOLOv8 model: {yolo_model_path}...")
                self.yolo_model = YOLO(yolo_model_path)
                print("YOLOv8 model loaded successfully!")
                self.detection_method = "YOLOv8"
            except Exception as e:
                print(f"YOLOv8 loading failed: {e}")
                self.use_yolo = False
        
        # Fallback to Haar Cascade
        if not self.use_yolo:
            if os.path.exists(cascade_path):
                self.cascade = cv2.CascadeClassifier(cascade_path)
                self.detection_method = "Haar Cascade"
                print("Using Haar Cascade detection")
            else:
                raise FileNotFoundError(f"Neither YOLO nor Cascade file available. Cascade path: {cascade_path}")

    def detect(self, frame):
        """
        frame: BGR image
        returns: list of dicts {x,y,w,h,roi_image, confidence}
        """
        if self.use_yolo and self.yolo_model:
            return self._detect_yolo(frame)
        else:
            return self._detect_cascade(frame)
    
    def _detect_yolo(self, frame):
        """YOLOv8 detection"""
        detections = []
        results = self.yolo_model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Convert to x,y,w,h format
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
                
                area = w * h
                if area < self.min_area:
                    continue
                
                roi = frame[y:y+h, x:x+w].copy()
                detections.append({
                    'x': x, 'y': y, 'w': w, 'h': h, 
                    'roi': roi, 'confidence': conf
                })
        
        return detections
    
    def _detect_cascade(self, frame):
        """Haar Cascade detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        detections = []
        
        for (x, y, w, h) in rects:
            area = w * h
            if area < self.min_area:
                continue
            roi = frame[y:y+h, x:x+w].copy()
            detections.append({
                'x': x, 'y': y, 'w': w, 'h': h, 
                'roi': roi, 'confidence': 0.0  # Cascade doesn't provide confidence
            })
        
        return detections

# ----------------------
# GUI & App
# ----------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection — v3.0 (YOLOv8 + AI)")
        self.root.geometry("1000x680")
        
        self.detector = None
        try:
            self.detector = PlateDetector(use_yolo=True)
            self.detection_method = self.detector.detection_method
        except FileNotFoundError as e:
            messagebox.showerror("Detector Missing", str(e))
            self.root.destroy()
            return

        # Webcam control
        self.cap = None
        self.video_thread = None
        self.running = False

        # Watchlist
        self.watchlist = set()
        self.load_watchlist()

        # Stats
        self.detection_count = 0

        # Build GUI
        self.build_ui()

        # Init DB
        init_db()

    def build_ui(self):
        # Top info bar
        info_frame = Frame(self.root, bg="#2C3E50", height=40)
        info_frame.pack(side="top", fill="x")
        
        self.info_label = Label(
            info_frame, 
            text=f"🚀 Detection Engine: {self.detection_method} | Detections: 0", 
            bg="#2C3E50", 
            fg="white", 
            font=("Helvetica", 11, "bold")
        )
        self.info_label.pack(pady=8)

        # Left frame: controls + listbox
        left = Frame(self.root, width=280, padx=8, pady=8, bg="#ECF0F1")
        left.pack(side="left", fill="y")

        Label(left, text="🎛️ Controls", font=("Helvetica", 14, "bold"), bg="#ECF0F1").pack(pady=(0,6))

        Button(left, text="📹 Start Webcam", width=24, command=self.start_webcam, bg="#3498DB", fg="white", font=("Helvetica", 10)).pack(pady=4)
        Button(left, text="⏹️ Stop Webcam", width=24, command=self.stop_webcam, bg="#E74C3C", fg="white", font=("Helvetica", 10)).pack(pady=4)
        Button(left, text="📁 Upload Image", width=24, command=self.upload_image, bg="#2ECC71", fg="white", font=("Helvetica", 10)).pack(pady=4)
        Button(left, text="📂 Process Folder (Batch)", width=24, command=self.batch_folder, bg="#9B59B6", fg="white", font=("Helvetica", 10)).pack(pady=4)
        Button(left, text="📊 View DB (Last 20)", width=24, command=self.show_last_detections, bg="#34495E", fg="white", font=("Helvetica", 10)).pack(pady=4)
        Button(left, text="💾 Export DB to CSV", width=24, command=self.export_db_csv, bg="#16A085", fg="white", font=("Helvetica", 10)).pack(pady=4)

        Label(left, text="🚨 Watchlist Alerts", font=("Helvetica", 12, "bold"), bg="#ECF0F1").pack(pady=(12,4))
        self.watch_entry_var = StringVar()
        Entry(left, textvariable=self.watch_entry_var, font=("Helvetica", 10)).pack(pady=(0,4))
        Button(left, text="➕ Add to Watchlist", width=24, command=self.add_watchlist, bg="#F39C12", fg="white").pack(pady=2)
        Button(left, text="🗑️ Clear Watchlist", width=24, command=self.clear_watchlist, bg="#C0392B", fg="white").pack(pady=2)

        Label(left, text="📜 Recent Detections", font=("Helvetica", 12, "bold"), bg="#ECF0F1").pack(pady=(12, 4))
        lb_frame = Frame(left, bg="#ECF0F1")
        lb_frame.pack(fill="both", expand=True)
        scrollbar = Scrollbar(lb_frame)
        scrollbar.pack(side="right", fill="y")
        self.recent_lb = Listbox(lb_frame, height=12, width=36, yscrollcommand=scrollbar.set, font=("Consolas", 9))
        self.recent_lb.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.recent_lb.yview)

        # Right frame: image display
        right = Frame(self.root, padx=8, pady=8, bg="white")
        right.pack(side="right", fill="both", expand=True)

        Label(right, text="🖼️ Live / Preview", font=("Helvetica", 14, "bold"), bg="white").pack()
        self.canvas_label = Label(right, bg="black")
        self.canvas_label.pack(padx=6, pady=6)

        Label(right, text="🔍 Cropped Plate Preview", font=("Helvetica", 12, "bold"), bg="white").pack(pady=(8,2))
        self.plate_label = Label(right, bg="gray")
        self.plate_label.pack(pady=6)

        Label(right, text="💡 Tip: Install Tesseract OCR for best text recognition results.", fg="gray", bg="white").pack(pady=(6,0))

    # ------------------
    # Watchlist methods
    # ------------------
    def load_watchlist(self):
        self.watchlist_file = "watchlist.txt"
        if os.path.exists(self.watchlist_file):
            with open(self.watchlist_file, "r") as f:
                lines = f.read().splitlines()
                self.watchlist = set([l.strip().upper() for l in lines if l.strip()])
        else:
            self.watchlist = set()

    def save_watchlist(self):
        with open(self.watchlist_file, "w") as f:
            for item in sorted(self.watchlist):
                f.write(item + "\n")

    def add_watchlist(self):
        val = self.watch_entry_var.get().strip().upper()
        if not val:
            return
        self.watchlist.add(val)
        self.save_watchlist()
        messagebox.showinfo("Watchlist", f"✅ {val} added to watchlist")
        self.watch_entry_var.set("")

    def clear_watchlist(self):
        if messagebox.askyesno("Clear watchlist", "Clear all watchlist entries?"):
            self.watchlist.clear()
            self.save_watchlist()
            messagebox.showinfo("Watchlist", "Watchlist cleared")

    # ------------------
    # Image & batch methods
    # ------------------
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        self.process_and_display(img, source_name=os.path.basename(path))

    def batch_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if not files:
            messagebox.showinfo("Batch", "No image files found in selected folder.")
            return

        for i, fp in enumerate(files):
            img = cv2.imread(fp)
            self.process_and_display(img, source_name=os.path.basename(fp), save_crop=True)
            self.root.update()
            time.sleep(0.05)
        messagebox.showinfo("Batch", f"✅ Processed {len(files)} images from folder.")

    # ------------------
    # Webcam / Video methods
    # ------------------
    def start_webcam(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam", "Unable to open webcam.")
            return
        self.running = True
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.canvas_label.config(image='')

    def _video_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.process_and_display(frame, live=True)
            time.sleep(0.03)
        self.running = False

    # ------------------
    # Core processing
    # ------------------
    def process_and_display(self, frame, source_name="frame", live=False, save_crop=False):
        display = frame.copy()
        detections = self.detector.detect(frame)
        chosen_plate_img = None
        chosen_text = ""

        for i, d in enumerate(detections):
            x, y, w, h = d['x'], d['y'], d['w'], d['h']
            roi = d['roi']
            conf = d.get('confidence', 0)
            
            # Draw rectangle & label
            color = (0, 255, 0) if conf > 0.7 else (255, 0, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
            label = f"Plate {conf:.2f}" if conf > 0 else "Plate"
            cv2.putText(display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # OCR
            text = ocr_plate(roi)
            if text:
                cv2.putText(display, text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Choose largest detection
            if chosen_plate_img is None or (w*h) > (chosen_plate_img.shape[1] * chosen_plate_img.shape[0]):
                chosen_plate_img = roi
                chosen_text = text

            if save_crop:
                out_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, roi)
                if text:
                    log_detection(text, out_path, self.detection_method)
                    self.recent_lb.insert(0, f"{datetime.now().strftime('%H:%M:%S')} - {text}")
                    self.detection_count += 1

                if text and text in self.watchlist:
                    self.alert_watchlist(text, out_path)

        # Live mode throttled save
        if chosen_plate_img is not None and live and not save_crop:
            if not hasattr(self, "_last_save_time"):
                self._last_save_time = 0
            if time.time() - self._last_save_time > 1.0:
                out_name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, chosen_plate_img)
                if chosen_text:
                    log_detection(chosen_text, out_path, self.detection_method)
                    self.recent_lb.insert(0, f"{datetime.now().strftime('%H:%M:%S')} - {chosen_text}")
                    self.detection_count += 1
                    if chosen_text in self.watchlist:
                        self.alert_watchlist(chosen_text, out_path)
                self._last_save_time = time.time()

        # Update info bar
        self.info_label.config(text=f"🚀 Detection Engine: {self.detection_method} | Detections: {self.detection_count}")

        # Display main image
        disp_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(disp_rgb)
        img_pil = img_pil.resize((640, 360), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(img_pil)
        self.canvas_label.imgtk = imgtk
        self.canvas_label.config(image=imgtk)

        # Display plate preview
        if chosen_plate_img is not None:
            plate_rgb = cv2.cvtColor(chosen_plate_img, cv2.COLOR_BGR2RGB)
            plate_pil = Image.fromarray(plate_rgb)
            plate_pil = plate_pil.resize((320, 120), Image.LANCZOS)
            plate_imgtk = ImageTk.PhotoImage(plate_pil)
            self.plate_label.imgtk = plate_imgtk
            self.plate_label.config(image=plate_imgtk)
        else:
            self.plate_label.config(image='')

    def alert_watchlist(self, plate_text, image_path):
        def _show():
            messagebox.showwarning("🚨 Watchlist Alert", f"Watchlist plate detected:\n\n{plate_text}\n\nSaved: {image_path}")
        self.root.after(0, _show)

    # ------------------
    # DB / export methods
    # ------------------
    def show_last_detections(self):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT plate_text, image_path, timestamp, detection_method FROM detections ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        if not rows:
            messagebox.showinfo("DB", "No detections logged yet.")
            return
        win = Tk()
        win.title("Last 20 Detections")
        txt = "TIMESTAMP | PLATE | METHOD | PATH\n" + "="*100 + "\n"
        for r in rows:
            txt += f"{r[2]} | {r[0]} | {r[3]} | {r[1]}\n"
        lbl = Label(win, text=txt, justify="left", font=("Consolas", 9))
        lbl.pack(padx=8, pady=8)
        win.geometry("1000x450")

    def export_db_csv(self):
        save_fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
        if not save_fp:
            return
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, plate_text, image_path, timestamp, detection_method FROM detections ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        import csv
        with open(save_fp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "plate_text", "image_path", "timestamp", "detection_method"])
            writer.writerows(rows)
        messagebox.showinfo("Export", f"✅ Exported {len(rows)} rows to {save_fp}")

# ----------------------
# Custom Training Guide (Optional)
# ----------------------
def train_yolo_custom():
    """
    Guide for training YOLOv8 on custom license plate dataset
    
    1. Prepare dataset in YOLO format:
       - images/ folder with .jpg files
       - labels/ folder with .txt files (one per image)
       - Format: class_id x_center y_center width height (normalized 0-1)
    
    2. Create data.yaml:
       ---
       path: /path/to/dataset
       train: images/train
       val: images/val
       nc: 1
       names: ['license_plate']
       ---
    
    3. Train:
       from ultralytics import YOLO
       model = YOLO('yolov8n.pt')
       model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)
    
    4. Export best weights:
       Use runs/detect/train/weights/best.pt
    
    5. Update YOLO_MODEL_NAME = 'path/to/best.pt' in this script
    """
    pass

# ----------------------
# Entry point
# ----------------------
def main():
    root = Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
