"""
main_v2.py
Enhanced License Plate Detection System (OCR + Webcam + Batch + SQLite logging)

Features:
- Haar cascade detection (existing) for license plates
- OCR using pytesseract (primary). EasyOCR fallback if Tesseract unavailable.
- Webcam / Video stream real-time detection
- Single image upload & batch folder processing
- Cropped plate saving to outputs/
- SQLite logging (plates.db) with datetime and image path
- Watchlist alert system (popup when a watched plate is detected)
- Simple Tkinter GUI with controls (Start/Stop webcam, Upload, Batch, Manage Watchlist)

Run:
    python main_v2.py

Notes:
 - Install dependencies (see requirements.txt)
 - You MUST install Tesseract OCR binary for best OCR accuracy:
   - Windows: https://github.com/tesseract-ocr/tesseract/wiki
   - Linux (Ubuntu): sudo apt install tesseract-ocr
   - MacOS: brew install tesseract
"""

import os
import threading
import time
import sqlite3
from datetime import datetime
from tkinter import (
    Tk, Label, Button, Frame, filedialog, Listbox, messagebox, Scrollbar, Entry, END, StringVar
)
from PIL import Image, ImageTk
import cv2
import numpy as np

# Try pytesseract first
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Attempt EasyOCR as fallback
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    _easyocr_reader = easyocr.Reader(['en'])  # keep global
except Exception:
    EASYOCR_AVAILABLE = False
    _easyocr_reader = None

# ----------------------
# Config / Paths
# ----------------------
CASCADE_FILE = "resources/haarcascade_russian_plate_number.xml"  # path to cascade file
OUTPUT_DIR = "outputs"
DB_FILE = "plates.db"
MIN_AREA = 500  # pixels to filter small detections

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
                    timestamp TEXT
                );""")
    conn.commit()
    conn.close()

def log_detection(plate_text, image_path):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO detections (plate_text, image_path, timestamp) VALUES (?, ?, ?)",
              (plate_text, image_path, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ----------------------
# OCR Utilities
# ----------------------
def ocr_plate(plate_img):
    """
    plate_img: BGR numpy array (cropped plate region)
    returns: cleaned text or empty string
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # try thresholding to improve OCR
    _,th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    text = ""
    if TESSERACT_AVAILABLE:
        try:
            # we assume single line -> psm 7 or 8 can help; whitelist digits+letters
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(th, config=config)
        except Exception:
            text = ""
    if not text.strip() and EASYOCR_AVAILABLE:
        try:
            # easyocr expects RGB
            rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            res = _easyocr_reader.readtext(rgb)
            # combine results
            text = " ".join([r[1] for r in res])
        except Exception:
            text = ""
    # basic cleaning
    text = text.strip()
    text = "".join([c for c in text if c.isalnum()])
    return text.upper()

# ----------------------
# Detection Utilities
# ----------------------
class PlateDetector:
    def __init__(self, cascade_path=CASCADE_FILE, min_area=MIN_AREA):
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.min_area = min_area

    def detect(self, frame):
        """
        frame: BGR image
        returns: list of dicts {x,y,w,h,roi_image}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        detections = []
        for (x, y, w, h) in rects:
            area = w * h
            if area < self.min_area:
                continue
            x2, y2 = x+w, y+h
            roi = frame[y:y2, x:x2].copy()
            detections.append({'x': x, 'y': y, 'w': w, 'h': h, 'roi': roi})
        return detections

# ----------------------
# GUI & App
# ----------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection — Enhanced v2")
        self.root.geometry("1000x640")
        self.detector = None
        try:
            self.detector = PlateDetector()
        except FileNotFoundError as e:
            messagebox.showerror("Cascade Missing", str(e))
            self.root.destroy()
            return

        # Webcam control
        self.cap = None
        self.video_thread = None
        self.running = False

        # watchlist
        self.watchlist = set()
        self.load_watchlist()

        # Build GUI
        self.build_ui()

        # init DB
        init_db()

    def build_ui(self):
        # Left frame: controls + listbox
        left = Frame(self.root, width=280, padx=8, pady=8)
        left.pack(side="left", fill="y")

        Label(left, text="Controls", font=("Helvetica", 14, "bold")).pack(pady=(0,6))

        Button(left, text="Start Webcam", width=24, command=self.start_webcam).pack(pady=4)
        Button(left, text="Stop Webcam", width=24, command=self.stop_webcam).pack(pady=4)
        Button(left, text="Upload Image", width=24, command=self.upload_image).pack(pady=4)
        Button(left, text="Process Folder (Batch)", width=24, command=self.batch_folder).pack(pady=4)
        Button(left, text="View DB (Last 20)", width=24, command=self.show_last_detections).pack(pady=4)
        Button(left, text="Export DB to CSV", width=24, command=self.export_db_csv).pack(pady=4)

        Label(left, text="Watchlist (alerts)", font=("Helvetica", 12)).pack(pady=(12,4))
        self.watch_entry_var = StringVar()
        Entry(left, textvariable=self.watch_entry_var).pack(pady=(0,4))
        Button(left, text="Add to Watchlist", width=24, command=self.add_watchlist).pack(pady=2)
        Button(left, text="Clear Watchlist", width=24, command=self.clear_watchlist).pack(pady=2)

        Label(left, text="Recent Detections", font=("Helvetica", 12)).pack(pady=(12, 4))
        # listbox with scrollbar
        lb_frame = Frame(left)
        lb_frame.pack(fill="both", expand=True)
        scrollbar = Scrollbar(lb_frame)
        scrollbar.pack(side="right", fill="y")
        self.recent_lb = Listbox(lb_frame, height=12, width=36, yscrollcommand=scrollbar.set)
        self.recent_lb.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.recent_lb.yview)

        # Right frame: image display
        right = Frame(self.root, padx=8, pady=8)
        right.pack(side="right", fill="both", expand=True)

        Label(right, text="Live / Preview", font=("Helvetica", 14, "bold")).pack()
        self.canvas_label = Label(right)
        self.canvas_label.pack(padx=6, pady=6)

        Label(right, text="Cropped Plate Preview", font=("Helvetica", 12)).pack(pady=(8,2))
        self.plate_label = Label(right)
        self.plate_label.pack(pady=6)

        # instructions
        Label(right, text="Notes: For best OCR, install Tesseract and provide clear plate images.", fg="gray").pack(pady=(6,0))

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
        messagebox.showinfo("Watchlist", f"{val} added to watchlist")

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

        for fp in files:
            img = cv2.imread(fp)
            self.process_and_display(img, source_name=os.path.basename(fp), save_crop=True)
            # small delay so GUI can update
            self.root.update()
            time.sleep(0.1)
        messagebox.showinfo("Batch", f"Processed {len(files)} images from folder.")

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
        # clear displayed image
        self.canvas_label.config(image='')

    def _video_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # process frame
            self.process_and_display(frame, live=True)
            # tiny sleep to throttle
            time.sleep(0.03)
        self.running = False

    # ------------------
    # Core processing
    # ------------------
    def process_and_display(self, frame, source_name="frame", live=False, save_crop=False):
        """
        frame: BGR image
        live: if True, show detection as live (does not save every detection to disk unless save_crop True)
        save_crop: whether to save cropped plates for this call (useful for batch)
        """
        display = frame.copy()
        detections = self.detector.detect(frame)
        chosen_plate_img = None
        chosen_text = ""

        # annotate detections
        for i, d in enumerate(detections):
            x,y,w,h = d['x'], d['y'], d['w'], d['h']
            roi = d['roi']
            # draw rectangle & label
            cv2.rectangle(display, (x,y), (x+w, y+h), (255,0,255), 2)
            cv2.putText(display, "Number Plate", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            # OCR the roi
            text = ocr_plate(roi)
            if text:
                cv2.putText(display, text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # choose the largest detection as primary
            if chosen_plate_img is None or (w*h) > (chosen_plate_img.shape[1] * chosen_plate_img.shape[0] if chosen_plate_img is not None else 0):
                chosen_plate_img = roi
                chosen_text = text

            # optionally save each crop
            if save_crop:
                out_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, roi)
                if text:
                    log_detection(text, out_path)
                    self.recent_lb.insert(0, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {text}")

                # check watchlist
                if text and text in self.watchlist:
                    self.alert_watchlist(text, out_path)

        # If we found a plate and NOT saving to disk in live mode: save crop to outputs once and log
        if chosen_plate_img is not None and live and not save_crop:
            # we save once per frame where plate found (throttle by time)
            # To avoid flooding, we'll not save every frame — implement a 1-per-1s throttle
            if not hasattr(self, "_last_save_time"):
                self._last_save_time = 0
            if time.time() - self._last_save_time > 1.0:
                out_name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, chosen_plate_img)
                if chosen_text:
                    log_detection(chosen_text, out_path)
                    # insert into recent list
                    self.recent_lb.insert(0, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {chosen_text}")
                    # watchlist check
                    if chosen_text in self.watchlist:
                        self.alert_watchlist(chosen_text, out_path)
                self._last_save_time = time.time()

        # update main display (convert BGR->RGB -> PIL -> ImageTk)
        disp_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(disp_rgb)
        img_pil = img_pil.resize((640, 360), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(img_pil)
        # keep reference
        self.canvas_label.imgtk = imgtk
        self.canvas_label.config(image=imgtk)

        # show chosen plate preview
        if chosen_plate_img is not None:
            plate_rgb = cv2.cvtColor(chosen_plate_img, cv2.COLOR_BGR2RGB)
            plate_pil = Image.fromarray(plate_rgb)
            plate_pil = plate_pil.resize((320, 120), Image.LANCZOS)
            plate_imgtk = ImageTk.PhotoImage(plate_pil)
            self.plate_label.imgtk = plate_imgtk
            self.plate_label.config(image=plate_imgtk)
        else:
            # clear
            self.plate_label.config(image='')

    def alert_watchlist(self, plate_text, image_path):
        # simple popup alert (runs in GUI thread)
        def _show():
            messagebox.showwarning("Watchlist Alert", f"Watchlist plate detected: {plate_text}\nSaved: {image_path}")

        # ensure executed in main thread
        self.root.after(0, _show)

    # ------------------
    # DB / export methods
    # ------------------
    def show_last_detections(self):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT plate_text, image_path, timestamp FROM detections ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        if not rows:
            messagebox.showinfo("DB", "No detections logged yet.")
            return
        win = Tk()
        win.title("Last 20 Detections")
        txt = ""
        for r in rows:
            txt += f"{r[2]} | {r[0]} | {r[1]}\n"
        lbl = Label(win, text=txt, justify="left", font=("Consolas", 10))
        lbl.pack(padx=8, pady=8)
        win.geometry("900x400")

    def export_db_csv(self):
        save_fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
        if not save_fp:
            return
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, plate_text, image_path, timestamp FROM detections ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        import csv
        with open(save_fp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "plate_text", "image_path", "timestamp"])
            writer.writerows(rows)
        messagebox.showinfo("Export", f"Exported {len(rows)} rows to {save_fp}")

# ----------------------
# Entry point
# ----------------------
def main():
    root = Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
