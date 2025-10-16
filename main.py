import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to detect number plate
def detect_number_plate(img_path):
    nPlateCascade = cv2.CascadeClassifier("resources/haarcascade_russian_plate_number.xml")
    minArea = 500
    color1 = (255, 0, 255)
    
    img = cv2.imread(img_path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlate = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlate:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), color1, 2)
            cv2.putText(img, "Number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color1, 2)
            imgNumberPlate = img[y:y + h, x:x + w]
            return img, imgNumberPlate

    return img, None

# Function to open file dialog and detect plate
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img, imgNumberPlate = detect_number_plate(file_path)
        
        if imgNumberPlate is not None:
            imgNumberPlate = cv2.cvtColor(imgNumberPlate, cv2.COLOR_BGR2RGB)
            imgNumberPlate = Image.fromarray(imgNumberPlate)
            imgNumberPlate = ImageTk.PhotoImage(imgNumberPlate)
            lbl_imgNumberPlate.configure(image=imgNumberPlate)
            lbl_imgNumberPlate.image = imgNumberPlate
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        lbl_img.configure(image=img)
        lbl_img.image = img

# Create the Tkinter window
root = Tk()
root.title("License Plate Detector")

# Create a frame to center all elements
frame = Frame(root)
frame.pack(pady=20, padx=20)

# Create a title label
title = Label(frame, text="License Plate Detection", font=("Helvetica", 16))
title.pack(pady=50)

# Create a button to upload image
btn_upload = Button(frame, text="Upload Image", command=open_file)
btn_upload.pack(pady=30)

# Create a label to display the uploaded image
lbl_img = Label(frame)
lbl_img.pack(pady=30)

# Create a label to display the detected number plate
lbl_imgNumberPlate = Label(frame)
lbl_imgNumberPlate.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()

