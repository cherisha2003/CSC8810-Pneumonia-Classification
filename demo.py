import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121

IMG_SIZE = 224

# -------------------------------
# LOAD MODEL + SCALER
# -------------------------------
print("Loading model...")
clf = joblib.load("model_baseline.pkl")
scaler = joblib.load("scaler.pkl")

print("Loading DenseNet...")
densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg')

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
def preprocess(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_image(img_path):
    try:
        img = preprocess(img_path)

        features = densenet.predict(img)
        features = scaler.transform(features)

        pred = clf.predict(features)[0]
        prob = clf.predict_proba(features)[0]

        label = "PNEUMONIA" if pred == 1 else "NORMAL"
        confidence = max(prob)

        show_result(img_path, label, confidence)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# -------------------------------
# DISPLAY RESULT
# -------------------------------
def show_result(img_path, label, confidence):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.title(f"{label} ({confidence:.2f})", fontsize=14)
    plt.axis("off")
    plt.show()

# -------------------------------
# FILE PICKER
# -------------------------------
def open_file():
    file_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if file_path:
        predict_image(file_path)

# -------------------------------
# GUI WINDOW
# -------------------------------
root = tk.Tk()
root.title("Pneumonia Detection Demo")
root.geometry("300x150")

btn = tk.Button(root, text="Upload X-ray Image", command=open_file, height=2)
btn.pack(expand=True)

label = tk.Label(root, text="Click to upload image")
label.pack()

root.mainloop()