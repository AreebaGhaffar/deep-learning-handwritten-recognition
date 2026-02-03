import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import pyttsx3
import os

class UAF_AI_Recognition(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("UAF Handwriting AI - Optimized") 
        # digit_letter_recognition_model
        self.geometry("1000x650") 

        # 1. Load Model with Error Catching
        try:
            self.model = tf.keras.models.load_model('handwriting_model.h5')
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model: {e}")
            self.destroy()
            return

        self.classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
        
        # 2. UI Setup
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="AI Settings", font=("Arial", 18, "bold")).pack(pady=20)
        
        self.auto_speak = ctk.CTkSwitch(self.sidebar, text="Auto-Speak")
        self.auto_speak.select()
        self.auto_speak.pack(pady=10)

        ctk.CTkButton(self.sidebar, text="Clear Canvas", fg_color="#d32f2f", command=self.clear).pack(pady=20, padx=10)

        # Drawing Canvas
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        # IMPORTANT: White background for user, but we will process it for AI
        self.canvas = tk.Canvas(self.canvas_frame, width=400, height=400, bg='white', cursor="pencil")
        self.canvas.pack(pady=20)
        
        self.image = Image.new("L", (400, 400), 0) # Internal black image
        self.draw = ImageDraw.Draw(self.image)

        self.result_label = ctk.CTkLabel(self.canvas_frame, text="Prediction: None", font=("Arial", 24, "bold"))
        self.result_label.pack(pady=10)

        # Bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.predict)

    def paint(self, event):
        x, y = event.x, event.y
        r = 15 # Thicker brush for better AI recognition
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255) # White ink on black for AI

    def predict(self, event=None):
        # 1. IMPROVED PREPROCESSING
        # Find the bounding box of the drawing to crop out empty space
        bbox = self.image.getbbox()
        if not bbox: return
        
        # Crop to the drawing and add padding to make it look like EMNIST data
        cropped = self.image.crop(bbox)
        padded = ImageOps.expand(cropped, border=40, fill=0)
        img = padded.resize((28, 28))
        
        # 2. Convert to Array
        img_array = np.array(img).astype('float32') / 255.0
        
        # EMNIST is often flipped/rotated compared to standard arrays
        img_array = np.rot90(img_array, k=3)
        img_array = np.fliplr(img_array)
        
        img_array = img_array.reshape(1, 28, 28, 1)

        # 3. Predict
        predictions = self.model.predict(img_array, verbose=0)
        char_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        result = self.classes[char_idx]

        self.result_label.configure(text=f"Prediction: {result} ({confidence*100:.1f}%)")

        # 4. FIXED VOICE LOGIC
        if self.auto_speak.get():
            self.speak_result(result)

    def speak_result(self, text):
        # We initialize and stop inside the function to prevent the "hang"
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop() # Force the engine to reset

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (400, 400), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.configure(text="Prediction: None")

if __name__ == "__main__":
    app = UAF_AI_Recognition()
    app.mainloop()