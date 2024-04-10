import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load your pre-trained model ('CNN.model')
model = load_model('CNN.model')  # Replace with the correct path to your CNN.model
data_dir = "data"
class_names = os.listdir(data_dir)

# Function to classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the selected image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        img = Image.fromarray(img)  # Convert to PIL Image
        img.thumbnail((600, 600))  # Resize image for display
        img = ImageTk.PhotoImage(img)  # Convert to ImageTk format
        
        # Display the selected image
        image_label.config(image=img)
        image_label.image = img  # Keep a reference to prevent image from being garbage collected
        
        # Make predictions
        img_array = cv2.resize(cv2.imread(file_path), (50, 50))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]
        
        # Display the result
        result_label.config(text=f'Result: {class_label}')

# Create a tkinter window
root = tk.Tk()
root.title("CNN Model")

# Styling
root.configure(bg='#f0f0f0')

# Title label styling
title_style = {'font': ('Helvetica', 24), 'bg': '#f0f0f0'}

# Create a label for the title
title_label = tk.Label(root, text="CNN Model", **title_style)
title_label.pack(pady=20)

# Result label styling
result_style = {'font': ('Helvetica', 24), 'bg': '#f0f0f0'}

# Create a label to display the result
result_label = tk.Label(root, text="", **result_style)
result_label.pack(pady=20)

# Create a label to display the selected image
image_label = tk.Label(root, bg='#f0f0f0')
image_label.pack(pady=20)

# Button style
button_style = {'font': ('Helvetica', 14), 'fg': 'white', 'bg': '#4CAF50', 'relief': 'flat', 'width': 20, 'height': 2}

# Create a button to select an image
classify_button = tk.Button(root, text="Select Image", command=classify_image, **button_style)
classify_button.pack(pady=10)


# Start the tkinter main loop
root.mainloop()
