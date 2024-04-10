import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from PIL import Image, ImageTk
import os
# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Load the trained AdaBoost model
adaboost_model = joblib.load('xgboost_model.joblib')

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # VGG16 input size
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features from an image
def extract_features(image_path):
    img_array = load_and_preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract features using the pre-trained model
    features = model.predict(img_array)[0]
    
    return features.reshape(1, -1)

# Function to predict using the trained AdaBoost model
def predict_image(image_features):
    prediction = adaboost_model.predict(image_features)[0]
    return prediction

# Function to handle image selection and prediction
def predict_from_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        image_features = extract_features(file_path)
        prediction = predict_image(image_features)
        if prediction == 0:
            result_label.config(text='Result: Adventorous')
        elif prediction == 1:
            result_label.config(text='Result: Amused')
        elif prediction == 2:
            result_label.config(text='Result: Angry')
        elif prediction == 3:
            result_label.config(text='Result: Sad')

# Display the selected image
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to prevent image from being garbage collected

# Tkinter GUI
root = tk.Tk()
root.title('XGBoost Image Prediction')

# Styling
root.configure(bg='#f0f0f0')

# Button Style
button_style = {'font': ('Arial', 14), 'fg': 'white', 'bg': '#4CAF50', 'relief': 'flat', 'width': 20, 'height': 2}

# Button to select an image
select_button = tk.Button(root, text='Select Image', command=predict_from_image, **button_style)
select_button.pack(pady=20)

# Label to display the prediction result
result_label = tk.Label(root, text='Result: ', font=('Arial', 24), bg='white')
result_label.pack(pady=10)

# Label to display the selected image
image_label = tk.Label(root, bg='white')
image_label.pack(pady=10)

root.mainloop()
