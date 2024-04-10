import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import PIL.Image
import PIL.ImageTk
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
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

# Function to classify an image using CNN model
def classify_cnn_image(file_path):
    # Load your pre-trained CNN model ('CNN.model')
    cnn_model = load_model('CNN.model')
    data_dir = "data"
    class_names = os.listdir(data_dir)

    # Load and preprocess the selected image
    img = cv2.imread(file_path)
    img = cv2.medianBlur(img, 1)
    img = cv2.resize(img, (50, 50))  # Adjust the size if needed
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    img = np.asarray(img)  # Convert to numpy array

    # Make predictions
    predictions = cnn_model.predict(img)
    class_index = np.argmax(predictions)
    class_label = class_names[class_index]

    return f'Result: {class_label}'


# Function to load and preprocess an image for VGG16 model
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # VGG16 input size
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array


# Function to extract features from an image for VGG16 model
def extract_features(image_path):
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    img_array = load_and_preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features using the pre-trained model
    features = model.predict(img_array)[0]

    return features.reshape(1, -1)


# Function to predict using the trained AdaBoost model
def predict_image(image_features):
    adaboost_model = joblib.load('xgboost_model.joblib')
    prediction = adaboost_model.predict(image_features)[0]
    return prediction


# Function to detect image using a saved model
def detect_image(filename):
    datadir = 'data'
    Categories = os.listdir(datadir)
    loaded_model = joblib.load('model.sav')

    img = imread(filename)
    plt.imshow(img)
    plt.show()
    img_resize = resize(img, (50, 50, 3))
    l = [img_resize.flatten()]
    probability = loaded_model.predict(l)

    return Categories[int(probability)]

def show_results(cnn_result, adaboost_result, detection_result):
    # Create a new window for displaying results
    result_window = tk.Toplevel()
    result_window.title('Results')
    
    # Configure the window size
    result_window.geometry('400x200')
    
    # Label to display results
    result_label = tk.Label(result_window, text=f"CNN Model: {cnn_result}\n xgBoost: {adaboost_result}\n KNN Model: {detection_result}", font=('Arial', 14))
    result_label.pack(pady=20)

# Function to handle image selection and prediction for all models
def predict_from_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        # Prediction using CNN model
        cnn_result = classify_cnn_image(file_path)

        # Prediction using VGG16 + AdaBoost model
        image_features = extract_features(file_path)
        adaboost_prediction = predict_image(image_features)
        if adaboost_prediction == 0:
            adaboost_result = 'Adventurous'
        elif adaboost_prediction == 1:
            adaboost_result = 'Amused'
        elif adaboost_prediction == 2:
            adaboost_result = 'Angry'
        elif adaboost_prediction == 3:
            adaboost_result = 'Sad'

        # Prediction using saved model
        detection_result = detect_image(file_path)

        # Display results
        show_results(cnn_result, adaboost_result, detection_result)


# Tkinter GUI
root = tk.Tk()
root.title('Image Classifier')
root.state('zoomed')
root.configure(bg='#D3D3D3')

# Stylish Label for Title
title_label = tk.Label(root, text='Image Classifier', font=('Arial', 30, 'bold'), fg='#333', bg='#D3D3D3')
title_label.pack(pady=20)

# Button to select an image with stylish design
select_button = tk.Button(root, text='Select Image', font=('Arial', 14), fg='white', bg='#4CAF50', relief='flat', width=20, height=2, command=predict_from_image)
select_button.pack(pady=20)

root.mainloop()
