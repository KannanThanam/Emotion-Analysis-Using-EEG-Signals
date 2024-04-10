import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # VGG16 input size
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features from images
def extract_features(images_folder):
    features = []
    labels = []
    label_encoder = LabelEncoder()

    for folder in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder)
        
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file)
                
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_array = load_and_preprocess_image(image_path)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Extract features using the pre-trained model
                    features.append(model.predict(img_array)[0])
                    labels.append(folder)

    # Encode labels
    encoded_labels = label_encoder.fit_transform(labels)

    return np.array(features), np.array(encoded_labels)

# Path to your dataset folder with subfolders
dataset_folder = 'data'

# Extract features and labels
features, labels = extract_features(dataset_folder)

# Create a DataFrame with features and labels
df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
df['label'] = labels


df.to_csv('image_features.csv', index=False)
