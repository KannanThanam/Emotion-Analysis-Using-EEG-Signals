import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tkinter import *
import tkinter.messagebox
import PIL.Image
import PIL.ImageTk
from tkinter import filedialog
import pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

datadir = 'data'
Categories = os.listdir(datadir)

v = ""
root = Tk()
root.title("Image Detection")
root.state('zoomed')
root.configure(bg='#f0f0f0')

# Load trained model
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to detect image
def detect(filename):
    img = imread(filename)
    img_resize = resize(img, (50, 50, 3))
    l = [img_resize.flatten()]
    probability = loaded_model.predict(l)
    print(int(probability))
    print("The predicted image is : " + Categories[int(probability)])
    value.set(Categories[int(probability)])

# Function to handle click action
def ClickAction(event=None):
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((350, 350))
    img = PIL.ImageTk.PhotoImage(img)
    global panel
    panel = Label(root, image=img)
    panel.image = img
    panel = panel.place(relx=0.3, rely=0.1)
    detect(filename)

# GUI setup
root.geometry("900x500")

# Load and display background image
background_image = PIL.Image.open("eeg_background.jpg")
background_image = background_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), PIL.Image.ANTIALIAS)
background_image = PIL.ImageTk.PhotoImage(background_image)
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Title
label = Label(root, text="Image Detection", font=("Arial", 24), fg="#333", bg="#f0f0f0")
label.place(relx=0.5, rely=0.02, anchor=CENTER)

# Stylish Button Design
load_button = Button(root, text='Load Image', font=("Arial", 14), fg='white', bg='#4CAF50', relief=FLAT, width=10, height=2, command=ClickAction)
load_button.place(relx=0.5, rely=0.9, anchor=CENTER)

# Result Label
value = StringVar()
result_label = Label(root, textvariable=value, font=("Arial", 18), fg="#333", bg="#f0f0f0")
result_label.place(relx=0.5, rely=0.85, anchor=CENTER)

root.mainloop()

