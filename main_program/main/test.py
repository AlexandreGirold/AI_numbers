import glob
import os
import load
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model

path_folder = r"path to your folder"
which_model = input("1 for normal model, 2 for convolution model : ")
which_model = int(which_model)

if which_model == 1 :
    model = load_model("my_model.h5")
elif which_model == 2 :
    model = load_model("my_model_conv.h5")
else :
    raise ValueError("Must be 1 or 2.")
index = 0
correct = 0
list = load.get_answer(path_folder)
for img in glob.glob(path_folder + os.sep + "*.png"):
    image = load.load_image(img)
    image = cv2.resize(image, (28, 28))
    #print(image.shape)
    image = np.dot(np.array([image]), [0.2989, 0.5870, 0.1140])
    image = np.expand_dims(image, axis=-1)
    #print(image.shape)
    pred = model.predict(image / 255)
    if np.argmax(pred) == int(list[index]):
        correct += 1
        index += 1
        #print(f"Answer might be : {np.argmax(pred)}")
    else:
        index += 1
        #print(f"Answer might be : {np.argmax(pred)}")
print(f"Average correct answer : {correct/index*100}%")

