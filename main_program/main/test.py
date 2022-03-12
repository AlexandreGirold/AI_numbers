import glob
import os
import load
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model

path_folder = r'C:\Users\sacha\Desktop\AI_numbers\main_program\images'

model = load_model("my_model.h5")
for img in glob.glob(path_folder + os.sep + "*.png"): 
    image = load.load_image(img)
    image = np.dot (np.invert(np.array([image])),[0.2989, 0.5870, 0.1140])
    image = np.expand_dims(image, axis=-1)
    print(image.shape)
    answer = model.predict(image/255)
    print (f'Answer : {np.argmax(answer)}')