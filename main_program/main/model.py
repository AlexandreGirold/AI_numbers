"model trained onf the keras images"
import glob
import os
import load
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model


mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
"X_train contains the hand written number, Y_train contains the number which is written"

"scaling down the X"

X_train = X_train/255
X_test = X_test/255
print(X_train.shape)

"not need to scale down the labels"

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))  # output layer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# model = load_model("my_model.h5")
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.summary()
model.fit(X_train, Y_train, epochs=10)

# loss, accuracy = model.evaluate(X_test, Y_test)
# print(accuracy)
# print(loss)

model.save("my_model.h5")
