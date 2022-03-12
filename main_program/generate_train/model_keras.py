"model trained onf the keras images"

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

dataset = tf.keras.datasets.mnist
(X_train, Y_train),(X_test, Y_test) = dataset.load_data()
"X_train contains the hand written number, y_train contains the number which is written"

"scaling down the X"

X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)

"not need to scale down the labels"

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(units =128, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax))#output layer

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer = 'adam', loss = loss_fn, metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 5)

loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)
print(loss)

