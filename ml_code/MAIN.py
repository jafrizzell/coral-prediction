import os
import tensorflow as tf
import pathlib

path = str(pathlib.Path(os.getcwd()).parent) + '/models/trial0.2.h5'

model = tf.keras.models.load_model(path)

conditions = [0, -100, 860, 12, 34.5, 5]

predictions = model.predict([conditions])[0][0]

print(predictions)