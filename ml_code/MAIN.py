import tensorflow as tf
import numpy as np

path = 'C:/Users/jafri/Documents/GitHub/coral-prediction/models/trial0.0.h5'

model = tf.keras.models.load_model(path)

conditions = [0, -100, 860, 12, 34.5, 5]

predictions = model.predict([conditions])[0][0]

print(predictions)