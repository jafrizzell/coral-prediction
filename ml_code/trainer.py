import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import RegscorePy
from math import sqrt
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

path = 'C:\Users\jafri\Documents\GitHub\coral-prediction\processed_data\combined_data_truncated.csv'

raw = pd.read_csv(path)

raw = raw.sample(frac=0.08, random_state=0)

print(raw.describe())

raw.pop('species')
raw.pop('round_d')

train = raw.sample(frac=0.8, random_state=0)
test = raw.drop(train.index)

train_features = train.copy()
test_features = test.copy()

train_labels = train_features['coral_present']
test_labels = test_features['coral_present']
train_features.pop('coral_present')
test_features.pop('coral_present')


def fit_and_evaluate(architecture):
    dnn_model = build_and_compile_model(architecture)

    history = dnn_model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=20)
    plot_loss(history)

    test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
    test_predictions = dnn_model.predict(test_features).flatten()
    r2= r2_score(np.asarray(test_labels).flatten(), test_predictions)
    mae = mean_absolute_error(np.asarray(test_labels).flatten(), test_predictions)
    aic = RegscorePy.aic.aic(np.asarray(test_labels, dtype=float).flatten(), np.asarray(test_predictions).astype(float), 4+2)
    rmse = sqrt(mean_squared_error(np.asarray(test_labels).flatten(), test_predictions))
    return dnn_model, aic, r2, mae, rmse, test_predictions


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 30])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # pass


def build_and_compile_model(arch):
    # Adjust the number of hidden layers and neurons per layer that results in best fit NN
    inputs = keras.Input(shape=(6,))
    if arch[1] == 0.0:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        dense1 = layers.Dropout(rate=0.2)(dense1)
        dense3 = layers.Dense(arch[2], activation='relu')(dense1)
        dense3 = layers.Dropout(rate=0.2)(dense3)
        outputs = layers.Dense(1)(dense3)

    elif arch[2] == 0.0:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        dense1 = layers.Dropout(rate=0.2)(dense1)
        dense2 = layers.Dense(arch[1], activation='relu')(dense1)
        dense2 = layers.Dropout(rate=0.2)(dense2)
        outputs = layers.Dense(1)(dense2)

    elif arch[1] == 0.0 and arch[2] == 0.0:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        dense1 = layers.Dropout(rate=0.2)(dense1)
        outputs = layers.Dense(1)(dense1)

    else:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        dense1 = layers.Dropout(rate=0.2)(dense1)
        dense2 = layers.Dense(arch[1], activation='relu')(dense1)
        dense2 = layers.Dropout(rate=0.2)(dense2)
        dense3 = layers.Dense(arch[2], activation='relu')(dense2)
        dense3 = layers.Dropout(rate=0.2)(dense3)
        outputs = layers.Dense(1)(dense3)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


models = []
aic_scores = []
r2_scores = []
maes = []
rmses = []
# l1 = np.linspace(32, 256, 8)
# l2 = np.linspace(0, 256, 9)
# l3 = np.linspace(0, 256, 9)
# parametric_space = list(product(*[l1, l2, l3]))
parametric_space = [[128, 128, 128]]
print(parametric_space)
start_t = time.time()
c = 1

for arch in parametric_space:
    print('Progress: ' + str(c) + '/' + str(len(parametric_space)))
    dnn_model, aic, r2, mae, rmse, test_predictions = fit_and_evaluate(arch)
    models.append(dnn_model)
    aic_scores.append(aic)
    r2_scores.append(r2)
    maes.append(mae)
    rmses.append(rmse)

    curr_time = time.time()
    diff_t = curr_time - start_t
    t_per_model = diff_t / c
    num_mods_rem = len(parametric_space) - c
    t_rem = t_per_model * num_mods_rem
    print("Estimated Time Remaining: " + time.strftime('%H:%M:%S', time.gmtime(t_rem)) + ' seconds')
    c += 1

parametric_space_t = np.asarray(parametric_space).transpose().tolist()
output_data = [parametric_space_t[0], parametric_space_t[1], parametric_space_t[1], aic_scores, maes, rmses, r2_scores]
output_data = np.asarray(output_data).transpose().tolist()
print(output_data)
oput = pd.DataFrame(output_data, columns=['L1', 'L2', 'L3', 'AIC', 'MAE', 'RMSE', 'R2'])
print(oput)
# oput.to_csv('Parametric_space_study.csv', index=False)

models[0].save('C:/Users/jafri/Documents/GitHub/coral-prediction/models/trial0.0.h5')

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions, s=0.8)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [-0.5, 1.5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()