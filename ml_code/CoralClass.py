import random

import tensorflow as tf
import numpy as np
from itertools import product
from statistics import mean, median


class CoralPrediction:
    def __init__(self):
        self.model = None
        self.params = np.zeros(6)
        self.test_lat = []
        self.test_long = []
        self.test_depth = []
        self.test_temp = []
        self.test_sal = []
        self.test_oxy = []

    def set_model(self, modelpath):
        self.model = tf.keras.models.load_model(modelpath)

    def predict(self, params):
        self.params = params
        missing = []
        for i in range(len(self.params)):
            if type(self.params[i]) != float and type(self.params[i]) != int:
                missing.append(i)

        if 0 in missing:
            self.test_lat = np.linspace(-90, 90, 90)
        else:
            self.test_lat = [self.params[0]]

        if 1 in missing:
            self.test_long = np.linspace(-180, 180, 180)
        else:
            self.test_long = [self.params[1]]

        if 2 in missing:
            self.test_depth = np.linspace(0, 3000, 50)
        else:
            self.test_depth = [self.params[2]]

        if 3 in missing:
            self.test_temp = np.linspace(-2, 28, 20)
        else:
            self.test_temp = [self.params[3]]

        if 4 in missing:
            self.test_sal = np.linspace(0, 41, 20)
        else:
            self.test_sal = [self.params[4]]

        if 5 in missing:
            self.test_oxy = np.linspace(0.2, 132, 40)
        else:
            self.test_oxy = [self.params[5]]

        cond_list = list(product(*[self.test_lat, self.test_long, self.test_depth, self.test_temp, self.test_sal, self.test_oxy]))
        if len(cond_list) >= 4000:
            cond_list = random.sample(cond_list, 4000)
        count = 0
        predictions = []
        for cond in cond_list:
            print(count, ' completed out of: ', len(cond_list))
            predictions.append(self.model.predict([list(cond)])[0][0])
            count += 1
        return mean(predictions), median(predictions)
