import os
import pathlib
import CoralClass

modelpath = str(pathlib.Path(os.getcwd()).parent) + '/models/trial0.3.h5'

CoralPredictor = CoralClass.CoralPrediction()
CoralPredictor.set_model(modelpath)
conditions = [38, -150, 4000, None, 20, None]
mean_pred, med_pred = CoralPredictor.predict(conditions)
print('\n----------------------------------------')
print('The average likelihood of coral growth is: ', mean_pred*100, '%')
print('The median likelihood of coral growth is: ', med_pred*100, '%')
