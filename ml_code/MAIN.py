import os
import pathlib
import CoralClass


def __main__(modelpath, conditions):
    CoralPredictor = CoralClass.CoralPrediction()
    CoralPredictor.set_model(str(pathlib.Path(os.getcwd())) + modelpath)
    mean_pred, med_pred = CoralPredictor.predict(conditions)
    print('\n----------------------------------------')
    print('The average likelihood of coral growth is: ', mean_pred*100, '%')
    print('The median likelihood of coral growth is: ', med_pred*100, '%')


if __name__ == '__main__':
    import sys
    args = sys.argv[2:]
    modelpath = sys.argv[1]
    pythonname = sys.argv[0]
    __main__(modelpath, args)

# modelpath = str(pathlib.Path(os.getcwd()).parent) + '/models/trial0.3.h5'
#
#
# conditions = [38, -150, 4000, None, 20, None]

