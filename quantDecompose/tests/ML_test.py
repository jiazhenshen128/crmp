from MachineLearningBasedModels import Engines
from MachineLearningBasedModels import TransformationFunctions
import pandas as pd
import numpy as np
from random import gauss
import matplotlib.pyplot as plt
import math


def func(x):
    return x + 1


def generate_fake_data1():
    data_length = 100
    observations = pd.Series([gauss(0, 5) + i for i in range(0, data_length)])
    observations.name = 'Observations'
    x = pd.DataFrame({'feature1': [20 * i + 100 for i in range(0, data_length)]})
    x.loc[[20, 66, 34], :] = np.nan
    return observations, x


def generate_fake_data2():
    data_length = 100
    observations = (pd.Series([gauss(0, 30) + i for i in range(0, data_length)]) > 50).astype(int)
    observations.name = 'Observations'
    x1 = pd.DataFrame({'feature1': [math.sin(i) for i in range(0, data_length)],
                       'feature2': [math.cos(i) for i in range(0, data_length)]})
    x2 = pd.DataFrame({'feature1': [i for i in range(0, data_length)]})
    return observations, x1, x2, data_length


def test_simplest_linear_model():
    observations, x = generate_fake_data1()
    mModel = Engines.linearModel(observations, TransformationFunctions.identity())
    mModel.xAndPart(x, partitionRatio=0.8, fillingValue=1000)
    mModel.fit()
    assert isinstance(mModel.trainError('MAE')[1], float), 'The training is not valid'
    assert isinstance(mModel.predict(x[3:7])[0], float), 'Can not predict'


def test_simplest_logistic_model():
    observations, x1, x2, data_length = generate_fake_data2()
    mModel = Engines.logisticRegression(observations, TransformationFunctions.identity())
    mModel.xAndPart(x1, partitionRatio=0.6)
    mModel.fit()
    assert isinstance(mModel.trainError('AUC')[1][0], float), 'The training is not valid'

    mModel.xAndPart(x2, partitionRatio=0.6)
    mModel.fit()
    assert isinstance(mModel.trainError('AUC')[1][0], float), 'The training is not valid'
    assert mModel.predict(x2[int(data_length/2)-10:int(data_length/2)+10])[0] in [0, 1], 'Can not predict'
    assert isinstance(mModel.predictProb(x2[2:8])[0][0], float), 'Can not predict probability'
