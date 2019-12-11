# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:47:56 2019
@author: JiazhenShen
"""

from RRModel import RRModel


class Foo:
    pass


myInput = Foo()

modelFile = "platform"

for i in range(1, 5):
    myInput.machineLearningEngine = 'CRF'
    myInput.output_horizon = i

    model = RRModel(myInput, model_file=modelFile)
    model.train(save=False)
