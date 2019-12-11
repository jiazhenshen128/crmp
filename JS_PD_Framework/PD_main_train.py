# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:47:56 2019
@author: JiazhenShen
"""

from PDModel import PDModel

modelFile ="platform"


class Foo:
    pass

myInput = Foo()

for myInput.output_horizon in range(1, 5):
    myInput.machineLearningEngine = 'CRF'

    model = PDModel(myInput, model_file=modelFile)
    model.train(save=False)

