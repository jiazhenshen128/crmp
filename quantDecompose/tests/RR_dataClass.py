# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 22:40:02 2018

@author: JiazhenShen
"""

import pandas as pd
import scipy.stats
from FinancialProxyCleanData import dataClass as dc
import numpy as np


class DataClass(dc.DataClass):
    def selectDefault(self, colInd=0, fillmissingby=False):
        assert self.DEVMODE, 'No reason to select default companies in the cutargetent vision if you are not in developing mode'
        assert self.proxy is not None, "Generate proxy firstly"

        # Select colInd column in self.proxy as the target candidate
        target1 = self.proxy.iloc[:, colInd]
        target1[target1 == 0] = 0.0001  # 0 -> a small number
        target1[target1 == 1] = 0.9999  # 1 -> be a limitation number

        # We only keep samples with some standard, although there may be clean enough.
        # Here, the RR models, we keep the samples whose proxy exist and number of missing values is limited.
        # But cleanning data here is not neceesary and core purposes here.
        # We also remove the volunteer brankrupt companies.
        missingn=np.sum(self._getCols('Total Assets').isnull().values,1) + np.sum(self._getCols('Total Current Liabilities').isnull().values,1)
        missingn+=np.sum(self._getCols('Working Capital').isnull().values,1) + np.sum(self._getCols('Total Non Current Liabilities (Incl Provisions)').isnull().values,1)
        self.targetposition = target1.notnull() & (target1 < 1) & (target1 > 0) & (missingn<1)  # &(np.sum(self.data.isnull().values, 1) < 6)#0and1've been changed

        # Target is its proxy in valid position
        self.target = target1[self.targetposition]
        self.target.name = 'target'
        # Every time we select the default and get target, we need to initialize
        # the valueableRatios and uniTestTable
        # self.uniTestTable = pd.DataFrame({'Variable':[], 'Coeff':[], 'PValue':[]})
        # self.uniTestTable.columns = ['Variable', 'Coeff', 'PValue']
        self.valuableRatios = pd.DataFrame()
        if hasattr(self, 'preData'):
            self.PvaluableRatios = pd.DataFrame() # Now it is initialized in __init__, so no worry about not running this method in using Mode
            
        if fillmissingby:  # not safe codes! transform choose the float columns automatically, might be problem
            d = self.data.loc[self.targetposition]   # self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes=='float64']].transform(lambda x:x.fillna(x.mean()))
            self.data.loc[self.targetposition,d.columns[d.dtypes=='float64']]=dd
            d = self.data.loc[self.targetposition]   # ,self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes=='float64']].transform(lambda x:scipy.stats.mstats.winsorize(x, limits = 0.05,inplace=True))
            self.data.loc[self.targetposition,d.columns[d.dtypes=='float64']]=dd
            
            if hasattr(self,'preData'):
                pass # not good way to do this

