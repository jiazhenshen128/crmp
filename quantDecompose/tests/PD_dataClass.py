# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:33:24 2018

@author: JiazhenShen
"""



import numpy as np
import pandas as pd
#from sklearn import linear_model
import statsmodels.api as sm
import scipy.stats
#from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

from FinancialProxyCleanData import dataClass as dC


class DataClass(dC.DataClass):
    def selectDefault(self, colInd=0, fillmissingby=False):
        assert self.DEVMODE, 'No reason to select default companies in the current vision if you are not in developing mode'
        assert self.proxy is not None, "Generate proxy firstly"

        # Select colInd column in self.proxy as the target candidate
        target1 = self.proxy.iloc[:, colInd]
        target1[target1 == 0] = 0.0001  # 0 -> a small number
        target1[target1 == 1] = 0.9999  # 1 -> be a limitation number

        # We do not use macro variables now, so it is not needed
        # accountingYear = pd.DatetimeIndex(self._getCol("Latest Accounts Date").values).year

        TS = self._getCol("Trading Status")  # Active / no-active label
        # We only keep samples with some standard, although there may be clean enough.
        # Here, the PD models, we keep the samples whose proxy exist and number missing values is smaller than 6.
        # We also remove non-active samples whose proxy is smaller than 1. These unusual companies are not useful as
        # they are not our clients.
        self.targetposition = target1.notnull() & ((np.sum(self.data.isnull().values, 1) < 6)) & ( (TS == "Active").values | ( (TS != "Active").values & (target1<1) )   ) # & (target1 <  1) & (target1 >  0) #0and1've been changed
        assert sum(self.targetposition)>0, 'No samples left after cleaning. The qualification of samples is too strict!'
        # Target is whether its proxy is < 1 and it is non-active
        self.target = ((target1 < 1) & (target1 > 0) & (TS != "Active").values)[self.targetposition]  # self.target = target1[self.targetposition]
        self.target.name = 'target'
        # Every time we select the default and get target,  we need to initialize
        # the valueableRatios and uniTestTable

        # uniTestTable is not used now because we remove the function of significant testing
        # self.uniTestTable = pd.DataFrame({'Variable':[], 'Coeff':[], 'PValue':[]})
        # self.uniTestTable.columns = ['Variable', 'Coeff', 'PValue']
        self.valuableRatios = pd.DataFrame()
        if hasattr(self, 'preData'):
            self.PvaluableRatios = pd.DataFrame()
            
        if fillmissingby:  # not safe codes! transform choose the float columns automatically,  might be problem
            d = self.data.loc[self.targetposition]   # self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes == 'float64']].transform(lambda x:x.fillna(x.mean()))
            self.data.loc[self.targetposition, d.columns[d.dtypes == 'float64']] = dd
            d = self.data.loc[self.targetposition]   # , self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes == 'float64']].transform(lambda x:scipy.stats.mstats.winsorize(x,  limits  =  0.05, inplace = True))
            self.data.loc[self.targetposition, d.columns[d.dtypes == 'float64']] = dd
            
            if hasattr(self, 'preData'):
                pass  # not good way to do this
        
        if self.iswithin:
            loopn = self.consPara[0] - len(self.withoutP)

            self.target = pd.Series(np.tile(self.target.values, loopn))

        assert sum(self.target) > 0, 'No bankrupt company samples left'
        assert sum(self.target) < len(self.target), 'No active company samples left'
