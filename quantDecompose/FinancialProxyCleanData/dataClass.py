# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:18:47 2018

@author: JiazhenShen
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA


class DataClass:
    def __init__(self, input_historical=4, output_horizon=1, trainData=pd.DataFrame(), preData=pd.DataFrame(), dataNote='',
                 iswithin=False, consPara=(5, " (Period #)")):
        # data: pandas dataframe
        # isTTA True means using TTA/TL as a proxy
        # DataNote: string information for main in plots and other marks
        # yearsTrain, years using for training set
        # yearsAcross, years using for prediction
        # Optional argument preData for prediction
        # optional argument for total years of data we have and the 
        # columns (counting from 1) we actually need, which must be numeric
        assert input_historical+output_horizon <= consPara[0], \
        f'yearsTrain+yearsAcross must<={consPara[0]}'
        assert input_historical > 0 and output_horizon > 0, \
        'yearsTrain,yearsAcross must>0'

        self.DEVMODE = ~preData.empty
        self.data = trainData.copy()
        self.dataNote = dataNote
        self.yearsTrain = input_historical
        self.yearsAcross = output_horizon
        self.names = trainData.columns
        self.iswithin =iswithin
        self.consPara = consPara
        
        # Prediction data is going to be moved, so the most recent data can be used

        if not preData.empty:
            self.preData = preData.copy()
            if not iswithin:
                cols = self.preData.columns.tolist()
                for i in range(0, self.yearsAcross):
                    # cols[consPara[1]:(consPara[2])] = cols[(consPara[1]-1):(consPara[2]-1)]
                    cols = [cols[-1]] + cols[:-1]
                tmppreData = self.preData[cols]

                tmppreData.columns = self.preData.columns
                self.preData = tmppreData
        self.consPara = consPara

        # 'None' attributes List
        self.targetposition = None  # .target' actually means Target
        self.target = None          # .target' actually means Target
        self.proxy = None  # .target' actually means Target, proxy means general(hat) Target without filters
        self.uniTestTable = None
        self.valuableRatios = None
        self.PvaluableRatios = None
        self.isTTA = None
        self.transFunClass = None
        self.transFun = None
        self.transInvFun = None
        self.Ttarget = None  # 'target' actually means Target, Ttarget means Transformed Target
        self.PvaluableRatios = pd.DataFrame()
        self.inputs = set()
        self.deriveMapping = dict()
        
        withoutP = [i for i in range(0,self.yearsAcross)]
        if self.yearsAcross+self.yearsTrain<consPara[0]:
            withoutP = list(set(withoutP+[i for i in range(self.yearsAcross+self.yearsTrain,consPara[0])]))


        self.withoutP = withoutP 
        if self.iswithin:
            self.withoutP = [3, 4]

    #######################Private Methods#################################

    def _getCol(self, name, isPreData=False):
        if isPreData:
            return self.preData.loc[:, name].copy()
        else:
            return self.data.loc[:, name].copy()

    def _getCols(self, name, isPreData=False):
        res = pd.DataFrame()
        for i in range(0, self.consPara[0]):
            myName = name+self.consPara[1].split(sep='#')[0]+str(i+1)+self.consPara[1].split(sep='#')[1]  # This is the format of the period presentation!
            res = pd.concat([res, self._getCol(myName, isPreData)], axis=1)
        return res
    
    def _getDummyVars(self, name, isPreData=False):
        if isPreData:
            res = pd.get_dummies(self.data.loc[:, name]).iloc[0:0, :].copy()
            target = self.preData.loc[:, name]
            for elem in res.columns:
                res[str(elem)] = (target == elem).apply(int)  
            return res
        else:                         
            return pd.get_dummies(self.data.loc[:, name].copy())

    def _getDCol(self, name, isPreData=False):
        if isPreData:
            return self._getCol(name, isPreData)
        else:
            assert hasattr(self, 'target'), 'selectdefault() first!'
            v = self._getCol(name)
            return v[self.targetposition]

    def _getDCols(self, name, isPreData=False):
        res = pd.DataFrame()
        for i in range(0, self.consPara[0]):
            myName = name +self.consPara[1].split(sep='#')[0]+str(i+1)+self.consPara[1].split(sep='#')[1]
            res = pd.concat([res, self._getDCol(myName, isPreData)], axis=1)

        # indn5na=np.sum(res.isnull().values,1)<5
        # res[indn5na]=res[indn5na].interpolate(limit=1,axis=1,limit_direction='backward')
        return res

    def _getDDummyVars(self, name, isPreData=False):
        if isPreData:
            return self._getDummyVars(name, isPreData)
        else:
            assert hasattr(self, 'target'), 'selectdefault() first!'
            v = self._getDummyVars(name)
            return v[self.targetposition]

    def _calculate_feature(self, isPreData, numeratorName, denominatorName, useLog, useBigLog, useDiff, categorical):
        if categorical:
            assert denominatorName == None and useLog == False and useDiff == False, \
                'Categorical variables cannot have these arguements'
            numerator = self._getDDummyVars(numeratorName, isPreData)
        else:
            numerator = self._getDCols(numeratorName, isPreData)

        if useLog:
            numerator = numerator.apply(np.log)
        # The case for non-ratio,such as log(Assets)
        if denominatorName:
            denominator = self._getDCols(denominatorName, isPreData)
            if useLog:
                numerator = numerator.apply(np.log)
            ratios = numerator / denominator.values
            ratios.values[denominator.values == 0] = np.nan
        else:
            ratios = numerator

        if useDiff:
            ratios = -ratios.diff(axis=1).shift(-1, axis=1)
            ratios.drop(ratios.columns[len(ratios.columns) - 1],
                        axis=1, inplace=True)

        if useBigLog:
            ratios = 1 - ratios
            ratios.apply(np.log)

        return ratios

        #####Regression and keep significant variables######

    def _reserve_features(self, period, ratios, varName, withoutP, fillingValue, categorical, numeratorName, denominatorName, useDiff):
        tmpColumn = pd.DataFrame()
        myRange = range(0, len(ratios.columns))
        if period != 'auto':
            myRange = period
        if categorical:
            self.inputs.add(numeratorName)
            tmpColumn = ratios
            #print(tmpColumn)
        else:
            for i in [ii for ii in myRange if ii not in withoutP]:
                myRatio = ratios.iloc[:, i]
                varNamei = varName + ' Period ' + str(i + 1)
                if not categorical:
                    myRatio.name = varNamei
                myRatio = myRatio.replace([np.inf, -np.inf], np.nan)
                myRatio[myRatio.isnull()] = fillingValue
                if self.iswithin:
                    if tmpColumn.empty:
                        tmpColumn = myRatio
                    else:
                        tmpColumn = pd.Series(np.concatenate([tmpColumn.values, myRatio.values], axis=0))
                        tmpColumn.name = varNamei
                else:
                    tmpColumn = pd.concat([tmpColumn, myRatio], axis=1, sort=True)
                    self.inputs.add(numeratorName + ' (Period '+str(i+1-self.yearsAcross)+')')
                    if useDiff:
                        self.inputs.add(numeratorName + ' (Period ' + str(i + 1 - 1 - self.yearsAcross) + ')')
                    if denominatorName:
                        self.inputs.add(denominatorName + ' (Period ' +str(i+1-self.yearsAcross)+')')
                        if useDiff:
                            self.inputs.add(denominatorName + ' (Period ' + str(i + 1 - 1 - self.yearsAcross) + ')')
        return tmpColumn
    
    #######################Public Methods#################################
    def additionClean(self, left, *right):
        #  initial the sum of right hand
        if not self.data.empty:
            rightSum=np.zeros([len(self.data), self.consPara[0]])
        if hasattr(self, 'preData'):
            PrightSum=np.zeros([len(self.preData), self.consPara[0]])
        #  Caculate the sum of right hand    
        for arg in right:
            self.deriveMapping[left] = [arg, ] + self.deriveMapping.get(left, [left,])
            if not self.data.empty:
                myCol=self._getCols(arg)
                rightSum += myCol.values
            if hasattr(self, 'preData'):
                PmyCol = self._getCols(arg, True)

                PrightSum = PrightSum + PmyCol.values

        # Take the sum of left hand and updata them
        if not self.data.empty:
            myLeft = self._getCols(left)
            myLeft=myLeft.where(myLeft.values > rightSum, rightSum).fillna(myLeft)  # Take tha max of two methods
            self.data[myLeft.columns] = myLeft

        if hasattr(self, 'preData'):
            PmyLeft = self._getCols(left, True)
            PmyLeft = PmyLeft.where((PmyLeft.values > PrightSum), PrightSum).fillna(PmyLeft)  # Take tha max of two methods
            self.preData[PmyLeft.columns] = PmyLeft

    def genProxy(self, isTTA):
        # data must include some specfic columns, the function now is for TTA/TL and TA/TL
        assert self.DEVMODE, 'No reason to generate Proxy if you are not in developing mode' 
        self.isTTA = isTTA
        TCA = self._getCols('Total Current Assets')
        TNCA = self._getCols('Total Non Current Assets')
        TAa = self._getCols('Total Assets')
        TAb = TCA+TNCA.values
        TA = TAa.where(TAa.values > TAb.values, TAb).fillna(TAa) #Take tha max of two methods

        if isTTA:
            IA=self._getCols('Intangible Assets')
            IA[IA.isnull()]=0.0 #isnull() is in Pandas 0.20, isna pandas 0.23
            TA=TA-IA.values
        
        TNCL=self._getCols('Total Non Current Liabilities (Incl Provisions)')
        #TNCLEP=self._getCols('Total Non Current Liabilities (Excl Provisions)')
        # Here can check some data,
        TCL=self._getCols('Total Non Current Liabilities (Incl Provisions)')
        TL=TNCL+TCL.values
        self.proxy = TA.divide(TL.values)
        # self.proxy.columns = 

    def selectDefault(self, colInd=0, fillmissingby=False):
        raise NotImplementedError()

    def transtarget(self, funC):
        self.transFunClass = funC
        self.transFun = funC.fun
        self.transInvFun = funC.invFun
        if self.DEVMODE:  # We do not need Ttarget in using mode
            self.Ttarget = self.target.apply(self.transInvFun)
            self.Ttarget.name = 'Ttarget'

    def deriveCol(self, newName, *Arg):
        ## Arg should be in the format: colname, sign, colname, sign...,colname
        assert len(Arg) % 2 == 1, 'The number of input is not reasonable'
        myNewNames = [' (Period '+str(i+1)+')' for i in range(0, self.consPara[0])] # The period format!
        if not self.data.empty:
            assert newName + myNewNames[0] not in self.data.columns, 'The new columns already exist.'  # The period format!
        if hasattr(self, 'preData'):
            assert newName + myNewNames[
                0] not in self.preData.columns, f"The new columns {newName + myNewNames[0]} already exist."  # The period format!
        for i in range(0, len(myNewNames)):
            myNewNames[i] = newName+myNewNames[i]
            self.inputs.add(myNewNames[i])
        counter = 1
        sign = 1
        if not self.data.empty:
            res = np.zeros([len(self.data), self.consPara[0]])
        if hasattr(self, 'preData'):
            Pres=np.zeros([len(self.preData), self.consPara[0]])
        for arg in Arg:
            if counter == 1:
                self.deriveMapping[newName] = [arg,]+self.deriveMapping.get(newName,[])
                if not self.data.empty:
                    myCol = self._getCols(arg)
                    res += sign*myCol.values
                if hasattr(self, 'preData'):
                    PmyCol = self._getCols(arg,True)
                    Pres = Pres + sign*PmyCol.values
            else:
                if arg == '+':
                    sign = 1
                elif arg == '-':
                    sign = -1
                else:
                    assert False, 'No support for the sign input'
            counter *= -1
        if not self.data.empty:
            respd = pd.DataFrame(res)
            respd.columns = myNewNames
            self.data = pd.concat([self.data,respd],axis=1)
        if hasattr(self,'preData'):
            Prespd = pd.DataFrame(Pres)
            Prespd.columns = myNewNames
            self.preData = pd.concat([self.preData,Prespd],axis=1)

    def macroX(self, Mdata, timeColName, newName):
        macroYears = pd.to_datetime(Mdata.iloc[:,0]).dt.year
        macrodict = dict(zip(macroYears,Mdata.iloc[:,1]))

        if self.DEVMODE:
            timeCol = self._getCol(timeColName)
            lastYears = timeCol.dt.year

            macros=pd.DataFrame(\
             (lambda lastYears=lastYears, macrodict=macrodict, consPara=self.consPara:\
             [(lambda macrodict=macrodict,lastYears=lastYears:\
                          [macrodict.get(k) for k in lastYears-i])()\
                                            for i in range(0,consPara[0])])()).T

            names=(lambda newName=newName:[newName+' (Period '+str(i+1)+')' \
                                          for i in range(0,len(macros.columns))])()

            macros.columns=names
            self.data=pd.concat([self.data,macros],axis=1)

        if hasattr(self,'preData'):
            timeCol=self._getCol(timeColName,True)
            lastYears=timeCol.dt.year


            macros=pd.DataFrame(\
         (lambda lastYears=lastYears, macrodict=macrodict, consPara=self.consPara:\
         [(lambda macrodict=macrodict,lastYears=lastYears:\
                      [macrodict.get(k) for k in lastYears-i])()\
                                        for i in range(0,consPara[0])])()).T

            names=(lambda newName=newName:[newName+' (Period '+str(i+1)+')' \
                                      for i in range(0,len(macros.columns))])()

            macros.columns=names
            self.preData=pd.concat([self.preData,macros],axis=1)

    def genRatio(self, numeratorName, denominatorName=None, historical_length='auto', useLog=False,useBigLog=False,\
                 useDiff=False, sigLevel=0.1, fillingValue=0.0, categorical=False):
        ## useLog means take log for denominator and numerator if they exist
        ## useDiff means use trend of the variables
        # Decide how many columns we will use

        withoutP = self.withoutP
        PwithoutP = withoutP
        if self.iswithin:
            PwithoutP = [1, 2, 3, 4]

        #####Create independent variables#####
        if not self.data.empty:
            ratios = self._calculate_feature(False, numeratorName, denominatorName, useLog, useBigLog, useDiff, categorical)

        # Do the same things for preData
        if hasattr(self, 'preData'):
            Pratios = self._calculate_feature(True, numeratorName, denominatorName, useLog, useBigLog, useDiff, categorical)

        shortNName=''.join([x[0] for x in numeratorName.split()])
        if denominatorName:
            shortDName = ''.join([x[0] for x in denominatorName.split()])
            varName = useDiff*'Trend_' + useLog * 'log' + shortNName + ' Over '\
            + useLog * 'log' + shortDName
        else:
            varName = useDiff * 'Trend_' + useLog * 'log' + shortNName

        #####Create independent variables#####


        #####keep significant variables######
        if isinstance(historical_length, int):
            period = [historical_length + self.yearsAcross - 1]

        elif historical_length=='auto':
            period = historical_length
        elif isinstance(historical_length, list):
            year2add=self.yearsAcross-1
            period = [i + year2add for i in historical_length]
        else:
            assert False, 'Wrong type for historical_length'

        if not self.data.empty:
            tmpColumn = self._reserve_features(period, ratios, varName, withoutP, fillingValue, categorical, numeratorName, denominatorName, useDiff)
            assert not set(tmpColumn.columns).issubset(self.valuableRatios.columns), f'This ratio {tmpColumn.columns} has been generated, do not generate again.'
            self.valuableRatios = pd.concat([self.valuableRatios, tmpColumn], axis=1)

        if hasattr(self,'preData'):
            tmpColumn = self._reserve_features(period, Pratios, varName, PwithoutP, fillingValue, categorical, numeratorName, denominatorName, useDiff)
            assert not set(tmpColumn.columns).issubset(
                self.PvaluableRatios.columns), 'This ratio has been chosen, do not choose again.'
            self.PvaluableRatios=pd.concat([self.PvaluableRatios,tmpColumn], axis=1)

    def pcaTran(self,n_components=8):
        pca = PCA(n_components=n_components,svd_solver='full')
        self.valuables = pca.fit_transform(self.valuableRatios.values)
        if hasattr(self,'preData'):
            self.Pvaluables=pca.transform(self.PvaluableRatios.values)

    def sigRatio(self, numeratorName, denominatorName=None, useLog=False, useBigLog=False, \
                 useDiff=False, sigLevel=0.1, fillingValue=0.0, categorical=False):

        assert self.DEVMODE, 'It is too difficult to do sigRatio because the Pdata is dependent on training data'
        assert self.Ttarget is not None, "The function of transtarget must be choosen"
        # useLog means take log for denominator and numerator if they exist
        # useDiff means use trend of the variables
        # Decide how many columns we will use
        withoutP = range(0, self.yearsAcross)
        if self.yearsAcross + self.yearsTrain < self.consPara[0]:
            withoutP = withoutP.append(range(self.yearsAcross + self.yearsTrain, self.consPara[0]))

        #####Create independent variables#####

        ratios = self._calculate_feature(False, period, numeratorName, denominatorName, useLog, useBigLog, useDiff, categorical)
        # Do the same things for preData
        if hasattr(self, 'preData'):
            Pratios = self._calculate_feature(True, period, numeratorName, denominatorName, useLog, useBigLog, useDiff, categorical)

        shortNName = ''.join([x[0] for x in numeratorName.split()])
        if denominatorName:
            shortDName = ''.join([x[0] for x in denominatorName.split()])
            varName = useDiff * 'Trend_' + useLog * 'log' + shortNName + ' Over ' \
                      + useLog * 'log' + shortDName
        else:
            varName = useDiff * 'Trend_' + useLog * 'log' + shortNName

        #####Create independent variables#####

        #####Regression and keep significant variables######
        for i in range(0, len(ratios.columns)):
            myRatio = ratios.iloc[:, i]
            varNamei = varName + ' Period ' + str(i + 1)
            if not categorical:
                myRatio.name = varNamei
            # df = pd.DataFrame({'target':self.target.values,varNamei:myRatio})
            # reg = linear_model.LinearRegression()
            # reg.fit(df[varNamei].reshape(-1,1),df['target'])
            myRatio = myRatio.replace([np.inf, -np.inf], np.nan)
            myRatio[myRatio.isnull()] = fillingValue

            myRatio2 = sm.add_constant(myRatio)
            est = sm.OLS(self.Ttarget, myRatio2)
            est2 = est.fit()
            # scores, pvalues = chi2(myRatio, self.target.values)
            pValue = est2.pvalues[1]
            self.uniTestTable.loc[len(self.uniTestTable)] \
                = [varNamei, est2.params[1], pValue]
            if (pValue < sigLevel) and (not (i in withoutP)):
                self.valuableRatios = pd.concat([self.valuableRatios, myRatio] \
                                                , axis=1)

                if hasattr(self, 'preData'):
                    PmyRatio = Pratios.iloc[:, i]
                    if not categorical:
                        PmyRatio.name = varNamei
                    PmyRatio[PmyRatio.isnull()] = fillingValue
                    self.PvaluableRatios \
                        = pd.concat([self.PvaluableRatios, PmyRatio], axis=1)


        

