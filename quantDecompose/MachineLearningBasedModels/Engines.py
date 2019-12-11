# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 19:07:09 2018

@author: JiazhenShen
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel


class Models:
    def __init__(self, targets, transFunClass):
        #self.dataClass.PvaluableRatios = None  # This is not safe and make developers confusing, need to improve
        self.transFunClass = transFunClass
        self.yBench = pd.Series(targets.values)
        self.yBench.name = targets.name
        self.y = pd.Series(self.yBench.apply(self.transFunClass.invFun).values)
        self.y.name = 'TRR'
        self.isClass = False
        self.x = None
        self.lastPvaluableRatios = None
        self.train = None
        self.x_tr = None
        self.x_te = None
        self.y_tr = None
        self.y_te = None
        self.yBench_tr = None
        self.yBench_te = None

    def xAndPart(self, x, partitionRatio=0.8, fillingValue=0.0):
        assert len(self.y)==len(x), 'The explantary variables have \
        incorrect length'
        self.x = pd.DataFrame(x.values)
        self.x[self.x.isnull()] = fillingValue
        self.x.columns = x.columns
        # self.dataf=pd.concat([self.y,x],axis=1)
        # Partition
        permuted_indices = np.random.permutation(len(self.x))
        inTrain = permuted_indices[:int(len(permuted_indices)*partitionRatio)]
        inTest = permuted_indices[int(len(permuted_indices)*partitionRatio):]
        self.x_tr = self.x.iloc[inTrain]
        self.x_te = self.x.iloc[inTest]
        self.y_tr = self.y.iloc[inTrain]
        self.y_te = self.y.iloc[inTest]
        self.yBench_tr = self.yBench.iloc[inTrain]
        self.yBench_te = self.yBench.iloc[inTest]

    def fit(self):
        raise NotImplementedError()

    def _predict(self,x):
        self.lastPvaluableRatios = x
        return self.train.predict(x)
    
    def _predictProb(self,x):
        self.lastPvaluableRatios = x
        return self.train.predict_proba(x)
    
    def predict(self, x, class2valueFun = ''):
        if self.isClass & (class2valueFun == 'mean'):
            tmpx = self._predictProb(x)
            tmpy = self.transFunClass.C[1:len(self.transFunClass.C)] - \
                   1 / self.transFunClass.Nclass / 2
            return (lambda lx=tmpx, ly=tmpy: [sum(lx[i] * ly) for i in range(0, len(tmpx))])()
        else:
            return self.transFunClass.fun(self._predict(x))

    def predictProb(self,x):
        if self.isClass:
            self.lastPvaluableRatios = x
            return self._predictProb(x)

    def _error(self, method, x, yBench):
        if method == 'MAE':
            pre = self.predict(x, class2valueFun='mean')
            preNaive = np.mean(yBench)

            error = pre - yBench
            errorNaive = preNaive - yBench

            MAE = np.mean(abs(error))
            MAENaive = np.mean(abs(errorNaive))
            return [MAE, MAENaive]
        if method == 'maxMAE':
            pre = self.predict(x, class2valueFun='max')
            preNaive = np.mean(yBench)

            error = pre - yBench
            errorNaive = preNaive - yBench

            MAE = np.mean(abs(error))
            MAENaive = np.mean(abs(errorNaive))
            return [MAE, MAENaive]
        if method == "AUC":
            pre = self._predictProb(x)
            fpr, tpr, thresholds = metrics.roc_curve(yBench.values.astype(int) + 1, pre[:, 1], pos_label=2)
            return (metrics.auc(fpr, tpr), fpr, tpr)
        if method == "AUC_Cut":
            pre = self._predictProb(x)[:, 1]
            real = yBench.values.astype(int)
            cut = min(pre[real == 1])
            pre_cut = pre[pre >= cut]
            real_cut = real[pre >= cut]

            fpr, tpr, thresholds = metrics.roc_curve(real_cut + 1, pre_cut, pos_label=2)
            return metrics.auc(fpr, tpr), fpr, tpr

    def trainError(self, method='MAE'):
            return self._error(method, self.x_tr, self.yBench_tr)

    def testError(self, method='MAE'):
            return self._error(method, self.x_te, self.yBench_te)


from sklearn import linear_model
class linearModel(Models):
    def fit(self):
        self.train=linear_model.LinearRegression()
        self.train.fit(self.x_tr,self.y_tr)
        
     
from sklearn.ensemble import RandomForestRegressor
class randomForest(Models):
    def fit(self,umax_depth=5,ucriterion='mae'):
        self.train=RandomForestRegressor(max_depth=umax_depth,random_state=0,\
                                         criterion=ucriterion,n_estimators=10)
        self.train.fit(self.x_tr,self.y_tr)
        
      
from sklearn import svm
class CalibratedsvmSvc(Models):
    def __init__(self, targets, transFunClass):
        Models.__init__(self, targets, transFunClass)
        self.isClass=True

    def fit(self,kernel='linear', x_tr=pd.DataFrame(), y_tr=pd.DataFrame(), cv=5, C=0.5):
        a = svm.SVC(C=C, kernel=kernel,probability=True)
        self.train=  CalibratedClassifierCV(a, method='sigmoid')
        
        if not(x_tr.empty and y_tr.empty):
            self.x_tr = x_tr
            self.y_tr = y_tr
        self.train.fit(self.x_tr, (self.y_tr))


from sklearn.linear_model import LogisticRegression
class logisticRegression(Models):
    def __init__(self, targets, transFunClass):
        Models.__init__(self, targets, transFunClass)
        self.isClass=True

    def fit(self, x_tr=pd.DataFrame(), y_tr=pd.DataFrame()):
        if not(x_tr.empty and y_tr.empty):
            self.x_tr = x_tr
            self.y_tr = y_tr
        lr = LogisticRegression(random_state=0, penalty = 'l1', C=10, solver= 'liblinear')
        self.train = lr
        self.train.fit(self.x_tr, self.y_tr)


class CalibratedlogisticRegression(Models):
    def __init__(self, targets, transFunClass):
        Models.__init__(self, targets, transFunClass)
        self.isClass=True

    def fit(self, x_tr=pd.DataFrame(), y_tr=pd.DataFrame()):
        #self.train=LogisticRegression(random_state=0, penalty = 'l1', C=10, class_weight = 'balanced')

        #self.train=lr
        if not(x_tr.empty and y_tr.empty):
            self.x_tr = x_tr
            self.y_tr = y_tr
        lr=LogisticRegression(random_state=0, penalty = 'l1', C=10, class_weight = 'balanced')
        self.train=CalibratedClassifierCV(lr, method='isotonic', cv=int(np.log(len(self.y_tr))))
        self.train.fit(self.x_tr, self.y_tr)


from sklearn.ensemble import RandomForestClassifier
class randomForestC(Models):
    def __init__(self, targets, transFunClass):
        Models.__init__(self, targets, transFunClass)
        self.isClass=True

    def fit(self,x_tr=pd.DataFrame(), y_tr=pd.DataFrame(),umax_depth=5,n_estimators=100, class_weight='balanced_subsample'):
        self.train=RandomForestClassifier(max_depth=umax_depth,random_state=0,\
                                          n_estimators=n_estimators, min_samples_leaf=0.02, class_weight = class_weight)
        if not(x_tr.empty and y_tr.empty):
            self.x_tr = x_tr
            self.y_tr = y_tr
        self.train.fit(self.x_tr,self.y_tr)


        
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
class CalibratedRandomForestC(Models):
    def __init__(self,targets, transFunClass):
        Models.__init__(self,targets, transFunClass)
        self.isClass=True

    def fit(self,x_tr=pd.DataFrame(), y_tr=pd.DataFrame(), criterion='gini', umax_depth=3,n_estimators=100, method='sigmoid', cv=5, class_weight = 'balanced_subsample'):
        if not(x_tr.empty and y_tr.empty):
            self.x_tr = x_tr
            self.y_tr = y_tr
        #rfc=RandomForestClassifier(criterion=criterion, max_depth=umax_depth,\
        #                                  n_estimators=n_estimators, class_weight = class_weight)
        #sel=SelectFromModel(rfc)
        #sel.fit(self.x_tr,self.y_tr)
        #self.x_tr=self.x_tr.iloc[:,sel.get_support()]
        #self.x_te=self.x_te.iloc[:,sel.get_support()]
        rfc=RandomForestClassifier(criterion=criterion, max_depth=umax_depth,\
                                          n_estimators=n_estimators, class_weight = class_weight)
        self.rfc = rfc
        self.train=CalibratedClassifierCV(rfc, method=method, cv=cv)
        #self.train=CalibratedClassifierCV(rfc, method='isotonic', cv=2)
        #self.train = rfc
        self.train.fit(self.x_tr,self.y_tr)

        
class DoubleRandomForestC(Models):
    def __init__(self, targets, transFunClass):
        Models.__init__(self, targets, transFunClass)
        self.isClass=True
        self.model1=CalibratedRandomForestC(targets, transFunClass)
        self.model2=CalibratedsvmSvc(targets, transFunClass)
    
    def fit(self,umax_depth=3,n_estimators=100):
        self.model1.fit(x_tr=self.x_tr, y_tr=self.y_tr, umax_depth=umax_depth, n_estimators=100, cv=int(np.log(len(self.y_tr))))
        pre=self.model1._predictProb(self.x_tr)[:,1]
        self.cut = np.percentile(pre[self.y_tr==1], 25)
        self.cut2= np.percentile(pre[self.y_tr==1], 60)
        self.y_tr_cut = self.y_tr[pre>=self.cut]
        self.x_tr_cut = self.x_tr[pre>=self.cut]
        self.model2.fit(x_tr=self.x_tr_cut, y_tr=self.y_tr_cut, cv=int(np.log(len(self.y_tr_cut))))
    
    def _predictProb(self,x):
        res1 = self.model1._predictProb(x)
        tmp_index = (res1[:,1] >= self.cut2).tolist()
        if sum(tmp_index)>0:
            res1[tmp_index,:] = self.model2._predictProb(x[tmp_index])
        return res1 

