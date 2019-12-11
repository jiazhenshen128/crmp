# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 20:07:07 2018

@author: JiazhenShen
"""

import numpy as np
import bisect


class sigmoid:
    def fun(self,v):
        return 1/(1+np.exp(-v))

    def invFun(self,v):
        return(-np.log(1/v-1))


class identity:    
    def fun(self,v):
        return v

    def invFun(self,v):
        return v


class segment:
    def __init__(self, Nclass=10):
        self.Nclass=Nclass
        self.C=np.array(range(0,self.Nclass+1))/self.Nclass
        self.vfun=np.vectorize((lambda x,\
                           a=self.C:bisect.bisect_left(a, x, lo=0, hi=len(a))
                           ))

    def invFun(self,v):
        return(self.vfun(v))

    def fun(self,v):    
        return (self.C-1/self.Nclass/2)[(lambda v=v:[int(np.round(x)) for x in v])()]


class segment2:
    def __init__(self,Nclass=10):
        self.Nclass=Nclass
        self.C=np.array(range(0,self.Nclass+1))/self.Nclass
        self.C=np.insert(self.C, 1, 0.001)
        self.C=np.insert(self.C, len(self.C)-1, 0.999)
        self.Nclass+=2
        self.vfun=np.vectorize((lambda x,\
                           a=self.C:bisect.bisect_left(a, x, lo=0, hi=len(a))
                           ))

    def invFun(self,v):
        return(self.vfun(v))

    def fun(self,v):    
        return (self.C-1/self.Nclass/2)[(lambda v=v:[int(np.round(x)) for x in v])()]


