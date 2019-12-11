# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:00:54 2018

@author: JiazhenShen
"""


import inspect
import numpy as np
#inspect.getmembers(sb)
def changeData(data_decompose):
    data_decompose.additionClean("Total Assets", "Total Current Assets", "Total Non Current Assets")
    data_decompose.deriveCol('Total Liabilities', "Total Current Liabilities", '+',
                             "Total Non Current Liabilities (Incl Provisions)")
    data_decompose.deriveCol('Total Tangible Assets', "Total Assets", "-", "Intangible Assets")
#    data11.macroX(RRrd.CPIdata,"Latest Accounts Date",'CPI')
#    data11.macroX(RRrd.inflationdata,"Latest Accounts Date",'Inflation')
#    data11.macroX(RRrd.GDPdata,"Latest Accounts Date",'GDP')


# The followings are for more variables clean. Now we only use the basic variables here.
# def changeData3(data1):
#     tmp=data1.data.iloc[:,(data1.consPara[1]-1):data1.consPara[2]]
#     data1.data=data1.data[np.sum(tmp.isnull().values,1)<(data1.consPara[2]-data1.consPara[1])*0.9]
#     tmp=data1.data.iloc[:,(data1.consPara[1]-1):data1.consPara[2]]
#     X_incomplete_normalized = (tmp.values)/np.max(tmp.values,0)
#     X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)
#     tmp[:]=X_filled_softimpute
#     data1.data.iloc[:,(data1.consPara[1]-1):data1.consPara[2]]=tmp
#
# def changeData2(data1):
#
#     X=data1._getCols("Total Sales / Turnover").values
#     XX=data1._getCols("Total Sales / Turnover")
#     Y=data1._getCols("Total Cost of Sales").values
#     YY=data1._getCols("Total Cost of Sales")
#     Z=data1._getCols("Gross Profit").values
#     ZZ=data1._getCols("Gross Profit")
#
#
#     data1._getCols("Gross Profit")[~((X>0)&(Y>0))]=(XX[~((X>0)&(Y>0))]-YY[~((X>0)&(Y>0))].values).values
#     data1._getCols("Total Cost of Sales")[~((X>0)&(Z>0))]=(XX[~((X>0)&(Z>0))]-ZZ[~((X>0)&(Z>0))].values).values
#     data1._getCols("Total Sales / Turnover")[~((Y>0)&(Z>0))]=(YY[~((Y>0)&(Z>0))]-ZZ[~((Y>0)&(Z>0))].values).values
#
#
#
#     data1.addictionClean("Total Non Current Assets","Leasehold","Freehold","Land & Buildings","Fixtures & Fittings","Plant",\
#     "Vehicles","Plant & Machinery","Other Tangible Assets","Property, Plant & Equipment",\
#     "Intangible Assets","Other Non Current Assets")
#
#     data1.addictionClean("Total Non Current Assets",\
#     "Finished Goods",\
#     "Raw Materials / Stocks",\
#     "Work in Progress",\
#     "Inventories",\
#     "Trade Debtors",\
#     "Group Loans (non-current)",\
#     "Director Loans (non-current)",\
#     "Other Current Receivables",\
#     "Trade And Other Receivables",\
#     "Cash At Bank",\
#     "Other Current Assets")
#
#     data1.addictionClean("Total Assets","Total Current Assets","Total Non Current Assets")
#     data1.addictionClean("Total Assets","Total Equity","Total Non Current Liabilities (Incl Provisions)","Total Current Liabilities")
#     data1.addictionClean("Total Assets","Total Current Assets","Total Non Current Assets")
#
#     new=1*(data1._getCols("Hire Purchase (non-current)")>1)
#     data1.data[new.columns]=new
#
#     new=1*(data1._getCols("Leasing (non-current)")>1)
#     data1.data[new.columns]=new
#
#     new=1*(data1._getCols("Hire Purchase and Leasing (non-current)")>1)
#     data1.data[new.columns]=new
#
#
#
#
#
#
    