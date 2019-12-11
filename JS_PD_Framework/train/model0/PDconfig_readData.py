# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:26:39 2018

@author: JiazhenShen
"""

import os
import sys
import pandas as pd


def readData():
    dataCons1 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData/A_raw_merged.xlsx")
    dataCons2 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData/LRD_raw_merged.xlsx")
    dataPD=pd.concat([dataCons1,dataCons2])
    dataPD.index=range(0,len(dataPD))
    return dataPD

# The following function is to change the sic code to the sector name according a SIC Mapping.xlsx. Because we find the
# the sectors are not significant, we do not use it.

# def sic2sector(data, colname):
#     SIC = data.loc[:, colname]
#     SIC_Mapping = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/TrainingData/0 - SIC Mapping.xlsx", header=None,
#                                 col_types=["text", "text", "text", "text",
#                                            "text", "skip", "skip", "skip",
#                                            "skip", "skip"])
#     sc = SIC_Mapping.iloc[1:, 1]
#     sc.index = range(0, len(sc))
#
#     let = SIC_Mapping.iloc[3:23, 4]
#     let.index = range(0, len(let))
#
#     startNum = [0, ]
#
#     for i in range(0, len(let)):
#         ll = let[i]
#         s = "Section " + str(ll.upper())
#         ind = sc[sc == s].index
#         startNum.append(sc.iloc[ind + 1].iloc[0])
#     letters = ['0', ]
#     letters += list(let.values)
#     secStart = pd.DataFrame({"startNum": startNum, "letters": letters})
#
#     sic = []
#     for i in range(0, len(SIC)):
#         s = SIC[i]
#         try:
#             sic.append(secStart.loc[sum(int(s) >= secStart["startNum"]) - 1, "letters"])
#         except ValueError:
#             sic.append(secStart.loc[sum(0 >= secStart["startNum"]) - 1, "letters"])
#
#     data.loc[:, colname] = sic

# sic2sector(dataPD1, 'UK SIC Code')
