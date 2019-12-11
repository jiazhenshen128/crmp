# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:07:46 2018

@author: JiazhenShen
"""
import pandas as pd
import os

def sic2sector(data, colname):
    SIC = data.loc[:, colname]
    SIC_Mapping = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData//0 - SIC Mapping.xlsx", header=None)
    sc = SIC_Mapping.iloc[1:, 1]
    sc.index = range(0, len(sc))

    let = SIC_Mapping.iloc[3:23, 4]
    let.index = range(0, len(let))

    startNum = [0, ]

    for i in range(0, len(let)):
        l = let[i]
        s = "Section " + str(l.upper())
        ind = sc[sc == s].index
        startNum.append(sc.iloc[ind + 1].iloc[0])
    letters = ['0', ]
    letters += list(let.values)
    secStart = pd.DataFrame({"startNum": startNum, "letters": letters})

    sic = []
    for i in range(0, len(SIC)):
        s = SIC[i]
        try:
            sic.append(secStart.loc[sum(int(s) >= secStart["startNum"]) - 1, "letters"])
        except ValueError:
            sic.append(secStart.loc[sum(0 >= secStart["startNum"]) - 1, "letters"])

    mappingA = {'a': 'A', 'b': 'B', 'c': 'A', 'd': 'B', 'e': 'B', 'f': 'B', 'g': 'A', 'h': 'B', 'i': 'C', 'j': 'C',
                'k': 'C', 'l': 'C', 'm': 'C', 'n': 'C', 'o': 'A', 'p': 'C', 'q': 'C', 'r': 'C', 's': 'C', 't': '0',
                'u': '0', '0':'0'}
    # # mappingA = {'a': 'A', 'b': 'B', 'c': 'B', 'd': 'B', 'e': 'B', 'f': 'B', 'g': 'C', 'h': 'C', 'i': 'C', 'j': 'C',
    # #             'k': 'C', 'l': 'C', 'm': 'C', 'n': 'C', 'o': 'D', 'p': 'D', 'q': 'D', 'r': 'D', 's': 'o', 't': '0',
    # #             'u': '0', '0':'0'}
    Sic = [mappingA[i] for i in sic]
    data.loc[:, colname] = Sic

def readData():
    dataL1 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData/Bankrupt_AllSizes_1.xlsx",
                        col_types = ["text", "text", "text",
                                      "text", "text", "text", "text", "date",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric"])

    dataL2 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData/Bankrupt_AllSizes_2.xlsx",
                        col_types = ["text", "text", "text",
                                      "text", "text", "text", "text", "date",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric"])


    dataL3 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData/Bankrupt_AllSizes_3.xlsx",
                        col_types = ["text", "text", "text",
                                      "text", "text", "text", "text", "date",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric"])


    dataL4 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData//Bankrupt_AllSizes_4.xlsx",
                        col_types = ["text", "text", "text",
                                      "text", "text", "text", "text", "date",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric", "numeric", "numeric",
                                      "numeric"])

    dataRR=pd.concat([dataL1,dataL2,dataL3,dataL4])
    dataRR.index=range(0,len(dataRR))


    # sic2sector(dataRR,'UK SIC Code')

    # print(dataRR['UK SIC Code'])

    return dataRR


import pandas as pd                               # For munging tabular data
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from UnifyData import FinancialSheet
import os

#
# def readData():
#     snow_user = os.getenv("snow_user")
#     snow_pwd = os.getenv("snow_pwd")
#     print(snow_user)
#
#     engine = create_engine(URL(
#         account='xb33235.eu-west-1',
#         user=snow_user,
#         password=snow_pwd,
#         database='ANALYTICS',
#         schema='ANALYTICS',
#         warehouse='reporting',
#         role='ANALYST',
#     ))
#
#     cs = engine.connect()
#
#     # Get the total ID we have
#     sql = f"SELECT * from ANALYTICS_BASE.EXPERIAN_RR_BANKRUPT"
#     df1 = pd.read_sql_query(sql, cs)
#     mapping_csv = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))) + '/name_mapping.csv'
#     df2 = FinancialSheet(df1, mapping_csv, max_period=1).data
#     df2.fillna(value=pd.np.nan, inplace=True)
#     for col in df2:
#         try:
#             df2[col] = df2[col].astype(float)
#         except:
#             continue
#     return df2

