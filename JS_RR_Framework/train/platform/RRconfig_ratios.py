# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:28:30 2018

@author: JiazhenShen
"""

def ratios(data_decompose,sigLevel=0.1):

    data_decompose.genRatio('Total Assets', 'Total Liabilities', historical_length=1)

    data_decompose.genRatio("Total Non Current Liabilities (Incl Provisions)", "Total Assets", historical_length=1)

    # data_decompose.genRatio('UK SIC Code', categorical=True)
