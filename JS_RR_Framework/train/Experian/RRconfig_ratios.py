# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:28:30 2018

@author: JiazhenShen
"""

def ratios(data_decompose,sigLevel=0.1):
    data_decompose.genRatio('Total Assets', 'Total Liabilities')

    data_decompose.genRatio('Total Tangible Assets', 'Total Liabilities')

    data_decompose.genRatio('Total Assets', useLog=True)

    data_decompose.genRatio("Total Tangible Assets", "Total Assets")

    data_decompose.genRatio("Working Capital", "Total Assets")

    data_decompose.genRatio("Total Current Liabilities", "Total Assets")

    data_decompose.genRatio("Total Non Current Liabilities (Incl Provisions)", "Total Assets")
    
  #  data_decompose.genRatio("Total Non Current Assets", "Total Assets", useLog=True)
