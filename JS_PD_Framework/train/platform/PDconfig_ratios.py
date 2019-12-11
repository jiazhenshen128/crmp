# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:57:04 2018

@author: JiazhenShen
"""

def ratios(data_decompose):
    data_decompose.genRatio("Total Sales / Turnover", "Total Assets", historical_length=1)
    data_decompose.genRatio('Total Assets', 'Total Liabilities', historical_length=1)
    data_decompose.genRatio("EBITDA", "Total Assets", historical_length=1)
    data_decompose.genRatio("Working Capital", "Total Assets", historical_length=1)
    data_decompose.genRatio("Retained Earnings", "Total Assets", historical_length=1)
#    data_decompose.genRatio("Total Liabilities", "EBITDA", historical_length=1)
#     data_decompose.genRatio("EBITDA", "Total Sales / Turnover", useLog=True, useDiff=True, historical_length=1)
#    data_decompose.genRatio("Property, Plant & Equipment", "Total Liabilities", historical_length=1)
#    data_decompose.genRatio("Cash At Bank", "Total Assets", historical_length=1)
#    data_decompose.genRatio('Borrowings (current)', "Total Tangible Assets")
#    data_decompose.genRatio('Borrowings (non-current)', "Total Tangible Assets")
#    data_decompose.genRatio('Total Tangible Assets', "Total Tangible Assets")
#     data_decompose.genRatio('Total Assets', useLog=True, historical_length=1)
#     data_decompose.genRatio("Total Current Assets", "Total Tangible Assets")
#     data_decompose.genRatio("Total Non Current Assets", "Total Tangible Assets")

#    data_decompose.genRatio("EBITDA", "Total Assets", useLog=True)
#     data_decompose.genRatio("EBITDA", "Total Assets", useDiff=True)
#    data_decompose.genRatio("Total Sales / Turnover", "Total Assets", useDiff=True)

#    data_decompose.genRatio("Total Non Current Liabilities (Incl Provisions)", "Property,  Plant & Equipment")
#    data_decompose.genRatio("Total Non Current Liabilities (Excl Provisions)", "Total Tangible Assets")
#      data_decompose.genRatio("Total Current Liabilities", "Total Tangible Assets")
#     data_decompose.genRatio("Total Liabilities", "Total Tangible Assets")

#    data_decompose.genRatio("Cash At Bank", "Working Capital")
#    data_decompose.genRatio("Cash At Bank", 'Total Tangible Assets')


#     data_decompose.genRatio('Total Tangible Assets', 'Total Liabilities')
#     data_decompose.genRatio('Total Liabilities', 'Total Assets', historical_length=1)
    
#    data_decompose.genRatio("UK SIC Code", categorical=True)
#    data_decompose.genRatio("CPI")
#    data_decompose.genRatio("CPI", useDiff=True)
#    data_decompose.genRatio("Inflation")
#    data_decompose.genRatio("Inflation", useDiff=True)
#    data_decompose.genRatio("GDP")
#    data_decompose.genRatio("GDP", useDiff=True)
