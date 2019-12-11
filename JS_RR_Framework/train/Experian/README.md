# RR Model0 Configurations  
These configurations are four files with four functions, which will be called in  *RRModel.py*. These functions are actually "inline" functions, i.e. the implementations can be copied and pasted directly in the *RRModel.py*. This design is purposed to separate these implementations with the others in *RRModel.py* so that people can easily change these configurations to train and use different models, independent with other codes.  
  
## RRconfig_readData.py  
In this model, we use *Bankrupt_AllSizes.xlsx* as the training data set. The final result is dataRR. The codes are only several lines. 
  
  
## RRconfig_changeData.py  
Here, we only to take the maximum of total assets and the sum total current assets and total non current assets for cleaning data. The process will be used for all periods. The details of the function is in [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData). The one line codes are as following:  
  
``` python 
def changeData(data_decompose):    
    data_decompose.additionClean("Total Assets", "Total Current Assets", "Total Non Current Assets")  
```  
  
## RRconfig_deriveColumns.py  
Because we only want to calculate total liabilities and total tangible assets in this model, the two lines are enough to do it. The *deriveCol()* is also one of method in  [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData).  The two line codes are as following:  
  
``` python 
def deriveColumns(data_decompose):    
    data_decompose.deriveCol('Total Liabilities', "Total Current Liabilities", '+', "Total Non Current Liabilities (Incl Provisions)")    
    data_decompose.deriveCol('Total Tangible Assets', "Total Assets", "-", "Intangible Assets" )  
```  
  
  
## RRconfig_ratios.py  
We select some features by trials, including TA/TL, TTA/TL, log(TA), TTA/TA, WC/TA, ... All available periods of them are choosen in this model. The features are constructed by the member method *genRatios()* of [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData).  The features are created in one line respectively as following:   
  
  
```  python 
def ratios(data_decompose):  
    data_decompose.genRatio('Total Assets', 'Total Liabilities')    
    data_decompose.genRatio('Total Tangible Assets', 'Total Liabilities')  
    data_decompose.genRatio('Total Assets', useLog=True)  
    data_decompose.genRatio("Total Tangible Assets", "Total Assets")   
    data_decompose.genRatio("Working Capital", "Total Assets")  
    data_decompose.genRatio("Total Current Liabilities", "Total Assets")   
    data_decompose.genRatio("Total Non Current Liabilities (Incl Provisions)", "Total Assets")
```