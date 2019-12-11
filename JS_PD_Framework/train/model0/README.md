# PD Model0 Configurations
These configurations are four files with four functions, which will be called in  *PDModel.py*. These functions are actually "inline" functions, i.e. the implementations can be copied and pasted directly in the *PDModel.py*. This design is purposed to separate these implementations with the others in *PDModel.py* so that people can easily change these configurations to train and use different models, independent with other codes.

## PDconfig_readData.py
In this model, we use *A_raw_merged.xlsx* and *LRD_raw_merged.xlsx* in the *TrainingData/* file as the training data set. The final result is dataPD. The codes are only several lines as following:

``` python 
dataCons1 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData/A_raw_merged.xlsx")  
dataCons2 = pd.read_excel(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/TrainingData/LRD_raw_merged.xlsx")  
dataPD = pd.concat([dataCons1,dataCons2])  
```

## PDconfig_changeData.py
Here, we only to take the maximum of total assets and the sum total current assets and total non current assets for cleaning data. The process will be used for all periods. The details of the function is in [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData). The one line codes are as following:

``` python 
def changeData(data_decompose):  
    data_decompose.additionClean("Total Assets", "Total Current Assets", "Total Non Current Assets")
```

## PDconfig_deriveColumns.py
Because we only want to calculate total liabilities and total tangible assets in this model, the two lines are enough to do it. The *deriveCol()* is also one of method in  [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData).  The two line codes are as following:

``` python 
def deriveColumns(data_decompose):  
    data_decompose.deriveCol('Total Liabilities', "Total Current Liabilities", '+', "Total Non Current Liabilities (Incl Provisions)")  
    data_decompose.deriveCol('Total Tangible Assets', "Total Assets", "-", "Intangible Assets" )
```


## PDconfig_ratios.py
We select some features by trials, including log(TA). TCA/TTA, TNCA/TTA, EBITDA/TTA, TCL/TTA, WC/TTA, ... All available periods of them are choosen in this model. The features are constructed by the member method *genRatios()* of [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData).  The features are created in one line respectively as following: 


```  python 
def ratios(data_decompose):
    data_decompose.genRatio('Total Assets', useLog=True)  
    data_decompose.genRatio("Total Current Assets", "Total Tangible Assets")  
    data_decompose.genRatio("Total Non Current Assets", "Total Tangible Assets")  
    data_decompose.genRatio("EBITDA", "Total Tangible Assets")    
    data_decompose.genRatio("Total Current Liabilities", "Total Tangible Assets")  
    data_decompose.genRatio("Total Liabilities", "Total Tangible Assets")  
    data_decompose.genRatio("Working Capital", "Total Tangible Assets")  
    data_decompose.genRatio("Property, Plant & Equipment", "Total Liabilities")  
    data_decompose.genRatio('Total Assets', 'Total Liabilities')  
    data_decompose.genRatio('Total Tangible Assets', 'Total Liabilities')  
    data_decompose.genRatio('Total Liabilities', 'Total Assets')
```