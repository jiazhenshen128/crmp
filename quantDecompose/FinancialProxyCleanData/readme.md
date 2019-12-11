# Financial Proxy and Clean Data Model
This models are the prepared process from the financial information to the data ready for prediction models. The model can **clean the data**, **generate proxies**, **generate prediction targets**,  and **select features** (This function is not updated, hence unsafe in the current version). 

This pre-process model is designed for the synchronization of the training data and daliy data.

The ***RR_dataClass*** and ***PD_dataClass***  are two subclasses of the ***dataClass***.


## Initialisation
**Inputs**:
1. **input_historical**:  a ***integer*** of how many years using as inputs in the model.
 5. **output_horizon**:   a ***integer*** of how many years using for predictions.
 6. **trainData**: (Optional) a ***pandas dataframe*** where each row is one company sample. The data is prepared for If it is not given, the model is not in developing mode.
3.  **preData**:  (Optional) a ***pandas dataframe*** for prediction. This data should be in the same format of the **data**, so that we can do the same thing with the data. In developing mode, this argument can be not given. 
 3. **dataNote**: (Optional) any ***string*** information for main in plots and other marks.  
 4.  **consPara**: a ***tuple*** containing some parameters for settings. The first element is a ***integer*** of how many years the data have in total. The second one is the format of presentation periods, e.g. " (Period #)".

> **Note**: The sum of **yearsTrain** and **yearsAcross** must not be larger than the  first element in the  **consPara**. 


## additionClean
Due to the wrong consistent variables in the data, we will use this function to make sure some formulas hold.

For example, if we want to make sure A=B+C+D+...., then we use code: **additionClean("A", "B", "C", "D", ...)** where A, B,.. are variables such as "Total Assets". But the function actually make A=max(A, B+C+....). We believe the larger one is more trustworthy.

## genProxy (inflexible function)
Due to the usual lack of the target indicators, we use this function to generate the proxy for indicators. The function can only be called in developing mode. 

In the ***RR_dataClass*** and ***PD_dataClass*** , the **genProxy** are the same one derived from the **genProxy** which can calculate **Total Assets/( Total Non Current Liabilities (Incl Provisions) + 'Total Current Liabilities)** or **Total tangible assets/( Total Non Current Liabilities (Incl Provisions) + 'Total Current Liabilities)**. 

The proxy generated is saved in **self.proxy**. The proxy is very crude in these stage. It is generated in all periods without filter any companies.

> **Note**: This function is not flexible but depends on what data is used and how proxy is defined. Therefore, this function is expected to be overloaded by the derived class.

## selectDefault
This function is pure virtual and it must be implemented in derived class. It can only be called after calling **genProxy** in developing mode.

Basically, the function will give the **self.RRposition** which is a ***bool array*** indexing the filtered companies. For instance, in ***RR_dataClass***, the **self.RRposition** index the company whose $0 \le self.proxy \le 1$ and data in some specific given columns are not missing. 

The main output here is **self.RR** which is the target values we hope to predict. 

> **Note**: The name RR used here is because the model is initially coded for RR models. But now models may be extended in many applications, hence, the word "RR" here can be understood as the word "target" in predictions.

## deriveCol
This member function can add new columns by addiction and difference operations with existed variables.

For instance, **deriveCol("new", "A", "+", "B", "-". "C", "-", ...)**, we can get a column whose name is "new" and the value is A+B-C-....

## Genrate Features
### genRatio
This function do not filter the features but reserve all the features generated in the right periods.

The inputs: 
**numeratorName**, a ***string*** like "Total Assets";
**denominatorName** (optional), a ***string*** like "Total Liabilities", if not given, it can be viewed as 1；
**historical_length**, an ***int*** or ***list of int***, e.g. if you want use 2017, 2018 to predict 2020, the it should be equal to [2, 3].
**useLog**, **true** or **false** to take log of numerator and denominator (if given)；
**useBigLog**, **true** or **false** to take log of numerator over denominator (if given)；
**useDiff**, **true** or **false** to take trend of numerator or denominator (if given)；
**categorical**, **true** or **false** to make numerator as dummy variables without denominator；

Outputs:
If the training data or prediction data is given, **self.valuableRatios** or **self.PvaluableRatios** can be generated respectively.

>**Note**: The right periods in the training data  are training periods. But in the prediction data, the model have moved the data in the initilization. So it will keep the most recent periods. For instance, using three years data to predict the target after two years will keep features of period 5, 4, 3 in training data but features of period 3, 2, 1 in the predicting data.

### sigRatio
The function can keep the significant ratio but the function can only be used in developing mode. The producting mode require further developed. 

### macroX
This function is to use macro variables as features. The method is not updated. It requires further developed before using it.

## pcaTran
The function transform **self.valuableRatios** or **self.PvaluableRatios** with PCA and keep the most principle **n_components** components.

## Samples
Please see tests/FP_samples.