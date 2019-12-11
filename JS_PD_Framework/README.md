
# PD Models Framework

--- 
**train/** is a file of different model files where their configurations are set there and the trained model with input names is saved for using later.  
  
The four configuration files include *PDconfig_readData.py*, *PDconfig_changeData.py*, *PDconfig_deriveColumns.py*, *PDconfig_ratios.py*.   
  
The trained model is saved in the format as  *PD{submodelname}{output_horizon}.pkl*. Its input names are saved in a CSV called *PD{submodelname}{output_horizon}inputs.csv*.  
  
---  
  *PD_main_train.ipynb* is the sample notebook of how to train a model. The python file is purposed to use the four configuration files to generate the trained model with input names in the corresponding file.  
    
*PD_main_test.ipynb* is the sample notebook of how to use the trained model. The python file is purposed to get the results of the new inputs written in this file using the trained model  in the corresponding file.  
  
--- 
- *PDconfig_readData.py* is a python file for using different training data (which should be Experian data).  Besides, the final training data frame must be named as *dataPD*.   
- *PDconfig_changeData* is purposed to change the data (Actually clean the [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData) defined in *quant_decompose* module). The file must be a python file containing a function named *changeData()* with a [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData) as the argument and use its methods, such as *additionClean*. To derive new columns with existing columns, the file can contaion a function called deriveColumns() with a [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData) and the modulated  *PDconfig_readData.py* as arguments. The typical function we call here is the *deriveCol()* and *macroX()* of [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData).   
- *PDconfig_ratios* is a file which create features. The file must contain a *ratios()* with a [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData) as the argument. Its member method genRatio is mainly used in this file.

# PDModel
PDModel is a model class using  [*quant_decompose*](https://github.com/chiraldev/quant_decompose)  library in the framework above.

## Public Methods
### 1. Constructor
#### Arguments
- *init_inputs*: Any class/structure which has member data *init_inputs.output_horizon* and *init_inputs.submodel*. Options: 'CLR' - Calibrated Logistical Regression, 'CRF' - Calibrated Random Forest, 'CRFSVM' - Double Moodel of Calibrated Random Forest and support vector machine, 'LR' - Logistical Regression, 'warning' - Logistical Regression removing periods.

- *model_file*: The model file name in **train/** of configurations.

#### Results
- *self.init_inputs* and *self.model_file*: trivial.
- *self.KPI*: key performance indicators, empty dictionary; 
- *self.train_data* and *self.Pdata*: data for training and predicting, empty data frames
- *self.consPara*: constant parameters, a list [5,], which means there are total five periods of training data. (The reason why it is in the format of list is we historically had several parameters which are abandoned now.)
- *self.path*: a string of path to save and name the results. If the machineLearningEngines is 'warning', then the output_horizon will be removed in the name.


---
### 2. train
The train function will use the four configurations and the initialization inputs to train the model. Hence, this method is more like a 'command' than a function.
#### Arguments
- *save*: whether the model pkl and inputs csv are saved, default: True
#### Results
- The model pkl and inputs csv are saved in **\train\\{model_file}\** 

---
### 3. genPdata
Generate predicting data, i.e. put new inputs in it. 
#### Arguments
1. List:
according to the inputs.csv, the arguments should be given by the order into the function.
2. DataFrame:
the inputs columns will be searched according to its column names. They should be unified through the [**UnifyData**](https://github.com/chiraldev/quant_decompose/tree/master/UnifyData).
#### Results
- *self.Pdata* will be a data frame with the inputs columns of all periods if *self.Pdata*  is empty. Then the first line will be the arguments in the corresponding columns and missing value in the others. For example, the inputs csv are A (Period 4), B(Period 2). Then the columns are A (Period 5), A (Period 4), A (Period 3), A (Period 2), A (Period 1), B (Period 5), B (Perid 4) ... If  *self.Pdata*  is not empty, then add one line below it.


---
### 4. use
The use function will use the trained model to predict the *self.Pdata*. This method is more like a 'command' than a function.
#### Results
- *self.KPI* will save the results we are interested in.

 ---
> **The following stuff is not necessary if the training and using models are only the purpose. Following the several lines in the Jupyters and adjusting the lines in configurations can easily train different models and use them for new inputs.  The following is the detailed steps of training and using for further develop.**

##  Steps in *train()*
1. Run the PDconfig_readData.py and assign it to *self.train_data*.
2. Run the private method *self._pre_data_process()* and return the so-called *data1* which belongs to [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData).
3. Run the *data1.genProxy()* (see [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData)).
4.  Run the *data1.selectDefault()* (see [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData)). Now the *data1* has *data1.RR* to train in the machine learning engine. 
> Note: Although this is PDmodel, the target variable is still called 'RR'. The 'RR' here is actually a 1 or 0 depending on whether the sample default. The strange name is a historical reason of developing models. Please Ignore the meaning of RR and view it as the target values.
5.  Run the *PDconfig_ratios.py* with *data1*. Now the *data1* has *data1.valuableRatios*  ready to train in the machine learning engine.
6.  Import different model class  in [*MachineLearningBasedModels*](https://github.com/chiraldev/quant_decompose/tree/master/MachineLearningBasedModels) according to the self.init_inputs.machineLearningEngine which is given when initialise this PD model. Then construct *myModel* with *data1.RR* and a segment transformation function.
7. Run *my_model.xAndPart(data1.valuableRatios)* (see [*MachineLearningBasedModels*](https://github.com/chiraldev/quant_decompose/tree/master/MachineLearningBasedModels))
8. Run *my_model.fit()* (see [*MachineLearningBasedModels*](https://github.com/chiraldev/quant_decompose/tree/master/MachineLearningBasedModels)). Som specific arguments are given here. 
9. Run the private method *self._view_errors()* to plot performance in training and testing sets.
10. Save the *my_model*  as a pkl and inputs required in a csv if *save=True*. The inputs required are very tricky to find. In data1, there will be a member data called *data1.inputs* which contains all the variables reserved when we run *data1.genRatios* in *RRconfig_ratios.py*. However, they are not always the raw inputs if we derive some columns by others. Hence, we record the raw variables of each derived variable in a dictionary *data1.derivaMapping*. Accordingly, we replace the derived ones with raw variables firstly and then save them as a csv.

##  Steps in *use()*
1. Because we want to ignore training again to save time, we set *self.train_data* as empty to confirm the training training mode will not start. 
2. Run the private method *self._pre_data_process()* and return the so-called *data1* which belongs to [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData) which is similar with training.
3.  Run the *PDconfig_ratios.py* with *data1*. Now the *data1* has *data1.PvaluableRatios*  ready as new feature inputs in the machine learning engine.
4. Read the trained model
5. Get the predictions and put them in the KPI

## Private Methods  
### *_pre_data_process()*
This private methods are both called in *use()* and *train()*
1. Initialize PD [*dataClass*](https://github.com/chiraldev/quant_decompose/tree/master/FinancialProxyCleanData) (Check whether it is *warning* model firstly)
2. Run *PDconfig_changeData*
3. Run *PDconfig_deriveColumn*

### *_initPdata(names)*
This method is called in *genPdata()*. Put any name list like *['A (Period 2)', 'B (Period 5)']* into the method and it will return a data frame only with column names: *['A (Period 1)', 'A (Period 2)', 'A (Period 3)', ... , 'B (Period 1)',...]*. 

### _view_errors(model1, myModel)
This is the static member function for plotting error measures. *model1* using 20% as the testing set while myModel using 90%.
We plot ROC (Receiver Operating Characteristics) Curves of them. The area under curve (AUC) has range ([0.5-1]). The higer, the better.
We also plot the AUC of the grey samples, which is defined as the samples whose predictions are larger than the smallest default sample's PD and smaller than the largest active sample's default.
The calibration curve measure how the probability value is similar with the true frequency.

