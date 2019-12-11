# Machine Learning Based Models
This models are high-level interface classes mainly based on **sklearn**. The models can realise the functions including **splitting data** (into training set and testing set), **fitting** (calibrating coefficients), **predicting** (classification and regression)and **testing performances** with data which has *transforming functions* 
> **Note:** Though the model was developed for recovery rate models, it can be also used in PD models and even other machine-learning based prediction models. If the machine learning cores are not coded, it is not difficult to extend the class.


## Engine Construction
Inputs:
1. A ***pandas.dataframe*** as the target data, i.e. the observations we want machines to learn and to predict.
2. A specific ***self-made class*** instance which has two member functions: **fun** and **invFun**. The transformation functions are provided and can be added in the *TransformationFunctions.py*.


Outputs:
1. **self.yBench**: equal to dataClass.RR
4. **self.y**: equal to dataClass.invFun(dataClass.RR)
5. **self.transFunClass**: equal to dataClass.transFunClass
	

## xAndPart
This member function is to put explanatory variables in the class and partition it as training part and testing part. 

The inputs of the function has **x**, which are a ***pandas.dataframe***  whose length must be equal to the length of **self.y**. The table **x** are the expalnatory variables for machine learning core.

It also has **partitionRatio** and **fillingvalue** as arguments to decide the percentage of the training set and the filling value for missing data in **x**.

The outputs are: **self.x_tr**, **self.x_te**, **self.y_tr**, **self.y_te**, **self.yBench_tr**, **self.yBench_te**.

## Fit
This member function is "pure virtual". It must be implemented in the derived class. 

Hence, one of the main differences between sub-classes, for example, ***logisticRegression*** and ***CalibratedRandomForestC*** is the **fit** function.

The arguments and parameters of this function is not considered in the production version.

## Predict
The prediction is probabilities for classification and values for regression. The output is the transformation function's values of outputs from the machine learning cores.

> **Note:** If we use transformation function like segments to change the regression problem as the classification problem, then we can use 'max' and 'mean' to use different way for outputing value. We use max as default in this function, but use mean as
default in Error Revaluation below, If we want the error of 'max' method, use 'maxMAE' instead of 'MAE'. 

## Error Revaluation (TrainError, TestError)
If *method=‘MAE’*, it will output the mean absolute error of model and the naive method (Using mean of the training data). 

If *method=‘AUC’*, it will output the area under the curve, false  positive rate, true positive rate.

If method is equal to the others, it is not a usual usage. Ignore them.

## Samples
Please read **tests/ML_samples.ipynb**.

 

