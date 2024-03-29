# Quant Decompose

There are four general modules for importing as tools: **FinancialProxyCleanData**, **MachineLearningBasedModels**, **Portfolio** 
and **UnifyData**.

**FinancialProxyCleanData** is aimed at generating proxy, cleaning data, selecting features and generating 'X and Y' 
for mathematical models easily. Using member functions, we can do the above with several lines and they are clear to change and
compare for any length of periods. 

**MachineLearningBasedModels** is purposed to provide a class with which we can train and test machine learning cores with
different transformation classes while partitioning data set with several lines.

**Portfolio** is a module which can output simulated loss distributions and key performance indicators after imputing
PDs, RRs and other necessary parameters. The model is intelligent to deal with the period horizon.

**UnifyData** is to unify different data column names with the same meanings.

## Install

```bash
pip install git+https://github.com/chiraldev/quant_decompose.git
```

Then the environment will have a package called **JSQuantTool**. 

## Import
The way to import the modules in python file:
```python
from FinancialProxyCleanData import DataClass
```
The **DataClass** is already a python class to use directly.

---

```python
from MachineLearningBasedModels import Engines
from MachineLearningBasedModels import TransformationFunctions
```
The **MLM** and **tfun** is a module file containing lots of classes. To use the classes, we can call the python classes like  
**MLM.logisticRegression** and **tfun.segment**.

If we want to import python classes directly, we can import as followings:
```python
from MachineLearningBasedModels.Engines import logisticRegression
from MachineLearningBasedModels.TransformationFunctions import segment
```

---
```python
from Portfolio import CreditPort
```
The **CreditPort** is already a python class to use directly.

---

```python
from UnifyData import FinancialSheet
```
The **FinancialSheet** is already a python class to use directly.

## Uninstall

For uninstall it, we can run the code:
```bash
pip uninstall JSQuantTool
```

If there are small update in git without publishing new version in setup.py. Please uninstall and install.

# Tests
Open the whole project and imput:
```
Pytest
```
