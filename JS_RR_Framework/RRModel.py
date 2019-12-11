import os
import dill
import importlib
import csv
from FinancialProxyCleanData import DataClass
import pandas as pd
import scipy.stats
import numpy as np


class RRDataClass(DataClass):
    def selectDefault(self, colInd=0, fillmissingby=False):
        assert self.DEVMODE, 'No reason to select default companies in the cutargetent vision if you are not in developing mode'
        assert self.proxy is not None, "Generate proxy firstly"

        # Select colInd column in self.proxy as the target candidate
        target1 = self.proxy.iloc[:, colInd]
        target1[target1 == 0] = 0.0001  # 0 -> a small number
        target1[target1 == 1] = 0.9999  # 1 -> be a limitation number

        # We only keep samples with some standard, although there may be clean enough.
        # Here, the RR models, we keep the samples whose proxy exist and number of missing values is limited.
        # But cleanning data here is not neceesary and core purposes here.
        # We also remove the volunteer brankrupt companies.
        missingn = np.sum(self._getCols('Total Assets').isnull().values, 1) + np.sum(
            self._getCols('Total Current Liabilities').isnull().values, 1)
        missingn += np.sum(self._getCols('Working Capital').isnull().values, 1) + np.sum(
            self._getCols('Total Non Current Liabilities (Incl Provisions)').isnull().values, 1)
        self.targetposition = target1.notnull() & (target1 < 1) & (target1 > 0) & (
                    missingn < 1)  # &(np.sum(self.data.isnull().values, 1) < 6)#0and1've been changed

        # Target is its proxy in valid position
        self.target = target1[self.targetposition]
        self.target.name = 'target'
        # Every time we select the default and get target, we need to initialize
        # the valueableRatios and uniTestTable
        # self.uniTestTable = pd.DataFrame({'Variable':[], 'Coeff':[], 'PValue':[]})
        # self.uniTestTable.columns = ['Variable', 'Coeff', 'PValue']
        self.valuableRatios = pd.DataFrame()
        if hasattr(self, 'preData'):
            self.PvaluableRatios = pd.DataFrame()  # Now it is initialized in __init__, so no worry about not running this method in using Mode

        if fillmissingby:  # not safe codes! transform choose the float columns automatically, might be problem
            d = self.data.loc[self.targetposition]  # self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes == 'float64']].transform(lambda x: x.fillna(x.mean()))
            self.data.loc[self.targetposition, d.columns[d.dtypes == 'float64']] = dd
            d = self.data.loc[self.targetposition]  # ,self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes == 'float64']].transform(
                lambda x: scipy.stats.mstats.winsorize(x, limits=0.05, inplace=True))
            self.data.loc[self.targetposition, d.columns[d.dtypes == 'float64']] = dd

            if hasattr(self, 'preData'):
                pass  # not good way to do this


class RRModel:
    def __init__(self, init_inputs, model_file):

        """
        :param init_inputs: Any class/structure which has member data *init_inputs.output_horizon*.
        (We do not input the members as arguments directly into the model just because it was more convenient in Django.
        All are historical reasons)
        :param model_file: The model file name in **train/** of configurations.

        #### Results
        - *self.init_inputs* and *self.model_file*: trivial.
        - *self.KPI*: key performance indicators, empty dictionary;
        - *self.train_data* and *self.Pdata*: data for training and predicting, empty data frames
        - *self.consPara*: constant parameters, a list [5,], which means there are total five periods of
        training data. (The reason why it is in the format of list is we historically had several parameters
        which are abandoned now.)
        - *self.path*: a string of path to save and name the results.
        """
        self.init_inputs = init_inputs
        self.init_inputs.machineLearningEngine = self.init_inputs.machineLearningEngine if hasattr(self.init_inputs,'machineLearningEngine') else 'CRF'
        self.model_file = model_file
        self.KPI = dict()
        self.train_data = pd.DataFrame()
        self.Pdata = pd.DataFrame()
        self.path = os.path.dirname(
            os.path.realpath(__file__)) + '/train/' + self.model_file + '/RR' + str(self.init_inputs.machineLearningEngine) + str(self.init_inputs.output_horizon)

        RRrd = importlib.import_module("train." + self.model_file + ".RRconfig_readData")
        consPara0 = RRrd.max_period if hasattr(RRrd, 'max_period') else 5
        consPara1 = " (Period #)"
        self.consPara = (consPara0, consPara1)

    def _pre_data_process(self):
        """
         This model is construct a corresponding DataClass and run the PDconfig_changeData.
        """
        # if self.init_inputs.document:
        #    self.Pdata = pd.read_excel(self.init_inputs.document.path)

        # 1.Construct PD dataClass
        data1 = RRDataClass(input_historical=5 - self.init_inputs.output_horizon, output_horizon=self.init_inputs.output_horizon,
                               trainData=self.train_data, preData=self.Pdata, dataNote="Company Set A", consPara=self.consPara)
        # 1.5 Make some changes to the data; We fill some total assets of Pdata by TCA and TNCA,
        # it is strange here, need more research!

        RRcd = importlib.import_module("train." + self.model_file + ".RRconfig_changeData")
        # from train.premodel import PD_changeData as RRcd
        RRcd.changeData(data1)

        return data1

    def _initPdata(self, names):
        NameSet = set()

        for n in names:
            NameSet.add(n.split(' (Period')[0])

        for n in NameSet:
            for i in range(0, self.consPara[0]):
                self.Pdata[(n+f" (Period {i+1})")] = []

    def genPdata(self, *args):
        """
        Generate predicting data, i.e. put new inputs in it.
        :param args: According to the inputs.csv, the arguments should be given by the order
        :return: - *self.Pdata* will be a data frame with the inputs columns of all periods if *self.Pdata*  is empty.
        Then the first line will be the arguments in the corresponding columns and missing value in the others.
        For example, the inputs csv are A (Period 4), B(Period 2). Then the columns are A (Period 5), A (Period 4),
        A (Period 3), A (Period 2), A (Period 1), B (Period 5), B (Perid 4) ...
        If  *self.Pdata*  is not empty, then add one line below it.
        """
        df = pd.read_csv(self.path + 'inputs.csv')

        if self.Pdata.empty:
            self._initPdata(df.columns)

        d = dict()

        #if not isinstance(args, pd.DataFrame):
        for arg in args:
            if not isinstance(arg, pd.DataFrame):
                for k in df.columns:
                    d[k] = float(arg)
        if len(d)!=0:
            df = df.append(d,ignore_index=True,sort=True)

        for arg in args:
            if isinstance(arg, pd.DataFrame):
                for row in arg.index:
                    d = dict()
                    for k in arg.columns:
                        if k in df.columns:
                            d[k] = arg[k][row]
                    df = df.append(d, ignore_index=True, sort=True)
        self.Pdata = self.Pdata.append(df, ignore_index=True, sort=True)
        self.Pdata_notnull = self.Pdata[self.Pdata.columns[self.Pdata.notnull().values[0]]]

    def train(self, save=False):
        """
        The train function will use the four configurations and the initialization inputs to train the model.
        Hence, this method is more like a 'command' than a function.
        :param save: whether the model pkl and inputs csv are saved, default: True
        results: - The model pkl and inputs csv are saved in **\train\\{model_file}\**
        """

        # 1. Run the RRconfig_readData.py and assign it to *self.train_data*.
        RRrd = importlib.import_module("train." + self.model_file + ".RRconfig_readData")
        self.train_data = RRrd.readData()

        # 2. Run the private method *self._pre_data_process()*
        # and return the so-called *data1* which belongs to [*dataClass*]
        data1 = self._pre_data_process()

        # 3.Generate the proxy: Run the *data1.genProxy()*
        data1.genProxy(True)  # using TTA/TA as the proxy

        # 4.Select the default data: Run the *data1.selectDefault()*
        data1.selectDefault()

        # 5.Run the *RRconfig_ratios.py* with *data1*.
        # Now the *data1* has *data1.valuableRatios*  ready to train in the machine learning engine.
        RRdsrt = importlib.import_module("train." + self.model_file + ".RRconfig_ratios")
        RRdsrt.ratios(data1)

        # 6.Training
        from MachineLearningBasedModels import Engines 
        from MachineLearningBasedModels import TransformationFunctions
        print('Samples of default companies for machine learning core: ', len(data1.target))

        if self.init_inputs.machineLearningEngine == 'LR':
            model1 = Engines.logisticRegression(data1.target, TransformationFunctions.segment(6))
            my_model = Engines.logisticRegression(data1.target, TransformationFunctions.segment(6))
        elif self.init_inputs.machineLearningEngine == 'CRF':
            model1 = Engines.CalibratedRandomForestC(data1.target, TransformationFunctions.segment(6))
            my_model = Engines.CalibratedRandomForestC(data1.target, TransformationFunctions.segment(6))
        else:
            assert False, 'No such model'

        model1.xAndPart(data1.valuableRatios)
        my_model.xAndPart(data1.valuableRatios, partitionRatio=0.9)
        if self.init_inputs.machineLearningEngine == 'CRF':
            model1.fit(umax_depth=3, n_estimators=1000, class_weight=None, method='isotonic',criterion = "entropy")
            my_model.fit(umax_depth=3, n_estimators=1000, class_weight=None, method='isotonic',criterion = "entropy")
        else:
            model1.fit()
            my_model.fit()

        # 7.Run the private method *self._view_errors()* to plot performance in training and testing sets.
        self._view_errors(model1, my_model)
        print('Training Completed!')

        # 8. Save the *my_model*  as a pkl and inputs required in a csv if *save=True*.
        # The inputs required are very tricky to find. In data1,
        # there will be a member data called *data1.inputs* which contains all the variables reserved
        # when we run *data1.genRatios* in *RRconfig_ratios.py*.
        # However, they are not always the raw inputs if we derive some columns by others.
        # Hence, we record the raw variables of each derived variable in a dictionary *data1.derivaMapping*.
        # Accordingly, we replace the derived ones with raw variables firstly and then save them as a csv.
        if save:
            with open(self.path + '.pkl', 'wb') as f:
                dill.dump(my_model, f)

            old_inputs = data1.inputs.copy()
            for c in old_inputs:
                if c.split(' (Period')[0] in data1.deriveMapping:
                    data1.inputs.remove(c)
                    for nc in data1.deriveMapping[c.split(' (Period')[0]]:
                        data1.inputs.add(nc + ' (Period' + c.split(' (Period')[1])
            cw = csv.writer(open(self.path + 'inputs.csv', 'w'))
            cw.writerow(list(data1.inputs))
            print('Model Saved!')
        return my_model, data1

    def use(self):
        """
        The use function will use the trained model to predict the *self.Pdata*.
        This method is more like a 'command' than a function.
        :return: *self.KPI* will save the results we are interested in.
        """
        # 1. Because we want to ignore training again to save time,
        # we set *self.train_data* as empty to confirm the training training mode will not start.
        self.train_data = pd.DataFrame()

        # 2. Run the private method *self._pre_data_process()*
        # and return the so-called *data1* which belongs to [*dataClass*]
        data1 = self._pre_data_process()
        RRdsrt = importlib.import_module("train." + self.model_file + ".RR_ratios")

        # 3.  Run the *RRconfig_ratios.py* with *data1*.
        # Now the *data1* has *data1.PvaluableRatios*  ready as new feature inputs in the machine learning engine.
        RRdsrt.ratios(data1)

        # 4. Read the trained model
        with open(self.path + '.pkl', 'rb') as f:
            my_model = dill.load(f)

        # 5. Get the predictions and put them in the KPI
        self.KPI['preRRdists'] = (100*my_model._predictProb(data1.PvaluableRatios)).tolist()
        self.KPI['preRRs'] = (100 * np.array(my_model.predict(data1.PvaluableRatios))).tolist()
        self.KPI['RRintervals'] = 50/len(self.KPI['preRRdists'][0])+100/len(self.KPI['preRRdists'][0])*np.array([i for i in range(0, len(self.KPI['preRRdists'][0]))])
        self.KPI['RRintervals'] = self.KPI['RRintervals'].tolist()
        self.KPI['numpreRRs'] = [(i + 1) for i in range(0, len(self.KPI['preRRs']))]

        # print(self.KPI)
        return self.KPI

    @staticmethod
    def _view_errors(model1, my_model):
        from matplotlib import pyplot as plt
        from sklearn.calibration import calibration_curve
        print('The MAEs of the model1, the naive method in the training set are {}'.format(model1.trainError()))

        print('The MAEs of the model1, the naive method in the testing set are {}'.format(model1.testError()))

        print('The MAEs of the my_model, the naive method in the training set are {}'.format(my_model.trainError()))

        print('The MAEs of the my_model, the naive method in the testing set are {}'.format(my_model.testError()))

        model1.errorNote = (my_model.trainError(), my_model.testError())

        sb = model1._predictProb(my_model.x_tr)
        plt.plot(np.transpose(sb))
        plt.xlabel("Ty")
        plt.ylabel("Probability")
        plt.title(
            'The MAEs of the model: {:.2f}'.format(my_model.testError()[0], 2) + ', the naive method: {:.2f}'.format(
                my_model.testError()[1], 2))
        plt.show()
