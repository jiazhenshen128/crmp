import os
import dill
import importlib
import csv
from FinancialProxyCleanData import DataClass
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA


class PDDataClass(DataClass):
    def selectDefault(self, colInd=0, fillmissingby=False):
        assert self.DEVMODE, 'No reason to select default companies in the current vision if you are not in developing mode'
        assert self.proxy is not None, "Generate proxy firstly"

        # Select colInd column in self.proxy as the target candidate
        target1 = self.proxy.iloc[:, colInd]
        target1[target1 == 0] = 0.0001  # 0 -> a small number
        target1[target1 == 1] = 0.9999  # 1 -> be a limitation number

        # We do not use macro variables now, so it is not needed
        # accountingYear = pd.DatetimeIndex(self._getCol("Latest Accounts Date").values).year

        TS = self._getCol("Trading Status")  # Active / no-active label
        # We only keep samples with some standard, although there may be clean enough.
        # Here, the PD models, we keep the samples whose proxy exist and number missing values is smaller than 6.
        # We also remove non-active samples whose proxy is smaller than 1. These unusual companies are not useful as
        # they are not our clients.
        self.targetposition = target1.notnull() & ((np.sum(self.data.isnull().values, 1) < 6)) & (
                    (TS == "Active").values | ((TS != "Active").values & (
                        target1 < 1)))  # & (target1 <  1) & (target1 >  0) #0and1've been changed
        assert sum(
            self.targetposition) > 0, 'No samples left after cleaning. The qualification of samples is too strict!'
        # Target is whether its proxy is < 1 and it is non-active
        self.target = ((target1 < 1) & (target1 > 0) & (TS != "Active").values)[
            self.targetposition]  # self.target = target1[self.targetposition]
        self.target.name = 'target'
        # Every time we select the default and get target,  we need to initialize
        # the valueableRatios and uniTestTable

        # uniTestTable is not used now because we remove the function of significant testing
        # self.uniTestTable = pd.DataFrame({'Variable':[], 'Coeff':[], 'PValue':[]})
        # self.uniTestTable.columns = ['Variable', 'Coeff', 'PValue']
        self.valuableRatios = pd.DataFrame()
        if hasattr(self, 'preData'):
            self.PvaluableRatios = pd.DataFrame()

        if fillmissingby:  # not safe codes! transform choose the float columns automatically,  might be problem
            d = self.data.loc[self.targetposition]  # self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes == 'float64']].transform(lambda x: x.fillna(x.mean()))
            self.data.loc[self.targetposition, d.columns[d.dtypes == 'float64']] = dd
            d = self.data.loc[self.targetposition]  # , self.data.columns[self.consPara[1]:]]
            dd = d.groupby(fillmissingby)[d.columns[d.dtypes == 'float64']].transform(
                lambda x: scipy.stats.mstats.winsorize(x, limits=0.05, inplace=True))
            self.data.loc[self.targetposition, d.columns[d.dtypes == 'float64']] = dd

            if hasattr(self, 'preData'):
                pass  # not good way to do this

        if self.iswithin:
            loopn = self.consPara[0] - len(self.withoutP)

            self.target = pd.Series(np.tile(self.target.values, loopn))

        assert sum(self.target) > 0, 'No bankrupt company samples left'
        assert sum(self.target) < len(self.target), 'No active company samples left'


class PDModel:
    def __init__(self, init_inputs, model_file):
        """
        :param init_inputs: Any class/structure which has member data *init_inputs.output_horizon*
        and *init_inputs.machineLearningEngine*. (We do not input the members as arguments directly into the model just
        because it was more convenient in Django. All are historical reasons)
        :param model_file: The model file name in **train/** of configurations.
        # Results
        - *self.init_inputs* and *self.model_file*: trivial.
        - *self.KPI*: key performance indicators, empty dictionary;
        - *self.train_data* and *self.Pdata*: data for training and predicting, empty data frames
        - *self.consPara*: constant parameters, a list [5,], which means there are total five periods of training data. (The reason why it is in the format of list is we historically had several parameters which are abandoned now.)
        - *self.path*: a string of path to save and name the results. If the submodes is 'warning', then the output_horizon will be removed in the name.
        """
        self.init_inputs = init_inputs
        self.model_file = model_file
        self.KPI = dict()
        self.train_data = pd.DataFrame()
        self.Pdata = pd.DataFrame()

        print('machineLearningEngine: ', self.init_inputs.machineLearningEngine)
        print('output_horizon: ', self.init_inputs.output_horizon)
        # Because warning model do not have the time horizon, hence the name of saving files should do not contain it.
        foo = '' if self.init_inputs.machineLearningEngine == 'warning' else str(self.init_inputs.output_horizon)
        self.path = os.path.dirname(os.path.realpath(
            __file__)) + '/train/' + self.model_file + '/PD' + self.init_inputs.machineLearningEngine + foo
        PDrd = importlib.import_module("train." + self.model_file + ".PDconfig_readData")
        consPara0 = PDrd.max_period if hasattr(PDrd, 'max_period') else 5
        consPara1 = " (Period #)"
        self.consPara = (consPara0, consPara1)

    def _pre_data_process(self):
        """
        This model is construct a corresponding DataClass and run the PDconfig_changeData.
        """
        # 1. Check whether it is *warning* model
        self.init_inputs.iswithin = (self.init_inputs.machineLearningEngine == 'warning')

        # 2. Construct PD dataClass
        data1 = PDDataClass(input_historical=self.consPara[0] - self.init_inputs.output_horizon, output_horizon=self.init_inputs.output_horizon,
                               trainData=self.train_data, preData=self.Pdata, dataNote="Company Set A",
                               iswithin=self.init_inputs.iswithin, consPara=self.consPara)

        # 3. Run PDconfig_changeData
        RRcd = importlib.import_module("train." + self.model_file + ".PDconfig_changeData")
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
        # 1. Run the PDconfig_readData.py and assign it to *self.train_data*.
        PDrd = importlib.import_module("train." + self.model_file + ".PDconfig_readData")
        self.train_data = PDrd.readData()

        # 2. Run the private method *self._pre_data_process()* and return
        # the so-called *data1* which belongs to [*dataClass*]
        data1: PDDataClass = self._pre_data_process()

        # 3.Generate the proxy: Run the *data1.genProxy()*
        data1.genProxy(True)  # using TTA/TA as the proxy

        # 4. Run the *data1.selectDefault()* data
        data1.selectDefault()

        # 5. Run the *PDconfig_ratios.py* with *data1*.
        # Now the *data1* has *data1.valuableRatios* ready to train in the machine learning engine.
        PDdsrt = importlib.import_module("train." + self.model_file + ".PDconfig_ratios")
        PDdsrt.ratios(data1)

        # 6.Training
        # My_model: partitionRatio=0.9 for main_use, model1: partitionRatio=0.9
        from MachineLearningBasedModels import Engines
        from MachineLearningBasedModels import TransformationFunctions
        print('Samples of default companies for machine learning core: ', np.sum(data1.target==1))
        print('Samples of active companies for machine learning core: ', np.sum(data1.target ==0))
        if self.init_inputs.machineLearningEngine == 'CLR':
            model1 = Engines.CalibratedlogisticRegression(data1.target, TransformationFunctions.identity())
            my_model = Engines.CalibratedlogisticRegression(data1.target, TransformationFunctions.identity())
        elif self.init_inputs.machineLearningEngine == 'CRF':
            model1 = Engines.CalibratedRandomForestC(data1.target, TransformationFunctions.identity())
            my_model = Engines.CalibratedRandomForestC(data1.target, TransformationFunctions.identity())
        elif self.init_inputs.machineLearningEngine == 'CRFSVM':
            model1 = Engines.DoubleRandomForestC(data1.target, TransformationFunctions.identity())
            my_model = Engines.DoubleRandomForestC(data1.target, TransformationFunctions.identity())
        elif self.init_inputs.machineLearningEngine == 'XGB':
            model1 = Engines.XGBoost(data1.target, TransformationFunctions.identity())
            my_model = Engines.XGBoost(data1.target, TransformationFunctions.identity())            
        elif self.init_inputs.machineLearningEngine == 'LR' or self.init_inputs.machineLearningEngine == 'warning':
            model1 = Engines.logisticRegression(data1.target, TransformationFunctions.identity())
            my_model = Engines.logisticRegression(data1.target, TransformationFunctions.identity())
        else:
            assert False, 'No sub-model for it'
        model1.xAndPart(data1.valuableRatios, partitionRatio=0.7)
        model1.fit()

        my_model.xAndPart(data1.valuableRatios, partitionRatio=0.9)
        my_model.fit()

        # 7.View Errors
        self._view_errors(model1, my_model)
        print("Training Completed!")

        # 8. Save the *my_model*  as a pkl and inputs required in a csv if *save=True*.
        # The inputs required are very tricky to find. In data1,
        # there will be a member data called *data1.inputs* which contains all the variables reserved
        # when we run *data1.genRatios* in *PDconfig_ratios.py*.
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
            print("Model Saved!")
        print("--------------------------------End--------------------------------")
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
        # which is similar with training.
        data1 = self._pre_data_process()

        # 3. Run the *PDconfig_ratios.py* with *data1*.
        # Now the *data1* has *data1.PvaluableRatios* ready as new feature inputs in the machine learning engine.
        PDdsrt = importlib.import_module("train." + self.model_file + ".PDconfig_ratios")
        PDdsrt.ratios(data1)
        # 4. Read the trained model
        with open(self.path + '.pkl', 'rb') as f:
            my_model = dill.load(f)

        # 5. Get the predictions and put them in the KPI

        self.KPI['prePDs'] = (100 * my_model.predictProb(data1.PvaluableRatios)[:, 1]).tolist()
        # Ignore the followings
        # self.KPI['prePDsWithin'] *= (1 - my_model._predictProb(data1.PvaluableRatios)[:, 1])
        # self.KPI['prePDsWithin'] = (100 * (1 - self.KPI['prePDsWithin'])).tolist()
        # self.KPI['numprePDs'] = [(i + 1) for i in range(0, len(self.KPI['prePDs']))]
        # self.KPI['companyName'] = list(Pdata.loc[:, 'Company Name'].values)
        return self.KPI

    @staticmethod
    def _view_errors(model1, my_model):
        from matplotlib import pyplot as plt
        from sklearn.calibration import calibration_curve

        sb = model1._predictProb(model1.x_te)
        plt.subplot(2, 1, 1)
        plt.hist(sb[model1.yBench_te==1][:,1])
        plt.title('PD predictions hist of default testing companies')
        plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(sb[model1.yBench_te==0][:,1])
        plt.title('PD predictions hist of active testing companies')
        plt.show()
        # plt.plot(np.transpose(sb))
        sb11, sb2, sb3 = model1.trainError("AUC")
        plt.title('Model1 Training Set')
        plt.plot(sb2, sb3, label="Training Set-AUC:" + str(sb11))

        sb12, sb2, sb3 = model1.testError("AUC")
        plt.plot(sb2, sb3, label="Testing Set-AUC:" + str(sb12))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title("AUC of large testing set")

        plt.legend()
        plt.plot(sb2, sb2, '--')
        plt.show()


        # sb11, sb2, sb3 = model1.trainError("AUC_Cut")
        # plt.plot(sb2, sb3, label="Training Set-AUC:" + str(sb11))
        #
        # sb12, sb2, sb3 = model1.testError("AUC_Cut")
        # plt.plot(sb2, sb3, label="Testing Set-AUC:" + str(sb12))
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title("AUC of grey samples of large testing set")
        # plt.legend()
        # plt.plot(sb2, sb2, '--')
        # plt.show()

        # model1.errorNote = (sb11, sb12)
        # fop, mpv = calibration_curve(model1.y_te.values, sb[:, 1], n_bins=10)
        # # plot perfectly calibrated
        # plt.plot([0, 1], [0, 1], linestyle='--')
        # # plot model reliability
        # plt.plot(mpv, fop, marker='.')
        # plt.title("Calibration Curve")
        # plt.show()
        # print((lambda sb: ([sum(sb[:, 1] > (i + 1) * 1 / 10) for i in range(0, 10)]))(sb))
        # print("------------------------------------------")

        # sb = my_model._predictProb(my_model.x_te)
        # sb11, sb2, sb3 = my_model.trainError("AUC_Cut")
        # plt.plot(sb2, sb3, label="Training Set-AUC:" + str(sb11))
        #
        # sb12, sb2, sb3 = my_model.testError("AUC_Cut")
        # plt.plot(sb2, sb3, label="Testing Set-AUC:" + str(sb12))
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.legend()
        # plt.plot(sb2, sb2, '--')
        # plt.title("AUC of small testing set")
        # plt.show()
        #
        print("AUC-Train: ", round(sb11, 2))
        print("AUC-Test: ", round(sb12, 2))
        #
        # sb = my_model._predictProb(my_model.x_te)
        # sb11, sb2, sb3 = my_model.trainError("AUC")
        # plt.plot(sb2, sb3, label="Training Set-AUC:" + str(sb11))
        #
        # sb12, sb2, sb3 = my_model.testError("AUC")
        # plt.plot(sb2, sb3, label="Testing Set-AUC:" + str(sb12))
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.legend()
        # plt.plot(sb2, sb2, '--')
        # plt.title("AUC of grey samples of small testing set")
        # plt.show()
        #
        # print("AUC-Train: " + str(sb11))
        # print("AUC-Test: " + str(sb12))
        # my_model.errorNote = (sb11, sb12)
        #
        # # reliability diagram
        # fop, mpv = calibration_curve(my_model.y_te.values, sb[:, 1], n_bins=10)
        #
        # # plot perfectly calibrated
        # plt.plot([0, 1], [0, 1], linestyle='--')
        # # plot model reliability
        # plt.plot(mpv, fop, marker='.')
        # plt.title("Calibration Curve")
        # plt.show()
        # print((lambda sb: ([sum(sb[:, 1] > (i + 1) * 1 / 10) for i in range(0, 10)]))(sb))


        # sb = my_model._predictProb(my_model.x_te)
        # sb11, sb2, sb3 = my_model.trainError("AUC_Cut")
        # plt.plot(sb2, sb3, label="Training Set-AUC:" + str(sb11))
        #
        # sb12, sb2, sb3 = my_model.testError("AUC_Cut")
        # plt.plot(sb2, sb3, label="Testing Set-AUC:" + str(sb12))
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.legend()
        # plt.plot(sb2, sb2, '--')
        # plt.show()


        # my_model.errorNote = (sb11, sb12)

        # sb = my_model._predictProb(my_model.x_te)
        # sb11, sb2, sb3 = my_model.trainError("AUC")
        # plt.plot(sb2, sb3, label="Training Set-AUC:" + str(sb11))
        #
        # sb12, sb2, sb3 = my_model.testError("AUC")
        # plt.plot(sb2, sb3, label="Testing Set-AUC:" + str(sb12))
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.legend()
        # plt.plot(sb2, sb2, '--')
        # plt.show()
        #
        # print("AUC-Train of my model: " + str(sb11))
        # print("AUC-Test of my model: " + str(sb12))
        # my_model.errorNote = (sb11, sb12)
        #
        # # reliability diagram
        # fop, mpv = calibration_curve(my_model.y_te.values, sb[:, 1], n_bins=10)
        # # plot perfectly calibrated
        # plt.plot([0, 1], [0, 1], linestyle='--')
        # # plot model reliability
        # plt.plot(mpv, fop, marker='.')
        # plt.show()
        # print((lambda sb: ([sum(sb[:, 1] > (i + 1) * 1 / 10) for i in range(0, 10)]))(sb))
# model1.rfc.fit(model1.x_tr,model1.y_tr)
# importances=model1.rfc.feature_importances_
# indices = np.argsort(importances)
# features = model1.x_tr.columns
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()