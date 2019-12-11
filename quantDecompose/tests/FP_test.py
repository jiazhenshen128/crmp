import pandas as pd
import os
from tests import PD_dataClass, RR_dataClass


class TestClass:
    def test_read_data(self):
        self.data = pd.read_excel(os.path.dirname(os.path.realpath(__file__))+"/TrainingData/Bankrupt_AllSizes_1.xlsx")

    def test_init(self):
        self.test_read_data()
        self.data1 = RR_dataClass.DataClass(input_historical=3, output_horizon=2, trainData=self.data)
        self.data2 = PD_dataClass.DataClass(input_historical=3, output_horizon=2, trainData=self.data)


    def test_additionClean(self):
        self.test_init()
        self.data1.data["Total Assets (Period 3)"][4] = 1000
        assert (self.data1.data["Total Assets (Period 3)"][4] !=
                (self.data1.data["Total Current Assets (Period 3)"][4] +
                 self.data1.data["Total Non Current Assets (Period 3)"][4])), \
            'The data can not be replaced deliberately, hence cannot conduct the test'
        self.data1.additionClean("Total Assets", "Total Current Assets", "Total Non Current Assets")
        assert (self.data1.data["Total Assets (Period 3)"][4] ==
                (self.data1.data["Total Current Assets (Period 3)"][4] +
                 self.data1.data["Total Non Current Assets (Period 3)"][4])), \
            'The equation is still not equal after running the function'

        self.data2.data["Total Assets (Period 3)"][4] = 1000
        assert (self.data2.data["Total Assets (Period 3)"][4] !=
                (self.data2.data["Total Current Assets (Period 3)"][4] +
                 self.data2.data["Total Non Current Assets (Period 3)"][4])), \
            'The data can not be replaced deliberately, hence cannot conduct the test'
        self.data2.additionClean("Total Assets", "Total Current Assets", "Total Non Current Assets")
        assert (self.data2.data["Total Assets (Period 3)"][4] ==
                (self.data2.data["Total Current Assets (Period 3)"][4] +
                 self.data2.data["Total Non Current Assets (Period 3)"][4])), \
            'The equation is still not equal after running the function'

    def test_genProxy(self):
        self.test_additionClean()
        self.data1.genProxy(isTTA=False)
        self.data1.genProxy(isTTA=True)

        self.data2.genProxy(isTTA=False)
        self.data2.genProxy(isTTA=True)

    def test_deriveCol(self):
        self.test_genProxy()
        self.data1.deriveCol('Total Liabilities', "Total Current Liabilities", '+',
                        "Total Non Current Liabilities (Incl Provisions)")
        self.data1.deriveCol('Total Tangible Assets', "Total Assets", "-", "Intangible Assets")
        self.data1.deriveCol('Net Assets', "Total Assets", "-", "Total Current Liabilities", "-",
                        "Total Non Current Liabilities (Incl Provisions)")
        assert not self.data1._getCols('Total Liabilities').empty, 'Cannot get the new variable'

        self.data2.deriveCol('Total Liabilities', "Total Current Liabilities", '+',
                        "Total Non Current Liabilities (Incl Provisions)")
        self.data2.deriveCol('Total Tangible Assets', "Total Assets", "-", "Intangible Assets")
        self.data2.deriveCol('Net Assets', "Total Assets", "-", "Total Current Liabilities", "-",
                        "Total Non Current Liabilities (Incl Provisions)")
        assert not self.data2._getCols('Total Liabilities').empty, 'Cannot get the new variable'


    def test_selectDefault(self):
        self.test_deriveCol()
        self.data1.selectDefault()
        self.data2.selectDefault()

    def test_chooseRatio(self):
        self.test_selectDefault()
        self.data1.genRatio('Total Assets', 'Total Liabilities')
        self.data1.genRatio('Total Assets', useLog=True)

        self.data2.genRatio('Total Assets', 'Total Liabilities')
        self.data2.genRatio('Total Assets', useLog=True)

    def test_production_mode(self):
        data = pd.read_excel(os.path.dirname(os.path.realpath(__file__))+"/TrainingData/Bankrupt_AllSizes_1.xlsx")
        pdata = pd.read_excel(os.path.dirname(os.path.realpath(__file__))+"/TrainingData/Company list Data.xlsx")
        data12 = RR_dataClass.DataClass(input_historical=3, output_horizon=2, trainData=data, preData=pdata)
        data12.additionClean("Total Assets", "Total Current Assets", "Total Non Current Assets")
        data12.genProxy(isTTA=True)
        data12.deriveCol('Total Liabilities', "Total Current Liabilities", '+',
                        "Total Non Current Liabilities (Incl Provisions)")
        data12.deriveCol('Total Tangible Assets', "Total Assets", "-", "Intangible Assets")
        data12.selectDefault()
        data12.genRatio('Total Assets', 'Total Liabilities')
        data12.genRatio('Total Assets', useLog=True)


        data22 = RR_dataClass.DataClass(input_historical=3, output_horizon=2, trainData=data, preData=pdata)
        data22.additionClean("Total Assets", "Total Current Assets", "Total Non Current Assets")
        data22.genProxy(isTTA=True)
        data22.deriveCol('Total Liabilities', "Total Current Liabilities", '+',
                        "Total Non Current Liabilities (Incl Provisions)")
        data22.deriveCol('Total Tangible Assets', "Total Assets", "-", "Intangible Assets")
        data22.selectDefault()
        data22.genRatio('Total Assets', 'Total Liabilities')
        data22.genRatio('Total Assets', useLog=True)

