from UnifyData import FinancialSheet
import pandas as pd
import os


def test_FinancialSheet():
    df = pd.DataFrame({'IA 1':[88.8,777],'TA 2':[88888,77777.0],'TCL 2':[88.88,77.77],
                       'TNCA 2':[888.8, 777.77], 'TNCA 1':[888.88, 77.77],'TNCL 1':[8888,777],'TNCL 2':[888.8,777.7],
                       'TA 1': [88888, 7777.77],'WC 1':[888.08,-77.7], 'WC 2':[8888.08,-7777.7],'TCA 1':[88,77],
                       'TCA 2': [888, 777], 'IA 2': [88.88, 77.77], 'TCL 1':[8888.88, 777.7]})
    max_period =10
    Pdata = FinancialSheet(df, mapping_csv=os.path.dirname(os.path.realpath(__file__))+'/UD_name_mapping.csv', max_period=max_period).data
    assert len(Pdata.columns) == max_period*7, 'The columns'


    return Pdata.columns
