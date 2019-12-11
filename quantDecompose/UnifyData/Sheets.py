import pandas as pd
import numpy as np
import os


class FinancialSheet:
    "A start at creating a balance sheet"
    def __init__(self, df, mapping_csv, max_period = 10):
        """
        :param TA: latest historical total assets in SMECAPITAL_ORIGINATION_TEMPFINANCIALS
        :param TL: latest historical total liabilities in SMECAPITAL_ORIGINATION_TEMPFINANCIALS
        :param WC: latest historical total working capital in SMECAPITAL_ORIGINATION_TEMPFINANCIALS
        :param RP: latest historical retained profit in SMECAPITAL_ORIGINATION_TEMPFINANCIALS
        :param OP: latest historical operating profit in SMECAPITAL_ORIGINATION_TEMPFINANCIALS
        :param TS: latest historical Total Sales/ Revenue/Turnover in SMECAPITAL_ORIGINATION_TEMPFINANCIALS
        :param EBITDA_LA: latest historical adjusted EBITDA in SMECAPITAL_ORIGINATION_TEMPFINANCIALS
        """

        self.data = pd.DataFrame()
        name_mapping = pd.read_csv(mapping_csv, header=None, index_col=0).T
        mapping_dict = dict()
        for value in name_mapping:
            for key in name_mapping[value]:

                mapping_dict[key] = value
            mapping_dict[value] = value

        for col in df:
            col_name = df[col].name
            period = ''.join(list(filter(str.isdigit, col_name)))
            if period != '':
                col_name = col_name.replace(period, "#")
            if col_name in mapping_dict:
                standard_name = mapping_dict[col_name]
                standard_name_period = standard_name.replace("#", period)
                self.data[standard_name_period] = df[col]
                if period != '':
                    for i in range(1, max_period+1):
                        standard_name_period = standard_name.replace("#", str(i))
                        if standard_name_period not in self.data.columns:
                            self.data[standard_name_period] = np.nan

            else:
                print('This column is not in mapping csv: ', col_name)
                self.data[col] = df[col]





# df = pd.read_csv('sampledf.csv')
# BalanceSheet(df)




