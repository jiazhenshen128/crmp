# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:47:56 2019
@author: JiazhenShen
"""


from PDModel import PDModel
from UnifyData import FinancialSheet
import os                                         # For accessing .env variables
import pandas as pd                               # For munging tabular data
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import numpy as np


def cumPD(myInput, model_file, Pdata):
    max_period = myInput.output_horizon
    res = 1
    for myInput.output_horizon in range(1, max_period):
        model = PDModel(myInput, model_file=model_file)
        model.genPdata(Pdata)
        prePD = model.use()['prePDs'][0]
        res *= 1-prePD
    return 1-res


class Foo:
    pass

#########################################
modelFile ="platform"

myInput = Foo()
myInput.machineLearningEngine = 'LR'
myInput.output_horizon = 1



################ Simple Test #########################

#model = PDModel(myInput, model_file=modelFile)
#model.genPdata(*[888.0,888,88.08,888,8,888,888,8888,8888888,8,8,8,888,88,8,888,8,888,8,8.9,8,8,8])
#model.genPdata(*[777,777,777,877.09,77,887,7888,78888,88877778888.09,78,8,78,8788,878,87,8788,78,8878,87,877,78,87,87])
#model.use()

############## Plantform Data ######################

# Connect
snow_user = os.getenv("snow_user")
snow_pwd = os.getenv("snow_pwd")
print(snow_user)

engine = create_engine(URL(
    account='xb33235.eu-west-1',
    user=snow_user,
    password=snow_pwd,
    database='ANALYTICS',
    schema='ANALYTICS',
    warehouse='reporting',
    role='ANALYST',
))

cs = engine.connect()

# Get the total ID we have
sql = f"SELECT * from ANALYTICS.ANALYTICS.SMECAPITAL_LOANS"
df1 = pd.read_sql_query(sql, cs)
maxID=max(df1['sys_transaction_id'])
print('The maximum loan ID of the plantform: '+str(maxID))

# Start for loop from 0 to max ID we have
csvdf = pd.DataFrame()
for loan_id in range(1,maxID+1):
    # Get the neccessary data
    sql = f"SELECT * from ANALYTICS.ANALYTICS_BASE.SMECAPITAL_ORIGINATION_TEMPFINANCIALS where loan_id={loan_id}"
    df4 = pd.read_sql_query(sql, cs)
    if df4.empty:  # If financial statement empty, then skip this ID and print ID
        print('Skip ID:' + str(loan_id))
        continue

    df4.fillna(value=pd.np.nan, inplace=True)
    df4a = df4[(df4.record_kinds=='Historical')]
    df4b = df4a[df4a.display_order==np.max(df4a.display_order)]

    EBITDA_latest = df4a[df4a.display_order==np.max(df4a.display_order)]['adjusted_ebitda'].values[0]
    WC = df4b['total_working_capital'].values[0]
    TA = df4b['total_assets'].values[0]
    if np.isnan(TA):  # If TA is missing, then skip this ID and print ID
        print('Skip ID:' + str(loan_id))
        continue
    RP = df4b['retained_profit'].values[0]
    OP = df4b['operating_profit'].values[0]
    TL = df4b['total_liabilities'].values[0]
    TCL = df4b['current_liabilities'].values[0]
    Turnover = df4b['turnover'].values[0]
    sql = f"SELECT * from ANALYTICS.ANALYTICS.SMECAPITAL_LOANS where SYS_TRANSACTION_ID={loan_id}"
    df5 = pd.read_sql_query(sql, cs)
    if df5.empty:
        print('Skip ID:' + str(loan_id))
        continue
    company_name = df5['borrowing_company'].values[0]

    # Now we unify and generate Pdata we want
    df = pd.DataFrame({'TCL 1': [TCL], 'IA 1': [0.0], 'TNCL 1': [TL-TCL], 'TA 1':[TA], 'WC 1':[WC], 'RP 1':[RP],
                       'EBITDA 1':[EBITDA_latest], 'OP 1':[OP], 'Turnover 1': Turnover})
    Pdata = FinancialSheet(df, mapping_csv=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/name_mapping.csv').data
    # Start our model for output_horizon = 1
    myInput.output_horizon = 1
    model = PDModel(myInput, model_file=modelFile)
    model.genPdata(Pdata)
    prePD = model.use()['prePDs'][0]
    # Our csv row for saving is using the notnull Pdata in the model. It is opportunity to see if the data is past
    # to the model correctly after unifying variable names and constructing Experian format tables
    # row = model.Pdata_notnull
    row = df
    row['Loan ID'] = [loan_id]
    row['Company Name'] = [company_name]
    row[f'PD{myInput.output_horizon}'] = [prePD]
    # Run the PD for output_horizon = 2,3,4 too.
    for myInput.output_horizon in range(2,5):
        model = PDModel(myInput, model_file=modelFile)
        model.genPdata(Pdata)
        prePD = model.use()['prePDs'][0]
        row[f'PD{myInput.output_horizon}'] = [prePD]
    # Append the row in the csv we will save
    csvdf = pd.concat([csvdf, row], ignore_index=True, sort=False)
    print(csvdf)

# Save the CSV
csvdf.to_csv('PD_plantform_data1.csv')






