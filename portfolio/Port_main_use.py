from Portfolio import CreditPort
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import re

modelFile = "platform"
id_label = "Loan ID"
output_horizon_within = 4
PDs_RRdists_exposures_years = []

for output_horizon in range(1,output_horizon_within+1):
    csv_files = [f for f in listdir(modelFile) if (isfile(join(modelFile, f)) and 'csv' in f)]
    PD_csv = [f for f in csv_files if 'PD' in f][0]
    RR_csv = [f for f in csv_files if 'RR' in f][0]
    PDdf = pd.read_csv(join(modelFile, PD_csv))
    RRdf = pd.read_csv(join(modelFile, RR_csv))
    port_df = pd.merge(PDdf, RRdf, on=id_label)
    print(port_df.columns)

    PDs = port_df[f'PD{output_horizon}'].values/100

    RR_dists = port_df[[c for c in port_df.columns if (f"RR{output_horizon} dis" in c)]].values/100

    RR_intervals = [float(re.findall("\d+\.\d+", s)[0])/100 for c in port_df.columns if (f"RR{output_horizon} dis" in c) for s in c.split() if len(re.findall("\d+\.\d+", s))==1]

    exposures = [1 for i in range(0, len(PDs))]
    correlation_matrix = np.ones((len(PDs), len(PDs)))*0.2
    np.fill_diagonal(correlation_matrix, 1)
    print(correlation_matrix)
    print('PDs:\n',PDs)
    print('RRs:\n',RR_dists)
    print(RR_intervals)
    PDs_RRdists_exposures = [PDs,RR_dists,exposures]
    PDs_RRdists_exposures_years =PDs_RRdists_exposures_years + [PDs_RRdists_exposures,]

print(PDs_RRdists_exposures_years)
myPort = CreditPort( RR_interval_middles=RR_intervals, PDs_RRdists_exposures_periods=PDs_RRdists_exposures_years, correlation_matrix=correlation_matrix)
print('--------------------KPI-------------------')
myPort.genKPI()

print('--------------------SIMULATE-------------------')
myPort.simulate_portfolio()
print(myPort.genKPI().keys())
