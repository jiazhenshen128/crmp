import numpy as np
from Portfolio.Port_modelClass import CreditPort


def test_simplest_linear_model():
    PDs = [0.5,0.1]
    RR_dists = [[0.2, 0.3, 0.5], [0.9, 0.05,0.05]]
    RR_intervals = [0, 0.2, 0.6, 1]
    exposures = [10 for i in range(0, len(PDs))]
    PDs_RRdists_1 = [PDs, RR_dists, exposures]
    PDs_RRdists_2 = [PDs, RR_dists, exposures]
    PDs_RRdists_3 = [PDs, RR_dists, exposures]
    PDs_RRdists_years = [PDs_RRdists_1, PDs_RRdists_2, PDs_RRdists_3]
    correlation_matrix = np.ones((len(PDs), len(PDs)))*0.2
    np.fill_diagonal(correlation_matrix, 1)
    myPort = CreditPort(RR_intervals = RR_intervals, PDs_RRdists_exposures_periods = PDs_RRdists_years, correlation_matrix=correlation_matrix)
    myPort.genKPI()
    myPort.simulate_portfolio()
    myPort.genKPI()
    myPort.simulate_portfolio(10000, 't', t_v=4)
    myPort.genKPI()
