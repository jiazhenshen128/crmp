# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:13:59 2018

@author: JiazhenShen
"""

import numpy as np
import bisect
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
from itertools import combinations


def ppf(x, xcdf, xvalue):
    print(len(xcdf))
    return [[xvalue[i][bisect.bisect_right(xcdf[i], x[j][i])] for i in range(0, len(xcdf))] for j in range(0, len(x))]


def multivariate_t(mu,corrM,dfreedom,simN):
    d = len(corrM)
    g = np.tile(np.random.gamma(dfreedom/2.,2./dfreedom,simN),(d,1)).T
    Z = np.random.multivariate_normal(np.zeros(d),corrM,simN)
    return mu+Z/np.sqrt(g)


def combine_dist(dist1_x,dist1_y,dist2_x,dist2_y):
    dist3_x = np.concatenate([dist1_x, dist2_x],axis=1)
    dist3_y = np.concatenate([dist1_y, dist2_y],axis=1)
    ind = np.argsort(-dist3_x, axis=1)
    dist_x = np.take_along_axis(dist3_x, ind, axis=1)
    dist_y = np.take_along_axis(dist3_y, ind, axis=1)
    print(np.insert(np.diff(dist_x,axis=1)==0,0,values=False,axis=1))
    dist_x[np.insert(np.diff(dist_x,axis=1)==0,0,values=False,axis=1)]  =  dist_x[np.insert(np.diff(dist_x,axis=1)==0,0,values=False,axis=1)] + 0.0000000001
    print(dist_x)
    return (dist_x, dist_y)


class CreditPort:
    def __init__(self, PDs_RRdists_exposures_periods, correlation_matrix, RR_interval_middles = None, RR_intervals = None ):
        # Inputs preparations
        self.PDs_periods = [np.array([[1-i,i] for i in PDs_RRdists[0]]) for PDs_RRdists in PDs_RRdists_exposures_periods]
        self.RR_dists_periods = [np.array(PDs_RRdists[1]) for PDs_RRdists in PDs_RRdists_exposures_periods]
        self.exposures_periods = [np.array(PDs_RRdists[2]) for PDs_RRdists in PDs_RRdists_exposures_periods]
        self.correlation_matrix = np.array(correlation_matrix)
        if RR_interval_middles:
            self.lgd_interval_middles = 1-np.array(RR_interval_middles)
        else:
            RR_intervals = np.array(RR_intervals)
            self.lgd_interval_middles = 1-(RR_intervals[1:]+RR_intervals[:-1])/2
        self.total_periods = len(PDs_RRdists_exposures_periods)


        # Adjusted items
        print('PDs_periods', self.PDs_periods)
        self.adjPDs_periods = np.array([self.PDs_periods[0][:,1], ])
        multipliers = np.array([PDs[:, 0] for PDs in self.PDs_periods])
        print('multipliers', multipliers)
        self.loss_distributions_periods = [[np.transpose(x*np.transpose(y)) for x, y in zip(self.adjPDs_periods[0], self.RR_dists_periods[0])], ]
        self.loss_distributions_x_periods = [[np.transpose(x * np.transpose(self.lgd_interval_middles)) for x in self.exposures_periods[0]], ]
        for period in range(1, self.total_periods):
            multiplier = np.array([np.cumprod(multipliers[:period],axis=0)[-1,:]])
            PDnow = self.PDs_periods[period][:,1]
            print(multiplier*PDnow)
            self.adjPDs_periods = np.append(self.adjPDs_periods, multiplier*PDnow, axis=0)
            print('-----------------', period)
            print(self.adjPDs_periods)
            print('-----------------', 'loss_distributions_periods')
            print(self.loss_distributions_periods)
            dist1_y = self.loss_distributions_periods[-1]
            dist2_y = [np.transpose(x * np.transpose(y)) for x, y in zip(self.adjPDs_periods[period], self.RR_dists_periods[period])]
            dist1_x = self.loss_distributions_x_periods[-1]
            dist2_x = [np.transpose(x * np.transpose(self.lgd_interval_middles)) for x in self.exposures_periods[period]]
            print('dist')
            print(dist1_y)
            print(dist2_y)
            print(dist1_x)
            print(dist2_x)
            loss_distributions_x_periodsnow, loss_distributions_periodsnow = combine_dist(dist1_x,dist1_y,dist2_x,dist2_y)
            print('-----------------', 'loss_distributions_periodsnow')
            print(loss_distributions_periodsnow)
            print(loss_distributions_x_periodsnow)
            print('-----------------!!!!!!!')
            print(self.loss_distributions_periods)
            print(loss_distributions_periodsnow)
            self.loss_distributions_periods.append(loss_distributions_periodsnow)
            self.loss_distributions_x_periods.append(loss_distributions_x_periodsnow)
            print('!!!!!!!!!!!!!!!')
            print(self.loss_distributions_periods)
            print(self.loss_distributions_x_periods)
            print('-----------------')

        # Some calculations independent of periods
        self.ERR_periods = [np.array([np.sum(np.array(x)*np.array((1-self.lgd_interval_middles)))
                                    for x in RR_dists]) for RR_dists in self.RR_dists_periods]
        self.ELs_periods = [x*y*(1-z) for x, y, z in zip(self.exposures_periods, self.adjPDs_periods, self.ERR_periods)]

        self.sigma2_RR_periods = [np.var(self.lgd_interval_middles*RR_dists, axis=1) for RR_dists in self.RR_dists_periods]
        self.sigma2_EDF_periods = [adjPDs*(1-adjPDs) for adjPDs in self.adjPDs_periods]

        # self.unexpectedLosses = [(exposures, ERR, sigma2_EDF, adjPDs, sigma2_losses) for exposures, ERR, sigma2_EDF, adjPDs, sigma2_losses in zip(self.exposures_periods, self.ERR_periods, self.sigma2_EDF_periods, self.adjPDs_periods, self.sigma2_losses_periods)]

        self.sigma2_adjloss_periods = [np.power(exposures,2)*np.power((1-ERR), 2)*sigma2_EDF+adjPDs*sigma2_RR
                                 for exposures, ERR, sigma2_EDF, adjPDs, sigma2_RR
                                 in zip(self.exposures_periods, self.ERR_periods, self.sigma2_EDF_periods, self.adjPDs_periods, self.sigma2_RR_periods)]
        print(self.sigma2_adjloss_periods)
        foo_adjEL_combinations_periods = [adjELs1*adjELs2 for adjELs1, adjELs2 in list(combinations(self.ELs_periods,2))]
        print(foo_adjEL_combinations_periods)
        self.unexpected_loss = np.sum(self.sigma2_adjloss_periods,axis=0) - 2*np.sum(foo_adjEL_combinations_periods, axis=0)
        print(self.unexpected_loss)

    def simulate_portfolio(self, sim_num=10000, sim_copula="Guassian", **kwargs):
        x = self.loss_distributions_periods[-1]
        x0 = 1-np.sum(x,axis=1)
        xpdf = np.insert(x, len(x[0]), x0, axis=1)
        xcdf = np.cumsum(xpdf, axis=1)
        xvalue = self.loss_distributions_x_periods[-1]
        xvalue = np.insert(xvalue, xvalue.shape[1],values=0, axis=1)
        print(xcdf,xvalue)

        if sim_copula == "Guassian":
            mvnorm = stats.multivariate_normal(mean=np.zeros((len(x)), ), cov=self.correlation_matrix)
            nrvs = mvnorm.rvs(sim_num)
            unifrvs = stats.norm.cdf(nrvs)
        if sim_copula == "t":
            trvs = multivariate_t(np.zeros((len(x)), ), self.correlation_matrix, kwargs['t_v'], sim_num)
            unifrvs = stats.t.cdf(trvs, kwargs['t_v'])
        if len(unifrvs.shape) == 1:
            unifrvs = np.array([unifrvs])
        self.unifrvs = unifrvs
        print(unifrvs.shape)
        simrvs = np.array(ppf(unifrvs, xcdf, xvalue))
        print(simrvs)
        self.sim_port_loss = np.sum(simrvs,axis=1)

        binwidth = 0.006
        print(np.mean(self.exposures_periods, axis=0))
        print(np.sum(np.mean(self.exposures_periods, axis=0)))
        bins = np.arange(min(self.sim_port_loss / np.sum(np.mean(self.exposures_periods, axis=0))),
                         max(self.sim_port_loss / np.sum(np.mean(self.exposures_periods, axis=0))) + binwidth,
                         binwidth)
        self.portHist = plt.hist(self.sim_port_loss / np.sum(np.mean(self.exposures_periods, axis=0)),
                                 weights=np.zeros_like(self.sim_port_loss) + 1. / self.sim_port_loss.size, bins=bins)
        plt.title("Histogram of portfolio loss with " + sim_copula + " copula")
        plt.show()

    def genKPI(self):
        KPI = {}
        KPI["AverageExposure"] = np.sum(np.mean(self.exposures_periods, axis=0))
        KPI["ExpectedLoss"] = round(100*sum(self.ELs_periods[-1])/KPI["AverageExposure"],2)
        print(KPI["ExpectedLoss"])
        tmpa=np.mean(self.exposures_periods,axis=0)*self.unexpected_loss/KPI["AverageExposure"]
        tmpb=np.matmul(tmpa,self.correlation_matrix)
        KPI["UnexpectedLoss"] = round(100*np.matmul(tmpb,np.transpose(tmpa)),2)
        print(KPI["UnexpectedLoss"])
        #KPI["PDs"] = self.adjPDs_periods.tolist()
        #KPI["RR_dists"] = self.RR_dists_periods
        KPI["LossDistributions"] = self.loss_distributions_periods[-1]
        #KPI["ExpectedRR"] = self.ERR_periods.tolist()
        if hasattr(self, 'portHist'):
            
            df=pd.DataFrame(self.sim_port_loss)
            df.sort_values(0,inplace=True,ascending=True)
            KPI["VAR90"] = round(df[0].quantile(0.90)/KPI["AverageExposure"],2)
            sim_port_lossR=self.sim_port_loss/KPI["AverageExposure"]
            KPI["Expectedshortfall90"] = round(np.mean(sim_port_lossR[sim_port_lossR > KPI["VAR90"]]), 2)
            if not (KPI["Expectedshortfall90"]>0):
                KPI["Expectedshortfall90"]=0
            KPI["VAR95"] = round(df[0].quantile(0.95)/KPI["AverageExposure"],2)
            KPI["Expectedshortfall95"] = round(np.mean(sim_port_lossR[sim_port_lossR > KPI["VAR95"]]), 2)
            if not (KPI["Expectedshortfall95"]>0):
                KPI["Expectedshortfall95"]=0
            KPI["VAR99"]=round(df[0].quantile(0.99)/KPI["AverageExposure"],2)
            KPI["Expectedshortfall99"] = round(np.mean(sim_port_lossR[sim_port_lossR > KPI["VAR99"]]), 2)
            if not (KPI["Expectedshortfall99"]>0):
                KPI["Expectedshortfall99"]=0
            KPI["LossDistributiony"] = (np.round(self.portHist[0], 3)).tolist()
            KPI["LossDistributionx"] = (np.round(100*self.portHist[1], 1)).tolist()

        return KPI



