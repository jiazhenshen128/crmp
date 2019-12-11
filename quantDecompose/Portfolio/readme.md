# Portfolio Models 
This model combine many companies' RRs, PDs and exposures period by period and output the KPIs 
and plot simulated loss distributions with necessary parameters.
    
    
## Initialisation 
Inputs:    
1. **RR_intervals**: Points determining intervals for RR distributes in, e.g. [0, 0.2, 0.6, 1] represents three intervals.  
1. **PDs_RRdists_exposures_periods**: a list of period-level 3-elements list:  
    - **PDs**:  a list or array of companies' probability of default, e.g. [0.5,0.1,0.2]  
	- **RR_dists**: a list of companies' distributions (lists) in intervals, e.g. [[0.2, 0.3, 0.5], [0.9, 0.05,0.05],[0.4, 0.3, 0.3]]  
    - **exposures**: list of exposures for different companies, e.g. [1,2.5,1.8]  
>  The input is a 4D list:         
>  Periods T $\times$  3 (PD,  $\times$ n loans            
>   Periods T $\times$ RR distributions,  $\times$ n loans $\times$ m RR-intervals        
>   Periods T $\times$  exposures)  $\times$ n loans      
> **Example:**       
> **[**       
> (Period T-1 = Next Period)        
> **[**         
> **[0.5, 0.1]** (PDs) **,          
> [[0.2, 0.3, 0.5], [0.9, 0.05, 0.05]]** (RR distributions)**,          
> [10, 1]** (exposures)          
> **],**       
> 
>(Period T)       
>**[**     
>**[0.6, 0.2], [[0.4, 0.3, 0.3], [0.7, 0.1, 0.2]], [10, 10]], [[0.4, 0.4], [[0.5, 0.1, 0.4], [0.8, 0.05, 0.15]], [102, 9]**        
>**]**       
>**]**       
5.  **correlation_matrix**: a 2D-array or list of correlation matrix.  
     
Outputs:    
1. **self.lgd_interval_middles**: an array of Loss given default (1-RR) intervals middle points.  
7.  **self.loss_distributions**: loss distributions of each companies  
8. **self.self.sigma2_losses**: The variances of LGD distributions  
9. **self.sigma2_EDF**:  The variances of PD binomial distributions.  
       
    
## simulate_portfolio  
This member function is to simulate the scenarios of the portfolio to see the comprehensive loss of the portfolio.   
  
**Inputs**:    
1. **sim_num**: a integer indicating how many scenarios it will simulate.  
2. **sim_copula**:  a string: 't' or 'Guassian' to indicate which copula to be used.  
  
**Outputs**:  
1. **self.unifrvs**: simulated uniform random variables  
4.  **self.sim_port_loss**: simulated portfolio losses  
   
    
## genKPI  
This function is going to generate the KPIs of the portfolio.  
  
1. **TotalExposure**: The sum of exposures  
2. **ExpectedLoss**: The expected value of loss (% of TotalExposure)  
3. **UnexpectedLoss**: The variance of the loss function (% of TotalExposure)  
4.  **VAR ($\alpha$: 90%, 95%, 99%)**: Value at Risk (Requring running simulate_portfolio() first)  
5. **Expectedshortfall ($\alpha$: 90%, 95%, 99%)**: Expected shortfall (% of TotalExposure) (Requring running simulate_portfolio() first)