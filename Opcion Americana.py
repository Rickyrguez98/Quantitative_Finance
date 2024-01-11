#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:11:00 2023

@author: rixon
"""


# PUT
import numpy as np

    # Inputs
n = 2     #number of steps
S = 50  #initial underlying asset price
r = 0.05    #risk-free interest rate
K = 52   #strike price
v = 0.26 #volatility
T = 1
c=2

dt = T/n 
u =  np.exp(v*np.sqrt(dt))
d =  1./u
p = (np.exp(r*dt)-d) / (u-d) 

#Binomial price tree
stockvalue = np.zeros((n+1,n+1))
decision = np.zeros((n+1,n+1))

stockvalue[0,0] = S
for i in range(1,n+1):
    stockvalue[i,0] = stockvalue[i-1,0]*u
    for j in range(1,i+1):
        stockvalue[i,j] = stockvalue[i-1,j-1]*d
        

#option value at final node
optionvalue= np.zeros((n+1,n+1))
for j in range(n+1):
    optionvalue[n,j] = max(K-stockvalue[n,j],0) # se cambia para compra
    
    

#Backward calculation for option price
for i in range(n-1,-1,-1):
    for j in range(i+1):
        optionvalue[i,j]= np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1])
    

print('Put: ',optionvalue[0,0])

#%% Call
    # Inputs
n = 5000     #number of steps
S = 50  #initial underlying asset price
r = 0.05    #risk-free interest rate
K = 52   #strike price
v = 0.26 #volatility
T = 1
c=2

dt = T/n 
u =  np.exp(v*np.sqrt(dt))
d =  1./u
p = (np.exp(r*dt)-d) / (u-d) 

#Binomial price tree
stockvalue = np.zeros((n+1,n+1))
decision = np.zeros((n+1,n+1))

stockvalue[0,0] = S
for i in range(1,n+1):
    stockvalue[i,0] = stockvalue[i-1,0]*u
    for j in range(1,i+1):
        stockvalue[i,j] = stockvalue[i-1,j-1]*d
        

#option value at final node
optionvalue= np.zeros((n+1,n+1))
for j in range(n+1):
    optionvalue[n,j] = max(stockvalue[n,j]-K,0) # se cambia para compra
    
    

#Backward calculation for option price
for i in range(n-1,-1,-1):
    for j in range(i+1):
        optionvalue[i,j]= np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1])
    

print('Call: ',optionvalue[0,0]) 


#%% Opcion americana PUT
    # Inputs
n = 5000     #number of steps
S = 50  #initial underlying asset price
r = 0.05    #risk-free interest rate
K = 52   #strike price
v = 0.26 #volatility
T = 1
c=2

dt = T/n 
u =  np.exp(v*np.sqrt(dt))
d =  1./u
p = (np.exp(r*dt)-d) / (u-d) 

#Binomial price tree
stockvalue = np.zeros((n+1,n+1))
decision = np.zeros((n+1,n+1))

stockvalue[0,0] = S
for i in range(1,n+1):
    stockvalue[i,0] = stockvalue[i-1,0]*u
    for j in range(1,i+1):
        stockvalue[i,j] = stockvalue[i-1,j-1]*d
        

#option value at final node
optionvalue= np.zeros((n+1,n+1))
for j in range(n+1):
    optionvalue[n,j] = max(K-stockvalue[n,j],0) # se cambia para compra
    
    

#Backward calculation for option price
for i in range(n-1,-1,-1):
    for j in range(i+1):
        F1 = np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1])
        F2 = max(K-stockvalue[i,j],0)
        optionvalue[i,j]= max(F1,F2)
    

print('Put Opcion Americana: ',optionvalue[0,0])


#%% Opcion americana Call
    # Inputs
n = 5000     #number of steps
S = 50  #initial underlying asset price
r = 0.05    #risk-free interest rate
K = 52   #strike price
v = 0.26 #volatility
T = 1
c=2

dt = T/n 
u =  np.exp(v*np.sqrt(dt))
d =  1./u
p = (np.exp(r*dt)-d) / (u-d) 

#Binomial price tree
stockvalue = np.zeros((n+1,n+1))
decision = np.zeros((n+1,n+1))

stockvalue[0,0] = S
for i in range(1,n+1):
    stockvalue[i,0] = stockvalue[i-1,0]*u
    for j in range(1,i+1):
        stockvalue[i,j] = stockvalue[i-1,j-1]*d
        

#option value at final node
optionvalue= np.zeros((n+1,n+1))
for j in range(n+1):
    optionvalue[n,j] = max(K-stockvalue[n,j],0) # se cambia para compra
    
    

#Backward calculation for option price
for i in range(n-1,-1,-1):
    for j in range(i+1):
        F1 = np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1])
        F2 = max(stockvalue[i,j]-K,0)
        optionvalue[i,j]= max(F1,F2)
    

print('Call Opcion Americana: ',optionvalue[0,0])

#%% Opcion americana BINARIA PUT Cash or nothing
    # Inputs
n = 5000     #number of steps
S = 40  #initial underlying asset price
r = 0.05    #risk-free interest rate
K = 52   #strike price
v = 0.26 #volatility
T = 1
c=500

dt = T/n 
u =  np.exp(v*np.sqrt(dt))
d =  1./u
p = (np.exp(r*dt)-d) / (u-d) 

#Binomial price tree
stockvalue = np.zeros((n+1,n+1))
decision = np.zeros((n+1,n+1))

stockvalue[0,0] = S
for i in range(1,n+1):
    stockvalue[i,0] = stockvalue[i-1,0]*u
    for j in range(1,i+1):
        stockvalue[i,j] = stockvalue[i-1,j-1]*d
        

#option value at final node
optionvalue= np.zeros((n+1,n+1))
for j in range(n+1):
    optionvalue[n,j] = c if (stockvalue[n,j] < K) else 0 # funcion de utilidad
    
    

#Backward calculation for option price
for i in range(n-1,-1,-1):
    for j in range(i+1):
        F1 = np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1])
        F2 = c if (stockvalue[i,j] < K) else 0
        optionvalue[i,j]= max(F1,F2)
    

print('Put Opcion Americana Binaria Cash or nothing: ',optionvalue[0,0])


#%% Opcion americana BINARIA CALL Cash or nothing
    # Inputs
n = 5000     #number of steps
S = 40  #initial underlying asset price
r = 0.05    #risk-free interest rate
K = 52   #strike price
v = 0.26 #volatility
T = 1
c=500

dt = T/n 
u =  np.exp(v*np.sqrt(dt))
d =  1./u
p = (np.exp(r*dt)-d) / (u-d) 

#Binomial price tree
stockvalue = np.zeros((n+1,n+1))
decision = np.zeros((n+1,n+1))

stockvalue[0,0] = S
for i in range(1,n+1):
    stockvalue[i,0] = stockvalue[i-1,0]*u
    for j in range(1,i+1):
        stockvalue[i,j] = stockvalue[i-1,j-1]*d
        

#option value at final node
optionvalue= np.zeros((n+1,n+1))
for j in range(n+1):
    optionvalue[n,j] = c if (stockvalue[n,j] > K) else 0 # funcion de utilidad
    
    

#Backward calculation for option price
for i in range(n-1,-1,-1):
    for j in range(i+1):
        F1 = np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1])
        F2 = c if (stockvalue[i,j] < K) else 0
        optionvalue[i,j]= max(F1,F2)
    

print('Call Opcion Americana Binaria Cash or nothing: ',optionvalue[0,0])

#%% Compound options

import numpy as np
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf

from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize

def e_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r +0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r -  0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call =(S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call



S, X2, r, sigma = 100, 100, 0.05, .3
T2 = 3
T1 = 1
t=0
X1=10

def SOpt_call(S,X1,X2,T1,T2,r,sigma):
    tau = T2-T1
    bound = [[0, None]]
    apply_constraints1 = lambda S: e_call(S, X2, tau, r, sigma)-X1
    my_constraints = ({'type':'eq',"fun":apply_constraints1})
    a = minimize(e_call, S, bounds=bound, constraints=my_constraints, 
                 method='SLSQP', args=(X2,tau,r,sigma))
    SOpt = a.x[0]
    
    return SOpt



def Call_call(S,X1,X2,T1,T2,r,sigma):
    SOpt = SOpt_call(S, X1, X2, T1, T2, r, sigma)
    ### Call_Call
    D1 = (np.log(S / SOpt) + (r +0.5 * sigma ** 2) * (T1-t)) / (sigma * np.sqrt(T1-t))
    D2 = D1-sigma*np.sqrt(T1-t)
    
    E1 = (np.log(S / X2) + (r +0.5 * sigma ** 2) * (T2-t)) / (sigma * np.sqrt(T2-t))
    E2 = E1-sigma*np.sqrt(T2-t)
    
    Corr = np.sqrt(T1/T2)
    dist = mvn(mean=np.array([0,0]), cov=np.array([[1,Corr],[Corr,1]]))
    N2D1E1 = dist.cdf(np.array([D1,E1]))
    N2D2E2 = dist.cdf(np.array([D2,E2]))
    ND2 = si.norm.cdf(D2, 0.0, 1.0)
    
    Call_call = S*N2D1E1 - X2*np.exp(-r*(T2-t))*N2D2E2 - X1*np.exp(-r*(T1-t))*ND2
    return Call_call


def Call_put(S,X1,X2,T1,T2,r,sigma):
    SOpt = SOpt_call(S, X1, X2, T1, T2, r, sigma)
    ### Call_Call
    D1 = (np.log(S / SOpt) + (r +0.5 * sigma ** 2) * (T1-t)) / (sigma * np.sqrt(T1-t))
    D2 = D1-sigma*np.sqrt(T1-t)
    
    E1 = (np.log(S / X2) + (r +0.5 * sigma ** 2) * (T2-t)) / (sigma * np.sqrt(T2-t))
    E2 = E1-sigma*np.sqrt(T2-t)
    
    Corr = np.sqrt(T1/T2)
    dist = mvn(mean=np.array([0,0]), cov=np.array([[1,Corr],[Corr,1]]))
    N2D1E1 = dist.cdf(np.array([D1,E1]))
    N2D2E2 = dist.cdf(np.array([D2,E2]))
    ND2 = si.norm.cdf(D2, 0.0, 1.0)
    
    Call_call = S*N2D1E1 + X2*np.exp(-r*(T2-t))*N2D2E2 + X1*np.exp(-r*(T1-t))*ND2
    return Call_call
#%%
S, X2, r, sigma = 100, 100, 0.05, .3
T2 = 3
T1 = 1
t=0
X1=10

EscA = e_call(S, X2, T2, r, sigma)
Eva_T1 = e_call(80, X2, T2-T1, r, sigma)

Call_Call = Call_call(S, X1, X2, T1, T2, r, sigma)
