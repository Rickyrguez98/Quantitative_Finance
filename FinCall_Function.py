#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:27:10 2023

@author: rixon
"""

import numpy as np
import scipy.stats as si
from scipy.optimize import minimize

#%%

S=20
k=20
r=0.03
sigma=0.3
t=0
T=1






#%%

#time

d1= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
d2= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

BinPut_AN=S*si.norm.cdf(-d1)
BinCall_AN=S*si.norm.cdf(d1)

BinPut_CN=k*np.exp(-r*(T-t))*si.norm.cdf(-d2)
BinCall_CN=k*np.exp(-r*(T-t))*si.norm.cdf(d2)

FinCall= BinCall_AN-BinCall_CN
FinPut= BinPut_AN-BinPut_CN

DeltaCall = si.norm.cdf(d1)
DeltaPut = si.norm.cdf(-d1)


Gamma = (np.exp(-0.5 * ((d1)**2)))/(S*sigma*np.sqrt(T-t)*np.sqrt(T-t)*np.sqrt(2*np.pi))

Speed = (-Gamma/S)*((d1/sigma*np.sqrt(T-t))+1)

#%%

def FinCall_Fuction(sigma,S,k,r,T,t):
    d1= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    BinPut_AN=S*si.norm.cdf(-d1)
    BinCall_AN=S*si.norm.cdf(d1)

    BinPut_CN=k*np.exp(-r*(T-t))*si.norm.cdf(-d2)
    BinCall_CN=k*np.exp(-r*(T-t))*si.norm.cdf(d2)

    FinCall= BinCall_AN-BinCall_CN
    FinPut= BinPut_AN-BinPut_CN

    DeltaCall = si.norm.cdf(d1)
    DeltaPut = si.norm.cdf(-d1)
    
    call = BinCall_AN-BinCall_CN
    
    return call


call = FinCall_Fuction(sigma, S, k, r, T, t)
newcall = FinCall_Fuction(0.34444605450618127, S, k, r, T, t)

bounds = [[None, None] ]

Market=3
    
apply_constraint1=lambda sigma: FinCall_Fuction(sigma,S,k,r,T,t)-Market
   
    
    #apply_constraint1=lambda S:e_callS(S)-X1
my_constraints = ({'type': 'eq', "fun": apply_constraint1}

                       )
a = minimize(FinCall_Fuction, sigma, bounds=bounds, 
             constraints=my_constraints, method='SLSQP',args=(S,k,r,T,t))

SigmaOpt =a.x[0]
    




#%% 2do Ejercicio

#%%Parameters
S_array=[100,110,120,110,130]
k_array=[100,100,100,100,100]
r_array=[0.05,0.05,0.05,0.05,0.05]
t_array=[0,0,0,0,0]
T_array=[6/6,5/6,4/6,3/6,2/6]
Market_array=[15,14,17,16,16]

Delta_array=[0,0,0,0,0]
Gamma_array=[0,0,0,0,0]
Speed_array=[0,0,0,0,0]

Sigma_array = [0.1,0.1,0.1,0.1,0.1]

S=S_array[0]
k=k_array[0]
r=r_array[0]
t=t_array[0]
T=T_array[0]
Market= Market_array[0]
sigma = Sigma_array[0]

#time




#%%

def FinCall_Fuction(sigma,S,k,r,T,t):
    d1= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    BinPut_AN=S*si.norm.cdf(-d1)
    BinCall_AN=S*si.norm.cdf(d1)

    BinPut_CN=k*np.exp(-r*(T-t))*si.norm.cdf(-d2)
    BinCall_CN=k*np.exp(-r*(T-t))*si.norm.cdf(d2)

    FinCall= BinCall_AN-BinCall_CN
    FinPut= BinPut_AN-BinPut_CN

    DeltaCall = si.norm.cdf(d1)
    DeltaPut = si.norm.cdf(-d1)
       
    call = BinCall_AN-BinCall_CN
    
    return call


for i in range(5):
    S=S_array[i]
    k=k_array[i]
    r=r_array[i]
    t=t_array[i]
    T=T_array[i]
    Market= Market_array[i]
    sigma = Sigma_array[i]

    bounds = [[None, None] ]
        
    apply_constraint1=lambda sigma: FinCall_Fuction(sigma,S,k,r,T,t)-Market
       
        
        #apply_constraint1=lambda S:e_callS(S)-X1
    my_constraints = ({'type': 'eq', "fun": apply_constraint1}
    
                           )
    a = minimize(FinCall_Fuction, sigma, bounds=bounds, 
                 constraints=my_constraints, method='SLSQP',args=(S,k,r,T,t))
    
    SigmaOpt =a.x[0]
    




    sigma= SigmaOpt
    
    d1= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2= (np.log(S/k)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    
    
    BinPut_AN=S*si.norm.cdf(-d1)
    BinCall_AN=S*si.norm.cdf(d1)
    
    BinPut_CN=k*np.exp(-r*(T-t))*si.norm.cdf(-d2)
    BinCall_CN=k*np.exp(-r*(T-t))*si.norm.cdf(d2)
    
    FinCall= BinCall_AN-BinCall_CN
    FinPut= BinPut_AN-BinPut_CN
    
    DeltaCall = si.norm.cdf(d1)
    DeltaPut = si.norm.cdf(-d1)
    
    
    Gamma = (np.exp(-0.5 * ((d1)**2)))/(S*sigma*np.sqrt(T-t)*np.sqrt(T-t)*np.sqrt(2*np.pi))
    
    Speed = (-Gamma/S)*((d1/sigma*np.sqrt(T-t))+1)
    
    Delta_array[i]=DeltaCall
    Gamma_array[i]=Gamma
    Speed_array[i]=Speed
    
    Sigma_array[i]=sigma

    print(SigmaOpt, DeltaCall, DeltaPut, Gamma, Speed)




