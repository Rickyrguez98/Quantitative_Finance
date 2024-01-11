#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:26:37 2023

@author: rixon
"""

from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
ticker = "IBM" # GOOG  #HPQ   #HPE
yahoo_financials = YahooFinancials(ticker)
data = yahoo_financials.get_historical_price_data("2010-01-01", "2022-08-01", "daily")
df = pd.DataFrame(data[ticker]["prices"])

data=pd.DataFrame(df.adjclose)

data["R"]=data.adjclose.pct_change()
data["r"]=np.log(data.adjclose)-np.log(data.adjclose.shift(1))



#%% Data 1


yahoo_financials = YahooFinancials(ticker)
data1 = yahoo_financials.get_historical_price_data("2000-01-01", "2020-08-01", "daily")
df1 = pd.DataFrame(data1[ticker]["prices"])

data1=pd.DataFrame(df1.adjclose)

data1["R"]=data1.adjclose.pct_change()
data1["r"]=np.log(data1.adjclose)-np.log(data1.adjclose.shift(1))

#%% Pruebas estadisticas
from scipy import stats

data = data.dropna()
data1 = data1.dropna()

alfa = 0.05

#Ho = son igual
#Ha =  no son iguales

'levene'
W, p2 = stats.levene(data.r, data1.r)
if p2<alfa:
    print('Ho puede ser rechazada (No son iguales)')
else:
    print('Ho NO puede ser rechazada (Son iguales)')
    
t,p = stats.ttest_ind(data.r, data1.r, equal_var=True)

if p<alfa:
    print('Ho puede ser rechazada (No son iguales)')
else:
    print('Ho NO puede ser rechazada (Son iguales)')
    



k3, p3 = stats.normaltest(data.r)
if p3 < alfa:
    print("Ho puede ser rechazada, No es normal") 
else:
    print("Ho NO puede ser rechazada, es normal")
    

k4, p4 = stats.normaltest(data1.r)
if p4 < alfa:
    print("Ho puede ser rechazada, No es normal") 
else:
    print("Ho NO puede ser rechazada, es normal")
    
