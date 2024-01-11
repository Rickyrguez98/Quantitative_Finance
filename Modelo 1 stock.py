#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:28:16 2023

@author: rixon
"""


from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Download AMZN
ticker = "IBM" # GOOG  #HPQ   #HPE
yahoo_financials = YahooFinancials(ticker)
data = yahoo_financials.get_historical_price_data("2010-01-01", "2022-08-01", "daily")
df = pd.DataFrame(data[ticker]["prices"])

data=pd.DataFrame(df.adjclose)



data["R"]=data.adjclose.pct_change()
data["r"]=np.log(data.adjclose)-np.log(data.adjclose.shift(1))

S0 = data.iloc[0]
mu = data.R.mean()
sigma = data.R.std()

k = len(data)

data_model = np.zeros(k)

data_model[0] = S0.iloc[0]

#%% Random Walk
w = np.random.normal(loc= 0 , scale= 1, size=k)

for i in range(1, k):
    data_model[i]= data_model[i-1]*np.exp((mu-0.5*sigma**2) + sigma * w[i])





#%% Plot
data.adjclose.plot()
pd.DataFrame(data_model).plot()

