import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# For data retrival
from pandas_datareader import data, wb
from datetime import datetime
import StockAnalysis as st
# Monte Carlo method
# Future price of the stock cannot be predicted from the past price : efficient market hypothesis

days = 365
dt = 1/days

mu = st.rets.mean()['GOOG']
sigma = st.rets.std()['GOOG']

def stock_monte_carlo(start_price):
    price = np.zeros(days)
    price[0] = start_price

    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in range(1,days):
        shock [x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))

    return price