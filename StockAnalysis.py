import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# For data retrival
from pandas_datareader import data, wb
from datetime import datetime

sns.set_style('whitegrid')
# Let us consider the stocks for various tech gaints
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = data.get_data_yahoo(stock,start,end)

# AAPL['Adj Close'].plot(legend = True,figsize=(10,4))
AAPL['Volume'].plot(legend = True,figsize=(10,4))
