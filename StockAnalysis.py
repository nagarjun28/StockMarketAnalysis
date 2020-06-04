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
    globals()[stock] = data.get_data_yahoo(stock, start, end)

# AAPL['Adj Close'].plot(legend = True,figsize=(10,4))
AAPL['Volume'].plot(legend=True, figsize=(10, 4))

# Finding moving averages for fav AAPL stock

ma_day = [10, 20, 50]
for ma in ma_day:
    column_name = "MA for days" + str(ma)
    AAPL[column_name] = AAPL['Adj Close'].rolling(ma).mean()
# To understand the flowing line and trading averages
AAPL[['Adj Close', 'MA for days10', 'MA for days20', 'MA for days50']].plot(subplots=False, figsize=(10, 4))

# To check for daily returns
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(10, 4), legend=True, linestyle='--', marker='o')

sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')
closing_df = data.get_data_yahoo(tech_list, start, end)['Adj Close']
tech_rets = closing_df.pct_change()
sns.jointplot('GOOG', 'MSFT', tech_rets, kind='scatter', color='seagreen')

# To plot pair to understand the corelation better
sns.pairplot(tech_rets.dropna())
sns.pairplot(closing_df.dropna())

sns.heatmap(tech_rets.corr(), annot=True)
sns.heatmap(closing_df.corr(), annot=True)

# Risk analysis for the stocks

rets = tech_rets.dropna()
area = np.pi * 20
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel("Expected return")
plt.ylabel("Risk")
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy=(x, y), xytext=(50, 50),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3')
    )

