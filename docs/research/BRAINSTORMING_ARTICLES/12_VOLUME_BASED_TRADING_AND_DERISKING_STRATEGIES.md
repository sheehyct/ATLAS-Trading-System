# Deciphering the Silent Language of Volume-Based Algorithmic Trading & Integrated De-Risking Strategies in Financial Markets | by Alexzap | InsiderFinance Wire

Member-only story

# Deciphering the Silent Language of Volume-Based Algorithmic Trading & Integrated De-Risking Strategies in Financial Markets

## Synergizing Volume Technical Analysis with Non-Volume Indicators, Strategy Backtesting, Essential Financial KPI’s & Robust ML Forecasting Models: PLTR Use Case

[

![Alexzap](https://miro.medium.com/v2/resize:fill:64:64/1*L1sRpfwSPETNMy1qzGUGBg.jpeg)





](https://medium.com/@alexzap922?source=post_page---byline--6e078ab8cc36---------------------------------------)

[Alexzap](https://medium.com/@alexzap922?source=post_page---byline--6e078ab8cc36---------------------------------------)

Following

69 min read

·

Jul 29, 2025

212

2

Listen

Share

More

**Keywords:** Financial markets, Python, algorithmic trading, integration, volume technical analysis, stock fundamentals, financial metrics, Machine Learning (ML) algorithms, Prophet

[“Nature laughs at the difficulties of integration”.](https://todayinsci.com/QuotationsCategories/I_Cat/Integration-Quotations.htm) — [Pierre-Simon Laplace](https://todayinsci.com/L/Laplace_Pierre/LaplacePierre-Quotations.htm)

_Integrated De-Risking in Algorithmic Trading: A Boon or a Bane Ad Nauseam?_

### Synopsis:

-   This is a full-fledged integrated algorithmic trading de-risking strategy in Python aimed at the identification and mitigation of (market, execution, model, and operational) risks while maximizing profitability in the fast-moving financial markets.
-   The strategy consists of the following three constituents:

1.  By jointly interpreting and backtesting volume and non-volume (trend-following, momentum, and volatility) trading indicators, we reduce the risk of false signals and minimize uncertainties of trading decisions.
2.  Incorporating both fundamental and technical analysis allows us to leverage the strengths of each method, viz. capturing short-term price fluctuations and evaluating the intrinsic value of a stock with relevant financial models and KPI’s.
3.  Implementing the multi-model approach for stock price prediction through the integration of the aforementioned trading indicators and well-established ML models such as [LSTM](https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/), [SciKit-Learn](https://scikit-learn.org/stable/), and [FB Prophet](https://medium.com/data-science/getting-started-predicting-time-series-data-with-facebook-prophet-c74ad3040525).

-   The proposed strategy is assessed using the [PLTR](https://www.palantir.com/) historical data as a use case.
-   Results are compared against U.S. Big Tech, Buy-and-Hold (B&H), and S&P 500 benchmark.
-   This assessment also includes best industry practices represented by independent third-party insights that may not be visible through other means.

![Image Design by the author via Canva.](https://miro.medium.com/v2/resize:fit:700/1*C3P4KeIEywyKJkXdu5EchQ.png)
Image Design by the author via Canva.

-   Nowadays, Python algorithmic (aka algo) trading (AT) has been established as a core dynamic ecosystem at the heart of quantitative finance \[1–5\] because of [speed, accuracy, and reduced costs](https://www.nasdaq.com/articles/advantages-algorithmic-trading-2019-06-07).
-   Other [benefits](https://www.utradealgos.com/blog/what-every-trader-should-know-about-algorithmic-trading-risks) of AT commonly include reduced emotion, enhanced liquidity, diversification, backtesting and optimization.
-   While AT offers many advantages, it comes with its own set of [risks](https://www.utradealgos.com/blog/what-every-trader-should-know-about-algorithmic-trading-risks) such as the execution, operational, market, model, regulatory and compliance risks (slippage, technological issues, data quality/privacy, market volatility, overfitting, concept/data drift, cybersecurity, etc.) \[6\].
-   The objective of this post is to address _the market and model risks_ by adopting integrated de-risking AT strategies, vide infra.
-   Our approach encompasses the following three components that cater to complementary aspects of AT risk management:

1.  A proper understanding of price movements and trend identification by combining [volume-based trading strategies](https://fenefx.com/en/blog/volume-based-trading-strategies/) \[5, 8\] with other technical analysis tools \[1, 7\]. The volume can be a sign that the current trend is weakening or about to reverse. That’s why [the volume indicators](https://wemastertrade.com/volume-indicators/) are visualized as a measure of the strength of a trend.
2.  Combining insights from fundamental and technical analysis \[7\]. While assessing a company’s intrinsic worth, traders can capture market trends and momentum to identify opportunities and find the ideal moment to enter/exit the market.
3.  Accurate [stock price prediction](https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning) using [machine learning](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-machine-learning), [deep learning](https://www.simplilearn.com/tutorials/deep-learning-tutorial/what-is-deep-learning) and statistical modeling such as [FB Prophet](https://facebook.github.io/prophet/). It is now possible to analyze vast amounts of stock data and uncover patterns that were once impossible to spot.

Let’s delve into the specifics of the proposed methodology!

## Contents

-   Core Libraries, Tools & APIs
-   Basic Imports & Installations
-   Reading & Plotting Input Stock Data
-   Stock Candlesticks with Indicators
-   Stock Volatility & Returns
-   Alpha & Beta Coefficients
-   On-Balance Volume (OBV)
-   Basic Momentum
-   Momentum Trading Strategy
-   Supervised ML Forecast of Strategy Position
-   Price Dynamics vs Volume Ratio
-   OBV-EMA Trading Strategy
-   VPT-MA Trading Strategy
-   VROC-MA Trading Strategy
-   VWAP Trading Strategy
-   A/D Line Trading Strategy
-   CMF Trading Strategy
-   MFI Trading Strategy
-   KO Trading Strategy
-   NVI Trading Strategy
-   Optimized Fisher-PVT Trading Strategy
-   VWAP & Dynamic Volume Profile Oscillator
-   Prophet Stock Price Prediction
-   LSTM Stock Price Prediction
-   Backtesting VWAP Crossover Strategy
-   Backtesting RSI Trading Strategy
-   Optimizing MACD Trading Strategy
-   Backtesting TSI Trading Strategy
-   Backtesting ADX Trading Strategy
-   Backtesting MACD-CHOP Trading Strategy
-   Optimizing CCI-VI Trading Strategy
-   Comparison of Essential Financial Ratios & KPI’s
-   Ichimoku Cloud Chart
-   Bollinger Bands
-   RSI for Big Tech
-   Other Technical Indicators
-   Third-Party Insights

## Core Libraries, Tools & APIs

-   [TwelveData](https://twelvedata.com/) API: Fetch historical stock data
-   [pandas-ta](https://pypi.org/project/pandas-ta/): An easy to use Python 3 Pandas extension with 130+ Technical Analysis Indicators. Can be called from a Pandas Dataframe or standalone like [TA-Lib](https://github.com/TA-Lib/ta-lib-python). Correlation tested with [TA-Lib](https://github.com/TA-Lib/ta-lib-python).
-   [Backtesting.py](https://kernc.github.io/backtesting.py/) is a Python framework for inferring viability of trading strategies on historical (past) data.
-   [FinanceToolkit](https://www.jeroenbouma.com/projects/financetoolkit/getting-started) is an open-source toolkit in which all relevant financial ratios (150+), indicators and performance KPI’s are implemented.
-   [Prophet](https://pypi.org/project/prophet/) Automatic Time-Series Forecasting Procedure
-   [SciKit-Learn](https://scikit-learn.org/stable/): ML in Python
-   [Keras](https://keras.io/getting_started/) is the high-level API of the [TensorFlow](https://www.tensorflow.org/) platform. It provides an approachable, highly-productive interface for solving ML problems
-   [Vectorbt](https://pypi.org/project/vectorbt/): Python library for backtesting and analyzing trading strategies at scale
-   [Backtrader](https://pypi.org/project/backtrader/): Backtesting Engine
-   [Statsmodels](https://www.statsmodels.org/stable/index.html) is a Python module that provides classes and functions for the estimation of many different statistical models
-   [Plotly](https://plotly.com/): Interactive data visualization
-   [Matplotlib](https://pypi.org/project/matplotlib/): Python plotting package
-   [Seaborn](https://pypi.org/project/seaborn/): Statistical data visualization.

## Basic Imports & Installations

-   Let’s begin with installing and importing the Python packages and modules needed

!pip install prophet, pandas-ta, backtesting,financetoolkit,vectorbt, backtrader, sklearn  
!pip install tensorflow  
!pip install keras  
  
  
import pandas as pd  
import numpy as np  
import requests  
import matplotlib.pyplot as plt  
  
import matplotlib.ticker as mtick  
from matplotlib import patheffects  
  
  
from math import floor  
from termcolor import colored as cl  
  
import math  
  
plt.style.use('fivethirtyeight')  
plt.rcParams\['figure.figsize'\] = (12,6)  
  
from plotly.subplots import make\_subplots  
import plotly.graph\_objects as go  
  
import statsmodels.api as sm  
from statsmodels import regression  
  
from sklearn.model\_selection import train\_test\_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy\_score  
from sklearn.metrics import classification\_report  
from sklearn.metrics import confusion\_matrix  
from sklearn.model\_selection import ParameterGrid  
  
from sklearn.metrics import mean\_absolute\_error, mean\_squared\_error  
from sklearn.model\_selection import TimeSeriesSplit  
from sklearn.preprocessing import MinMaxScaler  
  
from keras.models import Sequential  
from keras.layers import Dense, LSTM, Dropout  
from keras.optimizers import Adam  
  
import vectorbt as vbt  
import backtrader as bt  
  
  
import seaborn as sns  
  
from prophet import Prophet  
import itertools  
  
from prophet.diagnostics import cross\_validation, performance\_metrics  
  
import pandas\_ta as ta  
  
from backtesting import Strategy, Backtest  
  
from backtesting.lib import crossover  
  
from financetoolkit import Toolkit  

## Reading & Plotting Input Stock Data

-   One of the reliable API’s from which you can get historical daily price-volume stock market data is [TwelveData](https://twelvedata.com/). It enables access to various financial assets (stocks, ETFs, forex, etc.) from anywhere at any time.
-   Using the function get\_historical\_data \[1\] to fetch the [PLTR](https://www.palantir.com/) daily stock data from 2022–01–03 to 2025–07–24

def get\_historical\_data(symbol, start\_date):  
    api\_key = 'YOUR API KEY'  
    api\_url = f'https://api.twelvedata.com/time\_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api\_key}'  
    raw\_df = requests.get(api\_url).json()  
    df = pd.DataFrame(raw\_df\['values'\]).iloc\[::-1\].set\_index('datetime').astype(float)  
    df = df\[df.index >= start\_date\]  
    df.index = pd.to\_datetime(df.index)  
    return df  
  
stock\_symbol='PLTR'  
start\_date='2022-01-01'  
df = get\_historical\_data(stock\_symbol, start\_date)  
df.tail()  
  
           open      high      low        close      volume  
datetime       
2025\-07-18 154.86000 154.92000 151.899990 153.520000 45771600.0  
2025\-07-21 153.88000 155.44000 151.360000 151.789990 45072800.0  
2025\-07-22 150.85001 151.78999 145.061996 149.070007 49880800.0  
2025\-07-23 149.74001 155.00000 148.287000 154.630000 47869400.0  
2025\-07-24 153.97000 155.15000 152.580000 154.215000 2741478.0

The info() function provides a structured overview of the dataset

df.info()  
  
<class 'pandas.core.frame.DataFrame'\>  
DatetimeIndex: 892 entries, 2022\-01-03 to 2025\-07-24  
Data columns (total 5 columns):  
 \#   Column  Non-Null Count  Dtype    
\---  ------  --------------  -----    
 0   open    892 non-null    float64  
 1   high    892 non-null    float64  
 2   low     892 non-null    float64  
 3   close   892 non-null    float64  
 4   volume  892 non-null    float64  
dtypes: float64(5)  
memory usage: 41.8 KB

-   Using [Plotly](https://plotly.com/) to create the basic OHLC & Volume chart \[2\]

fig = make\_subplots(rows=2, cols=1, shared\_xaxes=True, vertical\_spacing=0.05, row\_heights = \[0.7, 0.3\])  
fig.add\_trace(go.Candlestick(x=df.index,  
                             open\=df\['open'\],  
                             high=df\['high'\],  
                             low=df\['low'\],  
                             close=df\['close'\],  
                             name='PLTR'),  
              row=1, col=1)  
  
  
\# Plotting volume chart on the second row   
fig.add\_trace(go.Bar(x=df.index,  
                     y=df\['volume'\],  
                     name='Volume',  
                     marker=dict(color='black', opacity=1.0)),  
              row=2, col=1)  
  
  
  
\# Configuring layout  
fig.update\_layout(title='PLTR Candlestick Chart',  
                  yaxis=dict(title='Price (USD)'),  
                  height=1000,  
                 template = 'plotly\_white')  
  
\# Configuring axes and subplots  
fig.update\_xaxes(rangeslider\_visible=False, row=1, col=1)  
fig.update\_xaxes(rangeslider\_visible=False, row=2, col=1)  
fig.update\_yaxes(title\_text='Price (USD)', row=1, col=1)  
fig.update\_yaxes(title\_text='Volume', row=2, col=1)  
  
fig.show()

![PLTR OHLC & Volume chart](https://miro.medium.com/v2/resize:fit:700/1*di3zrc4LCc8TXnjvyh4DxQ.png)
PLTR OHLC & Volume chart

## Stock Candlesticks with Indicators

-   Adding the popular trading indicators such as EMA, SMA, Bollinger Bands (BB), RSI, and ATR to the above chart \[2\]

df.rename(columns={'open': 'Open','close': 'Close','high':'High','low': 'Low','volume':'Volume'},inplace=True)  
\# Adding Moving Averages  
df\['EMA9'\] = df\['Close'\].ewm(span = 9, adjust = False).mean() \# Exponential 9-Period Moving Average  
df\['SMA20'\] = df\['Close'\].rolling(window=20).mean() \# Simple 20-Period Moving Average  
df\['SMA50'\] = df\['Close'\].rolling(window=50).mean() \# Simple 50-Period Moving Average  
df\['SMA100'\] = df\['Close'\].rolling(window=100).mean() \# Simple 100-Period Moving Average  
df\['SMA200'\] = df\['Close'\].rolling(window=200).mean() \# Simple 200-Period Moving Average  
  
\# Adding RSI for 14-periods   
delta = df\['Close'\].diff() \# Calculating delta  
gain = delta.where(delta > 0,0) \# Obtaining gain values  
loss = -delta.where(delta < 0,0) \# Obtaining loss values  
avg\_gain = gain.rolling(window=14).mean() \# Measuring the 14-period average gain value  
avg\_loss = loss.rolling(window=14).mean() \# Measuring the 14-period average loss value  
rs = avg\_gain/avg\_loss \# Calculating the RS  
df\['RSI'\] = 100 - (100 / (1 + rs)) \# Creating an RSI column to the Data Frame   
  
\# Adding Bollinger Bands 20-periods  
df\['BB\_UPPER'\] = df\['SMA20'\] + 2\*df\['Close'\].rolling(window=20).std() \# Upper Band  
df\['BB\_LOWER'\] = df\['SMA20'\] - 2\*df\['Close'\].rolling(window=20).std() \# Lower Band  
  
\# Adding ATR 14-periods  
df\['TR'\] = pd.DataFrame(np.maximum(np.maximum(df\['High'\] - df\['Low'\], abs(df\['High'\] - df\['Close'\].shift())), abs(df\['Low'\] - df\['Close'\].shift())), index = df.index)  
df\['ATR'\] = df\['TR'\].rolling(window = 14).mean() \# Creating an ART column to the Data Frame   
  
\# Plotting Candlestick charts with indicators  
fig = make\_subplots(rows=4, cols=1, shared\_xaxes=True, vertical\_spacing=0.05,row\_heights=\[0.6, 0.10, 0.10, 0.20\])  
  
\# Candlestick   
fig.add\_trace(go.Candlestick(x=df.index,  
                             open\=df\['Open'\],  
                             high=df\['High'\],  
                             low=df\['Low'\],  
                             close=df\['Close'\],  
                             name='PLTR'),  
              row=1, col=1)  
  
\# Moving Averages  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['EMA9'\],  
                         mode='lines',  
                         line=dict(color='#90EE90'),  
                         name='EMA9'),  
              row=1, col=1)  
  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['SMA20'\],  
                         mode='lines',  
                         line=dict(color='yellow'),  
                         name='SMA20'),  
              row=1, col=1)  
  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['SMA50'\],  
                         mode='lines',  
                         line=dict(color='orange'),  
                         name='SMA50'),  
              row=1, col=1)  
  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['SMA100'\],  
                         mode='lines',  
                         line=dict(color='purple'),  
                         name='SMA100'),  
              row=1, col=1)  
  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['SMA200'\],  
                         mode='lines',  
                         line=dict(color='red'),  
                         name='SMA200'),  
              row=1, col=1)  
  
\# Bollinger Bands  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['BB\_UPPER'\],  
                         mode='lines',  
                         line=dict(color='#00BFFF'),  
                         name='Upper Band'),  
              row=1, col=1)  
  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['BB\_LOWER'\],  
                         mode='lines',  
                         line=dict(color='#00BFFF'),  
                         name='Lower Band'),  
              row=1, col=1)  
  
\# Relative Strengh Index (RSI)  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['RSI'\],  
                         mode='lines',  
                         line=dict(color='#CBC3E3'),  
                         name='RSI'),  
              row=2, col=1)  
  
\# Adding marking lines at 70 and 30 levels  
fig.add\_shape(type\="line",  
              x0=df.index\[0\], y0=70, x1=df.index\[-1\], y1=70,  
              line=dict(color="red", width=2, dash="dot"),  
              row=2, col=1)  
fig.add\_shape(type\="line",  
              x0=df.index\[0\], y0=30, x1=df.index\[-1\], y1=30,  
              line=dict(color="#90EE90", width=2, dash="dot"),  
              row=2, col=1)  
  
\# Average True Range (ATR)  
fig.add\_trace(go.Scatter(x=df.index,  
                         y=df\['ATR'\],  
                         mode='lines',  
                         line=dict(color='#00BFFF'),  
                         name='ATR'),  
              row=3, col=1)  
  
  
\# Volume  
fig.add\_trace(go.Bar(x=df.index,  
                     y=df\['Volume'\],  
                     name='Volume',  
                     marker=dict(color='black', opacity=1.0)),  
              row=4, col=1)  
  
  
\# Layout  
fig.update\_layout(title='PLTR Candlestick with Indicators',  
                  yaxis=dict(title='Price (USD)'),  
                  height=1000,  
                 template = 'plotly\_white')  
  
\# Axes and subplots  
fig.update\_xaxes(rangeslider\_visible=False, row=1, col=1)  
fig.update\_xaxes(rangeslider\_visible=False, row=4, col=1)  
fig.update\_yaxes(title\_text='Price (USD)', row=1, col=1)  
fig.update\_yaxes(title\_text='RSI', row=2, col=1)  
fig.update\_yaxes(title\_text='ATR', row=3, col=1)  
fig.update\_yaxes(title\_text='Volume', row=4, col=1)  
  
fig.show()

![PLTR OHLC & Volume chart with 5 trading indicators such as EMA, SMA, BB, RSI, and ATR.](https://miro.medium.com/v2/resize:fit:700/1*XhqXje_e_YxSNQYVwsXcMQ.png)
PLTR OHLC & Volume chart with 5 trading indicators such as EMA, SMA, BB, RSI, and ATR.

## Stock Volatility & Returns

-   Calculating and plotting the PLTR daily/cumulative returns and the annualized volatility \[5\]

stock\_data=get\_historical\_data(stock\_symbol, start\_date)  
  
\# Calculating daily returns  
  
stock\_data\['Daily\_Return'\] = stock\_data\['close'\].pct\_change()  
  
\# Calculating cumulative returns  
daily\_pct\_change = stock\_data\['Daily\_Return'\]  
daily\_pct\_change.fillna(0, inplace=True)  
cumprod\_daily\_pct\_change = (1 + daily\_pct\_change).cumprod()  
  
\# Calculating the annualized volatility  
volatility= stock\_data\['Daily\_Return'\].std()\*np.sqrt(252)  
  
  
\# Plotting the daily returns  
plt.figure(figsize=(10, 6))  
stock\_data\['Daily\_Return'\].plot()  
plt.title(f'Daily Returns of {stock\_symbol}')  
plt.ylabel('Daily Returns')  
plt.xlabel('Date')  
plt.show()  
  
\# Plotting the cumulative returns  
  
fig = plt.figure(figsize=(12, 7))  
ax1 = fig.add\_subplot(1, 1, 1)  
cumprod\_daily\_pct\_change.plot(ax=ax1)  
ax1.set\_xlabel('Date')  
ax1.set\_ylabel('Cumulative Return')  
ax1.set\_title('PLTR Cumulative Returns')  
plt.show()  
  
print(volatility)  
  
0.7077963677957707

![Daily Returns of PLTR](https://miro.medium.com/v2/resize:fit:700/1*_EC9p8rHQWxAcHTj-kVl8A.png)
Daily Returns of PLTR

![Cumulative Returns of PLTR](https://miro.medium.com/v2/resize:fit:700/1*IUhfvRIlYfxFFhB3ptAuhw.png)
Cumulative Returns of PLTR

## Alpha & Beta Coefficients

-   Let’s calculate the alpha and beta coefficients of PLTR against SPY in 2025.
-   Reading the stock data

stock\_symbol='PLTR'  
start\_date='2025-01-01'  
pltr = get\_historical\_data(stock\_symbol, start\_date)  
  
stock\_symbol='SPY'  
start\_date='2025-01-01'  
spy = get\_historical\_data(stock\_symbol, start\_date)

-   Calculating the stock daily returns

return\_pltr = pltr.close.pct\_change()\[1:\]  
return\_spy = spy.close.pct\_change()\[1:\]

-   Implementing the statsmodels OLS linear regression \[9\]

\# Regression model  
X = return\_spy.values  
Y = return\_pltr.values  
  
def linreg(x,y):  
    x = sm.add\_constant(x)  
    model = regression.linear\_model.OLS(y,x).fit()  
  
    \# We are removing the constant  
    x = x\[:, 1\]  
    return model.params\[0\], model.params\[1\]  
  
alpha, beta = linreg(X,Y)  
print('alpha: ' + str(alpha))  
print('beta: ' + str(beta))  
  
alpha: 0.005327166788339888  
beta: 1.9632707300261591

-   Plotting the linear regression line

\# Plotting  
X2 = np.linspace(X.min(), X.max(), 100)  
Y\_hat = X2 \* beta + alpha  
plt.figure(figsize=(10,7))  
plt.scatter(X, Y, alpha=0.3) \# Plot the raw data  
plt.xlabel("SPY Daily Return")  
plt.ylabel("PLTR Daily Return")  
plt.plot(X2, Y\_hat, 'r', alpha=0.9)  
plt.show()

![PLTR vs SPY Daily Returns in 2025: Linear Regression Line.](https://miro.medium.com/v2/resize:fit:700/1*g0V5snR-MK-RZuEtGFbgJw.png)
PLTR vs SPY Daily Returns in 2025: Linear Regression Line.

## On-Balance Volume (OBV)

-   Calculating and visualizing the On-Balance Volume (OBV) for the selected stock \[5\]

\# Calculating the On-Balance Volume (OBV)  
obv = \[0\]  
for i in range(1, len(stock\_data)):  
    if stock\_data\['close'\].iloc\[i\] > stock\_data\['close'\].iloc\[i-1\]:  
        obv.append(obv\[-1\] + stock\_data\['volume'\].iloc\[i\])  
    elif stock\_data\['close'\].iloc\[i\] < stock\_data\['close'\].iloc\[i-1\]:  
        obv.append(obv\[-1\] - stock\_data\['volume'\].iloc\[i\])  
    else:  
        obv.append(obv\[-1\])  
  
stock\_data\['OBV'\] = obv  
  
\# Plotting OBV  
plt.figure(figsize=(10, 6))  
stock\_data\['OBV'\].plot()  
plt.title(f'On-Balance Volume (OBV) of {stock\_symbol}')  
plt.ylabel('OBV')  
plt.xlabel('Date')  
plt.show()

![On-Balance Volume (OBV) of PLTR](https://miro.medium.com/v2/resize:fit:700/1*80p4s9SExx8zPAUkCHffag.png)
On-Balance Volume (OBV) of PLTR

## Basic Momentum

-   Calculating and plotting the basic momentum indicator as the rate of change (ROC) in a stock’s price over a given period \[5\]

def calculate\_momentum(data, period=14):  
 return data\['close'\].diff(periods=period)  
  
\# Adding Momentum to DataFrame  
stock\_data\['Momentum'\] = calculate\_momentum(stock\_data)  
  
\# Plotting Momentum  
plt.figure(figsize=(10, 6))  
stock\_data\['Momentum'\].plot()  
plt.title(f'Momentum of {stock\_symbol}')  
plt.ylabel('Momentum')  
plt.xlabel('Date')  
plt.show()

![Momentum of PLTR.](https://miro.medium.com/v2/resize:fit:700/1*o-FJybmGUBSKs1GRv28IhA.png)
Momentum of PLTR.

## Momentum Trading Strategy

-   Calculating and visualizing the momentum buy and sell trading signals alongside the stock price \[5\]

def trading\_strategy(data):  
    buy\_signals = \[\]  
    sell\_signals = \[\]  
      
    for i in range(len(data)):  
        \# Buy if momentum is positive and OBV is increasing  
        if data\['Momentum'\].iloc\[i\] > 0 and data\['OBV'\].iloc\[i\] > data\['OBV'\].iloc\[i-1\]:  
            buy\_signals.append(data\['close'\].iloc\[i\])  
            sell\_signals.append(np.nan)  
        \# Sell if momentum is negative  
        elif data\['Momentum'\].iloc\[i\] < 0:  
            sell\_signals.append(data\['close'\].iloc\[i\])  
            buy\_signals.append(np.nan)  
        else:  
            buy\_signals.append(np.nan)  
            sell\_signals.append(np.nan)  
  
    return buy\_signals, sell\_signals  
  
stock\_data\['Buy\_Signals'\], stock\_data\['Sell\_Signals'\] = trading\_strategy(stock\_data)  
  
plt.figure(figsize=(12,6))  
plt.plot(stock\_data\['close'\], label='Close Price', alpha=0.5)  
plt.scatter(stock\_data.index, stock\_data\['Buy\_Signals'\], label='Buy Signal', marker='^', color='green')  
plt.scatter(stock\_data.index, stock\_data\['Sell\_Signals'\], label='Sell Signal', marker='v', color='red')  
plt.title(f'Momentum Trading Signals for {stock\_symbol}')  
plt.xlabel('Date')  
plt.ylabel('Price')  
plt.legend()  
plt.show()

![Momentum Trading Signals for PLTR.](https://miro.medium.com/v2/resize:fit:700/1*XGFqssNhoJbjyAvG2e3BCw.png)
Momentum Trading Signals for PLTR.

-   Defining the position

position = \[\]  
  
for i in range(len(stock\_data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(stock\_data\['close'\])):  
    if math.isnan(stock\_data\["Buy\_Signals"\]\[i\]) == False:  
        position\[i\] = 1  
    elif math.isnan(stock\_data\["Sell\_Signals"\]\[i\]) == False:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]

-   Calculating and plotting the expected strategy returns vs B&H \[1\]

rets = stock\_data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label="Strategy")  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label="Strategy")  
plt.legend()  
plt.show()

![Momentum Strategy vs B&H Daily & Cumulative Returns](https://miro.medium.com/v2/resize:fit:700/1*RInZ4yavEzJuks-lGwPKiw.png)
Momentum Strategy vs B&H Daily & Cumulative Returns

-   Comparing Momentum Strategy vs B&H Cumulative Returns on 2025–07–24

  
#B&H  
rets\_cum.tail(1)  
  
datetime  
2025\-07-24    7.357258  
  
  
#Strategy  
strat\_cum.tail(1)  
  
datetime  
2025\-07-24    204.517752  
  
  
#Ratio Strategy / B&H  
  
ratio=strat\_cum.tail(1)/rets\_cum.tail(1)  
print(ratio)  
  
datetime  
2025\-07-24    27.798092

## Supervised ML Forecast of Strategy Position

-   Let’s employ the binary classification model [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) to predict the AT position based on Momentum and OBV as the model features \[5\], viz.

X = stock\_data\[\['Momentum', 'OBV'\]\]  
y = position \# You can define target based on trading strategy  
  
\# Splitting the dataset  
X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.3)  
  
\# Training the model  
#model = RandomForestClassifier()  
model=DecisionTreeClassifier()  
model.fit(X\_train, y\_train)  
  
\# Testing the model  
predictions = model.predict(X\_test)  
print(f"Model Accuracy: {accuracy\_score(y\_test, predictions)}")  
Model Accuracy: 0.9850746268656716  
  
target\_names=\['0','1'\]  
print(classification\_report(y\_test, predictions, target\_names=target\_names))  
  
                precision    recall  f1-score   support  
  
           0       0.97      0.99      0.98       110  
           1       0.99      0.98      0.99       158  
  
    accuracy                           0.99       268  
   macro avg       0.98      0.99      0.98       268  
weighted avg       0.99      0.99      0.99       268

-   Plotting [the normalized confusion matrix](https://stackoverflow.com/questions/20927368/how-to-normalize-a-confusion-matrix)

cm = confusion\_matrix(y\_test, predictions)  
\# Normalise  
cmn = cm.astype('float') / cm.sum(axis=1)\[:, np.newaxis\]  
fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target\_names, yticklabels=target\_names)  
plt.ylabel('Actual')  
plt.xlabel('Predicted')  
plt.show(block=False)

![Decision Tree Classifier Normalized Confusion Matrix](https://miro.medium.com/v2/resize:fit:700/1*ntX_Y7YuIwJzxaSGiCBD3g.png)
Decision Tree Classifier Normalized Confusion Matrix

## Price Dynamics vs Volume Ratio

-   Calculating and plotting the ‘Volume Ratio’ — a powerful KPI for identifying irregular volume activity \[10\].

\# Reading the stock data  
  
stock\_symbol='PLTR'  
start\_date='2022-01-01'  
mydf = get\_historical\_data(stock\_symbol, start\_date)  
  
data=mydf.copy()  
  
#Calculating Volume\_Ratio  
  
data\['50\_day\_MA'\] = data\['close'\].rolling(window=50).mean()  \# Calculate the 50-day moving average for closing price  
data\['200\_day\_MA'\] = data\['close'\].rolling(window=200).mean()  \# Calculate the 200-day moving average for closing price  
data\['Volume\_MA20'\] = data\['volume'\].rolling(window=20).mean()  \# Calculate the 20-day moving average for volume  
data\['Volume\_Ratio'\] = data\['volume'\] / data\['Volume\_MA20'\]  \# Compute the ratio of current volume to the 20-day moving average volume  
  
\# Set the percentile threshold to capture top 5% of high-volume trading days  
percentile\_level = 0.95  
  
\# Calculate moving averages  
stock\_data=mydf.copy()  
stock\_data\['Short\_Term\_MA'\] = stock\_data\['close'\].rolling(window=50).mean()  
stock\_data\['Long\_Term\_MA'\] = stock\_data\['close'\].rolling(window=200).mean()  
  
\# Compute volume moving average and ratio  
stock\_data\['Avg\_Volume\_20'\] = stock\_data\['volume'\].rolling(window=20).mean()  
stock\_data\['Volume\_Index'\] = stock\_data\['volume'\] / stock\_data\['Avg\_Volume\_20'\]  
  
\# Determine dynamic threshold based on volume ratio  
dynamic\_threshold = stock\_data\['Volume\_Index'\].quantile(percentile\_level)  
highlight\_mask = stock\_data\['Volume\_Index'\] > dynamic\_threshold  
  
\# Prepare the figure for subplots  
fig, (price\_ax, volume\_ax, hist\_ax) = plt.subplots(3, figsize=(25, 14))  
\# Plot closing price in price\_ax  
price\_ax.plot(stock\_data.index, stock\_data\['close'\], label='Closing Price', color='blue')  
price\_ax.scatter(stock\_data.index\[highlight\_mask\], stock\_data\['close'\]\[highlight\_mask\], color='red', s=100, label='High Volume Days')  
price\_ax.set\_title(f'{stock\_symbol} Price Movement (2022–2025)')  
price\_ax.set\_ylabel('Price')  
price\_ax.legend(loc='upper left')  
\# Add secondary axis for volume in price\_ax  
volume\_ax2 = price\_ax.twinx()  
volume\_ax2.bar(stock\_data.index, stock\_data\['volume'\], color='gray', alpha=0.3, label='Volume')  
volume\_ax2.plot(stock\_data.index, stock\_data\['Avg\_Volume\_20'\], color='purple', label='20-Day Avg Volume', alpha=0.3)  
volume\_ax2.set\_ylabel('Volume')  
volume\_ax2.legend(loc='lower left')  
\# Plot volume ratio in volume\_ax  
volume\_ax.plot(stock\_data.index, stock\_data\['Volume\_Index'\], label='Volume to MA Ratio', color='green')  
volume\_ax.axhline(y=dynamic\_threshold, color='red', linestyle='--', label=f'{percentile\_level\*100:.0f}% Threshold')  
volume\_ax.set\_title(f'{stock\_symbol} Volume Ratio Over Time')  
volume\_ax.set\_ylabel('Volume Ratio')  
volume\_ax.legend(loc='upper right')  
\# Plot histogram of volume ratio in hist\_ax  
hist\_ax.hist(stock\_data\['Volume\_Index'\], bins=50, color='green', alpha=0.7, label='Distribution of Volume Ratios')  
hist\_ax.axvline(x=dynamic\_threshold, color='red', linestyle='--', label=f'{percentile\_level\*100:.0f}% Threshold')  
hist\_ax.set\_title(f'{stock\_symbol} Volume Ratio Histogram')  
hist\_ax.set\_xlabel('Volume Ratio')  
hist\_ax.set\_ylabel('Frequency')  
hist\_ax.legend(loc='upper right')  
\# Adjust the layout for better clarity and show the plots  
plt.tight\_layout()  
plt.show()

![PLTR Price Movement and Volume Ratio 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*yoSdIOb1zPjkczMNa8poJA.png)
PLTR Price Movement and Volume Ratio 2022–2025.

## OBV-EMA Trading Strategy

-   Implementing the OBV-EMA trading strategy as follows \[11\]

def calculate\_obv(data):  
    obv = \[0\]  
    for i in range(1, len(data)):  
        if data\['close'\].iloc\[i\] > data\['close'\].iloc\[i-1\]:  
            obv.append(obv\[-1\] + data\['volume'\].iloc\[i\])  
        elif data\['close'\].iloc\[i\] < data\['close'\].iloc\[i-1\]:  
            obv.append(obv\[-1\] - data\['volume'\].iloc\[i\])  
        else:  
            obv.append(obv\[-1\])  
    return obv  
  
\# Retrieve data  
  
stock\_symbol='PLTR'  
start\_date='2025-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
\# Calculate OBV  
data\['OBV'\] = calculate\_obv(data)  
data\['OBV\_EMA'\] = data\['OBV'\].ewm(span=30).mean()  \# 20-day EMA of OBV  
  
\# Generate buy and sell signals  
buy\_signal = (data\['OBV'\] > data\['OBV\_EMA'\]) & (data\['OBV'\].shift(1) <= data\['OBV\_EMA'\].shift(1))  
sell\_signal = (data\['OBV'\] < data\['OBV\_EMA'\]) & (data\['OBV'\].shift(1) >= data\['OBV\_EMA'\].shift(1))  
  
\# Plotting with adjusted subplot sizes  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[3, 1\]})  
  
\# Stock price plot with buy and sell signals  
ax1.plot(data\['close'\], label='Close Price', alpha=0.5)  
ax1.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green')  
ax1.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90,marker='v', color='red')  
ax1.set\_title(f'{stock\_symbol} Stock Price')  
ax1.set\_ylabel('Price')  
ax1.legend()  
  
\# OBV subplot  
ax2.plot(data\['OBV'\], label='OBV', color='blue')  
ax2.plot(data\['OBV\_EMA'\], label='30-day EMA of OBV', color='orange', alpha=0.6)  
ax2.set\_title(f'{stock\_symbol} On-Balance Volume (OBV)')  
ax2.set\_ylabel('OBV')  
ax2.legend()  
  
plt.tight\_layout()  
plt.show()

![PLTR OBV-EMA trading signals vs Close price, OBV and 30-day EMA of OBV indicators in 2025.](https://miro.medium.com/v2/resize:fit:700/1*SzM2HAYjCdmwayRScUop-w.png)
PLTR OBV-EMA trading signals vs Close price, OBV and 30-day EMA of OBV indicators in 2025.

-   Defining the position

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]

-   Comparing the OBV-EMA trading strategy vs B&H returns

rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend()  
plt.show()

![PLTR OBV-EMA Trading Strategy vs B&H Daily and Cumulative Returns in 2025.](https://miro.medium.com/v2/resize:fit:700/1*oZuFql6DuVfvoyo9fpus2A.png)
PLTR OBV-EMA Trading Strategy vs B&H Daily and Cumulative Returns in 2025.

## VPT-MA Trading Strategy

-   Implementing the VPT-MA trading strategy \[11\]

def calculate\_vpt(data):  
    vpt = \[0\]  
    for i in range(1, len(data)):  
        price\_change = data\['close'\].iloc\[i\] - data\['close'\].iloc\[i-1\]  
        vpt.append(vpt\[-1\] + (data\['volume'\].iloc\[i\] \* price\_change / data\['close'\].iloc\[i-1\]))  
    return vpt  
  
  
ticker=stock\_symbol  
  
\# Calculate VPT  
data\['VPT'\] = calculate\_vpt(data)  
data\['VPT\_MA'\] = data\['VPT'\].rolling(window=10).mean()  \# 10-day moving average  
  
\# Generate buy and sell signals  
buy\_signal = (data\['VPT'\] > data\['VPT\_MA'\]) & (data\['VPT'\].shift(1) <= data\['VPT\_MA'\].shift(1))  
sell\_signal = (data\['VPT'\] < data\['VPT\_MA'\]) & (data\['VPT'\].shift(1) >= data\['VPT\_MA'\].shift(1))  
  
\# Plotting  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[2, 1\]})  
  
\# Stock price plot with buy and sell signals  
ax1.plot(data\['close'\], label='Close Price', alpha=0.5)  
ax1.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green')  
ax1.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90,marker='v', color='red')  
ax1.set\_title(f'{ticker} Stock Price')  
ax1.set\_ylabel('Price')  
ax1.legend()  
  
\# VPT subplot  
ax2.plot(data\['VPT'\], label='VPT', color='blue')  
ax2.plot(data\['VPT\_MA'\], label='10-day MA of VPT', color='orange', alpha=0.6)  
ax2.set\_title(f'{ticker} Volume Price Trend (VPT)')  
ax2.set\_ylabel('VPT')  
ax2.legend()  
  
plt.tight\_layout()  
plt.show()

![PLTR VPT-MA trading signals vs Close price, VPT and 10-day MA of VPT indicators in 2025.](https://miro.medium.com/v2/resize:fit:700/1*tnHfAhc14zeDfnvuOxJfbw.png)
PLTR VPT-MA trading signals vs Close price, VPT and 10-day MA of VPT indicators in 2025.

-   Defining the position and comparing the VPT-MA trading strategy vs B&H returns

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend()  
plt.show()

![PLTR VPT-MA Trading Strategy vs B&H Daily and Cumulative Returns in 2025.](https://miro.medium.com/v2/resize:fit:700/1*GDgQv7YJ9XR8src4q5BnLQ.png)
PLTR VPT-MA Trading Strategy vs B&H Daily and Cumulative Returns in 2025.

## VROC-MA Trading Strategy

-   Considering the VROC-MA trading strategy 2022–2025 \[11\]

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
  
def calculate\_vroc(data, period=20):  
    vroc = ((data\['volume'\] - data\['volume'\].shift(period)) / data\['volume'\].shift(period)) \* 100  
    return vroc  
  
\# Calculate VROC  
vroc\_period = 60  \# 60-day VROC  
data\['VROC'\] = calculate\_vroc(data, vroc\_period)  
data\['VROC\_MA'\] = data\['VROC'\].rolling(window=vroc\_period).mean()  \# 60-day moving average of VROC  
  
\# Generate buy and sell signals  
buy\_signal = (data\['VROC'\] > data\['VROC\_MA'\]) & (data\['VROC'\].shift(1) <= data\['VROC\_MA'\].shift(1))  
sell\_signal = (data\['VROC'\] < data\['VROC\_MA'\]) & (data\['VROC'\].shift(1) >= data\['VROC\_MA'\].shift(1))  
  
\# Plotting  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[3, 1\]})  
  
\# Stock price plot with buy and sell signals  
ax1.plot(data\['close'\], label='Close Price', alpha=0.5)  
ax1.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green')  
ax1.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90,marker='v', color='red')  
ax1.set\_title(f'{ticker} Stock Price')  
ax1.set\_ylabel('Price')  
ax1.legend()  
  
\# VROC subplot  
ax2.plot(data\['VROC'\], label='VROC', color='blue')  
ax2.plot(data\['VROC\_MA'\], label=f'{vroc\_period}\-day MA of VROC', color='orange', alpha=0.6)  
ax2.set\_title(f'{ticker} Volume Rate of Change (VROC)')  
ax2.set\_ylabel('VROC (%)')  
ax2.legend()  
  
plt.tight\_layout()  
plt.show()

![PLTR VROC-MA trading signals vs Close price, VROC and 60-day MA of VROC indicators in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*buLeOCa0xSqKsiVVBn3CHA.png)
PLTR VROC-MA trading signals vs Close price, VROC and 60-day MA of VROC indicators in 2022-2025.

-   Defining the position and calculating the VROC-MA trading strategy vs B&H returns

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend(loc="upper left")  
plt.show()

![PLTR VROC-MA Trading Strategy vs B&H Daily and Cumulative Returns in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*pjXnRqMq9nSca_2iHVKBuA.png)
PLTR VROC-MA Trading Strategy vs B&H Daily and Cumulative Returns in 2022-2025.

## VWAP Trading Strategy

-   Let’s look at the VWAP trading strategy \[11\]

stock\_symbol='PLTR'  
start\_date='2025-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
def calculate\_vwap(data):  
    data\['Cumulative\_Volume\_Price'\] = (data\['close'\] \* data\['volume'\]).cumsum()  
    data\['Cumulative\_Volume'\] = data\['volume'\].cumsum()  
    vwap = data\['Cumulative\_Volume\_Price'\] / data\['Cumulative\_Volume'\]  
    return vwap  
  
  
\# Calculate VWAP  
data\['VWAP'\] = calculate\_vwap(data)  
  
\# Generate buy and sell signals  
buy\_signal = (data\['close'\] > data\['VWAP'\]) & (data\['close'\].shift(1) <= data\['VWAP'\].shift(1))  
sell\_signal = (data\['close'\] < data\['VWAP'\]) & (data\['close'\].shift(1) >= data\['VWAP'\].shift(1))  
  
\# Plotting  
plt.figure(figsize=(16, 6))  
plt.plot(data\['close'\], label='Close Price', alpha=0.5)  
plt.plot(data\['VWAP'\], label='VWAP', color='orange', alpha=0.6)  
plt.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green')  
plt.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90,marker='v', color='red')  
plt.title(f'{ticker} Stock Price and VWAP with Buy/Sell Signals')  
plt.ylabel('Price')  
plt.legend()  
plt.show()

![PLTR Stock Price and VWAP with Buy/Sell Signals in 2025.](https://miro.medium.com/v2/resize:fit:700/1*QRH6fTfg_773hJLTl3w80g.png)
PLTR Stock Price and VWAP with Buy/Sell Signals in 2025.

-   Formulating the position and calculating the VWAP trading strategy vs B&H returns

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend(loc="upper left")  
plt.show()

![PLTR VWAP Trading Strategy vs B&H Daily and Cumulative Returns in 2025.](https://miro.medium.com/v2/resize:fit:700/1*cg13K2WjYVGUJzk30L1n4Q.png)
PLTR VWAP Trading Strategy vs B&H Daily and Cumulative Returns in 2025.

## A/D Line Trading Strategy

-   Implementing the A/D line trading strategy \[11\]

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
def calculate\_ad\_line(data):  
    clv = ((data\['close'\] - data\['low'\]) - (data\['high'\] - data\['close'\])) / (data\['high'\] - data\['low'\])  
    clv.fillna(0, inplace=True)  \# Handling division by zero  
    ad\_line = (clv \* data\['volume'\]).cumsum()  
    return ad\_line  
  
  
\# Calculate Accumulation/Distribution Line  
data\['AD\_Line'\] = calculate\_ad\_line(data)  
  
\# Calculate rolling max and min for price for divergence detection  
lookback\_period = 30  \# Example lookback period  
shft=10  
data\['Rolling\_Max'\] = data\['close'\].rolling(window=lookback\_period).max()  
data\['Rolling\_Min'\] = data\['close'\].rolling(window=lookback\_period).min()  
  
\# Detect divergences for buy and sell signals  
buy\_signal = (data\['close'\] == data\['Rolling\_Min'\]) & (data\['AD\_Line'\] > data\['AD\_Line'\].shift(shft))  
sell\_signal = (data\['close'\] == data\['Rolling\_Max'\]) & (data\['AD\_Line'\] < data\['AD\_Line'\].shift(shft))  
  
\# Plotting  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[3, 1\]})  
  
\# Stock price plot with buy and sell signals  
ax1.plot(data\['close'\], label='Close Price', alpha=0.5)  
ax1.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green')  
ax1.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90,marker='v', color='red')  
ax1.set\_title(f'{ticker} Stock Price with Buy/Sell Signals')  
ax1.set\_ylabel('Price')  
ax1.legend()  
  
\# A/D Line subplot  
ax2.plot(data\['AD\_Line'\], label='Accumulation/Distribution Line', color='blue')  
ax2.set\_title(f'{ticker} Accumulation/Distribution Line')  
ax2.set\_ylabel('A/D Line')  
ax2.legend()  
  
plt.tight\_layout()  
plt.show()

![PLTR Stock Price and Accumulation/Distribution (A/D) Line with Buy/Sell Signals in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*xjkmrgQq5DaXmI0OPmUouw.png)
PLTR Stock Price and Accumulation/Distribution (A/D) Line with Buy/Sell Signals in 2022-2025.

-   Defining the position and calculating the A/D line trading strategy vs B&H returns

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend(loc="upper left")  
plt.show()

![PLTR A/D Line Trading Strategy vs B&H Daily and Cumulative Returns in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*FM59Twj8yzgNj1ObdAOgMw.png)
PLTR A/D Line Trading Strategy vs B&H Daily and Cumulative Returns in 2022-2025.

## CMF Trading Strategy

-   Examining ROI of the PLTR CMF trading strategy in 2025 \[11\]

stock\_symbol='PLTR'  
start\_date='2025-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
def calculate\_cmf(data, period=5):  
    mfv = ((data\['close'\] - data\['low'\]) - (data\['high'\] - data\['close'\])) / (data\['high'\] - data\['low'\]) \* data\['volume'\]  
    cmf = mfv.rolling(window=period).sum() / data\['volume'\].rolling(window=period).sum()  
    return cmf  
  
  
\# Calculate CMF  
cmf\_period = 5  \# 5-day CMF  
data\['CMF'\] = calculate\_cmf(data, cmf\_period)  
  
\# Define thresholds for buy and sell signals  
buy\_threshold = 0.05  \# Adjust this value as needed  
sell\_threshold = -0.05  \# Adjust this value as needed  
  
\# Generate buy and sell signals  
buy\_signal = (data\['CMF'\] > buy\_threshold) & (data\['CMF'\].shift(1) <= buy\_threshold)  
sell\_signal = (data\['CMF'\] < sell\_threshold) & (data\['CMF'\].shift(1) >= sell\_threshold)  
  
\# Plotting  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[3, 1\]})  
  
\# Stock price plot with buy and sell signals  
ax1.plot(data\['close'\], label='Close Price', alpha=0.5)  
ax1.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green')  
ax1.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90, marker='v', color='red')  
ax1.set\_title(f'{ticker} Stock Price')  
ax1.set\_ylabel('Price')  
ax1.legend()  
  
\# CMF subplot  
ax2.plot(data\['CMF'\], label='CMF', color='blue')  
ax2.axhline(buy\_threshold, color='green', linestyle='--')  
ax2.axhline(sell\_threshold, color='red', linestyle='--')  
ax2.set\_title(f'{ticker} Chaikin Money Flow (CMF)')  
ax2.set\_ylabel('CMF')  
ax2.legend()  
  
plt.tight\_layout()  
plt.show()

![PLTR Stock Price and CMF Indicator with Buy/Sell Signals in 2025.](https://miro.medium.com/v2/resize:fit:700/1*aAgzOzXqw9coVD5ScEgmrA.png)
PLTR Stock Price and CMF Indicator with Buy/Sell Signals in 2025.

-   Specifying the position and comparing returns of the CMF trading strategy vs B&H

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend(loc="upper left")  
plt.show()

![PLTR CMF Trading Strategy vs B&H Daily and Cumulative Returns in 2025.](https://miro.medium.com/v2/resize:fit:700/1*NkUt86d27jrwfREpJbBlGw.png)
PLTR CMF Trading Strategy vs B&H Daily and Cumulative Returns in 2025.

## MFI Trading Strategy

-   Exploring the PLTR MFI trading strategy in 2022-2025 \[11\]

def calculate\_mfi(data, period=14):  
    high = data\['high'\]  
    low = data\['low'\]  
    close = data\['close'\]  
    volume = data\['volume'\]  
    typical\_price = (high + low + close) / 3  
    money\_flow = typical\_price \* volume  
      
    positive\_flow = money\_flow.where(typical\_price > typical\_price.shift(1), 0)  
    negative\_flow = money\_flow.where(typical\_price < typical\_price.shift(1), 0)  
      
    positive\_mf\_sum = positive\_flow.rolling(window=period).sum()  
    negative\_mf\_sum = negative\_flow.rolling(window=period).sum()  
      
    mfi\_ratio = positive\_mf\_sum / (negative\_mf\_sum + 1e-10)    
    mfi = 100 - (100 / (1 + mfi\_ratio))  
    return mfi  
  
stock\_symbol='PLTR'  
start\_date='2022-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
  
\# Calculate MFI  
mfi\_period = 60  \# Typical period for MFI  
data\['MFI'\] = calculate\_mfi(data, mfi\_period)  
  
\# Define thresholds for buy and sell signals  
overbought\_threshold = 75  
oversold\_threshold = 40  
  
\# Generate buy and sell signals  
buy\_signal = (data\['MFI'\] < oversold\_threshold) & (data\['MFI'\].shift(1) >= oversold\_threshold)  
sell\_signal = (data\['MFI'\] > overbought\_threshold) & (data\['MFI'\].shift(1) <= overbought\_threshold)  
  
\# Plotting  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[3, 1\]})  
  
\# Stock price plot with buy and sell signals  
ax1.plot(data.index, data\['close'\], label='Close Price', alpha=0.5)  
ax1.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green', alpha=0.7)  
ax1.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90,marker='v', color='red', alpha=0.7)  
ax1.set\_title(f'{ticker} Stock Price')  
ax1.set\_ylabel('Price')  
ax1.legend()  
  
\# MFI subplot  
ax2.plot(data.index, data\['MFI'\], label='MFI', color='blue')  
ax2.axhline(overbought\_threshold, color='red', linestyle='--', label='Overbought Threshold')  
ax2.axhline(oversold\_threshold, color='green', linestyle='--', label='Oversold Threshold')  
ax2.set\_title(f'{ticker} Money Flow Index (MFI)')  
ax2.set\_ylabel('MFI')  
ax2.legend()  
  
plt.tight\_layout()  
plt.show()

![PLTR Stock Price and MFI Indicator with Buy/Sell Signals in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*uAjPN4t5N67XNaucIyRAHw.png)
PLTR Stock Price and MFI Indicator with Buy/Sell Signals in 2022-2025.

-   Defining the position and calculating returns of the MFI trading strategy vs B&H

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7)  
strat\_rets.plot(color = 'r', linewidth = 1)  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7)  
strat\_cum.plot(color = 'r', linewidth = 2)  
plt.show()

![PLTR MFI Trading Strategy vs B&H Daily and Cumulative Returns in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*DJtKt9w5-p0BrXr4n8u3FA.png)
PLTR MFI Trading Strategy vs B&H Daily and Cumulative Returns in 2022-2025.

## KO Trading Strategy

-   Diving into the PLTR KO trading strategy in 2022–2025 \[11\]

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
def calculate\_klinger\_oscillator(data, fast\_period=34, slow\_period=55):  
    \# Calculate the Volume Force (VF)  
    dm = ((data\['high'\] + data\['low'\]) / 2) - ((data\['high'\].shift(1) + data\['low'\].shift(1)) / 2)  
    cm = data\['close'\] - data\['close'\].shift(1)  
    vf = dm \* data\['volume'\] \* cm / dm.abs()  
      
    \# Calculate the fast and slow EMAs of VF  
    ko = vf.ewm(span=fast\_period).mean() - vf.ewm(span=slow\_period).mean()  
    return ko  
  
  
\# Calculate Klinger Oscillator  
data\['KO'\] = calculate\_klinger\_oscillator(data)  
  
\# Calculate the signal line (EMA of KO)  
signal\_line\_period = 150  \# Typical signal line period  
data\['KO\_Signal'\] = data\['KO'\].ewm(span=signal\_line\_period).mean()  
  
\# Generate buy and sell signals  
buy\_signal = (data\['KO'\] > data\['KO\_Signal'\]) & (data\['KO'\].shift(1) <= data\['KO\_Signal'\].shift(1))  
sell\_signal = (data\['KO'\] < data\['KO\_Signal'\]) & (data\['KO'\].shift(1) >= data\['KO\_Signal'\].shift(1))  
  
\# Plotting  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[3, 1\]})  
  
\# Stock price plot with buy and sell signals  
ax1.plot(data.index, data\['close'\], label='Close Price', alpha=0.5)  
ax1.scatter(data.index\[buy\_signal\], data\['close'\]\[buy\_signal\], label='Buy Signal', s=90,marker='^', color='green', alpha=0.7)  
ax1.scatter(data.index\[sell\_signal\], data\['close'\]\[sell\_signal\], label='Sell Signal', s=90,marker='v', color='red', alpha=0.7)  
ax1.set\_title(f'{ticker} Stock Price')  
ax1.set\_ylabel('Price')  
ax1.legend()  
  
\# KO subplot  
ax2.plot(data.index, data\['KO'\], label='Klinger Oscillator', color='blue')  
ax2.plot(data.index, data\['KO\_Signal'\], label='Signal Line', color='orange', alpha=0.7)  
ax2.scatter(data.index\[buy\_signal\], data\['KO'\]\[buy\_signal\], label='Buy Signal', s=190,marker='^', color='green', alpha=0.7)  
ax2.scatter(data.index\[sell\_signal\], data\['KO'\]\[sell\_signal\], label='Sell Signal', s=190,marker='v', color='red', alpha=0.7)  
ax2.set\_title(f'{ticker} Klinger Oscillator (KO)')  
ax2.set\_ylabel('KO')  
ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')  
ax2.legend(loc="upper left")  
  
plt.tight\_layout()  
plt.show()

![PLTR Stock Price and Klinger Oscillator (KO) with Buy/Sell Signals in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*7T7hnSkG8fitPu-uOkr5Pg.png)
PLTR Stock Price and Klinger Oscillator (KO) with Buy/Sell Signals in 2022–2025.

-   Defining the position and comparing returns of the KO trading strategy vs B&H

position = \[\]  
  
for i in range(len(data\['close'\])):  
        position.append(0)  
  
          
for i in range(len(data\['close'\])):  
    if buy\_signal.iloc\[i\] == True:  
        position\[i\] = 1  
    elif sell\_signal.iloc\[i\] == True:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = data.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend(loc="upper left")  
plt.show()

![PLTR KO Trading Strategy vs B&H Daily and Cumulative Returns in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*w3OBUQpedpC62ghhPdEYfA.png)
PLTR KO Trading Strategy vs B&H Daily and Cumulative Returns in 2022–2025.

## NVI Trading Strategy

-   Examining the PLTR NVI trading strategy in 2025 \[11\]

stock\_symbol='PLTR'  
start\_date='2025-01-01'  
data = get\_historical\_data(stock\_symbol, start\_date)  
  
def negative\_volume\_index(data):  
    data\['NVI'\] = np.nan  
    nvi = 1000  \# Initial NVI value  
    data\['NVI'\].iloc\[0\] = nvi  
  
    for i in range(1, len(data)):  
        if data\['volume'\].iloc\[i\] < data\['volume'\].iloc\[i - 1\]:  
            nvi += (data\['close'\].iloc\[i\] - data\['close'\].iloc\[i - 1\]) / data\['close'\].iloc\[i - 1\] \* nvi  
        data\['NVI'\].iloc\[i\] = nvi  
  
    return data  
  
def generate\_nvi\_signals(data, window):  
    data\['NVI\_SMA'\] = data\['NVI'\].rolling(window=window).mean()  
    buy\_signals = \[\]  
    sell\_signals = \[\]  
  
      
  
    for i in range(window, len(data)):  
        if data\['NVI'\].iloc\[i\] > data\['NVI\_SMA'\].iloc\[i\] and data\['NVI'\].iloc\[i - 1\] <= data\['NVI\_SMA'\].iloc\[i - 1\]:  
            buy\_signals.append((data.index\[i\], data\['close'\]\[i\]))  
        elif data\['NVI'\].iloc\[i\] < data\['NVI\_SMA'\].iloc\[i\] and data\['NVI'\].iloc\[i - 1\] >= data\['NVI\_SMA'\].iloc\[i - 1\]:  
            sell\_signals.append((data.index\[i\], data\['close'\].iloc\[i\]))  
  
    return buy\_signals, sell\_signals  
  
  
def plot\_nvi(data, buy\_signals, sell\_signals, window):  
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec\_kw={'height\_ratios': \[2, 1\]})  
    fig.autofmt\_xdate()  
  
    ax1.plot(data\['close'\]\[window:\], label='Close Price', alpha=0.5)  
  
    buy\_dates, buy\_prices = zip(\*buy\_signals)  
    sell\_dates, sell\_prices = zip(\*sell\_signals)  
  
    ax1.scatter(buy\_dates, buy\_prices, s=90,marker='^', color='g', label='Buy Signal')  
    ax1.scatter(sell\_dates, sell\_prices, s=90,marker='v', color='r', label='Sell Signal')  
  
    ax1.set\_title('PLTR Price and Negative Volume Index')  
    ax1.set\_ylabel('Price')  
    ax1.legend(loc='best')  
  
    ax2.plot(data\['NVI'\]\[window:\], label='Negative Volume Index', color='purple')  
    ax2.set\_xlabel('Date')  
    ax2.set\_ylabel('NVI')  
    ax2.legend(loc='best')  
  
    plt.show()  
  
df = get\_historical\_data(stock\_symbol, start\_date)  
  
\# Calculate Negative Volume Index  
df = negative\_volume\_index(df)  
  
\# Generate buy and sell signals based on NVI and a simple moving average with a specified window  
window = 3  
buy\_signals, sell\_signals = generate\_nvi\_signals(df, window)  
  
  
\# Plot the results with buy and sell signals  
plot\_nvi(df, buy\_signals, sell\_signals, window)

![PLTR Stock Price and Negative Volume Index (NVI) with Buy/Sell Signals in 2025.](https://miro.medium.com/v2/resize:fit:700/1*1NuE1-g5FfGclq1qbjA5qA.png)
PLTR Stock Price and Negative Volume Index (NVI) with Buy/Sell Signals in 2025.

-   Defining the position and calculating returns of the NVI trading strategy vs B&H

position = \[\]  
  
for i in range(len(df\['close'\])):  
        position.append(0)  
  
          
for i in range(len(df\['close'\])):  
    if nvi\_signal\[i\] == 1:  
        position\[i\] = 1  
    elif nvi\_signal\[i\] == -1:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
  
rets = df.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label="Strategy")  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label="Strategy")  
plt.legend()  
plt.show()

![PLTR NVI Trading Strategy vs B&H Daily and Cumulative Returns in 2025.](https://miro.medium.com/v2/resize:fit:700/1*jXQTmGFP8c-YKZkv_eziyA.png)
PLTR NVI Trading Strategy vs B&H Daily and Cumulative Returns in 2025.

## Optimized Fisher-PVT Trading Strategy

-   Let’s evaluate the trading optimization strategy \[12\] that captures both momentum and market volume**.** This strategy is based on the Fisher Transform and Price Volume Trend (PVT) \[12\].
-   Introducing the two functions to calculate PVT, the Fisher transform and its Signal Line

\# Function to calculate Price Volume Trend (PVT)  
def calculate\_pvt(df):  
    pvt = \[0\]  \# Initial PVT value  
    for i in range(1, len(df)):  
        pvt\_value = pvt\[-1\] + ((df\['Close'\]\[i\] - df\['Close'\]\[i-1\]) / df\['Close'\]\[i-1\]) \* df\['Volume'\]\[i\]  
        pvt.append(pvt\_value)  
    return pd.Series(pvt, index=df.index)  
  
\# Function to calculate Fisher Transform and its Signal line  
def calculate\_fisher\_transform(df, period=10):  
    high\_rolling = df\['High'\].rolling(window=period).max()  
    low\_rolling = df\['Low'\].rolling(window=period).min()  
      
    X = 2 \* ((df\['Close'\] - low\_rolling) / (high\_rolling - low\_rolling) - 0.5)  
    fisher = 0.5 \* np.log((1 + X) / (1 - X))  
    fisher\_signal = fisher.ewm(span=9).mean()  
      
    return fisher, fisher\_signal

-   Preparing the input data for optimization

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
df = get\_historical\_data(stock\_symbol, start\_date)  
  
  
  
data\_feed = bt.feeds.PandasData(dataname=df)  
  
df.rename(columns={'open': 'Open','close': 'Close','high':'High','low': 'Low','volume':'Volume'},inplace=True)  
  
print(df)  
  
              Open       High         Low       Close      Volume  
datetime                                                               
2022\-01-03   18.360000   18.56900   17.860000   18.530000  34283600.0  
2022\-01-04   18.610000   18.84000   17.790000   18.170000  39643200.0  
2022\-01-05   18.030001   18.56800   16.870000   16.960000  58445900.0  
2022\-01-06   16.870000   17.18000   16.055000   16.740000  49737100.0  
2022\-01-07   16.700000   17.33000   16.475000   16.560000  37989300.0  
...                ...        ...         ...         ...         ...  
2025\-07-21  153.880000  155.44000  151.360000  151.789990  45072800.0  
2025\-07-22  150.850010  151.78999  145.061996  149.070007  49880800.0  
2025\-07-23  149.740010  155.00000  148.287000  154.630000  48062000.0  
2025\-07-24  153.980000  155.63000  152.575000  154.860000  38814200.0  
2025\-07-25  155.655000  160.38000  155.630000  159.835000   6924333.0  
  
\[893 rows x 5 columns\]

-   Defining the 2D parameter grid

\# Define the range for the optimization  
shift\_values = range(1, 41)  \# PVT shift from 1 to 40  
fisher\_period\_values = range(5, 41)  \# Fisher period from 5 to 40  
  
\# Generate parameter grid  
param\_grid = {  
    'shift': shift\_values,  
    'fisher\_period': fisher\_period\_values  
}  
grid = ParameterGrid(param\_grid)

-   Optimizing the PVT shift and the Fisher period with [vectorbt](https://pypi.org/project/vectorbt/) backtesting

\# Store results for all combinations  
results = \[\]  
  
\# Optimize PVT shift and Fisher period  
for params in grid:  
    shift\_value = params\['shift'\]  
    fisher\_period = params\['fisher\_period'\]  
      
    \# Calculate Price Volume Trend (PVT)  
    df\['PVT'\] = calculate\_pvt(df)  
      
    \# Calculate Fisher Transform and Signal line  
    df\['Fisher'\], df\['Fisher\_Signal'\] = calculate\_fisher\_transform(df, period=fisher\_period)  
      
    \# Define Entry and Exit signals based on PVT and Fisher Transform  
    df\['Entry'\] = (df\['PVT'\] > df\['PVT'\].shift(shift\_value)) & (df\['Fisher'\] > df\['Fisher\_Signal'\])  
    df\['Exit'\] = (df\['PVT'\] < df\['PVT'\].shift(shift\_value)) & (df\['Fisher'\] < df\['Fisher\_Signal'\])  
      
    \# Filter data for the test period   
    df\_test = df\[(df.index.year >= 2022) & (df.index.year <= 2025)\]  
      
    \# Backtest using vectorbt  
    portfolio = vbt.Portfolio.from\_signals(  
        close=df\_test\['Close'\],  
        entries=df\_test\['Entry'\],  
        exits=df\_test\['Exit'\],  
        init\_cash=10000,  
        fees=0.002  
    )  
      
    \# Store the result  
    results.append({  
        'shift': shift\_value,  
        'fisher\_period': fisher\_period,  
        'performance': portfolio.stats()\['Total Return \[%\]'\]  
    })

-   Plotting the Heatmap of Total Return by Shift and Fisher Period

\# Convert results to DataFrame  
results\_df = pd.DataFrame(results)  
  
\# Pivot table for heatmap  
heatmap\_data = results\_df.pivot(index='fisher\_period', columns='shift', values='performance')  
  
\# Plot heatmap  
plt.figure(figsize=(12, 8))  
sns.heatmap(heatmap\_data, annot=False, fmt=".1f", cmap="coolwarm", cbar\_kws={'label': 'Total Return \[%\]'})  
plt.title("Heatmap of Total Return by Shift and Fisher Period")  
plt.xlabel("Shift")  
plt.ylabel("Fisher Period")  
plt.show()

![Heatmap of Total Return by Shift and Fisher Period](https://miro.medium.com/v2/resize:fit:700/1*xB_QZ2ScztHyZUdN16Pybw.png)
Heatmap of Total Return by Shift and Fisher Period

-   Visualizing Total Return by Shift and Fisher Period as 3D scatter plot

ax=plt.figure(figsize=(9,9))  
ax = plt.axes(projection='3d')  
  
  
\# Data for a three-dimensional line  
zline = results\_df\['performance'\]  
xline = results\_df\['shift'\]  
yline = results\_df\['fisher\_period'\]  
#ax.plot3D(xline, yline, zline, 'gray')  
  
\# Data for three-dimensional scattered points  
#zdata = 15 \* np.random.random(100)  
#xdata = np.sin(zdata) + 0.1 \* np.random.randn(100)  
#ydata = np.cos(zdata) + 0.1 \* np.random.randn(100)  
ax.scatter3D(xline, yline, zline, c=zline, cmap='Greens');  
plt.xlabel('Shift')  
plt.ylabel('Fisher Period')  
plt.title('Performance')  
plt.show()

![3D Scatter Plot of Total Return by Shift and Fisher Period](https://miro.medium.com/v2/resize:fit:700/1*xDrpIWutfEI0LADo42umhA.png)
3D Scatter Plot of Total Return by Shift and Fisher Period

-   Finding the best combination of parameters based on max (Total Return)

\# Find the best parameters based on Total Return  
best\_params = results\_df.loc\[results\_df\['performance'\].idxmax()\]  
  
print("Best parameters:")  
print(f"Shift: {best\_params\['shift'\]}")  
print(f"Fisher Period: {best\_params\['fisher\_period'\]}")  
print(f"Total Return: {best\_params\['performance'\]}")  
  
Best parameters:  
Shift: 20.0  
Fisher Period: 20.0  
Total Return: 1197.781537280152

-   Backtesting the Fisher-PVT trading strategy with the aforementioned optimized parameters

\# Calculate Price Volume Trend (PVT)  
\# df = stock input  
df\['PVT'\] = calculate\_pvt(df)  
  
\# Calculate Fisher Transform and Signal line  
df\['Fisher'\], df\['Fisher\_Signal'\] = calculate\_fisher\_transform(df, period=20)  
  
\# Define Entry and Exit signals based on PVT and Fisher Transform  
df\['Entry'\] = (df\['PVT'\] > df\['PVT'\].shift(20)) & (df\['Fisher'\] > df\['Fisher\_Signal'\])  
df\['Exit'\] = (df\['PVT'\] < df\['PVT'\].shift(20)) & (df\['Fisher'\] < df\['Fisher\_Signal'\])  
  
  
\# Backtest using vectorbt  
portfolio = vbt.Portfolio.from\_signals(  
    close=df\['Close'\],  
    entries=df\['Entry'\],  
    exits=df\['Exit'\],  
    init\_cash=10000,  
    fees=0.002  
)  
  
\# Display performance metrics  
print(portfolio.stats())  
  
\# Plot equity curve  
portfolio.plot().show()  
  
  
Start                         2022\-01-03 00:00:00  
End                           2025\-07-25 00:00:00  
Period                                        893  
Start Value                               10000.0  
End Value                           129778.153728  
Total Return \[%\]                      1197.781537  
Benchmark Return \[%\]                   762.574204  
Max Gross Exposure \[%\]                      100.0  
Total Fees Paid                       2385.307944  
Max Drawdown \[%\]                         37.58778  
Max Drawdown Duration                       132.0  
Total Trades                                   18  
Total Closed Trades                            18  
Total Open Trades                               0  
Open Trade PnL                                0.0  
Win Rate \[%\]                            55.555556  
Best Trade \[%\]                         238.084092  
Worst Trade \[%\]                        -16.687185  
Avg Winning Trade \[%\]                   47.690125  
Avg Losing Trade \[%\]                     -6.99769  
Avg Winning Trade Duration                   48.3  
Avg Losing Trade Duration                    9.75  
Profit Factor                             8.60504  
Expectancy                            6654.341874  
dtype: object

![Backtesting the optimized Fisher-PVT trading strategy.](https://miro.medium.com/v2/resize:fit:700/1*N2c9JXo-w17_Fi4SO6cPmQ.png)
Backtesting the optimized Fisher-PVT trading strategy.

## VWAP & Dynamic Volume Profile Oscillator

-   Let’s implement the Dynamic Volume Profile Oscillator (DVPO) to quantify deviations from its volatility scaled volume-weighted mean \[13\]. It blends volume, price, and volatility into a single signal. When price accelerates away from its volume-weighted anchor, the oscillator responds.
-   The end-to-end workflow consists of the following 6 steps \[13\]:
-   Step 1: Preparing the input stock data

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
df = get\_historical\_data(stock\_symbol, start\_date)  
df.rename(columns={'open': 'Open','close': 'Close','high':'High','low': 'Low','volume':'Volume'},inplace=True)  
\# Create standard column names for ease of use  
df\["Price"\] = df\["Close"\]  
df\["Vol"\]   = df\["Volume"\]  
df\["Date"\]  = df.index

-   Step 2: Compute Dynamic Volume Profile (VWAP & Deviation)

def compute\_vwap\_and\_dev(df, lookback=50, profile\_periods=10):  
    """  
    For each bar, every 'profile\_periods' bars we recalc:  
      VWAP = sum(Price \* Volume) / sum(Volume)  
      Dev  = volume-weighted average deviation from VWAP  
    Results are stored in df\["VWAP\_Level"\] and df\["Price\_Deviation"\].  
    """  
    n = len(df)  
    vwap\_vals = np.full(n, np.nan)  
    dev\_vals  = np.full(n, np.nan)  
    for i in range(n):  
        if i < lookback:  
            continue  
        if (i == lookback) or (i % profile\_periods == 0):  
            window\_slice = df.iloc\[i - lookback : i\]  
            vol\_sum = window\_slice\["Vol"\].sum()  
            if vol\_sum > 0:  
                sum\_price\_vol = (window\_slice\["Price"\] \* window\_slice\["Vol"\]).sum()  
                vwap\_current = sum\_price\_vol / vol\_sum  
                abs\_dev = (window\_slice\["Price"\] - vwap\_current).abs()  
                weights = window\_slice\["Vol"\] / vol\_sum  
                dev\_current = (abs\_dev \* weights).sum()  
            else:  
                vwap\_current = df\["Price"\].iloc\[i\]  
                dev\_current  = 0  
            for j in range(i, min(i + profile\_periods, n)):  
                vwap\_vals\[j\] = vwap\_current  
                dev\_vals\[j\]  = dev\_current  
    df\["VWAP\_Level"\]      = vwap\_vals  
    df\["Price\_Deviation"\] = dev\_vals  
  
\# Set parameters for volume profile computation  
LOOKBACK = 50           \# Number of bars used to compute VWAP and deviation. Higher = smoother, slower to adapt.  
PROFILE\_PERIODS = 10    \# How often to update VWAP/deviation. Smaller = more responsive, larger = more stable.  
  
compute\_vwap\_and\_dev(df, LOOKBACK, PROFILE\_PERIODS)

-   Step 3: Compute the Oscillator

def normalize(value, min\_val, max\_val):  
    """Scale 'value' to 0..100 based on the range \[min\_val, max\_val\]."""  
    rng = max\_val - min\_val  
    if rng <= 0:  
        return 50  
    return np.clip(((value - min\_val) / rng) \* 100, 0, 100)  
  
def ema(series, length):  
    """Exponential Moving Average."""  
    return series.ewm(span=length, adjust=False).mean()  
  
def sma(series, length):  
    """Simple Moving Average."""  
    return series.rolling(window=length, min\_periods=1).mean()  
  
def stdev(series, length):  
    """Rolling standard deviation."""  
    return series.rolling(window=length, min\_periods=1).std()  
  
\# Compute raw oscillator value.  
MEAN\_REVERSION = True     \# If True, oscillator reacts to price deviation from VWAP. If False, uses volume normalization.  
SENSITIVITY = 1.0         \# Affects how sharply the oscillator reacts to price deviations. Higher = more sensitive.  
SMOOTHING = 5             \# EMA length for smoothing the oscillator. Lower = more responsive, higher = smoother.  
  
\# Pre-calc volume SMA in case we need it (for non-mean reversion)  
df\["VolSMA"\] = df\["Vol"\].rolling(window=SMOOTHING, min\_periods=1).mean()  
df\["Osc\_Raw"\] = np.nan  
  
for i in range(len(df)):  
    if MEAN\_REVERSION:  
        vwap\_val = df\["VWAP\_Level"\].iloc\[i\]  
        dev\_val  = df\["Price\_Deviation"\].iloc\[i\]  
        if pd.isna(vwap\_val) or dev\_val == 0:  
            df.at\[df.index\[i\], "Osc\_Raw"\] = 50  
        else:  
            price\_term = df\["Price"\].iloc\[i\] - vwap\_val  
            df.at\[df.index\[i\], "Osc\_Raw"\] = 50 + (price\_term / (dev\_val \* SENSITIVITY)) \* 25  
    else:  
        vol\_sma = df\["VolSMA"\].iloc\[i\]  
        if i < LOOKBACK:  
            df.at\[df.index\[i\], "Osc\_Raw"\] = np.nan  
        else:  
            sub = df\["VolSMA"\].iloc\[i - LOOKBACK + 1 : i + 1\]  
            v\_min = sub.min()  
            v\_max = sub.max()  
            df.at\[df.index\[i\], "Osc\_Raw"\] = normalize(vol\_sma, v\_min, v\_max)  
  
\# Smooth the raw oscillator  
df\["Oscillator"\] = ema(df\["Osc\_Raw"\], SMOOTHING)

-   Step 4: Adaptive Midline and Zone Calculation

USE\_ADAPTIVE\_MID = True    \# If True, the midline adjusts over time using a moving average. If False, it's fixed at 50.  
MIDLINE\_PERIOD = 50        \# Controls how slowly or quickly the adaptive midline updates. Higher = smoother.  
ZONE\_WIDTH = 1.5           \# Multiplier for standard deviation bands around the midline. Wider zones mean more tolerance.  
  
if USE\_ADAPTIVE\_MID:  
    df\["Midline"\] = sma(df\["Oscillator"\], MIDLINE\_PERIOD)  
else:  
    df\["Midline"\] = 50  
  
df\["Osc\_Stdev"\] = stdev(df\["Oscillator"\], MIDLINE\_PERIOD) \* ZONE\_WIDTH  
df\["Upper\_Zone"\] = df\["Midline"\] + df\["Osc\_Stdev"\]  
df\["Lower\_Zone"\] = df\["Midline"\] - df\["Osc\_Stdev"\]

-   Step 5: Compute Fast & Slow Signal Lines

df\["Fast\_Signal"\] = ema(df\["Oscillator"\], 5)  
df\["Slow\_Signal"\] = ema(df\["Oscillator"\], 15)  
  
\# Determine bullish or bearish state based on oscillator vs. midline and signals.  
df\["is\_bullish"\] = (df\["Oscillator"\] > df\["Midline"\]) | (df\["Fast\_Signal"\] > df\["Slow\_Signal"\])  
df\["is\_bearish"\] = ~df\["is\_bullish"\]

-   Step 6: Plotting the Oscillator and Price Charts

COLOR\_BARS = True  
BULL\_COLOR = "#00FFBB"  
BEAR\_COLOR = "#FF009D"  
SHOW\_PRICE\_CHART = True  
TICKER='PLTR'  
  
if SHOW\_PRICE\_CHART:  
    fig, axes = plt.subplots(nrows=2, figsize=(16, 8), sharex=True,  
                             gridspec\_kw={"height\_ratios": \[2, 1\]})  
    ax\_price, ax\_osc = axes  
else:  
    fig, ax\_osc = plt.subplots(nrows=1, figsize=(12, 8))  
  
\# ----- Price Chart (Top Panel) -----  
if SHOW\_PRICE\_CHART:  
    ax\_price.set\_title(f"Dynamic Volume Profile Oscillator: {TICKER}", color="black")  
    ax\_price.set\_facecolor("#1F1B1B")  
    ax\_price.grid(True, alpha=0.2, color="black")  
    ax\_price.tick\_params(axis='x', colors='black')  
    ax\_price.tick\_params(axis='y', colors='black')  
  
    if COLOR\_BARS:  
        xvals = df\["Date"\]  
        yvals = df\["Price"\]  
        for i in range(1, len(df)):  
            c = BULL\_COLOR if df\["is\_bullish"\].iloc\[i\] else BEAR\_COLOR  
            ax\_price.plot(\[xvals\[i-1\], xvals\[i\]\],  
                          \[yvals\[i-1\], yvals\[i\]\],  
                          color=c, linewidth=1.5)  
    else:  
        ax\_price.plot(df\["Date"\], df\["Price"\], color="white", linewidth=1.5)  
  
    \# --- Add VWAP overlay ---  
    ax\_price.plot(df\["Date"\], df\["VWAP\_Level"\], color="white", linestyle="--", linewidth=1, label="VWAP")  
  
    ax\_price.set\_ylabel("Price", color="black")  
    ax\_price.legend(loc="best", facecolor="white")  
  
\# ----- Oscillator Chart (Bottom Panel) -----  
ax\_osc.set\_facecolor("#1F1B1B")  
ax\_osc.grid(True, alpha=0.2, color="black")  
ax\_osc.tick\_params(axis='x', colors='black')  
ax\_osc.tick\_params(axis='y', colors='black')  
ax\_osc.set\_ylabel("Oscillator", color="black")  
  
ax\_osc.plot(df\["Date"\], df\["Oscillator"\], color="white", label="Oscillator", linewidth=2)  
ax\_osc.plot(df\["Date"\], df\["Fast\_Signal"\], color="yellow", alpha=0.6, label="Fast Signal")  
ax\_osc.plot(df\["Date"\], df\["Slow\_Signal"\], color="orange", alpha=0.6, label="Slow Signal")  
ax\_osc.plot(df\["Date"\], df\["Midline"\], color="gray", label="Adaptive Midline", linewidth=1)  
  
\# Create gradient fills for the zones by subdividing the space into steps.  
dates = df\["Date"\]  
mid   = df\["Midline"\]  
up    = df\["Upper\_Zone"\]  
dn    = df\["Lower\_Zone"\]  
  
NUM\_STEPS = 10  
up\_diff = up - mid  
dn\_diff = mid - dn  
  
for step in range(NUM\_STEPS):  
    f1 = step / NUM\_STEPS  
    f2 = (step + 1) / NUM\_STEPS  
    y1\_up = mid + up\_diff \* f1  
    y2\_up = mid + up\_diff \* f2  
    alpha\_up = 0.05 + 0.03 \* step  
    ax\_osc.fill\_between(dates, y1\_up, y2\_up, color=BEAR\_COLOR, alpha=alpha\_up)  
  
    y1\_dn = mid - dn\_diff \* f1  
    y2\_dn = mid - dn\_diff \* f2  
    alpha\_dn = 0.05 + 0.03 \* step  
    ax\_osc.fill\_between(dates, y1\_dn, y2\_dn, color=BULL\_COLOR, alpha=alpha\_dn)  
  
ax\_osc.legend(loc="best", facecolor="white")  
  
fig.autofmt\_xdate()  
plt.tight\_layout()  
plt.show()

![PLTR Dynamic Volume Profile Oscillator, Close Price & VWAP in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*b5ACnUHLf1gF4uNonymoyQ.png)
PLTR Dynamic Volume Profile Oscillator, Close Price & VWAP in 2022–2025.

## Prophet Stock Price Prediction

-   Let’s explore the short-term close price forecasting using [FB Prophet](https://github.com/facebook/prophet) \[14–16\]. It works best with time series that have strong [seasonal effects](https://pypi.org/project/prophet/).
-   Preparing the input data for Prophet

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
df = get\_historical\_data(stock\_symbol, start\_date)  
  
data=df.copy()  
  
\# Prepare data for Prophet  
data.reset\_index(inplace=True)  
data.rename(columns={'open': 'Open','close': 'Close','high':'High','low': 'Low','volume':'Volume'},inplace=True)  
data = data\[\['datetime', 'Close'\]\].rename(columns={'datetime': 'ds', 'Close': 'y'})

-   Creating and fitting the Prophet model

\# Create and fit Prophet model  
model = Prophet(daily\_seasonality=True)  
model.fit(data)

-   Making Prophet predictions

\# Create future dataframe and make predictions  
future = model.make\_future\_dataframe(periods=30)  
forecast = model.predict(future)

-   Plotting these predictions and key four components

\# Plot the forecast  
fig1 = model.plot(forecast)  
plt.title("Stock Price Forecast")  
plt.xlabel("Date")  
plt.ylabel("Closing Price")  
plt.show()  
  
\# Plot components (trend, seasonality)  
fig2 = model.plot\_components(forecast)  
plt.show()

![Prophet PLTR Stock Price Forecast](https://miro.medium.com/v2/resize:fit:700/1*wl-uN8ZcMbBIyBEbzU824Q.png)
Prophet PLTR Stock Price Forecast

![Key Four Components of Prophet Forecast](https://miro.medium.com/v2/resize:fit:700/1*FYUT6EmHDTV0c7vtndwPaw.png)
Key Four Components of Prophet Forecast

-   Using [Plotly](https://plotly.com/) to select a custom time window

future\_dates=model.make\_future\_dataframe(periods=30)  
predictions=model.predict(future\_dates)  
  
from prophet.plot import plot\_plotly  
plot\_plotly(model, predictions)

![Prophet PLTR Stock Price Forecast: Plotly Version.](https://miro.medium.com/v2/resize:fit:700/1*Th-gJ4Sf7-OabZKVjN6jbQ.png)
Prophet PLTR Stock Price Forecast: Plotly Version.

-   Preparing the input data for Prophet by adding the 10-day Moving Average (MA) and an ‘Indicator’ column \[16\]

data1=df.copy()  
data1.rename(columns={'open': 'Open','close': 'Close','high':'High','low': 'Low','volume':'Volume'},inplace=True)  
data\_close = data1\[\['Close'\]\]  
data\_close.reset\_index(inplace=True)  
data\_close.columns = \['ds', 'y'\]  
data\_close.loc\[:, 'MA'\] = data\_close\['y'\].rolling(window=10).mean().fillna(0)  
data\_close.loc\[:, 'Indicator'\] = np.where(data\_close\['y'\] > data\_close\['MA'\], 'Buy', 'Sell')

-   The above Indicator signifies whether to “Buy” or “Sell” based on a comparison between the closing price (‘y’) and the moving  
    average (‘MA’). If the closing price is greater than the moving average, it is marked as “Buy”; otherwise, it is marked as “Sell”.
-   Defining the custom scoring function

def score\_func(y\_true, y\_pred):  
    return mean\_absolute\_error(y\_true, y\_pred)

-   Initializing the cross-validation process \[16\] to be repeated 5 times, with each split serving as a separate training and testing set.

for train\_index, test\_index in tscv.split(data\_close):  
    \# Get the training and testing data for this split  
    train\_data = data\_close.iloc\[train\_index\]  
    test\_data = data\_close.iloc\[test\_index\]  
    \# Fit the model on the training data  
    m = Prophet(yearly\_seasonality=True)  
    m.fit(train\_data)  
    \# Make predictions on the test data  
    future = m.make\_future\_dataframe(periods=365 \* 5)  
    forecast = m.predict(future)  
    test\_predictions = forecast.iloc\[-len(test\_data):\]\[\['yhat'\]\]  
    \# Calculate the score for this split  
    score = score\_func(test\_data\[\['y'\]\], test\_predictions)  
    \# Append the score to the scores list  
    scores.append(score)

-   Calculating the mean cross-validation score (CVS)

\# Calculate the mean score  
mean\_score = sum(scores) / len(scores)  
print('Score:' + str(mean\_score))  
  
Score:105.58539537811721

-   Calculating MAE, MSE, and RMSE of the Prophet cross-validation [procedure](https://facebook.github.io/prophet/docs/diagnostics.html)

df\_cv = cross\_validation(m, initial='365 days', period='30 days', horizon='180 days')  
df\_metrics = performance\_metrics(df\_cv)  
  
\# Calculate MAE, MSE, and RMSE  
mae = mean\_absolute\_error(df\_cv\['y'\], df\_cv\['yhat'\])  
mse = mean\_squared\_error(df\_cv\['y'\], df\_cv\['yhat'\])  
rmse = np.sqrt(mse)  
  
print(f'Mean Absolute Error: {mae:.2f}')  
print(f'Mean Squared Error: {mse:.2f}')  
print(f'Root Mean Squared Error: {rmse:.2f}')  
  
Mean Absolute Error: 11.40  
Mean Squared Error: 296.24  
Root Mean Squared Error: 17.21

-   Calculating the Prophet performance metrics with the manually selected [cutoffs](https://facebook.github.io/prophet/docs/diagnostics.html)

cutoffs = pd.to\_datetime(\['2022-09-01', '2024-03-01'\])  
df\_cv2 = cross\_validation(m, cutoffs=cutoffs, initial='750 days', period='30 days', horizon='180 days')  
  
from prophet.diagnostics import performance\_metrics  
df\_p = performance\_metrics(df\_cv2)  
df\_p.head()  
  
 horizon  mse       rmse     mae      mape     mdape    smape    coverage  
0 19 days 30.860388 5.555213 4.252605 0.430910 0.189928 0.603876 0.125000  
1 20 days 40.346234 6.351869 4.931778 0.512114 0.204154 0.689676 0.125000  
2 21 days 51.073255 7.146555 5.732567 0.600755 0.222081 0.774822 0.083333  
3 22 days 61.791382 7.860749 6.360483 0.686608 0.260917 0.852981 0.062500  
4 24 days 63.160335 7.947348 6.561904 0.691861 0.266610 0.856681 0.041667

-   Plotting the selected [cross-validation metrics](https://facebook.github.io/prophet/docs/diagnostics.html)

from prophet.plot import plot\_cross\_validation\_metric  
fig = plot\_cross\_validation\_metric(df\_cv, metric='mape')

![Prophet Cross-Validation Metrics: MAPE.](https://miro.medium.com/v2/resize:fit:700/1*g9gSUQ3lQso6321yqTAOxg.png)
Prophet Cross-Validation Metrics: MAPE.

fig = plot\_cross\_validation\_metric(df\_cv, metric='mae')

![Prophet Cross-Validation Metrics: MAE](https://miro.medium.com/v2/resize:fit:700/1*sx1GxacfPTO99W8jGeNYbA.png)
Prophet Cross-Validation Metrics: MAE

-   Implementing Prophet model tuning with the hyperparameter optimization ([HPO](https://facebook.github.io/prophet/docs/diagnostics.html))

param\_grid = {    
    'changepoint\_prior\_scale': \[0.001, 0.01, 0.1, 0.5\],  
    'seasonality\_prior\_scale': \[0.01, 0.1, 1.0, 10.0\],  
}  
  
\# Generate all combinations of parameters  
all\_params = \[dict(zip(param\_grid.keys(), v)) for v in itertools.product(\*param\_grid.values())\]  
rmses = \[\]  \# Store the RMSEs for each params here  
  
\# Use cross validation to evaluate all parameters  
for params in all\_params:  
    m = Prophet(\*\*params).fit(data\_close)  \# Fit model with given params  
    df\_cv = cross\_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")  
    df\_p = performance\_metrics(df\_cv, rolling\_window=1)  
    rmses.append(df\_p\['rmse'\].values\[0\])  
  
\# Find the best parameters  
tuning\_results = pd.DataFrame(all\_params)  
tuning\_results\['rmse'\] = rmses  
print(tuning\_results)  
  
     changepoint\_prior\_scale  seasonality\_prior\_scale   rmse  
0                     0.001                     0.01   5.560759  
1                     0.001                     0.10   5.977356  
2                     0.001                     1.00  19.163525  
3                     0.001                    10.00  32.285143  
4                     0.010                     0.01   2.484984  
5                     0.010                     0.10   3.003708  
6                     0.010                     1.00   3.033847  
7                     0.010                    10.00   2.796517  
8                     0.100                     0.01   1.968920  
9                     0.100                     0.10   2.680776  
10                    0.100                     1.00   4.702508  
11                    0.100                    10.00   4.367060  
12                    0.500                     0.01   1.392439  
13                    0.500                     0.10   3.677324  
14                    0.500                     1.00   9.025044  
15                    0.500                    10.00   9.462990

-   Finding the best parameters

best\_params = all\_params\[np.argmin(rmses)\]  
print(best\_params)  
  
{'changepoint\_prior\_scale': 0.5, 'seasonality\_prior\_scale': 0.01}

-   Plotting stock price forecasting with the HPO-tuned Prophet model

mbest = Prophet(changepoint\_prior\_scale=0.5, seasonality\_prior\_scale=0.01).fit(data\_close)   
  
future = mbest.make\_future\_dataframe(periods=60)  
forecast = mbest.predict(future)  
  
\# Plot the forecast  
plt.figure(figsize=(15, 8))  
fig1 = m.plot(forecast)  
\# Add labels and a title to the graph  
plt.xlabel('Date')  
plt.ylabel('Price')  
plt.title('Stock Price')  
\# Add gridlines to the graph  
plt.grid(True)

![PLTR Price Forecasting with the HPO-Tuned Prophet Model](https://miro.medium.com/v2/resize:fit:700/1*c5Pshm9F7GmIK4KPV6J0Tg.png)
PLTR Price Forecasting with the HPO-Tuned Prophet Model

## LSTM Stock Price Prediction

-   In this section, the Prophet model will be complemented by Long Short-Term Memory (LSTM) to predict stock prices \[17\].
-   Specifically, we’ll compare the performance of LSTM and Prophet, showcasing their prediction capabilities and error rates.
-   Typically, the [LSTM component](https://www.mdpi.com/1996-1073/18/2/278) captures nonlinear dependencies and long-term temporal patterns, while Prophet models seasonal trends and event-driven fluctuations.
-   The objective is to present a [hybrid forecasting model](https://www.mdpi.com/1996-1073/18/2/278) that integrates LSTM networks and Prophet models, leveraging their complementary strengths through a dynamic weighted ensemble methodology.
-   Preparing the input dataset (vide supra, data1) for LSTM

#Min-max scaler is used for scaling the data so that we can bring all the price values to a common scale.  
scaler = MinMaxScaler()  
scaled\_data = scaler.fit\_transform(data1\[\['Close'\]\])  
  
\# Splitting data into training and testing sets  
train\_size = int(len(scaled\_data) \* 0.8)  
train\_data, test\_data = scaled\_data\[:train\_size\], scaled\_data\[train\_size:\]  
  
\# Shaping data for LSTM  
def create\_dataset(dataset, time\_step=30):  
    X, y = \[\], \[\]  
    for i in range(time\_step, len(dataset)):  
        X.append(dataset\[i-time\_step:i\])  
        y.append(dataset\[i, 0\])  \# Predicting 'Close' price  
    return np.array(X), np.array(y)  
  
time\_step = 30  
X\_train, y\_train = create\_dataset(train\_data, time\_step)  
X\_test, y\_test = create\_dataset(test\_data, time\_step)  
  
\# Reshaping data to be suitable for LSTM  
X\_train = X\_train.reshape(X\_train.shape\[0\], X\_train.shape\[1\], X\_train.shape\[2\])  
X\_test = X\_test.reshape(X\_test.shape\[0\], X\_test.shape\[1\], X\_test.shape\[2\])

-   Creating the Sequential model imported from Keras

\# Creating LSTM model  
model = Sequential()  
model.add(LSTM(100, return\_sequences=True, input\_shape=(time\_step, X\_train.shape\[2\])))  
model.add(Dropout(0.1))  
model.add(LSTM(100, return\_sequences=True))  
model.add(Dropout(0.1))  
model.add(LSTM(50, return\_sequences=False))  
model.add(Dropout(0.1))  
model.add(Dense(50))  
model.add(Dense(25))  
model.add(Dense(1))

-   Compiling and training the above model

\# Compiling the model  
learning\_rate = 0.001  
optimizer = Adam(learning\_rate=learning\_rate)  
model.compile(optimizer=optimizer, loss='mean\_squared\_error')  
  
\# Training the model  
history=model.fit(X\_train, y\_train, batch\_size=16, epochs=50, validation\_split=0.2, verbose=1)  
  
model.summary()

![LSTM Model Summary.](https://miro.medium.com/v2/resize:fit:671/1*2Bjj2gisgztYUn2FSc6uYg.png)
LSTM Model Summary.

-   Plotting the LSTM model loss history curve

plt.plot(history.history\['loss'\])  
plt.plot(history.history\['val\_loss'\])  
plt.title('LSTM loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(\['train', 'val'\], loc='upper right')  
plt.show()

![LSTM model loss history curve](https://miro.medium.com/v2/resize:fit:700/1*VkJ-tYEUlLgg0gbl2J-mAg.png)
LSTM model loss history curve

-   Making test predictions

\# Making predictions  
predictions = model.predict(X\_test)  
predictions = scaler.inverse\_transform(np.hstack((predictions, np.zeros((predictions.shape\[0\], 0)))))\[:, 0\]  
  
\# Comparing with actual test data  
y\_test\_actual = scaler.inverse\_transform(np.hstack((y\_test.reshape(-1, 1), np.zeros((y\_test.shape\[0\], 0)))))\[:, 0\]  
  
plt.figure(figsize=(12, 6))  
plt.plot(y\_test\_actual, label='Actual Price')  
plt.plot(predictions, label='Predicted Price')  
plt.title("LSTM Stock Price Prediction: PLTR Test Data")  
plt.xlabel("Time")  
plt.ylabel("Price")  
plt.legend()  
plt.show()

![LSTM Test Data Predictions (30 Days).](https://miro.medium.com/v2/resize:fit:700/1*PV3P-rpnHTyDjss8HwcwoQ.png)
LSTM Test Data Predictions (30 Days).

-   Implementing the sliding window cross-validation

\# Sliding window validation  
  
from sklearn.metrics import mean\_squared\_error  
import math  
  
\# Parameters of Sliding window validation  
window\_size = int(len(scaled\_data) \* 0.8)  #80% for train  
step\_size = 100   
time\_step = 30   
window\_results = \[\]  
start\_index = 0  
  
while start\_index + window\_size + time\_step <= len(scaled\_data):  
    train\_data = scaled\_data\[start\_index:start\_index + window\_size\]  
    test\_data = scaled\_data\[start\_index + window\_size - time\_step:start\_index + window\_size + time\_step\]  
  
    X\_train, y\_train = create\_dataset(train\_data, time\_step)  
    X\_test, y\_test = create\_dataset(test\_data, time\_step)  
  
    model = Sequential()  
    model.add(LSTM(100, return\_sequences=True, input\_shape=(time\_step, X\_train.shape\[2\])))  
    model.add(Dropout(0.1))  
    model.add(LSTM(100, return\_sequences=True))  
    model.add(Dropout(0.1))  
    model.add(LSTM(50, return\_sequences=False))  
    model.add(Dropout(0.1))  
    model.add(Dense(50))  
    model.add(Dense(25))  
    model.add(Dense(1))  
  
    model.compile(optimizer=Adam(learning\_rate=0.001), loss='mean\_squared\_error')  
    model.fit(X\_train, y\_train, batch\_size=32, epochs=10, verbose=0)  
  
    predictions = model.predict(X\_test)  
    predictions = scaler.inverse\_transform(np.hstack((predictions, np.zeros((predictions.shape\[0\], 2)))))\[:, 0\]  
    y\_test\_actual = scaler.inverse\_transform(np.hstack((y\_test.reshape(-1, 1), np.zeros((y\_test.shape\[0\], 2)))))\[:, 0\]  
  
    rmse = math.sqrt(mean\_squared\_error(y\_test\_actual, predictions))  
    window\_results.append(rmse)  
    start\_index += step\_size  
  
average\_rmse = np.mean(window\_results)  
print(f"Sliding Window Average RMSE: {average\_rmse}")  
  
Sliding Window Average RMSE: 10.711130111573773  
  
\# cf. Prophet RMSE: 17.21

## Backtesting VWAP Crossover Strategy

-   Let’s turn our attention to a detailed analysis of VWAP (Volume Weighted Average Price) [indicator](/vwap-or-moving-average-for-algorithmic-trading-62b16af1cd2d) to build and backtest a trading bot on PLTR real-time data.
-   Preparing the input data for backtesting

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
dff = get\_historical\_data(stock\_symbol, start\_date)  
dff.rename(columns={'open': 'Open','close': 'Close','high':'High','low': 'Low','volume':'Volume'},inplace=True)

-   Backtesting the (window-optimized) VWAP 100–150 crossover strategy

def VWAP(prices, volume, n):  
    """  
    Return VWAP of \`values\`, at  
    each step taking into account \`n\` previous values.  
    """  
  
    values=(prices \* volume)  
    vol=pd.Series(volume)  
      
    return pd.Series(values).rolling(n).sum() / vol.rolling(n).sum()  
  
class VWAPCross(Strategy):  
    \# Define the two MA lags as \*class variables\*  
    \# for later optimization  
    n1 = 100  
    n2 = 150  
      
    def init(self):  
        \# Precompute the two moving averages  
        self.sma1 = self.I(VWAP, self.data.Close, self.data.Volume,self.n1)  
        self.sma2 = self.I(VWAP, self.data.Close, self.data.Volume, self.n2)  
      
    def next(self):  
        \# If sma1 crosses above sma2, close any existing  
        \# short trades, and buy the asset  
        if crossover(self.sma1, self.sma2):  
            self.position.close()  
            self.buy()  
  
        \# Else, if sma1 crosses below sma2, close any existing  
        \# long trades, and sell the asset  
        elif crossover(self.sma2, self.sma1):  
            self.position.close()  
            self.sell()  
  
bt = Backtest(dff, VWAPCross, cash=10000, commission=.002)  
stats = bt.run()  
stats  
  
Start                     2022\-01-03 00:00:00  
End                       2025\-07-25 00:00:00  
Duration                   1299 days 00:00:00  
Exposure Time \[%\]                     0.89586  
Equity Final \[$\]                 185955.48462  
Equity Peak \[$\]                  185955.48462  
Commissions \[$\]                      82.83102  
Return \[%\]                         1759.55485  
Buy & Hold Return \[%\]              1517.10794  
Return (Ann.) \[%\]                   128.15011  
Volatility (Ann.) \[%\]               138.03261  
CAGR \[%\]                             76.30291  
Sharpe Ratio                           0.9284  
Sortino Ratio                         4.17917  
Calmar Ratio                          3.15553  
Alpha \[%\]                           840.72764  
Beta                                  0.60564  
Max. Drawdown \[%\]                   -40.61127  
Avg. Drawdown \[%\]                    -6.94934  
Max. Drawdown Duration      120 days 00:00:00  
Avg. Drawdown Duration       17 days 00:00:00  
\# Trades                                    2  
Win Rate \[%\]                             50.0  
Best Trade \[%\]                          4.375  
Worst Trade \[%\]                      -2.75449  
Avg. Trade \[%\]                        0.74721  
Max. Trade Duration           8 days 00:00:00  
Avg. Trade Duration           5 days 00:00:00  
Profit Factor                         1.58832  
Expectancy \[%\]                        0.81025  
SQN                                   0.20882  
Kelly Criterion                       0.17275  
\_strategy                           VWAPCross  
\_equity\_curve                             ...  
\_trades                      Size  EntryBa...  
dtype: object  
  
bt.plot()

![Backtesting VWAP 100–150 Crossover Strategy](https://miro.medium.com/v2/resize:fit:700/1*cVCBvAcQw_z-AB9llkivBg.png)
Backtesting VWAP 100–150 Crossover Strategy

## Backtesting RSI Trading Strategy

-   Let’s build a PLTR trading strategy based on the Relative Strength Index (RSI) — a momentum oscillator that is used by traders to identify whether the market is in the state of overbought or oversold \[1, 18\].
-   Calculating the RSI-14 indicator

stock\_symbol='PLTR'  
start\_date='2022-01-01'  
df = get\_historical\_data(stock\_symbol, start\_date)  
  
def get\_rsi(close, lookback):  
    ret = close.diff()  
    up = \[\]  
    down = \[\]  
    for i in range(len(ret)):  
        if ret.iloc\[i\] < 0:  
            up.append(0)  
            down.append(ret.iloc\[i\])  
        else:  
            up.append(ret.iloc\[i\])  
            down.append(0)  
    up\_series = pd.Series(up)  
    down\_series = pd.Series(down).abs()  
    up\_ewm = up\_series.ewm(com = lookback - 1, adjust = False).mean()  
    down\_ewm = down\_series.ewm(com = lookback - 1, adjust = False).mean()  
    rs = up\_ewm/down\_ewm  
    rsi = 100 - (100 / (1 + rs))  
    rsi\_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set\_index(close.index)  
    rsi\_df = rsi\_df.dropna()  
    return rsi\_df\[3:\]  
  
df\['rsi\_14'\] = get\_rsi(df\['close'\], 14)  
df = df.dropna()

-   Implementing the RSI trading strategy and plotting the sell/buy signals

aln=50  
alk=75  
def implement\_rsi\_strategy(prices, rsi):    
    global aln,alk  
    buy\_price = \[\]  
    sell\_price = \[\]  
    rsi\_signal = \[\]  
    signal = 0  
  
    for i in range(len(rsi)):  
        if rsi.iloc\[i-1\] > aln and rsi.iloc\[i\] < aln:  
            if signal != 1:  
                buy\_price.append(prices.iloc\[i\])  
                sell\_price.append(np.nan)  
                signal = 1  
                rsi\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                rsi\_signal.append(0)  
        elif rsi.iloc\[i-1\] < alk and rsi.iloc\[i\] > alk:  
            if signal != -1:  
                buy\_price.append(np.nan)  
                sell\_price.append(prices.iloc\[i\])  
                signal = -1  
                rsi\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                rsi\_signal.append(0)  
        else:  
            buy\_price.append(np.nan)  
            sell\_price.append(np.nan)  
            rsi\_signal.append(0)  
              
    return buy\_price, sell\_price, rsi\_signal  
  
buy\_price, sell\_price, rsi\_signal = implement\_rsi\_strategy(df\['close'\], df\['rsi\_14'\])  
  
ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)  
ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)  
ax1.plot(df\['close'\], linewidth = 2.5, color = 'skyblue', label = 'PLTR')  
ax1.plot(df.index, buy\_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')  
ax1.plot(df.index, sell\_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')  
ax1.set\_title('PLTR RSI TRADE SIGNALS')  
ax1.set\_ylabel('PRICE USD')  
ax1.legend()  
ax2.plot(df\['rsi\_14'\], color = 'orange', linewidth = 2.5)  
ax2.axhline(aln, linestyle = '--', linewidth = 1.5, color = 'green')  
ax2.axhline(alk, linestyle = '--', linewidth = 1.5, color = 'red')  
ax2.set\_xlabel('DATE')  
  
ax2.set\_ylabel('RSI')  
plt.show()

![PLTR RSI Trade Signals](https://miro.medium.com/v2/resize:fit:700/1*rMqJ8JVsBwuEDBiJG6E_og.png)
PLTR RSI Trade Signals

-   Our return optimization results suggest that RSI readings below aln=50 (instead of 30) signal buy opportunities, indicating the asset is undervalued. Conversely, RSI readings above alk=75 (instead of 70) signal sell opportunities, suggesting the asset is overvalued. A value of 62.5 signifies a balance between bullish and bearish positions or a neutral stance.
-   Defining the position and calculating expected returns

position = \[\]  
for i in range(len(rsi\_signal)):  
    if rsi\_signal\[i\] > 1:  
        position.append(0)  
    else:  
        position.append(1)  
          
for i in range(len(df\['close'\])):  
    if rsi\_signal\[i\] == 1:  
        position\[i\] = 1  
    elif rsi\_signal\[i\] == -1:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
          
rsi = df\['rsi\_14'\]  
close\_price = df\['close'\]  
rsi\_signal = pd.DataFrame(rsi\_signal).rename(columns = {0:'rsi\_signal'}).set\_index(df.index)  
position = pd.DataFrame(position).rename(columns = {0:'rsi\_position'}).set\_index(df.index)  
  
frames = \[close\_price, rsi, rsi\_signal, position\]  
strategy = pd.concat(frames, join = 'inner', axis = 1)  
  
ret = pd.DataFrame(np.diff(df\['close'\])).rename(columns = {0:'returns'})  
rsi\_strategy\_ret = \[\]  
  
for i in range(len(ret)):  
    returns = ret\['returns'\].iloc\[i\]\*strategy\['rsi\_position'\].iloc\[i\]  
    rsi\_strategy\_ret.append(returns)  
      
rsi\_strategy\_ret\_df = pd.DataFrame(rsi\_strategy\_ret).rename(columns = {0:'rsi\_returns'})  
investment\_value = 10000  
rsi\_investment\_ret = \[\]  
  
for i in range(len(rsi\_strategy\_ret\_df\['rsi\_returns'\])):  
    number\_of\_stocks = floor(investment\_value/df\['close'\].iloc\[i\])  
    returns = number\_of\_stocks\*rsi\_strategy\_ret\_df\['rsi\_returns'\].iloc\[i\]  
    rsi\_investment\_ret.append(returns)  
  
rsi\_investment\_ret\_df = pd.DataFrame(rsi\_investment\_ret).rename(columns = {0:'investment\_returns'})  
total\_investment\_ret = round(sum(rsi\_investment\_ret\_df\['investment\_returns'\]), 2)  
profit\_percentage = floor((total\_investment\_ret/investment\_value)\*100)  
print(cl('Profit gained from the RSI strategy by investing $10k in PLTR : {}'.format(total\_investment\_ret), attrs = \['bold'\]))  
print(cl('Profit percentage of the RSI strategy : {}%'.format(profit\_percentage), attrs = \['bold'\]))  
  
Profit gained from the RSI strategy by investing $10k in PLTR : 22915.81  
Profit percentage of the RSI strategy : 229%

-   Comparing returns of the RSI trading strategy vs B&H

position=strategy\['rsi\_position'\]  
  
rets = df.close.pct\_change()\[1:\]  
strat\_rets = position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend()  
plt.show()

![PLTR RSI Trading Strategy vs B&H Daily and Cumulative Returns in 2022–2025.](https://miro.medium.com/v2/resize:fit:700/1*N7-zdgT8WxMLjgxQZByiDQ.png)
PLTR RSI Trading Strategy vs B&H Daily and Cumulative Returns in 2022–2025.

## Optimizing MACD Trading Strategy

-   In this section, we shift our focus on mastering the MACD (Moving Average Convergence Divergence) trading strategy \[19\] in the context of [vectorbt backtesting](https://github.com/polakowo/vectorbt/blob/master/examples/MACDVolume.ipynb) by optimizing MACD parameters to maximize the PLTR ROI for the period 2022–2025.
-   _Option 1_: Optimizing short windows suitable for high-frequency trading

df = get\_historical\_data('PLTR', '2022-01-01')  
price=df\['close'\]  
  
\# Define hyper-parameter space  
  
fast\_windows, slow\_windows, signal\_windows = vbt.utils.params.create\_param\_combs(  
    (product, (combinations, np.arange(2, 51, 1), 2), np.arange(2, 21, 1)))  
  
\# Run MACD indicator  
macd\_ind = vbt.MACD.run(  
    price,  
    fast\_window=fast\_windows,  
    slow\_window=slow\_windows,  
    signal\_window=signal\_windows  
)  
  
\# Long when MACD is above zero AND signal  
entries = macd\_ind.macd\_above(0) & macd\_ind.macd\_above(macd\_ind.signal)  
  
\# Short when MACD is below zero OR signal  
exits = macd\_ind.macd\_below(0) | macd\_ind.macd\_below(macd\_ind.signal)  
  
\# Build portfolio  
pf = vbt.Portfolio.from\_signals(  
    price.vbt.tile(len(fast\_windows)), entries, exits, fees=0.001, freq='1D')  
  
\# Draw all window combinations as a 3D volume  
fig = pf.total\_return().vbt.volume(  
    x\_level='macd\_fast\_window',  
    y\_level='macd\_slow\_window',  
    z\_level='macd\_signal\_window',  
      
    trace\_kwargs=dict(  
        colorbar=dict(  
            title='Total return',   
            tickformat='%'  
        )  
    )  
)  
fig.show()

![Total Return vs MACD short fast/slow and signal windows (Top-down perspective).](https://miro.medium.com/v2/resize:fit:700/1*4hQItSf7WkShbCp41561-Q.png)
Total Return vs MACD short fast/slow and signal windows (Top-down perspective).

![Same as above, but with the different 3D angle view.](https://miro.medium.com/v2/resize:fit:700/1*x5niRG4UMfDfMmVWGBEuow.png)
Same as above, but with the different 3D angle view.

-   Examining the backtesting summary

pf.stats()  
  
tart                                 2022\-01-03 00:00:00  
End                                   2025\-07-25 00:00:00  
Period                                  893 days 00:00:00  
Start Value                                         100.0  
End Value                                       184.31673  
Total Return \[%\]                                 84.31673  
Benchmark Return \[%\]                           756.988667  
Max Gross Exposure \[%\]                              100.0  
Total Fees Paid                                   7.22814  
Max Drawdown \[%\]                                46.720148  
Max Drawdown Duration         463 days 23:22:10.182599352  
Total Trades                                    35.055004  
Total Closed Trades                             34.306212  
Total Open Trades                                0.748792  
Open Trade PnL                                   5.725055  
Win Rate \[%\]                                    48.929664  
Best Trade \[%\]                                  58.413659  
Worst Trade \[%\]                                -22.062243  
Avg Winning Trade \[%\]                           15.329101  
Avg Losing Trade \[%\]                            -8.314508  
Avg Winning Trade Duration     15 days 11:20:16.837806523  
Avg Losing Trade Duration       8 days 17:37:41.389841864  
Profit Factor                                    1.600538  
Expectancy                                       2.905098  
Sharpe Ratio                                      0.67695  
Calmar Ratio                                     0.632216  
Omega Ratio                                      1.184638  
Sortino Ratio                                    1.094529

-   It is clear that the high-frequency MACD trading does not outperform B&H. Also, a Sharpe ratio of 0.67 (less than 1) suggests that the strategy is not providing sufficient return for the level of risk taken.
-   _Option 2_: Optimizing long windows appropriate for longer-term investing

pf = vbt.Portfolio.from\_holding(price, init\_cash=10000)  
pf.total\_profit()  
  
75698.86670264436  
  
fast\_ma = vbt.MA.run(price, 100)  
slow\_ma = vbt.MA.run(price, 150)  
entries = fast\_ma.ma\_crossed\_above(slow\_ma)  
exits = fast\_ma.ma\_crossed\_below(slow\_ma)  
  
pf = vbt.Portfolio.from\_signals(price, entries, exits, init\_cash=10000)  
pf.total\_profit()  
  
150728.74493927124  
  
#Optimizing MACD parameters  
  
windows = np.arange(50, 150)  
fast\_ma, slow\_ma = vbt.MA.run\_combs(price, window=windows, r=2, short\_names=\['fast', 'slow'\])  
entries = fast\_ma.ma\_crossed\_above(slow\_ma)  
exits = fast\_ma.ma\_crossed\_below(slow\_ma)  
  
pf\_kwargs = dict(size=np.inf, fees=0.002, freq='1D')  
pf = vbt.Portfolio.from\_signals(price, entries, exits, \*\*pf\_kwargs)  
  
fig = pf.total\_return().vbt.heatmap(  
    x\_level='fast\_window', y\_level='slow\_window',   
    trace\_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))  
fig.show()

![Total Return vs MACD long fast/slow windows (Heatmap).](https://miro.medium.com/v2/resize:fit:676/1*xSXWnx3RH83IYumlHXu_bw.png)
Total Return vs MACD long fast/slow windows (Heatmap).

-   Checking the backtesting summary

pf.stats()  
  
Start                                 2022\-01-03 00:00:00  
End                                   2025\-07-25 00:00:00  
Period                                  893 days 00:00:00  
Start Value                                         100.0  
End Value                                     1520.161179  
Total Return \[%\]                              1420.161179  
Benchmark Return \[%\]                           756.988667  
Max Gross Exposure \[%\]                              100.0  
Total Fees Paid                                  3.269366  
Max Drawdown \[%\]                                41.196388  
Max Drawdown Duration         144 days 09:10:06.545454546  
Total Trades                                     4.834141  
Total Closed Trades                              3.842424  
Total Open Trades                                0.991717  
Open Trade PnL                                1146.187596  
Win Rate \[%\]                                    69.790909  
Best Trade \[%\]                                 174.333215  
Worst Trade \[%\]                                  -2.07772  
Avg Winning Trade \[%\]                           79.528573  
Avg Losing Trade \[%\]                            -8.401962  
Avg Winning Trade Duration    119 days 11:43:02.798339818  
Avg Losing Trade Duration      16 days 09:52:05.419310913  
Profit Factor                                         inf  
Expectancy                                      76.235918  
Sharpe Ratio                                     1.952928  
Calmar Ratio                                     4.877447  
Omega Ratio                                      1.479731  
Sortino Ratio                                    3.512317

-   Observe that the high-frequency MACD trading significantly outperforms B&H. Also, a Sharpe ratio of ~1.95 indicates a very good risk-adjusted return for an investment. This means the investment has generated a high level of returns relative to the amount of risk taken. Generally, [Sharpe ratios above 1.0](https://www.investopedia.com/ask/answers/010815/what-good-sharpe-ratio.asp) are considered acceptable, with 2.0 and above signifying very good performance.

## Backtesting TSI Trading Strategy

-   Our next goal is to backtest a PLTR crossover trading strategy based on the [True Strength Index (TSI)](https://www.quantifiedstrategies.com/true-strength-index/) — a momentum indicator that is based on a double smoothing of price changes. It is an oscillator, swinging between limits of -100 and +100, with 0 as the centerline. The indicator consists of 2 lines — the TSI line and an exponential moving average of the TSI known as the signal line.
-   The actual TSI backtesting workflow consists of the following three steps \[1, 20\]:

1.  Data preparation, TSI calculation and visualization
2.  Creating the signal line crossover trading strategy
3.  Defining our position, backtesting and SPY benchmark comparison.

-   Implementing the TSI trading strategy and plotting the trading signals

def implement\_tsi\_strategy(prices, tsi, signal\_line):  
    buy\_price = \[\]  
    sell\_price = \[\]  
    tsi\_signal = \[\]  
    signal = 0  
      
    for i in range(len(prices)):  
        if tsi\[i-1\] < signal\_line\[i-1\] and tsi\[i\] > signal\_line\[i\]:  
            if signal != 1:  
                buy\_price.append(prices\[i\])  
                sell\_price.append(np.nan)  
                signal = 1  
                tsi\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                tsi\_signal.append(0)  
        elif tsi\[i-1\] > signal\_line\[i-1\] and tsi\[i\] < signal\_line\[i\]:  
            if signal != -1:  
                buy\_price.append(np.nan)  
                sell\_price.append(prices\[i\])  
                signal = -1  
                tsi\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                tsi\_signal.append(0)  
        else:  
            buy\_price.append(np.nan)  
            sell\_price.append(np.nan)  
            tsi\_signal.append(0)  
              
    return buy\_price, sell\_price, tsi\_signal  
  
buy\_price, sell\_price, tsi\_signal = implement\_tsi\_strategy(aapl\['close'\], aapl\['tsi'\], aapl\['signal\_line'\])  
  
\# TRUE STRENGTH INDEX TRADING SIGNALS PLOT  
  
ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)  
ax2 = plt.subplot2grid((11,1), (7,0), rowspan = 5, colspan = 1)  
ax1.plot(aapl\['close'\], linewidth = 2)  
ax1.plot(aapl.index, buy\_price, marker = '^', markersize = 12, color = 'green', linewidth = 0, label = 'BUY SIGNAL')  
ax1.plot(aapl.index, sell\_price, marker = 'v', markersize = 12, color = 'r', linewidth = 0, label = 'SELL SIGNAL')  
ax1.set\_ylabel('PRICE USD')  
ax1.legend()  
ax1.set\_title('PLTR TSI TRADING SIGNALS')  
ax2.plot(aapl\['tsi'\], linewidth = 2, color = 'orange', label = 'TSI LINE')  
ax2.plot(aapl\['signal\_line'\], linewidth = 2, color = '#FF006E', label = 'SIGNAL LINE')  
ax2.set\_title('PLTR TSI 25,13,12')  
ax2.set\_xlabel('DATE')  
ax2.set\_ylabel('TSI')  
ax2.legend()  
plt.show()

![PLTR TSI trading signals vs Close Price & Indicator.](https://miro.medium.com/v2/resize:fit:700/1*J33jxmer4lKdhqgGEC6ygw.png)
PLTR TSI trading signals vs Close Price & Indicator.

-   Defining our position, backtesting and comparison against SPY expected returns

\# STOCK POSITION  
  
position = \[\]  
for i in range(len(tsi\_signal)):  
    if tsi\_signal\[i\] > 1:  
        position.append(0)  
    else:  
        position.append(1)  
          
for i in range(len(aapl\['close'\])):  
    if tsi\_signal\[i\] == 1:  
        position\[i\] = 1  
    elif tsi\_signal\[i\] == -1:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
          
close\_price = aapl\['close'\]  
tsi = aapl\['tsi'\]  
signal\_line = aapl\['signal\_line'\]  
tsi\_signal = pd.DataFrame(tsi\_signal).rename(columns = {0:'tsi\_signal'}).set\_index(aapl.index)  
position = pd.DataFrame(position).rename(columns = {0:'tsi\_position'}).set\_index(aapl.index)  
  
frames = \[close\_price, tsi, signal\_line, tsi\_signal, position\]  
strategy = pd.concat(frames, join = 'inner', axis = 1)  
  
\# BACKTESTING  
  
aapl\_ret = pd.DataFrame(np.diff(aapl\['close'\])).rename(columns = {0:'returns'})  
tsi\_strategy\_ret = \[\]  
  
for i in range(len(aapl\_ret)):  
    returns = aapl\_ret\['returns'\]\[i\]\*strategy\['tsi\_position'\]\[i\]  
    tsi\_strategy\_ret.append(returns)  
      
tsi\_strategy\_ret\_df = pd.DataFrame(tsi\_strategy\_ret).rename(columns = {0:'tsi\_returns'})  
investment\_value = 10000  
tsi\_investment\_ret = \[\]  
  
for i in range(len(tsi\_strategy\_ret\_df\['tsi\_returns'\])):  
    number\_of\_stocks = floor(investment\_value/aapl\['close'\]\[i\])  
    returns = number\_of\_stocks\*tsi\_strategy\_ret\_df\['tsi\_returns'\]\[i\]  
    tsi\_investment\_ret.append(returns)  
  
tsi\_investment\_ret\_df = pd.DataFrame(tsi\_investment\_ret).rename(columns = {0:'investment\_returns'})  
total\_investment\_ret = round(sum(tsi\_investment\_ret\_df\['investment\_returns'\]), 2)  
profit\_percentage = floor((total\_investment\_ret/investment\_value)\*100)  
print(cl('Profit gained from the TSI strategy by investing $10k in PLTR : {}'.format(total\_investment\_ret), attrs = \['bold'\]))  
print(cl('Profit percentage of the TSI strategy : {}%'.format(profit\_percentage), attrs = \['bold'\]))  
  
Profit gained from the TSI strategy by investing $10k in PLTR : 11807.63  
Profit percentage of the TSI strategy : 118%  
  
\# SPY ETF COMPARISON  
  
def get\_benchmark(start\_date, investment\_value):  
    spy = get\_historical\_data('SPY', start\_date)\['close'\]  
    benchmark = pd.DataFrame(np.diff(spy)).rename(columns = {0:'benchmark\_returns'})  
      
    investment\_value = investment\_value  
    benchmark\_investment\_ret = \[\]  
      
    for i in range(len(benchmark\['benchmark\_returns'\])):  
        number\_of\_stocks = floor(investment\_value/spy\[i\])  
        returns = number\_of\_stocks\*benchmark\['benchmark\_returns'\]\[i\]  
        benchmark\_investment\_ret.append(returns)  
  
    benchmark\_investment\_ret\_df = pd.DataFrame(benchmark\_investment\_ret).rename(columns = {0:'investment\_returns'})  
    return benchmark\_investment\_ret\_df  
  
benchmark = get\_benchmark('2022-01-01', 10000)  
investment\_value = 10000  
total\_benchmark\_investment\_ret = round(sum(benchmark\['investment\_returns'\]), 2)  
benchmark\_profit\_percentage = floor((total\_benchmark\_investment\_ret/investment\_value)\*100)  
print(cl('Benchmark profit by investing $10k : {}'.format(total\_benchmark\_investment\_ret), attrs = \['bold'\]))  
print(cl('Benchmark Profit percentage : {}%'.format(benchmark\_profit\_percentage), attrs = \['bold'\]))  
print(cl('TSI Strategy profit is {}% higher than the Benchmark Profit'.format(profit\_percentage - benchmark\_profit\_percentage), attrs = \['bold'\]))  
  
Benchmark profit by investing $10k : 3443.75  
Benchmark Profit percentage : 34%  
TSI Strategy profit is 84% higher than the Benchmark Profit

-   Comparing the TSI strategy daily/cumulative returns vs B&H

pos=position\['tsi\_position'\]  
  
rets = close\_price.pct\_change()\[1:\]  
strat\_rets = pos\[1:\]\*rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_rets.plot(color = 'r', linewidth = 1,label='Strategy')  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='B&H')  
strat\_cum.plot(color = 'r', linewidth = 2,label='Strategy')  
plt.legend(loc="upper left")  
plt.show()

![TSI strategy daily/cumulative returns vs B&H](https://miro.medium.com/v2/resize:fit:700/1*akp8F-k0DL2-qwcgpWPnBg.png)
TSI strategy daily/cumulative returns vs B&H

## Backtesting ADX Trading Strategy

-   Let’s look at the [Average Directional Index (ADX)](https://capital.com/en-eu/learn/technical-analysis/average-directional-index) — a technical analysis tool that measures the strength of trends. To quantify a trend’s strength, the calculation of the ADX is based on the [moving average](https://capital.com/en-eu/learn/technical-analysis/moving-average) (MA) of a price range expansion over a certain lookback timeframe. Typically, this is a 14-day period.
-   As with the aforesaid MACD strategy, the ADX backtesting workflow consists of the following three steps \[1, 21\]:

1.  Data preparation, ADX calculation and visualization
2.  Creating the ADX trading strategy
3.  Defining our position, backtesting and SPY comparison.

-   Reading the stock data and defining the ADX threshold parameter alp

aapl = get\_historical\_data('PLTR', '2022-01-01')  
  
alp=25 #This is the optimal value that maximizes ROI during optimization, adjust it further if needed

-   Calculating the ADX indicator

def get\_adx(high, low, close, lookback):  
    plus\_dm = high.diff()  
    minus\_dm = low.diff()  
    plus\_dm\[plus\_dm < 0\] = 0  
    minus\_dm\[minus\_dm > 0\] = 0  
      
    tr1 = pd.DataFrame(high - low)  
    tr2 = pd.DataFrame(abs(high - close.shift(1)))  
    tr3 = pd.DataFrame(abs(low - close.shift(1)))  
    frames = \[tr1, tr2, tr3\]  
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)  
    atr = tr.rolling(lookback).mean()  
      
    plus\_di = 100 \* (plus\_dm.ewm(alpha = 1/lookback).mean() / atr)  
    minus\_di = abs(100 \* (minus\_dm.ewm(alpha = 1/lookback).mean() / atr))  
    dx = (abs(plus\_di - minus\_di) / abs(plus\_di + minus\_di)) \* 100  
    adx = ((dx.shift(1) \* (lookback - 1)) + dx) / lookback  
    adx\_smooth = adx.ewm(alpha = 1/lookback).mean()  
    return plus\_di, minus\_di, adx\_smooth  
  
aapl\['plus\_di'\] = pd.DataFrame(get\_adx(aapl\['high'\], aapl\['low'\], aapl\['close'\], 14)\[0\]).rename(columns = {0:'plus\_di'})  
aapl\['minus\_di'\] = pd.DataFrame(get\_adx(aapl\['high'\], aapl\['low'\], aapl\['close'\], 14)\[1\]).rename(columns = {0:'minus\_di'})  
aapl\['adx'\] = pd.DataFrame(get\_adx(aapl\['high'\], aapl\['low'\], aapl\['close'\], 14)\[2\]).rename(columns = {0:'adx'})  
aapl = aapl.dropna()

-   Implementing the ADX trading strategy

def implement\_adx\_strategy(prices, pdi, ndi, adx):  
    global alp  
    buy\_price = \[\]  
    sell\_price = \[\]  
    adx\_signal = \[\]  
    signal = 0  
      
    for i in range(len(prices)):  
        if adx.iloc\[i-1\] < alp and adx.iloc\[i\] > alp and pdi.iloc\[i\] > ndi.iloc\[i\]:  
            if signal != 1:  
                buy\_price.append(prices.iloc\[i\])  
                sell\_price.append(np.nan)  
                signal = 1  
                adx\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                adx\_signal.append(0)  
        elif adx.iloc\[i-1\] < alp and adx.iloc\[i\] > alp and ndi.iloc\[i\] > pdi.iloc\[i\]:  
            if signal != -1:  
                buy\_price.append(np.nan)  
                sell\_price.append(prices.iloc\[i\])  
                signal = -1  
                adx\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                adx\_signal.append(0)  
        else:  
            buy\_price.append(np.nan)  
            sell\_price.append(np.nan)  
            adx\_signal.append(0)  
              
    return buy\_price, sell\_price, adx\_signal  
  
buy\_price, sell\_price, adx\_signal = implement\_adx\_strategy(aapl\['close'\], aapl\['plus\_di'\], aapl\['minus\_di'\], aapl\['adx'\])

-   Visualizing the ADX trading signals

ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)  
ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)  
ax1.plot(aapl\['close'\], linewidth = 3, color = '#ff9800', alpha = 0.6)  
ax1.set\_title('PLTR CLOSING PRICE')  
ax1.plot(aapl.index, buy\_price, marker = '^', color = '#26a69a', markersize = 14, linewidth = 0, label = 'BUY SIGNAL')  
ax1.plot(aapl.index, sell\_price, marker = 'v', color = '#f44336', markersize = 14, linewidth = 0, label = 'SELL SIGNAL')  
ax1.set\_ylabel('PRICE USD')  
ax2.plot(aapl\['plus\_di'\], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)  
ax2.plot(aapl\['minus\_di'\], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)  
ax2.plot(aapl\['adx'\], color = '#2196f3', label = 'ADX 14', linewidth = 3)  
ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')  
ax2.legend()  
ax2.set\_title('PLTR ADX 14')  
ax2.set\_xlabel('DATE')  
ax2.set\_ylabel('ADX & DI')  
plt.show()

![PLTR ADX trading signals vs Close Price & Indicator.](https://miro.medium.com/v2/resize:fit:700/1*cCY0NKhUyK2IOgrqX7dU3A.png)
PLTR ADX trading signals vs Close Price & Indicator.

-   Defining our position

position = \[\]  
for i in range(len(adx\_signal)):  
    if adx\_signal\[i\] > 1:  
        position.append(0)  
    else:  
        position.append(1)  
          
for i in range(len(aapl\['close'\])):  
    if adx\_signal\[i\] == 1:  
        position\[i\] = 1  
    elif adx\_signal\[i\] == -1:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
          
close\_price = aapl\['close'\]  
plus\_di = aapl\['plus\_di'\]  
minus\_di = aapl\['minus\_di'\]  
adx = aapl\['adx'\]  
adx\_signal = pd.DataFrame(adx\_signal).rename(columns = {0:'adx\_signal'}).set\_index(aapl.index)  
position = pd.DataFrame(position).rename(columns = {0:'adx\_position'}).set\_index(aapl.index)  
  
frames = \[close\_price, plus\_di, minus\_di, adx, adx\_signal, position\]  
strategy = pd.concat(frames, join = 'inner', axis = 1)  
  
strategy  
  
           close    plus\_di   minus\_di adx adx\_signal adx\_position  
datetime        
2023\-03-27 8.040000 20.691480 21.614301 5.539491 0 1  
2023\-03-28 8.000000 19.742819 20.623331 3.798213 0 1  
2023\-03-29 8.220000 22.505232 19.693530 3.333553 0 1  
2023\-03-30 8.150000 24.179711 18.038531 4.417380 0 1  
2023\-03-31 8.450000 27.588294 18.323121 6.846790 0 1  
... ... ... ... ... ... ...  
2025\-07-21 151.789990 33.640851 16.213974 19.973148 0 1  
2025\-07-22 149.070007 31.752209 24.886207 20.926804 0 1  
2025\-07-23 154.630000 32.723725 22.002857 20.336025 0 1  
2025\-07-24 154.860000 31.502090 20.563794 20.289967 0 1  
2025\-07-25 158.800000 36.968756 19.504477 20.391889 0 1  
585 rows × 6 columns

-   ADX backtesting

aapl\_ret = pd.DataFrame(np.diff(aapl\['close'\])).rename(columns = {0:'returns'})  
adx\_strategy\_ret = \[\]  
  
for i in range(len(aapl\_ret)):  
    returns = aapl\_ret\['returns'\]\[i\]\*strategy\['adx\_position'\]\[i\]  
    adx\_strategy\_ret.append(returns)  
      
adx\_strategy\_ret\_df = pd.DataFrame(adx\_strategy\_ret).rename(columns = {0:'adx\_returns'})  
investment\_value = 10000  
number\_of\_stocks = floor(investment\_value/aapl\['close'\]\[-1\])  
adx\_investment\_ret = \[\]  
  
for i in range(len(adx\_strategy\_ret\_df\['adx\_returns'\])):  
    returns = number\_of\_stocks\*adx\_strategy\_ret\_df\['adx\_returns'\]\[i\]  
    adx\_investment\_ret.append(returns)  
  
adx\_investment\_ret\_df = pd.DataFrame(adx\_investment\_ret).rename(columns = {0:'investment\_returns'})  
total\_investment\_ret = round(sum(adx\_investment\_ret\_df\['investment\_returns'\]), 2)  
profit\_percentage = floor((total\_investment\_ret/investment\_value)\*100)  
print(cl('Profit gained from the ADX strategy by investing $10k in PLTR : {}'.format(total\_investment\_ret), attrs = \['bold'\]))  
print(cl('Profit percentage of the ADX strategy : {}%'.format(profit\_percentage), attrs = \['bold'\]))  
  
Profit gained from the ADX strategy by investing $10k in PLTR : 8975.74  
Profit percentage of the ADX strategy : 89%

-   Comparing ADX vs SPY returns

def get\_benchmark(start\_date, investment\_value):  
    spy = get\_historical\_data('SPY', start\_date)\['close'\]  
    benchmark = pd.DataFrame(np.diff(spy)).rename(columns = {0:'benchmark\_returns'})  
      
    investment\_value = investment\_value  
    number\_of\_stocks = floor(investment\_value/spy\[-1\])  
    benchmark\_investment\_ret = \[\]  
      
    for i in range(len(benchmark\['benchmark\_returns'\])):  
        returns = number\_of\_stocks\*benchmark\['benchmark\_returns'\]\[i\]  
        benchmark\_investment\_ret.append(returns)  
  
    benchmark\_investment\_ret\_df = pd.DataFrame(benchmark\_investment\_ret).rename(columns = {0:'investment\_returns'})  
    return benchmark\_investment\_ret\_df  
  
benchmark = get\_benchmark('2022-01-01', 10000)  
  
investment\_value = 10000  
total\_benchmark\_investment\_ret = round(sum(benchmark\['investment\_returns'\]), 2)  
benchmark\_profit\_percentage = floor((total\_benchmark\_investment\_ret/investment\_value)\*100)  
print(cl('Benchmark profit by investing $10k : {}'.format(total\_benchmark\_investment\_ret), attrs = \['bold'\]))  
print(cl('Benchmark Profit percentage : {}%'.format(benchmark\_profit\_percentage), attrs = \['bold'\]))  
print(cl('ADX Strategy profit is {}% higher than the Benchmark Profit'.format(profit\_percentage - benchmark\_profit\_percentage), attrs = \['bold'\]))  
  
Benchmark profit by investing $10k : 2390.85  
Benchmark Profit percentage : 23%  
ADX Strategy profit is 66% higher than the Benchmark Profit

-   Comparing the ADX strategy daily/cumulative returns vs B&H

rets = aapl.close.pct\_change().dropna()  
strat\_rets = strategy.adx\_position\[1:\]\*rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7)  
strat\_rets.plot(color = 'r', linewidth = 1)  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7)  
strat\_cum.plot(color = 'r', linewidth = 2)  
plt.show()

![](https://miro.medium.com/v2/resize:fit:700/1*7vaqV4ESMwDB97F5FzJN0g.png)
ADX strategy daily/cumulative returns vs B&H

-   We can see that the ADX strategy does not outperform B&H.

## Backtesting MACD-CHOP Trading Strategy

-   Let’s combine MACD with the [Choppiness Index (CHOP)](https://www.tradingview.com/scripts/choppinessindex/) designed to determine if the market is choppy (trading sideways) or not choppy (trading within a trend in either direction).
-   While MACD helps traders assess the strength and direction of a trend, CHOP can be used to determine the market’s trendiness only, i.e. higher values of CHOP mean more choppiness, whereas lower values of CHOP indicate directional trending.
-   The hybrid MACD-CHOP trading strategy consists of the following three [steps](https://github.com/Nikhil-Adithyan/Algorithmic-Trading-with-Python/blob/main/Trend/Choppiness_Index.py) \[1, 22\]:

1.  Data preparation and MACD/CHOP calculation
2.  Creating the MACD-CHOP trading strategy
3.  Defining our position, backtesting and comparison against B&H.

-   Reading the stock data, calculating the MACD/CHOP indicators and defining the CHOP global threshold parameter aln

tsla = get\_historical\_data('PLTR', '2022-01-01')  
  
aln=65 #This is the optimal value that maximizes ROI during optimization, adjust it further if needed  
  
def get\_ci(high, low, close, lookback):  
    tr1 = pd.DataFrame(high - low).rename(columns = {0:'tr1'})  
    tr2 = pd.DataFrame(abs(high - close.shift(1))).rename(columns = {0:'tr2'})  
    tr3 = pd.DataFrame(abs(low - close.shift(1))).rename(columns = {0:'tr3'})  
    frames = \[tr1, tr2, tr3\]  
    tr = pd.concat(frames, axis = 1, join = 'inner').dropna().max(axis = 1)  
    atr = tr.rolling(1).mean()  
    highh = high.rolling(lookback).max()  
    lowl = low.rolling(lookback).min()  
    ci = 100 \* np.log10((atr.rolling(lookback).sum()) / (highh - lowl)) / np.log10(lookback)  
    return ci  
  
tsla\['ci\_14'\] = get\_ci(tsla\['high'\], tsla\['low'\], tsla\['close'\], 14)  
tsla = tsla.dropna()  
  
def get\_macd(price, slow, fast, smooth):  
    exp1 = price.ewm(span = fast, adjust = False).mean()  
    exp2 = price.ewm(span = slow, adjust = False).mean()  
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})  
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})  
    hist = pd.DataFrame(macd\['macd'\] - signal\['signal'\]).rename(columns = {0:'hist'})  
    frames =  \[macd, signal, hist\]  
    df = pd.concat(frames, join = 'inner', axis = 1)  
    return df  
  
tsla\_macd = get\_macd(tsla\['close'\], 26, 12, 9)  

-   Implementing the MACD-CHOP trading strategy and plotting the trading signals

def implement\_ci\_macd\_strategy(prices, data, ci):  
    buy\_price = \[\]  
    sell\_price = \[\]  
    ci\_macd\_signal = \[\]  
    signal = 0  
      
    for i in range(len(prices)):  
        if data\['macd'\].iloc\[i\] > data\['signal'\].iloc\[i\] and ci.iloc\[i\] < aln:  
            if signal != 1:  
                buy\_price.append(prices.iloc\[i\])  
                sell\_price.append(np.nan)  
                signal = 1  
                ci\_macd\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                ci\_macd\_signal.append(0)  
        elif data\['macd'\].iloc\[i\] < data\['signal'\].iloc\[i\] and ci.iloc\[i\] < aln:  
            if signal != -1:  
                buy\_price.append(np.nan)  
                sell\_price.append(prices.iloc\[i\])  
                signal = -1  
                ci\_macd\_signal.append(signal)  
            else:  
                buy\_price.append(np.nan)  
                sell\_price.append(np.nan)  
                ci\_macd\_signal.append(0)  
        else:  
            buy\_price.append(np.nan)  
            sell\_price.append(np.nan)  
            ci\_macd\_signal.append(0)  
      
    return buy\_price, sell\_price, ci\_macd\_signal  
  
buy\_price, sell\_price, ci\_macd\_signal = implement\_ci\_macd\_strategy(tsla\['close'\], tsla\_macd, tsla\['ci\_14'\])  
  
#Plotting Trading Signals  
  
ax1 = plt.subplot2grid((19,1,), (0,0), rowspan = 5, colspan = 1)  
ax2 = plt.subplot2grid((19,1), (7,0), rowspan = 5, colspan = 1)  
ax3 = plt.subplot2grid((19,1), (14,0), rowspan = 5, colspan = 1)  
ax1.plot(tsla\['close'\], linewidth = 2.5, color = '#2196f3')  
ax1.plot(tsla.index, buy\_price, marker = '^', color = 'green', markersize = 12, label = 'BUY SIGNAL', linewidth = 0)  
ax1.plot(tsla.index, sell\_price, marker = 'v', color = 'r', markersize = 12, label = 'SELL SIGNAL', linewidth = 0)  
ax1.set\_ylabel('PRICE USD')  
ax1.legend()  
ax1.set\_title('PLTR TRADING SIGNALS')  
ax2.plot(tsla\['ci\_14'\], linewidth = 2.5, color = '#fb8c00')  
ax2.axhline(aln, linestyle = '--', linewidth = 1.5, color = 'grey')  
#ax2.axhline(61.8, linestyle = '--', linewidth = 1.5, color = 'grey')  
ax2.set\_title('PLTR CHOPPINESS INDEX 14')  
ax2.set\_ylabel('CHOP')  
ax3.plot(tsla\_macd\['macd'\], color = 'grey', linewidth = 1.5, label = 'MACD')  
ax3.plot(tsla\_macd\['signal'\], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')  
for i in range(len(tsla\_macd)):  
    if str(tsla\_macd\['hist'\]\[i\])\[0\] == '-':  
        ax3.bar(tsla\_macd.index\[i\], tsla\_macd\['hist'\]\[i\], color = '#ef5350')  
    else:  
        ax3.bar(tsla\_macd.index\[i\], tsla\_macd\['hist'\]\[i\], color = '#26a69a')  
ax3.legend()  
ax3.set\_title('PLTR MACD 26,12,9')  
ax3.set\_xlabel('DATE')  
ax3.set\_ylabel('MACD')  
plt.show()

![PLTR MACD-CHOP trading signals vs Close Price & Indicators.](https://miro.medium.com/v2/resize:fit:700/1*n_--fw41YGkJOgTU6Sn24g.png)
PLTR MACD-CHOP trading signals vs Close Price & Indicators.

-   Defining our position and comparing the MACD-CHOP trading strategy daily/cumulative returns vs B&H

position = \[\]  
for i in range(len(ci\_macd\_signal)):  
    if ci\_macd\_signal\[i\] > 1:  
        position.append(0)  
    else:  
        position.append(1)  
          
for i in range(len(tsla\['close'\])):  
    if ci\_macd\_signal\[i\] == 1:  
        position\[i\] = 1  
    elif ci\_macd\_signal\[i\] == -1:  
        position\[i\] = 0  
    else:  
        position\[i\] = position\[i-1\]  
          
ci = tsla\['ci\_14'\]  
close\_price = tsla\['close'\]  
ci\_macd\_signal = pd.DataFrame(ci\_macd\_signal).rename(columns = {0:'ci\_macd\_signal'}).set\_index(tsla.index)  
position = pd.DataFrame(position).rename(columns = {0:'ci\_macd\_position'}).set\_index(tsla.index)  
  
frames = \[close\_price, ci, ci\_macd\_signal, position\]  
strategy = pd.concat(frames, join = 'inner', axis = 1)  
  
strategy.head()  
  
rets = tsla.close.pct\_change()\[1:\]  
strat\_rets = strategy.ci\_macd\_position\[1:\] \* rets  
  
plt.title('Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label="B&H")  
strat\_rets.plot(color = 'r', linewidth = 1,label="Strategy")  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label="B&H")  
strat\_cum.plot(color = 'r', linewidth = 2,label="Strategy")  
plt.legend()  
plt.show()

![MACD-CHOP trading strategy daily/cumulative returns vs B&H](https://miro.medium.com/v2/resize:fit:700/1*VPd1cZXRcaj_MeXwPVyP7g.png)
MACD-CHOP trading strategy daily/cumulative returns vs B&H

## Optimizing CCI-VI Trading Strategy

-   This section is about implementing and backtesting the hybrid CCI-VI trading strategy \[23\], which consists of the Commodity Channel Index (CCI) and the Vortex Indicator (VI). [CCI](https://www.tradingview.com/scripts/commoditychannelindex/) is a momentum oscillator that can identify overbought/oversold levels, while [VI](https://www.quantifiedstrategies.com/vortex-indicator-trading-strategy/) is a mean reversion oscillator that is used to spot trend reversals and current trends.
-   Defining the functions to compute the CCI and VI indicators

\# Function to calculate Commodity Channel Index (CCI)  
def calculate\_cci(df, period):  
    tp = (df\['High'\] + df\['Low'\] + df\['Close'\]) / 3  \# Typical Price  
    ma = tp.rolling(window=period).mean()  
    md = (tp - ma).abs().rolling(window=period).mean()  
    cci = (tp - ma) / (0.015 \* md)  
    return cci  
  
\# Function to calculate Vortex Indicator (VI)  
def calculate\_vortex(df, period):  
    tr = np.maximum(df\['High'\] - df\['Low'\],   
                    np.maximum(abs(df\['High'\] - df\['Close'\].shift(1)), abs(df\['Low'\] - df\['Close'\].shift(1))))  
    vm\_plus = abs(df\['High'\] - df\['Low'\].shift(1))  
    vm\_minus = abs(df\['Low'\] - df\['High'\].shift(1))  
    vi\_plus = vm\_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()  
    vi\_minus = vm\_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()  
    return vi\_plus, vi\_minus

-   Reading the input data and iterating over CCI/VI periods

df = get\_historical\_data('PLTR', '2022-01-01')  
df.rename(columns={'open': 'Open','close': 'Close','high':'High','low': 'Low','volume':'Volume'},inplace=True)  
  
\# Define parameter ranges  
cci\_periods = range(5, 101,5)  \# Test CCI periods from 5 to 100  
vortex\_periods = range(5, 101,5)  \# Test Vortex periods from 5 to 100  
  
\# Store results  
results = \[\]  
  
\# Iterate over all parameter combinations  
for cci\_p, vortex\_p in product(cci\_periods, vortex\_periods):  
    df\['CCI'\] = calculate\_cci(df, period=cci\_p)  
    df\['VI+'\], df\['VI-'\] = calculate\_vortex(df, period=vortex\_p)  
      
    \# Define Entry and Exit signals  
    df\['Entry'\] = ((df\['CCI'\] > 100) & (df\['VI+'\] > df\['VI-'\])) | ((df\['CCI'\] < -100) & (df\['VI-'\] > df\['VI+'\]))  
    df\['Exit'\] = ((df\['VI+'\] < df\['VI-'\]) & (df\['CCI'\] > 0)) | ((df\['VI-'\] < df\['VI+'\]) & (df\['CCI'\] < 0))  
      
    \# Filter data for backtesting (2022-2025)  
    df\_filtered = df\[(df.index.year >= 2022) & (df.index.year <= 2025)\]  
  
    \# Run backtest  
    portfolio = vbt.Portfolio.from\_signals(  
        close=df\_filtered\['Close'\],  
        entries=df\_filtered\['Entry'\],  
        exits=df\_filtered\['Exit'\],  
        init\_cash=10000,  
        fees=0.002  
    )  
      
    \# Store performance metrics  
    stats = portfolio.stats()  
    total\_return = stats.loc\['Total Return \[%\]'\]  
  
    results.append((cci\_p, vortex\_p, total\_return))

-   Finding the best parameters that maximize ROI

\# Convert results to DataFrame  
results\_df = pd.DataFrame(results, columns=\['CCI Period', 'Vortex Period', 'Total Return'\])  
  
\# Find best parameter set based on highest Total Return \[%\]  
best\_params = results\_df.sort\_values(by='Total Return', ascending=False).iloc\[0\]  
  
print("Best Parameters:")  
print(best\_params)  
  
Best Parameters:  
CCI Period        75.000000  
Vortex Period     35.000000  
Total Return     959.862567  
Name: 286, dtype: float64

-   Plotting the optimization heatmap “Total Return \[%\] for Different CCI and Vortex Periods”

\# Plot parameter performance  
plt.figure(figsize=(12, 6))  
heatmap\_data = results\_df.pivot(index='CCI Period', columns='Vortex Period', values='Total Return')  
sns.heatmap(heatmap\_data, annot=False, fmt=".2f", cmap="coolwarm")  
plt.title("Total Return \[%\] for Different CCI and Vortex Periods")  
plt.show()

![Total Return [%] for Different CCI and Vortex Periods](https://miro.medium.com/v2/resize:fit:700/1*IQe2_I1yCMC526TevoUFdw.png)
Total Return [%] for Different CCI and Vortex Periods

-   Backtesting the optimized CCI-VI trading strategy with [vectorbt](https://pypi.org/project/vectorbt/)

df\['CCI'\] = calculate\_cci(df, period=34)  
df\['VI+'\], df\['VI-'\] = calculate\_vortex(df, period=30)  
  
\# Define Entry and Exit signals based on CCI and Vortex Indicator  
df\['Entry'\] = ((df\['CCI'\] > 100) & (df\['VI+'\] > df\['VI-'\])) | ((df\['CCI'\] < -100) & (df\['VI-'\] > df\['VI+'\]))  
df\['Exit'\] = ((df\['VI+'\] < df\['VI-'\]) & (df\['CCI'\] > 0)) | ((df\['VI-'\] < df\['VI+'\]) & (df\['CCI'\] < 0))  
  
\# Filter data for the test period (2022-2025)  
df = df\[(df.index.year >= 2022) & (df.index.year <= 2025)\]  
  
\# Backtest using vectorbt  
portfolio = vbt.Portfolio.from\_signals(  
    close=df\['Close'\],  
    entries=df\['Entry'\],  
    exits=df\['Exit'\],  
    init\_cash=10000,  
    fees=0.002  
)  
  
\# Display performance metrics  
print(portfolio.stats())  
  
\# Plot equity curve  
portfolio.plot().show()  
  
Start                         2022\-01-03 00:00:00  
End                           2025\-07-25 00:00:00  
Period                                        893  
Start Value                               10000.0  
End Value                            57975.408572  
Total Return \[%\]                       479.754086  
Benchmark Return \[%\]                   756.988667  
Max Gross Exposure \[%\]                      100.0  
Total Fees Paid                       1359.360019  
Max Drawdown \[%\]                        38.888889  
Max Drawdown Duration                       196.0  
Total Trades                                   18  
Total Closed Trades                            17  
Total Open Trades                               1  
Open Trade PnL                        1643.613778  
Win Rate \[%\]                            58.823529  
Best Trade \[%\]                         117.536957  
Worst Trade \[%\]                        -16.532055  
Avg Winning Trade \[%\]                    31.31611  
Avg Losing Trade \[%\]                    -8.561432  
Avg Winning Trade Duration                   38.3  
Avg Losing Trade Duration               20.857143  
Profit Factor                            4.887326  
Expectancy                            2725.399694  
dtype: object

![Backtesting the optimized CCI-VI trading strategy with vectorbt](https://miro.medium.com/v2/resize:fit:700/1*vYZdKttg8PNzwQz8Bp6oEA.png)
Backtesting the optimized CCI-VI trading strategy with vectorbt

## Comparison of Essential Financial Ratios & KPI’s

-   In this section, we’ll use the FinanceToolkit \[24\] to compare financial performance of PLTR, NVDA and other Big Tech companies vs S&P 500 benchmark for the period 2021–2025.
-   Downloading the stock historical data, a financial statement, and profitability ratios

companies = Toolkit(\["PLTR","NVDA"\], api\_key=API\_KEY, start\_date="2020-12-31")  
  
\# Historical   
historical\_data = companies.get\_historical\_data()  
  
\# Financial Statement   
income\_statement = companies.get\_income\_statement()  
  
\# Ratios   
profitability\_ratios = companies.ratios.collect\_profitability\_ratios()

-   Comparing the volatility of PLTR and NVDA vs S&P 500 on 2025–07–25

print(historical\_data\['Volatility'\].iloc\[-1\])  
  
PLTR        0.0438  
NVDA        0.0338  
Benchmark   0.0111  
Name: 2025\-07-25, dtype: float64

![Volatility of PLTR and NVDA vs S&P 500 on 2025–07–25](https://miro.medium.com/v2/resize:fit:661/1*7bfa2RlY2glKu9dXFMOezg.png)
Volatility of PLTR and NVDA vs S&P 500 on 2025–07–25

-   Comparing cumulative returns of PLTR and NVDA vs S&P 500

display(historical\_data)  
  
\# Copy to clipboard (this is just to paste the data in the README)  
pd.io.clipboards.to\_clipboard(  
    historical\_data.xs("PLTR", axis=1, level=1).iloc\[:-1\].head().to\_markdown(),  
    excel=False,  
)  
  
\# Create a line chart for cumulative returns  
ax = historical\_data\["Cumulative Return"\].plot(  
    figsize=(15, 5),  
    lw=2,  
)  
  
\# Customize the colors and line styles  
ax.set\_prop\_cycle(color=\["#007ACC", "#FF6F61", "#4CAF50"\])  
ax.set\_xlabel("Year")  
ax.set\_ylabel("Cumulative Return")  
ax.set\_title(  
    "Cumulative Returns of PLTR vs S&P 500 as Benchmark"  
)  
  
\# Add a legend  
ax.legend(\["PLTR", "NVDA", "S&P 500"\], loc="upper left")  
  
\# Add grid lines for clarity  
ax.grid(True, linestyle="--", alpha=0.7)  
  
plt.show()

![Cumulative returns of PLTR and NVDA vs S&P 500](https://miro.medium.com/v2/resize:fit:700/1*1xcQf5j7wGEz4lYBy428Kg.png)
Cumulative returns of PLTR and NVDA vs S&P 500

-   Displaying the income statements

display(income\_statement)  
  
<class 'pandas.core.frame.DataFrame'\>  
MultiIndex: 56 entries, ('NVDA', 'Revenue') to ('PLTR', 'Weighted Average Shares Diluted')  
Data columns (total 6 columns):  
 \#   Column  Non-Null Count  Dtype    
\---  ------  --------------  -----    
 0   2020    28 non-null     float64  
 1   2021    56 non-null     float64  
 2   2022    56 non-null     float64  
 3   2023    56 non-null     float64  
 4   2024    56 non-null     float64  
 5   2025    28 non-null     float64

-   Displaying profitability ratios for PLTR and NVDA

display(profitability\_ratios)

-   Visualizing the PLTR Profitability Ratios

pd.io.clipboards.to\_clipboard(  
    profitability\_ratios.loc\["PLTR"\].head().to\_markdown(), excel=False  
)  
  
ratios\_to\_plot = \[  
    "Return on Assets",  
    "Return on Equity",  
    "Return on Invested Capital",  
    "Return on Tangible Assets",  
\]  
  
\# Create the plot  
ax = (  
    (profitability\_ratios.dropna(axis=1) \* 100)  
    .loc\["PLTR", ratios\_to\_plot, :\]  
    .T.plot(figsize=(15, 5), title="PLTR Profitability Ratios", lw=2)  
)  
  
\# Customize the line styles and colors  
line\_styles = \["-", "--", "-.", ":"\]  
line\_colors = \["blue", "red", "green", "purple"\]  
for i, line in enumerate(ax.get\_lines()):  
    line.set\_linestyle(line\_styles\[i\])  
    line.set\_color(line\_colors\[i\])  
  
\# Customize the legend  
ax.legend(ratios\_to\_plot)  
  
\# Add labels and grid  
plt.xlabel("Year", fontsize=12)  
plt.ylabel("Percentage (%)", fontsize=12)  
plt.grid(True, linestyle="--", alpha=0.7)  
  
\# Customize the title  
plt.title("PLTR Profitability Ratios")  
  
\# Show the plot  
plt.show()

![PLTR Profitability Ratios 2022–2024.](https://miro.medium.com/v2/resize:fit:700/1*Z8_e4YSI858hDKGlE9aWXw.png)
PLTR Profitability Ratios 2022–2024.

-   The same considerations apply to the NVDA Profitability Ratios

![NVDA Profitability Ratios 2022–2024.](https://miro.medium.com/v2/resize:fit:700/1*0dcQD40yzA2uk0dBXzVcFg.png)
NVDA Profitability Ratios 2022–2024.

-   It is reasonable to add the Extended Dupont Analysis for NVDA

display(extended\_dupont\_analysis)  
  
\# Copy to clipboard (this is just to paste the data in the README)  
pd.io.clipboards.to\_clipboard(  
    extended\_dupont\_analysis.loc\["NVDA"\].to\_markdown(), excel=False  
)  
  
apple\_extended\_dupont\_analysis = extended\_dupont\_analysis.loc\["NVDA", :, :\]  
apple\_extended\_dupont\_analysis = apple\_extended\_dupont\_analysis.loc\[:, "2020":\]  
  
\# Create a stacked area chart for the components of Return on Equity  
ax = apple\_extended\_dupont\_analysis.iloc\[:-1\].T.plot.area(  
    figsize=(12, 6), stacked=True, colormap="tab20", alpha=0.7  
)  
ax.set\_xlabel("Year")  
ax.set\_ylabel("Ratio Value")  
ax.set\_title("Extended Dupont Analysis for NVDA")  
  
\# Create a line chart for Return on Equity on a separate axis  
ax2 = ax.twinx()  
apple\_extended\_dupont\_analysis.T\["Return on Equity"\].plot(  
    legend=True, ax=ax2, color="purple", linestyle="--", marker="o"  
)  
ax2.set\_ylabel("Return on Equity")  
  
\# Customize labels and legends  
ax.legend(loc="lower right", bbox\_to\_anchor=(0.98, 0.02))  
ax2.legend(loc="upper left", bbox\_to\_anchor=(0.02, 0.98))  
  
\# Add grid lines for clarity  
ax.grid(True, linestyle="--", alpha=0.7)  
  
\# Use a better color map for the area chart  
ax.set\_prop\_cycle(None)  
  
plt.show()

![Extended Dupont Analysis for NVDA](https://miro.medium.com/v2/resize:fit:700/1*WfroVN6JDSVtaxj9HVaJqg.png)
Extended Dupont Analysis for NVDA

-   Displaying VaR for PLTR and NVDA vs Benchmark

display(value\_at\_risk)  
  
  
pd.io.clipboards.to\_clipboard(value\_at\_risk.iloc\[:-1\].tail().to\_markdown(), excel=False)  
  
\# Filter out the occasional positive return  
value\_at\_risk = value\_at\_risk\[value\_at\_risk < 0\]  
  
\# Create an area chart for Value at Risk (VaR) with custom styling  
fig, ax = plt.subplots(figsize=(15, 5))  
  
\# Customize the colors and transparency  
ax.set\_prop\_cycle(color=\["#007ACC", "#FF6F61", "#4CAF50"\])  
ax.set\_xlabel("Year", fontsize=12)  
ax.set\_ylabel("VaR", fontsize=12)  
  
\# Add a title with a unique font style  
ax.set\_title("Value at Risk (VaR)")  
  
\# Stack the area chart for a better visual representation  
value\_at\_risk.plot.area(ax=ax, stacked=False, alpha=0.7)  
  
\# Add grid lines with a unique linestyle  
ax.grid(True, linestyle="--", alpha=0.5)  
  
\# Customize the ticks and labels with a unique font  
ax.tick\_params(axis="both", which="major", labelsize=10)  
  
\# Add a background color to the plot area for a unique touch  
ax.set\_facecolor("#F0F0F0")  
  
\# Add a unique border color to the plot area  
ax.spines\["top"\].set\_color("#E0E0E0")  
ax.spines\["bottom"\].set\_color("#E0E0E0")  
ax.spines\["left"\].set\_color("#E0E0E0")  
ax.spines\["right"\].set\_color("#E0E0E0")  
  
\# Set a unique background color for the entire plot  
fig.set\_facecolor("#F8F8F8")  
  
\# Add an annotation box at a specific date  
annotation\_date = value\_at\_risk.sort\_values(by="Benchmark").index\[0\]  
annotation\_value = value\_at\_risk.loc\[value\_at\_risk.sort\_values(by="PLTR").index\[0\]\]\[  
    "Benchmark"  
\]  
ax.annotate(  
    "Impact of the Initial Publications about COVID-19",  
    xy=(annotation\_date, annotation\_value),  
    xytext=(-180, -30),  
    textcoords="offset points",  
    arrowprops=dict(arrowstyle="->", color="black"),  
    fontsize=10,  
    bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"),  
)  
  
\# Display the prettier VaR chart with the annotation  
plt.show()

![VaR for PLTR and NVDA vs Benchmark](https://miro.medium.com/v2/resize:fit:700/1*zp4CxMZQfnfmGZvSCQjcuw.png)
VaR for PLTR and NVDA vs Benchmark

-   Comparing the quarterly Jensen’s alpha PLTR vs NVDA

df = companies.performance.get\_jensens\_alpha(period="quarterly")  
  
df.plot(kind='bar',figsize=(10,4))  
plt.title('Jensens alpha')  
plt.xticks(rotation=45)  
plt.grid()  
plt.show()

![Jensen’s alpha PLTR vs NVDA](https://miro.medium.com/v2/resize:fit:700/1*7BaLUXBpTgrHhXXSjXkUdA.png)
Jensen’s alpha PLTR vs NVDA

-   Comparing the 12-month rolling Sharpe ratio PLTR vs NVDA

companies.performance.get\_sharpe\_ratio(period="monthly", rolling=12).plot(  
    figsize=(15, 5),  
    title="12-Month Rolling Sharpe Ratio",  
    grid=True,  
    colormap="plasma",  
    lw=3,  
    linestyle="-",  
    ylabel="Sharpe Ratio",  
    xlabel="Date",  
)

![12-month rolling Sharpe ratio PLTR vs NVDA](https://miro.medium.com/v2/resize:fit:700/1*wYEdZb5MLgdimSXQ-fLq3w.png)
12-month rolling Sharpe ratio PLTR vs NVDA

-   Comparing the Piotroski score PLTR vs NVDA

companies.models.get\_piotroski\_score().loc\[:, "Piotroski Score", :\].T.plot.bar(  
    figsize=(15, 3), rot=0, title="Piotroski Score", colormap="tab20c", ylim=(0, 9)  
)  
plt.grid()

![Piotroski score PLTR vs NVDA](https://miro.medium.com/v2/resize:fit:700/1*hTbKsJIeeKuvoMRvMEBKLw.png)
Piotroski score PLTR vs NVDA

-   Comparing the Altman Z-score PLTR vs NVDA

companies.models.get\_altman\_z\_score().loc\[:, "Altman Z-Score", :\].T.plot.bar(  
    figsize=(15, 3), rot=0, title="Altman Z-Score", colormap="tab20c", ylim=(0, 100)  
)  
plt.grid()

![Altman Z-score PLTR vs NVDA](https://miro.medium.com/v2/resize:fit:700/1*SD56XRaZVzrCTImWdTHrTA.png)
Altman Z-score PLTR vs NVDA

-   Comparing weekly returns vs quarterly max drawdown for PLTR, NVDA and Benchmark

(companies.get\_historical\_data(period="weekly")\["Return"\] \* 100).plot(  
    figsize=(15, 3),  
    title="Returns for PLTR, NVDA & Benchmark",  
    grid=True,  
    xlabel="Date",  
    ylabel="Returns (%)",  
    colormap="coolwarm",  
)  
  
plt.legend(loc="upper left")  
  
  
(companies.risk.get\_maximum\_drawdown(period="quarterly") \* 100).plot.area(  
    stacked=False,  
    figsize=(15, 3),  
    title="Maximum Drawdown",  
    xlabel="Date",  
    ylabel="Drawdown (%)",  
    colormap="coolwarm",grid=True,  
)  
  
plt.legend(loc="lower left")

![Weekly returns vs quarterly max drawdown for PLTR, NVDA and Benchmark](https://miro.medium.com/v2/resize:fit:700/1*H1-0CcD9LQHaplQ81FaBPA.png)
Weekly returns vs quarterly max drawdown for PLTR, NVDA and Benchmark

-   Comparing the effective tax rate for PLTR vs NVDA

dff=companies.ratios.get\_effective\_tax\_rate()  
  
dff.plot(kind='bar',figsize=(10,4))  
plt.title('Effective Tax Rate')  
plt.xticks(rotation=45)  
plt.grid()  
plt.show()

![Effective tax rate for PLTR vs NVDA](https://miro.medium.com/v2/resize:fit:700/1*-6P-YLIn_ATUyerATWOWkQ.png)
Effective tax rate for PLTR vs NVDA

-   In necessary, we can compare the Trailing EPS vs Trailing EPS Growth

companies.ratios.get\_earnings\_per\_share(trailing=3).T.dropna().plot(  
    figsize=(15, 3),  
    title="Trailing Earnings per Share",  
    grid=True,  
    linestyle="-",  
    linewidth=2,  
    xlabel="Date",  
    ylabel="Earnings per Share",  
)  
  
companies.ratios.get\_earnings\_per\_share(trailing=2, growth=True).T.dropna().plot(  
    figsize=(15, 3),  
    title="Trailing Earnings per Share Growth",  
    grid=True,  
    linestyle="-",  
    linewidth=2,  
    xlabel="Date",  
    ylabel="Earnings per Share Growth",  
)

-   For the sake of comparison, let’s look at the profile of Big Tech vs PLTR (as of 2025–07–25)

df=companies.get\_profile()  
  
df1 = pd.DataFrame(df,  
                  index=df.index, columns=\['NVDA', 'PLTR', 'AAPL', 'AMZN','META','GOOGL','MSFT'\])  
  
dfb=df.iloc\[2\]  
dfb.plot(title='Beta')  
plt.grid()  
  
dfb=df.iloc\[3\]  
dfb.plot(title='Volume')  
plt.grid()  
  
dfb=df.iloc\[4\]  
dfb.plot(title='Market Capitalization')  
plt.grid()

![Beta: BigTech vs PLTR](https://miro.medium.com/v2/resize:fit:634/1*1hVr4Q_5YCj_Ff_49RGRfA.png)
Beta: BigTech vs PLTR

![Volume: BigTech vs PLTR](https://miro.medium.com/v2/resize:fit:665/1*k-EZOUhTf0hrUwuDdhJ4Rw.png)
Volume: BigTech vs PLTR

![Market Capitalization: BigTech vs PLTR](https://miro.medium.com/v2/resize:fit:633/1*EMCx1GkacLe8Tk1JFubbPg.png)
Market Capitalization: BigTech vs PLTR

-   Comparing cumulative returns of BigTech vs PLTR

\# Plot the Cumulative Returns  
companies.get\_historical\_data(period="quarterly")\["Cumulative Return"\].plot(  
    figsize=(15, 5),  
    title="Cumulative Returns of Big Tech",  
    grid=True,  
    linestyle="-",  
    linewidth=2,  
    colormap="plasma",  
    xlabel="Date",  
    ylabel="Cumulative Return",  
)

![Cumulative returns of BigTech vs PLTR](https://miro.medium.com/v2/resize:fit:700/1*Q1efH_QcerEzD8oaeA3-vQ.png)
Cumulative returns of BigTech vs PLTR

## Ichimoku Cloud Chart

-   In this and following three sections below, we’ll continue using the FinanceToolkit \[24\] to switch back from fundamental to technical analysis of PLTR, NVDA and other Big Tech companies vs S&P 500 benchmark for the period 2021–2025. Together, technical and fundamental analysis can be coupled to create a trading strategy geared towards providing alpha.
-   Plotting the Ichimoku Cloud for PLTR and NVDA

display(ichimoku\_cloud)  
  
\# Define your Ichimoku Cloud data DataFrames   
ichimoku\_data\_aapl = ichimoku\_cloud.xs("NVDA", level=1, axis=1)  
ichimoku\_data\_msft = ichimoku\_cloud.xs("PLTR", level=1, axis=1)  
  
\#   
pd.io.clipboards.to\_clipboard(ichimoku\_data\_aapl.tail().to\_markdown(), excel=False)  
  
\# Take the last 500 rows  
ichimoku\_data\_aapl = ichimoku\_data\_aapl.iloc\[-500:\]  
ichimoku\_data\_msft = ichimoku\_data\_msft.iloc\[-500:\]  
  
\# Create a figure and two axes for horizontal subplots  
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
  
\# Convert the PeriodIndex to a DatetimeIndex  
ichimoku\_data\_aapl.index = ichimoku\_data\_aapl.index.to\_timestamp()  
ichimoku\_data\_msft.index = ichimoku\_data\_msft.index.to\_timestamp()  
  
\# Plot Ichimoku Cloud for AAPL  
ax1.plot(  
    ichimoku\_data\_aapl.index,  
    ichimoku\_data\_aapl\["Conversion Line"\],  
    color="blue",  
    label="Conversion Line (Tenkan-sen)",  
    linewidth=2,  
)  
ax1.plot(  
    ichimoku\_data\_aapl.index,  
    ichimoku\_data\_aapl\["Base Line"\],  
    color="red",  
    label="Base Line (Kijun-sen)",  
    linewidth=2,  
)  
ax1.fill\_between(  
    ichimoku\_data\_aapl.index,  
    ichimoku\_data\_aapl\["Leading Span A"\],  
    ichimoku\_data\_aapl\["Leading Span B"\],  
    where=ichimoku\_data\_aapl\["Leading Span A"\] >= ichimoku\_data\_aapl\["Leading Span B"\],  
    facecolor="green",  
    alpha=0.2,  
    label="Bullish Cloud",  
)  
ax1.fill\_between(  
    ichimoku\_data\_aapl.index,  
    ichimoku\_data\_aapl\["Leading Span A"\],  
    ichimoku\_data\_aapl\["Leading Span B"\],  
    where=ichimoku\_data\_aapl\["Leading Span A"\] < ichimoku\_data\_aapl\["Leading Span B"\],  
    facecolor="red",  
    alpha=0.2,  
    label="Bearish Cloud",  
)  
  
\# Customize the legend and labels for AAPL  
ax1.legend(loc="upper left")  
ax1.set\_xlabel("Date", fontsize=14)  
ax1.set\_ylabel("Price", fontsize=14)  
ax1.set\_title("Ichimoku Cloud Chart NVDA", fontsize=16)  
ax1.grid(True, linestyle="--", alpha=0.5)  
ax1.set\_facecolor("#f7f7f7")  
ax1.tick\_params(axis="both", which="major", labelsize=12)  
  
\# Plot Ichimoku Cloud for MSFT  
ax2.plot(  
    ichimoku\_data\_msft.index,  
    ichimoku\_data\_msft\["Conversion Line"\],  
    color="blue",  
    label="Conversion Line (Tenkan-sen)",  
    linewidth=2,  
)  
ax2.plot(  
    ichimoku\_data\_msft.index,  
    ichimoku\_data\_msft\["Base Line"\],  
    color="red",  
    label="Base Line (Kijun-sen)",  
    linewidth=2,  
)  
ax2.fill\_between(  
    ichimoku\_data\_msft.index,  
    ichimoku\_data\_msft\["Leading Span A"\],  
    ichimoku\_data\_msft\["Leading Span B"\],  
    where=ichimoku\_data\_msft\["Leading Span A"\] >= ichimoku\_data\_msft\["Leading Span B"\],  
    facecolor="green",  
    alpha=0.2,  
    label="Bullish Cloud",  
)  
ax2.fill\_between(  
    ichimoku\_data\_msft.index,  
    ichimoku\_data\_msft\["Leading Span A"\],  
    ichimoku\_data\_msft\["Leading Span B"\],  
    where=ichimoku\_data\_msft\["Leading Span A"\] < ichimoku\_data\_msft\["Leading Span B"\],  
    facecolor="red",  
    alpha=0.2,  
    label="Bearish Cloud",  
)  
  
\# Customize the legend and labels for MSFT  
ax2.legend(loc="upper left")  
ax2.set\_xlabel("Date", fontsize=14)  
ax2.set\_ylabel("Price", fontsize=14)  
ax2.set\_title("Ichimoku Cloud Chart PLTR", fontsize=16)  
ax2.grid(True, linestyle="--", alpha=0.5)  
ax2.set\_facecolor("#f7f7f7")  
ax2.tick\_params(axis="both", which="major", labelsize=12)  
  
\# Adjust spacing between subplots  
plt.tight\_layout()  
  
\# Show the plot  
plt.show()

![Ichimoku Cloud Chart for PLTR and NVDA 2023–2025](https://miro.medium.com/v2/resize:fit:700/1*nZsUtyPHVeI6LAI4TjXduA.png)
Ichimoku Cloud Chart for PLTR and NVDA 2023–2025

## Bollinger Bands

-   Plotting the Bollinger Bands for PLTR

bollinger\_bands = companies.technicals.get\_bollinger\_bands()  
  
display(bollinger\_bands)  
  
bollinger\_bands.xs("PLTR", level=1, axis=1).plot(  
    figsize=(15, 3),  
    title="PLTR Bollinger Bands",  
    grid=True,  
    legend=True,  
    colormap="seismic",  
)

![Bollinger Bands for PLTR](https://miro.medium.com/v2/resize:fit:700/1*N_G22CeAs_57EWNJ2_FXJw.png)
Bollinger Bands for PLTR

-   Plotting the Bollinger Bands for NVDA

display(bollinger\_bands)  
  
bollinger\_bands.xs("NVDA", level=1, axis=1).plot(  
    figsize=(15, 3),  
    title="NVDA Bollinger Bands",  
    grid=True,  
    legend=True,  
    colormap="seismic",  
)

![Bollinger Bands for NVDA](https://miro.medium.com/v2/resize:fit:700/1*E08pORFH8Dm1mGksXriRbA.png)
Bollinger Bands for NVDA

## RSI for Big Tech

-   Comparing the RSI indicator for BigTech & PLTR vs Benchmark

\# Obtain the Relative Strength Index (RSI) for each company  
rsi = companies.technicals.get\_relative\_strength\_index()  
  
\# Show the RSI for each company  
display(rsi)  
  
\# Plot the RSI for each company  
rsi.plot(  
    subplots=True,  
    figsize=(15, 12),  
    title="Relative Strength Index for Big Tech Benchmark",  
    legend=True,  
    grid=True,  
    ylim=(0, 100),  
    sharex=True,  
)

![RSI indicator for BigTech & PLTR vs Benchmark](https://miro.medium.com/v2/resize:fit:700/1*XrxrsYZM_0___GbyJrN6nQ.png)
RSI indicator for BigTech & PLTR vs Benchmark

## Other Technical Indicators

-   Examining the full list of available technical indicators \[24\] for our portfolio of BigTech, PLTR & Benchmark

dff=companies.technicals.collect\_all\_indicators()  
  
for column in dff:  
    print(column)  
  
\# See Appendix A

_Below are some examples of technical indicators of interest listed above:_

-   McClellan Oscillator

plt.rcParams\["figure.figsize"\] = (20,3)  
dff\['McClellan Oscillator'\].plot()  
plt.legend(loc="upper left")  
plt.title('McClellan Oscillator')  
plt.grid()  
plt.show()

![McClellan Oscillator](https://miro.medium.com/v2/resize:fit:700/1*vgw1uKerHl5Kzcybyq6GUw.png)
McClellan Oscillator

-   Advancers — Decliners

plt.rcParams\["figure.figsize"\] = (20,3)  
dff\['Advancers - Decliners'\].plot()  
plt.legend(loc="upper left")  
plt.title('Advancers - Decliners')  
plt.grid()  
plt.show()

![Advancers — Decliners](https://miro.medium.com/v2/resize:fit:700/1*MtsPkyx9yezywfai2iEGaw.png)
Advancers — Decliners

-   Williams %R

osc='Williams %R'  
plt.rcParams\["figure.figsize"\] = (20,3)  
dff\[osc\].plot()  
plt.legend(loc="lower right")  
plt.title(osc)  
plt.grid()  
plt.show()

![Williams %R](https://miro.medium.com/v2/resize:fit:700/1*VeXVUun7d78WKS1bI5sKnQ.png)
Williams %R

-   Aroon Indicator

osc='Aroon Indicator Up'  
plt.rcParams\["figure.figsize"\] = (20,3)  
dff\[osc\].plot()  
plt.legend(loc="lower right")  
plt.title(osc)  
plt.grid()  
plt.show()

![Aroon Indicator Up](https://miro.medium.com/v2/resize:fit:700/1*lvR2jw_s4r7LFNh8f5HpJg.png)
Aroon Indicator Up

osc='Aroon Indicator Down'  
plt.rcParams\["figure.figsize"\] = (20,3)  
dff\[osc\].plot()  
plt.legend(loc="lower right")  
plt.title(osc)  
plt.grid()  
plt.show()

![Aroon Indicator Down](https://miro.medium.com/v2/resize:fit:700/1*Awd24Q_HeXw36bBaIeXnkQ.png)
Aroon Indicator Down

-   Ultimate Oscillator

osc='Ultimate Oscillator'  
plt.rcParams\["figure.figsize"\] = (20,3)  
dff\[osc\].plot()  
plt.legend(loc="lower left")  
plt.title(osc)  
plt.grid()  
plt.show()

![Ultimate Oscillator](https://miro.medium.com/v2/resize:fit:700/1*hJUyCITVrraHSS2_ZEyczA.png)
Ultimate Oscillator

-   Percentage Price Oscillator

osc='Percentage Price Oscillator'  
plt.rcParams\["figure.figsize"\] = (20,3)  
dff\[osc\].plot()  
plt.legend(loc="lower left")  
plt.title(osc)  
plt.grid()  
plt.show()

![Percentage Price Oscillator](https://miro.medium.com/v2/resize:fit:700/1*yTrjI1-0GZRWLMYBjg3vrQ.png)
Percentage Price Oscillator

-   Chande Momentum Oscillator

osc='Chande Momentum Oscillator'  
plt.rcParams\["figure.figsize"\] = (20,3)  
dff\[osc\].plot()  
plt.legend(loc="lower left")  
plt.title(osc)  
plt.grid()  
plt.show()

![Chande Momentum Oscillator](https://miro.medium.com/v2/resize:fit:700/1*2GmhpsSsAenyOFizYtDKbw.png)
Chande Momentum Oscillator

## Third-Party Insights

-   [Macroaxis Wealth Optimization Platform: Palantir Technologies Class Stock Today](https://www.macroaxis.com/stock/PLTR/Palantir-Technologies-Class)

![Source: Macroaxis](https://miro.medium.com/v2/resize:fit:700/1*iXNtprAL70_Mw-XxDIz9Wg.png)
Source: Macroaxis

-   [TradingView PLTR Financials](https://www.tradingview.com/symbols/NASDAQ-PLTR/financials-overview/): Financial position and solvency of the company

![Annual Debt Level and Coverage (Source: TradingView).](https://miro.medium.com/v2/resize:fit:700/1*p-r4tyqzNGJUF4art2CRxA.png)
Annual Debt Level and Coverage (Source: TradingView).

-   [Barchart Opinion: PLTR](https://www.barchart.com/stocks/quotes/PLTR/overview)

![Barchart Opinion PLTR](https://miro.medium.com/v2/resize:fit:700/1*AkdB4Ay52JXRZWRqd_dOgg.png)
Barchart Opinion: PLTR

-   [Zacks Ranking of PLTR](https://www.zacks.com/stock/quote/PLTR):

![Source: Zacks](https://miro.medium.com/v2/resize:fit:700/1*RR4kpqATQOz3B6_ptwBUEQ.png)
Source: Zacks

## Final Thoughts!

-   We have addressed _the market and model risks_ by adopting the integrated de-risking AT strategies summarized below.
-   First and foremost, the study demonstrates the great value of using [TwelveData](https://twelvedata.com/) API to fetch clean and structured historical stock data.
-   By jointly interpreting 25 trading indicators and backtesting 22 trading strategies, we reduced the risk of false signals and minimized uncertainties of trading decisions.
-   As we saw, combining several technical indicators can have many advantages including increased profitability of the resulting hybrid trading strategy (OBV-EMA, VPT-MA, VROC-MA, Fisher-PVT, MACD-CHOP, and CCI-VI).
-   Further, incorporating both fundamental and technical analysis allows us to capture short-term price fluctuations and evaluate the intrinsic value of a stock with different financial models and KPI’s: volatility, daily/cumulative returns, profitability ratios, Extended Dupont Analysis, VaR, max Drawdown, Beta, Jensen’s Alpha, Sharpe ratio, Piotroski score, Altman Z-score, etc.
-   It is worth mentioning that the following 23 trading indicators and financial metrics have played an important role throughout the study:
-   RSI, MACD, ADX, TSI, CCI, VI, CHOP, DVPO, OBV, VPT, VROC, VWAP, A/D, BB, Ichimoku, Fisher, PVT, NVI, KO, MFI, CMF, Alpha, and Beta.
-   We have implemented and tested accurate [stock price prediction](https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning) using [machine learning](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-machine-learning), [deep learning](https://www.simplilearn.com/tutorials/deep-learning-tutorial/what-is-deep-learning) and statistical modeling such as [FB Prophet](https://facebook.github.io/prophet/).
-   It is now possible to analyze vast amounts of stock data using the following algorithms:

1.  The binary classification model [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) to predict the AT position based on Momentum and OBV as the model features.
2.  The short-term close price forecasting using [FB Prophet](https://github.com/facebook/prophet) by adding the 10-day Moving Average (MA) as a model feature.
3.  The Prophet model has been complemented by Long Short-Term Memory (LSTM) to predict stock prices.

-   Specifically, we have compared the performance of LSTM and Prophet, showcasing their prediction capabilities and error rates.
-   Numerous examples have shown that financial data visualization (FDV) is crucial in AT for simplifying complex financial patterns and trends, enhancing our understanding of market dynamics, and facilitating better decision-making.
-   Relevant examples of FDV are OHLC with trading indicators, trading signals, volume ratio, dynamic volume profile oscillator, backtesting summary plots, the aforementioned 3rd party dashboards, etc.
-   In summary, this study lays the ground of more advanced applications in quantitative finance, from AT and AI-infused market analytics to better financial decision making.

## The Road Ahead

-   We are still building, backtesting and optimizing hybrid trading strategies based on other trading indicators listed in Appendix A.
-   Other promising area in stock market prediction involves integrating technical indicators as inputs to deep learning models.
-   The ongoing effort is focused on development of AI-powered AT bots that minimize risks and optimize core financials in real-time. This includes FDV-based data analysis, ML predictions, and dynamic risk KPIs.

## References

1.  [Algorithmic Trading with Python](https://github.com/Nikhil-Adithyan/Algorithmic-Trading-with-Python)
2.  [Data Science for Financial Markets](https://www.kaggle.com/code/lusfernandotorres/data-science-for-financial-markets)
3.  [Python Libraries Explained: Transforming Data for Effective Trading](https://blog.quantinsti.com/python-trading-library/)
4.  [FinanceToolkit](https://github.com/JerBouma/FinanceToolkit)
5.  [Advanced Momentum Trading Strategies with Volatility and Volume Indicators Using Python](https://eodhd.medium.com/advanced-momentum-trading-strategies-with-volatility-and-volume-indicators-using-python-412708c53603)
6.  [Risks Encountered in Algorithmic Trading: Top 5 Insights](https://www.utradealgos.com/blog/top-5-risks-encountered-in-algorithmic-trading-insights-and-solutions)
7.  [Combining Technical and Fundamental Analysis with Deep Learning for Stock Prediction](https://medium.com/@zhonghong9998/combining-technical-and-fundamental-analysis-with-deep-learning-for-stock-prediction-5a84c915048f)
8.  [Decoding Market Sentiment: A Deep Dive into Volume-Based Indicators and Forecasting using Machine learning](https://blog.gopenai.com/decoding-market-sentiment-a-deep-dive-into-volume-based-indicators-and-forecasting-using-machine-32f9efb8f8c1)
9.  [Asset Beta and Market Beta in Python](https://blog.quantinsti.com/asset-beta-market-beta-python/)
10.  [Unlocking Stock Trading Insights: Using the Volume Ratio for Smarter Decision-Making with Python](https://medium.datadriveninvestor.com/unlocking-stock-trading-insights-using-the-volume-ratio-for-smarter-decision-making-with-python-4914b9271a3d)
11.  [Top 9 Volume Indicators in Python](https://medium.com/@crisvelasquez/top-9-volume-indicators-in-python-e398791b98f9)
12.  [Algorithmic Trading with Fisher Transform & Price Volume Trend](https://medium.com/@kridtapon/algorithmic-trading-with-fisher-transform-price-volume-trend-0fb59aa37308)
13.  [Measuring Price Stretch from Volume Consensus](https://medium.com/@crisvelasquez/measuring-price-stretch-from-volume-consensus-5bfff300fcf0)
14.  [Predicting Stock Prices with Prophet: A Python Guide](https://medium.com/@serdarilarslan/predicting-stock-prices-with-prophet-a-python-guide-7b773d821fef)
15.  [Share Price Forecasting Using Facebook Prophet](https://www.geeksforgeeks.org/machine-learning/share-price-forecasting-using-facebook-prophet/)
16.  [Simple Stock Price Forecasting Model Using Prophet](https://colab.research.google.com/drive/1M3JtBChpDi8R0r2qEA9t5Jn8zjMr6SId?usp=sharing)
17.  [Stock Price Prediction using LSTM and Prophet](https://www.kaggle.com/code/bkcoban/stock-price-prediction-using-lstm-and-prophet)
18.  [Algorithmic Trading with Relative Strength Index in Python](https://eodhd.com/financial-academy/backtesting-strategies-examples/algorithmic-trading-with-relative-strength-index-in-python)
19.  [Optimizing Bitcoin Trading Strategies with MACD](https://medium.com/@kridtapon/optimizing-bitcoin-trading-strategies-with-macd-24299044b8d7)
20.  [Coding the True Strength Index and Backtesting a Trading Strategy in Python](https://eodhd.com/financial-academy/backtesting-strategies-examples/coding-the-true-strength-index-and-backtesting-a-trading-strategy-in-python)
21.  [Algorithmic Trading with Average Directional Index in Python](https://medium.com/codex/algorithmic-trading-with-average-directional-index-in-python-2b5a20ecf06a)
22.  [Detecting Ranging and Trending Markets with Choppiness Index in Python](https://eodhd.com/financial-academy/backtesting-strategies-examples/detecting-ranging-and-trending-markets-with-choppiness-index-in-python)
23.  [Data-Driven Trading: Building and Backtesting a Strategy with Python](https://medium.com/@kridtapon/data-driven-trading-building-and-backtesting-a-strategy-with-python-3b772765ad8b)
24.  [FinanceToolkit](https://www.jeroenbouma.com/projects/financetoolkit/getting-started)

## Explore More

-   [Backtesting Hybrid CI & MACD Trading Strategies for NVIDIA](https://medium.com/@alexzap922/backtesting-hybrid-ci-macd-trading-strategies-for-nvidia-b180d593e414)
-   [Creating & Using this Flask DIY Stock Screener App & Dashboard — It’s Money for Jam!](https://medium.com/insiderfinance/creating-using-this-flask-diy-stock-screener-app-dashboard-its-money-for-jam-e663a8eb31f6)
-   [Backtesting Volume Adjusted Moving Average (VAMA) Trading Strategy: Bayesian Optimization & Granger Causality](https://medium.com/insiderfinance/backtesting-volume-adjusted-moving-average-vama-trading-strategy-bayesian-optimization-granger-e09015a164a1)
-   [The Zero Lag DWT Crossover Strategy that Outperforms SMA, EMA & Buy-Hold](https://medium.com/insiderfinance/the-zero-lag-dwt-crossover-strategy-that-outperforms-sma-ema-buy-hold-c8ada0bf936a)
-   [Comparing Profitability of 4 Moving Average (MA) Algo-Trading Strategies in Python: A WMT Use Case](https://medium.com/@alexzap922/comparing-profitability-of-4-moving-average-ma-algo-trading-strategies-in-python-a-wmt-use-case-c0ab115fd470)
-   [Can Supervised ML Classifiers Improve ROI of Pairs Trading?](https://medium.com/python-in-plain-english/can-supervised-ml-classifiers-improve-roi-of-pairs-trading-1a6e1a1f9711)
-   [Can Financial Data Visualization Be a Business Game-Changer?](https://medium.com/insiderfinance/can-financial-data-visualization-be-a-business-game-changer-1f55fda1d635)
-   [Unlocking Palantir Technologies’ Potential with Profitable Algo Trading Strategies in Python — 2. Middle-Term Hull Volume Moving Average](https://medium.com/insiderfinance/unlocking-palantir-technologies-potential-with-profitable-algo-trading-strategies-in-python-2-60bbbde5e60f)
-   [Unlocking Palantir Technologies’ Potential with Profitable Algo Trading Strategies in Python — 1. ADX RSI](https://medium.com/insiderfinance/unlocking-palantir-technologies-potential-with-profitable-algo-trading-strategies-in-python-1-85023d734a36)
-   [Discovering the Best Integrated Platforms for Big Tech Quantitative Finance — 1. Finance Toolkit](https://medium.com/insiderfinance/discovering-the-best-integrated-platforms-for-big-tech-quantitative-finance-1-finance-toolkit-2e29b67eb4ea)
-   [Should I Algo Trade AMD with Gradient Boosting Classifier, Optimized SMA & Backtesting, or Just Buy & Hold?](https://medium.com/insiderfinance/should-i-algo-trade-amd-with-gradient-boosting-classifier-optimized-sma-backtesting-or-just-buy-13e5e8846b4d)
-   [Backtesting, Optimizing & Combining Multiple Algorithmic Trading Strategies Effectively: Bitcoin Use Case](https://medium.com/h7w/backtesting-optimizing-combining-multiple-algorithmic-trading-strategies-effectively-bitcoin-0344e1d5ca54)
-   [Creating & Backtesting 16 Popular Algo-Trading Strategies with Backtrader](https://medium.com/@alexzap922/creating-backtesting-16-popular-algo-trading-strategies-with-backtrader-21a45e93bc8d)
-   [Using Multiple ML/AI Methods for AMZN Stock Price Predictions](https://medium.com/insiderfinance/using-multiple-ml-ai-methods-for-amzn-stock-price-predictions-bd5020bb8096)
-   [Security De-Risking with Supervised ML & FRM KPIs: AAPL Use Case](https://medium.com/@alexzap922/security-de-risking-with-supervised-ml-frm-kpis-aapl-use-case-18b3aab3f14f)
-   [The Role of Risk Severity Matrix with Implications & Limitations in FinTech Explained](https://medium.com/@alexzap922/the-role-of-risk-severity-matrix-with-implications-limitations-in-fintech-explained-1cf7926f5349)
-   [NVDA vs BTC Algorithmic Trading: Backtest BB, MACD & AO Trading Strategies](https://medium.com/@alexzap922/nvda-vs-btc-algorithmic-trading-backtest-bb-macd-ao-trading-strategies-1db4f24d24ef)
-   [Backtesting TESLA Crossover Strategies: Noise-Resilient Wiener Filter MA vs SMA](https://medium.com/@alexzap922/backtesting-tesla-crossover-strategies-noise-resilient-wiener-filter-ma-vs-sma-2b9ab0fe2b63)
-   [BTC-USD Price Prediction using FB Prophet with Hyperparameter Optimization, Cross-Validation QC & Modified Algo-Trading Strategies](https://medium.com/@alexzap922/btc-usd-price-prediction-using-fb-prophet-with-hyperparameter-optimization-cross-validation-qc-7848b41dac30)
-   [5Y WMT Algo-Trading: Comparing DEMA vs SMA Crossover Strategies with Backtesting & SPY/Buy-Hold Benchmarking](https://medium.com/@alexzap922/5y-wmt-algo-trading-comparing-dema-vs-sma-crossover-strategies-with-backtesting-spy-buy-hold-94478353af33)
-   [LSTM SOL-USD Price Prediction with Data Leakage Mitigation & Residual QC Analysis](https://medium.com/@alexzap922/lstm-sol-usd-price-prediction-with-data-leakage-mitigation-residual-qc-analysis-6fd500abbf81)
-   [Resolving Bias-Variance Tradeoff in BTC-USD Price Prediction: Incorporating Technical Indicators into XGBoost Regression with Optuna Hyperparameter Optimization](https://medium.com/@alexzap922/resolving-bias-variance-tradeoff-in-btc-usd-price-prediction-incorporating-technical-indicators-0339c8d9f30a)
-   [Using Supervised Machine Learning Algorithms to Predict Pricing Trends & Reversals: In-Sample vs Out-of-Sample AMD Stock Prices](https://medium.com/@alexzap922/using-supervised-machine-learning-algorithms-to-predict-pricing-trends-reversals-in-sample-vs-23b79d7f7880)
-   [Backtesting BB-MA20 Algo-Trading Strategy INTC vs S&P 500](https://medium.com/@alexzap922/backtesting-bb-ma20-algo-trading-strategy-intc-vs-s-p-500-372efc47aa9f)
-   [A Market-Neutral Strategy](https://wp.me/pdMwZd-8ab)
-   [A Comprehensive Analysis of Best Trading Technical Indicators w/ TA-Lib — Tesla ‘23](https://wp.me/pdMwZd-859)
-   [Plotly Dash TA Stock Market App](https://wp.me/pdMwZd-7A3)
-   [Returns-Volatility Domain K-Means Clustering and LSTM Anomaly Detection of S&P 500 Stocks](https://wp.me/pdMwZd-7dS)
-   [NVIDIA Returns-Drawdowns MVA & RNN Mean Reversal Trading](https://wp.me/pdMwZd-7dI)
-   [Oracle Monte Carlo Stock Simulations](https://wp.me/pdMwZd-7e5)
-   [NVIDIA Rolling Volatility: GARCH & XGBoost](https://wp.me/pdMwZd-7by)
-   [IQR-Based Log Price Volatility Ranking of Top 19 Blue Chips](https://wp.me/pdMwZd-71E)
-   [Multiple-Criteria Technical Analysis of Blue Chips in Python](https://wp.me/pdMwZd-6V7)
-   [Datapane Stock Screener App from Scratch](https://wp.me/pdMwZd-61w)
-   [Data Visualization in Python — 1. Stock Technical Indicators](https://wp.me/pdMwZd-58A)
-   [JPM Breakouts: Auto ARIMA, FFT, LSTM & Stock Indicators](https://wp.me/pdMwZd-4Vu)
-   [Post-SVB Risk Aware Investing](https://wp.me/pdMwZd-4Rj)
-   [Applying a Risk-Aware Portfolio Rebalancing Strategy to ETF, Energy, Pharma, and Aerospace/Defense Stocks in 2023](https://wp.me/pdMwZd-4TI)
-   [Portfolio Optimization of 20 Dividend Growth Stocks](https://wp.me/pdMwZd-4Je)
-   [Donchian Channel Trading Systems](https://wp.me/pdMwZd-4Fd)
-   [Towards Max(ROI/Risk) Trading](https://wp.me/pdMwZd-4IB)
-   [The Qullamaggie’s TSLA Breakouts for Swing Traders](https://wp.me/pdMwZd-1uP)
-   [Algorithmic Testing Stock Portfolios to Optimize the Risk/Reward Ratio](https://wp.me/pdMwZd-192)
-   [Quant Trading using Monte Carlo Predictions and 62 AI-Assisted Trading Technical Indicators (TTI)](https://wp.me/pdMwZd-Rm)

## Appendix A: Full List of Technical Indicators

('McClellan Oscillator', 'NVDA')  
('McClellan Oscillator', 'PLTR')  
('McClellan Oscillator', 'AMZN')  
('McClellan Oscillator', 'MSFT')  
('McClellan Oscillator', 'AAPL')  
('McClellan Oscillator', 'META')  
('McClellan Oscillator', 'GOOGL')  
('McClellan Oscillator', 'Benchmark')  
('Advancers - Decliners', 'NVDA')  
('Advancers - Decliners', 'PLTR')  
('Advancers - Decliners', 'AMZN')  
('Advancers - Decliners', 'MSFT')  
('Advancers - Decliners', 'AAPL')  
('Advancers - Decliners', 'META')  
('Advancers - Decliners', 'GOOGL')  
('Advancers - Decliners', 'Benchmark')  
('On-Balance Volume', 'NVDA')  
('On-Balance Volume', 'PLTR')  
('On-Balance Volume', 'AMZN')  
('On-Balance Volume', 'MSFT')  
('On-Balance Volume', 'AAPL')  
('On-Balance Volume', 'META')  
('On-Balance Volume', 'GOOGL')  
('On-Balance Volume', 'Benchmark')  
('Accumulation/Distribution Line', 'NVDA')  
('Accumulation/Distribution Line', 'PLTR')  
('Accumulation/Distribution Line', 'AMZN')  
('Accumulation/Distribution Line', 'MSFT')  
('Accumulation/Distribution Line', 'AAPL')  
('Accumulation/Distribution Line', 'META')  
('Accumulation/Distribution Line', 'GOOGL')  
('Accumulation/Distribution Line', 'Benchmark')  
('Chaikin Oscillator', 'NVDA')  
('Chaikin Oscillator', 'PLTR')  
('Chaikin Oscillator', 'AMZN')  
('Chaikin Oscillator', 'MSFT')  
('Chaikin Oscillator', 'AAPL')  
('Chaikin Oscillator', 'META')  
('Chaikin Oscillator', 'GOOGL')  
('Chaikin Oscillator', 'Benchmark')  
('Money Flow Index', 'NVDA')  
('Money Flow Index', 'PLTR')  
('Money Flow Index', 'AMZN')  
('Money Flow Index', 'MSFT')  
('Money Flow Index', 'AAPL')  
('Money Flow Index', 'META')  
('Money Flow Index', 'GOOGL')  
('Money Flow Index', 'Benchmark')  
('Williams %R', 'NVDA')  
('Williams %R', 'PLTR')  
('Williams %R', 'AMZN')  
('Williams %R', 'MSFT')  
('Williams %R', 'AAPL')  
('Williams %R', 'META')  
('Williams %R', 'GOOGL')  
('Williams %R', 'Benchmark')  
('Aroon Indicator Up', 'AAPL')  
('Aroon Indicator Up', 'AMZN')  
('Aroon Indicator Up', 'Benchmark')  
('Aroon Indicator Up', 'GOOGL')  
('Aroon Indicator Up', 'META')  
('Aroon Indicator Up', 'MSFT')  
('Aroon Indicator Up', 'NVDA')  
('Aroon Indicator Up', 'PLTR')  
('Aroon Indicator Down', 'AAPL')  
('Aroon Indicator Down', 'AMZN')  
('Aroon Indicator Down', 'Benchmark')  
('Aroon Indicator Down', 'GOOGL')  
('Aroon Indicator Down', 'META')  
('Aroon Indicator Down', 'MSFT')  
('Aroon Indicator Down', 'NVDA')  
('Aroon Indicator Down', 'PLTR')  
('Commodity Channel Index', 'NVDA')  
('Commodity Channel Index', 'PLTR')  
('Commodity Channel Index', 'AMZN')  
('Commodity Channel Index', 'MSFT')  
('Commodity Channel Index', 'AAPL')  
('Commodity Channel Index', 'META')  
('Commodity Channel Index', 'GOOGL')  
('Commodity Channel Index', 'Benchmark')  
('Relative Vigor Index', 'NVDA')  
('Relative Vigor Index', 'PLTR')  
('Relative Vigor Index', 'AMZN')  
('Relative Vigor Index', 'MSFT')  
('Relative Vigor Index', 'AAPL')  
('Relative Vigor Index', 'META')  
('Relative Vigor Index', 'GOOGL')  
('Relative Vigor Index', 'Benchmark')  
('Force Index', 'NVDA')  
('Force Index', 'PLTR')  
('Force Index', 'AMZN')  
('Force Index', 'MSFT')  
('Force Index', 'AAPL')  
('Force Index', 'META')  
('Force Index', 'GOOGL')  
('Force Index', 'Benchmark')  
('Ultimate Oscillator', 'NVDA')  
('Ultimate Oscillator', 'PLTR')  
('Ultimate Oscillator', 'AMZN')  
('Ultimate Oscillator', 'MSFT')  
('Ultimate Oscillator', 'AAPL')  
('Ultimate Oscillator', 'META')  
('Ultimate Oscillator', 'GOOGL')  
('Ultimate Oscillator', 'Benchmark')  
('Percentage Price Oscillator', 'NVDA')  
('Percentage Price Oscillator', 'PLTR')  
('Percentage Price Oscillator', 'AMZN')  
('Percentage Price Oscillator', 'MSFT')  
('Percentage Price Oscillator', 'AAPL')  
('Percentage Price Oscillator', 'META')  
('Percentage Price Oscillator', 'GOOGL')  
('Percentage Price Oscillator', 'Benchmark')  
('Detrended Price Oscillator', 'NVDA')  
('Detrended Price Oscillator', 'PLTR')  
('Detrended Price Oscillator', 'AMZN')  
('Detrended Price Oscillator', 'MSFT')  
('Detrended Price Oscillator', 'AAPL')  
('Detrended Price Oscillator', 'META')  
('Detrended Price Oscillator', 'GOOGL')  
('Detrended Price Oscillator', 'Benchmark')  
('Average Directional Index', 'NVDA')  
('Average Directional Index', 'PLTR')  
('Average Directional Index', 'AMZN')  
('Average Directional Index', 'MSFT')  
('Average Directional Index', 'AAPL')  
('Average Directional Index', 'META')  
('Average Directional Index', 'GOOGL')  
('Average Directional Index', 'Benchmark')  
('Chande Momentum Oscillator', 'NVDA')  
('Chande Momentum Oscillator', 'PLTR')  
('Chande Momentum Oscillator', 'AMZN')  
('Chande Momentum Oscillator', 'MSFT')  
('Chande Momentum Oscillator', 'AAPL')  
('Chande Momentum Oscillator', 'META')  
('Chande Momentum Oscillator', 'GOOGL')  
('Chande Momentum Oscillator', 'Benchmark')  
('Ichimoku Conversion Line', 'AAPL')  
('Ichimoku Conversion Line', 'AMZN')  
('Ichimoku Conversion Line', 'Benchmark')  
('Ichimoku Conversion Line', 'GOOGL')  
('Ichimoku Conversion Line', 'META')  
('Ichimoku Conversion Line', 'MSFT')  
('Ichimoku Conversion Line', 'NVDA')  
('Ichimoku Conversion Line', 'PLTR')  
('Ichimoku Base Line', 'AAPL')  
('Ichimoku Base Line', 'AMZN')  
('Ichimoku Base Line', 'Benchmark')  
('Ichimoku Base Line', 'GOOGL')  
('Ichimoku Base Line', 'META')  
('Ichimoku Base Line', 'MSFT')  
('Ichimoku Base Line', 'NVDA')  
('Ichimoku Base Line', 'PLTR')  
('Ichimoku Leading Span A', 'AAPL')  
('Ichimoku Leading Span A', 'AMZN')  
('Ichimoku Leading Span A', 'Benchmark')  
('Ichimoku Leading Span A', 'GOOGL')  
('Ichimoku Leading Span A', 'META')  
('Ichimoku Leading Span A', 'MSFT')  
('Ichimoku Leading Span A', 'NVDA')  
('Ichimoku Leading Span A', 'PLTR')  
('Ichimoku Leading Span B', 'AAPL')  
('Ichimoku Leading Span B', 'AMZN')  
('Ichimoku Leading Span B', 'Benchmark')  
('Ichimoku Leading Span B', 'GOOGL')  
('Ichimoku Leading Span B', 'META')  
('Ichimoku Leading Span B', 'MSFT')  
('Ichimoku Leading Span B', 'NVDA')  
('Ichimoku Leading Span B', 'PLTR')  
('Stochastic %K', 'AAPL')  
('Stochastic %K', 'AMZN')  
('Stochastic %K', 'Benchmark')  
('Stochastic %K', 'GOOGL')  
('Stochastic %K', 'META')  
('Stochastic %K', 'MSFT')  
('Stochastic %K', 'NVDA')  
('Stochastic %K', 'PLTR')  
('Stochastic %D', 'AAPL')  
('Stochastic %D', 'AMZN')  
('Stochastic %D', 'Benchmark')  
('Stochastic %D', 'GOOGL')  
('Stochastic %D', 'META')  
('Stochastic %D', 'MSFT')  
('Stochastic %D', 'NVDA')  
('Stochastic %D', 'PLTR')  
('MACD Line', 'AAPL')  
('MACD Line', 'AMZN')  
('MACD Line', 'Benchmark')  
('MACD Line', 'GOOGL')  
('MACD Line', 'META')  
('MACD Line', 'MSFT')  
('MACD Line', 'NVDA')  
('MACD Line', 'PLTR')  
('MACD Signal Line', 'AAPL')  
('MACD Signal Line', 'AMZN')  
('MACD Signal Line', 'Benchmark')  
('MACD Signal Line', 'GOOGL')  
('MACD Signal Line', 'META')  
('MACD Signal Line', 'MSFT')  
('MACD Signal Line', 'NVDA')  
('MACD Signal Line', 'PLTR')  
('Relative Strength Index', 'NVDA')  
('Relative Strength Index', 'PLTR')  
('Relative Strength Index', 'AMZN')  
('Relative Strength Index', 'MSFT')  
('Relative Strength Index', 'AAPL')  
('Relative Strength Index', 'META')  
('Relative Strength Index', 'GOOGL')  
('Relative Strength Index', 'Benchmark')  
('Balance of Power', 'NVDA')  
('Balance of Power', 'PLTR')  
('Balance of Power', 'AMZN')  
('Balance of Power', 'MSFT')  
('Balance of Power', 'AAPL')  
('Balance of Power', 'META')  
('Balance of Power', 'GOOGL')  
('Balance of Power', 'Benchmark')  
('Simple Moving Average (SMA)', 'NVDA')  
('Simple Moving Average (SMA)', 'PLTR')  
('Simple Moving Average (SMA)', 'AMZN')  
('Simple Moving Average (SMA)', 'MSFT')  
('Simple Moving Average (SMA)', 'AAPL')  
('Simple Moving Average (SMA)', 'META')  
('Simple Moving Average (SMA)', 'GOOGL')  
('Simple Moving Average (SMA)', 'Benchmark')  
('Exponential Moving Average (EMA)', 'NVDA')  
('Exponential Moving Average (EMA)', 'PLTR')  
('Exponential Moving Average (EMA)', 'AMZN')  
('Exponential Moving Average (EMA)', 'MSFT')  
('Exponential Moving Average (EMA)', 'AAPL')  
('Exponential Moving Average (EMA)', 'META')  
('Exponential Moving Average (EMA)', 'GOOGL')  
('Exponential Moving Average (EMA)', 'Benchmark')  
('Double Exponential Moving Average (DEMA)', 'NVDA')  
('Double Exponential Moving Average (DEMA)', 'PLTR')  
('Double Exponential Moving Average (DEMA)', 'AMZN')  
('Double Exponential Moving Average (DEMA)', 'MSFT')  
('Double Exponential Moving Average (DEMA)', 'AAPL')  
('Double Exponential Moving Average (DEMA)', 'META')  
('Double Exponential Moving Average (DEMA)', 'GOOGL')  
('Double Exponential Moving Average (DEMA)', 'Benchmark')  
('TRIX', 'NVDA')  
('TRIX', 'PLTR')  
('TRIX', 'AMZN')  
('TRIX', 'MSFT')  
('TRIX', 'AAPL')  
('TRIX', 'META')  
('TRIX', 'GOOGL')  
('TRIX', 'Benchmark')  
('Triangular Moving Average', 'NVDA')  
('Triangular Moving Average', 'PLTR')  
('Triangular Moving Average', 'AMZN')  
('Triangular Moving Average', 'MSFT')  
('Triangular Moving Average', 'AAPL')  
('Triangular Moving Average', 'META')  
('Triangular Moving Average', 'GOOGL')  
('Triangular Moving Average', 'Benchmark')  
('Bollinger Band Upper', 'AAPL')  
('Bollinger Band Upper', 'AMZN')  
('Bollinger Band Upper', 'Benchmark')  
('Bollinger Band Upper', 'GOOGL')  
('Bollinger Band Upper', 'META')  
('Bollinger Band Upper', 'MSFT')  
('Bollinger Band Upper', 'NVDA')  
('Bollinger Band Upper', 'PLTR')  
('Bollinger Band Middle', 'AAPL')  
('Bollinger Band Middle', 'AMZN')  
('Bollinger Band Middle', 'Benchmark')  
('Bollinger Band Middle', 'GOOGL')  
('Bollinger Band Middle', 'META')  
('Bollinger Band Middle', 'MSFT')  
('Bollinger Band Middle', 'NVDA')  
('Bollinger Band Middle', 'PLTR')  
('Bollinger Band Lower', 'AAPL')  
('Bollinger Band Lower', 'AMZN')  
('Bollinger Band Lower', 'Benchmark')  
('Bollinger Band Lower', 'GOOGL')  
('Bollinger Band Lower', 'META')  
('Bollinger Band Lower', 'MSFT')  
('Bollinger Band Lower', 'NVDA')  
('Bollinger Band Lower', 'PLTR')  
('True Range', 'NVDA')  
('True Range', 'PLTR')  
('True Range', 'AMZN')  
('True Range', 'MSFT')  
('True Range', 'AAPL')  
('True Range', 'META')  
('True Range', 'GOOGL')  
('True Range', 'Benchmark')  
('Average True Range', 'NVDA')  
('Average True Range', 'PLTR')  
('Average True Range', 'AMZN')  
('Average True Range', 'MSFT')  
('Average True Range', 'AAPL')  
('Average True Range', 'META')  
('Average True Range', 'GOOGL')  
('Average True Range', 'Benchmark')  
('Keltner Channel Upper', 'AAPL')  
('Keltner Channel Upper', 'AMZN')  
('Keltner Channel Upper', 'Benchmark')  
('Keltner Channel Upper', 'GOOGL')  
('Keltner Channel Upper', 'META')  
('Keltner Channel Upper', 'MSFT')  
('Keltner Channel Upper', 'NVDA')  
('Keltner Channel Upper', 'PLTR')  
('Keltner Channel Middle', 'AAPL')  
('Keltner Channel Middle', 'AMZN')  
('Keltner Channel Middle', 'Benchmark')  
('Keltner Channel Middle', 'GOOGL')  
('Keltner Channel Middle', 'META')  
('Keltner Channel Middle', 'MSFT')  
('Keltner Channel Middle', 'NVDA')  
('Keltner Channel Middle', 'PLTR')  
('Keltner Channel Lower', 'AAPL')  
('Keltner Channel Lower', 'AMZN')  
('Keltner Channel Lower', 'Benchmark')  
('Keltner Channel Lower', 'GOOGL')  
('Keltner Channel Lower', 'META')  
('Keltner Channel Lower', 'MSFT')  
('Keltner Channel Lower', 'NVDA')  
('Keltner Channel Lower', 'PLTR')

## Contacts

-   [Website](https://newdigitals.org/)
-   [GitHub](https://github.com/alva922)
-   [Facebook](https://www.facebook.com/profile.php?id=100076281754699)
-   [X/Twitter](https://twitter.com/alzapress)
-   [Pinterest](https://nl.pinterest.com/alexzap922/)
-   [Mastodon](https://mastodon.social/@alexzap)
-   [Tumblr](https://alva922.tumblr.com/)

## Disclaimer

-   I declare that this article is written by me and not with any generative AI tool such as ChatGPT.
-   I declare that no data privacy policy is breached, and that any data associated with the contents here are obtained legitimately to the best of my knowledge.
-   The following disclaimer clarifies that the information provided in this article is for educational use only and should not be considered financial or investment advice.
-   The information provided does not take into account your individual financial situation, objectives, or risk tolerance.
-   Any investment decisions or actions you undertake are solely your responsibility.
-   You should independently evaluate the suitability of any investment based on your financial objectives, risk tolerance, and investment timeframe.
-   It is recommended to seek advice from a certified financial professional who can provide personalized guidance tailored to your specific needs.
-   The tools, data, content, and information offered are impersonal and not customized to meet the investment needs of any individual. As such, the tools, data, content, and information are provided solely for informational and educational purposes only.
-   _All images unless otherwise noted are by the author._

### A Message from InsiderFinance

![](https://miro.medium.com/v2/resize:fit:301/0*10x5_2smmKq8oIlf.png)

Thanks for being a part of our community! Before you go:

-   👏 Clap for the story and follow the author 👉
-   📰 View more content in the [InsiderFinance Wire](https://wire.insiderfinance.io/)
-   📚 Take our [FREE Masterclass](https://learn.insiderfinance.io/p/mastering-the-flow)
-   **📈 Discover** [**Powerful Trading Tools**](https://insiderfinance.io/?utm_source=wire&utm_medium=message)

## Embedded Content