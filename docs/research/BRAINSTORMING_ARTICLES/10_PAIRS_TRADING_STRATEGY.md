# Why Pairs Trading Strategy So Popular within (Quant) Financial Circles? | by Alexzap | InsiderFinance Wire

Member-only story

# Why Pairs Trading Strategy So Popular within (Quant) Financial Circles?

## Unleash the Power of Statistical Arbitrage, Mean Reversion, Sector Rotation, Options Trading, and Long-Short Equity (with SWOT Analysis and Python Code Examples)

[

![Alexzap](https://miro.medium.com/v2/resize:fill:48:48/1*L1sRpfwSPETNMy1qzGUGBg.jpeg)





](/@alexzap922?source=post_page---byline--5bc5a3637d0b---------------------------------------)

[Alexzap](/@alexzap922?source=post_page---byline--5bc5a3637d0b---------------------------------------)

Following

21 min read

·

May 24, 2025

251

4

Listen

Share

More

[“There is a time to go long, a time to go short and a time to go fishing.” — Jesse Livermore](https://fbs.com/fbs-academy/traders-blog/20-trading-quotes-that-will-change-your-trading)

![Canva Design Template Customized by the Author.](https://miro.medium.com/v2/resize:fit:1050/1*xRBxGfHr4jcMc3weeHXkVQ.png)
Canva Design Template Customized by the Author.

-   The [pairs trading strategy](https://daehkim.github.io/pair-trading/) relies on monitoring the correlation between a pair of highly correlated stocks. Specifically, a _long position_ is opened on the stock that rises and a _short position_ is opened on the stock that falls.
-   In this post, we’ll discuss essentials and gain hands-on experience in using the key 5 components of the pairs trading strategy, i.e. statistical arbitrage, mean reversion, sector rotation, options trading, and long-short equity (including SWOT analysis and working Python code examples).
-   Let’s delve into the specifics of this celebrated trading strategy!

## Contents

-   FAANG Correlation Matchups
-   Spread-ADX-RSI Pairs Trading Strategy
-   Pairs Trading Long-Short Equity ETFs
-   Sector Rotation Strategy
-   Statistical Arbitrage Model
-   Pairs Trading with Options
-   Visualizing Greeks in Finance Toolkit
-   Pairs Trading-as-a-Service (PTaaS)

## FAANG Correlation Matchups

-   In [pairs trading](https://daehkim.github.io/pair-trading/), identification of correlated stocks and generation of pairs is of paramount importance.
-   In [correlation analysis](/@junjunzaragosa2309/stocks-market-index-correlation-analysis-530e81f33aba) between stock prices, we aim to quantify the degree to which their price movements are related. Over a given time period, the two securities move together when the [Correlation Coefficient](https://www.macroaxis.com/invest/marketCorrelation) (CC) is positive. Conversely, the two assets move in opposite directions when CC<0.
-   The stock correlation analysis primarily involves \[1\]:

1.  Calculating [daily returns as percentage](https://www.investopedia.com/terms/r/rateofreturn.asp) aka Rate of Return (RoR);
2.  Calculating the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
3.  Backtesting the stock pair with max CC.
4.  Examining the portfolio/stock volatility.

-   Importing the necessary Python libraries and using [twelvedata API](https://twelvedata.com/) to download the FAANG historical data 2020–2025

import requests  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from termcolor import colored as cl  
from math import floor  
  
plt.rcParams\['figure.figsize'\] = (20,10)  
plt.style.use('fivethirtyeight')  
  
\# EXTRACTING STOCK DATA  
  
def get\_historical\_data(symbol, start\_date, end\_date):  
    api\_key = 'YOUR API KEY'  
    api\_url = f'https://api.twelvedata.com/time\_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api\_key}'  
    raw\_df = requests.get(api\_url).json()  
    df = pd.DataFrame(raw\_df\['values'\]).iloc\[::-1\].set\_index('datetime').astype(float)  
    df = df\[df.index >= start\_date\]  
    df = df\[df.index <= end\_date\]  
    df.index = pd.to\_datetime(df.index)  
    return df  
  
fb = get\_historical\_data('META', '2020-01-01', '2025-05-20')  
amzn = get\_historical\_data('AMZN', '2020-01-01', '2025-05-20')  
aapl = get\_historical\_data('AAPL', '2020-01-01', '2025-05-20')  
nflx = get\_historical\_data('NFLX', '2020-01-01', '2025-05-20')  
googl = get\_historical\_data('GOOGL', '2020-01-01', '2025-05-20')

-   Calculating the FAANG cumulative returns

b\_rets, fb\_rets.name = fb\['close'\] / fb\['close'\].iloc\[0\], 'fb'  
amzn\_rets, amzn\_rets.name = amzn\['close'\] / amzn\['close'\].iloc\[0\], 'amzn'  
aapl\_rets, aapl\_rets.name = aapl\['close'\] / aapl\['close'\].iloc\[0\], 'aapl'  
nflx\_rets, nflx\_rets.name = nflx\['close'\] / nflx\['close'\].iloc\[0\], 'nflx'  
googl\_rets, googl\_rets.name = googl\['close'\] / googl\['close'\].iloc\[0\], 'googl'  
  
plt.plot(fb\_rets, label = 'META')  
plt.plot(amzn\_rets, label = 'AMZN')  
plt.plot(aapl\_rets, label = 'AAPL')  
plt.plot(nflx\_rets, label = 'NFLX')  
plt.plot(googl\_rets, label = 'GOOGL', color = 'purple')  
plt.legend(fontsize = 16)  
plt.title('FAANG CUMULATIVE RETURNS')  
plt.show()

![FAANG CUMULATIVE RETURNS](https://miro.medium.com/v2/resize:fit:1050/1*UCYYN2T3LqFaZMQxSI7mNg.png)
FAANG CUMULATIVE RETURNS

-   Creating the FAANG correlation matrix

rets = \[fb\_rets, amzn\_rets, aapl\_rets, nflx\_rets, googl\_rets\]  
rets\_df = pd.DataFrame(rets).T.dropna()  
rets\_corr = rets\_df.corr()  
  
plt.style.use('default')  
sns.heatmap(rets\_corr, annot = True, linewidths = 0.5)  
plt.show()

![FAANG Correlation Matrix](https://miro.medium.com/v2/resize:fit:792/1*TVPHmoT_mo3p9Q_caBwEhQ.png)
FAANG Correlation Matrix

-   Backtesting the META-NFLX stock pair with max CC = 0.91

investment\_value = 100000  
N = 2  
nflx\_allocation = investment\_value / N  
fb\_allocation = investment\_value / N  
  
nflx\_stocks = floor(nflx\_allocation / nflx\['close'\].iloc\[0\])  
fb\_stocks = floor(fb\_allocation / fb\['close'\].iloc\[0\])  
  
nflx\_investment\_rets = nflx\_rets \* nflx\_stocks  
fb\_investment\_rets = fb\_rets \* fb\_stocks  
total\_rets = round(sum(((nflx\_investment\_rets + fb\_investment\_rets) / 2).dropna()), 3)  
total\_rets\_pct = round((total\_rets / investment\_value) \* 100, 3)  
  
print(cl(f'Profit gained from the investment : {total\_rets} USD', attrs = \['bold'\]))  
print(cl(f'Profit percentage of our investment : {total\_rets\_pct}%', attrs = \['bold'\]))  
  
Profit gained from the investment : 405317.809 USD  
Profit percentage of our investment : 405.318%

-   Examining the portfolio vs FAANG volatility

rets\_df\['Portfolio'\] = (rets\_df\[\['fb', 'nflx'\]\].sum(axis = 1)) / 2  
daily\_pct\_change = rets\_df.pct\_change()  
volatility = round(np.log(daily\_pct\_change + 1).std() \* np.sqrt(252), 5)  
  
companies = \['META', 'APPLE', 'AMAZON', 'NFLX', 'GOOGL', 'PORTFOLIO'\]  
for i in range(len(volatility)):  
    if i == 5:  
        print(cl(f'{companies\[i\]} VOLATILITY : {volatility\[i\]}', attrs = \['bold'\], color = 'green'))  
    else:  
        print(cl(f'{companies\[i\]} VOLATILITY : {volatility\[i\]}', attrs = \['bold'\]))  
  
META VOLATILITY : 0.45491  
APPLE VOLATILITY : 0.36358  
AMAZON VOLATILITY : 0.32734  
NFLX VOLATILITY : 0.46442  
GOOGL VOLATILITY : 0.32953  
PORTFOLIO VOLATILITY : 0.39321  
  
import matplotlib.pyplot as plt  
import numpy as np  
  
fru = companies  
sal = volatility  
  
plt.bar(fru, sal)  
plt.title('Asset Volatility')  
plt.xlabel('Assets')  
plt.ylabel('Volatility')  
plt.grid()  
plt.show()

![Asset Volatility](https://miro.medium.com/v2/resize:fit:873/1*pGOIJPAGyX4iqjLQj5cYKQ.png)
Asset Volatility

-   We can see that Vol(Portfolio) < Vol(META) ~Vol(NFLX), where Vol means Volatility.

## Spread-ADX-RSI Pairs Trading Strategy

-   Let’s evaluate profitability of the NVDA-IBM pairs trading strategy using the price spread and the trend strength and momentum technical indicators such as ADX and RSI \[2\].
-   Importing and installing the necessary Python libraries and using the aforementioned function get\_historical\_data to download the NVDA and IBM historical data

!pip install ta  
  
import pandas as pd  
import requests  
import math  
import numpy as np  
from termcolor import colored as cl  
import matplotlib.pyplot as plt  
from ta.momentum import RSIIndicator  
from ta.trend import ADXIndicator  
  
plt.rcParams\['figure.figsize'\] = (20,10)  
plt.style.use('fivethirtyeight')  
  
nvda = get\_historical\_data('NVDA', '2020-01-01', '2025-05-20')  
pltr = get\_historical\_data('IBM', '2020-01-01', '2025-05-20')  
  
\# MERGING HISTORICAL DATA  
  
df = pd.DataFrame(columns = \['nvda','ibm'\])  
df.nvda = nvda.close  
df.ibm = pltr.close  
df.index = nvda.index  
  
df.tail(10)  
  
            nvda       ibm  
datetime    
2025\-05-07 117.059998 253.37000  
2025\-05-08 117.370000 254.14000  
2025\-05-09 116.650000 249.20000  
2025\-05-12 123.000000 253.69000  
2025\-05-13 129.929990 258.59000  
2025\-05-14 135.340000 257.82001  
2025\-05-15 134.830000 266.67999  
2025\-05-16 135.399990 266.76001  
2025\-05-19 135.570010 268.41000  
2025\-05-20 134.390000 266.98500

-   Calculating the price spread and ADX/RSI technical indicators

beta = np.polyfit(df.ibm, df.nvda, 1)\[0\]  
spread = df.nvda - beta \* df.ibm  
  
df\['spread'\] = spread  
df\['adx'\] = ADXIndicator(high=spread, low=spread, close=spread, window=14).adx()  
df\['rsi'\] = RSIIndicator(close=spread, window=14).rsi()  
  
df = df.dropna()  
df.tail()  
  
           nvda      ibm        spread     adx      rsi  
datetime       
2025\-05-14 135.34000 257.82001 -125.627304 9.796106 59.575127  
2025\-05-15 134.83000 266.67999 -135.105440 9.210861 49.198663  
2025\-05-16 135.39999 266.76001 -134.616447 8.597862 49.685565  
2025\-05-19 135.57001 268.41000 -136.116559 8.246502 48.160587  
2025\-05-20 134.39000 266.98500 -135.854174 7.877671 48.458569  
  
df\['adx'\] = ADXIndicator(high=df\['spread'\], low=df\['spread'\], close=df\['spread'\], window=14).adx()

-   Backtesting the strategy

def implement\_pairs\_trading\_strategy(df, investment):  
    in\_position = False  
    equity = investment  
    nvda\_shares = 0  
    amd\_shares = 0  
  
    for i in range(1, len(df)):  
        \# Enter the market (Buy NVDA, Sell IBM) if RSI < 30 and ADX > 25  
        if df\['rsi'\]\[i\] < 30 and 20 < df\['adx'\]\[i\] < 25 and not in\_position:  
            \# Allocate 50% of equity for buying NVDA and 50% for shorting IBM  
            nvda\_allocation = equity \* 0.5  
            amd\_allocation = equity \* 0.5  
  
            nvda\_shares = math.floor(nvda\_allocation / df\['nvda'\]\[i\])  
            equity -= nvda\_shares \* df\['nvda'\]\[i\]  
  
            amd\_shares = math.floor(amd\_allocation / df\['ibm'\]\[i\])  
            equity += amd\_shares \* df\['ibm'\]\[i\]  \# Shorting IBM adds to equity  
  
            in\_position = True  
            print(cl('ENTER MARKET:', color='green', attrs=\['bold'\]),  
                  f'Bought {nvda\_shares} NVDA shares at ${df\["nvda"\]\[i\]}, '  
                  f'Sold {amd\_shares} IBM shares at ${df\["ibm"\]\[i\]} on {df.index\[i\]}')  
  
        \# Exit the market (Sell NVDA, Buy IBM) if RSI > 70 and ADX > 25  
        elif df\['rsi'\]\[i\] > 70 and 20 < df\['adx'\]\[i\] < 25 and in\_position:  
            equity += nvda\_shares \* df\['nvda'\]\[i\]  \# Selling NVDA adds to equity  
            nvda\_shares = 0  
  
            equity -= amd\_shares \* df\['ibm'\]\[i\]  \# Buying IBM subtracts from equity  
            amd\_shares = 0  
  
            in\_position = False  
            print(cl('EXIT MARKET:', color='red', attrs=\['bold'\]),  
                  f'Sold NVDA and Bought IBM on {df.index\[i\]} at IBM=${df\["nvda"\]\[i\]}, IBM=${df\["ibm"\]\[i\]}')  
  
    \# Closing any remaining positions at the end  
    if in\_position:  
        equity += nvda\_shares \* df\['nvda'\].iloc\[-1\]  
        equity -= amd\_shares \* df\['ibm'\].iloc\[-1\]  
        print(cl(f'\\nClosing positions at NVDA=${df\["nvda"\].iloc\[-1\]}, '  
                 f'IBM=${df\["ibm"\].iloc\[-1\]} on {df.index\[-1\]}', attrs=\['bold'\]))  
      
    \# Calculating earnings and ROI  
    earning = round(equity - investment, 2)  
    roi = round((earning / investment) \* 100, 2)  
  
    print('')  
    print(cl('PAIRS TRADING BACKTESTING RESULTS:', attrs=\['bold'\]))  
    print(cl(f'EARNING: ${earning} ; ROI: {roi}%', attrs=\['bold'\]))  
  
investment = 100000  
implement\_pairs\_trading\_strategy(df, investment)  
  
ENTER MARKET: Bought 3894 NVDA shares at $12.83925, Sold 383 IBM shares at $130.38242 on 2021\-03-26 00:00:00  
EXIT MARKET: Sold NVDA and Bought IBM on 2021\-10\-25 00:00:00 at IBM=$23.166, IBM=$122.026772  
ENTER MARKET: Bought 3602 NVDA shares at $19.902, Sold 515 IBM shares at $139.10001 on 2022\-04-25 00:00:00  
EXIT MARKET: Sold NVDA and Bought IBM on 2023\-01-26 00:00:00 at IBM=$19.802, IBM=$134.45  
ENTER MARKET: Bought 1601 NVDA shares at $45.417, Sold 497 IBM shares at $146.17999 on 2023\-08-07 00:00:00  
EXIT MARKET: Sold NVDA and Bought IBM on 2024\-03-25 00:00:00 at IBM=$95.0019989, IBM=$188.78999  
ENTER MARKET: Bought 981 NVDA shares at $103.73, Sold 533 IBM shares at $191.039993 on 2024\-07-30 00:00:00  
EXIT MARKET: Sold NVDA and Bought IBM on 2024\-10\-29 00:00:00 at IBM=$141.25, IBM=$210.42999  
ENTER MARKET: Bought 986 NVDA shares at $116.66, Sold 441 IBM shares at $260.73001 on 2025\-02-03 00:00:00  
  
Closing positions at NVDA=$134.39, IBM=$266.985 on 2025\-05-20 00:00:00  
  
PAIRS TRADING BACKTESTING RESULTS:  
EARNING: $144851.13 ; ROI: 144.85%

## Pairs Trading Long-Short Equity ETFs

-   In this section, we’ll generate and backtest pairs trading long-short signals by focusing on the ETF pairs such as SPY & QQQ \[3\].
-   Downloading the ETF Close prices

import yfinance as yf  
import pandas as pd  
  
\# Define the ticker symbols  
tickers = \['SPY', 'QQQ'\]  
  
\# Fetch the data  
data = yf.download(tickers, start='2020-01-01')  
  
\# Select Close prices  
closing\_prices = data\['Close'\]  
  
closing\_prices.tail()  
  
Ticker     QQQ        SPY  
Date    
2025\-05-16 521.510010 594.200012  
2025\-05-19 522.010010 594.849976  
2025\-05-20 520.270020 592.849976  
2025\-05-21 513.039978 582.859985  
2025\-05-22 514.000000 583.090027

-   Calculating the SPY-QQQ spread over time and the correlation coefficient

import matplotlib.pyplot as plt  
  
\# Calculate the spread  
closing\_prices\_clean\['Spread'\] = closing\_prices\_clean\['SPY'\] - closing\_prices\_clean\['QQQ'\]  
  
\# Plot the spread  
plt.figure(figsize=(12,6))  
plt.plot(closing\_prices\_clean.index, closing\_prices\_clean\['Spread'\])  
plt.title('Spread over time')  
plt.ylabel('Spread')  
plt.xlabel('Date')  
  
plt.show()  
  
\# Calculate correlation  
correlation = closing\_prices\_clean\['SPY'\].corr(closing\_prices\_clean\['QQQ'\])  
print(f"\\nCorrelation between SPY and QQQ: {correlation}")  
  
Correlation between SPY and QQQ: 0.984235404773848

![SPY-QQQ spread over time](https://miro.medium.com/v2/resize:fit:1050/1*90Lo___vnJFlyjM002YvBA.png)
SPY-QQQ spread over time

-   Calculating the Z-score of the spread

\# Calculate the z-score of the spread  
closing\_prices\_clean\['Z-Score'\] = (closing\_prices\_clean\['Spread'\] - closing\_prices\_clean\['Spread'\].mean()) / closing\_prices\_clean\['Spread'\].std()

-   Generating trading signals based on the Z-score with threshold = 1

\# Set the z-score threshold  
threshold = 1  
  
\# Create columns for long and short signals  
closing\_prices\_clean\['Long\_Entry'\] = 0  
closing\_prices\_clean\['Short\_Entry'\] = 0  
closing\_prices\_clean\['Long\_Exit'\] = 0  
closing\_prices\_clean\['Short\_Exit'\] = 0  
  
\# Generate trading signals based on z-score  
closing\_prices\_clean.loc\[closing\_prices\_clean\['Z-Score'\] <= -threshold, 'Long\_Entry'\] = 1  
closing\_prices\_clean.loc\[closing\_prices\_clean\['Z-Score'\] >= threshold, 'Short\_Entry'\] = 1  
closing\_prices\_clean.loc\[closing\_prices\_clean\['Z-Score'\] \* closing\_prices\_clean\['Z-Score'\].shift(1) < 0, 'Long\_Exit'\] = 1  
closing\_prices\_clean.loc\[closing\_prices\_clean\['Z-Score'\] \* closing\_prices\_clean\['Z-Score'\].shift(1) < 0, 'Short\_Exit'\] = 1  
  
#Plotting   
plt.figure(figsize=(12,6))  
closing\_prices\_clean\['Z-Score'\].plot()  
plt.axhline(threshold, color='red', linestyle='--')  
plt.axhline(-threshold, color='red', linestyle='--')  
plt.ylabel('Z-Score')  
plt.title('Z-Score with Threshold=+/-1')  
#plt.grid()  
plt.show()

![Z-Score with Threshold=+/-1](https://miro.medium.com/v2/resize:fit:1050/1*K8BPmnM0Sx91ODqK01bS9Q.png)
Z-Score with Threshold=+/-1

-   Plotting the buy/sell trading signals

import matplotlib.pyplot as plt  
  
plt.figure(figsize=(15,7))  
  
\# Plot spread  
closing\_prices\_clean\['Spread'\].plot(label='Spread', color='b')  
  
\# Plot buy signals  
buy\_signals = closing\_prices\_clean\['Spread'\]\[closing\_prices\_clean\['Long\_Entry'\] == 1\]  
sell\_signals = closing\_prices\_clean\['Spread'\]\[closing\_prices\_clean\['Short\_Entry'\] == 1\]  
plt.plot(buy\_signals, color='g', linestyle='None', marker='^', markersize=5, label='Buy Signal')  
plt.plot(sell\_signals, color='r', linestyle='None', marker='v', markersize=5, label='Sell Signal')  
  
\# Customize and show the plot  
plt.legend()  
  
plt.show()

![Buy/sell trading signals vs spread.](https://miro.medium.com/v2/resize:fit:1050/1*PqYXdgFxJK_sr2gyvb8C0A.png)
Buy/sell trading signals vs spread.

-   Backtesting the trading strategy

import pandas as pd  
from scipy.stats import zscore  
import matplotlib.pyplot as plt  
import seaborn as sns  
  
def backtest(closing\_prices\_clean):  
    \# Prepare an empty DataFrame for results  
    results = pd.DataFrame(index=closing\_prices\_clean.index)  
      
    \# Calculate zscore  
    results\['zscore'\] = zscore(closing\_prices\_clean\['Spread'\])  
      
    \# Initiate values  
    results\['Spread'\] = closing\_prices\_clean\['Spread'\]  
    results\['Long\_Entry'\] = closing\_prices\_clean\['Long\_Entry'\]  
    results\['Short\_Entry'\] = closing\_prices\_clean\['Short\_Entry'\]  
    results\['Long\_Exit'\] = closing\_prices\_clean\['Long\_Exit'\]  
    results\['Short\_Exit'\] = closing\_prices\_clean\['Short\_Exit'\]  
    results\['Returns'\] = 0  
    results\['Profit'\] = 0  
    position = 0  
    profit = 0  
      
    for i in range(1, len(closing\_prices\_clean)):  
        \# Check if we have a long or short position  
        if position == 1:  
            results.loc\[results.index\[i\], 'Returns'\] = closing\_prices\_clean.iloc\[i\]\['Spread'\] - closing\_prices\_clean.iloc\[i-1\]\['Spread'\]  
            profit += results.loc\[results.index\[i\], 'Returns'\]  
            if closing\_prices\_clean.iloc\[i\]\['Long\_Exit'\] == 1:  
                position = 0  
        elif position == -1:  
            results.loc\[results.index\[i\], 'Returns'\] = closing\_prices\_clean.iloc\[i-1\]\['Spread'\] - closing\_prices\_clean.iloc\[i\]\['Spread'\]  
            profit += results.loc\[results.index\[i\], 'Returns'\]  
            if closing\_prices\_clean.iloc\[i\]\['Short\_Exit'\] == 1:  
                position = 0  
  
        \# Update profit   
        results.loc\[results.index\[i\], 'Profit'\] = profit  
  
        \# Check if we should enter a position  
        if position == 0:  
            if closing\_prices\_clean.iloc\[i\]\['Long\_Entry'\] == 1:  
                position = 1  
            elif closing\_prices\_clean.iloc\[i\]\['Short\_Entry'\] == 1:  
                position = -1  
  
    return results  
  
\# Run the backtest  
results = backtest(closing\_prices\_clean)  
  
\# Print the total returns  
total\_returns = results\['Returns'\].sum()  
print(f"Total returns: {total\_returns \* 100:.2f}%")  
  
\# Save the results to a CSV file  
results.to\_csv('results.csv')  
  
\# Set the style of seaborn for better looking plots  
sns.set(style="whitegrid")  
  
\# Create a figure and a set of subplots  
fig, axs = plt.subplots(2,figsize=(12, 10))  
  
\# Plot the Spread, Long\_Entry, and Short\_Entry  
axs\[0\].plot(results.index, results\['Spread'\], color='blue', label='Spread')  
axs\[0\].plot(results\[results\['Long\_Entry'\] == 1\].index, results\[results\['Long\_Entry'\] == 1\]\['Spread'\], 'g^', label='Long Entry')  
axs\[0\].plot(results\[results\['Short\_Entry'\] == 1\].index, results\[results\['Short\_Entry'\] == 1\]\['Spread'\], 'rv', label='Short Entry')  
axs\[0\].set\_title('Spread and Entry Points')  
axs\[0\].set\_xlabel('Date')  
axs\[0\].set\_ylabel('Spread')  
axs\[0\].legend()  
  
\# Plot the Profit  
axs\[1\].plot(results.index, results\['Profit'\], color='purple', label='Profit')  
axs\[1\].set\_title('Profit Over Time')  
axs\[1\].set\_xlabel('Date')  
axs\[1\].set\_ylabel('Profit')  
axs\[1\].legend()  
  
\# Automatically adjust subplot params so that the subplot fits in to the figure area  
plt.tight\_layout()  
  
\# Display the figure  
plt.show()

![Backtesting the trading strategy](https://miro.medium.com/v2/resize:fit:1050/1*bXJXdy0Z9YzdtnWH3Yq8iw.png)
Backtesting the trading strategy

#Result of backtesting  
  
Total returns: 13851.85%

## Sector Rotation Strategy

-   This section illustrates the sector rotation strategy \[5\] using 19 different sectors in India’s BSE.
-   Reading the input data and defining the necessary functions \[5\]

import pandas as pd  
import datetime as dt  
import numpy as np  
  
index\_code = pd.read\_excel('indices.xlsx', engine='openpyxl')  
indices = pd.read\_csv('indices\_price.csv', header = 0, index\_col = 0)  
indices = indices.drop('BSE500', axis=1)  
  
def buy\_sell\_dates(dates, period):  
    '''  
    Calculates buy and sell dates based on defined period  
    '''  
    buy\_dates = \[\]  
    sell\_dates = \[\]  
    for i in range(0, len(dates), period):  
        buy\_dates.append(dates\[i\])  
    for i in range(period, len(dates), period):  
        sell\_dates.append(dates\[i\])  
    if(len(buy\_dates)>len(sell\_dates)):  
        del buy\_dates\[-1\]  
    return buy\_dates, sell\_dates  
  
def highlight\_max(data, color='yellow'):  
    '''  
    Highlights the maximum value in a Series or DataFrame in yellow  
    '''  
    attr = 'background-color: {}'.format(color)  
    if data.ndim == 1:  \# Series from .apply(axis=0) or axis=1  
        is\_max = data == data.max()  
        return \[attr if v else '' for v in is\_max\]  
    else:  \# from .apply(axis=None)  
        is\_max = data == data.max().max()  
        return pd.DataFrame(np.where(is\_max, attr, ''),  
                            index=data.index, columns=data.columns)  
      
def calculate\_trade\_profits(period\_returns, buy\_dates, sell\_dates):  
    '''  
    With buy and sell dates as inputs, this function looks up the period returns  
    It then computes the trade return of doing buy and sell trades  
    It creates a pandast dataframe with trade profits by trade number and   
    number of sectors. The sectors automatically vary from 1 sector to everything.  
    '''  
    trade\_profits = np.zeros((len(buy\_dates), period\_returns.shape\[1\]))  
    for a in range(len(buy\_dates)):  
        returns = period\_returns.T\[buy\_dates\[a\]\].sort\_values(ascending=False)  
        for i in range(len(returns)):  
            sector\_list = returns.index\[:i+1\]              
            trade\_profit = period\_returns.loc\[sell\_dates\[a\], sector\_list\].mean()   
            trade\_profits\[a, i\] = trade\_profit  
    columns = \[str(x+1)+' Sectors' for x in range(trade\_profits.shape\[1\])\]  
    index = \['Trade Number ' + str(x+1) for x in range(trade\_profits.shape\[0\])\]  
    return pd.DataFrame(trade\_profits, columns = columns, index = index)

-   Computing the 30, 60 and 90 days returns, sell dates, profits, and the rate of change

#Returns  
ninety\_rets = indices.pct\_change(90).dropna().loc\[:'23-October-2020',\]  
rets\_index = ninety\_rets.index.to\_list()  
thirty\_rets = indices.pct\_change(30).dropna().loc\[rets\_index\[0\]:'23-October-2020',\]  
sixty\_rets = indices.pct\_change(60).dropna().loc\[rets\_index\[0\]:'23-October-2020',\]  
#Sell dates  
thirty\_buy\_dates, thirty\_sell\_dates = buy\_sell\_dates(rets\_index, 30)  
sixty\_buy\_dates, sixty\_sell\_dates = buy\_sell\_dates(rets\_index, 60)  
ninety\_buy\_dates, ninety\_sell\_dates = buy\_sell\_dates(rets\_index, 90)  
#Profits  
thirty\_trade\_profits = calculate\_trade\_profits(thirty\_rets, thirty\_buy\_dates, thirty\_sell\_dates)  
sixty\_trade\_profits = calculate\_trade\_profits(sixty\_rets, sixty\_buy\_dates, sixty\_sell\_dates)  
ninety\_trade\_profits = calculate\_trade\_profits(ninety\_rets, ninety\_buy\_dates, ninety\_sell\_dates)  
#Rate of change  
results\_rateofchange = pd.DataFrame({'30 Days':((((1+thirty\_trade\_profits).astype('object').product())\*\*0.111)-1)\*100,  
             '60 Days':((((1+sixty\_trade\_profits).astype('object').product())\*\*0.111)-1)\*100,  
             '90 Days':((((1+ninety\_trade\_profits).astype('object').product())\*\*0.111)-1)\*100})

-   Computing and plotting the average rate of change for each strategy

results\_rateofchange.style.apply(highlight\_max)

![Average rate of change based on each strategy (highlighting max)](https://miro.medium.com/v2/resize:fit:458/1*xmvo8kPf72sEF-kCv_9DkQ.png)
Average rate of change for each strategy (highlighting max)

import matplotlib.pyplot as plt   
plt.figure(figsize=(10,6))  
plt.rcParams.update({'font.size': 20})  
results\_rateofchange.plot.bar(rot=45)  
plt.title('Rate of Change')  
plt.show()

![Bar Plot: Average rate of change based on each strategy](https://miro.medium.com/v2/resize:fit:1050/1*q3zVScjbMBFnySOplfmnkQ.png)
Bar Plot: Average rate of change for each strategy

-   Computing and plotting the percentage of profitable trades for each strategy

profitable\_trades = pd.DataFrame({'30 Days':thirty\_trade\_profits.gt(0).mean()\*100,  
             '60 Days':sixty\_trade\_profits.gt(0).mean()\*100,  
             '90 Days':ninety\_trade\_profits.gt(0).mean()\*100})  
  
profitable\_trades.style.apply(highlight\_max)

![Percentage of profitable trades for each strategy (highlighting max).](https://miro.medium.com/v2/resize:fit:437/1*9FNUu7AAnYipu0MYt_YHTw.png)
Percentage of profitable trades for each strategy (highlighting max).

import matplotlib.pyplot as plt   
plt.figure(figsize=(10,6))  
plt.rcParams.update({'font.size': 20})  
profitable\_trades.plot.bar(rot=45)  
plt.title(' Percentage of Profitable Trades for each Strategy')  
plt.show()

![Bar Plot: Percentage of profitable trades for each strategy](https://miro.medium.com/v2/resize:fit:1050/1*heUqzYYD2PKt2QVVZSbq4Q.png)
Bar Plot: Percentage of profitable trades for each strategy

-   Computing and plotting the Return/Risk ratio for each strategy

sixty\_sharpe = (results\_rateofchange\['60 Days'\]/100)/(sixty\_trade\_profits.std()\*np.sqrt(250/60))  
ninety\_sharpe = (results\_rateofchange\['90 Days'\]/100)/(sixty\_trade\_profits.std()\*np.sqrt(250/90))  
return\_to\_risk = pd.DataFrame({'30 Days':thirty\_sharpe,  
             '60 Days':sixty\_sharpe,  
             '90 Days':ninety\_sharpe})  
\# Return to Risk Ratio for each trading strategy  
return\_to\_risk.style.apply(highlight\_max)

![Return/Risk ratio for each strategy (highlighting max).](https://miro.medium.com/v2/resize:fit:410/1*dCT6cwuf_0e8TmZDHr4nCg.png)
Return/Risk ratio for each strategy (highlighting max).

import matplotlib.pyplot as plt   
plt.figure(figsize=(10,6))  
plt.rcParams.update({'font.size': 20})  
return\_to\_risk.plot.bar(rot=45)  
plt.title(' Return to Risk Ratio for each Trading Strategy')  
plt.show()

![Bar Plot: Return to Risk Ratio for each Trading Strategy](https://miro.medium.com/v2/resize:fit:1050/1*OAsr9eKEcA3V5gz8EEBtjg.png)
Bar Plot: Return to Risk Ratio for each Trading Strategy

-   Comparing the 30, 60 and 90 days returns for the specific date 25-March-2021 \[5\]

thirty\_rets.iloc\[-1:, \].T.sort\_values(by='25-March-2021', ascending=False)\[:3\]  
  
  
SIPOWE   0.100125  
SPBSUTIP 0.076847  
SPBSBMIP 0.073135

![30 days returns for the specific date 25-March-2021](https://miro.medium.com/v2/resize:fit:1050/1*gLnbxSlMbKKlwGO_NFh9tQ.png)
30 days returns for the specific date 25-March-2021

sixty\_rets.iloc\[-1:, \].T.sort\_values(by='25-March-2021', ascending=False)\[:3\]  
  
  
SPBSBMIP 0.213961  
SIPOWE   0.203544  
SPBSIDIP 0.186673

![60 days returns for the specific date 25-March-2021](https://miro.medium.com/v2/resize:fit:1050/1*vQxa3vedx5FEgzt251vzDQ.png)
60 days returns for the specific date 25-March-2021

ninety\_rets.iloc\[-1:, \].T.sort\_values(by='25-March-2021', ascending=False)\[:3\]  
  
SI1200   0.429335  
SPBSIDIP 0.396271  
SPBSBMIP 0.353286

![90 days returns for the specific date 25-March-2021](https://miro.medium.com/v2/resize:fit:1050/1*k-vzp-_mvL9Lleni3LVxcA.png)
90 days returns for the specific date 25-March-2021

-   Checking stock indexes vs codes

index\_code\[(index\_code.CODE=='SPBSBMIP')|(index\_code.CODE=='SIPOWE')|  
           (index\_code.CODE=='SPBSIDIP')|(index\_code.CODE=='SI1200')|  
           (index\_code.CODE=='SPBSUTIP')\]  
  
  INDEX                CODE  
3  BSE Basic Materials SPBSBMIP  
11 BSE Industrials     SPBSIDIP  
13 BSE METAL           SI1200  
15 BSE POWER           SIPOWE  
19 BSE Utilities       SPBSUTIP

## Statistical Arbitrage Model

-   In this section, we are going to consider the Cointegrated Augmented Dickey-Fuller ([CADF](https://www.quantstart.com/articles/Cointegrated-Augmented-Dickey-Fuller-Test-for-Pairs-Trading-Evaluation-in-R/)) test \[6\] applied to the BBY and AAL close prices.
-   According to the [Macroaxis Correlation Matchups](https://www.macroaxis.com/invest/marketCorrelation), these two securities move together with a correlation of +0.89.
-   Importing the necessary Python libraries and downloading the stock data with [Twelve Data](https://twelvedata.com/) financial APIs

import numpy as np \# linear algebra  
import pandas as pd \# data processing, CSV file I/O (e.g. pd.read\_csv)  
import requests  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from termcolor import colored as cl  
from math import floor  
  
plt.rcParams\['figure.figsize'\] = (20,10)  
plt.style.use('fivethirtyeight')  
  
  
  
def get\_historical\_data(symbol, start\_date):  
    api\_key = 'YOUR API KEY'  
    api\_url = f'https://api.twelvedata.com/time\_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api\_key}'  
    raw\_df = requests.get(api\_url).json()  
    df = pd.DataFrame(raw\_df\['values'\]).iloc\[::-1\].set\_index('datetime').astype(float)  
    df = df\[df.index >= start\_date\]  
    df.index = pd.to\_datetime(df.index)  
    return df  
  
sdef = get\_historical\_data('BBY', '2020-01-01')  
sdef.tail()  
  
           open   high     low       close  volume  
datetime       
2025\-05-19 72.34 72.640000 71.050003 71.60 3356400.0  
2025\-05-20 71.50 72.770000 70.880000 71.15 4254600.0  
2025\-05-21 70.00 71.660000 69.470000 70.15 3541100.0  
2025\-05-22 69.90 71.089996 69.770000 70.76 3125000.0  
2025\-05-23 67.75 70.720000 67.560000 69.92 3591100.0  
  
cdef = get\_historical\_data('AAL', '2020-01-01')  
cdef.tail()  
  
           open  high   low   close volume  
datetime       
2025\-05-19 11.68 11.93 11.50 11.86 43362700.0  
2025\-05-20 11.83 11.96 11.57 11.65 44004700.0  
2025\-05-21 11.46 11.56 11.19 11.24 59079100.0  
2025\-05-22 11.24 11.56 11.18 11.40 50411200.0  
2025\-05-23 11.06 11.28 11.02 11.19 46736600.0

-   Implementing the CADF test \[6\]

#COINTEGRATED AUGMENTED DICKEY FULLER TEST  
\# Code taken from https://gist.github.com/jcorrius/e79c6372a24c0f402f4bcb29b0fca05b  
\# Conducts (Cointegrated) Augmented Dickey Fuller unit root test  
from statsmodels.regression.linear\_model import OLS  
from statsmodels.tsa.tsatools import lagmat, add\_trend  
from statsmodels.tsa.adfvalues import mackinnonp  
  
   
def adf(ts, maxlag=1):  
    """  
    Augmented Dickey-Fuller unit root test  
    """  
    \# Get the dimension of the array  
    nobs = ts.shape\[0\]  
           
    \# Calculate the discrete difference  
    tsdiff = np.diff(ts)  
       
    \# Create a 2d array of lags, trim invalid observations on both sides  
    tsdall = lagmat(tsdiff\[:, None\], maxlag, trim='both', original='in')  
    \# Get dimension of the array  
    nobs = tsdall.shape\[0\]   
       
    \# replace 0 xdiff with level of x  
    tsdall\[:, 0\] = ts\[-nobs - 1:-1\]    
    tsdshort = tsdiff\[-nobs:\]  
       
    \# Calculate the linear regression using an ordinary least squares model      
    results = OLS(tsdshort, add\_trend(tsdall\[:, :maxlag + 1\], 'c')).fit()  
    adfstat = results.tvalues\[0\]  
       
    \# Get approx p-value from a precomputed table (from stattools)  
    pvalue = mackinnonp(adfstat, 'c', N=1)  
    return pvalue  
   
def cadf(x, y):  
    """  
    Returns the result of the Cointegrated Augmented Dickey-Fuller Test  
    """  
    \# Calculate the linear regression between the two time series  
    ols\_result = OLS(x, y).fit()  
       
    \# Augmented Dickey-Fuller unit root test  
    return adf(ols\_result.resid)  
  
\# returning p value of cointegration test at 5% alpha  
x=list(sdef\["close"\])  
y=list(cdef\["close"\])  
cadf(x,y)  
  
0.0008439749539679426

-   The CADF test provides us with a very low p-value << 5%. We can likely reject the null hypothesis of the presence of a unit root and conclude that we have a stationary series and hence a cointegrated pair.

## Pairs Trading with Options

-   Referring to the recent university-based study \[4\], the objective of this section is to discuss option strategies in a pair-trading framework, including explaining how to setup a pair trade and how it works, conducting the SWOT analysis of pairs trading options, and more.
-   A typical pair trade is buying one stock you think will increase while at the same time selling a stock in a similar industry, sector, or market direction.
-   Let’s understand how options work to make a pair trade using options.
-   **Step 1**: we must be aware of option leverage. Each stock _call_ option contract purchased gives the investor the right to _buy_ 100 shares of stock at the given price P, while a _put_ gives the investor the right to _sell_ 100 shares of stock at P.
-   **Step 2**: about the “greeks” of options in a portfolio setting.
-   First, _delta_ designates how a $1 change in stock price impacts option premiums. A delta of 0.50 signifies that a dollar increase in stock price leads to an increase of 50 cents in the option premium.
-   Second, _gamma_ (first derivative of delta) or the rate at which delta increases or decreases can be used as well.
-   Next, we should examine _theta_ or time premium decay. This shows how fast the time premium of the option reduces or decays over time. This is important because when purchasing calls and puts as time passes the option time premium approaches zero. Thus, options with longer time periods have higher premiums and those premiums decay as you approach expiration.
-   Finally, we can consider _Vega_ or the how volatility impacts option prices. As volatility of the underlining stock increases so does option premiums on that stock. With all else equal, the option premium increases in value when the implied volatility of the underlying stock increases. So, even with no price movement in the underlining stock the option will move down with time and can move up and down with changes in volatility.

**Summary of the “greeks”:**

-   _Delta =_ How stock price changes impact option premiums
-   _Gamma_ = First derivative of delta (the rate at which delta increases or decreases)
-   _Theta_ = Time premium decay (how fast the time premium decays over time)
-   _Vega =_ How the option premium will change with changes in implied volatility of the stock

[**Best Industry Practices:**](https://www.schaeffersresearch.com/content/education/2017/05/25/best-practices-for-pairs-trading)

-   Once you have identified two stocks, we like to pair them by utilizing call and put options that have similar [**deltas**](http://www.schaeffersresearch.com/education/options-basics/key-option-concepts/option-greeks).
-   Try to play 3–6 month options, considering that you are paying for time premium on two options.
-   Combine fundamental, technical, and sentiment analysis to look for stocks that can buck the trend of expectations.
-   Enter the [**summer months**](http://www.schaeffersresearch.com/content/analysis/2017/05/03/indicator-of-the-week-the-best-day-of-the-year-to-sell-stocks), which historically haven’t been quite as volatile as other times of the year. On the other hand, option premiums are at some of their lowest levels in years.

**SWOT Analysis of Pairs + Options**

_Strengths_:

-   [Options](https://sharecouncil.co/en/blog/the-benefits-of-options-flexibility-and-potential-returns) give investors the **flexible** right, but not the obligation, to buy (call option) or sell (put option) underlying assets at a predetermined price within a specified period.
-   At the same time, [pairs trading](https://blog.quantinsti.com/pairs-trading-basics/) helps in the mitigation of risks of call/put options as the pairs strategy involves dealing with two securities so if one is underperforming then there are chances that the other absorbs the losses.
-   The stock that you have a call on could outperform the stock you have a put on, and the relative outperformance could lead to the overall profitability of the [pairs trade](https://www.schaeffersresearch.com/content/education/2017/05/25/best-practices-for-pairs-trading).
-   Even if you are dead wrong on the losing side of the trade, the most you can lose is 100% of the premium paid. With [options](https://www.schaeffersresearch.com/content/education/2017/05/25/best-practices-for-pairs-trading), you have the ability to gain more than 100% on the winning side. If the entire sector makes a huge move in one direction, you may lose 100% on your losing side and gain well over 100% on the winning side.

_Weaknesses_:

-   Options do have several [disadvantages](https://steadyoptions.com/articles/ep-pros-cons-options-trading/) such as the high sellers’ risk, low liquidity, and trading/commission costs.
-   A key [disadvantage](https://www.schaeffersresearch.com/content/education/2017/05/25/best-practices-for-pairs-trading) is that you are opening two trades at the same time, so generally [**premiums**](http://www.schaeffersresearch.com/education/options-basics/key-option-concepts/understanding-option-pricing) will be higher than simply buying a directional call or put without a pair.
-   Unfortunately, [pairs trading](https://blog.quantinsti.com/pairs-trading-basics/) relies on the securities having a _high_ statistical correlation. Most of the traders require a correlation of at least 0.80 which is very challenging to recognize. Besides, some traders highly discourage pairs trading because of its higher commission charges.

_Opportunities_:

-   The pairs trading strategy can be particularly attractive during periods of high market volatility or when market participants exhibit a tendency to overreact to news, creating temporary price deviations.
-   Here are the 3 [key advantages](https://www.investopedia.com/articles/optioninvestor/06/options4advantages.asp) options offer:

1.  They can provide increased cost-efficiency.
2.  They can be less risky than [equities](https://www.investopedia.com/terms/e/equity.asp).
3.  They can deliver higher percentage returns.

_Threats_:

-   [Example](https://www.reddit.com/r/Daytrading/comments/14hk7o8/why_are_options_so_dangerous/): Let’s say an average options contract costs $100. This is a cheap way to control 100 shares of a given stock (never mind delta and the other Greeks right now), which might cost thousands more to have that many in actual shares, so small accounts can take advantage of that leverage. Great, I can buy like 10 of these! Well, depending on the price movement of the underlying stock, that contract can increase and decrease and decrease substantially on any given day, going to 105, 110, 120, 150 etc. Let’s say you bought 10 of those $100 contracts and the price collapsed to $50 overnight. This can happen for a number of reasons other than just the price of the stock itself dropping, and now you’re down $500 on a single trade that only cost $1000 to get in. Some people might even have blown their entire account on a single trade. This is why it’s dangerous.
-   Ideally, [pairs trading](https://fsgjournal.nl/article/2024-04-17-pairs-trading-riding-high-or-risking-all) seeks profit by exploiting price disparities between correlated assets. Issues arise when prices diverge, correlations are misestimated, model assumptions don’t hold and shorting restrictions limit trading ability.
-   Generally, this trade relies on [volatility](https://www.schaeffersresearch.com/content/education/2017/05/25/best-practices-for-pairs-trading) and a directional move by one or both of the underlying stocks. If volatility suppresses or both stocks trade sideways, you could lose on both legs as the time premium decays and both the call and put are sitting at losses.

## Visualizing Greeks in Finance Toolkit

-   The purpose of this section is to show how the Finance Toolkit \[7\] can help effectively visualize the aforementioned greeks.
-   Importing and installing the necessary Python libraries, reading the tech stock data and collecting all greeks with expiration\_time\_range=180

!pip install financetoolkit  
  
import matplotlib.pyplot as plt  
import matplotlib.ticker as mtick  
import pandas as pd  
import seaborn as sns  
from matplotlib import patheffects  
from matplotlib import cm  
  
from financetoolkit import Toolkit  
  
API\_KEY = "YOUR API KEY"  
  
companies = Toolkit(\["NVDA", "AMD"\], api\_key=API\_KEY, start\_date="2017-12-31")  
  
all\_greeks = companies.options.collect\_all\_greeks(expiration\_time\_range=180)

-   Plotting the 1–6 months Greek Sensitivities for NVDA

fig, ax = plt.subplots(figsize=(15, 10), ncols=2, nrows=2)  
  
delta\_over\_time\_df = pd.DataFrame()  
dates = all\_greeks.columns.get\_level\_values(0)  
  
\# Loop through different times  
for i, time in enumerate(range(30, 210, 30)):  
  
    try:  
        period\_column = dates\[time\]  
    except IndexError:  
        period\_column = dates\[-1\]  
  
    color = cm.viridis(i / 5)  \# Using viridis colormap for color variation  
  
    \# Delta plot  
    ax\[0, 0\].plot(all\_greeks.loc\["NVDA", (period\_column, "Delta")\], color=color)  
  
    \# Gamma plot  
    ax\[0, 1\].plot(all\_greeks.loc\["NVDA", (period\_column, "Gamma")\], color=color)  
  
    \# Theta plot  
    ax\[1, 0\].plot(all\_greeks.loc\["NVDA", (period\_column, "Theta")\], color=color)  
  
    \# Vega plot  
    ax\[1, 1\].plot(all\_greeks.loc\["NVDA", (period\_column, "Vega")\], color=color)  
  
    delta\_over\_time\_df = pd.concat(  
        \[delta\_over\_time\_df, all\_greeks.loc\["NVDA", (period\_column, "Delta")\]\], axis=1  
    )  
  
date\_labels = \[  
    "1 Month",  
    "2 Months",  
    "3 Months",  
    "4 Months",  
    "5 Months",  
    "6 Months",  
\]  
  
delta\_over\_time\_df.columns = date\_labels  
  
\# Show the DataFrame  
display(delta\_over\_time\_df.iloc\[7:12\])  
  
\# Copy to clipboard (this is just to paste the data in the README)  
pd.io.clipboards.to\_clipboard(delta\_over\_time\_df.iloc\[7:12\].to\_markdown(), excel=False)  
  
\# Titles and labels  
for number1, number2 in \[(0, 0), (1, 0), (0, 1), (1, 1)\]:  
    ax\[number1, number2\].set\_xlim(  
        \[all\_greeks.loc\["NVDA"\].index.min(), all\_greeks.loc\["NVDA"\].index.max()\]  
    )  
    ax\[number1, number2\].grid(True, linestyle="--", alpha=0.7)  
    ax\[number1, number2\].set\_xlabel("Strike Price")  
    ax\[number1, number2\].set\_facecolor("#F5F5F5")  
  
ax\[0, 0\].set\_title("Delta")  
ax\[0, 1\].set\_title("Gamma")  
ax\[1, 0\].set\_title("Theta")  
ax\[1, 1\].set\_title("Vega")  
  
\# Adjust layout  
fig.legend(  
    date\_labels,  
    loc="upper center",  
    ncol=6,  
    bbox\_to\_anchor=(0.5, 0),  
    frameon=False,  
)  
fig.suptitle(  
    "Greek Sensitivities for NVDA", fontsize=30, x=0.5, y=0.98, fontfamily="cursive"  
)  
  
fig.tight\_layout()  
  
\# Show the plot  
plt.show()

![1–6 months Greek Sensitivities for NVDA](https://miro.medium.com/v2/resize:fit:1050/1*OEANeWpFgwYp8m7YRs-gZQ.png)
1–6 months Greek Sensitivities for NVDA

## Pairs Trading-as-a-Service (PTaaS)

-   PTaaS refers to a model where a third-party provider manages the entire trading process, including execution, risk management, and potentially other related services, for a client, [according to Luxoft](https://www.luxoft.com/files/pdfs/banking/Trading-System-as-a-Service.pdf).
-   In this section, we’ll demonstrate how PTaaS can unlock opportunities in the financial world.
-   PTaaS can make [scale](https://www.mckinsey.com/industries/financial-services/our-insights/how-trading-as-a-service-unlocks-opportunities-for-banks) benefits more widely available, with opportunities for various market players.
-   PTaaS enables you to outsource [risk](https://www.vestmark.com/outsourced-services/model-trading-service) to a highly scalable platform with automated workflows and an experienced team.
-   Since the market is extremely dynamic, PTaaS can model feature parameters to dynamically balance between price and execution probability and use that as inputs for the [optimization](https://www.ctrmcenter.com/publications/interviews/trading-as-a-service-in-intraday-power-markets/).
-   PTaaS can discover insights faster that will help quants automate their complex analytic tasks. It enables quants to integrate their own tools, parameters, pricing models and workflows so domain experts can personalize their experience.

## **Final Thoughts!**

-   _FAANG Correlation Matchups 2020–2025_:

Backtesting the META-NFLX stock pair with max CC = 0.91

Profit percentage of our investment : 405.318%

Vol(Portfolio) < Vol(META) ~Vol(NFLX), where Vol means Volatility.

-   _Spread-ADX-RSI Pairs Trading Strategy_:

PAIRS TRADING BACKTESTING RESULTS:  
EARNING: $144851.13 ; ROI: 144.85%

-   _Pairs Trading Long-Short Equity ETFs SPY & QQQ 2020–2025_:

#Result of backtesting  
  
Total returns: 13851.85%

-   _Sector Rotation Strategy_ using 19 different sectors in India’s BSE: we have examined the 30, 60 and 90 days returns, sell dates, profits, the rate of change, the percentage of profitable trades and Return/Risk ratio for each strategy.
-   _Statistical Arbitrage Model_:

The CADF test for the BBY-AAL pair provides us with a very low p-value << 5%

0.0008439749539679426

Hence, we can likely reject the null hypothesis of the presence of a unit root and conclude that we have a stationary series and hence a cointegrated pair.

This is also consistent with the [Macroaxis Correlation Matchups](https://www.macroaxis.com/invest/marketCorrelation): these two securities move together with a correlation of +0.89.

-   _Pairs Trading with Options:_

Summary of the “greeks”, best industry practices, and SWOT analysis.

-   _Visualizing Greeks in Finance Toolkit_:

Example of 1–6 months Greek Sensitivities for NVDA (Delta, Gamma, Theta, and Vega).

-   _Pairs Trading-as-a-Service (PTaaS):_

Is PTaaS going to capture growth in the FinTech sector?

## References

1.  [Creating a Diversified Portfolio with Correlation Matrix in Python](https://www.insightbig.com/post/creating-a-diversified-portfolio-with-correlation-matrix-in-python)
2.  [Developing a Profitable Pairs Trading Strategy with Python](https://www.insightbig.com/post/developing-a-profitable-pairs-trading-strategy-with-python)
3.  [ETF Pairs Trading Signals (SPY, QQQ)](https://www.kaggle.com/code/christopherchiarilli/etf-pairs-trading-signals-spy-qqq)
4.  [Pair Trading with Options](https://www.aabri.com/LV2014Manuscripts/LV14042.pdf)
5.  [Sector Rotation Strategy](https://github.com/everyfin-in/sector-rotation-strategy/blob/main/EVERYFIN_sector_rotation_strategy.ipynb)
6.  [Statistical Arbitrage Model](https://www.kaggle.com/code/abhimanyunag/statistical-arbitrage-model/notebook)
7.  [Finance Toolkit](https://github.com/JerBouma/FinanceToolkit)

## Explore More

-   [A Deeper Dive into Financial Functions in Python with Beginner-Friendly FinTech Examples, GUI & Data Visualization Use Cases](/@alexzap922/a-deeper-dive-into-financial-functions-in-python-with-beginner-friendly-fintech-examples-gui-f952a5625942)
-   [An Integrated Quant Trading Analysis of US Big Techs using Quantstats, TA, PyPortfolioOpt, and FinanceToolkit](/insiderfinance/an-integrated-quant-trading-analysis-of-us-big-techs-using-quantstats-ta-pyportfolioopt-and-5287b6cd9163)
-   [Strategic Asset Allocation (SAA) & Portfolio Optimization (PO): Go-To’s for Quants](/@alexzap922/strategic-asset-allocation-saa-portfolio-optimization-po-go-tos-for-quants-2da46c7e6c0e)
-   [2Y Big Techs Portfolio Diversification, Risk-Return Tradeoff and LSTM Price Prediction: AAPL, NVDA, META & AMZN](/@alexzap922/2y-big-techs-portfolio-diversification-risk-return-tradeoff-and-lstm-price-prediction-aapl-nvda-b8cb1992697f)

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

### A Message from InsiderFinance

![](https://miro.medium.com/v2/resize:fit:452/0*10x5_2smmKq8oIlf.png)

Thanks for being a part of our community! Before you go:

-   👏 Clap for the story and follow the author 👉
-   📰 View more content in the [InsiderFinance Wire](https://wire.insiderfinance.io/)
-   📚 Take our [FREE Masterclass](https://learn.insiderfinance.io/p/mastering-the-flow)
-   **📈 Discover** [**Powerful Trading Tools**](https://insiderfinance.io/?utm_source=wire&utm_medium=message)

## Embedded Content

---