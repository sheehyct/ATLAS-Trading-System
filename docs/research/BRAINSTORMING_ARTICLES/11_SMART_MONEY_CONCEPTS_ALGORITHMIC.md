# Fair Value Gap (FVG) Algo-Trading with Smart Money Concepts (SMC) | by Alexzap | Sep, 2025 | InsiderFinance Wire

Member-only story

# Fair Value Gap (FVG) Algo-Trading with Smart Money Concepts (SMC)

## Advanced Technical Analysis using SMC FVG Trading Signals in Python — ORCL Stock in Focus

[

![Alexzap](https://miro.medium.com/v2/resize:fill:48:48/1*L1sRpfwSPETNMy1qzGUGBg.jpeg)





](/@alexzap922?source=post_page---byline--982a4e4c92d7---------------------------------------)

[Alexzap](/@alexzap922?source=post_page---byline--982a4e4c92d7---------------------------------------)

Following

7 min read

·

Sep 14, 2025

118

2

Listen

Share

More

[**“All models are wrong, but some are useful**”.](https://en.wikipedia.org/wiki/All_models_are_wrong)\- [George E. P. Box](https://en.wikipedia.org/wiki/George_E._P._Box)

Let’s break down the mechanics of SMC trading strategies based on the KISS (Keep It Simple, Stupid!) principle.

![Modified Canva Image Design Template by @red-hawk-eye](https://miro.medium.com/v2/resize:fit:750/1*vaglIrjLkSAQPJfW-eLU4A.png)
Modified Canva Image Design Template by @red-hawk-eye

-   Nowadays the use of algorithmic (aka algo-) trading \[1, 2\] has become more predominant in [major financial markets](https://www.sciencedirect.com/science/article/abs/pii/S0378426621000480).
-   According to [Fortune Business Insights](https://share.google/kqYc4zbPAcnfpnhCL), the global algo-trading market size was valued at USD 2.36 billion in 2024 and is projected to grow from USD 2.53 billion in 2025 to USD 4.06 billion by 2032, exhibiting a CAGR of 7.0% during the forecast period.
-   Most technical analysis-based [algo-trading strategies](/@jenish064/build-an-algorithmic-trading-api-based-on-technical-analysis-69627ba9ebc5) involve the technical indicators \[1\]. Each indicator is designed to serve a certain [purpose](/@jenish064/build-an-algorithmic-trading-api-based-on-technical-analysis-69627ba9ebc5) by choosing among long-term investments, short-term intraday, swings, derivatives, or commodities.
-   However, most popular technical indicators \[1\] are _lagging_ in that they all respond to what has occurred and not what’s going to occur. These indicators often create a _false confidence_ and therefore are terrible in choppy or fast-moving markets \[3\].
-   [The Inner Circle Trader (ICT) Trading Strategy](https://howtotrade.com/wp-content/uploads/2023/11/ICT-Trading-Strategy-1.pdf) developed by Michael J. Huddleston attempts to address the these shortcomings of lagging indicators based on the actions of institutional traders in the market (e.g. central banks in forex trading) \[3\].
-   ICT focuses on recognizing the [footprints](https://eplanetbrokers.com/training/ict-trading-strategy-explained/) left behind by banks, hedge funds, and other big players. These show up through things like order blocks, liquidity grabs, fair value gaps, and shifts in market structure.
-   A crucial insight that the ICT method provides is the identification of the precise candle where the [order flow](https://howtotrade.com/blog/order-flow-trading/) starts.
-   Michael distilled his trading insights into the [_Smart Money Concept (SMC)_](https://eplanetbrokers.com/training/ict-trading-strategy-explained/) \[4\]_._ Instead of relying solely on indicators that lag price and basic chart patterns, SMC practitioners \[4\] analyze key market structures and patterns that signal institutional activity, viz.

1.  **Order Blocks**: Areas where institutions place large orders that drive future price movement
2.  **Liquidity:** Concentrations of stop losses and pending orders that institutions target
3.  **Market Structure:** The hierarchical pattern of higher highs/lows or lower highs/lows
4.  **Fair Value Gaps (FVG):** Quick moves that leave imbalances in price, creating future reversal zones
5.  **Breaker Blocks:** Order blocks that have been broken through and now serve as strong support/resistance.

-   In this post, we’ll look at the FVG trading signals available through [smartmoneyconcepts](https://pypi.org/project/smartmoneyconcepts/) in Python \[5\]

!pip install smartmoneyconcepts

-   Typically, a [FVG](https://share.google/q5KeLhrfTPwi86hlG) is formed when the low of a candle is higher than the high of the next candle (for bearish FVGs) or when the high of a candle is lower than the low of the next candle (for bullish FVGs). It’s essentially a gap in the normal flow of price action, as shown below.

![Bullish & Bearish FVGs.](https://miro.medium.com/v2/resize:fit:1050/1*Sppk4TgrPxccXpXwdPH4JQ.png)
Bullish & Bearish FVGs.

-   Our objective is twofold: (1) the automatic identification of FVGs with the smc.fvg method \[5\]; (2) Profitability analysis of FVG trading signals as compared to the Buy&Hold strategy for the same period of time.
-   Below we’ll download the Oracle (NYSE: ORCL) daily stock prices from 2025–01–01 to 2025–09–12. According to [TradingView](https://www.tradingview.com/chart/ORCL/J4YfcruS-ORACLE/), the market views Oracle as a frontrunner in the expanding AI infrastructure space, driving a surge in investor confidence and buying activity.

_Let’s get into specifics!_

-   Importing the necessary Python libraries and fetching the ORCL stock data with [TwelveData API](https://twelvedata.com/) \[1\]

import pandas as pd  
import numpy as np  
import requests  
import matplotlib.pyplot as plt  
from smartmoneyconcepts import smc  
import plotly.graph\_objects as go  
  
def get\_historical\_data(symbol, start\_date):  
    api\_key = 'YOUR API KEY'  
    api\_url = f'https://api.twelvedata.com/time\_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api\_key}'  
    raw\_df = requests.get(api\_url).json()  
    df = pd.DataFrame(raw\_df\['values'\]).iloc\[::-1\].set\_index('datetime').astype(float)  
    df = df\[df.index >= start\_date\]  
    df.index = pd.to\_datetime(df.index)  
    return df  
  
ohlc = get\_historical\_data('ORCL', '2025-01-01')  
  
ohlc.tail()  
  
           open    high      low        close     volume  
datetime       
2025\-09-08 239.890 242.42000 235.310000 238.48000 18803000.0  
2025\-09-09 239.940 243.49001 234.560000 241.50999 41178700.0  
2025\-09-10 319.190 345.72000 312.089996 328.32999 131618100.0  
2025\-09-11 330.340 331.00000 304.600010 307.85999 69857800.0  
2025\-09-12 306.035 307.73500 291.750000 293.97000 1909846.0

-   Using Plotly to visualize the ORCL OHLC chart

fig = go.Figure(data=go.Ohlc(x=ohlc.index,  
                             open\=ohlc.open,  
                             high=ohlc.high,  
                             low=ohlc.low,  
                             close=ohlc.close),  
  
                layout=go.Layout(  
        title=go.layout.Title(text="ORCL OHLC 2025"))  
               )  
fig.update\_xaxes(title\_text="Date")  
fig.update\_yaxes(title\_text="Price USD")  
\# show the figure  
fig.show()

![ORCL OHLC 2025](https://miro.medium.com/v2/resize:fit:1050/1*2jNjJp_uvvhIXRGt7HoFEQ.png)
ORCL OHLC 2025

-   Calculating the FVG indicator \[5\]

myfvg=smc.fvg(ohlc, join\_consecutive=False)

-   The output of smc.fvg consists of the following four columns

df = pd.DataFrame(index=(ohlc.index))  
mylist=myfvg\["FVG"\]  
se = pd.Series(mylist)  
df\['FVG'\]=se.values  
  
mylist=myfvg\["Top"\]  
se = pd.Series(mylist)  
df\['Top'\]=se.values  
  
mylist=myfvg\["Bottom"\]  
se = pd.Series(mylist)  
df\['Bottom'\]=se.values  
  
mylist=myfvg\["MitigatedIndex"\]  
se = pd.Series(mylist)  
df\['MitigatedIndex'\]=se.values  
  
df.tail()  
  
           FVG Top        Bottom    MitigatedIndex  
datetime      
2025\-09-08 NaN NaN        NaN       NaN  
2025\-09-09 1.0 312.089996 242.42000 172.0  
2025\-09-10 1.0 304.600010 243.49001 173.0  
2025\-09-11 -1.0 312.089996 307.97000 0.0  
2025\-09-12 NaN  NaN       NaN       NaN

-   In principle, we can restrict ourselves to the simplified FVG method

    def fvgonly(cls, ohlc: DataFrame, join\_consecutive=False) -> Series:  
        """  
        FVG - Fair Value Gap  
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.  
        Or when the previous low is higher than the next high if the current candle is bearish.  
  
        parameters:  
        join\_consecutive: bool - if there are multiple FVG in a row then they will be merged into one using the highest top and the lowest bottom  
  
        returns:  
        FVG = 1 if bullish fair value gap, -1 if bearish fair value gap  
        """  
  
        fvg = np.where(  
            (  
                (ohlc\["high"\].shift(1) < ohlc\["low"\].shift(-1))  
                & (ohlc\["close"\] > ohlc\["open"\])  
            )  
            | (  
                (ohlc\["low"\].shift(1) > ohlc\["high"\].shift(-1))  
                & (ohlc\["close"\] < ohlc\["open"\])  
            ),  
            np.where(ohlc\["close"\] > ohlc\["open"\], 1, -1),  
            np.nan,  
        )  
  
        return pd.concat(  
            \[  
                pd.Series(fvg, name="FVG"),  
            \],  
            axis=1,  
        )  
  
myfvgonly=smc.fvgonly(ohlc, join\_consecutive=False)  
  
print(myfvgonly)  
  
FVG  
0    NaN  
1    NaN  
2    NaN  
3   -1.0  
4    NaN  
..   ...  
169  NaN  
170  1.0  
171  1.0  
172 -1.0  
173  NaN  
  
\[174 rows x 1 columns\]

-   Plotting the ORCL FVG Top/Bottom (output of smc.fvg)

plt.plot(df.index,ohlc\['close'\],color='y',alpha=0.5,label='Close')  
plt.scatter(df.index,myfvg\["Top"\],color = 'green',alpha=0.99,label='Top')  
plt.scatter(df.index,myfvg\["Bottom"\],color = 'red',alpha=0.99,label='Bottom')  
plt.legend()  
plt.xlabel('Date')  
plt.ylabel('Price USD')  
plt.title('ORCL FVG Top/Bottom')  
plt.show()

![ORCL FVG Top/Bottom](https://miro.medium.com/v2/resize:fit:1050/1*CjVQEe6QGw-fPcg4EGix3w.png)
ORCL FVG Top/Bottom

-   Visualizing the FVG [trading signals](https://eodhd.medium.com/visualizing-trading-signals-in-python-3cab01cc5847)

bullsignals = ohlc\['close'\]\[df\["FVG"\] == 1\]  
bearsignals = ohlc\['close'\]\[df\["FVG"\] == -1\]  
  
plt.plot(ohlc\['close'\],alpha=0.4,label='Close')  
for idx in bullsignals.index.tolist():  
  plt.plot(  
      idx,  
      ohlc.loc\[idx\]\["close"\],  
      "g\*",  
      markersize=25,   
  )  
  
for idx in bearsignals.index.tolist():  
  plt.plot(  
      idx,  
      ohlc.loc\[idx\]\["close"\],  
      "r\*",  
      markersize=25,  
  )  
  
plt.legend()  
plt.title('ORCL FVG Trading Signals')  
plt.xlabel('Date')  
plt.ylabel('Price USD')  
  
plt.show()

![ORCL FVG Buy & Sell Trading Signals (Green & Red Star Symbols, respectively).](https://miro.medium.com/v2/resize:fit:1050/1*SK8lt4NwqkPx29kYjary7A.png)
ORCL FVG Buy & Sell Trading Signals (Green & Red Star Symbols, respectively).

-   Creating positions (shift signals to avoid look-ahead bias) \[2\]

data=df.copy()  
data\['Position'\] = data\['FVG'\].shift(1)

-   Calculating and plotting the daily and cumulative returns (backtesting)

\# Calculate daily percentage change in stock prices  
data\['Daily Return'\] = ohlc\['close'\].pct\_change()  
  
\# Calculate returns based on the strategy  
data\['Strategy Return'\] = data\['Position'\] \* data\['Daily Return'\]  
  
\# Calculate cumulative returns  
data\['Cumulative Market Return'\] = (1 + data\['Daily Return'\]).cumprod()  
data\['Cumulative Strategy Return'\] = (1 + data\['Strategy Return'\]).cumprod()  
  
plt.style.use('fivethirtyeight')  
plt.rcParams\['figure.figsize'\] = (10,5)  
  
rets = ohlc.close.pct\_change()\[1:\]  
strat\_rets = data\['Position'\]\[1:\] \* rets  
  
plt.title('ORCL Daily Returns')  
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='Buy&Hold')  
strat\_rets.plot(color = 'r', linewidth = 1,linestyle='dashed', marker='s',label='FVG')  
plt.legend()  
plt.show()  
  
rets\_cum = (1 + rets).cumprod() - 1   
strat\_cum = (1 + strat\_rets).cumprod() - 1  
  
plt.title('ORCL Cumulative Returns')  
rets\_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='Buy&Hold')  
strat\_cum.plot(color = 'r', linewidth = 2,linestyle='dashed', marker='s',label='FVG')  
plt.legend()  
plt.show()

![ORCL Daily Returns: FVG vs Buy&Hold.](https://miro.medium.com/v2/resize:fit:1050/1*v_tUTtIkEm3WwjQsOTpcXA.png)
ORCL Daily Returns: FVG vs Buy&Hold.

![ORCL Cumulative Returns: FVG vs Buy&Hold.](https://miro.medium.com/v2/resize:fit:1050/1*OtG-zrCoGEoFr1LYt4iVLQ.png)
ORCL Cumulative Returns: FVG vs Buy&Hold.

## Conclusions

-   We have implemented and (back)tested the FVG trading strategy using [smartmoneyconcepts](https://pypi.org/project/smartmoneyconcepts/) in Python.
-   In terms of the cumulative return, this strategy is shown to significantly outperform the buy & hold approach for the Oracle stock in Q1-Q3 2025.
-   However, these are strictly backtesting findings. Like any strategy, there is no guarantee of future performance, and this strategy should not be taken as financial advice.

## References

1.  [Algorithmic Trading with Python](https://github.com/Nikhil-Adithyan/Algorithmic-Trading-with-Python)
2.  [Automated Trading using Python](https://www.geeksforgeeks.org/python/automated-trading-using-python/)
3.  [The Only TradingView Indicators That Actually Work (NO RSI, MACD BS)](/@market.muse/the-only-tradingview-indicators-that-actually-work-no-rsi-macd-bs-92a83df6f5cc)
4.  [Smart Money Concept: Trading.](/@market.muse/smart-money-concept-trading-9c2f855da80d)
5.  [GitHub Repo: smartmoneyconcepts](https://github.com/joshyattridge/smart-money-concepts/tree/master/smartmoneyconcepts)

## Explore More

-   [Oracle Monte Carlo Stock Simulations](https://newdigitals.org/2023/10/20/oracle-monte-carlo-stock-simulations/)
-   [Justify the ROI of AI: Comparing Fundamentals, Risks, Volatility & Returns of Top 15 US Tech Stocks](/insiderfinance/justify-the-roi-of-ai-comparing-fundamentals-risks-volatility-returns-of-top-15-us-tech-stocks-58984625f4fc)

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

![](https://miro.medium.com/v2/resize:fit:452/0*10x5_2smmKq8oIlf.png)

Thanks for being a part of our community! Before you go:

-   👏 Clap for the story and follow the author 👉
-   📰 View more content in the [InsiderFinance Wire](https://wire.insiderfinance.io/)
-   📚 Take our [FREE Masterclass](https://learn.insiderfinance.io/p/mastering-the-flow)
-   **📈 Discover** [**Powerful Trading Tools**](https://insiderfinance.io/?utm_source=wire&utm_medium=message)

## Embedded Content

---