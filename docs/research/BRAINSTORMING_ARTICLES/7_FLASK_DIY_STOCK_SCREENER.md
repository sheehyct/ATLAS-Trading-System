# Creating & Using this Flask DIY Stock Screener App & Dashboard — It’s Money for Jam! | by Alexzap | InsiderFinance Wire

Member-only story

# Creating & Using this Flask DIY Stock Screener App & Dashboard — It’s Money for Jam!

## Combining Stock Candlesticks, Volume, Sought-After Trading Indicators, Support/Resistance Levels, and the Twelve Data API endpoints into a Simple Yet Powerful Plotly/Dash UI Framework in Python

[

![Alexzap](https://miro.medium.com/v2/resize:fill:48:48/1*L1sRpfwSPETNMy1qzGUGBg.jpeg)





](/@alexzap922?source=post_page---byline--e663a8eb31f6---------------------------------------)

[Alexzap](/@alexzap922?source=post_page---byline--e663a8eb31f6---------------------------------------)

Following

21 min read

·

Jul 18, 2025

154

1

Listen

Share

More

[“Investors should purchase stocks like they purchase groceries, not like they purchase perfume.”- Benjamin Graham](https://www.adityabirlacapital.com/abc-of-money/motivational-quotes-for-stock-market-investing)

![Photo by rc.xyz NFT gallery on Unsplash](https://miro.medium.com/v2/resize:fit:1050/0*7YAKuCW2kcVL5azV)
Photo by rc.xyz NFT gallery on Unsplash

-   In this fast-paced data-driven world, where Python data science is gaining popularity with the hypersonic speed, [Plotly / Dash](https://github.com/plotly/dash) has emerged as one of the leading players in the market.
-   Built on top of [Plotly.js](https://github.com/plotly/plotly.js), [React](https://reactjs.org/) and [Flask](https://palletsprojects.com/p/flask/), Dash ties modern UI elements like dropdowns, sliders, and graphs directly to your web app \[1, 2\].
-   We can start off quickly with a simple [example](https://github.com/plotly/dash?tab=readme-ov-file) of a Dash App that dynamically exports data from Google Finance, not to mention [6 examples](https://plotly.com/python/ohlc-charts/) of OHLC charts with Yahoo Finance.
-   Reservations notwithstanding, the ascent of Dash in the digital financial landscape has been nothing short of impressive \[4–9\]. The seamless integration of core [features](https://github.com/plotly/dash) (Low-Code, Enterprise AI, etc.), its user-friendly GUI and ease of use for newbies have contributed to its continuously growing user base.
-   In this post, we’ll delve into the intricacies of the simple yet powerful Stock Screener Dash App by leveraging the basic features of Plotly time-series data visualization \[4, 5\] along with the [Twelve Data API](https://publicapi.dev/twelve-data-api) endpoints to get free access to a wide range of financial data and market information \[10\].
-   You can get the full source code in Appendix A.

## Prerequisites

-   [Twelve Data API](https://publicapi.dev/twelve-data-api) is needed to retrieve real-time stock quote data.
-   Installed [Plotly Dash](https://plotly.com/python/getting-started/), the Technical Analysis (ta) [Library](https://pypi.org/project/ta/), and the HTTP library [requests](https://pypi.org/project/requests/).

## Essential Imports

-   The first and foremost step is to import all the required packages into our Python environment

!pip install ta  
  
import pandas as pd #data formatting, clearing, manipulating, wrangling.  
import numpy as np  
from datetime import date  
import plotly.graph\_objects as go #creating charts and visualizations  
from plotly.subplots import make\_subplots  
from ta.trend import MACD  
from ta.momentum import RSIIndicator  
from ta.momentum import StochasticOscillator   
from dash import html, dcc, Dash  
from dash.dependencies import Input, Output, State  
  
import requests #for making API calls in order to extract data

## Reading Stock Historical Data

-   Obtaining the stock historical data with [Twelve Data API](https://publicapi.dev/twelve-data-api) form start\_date to today \[10\]

def get\_historical\_data(symbol, start\_date):  
    api\_key = 'YOUR API KEY'  
    api\_url = f'https://api.twelvedata.com/time\_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api\_key}'  
    raw\_df = requests.get(api\_url).json()  
    df = pd.DataFrame(raw\_df\['values'\]).iloc\[::-1\].set\_index('datetime').astype(float)  
    df = df\[df.index >= start\_date\]  
    df.index = pd.to\_datetime(df.index)  
    return df

-   Make sure to replace `YOUR API KEY` with your secret API key.
-   Using the `get` function provided by the Requests package, we are making an API call to get the daily historical data of the stock defined by symbol.

## Make Candlestick Charts

-   We’ll make interactive [candlestick charts](https://plotly.com/python/candlestick-charts/) in Python with Plotly.

def makeCandlestick(fig, stockDF):  
    #sets parameters for subplots  
    fig = make\_subplots(rows = 6, cols = 1, shared\_xaxes = True,  
                    vertical\_spacing = 0.01,  
                    row\_heights = \[0.6, 0.1, 0.15, 0.15,0.15,0.15\])  
  
  
    #plots candlestick values using stockDF  
    fig.add\_trace(go.Candlestick(x = stockDF.index,  
                                 open = stockDF\['Open'\],  
                                 high = stockDF\['High'\],  
                                 low = stockDF\['Low'\],  
                                 close = stockDF\['Close'\],  
                                 name = 'Open/Close'))  
      
    return fig

-   The [candlestick chart](https://en.wikipedia.org/wiki/Candlestick_chart) is a style of financial chart describing open, high, low and close for a given `x` coordinate (time). The boxes represent the spread between the Open and Close values and the lines represent the spread between the Low and High values. Sample points where the close value is higher (lower) then the open value are called increasing (decreasing). By default, increasing candles are drawn in green whereas decreasing are drawn in red.

## Calculate & Plot Stock Indicators

-   Calculating the Awesome Oscillator (AO) \[12\] to spot potential trend reversals and gauge market strength

def calculate\_awesome\_oscillator(df, short\_window=5, long\_window=34):  
    median\_price = (df\["High"\] + df\["Low"\]) / 2  
    short\_sma = median\_price.rolling(window=short\_window).mean()  
    long\_sma = median\_price.rolling(window=long\_window).mean()  
    AO = short\_sma - long\_sma  
    return AO  
  
def generate\_awesome\_oscillator\_color(AO):  
    awesome\_oscillator\_color = \[\]  
    awesome\_oscillator\_color.clear()  
    for i in range (0,len(AO)):  
        if AO\[i\] >= 0 and AO\[i-1\] < AO\[i\]:  
            awesome\_oscillator\_color.append('#26A69A')  
            #print(i,'green')  
        elif AO\[i\] >= 0 and AO\[i-1\] > AO\[i\]:  
            awesome\_oscillator\_color.append('#FF5252')  
            #print(i,'faint green')  
        elif AO\[i\] < 0 and AO\[i-1\] > AO\[i\] :  
            #print(i,'red')  
            awesome\_oscillator\_color.append('#FF5252')  
        elif AO\[i\] < 0 and AO\[i-1\] < AO\[i\] :  
            #print(i,'faint red')  
            awesome\_oscillator\_color.append('#26A69A')  
        else:  
            awesome\_oscillator\_color.append('#000000')  
    return awesome\_oscillator\_color

-   AO is a momentum indicator that compares recent market momentum to the broader trend. It can help us identify trend strength by spotting potential reversals and showing momentum.
-   Calculating the Exponential Moving Averages (MA) \[11\]

def makeMA(fig, stockDF):  
    #create moving average values  
    stockDF\["MA5"\] = stockDF\["Close"\].ewm(span=5, adjust=False).mean()  
    stockDF\["MA15"\] = stockDF\["Close"\].ewm(span=15, adjust=False).mean()  
    stockDF\["MA50"\] = stockDF\["Close"\].ewm(span=50, adjust=False).mean()  
    stockDF\["MA100"\] = stockDF\["Close"\].ewm(span=100, adjust=False).mean()  
  
  
    #plots moving average values; the 50-day and 200-day averages  
    #are visible by default, and the 5-day and 15-day are accessed via legend  
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA5'\], opacity = 0.7,   
                        line = dict(color = 'blue', width = 2), name = 'MA 5'))  
              
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA15'\], opacity = 0.7,   
                        line = dict(color = 'orangered', width = 2), name = 'MA 15'))  
  
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA50'\], opacity = 0.7,  
                        line = dict(color = 'purple', width = 2), name = 'MA 50'))  
  
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA100'\], opacity = 0.7,  
                        line = dict(color = 'black', width = 2), name = 'MA 100'))  
  
    return fig

-   Unlike simple MA, [Exponential MA](/@mburakbedir/understanding-exponential-moving-averages-ema-and-their-applications-in-python-eccb9518d729) give more weight to recent data points, making them more responsive to recent price changes. This responsiveness is particularly useful for traders who need to make timely decisions based on the latest market information.
-   Creating the Volume bar plot

def makeVolume(fig, stockDF):  
    #sets colors of volume bars  
    colors = \['green' if row\['Open'\] - row\['Close'\] >= 0  
          else 'red' for index, row in stockDF.iterrows()\]  
  
  
    #Plot volume trace  
    fig.add\_trace(go.Bar(x = stockDF.index,  
                         y = stockDF\['Volume'\],  
                         marker\_color = colors,  
                         showlegend = False,  
                         name = "Volume"  
                         ), row = 2, col = 1)  
  
    return fig

-   [Trading volume](https://www.tradu.com/eu/guide/stocks/does-trading-volume-affect-stock-price/) doesn’t affect stock price directly, but it does influence how shares move. Normally, trading volumes increase when there is price volatility in the market. For example, high trading volumes normally occur due to news or events that impact the value of the stocks.
-   Plotting the AO histogram

def makeOA(fig, stockDF):  
      
    AO = calculate\_awesome\_oscillator(stockDF,5,34)  
  
    \# List of Color Assiging To Awesome Oscillator  
    awesome\_oscillator\_color = generate\_awesome\_oscillator\_color(AO)  
    \# Data Extracted And New Variable Applied  
    awesome\_oscillator = AO  
  
              
    #Sets color for AO  
    colors = awesome\_oscillator\_color  
  
  
    #Plots AO values  
    fig.add\_trace(go.Bar(x = stockDF.index,  
                         y = AO,  
                         marker\_color = colors,  
                         showlegend = False,  
                         name = "Histogram"  
                         ), row = 6, col = 1)  
  
    return fig

-   Calculating and plotting the Relative Strength Index (RSI) indicator \[10\]

def makeRSI(fig, stockDF):  
    #Create RSI values  
    rsi = RSIIndicator(close = stockDF\["Close"\],  
                       window = 14)  
  
  
    #Plots RSI values  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = rsi.rsi(),  
                             line = dict(color = 'black', width = 2),  
                             showlegend = False,  
                             name = "RSI"  
                             ), row = 3, col = 1)  
  
  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[30 for val in range(len(stockDF))\],  
                             line = dict(color = 'red', width = 1),  
                             showlegend = False,  
                             name = "Oversold"  
                             ), row = 3, col = 1)  
  
  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[70 for val in range(len(stockDF))\],  
                             line = dict(color = 'green', width = 1),  
                             showlegend = False,  
                             name = "Overbought"  
                             ), row = 3, col = 1)  
  
  
    return fig

-   This is a popular [momentum oscillator](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI) that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30.

## Deal with Support/Resistance Levels

-   Finding and returning the resistance and support levels using fractals. [Fractals](https://mondfx.com/fractal-in-forex/) help us recognize key turning points on the chart and use them for entry, exit, and stop-loss placement.

def makeCurrentPrice(fig, stockDF):  
    #Plots the last closing price of stock   
    fig.add\_trace(go.Scatter(x = stockDF.index,  
              y = \[stockDF\['Close'\].iat\[-1\] for price in range(len(stockDF))\],  
              opacity = 0.7, line = dict(color = 'red', width = 2, dash = 'dot'),  
              name = "Current Price: " + str(round(stockDF\['Close'\].iat\[-1\], 2))))  
  
    return fig  
  
  
def supportLevel(stockDF, index):  
    #Finds and returns support levels using fractals;  
    #if there are two higher lows on each side of the current stockDF\['Low'\] value,   
    #return this value  
    support = stockDF\['Low'\]\[index\] < stockDF\['Low'\]\[index - 1\] and \\  
              stockDF\['Low'\]\[index\] < stockDF\['Low'\]\[index + 1\] and \\  
              stockDF\['Low'\]\[index + 1\] < stockDF\['Low'\]\[index + 2\] and \\  
              stockDF\['Low'\]\[index - 1\] < stockDF\['Low'\]\[index - 2\]  
  
    return support  
  
  
def resistanceLevel(stockDF, index):  
    #Finds and returns resistance levels using fractals;  
    #If there are two lower highs on each side of the current stock\['High'\] value,  
    #return this value  
    resistance = stockDF\['High'\]\[index\] > stockDF\['High'\]\[index - 1\] and \\  
              stockDF\['High'\]\[index\] > stockDF\['High'\]\[index + 1\] and \\  
              stockDF\['High'\]\[index + 1\] > stockDF\['High'\]\[index + 2\] and \\  
              stockDF\['High'\]\[index - 1\] > stockDF\['High'\]\[index - 2\]  
  
    return resistance

-   The [fractal pattern](https://mondfx.com/fractal-in-forex/) consists of 5 consecutive candlesticks, where the middle candlestick forms the highest (in a bearish fractal) or lowest (in a bullish fractal) point, while the two candlesticks on either side must have lower highs or higher lows.
-   Finding and plotting the key support/resistance levels

def isFarFromLevel(stockDF, level, levels):  
    #If a level is found near another level, it returns false;  
  
    ##.88 for longer term .97 for short term  
    s = np.mean(stockDF\['High'\] - (stockDF\['Low'\] \* .89))  
  
    return np.sum(\[abs(level - x) < s for x in levels\]) == 0  
  
  
def makeLevels(fig, stockDF):  
    #Traverses through stockDF and finds key support/resistance levels  
    levels = \[\]  
    for index in range(2, stockDF.shape\[0\] - 2):  
        if supportLevel(stockDF, index):  
            support = stockDF\['Low'\]\[index\]  
            if isFarFromLevel(stockDF, support, levels):  
                levels.append((support))  
              
        elif resistanceLevel(stockDF, index):  
            resistance = stockDF\['High'\]\[index\]  
            if isFarFromLevel(stockDF, resistance, levels):  
                levels.append((resistance))  
  
    levels.sort()  
  
    #Plots the key levels within levels   
    for i in range(len(levels)):  
        fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[levels\[i\] for val in range(len(stockDF))\],  
                             line = dict(color = "black"),  
                             name = "Sup/Res: " + str(round(levels\[i\], 2)),  
                             hoverinfo = "skip",  
                             opacity = 0.3))  
  
    return fig

-   Implementing [the Fibonacci retracements](https://zerodha.com/varsity/chapter/fibonacci-retracements/). [The Fibonacci ratios](http://coderzcolumn.com/tutorials/machine-learning/scikit-plot-visualizing-machine-learning-algorithm-results-and-performance) commonly used are 100%, 61.8%, 50%, 38.2%, 23.6% — these are shown as horizontal lines on a chart and may identify areas of support and resistance.

def findAbsMax(stockDF):  
    absMax = 0  
    for i in range(len(stockDF)):  
        if stockDF\["Close"\]\[i\] > absMax:  
            absMax = stockDF\["Close"\]\[i\]  
          
    return absMax  
  
  
def findAbsLow(stockDF):  
    absLow = 50  
    for i in range(len(stockDF)):  
        if stockDF\["Close"\]\[i\] < absLow:  
            absLow = stockDF\["Close"\]\[i\]  
  
    return absLow  
  
  
def makeFibLevels(fig, stockDF):  
    fibRatios = \[.236, .382, .5, .618, .786, 1\]  
    fibLevels = \[\]  
    absMax = findAbsMax(stockDF)  
    absLow = findAbsLow(stockDF)  
    dif = absMax - absLow  
  
    for i in range(len(fibRatios)):  
        fibLevels.append(dif \* fibRatios\[i\])  
#for prices that are above the last resistance/support line within fibLevels,  
    #look to see if there can be any levels drawn using fractals that are also not   
    #too close to the current last support/resistance;  
    #We really are just looking for the last resistance level;  
    fractal = fibLevels\[-1\] + (fibLevels\[-1\] \* .17)  
    if (fibLevels\[-1\] < fractal) and (fractal < absMax):  
         fibLevels.append(fractal)  
  
    for i in range(len(fibLevels)):  
        fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[fibLevels\[i\] for val in range(len(stockDF))\],  
                             line = dict(color = "black"),  
                             name = "Sup/Res: " + str(round(fibLevels\[i\], 2)),  
                             hoverinfo = "skip",  
                             opacity = 0.3))  
      
    return fig  
  
  
def graphLayout(fig, choice):  
    #Sets the layout of the graph and legend  
    fig.update\_layout(title\_text = choice + ' Price Action',   
                  title\_x = 0.5,   
                  legend\_title\_text = "Legend Items",  
                  dragmode = "pan",   
                  xaxis\_rangeslider\_visible = False,   
                  hovermode = "x",   
                  legend = dict(bgcolor="#E2E2E2",  
                           bordercolor="Black",  
                           borderwidth=2)  
                                 
                 )  
  
    subplotLabels(fig)  
  
    return fig

-   [Fibonacci analysis](https://zerodha.com/varsity/chapter/fibonacci-retracements/) can be applied when there is a noticeable up-move or down-move in prices. Whenever the stock moves either upwards or downwards sharply, it usually tends to retrace back before its next move. For example, if the stock has run up from $50 to $100, it is likely to retrace back to probably $70 before moving $120.

## Set Subplot Labels & Basic Dash Layout

-   Adjusting x and y axes properties in Plotly — axes titles, styling and coloring axes and grid lines, ticks, and tick labels.

def subplotLabels(fig):  
    #Sets subplot labels  
    fig.update\_yaxes(title\_text = "Price", row = 1, col = 1)  
    fig.update\_yaxes(title\_text = "Volume", row = 2, col = 1)  
    fig.update\_yaxes(title\_text = "RSI", row = 3, col = 1)  
    fig.update\_yaxes(title\_text = "MACD", showgrid = False, row = 4, col = 1)  
    fig.update\_yaxes(title\_text = "Stoch", showgrid = False, row = 5, col = 1)  
    fig.update\_yaxes(title\_text = "AO", showgrid = False, row = 6, col = 1)  
  
    return fig  
  
  
def xAxes(fig):  
    #Remove none trading days from dataset and sets behavior for x-axis mouse-hovering  
    fig.update\_xaxes(rangebreaks = \[dict(bounds = \["sat", "mon"\])\],   
                 autorange = True,   
                 showspikes = True,   
                 spikedash = "dot",  
                 spikethickness = 1,   
                 spikemode = "across",   
                 spikecolor = "black")  
      
    return fig  
  
  
  
fig = go.Figure()  
config = dict({'scrollZoom': True})

-   Instead of writing HTML or using an HTML templating engine, we can compose our layout with the Dash HTML Components module (`[dash.html](https://dash.plotly.com/dash-html-components)`)

from dash import html  
  
stockApp = Dash(\_\_name\_\_, meta\_tags=\[{'name': 'viewport',   
                       'content':'width=device-width, initial-scale=1.0'}\])  
  
application = stockApp.server  
  
stockApp.layout = html.Div(\[  
    dcc.Graph(figure = fig, config = config,  
  
              style = {'width': '99vw', 'height': '93vh'},  
              id = "stockGraph"  
             ),  
  
             html.Div(\[  
                dcc.Input(  
                    id = "userInput",  
                    type = "text",  
                    placeholder = "Ticker Symbol"  
                         ),  
              
            html.Button("Submit", id = "btnSubmit")\]),  
                      \],  
            )

## Make & Plot Plotly Charts

-   Making and plotting the above Plotly charts with the starting date 2024–03–01 (user parameter) and the default stock symbol NVDA. Merging them in a single dashboard, giving the possibility to the user to access them by taking advantage of an interactive Plotly menu.

@stockApp.callback(    Output("stockGraph", "figure"),  
    Input("btnSubmit", "n\_clicks"),  
    State("userInput", "value"))  
  
def update\_figure(n, tickerChoice):  
    #set choice to something if !isPostBack  
    if tickerChoice == None:  
        tickerChoice = 'NVDA'  
  
  
    #make stockDF      
      
    aapl = get\_historical\_data(tickerChoice, '2024-03-01')  
    stockDF = aapl.rename(columns={'open': 'Open', 'high': 'High','low': 'Low', 'close': 'Close','volume': 'Volume'})  
  
    #make go Figure object as fig  
    fig = go.Figure()  
  
    #make and plot candlestick chart  
    fig = makeCandlestick(fig, stockDF)  
  
    #update layout properties  
    fig = graphLayout(fig, tickerChoice.upper())  
  
    #updates x-axis parameters  
    fig = xAxes(fig)  
  
    #make and plot subplots charts and moving averages  
    fig = makeMA(fig, stockDF)  
    fig = makeVolume(fig, stockDF)  
    fig = makeMACD(fig, stockDF)  
    fig = makeRSI(fig, stockDF)  
    fig = makeOA(fig, stockDF)  
  
    #make and plot stock's last closing price  
    fig = makeCurrentPrice(fig, stockDF)  
  
    #make and plot stock's resistance/support values using fibonacci retracement  
    fig = makeFibLevels(fig, stockDF)  
  
      
    return fig

## Serving Flask App

-   Running the entire application with port = 8080

if \_\_name\_\_ == '\_\_main\_\_':  
    application.run(debug = False, port = 8080)

Following Dash [tutorials](https://dash.plotly.com/) to create a simple app, we run it locally through our terminal, put in the default local host IP and port (http://127.0.0.1:8080/) in any browser and voilà.

## Interpretations

-   Below you can see the Stock Screener described in Appendix A. This is a highly interactive tool that lets you find stocks that match your strategy.
-   Example:

1.  “Show me oversold stocks with high volume.”
2.  “Look for a positive crossover between the 100 and 50 EMAs.”

![Full Stock Screener App (see Appendix A): Interactive NVDA Candlesticks, Exponential MA (EMA), Support/Resistance Levels, Volume, RSI, MACD, Stoch, and AO indicators.](https://miro.medium.com/v2/resize:fit:1050/1*5SSkHTFmnAbyoNhg5O3srg.jpeg)
Full Stock Screener App (see Appendix A): Interactive NVDA Candlesticks, Exponential MA (EMA), Support/Resistance Levels, Volume, RSI, MACD, Stoch, and AO indicators.

-   We can examine the stock [candlesticks](https://groww.in/blog/how-to-read-candlestick-charts) (at the very top) as a visual representation of the size of price fluctuations. Each candle has three parts: the body and upper/lower shadow.
-   We can look for [candle chart patterns](https://groww.in/blog/how-to-read-candlestick-charts) (Doji, Hammer, etc.) understand investor sentiment and the relationship between demand and supply, spot bullish and bearish trends, etc.
-   The candlesticks are especially important for [intra-day investing](https://groww.in/blog/how-to-read-candlestick-charts) in stocks where the trader buys and sells stocks on the same day without any open positions left by the end of the day.

![NVDA Current Price on Jul 17, 2025.](https://miro.medium.com/v2/resize:fit:488/1*LXphu9bte_rYtBrVKJ-bJA.jpeg)
NVDA Current Price on Jul 17, 2025.

-   [The trading volume](https://www.cabotwealth.com/daily/stock-market/trading-volume-important) (second row down) is important because it reflects overall market activity, is a marker of liquidity, and can signal the strength behind moves higher (or lower).
-   It serves as a [warning](https://www.cabotwealth.com/daily/stock-market/trading-volume-important) as to whether a stock is on the verge of breaking into upside territory (high volume) or into a downside trend (low volume). High volume also gives investors more flexibility to determine when it’s the right time to sell since it translates to greater liquidity.

![NVDA Volume on May 6, 2025.](https://miro.medium.com/v2/resize:fit:392/1*KSsLMNV29vZ5wK9AQkTQRw.jpeg)
NVDA Volume on May 6, 2025.

-   The [RSI indicator](https://www.oanda.com/us-en/trade-tap-blog/trading-knowledge/understanding-the-relative-strength-index/) (third row down) is a momentum oscillator that measures the speed and change of price movements. It stands out because it’s simple to use, has been around for a long time and works well.
-   Think of it as a speedometer for market momentum. When the RSI is above 70, it signals that prices might be overbought or overvalued. Conversely, when an asset’s RSI falls below 30, it indicates that the asset may be oversold or undervalued. We can use these signals as warnings, as they often precede price retracements.

![NVDA RSI on Jun 17, 2025.](https://miro.medium.com/v2/resize:fit:701/1*3iqBKUPODKGDiP1RBMPzEQ.jpeg)
NVDA RSI on Jun 17, 2025.

-   The [MACD indicator](https://commodity.com/technical-analysis/macd/) (forth row down) is a versatile tool. A potential buy/sell signal is generated when the MACD (blue line) crosses above/below the MACD Signal Line (red line).
-   The MACD Histogram is simply the difference between the MACD line (blue line) and the MACD signal line (red line).
-   Case 1: The MACD histogram can be shrinking in height. This occurs because there is a change in direction or a slowdown in the stock.
-   Case 2: The MACD histogram is increasing in height. This occurs because the MACD is accelerating faster in the direction of the prevailing market trend.

![NVDA MACD on Jul 11, 2025.](https://miro.medium.com/v2/resize:fit:549/1*fmRFK2915pIM3kou7TAQpg.jpeg)
NVDA MACD on Jul 11, 2025.

-   The [Stochastic indicator](https://tradeciety.com/how-to-use-the-stochastic-indicator) (5th row down) analyses price movements (aka momentum) and tells us how fast and how strong the price moves.
-   Generally, traders would say that a [Stochastic](https://tradeciety.com/how-to-use-the-stochastic-indicator) over 80 suggests that the price is overbought and when the Stochastic is below 20, the price is considered oversold.
-   Notably, %K is referred to sometimes as the [fast stochastic](https://www.investopedia.com/ask/answers/05/062405.asp) indicator. The “slow” stochastic indicator is taken as %D = 3-period moving average of %K.

![NVDA Stoch on Jul 16, 2025.](https://miro.medium.com/v2/resize:fit:536/1*_aeViWFBqbQ_FMgHxlJsIg.jpeg)
NVDA Stoch on Jul 16, 2025.

-   Finally, the [awesome oscillator (AO)](https://www.ig.com/en/trading-strategies/a-traders-guide-to-using-the-awesome-oscillator-200130) at the very bottom is a market momentum indicator which compares recent market movements to historic market movements. It uses a zero line in the center, either side of which price movements are plotted according to a comparison of two different MA.
-   Example: [The AO saucer](https://www.ig.com/en/trading-strategies/a-traders-guide-to-using-the-awesome-oscillator-200130) is a trading signal that many analysts use to identify potential rapid changes in momentum. The saucer strategy involves looking for changes in three consecutive bars that are on the same side of the zero line.

![NVDA AO Histogram on Jul 17, 2025](https://miro.medium.com/v2/resize:fit:678/1*tqzjHKY-j3tI3Qvdo3rDQQ.jpeg)
NVDA AO Histogram on Jul 17, 2025

-   As with all technical indicators, AO signals are no guarantee that a market will behave in a certain way. Because of this, many traders will take steps to manage their risk by combining several technical indicators to optimize the advantages of each. Put them together, and they counteract each other’s negatives.

## Conclusions

-   The present Stock Screener is a simple yet powerful tool designed to help traders and investors identify high-probability trade setups.
-   Using the Screener can eliminate noise and allow you to focus on high-probability setups that match your risk profile, time horizon, and trading goals.
-   _Benefits:_

1.  Real-time market scanning
2.  Reduced emotional bias
3.  Minimized false signals
4.  Improved decision-making.

-   Although the present trading indicators are very simple tools and only looks at a few key data points on your charts, they can provide meaningful information.
-   However, a wrong application of these trading tools leads to incorrect trading decisions as well.
-   Multiple indicators can give the same type of signal and can help us decide with more certainty whether the position should be opened or not.
-   Not least of all, the Screener contains [Support and Resistance Levels](https://mondfx.com/support-and-resistance-levels/) that play a crucial role in determining entry and exit points in trading. These levels appear as horizontal lines on the chart and are of significant importance.

## References

1.  [How to create a beautiful, interactive dashboard layout in Python with Plotly Dash](/plotly/how-to-create-a-beautiful-interactive-dashboard-layout-in-python-with-plotly-dash-a45c57bb2f3c)
2.  [Dash in 20 Minutes](https://dash.plotly.com/tutorial)
3.  [OHLC Charts in Python](https://plotly.com/python/ohlc-charts/)
4.  [Python: Adding Features To Your Stock Market Dashboard With Plotly](/@jsteinb/python-adding-features-to-your-stock-market-dashboard-with-plotly-4208d8bc3bd5)
5.  [Building Interactive Trading Dashboards with Python](https://www.pyquantnews.com/free-python-resources/building-interactive-trading-dashboards-with-python)
6.  [Build a Dynamic Stock Data Dashboard: Visualizing Financial Performance with Python and Vectorbt](/pythoneers/build-a-dynamic-stock-data-dashboard-visualizing-financial-performance-with-python-and-vectorbt-95e72153629a)
7.  [Building A Simple Stock Screener Using Streamlit and Python Plotly Library](https://python.plainenglish.io/building-a-simple-stock-screener-using-streamlit-and-python-plotly-library-a6f04a2e40f9)
8.  [Building a Stock Price Dashboard with Streamlit, Python, and APIs](/@cameronjosephjones/building-a-stock-price-dashboard-with-streamlit-python-and-apis-bc57011758d4)
9.  [Build a Real-time Stock Price Dashboard With Python, QuestDB and Plotly](https://hackernoon.com/build-a-real-time-stock-price-dashboard-with-python-questdb-and-plotly)
10.  [Algorithmic Trading with Python](https://github.com/Nikhil-Adithyan/Algorithmic-Trading-with-Python)
11.  [Top 36 Moving Average Methods For Stock Prices in Python \[1/4\]](/@crisvelasquez/36-moving-average-methods-in-python-for-stock-price-analysis-1-4-4ce0c182093c)
12.  [How to Calculate Awesome Oscillator in Python](/@huzaifazahoor654/how-to-calculate-awesome-oscillator-in-python-de0973dbbdf1)

## Explore More

-   [Plotly Dash TA Stock Market App](/@alexzap922/plotly-dash-ta-stock-market-app-6cbcaa349cf3)
-   [Datapane Stock Screener App from Scratch](https://newdigitals.org/2023/05/24/datapane-stock-screener-api/)
-   [Advanced Integrated Data Visualization (AIDV) in Python — 1. Stock Technical Indicators](https://wp.me/pdMwZd-58A)
-   [The Donchian Channel vs Buy-and-Hold Breakout Trading Systems — $MO Use-Case](https://wp.me/pdMwZd-4Fd)
-   [A Comparative Analysis of The 3 Best U.S. Growth Stocks in Q1’23–1. WMT](https://wp.me/pdMwZd-4xZ)
-   [Python Technical Analysis for BioTech — Get Buy Alerts on ABBV in 2023](https://wp.me/pdMwZd-3Tq)
-   [Basic Stock Price Analysis in Python](https://wp.me/pdMwZd-Q2)
-   [Explore The Best Moving Averages For Big Tech Swing Trading — 1. NVDA](/insiderfinance/explore-the-best-moving-averages-for-big-tech-swing-trading-1-nvda-f0ddf3e00f5d)
-   [Optimized Keltner Channel Algo-Trading Strategy with Parameter Tuning: Energy Flagship XOM ROI Backtesting vs SPY ETF Benchmark](/@alexzap922/optimized-keltner-channel-algo-trading-strategy-with-parameter-tuning-energy-flagship-xom-roi-dec923c25e2d)
-   [NVDA vs BTC Algorithmic Trading: Backtest BB, MACD & AO Trading Strategies](/@alexzap922/nvda-vs-btc-algorithmic-trading-backtest-bb-macd-ao-trading-strategies-1db4f24d24ef)
-   [Thinking Outside the Box: Top 20 Plotly/Python Code Examples of Amazing Data Visualization That Will Blow Up Your Mind!](/python-in-plain-english/thinking-outside-the-box-top-20-plotly-python-code-examples-of-amazing-data-visualization-that-482fbf3c332b)
-   [Comparing Profitability of 4 Moving Average (MA) Algo-Trading Strategies in Python: A WMT Use Case](/@alexzap922/comparing-profitability-of-4-moving-average-ma-algo-trading-strategies-in-python-a-wmt-use-case-c0ab115fd470)

## Appendix A: Full Source Code

import pandas as pd  
import numpy as np  
from datetime import date  
import plotly.graph\_objects as go  
from plotly.subplots import make\_subplots  
from ta.trend import MACD  
from ta.momentum import RSIIndicator  
from ta.momentum import StochasticOscillator   
from dash import html, dcc, Dash  
from dash.dependencies import Input, Output, State  
  
  
import requests  
  
def get\_historical\_data(symbol, start\_date):  
    api\_key = 'a07d718849d64be78e8a7d5669e4e3af'  
    api\_url = f'https://api.twelvedata.com/time\_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api\_key}'  
    raw\_df = requests.get(api\_url).json()  
    df = pd.DataFrame(raw\_df\['values'\]).iloc\[::-1\].set\_index('datetime').astype(float)  
    df = df\[df.index >= start\_date\]  
    df.index = pd.to\_datetime(df.index)  
    return df  
  
def calculate\_awesome\_oscillator(df, short\_window=5, long\_window=34):  
    median\_price = (df\["High"\] + df\["Low"\]) / 2  
    short\_sma = median\_price.rolling(window=short\_window).mean()  
    long\_sma = median\_price.rolling(window=long\_window).mean()  
    AO = short\_sma - long\_sma  
    return AO  
  
def generate\_awesome\_oscillator\_color(AO):  
    awesome\_oscillator\_color = \[\]  
    awesome\_oscillator\_color.clear()  
    for i in range (0,len(AO)):  
        if AO\[i\] >= 0 and AO\[i-1\] < AO\[i\]:  
            awesome\_oscillator\_color.append('#26A69A')  
            #print(i,'green')  
        elif AO\[i\] >= 0 and AO\[i-1\] > AO\[i\]:  
            awesome\_oscillator\_color.append('#FF5252')  
            #print(i,'faint green')  
        elif AO\[i\] < 0 and AO\[i-1\] > AO\[i\] :  
            #print(i,'red')  
            awesome\_oscillator\_color.append('#FF5252')  
        elif AO\[i\] < 0 and AO\[i-1\] < AO\[i\] :  
            #print(i,'faint red')  
            awesome\_oscillator\_color.append('#26A69A')  
        else:  
            awesome\_oscillator\_color.append('#000000')  
    return awesome\_oscillator\_color  
  
def makeCandlestick(fig, stockDF):  
    #sets parameters for subplots  
    fig = make\_subplots(rows = 6, cols = 1, shared\_xaxes = True,  
                    vertical\_spacing = 0.01,  
                    row\_heights = \[0.6, 0.1, 0.15, 0.15,0.15,0.15\])  
  
  
    #plots candlestick values using stockDF  
    fig.add\_trace(go.Candlestick(x = stockDF.index,  
                                 open = stockDF\['Open'\],  
                                 high = stockDF\['High'\],  
                                 low = stockDF\['Low'\],  
                                 close = stockDF\['Close'\],  
                                 name = 'Open/Close'))  
      
    return fig  
def makeMA(fig, stockDF):  
    #create moving average values  
    stockDF\["MA5"\] = stockDF\["Close"\].ewm(span=5, adjust=False).mean()  
    stockDF\["MA15"\] = stockDF\["Close"\].ewm(span=15, adjust=False).mean()  
    stockDF\["MA50"\] = stockDF\["Close"\].ewm(span=50, adjust=False).mean()  
    stockDF\["MA100"\] = stockDF\["Close"\].ewm(span=100, adjust=False).mean()  
  
  
    #plots moving average values; the 50-day and 200-day averages  
    #are visible by default, and the 5-day and 15-day are accessed via legend  
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA5'\], opacity = 0.7,   
                        line = dict(color = 'blue', width = 2), name = 'MA 5'))  
              
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA15'\], opacity = 0.7,   
                        line = dict(color = 'orangered', width = 2), name = 'MA 15'))  
  
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA50'\], opacity = 0.7,  
                        line = dict(color = 'purple', width = 2), name = 'MA 50'))  
  
    fig.add\_trace(go.Scatter(x = stockDF.index, y = stockDF\['MA100'\], opacity = 0.7,  
                        line = dict(color = 'black', width = 2), name = 'MA 100'))  
  
    return fig  
  
def makeVolume(fig, stockDF):  
    #sets colors of volume bars  
    colors = \['green' if row\['Open'\] - row\['Close'\] >= 0  
          else 'red' for index, row in stockDF.iterrows()\]  
  
  
    #Plot volume trace  
    fig.add\_trace(go.Bar(x = stockDF.index,  
                         y = stockDF\['Volume'\],  
                         marker\_color = colors,  
                         showlegend = False,  
                         name = "Volume"  
                         ), row = 2, col = 1)  
  
    return fig  
def makeMACD(fig, stockDF):  
    #Create MACD values  
    macd = MACD(close = stockDF\["Close"\],  
                window\_slow = 26,  
                window\_fast = 12,  
                window\_sign = 9)  
  
    \# Stochastic  
    stoch = StochasticOscillator(high=stockDF\['High'\],  
                             close=stockDF\['Close'\],  
                             low=stockDF\['Low'\],  
                             window=14,   
                             smooth\_window=3)  
  
              
    #Sets color for MACD  
    colors = \['green' if val >= 0   
                      else 'red' for val in macd.macd\_diff()\]  
  
  
    #Plots MACD values  
    fig.add\_trace(go.Bar(x = stockDF.index,  
                         y = macd.macd\_diff(),  
                         marker\_color = colors,  
                         showlegend = False,  
                         name = "Histogram"  
                         ), row = 4, col = 1)  
  
  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = macd.macd(),  
                             line = dict(color = 'red', width = 1),  
                             showlegend = False,  
                             name = "MACD"  
                             ), row = 4, col = 1)  
  
  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = macd.macd\_signal(),  
                             line = dict(color = 'blue', width = 2),  
                             showlegend = False,  
                             name = "Signal"  
                             ), row = 4, col = 1)  
    \# Plot stochastics trace on 4th row  
    fig.add\_trace(go.Scatter(x=stockDF.index,  
                         y=stoch.stoch(),  
                         line=dict(color='black', width=2),name = "% K"  
                        ), row=5, col=1)  
    fig.add\_trace(go.Scatter(x=stockDF.index,  
                         y=stoch.stoch\_signal(),  
                         line=dict(color='blue', width=1),name = "% D"  
                        ), row=5, col=1)  
  
  
    #########################  
  
    return fig  
  
#https://github.com/matplotlib/mplfinance/blob/master/examples/indicators/awesome\_oscillator.ipynb  
  
def makeOA(fig, stockDF):  
      
    AO = calculate\_awesome\_oscillator(stockDF,5,34)  
  
    \# List of Color Assiging To Awesome Oscillator  
    awesome\_oscillator\_color = generate\_awesome\_oscillator\_color(AO)  
    \# Data Extracted And New Variable Applied  
    awesome\_oscillator = AO  
  
              
    #Sets color for AO  
    colors = awesome\_oscillator\_color  
  
  
    #Plots AO values  
    fig.add\_trace(go.Bar(x = stockDF.index,  
                         y = AO,  
                         marker\_color = colors,  
                         showlegend = False,  
                         name = "Histogram"  
                         ), row = 6, col = 1)  
  
    return fig  
  
def makeRSI(fig, stockDF):  
    #Create RSI values  
    rsi = RSIIndicator(close = stockDF\["Close"\],  
                       window = 14)  
  
  
    #Plots RSI values  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = rsi.rsi(),  
                             line = dict(color = 'black', width = 2),  
                             showlegend = False,  
                             name = "RSI"  
                             ), row = 3, col = 1)  
  
  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[30 for val in range(len(stockDF))\],  
                             line = dict(color = 'red', width = 1),  
                             showlegend = False,  
                             name = "Oversold"  
                             ), row = 3, col = 1)  
  
  
    fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[70 for val in range(len(stockDF))\],  
                             line = dict(color = 'green', width = 1),  
                             showlegend = False,  
                             name = "Overbought"  
                             ), row = 3, col = 1)  
  
  
    return fig  
  
def makeCurrentPrice(fig, stockDF):  
    #Plots the last closing price of stock   
    fig.add\_trace(go.Scatter(x = stockDF.index,  
              y = \[stockDF\['Close'\].iat\[-1\] for price in range(len(stockDF))\],  
              opacity = 0.7, line = dict(color = 'red', width = 2, dash = 'dot'),  
              name = "Current Price: " + str(round(stockDF\['Close'\].iat\[-1\], 2))))  
  
    return fig  
  
  
def supportLevel(stockDF, index):  
    #Finds and returns support levels using fractals;  
    #if there are two higher lows on each side of the current stockDF\['Low'\] value,   
    #return this value  
    support = stockDF\['Low'\]\[index\] < stockDF\['Low'\]\[index - 1\] and \\  
              stockDF\['Low'\]\[index\] < stockDF\['Low'\]\[index + 1\] and \\  
              stockDF\['Low'\]\[index + 1\] < stockDF\['Low'\]\[index + 2\] and \\  
              stockDF\['Low'\]\[index - 1\] < stockDF\['Low'\]\[index - 2\]  
  
    return support  
  
  
def resistanceLevel(stockDF, index):  
    #Finds and returns resistance levels using fractals;  
    #If there are two lower highs on each side of the current stock\['High'\] value,  
    #return this value  
    resistance = stockDF\['High'\]\[index\] > stockDF\['High'\]\[index - 1\] and \\  
              stockDF\['High'\]\[index\] > stockDF\['High'\]\[index + 1\] and \\  
              stockDF\['High'\]\[index + 1\] > stockDF\['High'\]\[index + 2\] and \\  
              stockDF\['High'\]\[index - 1\] > stockDF\['High'\]\[index - 2\]  
  
    return resistance  
  
def isFarFromLevel(stockDF, level, levels):  
    #If a level is found near another level, it returns false;  
  
    ##.88 for longer term .97 for short term  
    s = np.mean(stockDF\['High'\] - (stockDF\['Low'\] \* .89))  
  
    return np.sum(\[abs(level - x) < s for x in levels\]) == 0  
  
  
def makeLevels(fig, stockDF):  
    #Traverses through stockDF and finds key support/resistance levels  
    levels = \[\]  
    for index in range(2, stockDF.shape\[0\] - 2):  
        if supportLevel(stockDF, index):  
            support = stockDF\['Low'\]\[index\]  
            if isFarFromLevel(stockDF, support, levels):  
                levels.append((support))  
              
        elif resistanceLevel(stockDF, index):  
            resistance = stockDF\['High'\]\[index\]  
            if isFarFromLevel(stockDF, resistance, levels):  
                levels.append((resistance))  
  
    levels.sort()  
  
    #Plots the key levels within levels   
    for i in range(len(levels)):  
        fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[levels\[i\] for val in range(len(stockDF))\],  
                             line = dict(color = "black"),  
                             name = "Sup/Res: " + str(round(levels\[i\], 2)),  
                             hoverinfo = "skip",  
                             opacity = 0.3))  
  
    return fig  
  
def findAbsMax(stockDF):  
    absMax = 0  
    for i in range(len(stockDF)):  
        if stockDF\["Close"\]\[i\] > absMax:  
            absMax = stockDF\["Close"\]\[i\]  
          
    return absMax  
  
  
def findAbsLow(stockDF):  
    absLow = 50  
    for i in range(len(stockDF)):  
        if stockDF\["Close"\]\[i\] < absLow:  
            absLow = stockDF\["Close"\]\[i\]  
  
    return absLow  
  
  
def makeFibLevels(fig, stockDF):  
    fibRatios = \[.236, .382, .5, .618, .786, 1\]  
    fibLevels = \[\]  
    absMax = findAbsMax(stockDF)  
    absLow = findAbsLow(stockDF)  
    dif = absMax - absLow  
  
    for i in range(len(fibRatios)):  
        fibLevels.append(dif \* fibRatios\[i\])  
#for prices that are above the last resistance/support line within fibLevels,  
    #look to see if there can be any levels drawn using fractals that are also not   
    #too close to the current last support/resistance;  
    #We really are just looking for the last resistance level;  
    fractal = fibLevels\[-1\] + (fibLevels\[-1\] \* .17)  
    if (fibLevels\[-1\] < fractal) and (fractal < absMax):  
         fibLevels.append(fractal)  
  
    for i in range(len(fibLevels)):  
        fig.add\_trace(go.Scatter(x = stockDF.index,  
                             y = \[fibLevels\[i\] for val in range(len(stockDF))\],  
                             line = dict(color = "black"),  
                             name = "Sup/Res: " + str(round(fibLevels\[i\], 2)),  
                             hoverinfo = "skip",  
                             opacity = 0.3))  
      
    return fig  
  
  
def graphLayout(fig, choice):  
    #Sets the layout of the graph and legend  
    fig.update\_layout(title\_text = choice + ' Price Action',   
                  title\_x = 0.5,   
                  legend\_title\_text = "Legend Items",  
                  dragmode = "pan",   
                  xaxis\_rangeslider\_visible = False,   
                  hovermode = "x",   
                  legend = dict(bgcolor="#E2E2E2",  
                           bordercolor="Black",  
                           borderwidth=2)  
                                 
                 )  
  
    subplotLabels(fig)  
  
    return fig  
  
def subplotLabels(fig):  
    #Sets subplot labels  
    fig.update\_yaxes(title\_text = "Price", row = 1, col = 1)  
    fig.update\_yaxes(title\_text = "Volume", row = 2, col = 1)  
    fig.update\_yaxes(title\_text = "RSI", row = 3, col = 1)  
    fig.update\_yaxes(title\_text = "MACD", showgrid = False, row = 4, col = 1)  
    fig.update\_yaxes(title\_text = "Stoch", showgrid = False, row = 5, col = 1)  
    fig.update\_yaxes(title\_text = "AO", showgrid = False, row = 6, col = 1)  
  
    return fig  
  
  
def xAxes(fig):  
    #Remove none trading days from dataset and sets behavior for x-axis mouse-hovering  
    fig.update\_xaxes(rangebreaks = \[dict(bounds = \["sat", "mon"\])\],   
                 autorange = True,   
                 showspikes = True,   
                 spikedash = "dot",  
                 spikethickness = 1,   
                 spikemode = "across",   
                 spikecolor = "black")  
      
    return fig  
  
  
  
fig = go.Figure()  
config = dict({'scrollZoom': True})  
  
stockApp = Dash(\_\_name\_\_, meta\_tags=\[{'name': 'viewport',   
                       'content':'width=device-width, initial-scale=1.0'}\])  
  
application = stockApp.server  
  
stockApp.layout = html.Div(\[  
    dcc.Graph(figure = fig, config = config,  
  
              style = {'width': '99vw', 'height': '93vh'},  
              id = "stockGraph"  
             ),  
  
             html.Div(\[  
                dcc.Input(  
                    id = "userInput",  
                    type = "text",  
                    placeholder = "Ticker Symbol"  
                         ),  
              
            html.Button("Submit", id = "btnSubmit")\]),  
                      \],  
            )  
  
@stockApp.callback(    Output("stockGraph", "figure"),  
    Input("btnSubmit", "n\_clicks"),  
    State("userInput", "value"))  
  
def update\_figure(n, tickerChoice):  
    #set choice to something if !isPostBack  
    if tickerChoice == None:  
        tickerChoice = 'AAPL'  
  
  
    #make stockDF      
    #today = date.today()  
    #stockDF = yf.download(tickerChoice, start = '2020-01-01', end = today )  
    aapl = get\_historical\_data(tickerChoice, '2024-03-01')  
    stockDF = aapl.rename(columns={'open': 'Open', 'high': 'High','low': 'Low', 'close': 'Close','volume': 'Volume'})  
  
    #make go Figure object as fig  
    fig = go.Figure()  
  
    #make and plot candlestick chart  
    fig = makeCandlestick(fig, stockDF)  
  
    #update layout properties  
    fig = graphLayout(fig, tickerChoice.upper())  
  
    #updates x-axis parameters  
    fig = xAxes(fig)  
  
    #make and plot subplots charts and moving averages  
    fig = makeMA(fig, stockDF)  
    fig = makeVolume(fig, stockDF)  
    fig = makeMACD(fig, stockDF)  
    fig = makeRSI(fig, stockDF)  
    fig = makeOA(fig, stockDF)  
  
    #make and plot stock's last closing price  
    fig = makeCurrentPrice(fig, stockDF)  
  
    #make and plot stock's resistance/support values using fibonacci retracement  
    fig = makeFibLevels(fig, stockDF)  
  
      
    return fig  
  
if \_\_name\_\_ == '\_\_main\_\_':  
    application.run(debug = False, port = 8080)

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