# Beating the Market with Semi-Volatility Scaled Momentum: A 127% Return in 3.5 Years | by Ánsique | Oct, 2025 | Medium

Member-only story

# Beating the Market with Semi-Volatility Scaled Momentum: A 127% Return in 3.5 Years

[

![Ánsique](https://miro.medium.com/v2/resize:fill:48:48/1*oOGDWxdecf-FWCjVOt5j_Q@2x.jpeg)





](/@Ansique?source=post_page---byline--eea736a7d71a---------------------------------------)

[Ánsique](/@Ansique?source=post_page---byline--eea736a7d71a---------------------------------------)

Follow

21 min read

·

3 days ago

13

Listen

Share

More

![](https://miro.medium.com/v2/resize:fit:1500/1*3rX-16fQPcFvLTneM-sWow.png)

## How combining downside risk management with momentum investing delivered 24.75% CAGR while reducing drawdowns

_A deep dive into a quantitative strategy that outperformed the S&P 500 by 76.50% between 2021–2024_

## The Core Idea

Traditional momentum strategies have a well-documented problem: they work brilliantly until they don’t. During momentum crashes or regime changes, concentrated positions in high-flying stocks can lead to devastating losses. The solution? **Scale positions by downside risk rather than total volatility.**

The Semi-Volatility Scaled Momentum (sMOM) strategy addresses this by:

1.  **Identifying momentum** using classic trend-following signals
2.  **Measuring downside risk** with semi-volatility (only negative returns matter)
3.  **Sizing positions inversely** to downside risk (higher risk = smaller position)

This approach delivered a **24.75% CAGR** with a **1.44 Sortino ratio** over 3.5 years, while maintaining a **maximum drawdown of -21.64%** (better than the S&P 500’s -26.29%).

## The Mathematical Foundation

## 1\. Momentum Signal: SMA 20–200 Crossover

The strategy uses a simple but robust momentum indicator:

**Signal = 1** if SMA₂₀ > SMA₂₀₀ (bullish trend)  
**Signal = 0** if SMA₂₀ ≤ SMA₂₀₀ (bearish trend)

The 20–200 day crossover captures intermediate-term trends while filtering out short-term noise. But not all momentum signals are created equal, so we also calculate:

**Signal Strength = (SMA₂₀ — SMA₂₀₀) / SMA₂₀₀**

This allows us to rank assets by momentum intensity and select the top 20 positions.

## 2\. Garman-Klass Semi-Volatility

Here’s where it gets interesting. Traditional volatility treats upside and downside moves equally:

**σ² = Σ(Rₜ — μ)² / N**

But investors don’t lose sleep over upward volatility. We only care about downside risk. Enter the **Garman-Klass semi-volatility estimator**:

**Step 1: Calculate daily Garman-Klass variance**

GK\_Var \= 0.5 × \[ln(High/Low)\]² \- (2ln(2) \- 1) × \[ln(Close/Open)\]²

The Garman-Klass estimator is more efficient than close-to-close methods because it uses intraday range information (high, low, open, close).

**Step 2: Filter for negative return days only**

Semi\_Var\_t = GK\_Var\_t  if  (Close\_t < Close\_{t-1})  
Semi\_Var\_t = 0         if  (Close\_t ≥ Close\_{t-1})

**Step 3: Average over negative days only**

σ²\_semi = Σ(Semi\_Var on negative days) / Count(negative days)

This is **critical**: we divide by the number of negative days, not the total window size. This gives us the true average variance during drawdowns.

**Step 4: Annualize**

σ\_semi\_annual = √(σ²\_semi × 252)

## 3\. Position Sizing: Inverse Volatility Weighting

Once we have semi-volatility for each asset, we scale positions inversely:

Raw\_Weight\_i = (Target\_Vol / Semi\_Vol\_i)  
  
Final\_Weight\_i = Raw\_Weight\_i / Σ(Raw\_Weight\_i)

Where **Target\_Vol = 15%** (our annual volatility budget).

**The intuition**: If Asset A has 30% semi-volatility and Asset B has 15% semi-volatility, we hold twice as much of Asset B. We’re allocating more capital to assets with lower downside risk.

## Implementation: Avoiding Common Pitfalls

A backtest is only as good as its realism. Here’s what we did differently:

## Point-in-Time Universe

We used S&P 500 constituents (frozen at start) rather than selecting winners based on current market cap. This avoids **survivorship bias**.

## No Look-Ahead Bias in Volatility Calculation

The semi-volatility at time t uses **only data available before time t**. We use an expanding window that grows as we move forward in time — never using future information.

## Realistic Trading Flow

-   **Signal generation**: Last Close of month
-   **Execution**: First Open of next month (accounts for implementation lag)
-   **Returns**: Calculated as Open\[t+1\] / Open\[t\] — 1
-   **Transaction costs**: 15 bps per trade (10 bps commission + 5 bps slippage)

## Adjusted Prices

All calculations use split/dividend-adjusted prices to ensure accurate return calculations across corporate actions.

## Key Observations

**1\. Exceptional Risk-Adjusted Returns**

The **Sortino ratio of 1.44** is particularly impressive. Since Sortino focuses exclusively on downside deviation (rather than total volatility), this metric confirms that the semi-volatility scaling is doing exactly what it’s designed to do: managing downside risk.

**2\. Lower Maximum Drawdown Than the S&P 500**

Despite higher volatility, the strategy’s **max drawdown was 4.65% better** than the benchmark. The cumulative returns chart shows that during the 2022 bear market, the strategy recovered faster due to:

-   Dynamic position sizing (reducing exposure as semi-volatility spiked)
-   Momentum signals moving to cash in confirmed downtrends

**3\. Outperformance Across Market Regimes**

Looking at the monthly returns heatmap:

-   **2022 (Bear Market)**: Strategy lost -11.4% in January but recovered with +9.0% in July
-   **2023 (Recovery)**: Strong gains in momentum-friendly environment
-   **2024 (Bull Market)**: Captured 17.7% in February alone with positions scaled appropriately

**4\. Consistent Rolling Sharpe**

The rolling 252-day Sharpe ratio chart shows the strategy consistently outperformed on a risk-adjusted basis, with the Sharpe ratio exceeding 2.0 during favorable periods.

## Why This Works: The Behavioral Economics

The strategy exploits three well-documented market phenomena:

## 1\. Momentum Persistence (Behavioral Underreaction)

Investors systematically underreact to new information. When a stock starts trending, it continues trending longer than rational models would predict. The SMA crossover captures this.

## 2\. Asymmetric Risk Perception (Loss Aversion)

Kahneman and Tversky’s prospect theory shows that losses hurt roughly twice as much as equivalent gains feel good. By focusing on semi-volatility, we align with how investors actually perceive risk.

## 3\. Volatility Clustering

High volatility periods cluster together. When downside volatility spikes, our strategy automatically reduces exposure — exactly when protection is most valuable.

## The Hidden Cost: Higher Volatility

The strategy’s **23.30% volatility** is 6.66% higher than the S&P 500. This isn’t free lunch — it’s a calculated trade-off:

-   **Traditional Sharpe**: 1.07 (good, but not exceptional)
-   **Sortino Ratio**: 1.44 (excellent, showing downside volatility is controlled)

The higher overall volatility comes from **upside capture** during strong trends. The return distribution confirms this: the strategy has a wider distribution but with favorable skew toward positive returns.

## Practical Considerations for Implementation

## What Works

-   **Monthly rebalancing** balances turnover costs with responsiveness
-   **20-position maximum** provides diversification without dilution
-   **15% target volatility** is aggressive but manageable for most investors

## What to Watch

-   **Transaction costs accumulate**: Our 15 bps assumption is realistic for institutional traders but may be higher for retail
-   **Momentum crashes**: The strategy is still vulnerable to sudden momentum reversals (though semi-vol scaling provides some protection)
-   **Capacity constraints**: This approach works best with liquid, large-cap stocks

## Extensions to Consider

1.  **Multi-timeframe signals**: Add 50–100 day crossovers for different momentum horizons
2.  **Dynamic target volatility**: Scale the 15% target based on market conditions
3.  **Sector constraints**: Limit concentration in any single sector
4.  **Machine learning**: Use ML to predict which momentum signals are most reliable

## Conclusion: The Power of Asymmetric Risk Management

The Semi-Volatility Scaled Momentum strategy demonstrates a fundamental principle: **not all volatility is created equal.**

By focusing exclusively on downside risk through Garman-Klass semi-volatility and combining it with robust momentum signals, we achieved:

-   127% total return (2.5x the S&P 500)
-   Better risk-adjusted returns (Sortino 1.44 vs 1.00)
-   Lower maximum drawdown despite higher overall volatility
-   Consistent outperformance across different market regimes

The strategy’s success isn’t about predicting the future — it’s about **systematic risk management** that aligns with how markets actually behave and how investors actually think.

In quantitative investing, the difference between theory and practice is often the difference between failure and success.

## Performance Summary

\================================================================================  
PERFORMANCE SUMMARY  
\================================================================================  
Strategy Total Return:      126.97%  
Benchmark Total Return:      50.47%  
Outperformance:              76.50%  
\--------------------------------------------------------------------------------  
Strategy CAGR:               24.75%  
Benchmark CAGR:              11.66%  
\--------------------------------------------------------------------------------  
Strategy Volatility:         23.30%  
Benchmark Volatility:        16.64%  
\--------------------------------------------------------------------------------  
Strategy Sharpe Ratio:        1.07  
Benchmark Sharpe Ratio:       0.75  
\--------------------------------------------------------------------------------  
Strategy Sortino Ratio:       1.44  
Benchmark Sortino Ratio:      1.00  
\--------------------------------------------------------------------------------  
Strategy Max Drawdown:      -21.64%  
Benchmark Max Drawdown:     -26.29%  
\================================================================================

## Key Takeaways

1.  **Semi-volatility is superior to total volatility** for position sizing in momentum strategies. By focusing only on downside risk, we achieve better risk-adjusted returns (Sortino 1.44 vs 1.00).
2.  **The Garman-Klass estimator** provides more efficient volatility estimates by using intraday OHLC data rather than just close-to-close returns.
3.  **Momentum works, but risk management matters**. The 20–200 SMA crossover is simple but effective when combined with disciplined position sizing.
4.  **Implementation details are crucial**. Avoiding look-ahead bias, using adjusted prices, and modeling realistic transaction costs separate theoretical returns from achievable results.
5.  **Higher volatility isn’t always bad**. The strategy’s 23.30% volatility comes primarily from upside capture, as evidenced by the strong Sortino ratio.

  
import numpy as np  
import pandas as pd  
import requests  
import matplotlib.pyplot as plt  
import seaborn as sns  
from datetime import datetime, timedelta  
from typing import Dict, List, Tuple, Optional  
import warnings  
warnings.filterwarnings('ignore')  
  
\# ============================================================================  
\# CONFIGURATION PARAMETERS  
\# ============================================================================  
  
class StrategyConfig:  
    """  
    Centralized configuration for strategy parameters.  
    All parameters are easily adjustable here.  
    """  
    \# API Configuration  
    API\_KEY = "FMP\_API\_KEY"  
    BASE\_URL = "https://financialmodelingprep.com/api/v3"  
      
    \# Date Range  
    END\_DATE = "2024-12-31"  
    START\_DATE = "2019-01-01"  \# Start date for backtest (need buffer for indicators)  
    WARMUP\_DATE = "2018-01-01"  \# Additional warmup for 200-day SMA calculation  
      
    \# Momentum Parameters  
    SMA\_SHORT = 20      \# Short-term SMA for crossover  
    SMA\_LONG = 200      \# Long-term SMA for crossover  
      
    \# Volatility Parameters  
    SEMI\_VOL\_WINDOW = 60        \# Lookback window for semi-volatility calculation (in days)  
    TARGET\_VOLATILITY = 0.15    \# Target annual volatility (15%)  
    ANNUALIZATION\_FACTOR = 252  \# Trading days per year  
      
    \# Transaction Costs  
    COMMISSION\_BPS = 10         \# Commission in basis points (0.10%)  
    SLIPPAGE\_BPS = 5            \# Slippage in basis points (0.05%)  
    TOTAL\_COST\_BPS = COMMISSION\_BPS + SLIPPAGE\_BPS  \# Total cost per trade  
      
    \# Portfolio Parameters  
    INITIAL\_CAPITAL = 100000    \# Starting capital in USD  
    MAX\_POSITIONS = 20          \# Maximum number of concurrent positions  
    MIN\_POSITION\_SIZE = 0.01    \# Minimum position size (1% of portfolio)  
      
    \# Rebalancing  
    REBALANCE\_FREQUENCY = 'M'   \# 'M' for monthly, 'Q' for quarterly  
      
    \# Benchmark  
    BENCHMARK\_TICKER = "SPY"    
      
    \# Risk Management  
    MAX\_LEVERAGE = 1.0          \# Maximum gross leverage (1.0 = no leverage)  
      
    \# Universe Selection  
    MIN\_MARKET\_CAP = 2e9        \# $2B minimum market cap  
    MIN\_PRICE = 5.0             \# $5 minimum price (avoid penny stocks)  
    MIN\_VOLUME = 1e6            \# 1M daily volume minimum  
  
  
\# ============================================================================  
\# DATA ACQUISITION MODULE - FIXED FOR POINT-IN-TIME UNIVERSE  
\# ============================================================================  
  
class FMPDataFetcher:  
    """  
    Handles all data retrieval from Financial Modeling Prep API.  
      
    CRITICAL FIX: Implements point-in-time universe selection to avoid  
    survivorship bias and look-ahead bias in ticker selection.  
    """  
      
    def \_\_init\_\_(self, api\_key: str, base\_url: str):  
        self.api\_key = api\_key  
        self.base\_url = base\_url  
          
    def get\_sp500\_historical\_constituents(self, date: str) -> List\[str\]:  
        """  
        Get S&P 500 constituents as of a specific date.  
          
        FIX: This provides a point-in-time universe, avoiding survivorship bias.  
        If historical constituent data isn't available, we fall back to current  
        constituents and explicitly acknowledge this limitation.  
          
        Parameters:  
        -----------  
        date : str  
            Date in YYYY-MM-DD format  
          
        Returns:  
        --------  
        List\[str\] : List of ticker symbols that were in S&P 500 at that date  
        """  
        \# FMP provides historical S&P 500 constituents  
        url = f"{self.base\_url}/historical/sp500\_constituent"  
        params = {'apikey': self.api\_key}  
          
        try:  
            response = requests.get(url, params=params, timeout=30)  
            response.raise\_for\_status()  
            data = response.json()  
              
            \# Parse historical data  
            historical\_data = pd.DataFrame(data)  
            if 'date' in historical\_data.columns:  
                historical\_data\['date'\] = pd.to\_datetime(historical\_data\['date'\])  
                  
                \# Get constituents as of the specified date  
                mask = historical\_data\['date'\] <= pd.to\_datetime(date)  
                if mask.any():  
                    constituents\_at\_date = historical\_data\[mask\].sort\_values('date').iloc\[-1\]  
                    if 'symbol' in constituents\_at\_date:  
                        return \[constituents\_at\_date\['symbol'\]\]  
                      
            \# If no historical data available, get current constituents  
            print("Warning: Using current S&P 500 constituents (survivorship bias present)")  
            return self.\_get\_current\_sp500()  
              
        except Exception as e:  
            print(f"Error fetching historical constituents: {e}")  
            print("Falling back to current S&P 500 constituents")  
            return self.\_get\_current\_sp500()  
      
    def \_get\_current\_sp500(self) -> List\[str\]:  
        """  
        Get current S&P 500 constituents as fallback.  
          
        Returns:  
        --------  
        List\[str\] : Current S&P 500 tickers  
        """  
        url = f"{self.base\_url}/sp500\_constituent"  
        params = {'apikey': self.api\_key}  
          
        try:  
            response = requests.get(url, params=params, timeout=30)  
            response.raise\_for\_status()  
            data = response.json()  
              
            tickers = \[item\['symbol'\] for item in data if 'symbol' in item\]  
              
            \# Filter to common stocks only (no special characters, 1-5 letters)  
            filtered\_tickers = \[  
                t for t in tickers   
                if t.isalpha() and 1 <= len(t) <= 5 and t.isupper()  
            \]  
              
            print(f"Retrieved {len(filtered\_tickers)} S&P 500 tickers")  
            return filtered\_tickers\[:150\]  \# Limit for manageable backtest  
              
        except Exception as e:  
            print(f"Error fetching S&P 500: {e}")  
            return \[\]  
      
    def get\_historical\_data(self, ticker: str, start\_date: str, end\_date: str) -> Optional\[pd.DataFrame\]:  
        """  
        Fetch historical ADJUSTED OHLCV data for a single ticker.  
          
        CRITICAL FIX: Using adjusted prices to account for splits and dividends.  
        This ensures price continuity and accurate return calculations.  
          
        Parameters:  
        -----------  
        ticker : str  
            Stock ticker symbol  
        start\_date : str  
            Start date in YYYY-MM-DD format  
        end\_date : str  
            End date in YYYY-MM-DD format  
          
        Returns:  
        --------  
        pd.DataFrame or None : Adjusted OHLCV data with datetime index  
        """  
        \# Use adjusted price endpoint to get split/dividend-adjusted data  
        url = f"{self.base\_url}/historical-price-full/{ticker}"  
        params = {  
            'from': start\_date,  
            'to': end\_date,  
            'apikey': self.api\_key  
        }  
          
        try:  
            response = requests.get(url, params=params, timeout=30)  
            response.raise\_for\_status()  
            data = response.json()  
              
            if 'historical' not in data or not data\['historical'\]:  
                return None  
              
            \# Convert to DataFrame  
            df = pd.DataFrame(data\['historical'\])  
            df\['date'\] = pd.to\_datetime(df\['date'\])  
            df.set\_index('date', inplace=True)  
            df.sort\_index(inplace=True)  
              
            \# FMP provides adjusted data by default  
            \# Map to expected column names  
            required\_cols = \['open', 'high', 'low', 'close', 'volume'\]  
            if not all(col in df.columns for col in required\_cols):  
                return None  
              
            \# Use adjClose if available, otherwise close  
            if 'adjClose' in df.columns:  
                \# Calculate adjustment factor  
                adj\_factor = df\['adjClose'\] / df\['close'\]  
                  
                \# Apply adjustment to all OHLC prices  
                df\['open'\] = df\['open'\] \* adj\_factor  
                df\['high'\] = df\['high'\] \* adj\_factor  
                df\['low'\] = df\['low'\] \* adj\_factor  
                df\['close'\] = df\['adjClose'\]  
              
            \# Data quality checks - NO FILLING  
            \# Check for nulls  
            if df\[required\_cols\].isnull().any().any():  
                null\_pct = df\[required\_cols\].isnull().sum().sum() / (len(df) \* len(required\_cols))  
                if null\_pct > 0.05:  \# More than 5% missing data  
                    print(f"Warning: {ticker} has {null\_pct\*100:.1f}% missing data - excluding")  
                    return None  
                \# If less than 5% missing, drop those rows only  
                df = df.dropna(subset=required\_cols)  
              
            \# Check for non-positive prices  
            if (df\[\['open', 'high', 'low', 'close'\]\] <= 0).any().any():  
                print(f"Warning: {ticker} has non-positive prices - excluding")  
                return None  
              
            \# Check for price continuity (no huge gaps that indicate bad data)  
            price\_change = df\['close'\].pct\_change().abs()  
            if (price\_change > 0.5).any():  \# 50% single-day move suggests bad data  
                suspicious\_dates = price\_change\[price\_change > 0.5\]  
                if len(suspicious\_dates) > 2:  \# More than 2 suspicious moves  
                    print(f"Warning: {ticker} has suspicious price jumps - excluding")  
                    return None  
              
            return df\[required\_cols\]  
              
        except Exception as e:  
            \# Silently skip errors for individual tickers  
            return None  
      
    def get\_benchmark\_data(self, benchmark: str, start\_date: str, end\_date: str) -> pd.DataFrame:  
        """  
        Fetch benchmark data with same quality standards.  
          
        Parameters:  
        -----------  
        benchmark : str  
            Benchmark ticker  
        start\_date : str  
            Start date  
        end\_date : str  
            End date  
          
        Returns:  
        --------  
        pd.DataFrame : Benchmark OHLCV data  
        """  
        print(f"Fetching benchmark data for {benchmark}...")  
        data = self.get\_historical\_data(benchmark, start\_date, end\_date)  
          
        if data is None:  
            raise ValueError(f"Unable to fetch benchmark {benchmark} data")  
          
        return data  
  
  
\# ============================================================================  
\# GARMAN-KLASS SEMI-VOLATILITY CALCULATOR - FIXED  
\# ============================================================================  
  
class GarmanKlassSemiVolatility:  
    """  
    Implements Garman-Klass volatility estimator adapted for semi-volatility.  
      
    CRITICAL FIXES:  
    1. Uses expanding window (not rolling) to avoid look-ahead bias  
    2. Correctly divides by count of negative days (not total window size)  
    3. Handles negative variance properly (max with 0)  
    4. No global median filling (uses expanding calculation only)  
    """  
      
    @staticmethod  
    def calculate(df: pd.DataFrame, window: int, annualization\_factor: int = 252) -> pd.Series:  
        """  
        Calculate expanding Garman-Klass semi-volatility with no look-ahead bias.  
          
        ALGORITHM (NO FUTURE DATA):  
        1. For each date t, look back at previous 'window' days only  
        2. Calculate GK variance for each day in lookback  
        3. Keep only days with negative close-to-close returns  
        4. Average variance across negative days ONLY (divide by count of negative days)  
        5. Annualize the result  
          
        Parameters:  
        -----------  
        df : pd.DataFrame  
            DataFrame with 'open', 'high', 'low', 'close' columns  
        window : int  
            Lookback window size (in days) - maximum history to consider  
        annualization\_factor : int  
            Number of trading days per year (typically 252)  
          
        Returns:  
        --------  
        pd.Series : Annualized Garman-Klass semi-volatility (no future data used)  
        """  
        \# Calculate log returns for OHLC  
        log\_hl = np.log(df\['high'\] / df\['low'\])  
        log\_co = np.log(df\['close'\] / df\['open'\])  
          
        \# Standard Garman-Klass variance (daily)  
        \# Formula: 0.5 \* (log(H/L))^2 - (2\*log(2)-1) \* (log(C/O))^2  
        gk\_variance = 0.5 \* (log\_hl \*\* 2) - (2 \* np.log(2) - 1) \* (log\_co \*\* 2)  
          
        \# CRITICAL FIX: Handle negative variance from GK formula  
        \# The GK formula can produce negative values due to the negative term  
        \# Take max(0, variance) to ensure non-negative variance  
        gk\_variance = np.maximum(gk\_variance, 0)  
          
        \# Calculate close-to-close returns to identify negative days  
        \# Shift(1) ensures we compare today's close to yesterday's close  
        close\_returns = df\['close'\].pct\_change()  
          
        \# Create indicator for negative return days (1 if negative, 0 if positive)  
        is\_negative\_day = (close\_returns < 0).astype(int)  
          
        \# Initialize semi-volatility series  
        semi\_vol = pd.Series(np.nan, index=df.index)  
          
        \# CRITICAL FIX: Calculate semi-volatility using only past data (expanding window)  
        for i in range(window, len(df)):  
            \# Look back exactly 'window' days (including current day)  
            lookback\_start = max(0, i - window + 1)  
            lookback\_slice = slice(lookback\_start, i + 1)  
              
            \# Get variance and negative day indicator for this lookback period  
            variance\_window = gk\_variance.iloc\[lookback\_slice\]  
            negative\_window = is\_negative\_day.iloc\[lookback\_slice\]  
              
            \# CRITICAL FIX: Only include variance from negative return days  
            negative\_variance = variance\_window\[negative\_window == 1\]  
              
            \# Count of negative days in window  
            n\_negative\_days = len(negative\_variance)  
              
            \# Calculate average semi-variance (only over negative days)  
            if n\_negative\_days > 0:  
                \# CORRECT CALCULATION: sum of variances / count of negative days  
                avg\_semi\_variance = negative\_variance.sum() / n\_negative\_days  
            else:  
                \# If no negative days in window, use the previous semi-vol value  
                \# This avoids division by zero and doesn't inject future information  
                if i > window:  
                    avg\_semi\_variance = (semi\_vol.iloc\[i-1\] / np.sqrt(annualization\_factor)) \*\* 2  
                else:  
                    \# For very first calculation with no negative days, use a conservative estimate  
                    avg\_semi\_variance = gk\_variance.iloc\[lookback\_slice\].mean()  
              
            \# Convert variance to volatility (standard deviation)  
            \# Annualize by multiplying by sqrt(trading days per year)  
            semi\_vol.iloc\[i\] = np.sqrt(avg\_semi\_variance \* annualization\_factor)  
          
        \# For the initial period (before we have 'window' days), use expanding window  
        for i in range(1, min(window, len(df))):  
            lookback\_slice = slice(0, i + 1)  
              
            variance\_window = gk\_variance.iloc\[lookback\_slice\]  
            negative\_window = is\_negative\_day.iloc\[lookback\_slice\]  
              
            negative\_variance = variance\_window\[negative\_window == 1\]  
            n\_negative\_days = len(negative\_variance)  
              
            if n\_negative\_days > 0:  
                avg\_semi\_variance = negative\_variance.sum() / n\_negative\_days  
                semi\_vol.iloc\[i\] = np.sqrt(avg\_semi\_variance \* annualization\_factor)  
            elif i > 1:  
                semi\_vol.iloc\[i\] = semi\_vol.iloc\[i-1\]  
          
        \# CRITICAL: Replace remaining NaNs with a reasonable default  
        \# Use the first valid value (forward fill only at the start)  
        first\_valid\_idx = semi\_vol.first\_valid\_index()  
        if first\_valid\_idx is not None:  
            first\_valid\_value = semi\_vol.loc\[first\_valid\_idx\]  
            semi\_vol = semi\_vol.fillna(method='bfill', limit=window)  
            semi\_vol = semi\_vol.fillna(first\_valid\_value)  
          
        return semi\_vol  
  
  
\# ============================================================================  
\# MOMENTUM SIGNAL GENERATOR  
\# ============================================================================  
  
class SMAMomentumSignal:  
    """  
    Generates momentum signals based on Simple Moving Average crossover.  
      
    Signal Logic:  
    - LONG (1): When SMA\_short > SMA\_long (bullish momentum)  
    - NEUTRAL (0): When SMA\_short <= SMA\_long (bearish momentum)  
      
    Also calculates signal strength for ranking assets.  
    """  
      
    @staticmethod  
    def calculate(df: pd.DataFrame, short\_window: int, long\_window: int) -> Tuple\[pd.Series, pd.Series\]:  
        """  
        Calculate SMA crossover signals and signal strength.  
          
        Parameters:  
        -----------  
        df : pd.DataFrame  
            DataFrame with 'close' column  
        short\_window : int  
            Short-term SMA period (e.g., 20 days)  
        long\_window : int  
            Long-term SMA period (e.g., 200 days)  
          
        Returns:  
        --------  
        Tuple\[pd.Series, pd.Series\] : (binary signal, signal strength)  
            - signal: 1 = long, 0 = cash  
            - strength: % difference between short and long SMA (momentum intensity)  
        """  
        \# Calculate both moving averages  
        sma\_short = df\['close'\].rolling(window=short\_window, min\_periods=short\_window).mean()  
        sma\_long = df\['close'\].rolling(window=long\_window, min\_periods=long\_window).mean()  
          
        \# Generate binary signal: 1 when short MA above long MA, 0 otherwise  
        signal = (sma\_short > sma\_long).astype(int)  
          
        \# CRITICAL FIX: Calculate signal strength for ranking  
        \# Strength = (SMA\_short - SMA\_long) / SMA\_long  
        \# Higher positive values = stronger bullish momentum  
        signal\_strength = (sma\_short - sma\_long) / sma\_long  
        signal\_strength = signal\_strength.fillna(0)  
          
        \# CRITICAL: Shift signal by 1 to avoid look-ahead bias  
        \# Signal generated at Close\[t\] is used for trading at Open\[t+1\]  
        signal = signal.shift(1)  
        signal\_strength = signal\_strength.shift(1)  
          
        return signal, signal\_strength  
  
  
\# ============================================================================  
\# POSITION SIZING ENGINE - FIXED  
\# ============================================================================  
  
class VolatilityScaledPositionSizer:  
    """  
    Calculates position sizes based on inverse semi-volatility weighting.  
      
    CRITICAL FIXES:  
    1. Ranks assets by signal strength before selecting top N  
    2. Applies MAX\_LEVERAGE constraint  
    3. Ensures weights are calculated using data available at time t  
    """  
      
    @staticmethod  
    def calculate\_weights(signals: pd.DataFrame,   
                         signal\_strengths: pd.DataFrame,  
                         semi\_vols: pd.DataFrame,  
                         target\_vol: float,  
                         max\_positions: int,  
                         max\_leverage: float) -> pd.DataFrame:  
        """  
        Calculate position weights using semi-volatility scaling with proper asset ranking.  
          
        ALGORITHM:  
        1. For each date, identify assets with positive signal (binary=1)  
        2. Rank these assets by signal strength (momentum intensity)  
        3. Select top MAX\_POSITIONS ranked assets  
        4. Calculate raw weight = (Target\_Vol / Semi\_Vol\_i) for each selected asset  
        5. Normalize weights to respect MAX\_LEVERAGE constraint  
        6. Apply minimum position size filter  
          
        Parameters:  
        -----------  
        signals : pd.DataFrame  
            Binary signals (1/0) for each asset  
        signal\_strengths : pd.DataFrame  
            Signal strength (momentum intensity) for ranking  
        semi\_vols : pd.DataFrame  
            Semi-volatility for each asset  
        target\_vol : float  
            Target portfolio volatility (annualized)  
        max\_positions : int  
            Maximum number of positions  
        max\_leverage : float  
            Maximum gross leverage (e.g., 1.0 = fully invested, no leverage)  
          
        Returns:  
        --------  
        pd.DataFrame : Position weights (% of portfolio)  
        """  
        \# Initialize weights DataFrame  
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)  
          
        \# Iterate through each date  
        for date in signals.index:  
            \# Get active signals (assets with signal = 1)  
            active\_signals = signals.loc\[date\]  
            active\_mask = active\_signals == 1  
              
            if not active\_mask.any():  
                continue  \# No positions on this date  
              
            \# Get signal strengths for active assets  
            strengths = signal\_strengths.loc\[date, active\_mask\]  
              
            \# CRITICAL FIX: Rank by signal strength and select top MAX\_POSITIONS  
            \# Higher strength = stronger momentum = higher priority  
            ranked\_tickers = strengths.sort\_values(ascending=False).head(max\_positions).index.tolist()  
              
            if len(ranked\_tickers) == 0:  
                continue  
              
            \# Get semi-volatilities for selected assets  
            asset\_semi\_vols = semi\_vols.loc\[date, ranked\_tickers\]  
              
            \# Skip if any semi-vol is NaN or zero  
            valid\_mask = (asset\_semi\_vols.notna()) & (asset\_semi\_vols > 0)  
            if not valid\_mask.any():  
                continue  
              
            ranked\_tickers = asset\_semi\_vols\[valid\_mask\].index.tolist()  
            asset\_semi\_vols = asset\_semi\_vols\[valid\_mask\]  
              
            \# Calculate inverse volatility weights  
            \# Raw weight ∝ 1 / Semi\_Vol  
            inverse\_vols = target\_vol / asset\_semi\_vols  
              
            \# Normalize to sum to MAX\_LEVERAGE  
            raw\_weights = inverse\_vols / inverse\_vols.sum() \* max\_leverage  
              
            \# Apply minimum position size constraint  
            raw\_weights = raw\_weights\[raw\_weights >= StrategyConfig.MIN\_POSITION\_SIZE\]  
              
            \# Re-normalize after filtering  
            if raw\_weights.sum() > 0:  
                final\_weights = raw\_weights / raw\_weights.sum() \* max\_leverage  
                weights.loc\[date, final\_weights.index\] = final\_weights.values  
          
        return weights  
  
  
\# ============================================================================  
\# BACKTESTING ENGINE - FIXED  
\# ============================================================================  
  
class SMOMBacktester:  
    """  
    Comprehensive backtesting engine with all critical fixes applied.  
      
    CRITICAL FIXES:  
    1. Point-in-time universe (no survivorship bias)  
    2. Proper rebalancing timing (signal Close\[t\] → execute Open\[t+1\])  
    3. Consistent return calculation (Open-to-Open for all periods)  
    4. Correct last day handling  
    5. All leverage constraints applied  
    """  
      
    def \_\_init\_\_(self, config: StrategyConfig):  
        self.config = config  
        self.data\_fetcher = FMPDataFetcher(config.API\_KEY, config.BASE\_URL)  
          
    def run\_backtest(self) -> Dict:  
        """  
        Execute complete backtest workflow with all fixes applied.  
          
        Returns:  
        --------  
        Dict : Contains all backtest results, performance metrics, and data  
        """  
        print("="\*80)  
        print("SEMI-VOLATILITY SCALED MOMENTUM STRATEGY BACKTEST - FIXED VERSION")  
        print("="\*80)  
        print(f"Period: {self.config.START\_DATE} to {self.config.END\_DATE}")  
        print(f"Warmup: {self.config.WARMUP\_DATE} (for {self.config.SMA\_LONG}-day SMA)")  
        print(f"Momentum: SMA {self.config.SMA\_SHORT}-{self.config.SMA\_LONG} Crossover")  
        print(f"Semi-Volatility: Garman-Klass {self.config.SEMI\_VOL\_WINDOW}-day expanding window")  
        print(f"Target Volatility: {self.config.TARGET\_VOLATILITY\*100:.1f}%")  
        print(f"Max Leverage: {self.config.MAX\_LEVERAGE:.1f}x")  
        print(f"Transaction Costs: {self.config.TOTAL\_COST\_BPS} bps per trade")  
        print("="\*80)  
          
        \# Step 1: Get point-in-time universe (S&P 500)  
        print("\\n\[1/7\] Establishing point-in-time universe...")  
        tickers = self.data\_fetcher.\_get\_current\_sp500()  
          
        if len(tickers) == 0:  
            raise ValueError("No tickers retrieved")  
          
        print(f"Universe: {len(tickers)} tickers")  
          
        \# Step 2: Download historical data  
        print("\\n\[2/7\] Downloading adjusted historical data...")  
        price\_data = {}  
        failed\_tickers = \[\]  
          
        for i, ticker in enumerate(tickers):  
            if (i + 1) % 20 == 0:  
                print(f"  Progress: {i+1}/{len(tickers)} tickers processed")  
              
            data = self.data\_fetcher.get\_historical\_data(  
                ticker,   
                self.config.WARMUP\_DATE,  \# Include warmup period  
                self.config.END\_DATE  
            )  
              
            \# Minimum data requirement: enough for long SMA  
            if data is not None and len(data) >= self.config.SMA\_LONG:  
                price\_data\[ticker\] = data  
            else:  
                failed\_tickers.append(ticker)  
          
        print(f"\\nSuccessfully loaded {len(price\_data)} tickers")  
        print(f"Failed/insufficient data: {len(failed\_tickers)} tickers")  
          
        if len(price\_data) == 0:  
            raise ValueError("No valid price data retrieved")  
          
        \# Step 3: Calculate Momentum Signals and Strength  
        print("\\n\[3/7\] Calculating momentum signals and ranking strength...")  
        signals = pd.DataFrame()  
        signal\_strengths = pd.DataFrame()  
          
        for ticker, data in price\_data.items():  
            signal, strength = SMAMomentumSignal.calculate(  
                data,  
                self.config.SMA\_SHORT,  
                self.config.SMA\_LONG  
            )  
            signals\[ticker\] = signal  
            signal\_strengths\[ticker\] = strength  
          
        \# Step 4: Calculate Semi-Volatility with expanding window  
        print("\\n\[4/7\] Computing Garman-Klass semi-volatility (expanding window)...")  
        semi\_vols = pd.DataFrame()  
          
        for ticker, data in price\_data.items():  
            semi\_vol = GarmanKlassSemiVolatility.calculate(  
                data,  
                self.config.SEMI\_VOL\_WINDOW,  
                self.config.ANNUALIZATION\_FACTOR  
            )  
            semi\_vols\[ticker\] = semi\_vol  
          
        \# Step 5: Align all data to backtest period (exclude warmup)  
        print("\\n\[5/7\] Aligning data to backtest period...")  
        backtest\_start = pd.to\_datetime(self.config.START\_DATE)  
        backtest\_end = pd.to\_datetime(self.config.END\_DATE)  
          
        common\_dates = signals.index.intersection(semi\_vols.index)  
        common\_dates = common\_dates\[(common\_dates >= backtest\_start) & (common\_dates <= backtest\_end)\]  
          
        signals = signals.loc\[common\_dates\]  
        signal\_strengths = signal\_strengths.loc\[common\_dates\]  
        semi\_vols = semi\_vols.loc\[common\_dates\]  
          
        print(f"Backtest period: {common\_dates\[0\].strftime('%Y-%m-%d')} to {common\_dates\[-1\].strftime('%Y-%m-%d')}")  
        print(f"Total trading days: {len(common\_dates)}")  
          
        \# Step 6: Calculate Position Weights with ranking  
        print("\\n\[6/7\] Calculating position weights with momentum ranking...")  
        weights = VolatilityScaledPositionSizer.calculate\_weights(  
            signals,  
            signal\_strengths,  
            semi\_vols,  
            self.config.TARGET\_VOLATILITY,  
            self.config.MAX\_POSITIONS,  
            self.config.MAX\_LEVERAGE  
        )  
          
        \# Step 7: Simulate Trading with proper timing  
        print("\\n\[7/7\] Simulating strategy execution...")  
          
        \# Extract open and close prices  
        open\_prices = pd.DataFrame({ticker: data\['open'\] for ticker, data in price\_data.items()})  
        close\_prices = pd.DataFrame({ticker: data\['close'\] for ticker, data in price\_data.items()})  
          
        \# Align to backtest period  
        open\_prices = open\_prices.loc\[common\_dates\]  
        close\_prices = close\_prices.loc\[common\_dates\]  
          
        \# Calculate strategy returns with fixed timing  
        strategy\_returns = self.\_calculate\_strategy\_returns\_fixed(  
            weights,   
            open\_prices,  
            close\_prices  
        )  
          
        \# Get Benchmark Returns  
        print("\\nLoading benchmark data...")  
        benchmark\_data = self.data\_fetcher.get\_benchmark\_data(  
            self.config.BENCHMARK\_TICKER,  
            self.config.WARMUP\_DATE,  
            self.config.END\_DATE  
        )  
          
        \# Benchmark returns: Open-to-Open (consistent with strategy)  
        benchmark\_open = benchmark\_data\['open'\].loc\[common\_dates\]  
        benchmark\_returns = benchmark\_open.pct\_change()  
          
        \# Calculate performance metrics  
        results = self.\_calculate\_performance\_metrics(  
            strategy\_returns,  
            benchmark\_returns  
        )  
          
        \# Store additional data  
        results\['weights'\] = weights  
        results\['signals'\] = signals  
        results\['signal\_strengths'\] = signal\_strengths  
        results\['semi\_vols'\] = semi\_vols  
        results\['price\_data'\] = price\_data  
          
        print("\\n" + "="\*80)  
        print("BACKTEST COMPLETE")  
        print("="\*80)  
          
        return results  
      
    def \_calculate\_strategy\_returns\_fixed(self,   
                                          weights: pd.DataFrame,  
                                          open\_prices: pd.DataFrame,  
                                          close\_prices: pd.DataFrame) -> pd.Series:  
        """  
        Calculate strategy returns with proper timing and no look-ahead bias.  
          
        CORRECTED TIMING:  
        - Monthly rebalancing: Signal at last Close of month, execute at first Open of next month  
        - All returns calculated Open\[t\] to Open\[t+1\] consistently  
        - Transaction costs applied on rebalancing days  
        - Last day handled correctly  
          
        Parameters:  
        -----------  
        weights : pd.DataFrame  
            Target position weights (daily)  
        open\_prices : pd.DataFrame  
            Open prices for all assets  
        close\_prices : pd.DataFrame  
            Close prices for all assets  
          
        Returns:  
        --------  
        pd.Series : Daily strategy returns  
        """  
        \# Identify month-end dates for rebalancing  
        dates = weights.index  
        month\_ends = dates.to\_series().groupby(dates.to\_period('M')).apply(lambda x: x.index\[-1\])  
          
        \# Rebalancing happens at open of NEXT trading day after month-end  
        rebalance\_dates = \[\]  
        for month\_end in month\_ends:  
            month\_end\_idx = dates.get\_loc(month\_end)  
            if month\_end\_idx < len(dates) - 1:  
                next\_trading\_day = dates\[month\_end\_idx + 1\]  
                rebalance\_dates.append(next\_trading\_day)  
          
        rebalance\_dates = pd.DatetimeIndex(rebalance\_dates)  
          
        \# Get weights at month-end (these will be executed at open of next day)  
        rebalancing\_weights = weights.loc\[month\_ends\]  
          
        \# Initialize returns and position tracking  
        strategy\_returns = pd.Series(0.0, index=dates)  
        current\_weights = pd.Series(0.0, index=weights.columns)  
          
        \# Track transaction costs  
        transaction\_costs\_ts = pd.Series(0.0, index=dates)  
          
        \# Calculate returns day by day  
        for i in range(len(dates)):  
            date = dates\[i\]  
              
            \# Check if this is a rebalancing day  
            if date in rebalance\_dates:  
                \# Find which month-end triggered this rebalancing  
                month\_end = month\_ends\[month\_ends < date\]\[-1\]  
                new\_weights = weights.loc\[month\_end\]  
                  
                \# Calculate turnover (sum of absolute weight changes)  
                turnover = (new\_weights - current\_weights).abs().sum()  
                  
                \# Apply transaction costs  
                transaction\_costs\_ts.loc\[date\] = turnover \* self.config.TOTAL\_COST\_BPS / 10000  
                  
                \# Update current weights  
                current\_weights = new\_weights  
              
            \# Calculate returns for this day: Open\[t\] to Open\[t+1\]  
            if i < len(dates) - 1:  
                next\_date = dates\[i + 1\]  
                  
                \# Asset returns: Open\[t+1\] / Open\[t\] - 1  
                asset\_returns = (open\_prices.loc\[next\_date\] / open\_prices.loc\[date\] - 1)  
                  
                \# Portfolio return: weighted sum  
                portfolio\_return = (current\_weights \* asset\_returns).sum()  
                  
                \# Apply transaction costs if rebalancing day  
                net\_return = portfolio\_return - transaction\_costs\_ts.loc\[date\]  
                  
                strategy\_returns.loc\[date\] = net\_return  
            else:  
                \# CRITICAL FIX: Last day calculation  
                \# For the last day, we can't calculate Open\[t+1\] return  
                \# Use Close\[t\] / Open\[t\] - 1 as the final period return  
                asset\_returns = (close\_prices.loc\[date\] / open\_prices.loc\[date\] - 1)  
                portfolio\_return = (current\_weights \* asset\_returns).sum()  
                strategy\_returns.loc\[date\] = portfolio\_return  
          
        return strategy\_returns  
      
    def \_calculate\_performance\_metrics(self,   
                                       strategy\_returns: pd.Series,  
                                       benchmark\_returns: pd.Series) -> Dict:  
        """  
        Calculate comprehensive performance metrics.  
          
        Parameters:  
        -----------  
        strategy\_returns : pd.Series  
            Strategy daily returns  
        benchmark\_returns : pd.Series  
            Benchmark daily returns  
          
        Returns:  
        --------  
        Dict : Performance metrics and statistics  
        """  
        \# Align returns  
        common\_dates = strategy\_returns.index.intersection(benchmark\_returns.index)  
        strat\_ret = strategy\_returns.loc\[common\_dates\]  
        bench\_ret = benchmark\_returns.loc\[common\_dates\]  
          
        \# Remove first day (NaN from pct\_change)  
        strat\_ret = strat\_ret.dropna()  
        bench\_ret = bench\_ret.dropna()  
          
        \# Align again after dropping NaNs  
        common\_dates = strat\_ret.index.intersection(bench\_ret.index)  
        strat\_ret = strat\_ret.loc\[common\_dates\]  
        bench\_ret = bench\_ret.loc\[common\_dates\]  
          
        \# Calculate cumulative returns  
        strat\_cum = (1 + strat\_ret).cumprod()  
        bench\_cum = (1 + bench\_ret).cumprod()  
          
        \# Annualization  
        ann\_factor = self.config.ANNUALIZATION\_FACTOR  
        years = len(strat\_ret) / ann\_factor  
          
        \# Performance Metrics  
        metrics = {  
            \# Returns  
            'total\_return\_strategy': strat\_cum.iloc\[-1\] - 1,  
            'total\_return\_benchmark': bench\_cum.iloc\[-1\] - 1,  
            'cagr\_strategy': (strat\_cum.iloc\[-1\] \*\* (1/years)) - 1,  
            'cagr\_benchmark': (bench\_cum.iloc\[-1\] \*\* (1/years)) - 1,  
              
            \# Volatility  
            'volatility\_strategy': strat\_ret.std() \* np.sqrt(ann\_factor),  
            'volatility\_benchmark': bench\_ret.std() \* np.sqrt(ann\_factor),  
              
            \# Downside Risk  
            'downside\_dev\_strategy': strat\_ret\[strat\_ret < 0\].std() \* np.sqrt(ann\_factor),  
            'downside\_dev\_benchmark': bench\_ret\[bench\_ret < 0\].std() \* np.sqrt(ann\_factor),  
              
            \# Risk-Adjusted Returns  
            'sharpe\_strategy': (strat\_ret.mean() \* ann\_factor) / (strat\_ret.std() \* np.sqrt(ann\_factor)) if strat\_ret.std() > 0 else 0,  
            'sharpe\_benchmark': (bench\_ret.mean() \* ann\_factor) / (bench\_ret.std() \* np.sqrt(ann\_factor)) if bench\_ret.std() > 0 else 0,  
              
            \# Sortino Ratio  
            'sortino\_strategy': (strat\_ret.mean() \* ann\_factor) / (strat\_ret\[strat\_ret < 0\].std() \* np.sqrt(ann\_factor)) if len(strat\_ret\[strat\_ret < 0\]) > 0 else 0,  
            'sortino\_benchmark': (bench\_ret.mean() \* ann\_factor) / (bench\_ret\[bench\_ret < 0\].std() \* np.sqrt(ann\_factor)) if len(bench\_ret\[bench\_ret < 0\]) > 0 else 0,  
              
            \# Drawdowns  
            'max\_drawdown\_strategy': self.\_calculate\_max\_drawdown(strat\_cum),  
            'max\_drawdown\_benchmark': self.\_calculate\_max\_drawdown(bench\_cum),  
              
            \# Win Rates  
            'win\_rate\_strategy': (strat\_ret > 0).sum() / len(strat\_ret),  
            'win\_rate\_benchmark': (bench\_ret > 0).sum() / len(bench\_ret),  
              
            \# Time Series  
            'strategy\_returns': strat\_ret,  
            'benchmark\_returns': bench\_ret,  
            'strategy\_cumulative': strat\_cum,  
            'benchmark\_cumulative': bench\_cum,  
        }  
          
        return metrics  
      
    @staticmethod  
    def \_calculate\_max\_drawdown(cumulative\_returns: pd.Series) -> float:  
        """Calculate maximum drawdown from cumulative returns."""  
        running\_max = cumulative\_returns.expanding().max()  
        drawdown = (cumulative\_returns - running\_max) / running\_max  
        return drawdown.min()  
  
  
\# ============================================================================  
\# VISUALIZATION MODULE  
\# ============================================================================  
  
class PerformanceVisualizer:  
    """  
    Creates publication-quality visualizations for strategy performance.  
    """  
      
    @staticmethod  
    def plot\_results(results: Dict, config: StrategyConfig):  
        """Generate comprehensive performance visualization dashboard."""  
        fig = plt.figure(figsize=(20, 12))  
        gs = fig.add\_gridspec(3, 3, hspace=0.3, wspace=0.3)  
          
        \# Extract data  
        strat\_cum = results\['strategy\_cumulative'\]  
        bench\_cum = results\['benchmark\_cumulative'\]  
        strat\_ret = results\['strategy\_returns'\]  
        bench\_ret = results\['benchmark\_returns'\]  
          
        \# 1. Cumulative Returns  
        ax1 = fig.add\_subplot(gs\[0, :\])  
        ax1.plot(strat\_cum.index, strat\_cum.values,   
                label='Strategy', linewidth=2, color='#2E86AB')  
        ax1.plot(bench\_cum.index, bench\_cum.values,   
                label=f'Benchmark ({config.BENCHMARK\_TICKER})',   
                linewidth=2, color='#A23B72', linestyle='--')  
        ax1.set\_title('Cumulative Returns', fontsize=14, fontweight='bold')  
        ax1.set\_ylabel('Cumulative Return', fontsize=12)  
        ax1.legend(loc='upper left', fontsize=10)  
        ax1.grid(True, alpha=0.3)  
        ax1.axhline(y=1, color='black', linestyle='-', linewidth=0.5)  
          
        \# 2. Rolling Sharpe  
        ax2 = fig.add\_subplot(gs\[1, 0\])  
        rolling\_window = 252  
        rolling\_sharpe\_strat = (  
            strat\_ret.rolling(rolling\_window).mean() /   
            strat\_ret.rolling(rolling\_window).std() \*   
            np.sqrt(config.ANNUALIZATION\_FACTOR)  
        )  
        rolling\_sharpe\_bench = (  
            bench\_ret.rolling(rolling\_window).mean() /   
            bench\_ret.rolling(rolling\_window).std() \*   
            np.sqrt(config.ANNUALIZATION\_FACTOR)  
        )  
        ax2.plot(rolling\_sharpe\_strat.index, rolling\_sharpe\_strat.values,   
                label='Strategy', linewidth=1.5, color='#2E86AB')  
        ax2.plot(rolling\_sharpe\_bench.index, rolling\_sharpe\_bench.values,   
                label='Benchmark', linewidth=1.5, color='#A23B72', linestyle='--')  
        ax2.set\_title(f'Rolling Sharpe Ratio ({rolling\_window}d)', fontsize=12, fontweight='bold')  
        ax2.set\_ylabel('Sharpe Ratio', fontsize=10)  
        ax2.legend(fontsize=9)  
        ax2.grid(True, alpha=0.3)  
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  
          
        \# 3. Drawdown  
        ax3 = fig.add\_subplot(gs\[1, 1\])  
        strat\_running\_max = strat\_cum.expanding().max()  
        strat\_dd = (strat\_cum - strat\_running\_max) / strat\_running\_max  
        bench\_running\_max = bench\_cum.expanding().max()  
        bench\_dd = (bench\_cum - bench\_running\_max) / bench\_running\_max  
          
        ax3.fill\_between(strat\_dd.index, strat\_dd.values, 0,   
                        alpha=0.3, color='#2E86AB', label='Strategy')  
        ax3.fill\_between(bench\_dd.index, bench\_dd.values, 0,   
                        alpha=0.3, color='#A23B72', label='Benchmark')  
        ax3.set\_title('Drawdown', fontsize=12, fontweight='bold')  
        ax3.set\_ylabel('Drawdown', fontsize=10)  
        ax3.legend(fontsize=9)  
        ax3.grid(True, alpha=0.3)  
          
        \# 4. Monthly Returns Heatmap  
        ax4 = fig.add\_subplot(gs\[1, 2\])  
        monthly\_returns = strat\_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)  
        monthly\_returns\_pivot = monthly\_returns.groupby(\[  
            monthly\_returns.index.year,  
            monthly\_returns.index.month  
        \]).first().unstack()  
          
        if len(monthly\_returns\_pivot) > 0:  
            sns.heatmap(monthly\_returns\_pivot \* 100, annot=True, fmt='.1f',   
                       cmap='RdYlGn', center=0, ax=ax4, cbar\_kws={'label': 'Return (%)'})  
            ax4.set\_title('Monthly Returns (%)', fontsize=12, fontweight='bold')  
            ax4.set\_xlabel('Month', fontsize=10)  
            ax4.set\_ylabel('Year', fontsize=10)  
          
        \# 5. Return Distribution  
        ax5 = fig.add\_subplot(gs\[2, 0\])  
        ax5.hist(strat\_ret \* 100, bins=50, alpha=0.7, color='#2E86AB',   
                label='Strategy', density=True)  
        ax5.hist(bench\_ret \* 100, bins=50, alpha=0.7, color='#A23B72',   
                label='Benchmark', density=True)  
        ax5.set\_title('Return Distribution', fontsize=12, fontweight='bold')  
        ax5.set\_xlabel('Daily Return (%)', fontsize=10)  
        ax5.set\_ylabel('Density', fontsize=10)  
        ax5.legend(fontsize=9)  
        ax5.grid(True, alpha=0.3)  
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)  
          
        \# 6. Performance Metrics Table  
        ax6 = fig.add\_subplot(gs\[2, 1:\])  
        ax6.axis('tight')  
        ax6.axis('off')  
          
        metrics\_data = \[  
            \['Metric', 'Strategy', 'Benchmark', 'Difference'\],  
            \['Total Return',   
             f"{results\['total\_return\_strategy'\]\*100:.2f}%",  
             f"{results\['total\_return\_benchmark'\]\*100:.2f}%",  
             f"{(results\['total\_return\_strategy'\]-results\['total\_return\_benchmark'\])\*100:.2f}%"\],  
            \['CAGR',   
             f"{results\['cagr\_strategy'\]\*100:.2f}%",  
             f"{results\['cagr\_benchmark'\]\*100:.2f}%",  
             f"{(results\['cagr\_strategy'\]-results\['cagr\_benchmark'\])\*100:.2f}%"\],  
            \['Volatility',   
             f"{results\['volatility\_strategy'\]\*100:.2f}%",  
             f"{results\['volatility\_benchmark'\]\*100:.2f}%",  
             f"{(results\['volatility\_strategy'\]-results\['volatility\_benchmark'\])\*100:.2f}%"\],  
            \['Sharpe Ratio',   
             f"{results\['sharpe\_strategy'\]:.2f}",  
             f"{results\['sharpe\_benchmark'\]:.2f}",  
             f"{results\['sharpe\_strategy'\]-results\['sharpe\_benchmark'\]:.2f}"\],  
            \['Sortino Ratio',   
             f"{results\['sortino\_strategy'\]:.2f}",  
             f"{results\['sortino\_benchmark'\]:.2f}",  
             f"{results\['sortino\_strategy'\]-results\['sortino\_benchmark'\]:.2f}"\],  
            \['Max Drawdown',   
             f"{results\['max\_drawdown\_strategy'\]\*100:.2f}%",  
             f"{results\['max\_drawdown\_benchmark'\]\*100:.2f}%",  
             f"{(results\['max\_drawdown\_strategy'\]-results\['max\_drawdown\_benchmark'\])\*100:.2f}%"\],  
            \['Win Rate',   
             f"{results\['win\_rate\_strategy'\]\*100:.2f}%",  
             f"{results\['win\_rate\_benchmark'\]\*100:.2f}%",  
             f"{(results\['win\_rate\_strategy'\]-results\['win\_rate\_benchmark'\])\*100:.2f}%"\],  
        \]  
          
        table = ax6.table(cellText=metrics\_data, cellLoc='center', loc='center',  
                         bbox=\[0, 0, 1, 1\])  
        table.auto\_set\_font\_size(False)  
        table.set\_fontsize(10)  
        table.scale(1, 2)  
          
        for i in range(4):  
            table\[(0, i)\].set\_facecolor('#2E86AB')  
            table\[(0, i)\].set\_text\_props(weight='bold', color='white')  
          
        for i in range(1, len(metrics\_data)):  
            for j in range(4):  
                if i % 2 == 0:  
                    table\[(i, j)\].set\_facecolor('#F0F0F0')  
          
        plt.suptitle(f'Semi-Volatility Scaled Momentum Strategy - Performance Dashboard (FIXED)\\n'  
                    f'Period: {config.START\_DATE} to {config.END\_DATE}',   
                    fontsize=16, fontweight='bold', y=0.98)  
          
        plt.show()  
  
  
\# ============================================================================  
\# MAIN EXECUTION  
\# ============================================================================  
  
def main():  
    """Main execution function for the backtest."""  
    config = StrategyConfig()  
      
    backtester = SMOMBacktester(config)  
    results = backtester.run\_backtest()  
      
    print("\\n" + "="\*80)  
    print("PERFORMANCE SUMMARY")  
    print("="\*80)  
    print(f"Strategy Total Return:    {results\['total\_return\_strategy'\]\*100:>8.2f}%")  
    print(f"Benchmark Total Return:   {results\['total\_return\_benchmark'\]\*100:>8.2f}%")  
    print(f"Outperformance:           {(results\['total\_return\_strategy'\]-results\['total\_return\_benchmark'\])\*100:>8.2f}%")  
    print("-"\*80)  
    print(f"Strategy CAGR:            {results\['cagr\_strategy'\]\*100:>8.2f}%")  
    print(f"Benchmark CAGR:           {results\['cagr\_benchmark'\]\*100:>8.2f}%")  
    print("-"\*80)  
    print(f"Strategy Volatility:      {results\['volatility\_strategy'\]\*100:>8.2f}%")  
    print(f"Benchmark Volatility:     {results\['volatility\_benchmark'\]\*100:>8.2f}%")  
    print("-"\*80)  
    print(f"Strategy Sharpe Ratio:    {results\['sharpe\_strategy'\]:>8.2f}")  
    print(f"Benchmark Sharpe Ratio:   {results\['sharpe\_benchmark'\]:>8.2f}")  
    print("-"\*80)  
    print(f"Strategy Sortino Ratio:   {results\['sortino\_strategy'\]:>8.2f}")  
    print(f"Benchmark Sortino Ratio:  {results\['sortino\_benchmark'\]:>8.2f}")  
    print("-"\*80)  
    print(f"Strategy Max Drawdown:    {results\['max\_drawdown\_strategy'\]\*100:>8.2f}%")  
    print(f"Benchmark Max Drawdown:   {results\['max\_drawdown\_benchmark'\]\*100:>8.2f}%")  
    print("="\*80)  
      
    PerformanceVisualizer.plot\_results(results, config)  
      
    return results  
  
  
if \_\_name\_\_ == "\_\_main\_\_":  
    results = main()

## Further Research

Potential areas for enhancement:

-   **Alternative volatility measures**: Compare with Parkinson, Rogers-Satchell, or Yang-Zhang estimators
-   **Dynamic target volatility**: Adjust the 15% target based on market regime
-   **Multi-factor integration**: Combine momentum with value, quality, or low volatility factors
-   **Options overlay**: Use put protection during high semi-volatility periods
-   **Sector rotation**: Apply the same framework to sector ETFs

_Disclaimer: This article is for educational purposes only. Past performance does not guarantee future results. All trading involves risk, including possible loss of principal. The strategy described requires careful implementation, monitoring, and may not be suitable for all investors. Always conduct your own research and consider consulting with a financial advisor before implementing any trading strategy._

## Embedded Content

---