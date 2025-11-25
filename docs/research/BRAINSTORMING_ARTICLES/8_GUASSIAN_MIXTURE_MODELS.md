# Trading Market Regimes: A Gaussian Mixture Model Approach to Risk-Adjusted Returns | by Ánsique | Oct, 2025 | Medium

Member-only story

# Trading Market Regimes: A Gaussian Mixture Model Approach to Risk-Adjusted Returns

[

![Ánsique](https://miro.medium.com/v2/resize:fill:48:48/1*oOGDWxdecf-FWCjVOt5j_Q@2x.jpeg)





](/@Ansique?source=post_page---byline--f854dc2ac0f7---------------------------------------)

[Ánsique](/@Ansique?source=post_page---byline--f854dc2ac0f7---------------------------------------)

Follow

27 min read

·

2 days ago

12

Listen

Share

More

## How machine learning regime detection achieved a 1.00 Sharpe ratio with half the drawdown of buy-and-hold

![](https://miro.medium.com/v2/resize:fit:1500/1*PpNhZawMsYj1PHo22SOgbA.png)

_In the world of quantitative trading, one of the most persistent challenges is understanding when market dynamics shift. Are we in a bull market? A correction? A period of high volatility that requires caution? Rather than trying to predict these shifts after the fact, what if we could systematically detect regime changes in real-time and adjust our positions accordingly?_

_This is the premise behind regime-switching strategies, and today I’m sharing the results of a production-level implementation using Gaussian Mixture Models (GMM) that achieved compelling risk-adjusted returns over a 5-year period._

## The Core Insight: Markets Have Distinct “Regimes”

Markets don’t behave uniformly over time. Some periods are characterized by steady upward momentum with low volatility. Others exhibit choppy, range-bound behaviour. And occasionally, we experience sharp selloffs with elevated volatility.

The key insight is this: **if we can systematically identify which regime we’re in, we can adjust our exposure accordingly** — going long during favourable conditions and moving to cash during unfavourable ones.

## The Strategy Architecture

## Feature Engineering: What Defines a Regime?

Rather than using arbitrary technical indicators, I focused on two fundamental characteristics that capture market state:

**1\. Yang-Zhang Volatility (20-day window)**

Unlike simple historical volatility that only uses closing prices, the Yang-Zhang estimator is a sophisticated range-based measure that incorporates:

-   Overnight gaps (Open vs previous Close)
-   Intraday movement (High-Low ranges)
-   Open-to-Close dynamics

The formula combines three components:

σ²\_YZ = σ²\_overnight + k·σ²\_open\_close + (1\-k)·σ²\_Rogers\_Satchell

This gives us a more comprehensive view of realized volatility, capturing information that close-to-close calculations miss. **Why it matters**: Bullish regimes typically show higher volatility (counterintuitively), while choppy markets show moderate volatility.

**2\. SMA 20 vs 50 Crossover (Normalized)**

Instead of a binary crossover signal, I use a continuous normalized difference:

Signal = (SMA\_20 - SMA\_50) / SMA\_50

This captures both the direction and magnitude of trend strength. Positive values indicate bullish momentum, negative values indicate bearish momentum, and the magnitude tells us how strong that momentum is. **Normalization makes it scale-invariant** — it works the same whether SPY is at $300 or $600.

**Critical**: Both features are **lagged by 1 day** to prevent look-ahead bias. When we generate a signal at Close\[t\], we use features from t-1.

## The Algorithm: Gaussian Mixture Models

## Why GMM Instead of HMM?

While Hidden Markov Models (HMMs) are popular for regime detection, I chose Gaussian Mixture Models for several reasons:

1.  **Simplicity**: GMM assumes features come from a mixture of Gaussian distributions without modelling temporal transitions
2.  **Interpretability**: Each cluster centre represents a distinct market state
3.  **Robustness**: No need to model transition probabilities, which can be unstable
4.  **Speed**: Faster training and convergence

GMM works by assuming our 2D feature space (volatility × momentum) contains K=3 clusters representing Bearish, Neutral, and Bullish regimes. The algorithm finds:

-   **Cluster centres**: The “typical” feature values for each regime
-   **Covariances**: How features vary within each regime
-   **Mixing weights**: How common each regime is

## Walk-Forward Validation: Preventing Overfitting

Here’s where most backtests fail: **they use future data to make past decisions**.

This strategy uses **strict walk-forward validation**:

1.  **Expanding window training**: Start with 252 days (1 year) of data
2.  **Fit models**: Train StandardScaler → GMM on historical data only
3.  **Create regime mapping**: Analyse which clusters had positive forward returns **in the training period**
4.  **Freeze models**: Lock scaler, GMM, and regime labels
5.  **Predict forward**: Use frozen models for next 63 days (1 quarter)
6.  **Refit**: Expand window and repeat

This resulted in 20 refits over the backtest period, with all clusters remaining sufficiently populated (≥10 samples).

**Regime Assignment Logic**:

-   Calculate forward returns for each cluster in training data
-   Sort clusters by mean forward return
-   Assign: Lowest → Bearish, Middle → Neutral, Highest → Bullish

This ensures regime labels reflect actual future returns, not arbitrary technical criteria.

## Trading Logic: When to Be Long

The strategy employs a **long-only approach**:

-   **Long**: When in Bullish regime (407 days, 27.9%)
-   **Cash**: When in Neutral (555 days, 38.0%) or Bearish (246 days, 16.8%)

**Execution mechanics**:

-   Signal generated at Close\[t\] based on current regime
-   Execution at Open\[t+1\] (realistic slippage)
-   Portfolio marked-to-market at Close\[t+1\]
-   All returns calculated Close-to-Close for consistency
-   Commission: 0.1% per trade (10 bps)

With only **38 trades over 5+ years** (19 buys, 19 sells), the strategy shows remarkably low turnover, minimizing transaction costs and taxes.

## Results: Superior Risk-Adjusted Performance

## Performance Metrics (March 2019 — December 2024)

\----------------------------------------------------------------------------------------------------  
PERFORMANCE ANALYSIS  
\----------------------------------------------------------------------------------------------------  
  
\====================================================================================================  
PERFORMANCE METRICS (WALK-FORWARD VALIDATED \- NO LOOK-AHEAD BIAS)  
\====================================================================================================  
  
Strategy:  
  Total Return:        107.01%  
  CAGR:                 13.39%  
  Volatility:           11.25%  
  Sharpe Ratio:           1.00  
  Max Drawdown:        \-14.68%  
  Win Rate:             16.45%  
  Final Value:      $207,011.99  
  
SPY:  
  Total Return:        108.70%  
  CAGR:                 13.55%  
  Volatility:           20.08%  
  Sharpe Ratio:           0.63  
  Max Drawdown:        \-34.10%  
  Final Value:      $208,493.98  
  
URTH:  
  Total Return:         75.57%  
  CAGR:                 10.21%  
  Volatility:           19.46%  
  Sharpe Ratio:           0.49  
  Max Drawdown:        \-34.01%  
  Final Value:      $175,394.79  
  
Relative to URTH (Global Benchmark):  
  Excess Return:        31.44%  
  Excess CAGR:           3.18%  
  Sharpe Advantage:       0.50  
  
Relative to SPY:  
  Excess Return:        \-1.69%  
  Excess CAGR:          \-0.16%  
  Sharpe Advantage:       0.36  
\====================================================================================================

## Key Takeaways

**1\. Comparable Returns, Half the Risk**

The strategy delivered 107% returns, nearly matching SPY’s 109%. But here’s the critical difference: it achieved this with **11.25% volatility vs SPY’s 20.08%**. That’s a 44% reduction in volatility.

**2\. Exceptional Risk-Adjusted Performance**

The Sharpe ratio of **1.00 vs SPY’s 0.63** represents a 59% improvement. For every unit of risk taken, the strategy generated substantially more return.

**3\. Dramatically Reduced Drawdowns**

Maximum drawdown of **\-14.68% vs -34.10%** for SPY. During the 2020 COVID crash and 2022 bear market, the strategy’s regime detection moved to cash, avoiding the worst declines.

**4\. Outperformance vs Global Equities**

Against URTH (global benchmark), the strategy showed clear alpha:

-   Excess return: +31.44%
-   Excess CAGR: +3.18%
-   Sharpe advantage: +0.50

## Limitations and Considerations

## 1\. Sample Period Matters

The backtest period (2019–2024) included:

-   The 2020 COVID crash (regime detection worked well)
-   The 2022 bear market (moved to cash appropriately)
-   Multiple regime transitions
-   Limited to one business cycle

**Caveat**: Performance in prolonged sideways markets (e.g., 2000s) remains untested.

## 2\. Regime Detection Lag

The strategy detects regime changes but doesn’t predict them. There’s an inherent lag:

-   We observe feature changes
-   GMM classifies current state
-   We act on classification

This means we might miss the first few days of a new regime or hold slightly too long.

## 3\. Opportunity Cost in Bull Markets

Being in cash 55% of the time means missing gains during:

-   Neutral regime rallies
-   Early stage recoveries before Bullish classification

In strong bull markets (e.g., 2023–2024), buy-and-hold would outperform. **This is a feature, not a bug** — we accept lower returns in exchange for dramatically reduced risk.

## 4\. Three-Regime Framework

The K=3 cluster assumption is somewhat arbitrary. Markets might have:

-   More than 3 distinct regimes
-   Overlapping regime characteristics
-   Regime subtypes (e.g., “low-vol bullish” vs “high-vol bullish”)

However, increasing K introduces complexity and risks overfitting.

## 5\. Parameter Sensitivity

Key parameters that could affect performance:

-   SMA windows (20 vs 50)
-   Yang-Zhang window (20 days)
-   Refit frequency (63 days)
-   Minimum cluster samples (10)

While these were chosen based on theory and common practice, they weren’t exhaustively optimized (by design, to avoid overfitting).

## Extensions and Future Work

Several promising directions to explore:

## 1\. Multi-Asset Regime Detection

Apply the same framework to:

-   International equities (EEM, EFA)
-   Fixed income (TLT, IEF)
-   Commodities (GLD, DBC)
-   Crypto (BTC, ETH)

Different assets might exhibit different regime structures.

## 2\. Dynamic Position Sizing

Instead of binary long/cash:

-   100% long in Bullish
-   50% long in Neutral
-   0% long in Bearish

This could capture some upside in neutral regimes while maintaining risk control.

## 3\. Regime-Based Portfolio Allocation

Use regime detection for:

-   Stocks vs bonds allocation
-   Risk-on vs risk-off positioning
-   Factor rotation (value vs growth)

## 4\. Incorporate Additional Features

Potential regime indicators:

-   Credit spreads (HYG vs LQD)
-   Term structure (2Y-10Y spread)
-   Put/call ratios
-   Market breadth (AD line)
-   Sentiment indicators

## 5\. Long/Short Implementation

Enable short positions in Bearish regimes:

-   Long in Bullish
-   Cash in Neutral
-   Short in Bearish

This could enhance returns but adds complexity and risk.

## Conclusion: The Case for Regime-Aware Investing

The results speak for themselves: **matching market returns with half the volatility and less than half the drawdown**.

But beyond the numbers, this strategy represents a fundamentally different approach to investing. Rather than being fully invested at all times (traditional buy-and-hold) or trying to time the market based on predictions (tactical trading), regime detection offers a middle path:

**Respond to observable market conditions, don’t predict the future.**

When volatility and momentum characteristics suggest favourable conditions, be long. When they don’t, step aside. It’s that simple.

The financial industry has long recognized that returns are non-stationary — volatility clusters, trends persist, correlations break down. Yet most retail investors are told to “stay invested” regardless of conditions. Regime-switching strategies acknowledge that **not all market environments are created equal**.

For investors willing to accept lower returns in raging bull markets in exchange for sleeping soundly during crashes, regime detection strategies deserve serious consideration.

## Implementation

The complete strategy code is available with:

-   Full walk-forward validation
-   Benchmark comparisons
-   Visualization tools
-   Export functionality

Key files generated:

-   `gmm_walkforward_results_*.csv` - Full backtest results
-   `gmm_walkforward_trades_*.csv` - All executed trades
-   `gmm_walkforward_folds_*.csv` - Refit metadata for validation

All code is production-ready with:

-   Comprehensive error handling
-   Data validation
-   Non-negativity guards
-   Robust cluster handling
-   Proper short mechanics (if enabled)

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from datetime import datetime  
import requests  
import warnings  
from typing import Dict, Tuple, Optional, List  
from sklearn.mixture import GaussianMixture  
from sklearn.preprocessing import StandardScaler  
from dataclasses import dataclass  
  
warnings.filterwarnings('ignore')  
  
\# ============================================================================  
\# CONFIGURATION  
\# ============================================================================  
  
class StrategyConfig:  
    """  
    Centralized configuration for all strategy parameters.  
      
    All parameters are configurable to allow easy experimentation and optimization.  
    No hardcoded values in the main logic - everything flows from this config.  
    """  
      
    \# API Configuration  
    FMP\_API\_KEY = "FMP\_API\_KEY"  
      
    \# Date Range - 5 years of data for realistic backtest  
    START\_DATE = "2019-01-01"  
    END\_DATE = "2024-12-31"  
      
    \# Trading Parameters  
    INITIAL\_CAPITAL = 100000.0  \# Starting capital in USD  
    COMMISSION\_RATE = 0.001      \# 0.1% per trade (realistic for retail/institutional)  
      
    \# Tickers  
    MAIN\_TICKER = "SPY"                    \# Primary trading instrument  
    BENCHMARK\_TICKER = "URTH"              \# Global equity benchmark  
      
    \# Feature Engineering Parameters  
    SMA\_SHORT = 20      \# Short-term moving average window  
    SMA\_LONG = 50       \# Long-term moving average window  
    YZ\_WINDOW = 20      \# Yang-Zhang volatility estimation window  
      
    \# GMM Parameters  
    GMM\_N\_COMPONENTS = 3           \# Number of regimes (Bearish, Neutral, Bullish)  
    GMM\_COVARIANCE\_TYPE = "full"   \# Covariance type: 'full', 'tied', 'diag', 'spherical'  
    GMM\_MAX\_ITER = 100             \# Maximum iterations for EM algorithm  
    GMM\_N\_INIT = 10                \# Number of initializations (best is kept)  
    GMM\_RANDOM\_STATE = 42          \# For reproducibility  
      
    \# Walk-Forward Parameters  
    MIN\_TRAINING\_DAYS = 252        \# 1 year minimum before first prediction  
    REFIT\_FREQUENCY = 63           \# Refit every quarter (63 trading days ~= 3 months)  
    MIN\_CLUSTER\_SAMPLES = 10       \# Minimum samples per cluster for valid regime mapping  
      
    \# Regime Trading Logic  
    LONG\_REGIME = "Bullish"        \# Which regime to be long in  
    SHORT\_REGIME = None            \# Which regime to be short in (None = cash instead)  
    \# Examples:   
    \# - Long only: LONG\_REGIME="Bullish", SHORT\_REGIME=None  
    \# - Long/Short: LONG\_REGIME="Bullish", SHORT\_REGIME="Bearish"  
    \# - Long/Flat: LONG\_REGIME="Bullish", SHORT\_REGIME=None  
  
  
\# ============================================================================  
\# DATA ACQUISITION  
\# ============================================================================  
  
class DataFetcher:  
    """  
    Handles all data acquisition from FMP API.  
      
    Key principles:  
    - No data imputation (no bfill/ffill)  
    - Direct API calls only  
    - Proper error handling  
    - Data validation  
    """  
      
    def \_\_init\_\_(self, api\_key: str):  
        self.api\_key = api\_key  
        self.base\_url = "https://financialmodelingprep.com/api/v3"  
      
    def fetch\_historical\_data(self, ticker: str, start\_date: str, end\_date: str) -> pd.DataFrame:  
        """  
        Fetch historical OHLCV data from FMP API.  
          
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
        pd.DataFrame with columns: Date, Open, High, Low, Close, Volume  
          
        Critical: NO data filling - missing data stays missing and will be handled  
        downstream through proper NaN handling in feature engineering.  
        """  
        url = f"{self.base\_url}/historical-price-full/{ticker}"  
        params = {"from": start\_date, "to": end\_date, "apikey": self.api\_key}  
          
        try:  
            response = requests.get(url, params=params, timeout=30)  
            response.raise\_for\_status()  
            data = response.json()  
              
            if "historical" not in data:  
                raise ValueError(f"No data returned for {ticker}")  
              
            df = pd.DataFrame(data\["historical"\])  
            df\["date"\] = pd.to\_datetime(df\["date"\])  
            df = df.sort\_values("date").reset\_index(drop=True)  
              
            \# Standardize column names  
            df = df.rename(columns={  
                "date": "Date", "open": "Open", "high": "High",  
                "low": "Low", "close": "Close", "volume": "Volume"  
            })  
              
            \# Keep only necessary columns  
            df = df\[\["Date", "Open", "High", "Low", "Close", "Volume"\]\]  
              
            print(f"✓ Fetched {len(df)} records for {ticker} ({df\['Date'\].min().date()} to {df\['Date'\].max().date()})")  
              
            return df  
              
        except requests.exceptions.RequestException as e:  
            print(f"✗ Network error fetching data for {ticker}: {str(e)}")  
            raise  
        except Exception as e:  
            print(f"✗ Error processing data for {ticker}: {str(e)}")  
            raise  
  
  
\# ============================================================================  
\# FEATURE ENGINEERING  
\# ============================================================================  
  
class FeatureEngine:  
    """  
    Calculates technical features with strict look-ahead bias prevention.  
      
    All features are lagged by 1 day to ensure T-1 data is used for T predictions.  
    This is CRITICAL for walk-forward validation integrity.  
    """  
      
    @staticmethod  
    def calculate\_yang\_zhang\_volatility(df: pd.DataFrame, window: int) -> pd.Series:  
        """  
        Calculate Yang-Zhang volatility estimator with non-negativity guard.  
          
        The Yang-Zhang estimator is a range-based volatility measure that combines:  
        - Overnight volatility: (Open\[t\] / Close\[t-1\])  
        - Open-to-Close volatility: (Close\[t\] / Open\[t\])  
        - High-Low range: Rogers-Satchell component  
          
        Formula:  
        σ²\_YZ = σ²\_overnight + k·σ²\_open\_close + (1-k)·σ²\_RS  
          
        where:  
        - σ²\_overnight = Var(ln(O\[t\]/C\[t-1\]))  
        - σ²\_open\_close = Var(ln(C\[t\]/O\[t\]))  
        - σ²\_RS = Rogers-Satchell = E\[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)\]  
        - k = weighting factor ≈ 0.34 (standard value)  
          
        IMPROVEMENT: Added np.clip to ensure variance is non-negative before sqrt  
        to handle edge cases where floating point errors could produce tiny negative values.  
          
        Parameters:  
        -----------  
        df : pd.DataFrame  
            Must contain Open, High, Low, Close columns  
        window : int  
            Rolling window size for volatility estimation  
          
        Returns:  
        --------  
        pd.Series : Annualized Yang-Zhang volatility (always non-negative)  
        """  
        \# Overnight component: ln(Open\[t\] / Close\[t-1\])  
        overnight = np.log(df\["Open"\] / df\["Close"\].shift(1))  
        overnight\_var = overnight.rolling(window=window, min\_periods=window).var()  
          
        \# Open-to-Close component: ln(Close\[t\] / Open\[t\])  
        open\_close = np.log(df\["Close"\] / df\["Open"\])  
        open\_close\_var = open\_close.rolling(window=window, min\_periods=window).var()  
          
        \# Rogers-Satchell component  
        \# RS = E\[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)\]  
        high\_close = np.log(df\["High"\] / df\["Close"\])  
        high\_open = np.log(df\["High"\] / df\["Open"\])  
        low\_close = np.log(df\["Low"\] / df\["Close"\])  
        low\_open = np.log(df\["Low"\] / df\["Open"\])  
          
        rs\_component = high\_close \* high\_open + low\_close \* low\_open  
        rs\_var = rs\_component.rolling(window=window, min\_periods=window).mean()  
          
        \# Combine components with k = 0.34 (standard weighting)  
        k = 0.34  
        yang\_zhang\_var = overnight\_var + k \* open\_close\_var + (1 - k) \* rs\_var  
          
        \# CRITICAL: Clip variance to ensure non-negativity before sqrt  
        \# This prevents NaN from sporadic floating point errors that could produce  
        \# tiny negative variances (e.g., -1e-16)  
        yang\_zhang\_var\_clipped = np.clip(yang\_zhang\_var, 0, np.inf)  
          
        \# Convert to annualized standard deviation  
        yang\_zhang\_vol = np.sqrt(yang\_zhang\_var\_clipped \* 252)  
          
        return yang\_zhang\_vol  
      
    @staticmethod  
    def calculate\_sma\_crossover\_normalized(close: pd.Series, short\_window: int,   
                                          long\_window: int) -> pd.Series:  
        """  
        Calculate normalized SMA crossover signal.  
          
        Instead of binary crossover (1/-1), we use continuous normalized difference:  
        Signal = (SMA\_short - SMA\_long) / SMA\_long  
          
        This captures:  
        - Positive values: Short MA above long MA (bullish momentum)  
        - Negative values: Short MA below long MA (bearish momentum)  
        - Magnitude: Strength of trend  
          
        Normalization by SMA\_long makes the signal scale-invariant (works across  
        different price levels and assets).  
          
        Parameters:  
        -----------  
        close : pd.Series  
            Close prices  
        short\_window : int  
            Short moving average window  
        long\_window : int  
            Long moving average window  
          
        Returns:  
        --------  
        pd.Series : Normalized crossover signal (continuous, typically in \[-0.1, 0.1\])  
        """  
        \# Calculate moving averages with minimum periods = window  
        \# This ensures we don't have incomplete averages at the start  
        sma\_short = close.rolling(window=short\_window, min\_periods=short\_window).mean()  
        sma\_long = close.rolling(window=long\_window, min\_periods=long\_window).mean()  
          
        \# Normalized difference: percentage above/below long MA  
        \# Dividing by sma\_long makes it scale-invariant  
        crossover\_signal = (sma\_short - sma\_long) / sma\_long  
          
        return crossover\_signal  
      
    @staticmethod  
    def prepare\_features(df\_main: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:  
        """  
        Prepare all features with proper lagging to prevent look-ahead bias.  
          
        CRITICAL LOOK-AHEAD PREVENTION:  
        1. Calculate features using only past data (rolling windows)  
        2. Lag ALL features by 1 day: feature\[t-1\] predicts returns\[t\]  
        3. This ensures when we generate signal at Close\[t\], we're using data available at t  
          
        Feature Flow:  
        - Raw data at t-1 → Feature calculation → Feature\[t-1\] → Signal\[t-1\] → Execute\[t\]  
          
        Parameters:  
        -----------  
        df\_main : pd.DataFrame  
            Main price data (SPY)  
        config : StrategyConfig  
            Configuration object with all parameters  
          
        Returns:  
        --------  
        pd.DataFrame with lagged features and returns  
        """  
        df = df\_main.copy()  
          
        print("\\nCalculating features...")  
          
        \# ===== RETURN CALCULATIONS =====  
        \# CONSISTENT Close-to-Close returns for all daily calculations  
        \# This ensures strategy and benchmarks are on the same basis  
        df\["Returns\_CC"\] = df\["Close"\].pct\_change()  
          
        \# Open-to-Open returns: only used for regime mapping (forward returns)  
        df\["Returns\_OO"\] = df\["Open"\].pct\_change()  
          
        \# ===== FEATURE 1: Yang-Zhang Volatility =====  
        \# More sophisticated than simple volatility, uses full OHLC information  
        df\["YangZhang\_Vol"\] = FeatureEngine.calculate\_yang\_zhang\_volatility(  
            df, config.YZ\_WINDOW  
        )  
        print(f"✓ Yang-Zhang volatility calculated (window={config.YZ\_WINDOW})")  
          
        \# ===== FEATURE 2: SMA Crossover (Normalized) =====  
        \# Continuous momentum signal, not binary  
        df\["SMA\_Cross\_Norm"\] = FeatureEngine.calculate\_sma\_crossover\_normalized(  
            df\["Close"\], config.SMA\_SHORT, config.SMA\_LONG  
        )  
        print(f"✓ SMA crossover calculated (SMA{config.SMA\_SHORT} vs SMA{config.SMA\_LONG})")  
          
        \# ===== CRITICAL: LAG ALL FEATURES BY 1 DAY =====  
        \# This is THE key step to prevent look-ahead bias  
        \# When we predict regime at Close\[t\], we use features from t-1  
        feature\_cols = \["YangZhang\_Vol", "SMA\_Cross\_Norm"\]  
        for col in feature\_cols:  
            df\[f"{col}\_lag"\] = df\[col\].shift(1)  
          
        print("✓ Features lagged by 1 day to prevent look-ahead bias")  
          
        \# Show feature statistics  
        lagged\_features = \[f"{col}\_lag" for col in feature\_cols\]  
        valid\_data = df\[lagged\_features\].dropna()  
          
        if len(valid\_data) > 0:  
            print(f"\\nFeature statistics (after lagging):")  
            print(f"  Valid observations: {len(valid\_data)}")  
            for col in lagged\_features:  
                print(f"  {col}: mean={valid\_data\[col\].mean():.4f}, std={valid\_data\[col\].std():.4f}")  
          
        return df  
  
  
\# ============================================================================  
\# WALK-FORWARD GMM REGIME DETECTOR  
\# ============================================================================  
  
@dataclass  
class RegimeMapping:  
    """  
    Stores regime mapping from training period.  
      
    This mapping is FROZEN for the prediction period to prevent look-ahead bias.  
    The mapping tells us which GMM cluster corresponds to which economic regime  
    (Bearish/Neutral/Bullish) based on PAST forward returns.  
    """  
    cluster\_to\_regime: Dict\[int, str\]      \# Maps cluster ID → regime name  
    mean\_returns: Dict\[int, float\]         \# Mean forward returns per cluster  
    cluster\_counts: Dict\[int, int\]         \# Number of samples per cluster  
    training\_end\_date: pd.Timestamp        \# Last date of training data used  
    is\_valid: bool                         \# Whether mapping has sufficient samples  
  
  
class WalkForwardGMMRegimeDetector:  
    """  
    GMM Regime Detector with STRICT walk-forward validation.  
      
    GMM (Gaussian Mixture Model) vs HMM:  
    - GMM: Assumes features come from mixture of Gaussian distributions  
    - No temporal dependencies (unlike HMM)  
    - Simpler, faster, often more robust  
    - Good for regime detection when temporal transition probabilities not important  
      
    WALK-FORWARD PROCESS (NO LOOK-AHEAD):  
    =====================================  
    For each prediction window:  
      
    1. TRAIN SCALER: Fit StandardScaler on expanding training window  
       → Learn mean/std of features from past data only  
      
    2. TRAIN GMM: Fit GaussianMixture on normalized training features  
       → Learn cluster centers and covariances from past data  
      
    3. CREATE REGIME MAPPING: Analyze forward returns in training period  
       → Determine which cluster = Bearish, Neutral, Bullish  
       → CRITICAL: Use only training period forward returns  
       → ROBUST: Handle underpopulated clusters (min sample threshold)  
      
    4. FREEZE MODELS: Lock scaler, GMM, and regime mapping  
      
    5. PREDICT: Use frozen models to predict regime for next period  
      
    6. REFIT: After N days, repeat process with expanded training set  
      
    This ensures predictions at time T use ONLY information available at T-1.  
    """  
      
    def \_\_init\_\_(self, config: StrategyConfig):  
        self.config = config  
        \# Feature columns to use (already lagged in feature engineering step)  
        self.feature\_cols = \["YangZhang\_Vol\_lag", "SMA\_Cross\_Norm\_lag"\]  
        self.previous\_mapping = None  \# Store last valid mapping for fallback  
          
    def create\_regime\_mapping(self, X\_train: np.ndarray, gmm\_model: GaussianMixture,  
                             returns\_forward: np.ndarray, train\_end\_date: pd.Timestamp) -> RegimeMapping:  
        """  
        Create regime mapping with robust cluster handling.  
          
        IMPROVEMENTS:  
        1. Clean slicing: Remove last observation before analysis (no forward return)  
        2. Minimum sample enforcement: Require MIN\_CLUSTER\_SAMPLES per cluster  
        3. NaN for empty clusters: Don't use 0.0 fallback  
        4. Fallback mechanism: Use previous valid mapping if current is invalid  
          
        Parameters:  
        -----------  
        X\_train : np.ndarray  
            Training features  
        gmm\_model : GaussianMixture  
            Fitted GMM model  
        returns\_forward : np.ndarray  
            Forward returns (already shifted)  
        train\_end\_date : pd.Timestamp  
            End date of training period  
          
        Returns:  
        --------  
        RegimeMapping object with cluster→regime mapping  
        """  
        \# Predict clusters for training period  
        clusters\_all = gmm\_model.predict(X\_train)  
          
        \# CLEAN SLICING: Remove last observation (no forward return available)  
        \# This is cleaner than manual mask\[-1\] = False  
        clusters\_train = clusters\_all\[:-1\]  
        returns\_train = returns\_forward\[:-1\]  
          
        \# Analyze each cluster  
        cluster\_returns = {}  
        cluster\_counts = {}  
          
        for cluster\_id in range(self.config.GMM\_N\_COMPONENTS):  
            mask = clusters\_train == cluster\_id  
            count = mask.sum()  
            cluster\_counts\[cluster\_id\] = count  
              
            if count >= self.config.MIN\_CLUSTER\_SAMPLES:  
                \# Sufficient samples: calculate mean return  
                cluster\_returns\[cluster\_id\] = returns\_train\[mask\].mean()  
            else:  
                \# Insufficient samples: use NaN (not 0.0)  
                \# This signals that the cluster is unreliable  
                cluster\_returns\[cluster\_id\] = np.nan  
          
        \# Check if mapping is valid (all clusters have sufficient samples)  
        valid\_clusters = \[cid for cid, ret in cluster\_returns.items() if not np.isnan(ret)\]  
        is\_valid = len(valid\_clusters) == self.config.GMM\_N\_COMPONENTS  
          
        if is\_valid:  
            \# Sort by returns to assign regime labels  
            sorted\_clusters = sorted(cluster\_returns.items(), key=lambda x: x\[1\])  
            cluster\_to\_regime = {  
                sorted\_clusters\[0\]\[0\]: "Bearish",  
                sorted\_clusters\[1\]\[0\]: "Neutral",  
                sorted\_clusters\[2\]\[0\]: "Bullish"  
            }  
              
            print(f"✓ Valid regime mapping created:")  
            for cluster\_id, regime in cluster\_to\_regime.items():  
                print(f"  Cluster {cluster\_id} → {regime}: "  
                      f"{cluster\_counts\[cluster\_id\]} obs, "  
                      f"avg forward return: {cluster\_returns\[cluster\_id\]:.6f}")  
        else:  
            \# Invalid mapping: try to use previous valid mapping  
            if self.previous\_mapping is not None and self.previous\_mapping.is\_valid:  
                print(f"⚠ Some clusters underpopulated, using previous fold's mapping")  
                cluster\_to\_regime = self.previous\_mapping.cluster\_to\_regime  
                print(f"  Carried forward mapping from {self.previous\_mapping.training\_end\_date.date()}")  
            else:  
                \# No previous mapping available: assign default ordering  
                print(f"⚠ Some clusters underpopulated and no previous mapping available")  
                print(f"  Using default cluster→regime assignment (0→Bearish, 1→Neutral, 2→Bullish)")  
                cluster\_to\_regime = {  
                    0: "Bearish",  
                    1: "Neutral",  
                    2: "Bullish"  
                }  
              
            \# Show which clusters are problematic  
            for cluster\_id, count in cluster\_counts.items():  
                status = "OK" if count >= self.config.MIN\_CLUSTER\_SAMPLES else "UNDERPOPULATED"  
                ret\_str = f"{cluster\_returns\[cluster\_id\]:.6f}" if not np.isnan(cluster\_returns\[cluster\_id\]) else "NaN"  
                print(f"  Cluster {cluster\_id}: {count} obs, return: {ret\_str} \[{status}\]")  
          
        mapping = RegimeMapping(  
            cluster\_to\_regime=cluster\_to\_regime,  
            mean\_returns=cluster\_returns,  
            cluster\_counts=cluster\_counts,  
            training\_end\_date=train\_end\_date,  
            is\_valid=is\_valid  
        )  
          
        \# Update previous mapping if current is valid  
        if is\_valid:  
            self.previous\_mapping = mapping  
          
        return mapping  
          
    def walk\_forward\_predict(self, df: pd.DataFrame) -> Tuple\[pd.DataFrame, List\[Dict\]\]:  
        """  
        Perform walk-forward regime prediction with NO look-ahead bias.  
          
        Returns:  
        --------  
        Tuple of:  
        - df\_clean: DataFrame with regime predictions  
        - folds\_executed: List of dictionaries with fold metadata for validation  
        """  
        print("\\n" + "="\*80)  
        print("WALK-FORWARD GMM REGIME DETECTION (NO LOOK-AHEAD BIAS)")  
        print("="\*80)  
          
        \# ===== DATA CLEANING =====  
        \# Drop rows where features are NaN (e.g., initial period before windows fill)  
        df\_clean = df.dropna(subset=self.feature\_cols).copy().reset\_index(drop=True)  
        print(f"\\nTotal observations after cleaning: {len(df\_clean)}")  
        print(f"Date range: {df\_clean\['Date'\].min().date()} to {df\_clean\['Date'\].max().date()}")  
          
        \# Initialize result columns  
        df\_clean\["Regime\_Cluster"\] = np.nan  \# GMM cluster ID (0, 1, 2)  
        df\_clean\["Regime"\] = None             \# Economic label (Bearish, Neutral, Bullish)  
          
        \# ===== WALK-FORWARD SETUP =====  
        min\_train = self.config.MIN\_TRAINING\_DAYS  
        refit\_freq = self.config.REFIT\_FREQUENCY  
          
        print(f"\\nWalk-forward configuration:")  
        print(f"  Minimum training: {min\_train} days")  
        print(f"  Refit frequency: {refit\_freq} days")  
        print(f"  Min cluster samples: {self.config.MIN\_CLUSTER\_SAMPLES}")  
        print(f"  Total predictions needed: {len(df\_clean) - min\_train}")  
          
        \# Track all folds for validation  
        folds\_executed = \[\]  
          
        \# Initialize model objects  
        last\_refit\_idx = min\_train  
        scaler = None  
        gmm\_model = None  
        regime\_mapping = None  
          
        fold\_num = 0  
          
        \# ===== MAIN WALK-FORWARD LOOP =====  
        for i in range(min\_train, len(df\_clean)):  
            days\_since\_refit = i - last\_refit\_idx  
              
            \# ===== CHECK IF REFIT NEEDED =====  
            if (scaler is None) or (days\_since\_refit >= refit\_freq):  
                fold\_num += 1  
                train\_start = 0           \# Expanding window: always start from beginning  
                train\_end = i             \# End at current position  
                  
                print(f"\\n{'='\*80}")  
                print(f"FOLD {fold\_num}: Refitting models")  
                print(f"{'='\*80}")  
                print(f"Training window: index {train\_start} to {train\_end-1}")  
                print(f"Training dates: {df\_clean.loc\[train\_start, 'Date'\].date()} to "  
                      f"{df\_clean.loc\[train\_end-1, 'Date'\].date()}")  
                print(f"Training size: {train\_end - train\_start} days")  
                  
                \# ===== STEP 1: FIT SCALER =====  
                \# Extract raw features from training window  
                X\_train\_raw = df\_clean.loc\[train\_start:train\_end-1, self.feature\_cols\].values  
                  
                \# Fit StandardScaler: learns mean and std from training data  
                scaler = StandardScaler()  
                X\_train\_scaled = scaler.fit\_transform(X\_train\_raw)  
                  
                print(f"✓ Scaler fitted on {len(X\_train\_scaled)} training observations")  
                  
                \# ===== STEP 2: FIT GMM =====  
                \# GaussianMixture finds K clusters in feature space  
                gmm\_model = GaussianMixture(  
                    n\_components=self.config.GMM\_N\_COMPONENTS,  
                    covariance\_type=self.config.GMM\_COVARIANCE\_TYPE,  
                    max\_iter=self.config.GMM\_MAX\_ITER,  
                    n\_init=self.config.GMM\_N\_INIT,  
                    random\_state=self.config.GMM\_RANDOM\_STATE  
                )  
                gmm\_model.fit(X\_train\_scaled)  
                  
                print(f"✓ GMM fitted (converged: {gmm\_model.converged\_})")  
                  
                \# ===== STEP 3: CREATE REGIME MAPPING =====  
                \# Get forward returns for training period  
                returns\_forward\_train = df\_clean.loc\[train\_start:train\_end-1, "Returns\_OO"\].shift(-1).values  
                  
                \# Create mapping with robust cluster handling  
                regime\_mapping = self.create\_regime\_mapping(  
                    X\_train\_scaled,   
                    gmm\_model,   
                    returns\_forward\_train,  
                    df\_clean.loc\[train\_end-1, "Date"\]  
                )  
                  
                \# Update refit tracker  
                last\_refit\_idx = i  
                  
                \# ===== STORE FOLD METADATA =====  
                folds\_executed.append({  
                    'fold': fold\_num,  
                    'train\_start': train\_start,  
                    'train\_end': train\_end,  
                    'train\_start\_date': df\_clean.loc\[train\_start, 'Date'\],  
                    'train\_end\_date': df\_clean.loc\[train\_end-1, 'Date'\],  
                    'regime\_mapping': regime\_mapping.cluster\_to\_regime.copy(),  
                    'mean\_returns': regime\_mapping.mean\_returns.copy(),  
                    'cluster\_counts': regime\_mapping.cluster\_counts.copy(),  
                    'is\_valid': regime\_mapping.is\_valid  
                })  
              
            \# ===== STEP 4: PREDICT CURRENT OBSERVATION =====  
            \# Use FROZEN models (scaler, GMM, mapping) to predict regime  
              
            \# Extract current observation features  
            X\_current\_raw = df\_clean.loc\[i:i, self.feature\_cols\].values  
              
            \# Transform using frozen scaler  
            X\_current\_scaled = scaler.transform(X\_current\_raw)  
              
            \# Predict cluster using frozen GMM  
            cluster\_pred = gmm\_model.predict(X\_current\_scaled)\[0\]  
              
            \# Map to regime using frozen mapping  
            regime\_pred = regime\_mapping.cluster\_to\_regime\[cluster\_pred\]  
              
            \# Store predictions  
            df\_clean.loc\[i, "Regime\_Cluster"\] = cluster\_pred  
            df\_clean.loc\[i, "Regime"\] = regime\_pred  
          
        \# ===== VALIDATION SUMMARY =====  
        print(f"\\n{'='\*80}")  
        print(f"WALK-FORWARD COMPLETE")  
        print(f"{'='\*80}")  
        print(f"Total folds executed: {len(folds\_executed)}")  
        valid\_folds = sum(1 for f in folds\_executed if f\['is\_valid'\])  
        print(f"Valid folds (all clusters sufficiently populated): {valid\_folds}/{len(folds\_executed)}")  
        print(f"Predictions generated: {df\_clean\['Regime'\].notna().sum()}")  
          
        \# Show final regime distribution  
        regime\_counts = df\_clean\["Regime"\].value\_counts()  
        print(f"\\nFinal regime distribution:")  
        for regime in \["Bearish", "Neutral", "Bullish"\]:  
            count = regime\_counts.get(regime, 0)  
            pct = count / len(df\_clean) \* 100 if len(df\_clean) > 0 else 0  
            print(f"  {regime}: {count} days ({pct:.1f}%)")  
          
        return df\_clean, folds\_executed  
  
  
\# ============================================================================  
\# BACKTESTING ENGINE  
\# ============================================================================  
  
class BacktestEngine:  
    """  
    Production-level backtesting engine with proper long/short handling.  
      
    EXECUTION LOGIC:  
    ===============  
    1. Signal generated at Close\[t\] based on regime  
    2. Execution happens at Open\[t+1\]  
    3. Portfolio marked-to-market at Close\[t+1\]  
    4. Returns = Close\[t+1\] / Close\[t\] - 1 (CONSISTENT for strategy and benchmarks)  
      
    POSITION STATES:  
    - Position = 1: Long (positive shares)  
    - Position = 0: Flat (cash)  
    - Position = -1: Short (negative shares, borrowed stock)  
      
    SHORT MECHANICS:  
    - Borrow shares and sell them (receive cash + commission)  
    - Mark-to-market: Holdings = -shares \* Close (negative value)  
    - Cover: Buy back shares (pay cash + commission)  
    - PnL: Profit when price falls, loss when price rises  
    """  
      
    def \_\_init\_\_(self, initial\_capital: float, commission\_rate: float):  
        self.initial\_capital = initial\_capital  
        self.commission\_rate = commission\_rate  
        self.trades = \[\]  \# Store all executed trades for analysis  
      
    def run\_backtest(self, df: pd.DataFrame, long\_regime: str,   
                    short\_regime: Optional\[str\] = None) -> pd.DataFrame:  
        """  
        Execute backtest with proper long/short/flat handling.  
          
        Parameters:  
        -----------  
        df : pd.DataFrame  
            Data with Regime column  
        long\_regime : str  
            Which regime to go long (e.g., "Bullish")  
        short\_regime : str or None  
            Which regime to go short (None = cash instead)  
          
        Returns:  
        --------  
        pd.DataFrame with backtest results  
        """  
        results = df.reset\_index(drop=True).copy()  
          
        print("\\n" + "-"\*80)  
        print("EXECUTING BACKTEST")  
        print("-"\*80)  
          
        \# ===== GENERATE SIGNALS =====  
        \# Signal at Close\[t\] determines position for next day  
        results\["Signal"\] = 0  \# Default: cash (flat)  
          
        if long\_regime:  
            results.loc\[results\["Regime"\] == long\_regime, "Signal"\] = 1  
            print(f"Long signal: {long\_regime} regime")  
          
        if short\_regime:  
            results.loc\[results\["Regime"\] == short\_regime, "Signal"\] = -1  
            print(f"Short signal: {short\_regime} regime")  
        else:  
            print(f"Short signal: None (cash instead)")  
          
        \# ===== INITIALIZE TRACKING =====  
        results\["Position"\] = 0       \# 1=long, 0=flat, -1=short  
        results\["Cash"\] = 0.0  
        results\["Holdings"\] = 0.0     \# Can be negative for shorts  
        results\["Portfolio\_Value"\] = 0.0  
        results\["Strategy\_Returns"\] = 0.0  
        results\["Trades"\] = ""  
          
        \# Starting state  
        cash = self.initial\_capital  
        shares = 0  \# Can be negative for shorts  
          
        results.loc\[0, "Cash"\] = cash  
        results.loc\[0, "Portfolio\_Value"\] = cash  
          
        \# ===== MAIN BACKTEST LOOP =====  
        for i in range(1, len(results)):  
            \# Desired position based on yesterday's signal  
            desired\_position = int(results.loc\[i-1, "Signal"\])  
            current\_position = int(results.loc\[i-1, "Position"\])  
              
            \# Check if we need to trade  
            if desired\_position != current\_position:  
                \# Execute at Open\[i\]  
                execution\_price = results.loc\[i, "Open"\]  
                  
                \# ===== BRANCH 1: OPENING LONG POSITION =====  
                if desired\_position == 1 and current\_position == 0:  
                    \# Buy shares with available cash  
                    shares = cash / (execution\_price \* (1 + self.commission\_rate))  
                    commission = shares \* execution\_price \* self.commission\_rate  
                    cash = 0  \# All cash deployed  
                      
                    self.trades.append({  
                        "Date": results.loc\[i, "Date"\],  
                        "Type": "BUY",  
                        "Price": execution\_price,  
                        "Shares": shares,  
                        "Commission": commission,  
                        "Regime": results.loc\[i-1, "Regime"\]  
                    })  
                    results.loc\[i, "Trades"\] = "BUY"  
                  
                \# ===== BRANCH 2: CLOSING LONG POSITION =====  
                elif desired\_position == 0 and current\_position == 1:  
                    \# Sell all shares  
                    commission = shares \* execution\_price \* self.commission\_rate  
                    cash = shares \* execution\_price - commission  
                    shares = 0  
                      
                    self.trades.append({  
                        "Date": results.loc\[i, "Date"\],  
                        "Type": "SELL",  
                        "Price": execution\_price,  
                        "Shares": shares,  
                        "Commission": commission,  
                        "Regime": results.loc\[i-1, "Regime"\]  
                    })  
                    results.loc\[i, "Trades"\] = "SELL"  
                  
                \# ===== BRANCH 3: OPENING SHORT POSITION =====  
                elif desired\_position == -1 and current\_position == 0:  
                    \# Short: Borrow and sell shares  
                    \# Receive cash from sale minus commission  
                    shares\_to\_short = cash / (execution\_price \* (1 - self.commission\_rate))  
                    commission = shares\_to\_short \* execution\_price \* self.commission\_rate  
                    cash = cash + shares\_to\_short \* execution\_price - commission  
                    shares = -shares\_to\_short  \# Negative shares = short position  
                      
                    self.trades.append({  
                        "Date": results.loc\[i, "Date"\],  
                        "Type": "SHORT",  
                        "Price": execution\_price,  
                        "Shares": shares\_to\_short,  \# Store as positive for clarity  
                        "Commission": commission,  
                        "Regime": results.loc\[i-1, "Regime"\]  
                    })  
                    results.loc\[i, "Trades"\] = "SHORT"  
                  
                \# ===== BRANCH 4: CLOSING SHORT POSITION =====  
                elif desired\_position == 0 and current\_position == -1:  
                    \# Cover: Buy back borrowed shares  
                    shares\_to\_cover = abs(shares)  
                    commission = shares\_to\_cover \* execution\_price \* self.commission\_rate  
                    cash = cash - shares\_to\_cover \* execution\_price - commission  
                    shares = 0  
                      
                    self.trades.append({  
                        "Date": results.loc\[i, "Date"\],  
                        "Type": "COVER",  
                        "Price": execution\_price,  
                        "Shares": shares\_to\_cover,  
                        "Commission": commission,  
                        "Regime": results.loc\[i-1, "Regime"\]  
                    })  
                    results.loc\[i, "Trades"\] = "COVER"  
                  
                \# ===== BRANCH 5: LONG TO SHORT =====  
                elif desired\_position == -1 and current\_position == 1:  
                    \# Close long first  
                    commission\_sell = shares \* execution\_price \* self.commission\_rate  
                    cash = shares \* execution\_price - commission\_sell  
                      
                    \# Then open short  
                    shares\_to\_short = cash / (execution\_price \* (1 - self.commission\_rate))  
                    commission\_short = shares\_to\_short \* execution\_price \* self.commission\_rate  
                    cash = cash + shares\_to\_short \* execution\_price - commission\_short  
                    shares = -shares\_to\_short  
                      
                    total\_commission = commission\_sell + commission\_short  
                      
                    self.trades.append({  
                        "Date": results.loc\[i, "Date"\],  
                        "Type": "SELL+SHORT",  
                        "Price": execution\_price,  
                        "Shares": shares\_to\_short,  
                        "Commission": total\_commission,  
                        "Regime": results.loc\[i-1, "Regime"\]  
                    })  
                    results.loc\[i, "Trades"\] = "SELL+SHORT"  
                  
                \# ===== BRANCH 6: SHORT TO LONG =====  
                elif desired\_position == 1 and current\_position == -1:  
                    \# Cover short first  
                    shares\_to\_cover = abs(shares)  
                    commission\_cover = shares\_to\_cover \* execution\_price \* self.commission\_rate  
                    cash = cash - shares\_to\_cover \* execution\_price - commission\_cover  
                      
                    \# Then go long  
                    shares\_to\_buy = cash / (execution\_price \* (1 + self.commission\_rate))  
                    commission\_buy = shares\_to\_buy \* execution\_price \* self.commission\_rate  
                    cash = 0  
                    shares = shares\_to\_buy  
                      
                    total\_commission = commission\_cover + commission\_buy  
                      
                    self.trades.append({  
                        "Date": results.loc\[i, "Date"\],  
                        "Type": "COVER+BUY",  
                        "Price": execution\_price,  
                        "Shares": shares\_to\_buy,  
                        "Commission": total\_commission,  
                        "Regime": results.loc\[i-1, "Regime"\]  
                    })  
                    results.loc\[i, "Trades"\] = "COVER+BUY"  
              
            \# ===== UPDATE PORTFOLIO STATE =====  
            \# Position: 1 if long, -1 if short, 0 if flat  
            if shares > 0:  
                results.loc\[i, "Position"\] = 1  
            elif shares < 0:  
                results.loc\[i, "Position"\] = -1  
            else:  
                results.loc\[i, "Position"\] = 0  
              
            results.loc\[i, "Cash"\] = cash  
              
            \# Mark-to-market at Close (CONSISTENT valuation basis)  
            \# For shorts: holdings are negative (liability)  
            results.loc\[i, "Holdings"\] = shares \* results.loc\[i, "Close"\]  
              
            \# Total portfolio value = cash + holdings  
            \# For shorts: holdings < 0, so portfolio value decreases as price rises  
            results.loc\[i, "Portfolio\_Value"\] = cash + shares \* results.loc\[i, "Close"\]  
          
        \# ===== CALCULATE RETURNS =====  
        \# Close-to-Close percentage change (CONSISTENT with benchmarks)  
        results\["Strategy\_Returns"\] = results\["Portfolio\_Value"\].pct\_change()  
          
        \# Cumulative returns (compound growth)  
        results\["Cumulative\_Returns"\] = (1 + results\["Strategy\_Returns"\]).cumprod()  
          
        \# ===== SUMMARY =====  
        total\_trades = len(self.trades)  
        trade\_types = {}  
        for t in self.trades:  
            trade\_types\[t\["Type"\]\] = trade\_types.get(t\["Type"\], 0) + 1  
          
        print(f"\\n✓ Backtest completed")  
        print(f"  Total trades: {total\_trades}")  
        for trade\_type, count in trade\_types.items():  
            print(f"    {trade\_type}: {count}")  
        print(f"  Initial capital: ${self.initial\_capital:,.2f}")  
        print(f"  Final value: ${results.iloc\[-1\]\['Portfolio\_Value'\]:,.2f}")  
        print(f"  Total return: {(results.iloc\[-1\]\['Portfolio\_Value'\] / self.initial\_capital - 1) \* 100:.2f}%")  
          
        return results  
      
    def add\_buy\_hold\_benchmark(self, results: pd.DataFrame, price\_df: pd.DataFrame,   
                              ticker\_name: str) -> pd.DataFrame:  
        """  
        Add buy-and-hold benchmark with hardened data validation.  
          
        IMPROVEMENTS:  
        1. Inner join to ensure date alignment  
        2. Validate that benchmark data exists before computing  
        3. Assert non-NaN prices before calculations  
        4. Consistent Close-to-Close returns  
          
        Parameters:  
        -----------  
        results : pd.DataFrame  
            Main results dataframe  
        price\_df : pd.DataFrame  
            Benchmark price data  
        ticker\_name : str  
            Name for the benchmark (e.g., 'SPY', 'URTH')  
          
        Returns:  
        --------  
        pd.DataFrame with benchmark columns added  
        """  
        print(f"\\nAdding {ticker\_name} buy-and-hold benchmark...")  
          
        \# ===== HARDENED MERGE: INNER JOIN =====  
        \# Only keep dates that exist in both datasets  
        price\_aligned = price\_df\[\["Date", "Open", "Close"\]\].rename(  
            columns={"Open": f"{ticker\_name}\_Open", "Close": f"{ticker\_name}\_Close"}  
        )  
          
        \# Merge with inner join to intersect dates  
        results\_orig\_len = len(results)  
        results = results.merge(price\_aligned, on="Date", how="inner")  
        results\_new\_len = len(results)  
          
        if results\_new\_len < results\_orig\_len:  
            print(f"⚠ Inner join reduced data from {results\_orig\_len} to {results\_new\_len} rows")  
          
        \# ===== VALIDATE DATA EXISTS =====  
        \# Drop initial rows where benchmark data is NaN  
        bench\_open\_col = f"{ticker\_name}\_Open"  
        bench\_close\_col = f"{ticker\_name}\_Close"  
          
        initial\_len = len(results)  
        results = results.dropna(subset=\[bench\_open\_col, bench\_close\_col\]).reset\_index(drop=True)  
        dropped = initial\_len - len(results)  
          
        if dropped > 0:  
            print(f"⚠ Dropped {dropped} rows with missing benchmark data")  
          
        if len(results) == 0:  
            raise ValueError(f"No valid data after merging {ticker\_name} benchmark")  
          
        \# ===== VALIDATE FIRST PRICES =====  
        first\_open = results.loc\[0, bench\_open\_col\]  
        first\_close = results.loc\[0, bench\_close\_col\]  
          
        \# Assert non-NaN before proceeding  
        assert not np.isnan(first\_open), f"First Open price is NaN for {ticker\_name}"  
        assert not np.isnan(first\_close), f"First Close price is NaN for {ticker\_name}"  
        assert first\_open > 0, f"First Open price is non-positive for {ticker\_name}"  
        assert first\_close > 0, f"First Close price is non-positive for {ticker\_name}"  
          
        print(f"✓ Benchmark data validated: {len(results)} rows, first Open=${first\_open:.2f}, first Close=${first\_close:.2f}")  
          
        \# ===== CALCULATE BENCHMARK PORTFOLIO =====  
        \# Buy at first Open with commission  
        shares = self.initial\_capital / (first\_open \* (1 + self.commission\_rate))  
        commission\_buy = shares \* first\_open \* self.commission\_rate  
          
        \# Portfolio value during holding: mark-to-market at Close (CONSISTENT)  
        col\_prefix = f"{ticker\_name}\_BH"  
        results\[f"{col\_prefix}\_Value"\] = shares \* results\[bench\_close\_col\]  
          
        \# Adjust first day for buy commission  
        results.loc\[0, f"{col\_prefix}\_Value"\] = self.initial\_capital - commission\_buy  
          
        \# Adjust last day for sell commission (if sold at Close)  
        last\_idx = len(results) - 1  
        last\_close = results.loc\[last\_idx, bench\_close\_col\]  
        commission\_sell = shares \* last\_close \* self.commission\_rate  
        results.loc\[last\_idx, f"{col\_prefix}\_Value"\] = shares \* last\_close - commission\_sell  
          
        \# Calculate returns (Close-to-Close, CONSISTENT)  
        results\[f"{col\_prefix}\_Returns"\] = results\[bench\_close\_col\].pct\_change()  
        results\[f"{col\_prefix}\_Cumulative"\] = (1 + results\[f"{col\_prefix}\_Returns"\]).cumprod()  
          
        \# ===== SUMMARY =====  
        final\_value = results.loc\[last\_idx, f"{col\_prefix}\_Value"\]  
        total\_return = (final\_value / self.initial\_capital - 1) \* 100  
          
        print(f"✓ {ticker\_name} benchmark added:")  
        print(f"  Final value: ${final\_value:,.2f}")  
        print(f"  Total return: {total\_return:.2f}%")  
          
        return results  
  
  
\# ============================================================================  
\# PERFORMANCE METRICS  
\# ============================================================================  
  
class PerformanceMetrics:  
    """  
    Calculate comprehensive performance metrics.  
      
    All metrics calculated on Close-to-Close returns for consistency.  
    """  
      
    @staticmethod  
    def calculate\_all\_metrics(results: pd.DataFrame, rf\_rate: float = 0.02) -> Dict:  
        """  
        Calculate metrics for strategy and all benchmarks.  
          
        Parameters:  
        -----------  
        results : pd.DataFrame  
            Backtest results with returns  
        rf\_rate : float  
            Risk-free rate for Sharpe calculation (default 2% = US Treasury)  
          
        Returns:  
        --------  
        Dict with metrics for each strategy/benchmark  
        """  
        metrics = {}  
          
        \# ===== STRATEGY METRICS =====  
        strategy\_returns = results\["Strategy\_Returns"\].dropna()  
        n\_years = len(strategy\_returns) / 252  \# Convert trading days to years  
          
        \# Calculate max drawdown  
        cumulative = (strategy\_returns + 1).cumprod()  
        running\_max = cumulative.expanding().max()  
        drawdown = (cumulative / running\_max - 1)  
        max\_dd = drawdown.min()  
          
        metrics\["Strategy"\] = {  
            "Total Return": (results.iloc\[-1\]\["Portfolio\_Value"\] / results.iloc\[0\]\["Portfolio\_Value"\]) - 1,  
            "CAGR": ((results.iloc\[-1\]\["Portfolio\_Value"\] / results.iloc\[0\]\["Portfolio\_Value"\]) \*\* (1/n\_years) - 1) if n\_years > 0 else 0,  
            "Volatility": strategy\_returns.std() \* np.sqrt(252),  
            "Sharpe": np.sqrt(252) \* (strategy\_returns.mean() - rf\_rate/252) / strategy\_returns.std() if strategy\_returns.std() > 0 else 0,  
            "Max Drawdown": max\_dd,  
            "Win Rate": (strategy\_returns > 0).sum() / len(strategy\_returns),  
            "Final Value": results.iloc\[-1\]\["Portfolio\_Value"\]  
        }  
          
        \# ===== BENCHMARK METRICS =====  
        for col\_prefix in \["SPY\_BH", "URTH\_BH"\]:  
            if f"{col\_prefix}\_Returns" in results.columns:  
                bench\_returns = results\[f"{col\_prefix}\_Returns"\].dropna()  
                bench\_cumulative = (bench\_returns + 1).cumprod()  
                bench\_running\_max = bench\_cumulative.expanding().max()  
                bench\_drawdown = (bench\_cumulative / bench\_running\_max - 1)  
                bench\_max\_dd = bench\_drawdown.min()  
                  
                metrics\[col\_prefix.replace("\_BH", "")\] = {  
                    "Total Return": (results.iloc\[-1\]\[f"{col\_prefix}\_Value"\] / results.iloc\[0\]\[f"{col\_prefix}\_Value"\]) - 1,  
                    "CAGR": ((results.iloc\[-1\]\[f"{col\_prefix}\_Value"\] / results.iloc\[0\]\[f"{col\_prefix}\_Value"\]) \*\* (1/n\_years) - 1) if n\_years > 0 else 0,  
                    "Volatility": bench\_returns.std() \* np.sqrt(252),  
                    "Sharpe": np.sqrt(252) \* (bench\_returns.mean() - rf\_rate/252) / bench\_returns.std() if bench\_returns.std() > 0 else 0,  
                    "Max Drawdown": bench\_max\_dd,  
                    "Final Value": results.iloc\[-1\]\[f"{col\_prefix}\_Value"\]  
                }  
          
        return metrics  
      
    @staticmethod  
    def print\_metrics(metrics: Dict):  
        """Print formatted metrics table."""  
        print("\\n" + "="\*100)  
        print("PERFORMANCE METRICS (WALK-FORWARD VALIDATED - NO LOOK-AHEAD BIAS)")  
        print("="\*100)  
          
        \# Print each strategy/benchmark  
        for name, m in metrics.items():  
            print(f"\\n{name}:")  
            print(f"  Total Return:     {m\['Total Return'\]:>10.2%}")  
            print(f"  CAGR:             {m\['CAGR'\]:>10.2%}")  
            print(f"  Volatility:       {m\['Volatility'\]:>10.2%}")  
            print(f"  Sharpe Ratio:     {m\['Sharpe'\]:>10.2f}")  
            print(f"  Max Drawdown:     {m\['Max Drawdown'\]:>10.2%}")  
            if 'Win Rate' in m:  
                print(f"  Win Rate:         {m\['Win Rate'\]:>10.2%}")  
            print(f"  Final Value:      ${m\['Final Value'\]:>10,.2f}")  
          
        \# ===== COMPARATIVE METRICS =====  
        if "Strategy" in metrics and "URTH" in metrics:  
            print(f"\\nRelative to URTH (Global Benchmark):")  
            print(f"  Excess Return:    {metrics\['Strategy'\]\['Total Return'\] - metrics\['URTH'\]\['Total Return'\]:>10.2%}")  
            print(f"  Excess CAGR:      {metrics\['Strategy'\]\['CAGR'\] - metrics\['URTH'\]\['CAGR'\]:>10.2%}")  
            print(f"  Sharpe Advantage: {metrics\['Strategy'\]\['Sharpe'\] - metrics\['URTH'\]\['Sharpe'\]:>10.2f}")  
          
        if "Strategy" in metrics and "SPY" in metrics:  
            print(f"\\nRelative to SPY:")  
            print(f"  Excess Return:    {metrics\['Strategy'\]\['Total Return'\] - metrics\['SPY'\]\['Total Return'\]:>10.2%}")  
            print(f"  Excess CAGR:      {metrics\['Strategy'\]\['CAGR'\] - metrics\['SPY'\]\['CAGR'\]:>10.2%}")  
            print(f"  Sharpe Advantage: {metrics\['Strategy'\]\['Sharpe'\] - metrics\['SPY'\]\['Sharpe'\]:>10.2f}")  
          
        print("="\*100 + "\\n")  
  
  
\# ============================================================================  
\# VISUALIZATION  
\# ============================================================================  
  
def plot\_results(results: pd.DataFrame, config: StrategyConfig):  
    """  
    Create comprehensive visualization of backtest results.  
      
    Three subplots:  
    1. Price with regime coloring  
    2. Cumulative returns comparison  
    3. Strategy drawdown  
    """  
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))  
      
    \# ===== PLOT 1: REGIME SCATTER =====  
    regime\_colors = {"Bearish": "red", "Neutral": "gray", "Bullish": "green"}  
      
    for regime, color in regime\_colors.items():  
        mask = results\["Regime"\] == regime  
        if mask.sum() > 0:  
            axes\[0\].scatter(  
                results.loc\[mask, "Date"\],   
                results.loc\[mask, "Close"\],  
                c=color,   
                label=regime,   
                alpha=0.6,   
                s=10  
            )  
      
    axes\[0\].set\_ylabel("Price ($)", fontsize=12)  
    axes\[0\].set\_title(  
        f"{config.MAIN\_TICKER} Price with GMM Regimes (Walk-Forward Validated)",   
        fontsize=14,   
        fontweight="bold"  
    )  
    axes\[0\].legend(loc="best")  
    axes\[0\].grid(alpha=0.3)  
      
    \# ===== PLOT 2: CUMULATIVE RETURNS =====  
    axes\[1\].plot(  
        results\["Date"\],   
        100 \* results\["Cumulative\_Returns"\],  
        label="GMM Strategy",   
        linewidth=2,   
        color="blue"  
    )  
      
    if "SPY\_BH\_Cumulative" in results.columns:  
        axes\[1\].plot(  
            results\["Date"\],   
            100 \* results\["SPY\_BH\_Cumulative"\],  
            label="SPY Buy & Hold",   
            linewidth=2,   
            alpha=0.7,   
            color="orange"  
        )  
      
    if "URTH\_BH\_Cumulative" in results.columns:  
        axes\[1\].plot(  
            results\["Date"\],   
            100 \* results\["URTH\_BH\_Cumulative"\],  
            label="URTH Buy & Hold",   
            linewidth=2,   
            alpha=0.7,   
            color="purple"  
        )  
      
    axes\[1\].set\_ylabel("Cumulative Returns (Base 100)", fontsize=12)  
    axes\[1\].set\_title("Strategy Performance vs Benchmarks (Close-to-Close Returns)", fontsize=14, fontweight="bold")  
    axes\[1\].legend(loc="best")  
    axes\[1\].grid(alpha=0.3)  
      
    \# ===== PLOT 3: DRAWDOWN =====  
    strategy\_cum = (1 + results\["Strategy\_Returns"\]).cumprod()  
    strategy\_dd = (strategy\_cum / strategy\_cum.expanding().max() - 1) \* 100  
      
    axes\[2\].fill\_between(  
        results\["Date"\],   
        strategy\_dd,   
        0,   
        alpha=0.3,   
        color="red",  
        label="Drawdown"  
    )  
    axes\[2\].plot(  
        results\["Date"\],   
        strategy\_dd,   
        color="darkred",   
        linewidth=1  
    )  
      
    axes\[2\].set\_xlabel("Date", fontsize=12)  
    axes\[2\].set\_ylabel("Drawdown (%)", fontsize=12)  
    axes\[2\].set\_title("Strategy Drawdown", fontsize=14, fontweight="bold")  
    axes\[2\].grid(alpha=0.3)  
    axes\[2\].legend(loc="best")  
      
    plt.tight\_layout()  
    plt.show()  
  
  
def plot\_regime\_characteristics(results: pd.DataFrame, config: StrategyConfig):  
    """  
    Plot regime characteristics to understand what each regime captures.  
    """  
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  
      
    df\_analysis = results.dropna(subset=\["Regime"\])  
      
    regime\_order = \["Bearish", "Neutral", "Bullish"\]  
    colors = \["red", "gray", "green"\]  
      
    \# ===== PLOT 1: YANG-ZHANG VOLATILITY BY REGIME =====  
    for i, regime in enumerate(regime\_order):  
        mask = df\_analysis\["Regime"\] == regime  
        if mask.sum() > 0:  
            data = df\_analysis.loc\[mask, "YangZhang\_Vol"\]  
            axes\[0, 0\].hist(  
                data,   
                bins=30,   
                alpha=0.6,   
                label=regime,   
                color=colors\[i\]  
            )  
      
    axes\[0, 0\].set\_xlabel("Yang-Zhang Volatility", fontsize=11)  
    axes\[0, 0\].set\_ylabel("Frequency", fontsize=11)  
    axes\[0, 0\].set\_title("Volatility Distribution by Regime", fontsize=12, fontweight="bold")  
    axes\[0, 0\].legend()  
    axes\[0, 0\].grid(alpha=0.3)  
      
    \# ===== PLOT 2: SMA CROSSOVER BY REGIME =====  
    for i, regime in enumerate(regime\_order):  
        mask = df\_analysis\["Regime"\] == regime  
        if mask.sum() > 0:  
            data = df\_analysis.loc\[mask, "SMA\_Cross\_Norm"\]  
            axes\[0, 1\].hist(  
                data,   
                bins=30,   
                alpha=0.6,   
                label=regime,   
                color=colors\[i\]  
            )  
      
    axes\[0, 1\].set\_xlabel("SMA Crossover (Normalized)", fontsize=11)  
    axes\[0, 1\].set\_ylabel("Frequency", fontsize=11)  
    axes\[0, 1\].set\_title("SMA Crossover Distribution by Regime", fontsize=12, fontweight="bold")  
    axes\[0, 1\].legend()  
    axes\[0, 1\].grid(alpha=0.3)  
      
    \# ===== PLOT 3: DAILY RETURNS BY REGIME =====  
    regime\_returns = \[\]  
      
    for regime in regime\_order:  
        mask = df\_analysis\["Regime"\] == regime  
        if mask.sum() > 0:  
            regime\_returns.append(df\_analysis.loc\[mask, "Returns\_CC"\].dropna())  
      
    bp = axes\[1, 0\].boxplot(  
        regime\_returns,   
        labels=regime\_order,  
        patch\_artist=True  
    )  
      
    for patch, color in zip(bp\['boxes'\], colors):  
        patch.set\_facecolor(color)  
        patch.set\_alpha(0.6)  
      
    axes\[1, 0\].set\_ylabel("Daily Returns", fontsize=11)  
    axes\[1, 0\].set\_title("Daily Return Distribution by Regime", fontsize=12, fontweight="bold")  
    axes\[1, 0\].grid(alpha=0.3, axis='y')  
    axes\[1, 0\].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)  
      
    \# ===== PLOT 4: REGIME TRANSITIONS =====  
    regime\_transitions = pd.crosstab(  
        df\_analysis\["Regime"\],   
        df\_analysis\["Regime"\].shift(-1),  
        normalize="index"  
    )  
      
    for regime in regime\_order:  
        if regime not in regime\_transitions.index:  
            regime\_transitions.loc\[regime\] = 0  
        if regime not in regime\_transitions.columns:  
            regime\_transitions\[regime\] = 0  
      
    regime\_transitions = regime\_transitions.loc\[regime\_order, regime\_order\]  
      
    sns.heatmap(  
        regime\_transitions,  
        annot=True,  
        fmt=".2f",  
        cmap="YlOrRd",  
        ax=axes\[1, 1\],  
        cbar\_kws={"label": "Transition Probability"}  
    )  
      
    axes\[1, 1\].set\_xlabel("Next Regime", fontsize=11)  
    axes\[1, 1\].set\_ylabel("Current Regime", fontsize=11)  
    axes\[1, 1\].set\_title("Regime Transition Matrix", fontsize=12, fontweight="bold")  
      
    plt.tight\_layout()  
    plt.show()  
  
  
\# ============================================================================  
\# MAIN EXECUTION  
\# ============================================================================  
  
def main():  
    """  
    Main execution function with complete walk-forward validation pipeline.  
    """  
      
    print("\\n" + "="\*100)  
    print("GMM REGIME DETECTION STRATEGY - PRODUCTION-LEVEL IMPLEMENTATION")  
    print("="\*100)  
      
    \# ===== INITIALIZE CONFIGURATION =====  
    config = StrategyConfig()  
      
    print(f"\\nConfiguration:")  
    print(f"  Date Range: {config.START\_DATE} to {config.END\_DATE}")  
    print(f"  Ticker: {config.MAIN\_TICKER}")  
    print(f"  Benchmark: {config.BENCHMARK\_TICKER}")  
    print(f"  Initial Capital: ${config.INITIAL\_CAPITAL:,.0f}")  
    print(f"  Commission Rate: {config.COMMISSION\_RATE:.3%}")  
    print(f"\\n  Features:")  
    print(f"    - SMA {config.SMA\_SHORT} vs {config.SMA\_LONG} (normalized)")  
    print(f"    - Yang-Zhang volatility (window={config.YZ\_WINDOW})")  
    print(f"\\n  GMM Parameters:")  
    print(f"    - Components: {config.GMM\_N\_COMPONENTS}")  
    print(f"    - Covariance: {config.GMM\_COVARIANCE\_TYPE}")  
    print(f"    - Min cluster samples: {config.MIN\_CLUSTER\_SAMPLES}")  
    print(f"\\n  Walk-Forward:")  
    print(f"    - Min training: {config.MIN\_TRAINING\_DAYS} days")  
    print(f"    - Refit frequency: {config.REFIT\_FREQUENCY} days")  
    print(f"\\n  Trading Logic:")  
    print(f"    - Long regime: {config.LONG\_REGIME}")  
    print(f"    - Short regime: {config.SHORT\_REGIME if config.SHORT\_REGIME else 'None (cash)'}")  
      
    \# ===== DATA ACQUISITION =====  
    print("\\n" + "-"\*100)  
    print("DATA ACQUISITION")  
    print("-"\*100)  
      
    fetcher = DataFetcher(config.FMP\_API\_KEY)  
      
    df\_spy = fetcher.fetch\_historical\_data(  
        config.MAIN\_TICKER,   
        config.START\_DATE,   
        config.END\_DATE  
    )  
      
    df\_urth = fetcher.fetch\_historical\_data(  
        config.BENCHMARK\_TICKER,   
        config.START\_DATE,   
        config.END\_DATE  
    )  
      
    \# ===== FEATURE ENGINEERING =====  
    print("\\n" + "-"\*100)  
    print("FEATURE ENGINEERING")  
    print("-"\*100)  
      
    df\_features = FeatureEngine.prepare\_features(df\_spy, config)  
      
    \# ===== WALK-FORWARD REGIME DETECTION =====  
    detector = WalkForwardGMMRegimeDetector(config)  
    df\_regimes, folds = detector.walk\_forward\_predict(df\_features)  
      
    \# ===== BACKTESTING =====  
    print("\\n" + "-"\*100)  
    print("BACKTESTING")  
    print("-"\*100)  
      
    engine = BacktestEngine(config.INITIAL\_CAPITAL, config.COMMISSION\_RATE)  
      
    results = engine.run\_backtest(  
        df\_regimes,  
        long\_regime=config.LONG\_REGIME,  
        short\_regime=config.SHORT\_REGIME  
    )  
      
    \# ===== ADD BENCHMARKS =====  
    print("\\nAdding benchmarks...")  
      
    \# SPY benchmark  
    results = engine.add\_buy\_hold\_benchmark(results, df\_regimes, "SPY")  
      
    \# URTH benchmark  
    results = engine.add\_buy\_hold\_benchmark(results, df\_urth, "URTH")  
      
    \# ===== PERFORMANCE ANALYSIS =====  
    print("\\n" + "-"\*100)  
    print("PERFORMANCE ANALYSIS")  
    print("-"\*100)  
      
    metrics = PerformanceMetrics.calculate\_all\_metrics(results)  
    PerformanceMetrics.print\_metrics(metrics)  
      
    \# ===== REGIME ANALYSIS =====  
    print("\\n" + "-"\*100)  
    print("REGIME STATISTICS")  
    print("-"\*100)  
      
    regime\_stats = results.groupby("Regime").agg({  
        "Returns\_CC": \["count", "mean", "std"\],  
        "YangZhang\_Vol": "mean",  
        "SMA\_Cross\_Norm": "mean"  
    }).round(4)  
      
    print("\\nRegime characteristics:")  
    print(regime\_stats)  
      
    \# ===== VISUALIZATIONS =====  
    print("\\n" + "-"\*100)  
    print("GENERATING VISUALIZATIONS")  
    print("-"\*100)  
      
    plot\_results(results, config)  
    plot\_regime\_characteristics(results, config)  
      
    \# ===== EXPORT RESULTS =====  
    timestamp = datetime.now().strftime('%Y%m%d\_%H%M%S')  
      
    results.to\_csv(f"gmm\_walkforward\_results\_{timestamp}.csv", index=False)  
    pd.DataFrame(engine.trades).to\_csv(f"gmm\_walkforward\_trades\_{timestamp}.csv", index=False)  
    pd.DataFrame(folds).to\_csv(f"gmm\_walkforward\_folds\_{timestamp}.csv", index=False)  
      
    print(f"\\n✓ Results exported with timestamp {timestamp}")  
      
    \# ===== FINAL SUMMARY =====  
    print("\\n" + "="\*100)  
    print("EXECUTION COMPLETE - PRODUCTION-READY")  
    print("="\*100)  
      
    print(f"\\nKey Results:")  
    print(f"  Strategy Return: {metrics\['Strategy'\]\['Total Return'\]:.2%}")  
    print(f"  Strategy CAGR: {metrics\['Strategy'\]\['CAGR'\]:.2%}")  
    print(f"  Strategy Sharpe: {metrics\['Strategy'\]\['Sharpe'\]:.2f}")  
      
    if "URTH" in metrics:  
        print(f"  URTH Return: {metrics\['URTH'\]\['Total Return'\]:.2%}")  
        print(f"  URTH CAGR: {metrics\['URTH'\]\['CAGR'\]:.2%}")  
        print(f"  Excess Return: {metrics\['Strategy'\]\['Total Return'\] - metrics\['URTH'\]\['Total Return'\]:.2%}")  
      
    print("\\n" + "="\*100 + "\\n")  
      
    return results, metrics, folds  
  
  
if \_\_name\_\_ == "\_\_main\_\_":  
    results, metrics, folds = main()

## Final Thoughts

In an era of increasing market volatility and uncertainty, strategies that dynamically adjust risk exposure will likely outperform static approaches on a risk-adjusted basis.

The 1.00 Sharpe ratio achieved here isn’t magic — it’s the result of:

1.  Sound feature engineering (volatility + momentum)
2.  Appropriate algorithm selection (GMM)
3.  Rigorous validation (walk-forward)
4.  Disciplined execution (systematic rules)

As more investors embrace quantitative approaches, the key differentiator won’t be finding alpha (increasingly difficult) but **managing risk intelligently**.

Regime detection is one powerful tool in that arsenal.

_Disclaimer: This article is for educational purposes only and does not constitute investment advice. Past performance does not guarantee future results. All trading involves risk of loss._

**Questions? Thoughts?** Leave a comment below or connect with me to discuss regime-switching strategies, quantitative finance, or machine learning applications in trading.

_If you found this valuable, consider following for more deep dives into systematic trading strategies._

## Embedded Content

---