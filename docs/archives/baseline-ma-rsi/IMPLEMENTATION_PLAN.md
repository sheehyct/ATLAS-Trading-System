# Implementation Plan: Step-by-Step Build Guide

## Overview

This document provides the detailed implementation sequence for both strategy branches. Each phase includes code examples, testing requirements, and validation checkpoints.

**Estimated Timeline:**
- Phase 1 (Setup): 1 week
- Phase 2 (Branch 2 - Baseline): 2-3 weeks
- Phase 3 (Branch 1 - TFC): 2-3 weeks
- Phase 4 (Validation): 1-2 weeks
- Phase 5 (Paper Trading): 6+ months

**Recommendation:** Build Branch 2 (baseline) first to establish benchmark, then Branch 1 (TFC).

---

## Phase 1: Environment Setup & Data Validation

### 1.1 Verify VectorBT Pro Installation

```bash
# Check VectorBT Pro is installed
python -c "import vectorbtpro as vbt; print(vbt.__version__)"

# Should output version 1.x.x or higher
```

**If not installed:**
```bash
# Follow VectorBT Pro installation guide
# Requires GitHub access token for private repo
```

### 1.2 Configure Alpaca API

**Create `.env` file in project root:**
```bash
# .env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Paper Account 1 (Conservative - $10k)
ALPACA_PAPER_1_KEY=...
ALPACA_PAPER_1_SECRET=...

# Paper Account 2 (Moderate - $25k)
ALPACA_PAPER_2_KEY=...
ALPACA_PAPER_2_SECRET=...

# Paper Account 3 (Aggressive - $50k)
ALPACA_PAPER_3_KEY=...
ALPACA_PAPER_3_SECRET=...
```

**Test Alpaca connection:**
```python
# test_alpaca_connection.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv

load_dotenv()

client = StockHistoricalDataClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY')
)

# Test data fetch
request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Day,
    start="2024-01-01",
    end="2024-12-31"
)

bars = client.get_stock_bars(request)
print(f"✅ Fetched {len(bars.df)} bars for SPY")
print(bars.df.head())
```

### 1.3 Validate Existing TFC Code

**Test TFC calculation works:**
```python
# test_tfc_calculation.py
from data.alpaca import fetch_stock_data
from data.mtf_manager import MultiTimeframeManager
from core.analyzer import STRATAnalyzer

# Fetch data
symbol = "SPY"
start_date = "2024-01-01"
end_date = "2024-12-31"

data = fetch_stock_data(symbol, start_date, end_date)

# Create multi-timeframe data
mtf = MultiTimeframeManager(data)
hourly = mtf.resample('1H')
daily = mtf.resample('1D')
weekly = mtf.resample('1W')

# Calculate bar classifications
analyzer = STRATAnalyzer()
hourly_class = analyzer.classify_bars(hourly)
daily_class = analyzer.classify_bars(daily)
weekly_class = analyzer.classify_bars(weekly)

# Calculate TFC (you'll need to add this method if it doesn't exist)
# tfc_score = analyzer.calculate_tfc(hourly_class, daily_class, weekly_class)

print("✅ TFC calculation successful")
print(f"Hourly bars classified: {len(hourly_class)}")
print(f"Daily bars classified: {len(daily_class)}")
print(f"Weekly bars classified: {len(weekly_class)}")
```

**Checkpoint:** All tests pass before proceeding to Phase 2.

---

## Phase 2: Branch 2 - Baseline Implementation (Build This First)

### 2.1 Create Baseline Strategy Module

**Create `strategies/baseline_ma_rsi.py`:**

```python
"""
Baseline Strategy: 200-day MA + RSI(2) Mean Reversion

Research-proven strategy with documented 75% win rates over 30 years.
Serves as benchmark for TFC-based approach.
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class BaselineStrategy:
    """
    Simple momentum-mean reversion strategy.

    Entry Rules:
    - Long: Price > 200-day MA AND RSI(2) < 15
    - Short: Price < 200-day MA AND RSI(2) > 85

    Exit Rules:
    - Profit target: 2x ATR
    - Stop loss: 2x ATR
    - Time exit: 14 days max hold
    - Overbought/oversold: RSI(2) crosses opposite threshold
    """

    def __init__(
        self,
        ma_period: int = 200,
        rsi_period: int = 2,
        rsi_oversold: float = 15.0,
        rsi_overbought: float = 85.0,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        max_hold_days: int = 14,
    ):
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.max_hold_days = max_hold_days

    def generate_signals(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate long and short entry/exit signals.

        Returns:
            long_entries, long_exits, short_entries, short_exits
        """
        # Calculate indicators
        ma200 = vbt.talib("SMA").run(close, timeperiod=self.ma_period).real
        rsi = vbt.talib("RSI").run(close, timeperiod=self.rsi_period).real

        # Use close for ATR if high/low not provided
        if high is None or low is None:
            high = close
            low = close
        atr = vbt.talib("ATR").run(high, low, close, timeperiod=self.atr_period).real

        # Long signals
        uptrend = close > ma200
        oversold = rsi < self.rsi_oversold
        long_entries = uptrend & oversold

        overbought = rsi > self.rsi_overbought
        long_exits = overbought

        # Short signals
        downtrend = close < ma200
        short_entries = downtrend & overbought
        short_exits = oversold

        return long_entries, long_exits, short_entries, short_exits, atr

    def backtest(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        open_price: Optional[pd.Series] = None,
        init_cash: float = 10000.0,
        fees: float = 0.002,  # 0.2% per trade
        slippage: float = 0.001,  # 0.1% slippage
    ) -> vbt.Portfolio:
        """
        Run backtest and return portfolio object.
        """
        # Generate signals
        long_entries, long_exits, short_entries, short_exits, atr = \
            self.generate_signals(close, high, low)

        # Calculate stops and targets
        stop_distance = atr * self.atr_multiplier

        # Position sizing: 2% risk per trade
        position_size = init_cash * 0.02 / stop_distance

        # Run portfolio simulation
        pf = vbt.PF.from_signals(
            close=close,
            open=open_price,
            high=high,
            low=low,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            size=position_size,
            size_type="amount",
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            sl_stop=stop_distance,  # Stop loss
            tp_stop=stop_distance * 2,  # Take profit (2:1 ratio)
            td_stop=pd.Timedelta(days=self.max_hold_days),  # Time-based exit
        )

        return pf


# Example usage
if __name__ == "__main__":
    # Fetch data
    from data.alpaca import fetch_stock_data

    data = fetch_stock_data("SPY", "2020-01-01", "2024-12-31")

    # Run baseline strategy
    strategy = BaselineStrategy()
    pf = strategy.backtest(
        close=data['close'],
        high=data['high'],
        low=data['low'],
        open_price=data['open'],
    )

    # Print results
    print(f"Total Return: {pf.total_return():.2%}")
    print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {pf.max_drawdown():.2%}")
    print(f"Win Rate: {pf.trades.win_rate():.2%}")
    print(f"Total Trades: {pf.trades.count()}")
```

### 2.2 Test Baseline Strategy

**Create `tests/test_baseline_strategy.py`:**

```python
import pytest
from strategies.baseline_ma_rsi import BaselineStrategy
from data.alpaca import fetch_stock_data
import pandas as pd


def test_baseline_strategy_signals():
    """Test that signals are generated correctly."""
    # Create synthetic data
    close = pd.Series([100, 102, 98, 95, 97, 99, 101, 103, 100, 98],
                      index=pd.date_range('2024-01-01', periods=10, freq='D'))

    strategy = BaselineStrategy(ma_period=5, rsi_period=2)
    long_entries, long_exits, short_entries, short_exits, atr = \
        strategy.generate_signals(close)

    # Signals should be boolean Series
    assert isinstance(long_entries, pd.Series)
    assert long_entries.dtype == bool

    # Should have same index as input
    assert len(long_entries) == len(close)


def test_baseline_strategy_backtest():
    """Test backtest runs without errors."""
    # Fetch real data
    data = fetch_stock_data("SPY", "2024-01-01", "2024-06-30")

    strategy = BaselineStrategy()
    pf = strategy.backtest(
        close=data['close'],
        high=data['high'],
        low=data['low'],
        open_price=data['open'],
    )

    # Portfolio should have metrics
    assert pf.total_return() is not None
    assert pf.sharpe_ratio() is not None
    assert pf.max_drawdown() is not None

    # Win rate should be between 0 and 1
    win_rate = pf.trades.win_rate()
    assert 0 <= win_rate <= 1


def test_baseline_vs_buy_hold():
    """Baseline should beat buy-and-hold on SPY (ideally)."""
    data = fetch_stock_data("SPY", "2020-01-01", "2024-12-31")

    # Run strategy
    strategy = BaselineStrategy()
    pf = strategy.backtest(
        close=data['close'],
        high=data['high'],
        low=data['low'],
        open_price=data['open'],
    )

    # Buy-and-hold return
    buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
    strategy_return = pf.total_return()

    print(f"Buy-and-hold: {buy_hold_return:.2%}")
    print(f"Strategy: {strategy_return:.2%}")
    print(f"Sharpe: {pf.sharpe_ratio():.2f}")
    print(f"Win Rate: {pf.trades.win_rate():.2%}")

    # This test may fail if strategy doesn't work - that's OK, it's a check
    # Don't assert, just print results for analysis
```

**Run tests:**
```bash
pytest tests/test_baseline_strategy.py -v
```

### 2.3 Walk-Forward Optimization

**Create `optimization/walk_forward_baseline.py`:**

```python
"""
Walk-forward optimization for baseline strategy.

Tests parameter combinations on rolling windows to prevent overfitting.
"""

import vectorbtpro as vbt
import pandas as pd
from strategies.baseline_ma_rsi import BaselineStrategy
from data.alpaca import fetch_stock_data


def walk_forward_analysis(
    symbol: str = "SPY",
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    train_period_days: int = 365,  # 1 year training
    test_period_days: int = 90,    # 3 months testing
    step_forward_days: int = 30,   # 1 month step
):
    """
    Perform walk-forward analysis.

    Returns:
        DataFrame with results from each window
    """
    # Fetch data
    data = fetch_stock_data(symbol, start_date, end_date)

    results = []
    current_date = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    while current_date + pd.Timedelta(days=train_period_days + test_period_days) < end_ts:
        # Training window
        train_start = current_date
        train_end = current_date + pd.Timedelta(days=train_period_days)

        # Testing window
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_period_days)

        # Split data
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]

        if len(train_data) < 50 or len(test_data) < 20:
            # Not enough data
            current_date += pd.Timedelta(days=step_forward_days)
            continue

        # Optimize on training data (simple grid search)
        best_sharpe = -999
        best_params = None

        for rsi_threshold in [10, 15, 20]:
            strategy = BaselineStrategy(
                rsi_oversold=rsi_threshold,
                rsi_overbought=100 - rsi_threshold
            )

            pf_train = strategy.backtest(
                close=train_data['close'],
                high=train_data['high'],
                low=train_data['low'],
                open_price=train_data['open'],
            )

            sharpe = pf_train.sharpe_ratio()
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'rsi_threshold': rsi_threshold}

        # Test on out-of-sample data
        test_strategy = BaselineStrategy(
            rsi_oversold=best_params['rsi_threshold'],
            rsi_overbought=100 - best_params['rsi_threshold']
        )

        pf_test = test_strategy.backtest(
            close=test_data['close'],
            high=test_data['high'],
            low=test_data['low'],
            open_price=test_data['open'],
        )

        # Store results
        results.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'optimal_params': best_params,
            'train_sharpe': best_sharpe,
            'test_sharpe': pf_test.sharpe_ratio(),
            'test_return': pf_test.total_return(),
            'test_win_rate': pf_test.trades.win_rate(),
            'test_max_dd': pf_test.max_drawdown(),
            'test_trades': pf_test.trades.count(),
        })

        print(f"Window {test_start.date()}: Train Sharpe={best_sharpe:.2f}, "
              f"Test Sharpe={pf_test.sharpe_ratio():.2f}, Trades={pf_test.trades.count()}")

        # Step forward
        current_date += pd.Timedelta(days=step_forward_days)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running walk-forward analysis on baseline strategy...")
    results = walk_forward_analysis()

    print("\n=== Walk-Forward Results ===")
    print(f"Average Test Sharpe: {results['test_sharpe'].mean():.2f}")
    print(f"Average Test Return: {results['test_return'].mean():.2%}")
    print(f"Average Win Rate: {results['test_win_rate'].mean():.2%}")
    print(f"Average Max Drawdown: {results['test_max_dd'].mean():.2%}")

    # Check for overfitting
    sharpe_diff = results['train_sharpe'].mean() - results['test_sharpe'].mean()
    print(f"\nTrain-Test Sharpe Gap: {sharpe_diff:.2f}")
    if sharpe_diff > 0.5:
        print("⚠️  WARNING: Large train-test gap indicates potential overfitting")
    else:
        print("✅ Train-test gap acceptable, strategy appears robust")

    # Save results
    results.to_csv('baseline_walk_forward_results.csv', index=False)
    print("\n✅ Results saved to baseline_walk_forward_results.csv")
```

**Checkpoint:** Walk-forward results show consistent out-of-sample performance (test Sharpe within 0.3 of train Sharpe).

---

## Phase 3: Branch 1 - TFC Confidence Score Implementation

### 3.1 Create Confidence Scoring Module

**Create `strategies/tfc_confidence_score.py`:**

```python
"""
TFC Confidence Score Strategy: Multi-factor momentum-mean reversion

Combines TFC (Time Frame Continuity) with traditional indicators
in a weighted confidence scoring system.
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from core.analyzer import STRATAnalyzer
from data.mtf_manager import MultiTimeframeManager


class TFCConfidenceStrategy:
    """
    Advanced strategy using multi-factor confidence scoring.

    Confidence Score (0-100):
    - TFC Score: 30% weight (multi-timeframe trend alignment)
    - RSI Score: 40% weight (mean reversion strength)
    - MACD Score: 20% weight (momentum confirmation)
    - Volume Score: 10% weight (volume surge confirmation)

    Entry Thresholds:
    - High confidence (70+): Full position (2% risk)
    - Medium confidence (50-69): Half position (1% risk)
    - Low confidence (<50): No trade
    """

    def __init__(
        self,
        tfc_weight: float = 0.30,
        rsi_weight: float = 0.40,
        macd_weight: float = 0.20,
        volume_weight: float = 0.10,
        high_confidence_threshold: float = 70.0,
        medium_confidence_threshold: float = 50.0,
        rsi_period: int = 2,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        max_hold_days: int = 14,
    ):
        # Weights must sum to 1.0
        total_weight = tfc_weight + rsi_weight + macd_weight + volume_weight
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"

        self.tfc_weight = tfc_weight
        self.rsi_weight = rsi_weight
        self.macd_weight = macd_weight
        self.volume_weight = volume_weight
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.max_hold_days = max_hold_days

        self.analyzer = STRATAnalyzer()

    def calculate_tfc_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate TFC score (0-100) from multi-timeframe data.

        Uses existing STRAT TFC logic from core/analyzer.py
        """
        # Create multi-timeframe manager
        mtf = MultiTimeframeManager(data)

        # Resample to different timeframes
        hourly = mtf.resample('1H')
        daily = mtf.resample('1D')
        weekly = mtf.resample('1W')

        # Classify bars on each timeframe
        hourly_class = self.analyzer.classify_bars(hourly)
        daily_class = self.analyzer.classify_bars(daily)
        weekly_class = self.analyzer.classify_bars(weekly)

        # Calculate alignment (simplified - you'll need to implement full TFC logic)
        # For now, use directional alignment as proxy

        # This is a placeholder - replace with your actual TFC calculation
        tfc_score = pd.Series(50.0, index=data.index)  # Neutral by default

        # TODO: Implement actual TFC calculation from your existing code
        # tfc_score = self.analyzer.calculate_tfc(hourly_class, daily_class, weekly_class)

        return tfc_score

    def calculate_confidence_score(
        self,
        data: pd.DataFrame,
        tfc_score: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calculate multi-factor confidence score (0-100).
        """
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # Factor 1: TFC Score (0-30 points)
        if tfc_score is None:
            tfc_score = self.calculate_tfc_score(data)
        tfc_contribution = tfc_score * self.tfc_weight

        # Factor 2: RSI Score (0-40 points)
        rsi = vbt.talib("RSI").run(close, timeperiod=self.rsi_period).real
        rsi_score = np.where(
            rsi < 10, 40,  # Extreme oversold = max score
            np.where(rsi < 15, 30,  # Very oversold
            np.where(rsi < 20, 20,  # Oversold
            np.where(rsi > 90, 40,  # Extreme overbought (for shorts)
            np.where(rsi > 85, 30,  # Very overbought
            np.where(rsi > 80, 20,  # Overbought
            0)))))  # Neutral
        )
        rsi_contribution = rsi_score * self.rsi_weight

        # Factor 3: MACD Score (0-20 points)
        macd_result = vbt.talib("MACD").run(close)
        macd_bullish = macd_result.macd > macd_result.macdsignal
        macd_score = np.where(macd_bullish, 20, 0)
        macd_contribution = macd_score * self.macd_weight

        # Factor 4: Volume Score (0-10 points)
        volume_ma = volume.rolling(20).mean()
        volume_surge = volume > (volume_ma * 1.5)
        volume_score = np.where(volume_surge, 10, 0)
        volume_contribution = volume_score * self.volume_weight

        # Total confidence
        confidence = (
            tfc_contribution +
            rsi_contribution +
            macd_contribution +
            volume_contribution
        )

        return pd.Series(confidence, index=close.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        confidence: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Generate entry/exit signals based on confidence thresholds.
        """
        if confidence is None:
            confidence = self.calculate_confidence_score(data)

        close = data['close']
        high = data['high']
        low = data['low']

        # Calculate ATR for stops
        atr = vbt.talib("ATR").run(high, low, close, timeperiod=self.atr_period).real

        # Entry signals at different confidence levels
        high_confidence = confidence >= self.high_confidence_threshold
        medium_confidence = (confidence >= self.medium_confidence_threshold) & \
                           (confidence < self.high_confidence_threshold)

        # RSI for entry timing
        rsi = vbt.talib("RSI").run(close, timeperiod=self.rsi_period).real
        oversold = rsi < 15
        overbought = rsi > 85

        # Combine confidence with oversold/overbought
        high_conf_long = high_confidence & oversold
        medium_conf_long = medium_confidence & oversold

        # Exit on overbought
        exits = overbought

        return {
            'high_confidence_entries': high_conf_long,
            'medium_confidence_entries': medium_conf_long,
            'exits': exits,
            'confidence': confidence,
            'atr': atr,
        }

    def backtest(
        self,
        data: pd.DataFrame,
        init_cash: float = 10000.0,
        fees: float = 0.002,
        slippage: float = 0.001,
    ) -> vbt.Portfolio:
        """
        Run backtest with confidence-based position sizing.
        """
        signals = self.generate_signals(data)

        close = data['close']
        high = data['high']
        low = data['low']
        open_price = data['open']

        # Combine high and medium confidence entries
        all_entries = signals['high_confidence_entries'] | signals['medium_confidence_entries']

        # Position sizing based on confidence
        # High confidence: 2% risk, Medium: 1% risk
        stop_distance = signals['atr'] * self.atr_multiplier
        high_conf_size = init_cash * 0.02 / stop_distance
        medium_conf_size = init_cash * 0.01 / stop_distance

        position_size = np.where(
            signals['high_confidence_entries'], high_conf_size,
            np.where(signals['medium_confidence_entries'], medium_conf_size, 0)
        )

        # Run portfolio simulation
        pf = vbt.PF.from_signals(
            close=close,
            open=open_price,
            high=high,
            low=low,
            entries=all_entries,
            exits=signals['exits'],
            size=position_size,
            size_type="amount",
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            sl_stop=stop_distance,
            tp_stop=stop_distance * 2,
            td_stop=pd.Timedelta(days=self.max_hold_days),
        )

        return pf, signals


# Example usage
if __name__ == "__main__":
    from data.alpaca import fetch_stock_data

    # Fetch data
    data = fetch_stock_data("SPY", "2020-01-01", "2024-12-31")

    # Run TFC strategy
    strategy = TFCConfidenceStrategy()
    pf, signals = strategy.backtest(data)

    # Print results
    print(f"Total Return: {pf.total_return():.2%}")
    print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
    print(f"Max Drawdown: {pf.max_drawdown():.2%}")
    print(f"Win Rate: {pf.trades.win_rate():.2%}")
    print(f"Total Trades: {pf.trades.count()}")

    # Analyze by confidence level
    high_conf_trades = signals['high_confidence_entries'].sum()
    medium_conf_trades = signals['medium_confidence_entries'].sum()
    print(f"\nHigh Confidence Signals: {high_conf_trades}")
    print(f"Medium Confidence Signals: {medium_conf_trades}")
```

**Checkpoint:** TFC confidence strategy runs without errors and produces reasonable signals.

---

## Phase 4: Comparison & Validation

### 4.1 Create Comparison Script

**Create `comparison/compare_strategies.py`:**

```python
"""Compare Baseline vs TFC strategies side-by-side."""

from strategies.baseline_ma_rsi import BaselineStrategy
from strategies.tfc_confidence_score import TFCConfidenceStrategy
from data.alpaca import fetch_stock_data
import pandas as pd


def compare_strategies(
    symbol: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
):
    """
    Run both strategies and compare results.
    """
    # Fetch data
    data = fetch_stock_data(symbol, start_date, end_date)

    # Run Baseline
    print("Running Baseline Strategy...")
    baseline = BaselineStrategy()
    pf_baseline = baseline.backtest(
        close=data['close'],
        high=data['high'],
        low=data['low'],
        open_price=data['open'],
    )

    # Run TFC
    print("Running TFC Confidence Strategy...")
    tfc_strategy = TFCConfidenceStrategy()
    pf_tfc, signals_tfc = tfc_strategy.backtest(data)

    # Compare results
    results = pd.DataFrame({
        'Metric': [
            'Total Return',
            'Sharpe Ratio',
            'Max Drawdown',
            'Win Rate',
            'Total Trades',
            'Avg Trade Return',
        ],
        'Baseline': [
            f"{pf_baseline.total_return():.2%}",
            f"{pf_baseline.sharpe_ratio():.2f}",
            f"{pf_baseline.max_drawdown():.2%}",
            f"{pf_baseline.trades.win_rate():.2%}",
            pf_baseline.trades.count(),
            f"{pf_baseline.trades.returns.mean():.2%}",
        ],
        'TFC Confidence': [
            f"{pf_tfc.total_return():.2%}",
            f"{pf_tfc.sharpe_ratio():.2f}",
            f"{pf_tfc.max_drawdown():.2%}",
            f"{pf_tfc.trades.win_rate():.2%}",
            pf_tfc.trades.count(),
            f"{pf_tfc.trades.returns.mean():.2%}",
        ],
    })

    print("\n=== Strategy Comparison ===")
    print(results.to_string(index=False))

    # Determine winner
    baseline_sharpe = pf_baseline.sharpe_ratio()
    tfc_sharpe = pf_tfc.sharpe_ratio()

    if tfc_sharpe > baseline_sharpe + 0.2:
        print("\n✅ TFC Confidence strategy significantly outperforms baseline")
        print("   → TFC adds value, proceed with this approach")
    elif abs(tfc_sharpe - baseline_sharpe) < 0.2:
        print("\n⚠️  Strategies perform similarly")
        print("   → TFC doesn't add significant value, consider using simpler baseline")
    else:
        print("\n❌ Baseline outperforms TFC")
        print("   → TFC may be overfitting or adding unnecessary complexity")

    return pf_baseline, pf_tfc, results


if __name__ == "__main__":
    compare_strategies()
```

### 4.2 Run Walk-Forward on Both

Compare walk-forward results for both strategies to determine which is more robust.

**Checkpoint:** Decide which strategy to deploy to paper trading based on walk-forward results.

---

## Phase 5: Paper Trading Deployment

### 5.1 Create Paper Trading Bot

**Create `live/paper_trading_bot.py`:**

```python
"""
Paper trading bot for live strategy testing.

Runs strategy in real-time using Alpaca paper trading accounts.
"""

import vectorbtpro as vbt
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os
from dotenv import load_dotenv
import time
import pandas as pd

load_dotenv()


class PaperTradingBot:
    """
    Live paper trading bot.

    Monitors market, generates signals, places orders via Alpaca API.
    """

    def __init__(
        self,
        strategy,
        account_config: str = "PAPER_1",  # PAPER_1, PAPER_2, or PAPER_3
        symbols: list = ["SPY"],
        check_interval_minutes: int = 60,
    ):
        self.strategy = strategy
        self.symbols = symbols
        self.check_interval = check_interval_minutes

        # Initialize Alpaca client
        api_key = os.getenv(f'ALPACA_{account_config}_KEY')
        secret_key = os.getenv(f'ALPACA_{account_config}_SECRET')

        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )

        print(f"✅ Paper trading bot initialized with {account_config}")
        print(f"   Monitoring: {symbols}")
        print(f"   Check interval: {check_interval_minutes} minutes")

    def fetch_latest_data(self, symbol: str, lookback_days: int = 365):
        """Fetch latest market data for signal generation."""
        from data.alpaca import fetch_stock_data

        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=lookback_days)

        return fetch_stock_data(symbol, start_date, end_date)

    def check_for_signals(self):
        """Check each symbol for entry/exit signals."""
        for symbol in self.symbols:
            print(f"\nChecking {symbol}...")

            # Fetch data
            data = self.fetch_latest_data(symbol)

            # Generate signals
            signals = self.strategy.generate_signals(data)

            # Check latest signal
            latest_entry = signals['high_confidence_entries'].iloc[-1] or \
                          signals.get('medium_confidence_entries', pd.Series([False])).iloc[-1]
            latest_exit = signals['exits'].iloc[-1]

            # Check current positions
            positions = self.trading_client.get_all_positions()
            has_position = any(p.symbol == symbol for p in positions)

            if latest_entry and not has_position:
                print(f"📈 ENTRY SIGNAL for {symbol}")
                self.place_entry_order(symbol, data, signals)

            elif latest_exit and has_position:
                print(f"📉 EXIT SIGNAL for {symbol}")
                self.place_exit_order(symbol)

            else:
                print(f"   No action for {symbol}")

    def place_entry_order(self, symbol: str, data: pd.DataFrame, signals: dict):
        """Place entry order with position sizing."""
        # Calculate position size based on 2% risk
        account = self.trading_client.get_account()
        equity = float(account.equity)

        atr = signals['atr'].iloc[-1]
        stop_distance = atr * self.strategy.atr_multiplier
        position_size_dollars = equity * 0.02 / stop_distance

        current_price = data['close'].iloc[-1]
        shares = int(position_size_dollars / current_price)

        if shares < 1:
            print(f"   ⚠️  Position size too small ({shares} shares), skipping")
            return

        # Place market order
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            print(f"   ✅ Order placed: {shares} shares of {symbol}")
            print(f"      Order ID: {order.id}")
        except Exception as e:
            print(f"   ❌ Order failed: {e}")

    def place_exit_order(self, symbol: str):
        """Exit existing position."""
        try:
            self.trading_client.close_position(symbol)
            print(f"   ✅ Position closed for {symbol}")
        except Exception as e:
            print(f"   ❌ Exit failed: {e}")

    def run(self):
        """Main loop - check for signals periodically."""
        print("\n🤖 Paper trading bot started")
        print("   Press Ctrl+C to stop\n")

        try:
            while True:
                print(f"\n⏰ {pd.Timestamp.now()}")
                self.check_for_signals()

                print(f"\n💤 Sleeping for {self.check_interval} minutes...")
                time.sleep(self.check_interval * 60)

        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")


# Example usage
if __name__ == "__main__":
    from strategies.baseline_ma_rsi import BaselineStrategy

    strategy = BaselineStrategy()

    bot = PaperTradingBot(
        strategy=strategy,
        account_config="PAPER_2",  # Use moderate account
        symbols=["SPY", "QQQ"],
        check_interval_minutes=60,
    )

    bot.run()
```

### 5.2 Deploy to 3 Paper Accounts

**Run 3 instances with different configurations:**

```bash
# Terminal 1: Conservative account
python live/paper_trading_bot.py --account PAPER_1 --strategy baseline --confidence-threshold 75

# Terminal 2: Moderate account
python live/paper_trading_bot.py --account PAPER_2 --strategy baseline --confidence-threshold 60

# Terminal 3: Aggressive account
python live/paper_trading_bot.py --account PAPER_3 --strategy tfc --confidence-threshold 50
```

**Monitor for 6+ months, track:**
- Daily performance logs
- Weekly Sharpe ratio calculations
- Monthly comparison reports
- Divergence between accounts

---

## Success Criteria Summary

**After completing all phases, you should have:**

✅ Two working strategies (Baseline and TFC)
✅ Walk-forward validation results
✅ Side-by-side comparison data
✅ 3 paper trading accounts running live
✅ 6+ months of live paper performance data

**Decision point:**
- If TFC > Baseline (Sharpe +0.2): Deploy TFC to live trading
- If similar performance: Use Baseline (simpler)
- If Baseline > TFC: Abandon TFC, use proven approach

---

## Next Steps

1. Read BRANCH_COMPARISON.md for detailed branch analysis
2. Read VALIDATION_PROTOCOL.md for testing requirements
3. Choose which branch to build first (recommend Baseline)
4. Begin Phase 1 implementation
5. Track progress using todo list

**Questions before starting? Review STRATEGY_OVERVIEW.md again.**
