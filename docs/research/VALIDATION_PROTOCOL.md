# Validation Protocol: Testing & Quality Assurance

## Overview

This document defines the testing methodology to ensure strategy robustness before deploying real capital. Follow this protocol strictly to avoid the 44% failure rate documented in research.

**Core Principle:** Out-of-sample performance must match in-sample performance. If not, strategy is overfitted.

---

## Validation Phases

### Phase 1: Unit Testing (Week 1)
### Phase 2: Backtest Validation (Week 2-3)
### Phase 3: Walk-Forward Analysis (Week 4-5)
### Phase 4: Paper Trading (Month 3-8)
### Phase 5: Live Trading (Month 9+)

---

## Phase 1: Unit Testing

### 1.1 Data Integrity Tests

**Test data fetching and quality:**

```python
# tests/test_data_quality.py
import pytest
from data.alpaca import fetch_stock_data
import pandas as pd


def test_alpaca_data_fetches():
    """Test that Alpaca data fetches without errors."""
    data = fetch_stock_data("SPY", "2024-01-01", "2024-06-30")

    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])


def test_no_missing_trading_days():
    """Test that we don't have phantom weekend/holiday bars."""
    data = fetch_stock_data("SPY", "2024-01-01", "2024-06-30")

    # Check no weekend bars
    weekends = data.index.dayofweek.isin([5, 6])
    assert weekends.sum() == 0, f"Found {weekends.sum()} weekend bars"


def test_no_data_gaps():
    """Test for unexpected data gaps (>5 consecutive missing days)."""
    data = fetch_stock_data("SPY", "2024-01-01", "2024-06-30")

    # Calculate gaps
    gaps = data.index.to_series().diff()
    large_gaps = gaps[gaps > pd.Timedelta(days=5)]

    # Allow for holidays, but flag large gaps
    if len(large_gaps) > 3:
        print(f"Warning: {len(large_gaps)} large gaps found")


def test_split_adjustment():
    """Test that stock splits are adjusted."""
    # Use a stock that had a recent split
    data = fetch_stock_data("NVDA", "2024-01-01", "2024-08-01")

    # Prices should be continuous (no sudden 10x jumps)
    price_changes = data['close'].pct_change()
    extreme_changes = price_changes[abs(price_changes) > 0.5]

    assert len(extreme_changes) == 0, \
        f"Found {len(extreme_changes)} extreme price changes (possible unadjusted split)"


def test_ohlc_consistency():
    """Test that OHLC relationships are valid."""
    data = fetch_stock_data("SPY", "2024-01-01", "2024-06-30")

    # High should be >= Low
    assert (data['high'] >= data['low']).all(), "Found bars where high < low"

    # High should be >= Open and Close
    assert (data['high'] >= data['open']).all(), "Found bars where high < open"
    assert (data['high'] >= data['close']).all(), "Found bars where high < close"

    # Low should be <= Open and Close
    assert (data['low'] <= data['open']).all(), "Found bars where low > open"
    assert (data['low'] <= data['close']).all(), "Found bars where low > close"
```

**Run tests:**
```bash
pytest tests/test_data_quality.py -v
```

**Acceptance criteria:** All tests pass.

### 1.2 Indicator Calculation Tests

```python
# tests/test_indicators.py
import pytest
import vectorbtpro as vbt
import pandas as pd
import numpy as np


def test_rsi_calculation():
    """Test RSI calculates correctly."""
    # Create simple data: trending up
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                      index=pd.date_range('2024-01-01', periods=10, freq='D'))

    rsi = vbt.talib("RSI").run(close, timeperiod=2).real

    # Uptrend should have high RSI
    assert rsi.iloc[-1] > 70, f"Expected RSI > 70 in uptrend, got {rsi.iloc[-1]}"


def test_ma_calculation():
    """Test moving average calculates correctly."""
    close = pd.Series([100] * 10, index=pd.date_range('2024-01-01', periods=10, freq='D'))

    ma = vbt.talib("SMA").run(close, timeperiod=5).real

    # MA of constant series should equal the constant
    assert np.allclose(ma.dropna(), 100), "MA of constant series incorrect"


def test_atr_calculation():
    """Test ATR is positive and reasonable."""
    data = fetch_stock_data("SPY", "2024-01-01", "2024-06-30")

    atr = vbt.talib("ATR").run(
        data['high'],
        data['low'],
        data['close'],
        timeperiod=14
    ).real

    # ATR should be positive
    assert (atr.dropna() > 0).all(), "Found negative ATR values"

    # ATR should be reasonable (< 10% of price for SPY)
    assert (atr.dropna() < data['close'] * 0.10).all(), "ATR unreasonably large"
```

### 1.3 Signal Generation Tests

```python
# tests/test_signal_generation.py
import pytest
from strategies.baseline_ma_rsi import BaselineStrategy


def test_signals_are_boolean():
    """Test that signals are boolean Series."""
    strategy = BaselineStrategy()

    close = pd.Series([100, 102, 98, 95, 97, 99, 101, 103, 100, 98],
                      index=pd.date_range('2024-01-01', periods=10, freq='D'))

    long_entries, long_exits, short_entries, short_exits, atr = \
        strategy.generate_signals(close)

    assert long_entries.dtype == bool
    assert long_exits.dtype == bool
    assert short_entries.dtype == bool
    assert short_exits.dtype == bool


def test_no_simultaneous_entry_exit():
    """Test that entry and exit don't occur on same bar."""
    strategy = BaselineStrategy()

    data = fetch_stock_data("SPY", "2024-01-01", "2024-06-30")
    long_entries, long_exits, _, _, _ = strategy.generate_signals(data['close'])

    # Entry and exit should not both be True on same bar
    simultaneous = long_entries & long_exits
    assert simultaneous.sum() == 0, \
        f"Found {simultaneous.sum()} bars with simultaneous entry and exit"


def test_signal_count_reasonable():
    """Test that we don't generate too many or too few signals."""
    strategy = BaselineStrategy()

    data = fetch_stock_data("SPY", "2024-01-01", "2024-12-31")  # ~250 trading days
    long_entries, _, _, _, _ = strategy.generate_signals(data['close'])

    entry_count = long_entries.sum()

    # Expect 10-50 signals per year (not 0, not 200)
    assert 10 <= entry_count <= 50, \
        f"Signal count {entry_count} outside expected range [10, 50]"
```

**Acceptance criteria:** All tests pass.

---

## Phase 2: Backtest Validation

### 2.1 Conservative Assumptions

**Always use these assumptions:**

```python
# Backtest configuration
FEES = 0.002  # 0.2% per trade (research-recommended)
SLIPPAGE = 0.001  # 0.1% slippage
INIT_CASH = 10000.0  # $10k starting capital

# Run backtest
pf = strategy.backtest(
    data=data,
    init_cash=INIT_CASH,
    fees=FEES,
    slippage=SLIPPAGE,
)
```

**Research warning applied:**
> "Correcting transaction cost assumptions from 0.1% to the more realistic **0.2% per trade transformed profitable backtests into unprofitable reality**"

**If strategy isn't profitable with 0.2% fees, it won't work live.**

### 2.2 Multi-Regime Testing

**Test across different market conditions:**

```python
# Test periods
BULL_MARKET = ("2019-01-01", "2020-02-01")  # Pre-COVID bull run
COVID_CRASH = ("2020-02-01", "2020-04-01")  # Crash
RECOVERY = ("2020-04-01", "2021-12-31")     # Recovery rally
BEAR_2022 = ("2022-01-01", "2022-12-31")    # 2022 bear market
SIDEWAYS = ("2023-01-01", "2024-12-31")     # Sideways/choppy

# Run strategy on each
results = {}
for period_name, (start, end) in [("Bull", BULL_MARKET), ("Crash", COVID_CRASH), ...]:
    data = fetch_stock_data("SPY", start, end)
    pf = strategy.backtest(data)

    results[period_name] = {
        'return': pf.total_return(),
        'sharpe': pf.sharpe_ratio(),
        'max_dd': pf.max_drawdown(),
    }

# Analyze
for period, metrics in results.items():
    print(f"{period}: Return={metrics['return']:.2%}, Sharpe={metrics['sharpe']:.2f}")
```

**Acceptance criteria:**
- ✅ Positive returns in 4/5 regimes
- ✅ Sharpe > 0.5 in 3/5 regimes
- ✅ Max drawdown < 40% in all regimes
- ❌ If strategy only works in bull markets → it's a momentum-only strategy (not robust)

### 2.3 Out-of-Sample Testing

**Reserve 30% of data for testing:**

```python
# Split data
full_data = fetch_stock_data("SPY", "2019-01-01", "2024-12-31")

split_date = "2023-01-01"

# In-sample: train and optimize
train_data = full_data[:split_date]

# Out-of-sample: test only (never optimize on this)
test_data = full_data[split_date:]

# Optimize on training data
# ... (optimize parameters)

# Test on out-of-sample data (use optimized params, don't change them)
pf_test = strategy.backtest(test_data)

print(f"Out-of-sample Sharpe: {pf_test.sharpe_ratio():.2f}")
print(f"Out-of-sample Return: {pf_test.total_return():.2%}")
```

**Acceptance criteria:**
- ✅ Out-of-sample Sharpe within 0.3 of in-sample Sharpe
- ✅ Out-of-sample return positive
- ❌ If out-of-sample Sharpe < in-sample Sharpe by >0.5 → overfitted

---

## Phase 3: Walk-Forward Analysis

### 3.1 Setup Walk-Forward Windows

**Research recommendation:**
> "Walk-forward divides historical data into segments, optimizes parameters on in-sample periods, tests on following out-of-sample periods... **1-2 years for in-sample training, 3-6 months for out-of-sample testing**"

```python
# Walk-forward configuration
TRAIN_PERIOD = 365  # days (1 year)
TEST_PERIOD = 90    # days (3 months)
STEP_FORWARD = 30   # days (1 month)

# This creates overlapping windows:
# Window 1: Train Jan2019-Dec2019, Test Jan2020-Mar2020
# Window 2: Train Feb2019-Jan2020, Test Feb2020-Apr2020
# ... (rolling forward monthly)
```

### 3.2 Parameter Stability Test

**After walk-forward, analyze parameter stability:**

```python
# Example walk-forward results
wf_results = pd.DataFrame({
    'window': [1, 2, 3, 4, 5],
    'optimal_rsi_threshold': [15, 12, 18, 14, 16],
    'test_sharpe': [1.2, 0.9, 1.1, 1.3, 0.8],
})

# Check parameter stability
param_std = wf_results['optimal_rsi_threshold'].std()
print(f"Parameter stability (std dev): {param_std:.2f}")

if param_std > 5:
    print("⚠️  WARNING: Parameters vary widely across windows")
    print("   → Strategy may be unstable or overfitting")
else:
    print("✅ Parameters stable across windows")
```

**Acceptance criteria:**
- ✅ Parameter std dev < 20% of mean value
- ✅ Test Sharpe relatively stable (std dev < 0.5)
- ❌ If parameters jump around wildly → not robust

### 3.3 Walk-Forward Efficiency Test

**Research metric:**
> "The robustness score measures the **percentage of walk-forward conditions passing performance thresholds**"

```python
# Calculate walk-forward efficiency
passing_windows = (wf_results['test_sharpe'] > 0.5).sum()
total_windows = len(wf_results)

wf_efficiency = passing_windows / total_windows

print(f"Walk-Forward Efficiency: {wf_efficiency:.1%}")
print(f"({passing_windows}/{total_windows} windows passed)")

if wf_efficiency >= 0.70:
    print("✅ Strategy robust (70%+ windows profitable)")
elif wf_efficiency >= 0.50:
    print("⚠️  Strategy marginal (50-70% windows profitable)")
else:
    print("❌ Strategy unreliable (<50% windows profitable)")
```

**Acceptance criteria:**
- ✅ WF Efficiency > 70%
- ⚠️ WF Efficiency 50-70% (marginal, monitor closely)
- ❌ WF Efficiency < 50% (strategy doesn't work consistently)

---

## Phase 4: Paper Trading Validation

### 4.1 Three-Account Testing Strategy

**Account configurations:**

| Account | Size | Risk/Trade | Confidence Threshold | Purpose |
|---------|------|------------|---------------------|---------|
| Paper 1 | $10k | 1% | 75+ (high conf only) | Conservative test |
| Paper 2 | $25k | 2% | 60+ (med+high conf) | Moderate test |
| Paper 3 | $50k | 2% | 50+ (all signals) | Aggressive test |

**Duration: Minimum 6 months (100+ trades)**

### 4.2 Weekly Monitoring

**Every Friday, record:**

```python
# Weekly tracking metrics
weekly_metrics = {
    'account': 'Paper_2',
    'week_ending': '2024-12-06',
    'equity': 26850.00,
    'total_return': 0.074,  # 7.4%
    'week_return': 0.012,   # 1.2% this week
    'sharpe_ytd': 1.15,
    'max_dd_ytd': -0.18,    # -18%
    'trades_ytd': 28,
    'win_rate_ytd': 0.61,   # 61%
    'avg_trade_return': 0.026,  # 2.6%
}
```

**Create `logs/paper_trading_weekly.csv` and append weekly.**

### 4.3 Monthly Comparison Report

**Compare all 3 accounts:**

```python
# Monthly comparison (Month 3 example)
import pandas as pd

results = pd.DataFrame({
    'Account': ['Paper 1', 'Paper 2', 'Paper 3'],
    'Strategy': ['Baseline', 'Baseline', 'TFC'],
    'Confidence Threshold': [75, 60, 50],
    'Total Return': [0.058, 0.074, 0.062],
    'Sharpe Ratio': [1.25, 1.15, 0.95],
    'Max Drawdown': [-0.12, -0.18, -0.22],
    'Win Rate': [0.68, 0.61, 0.58],
    'Total Trades': [15, 28, 42],
})

print(results)
```

**Analysis questions:**
1. Which account has best Sharpe?
2. Is higher confidence threshold better? (Paper 1 vs Paper 2)
3. Does TFC beat Baseline? (Paper 2 vs Paper 3)
4. Are drawdowns acceptable? (<25% threshold)
5. Is win rate consistent with backtest? (±10%)

### 4.4 Paper-to-Backtest Gap Analysis

**Compare paper trading to backtest:**

```python
# After 3 months of paper trading
backtest_sharpe = 1.30  # From Phase 2
paper_sharpe = 1.15     # From Paper Account 2

gap = backtest_sharpe - paper_sharpe

print(f"Backtest Sharpe: {backtest_sharpe:.2f}")
print(f"Paper Sharpe: {paper_sharpe:.2f}")
print(f"Gap: {gap:.2f}")

if gap < 0.2:
    print("✅ Paper trading matches backtest")
elif gap < 0.5:
    print("⚠️  Paper trading slightly worse than backtest")
    print("   → Likely due to slippage/execution")
else:
    print("❌ Large gap between backtest and paper")
    print("   → Strategy may not work in live markets")
    print("   → Check for overfitting or data issues")
```

**Acceptance criteria:**
- ✅ Gap < 0.3 Sharpe points
- ⚠️ Gap 0.3-0.5 (monitor, may improve)
- ❌ Gap > 0.5 (strategy likely overfitted)

### 4.5 Failure Criteria (Stop Paper Trading)

**Abort paper trading if ANY of these occur:**

| Metric | Threshold | Action |
|--------|-----------|--------|
| **Sharpe Ratio** | < 0.3 after 6 months | STOP - strategy doesn't work |
| **Win Rate** | < 45% after 100 trades | STOP - signal quality poor |
| **Max Drawdown** | > 40% at any point | STOP - position sizing broken |
| **Degrading Performance** | Sharpe declining 3 months in row | STOP - strategy decaying |
| **Paper-Backtest Gap** | > 0.8 Sharpe points | STOP - severely overfitted |

**If any failure criteria met:**
1. Stop paper trading immediately
2. Analyze what went wrong
3. Fix issues in backtest
4. Revalidate (restart Phase 2)
5. Don't move to live trading

---

## Phase 5: Live Trading Transition

### 5.1 Pre-Live Checklist

**Only proceed to live trading if ALL are true:**

- [ ] Paper trading ran for minimum 6 months
- [ ] 100+ paper trades executed
- [ ] Sharpe ratio > 0.8 in paper
- [ ] Win rate > 55% in paper
- [ ] Max drawdown < 25% in paper
- [ ] Paper performance matches backtest (gap < 0.3 Sharpe)
- [ ] No degradation over time (last 3 months stable or improving)
- [ ] You understand why strategy works (can explain edge)
- [ ] You're emotionally prepared for losses (8-10 in a row possible)
- [ ] You have 6+ months of living expenses (don't need trading income)

**If ANY checkbox is unchecked, DO NOT go live.**

### 5.2 Initial Live Capital

**Start small:**

```
Paper account showed: $25k paper → 7.4% return in 3 months

Initial live size: $1,000 - $5,000 MAX

NOT: $25,000 (your full paper size)
NOT: $100,000 (your actual capital)

Start with 5-20% of intended size.
```

**Why:**
- Emotions change with real money
- Execution may differ (real fills vs paper)
- Strategy may degrade (market conditions change)
- You may panic and override system (common beginner mistake)

### 5.3 Scaling Plan

**Only scale up if live performance matches paper:**

| Month | Live Capital | Condition |
|-------|--------------|-----------|
| 1-2 | $1,000 - $2,000 | Initial live test |
| 3-4 | $3,000 - $5,000 | If Sharpe > 0.8 in months 1-2 |
| 5-6 | $5,000 - $10,000 | If Sharpe > 0.8 in months 3-4 |
| 7+ | $10,000 - $25,000 | If Sharpe > 0.8 in months 5-6 |

**Never scale up after losing months. Only after wins.**

### 5.4 Live Monitoring (Daily)

**Track every trade in spreadsheet:**

```python
# logs/live_trades.csv
trade_log = {
    'date': '2024-12-06',
    'symbol': 'SPY',
    'action': 'BUY',
    'shares': 15,
    'entry_price': 580.50,
    'confidence': 72,
    'expected_stop': 575.00,
    'expected_target': 591.00,
    'actual_fill': 580.58,  # Slippage: $0.08
    'fees': 0.00,  # Alpaca commission-free
}

# On exit:
trade_log_exit = {
    'exit_date': '2024-12-10',
    'exit_price': 588.20,
    'exit_reason': 'Target',
    'actual_return': 0.013,  # 1.3%
    'expected_return': 0.018,  # 1.8% (difference = slippage)
}
```

**Review weekly:**
- Are fills matching paper trading?
- Is slippage higher than expected?
- Are you following the system or overriding?
- Is win rate matching paper trading?

---

## Red Flags & Warning Signs

### Overfitting Indicators

| Sign | What It Means | Action |
|------|---------------|--------|
| Backtest Sharpe > 3.0 | Almost certainly overfitted | Re-validate with simpler params |
| Perfect equity curve | No realistic drawdowns | Lookahead bias in code |
| 90%+ win rate | Too good to be true | Data leakage somewhere |
| Huge gap (in-sample vs out-sample) | Parameters don't generalize | Reduce parameters |
| Parameters change wildly in WF | Unstable strategy | Use fixed params |

**Research warning:**
> "**Backtested Sharpe ratios exceeding 3.0 typically indicate overfitting** rather than genuine edge"

### Strategy Decay Indicators

| Sign | What It Means | Action |
|------|---------------|--------|
| Sharpe declining monthly | Strategy aging poorly | Monitor 1-2 more months, then stop |
| Win rate dropping | Signal quality degrading | Retrain or halt |
| More whipsaws (stop-outs) | Market regime changed | Widen stops or pause |
| Slippage increasing | Liquidity drying up | Switch to more liquid symbols |

### Execution Issues

| Sign | What It Means | Action |
|------|---------------|--------|
| Paper works, live doesn't | Execution problems | Check fills, reduce size |
| Frequent partial fills | Liquidity issues | Trade only liquid stocks (>1M vol/day) |
| Stops not filling at expected price | Slippage worse than assumed | Widen stops, reduce size |
| Emotional override (skipping signals) | Psychology issues | Go back to paper trading |

---

## Success Criteria Summary

**After completing all 5 phases:**

### Minimum Viable Strategy (Deploy to Live)

✅ Sharpe > 0.8 in paper (6+ months)
✅ Win rate > 55%
✅ Max drawdown < 25%
✅ 100+ trades executed
✅ Paper matches backtest (gap < 0.3)
✅ No degradation over time
✅ All unit tests pass
✅ Walk-forward efficiency > 70%

### Exceptional Strategy (Scale Up Faster)

✅ Sharpe > 1.2 in paper
✅ Win rate > 65%
✅ Max drawdown < 20%
✅ Paper beats backtest (live edge)
✅ Improving over time

### Failed Strategy (Abandon)

❌ Sharpe < 0.5 in paper
❌ Win rate < 50%
❌ Max drawdown > 35%
❌ Degrading performance
❌ Large paper-backtest gap (>0.5)

---

## Final Checklist

Before deploying ANY capital:

**Technical Validation:**
- [ ] All unit tests pass
- [ ] Backtest Sharpe > 1.0 with 0.2% fees
- [ ] Out-of-sample performance validated
- [ ] Walk-forward efficiency > 70%
- [ ] Parameter stability confirmed
- [ ] No overfitting signs (Sharpe < 3.0)

**Live Validation:**
- [ ] Paper trading 6+ months
- [ ] 100+ trades in paper
- [ ] Paper Sharpe > 0.8
- [ ] Paper win rate > 55%
- [ ] Paper drawdown < 25%
- [ ] Paper matches backtest
- [ ] No performance degradation

**Psychological Readiness:**
- [ ] Prepared for 8-10 losing trades in a row
- [ ] Won't panic and override system
- [ ] Have 6+ months living expenses saved
- [ ] Starting with <20% of intended capital
- [ ] Can objectively stop if failure criteria met

**If ALL boxes checked → Proceed to live trading with small size.**

**If ANY box unchecked → Fix issue first.**

---

## Next Steps

1. Implement unit tests (Phase 1)
2. Run backtests with conservative assumptions (Phase 2)
3. Execute walk-forward analysis (Phase 3)
4. Deploy to paper trading (Phase 4)
5. Monitor for 6+ months
6. Re-evaluate using this checklist
7. Only then consider live trading (Phase 5)

**Remember:**
> "**44% of published trading strategies fail to replicate on new data**"

Your job: Be in the 56% that succeed. This protocol is how.
