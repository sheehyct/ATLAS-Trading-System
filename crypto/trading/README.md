# Crypto Derivatives Trading Module - Technical Reference

**Last Updated:** January 23, 2026
**Session:** CRYPTO Capital Efficiency Analysis

---

## Overview

This module provides trading infrastructure for Coinbase CFM (US-regulated crypto derivatives). Key components include position sizing, fee calculations, and beta-based capital efficiency analysis.

## Module Structure

```
crypto/trading/
├── __init__.py      # Exports all public functions
├── sizing.py        # Position sizing (risk-based and leverage-first)
├── fees.py          # Coinbase CFM fee calculations
├── beta.py          # Beta analysis and capital efficiency
└── derivatives.py   # Derivatives-specific utilities
```

---

## Key Discoveries (Jan 23, 2026)

### 1. Leverage Tier Correction

**CRITICAL:** Not all instruments have 10x intraday leverage.

| Asset | Intraday Leverage | Swing Leverage | Initial Margin |
|-------|-------------------|----------------|----------------|
| BTC   | 10x               | 4x             | 10% / 25%      |
| ETH   | 10x               | 4x             | 10% / 25%      |
| SOL   | **5x**            | 3x             | 20% / 33%      |
| XRP   | **5x**            | 3x             | 20% / 33%      |
| ADA   | **5x**            | 3x             | 20% / 33%      |

### 2. Beta to BTC (Volatility Multiplier)

Empirical beta values calculated from Day Down/Day Up price ranges:

| Asset | Beta | Interpretation |
|-------|------|----------------|
| BTC   | 1.00 | Baseline |
| ETH   | 1.98 | Moves ~2x BTC on same structure |
| SOL   | 1.55 | Moves ~1.5x BTC |
| XRP   | 1.77 | Moves ~1.8x BTC |
| ADA   | 2.20 | Moves ~2.2x BTC (highest beta) |

**Key Insight:** Lower leverage can be offset by higher beta. ADA at 5x leverage with 2.2x beta outperforms BTC at 10x leverage with 1.0x beta.

### 3. Effective Multiplier (Capital Efficiency)

```
Effective Multiplier = Leverage × Beta
```

**Intraday Rankings:**
1. **ETH: 19.8** (10x × 1.98) - BEST capital efficiency
2. **ADA: 11.0** (5x × 2.20)
3. **BTC: 10.0** (10x × 1.00)
4. **XRP: 8.85** (5x × 1.77)
5. **SOL: 7.75** (5x × 1.55) - WORST capital efficiency

### 4. Fee Structure

Coinbase CFM uses percentage fees with a minimum floor:

- **Taker Fee:** 0.02% (0.0002)
- **Maker Fee:** 0.00%
- **Minimum Fee:** $0.15 per contract

The actual fee is `max(notional × rate, $0.15 × num_contracts)`.

**Fee Impact by Instrument ($1k account, full leverage):**

| Asset | Contracts | Notional | Round-Trip Fee |
|-------|-----------|----------|----------------|
| BTC   | 11        | $10,000  | $3.30          |
| ETH   | 33        | $10,000  | $9.90          |
| SOL   | 15        | $5,000   | $4.50          |
| XRP   | 10        | $5,000   | $3.00          |
| ADA   | 25        | $5,000   | $7.50          |

---

## API Reference

### fees.py

```python
from crypto.trading.fees import (
    calculate_fee,
    calculate_round_trip_fee,
    calculate_breakeven_move,
    calculate_num_contracts,
    calculate_notional_from_contracts,
    create_coinbase_fee_func,
    analyze_fee_impact,
)

# Single trade fee
fee = calculate_fee(notional_value=5000, num_contracts=12, is_maker=False)
# Returns: max(5000 * 0.0002, 12 * 0.15) = max(1.0, 1.80) = $1.80

# Round-trip fee
rt_fee = calculate_round_trip_fee(5000, num_contracts=12)
# Returns: $3.60

# Breakeven move percentage
breakeven = calculate_breakeven_move(5000, num_contracts=12)
# Returns: 0.00072 (0.072% move needed to cover fees)

# Contract calculations
num = calculate_num_contracts(notional_value=5000, price=0.35, symbol="ADA")
# Returns: 14 contracts (5000 / (0.35 × 1000))

notional = calculate_notional_from_contracts(12, price=0.3415, symbol="ADA")
# Returns: $4,098 (12 × 1000 × 0.3415)

# VectorBT Pro integration
fee_func = create_coinbase_fee_func("ADA", is_maker=False)
pf = vbt.Portfolio.from_signals(close=data, entries=entries, exits=exits, fees=fee_func)

# Full analysis
analysis = analyze_fee_impact(
    account_value=1000,
    leverage=5,
    price=0.35,
    symbol="ADA",
    stop_percent=0.02,
    target_percent=0.04,
)
# Returns dict with notional, fees, breakeven, net R:R ratio, etc.
```

### beta.py

```python
from crypto.trading.beta import (
    calculate_effective_multiplier,
    get_effective_multipliers,
    rank_by_capital_efficiency,
    project_pnl_on_btc_move,
    compare_instruments_on_btc_move,
    calculate_rolling_beta,
    calculate_beta_from_ranges,
    select_best_instrument,
    CRYPTO_BETA_TO_BTC,
    INTRADAY_LEVERAGE,
)

# Get effective multiplier for a symbol
mult = calculate_effective_multiplier("ADA", "intraday")
# Returns: 11.0 (5x × 2.2)

# Get all multipliers
mults = get_effective_multipliers("intraday")
# Returns: {'BTC': 10.0, 'ETH': 19.8, 'SOL': 7.75, 'XRP': 8.85, 'ADA': 11.0}

# Rank by capital efficiency
ranking = rank_by_capital_efficiency("intraday")
# Returns: [('ETH', 19.8), ('ADA', 11.0), ('BTC', 10.0), ('XRP', 8.85), ('SOL', 7.75)]

# Project P/L on expected BTC move
projection = project_pnl_on_btc_move(
    btc_move_percent=0.03,  # 3% BTC move
    account_value=1000,
    symbol="ADA",
    leverage_tier="intraday",
)
# Returns: {'expected_pnl': 330.0, 'pnl_percent': 0.33, ...}

# Compare all instruments
df = compare_instruments_on_btc_move(0.0358, 1000, "intraday")
# Returns DataFrame sorted by expected P/L

# Calculate rolling beta from price data
beta_series = calculate_rolling_beta(ada_prices, btc_prices, window=20)

# Calculate beta from Day Up/Down levels
beta = calculate_beta_from_ranges(
    asset_high=0.3737, asset_low=0.3464,  # ADA
    btc_high=90273, btc_low=87156,        # BTC
)
# Returns: 2.20

# Select best instrument from multiple signals
signals = [
    {'symbol': 'BTC', 'entry': 90000, 'stop': 89000, 'target': 92000},
    {'symbol': 'ADA', 'entry': 0.36, 'stop': 0.35, 'target': 0.40},
]
best = select_best_instrument(signals, account_value=1000, leverage_tier="intraday")
# Returns ADA signal (higher effective multiplier)
```

### sizing.py

```python
from crypto.trading.sizing import (
    calculate_position_size,
    calculate_position_size_leverage_first,
    should_skip_trade,
)

# Risk-based sizing (2% risk per trade)
size, leverage, risk = calculate_position_size(
    account_value=1000,
    risk_percent=0.02,
    entry_price=90000,
    stop_price=89000,
    max_leverage=10.0,
)

# Leverage-first sizing (full leverage deployment)
size, leverage, risk = calculate_position_size_leverage_first(
    account_value=1000,
    entry_price=0.35,
    stop_price=0.34,
    leverage=5.0,  # ADA intraday max
)

# Check if trade should be skipped
skip, reason = should_skip_trade(
    account_value=1000,
    risk_percent=0.02,
    entry_price=100,
    stop_price=85,  # 15% stop
    max_leverage=5.0,
)
# Returns: (True, "Setup requires 13.3x leverage...")
```

---

## Contract Specifications

| Asset | Contract Size | Symbol Prefix | Example Notional @ Price |
|-------|---------------|---------------|--------------------------|
| BTC   | 0.01 BTC      | BIP           | $900 @ $90,000           |
| ETH   | 0.10 ETH      | ETP           | $300 @ $3,000            |
| SOL   | 5 SOL         | -             | $650 @ $130              |
| XRP   | 500 XRP       | -             | $1,000 @ $2.00           |
| ADA   | 1,000 ADA     | -             | $400 @ $0.40             |

---

## Trading Capabilities

**Available on Coinbase CFM:**
- Long positions (all assets)
- **Short positions (perpetual futures)** - CAN short perps
- Intraday leverage (10x BTC/ETH, 5x alts)
- Swing leverage (4x BTC/ETH, 3x alts)

**Account Constraints:**
- Cash account (no portfolio margin)
- Must close intraday positions by 4PM ET for 10x/5x leverage
- Weekend leverage reduced (Friday 4PM to Sunday 6PM ET)

---

## Statistical Arbitrage Potential

**Since shorting IS available on perps, stat arb strategies are viable:**

1. **Pairs Trading (BTC/Altcoin)**
   - Long lagging asset, short leading asset
   - Exploit temporary beta deviations
   - Requires cointegration testing

2. **Beta Mean Reversion**
   - When ADA moves only 1.5x (vs expected 2.2x), long ADA / short BTC
   - Wait for beta to revert to mean
   - Risk: Beta itself is non-stationary

3. **Lead-Lag Exploitation**
   - BTC often leads altcoin moves by minutes/hours
   - Enter altcoin position when BTC breaks structure
   - Exit when altcoin catches up

**Research References:**
- Leung & Nguyen (2019): Cointegrated cryptocurrency portfolios
- EUR thesis (2023): 12% monthly returns on crypto pairs trading
- DRL paper (2024): Sharpe 2.43 with stat arb + reinforcement learning

---

## Configuration (crypto/config.py)

Key constants updated in this session:

```python
# Leverage tiers (corrected)
LEVERAGE_TIERS = {
    "intraday": {"BTC": 10.0, "ETH": 10.0, "SOL": 5.0, "XRP": 5.0, "ADA": 5.0},
    "swing": {"BTC": 4.0, "ETH": 4.0, "SOL": 3.0, "XRP": 3.0, "ADA": 3.0},
}

# Beta to BTC
CRYPTO_BETA_TO_BTC = {"BTC": 1.00, "ETH": 1.98, "SOL": 1.55, "XRP": 1.77, "ADA": 2.20}

# Effective multipliers (computed)
EFFECTIVE_MULTIPLIER_INTRADAY = {"BTC": 10.0, "ETH": 19.8, "SOL": 7.75, "XRP": 8.85, "ADA": 11.0}
```

---

## Usage Examples

### Example 1: Instrument Selection for STRAT Signal

```python
from crypto.trading import rank_by_capital_efficiency, analyze_fee_impact

# Multiple STRAT signals available - which to take?
ranking = rank_by_capital_efficiency("intraday")
print(f"Best capital efficiency: {ranking[0]}")  # ('ETH', 19.8)

# Check fee impact on chosen instrument
analysis = analyze_fee_impact(
    account_value=1000, leverage=10, price=3000,
    symbol="ETH", stop_percent=0.015, target_percent=0.03
)
print(f"Net R:R after fees: {analysis['net_rr_ratio']:.2f}")
```

### Example 2: P/L Projection on BTC Move

```python
from crypto.trading import compare_instruments_on_btc_move

# If BTC moves 3%, what's my P/L on each instrument?
df = compare_instruments_on_btc_move(0.03, account_value=1000, leverage_tier="intraday")
print(df[['symbol', 'expected_pnl', 'pnl_percent']])

#   symbol  expected_pnl  pnl_percent
# 0    ETH         594.0        0.594
# 1    ADA         330.0        0.330
# 2    BTC         300.0        0.300
# 3    XRP         265.5        0.266
# 4    SOL         232.5        0.233
```

### Example 3: VectorBT Backtesting with Accurate Fees

```python
import vectorbtpro as vbt
from crypto.trading import create_coinbase_fee_func

# Create fee function for backtesting
fee_func = create_coinbase_fee_func("ADA", is_maker=False)

# Run backtest with accurate fees
pf = vbt.Portfolio.from_signals(
    close=ada_prices,
    entries=entries,
    exits=exits,
    fees=fee_func,
    freq="1h",
)
print(pf.stats())
```

---

## Future Development

1. **Rolling Beta Recalculation** - Automate weekly beta updates
2. **Stat Arb Module** - Pairs trading with cointegration testing
3. **Lead-Lag Detection** - Identify BTC-to-altcoin timing patterns
4. **Regime-Based Instrument Selection** - Adjust preferences by market state

---

## Session Reference

- **Session:** CRYPTO Capital Efficiency Analysis (Jan 23, 2026)
- **Transcript:** `/mnt/transcripts/2026-01-23-17-22-49-crypto-leverage-beta-analysis.txt`
- **Key Insight:** ETH offers best capital efficiency (19.8x effective multiplier)
- **Trade Post-Mortem:** ADA loss validated beta math ($263 loss on 6.41% adverse move)
