# STRAT Options - Options Integration for STRAT Patterns

**Purpose:** Implementation guide for integrating options trading with STRAT patterns  
**Parent:** [SKILL.md](SKILL.md)

---

## Table of Contents

1. [Strike Selection Methodology](#1-strike-selection-methodology)
2. [Expected ROI Calculation](#2-expected-roi-calculation)
3. [Greeks Analysis](#3-greeks-analysis)
4. [Expiration Selection](#4-expiration-selection)
5. [Position Sizing for Options](#5-position-sizing-for-options)
6. [Liquidity Requirements](#6-liquidity-requirements)
7. [Cheap Options Strategy](#7-cheap-options-strategy)

---

## 1. Strike Selection Methodology

### Core Principle

**Strike must be within entry-to-target range** - not entry-to-stop.

**Why:** Options need price to reach target, not just trigger. If strike is between entry and stop, option expires worthless even if pattern is correct.

---

### Strike Selection Formula

**For Calls (Bullish STRAT):**
```
Optimal Strike Range = [Entry, Target_1]
Best Strike = Entry + (0.3 × (Target_1 - Entry))

Example:
Entry: $100
Target 1: $104
Optimal Range: $100 - $104
Best Strike: $100 + (0.3 × $4) = $101.20 → Round to $101
```

**For Puts (Bearish STRAT):**
```
Optimal Strike Range = [Target_1, Entry]
Best Strike = Entry - (0.3 × (Entry - Target_1))

Example:
Entry: $100
Target 1: $96
Optimal Range: $96 - $100
Best Strike: $100 - (0.3 × $4) = $98.80 → Round to $99
```

---

### Implementation

```python
def select_option_strike(entry, target, stop, bias, available_strikes):
    """
    Select optimal strike based on STRAT pattern.
    
    Args:
        entry: Entry price from pattern
        target: Target price (T1)
        stop: Stop price
        bias: 'bull' or 'bear'
        available_strikes: List of available strike prices
    
    Returns:
        (optimal_strike, strike_type, rationale)
    """
    if bias == 'bull':
        # Call option
        strike_range_min = entry
        strike_range_max = target
        optimal_strike_calc = entry + (0.3 * (target - entry))
        strike_type = 'call'
    else:
        # Put option
        strike_range_min = target
        strike_range_max = entry
        optimal_strike_calc = entry - (0.3 * (entry - target))
        strike_type = 'put'
    
    # Find closest available strike
    valid_strikes = [s for s in available_strikes 
                     if strike_range_min <= s <= strike_range_max]
    
    if not valid_strikes:
        return None, None, "No strikes in valid range"
    
    # Find closest to optimal
    closest_strike = min(valid_strikes, key=lambda x: abs(x - optimal_strike_calc))
    
    return closest_strike, strike_type, f"Within range {strike_range_min}-{strike_range_max}"
```

---

### Strike Selection by Pattern

**2-1-2 Pattern:**
```
Entry: $100
Stop: $98
Target: $104

Call Strike Range: $100 - $104
Recommended: $101 or $102
Avoid: $105+ (above target)
Avoid: $98-$99 (too deep ITM, low leverage)
```

**3-1-2 Pattern (Wider Stop):**
```
Entry: $100
Stop: $95
Target: $107

Call Strike Range: $100 - $107
Recommended: $102 or $103
Avoid: $108+ (above target)
Avoid: $95-$98 (too deep ITM)
```

---

## 2. Expected ROI Calculation

### Breakeven Analysis

**Call Option Breakeven:**
```
Breakeven = Strike + Premium Paid

Example:
Strike: $101
Premium: $2.50
Breakeven: $103.50
```

**Put Option Breakeven:**
```
Breakeven = Strike - Premium Paid

Example:
Strike: $99
Premium: $2.20
Breakeven: $96.80
```

---

### ROI Calculation

```python
def calculate_option_roi(strike, premium, entry, target, bias):
    """
    Calculate expected ROI if target is hit.
    
    Args:
        strike: Option strike price
        premium: Premium paid
        entry: Stock entry price
        target: Stock target price
        bias: 'bull' or 'bear'
    
    Returns:
        (breakeven, target_value, roi_pct, meets_threshold)
    """
    if bias == 'bull':
        breakeven = strike + premium
        
        # Value at target = Max(Target - Strike, 0)
        target_value = max(target - strike, 0)
        
        # ROI = (Target Value - Premium) / Premium × 100
        roi = ((target_value - premium) / premium) * 100
        
    else:  # Put
        breakeven = strike - premium
        
        # Value at target = Max(Strike - Target, 0)
        target_value = max(strike - target, 0)
        
        # ROI = (Target Value - Premium) / Premium × 100
        roi = ((target_value - premium) / premium) * 100
    
    # Threshold: ROI should be at least 100%
    meets_threshold = roi >= 100
    
    return breakeven, target_value, roi, meets_threshold
```

---

### ROI Requirements

**Minimum ROI by Quality:**
| Setup Quality | Min ROI | Typical Premium | Max Premium |
|---------------|---------|-----------------|-------------|
| **A+ (12-15)** | 75% | $1-$3 | $5 |
| **A (9-11)** | 100% | $0.50-$2 | $3 |
| **B (6-8)** | 150% | $0.25-$1 | $2 |
| **C (3-5)** | 200% | $0.10-$0.50 | $1 |

**Why higher ROI for lower quality:**
- Lower win rate requires higher payoff
- Cheaper options = higher leverage
- Risk/reward balance

---

### Example ROI Analysis

**Scenario: 2-1-2 Bull Pattern**
```
Stock Entry: $100
Stock Target: $104
Stop: $98
Pattern Quality: A (score 9)

Option 1: $101 Call @ $2.50
Breakeven: $103.50
Target Value: $3.00 ($104 - $101)
ROI: ($3.00 - $2.50) / $2.50 = 20%
Result: ❌ Reject (ROI < 100%)

Option 2: $102 Call @ $1.50
Breakeven: $103.50
Target Value: $2.00 ($104 - $102)
ROI: ($2.00 - $1.50) / $1.50 = 33%
Result: ❌ Reject (ROI < 100%)

Option 3: $103 Call @ $0.80
Breakeven: $103.80
Target Value: $1.00 ($104 - $103)
ROI: ($1.00 - $0.80) / $0.80 = 25%
Result: ❌ Reject (ROI < 100%)

Conclusion: Options not favorable for this setup.
Trade stock instead.
```

**Scenario: Better Setup**
```
Stock Entry: $100
Stock Target: $108 (3R pattern)
Stop: $98
Pattern Quality: A+ (score 12)

Option: $102 Call @ $1.80
Breakeven: $103.80
Target Value: $6.00 ($108 - $102)
ROI: ($6.00 - $1.80) / $1.80 = 233%
Result: ✅ Accept (ROI > 75% for A+)

Position: Buy $102 calls
Risk: $1.80 per contract
Reward: $4.20 per contract (R:R = 2.3:1)
```

---

## 3. Greeks Analysis

### Delta Requirements

**Delta** = Rate of change in option price relative to stock price

**Optimal Delta Range:** 0.50 - 0.80

**Why:**
- <0.50: Too far OTM, low probability
- >0.80: Too expensive, low leverage
- 0.50-0.80: Balance of probability and leverage

---

### Delta by Strike Selection

```python
def evaluate_delta(strike, stock_price, option_delta, bias):
    """
    Evaluate if Delta is appropriate for STRAT options.
    
    Args:
        strike: Option strike
        stock_price: Current stock price
        option_delta: Option delta value
        bias: 'bull' or 'bear'
    
    Returns:
        (is_valid, rating, recommendation)
    """
    delta_abs = abs(option_delta)
    
    if delta_abs < 0.30:
        return False, 'TOO LOW', 'Strike too far OTM'
    elif delta_abs < 0.50:
        return False, 'LOW', 'Consider closer strike'
    elif delta_abs <= 0.80:
        return True, 'OPTIMAL', 'Good balance of probability and leverage'
    elif delta_abs <= 0.90:
        return False, 'HIGH', 'Too expensive, low leverage'
    else:
        return False, 'TOO HIGH', 'Essentially stock, no leverage benefit'
```

---

### Theta (Time Decay)

**Theta** = Rate of time decay per day

**Rules:**
1. Lower theta = better for swing trades (5+ days)
2. Higher theta acceptable for day trades (0-1 DTE)
3. Avoid high theta for uncertain timing

**Theta by Expiration:**
```
30+ DTE: Theta = -$0.02 to -$0.05/day (acceptable)
7-30 DTE: Theta = -$0.05 to -$0.15/day (moderate)
0-7 DTE: Theta = -$0.15 to -$0.50/day (fast decay)
```

**Example:**
```
Setup: 2-1-2 bull, expected 3-5 days to target

Option 1: 30 DTE, Theta = -$0.03
Time decay over 5 days: $0.15

Option 2: 7 DTE, Theta = -$0.20
Time decay over 5 days: $1.00

Choice: Option 1 (lower decay)
```

---

### Vega (Volatility Sensitivity)

**Vega** = Change in option price per 1% change in IV

**For STRAT patterns:**
- Positive vega is good (benefit from volatility expansion)
- 3-1-2 patterns (outside bars) = higher vega benefit
- 2-1-2 patterns (inside bars) = lower vega impact

**Vega Considerations:**
```python
def evaluate_vega_risk(pattern_type, current_iv, historical_iv, vega):
    """
    Assess vega risk for STRAT pattern.
    
    Args:
        pattern_type: '212', '312', '22', etc.
        current_iv: Current implied volatility (%)
        historical_iv: Historical avg IV (%)
        vega: Option vega value
    
    Returns:
        (risk_level, recommendation)
    """
    iv_percentile = (current_iv / historical_iv) * 100
    
    # High IV = risky for buying options
    if iv_percentile > 120:
        return 'HIGH RISK', 'IV elevated, consider stock or spreads'
    
    # 3-1-2 patterns can benefit from volatility
    if pattern_type == '312' and iv_percentile < 80:
        return 'FAVORABLE', 'Low IV, outside bar pattern benefits from vol expansion'
    
    # Standard case
    if iv_percentile < 100:
        return 'ACCEPTABLE', 'IV normal range'
    else:
        return 'ELEVATED', 'Consider reducing premium or using spreads'
```

---

## 4. Expiration Selection

### Expiration Matching Pattern Timeframe

**Rule:** Expiration should be 2-3x expected time to target

| Pattern TF | Expected Duration | Minimum DTE | Recommended DTE |
|------------|-------------------|-------------|-----------------|
| **Daily** | 3-7 days | 7 DTE | 14-30 DTE |
| **60min** | 1-3 days | 3 DTE | 7-14 DTE |
| **15min** | 0-1 day | 1 DTE | 2-7 DTE |

---

### Expiration Selection Logic

```python
def select_expiration(pattern_tf, pattern_quality, available_expirations):
    """
    Select optimal expiration for STRAT pattern.
    
    Args:
        pattern_tf: 'daily', '60min', '15min'
        pattern_quality: Quality score (0-15)
        available_expirations: List of available DTE values
    
    Returns:
        (selected_dte, rationale)
    """
    # Base DTE by timeframe
    if pattern_tf == 'daily':
        min_dte = 7
        optimal_dte = 21
    elif pattern_tf == '60min':
        min_dte = 3
        optimal_dte = 10
    else:  # 15min
        min_dte = 1
        optimal_dte = 5
    
    # Adjust for quality
    if pattern_quality >= 10:  # A+
        # Can use shorter expiration (high confidence)
        optimal_dte = int(optimal_dte * 0.7)
    elif pattern_quality <= 6:  # B or lower
        # Use longer expiration (need more time)
        optimal_dte = int(optimal_dte * 1.5)
    
    # Find closest available
    valid_expirations = [dte for dte in available_expirations if dte >= min_dte]
    
    if not valid_expirations:
        return None, "No valid expirations available"
    
    selected = min(valid_expirations, key=lambda x: abs(x - optimal_dte))
    
    return selected, f"Optimal: {optimal_dte} days, Selected: {selected} days"
```

---

### Weekly vs Monthly Options

**Weekly Options (Best For):**
- High quality setups (A/A+)
- Short-term patterns (15min, 60min)
- Near-term targets (1-2 days)
- High conviction trades

**Monthly Options (Best For):**
- Medium quality setups (B/C)
- Daily patterns
- Longer-term targets (5-10 days)
- Lower conviction trades

---

## 5. Position Sizing for Options

### Options Risk Management

**Key Principle:** Options are higher risk → reduce position size

**Options Position Sizing:**
```
Options Risk = Stock Risk × 0.5

Example:
Stock position: 1% risk
Options position: 0.5% risk
```

---

### Position Size Calculation

```python
def calculate_options_position_size(account_size, risk_pct, option_premium, 
                                   quality_score, bias_confidence):
    """
    Calculate number of option contracts.
    
    Args:
        account_size: Total account value
        risk_pct: Base risk percentage (e.g., 0.01 for 1%)
        option_premium: Price per contract
        quality_score: Pattern quality (0-15)
        bias_confidence: Directional confidence (0-1)
    
    Returns:
        (num_contracts, risk_amount, rationale)
    """
    # Base risk (50% of stock risk)
    options_risk_pct = risk_pct * 0.5
    
    # Adjust for quality
    if quality_score >= 10:
        multiplier = 1.0  # Full size for A+
    elif quality_score >= 7:
        multiplier = 0.75  # 75% for A
    elif quality_score >= 5:
        multiplier = 0.50  # 50% for B
    else:
        multiplier = 0.25  # 25% for C, or skip
    
    # Calculate risk amount
    risk_amount = account_size * options_risk_pct * multiplier
    
    # Calculate contracts
    # Risk per contract = full premium (can go to $0)
    num_contracts = int(risk_amount / (option_premium * 100))
    
    return num_contracts, risk_amount, f"Quality: {quality_score}, Multiplier: {multiplier}"
```

---

### Example Position Sizing

**Scenario:**
```
Account: $100,000
Stock risk per trade: 1% = $1,000
Pattern quality: A (score 9)
Option: $102 Call @ $1.50

Options risk: 1% × 0.5 = 0.5% = $500
Quality multiplier: 0.75 (A setup)
Adjusted risk: $500 × 0.75 = $375

Risk per contract: $1.50 × 100 = $150
Number of contracts: $375 / $150 = 2.5 → 2 contracts

Position:
- 2 contracts
- Total cost: $300
- Total risk: $300 (can go to $0)
- Max loss: 0.3% of account
```

---

### Options vs Stock Position Comparison

**Example: 2-1-2 Bull at $100, Target $104**

**Stock Position:**
```
Account: $100,000
Risk: 1% = $1,000
Entry: $100
Stop: $98
Risk per share: $2
Shares: 500
Total cost: $50,000
Max loss: $1,000 (2% of entry)
Max gain: $2,000 at T1 (4% return)
```

**Options Position:**
```
Account: $100,000
Risk: 0.5% = $500
Strike: $102
Premium: $1.50
Breakeven: $103.50
Contracts: 3 ($450 total)
Max loss: $450
Max gain: $600 at T1 (133% ROI)
Higher leverage, limited risk
```

---

## 6. Liquidity Requirements

### Minimum Liquidity Standards

**For STRAT Options Trading:**

| Metric | Minimum | Preferred | Ideal |
|--------|---------|-----------|-------|
| **Daily Volume** | 100 | 500 | 1,000+ |
| **Open Interest** | 1,000 | 5,000 | 10,000+ |
| **Bid-Ask Spread** | <$0.10 | <$0.05 | <$0.02 |
| **Bid-Ask %** | <5% | <3% | <1% |

---

### Liquidity Validation

```python
def validate_option_liquidity(volume, open_interest, bid, ask, min_requirements=True):
    """
    Validate if option meets liquidity requirements.
    
    Args:
        volume: Daily volume
        open_interest: Open interest
        bid: Bid price
        ask: Ask price
        min_requirements: If True, use minimum standards; if False, use preferred
    
    Returns:
        (is_valid, warnings, score)
    """
    warnings = []
    score = 0
    
    # Volume check
    min_vol = 100 if min_requirements else 500
    if volume < min_vol:
        warnings.append(f"Low volume: {volume} < {min_vol}")
    else:
        score += 1
    
    # Open interest check
    min_oi = 1000 if min_requirements else 5000
    if open_interest < min_oi:
        warnings.append(f"Low OI: {open_interest} < {min_oi}")
    else:
        score += 1
    
    # Spread check
    spread = ask - bid
    spread_pct = (spread / bid) * 100 if bid > 0 else 100
    
    max_spread = 0.10 if min_requirements else 0.05
    max_spread_pct = 5 if min_requirements else 3
    
    if spread > max_spread:
        warnings.append(f"Wide spread: ${spread:.2f} > ${max_spread}")
    elif spread_pct > max_spread_pct:
        warnings.append(f"Wide spread %: {spread_pct:.1f}% > {max_spread_pct}%")
    else:
        score += 1
    
    is_valid = score >= 2  # Must pass at least 2 of 3 checks
    
    return is_valid, warnings, score
```

---

### Impact of Low Liquidity

**Wide Spreads:**
- Slippage on entry: -2-5%
- Slippage on exit: -2-5%
- Total cost: -4-10% of trade value

**Example:**
```
Option: $102 Call
Fair Value: $1.50
Bid: $1.40
Ask: $1.60
Spread: $0.20 (13% of bid)

Buy at ask: $1.60
Sell at bid: $1.40
Immediate loss: $0.20 (12.5%)

Even if stock moves correctly, must overcome spread cost.
If target value = $2.00:
Gross profit: $0.40
Net profit after spread: $0.20
ROI: 12.5% (vs 33% with tight spread)
```

---

## 7. Cheap Options Strategy

### Definition

**Cheap Option** = Option costing <$1.00 with high ROI potential (200%+)

**Use Cases:**
- Low quality setups (C or lower)
- High risk/high reward
- Small position size
- 0DTE or weekly options

---

### Cheap Options Selection

```python
def find_cheap_options(strikes, premiums, entry, target, bias, max_premium=1.00):
    """
    Find cheap options with high ROI potential.
    
    Args:
        strikes: List of strike prices
        premiums: List of corresponding premiums
        entry: Stock entry price
        target: Stock target price
        bias: 'bull' or 'bear'
        max_premium: Maximum premium (default $1.00)
    
    Returns:
        List of (strike, premium, roi, meets_threshold) tuples
    """
    cheap_options = []
    
    for strike, premium in zip(strikes, premiums):
        if premium > max_premium:
            continue
        
        # Calculate ROI
        if bias == 'bull':
            target_value = max(target - strike, 0)
        else:
            target_value = max(strike - target, 0)
        
        if target_value <= 0 or premium <= 0:
            continue
        
        roi = ((target_value - premium) / premium) * 100
        
        # Require minimum 200% ROI for cheap options
        if roi >= 200:
            cheap_options.append((strike, premium, roi, True))
    
    # Sort by ROI descending
    cheap_options.sort(key=lambda x: x[2], reverse=True)
    
    return cheap_options
```

---

### Cheap Options Management

**Entry Rules:**
1. Only use for C-quality setups or lower
2. Maximum 1-2% of account per trade
3. Position size: 2-5 contracts max
4. Expected ROI: 200% minimum

**Exit Rules:**
1. **Target hit:** Exit 50% immediately, hold rest
2. **50% profit:** Exit 50% (lock profit)
3. **End of day:** Exit if 0DTE, otherwise hold
4. **Stop:** No stop - risk full premium

**Example:**
```
Setup: 2-1-2 bull (C quality)
Entry: $100
Target: $106 (weak setup, extended target)

Cheap Option: $104 Call @ $0.60
Breakeven: $104.60
Target value: $2.00 ($106 - $104)
ROI: ($2.00 - $0.60) / $0.60 = 233%

Position: 5 contracts × $60 = $300 total risk
Max loss: $300 (0.3% of $100k account)
Max gain: $700 (233% ROI)

Management:
- Price hits $1.20 (100% gain): Exit 2 contracts, hold 3
- Price hits $2.00 (target): Exit remaining 3 contracts
- Profit: $300 + $420 = $720 (240% total ROI)
```

---

### 0DTE (Zero Days to Expiration)

**0DTE Strategy for STRAT:**
1. Only for A-quality setups or better
2. Use 15min or 60min patterns only
3. Enter by 10:00 AM EST
4. Exit by 3:45 PM EST
5. Premium: $0.25-$0.75 max

**Risk Parameters:**
- **Position size:** 50% of normal (higher risk)
- **Stop:** Tight (2% max)
- **Profit target:** 50-100% ROI (quick exit)
- **Time decay:** Extreme - must hit target fast
- **Exit:** 3:45 PM latest (no overnight hold)

---

## Summary

**Options Integration Checklist:**
- ✅ Strike within entry-to-target range
- ✅ Expected ROI ≥ 100% (cheap option strategy ≥ 200%)
- ✅ Breakeven beaten by target
- ✅ Delta 0.50-0.80 (responsive but not expensive)
- ✅ Liquidity: 100 vol, 1000 OI, <5% spread
- ✅ Position size: 1-2% risk, <20% total
- ✅ Expiration matches pattern timeframe
- ✅ Account for transaction costs (4-5%)

**Next Steps:**
- For entry/exit execution → Read [EXECUTION.md](EXECUTION.md)
- For pattern detection → Read [PATTERNS.md](PATTERNS.md)
- For timeframe analysis → Read [TIMEFRAMES.md](TIMEFRAMES.md)
