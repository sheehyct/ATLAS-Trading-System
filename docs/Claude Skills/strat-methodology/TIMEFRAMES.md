# STRAT Timeframes - Multi-Timeframe Continuity Analysis

**Purpose:** Implementation guide for timeframe continuity, MOAF detection, and trade quality scoring  
**Parent:** [SKILL.md](SKILL.md)

---

## Table of Contents

1. [The 4 C's Framework](#1-the-4-cs-framework)
2. [MOAF - Mother of All Flips](#2-moaf-mother-of-all-flips)
3. [Timeframe Relationships](#3-timeframe-relationships)
4. [Continuity Scoring](#4-continuity-scoring)
5. [Trade Quality Matrix](#5-trade-quality-matrix)

---

## 1. The 4 C's Framework

### Overview

The 4 C's determine trade quality and position sizing based on multi-timeframe alignment.

| C | Name | Description | Position Size |
|---|------|-------------|---------------|
| **Combo** | Multiple TF same setup | 2+ timeframes show identical pattern | 100% |
| **Confirm** | Lower confirms higher | LTF triggers before HTF | 75% |
| **Continue** | Multiple setups same direction | Sequential setups in same trend | 50% |
| **Consolidate** | Controlled pullback | Pullback respects structure | 25% |

---

### 1.1 Combo (Multiple Timeframes Same Setup)

**Definition:** Same STRAT pattern appears on 2+ timeframes simultaneously.

**Example:**
```
Daily: 2-1-2 bull at $100 (trigger)
60min: 2-1-2 bull at $99.80 (trigger)
15min: 2-1-2 bull at $99.90 (trigger)

All three show bullish 2-1-2 = COMBO
```

**Detection Logic:**
```python
def detect_combo(patterns_by_tf, current_idx):
    """
    Detect Combo: Same pattern on multiple timeframes.
    
    Args:
        patterns_by_tf: Dict of {timeframe: pattern_array}
        current_idx: Current bar index
    
    Returns:
        (is_combo, matching_timeframes, bias)
    """
    pattern_counts = {'212_bull': 0, '212_bear': 0, '312_bull': 0, '312_bear': 0}
    matching_tfs = []
    
    for tf, patterns in patterns_by_tf.items():
        if current_idx >= len(patterns):
            continue
        
        pattern = patterns[current_idx]
        if pattern in pattern_counts:
            pattern_counts[pattern] += 1
            matching_tfs.append(tf)
    
    # Combo requires 2+ timeframes with same pattern
    max_count = max(pattern_counts.values())
    if max_count >= 2:
        dominant_pattern = max(pattern_counts, key=pattern_counts.get)
        bias = 'bull' if 'bull' in dominant_pattern else 'bear'
        return True, matching_tfs, bias
    
    return False, [], None
```

**Trading Combo:**
- **Position size:** 100% (full size)
- **Entry:** First timeframe to trigger
- **Stop:** Widest stop across all timeframes
- **Target:** Highest target across all timeframes
- **Win rate:** 70-80% (highest probability)

**Example Trade:**
```
Daily 2-1-2 bull: Entry $100, Stop $98, Target $104
60min 2-1-2 bull: Entry $99.80, Stop $98.50, Target $103
Combo entry: $99.80 (first trigger)
Combo stop: $98 (wider daily stop)
Combo target: $104 (higher daily target)
Position size: 100%
```

---

### 1.2 Confirm (Lower Confirms Higher)

**Definition:** Lower timeframe triggers before higher timeframe, confirming HTF direction.

**Example:**
```
Daily: 2-1-2 bull forming (trigger = $100)
60min: 2-1-2 bull TRIGGERED at $99.50
15min: Already bullish

Lower TF confirms higher TF bias before HTF trigger
```

**Detection Logic:**
```python
def detect_confirm(htf_pattern, htf_trigger, ltf_pattern, ltf_triggered, ltf_bias):
    """
    Detect Confirm: LTF triggers before HTF in same direction.
    
    Args:
        htf_pattern: Higher TF pattern type
        htf_trigger: Higher TF trigger price
        ltf_pattern: Lower TF pattern type
        ltf_triggered: Whether LTF already triggered
        ltf_bias: Lower TF bias ('bull' or 'bear')
    
    Returns:
        (is_confirm, expected_bias)
    """
    # Check if HTF has valid pattern forming
    if htf_pattern is None:
        return False, None
    
    # Determine HTF bias
    htf_bias = 'bull' if 'bull' in htf_pattern else 'bear'
    
    # LTF must be triggered and match HTF bias
    if ltf_triggered and ltf_bias == htf_bias:
        return True, htf_bias
    
    return False, None
```

**Trading Confirm:**
- **Position size:** 75% (high confidence)
- **Entry:** LTF trigger (early entry)
- **Stop:** HTF stop (wider but safer)
- **Target:** HTF target
- **Win rate:** 65-75%

**Example Trade:**
```
Daily 2-1-2 bull: Trigger $100 (not hit yet)
60min 2-1-2 bull: Triggered at $99.50
Entry: $99.50 (60min trigger)
Stop: $98 (daily stop)
Target: $104 (daily target)
Position size: 75%
```

**Advantages:**
- Earlier entry than HTF
- Better risk/reward
- HTF provides larger target

**Risks:**
- HTF may not trigger
- Early entry = more heat
- Requires discipline to use HTF stop

---

### 1.3 Continue (Multiple Setups Same Direction)

**Definition:** Sequential STRAT patterns in same direction showing trend continuation.

**Example:**
```
Bar 1-3: 2-1-2 bull (completed)
Bar 4-6: 2-2 bull (forming)
Bar 7-9: 2-1-2 bull (new setup)

Multiple sequential bullish patterns = CONTINUE
```

**Detection Logic:**
```python
def detect_continue(pattern_history, lookback=20):
    """
    Detect Continue: Multiple setups in same direction.
    
    Args:
        pattern_history: List of (bar_idx, pattern, bias) tuples
        lookback: Bars to look back
    
    Returns:
        (is_continue, dominant_bias, pattern_count)
    """
    recent_patterns = [p for p in pattern_history if p[0] >= len(pattern_history) - lookback]
    
    bull_count = sum(1 for p in recent_patterns if p[2] == 'bull')
    bear_count = sum(1 for p in recent_patterns if p[2] == 'bear')
    
    # Require 2+ patterns in same direction
    if bull_count >= 2 and bull_count > bear_count:
        return True, 'bull', bull_count
    elif bear_count >= 2 and bear_count > bull_count:
        return True, 'bear', bear_count
    
    return False, None, 0
```

**Trading Continue:**
- **Position size:** 50% (trend following)
- **Entry:** Latest setup trigger
- **Stop:** Latest setup stop
- **Target:** Extended (3R+)
- **Win rate:** 55-65%

**Management:**
- Trail stops on each new setup
- Scale out at each setup target
- Roll profits into new setups

**Example Trade:**
```
Setup 1: 2-1-2 bull, Entry $100, Exit $104 (1R profit)
Setup 2: 2-2 bull, Entry $104, Exit $108 (1R profit)  
Setup 3: 2-1-2 bull, Entry $108, target $112+
Position size: 50% per setup
```

---

### 1.4 Consolidate (Controlled Pullback)

**Definition:** Pullback that respects prior support/resistance and key STRAT levels.

**Example:**
```
Uptrend: $90 → $95 → $100
Pullback: $100 → $97 (holds $95 support)
Structure: Consolidation with inside bars
Pattern: Forms 2-1-2 bull at support

Controlled pullback = CONSOLIDATE
```

**Detection Logic:**
```python
def detect_consolidate(prices, supports, current_idx, lookback=10):
    """
    Detect Consolidate: Pullback respecting structure.
    
    Args:
        prices: Price array
        supports: Array of support levels
        current_idx: Current bar
        lookback: Bars for pullback analysis
    
    Returns:
        (is_consolidate, pullback_depth, structure_held)
    """
    # Identify recent high
    recent_high = np.max(prices[current_idx-lookback:current_idx])
    current_price = prices[current_idx]
    
    # Calculate pullback depth
    pullback_pct = (recent_high - current_price) / recent_high * 100
    
    # Check if any support level held
    structure_held = False
    for support in supports:
        if current_price >= support and current_price <= recent_high:
            structure_held = True
            break
    
    # Consolidate = pullback 30-50% that holds structure
    if 30 <= pullback_pct <= 50 and structure_held:
        return True, pullback_pct, structure_held
    
    return False, pullback_pct, structure_held
```

**Trading Consolidate:**
- **Position size:** 25% (cautious)
- **Entry:** Pattern trigger at support
- **Stop:** Below support level
- **Target:** Conservative (1-2R)
- **Win rate:** 45-55%

**Risk factors:**
- Lowest probability setup
- Could be trend reversal
- Requires strong R/R to justify

---

## 2. MOAF - Mother of All Flips

### Definition

**MOAF** = Institutional timeframe (daily+) flips direction while retail timeframes (15min-60min) lag.

**Characteristics:**
- Occurs on daily or weekly timeframe
- Creates multi-day/week directional bias
- Lower timeframes eventually align
- High conviction trade opportunity

---

### 2.1 MOAF Detection

**Bullish MOAF:**
```
Daily: Flips bullish (2-1-2 bull or 2-2 bull)
60min: Still bearish or neutral
15min: Still bearish

Daily leads, lower TFs will follow
```

**Bearish MOAF:**
```
Daily: Flips bearish (2-1-2 bear or 2-2 bear)
60min: Still bullish or neutral
15min: Still bullish

Daily leads, lower TFs will follow
```

**Detection Logic:**
```python
def detect_moaf(daily_bias, h60_bias, h15_bias, daily_flip):
    """
    Detect MOAF: HTF flips while LTF lags.
    
    Args:
        daily_bias: Current daily bias ('bull', 'bear', 'neutral')
        h60_bias: Current 60min bias
        h15_bias: Current 15min bias
        daily_flip: Whether daily just flipped (Boolean)
    
    Returns:
        (is_moaf, moaf_direction)
    """
    if not daily_flip:
        return False, None
    
    # MOAF = Daily flips but lower TFs haven't yet
    if daily_bias == 'bull' and (h60_bias != 'bull' or h15_bias != 'bull'):
        return True, 'bull'
    elif daily_bias == 'bear' and (h60_bias != 'bear' or h15_bias != 'bear'):
        return True, 'bear'
    
    return False, None
```

---

### 2.2 Trading MOAF

**Strategy:**
1. Identify daily flip
2. Wait for 60min to align
3. Enter on 15min pattern in daily direction
4. Use daily stop
5. Target daily target

**Example:**
```
Day 1: Daily flips bullish (2-1-2 bull triggered)
Day 1-2: 60min still showing bearish patterns
Day 3: 60min flips bullish
Day 3: Enter 15min bullish pattern
Stop: Daily low
Target: Daily target
Position size: 100% (highest conviction)
```

**MOAF Phases:**
```
Phase 1: Daily flips (note the flip)
Phase 2: 60min conflicted (wait)
Phase 3: 60min aligns (prepare)
Phase 4: 15min pattern (enter)
Phase 5: All TF aligned (scale in)
```

---

### 2.3 Institutional vs Retail Flips

**Institutional Flips (MOAF):**
- Daily/Weekly timeframes
- Large position accumulation
- Multi-day directional bias
- High follow-through

**Retail Flips:**
- 5min/15min timeframes
- Small reactive moves
- Intraday noise
- Low follow-through

**How to distinguish:**
```python
def classify_flip(timeframe, volume_ratio, follow_through_bars):
    """
    Classify flip as institutional or retail.
    
    Args:
        timeframe: Timeframe of flip ('5min', '15min', '60min', 'daily', 'weekly')
        volume_ratio: Volume vs 20-bar average
        follow_through_bars: Bars maintaining direction after flip
    
    Returns:
        'institutional', 'retail', or 'uncertain'
    """
    institutional_tf = timeframe in ['daily', 'weekly']
    high_volume = volume_ratio > 1.5
    strong_follow = follow_through_bars >= 3
    
    if institutional_tf and high_volume and strong_follow:
        return 'institutional'
    elif timeframe in ['5min', '15min'] and volume_ratio < 1.2:
        return 'retail'
    else:
        return 'uncertain'
```

**Trading implications:**
- Trade WITH institutional flips
- Fade retail flips when against institutional bias
- Never trade retail flips against MOAF

---

## 3. Timeframe Relationships

### 3.1 Optimal Timeframe Ratios

**Best practice:** Use 4:1 ratio between timeframes

| Higher TF | Lower TF | Ratio | Use Case |
|-----------|----------|-------|----------|
| Daily | 60min | 6.5:1 | Swing trading |
| 60min | 15min | 4:1 | Day trading |
| Daily | 15min | 26:1 | Position sizing |
| Weekly | Daily | 5:1 | Long-term trends |

**Why 4:1 ratio:**
- Sufficient bars for pattern formation
- Clear higher TF structure on lower TF
- Proper mother bar relationships

---

### 3.2 Timeframe Analysis Matrix

```python
def analyze_timeframes(daily_data, h60_data, h15_data):
    """
    Comprehensive timeframe analysis.
    
    Returns:
        dict with pattern, bias, and quality for each TF
    """
    analysis = {
        'daily': {
            'bars': classify_bars_nb(daily_data['High'].values, daily_data['Low'].values),
            'patterns': None,
            'bias': None,
            'quality': 0
        },
        'h60': {
            'bars': classify_bars_nb(h60_data['High'].values, h60_data['Low'].values),
            'patterns': None,
            'bias': None,
            'quality': 0
        },
        'h15': {
            'bars': classify_bars_nb(h15_data['High'].values, h15_data['Low'].values),
            'patterns': None,
            'bias': None,
            'quality': 0
        }
    }
    
    # Detect patterns for each TF
    for tf in analysis:
        # ... pattern detection logic ...
        pass
    
    # Calculate trade quality
    combo = detect_combo(...)
    confirm = detect_confirm(...)
    continue_sig = detect_continue(...)
    consolidate = detect_consolidate(...)
    moaf = detect_moaf(...)
    
    return {
        'analysis': analysis,
        'combo': combo,
        'confirm': confirm,
        'continue': continue_sig,
        'consolidate': consolidate,
        'moaf': moaf,
        'overall_quality': calculate_quality(combo, confirm, continue_sig, consolidate, moaf)
    }
```

---

## 4. Continuity Scoring

### Scoring System

**Quality Score Formula:**
```
Quality Score = (
    Combo * 4 +
    Confirm * 3 +
    Continue * 2 +
    Consolidate * 1 +
    MOAF * 5
)

Max Score: 15 (all factors present)
Min Score: 0 (no alignment)
```

**Implementation:**
```python
def calculate_quality_score(combo, confirm, continue_sig, consolidate, moaf):
    """
    Calculate trade quality score (0-15).
    
    Returns:
        (score, quality_level, recommended_size)
    """
    score = 0
    
    if combo:
        score += 4
    if confirm:
        score += 3
    if continue_sig:
        score += 2
    if consolidate:
        score += 1
    if moaf:
        score += 5
    
    # Determine quality level
    if score >= 10:
        quality = 'A+'
        size = 1.0
    elif score >= 7:
        quality = 'A'
        size = 0.75
    elif score >= 5:
        quality = 'B'
        size = 0.50
    elif score >= 3:
        quality = 'C'
        size = 0.25
    else:
        quality = 'D'
        size = 0.0  # Skip trade
    
    return score, quality, size
```

---

## 5. Trade Quality Matrix

### Position Sizing by Quality

| Score | Quality | Position Size | Expected Win Rate | Typical Setup |
|-------|---------|---------------|-------------------|---------------|
| **12-15** | A+ | 100-150% | 75-85% | MOAF + Combo |
| **9-11** | A | 75-100% | 65-75% | Combo or MOAF |
| **6-8** | B | 50-75% | 55-65% | Confirm or Continue |
| **3-5** | C | 25-50% | 45-55% | Consolidate |
| **0-2** | D | 0-25% | <45% | Skip |

---

### Quality-Based Management

**A+ Trades (Score 12-15):**
- Full size or overweight
- Widest stops
- Highest targets (3R+)
- Hold through minor pullbacks
- Scale in on dips

**A Trades (Score 9-11):**
- Full size
- Standard stops
- 2-3R targets
- Standard management
- Don't add to losers

**B Trades (Score 6-8):**
- Half size
- Tighter stops
- 1.5-2R targets
- Quick to break even
- Don't scale in

**C Trades (Score 3-5):**
- Quarter size or skip
- Very tight stops
- 1R target
- Exit quickly if wrong
- Never add to position

**D Trades (Score 0-2):**
- **DO NOT TRADE**
- Wait for higher quality setup

---

### Example Quality Assessment

**Scenario 1: Perfect Setup**
```
Daily: 2-1-2 bull (MOAF - daily just flipped)
60min: 2-1-2 bull (Combo - same pattern)
15min: 2-1-2 bull (Combo - same pattern)

Score Breakdown:
- Combo: +4 (all TF same pattern)
- Confirm: +3 (lower TF confirming)
- MOAF: +5 (daily flip)
Total: 12 points = A+ quality

Position Size: 100-150%
Expected Win Rate: 75-85%
```

**Scenario 2: Mediocre Setup**
```
Daily: Neutral (inside bars)
60min: 2-1-2 bull
15min: 2-1-2 bull

Score Breakdown:
- Combo: +4 (60min + 15min match)
Total: 4 points = C quality

Position Size: 25-50%
Expected Win Rate: 45-55%
```

**Scenario 3: Poor Setup**
```
Daily: Bearish trend
60min: 2-1-2 bull
15min: 2-1-2 bull

Score Breakdown:
- None (counter-trend)
Total: 0 points = D quality

Position Size: 0% - SKIP TRADE
```

---


---

## 6. Weekly/Monthly Timeframe Analysis

### 6.1 Extended Timeframe Hierarchy

**Full Hierarchy (highest to lowest authority):**
```
Monthly > Weekly > Daily > 60min > 15min > 5min
```

**Role of Each Timeframe:**

| Timeframe | Role | Typical Hold | Primary Use |
|-----------|------|--------------|-------------|
| **Monthly** | Macro bias | Weeks to months | Trend direction, major S/R |
| **Weekly** | Anchor timeframe | Days to weeks | Swing trade bias, position trades |
| **Daily** | Execution bias | 1-7 days | Swing entries, daily management |
| **60min** | Entry refinement | Hours to 1-2 days | Entry timing, intraday swings |
| **15min** | Precision entry | Minutes to hours | Scalps, precise entries |

---

### 6.2 Weekly Timeframe Patterns

**Weekly bars follow identical classification rules:**
- Type 1 (Inside): `Weekly_H <= Prior_Weekly_H AND Weekly_L >= Prior_Weekly_L`
- Type 2U (Up): `Weekly_H > Prior_Weekly_H AND Weekly_L >= Prior_Weekly_L`
- Type 2D (Down): `Weekly_H <= Prior_Weekly_H AND Weekly_L < Prior_Weekly_L`
- Type 3 (Outside): `Weekly_H > Prior_Weekly_H AND Weekly_L < Prior_Weekly_L`

**Weekly Pattern Significance:**
- Weekly inside bar (Type 1) = Major consolidation, explosive move pending
- Weekly 2-1-2 = High conviction swing setup (5-15 day hold)
- Weekly 3-1-2 = Major reversal signal (institutional repositioning)
- Weekly MOAF = Trend change, multi-week directional bias

**Weekly Pattern Expected Duration:**

| Pattern | Expected Duration | Target Timeframe |
|---------|-------------------|------------------|
| Weekly 2-1-2 | 5-15 trading days | Daily/Weekly |
| Weekly 3-1-2 | 10-20 trading days | Weekly |
| Weekly 2-2 | 3-10 trading days | Daily |
| Weekly Rev Strat | 15-30 trading days | Weekly/Monthly |

---

### 6.3 Monthly Timeframe Patterns

**Monthly = Macro Trend Bias**

Monthly patterns are rare but extremely significant:
- Monthly inside bar = Multi-week consolidation, major breakout pending
- Monthly 2U/2D = Established trend direction (trade with, not against)
- Monthly Type 3 = Volatility expansion, trend uncertainty
- Monthly MOAF = Major trend reversal (quarterly/yearly significance)

**Monthly Pattern Frequency:**
```
Typical year for SPY:
- Monthly Type 1: 1-2 occurrences
- Monthly Type 2: 8-10 occurrences  
- Monthly Type 3: 1-2 occurrences
- Monthly MOAF: 0-1 occurrences
```

**Monthly Bias Application:**
```python
def get_monthly_bias(monthly_bar_type, monthly_close_position):
    """
    Determine macro bias from monthly timeframe.
    
    Args:
        monthly_bar_type: Current month bar type (1, 2, -2, 3)
        monthly_close_position: 'upper_half' or 'lower_half' of range
    
    Returns:
        ('bullish', 'bearish', or 'neutral', confidence_level)
    """
    if monthly_bar_type == 2:  # 2U
        return 'bullish', 'high'
    elif monthly_bar_type == -2:  # 2D
        return 'bearish', 'high'
    elif monthly_bar_type == 1:  # Inside
        # Bias based on close position within range
        if monthly_close_position == 'upper_half':
            return 'bullish', 'low'
        else:
            return 'bearish', 'low'
    else:  # Type 3
        # Expansion - bias based on close
        if monthly_close_position == 'upper_half':
            return 'bullish', 'medium'
        else:
            return 'bearish', 'medium'
```

---

### 6.4 Weekly-Daily Continuity Analysis

**Optimal Weekly-Daily Alignment:**

| Weekly Bar | Daily Setup | Alignment | Trade Quality |
|------------|-------------|-----------|---------------|
| 2U | 2-1-2 Bull | ✅ Full | A+ (position size 100%) |
| 2U | 2-1-2 Bear | ❌ Counter | Skip or reduce 75% |
| 2D | 2-1-2 Bear | ✅ Full | A+ (position size 100%) |
| 2D | 2-1-2 Bull | ❌ Counter | Skip or reduce 75% |
| 1 (Inside) | Any | ⚠️ Neutral | Wait for weekly resolution |
| 3 | Matches close | ✅ Partial | B+ (position size 75%) |

**Weekly-Daily Continuity Score:**

```python
def calculate_weekly_daily_continuity(weekly_type, weekly_close_pos, 
                                       daily_pattern, daily_bias):
    """
    Score weekly-to-daily continuity.
    
    Returns:
        score (0-5), recommendation
    """
    score = 0
    
    # Weekly directional alignment
    if weekly_type == 2 and daily_bias == 'bull':
        score += 3  # Weekly 2U + daily bull = strong
    elif weekly_type == -2 and daily_bias == 'bear':
        score += 3  # Weekly 2D + daily bear = strong
    elif weekly_type == 1:
        score += 1  # Inside = neutral, slight bonus for any setup
    elif weekly_type == 3:
        # Outside bar - check close position
        if weekly_close_pos == 'upper' and daily_bias == 'bull':
            score += 2
        elif weekly_close_pos == 'lower' and daily_bias == 'bear':
            score += 2
    
    # Pattern quality bonus
    if daily_pattern in ['2-1-2', '3-1-2']:
        score += 1
    if daily_pattern == 'MOAF':
        score += 2
    
    # Recommendations
    if score >= 4:
        return score, 'HIGH CONVICTION - Full size'
    elif score >= 2:
        return score, 'MODERATE - 50-75% size'
    else:
        return score, 'LOW/COUNTER - Skip or 25% size'
```

---

### 6.5 Monthly-Weekly-Daily Cascade

**The Cascade Principle:**
```
Monthly sets MACRO direction (weeks to months)
  └── Weekly confirms SWING direction (days to weeks)
        └── Daily provides ENTRY timing (hours to days)
              └── 60min/15min refine EXECUTION (minutes to hours)
```

**Cascade Alignment Scoring:**

| Monthly | Weekly | Daily | Score | Action |
|---------|--------|-------|-------|--------|
| 2U | 2U | 2-1-2 Bull | 5/5 | Maximum conviction, full size |
| 2U | 2U | 2-1-2 Bear | 2/5 | Counter-trend, skip |
| 2U | 1 | 2-1-2 Bull | 3/5 | Waiting for weekly, reduced size |
| 2U | 2D | Any | 1/5 | Conflicting signals, skip |
| 1 | 2U | 2-1-2 Bull | 3/5 | Monthly neutral, follow weekly |
| 2D | 2D | 2-1-2 Bear | 5/5 | Maximum conviction, full size |

**Implementation:**

```python
def cascade_analysis(monthly_type, weekly_type, daily_pattern, daily_bias):
    """
    Analyze monthly-weekly-daily cascade for trade quality.
    
    Returns:
        (cascade_score, alignment_description, position_size_pct)
    """
    # Normalize types
    monthly_dir = 'bull' if monthly_type == 2 else ('bear' if monthly_type == -2 else 'neutral')
    weekly_dir = 'bull' if weekly_type == 2 else ('bear' if weekly_type == -2 else 'neutral')
    
    # Full alignment check
    all_bull = monthly_dir == 'bull' and weekly_dir == 'bull' and daily_bias == 'bull'
    all_bear = monthly_dir == 'bear' and weekly_dir == 'bear' and daily_bias == 'bear'
    
    if all_bull or all_bear:
        return 5, 'FULL CASCADE ALIGNMENT', 100
    
    # Partial alignment (2 of 3)
    directions = [monthly_dir, weekly_dir, daily_bias]
    bull_count = directions.count('bull')
    bear_count = directions.count('bear')
    neutral_count = directions.count('neutral')
    
    if bull_count >= 2 or bear_count >= 2:
        return 3, 'PARTIAL ALIGNMENT', 50
    
    if neutral_count >= 2:
        return 2, 'NEUTRAL/WAITING', 25
    
    # Conflicting
    return 1, 'CONFLICTING - SKIP', 0
```

---

### 6.6 Position Sizing by Timeframe

**Extended Position Sizing Matrix:**

| Primary TF | Cascade Score | Base Size | Hold Time |
|------------|---------------|-----------|-----------|
| Monthly | 5/5 | 100% | 2-8 weeks |
| Monthly | 3/5 | 50% | 1-4 weeks |
| Weekly | 5/5 | 100% | 5-15 days |
| Weekly | 3/5 | 50% | 3-10 days |
| Daily | 5/5 | 100% | 1-5 days |
| Daily | 3/5 | 50% | 1-3 days |

**Risk Adjustment by Timeframe:**

```python
def adjust_risk_for_timeframe(base_risk_pct, primary_timeframe):
    """
    Adjust risk percentage based on primary trading timeframe.
    
    Weekly/Monthly patterns have wider stops but higher conviction.
    
    Args:
        base_risk_pct: Base risk per trade (e.g., 1.0 for 1%)
        primary_timeframe: 'monthly', 'weekly', 'daily', '60min', '15min'
    
    Returns:
        adjusted_risk_pct
    """
    multipliers = {
        'monthly': 0.5,   # Half size due to wide stops
        'weekly': 0.75,   # Reduced size for wider stops
        'daily': 1.0,     # Standard
        '60min': 1.0,     # Standard
        '15min': 0.75     # Reduced for noise/whipsaws
    }
    
    return base_risk_pct * multipliers.get(primary_timeframe, 1.0)
```

---

### 6.7 Weekly/Monthly MOAF

**Weekly MOAF = Major Swing Reversal**

Characteristics:
- Weekly bar flips from 2U to 2D (or vice versa)
- Often occurs at major support/resistance
- Creates multi-week directional bias
- Daily bars will eventually align

**Weekly MOAF Detection:**

```python
def detect_weekly_moaf(weekly_bars):
    """
    Detect Mother of All Flips on weekly timeframe.
    
    Args:
        weekly_bars: Array of weekly bar types
    
    Returns:
        List of (index, direction) for MOAF occurrences
    """
    moafs = []
    
    for i in range(1, len(weekly_bars)):
        prev = weekly_bars[i-1]
        curr = weekly_bars[i]
        
        # Bull MOAF: 2D -> 2U
        if prev == -2 and curr == 2:
            moafs.append((i, 'bull'))
        # Bear MOAF: 2U -> 2D
        elif prev == 2 and curr == -2:
            moafs.append((i, 'bear'))
    
    return moafs
```

**Monthly MOAF = Trend Change**

- Extremely rare (0-1 per year typically)
- Signals major market regime change
- All lower timeframes eventually align
- Highest conviction signal in STRAT

---

### 6.8 Practical Weekly/Monthly Workflow

**Step 1: Monthly Analysis (Sunday)**
```
1. Classify current monthly bar type
2. Identify monthly S/R levels (prior month H/L)
3. Determine macro bias
4. Note: If monthly inside, prepare for breakout
```

**Step 2: Weekly Analysis (Sunday/Monday)**
```
1. Classify current weekly bar type
2. Check weekly-monthly alignment
3. Identify weekly triggers (prior week H/L)
4. Calculate position size multiplier
```

**Step 3: Daily Execution (Daily)**
```
1. Wait for daily pattern formation
2. Verify cascade alignment
3. Enter on daily trigger with weekly context
4. Manage position based on daily bars
```

**Example Workflow:**
```
Monthly: 2U (bullish macro)
Weekly: Type 1 inside (consolidation)
  → Weekly high = $450, low = $440
  → Waiting for weekly breakout

Daily: Forms 2-1-2 bull pattern
  → Trigger = $448
  → Stop = $445

Cascade score: 3/5 (monthly bull, weekly neutral, daily bull)
Position size: 50% (waiting for weekly confirmation)

If weekly breaks $450:
  → Cascade becomes 5/5
  → Add to 100% position
```

## Summary

**Timeframe Hierarchy:**
1. Weekly > Daily > 60min > 15min > 5min
2. Always respect higher timeframes
3. Enter on lower TF, manage on higher TF
4. MOAF = highest conviction signal

**4 C's Priority:**
1. **Combo** - Best (4 points)
2. **MOAF** - Highest conviction (5 points)
3. **Confirm** - Strong (3 points)
4. **Continue** - Moderate (2 points)
5. **Consolidate** - Weakest (1 point)

**Key Rules:**
- Never trade against daily bias
- MOAF overrides all other signals
- Score <3 = skip trade
- Quality determines size, not conviction

**Next Steps:**
- For entry/exit mechanics → Read [EXECUTION.md](EXECUTION.md)
- For pattern detection → Read [PATTERNS.md](PATTERNS.md)
- For options integration → Read [OPTIONS.md](OPTIONS.md)
