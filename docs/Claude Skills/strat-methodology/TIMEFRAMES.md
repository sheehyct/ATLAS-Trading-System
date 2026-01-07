# STRAT Timeframes - Multi-Timeframe Continuity Analysis

**Purpose:** Implementation guide for timeframe continuity, MOAF detection, and trade quality scoring  
**Parent:** [SKILL.md](SKILL.md)

---

## Table of Contents

1. [Timeframe Continuity Fundamentals](#1-timeframe-continuity-fundamentals)
   - 1.1 The 4 C's Framework (Control/Confirm/Conflict/Change)
   - 1.2 Participation Groups
   - 1.3 Control Analysis
   - 1.4 TFC Scoring
   - 1.5 Green/Red Candle Contribution
   - 1.6 TFC and Position Sizing
   - 1.7 MOAF (Mother of All Frames - 13:30 EST)
2. [MOAF - Mother of All Flips](#2-moaf-mother-of-all-flips)
3. [Timeframe Relationships](#3-timeframe-relationships)
4. [Continuity Scoring](#4-continuity-scoring)
5. [Trade Quality Matrix](#5-trade-quality-matrix)

---

## 1. Timeframe Continuity Fundamentals

### 1.1 The 4 C's Framework

The 4 C's are diagnostic questions to evaluate timeframe alignment:

| C | Question | What It Reveals |
|---|----------|-----------------|
| **Control** | Which participation group(s) control current price direction? | Identifies dominant force |
| **Confirm** | Are all participation groups confirming each other's direction? | Checks alignment |
| **Conflict** | Are any participation groups in conflict? | Identifies divergence |
| **Change** | Are any groups changing the continuity or direction of others? | Spots transitions |

**IMPORTANT:** The 4 C's are analytical questions, NOT position sizing rules or pattern categories.

### 1.2 Participation Groups

| Group | Timeframe | Represents | Control Duration |
|-------|-----------|------------|------------------|
| Monthly | 1 bar/month | Institutional players | Weeks to months |
| Weekly | 1 bar/week | Swing traders, funds | Days to weeks |
| Daily | 1 bar/day | Day-to-day participants | Hours to days |
| 60-min | 1 bar/hour | Intraday participants | Minutes to hours |

**Standard Timeframes:** Monthly, Weekly, Daily, 60-minute

**High VIX Environment (25-30+):** Time compression shifts analysis:
- Monthly analysis -> 60-minute
- Weekly analysis -> 30-minute
- Daily analysis -> 15-minute
- Hourly analysis -> 5-minute (or 1-minute)

### 1.3 Control Analysis

**Who controls RIGHT NOW:**
- When 60-min and Daily confirm each other, they show IMMEDIATE control
- This may override Weekly/Monthly direction temporarily
- Shorter timeframes = shorter duration of control
- Watch for flips - control can shift quickly

**Who controls LONGER TERM:**
- Monthly and Weekly represent larger, more persistent positioning
- When these flip, it signals significant institutional change
- Trades aligned with Monthly/Weekly have more staying power

### 1.4 TFC Scoring

TFC Score counts how many timeframes show DIRECTIONAL alignment:

| Bar Type | Counts Toward TFC? | Direction Determined By |
|----------|-------------------|------------------------|
| Type 1 (Inside) | NO - indecision | N/A |
| Type 2U | YES | Bullish (broke high) |
| Type 2D | YES | Bearish (broke low) |
| Type 3 (Outside) | YES | Green = Bullish, Red = Bearish |

**Scoring Examples:**

```
4/4 TFC (FTFC): All four timeframes directional and aligned
3/4 TFC: Three timeframes aligned, one in conflict or indecision
2/4 TFC: Two timeframes aligned - mixed/conflicted
1/4 TFC: One timeframe aligned - counter-trend territory
```

**Example Calculation:**

| TF | Bar Type | Color | TFC Contribution |
|----|----------|-------|------------------|
| M | 2U | Green | Bullish (counts) |
| W | 1 | - | Indecision (does NOT count) |
| D | 2U | Green | Bullish (counts) |
| H | 2U | Green | Bullish (counts) |

**Result:** 3/4 TFC Bullish (Weekly indecision means institutions have not committed)

### 1.5 Green/Red Candle Contribution

Green/Red (close vs open) serves two purposes:

**For Type 2 Bars - Conviction Modifier:**
- 2U + Green = Broke high AND buyers held = strong bullish
- 2U + Red = Broke high BUT sellers took over = weaker/conflicted
- 2D + Red = Broke low AND sellers held = strong bearish
- 2D + Green = Broke low BUT buyers stepped in = weaker/conflicted

**For Type 3 Bars - Direction Determination:**
- Type 3 + Green = Counts as bullish TFC
- Type 3 + Red = Counts as bearish TFC

### 1.6 TFC and Position Sizing

TFC score CAN inform position sizing decisions:

| TFC Score | Conviction Level | Position Size Consideration |
|-----------|------------------|----------------------------|
| 4/4 (FTFC) | Highest | Full position acceptable |
| 3/4 | High | Near-full position |
| 2/4 | Moderate/Conflicted | Reduced position |
| 1/4 | Low (counter-trend) | Minimal position or avoid |

**Additional Factors:**
- Higher timeframes green while directional = more significance
- Higher timeframes red despite being directional = caution warranted
- Balance TFC score against WHO is in control right now

**Example - Caution Scenario:**

| TF | Bar Type | Color | Notes |
|----|----------|-------|-------|
| M | 2U | Red | Directional but selling pressure |
| W | 2U | Red | Directional but selling pressure |
| D | 2U | Green | Directional, buyers holding |
| H | 2U | Green | Directional, buyers holding |

TFC = 4/4 Bullish, BUT Monthly and Weekly showing red suggests caution. Consider smaller position than if all were green.

### 1.7 MOAF (Mother of All Frames)

**What:** 13:30 EST - when the 2nd 4-hour candle of the trading session opens.

**Structure:**
- Market hours: 09:30-16:00 EST
- First 4H candle: 09:30-13:30 EST
- Second 4H candle: 13:30-16:00 EST

**At 13:30 EST, multiple timeframes get new candles simultaneously:**
- New 4H candle (2nd of the day)
- New 1H candle (4th hour from market open)
- New 30m, 15m, 5m candles all align

**Why It Matters:**
- When 2nd 4H candle breaks above/below first 4H range, this CAN indicate increased institutional participation
- Institutions use larger timeframes and multi-timeframe analysis
- A directional break on 4H aligns with how they analyze markets

**Important Nuance:**
- MOAF does NOT mean institutions specifically watch 13:30
- It DOES mean a 4H range break aligns with institutional analysis methods
- Increased participation is potential, not guaranteed

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
Daily: Flips bullish (2-1-2 bull or 2-2 reversal bull)
60min: Still bearish or neutral
15min: Still bearish

Daily leads, lower TFs will follow
```

**Bearish MOAF:**
```
Daily: Flips bearish (2-1-2 bear or 2-2 reversal bear)
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
