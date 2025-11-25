# VIX Acceleration + STRAT Cross-Asset Regime Enhancement

**Date:** November 14, 2025
**Status:** EXPLORATORY - Theory validated, implementation deferred
**Related Sessions:** Session 34 (exploration), Session 35+ (implementation)
**Origin:** Claude Desktop discussion on DEM/WEM dealer ranges led to jump model gap analysis

---

## Executive Summary

**Critical Gap Identified:** Current academic jump model lacks VIX integration and cross-asset confirmation, making it unsuitable for flash crash detection (e.g., August 5th VXX 20x trade).

**Proposed Solution:** 3-layer regime detection architecture:
1. Layer 1A: Academic clustering (existing - slow trends, weeks-months)
2. Layer 1B-1: VIX acceleration (NEW - flash crashes, minutes-hours)
3. Layer 1B-2: STRAT cross-asset timeframe continuity (NEW - institutional flow, days-weeks)

**Implementation Priority:** VIX acceleration first (highest ROI), STRAT cross-asset second, DEM/WEM third (lower priority).

---

## Context: How We Got Here

### Claude Desktop Discussion (Nov 14, 2025)

**User asked about DEM/WEM (Daily/Weekly Expected Move) for SPY:**
- DEM = ATM straddle premium for daily options (dealer range estimate)
- WEM = ATM straddle premium for weekly options
- "Lobster and cracked crab" = dealers profit when price stays inside ranges

**Assessment:** DEM/WEM are descriptive (where market WAS priced), not predictive. Useful as filters for "when NOT to trade" but not primary signals.

**Key Realization:** Discussion revealed that jump model lacks VIX integration entirely.

### Current Jump Model Status

**What's Actually Implemented:**

1. **Academic Statistical Jump Model** (`regime/academic_jump_model.py`)
   - 912 lines, 31/31 tests passing
   - Features: Downside Deviation + Sortino Ratios (20d, 60d)
   - Method: Coordinate descent clustering with temporal penalty
   - Performance: 77% March 2020 crash detection
   - **NO VIX DATA** - purely price-based

2. **Old Jump Model** (`regime/jump_model.py`)
   - Deprecated logistic function approach
   - Failed March 2020 (4.2% detection)
   - Should be deleted

**The Critical Gap:**

August 5th 2024 VXX trade (20x profit) came from VIX spike detection (VIX ~15 → ~40 in hours). Academic model uses 20-60 day smoothed ratios - would have lagged by DAYS.

**Current model: Excellent for stable regime classification (bull/bear over weeks). Inadequate for flash crash detection (VIX doubling in hours).**

---

## Proposed Architecture: 3-Layer Regime Detection

### Layer 1A: Academic Clustering (Existing)

**Purpose:** Stable baseline regime (bull/bear/neutral)
**Speed:** Weeks to months
**Method:** Coordinate descent clustering on 20-60 day features
**Status:** ✅ IMPLEMENTED (academic_jump_model.py)

**Strengths:**
- 33 years academic validation
- Low turnover (~1 switch/year with lambda=50)
- Proven Sharpe improvements (+42% to +158%)

**Weaknesses:**
- Backward-looking (20-60 day smoothing)
- Misses flash crashes (too slow)
- Lambda parameter struggles (Sessions 24-26 calibration issues)

### Layer 1B-1: VIX Acceleration (NEW - HIGH PRIORITY)

**Purpose:** Flash crash detection (August 5th scenarios)
**Speed:** Minutes to hours
**Method:** Z-score of VIX percentage change

**Implementation:**

```python
# regime/vix_acceleration.py

def detect_vix_acceleration(vix_data, lookback=20, z_threshold=3.0):
    """
    Detect VIX spikes using acceleration metrics.

    Returns:
    - 'FLASH_CRASH': VIX spike >3 sigma (like August 5th)
    - 'ELEVATED': VIX spike >2 sigma
    - 'NORMAL': Below thresholds
    """
    vix_change_1d = vix_data.pct_change()
    vix_zscore = (vix_change_1d - vix_change_1d.rolling(lookback).mean()) / \
                  vix_change_1d.rolling(lookback).std()

    if vix_zscore > z_threshold:
        return 'FLASH_CRASH'
    elif vix_zscore > 2.0:
        return 'ELEVATED'
    else:
        return 'NORMAL'
```

**Validation Targets:**
- August 5, 2024: Should trigger FLASH_CRASH
- March 2020: Should trigger ELEVATED early in crash
- 2020-2024 backtest: Measure false positive rate

**Why This Solves Lambda Issues:**

Academic model struggles because we're forcing ONE lambda to do TWO jobs:
- Low lambda (5-15): Responsive but noisy
- High lambda (50-150): Stable but laggy

**Solution:** Separate fast/slow detection:
- VIX handles flash crashes (no lambda needed)
- Academic handles slow trends (use high lambda=50-100)

### Layer 1B-2: STRAT Cross-Asset Timeframe Continuity (NEW - MEDIUM PRIORITY)

**Purpose:** Institutional flow detection (big money repositioning)
**Speed:** Days to weeks
**Method:** Simultaneous bar type changes across major indices

**Core Assets (4):**
- SPY (S&P 500 broad market)
- QQQ (Nasdaq tech-heavy)
- IWM (Russell 2000 small cap)
- DIA (Dow 30 blue chip)

**Why These 4:** Cover size spectrum (large → small cap) and sector tilt (tech vs value). When ALL 4 flip simultaneously, it's institutional repositioning, not sector rotation.

**Severity Levels:**

| Level | Trigger | Action | Conviction |
|-------|---------|--------|------------|
| **LEVEL 1: CAUTION** | All 4 daily bars go 2D/bearish 3 | Reduce positions, tighten stops | 40% |
| **LEVEL 2: HIGH CONVICTION** | All 4 weekly bars go 2D/bearish 3 | Exit longs, consider shorts | 75% |
| **LEVEL 3: TRIGGERED** | All 4 monthly bars go 2D/bearish 3 | Full regime change to CRASH | 95% |

**Inverse logic for bullish:** All 4 go 2U + weekly/monthly up = TREND_BULL escalation

**Implementation:**

```python
# regime/strat_cross_asset.py

def calculate_live_bar_type(today_high, today_low, yesterday_high, yesterday_low):
    """
    Returns current bar type based on what happened SO FAR today.
    Classification is IMMEDIATE and FINAL once price breaks levels.

    CRITICAL CORRECTION: Bar type determined by range expansion, NOT closing price.
    - Type 1: Price stayed inside previous range all day
    - Type 2U: Price broke ABOVE previous high (even if closes back inside)
    - Type 2D: Price broke BELOW previous low (even if closes back inside)
    - Type 3: Price broke BOTH sides (regardless of where it closes)
    """
    broke_above = (today_high > yesterday_high)
    broke_below = (today_low < yesterday_low)

    if broke_above and broke_below:
        return '3'  # Took out both sides (LOCKED)
    elif broke_above:
        return '2U'  # Took out high only (could still become 3)
    elif broke_below:
        return '2D'  # Took out low only (could still become 3)
    else:
        return '1'  # Still inside range (could become 2 or 3)

def detect_strat_cross_asset(assets_data, date):
    """
    Check if all 4 indices simultaneously bearish across timeframes.

    assets_data = {
        'SPY': {'daily': OHLC, 'weekly': OHLC, 'monthly': OHLC},
        'QQQ': {...},
        'IWM': {...},
        'DIA': {...}
    }
    """
    # Daily alignment
    daily_bearish = all([
        calculate_live_bar_type(
            asset['daily']['current_high'],
            asset['daily']['current_low'],
            asset['daily']['prev_high'],
            asset['daily']['prev_low']
        ) in ['2D', '3']
        for asset in assets_data.values()
    ])

    # Weekly alignment (requires daily + weekly)
    weekly_bearish = all([...])

    # Monthly alignment (requires all 3)
    monthly_bearish = all([...])

    if monthly_bearish:
        return 'MONTHLY_ALIGNED'  # Trigger
    elif weekly_bearish:
        return 'WEEKLY_ALIGNED'  # High conviction
    elif daily_bearish:
        return 'DAILY_CAUTION'  # Early warning
    else:
        return 'NORMAL'
```

**Cascading Break Mechanism (CRITICAL UNDERSTANDING):**

**Theoretical Example (User-provided):**

```
Oct 31st close:
- Daily low:   450
- Weekly low:  450  ← All 3 timeframes share same support
- Monthly low: 450

Nov 3rd at 1:35pm - Single 5min candle hits 449.99:

CASCADE HAPPENS INSTANTLY:
5min → breaks 450 → goes 2D
   ↓ triggers
15min → breaks 450 → goes 2D
   ↓ triggers
30min → breaks 450 → goes 2D
   ↓ triggers
1H → breaks 450 → goes 2D
   ↓ triggers
4H → breaks 450 → goes 2D (MOAF confirmed)
   ↓ triggers
Daily → breaks 450 → goes 2D
   ↓ triggers
Weekly → breaks 450 → goes 2D
   ↓ triggers
Monthly → breaks 450 → goes 2D

ALL at 1:35pm, NOT spread throughout day/week/month.
```

**Feedback Loop:**
1. Monthly low breaks → stops triggered
2. Stops trigger → selling pressure
3. Selling pressure → breaks next level
4. Repeat → cascading liquidation

**This is EXACTLY what happened March 2020, August 2024, etc.**

**Why This Matters for Monitoring:**

Original incorrect assumption: "Weekly bars change slower, check every 30-60 min. Monthly bars rarely flip intraday, check EOD only."

**WRONG.** When timeframes CONVERGE on same level (common at month/week closes, major support/resistance), breaking ONE level = breaking ALL timeframes simultaneously.

**CORRECT:** ALL timeframes must be monitored at SAME frequency (5 minutes) to detect cascading breaks in real-time.

---

## Unified Regime Detection Logic

```python
def detect_regime(spy_data, vix_data, multi_asset_data):
    """
    Priority stack: VIX flash crash > STRAT cross-asset > Academic clustering
    """
    # LAYER 1B-1: VIX Flash Crash (highest priority, fastest)
    vix_severity = detect_vix_acceleration(vix_data)
    if vix_severity == 'FLASH_CRASH':
        return 'CRASH'  # Immediate override

    # LAYER 1B-2: STRAT Cross-Asset Continuity (institutional flow)
    strat_severity = detect_strat_cross_asset(multi_asset_data)
    if strat_severity == 'MONTHLY_ALIGNED':
        return 'CRASH'  # High conviction regime change
    elif strat_severity == 'WEEKLY_ALIGNED':
        return 'TREND_BEAR'  # Exit longs
    elif strat_severity == 'DAILY_CAUTION':
        # Reduce position sizes (don't override regime yet)
        pass

    # LAYER 1A: Academic clustering (base case)
    return academic_model.online_inference(spy_data)
```

**Why This Order:**
1. VIX = fastest (catches flash crashes in minutes-hours)
2. STRAT cross-asset = predictive (catches building pressure before break)
3. Academic = stable baseline (slow trends over weeks-months)

---

## Monitoring Architecture

### API Feasibility Analysis

**Required Monitoring Frequency:**

| Signal | Frequency | Reason |
|--------|-----------|--------|
| VIX acceleration | 5 min | Flash crashes happen in minutes-hours |
| Daily bar cross-asset | 5 min | Can cascade from intraday break |
| Weekly bar cross-asset | 5 min | Can cascade from daily break |
| Monthly bar cross-asset | 5 min | Can cascade from weekly break |

**Why 5 min for ALL timeframes:** Cannot predict when pivots will align. Must monitor continuously to detect cascading breaks in real-time.

**API Call Math:**

Per 5-minute poll:
- VIX: 1 call (1min data)
- SPY/QQQ/IWM/DIA: 4 calls (1min data each)
- Total: 5 calls per 5 min

Per hour: 5 × 12 = 60 calls
Per day: 60 × 6.5 = 390 calls

**Alpaca rate limit:** 200 requests/minute = 12,000 requests/hour

**Usage: 390 / 78,000 = 0.5% of daily capacity** ✅ Totally feasible

### Monitoring Loop Implementation

```python
def monitor_regime_realtime():
    """
    5-minute monitoring for VIX + STRAT cross-asset (all timeframes).
    """
    # Initialize at market open (9:30am)
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
    reference_bars = {
        symbol: initialize_reference_bars(symbol)
        for symbol in symbols
    }

    while market_is_open():
        now = datetime.now()

        # --- VIX MONITORING ---
        vix_data = fetch_alpaca_data('VIX', '1Min', period_days=20)
        vix_severity = detect_vix_acceleration(vix_data)

        if vix_severity == 'FLASH_CRASH':
            print(f"[{now}] VIX FLASH CRASH")
            update_regime('CRASH')
            send_alert('VIX_FLASH_CRASH')

        # --- STRAT CROSS-ASSET (ALL TIMEFRAMES) ---
        all_bars = {}
        cascade_risks = {}

        for symbol in symbols:
            bars = poll_all_timeframes(symbol, reference_bars[symbol])
            all_bars[symbol] = bars
            cascade_risks[symbol] = bars['cascade_potential']

        # Check daily alignment
        all_daily_bearish = all([
            all_bars[s]['daily_bar'] in ['2D', '3']
            for s in symbols
        ])

        # Check weekly alignment
        all_weekly_bearish = all([
            all_bars[s]['weekly_bar'] in ['2D', '3']
            for s in symbols
        ])

        # Check monthly alignment
        all_monthly_bearish = all([
            all_bars[s]['monthly_bar'] in ['2D', '3']
            for s in symbols
        ])

        # Check cascade risk
        high_cascade_risk = any([
            cascade_risks[s]['cascade_risk'] == 'HIGH'
            for s in symbols
        ])

        # --- REGIME CLASSIFICATION ---
        if all_monthly_bearish:
            print(f"[{now}] MONTHLY TRIGGERED")
            update_regime('CRASH')
            send_alert('STRAT_MONTHLY_TRIGGER')

        elif all_weekly_bearish:
            print(f"[{now}] WEEKLY CONFIRMED")
            update_regime('TREND_BEAR')
            send_alert('STRAT_WEEKLY_CONFIRM')

        elif all_daily_bearish:
            print(f"[{now}] DAILY CAUTION")
            if high_cascade_risk:
                print(f"[{now}] WARNING: Cascade risk HIGH")
            update_regime('DAILY_CAUTION')

        time.sleep(300)  # 5 minutes
```

### Data Requirements Per Symbol

**ONE-TIME FETCH at 9:30am:**
```python
def initialize_reference_bars(symbol):
    """Fetch previous timeframe bars once at market open."""

    daily_data = fetch_alpaca_data(symbol, '1D', period_days=2)
    prev_day_high = daily_data.iloc[-1]['High']
    prev_day_low = daily_data.iloc[-1]['Low']

    weekly_data = fetch_alpaca_data(symbol, '1W', period_weeks=2)
    prev_week_high = weekly_data.iloc[-1]['High']
    prev_week_low = weekly_data.iloc[-1]['Low']

    monthly_data = fetch_alpaca_data(symbol, '1M', period_months=2)
    prev_month_high = monthly_data.iloc[-1]['High']
    prev_month_low = monthly_data.iloc[-1]['Low']

    return {
        'daily': {'high': prev_day_high, 'low': prev_day_low},
        'weekly': {'high': prev_week_high, 'low': prev_week_low},
        'monthly': {'high': prev_month_high, 'low': prev_month_low}
    }
```

**INTRADAY POLLING (every 5 min):**
```python
def poll_all_timeframes(symbol, reference_bars):
    """
    Fetch current intraday high/low.
    Classify against ALL 3 timeframes SIMULTANEOUSLY.
    """
    intraday = fetch_alpaca_data(symbol, '1Min', period_minutes=400)

    current_high = intraday['High'].max()
    current_low = intraday['Low'].min()

    # Classify bar type for ALL timeframes using SAME current high/low
    daily_bar = classify_bar_type(
        current_high, current_low,
        reference_bars['daily']['high'],
        reference_bars['daily']['low']
    )

    weekly_bar = classify_bar_type(
        current_high, current_low,
        reference_bars['weekly']['high'],
        reference_bars['weekly']['low']
    )

    monthly_bar = classify_bar_type(
        current_high, current_low,
        reference_bars['monthly']['high'],
        reference_bars['monthly']['low']
    )

    # Detect aligned pivots (cascade risk)
    cascade_potential = detect_aligned_pivots(reference_bars)

    return {
        'current_high': current_high,
        'current_low': current_low,
        'daily_bar': daily_bar,
        'weekly_bar': weekly_bar,
        'monthly_bar': monthly_bar,
        'cascade_potential': cascade_potential
    }
```

**Cascade Risk Detection:**
```python
def detect_aligned_pivots(reference_bars, threshold=0.01):
    """
    Detect if daily/weekly/monthly pivots are aligned (cascade risk).
    threshold: 1% = pivots considered "aligned" if within 1%
    """
    daily_low = reference_bars['daily']['low']
    weekly_low = reference_bars['weekly']['low']
    monthly_low = reference_bars['monthly']['low']

    low_range = max(daily_low, weekly_low, monthly_low) - \
                min(daily_low, weekly_low, monthly_low)
    low_aligned = (low_range / daily_low) < threshold

    daily_high = reference_bars['daily']['high']
    weekly_high = reference_bars['weekly']['high']
    monthly_high = reference_bars['monthly']['high']

    high_range = max(daily_high, weekly_high, monthly_high) - \
                 min(daily_high, weekly_high, monthly_high)
    high_aligned = (high_range / daily_high) < threshold

    return {
        'lows_aligned': low_aligned,
        'highs_aligned': high_aligned,
        'cascade_risk': 'HIGH' if (low_aligned or high_aligned) else 'NORMAL'
    }
```

---

## DEM/WEM Assessment (Lower Priority)

### What They Are

- **DEM (Daily Expected Move):** ATM call premium + ATM put premium (nearest expiration)
- **WEM (Weekly Expected Move):** Same calculation using weekly options
- Represents ~1 standard deviation price range based on implied volatility

### "Dealer Paradise" Concept

When price stays INSIDE expected move ranges:
- Options sellers (market makers) profit
- Theta decay accelerates
- Low realized vol < implied vol
- Dealers capture premium without aggressive hedging

### Assessment: 6/10 Value

**Pros:**
- Easy to implement (options chain ATM straddle premiums)
- Useful filter for "dealer paradise" periods (stand aside)
- Adds risk management layer
- Real-time calculation possible

**Cons:**
- Doesn't address VIX gap (primary issue)
- Adds complexity without adding alpha
- Requires options data (another API dependency)
- Lagging indicator (same critique as academic model for flash crashes)
- Designed for options sellers, not breakout traders

### Recommended Use (If Implemented)

```python
# GOOD USE - Regime Filter
if price_within_DEM and realized_vol < implied_vol * 0.7:
    regime = "dealer_paradise"  # Stand aside
    position_size_multiplier = 0.0

# GOOD USE - Breakout Confirmation
if price_breaks_WEM_with_volume and vix_acceleration:
    regime = "potential_trend"  # Your wheelhouse
    position_size_multiplier = 1.0

# BAD USE - Don't Do This
if price_at_DEM_upper:
    enter_long()  # This is NOT your edge
```

**Bottom Line:** Use as FILTER ("when NOT to trade"), not as entry signal. Your edge is catching what BREAKS these ranges, not trading within them.

**Priority:** Implement AFTER VIX and STRAT cross-asset are working.

---

## Implementation Roadmap

### Phase 1: VIX Acceleration Layer (HIGH PRIORITY)

**Why First:**
- Addresses August 5th success case (your proven edge)
- Solves lambda calibration issues (separate fast/slow detection)
- Adds true predictive edge (academic model is backward-looking)
- Simplest to implement (1-2 hours)

**Tasks:**
1. Create `regime/vix_acceleration.py`
2. Implement `detect_vix_acceleration(vix_data, lookback=20, z_threshold=3.0)`
3. Test on August 5, 2024 (should trigger FLASH_CRASH)
4. Test on March 2020 (should trigger ELEVATED early)
5. Backtest 2020-2024: Measure false positive rate
6. Integrate with academic model (VIX overrides clustering)

**VectorBT Pro Verification Required:**
- Search for VIX data fetching examples
- Verify percentage change calculations
- Test z-score rolling window operations

### Phase 2: STRAT Cross-Asset Layer (MEDIUM PRIORITY)

**Why Second:**
- Complements VIX (different timescale)
- Adds institutional flow signal
- Higher conviction when combined with VIX
- More complex (2-3 hours)

**Tasks:**
1. Create `regime/strat_cross_asset.py`
2. Implement `calculate_live_bar_type()` function
3. Implement `detect_strat_cross_asset()` with daily/weekly/monthly checks
4. Implement `detect_aligned_pivots()` for cascade risk
5. Test on March 2020 (should show daily → weekly → monthly escalation)
6. Backtest 2020-2024: Validate cross-asset confirmation
7. Integrate with unified regime detection

**VectorBT Pro Verification Required:**
- Search for multi-timeframe data fetching
- Verify resampling daily → weekly → monthly
- Test intraday high/low tracking

### Phase 3: DEM/WEM Filter Layer (LOW PRIORITY)

**Why Third:**
- Lower ROI than VIX/STRAT
- Requires additional options data
- Risk management enhancement, not core alpha

**Tasks:**
1. Create `regime/dealer_ranges.py`
2. Implement `calculate_dem_wem(options_chain)`
3. Implement `assess_dealer_environment(price, dem_range, wem_range)`
4. Integrate as position sizing filter
5. Backtest: Measure impact on trade frequency

**VectorBT Pro Verification Required:**
- Search for options chain data fetching
- Verify ATM strike selection logic

### Phase 4: Monitoring Infrastructure (AFTER PHASES 1-2)

**Tasks:**
1. Create `regime/monitoring.py`
2. Implement `monitor_regime_realtime()` loop
3. Implement state persistence (`regime/state.py`)
4. Implement alerting (Discord/email/file flags)
5. Test during market hours (paper trading mode)
6. Validate API call usage vs Alpaca limits

---

## Critical Implementation Requirements

### 1. VectorBT Pro Compliance (MANDATORY)

**ZERO TOLERANCE for skipping 5-step workflow:**

```
1. SEARCH - mcp__vectorbt-pro__search() for patterns/examples
2. VERIFY - resolve_refnames() to confirm methods exist
3. FIND - mcp__vectorbt-pro__find() for real-world usage
4. TEST - mcp__vectorbt-pro__run_code() minimal example
5. IMPLEMENT - Only after 1-4 pass successfully
```

**Example for VIX implementation:**
```python
# Step 1: Search
mcp__vectorbt-pro__search("VIX data fetching percentage change")

# Step 2: Verify
mcp__vectorbt-pro__resolve_refnames(["vbt.YFData.pull", "pd.Series.pct_change"])

# Step 3: Find
mcp__vectorbt-pro__find(["YFData"])

# Step 4: Test
mcp__vectorbt-pro__run_code("""
import vectorbtpro as vbt
vix = vbt.YFData.pull('VIX', start='2024-08-01', end='2024-08-10').get()
vix_change = vix['Close'].pct_change()
print(vix_change.tail())
""")

# Step 5: Implement (only after above passes)
```

### 2. Data Source Validation

**VIX Data:**
- Alpaca: Check if VIX available (may be limited)
- Alternative: CBOE VIX via yfinance
- Fallback: VXX ETF as proxy

**Multi-Timeframe Data:**
- Fetch daily bars, resample to weekly/monthly
- Validate resampling logic (weekly = Friday close, monthly = last trading day)
- Handle holidays/partial weeks correctly

### 3. Edge Case Handling

**Market Gaps:**
- Monday open gaps down through all 3 timeframe lows
- Handle pre-market data if available
- Default to 9:30am ET open if no pre-market

**Partial Bars:**
- First 5 minutes of day: Bar type may be unstable
- Consider minimum time threshold (e.g., 15 min into session)

**Holiday Weeks:**
- Short weeks (3-4 days): Weekly bar calculation
- Month-end falling on holiday: Use last trading day

---

## Validation Targets

### VIX Acceleration Layer

**Test Case 1: August 5, 2024**
- VIX spike: ~15 → ~40 (167% intraday)
- Expected: FLASH_CRASH trigger
- Timing: Should trigger within 30 minutes of spike start

**Test Case 2: March 2020**
- VIX progression: 15 → 30 → 50 → 82 over 2 weeks
- Expected: ELEVATED trigger early (week 1), FLASH_CRASH at peak
- Should detect regime change before academic model

**Test Case 3: Normal Volatility (2021-2022)**
- VIX range: 15-30 (normal fluctuations)
- Expected: Minimal false positives (<5 per year)
- Should NOT trigger on routine 10-20% VIX moves

### STRAT Cross-Asset Layer

**Test Case 1: March 2020 Daily → Weekly → Monthly Cascade**
- Week 1: All 4 indices daily bars bearish → DAILY_CAUTION
- Week 2: All 4 indices weekly bars bearish → WEEKLY_CONFIRMED
- Week 3: All 4 indices monthly bars bearish → MONTHLY_TRIGGERED
- Expected: Clear escalation pattern

**Test Case 2: Sector Rotation (No Trigger)**
- Tech selloff: QQQ bearish, SPY/IWM/DIA neutral
- Expected: NO TRIGGER (not cross-asset confirmation)
- Filter out sector-specific moves

**Test Case 3: Aligned Pivots Cascade**
- All 4 indices with lows within 1% (aligned)
- Single intraday break triggers all timeframes simultaneously
- Expected: Cascade risk warning + immediate escalation to WEEKLY_CONFIRMED

### Combined Layers

**Test Case: August 5, 2024**
- VIX: FLASH_CRASH trigger (Layer 1B-1)
- STRAT: Daily bearish across all 4 (Layer 1B-2)
- Expected: Both layers confirm → highest conviction CRASH signal

---

## Open Questions for Implementation

1. **VIX Data Source:**
   - Is VIX available via Alpaca API?
   - Should we use VXX ETF as proxy?
   - Latency requirements: Real-time or 15-min delayed?

2. **Monitoring Deployment:**
   - Run as separate process or integrated into main trading loop?
   - Use cron/scheduler or persistent background service?
   - How to handle overnight gaps (futures data)?

3. **Alerting Preferences:**
   - Discord webhook for alerts?
   - Email notifications?
   - File flags for other processes to read?

4. **Position Sizing Integration:**
   - How does regime severity affect position size?
   - FLASH_CRASH = exit all positions immediately?
   - DAILY_CAUTION = reduce to 50% size?

5. **Backtest Integration:**
   - Simulate intraday monitoring in backtests?
   - Or use EOD regime assignments for simplicity?
   - How to validate monitoring loop logic without live data?

---

## References

### Source Documents

1. **Claude Desktop Discussion:**
   - File: `c:\Users\sheeh\Downloads\ATLAS_DEM_WEM_Enhancement_Discussion.md`
   - Date: November 14, 2025
   - Key insights: DEM/WEM assessment, VIX gap analysis, monitoring feasibility

2. **Chat Transcript:**
   - File: `c:\Users\sheeh\Downloads\Claude-DEM and WEM market maker levels for SPY.md`
   - Date: November 14, 2025
   - Context: Market analysis November 11-14, 2025 (SPY/VIX action)

### OpenMemory Context

- Session 11-17: Academic jump model implementation phases
- Session 24-26: Lambda calibration issues (archived)
- Session 29: STRAT skill refinement
- Session 33: STRAT-ATLAS integration
- Session 34 Part 1: 52-week high momentum strategy

### Code Files

- `regime/academic_jump_model.py` (912 lines, current implementation)
- `regime/jump_model.py` (490 lines, deprecated)
- `strat/atlas_integration.py` (150 lines, signal quality matrix)
- `~/.claude/skills/strat-methodology/` (STRAT skill files)

### Academic References

- Shu et al., Princeton 2024: Jump model clustering approach
- Novy-Marx (2012): Momentum strategies
- George & Hwang (2004): 52-week high momentum

---

## Status and Next Steps

**Current Status:** EXPLORATORY - Theory validated, no code implemented

**Next Session Priorities:**

**Option A:** Debug 52-week high signal generation first (fix existing strategy)
- Only 3 trades in 20 years (expected hundreds)
- Performance below architecture targets
- Original Session 35 plan

**Option B:** Implement VIX acceleration detector first (fix regime foundation)
- Addresses critical gap in jump model
- Enables August 5th-style trades algorithmically
- 1-2 hour implementation

**Option C:** Do both (debug signals, then add VIX if time permits)

**USER DECISION REQUIRED before proceeding.**

---

## Conclusion

This exploration revealed a critical architectural gap: **The current jump model lacks VIX integration**, making it unsuitable for flash crash detection despite that being a proven edge (August 5th 20x VXX trade).

**The proposed 3-layer architecture solves this:**
- Layer 1A (Academic): Stable trends (existing)
- Layer 1B-1 (VIX): Flash crashes (NEW - high priority)
- Layer 1B-2 (STRAT): Institutional flow (NEW - medium priority)

**Implementation is feasible:**
- API usage: 390 calls/day (0.5% of Alpaca capacity)
- Monitoring frequency: 5 minutes for all signals
- Clear validation targets (August 5th, March 2020)

**Next step:** Choose implementation priority (VIX first recommended) and execute with full VectorBT Pro compliance workflow.
