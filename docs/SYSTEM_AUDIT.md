# ATLAS STRAT System Audit

**Session:** 83K-54
**Date:** December 7, 2025
**Purpose:** Document EXACTLY what the code does before VPS deployment

---

## Executive Summary

This audit traces through all 7 critical decision points in the STRAT options trading system to document:
1. What the code currently does
2. Where each decision came from (user discussion vs Claude's choosing)
3. Potential issues or concerns

**Key Finding:** Most core logic follows STRAT methodology (discussed with user). However, several operational parameters were pragmatic defaults set by Claude without explicit discussion.

---

## Component 1: Entry Logic

### Current Implementation

**Files:**
- `strat/pattern_detector.py` (lines 49-169 for 3-1-2, lines 172-336 for 2-1-2, lines 464-652 for 2-2, lines 655-858 for 3-2, lines 861-988 for 3-2-2)
- `strat/tier1_detector.py` (lines 346-481)

**Logic:**

| Pattern | Entry Price | Code Location |
|---------|-------------|---------------|
| 3-1-2 Bullish | Inside bar high `high[i-1]` | pattern_detector.py:113 |
| 3-1-2 Bearish | Inside bar low `low[i-1]` | pattern_detector.py:125 |
| 2-1-2 Bullish | Inside bar high `high[i-1]` | pattern_detector.py:242 |
| 2-1-2 Bearish | Inside bar low `low[i-1]` | pattern_detector.py:255 |
| 2-2 Bullish (2D-2U) | Previous bar high `high[i-1]` | pattern_detector.py:554 |
| 2-2 Bearish (2U-2D) | Previous bar low `low[i-1]` | pattern_detector.py:576 |
| 3-2 Bullish | Outside bar high `high[i-1]` | pattern_detector.py:734 |
| 3-2 Bearish | Outside bar low `low[i-1]` | pattern_detector.py:768 |
| 3-2-2 Bullish | 2D bar high `high[i-1]` | pattern_detector.py:941 |
| 3-2-2 Bearish | 2U bar low `low[i-1]` | pattern_detector.py:953 |

**Confirmation Logic:**
- Entry triggers when price breaks the entry level on subsequent bar
- Backtest (`backtest_strat_equity_validation.py:1391-1430`) checks `high >= entry_trigger` (bullish) or `low <= entry_trigger` (bearish)
- Slippage model: Worst-case entry capped at 0.2% from trigger (Session 78 fix)

### Origin

- [x] Discussed with user
- [ ] Claude's choosing (pragmatic gap-fill)
- [ ] Unknown

**Evidence:** Session 83K-4 (OpenMemory) explicitly discusses STRAT entry methodology: "Entry when underlying breaks pattern entry" per Rob Smith's STRAT methodology. Entry is the previous bar's high/low because patterns are detected at end of trigger bar but entry happens LIVE when price breaks the level.

### Notes

This is CORRECT per STRAT methodology. The "i-1" bar is where the entry trigger level is set, not where we enter. We enter when the CURRENT bar (i) breaks that level during its formation.

---

## Component 2: Stop Loss

### Current Implementation

**Files:**
- `strat/pattern_detector.py` (same line ranges as above)

**Logic:**

| Pattern | Stop Price | Code Location |
|---------|------------|---------------|
| 3-1-2 Bullish | Outside bar low `low[i-2]` | pattern_detector.py:114 |
| 3-1-2 Bearish | Outside bar high `high[i-2]` | pattern_detector.py:126 |
| 2-1-2 Bullish | Inside bar low `low[i-1]` | pattern_detector.py:243 |
| 2-1-2 Bearish | Inside bar high `high[i-1]` | pattern_detector.py:256 |
| 2-2 Bullish | 2D bar low `low[i-1]` | pattern_detector.py:555 |
| 2-2 Bearish | 2U bar high `high[i-1]` | pattern_detector.py:577 |
| 3-2 Bullish | Outside bar low `low[i-1]` | pattern_detector.py:735 |
| 3-2 Bearish | Outside bar high `high[i-1]` | pattern_detector.py:769 |
| 3-2-2 Bullish | 2D bar low `low[i-1]` | pattern_detector.py:942 |
| 3-2-2 Bearish | 2U bar high `high[i-1]` | pattern_detector.py:954 |

**Stop Logic Summary:**
- 3-1-2: Uses outside bar's opposite extreme (wider stop for wider pattern)
- 2-1-2: Uses inside bar's opposite extreme (tighter stop)
- 2-2/3-2/3-2-2: Uses trigger bar's opposite extreme

### Origin

- [x] Discussed with user
- [ ] Claude's choosing (pragmatic gap-fill)
- [ ] Unknown

**Evidence:** Session 83K-4 (OpenMemory): "EXACT STOP - Pattern invalidation (reversal bar) gives exact exit point." This is standard STRAT methodology - stop is at the structural level that invalidates the pattern.

### Notes

The different stop placement for 3-1-2 (outside bar extreme) vs 2-1-2 (inside bar extreme) is CORRECT per STRAT. The wider pattern (3-1-2 has outside bar) gets wider stop; the narrower pattern (2-1-2 inside bar) gets tighter stop.

---

## Component 3: Target/Magnitude

### Current Implementation

**Files:**
- `strat/pattern_detector.py` (lines 376-461 for geometry validation/fallback)

**Logic:**

| Pattern | Target Price | Fallback | Code Location |
|---------|--------------|----------|---------------|
| 3-1-2 Bullish | Outside bar high `high[i-2]` | N/A | pattern_detector.py:115 |
| 3-1-2 Bearish | Outside bar low `low[i-2]` | N/A | pattern_detector.py:127 |
| 2-1-2 Bullish | First directional bar high `high[i-2]` | N/A | pattern_detector.py:244 |
| 2-1-2 Bearish | First directional bar low `low[i-2]` | N/A | pattern_detector.py:257 |
| 2-2 Bullish | Bar[i-2] high | Measured move 1.5x R:R | pattern_detector.py:558-565 |
| 2-2 Bearish | Bar[i-2] low | Measured move 1.5x R:R | pattern_detector.py:580-587 |
| 3-2 Bullish | Previous outside bar high | Measured move 1.5x R:R | pattern_detector.py:747-757 |
| 3-2 Bearish | Previous outside bar low | Measured move 1.5x R:R | pattern_detector.py:779-790 |
| 3-2-2 Bullish | Outside bar high `high[i-2]` | N/A | pattern_detector.py:943 |
| 3-2-2 Bearish | Outside bar low `low[i-2]` | N/A | pattern_detector.py:955 |

**Geometry Validation:**
- `validate_target_geometry_nb()` (lines 376-417) checks that target is ABOVE entry for bullish, BELOW for bearish
- If geometry invalid, falls back to `calculate_measured_move_nb()` (lines 420-461)
- Measured move uses 1.5x R:R multiplier (hardcoded)

### Origin

- [x] Discussed with user (structural levels)
- [x] Claude's choosing (1.5x fallback multiplier)
- [ ] Unknown

**Evidence:**
- Session 83K-4: "EXACT TARGET - Magnitude gives exact profit target" per STRAT
- Session 76 (docstrings): Geometry validation and 1.5x fallback was added to handle edge cases where structural target would be inverted (target below entry for bullish)

### Notes

**CONCERN:** The 1.5x R:R fallback multiplier was Claude's choosing. This affects patterns where structural target creates invalid geometry. Consider:
- Is 1.5x appropriate?
- Should these patterns be skipped instead of using fallback?

---

## Component 4: Time Exit

### Current Implementation

**Files:**
- `scripts/backtest_strat_equity_validation.py` (lines 107-118)

**Logic:**

```python
'max_holding_bars': {
    '1H': 60,   # 60 market hours (~10 trading days)
    '1D': 30,   # 30 days (~6 weeks)
    '1W': 20,   # 20 weeks (~5 months)
    '1M': 12,   # 12 months (1 year)
}
```

If pattern does not hit target or stop within max_holding_bars, position is exited at close price.

For hourly LIVE trading:
- `options_module.py:1461-1477` forces exit by 15:30 ET (TIME_EXIT)
- No overnight holds for hourly patterns

### Origin

- [ ] Discussed with user
- [x] Claude's choosing (pragmatic gap-fill)
- [ ] Unknown

**Evidence:** Session 83K-53 discovered this. User stated: "The 30-bar max_holding_bars window was set as a default on Nov 22, 2025 without explicit user discussion."

**Session 83K-53 Fix:** Changed from hardcoded 30 bars to timeframe-specific windows:
- 1H: Increased from 30 to 60 (30 was ~1.25 days - too short)
- 1D: Kept at 30
- 1W: Reduced to 20
- 1M: Reduced to 12

### Notes

**CRITICAL CONCERN:** These values are educated guesses based on bars-to-magnitude analysis, but:
1. Original 30-bar default was never discussed with user
2. New values (60/30/20/12) are based on backtest analysis, not live trading validation
3. The relationship between these bars and option DTE is not explicit

**RECOMMENDATION:** Validate in Phase 2 (SPEC) whether these align with user's trading intent.

---

## Component 5: DTE Selection

### Current Implementation

**Files:**
- `strat/options_module.py` (lines 219-228, 839-864)

**Logic:**

```python
default_dte_weekly: int = 35     # Weekly patterns
default_dte_monthly: int = 75    # Monthly patterns
default_dte_daily: int = 21      # Daily patterns
default_dte_hourly: int = 3      # Hourly patterns (CRITICAL ISSUE)
```

DTE calculation (`_calculate_expiration`, lines 809-864):
1. Start from signal timestamp
2. Add `target_dte` days
3. Find next Friday
4. Adjust for market holidays (e.g., Good Friday)

### Origin

- [ ] Discussed with user
- [x] Claude's choosing (pragmatic gap-fill)
- [ ] Unknown

**Evidence:**
- Session 83K-53 findings show hourly DTE = 3 days is TOO SHORT
- User recommended increasing to 7 days in `.session_startup_prompt.md`
- Weekly/Monthly DTEs appear reasonable but were not explicitly discussed

### Notes

**CRITICAL ISSUE - HOURLY DTE:**
- 3-day DTE for hourly means theta decay is SEVERE
- Session 83K-53 analysis showed hourly patterns have 36.2% win rate (vs 70-88% for other TFs)
- Likely cause: Options expire before pattern can hit magnitude

**DTE-to-Holding-Bars Relationship (Session 83K-53):**

| Timeframe | Max Holding Bars | Equivalent Days | Current DTE | Margin |
|-----------|-----------------|-----------------|-------------|--------|
| 1H | 60 bars | ~10 days | 3 days | NEGATIVE |
| 1D | 30 bars | ~6 weeks | 21 days | +3 weeks |
| 1W | 20 bars | ~5 months | 35 days | OK |
| 1M | 12 bars | ~1 year | 75 days | OK |

**RECOMMENDATION:** Hourly DTE should be at least 7 days to cover 60-bar holding window with margin for theta.

---

## Component 6: Position Sizing

### Current Implementation

**Files:**
- `strat/signal_automation/executor.py` (lines 107-138, 498-512)

**Logic:**

```python
# ExecutorConfig defaults
max_capital_per_trade: float = 300.0  # Max $ per trade
max_concurrent_positions: int = 5     # Max open positions

# Position size calculation (lines 498-512)
estimated_premium = underlying_price * 0.03  # Rough: 3% of underlying
max_contracts = int(max_capital_per_trade / (estimated_premium * 100))
return max(1, min(max_contracts, 5))  # 1-5 contracts
```

**Example Calculation:**
- SPY at $600
- Estimated premium: $600 * 0.03 = $18/share = $1,800/contract
- With $300 max: $300 / $1,800 = 0.16 contracts -> 1 contract
- Result: 1 contract (minimum)

### Origin

- [ ] Discussed with user
- [x] Claude's choosing (pragmatic gap-fill)
- [ ] Unknown

**Evidence:**
- Session 83K-45 introduced signal automation with $300 default capital
- User confirmed $3k account in Session 83K-42 discussion
- $300/trade (~10% of $3k) was Claude's pragmatic choice for risk management

### Notes

**CONCERNS:**

1. **Premium Estimation:** 3% of underlying is VERY rough:
   - ATM SPY options are ~$5-15/share depending on DTE/VIX
   - OTM options can be $1-5/share
   - This estimation could be 2-6x off

2. **Capital Calculation:** Does not consider:
   - Actual option price (just estimates)
   - Account buying power
   - Margin requirements

3. **No Dynamic Adjustment:** Position size is fixed regardless of:
   - Pattern quality/confidence
   - VIX level
   - Account performance

**RECOMMENDATION:** Use actual option prices from Alpaca API instead of 3% estimate.

---

## Component 7: Options Selection (Strike/Delta)

### Current Implementation

**Files:**
- `strat/options_module.py` (lines 672-807)

**Logic - Data-Driven Delta Targeting:**

```python
target_delta: float = 0.65           # Target delta
delta_range: Tuple[float, float] = (0.50, 0.80)  # Acceptable range
max_theta_pct: float = 0.30          # Max 30% theta cost
```

Algorithm (`_select_strike_data_driven`, lines 672-807):
1. Generate candidate strikes within entry-target range (expanded for ITM)
2. Calculate Greeks for each candidate using Black-Scholes
3. Filter: Keep only strikes with delta in 0.50-0.80 range
4. Score: 70% delta proximity to target + 30% theta cost acceptability
5. Select best scoring strike
6. Fallback to 0.3x geometric formula if no valid strikes

**Fallback Formula (`_fallback_to_geometric`, lines 631-670):**
```python
# For calls: strike = entry + 0.3 * (target - entry)
# For puts: strike = entry - 0.3 * (entry - target)
```

### Origin

- [x] Discussed with user (delta range)
- [x] Claude's choosing (scoring weights, theta threshold)
- [ ] Unknown

**Evidence:**
- Session 73: Delta targeting 0.40-0.55 first mentioned
- Later expanded to 0.50-0.80 for better delta capture
- The 70/30 scoring weights and 30% theta threshold were Claude's design choices

### Notes

**CONCERNS:**

1. **Delta Range Changed:** Original spec was 0.40-0.55, code uses 0.50-0.80
   - Higher delta = more ITM = higher premium but better delta P&L
   - Lower delta = more OTM = cheaper but more theta decay exposure

2. **Theta Threshold:** 30% was relaxed from original 10%
   - Session 78 comment: "relaxed from 10%"
   - This allows strikes with significant theta cost

3. **Scoring Weights:** 70% delta / 30% theta is arbitrary
   - No validation that this produces optimal outcomes
   - Different market conditions may warrant different weights

4. **Fallback Usage:** When does geometric fallback trigger?
   - If ALL candidates fail delta or theta filters
   - This means fallback strikes may have poor Greeks

---

## Summary of Origins

| Component | User Discussed | Claude's Choosing | Unknown |
|-----------|---------------|-------------------|---------|
| 1. Entry Logic | YES | - | - |
| 2. Stop Loss | YES | - | - |
| 3. Target/Magnitude | PARTIAL | 1.5x fallback | - |
| 4. Time Exit | - | YES | - |
| 5. DTE Selection | - | YES | - |
| 6. Position Sizing | - | YES | - |
| 7. Options Selection | PARTIAL | Scoring weights | - |

**CORE STRAT METHODOLOGY (1-3):** User-discussed and follows Rob Smith's STRAT methodology.

**OPERATIONAL PARAMETERS (4-7):** Mostly Claude's pragmatic choices that need validation.

---

## Recommended Phase 2 Questions

For the SPEC phase, the following questions should be resolved with user:

### Time Exit
1. What maximum holding period makes sense for each timeframe?
2. Should we use bars or calendar days?
3. What happens at time exit - market order, limit order, next open?

### DTE Selection
1. Should hourly DTE increase from 3 to 7+ days?
2. Should DTE be tied to max_holding_bars?
3. Any preference for weeklies vs monthlies for specific patterns?

### Position Sizing
1. What percentage of account per trade (currently ~10%)?
2. Should size vary by pattern quality, VIX, or other factors?
3. Max concurrent positions (currently 5)?

### Options Selection
1. Target delta: 0.50-0.80 acceptable or prefer narrower range?
2. How important is theta cost in strike selection?
3. Is the geometric fallback acceptable or should those trades be skipped?

---

## Audit Checklist

- [x] Entry Logic documented
- [x] Stop Loss documented
- [x] Target/Magnitude documented
- [x] Time Exit documented
- [x] DTE Selection documented
- [x] Position Sizing documented
- [x] Options Selection documented
- [x] Origin of each decision documented
- [x] Concerns/issues flagged
- [x] Phase 2 questions prepared

**Audit Complete:** December 7, 2025 (Session 83K-54)

---

*Next: Phase 2 (SPEC) - Define what system SHOULD do based on this audit + user input*
