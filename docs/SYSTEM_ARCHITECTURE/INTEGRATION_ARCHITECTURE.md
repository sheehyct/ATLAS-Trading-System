# ATLAS-STRAT Integration Architecture

**Version:** 1.0
**Date:** 2025-11-10
**Status:** Design specification

---

## Overview

This document defines how ATLAS Layer 1 (regime detection) and STRAT Layer 2 (pattern recognition) interact, including three distinct deployment modes and the optional signal confluence framework.

**Critical principle:** STRAT and ATLAS are peer systems, not hierarchical. STRAT can operate independently or in conjunction with ATLAS based on trader preference and capital constraints.

---

## Three Deployment Modes

### Mode 1: Standalone ATLAS (Regime-Only Trading)

**Use case:** Trader wants broad regime-based position sizing without pattern-level precision.

**Operation:**
- ATLAS detects market regime (TREND_BULL, TREND_NEUTRAL, TREND_BEAR, CRASH)
- Existing strategies (ORB, mean reversion, pairs trading) use regime for filtering
- No STRAT pattern detection required
- Suitable for $10k+ capital (equity-focused strategies)

**Signal flow:**
```
Market Data → ATLAS Regime Detection → Strategy Filters → Position Sizing → Execution
```

**Advantages:**
- Simpler system (fewer moving parts)
- Lower computational requirements
- Proven academic validation (33 years historical testing)

**Disadvantages:**
- Coarser entry/exit timing
- Miss pattern-specific opportunities
- Regime changes lag actual price reversals

---

### Mode 2: Standalone STRAT (Pattern-Only Trading)

**Use case:** Trader prioritizes price action patterns without regime context.

**Operation:**
- STRAT detects bar patterns (3-1-2, 2-1-2, 2-2, Rev Strat)
- Entry/exit determined purely by governing range breaks
- Multi-timeframe continuity used for signal quality assessment
- Suitable for $3k+ capital (options-focused implementation)

**Signal flow:**
```
Market Data → STRAT Pattern Detection → Timeframe Continuity → Options Execution
```

**Advantages:**
- Capital efficient (27x leverage via options)
- Precise entry/exit levels (governing range methodology)
- Works in all market conditions (not regime-dependent)
- Lower capital requirement ($3k vs $10k)

**Disadvantages:**
- No macro regime awareness
- Patterns can fail during regime transitions
- Options theta decay during low-volatility inside bar sequences

---

### Mode 3: Integrated ATLAS+STRAT (Confluence Trading)

**Use case:** Trader wants highest-quality signals using both systems.

**Operation:**
- ATLAS provides regime context
- STRAT provides precise entry/exit patterns
- Signals rated by confluence level (high/medium/reject)
- Best for traders with sufficient capital to deploy both systems

**Signal flow:**
```
Market Data → ATLAS Regime Detection ┐
                                      ├→ Signal Quality Matrix → Position Sizing → Execution
Market Data → STRAT Pattern Detection ┘
```

**Advantages:**
- Highest signal quality (confluence reduces false signals)
- Regime awareness prevents counter-trend pattern trading
- Pattern precision improves regime-based entry timing

**Disadvantages:**
- Most complex implementation
- Requires both systems operational
- Potential for signal conflicts requiring resolution rules

---

## Signal Quality Matrix (Mode 3 Only)

When operating in integrated mode, signals are classified based on ATLAS-STRAT agreement:

### High Quality Signals (Trade Aggressively)

| ATLAS Regime | STRAT Pattern | Timeframe Continuity | Action |
|--------------|---------------|---------------------|---------|
| TREND_BULL | 3-1-2 Bullish | Control/Confirm (>67%) | Full size, hold for 2R+ targets |
| TREND_BULL | 2-2 Bullish Continuation | Confirm (>50%) | Full size, trail aggressively |
| TREND_BEAR | 3-1-2 Bearish | Control/Confirm (>67%) | Full size, hold for 2R+ targets |
| TREND_BEAR | 2-2 Bearish Continuation | Confirm (>50%) | Full size, trail aggressively |

**Rationale:** Both systems agree on directional bias, timeframes aligned, highest probability.

### Medium Quality Signals (Reduce Size 50%)

| ATLAS Regime | STRAT Pattern | Timeframe Continuity | Action |
|--------------|---------------|---------------------|---------|
| TREND_NEUTRAL | Any STRAT Pattern | Confirm (>50%) | Half size, take profit at 1R |
| TREND_BULL | 2-1-2 Bullish (Reversal) | Conflict (<50%) | Half size, tight stop |
| TREND_BEAR | 2-1-2 Bearish (Reversal) | Conflict (<50%) | Half size, tight stop |
| Any Regime | Any Pattern | Conflict | Half size, scale out at 1R |

**Rationale:** Mixed signals suggest lower probability, reduce risk exposure.

### Reject Signals (Do Not Trade)

| ATLAS Regime | STRAT Pattern | Reason |
|--------------|---------------|---------|
| CRASH | Any Bullish Pattern | Regime override: crash conditions invalidate bullish patterns |
| TREND_BULL | 2-1-2/3-1-2 Bearish | Counter-trend pattern vs regime (likely false breakdown) |
| TREND_BEAR | 2-1-2/3-1-2 Bullish | Counter-trend pattern vs regime (likely false breakout) |
| Any | Any Pattern with Change | Timeframe continuity shows reversal in progress |

**Rationale:** Conflicting signals between systems or within STRAT's timeframe analysis.

---

## ATLAS Veto Power (CRASH Regime)

**Special rule:** ATLAS CRASH regime has absolute veto power over all bullish signals.

**Logic:**
```python
if atlas_regime == "CRASH" and strat_pattern in ["3-1-2 Bullish", "2-1-2 Bullish", "2-2 Bullish"]:
    signal_action = "REJECT"
    reason = "CRASH regime invalidates bullish patterns"
```

**Historical validation:**
- March 2020: ATLAS detected 77% CRASH regime
- During this period, bullish STRAT patterns would have failed
- Veto power prevented losses from counter-trend trades

**Exception:** Rev Strat (reversal off lows) can override CRASH veto if:
- VIX drops >20% from peak (volatility collapsing)
- STRAT shows 3-timeframe bullish alignment (Control condition)
- Pattern forms after >10% market decline (oversold bounce)

---

## Layer Independence Principle

**Critical design decision:** STRAT is a peer to ATLAS, not subordinate.

**Implications:**
1. STRAT can trade without ATLAS regime signals
2. ATLAS can trade without STRAT patterns
3. Integration is optional, not mandatory
4. Each layer must be independently profitable

**Rationale:**
- Trader choice: Some traders prefer patterns, others prefer regimes
- Capital flexibility: $3k accounts can run STRAT only
- System robustness: If one layer fails, the other continues
- Development timeline: Layers developed and tested independently

**Code architecture:**
```python
# Bad: STRAT depends on ATLAS
if atlas.regime == "TREND_BULL":
    strat_signal = strat.detect_pattern()  # Wrong: STRAT needs ATLAS to function

# Good: STRAT operates independently
strat_signal = strat.detect_pattern()
atlas_regime = atlas.get_current_regime()

# Integration logic separate
if deployment_mode == "integrated":
    final_signal = quality_matrix.evaluate(strat_signal, atlas_regime)
else:
    final_signal = strat_signal  # STRAT standalone mode
```

---

## Mixed Deployment Strategy

**Practical scenario:** Trader has $13k total capital but wants to test ATLAS before full commitment.

**Proposed allocation:**
- $3k live capital: STRAT standalone (options)
- $10k paper capital: ATLAS (equity strategies)

**Benefits:**
1. Real-money validation of STRAT (lower capital risk)
2. Paper trading validation of ATLAS (no capital risk)
3. Data collection on both systems simultaneously
4. Gradual integration as confidence builds

**Integration path:**
```
Month 1-3: STRAT live ($3k) + ATLAS paper ($10k) → Independent operation
Month 4-6: Evaluate performance, tune parameters
Month 7+:   If both profitable, transition ATLAS to live capital
Month 10+:  Begin integrated mode (confluence trading) with proven systems
```

**Risk management:**
- STRAT losses capped at $3k (premium risk only)
- ATLAS validated with zero capital risk
- Integration deferred until both systems proven independently

---

## Implementation Considerations

### Data Synchronization

**Challenge:** ATLAS and STRAT may use different timeframes.

**Example:**
- ATLAS: Daily regime detection (downside deviation, Sortino ratio)
- STRAT: 15min pattern detection (bar classification)

**Solution:**
```python
# Align data to common timestamps
atlas_daily_regime = atlas.get_regime_at_timestamp(current_timestamp)
strat_15min_pattern = strat.get_pattern_at_timestamp(current_timestamp)

# Use ATLAS regime from start-of-day for entire day's STRAT patterns
if current_timestamp.time() < market_close:
    atlas_regime_for_day = atlas.get_regime_at_timestamp(current_timestamp.replace(hour=9, minute=30))
```

**Key point:** ATLAS regime typically updates once per day, STRAT patterns update every bar.

### Computational Efficiency

**ATLAS:**
- Features: Calculated once per day on daily close
- Optimization: Can be cached, low computational cost
- Regime: Inference runs in <50ms for 250 days of history

**STRAT:**
- Bar classification: Must run every bar (15min, 60min, daily)
- Pattern detection: Runs on every timeframe independently
- Continuity: Requires multi-timeframe alignment (3+ timeframes)

**Optimization:**
```python
# Cache bar types, only reclassify new bars
if new_bar_received:
    bar_types[-1] = classify_bar_nb(high[-2:], low[-2:])  # Only latest bar
    patterns[-1] = detect_pattern_nb(bar_types[-3:])     # Only latest pattern

# Don't recompute entire history every bar
```

### Signal Conflicts

**Scenario:** ATLAS says TREND_BEAR, STRAT shows 3-1-2 bullish.

**Resolution hierarchy (integrated mode):**
1. Check signal quality matrix → Likely "REJECT" (counter-trend)
2. If CRASH regime → Automatic reject (ATLAS veto power)
3. If TREND_NEUTRAL → Downgrade to "MEDIUM" quality
4. If timeframe continuity is "Control" (all bullish) → Override to "MEDIUM" (pattern strength high)

**Code:**
```python
def resolve_signal_conflict(atlas_regime, strat_pattern, continuity_score):
    # CRASH veto
    if atlas_regime == "CRASH" and strat_pattern > 0:  # Bullish pattern
        return "REJECT", "CRASH regime veto"

    # Counter-trend check
    if (atlas_regime == "TREND_BEAR" and strat_pattern > 0) or \
       (atlas_regime == "TREND_BULL" and strat_pattern < 0):
        if continuity_score >= 0.9:  # Control (all timeframes aligned)
            return "MEDIUM", "Strong pattern overrides regime"
        else:
            return "REJECT", "Counter-trend to regime"

    # Neutral regime
    if atlas_regime == "TREND_NEUTRAL":
        return "MEDIUM", "No regime bias, pattern-dependent"

    # Aligned signals
    return "HIGH", "Regime and pattern agree"
```

---

## Testing Strategy

### Unit Tests (Per Layer)

**ATLAS tests** (regime/tests/test_academic_validation.py):
- Feature calculation accuracy
- Regime detection on synthetic data
- March 2020 crash detection
- Cross-validation parameter selection

**STRAT tests** (tests/test_strat/ - to be created):
- Bar classification for all 4 types
- Pattern detection for all 6 patterns
- Entry/exit price calculations
- Timeframe continuity scoring

### Integration Tests (Mode 3 Only)

**Signal quality matrix tests:**
```python
def test_high_quality_signal():
    """Test HIGH quality: TREND_BULL + 3-1-2 Bullish + Control continuity"""
    atlas_regime = "TREND_BULL"
    strat_pattern = 312  # 3-1-2 bullish
    continuity = 1.0     # Control (all timeframes aligned)

    quality = quality_matrix.evaluate(atlas_regime, strat_pattern, continuity)
    assert quality == "HIGH"
    assert position_size(quality) == 1.0  # Full size

def test_crash_veto():
    """Test CRASH regime vetoes bullish patterns"""
    atlas_regime = "CRASH"
    strat_pattern = 212  # 2-1-2 bullish
    continuity = 0.8     # Confirm

    quality = quality_matrix.evaluate(atlas_regime, strat_pattern, continuity)
    assert quality == "REJECT"
    assert position_size(quality) == 0.0  # No trade
```

### Walk-Forward Validation

**Test all three modes independently:**
1. Standalone ATLAS: 8-year rolling window, regime-based strategy filtering
2. Standalone STRAT: 2-year validation on SPY 15min patterns
3. Integrated mode: 2-year validation with quality matrix

**Performance comparison:**
```
Expected results (hypothetical):
- Standalone ATLAS: Sharpe 1.2, Max DD 18%
- Standalone STRAT: Sharpe 1.5, Max DD 22%
- Integrated mode: Sharpe 1.8, Max DD 15%  (confluence improves both metrics)
```

**If integrated mode underperforms:**
- Signal quality matrix is incorrectly calibrated
- Layer conflicts causing missed opportunities
- Fall back to best standalone system

---

## Deployment Recommendations

### For $3k Capital
**Recommended:** Mode 2 (Standalone STRAT)
- Options provide capital efficiency
- Pattern detection works without regime context
- Lower complexity, faster iteration

### For $10k Capital
**Recommended:** Mode 1 (Standalone ATLAS) or Mode 2 (Standalone STRAT)
- ATLAS: Equity strategies with regime filtering
- STRAT: Options strategies with pattern precision
- Choose based on trader preference (regimes vs patterns)

### For $20k+ Capital
**Recommended:** Mode 3 (Integrated ATLAS+STRAT)
- Sufficient capital to deploy both systems
- Confluence trading for highest signal quality
- Diversification across regime-based and pattern-based approaches

### Mixed Deployment (Any Capital Level)
**Recommended for validation phase:**
- Live STRAT (smallest viable capital, e.g., $3k)
- Paper ATLAS (simulated $10k)
- Proves both systems independently before integration

---

## Future Enhancements

### Layer 4 Integration

When credit spread monitoring (Layer 4) is implemented:

**Additional veto power:**
```python
if credit_spread_regime == "CRISIS" and atlas_regime != "CRASH":
    # Credit markets see stress before equities
    override_regime = "CRASH"
    reject_all_long_signals = True
```

**Three-way confluence:**
- ATLAS regime + STRAT pattern + Credit spreads
- Highest quality: All three agree
- Medium quality: Two agree
- Reject: Majority conflict or any CRISIS/CRASH signal

### Machine Learning Signal Weighting

**Future consideration:** Use historical performance to weight signals dynamically.

**Example:**
```python
# Learn optimal weights from backtest
weight_atlas = 0.6  # ATLAS correct 60% of time historically
weight_strat = 0.7  # STRAT correct 70% of time
weight_credit = 0.8  # Credit spreads correct 80% (best leading indicator)

confidence_score = (
    weight_atlas * atlas_signal +
    weight_strat * strat_signal +
    weight_credit * credit_signal
) / (weight_atlas + weight_strat + weight_credit)

if confidence_score > 0.7:
    quality = "HIGH"
```

**Caveat:** Adds complexity, deferred until all three layers proven independently.

---

## Summary

**Three deployment modes:**
1. Standalone ATLAS (regime-only)
2. Standalone STRAT (pattern-only)
3. Integrated ATLAS+STRAT (confluence)

**Key principles:**
- STRAT and ATLAS are peers, not hierarchical
- Each layer must be independently profitable
- Integration is optional based on capital and preference
- CRASH regime has veto power over bullish signals

**Recommended path:**
1. Validate ATLAS Layer 1 (complete)
2. Implement STRAT Layer 2 (in progress)
3. Test both standalone (Mode 1 and Mode 2)
4. Only after both proven, implement Mode 3 integration
5. Defer Layer 4 until Layers 1-2 validated in live trading
