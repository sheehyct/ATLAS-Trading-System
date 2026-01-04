# Claude Code Handoff: Timeframe Continuity Integration

**Session:** EQUITY-40
**Date:** 2025-12-31
**Purpose:** Enable parallel development of TFC integration across equity and crypto systems

---

## Git Information

### Branch to Review
```bash
git fetch origin claude/integrate-timeframe-continuity-1hWcj
git checkout claude/integrate-timeframe-continuity-1hWcj
```

### Related Documentation
- `docs/TFC_INTEGRATION_PLAN.md` - Full implementation plan with code examples
- `docs/HANDOFF.md` - Current system state (read first for context)
- `docs/CLAUDE.md` - Development rules and conventions

### Key Commits on This Branch
```
cc98551 docs: comprehensive TFC integration plan for position sizing and prioritization (EQUITY-40)
b2b6f48 docs: update session docs and add quality-momentum skeleton (EQUITY-40)
f27f9f4 feat(target-methodology): implement single source of truth for pattern targets (EQUITY-39)
b68491b feat(pattern-detector): add unified pattern detector for consistent backtest/paper trading (EQUITY-38)
```

---

## Problem Statement

Timeframe Continuity (TFC) scores are **calculated and stored** in every signal but **not used** for:
1. Position sizing
2. Trade prioritization
3. Execution filtering

This means a 5/5 TFC signal (full alignment) gets the same position size as a 1/5 signal (weak alignment), despite research showing higher TFC correlates with higher win rates.

---

## Current Architecture Summary

### Equity System (`strat/`)

| File | Purpose | TFC Status |
|------|---------|------------|
| `strat/timeframe_continuity.py` | `TimeframeContinuityChecker` class | ✅ Calculates TFC |
| `strat/paper_signal_scanner.py` | `get_tfc_score()` method | ✅ Stores in context |
| `strat/unified_pattern_detector.py` | Single source of truth for patterns | ❌ No TFC integration |
| `strat/signal_automation/signal_store.py` | Signal persistence | ❌ No TFC priority |
| `strat/signal_automation/executor.py` | Order execution | ❌ Fixed position sizing |
| `strategies/strat_options_strategy.py` | Backtesting | ❌ No TFC filtering |

### Crypto System (`crypto/`)

| File | Purpose | TFC Status |
|------|---------|------------|
| `crypto/data/state.py` | `get_continuity_score()` method | ✅ Calculates TFC |
| `crypto/scanning/signal_scanner.py` | `get_tfc_score()` method | ✅ Stores in context |
| `crypto/config.py` | `MIN_CONTINUITY_SCORE = 2` | ⚠️ Config exists but not enforced |
| `crypto/simulation/paper_trader.py` | Paper trading | ❌ Fixed position sizing |

### TFC Score Ranges
- **Equity:** 0-5 (timeframes: 1M, 1W, 1D, 4H, 1H)
- **Crypto:** 0-4 (timeframes: 1w, 1d, 4h, 1h)

---

## Proposed Changes

### Phase 1: Create Unified TFC Module

**New files to create:**
```
strat/tfc/__init__.py
strat/tfc/unified_tfc.py      # TFCResult, UnifiedTFCChecker
strat/tfc/position_sizing.py  # calculate_position_size()
strat/tfc/prioritization.py   # calculate_signal_priority()
```

**Key classes:**
- `TFCStrength` enum: FULL, STRONG, MODERATE, WEAK, NONE
- `TFCResult` dataclass: score, max_score, strength, position_multiplier, priority_score
- `UnifiedTFCChecker`: Supports both equity (5 TF) and crypto (4 TF)

### Phase 2: Position Sizing Integration

**Modify:** `strat/signal_automation/executor.py`

**Logic:**
```python
# Current: Fixed position size
shares = 100  # Always same

# Proposed: TFC-adjusted
tfc_multiplier = {
    TFCStrength.FULL: 1.5,      # 50% larger
    TFCStrength.STRONG: 1.25,   # 25% larger
    TFCStrength.MODERATE: 1.0,  # Base
    TFCStrength.WEAK: 0.75,     # 25% smaller
    TFCStrength.NONE: 0.5,      # 50% smaller
}
shares = base_shares * tfc_multiplier[tfc.strength]
```

### Phase 3: Trade Prioritization

**Modify:** `strat/signal_automation/signal_store.py`

**Add field:** `tfc_priority: int` to `StoredSignal`

**Priority formula:**
```python
priority = (tfc_score / tfc_max * 50)      # 0-50 points
         + timeframe_weight                 # 0-100 points (1M=100, 1H=20)
         + (risk_reward / 3 * 30)          # 0-30 points
         + magnitude_points                 # 0-20 points
```

### Phase 4: Crypto Integration

**Modify:** `crypto/data/state.py`, `crypto/scanning/signal_scanner.py`

Replace `get_continuity_score()` with `UnifiedTFCChecker(asset_type='crypto')`

### Phase 5: Backtest Integration

**Modify:** `strategies/strat_options_strategy.py`

**Add to config:**
```python
@dataclass
class STRATOptionsConfig:
    use_tfc_sizing: bool = True
    min_tfc_score: int = 2
    tfc_position_scaling: bool = True
```

---

## Refactoring Opportunities

### strat_options_strategy.py (875 lines)

1. **Lines 261-324:** Pattern detection duplicates `unified_pattern_detector.py` - should use unified detector
2. **Lines 456-508:** Continuation bar filter should be extracted to `strat/pattern_utils.py`
3. **Lines 326-364:** Hourly time filter duplicates `paper_signal_scanner.py` - create shared `strat/time_filters.py`
4. **Lines 747-805:** Sharpe calculation is complex - simplify with pandas built-ins

---

## File Dependencies for Implementation

### If implementing position sizing:
```
Read first:
- strat/signal_automation/executor.py
- strat/signal_automation/config.py

Create:
- strat/tfc/unified_tfc.py
- strat/tfc/position_sizing.py
```

### If implementing prioritization:
```
Read first:
- strat/signal_automation/signal_store.py
- strat/signal_automation/daemon.py

Create:
- strat/tfc/prioritization.py
```

### If implementing crypto integration:
```
Read first:
- crypto/data/state.py
- crypto/config.py
- crypto/scanning/signal_scanner.py
```

---

## Testing Requirements

1. Unit tests for `UnifiedTFCChecker` with equity and crypto timeframes
2. Unit tests for position size calculations at each TFC level
3. Integration tests verifying signal store ordering by priority
4. Backtest comparison: TFC-sized vs fixed-size positions

---

## Implementation Order Recommendation

For parallel development, these can be worked independently:

| Task | Dependencies | Can Parallelize With |
|------|--------------|---------------------|
| Create `unified_tfc.py` | None | - |
| Create `position_sizing.py` | `unified_tfc.py` | `prioritization.py` |
| Create `prioritization.py` | `unified_tfc.py` | `position_sizing.py` |
| Modify executor | `position_sizing.py` | Modify crypto |
| Modify signal_store | `prioritization.py` | Modify crypto |
| Modify crypto system | `unified_tfc.py` | Modify equity |
| Refactor strat_options_strategy | `unified_tfc.py` | Any |

---

## Code Examples

Full code examples with docstrings are in `docs/TFC_INTEGRATION_PLAN.md`.

Key function signatures:

```python
# strat/tfc/unified_tfc.py
class UnifiedTFCChecker:
    def check_continuity(
        self,
        bar_classifications: Dict[str, int],  # {'1D': 2, '1W': -2, ...}
        direction: int,                        # 1=bullish, -1=bearish
        detection_timeframe: Optional[str] = None
    ) -> TFCResult: ...

# strat/tfc/position_sizing.py
def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    tfc: TFCResult,
    timeframe: str,
    config: Optional[PositionSizeConfig] = None
) -> dict: ...  # Returns {shares, dollar_risk, risk_percent, tfc_multiplier, size_rationale}

# strat/tfc/prioritization.py
def calculate_signal_priority(
    tfc_score: int,
    tfc_max: int,
    timeframe: str,
    risk_reward: float,
    magnitude_pct: float
) -> int: ...  # Returns 0-200 priority score
```

---

## Questions for Review

1. Should minimum TFC threshold be configurable per timeframe? (e.g., require higher TFC for hourly)
2. Should TFC affect target/stop levels, or only position size?
3. Should crypto use the same multipliers as equity, or have separate config?
4. Should we add TFC to Discord alerts with visual indicators (emoji)?

---

## Contact

This handoff was created by Claude Code session EQUITY-40.
For questions about the existing codebase, refer to `docs/HANDOFF.md`.
