# Timeframe Continuity (TFC) Full Integration Plan

**Created:** Session EQUITY-40
**Branch:** `claude/integrate-timeframe-continuity-1hWcj`
**Status:** Planning Complete - Ready for Implementation

---

## Executive Summary

This document outlines the complete plan to integrate Timeframe Continuity (TFC) into **position sizing**, **trade prioritization**, and **cross-timeframe pattern analysis** for both equity options and crypto perpetual futures trading.

### Key Finding: TFC is Calculated but Not Applied

The TFC score is currently calculated and stored in every signal, but **not used for**:
1. Position sizing (higher TFC = larger position)
2. Trade prioritization (when multiple signals compete)
3. Risk management (higher TFC = tighter stops or wider targets)
4. Execution filtering (minimum TFC threshold before execution)

---

## Current State Analysis

### What Exists

| Component | Equity | Crypto | TFC Integration |
|-----------|--------|--------|-----------------|
| TFC Checker | `strat/timeframe_continuity.py` | `crypto/data/state.py::get_continuity_score()` | Exists |
| Signal Scanner | `strat/paper_signal_scanner.py::get_tfc_score()` | `crypto/scanning/signal_scanner.py::get_tfc_score()` | Calculated |
| Signal Context | `SignalContext.tfc_score` | `CryptoSignalContext.tfc_score` | Stored |
| Discord Alerts | Shows TFC in alerts | Shows TFC in alerts | Displayed |
| Position Sizing | N/A | N/A | **NOT INTEGRATED** |
| Trade Prioritization | Timeframe-based only | Timeframe-based only | **NOT INTEGRATED** |
| Backtesting | Flexible continuity filter available | Uses MIN_CONTINUITY_SCORE | Partial |

### Key Files

#### Equity Architecture
- `strat/timeframe_continuity.py` - TimeframeContinuityChecker class (613 lines)
- `strat/unified_pattern_detector.py` - Single source of truth for pattern detection (542 lines)
- `strat/paper_signal_scanner.py` - Scanner with TFC calculation (1644 lines)
- `strat/signal_automation/daemon.py` - Main orchestrator
- `strat/signal_automation/signal_store.py` - Signal persistence
- `strat/signal_automation/executor.py` - Order execution
- `strategies/strat_options_strategy.py` - Backtesting strategy (875 lines)

#### Crypto Architecture
- `crypto/scanning/signal_scanner.py` - Pattern detection with TFC (1301 lines)
- `crypto/data/state.py` - State management with get_continuity_score()
- `crypto/config.py` - Configuration with MIN_CONTINUITY_SCORE
- `crypto/simulation/paper_trader.py` - Paper trading

---

## Phase 1: Unified TFC Module (Single Source of Truth)

### New File: `strat/tfc/unified_tfc.py`

```python
"""
Unified Timeframe Continuity (TFC) Module - Single Source of Truth

Used by:
- Equity paper trading and backtesting
- Crypto paper trading and backtesting
- Position sizing across both systems
- Trade prioritization in signal stores
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class TFCStrength(Enum):
    """TFC strength levels for position sizing."""
    FULL = 5      # 5/5 aligned - Maximum conviction
    STRONG = 4    # 4/5 aligned - High conviction
    MODERATE = 3  # 3/5 aligned - Medium conviction
    WEAK = 2      # 2/5 aligned - Low conviction
    NONE = 1      # 1/5 or less - No conviction


@dataclass
class TFCResult:
    """Unified TFC result used across equity and crypto."""
    score: int                           # Raw score (0-5 equity, 0-4 crypto)
    max_score: int                       # Maximum possible (5 equity, 4 crypto)
    strength: TFCStrength                # Categorical strength level
    aligned_timeframes: List[str]        # Which timeframes are aligned
    required_timeframes: List[str]       # Timeframes checked
    direction: str                       # 'bullish' or 'bearish'
    passes_flexible: bool                # True if meets flexible requirements

    @property
    def percentage(self) -> float:
        """TFC as percentage for position sizing."""
        return self.score / self.max_score if self.max_score > 0 else 0.0

    @property
    def position_multiplier(self) -> float:
        """
        Position size multiplier based on TFC.

        Returns:
            1.0 = Base position size
            1.25 = 25% larger for strong TFC
            1.5 = 50% larger for full TFC
            0.5 = 50% smaller for weak TFC
        """
        if self.strength == TFCStrength.FULL:
            return 1.5
        elif self.strength == TFCStrength.STRONG:
            return 1.25
        elif self.strength == TFCStrength.MODERATE:
            return 1.0
        elif self.strength == TFCStrength.WEAK:
            return 0.75
        else:
            return 0.5

    @property
    def priority_score(self) -> int:
        """Priority score for trade ordering (0-50 range)."""
        return self.score * 10


class UnifiedTFCChecker:
    """
    Unified timeframe continuity checker.
    Supports both equity (5 timeframes) and crypto (4 timeframes).
    """

    EQUITY_TIMEFRAMES = ['1M', '1W', '1D', '4H', '1H']
    CRYPTO_TIMEFRAMES = ['1w', '1d', '4h', '1h']

    EQUITY_FLEXIBLE_REQUIREMENTS = {
        '1H': ['1W', '1D', '1H'],
        '1D': ['1M', '1W', '1D'],
        '1W': ['1M', '1W'],
        '1M': ['1M'],
    }

    CRYPTO_FLEXIBLE_REQUIREMENTS = {
        '1h': ['1d', '4h', '1h'],
        '4h': ['1w', '1d', '4h'],
        '1d': ['1w', '1d'],
        '1w': ['1w'],
    }

    def __init__(self, asset_type: str = 'equity'):
        self.asset_type = asset_type
        if asset_type == 'equity':
            self.timeframes = self.EQUITY_TIMEFRAMES
            self.flexible_requirements = self.EQUITY_FLEXIBLE_REQUIREMENTS
            self.max_score = 5
        else:
            self.timeframes = self.CRYPTO_TIMEFRAMES
            self.flexible_requirements = self.CRYPTO_FLEXIBLE_REQUIREMENTS
            self.max_score = 4

    def check_continuity(
        self,
        bar_classifications: Dict[str, int],
        direction: int,
        detection_timeframe: Optional[str] = None
    ) -> TFCResult:
        """Check timeframe continuity from bar classifications."""
        target_class = 2 * direction
        direction_str = 'bullish' if direction > 0 else 'bearish'

        if detection_timeframe and detection_timeframe in self.flexible_requirements:
            required_tfs = self.flexible_requirements[detection_timeframe]
        else:
            required_tfs = self.timeframes

        aligned_tfs = []
        for tf in required_tfs:
            if tf in bar_classifications and bar_classifications[tf] == target_class:
                aligned_tfs.append(tf)

        score = len(aligned_tfs)

        if score >= self.max_score:
            strength = TFCStrength.FULL
        elif score >= self.max_score - 1:
            strength = TFCStrength.STRONG
        elif score >= self.max_score - 2:
            strength = TFCStrength.MODERATE
        elif score >= 2:
            strength = TFCStrength.WEAK
        else:
            strength = TFCStrength.NONE

        min_required = max(1, len(required_tfs) // 2)
        passes_flexible = score >= min_required

        return TFCResult(
            score=score,
            max_score=self.max_score,
            strength=strength,
            aligned_timeframes=aligned_tfs,
            required_timeframes=required_tfs,
            direction=direction_str,
            passes_flexible=passes_flexible,
        )
```

---

## Phase 2: TFC-Based Position Sizing

### New File: `strat/tfc/position_sizing.py`

```python
"""TFC-Aware Position Sizing"""

from dataclasses import dataclass
from typing import Optional
from strat.tfc.unified_tfc import TFCResult, TFCStrength


@dataclass
class PositionSizeConfig:
    """Configuration for TFC-aware position sizing."""
    base_risk_per_trade: float = 0.01  # 1% of capital
    max_risk_per_trade: float = 0.02   # 2% max

    full_tfc_multiplier: float = 1.5
    strong_tfc_multiplier: float = 1.25
    moderate_tfc_multiplier: float = 1.0
    weak_tfc_multiplier: float = 0.75
    no_tfc_multiplier: float = 0.5

    min_tfc_score: int = 2
    hourly_size_cap: float = 0.5

    def get_multiplier(self, tfc: TFCResult) -> float:
        if tfc.strength == TFCStrength.FULL:
            return self.full_tfc_multiplier
        elif tfc.strength == TFCStrength.STRONG:
            return self.strong_tfc_multiplier
        elif tfc.strength == TFCStrength.MODERATE:
            return self.moderate_tfc_multiplier
        elif tfc.strength == TFCStrength.WEAK:
            return self.weak_tfc_multiplier
        return self.no_tfc_multiplier


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    tfc: TFCResult,
    timeframe: str,
    config: Optional[PositionSizeConfig] = None
) -> dict:
    """Calculate position size with TFC adjustment."""
    config = config or PositionSizeConfig()

    if tfc.score < config.min_tfc_score:
        return {
            'shares': 0,
            'dollar_risk': 0,
            'risk_percent': 0,
            'tfc_multiplier': 0,
            'size_rationale': f'TFC {tfc.score}/{tfc.max_score} below min {config.min_tfc_score}'
        }

    per_share_risk = abs(entry_price - stop_price)
    if per_share_risk <= 0:
        return {
            'shares': 0,
            'dollar_risk': 0,
            'risk_percent': 0,
            'tfc_multiplier': 0,
            'size_rationale': 'Invalid risk (entry == stop)'
        }

    base_risk = account_equity * config.base_risk_per_trade
    tfc_multiplier = config.get_multiplier(tfc)
    adjusted_risk = base_risk * tfc_multiplier

    if timeframe.upper() in ['1H', '60MIN', '60M']:
        adjusted_risk = min(adjusted_risk, base_risk * config.hourly_size_cap)

    final_risk = min(adjusted_risk, account_equity * config.max_risk_per_trade)
    shares = int(final_risk / per_share_risk)

    return {
        'shares': shares,
        'dollar_risk': shares * per_share_risk,
        'risk_percent': (shares * per_share_risk) / account_equity if account_equity > 0 else 0,
        'tfc_multiplier': tfc_multiplier,
        'size_rationale': f'TFC {tfc.score}/{tfc.max_score} ({tfc.strength.name}) -> {tfc_multiplier:.2f}x'
    }
```

---

## Phase 3: TFC-Based Trade Prioritization

### New File: `strat/tfc/prioritization.py`

```python
"""TFC-Based Trade Prioritization"""

TIMEFRAME_WEIGHTS = {
    '1M': 100, '1W': 80, '1D': 60, '4H': 40, '1H': 20,
    '1w': 80, '1d': 60, '4h': 40, '1h': 20, '15m': 10,
}


def calculate_signal_priority(
    tfc_score: int,
    tfc_max: int,
    timeframe: str,
    risk_reward: float,
    magnitude_pct: float
) -> int:
    """
    Calculate composite priority score.

    Components:
    - TFC: 0-50 points
    - Timeframe: 0-100 points
    - R:R: 0-30 points
    - Magnitude: 0-20 points
    """
    tfc_points = int((tfc_score / tfc_max) * 50) if tfc_max > 0 else 0
    tf_points = TIMEFRAME_WEIGHTS.get(timeframe, 30)
    rr_points = int(min(risk_reward / 3.0, 1.0) * 30)

    if 1.0 <= magnitude_pct <= 5.0:
        mag_points = 20
    elif magnitude_pct < 1.0:
        mag_points = int(magnitude_pct * 10)
    else:
        mag_points = max(0, 20 - int((magnitude_pct - 5) * 2))

    return tfc_points + tf_points + rr_points + mag_points
```

---

## Phase 4: Integration Points

### 4.1 Signal Store (`strat/signal_automation/signal_store.py`)

Add `tfc_priority` field to `StoredSignal` and sort by it.

### 4.2 Executor (`strat/signal_automation/executor.py`)

Use `calculate_position_size()` instead of fixed position sizing.

### 4.3 Crypto State (`crypto/data/state.py`)

Replace `get_continuity_score()` with `UnifiedTFCChecker`.

### 4.4 Discord Alerts

Add TFC strength indicator (emoji) and priority score.

### 4.5 Backtesting (`strategies/strat_options_strategy.py`)

Add TFC filtering and position scaling options.

---

## strat_options_strategy.py Refactoring Suggestions

### 1. Remove Duplicate Pattern Detection

**Current:** Re-implements pattern detection
**Suggested:** Use `unified_pattern_detector.detect_all_patterns()`

### 2. Extract Continuation Bar Logic

**Current:** 50+ line method in strategy class
**Suggested:** Move to `strat/pattern_utils.py`

### 3. Add TFC Configuration

```python
@dataclass
class STRATOptionsConfig:
    # ... existing fields ...
    use_tfc_sizing: bool = True
    min_tfc_score: int = 2
    tfc_position_scaling: bool = True
```

### 4. Consolidate Time Filters

**Current:** Duplicated hourly time filter logic
**Suggested:** Create shared `strat/time_filters.py`

### 5. Simplify Sharpe Calculation

**Current:** Complex 40-line method
**Suggested:** Use pandas built-ins more effectively

---

## Implementation Roadmap

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Create unified TFC module | 1-2 days |
| 2 | TFC position sizing | 1 day |
| 3 | TFC prioritization | 0.5 days |
| 4 | Signal store integration | 0.5 days |
| 5 | Executor integration | 0.5 days |
| 6 | Crypto integration | 1 day |
| 7 | Discord enhancement | 0.5 days |
| 8 | Backtest integration | 1 day |
| 9 | Refactor strat_options_strategy | 1-2 days |
| 10 | Tests | 1 day |

**Total: 8-10 days**

---

## Testing Strategy

1. Unit tests for `UnifiedTFCChecker`
2. Unit tests for `calculate_position_size()`
3. Unit tests for `calculate_signal_priority()`
4. Integration tests for signal store ordering
5. Integration tests for executor position sizing
6. Backtest comparison: TFC-sized vs fixed-size
7. Paper trading validation

---

## Success Metrics

1. **Position Sizing Accuracy**: Full TFC positions 50% larger, weak TFC 50% smaller
2. **Priority Ordering**: Higher TFC signals execute first
3. **Win Rate by TFC**: Track win rate segmented by TFC score
4. **Backtest Performance**: Compare risk-adjusted returns with/without TFC sizing

---

## Related Sessions

- Session 55-57: Flexible continuity concept
- Session 65-66: 2D timeframe research
- EQUITY-36: Intraday target reduction
- EQUITY-38: Unified pattern detector
- EQUITY-39: Single source of truth for targets
- EQUITY-40: This integration plan
