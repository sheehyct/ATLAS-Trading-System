# ATLAS Trading System - Implementation Plan (Session 74)

**Date:** 2025-11-25
**Branch:** claude/verify-repository-01A9Cm8YvZBytpMcojWQeo8u
**Author:** Claude Code Analysis Session
**Status:** Ready for Implementation

---

## Executive Summary

This document consolidates all findings from the Session 74 analysis session, providing a prioritized implementation plan that can be executed by any Claude instance (Web or Desktop).

**Session 74 Deliverables:**
1. Bug scan of `strat/options_module.py` and `strat/greeks.py`
2. UI/Alert System architecture specification
3. Strategy skeleton code for 3 missing strategies

---

## Current System State

### What's Working (141 Tests Passing)
- STRAT pattern detection (2-1-2, 3-1-2, 2-2 patterns)
- Bar classification system
- Timeframe continuity checking
- Options module with data-driven strike selection (94.3% delta accuracy)
- Black-Scholes Greeks calculation
- 52-Week High Momentum strategy (validated)
- Opening Range Breakout strategy (needs modification)

### What Was Created This Session
| File | Type | Status |
|------|------|--------|
| `docs/SYSTEM_ARCHITECTURE/UI_ALERT_SYSTEM_SPECIFICATION.md` | Design Doc | Complete |
| `strategies/quality_momentum.py` | Skeleton | Ready for Implementation |
| `strategies/semi_vol_momentum.py` | Skeleton | Ready for Implementation |
| `strategies/ibs_mean_reversion.py` | Skeleton | Ready for Implementation |
| `strategies/__init__.py` | Updated | Complete |
| `docs/IMPLEMENTATION_PLAN_SESSION_74.md` | This Doc | Complete |

---

## Priority 1: Bug Fixes (HIGH)

### 1.1 Timezone Handling in DTE Calculations

**Severity:** HIGH
**Files:** `strat/options_module.py` (lines 223-231, 886-902)
**Issue:** Timezone stripping without conversion can cause incorrect DTE calculations

**Current Code (line 223-231):**
```python
signal_dt = signal.timestamp
if hasattr(signal_dt, 'tzinfo') and signal_dt.tzinfo is not None:
    signal_dt = signal_dt.replace(tzinfo=None)  # BUG: Strips without converting
```

**Recommended Fix:**
```python
import pytz

signal_dt = signal.timestamp
if hasattr(signal_dt, 'tzinfo') and signal_dt.tzinfo is not None:
    # Convert to Eastern Time before stripping (market time)
    et = pytz.timezone('America/New_York')
    signal_dt = signal_dt.astimezone(et).replace(tzinfo=None)
```

**Same fix needed at line 886-902** (backtester DTE calculation)

**Test Required:** Create test with UTC timestamps, verify DTE matches expected ET-based calculation

---

### 1.2 Entry Equals Target Edge Case

**Severity:** HIGH
**File:** `strat/options_module.py` (line 509)
**Issue:** If `entry == target`, `expected_move = 0`, causing empty candidate list

**Current Code:**
```python
expected_move = abs(target - entry)
itm_expansion = expected_move * 1.0  # Becomes 0 if entry == target
```

**Recommended Fix:**
```python
expected_move = abs(target - entry)
if expected_move < 0.01:  # Minimum $0.01 move
    logger.warning(f"Entry equals target for {signal.symbol}, using fallback")
    return self._fallback_to_geometric(signal, underlying_price, option_type)
itm_expansion = expected_move * 1.0
```

**Test Required:** Unit test with `entry_price == target_price` scenario

---

### 1.3 Column Validation in Backtester

**Severity:** MEDIUM
**File:** `strat/options_module.py` (lines 830-832)
**Issue:** KeyError if neither 'high'/'High' column exists

**Recommended Fix (add at start of `backtest_trades()`):**
```python
# Validate required columns
required_cols_lower = ['high', 'low', 'close']
required_cols_upper = ['High', 'Low', 'Close']

for col in required_cols_lower:
    if col not in price_data.columns and col.capitalize() not in price_data.columns:
        raise ValueError(f"price_data missing '{col}' or '{col.capitalize()}' column")
```

---

## Priority 2: UI/Alert System Implementation (MEDIUM)

**Specification:** `docs/SYSTEM_ARCHITECTURE/UI_ALERT_SYSTEM_SPECIFICATION.md`

### Phase 2.1: Core Infrastructure (1 session)
```
alerts/
├── __init__.py
├── alert_engine.py          # AlertEngine class
├── portfolio_monitor.py     # PortfolioMonitor class
└── channels/
    ├── __init__.py
    ├── base.py              # NotificationChannel ABC
    └── console.py           # ConsoleChannel
```

**Tasks:**
- [ ] Create `alerts/` directory structure
- [ ] Implement `Alert`, `AlertRule`, `AlertType`, `AlertSeverity` dataclasses
- [ ] Implement `AlertEngine` with rule registration and processing
- [ ] Implement `ConsoleChannel` with color-coded output
- [ ] Add basic unit tests

### Phase 2.2: Notification Channels (1 session)
```
alerts/channels/
├── email.py                 # EmailChannel (SMTP)
├── push.py                  # PushoverChannel
└── webhook.py               # WebhookChannel (Discord/Slack)
```

**Tasks:**
- [ ] Implement `EmailChannel` with HTML formatting
- [ ] Implement `PushoverChannel` for mobile alerts
- [ ] Implement `WebhookChannel` (Discord/Slack formats)
- [ ] Add `.env` configuration variables
- [ ] Update `.env.template`

### Phase 2.3: Integration (1 session)
**Tasks:**
- [ ] Implement `PortfolioMonitor` class
- [ ] Implement `PatternAlertIntegration` class
- [ ] Create `config/alert_settings.py` factory
- [ ] Create `scripts/monitor_portfolio.py` CLI
- [ ] Integration tests with OptionsExecutor
- [ ] Documentation updates

---

## Priority 3: Strategy Implementation (MEDIUM)

### 3.1 Quality-Momentum (Phase 1 Strategy)

**File:** `strategies/quality_momentum.py` (SKELETON created)
**Priority:** Phase 1 (foundation strategy)

**TODO Items:**
1. Integrate fundamental data source (options):
   - Alpha Vantage (limited free tier)
   - Financial Modeling Prep API
   - Yahoo Finance (yfinance)
   - Tiingo fundamentals
2. Implement quality score calculation:
   - ROE rank (40%)
   - Earnings quality / accruals (30%)
   - Inverse leverage (30%)
3. Implement quarterly rebalance logic
4. Add multi-stock portfolio support
5. Write unit tests

**Requires Local Testing:** Yes (fundamental data API keys)

### 3.2 Semi-Volatility Momentum (Phase 2 Strategy)

**File:** `strategies/semi_vol_momentum.py` (SKELETON created)
**Priority:** Phase 2

**TODO Items:**
1. Validate volatility calculation against academic paper
2. Test scaling mechanism (0.5x to 2.0x)
3. Test circuit breaker (22% vol threshold)
4. Backtest against SPY 2010-2024
5. Compare Sharpe improvement (target: 0.8 -> 1.6)

**Requires Local Testing:** Yes (backtesting)

### 3.3 IBS Mean Reversion (Phase 2 Strategy)

**File:** `strategies/ibs_mean_reversion.py` (SKELETON created)
**Priority:** Phase 2

**TODO Items:**
1. Implement time-based exit (3-day max hold)
   - Option A: VBT custom callback
   - Option B: Post-process trades
   - Option C: Stateful signal generation
2. Test IBS calculation edge cases (doji bars)
3. Validate 65-75% win rate target
4. Test volume confirmation filter
5. Backtest in choppy market periods

**Requires Local Testing:** Yes (backtesting)

---

## Priority 4: Low-Severity Fixes (LOW)

### 4.1 Add Logging for Default IV Fallback

**File:** `strat/options_module.py` (line 284)

```python
if close_col not in price_data.columns:
    logger.warning(f"No 'close'/'Close' column found. Using default IV 0.20")
    return 0.20
```

### 4.2 Make Risk-Free Rate Configurable

**Files:** `strat/options_module.py:532`, `strat/greeks.py:486`
**Current:** Hardcoded `r = 0.05`

**Recommendation:** Add to config or fetch from data source

### 4.3 Improve Position Sizing Accuracy

**File:** `strat/options_module.py` (line 265-266)
**Current:** Uses rough 5% estimate for option premium

**Recommendation:** Use calculated `greeks.option_price` for more accurate sizing

---

## Implementation Order

### Recommended Sequence

```
Session A (Bug Fixes):
├── 1.1 Timezone handling (2 locations)
├── 1.2 Entry==Target guard
├── 1.3 Column validation
└── Run existing 141 tests to verify no regressions

Session B (UI/Alert Core):
├── 2.1 Core infrastructure
├── AlertEngine, ConsoleChannel
└── Basic tests

Session C (UI/Alert Channels):
├── 2.2 Email, Push, Webhook channels
├── Configuration
└── Integration tests

Session D (Strategy - Semi-Vol):
├── 3.2 Semi-Vol Momentum implementation
├── Volatility scaling tests
└── Backtest validation

Session E (Strategy - IBS):
├── 3.3 IBS Mean Reversion implementation
├── Time-based exit mechanism
└── Backtest validation

Session F (Strategy - Quality-Momentum):
├── 3.1 Quality-Momentum implementation
├── Fundamental data integration
└── Multi-stock portfolio tests
```

---

## Files Modified This Session

```
docs/SYSTEM_ARCHITECTURE/UI_ALERT_SYSTEM_SPECIFICATION.md  [NEW]
docs/IMPLEMENTATION_PLAN_SESSION_74.md                      [NEW]
strategies/quality_momentum.py                              [NEW]
strategies/semi_vol_momentum.py                             [NEW]
strategies/ibs_mean_reversion.py                            [NEW]
strategies/__init__.py                                      [MODIFIED]
```

---

## Handoff Checklist

For the implementing Claude instance:

### Before Starting
- [ ] Verify on correct branch: `claude/verify-repository-01A9Cm8YvZBytpMcojWQeo8u`
- [ ] Run `pytest` to confirm 141 tests passing
- [ ] Read `docs/CLAUDE.md` for coding conventions
- [ ] Read `docs/HANDOFF.md` for session context

### During Implementation
- [ ] Follow existing code patterns in `strat/` directory
- [ ] Use type hints (Python 3.10+ style)
- [ ] Add docstrings to all public methods
- [ ] Run tests after each significant change
- [ ] Commit incrementally with descriptive messages

### After Completion
- [ ] Run full test suite: `pytest`
- [ ] Update `docs/HANDOFF.md` with session summary
- [ ] Push changes to branch
- [ ] Report any new issues discovered

---

## Test Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_options_module.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=strat --cov=strategies

# Run only fast tests (skip slow/integration)
pytest -m "not slow"
```

---

## Environment Notes

- Python 3.13.7 with UV package manager
- VectorBT Pro 2025.10.15
- API keys in `.env` (gitignored)
- All tests should pass without API keys (mocked)

---

## Questions for Next Session

If any ambiguity:

1. **Timezone handling:** Should all datetime objects be converted to ET, or should we store in UTC and convert at display time?

2. **Alert channels:** Should email alerts be batched (digest mode) or sent immediately?

3. **Quality-Momentum:** Which fundamental data source is preferred? (Tiingo already integrated for price data)

4. **Time-based exits:** Which VBT approach is preferred for IBS max_hold_days exit?

---

## Summary

This session completed:
1. **Bug Scan:** Found 2 HIGH, 2 MEDIUM, 5 LOW severity issues in options module
2. **UI/Alert Design:** Created comprehensive 3-phase implementation spec
3. **Strategy Skeletons:** Created 3 new strategy files ready for implementation

**Total estimated implementation time:** 5-6 sessions
**Recommended start:** Bug fixes (Priority 1) - quick wins with high impact
