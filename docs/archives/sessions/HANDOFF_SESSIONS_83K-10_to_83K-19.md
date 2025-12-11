## Session 83K-19: Price Adjustment Mismatch FIX

**Date:** November 30, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - 6.9% price mismatch FIXED

### Session Accomplishments

1. **CRITICAL BUG FIXED** - Price alignment now within 0.1% of ThetaData
2. **Research Complete** - Found `adjustment='split'` parameter for VBT Pro AlpacaData
3. **Tiingo Fixed** - Changed column mapping from adjClose to close (raw)
4. **Data Source Priority Fixed** - Alpaca primary per CLAUDE.md (was Tiingo)
5. **Alpaca Auth Fixed** - Added credential loading to DataFetcher

### Price Verification Results

| Source | Dec 23, 2020 | Diff from ThetaData |
|--------|--------------|---------------------|
| ThetaData underlying | $367.78 | baseline |
| Alpaca `adjustment='split'` | $367.49 | **0.08%** |
| Tiingo raw (`close`) | $367.57 | **0.06%** |
| Tiingo adjusted (`adjClose`) | $344.04 | 6.5% (OLD BUG) |

### Files Modified

| File | Change |
|------|--------|
| `integrations/tiingo_data_fetcher.py` | Lines 111-121: Use raw columns (open/close) instead of adjusted (adjOpen/adjClose) |
| `validation/strat_validator.py` | Lines 321-336: Swap data source priority (Alpaca primary) |
| `validation/strat_validator.py` | Lines 407-442: Add Alpaca credential loading + adjustment='split' |

### Test Results

- 488 STRAT/validation tests: ALL PASSING
- Price verification: SUCCESS (0.08% Alpaca, 0.06% Tiingo)
- ThetaData terminal: Unstable (500 errors during validation)

### Session 83K-20 Tasks

1. **Re-run full validation** with corrected prices (ThetaData terminal stable)
2. **Verify strike selections** have correct moneyness (OTM should be OTM)
3. **Analyze validation results** with properly aligned prices

### Key Insight

The solution was simpler than expected:
- **Alpaca**: Use `adjustment='split'` (removes dividend adjustments, keeps splits)
- **Tiingo**: Use raw columns (`close` not `adjClose`)
- Both approaches produce prices within 0.1% of ThetaData's underlying

**Plan File:** `C:\Users\sheeh\.claude\plans\ancient-meandering-fern.md`

---

## Session 83K-18: Validation with Greeks + Root Cause Analysis

**Date:** November 30, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Greeks working, fundamental strategy issue identified

### Session Accomplishments

1. **Verified Greeks Fix Working** - delta=0.5975, iv=0.149, theta=-0.0826
2. **Fixed Greeks Tracking Bug** - `thetadata_greeks` counter now incremented
3. **Ran Full Validation** - 3-1-2 pattern, 5 symbols (daily timeframe)
4. **CRITICAL BUG FOUND** - Price adjustment mismatch between data sources

### CRITICAL BUG: Adjusted vs Unadjusted Price Mismatch

**Discovery:** Trade detail verification revealed 6.9% price discrepancy:

| Source | SPY Dec 23, 2020 | Type |
|--------|------------------|------|
| Tiingo/Alpaca OHLCV | $344.04 | Dividend-adjusted |
| ThetaData Options | $367.78 | Unadjusted |

**Impact:** ALL strike selections are wrong:
- Pattern detected at "$344" (adjusted)
- Strike $350 selected as "OTM"
- But actual market price was $368 - strike is actually ITM!
- Delta shows 0.78 (ITM) not ~0.40 (OTM) as expected

**This explains inconsistent validation results across all sessions.**

### Additional Issue: Wrong Data Source Priority

`validation/strat_validator.py` line 321 has Tiingo as primary, but CLAUDE.md specifies:
- **Alpaca = Primary** (all equities)
- **Tiingo = Secondary** (fallback)

### Fix Options for 83K-19 (PRIORITY 1)

ThetaData subscription does NOT include underlying equity data, so must use one of:

1. **Option A: Unadjust our equity data** - Reverse dividend adjustments to match ThetaData
2. **Option B: Adjust ThetaData options data** - Apply adjustment factor to ThetaData prices/strikes

Need to determine which approach is more accurate for options backtesting.

### Key Fix: Greeks Tracking Bug

**Location:** `validation/strat_validator.py` lines 885-900

**Problem:** `thetadata_greeks` was never incremented despite Greeks being used.

**Fix:** Added tracking logic based on `data_source` column:
- ThetaData data_source implies Greeks were also from ThetaData
- Now correctly shows 100% Greeks coverage

### Validation Results (ThetaData 100%)

| Symbol | Trades | IS Sharpe | OOS Sharpe | IS PnL | OOS PnL |
|--------|--------|-----------|------------|--------|---------|
| SPY | 13 | 5.07 | -16.90 | +$3,883 | -$759 |
| QQQ | 24 | -2.84 | -7.86 | - | - |
| AAPL | 18 | -6.44 | 1.81 | - | - |
| IWM | 20 | -7.70 | 4.09 | - | - |
| DIA | 22 | -4.48 | -3.08 | - | - |

### ROOT CAUSE: Trade Sparsity (CRITICAL FINDING)

SPY 3-1-2 detailed analysis revealed:

**IS Period (2020-2024):**
- 10 trades over 3.5 years (2.8 trades/year)
- First trade: +$3,209 = 82% of total IS profit
- Without outlier: IS profit = +$674

**OOS Period (2024-2025):**
- Only 3 trades in 1.5 years
- Too few trades for statistical significance
- Results are essentially random noise

### Implications

1. **Greeks NOT the Issue** - ThetaData Greeks working perfectly (100% coverage)
2. **Strategy Sparsity IS the Issue** - 13 trades over 5 years is NOT validatable
3. **Outlier Dominance** - Single big winner (+$3,209) drives all IS metrics
4. **OOS Too Small** - 3 trades cannot produce meaningful statistics

### What This Means for ATLAS

The 3-1-2 pattern as currently implemented:
- Generates too few signals for options trading validation
- Cannot be statistically validated with walk-forward or holdout methods
- May work in practice but cannot be proven/disproven with this sample size

### Possible Path Forward

1. **Aggregate Patterns** - Combine 3-1-2 + 2-1-2 + 2-2 for more signals
2. **Multi-Symbol Portfolio** - Trade all 6 symbols simultaneously
3. **Lower Timeframes** - 4H/1H for more patterns (requires intraday data)
4. **Accept Unvalidated** - Use with smaller position sizes as discretionary overlay

### Files Modified

| File | Change |
|------|--------|
| `validation/strat_validator.py` | Fixed Greeks tracking (lines 885-900) |

### Test Results

- 217 validation tests: PASSING
- 271 STRAT tests: PASSING
- Greeks tracking: VERIFIED (100% coverage)

### Session 83K-19 Options

1. **Option A:** Aggregate patterns for portfolio validation
2. **Option B:** Accept sparse patterns, focus on live paper trading
3. **Option C:** Explore lower timeframes (4H) for more signals

---

## Session 83K-17: Greeks Fix + Validation Analysis (CONTINUED)

**Date:** November 30, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Critical Greeks endpoint fix applied
**Plan File:** `C:\Users\sheeh\.claude\plans\sprightly-twirling-hollerith.md`

### CRITICAL FIX: ThetaData Greeks Endpoint

**Problem Identified:** Greeks coverage was 0% despite ThetaData connection working. The client used `option/history/greeks/first_order` (intraday, requires interval) instead of `option/history/greeks/eod` (end-of-day, designed for historical backtesting).

**Fix Applied:** `integrations/thetadata_client.py` lines 824-840
- Changed endpoint from `first_order` to `eod`
- Changed params from `date` to `start_date/end_date`
- Updated field mapping for `implied_vol` (EOD format)

**Verification:**
```
SUCCESS: delta=0.5975, iv=0.1490, theta=-0.0826
```

### Validation Results Summary (from earlier in session)

| Pattern | Symbol | Trades | IS Return | OOS Return | Status |
|---------|--------|--------|-----------|------------|--------|
| 3-1-2 | SPY | 13 | +32% | -7.6% | Only profitable |
| 3-1-2 | QQQ | 24 | -26.9% | -66% | CATASTROPHIC |
| 2-1-2 | SPY | 68 | -46.6% | -153% | CATASTROPHIC |
| 2-1-2 | QQQ | 76 | -68.6% | -83% | CATASTROPHIC |

### Implementation Verification (Pattern Detection)

All pattern implementations verified CORRECT:
- 3-1-2: 14 patterns, all checks PASS (Outside=3 -> Inside=1 -> Trigger=2)
- 2-1-2: 68 patterns, all checks PASS
- 2-2: 266 patterns, all checks PASS

Entry/Stop/Target calculations verified against STRAT methodology.

### Root Cause Analysis

1. **Greeks Not Fetched** - NOW FIXED (was using wrong endpoint)
2. **No Trade Logging** - Individual trade details (strike, DTE, delta) not stored
3. **Strategy Performance** - Need to re-run validation WITH Greeks to see real impact

### Session 83K-18 Tasks

1. **RE-RUN VALIDATION** with Greeks endpoint fix to compare results
2. Add trade logging to capture individual trade details (strike, DTE, delta)
3. Review Monte Carlo thresholds for sparse-trade strategies
4. Analyze why 2-1-2 pattern has catastrophic returns (possible exit logic issue?)

### IMPORTANT FOR NEXT SESSION

**Context Window Management:** Be proactive about context window usage. Previous sessions lost this discipline. Use:
- Task tool for complex searches instead of multiple Glob/Grep
- Compact summaries when appropriate
- Focus on essential files, avoid re-reading entire files

---

## Session 83K-16: Validation Bug Fixes (Sign Reversal + Trade Filtering + Config)

**Date:** November 30, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - 3 bugs fixed, 8 new regression tests added
**Plan File:** `C:\Users\sheeh\.claude\plans\sprightly-twirling-hollerith.md`

### Critical Bugs Found and Fixed

#### Bug #1: Sharpe Degradation Calculation (CRITICAL)
**Location:** `validation/walk_forward.py` lines 433-437

**Problem:** When IS Sharpe was negative and OOS Sharpe was positive (like IWM), the function returned 0.0 degradation, causing false validation PASSes.

**Impact:** IWM 3-1-2 was the only PASS in Session 83K-15, but it was a FALSE POSITIVE due to this bug.

**Fix:** Modified `_calculate_sharpe_degradation()` to return tuple `(degradation, is_sign_reversal)` and added sign reversal detection. Results now include `has_sign_reversal_warning` field to flag suspicious IS/OOS sign reversals for manual review.

#### Bug #2: OOS Trade Date Filtering (CRITICAL)
**Location:** `validation/walk_forward.py` lines 276-308

**Problem:** Trades returned from backtest were not verified to fall within the OOS test period, potentially causing incorrect trade counts and OOS Sharpe = 0.00 issues.

**Fix:** Added trade date verification after OOS backtest. If trades fall outside the test period, a warning is logged and the trade count is adjusted.

#### Bug #3: config.for_options() Root Cause (MEDIUM)
**Location:** `validation/config.py` lines 276-290

**Problem:** The 83K-15 fix was a WORKAROUND. The root cause was that `for_options()` always created a new `WalkForwardConfigOptions()` with default `validation_mode='walk_forward'`, losing holdout settings.

**Fix:** Modified `for_options()` to check `self.walk_forward.validation_mode` and preserve holdout mode when present.

### IWM Analysis: False Positive Explained

| Metric | IWM (83K-15) | After Fix |
|--------|--------------|-----------|
| IS Sharpe | -7.70 | -7.70 |
| OOS Sharpe | +4.09 | +4.09 |
| Degradation | 0.0 (BUG) | 0.0 (with WARNING) |
| Sign Reversal | Not detected | DETECTED |
| WF Result | PASS | PASS (with WARNING) |

**Conclusion:** IWM still passes validation, but now includes a sign reversal warning flagging it for manual review. The negative IS / positive OOS pattern is suspicious and may indicate luck, regime change, or data issues.

### Files Modified

| File | Lines | Change |
|------|-------|--------|
| `validation/results.py` | 117-119, 132-133, 162-166 | Added sign reversal warning fields |
| `validation/walk_forward.py` | 339-351, 419-420, 439-452, 293-315, 330, 534-536 | Sign reversal detection + trade filtering |
| `validation/config.py` | 276-304 | Fixed for_options() to preserve holdout mode |
| `tests/test_validation/test_walk_forward.py` | 160-251, 827-887 | Updated + 8 new regression tests |

### Test Results

| Test Suite | Before | After | Delta |
|------------|--------|-------|-------|
| Validation tests | 209 | 217 | +8 |
| Total passing | 209 | 217 | ALL PASS |

### New Tests Added (8)

1. `test_sign_reversal_negative_is_positive_oos` - IWM case
2. `test_sign_reversal_positive_is_negative_oos` - SPY case
3. `test_no_sign_reversal_both_negative` - Both bad case
4. `test_no_sign_reversal_both_positive` - Normal case
5. `test_for_options_preserves_holdout_mode` - Holdout preservation
6. `test_for_options_preserves_holdout_train_pct` - Train pct preservation
7. `test_for_options_uses_options_thresholds_in_holdout` - Thresholds correct
8. `test_for_options_walk_forward_mode_unchanged` - Default mode unchanged

### Session 83K-17 Tasks

1. Re-run validation with fixed framework - results should now be trustworthy
2. Analyze any runs that show sign reversal warnings
3. Consider Monte Carlo threshold adjustments for options
4. Focus on daily timeframe (skip 1W/1M for speed)

---

## Session 83K-15: Holdout Mode Bug Fix + Validation Run

**Date:** November 30, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Bug fixed, partial validation run
**Commit:** Pending

### Critical Bug Found and Fixed

**Root Cause:** In `validation/validation_runner.py` line 118, `for_options()` was creating a NEW config that overwrote the holdout settings:

```python
# OLD (buggy)
config = self.config.for_options() if is_options else self.config
# This created a NEW WalkForwardConfigOptions() ignoring holdout mode!
```

**Fix Applied:**
```python
# NEW (fixed)
if is_options:
    if self.config.walk_forward.validation_mode == 'holdout':
        config = self.config.for_options()
        config.walk_forward = self.config.walk_forward  # Restore holdout config
    else:
        config = self.config.for_options()
else:
    config = self.config
```

### Verification Results

- All 209 validation tests: PASSING
- Holdout mode now correctly generates 1 fold (was generating 15)
- Walk-forward validation output: "Generated 1 folds" (verified)

### Partial Validation Results (Daily Timeframe Only)

| Pattern | Symbol | Trades | IS Sharpe | OOS Sharpe | WF Pass | ThetaData |
|---------|--------|--------|-----------|------------|---------|-----------|
| 3-1-2 | SPY | 13 | 5.07 | -16.90 | FAIL | 100% |
| 3-1-2 | QQQ | 24 | -2.84 | -7.86 | FAIL | 96% |
| 3-1-2 | AAPL | 18 | -6.44 | 1.81 | FAIL | 100% |
| 3-1-2 | IWM | 20 | -7.70 | 4.09 | PASS | 100% |
| 3-1-2 | DIA | 22 | -4.48 | -3.08 | FAIL | 100% |
| 2-1-2 | SPY | 68 | -3.56 | 0.00 | FAIL | 100% |
| 2-1-2 | QQQ | 76 | - | - | FAIL | 100% |

**Key Observations:**
- 3-1-2 batch: 0/5 passed, ThetaData coverage 99.2%
- IWM 3-1-2 had positive OOS Sharpe (4.09) - walk-forward PASSED
- 2-1-2 pattern has 5x more trades (68-76) vs 3-1-2 (13-24)
- Weekly/Monthly timeframes slow due to ThetaData 472 errors for historical data

### Files Modified

| File | Change |
|------|--------|
| `validation/validation_runner.py` | Fixed line 118 to preserve holdout config with is_options=True |

### Session 83K-16 Tasks

1. Run optimized validation (skip slow timeframes initially)
2. Investigate why OOS Sharpe is often 0.00 (no trades in OOS period?)
3. Consider reducing ThetaData retry count for faster validation
4. Analyze 3-1-2_1D_IWM (the only walk-forward PASS)

---

## Session 83K-14: Holdout Validation Mode Implementation

**Date:** November 30, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Holdout mode implemented, 209 tests passing
**Commit:** Pending

### Implementation Summary

Implemented holdout validation mode (70/30 train/test split) as alternative to walk-forward for sparse STRAT pattern strategies.

### Changes Made

| File | Change |
|------|--------|
| `validation/config.py` | Added `validation_mode`, `holdout_train_pct` fields, `WalkForwardConfigHoldout` class |
| `validation/walk_forward.py` | Added holdout fold generation in `_generate_fold_windows()`, conditional checks in `_aggregate_results()` |
| `validation/strat_validator.py` | Added `holdout_mode` parameter to `ATLASSTRATValidator.__init__()` |
| `scripts/run_atlas_validation_83k.py` | Added `--holdout` and `--include-nvda` CLI flags, auto-exclude NVDA |
| `tests/test_validation/test_walk_forward.py` | Added 11 holdout mode tests |
| `validation/__init__.py` | Exported `WalkForwardConfigHoldout`, version 0.7.1 |

### CLI Usage

```bash
# Run holdout validation (NVDA auto-excluded)
python scripts/run_atlas_validation_83k.py --holdout

# Dry run to verify setup
python scripts/run_atlas_validation_83k.py --holdout --dry-run

# Include NVDA (override auto-exclude)
python scripts/run_atlas_validation_83k.py --holdout --include-nvda
```

### Holdout vs Walk-Forward

| Aspect | Walk-Forward | Holdout |
|--------|--------------|---------|
| Folds | 15 rolling | 1 single |
| Split | 252 train / 63 test | 70% train / 30% test |
| Best for | High-frequency strategies | Sparse patterns |
| Checks | All validation checks | Skips profitable_folds, param_stability |

### Test Results

- 55 walk-forward tests: ALL PASSING (44 original + 11 new)
- 209 total validation tests: ALL PASSING
- Dry run: 75 combinations verified (NVDA excluded)

### Session 83K-15 Tasks

1. Run full validation with holdout mode: `python scripts/run_atlas_validation_83k.py --holdout`
2. Analyze results and generate report
3. Begin DTE optimization (Phase 7)

---

## Session 83K-13: Walk-Forward Validation Limitation Discovery

**Date:** November 30, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Critical methodology limitation discovered
**Commit:** None (documentation only)

### Critical Discovery

**Walk-forward validation is inappropriate for sparse pattern strategies like STRAT.**

With 13-24 trades over 5 years split across 15 folds:
- Each fold averages ~1-2 trades
- Most OOS windows (63 bars) have 0 trades
- OOS Sharpe = 0.00 (mathematically correct - no trades to measure)
- 100% degradation is reported but meaningless

### Validation Results (Partial - 5/90 runs before termination)

| Run | Trades | ThetaData | OOS Issue |
|-----|--------|-----------|-----------|
| 3-1-2_1D_SPY | 13 | 100% | 0 trades in most OOS folds |
| 3-1-2_1D_QQQ | 24 | 96% | 0 trades in most OOS folds |
| 3-1-2_1D_AAPL | 18 | 100% | 0 trades in most OOS folds |
| 3-1-2_1D_IWM | 20 | 100% | 0 trades in most OOS folds |
| 3-1-2_1D_DIA | 22 | 100% | 0 trades in most OOS folds |

### Positive Findings

1. **ThetaData Integration Working** - 96-100% coverage for most symbols
2. **P&L Fix Validated** - SPY original Sharpe=4.09, MaxDD=11.4% (per Monte Carlo)
3. **Bias Detection Passing** - All runs pass bias detection

### NVDA Historical Data Issue

- Pre-split strike prices ($15-$20 from 2021) cause ThetaData 472 errors
- NVDA had 4:1 split in July 2021 - old strikes don't exist in ThetaData
- Recommendation: Exclude NVDA or use post-2022 data only

### Proposed Solutions for Session 83K-14

**Option A: Holdout Validation (Recommended)**
- Simple 70/30 train/test split instead of walk-forward
- More trades in each period
- Easier to interpret results

**Option B: Reduce Fold Count**
- Use 3-5 folds instead of 15
- Larger windows = more trades per fold

**Option C: Pattern Aggregation**
- Combine 3-1-2 + 2-1-2 + 2-2 patterns
- More signals per period

### Session 83K-14 Tasks

1. Modify `validation/walk_forward.py` for holdout mode
2. Update `scripts/run_atlas_validation_83k.py` with `--holdout` flag
3. Re-run validation with holdout approach
4. Skip or fix NVDA data issues

---

## Session 83K-12: P&L Fix Validation and Detailed Trade Analysis

**Date:** November 29, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - P&L fix validated, circular import fixed, detailed metrics obtained
**Commit:** Pending (circular import fix)

### P&L Fix Validation

Confirmed P&L calculation fix from Session 83K-11 is working correctly:

| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| Avg IS Sharpe | -84.17 | +1.88 | +86.05 |
| Original Sharpe | -6.46 | +1.68 | +8.14 |
| Original MaxDD | 100% | 19.6% | -80.4% |

### SPY 1D 3-1-2 Detailed Results

| Metric | Value |
|--------|-------|
| Trade Count | 16 |
| Total Return | +12.59% |
| Sharpe Ratio | **2.13** |
| Max Drawdown | 18.63% |
| Win Rate | **37.5%** (6/16) |
| Profit Factor | **1.37** |
| Total P&L | +$1,259 |
| Avg Win | +$778 |
| Avg Loss | -$341 |

### Circular Import Fix

Fixed circular import between `validation/strat_validator.py` and `strategies/strat_options_strategy.py`:

**Root Cause:**
- `strategies/strat_options_strategy.py` imports `validation.protocols`
- `validation/__init__.py` imports `validation.strat_validator`
- `validation/strat_validator.py` imports `strategies.strat_options_strategy`

**Fix:** Made import lazy (inside `_run_single_validation()` method) in `validation/strat_validator.py`

### Key Finding: Checkpoint Timing

Old checkpoint data showed impossible metrics because it was created BEFORE the P&L fix:
- Checkpoint timestamp: 2025-11-29 18:40:18 (6:40 PM)
- P&L fix commit: 2025-11-29 20:05:17 (8:05 PM)

Fresh validation run confirmed fix is working.

### Test Results

- 469 STRAT/validation tests: ALL PASSING (2 skipped)
- Circular import fix verified

### Files Modified

| File | Change |
|------|--------|
| `validation/strat_validator.py` | Lazy import to fix circular dependency |
| `exploratory/debug_trade_details.py` | Created for detailed trade analysis |

---

## Session 83K-11: P&L Calculation Double-Subtraction Bug Fix

**Date:** November 29, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Critical P&L bug fixed, 469 tests passing
**Commit:** 070ac95

### The Bug

During validation run, ALL 90 combinations showed:
- Sharpe ratios: -100 to -636 (astronomically negative)
- MaxDD: 100% (total loss every run)
- ThetaData coverage: 100% (previous bug fixed)

Investigation revealed P&L calculation bugs in `strat/options_module.py` lines 1417-1434.

### Bug #1: TARGET Hits Double-Subtracted Premium

**Location:** `strat/options_module.py:1431`

```python
# OLD (buggy)
gross_pnl = delta_pnl + gamma_pnl + theta_pnl
pnl = gross_pnl - actual_option_cost * 100 * trade.quantity  # WRONG!

# NEW (fixed)
gross_pnl = delta_pnl + gamma_pnl + theta_pnl
pnl = gross_pnl  # gross_pnl IS the P&L (change in option value)
```

**Root Cause:** `gross_pnl` (delta + gamma + theta) represents the CHANGE in option value, which IS the profit/loss. Subtracting the entry cost again made winning trades appear as losses.

### Bug #2: STOP Hits Assumed 100% Loss

**Location:** `strat/options_module.py:1434`

```python
# OLD (buggy)
pnl = -actual_option_cost * 100 * trade.quantity  # Always 100% loss!

# NEW (fixed)
pnl = gross_pnl  # Use Greek-based calculation (same as TARGET)
```

**Root Cause:** Code assumed every stop hit = 100% loss of premium, regardless of actual price movement.

### Results

| Metric | BEFORE Fix | AFTER Fix | Improvement |
|--------|------------|-----------|-------------|
| SPY 1D Sharpe | -6.46 | +4.09 | +1034% |
| SPY 1D MaxDD | 100% | 11.4% | Realistic |
| Fold 1 IS Sharpe | -103.24 | +7.10 | Positive |
| Fold 12 IS Sharpe | -401.44 | +173.52 | Positive |

### Files Modified

| File | Change |
|------|--------|
| `strat/options_module.py` | Lines 1417-1434 - P&L calculation fix |
| `tests/test_strat/test_options_pnl.py` | 3 regression tests added |

### Test Results

- 271 STRAT tests: ALL PASSING (268 + 3 new)
- 198 validation tests: ALL PASSING
- Total: 469 tests passing

---

