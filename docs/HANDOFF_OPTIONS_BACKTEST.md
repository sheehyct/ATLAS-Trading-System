# HANDOFF - STRAT Options Backtest with ThetaData

**Last Updated:** December 14, 2025 (Session 85)
**Status:** CRITICAL BUG FOUND - Direction misclassification invalidates previous results
**Priority:** URGENT - Multi-session fix required before any production deployment

---

## Session 85 Summary - CRITICAL BUG DISCOVERED

### Session 85 Accomplishments (Before Bug Discovery)

1. **Risk Level Analysis** - Found 7% risk closest to SPY B&H (+107.9% vs +121.5%)
2. **QuantStats Reports** - Generated 6 PDF tearsheets with ThetaData pricing
3. **Greeks Verification** - 100% of strikes in target delta range (0.30-0.80)

### CRITICAL BUG: Direction Misclassification

**Location:** `scripts/backtest_strat_options_thetadata.py` line 343

**Buggy Code:**
```python
direction = 1 if 'Up' in pattern_type or '2U' in pattern_type else -1
```

**Problem:** Pattern `2U-1-2D` contains "2U" so it's classified as BULLISH (call) when it should be BEARISH (put).

**Impact:**
| Pattern | Source Data | Backtest Result | Error |
|---------|-------------|-----------------|-------|
| 2U-1-2D | 40 bearish | 32 as CALLS | WRONG - should be PUTS |
| Total Bearish | 79 patterns | 29 trades | 50 patterns misclassified |

**Root Cause:** The CSV has a correct `direction` column but the backtest ignores it.

**Fix Required:**
```python
direction = 1 if row.get('direction') == 'bullish' else -1
```

---

## SYSTEMIC ISSUE: Pattern Naming Convention

### Problem

The codebase inconsistently uses pattern names without proper directional bar suffixes:
- "2-2 Up" should be "2U-2U" or "2D-2U" (specifying both bars)
- "2-1-2" should be "2U-1-2U" or "2D-1-2D" (specifying direction)

Per CLAUDE.md Section 13: **Every directional bar MUST be classified as 2U (bullish) or 2D (bearish). Never use just "2".**

### Affected Files (Potentially)

| File | Issue | Severity |
|------|-------|----------|
| `scripts/backtest_strat_options_thetadata.py` | Direction inference bug (line 343) | CRITICAL |
| `strat/paper_signal_scanner.py` | Uses "2-2" without direction | HIGH |
| `strat/paper_trading.py` | VPS paper trading | HIGH |
| `strat/tier1_detector.py` | Uses "2-2 Up/Down" (somewhat ok) | MEDIUM |
| `strat/options_module.py` | Mixed notation | MEDIUM |
| `crypto/scanning/signal_scanner.py` | Uses "2-2" | MEDIUM |
| Validation CSVs | Contains "2-2 Up" instead of "2U-2U" | HIGH |

### VPS Paper Trading Risk

The deployed VPS paper trading system uses `strat/paper_trading.py` and `strat/paper_signal_scanner.py`. If these have similar direction logic issues, live paper trades may be affected.

---

## Previous Results (NOW SUSPECT)

The following results were generated with the direction bug and should be re-validated after fix:

### SPY Backtest Results (ThetaData, 2018-2024) - SUSPECT

| Risk | Return | Note |
|------|--------|------|
| 2% | +37.5% | Bearish patterns misclassified |
| 5% | +81.7% | Bearish patterns misclassified |
| 7% | +107.9% | Bearish patterns misclassified |
| 10% | +382.7% | Bearish patterns misclassified |

### Drawdown Analysis Finding

- Peak: $27,737 (Sept 2018)
- Low: $11,748 (June 2022) - 57.6% drawdown
- Recovery: 2023 (1700+ days)
- **This may be partially explained by bearish patterns being traded as calls during bearish markets**

---

## Next Session Tasks (MULTI-SESSION)

### Phase 1: Fix Direction Bug (Session 86)

1. **Fix backtest script**
   - Use `direction` column from CSV instead of pattern name inference
   - Location: `scripts/backtest_strat_options_thetadata.py` line 343

2. **Re-run all backtests**
   - Generate new results with correct direction handling
   - Compare before/after to quantify bug impact

3. **Update QuantStats reports**
   - Regenerate PDFs with corrected data

### Phase 2: Audit All Pattern Logic (Session 87+)

1. **Audit strat-methodology skill**
   - Ensure proper bar classification terminology
   - Location: `C:/Users/sheeh/.claude/skills/strat-methodology/`

2. **Audit paper trading system**
   - `strat/paper_signal_scanner.py`
   - `strat/paper_trading.py`
   - Check if VPS deployment is affected

3. **Audit pattern detector**
   - `strat/pattern_detector.py`
   - `strat/tier1_detector.py`

4. **Standardize pattern naming**
   - Replace "2-2 Up" with "2D-2U" or "2U-2U" throughout
   - Replace "2-2 Down" with "2U-2D" or "2D-2D" throughout
   - Update validation CSVs if needed

### Phase 3: Validation (Session 88+)

1. **Verify VPS paper trading**
   - Check live paper trades for direction correctness
   - Audit recent trade history

2. **Update MASTER_FINDINGS_REPORT.md**
   - Add corrected results
   - Document the bug and fix

---

## Files to Review

### Critical (Fix First)
- `scripts/backtest_strat_options_thetadata.py` - Line 343 direction bug

### High Priority (Audit)
- `strat/paper_signal_scanner.py` - VPS scanner
- `strat/paper_trading.py` - VPS trading
- `scripts/strat_validation_*.csv` - Pattern naming

### Medium Priority (Standardize)
- `strat/pattern_detector.py`
- `strat/tier1_detector.py`
- `strat/options_module.py`
- `crypto/scanning/signal_scanner.py`

### Skills (Verify Terminology)
- `C:/Users/sheeh/.claude/skills/strat-methodology/`

---

## Commands Reference

```bash
# Start ThetaData terminal (required for backtests)
cd C:\thetaterminal && java -jar thetaterminalv3.jar

# Run backtest (after fix)
uv run python scripts/backtest_strat_options_thetadata.py --symbol SPY --risk 7

# Generate QuantStats report (after fix)
uv run python scripts/generate_strat_options_quantstats.py --symbol SPY --risk 7 --thetadata
```

---

## OpenMemory Queries

```
"Session 85 direction bug"
"STRAT bar classification"
"2U-1-2D misclassification"
```

---

## Key Lesson

**ALWAYS use explicit direction from data, never infer from pattern names.**

Pattern names like "2U-1-2D" are ambiguous because:
- They contain both "2U" and "2D"
- String matching like `'2U' in pattern` will incorrectly match
- The EXIT bar (last bar) determines trade direction, not the first bar
