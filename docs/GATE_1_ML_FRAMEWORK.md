# Gate 1 ML Optimization Framework

**Created:** December 4, 2025 (Session 83K-39)
**Status:** DESIGN COMPLETE - Ready for implementation
**Prerequisites:** Gate 0 Economic Logic documented in GATE_0_ECONOMIC_LOGIC.md

---

## Executive Summary

This document defines the ML optimization framework for patterns that have passed Gate 1 sample size requirements. Based on analysis of non-hourly (Daily/Weekly/Monthly) trade data, **only 2 patterns qualify for ML optimization**.

**Hourly trades excluded** - Session 83K-38/39 confirmed hourly is NOT profitable (-$240 avg P&L).

---

## Gate 1 Status

### Pattern Eligibility (Non-Hourly Only)

| Pattern | Trades | Simple ML (200+) | Complex ML (500+) | Status |
|---------|--------|------------------|-------------------|--------|
| **2-2** | 576 | PASS | PASS | **ML READY** |
| **3-2** | 519 | PASS | PASS | **ML READY** |
| 3-2-2 | 148 | FAIL | FAIL | Need 52+ more trades |
| 2-1-2 | 172 | FAIL | FAIL | Need 28+ more trades |
| 3-1-2 | 44 | FAIL | FAIL | Need 156+ more trades |

### Timeframe Distribution for ML-Ready Patterns

**2-2 Pattern (576 trades):**
| Timeframe | Trades | Avg P&L |
|-----------|--------|---------|
| Daily | 459 | $368 |
| Weekly | 90 | $1,044 |
| Monthly | 27 | $3,016 |

**3-2 Pattern (519 trades):**
| Timeframe | Trades | Avg P&L |
|-----------|--------|---------|
| Daily | 387 | $463 |
| Weekly | 112 | $1,818 |
| Monthly | 20 | $3,318 |

---

## Approved ML Applications

Per ML_IMPLEMENTATION_GUIDE_STRAT.md Section 7:

### Application 1: Delta/Strike Optimization

**Objective:** Given a validated STRAT signal, predict optimal delta for options positioning.

**Features (Documented Economic Rationale):**

| Feature | Description | Economic Rationale |
|---------|-------------|-------------------|
| `iv_percentile_30d` | IV as percentile of 30-day range | High IV = expensive options, favor ITM |
| `iv_rank_52w` | IV as percentile of 52-week range | Extreme IV affects premium value |
| `dte` | Days to expiration | Time decay rate affects optimal strike |
| `atr_percentile` | ATR as percentile of 20-day | Volatility affects magnitude probability |
| `magnitude_pct` | Expected move from entry to target | Larger moves favor OTM |
| `timeframe_encoded` | Numeric timeframe (1D=1, 1W=5, 1M=21) | Higher TF = longer hold = more decay |
| `vix_bucket` | VIX level category (<15, 15-25, 25-35, >35) | Market regime affects delta selection |

**Output:** Delta bucket recommendation (0.40-0.50, 0.50-0.65, 0.65-0.80)

**Model Selection:**
- Primary: Ridge Regression with alpha=1.0 (interpretable coefficients)
- Secondary: Gradient Boosting with max_depth=3 (if Ridge underperforms)

**Baseline to Beat:** Static delta=0.65 (current system default)

### Application 2: DTE Selection

**Objective:** Predict optimal days-to-expiration based on expected time-to-magnitude.

**Features (Documented Economic Rationale):**

| Feature | Description | Economic Rationale |
|---------|-------------|-------------------|
| `pattern_type` | Encoded pattern (2-2=1, 3-2=2) | Different patterns resolve at different speeds |
| `timeframe_encoded` | Numeric timeframe | Higher TF = longer resolution |
| `magnitude_pct` | Expected move percentage | Larger moves need more time |
| `atr_ratio` | ATR / magnitude | Low ratio = easier target |
| `day_of_week` | Mon=1 to Fri=5 | Weekly effects on resolution |
| `iv_percentile` | IV environment | High IV may accelerate moves |
| `continuity_score` | TF alignment (0-5 scale) | Higher = faster resolution |

**Output:** DTE bucket recommendation (7-14, 14-21, 21-35, 35-60)

**Model Selection:**
- Primary: Random Forest with max_depth=4, min_samples_leaf=25
- Secondary: XGBoost with reg_lambda=1.0 (if RF underperforms)

**Baseline to Beat:** Static DTE rules:
- Daily: 21 days
- Weekly: 35 days
- Monthly: 75 days

### Application 3: Position Sizing

**Objective:** Adjust position size based on setup quality score.

**Features (Documented Economic Rationale):**

| Feature | Description | Economic Rationale |
|---------|-------------|-------------------|
| `continuity_score` | TF alignment (0-5) | Higher = more reliable |
| `vix_bucket` | Market regime | High VIX = reduce size |
| `magnitude_pct` | Expected move | Larger = potentially higher quality |
| `pattern_type` | Pattern classification | Some patterns more reliable |
| `timeframe_encoded` | Trade timeframe | Higher TF = higher quality |
| `symbol_liquidity` | Avg daily volume bucket | Low liquidity = reduce size |
| `recent_win_rate` | Rolling 20-trade win rate | Cold streaks = reduce size |

**Output:** Position size multiplier (0.5x, 0.75x, 1.0x, 1.25x, 1.5x)

**Model Selection:**
- Primary: Logistic Regression with L2 regularization
- Secondary: Ordinal Regression (if treating as ordered categories)

**Baseline to Beat:** Static 1.0x sizing (equal weight all trades)

---

## Validation Protocol

### Train/Validation/Test Split

Per ML_IMPLEMENTATION_GUIDE_STRAT.md Section 5:

```
Total: 576 trades (2-2) or 519 trades (3-2)

Split (60/20/20):
- Train: 60% (345 trades for 2-2, 311 for 3-2)
- Validation: 20% (115 trades for 2-2, 104 for 3-2)
- Test: 20% (116 trades for 2-2, 104 for 3-2) - SACRED

Important:
- Temporal splits ONLY (no random shuffling)
- Test set never touched until final evaluation
- Purge 5 days at split boundaries
- Embargo 2 days after each split
```

### Walk-Forward Validation

```
Window Configuration:
- Initial training: 200 trades
- Test window: 50 trades
- Step size: 25 trades
- Total folds: ~6-8 folds depending on pattern

Each fold:
1. Train on expanding window
2. Test on next 50 trades
3. Record metrics
4. Step forward 25 trades
5. Repeat
```

### Performance Retention Requirements

Per ML_IMPLEMENTATION_GUIDE_STRAT.md Section 2.5:

| Metric | Required Retention |
|--------|-------------------|
| Sharpe | >80% of training |
| Win Rate | >85% of training |
| Profit Factor | >80% of training |
| Max Drawdown | <120% of training |

### 50% Haircut Rule

ALL backtest improvements must be haircut by 50% for deployment expectations:

| Backtest Improvement | Expected Live Improvement |
|---------------------|---------------------------|
| +20% Sharpe | +10% Sharpe |
| +$100/trade | +$50/trade |
| +10% win rate | +5% win rate |

---

## Implementation Roadmap

### Phase 1: Data Preparation (Session 83K-40)

1. **Export ML-ready trade data:**
   - Merge all non-hourly trade CSVs
   - Add computed features (IV percentile, ATR ratio, etc.)
   - Validate feature completeness

2. **Feature engineering:**
   - Calculate all features from OHLCV + options data
   - Document each feature's calculation
   - Handle missing values (ThetaData gaps)

3. **Create temporal splits:**
   - Sort by entry_date
   - Apply 60/20/20 split with purging/embargo
   - Quarantine test set

### Phase 2: Delta Optimization (Session 83K-41)

1. **Baseline measurement:**
   - Calculate metrics with static delta=0.65
   - Record Sharpe, win rate, avg P&L

2. **Model training:**
   - Train Ridge Regression on training set
   - Hyperparameter tuning on validation set
   - Feature importance analysis

3. **Validation:**
   - Walk-forward validation (6-8 folds)
   - Check retention requirements
   - Compare to baseline

4. **If passed:**
   - Apply 50% haircut
   - Document expected improvement
   - Move to Phase 3

### Phase 3: DTE Optimization (Session 83K-42)

1. **Baseline measurement:**
   - Calculate metrics with static DTE rules

2. **Model training:**
   - Train Random Forest on training set
   - Feature importance analysis

3. **Validation:**
   - Walk-forward validation
   - Retention checks

4. **Combine with Delta model:**
   - Test Delta + DTE together
   - Ensure no degradation

### Phase 4: Position Sizing (Session 83K-43)

1. **Baseline measurement:**
   - Equal-weight (1.0x) all trades

2. **Model training:**
   - Train Logistic Regression
   - Risk-adjusted metrics focus

3. **Validation:**
   - Walk-forward validation
   - Ensure no drawdown increase

4. **Full Stack Test:**
   - Combine Delta + DTE + Sizing
   - Final validation metrics

### Phase 5: Paper Trading (Session 83K-44+)

1. **Shadow mode (2 weeks):**
   - ML runs but doesn't affect trades
   - Log predictions vs rule-based

2. **Paper trading (6 months minimum):**
   - A/B comparison with baseline
   - 50+ trades with ML optimization
   - Monitor for drift

---

## File Structure

```
strat/
  ml/
    __init__.py
    delta_optimizer.py      # Delta/strike ML model
    dte_selector.py         # DTE ML model
    position_sizer.py       # Position sizing ML model
    feature_engineering.py  # Feature calculation
    validation.py           # Walk-forward, retention checks

scripts/
    prepare_ml_data.py      # Data preparation script
    train_delta_model.py    # Delta model training
    train_dte_model.py      # DTE model training
    train_sizing_model.py   # Sizing model training
    evaluate_ml_stack.py    # Combined evaluation

validation_results/
  ml/
    delta_model/
      training_metrics.json
      validation_metrics.json
      walk_forward_results.json
    dte_model/
      ...
    sizing_model/
      ...
    combined/
      final_test_results.json  # ONLY after all validation passes
```

---

## Monitoring and Rollback

### Daily Monitoring

| Metric | Alert Threshold |
|--------|-----------------|
| Prediction drift | >25% from training distribution |
| Feature drift | >20% for any feature |
| Rolling Sharpe (20 trades) | <50% of validation Sharpe |
| Consecutive losses | >5 trades |

### Rollback Triggers

Immediately revert to rule-based if:
- Sharpe drops >20% vs baseline
- Max DD increases >15 percentage points
- Win rate drops >10 percentage points
- 5+ consecutive losses

### Recalibration Schedule

- Monthly: Check feature stability
- Quarterly: Consider recalibration if performance decay
- Never: Retrain on test set data

---

## Prohibited Actions

Per ML_IMPLEMENTATION_GUIDE_STRAT.md Section 8:

1. **NO signal generation** - STRAT patterns are rule-based
2. **NO direction prediction** - Timeframe continuity determines direction
3. **NO pattern classification** - Bar types are deterministic
4. **NO neural networks** - Insufficient data
5. **NO deep learning** - Insufficient data
6. **NO automated feature discovery** - All features need economic rationale

---

## Next Steps

1. **Session 83K-40:** Phase 1 - Data Preparation
   - Merge trade CSVs
   - Engineer features
   - Create temporal splits

2. **Session 83K-41:** Phase 2 - Delta Optimization
   - Train and validate delta model
   - Compare to baseline

3. Continue through phases as each passes validation

---

## References

- ML_IMPLEMENTATION_GUIDE_STRAT.md - Full ML guidelines
- GATE_0_ECONOMIC_LOGIC.md - Economic logic documentation
- MASTER_FINDINGS_REPORT.md - Validation statistics
- HANDOFF.md - Session history

---

*Gate 1 ML Framework Design Complete - Session 83K-39*
