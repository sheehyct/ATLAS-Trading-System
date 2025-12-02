# Machine Learning Implementation Guide for STRAT Trading System

**Version:** 1.0  
**Created:** December 2, 2025  
**Purpose:** Define strict gates, validation requirements, and best practices for ML implementation  
**Audience:** Claude Code, development team  
**Status:** REFERENCE DOCUMENT - Do not implement ML until gates are passed

---

## Executive Summary

This document defines **when** and **how** machine learning may be implemented in the STRAT options trading system. ML is NOT a replacement for rule-based edge detection—it is a potential optimization layer for secondary parameters AFTER the base edge has been statistically validated.

**Core Principle:** ML can potentially sharpen a dull blade, but it cannot create a blade from nothing. The structural edge must exist first.

---

## Table of Contents

1. [Fundamental Philosophy](#1-fundamental-philosophy)
2. [Gate System: Prerequisites for ML](#2-gate-system-prerequisites-for-ml)
3. [Legitimate vs Illegitimate ML Use Cases](#3-legitimate-vs-illegitimate-ml-use-cases)
4. [Sample Size Requirements](#4-sample-size-requirements)
5. [Validation Protocol](#5-validation-protocol)
6. [Overfitting Prevention](#6-overfitting-prevention)
7. [Approved ML Applications](#7-approved-ml-applications)
8. [Prohibited ML Applications](#8-prohibited-ml-applications)
9. [Implementation Checklist](#9-implementation-checklist)
10. [Model Selection Guidelines](#10-model-selection-guidelines)
11. [Feature Engineering Rules](#11-feature-engineering-rules)
12. [Performance Evaluation](#12-performance-evaluation)
13. [Deployment Protocol](#13-deployment-protocol)
14. [Monitoring and Maintenance](#14-monitoring-and-maintenance)

---

## 1. Fundamental Philosophy

### 1.1 Why STRAT Works (Economic Reasoning)

STRAT's edge comes from **timeframe continuity indicating institutional participation**, not from pattern frequency or statistical correlations.

```
STRAT Edge Source:
┌─────────────────────────────────────────────────────────────┐
│  Full Timeframe Continuity (M/W/D aligned)                  │
│           ↓                                                 │
│  Indicates: Large institutional flows in same direction    │
│           ↓                                                 │
│  Result: Higher probability of magnitude target hit        │
└─────────────────────────────────────────────────────────────┘
```

**This is a structural explanation, not a statistical one.** ML cannot "understand" this—it can only find correlations. Those correlations may be:
- Real (continuity → institutional flow → magnitude hit)
- Spurious (artifacts of training period)
- Backwards (magnitude causes apparent continuity post-hoc)

### 1.2 ML's Proper Role

```
CORRECT HIERARCHY:
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Rule-Based Signal Generation (STRAT patterns)     │
│          - Bar classification (1, 2U, 2D, 3)               │
│          - Timeframe continuity check                       │
│          - Pattern recognition (3-1-2, 2-1-2, etc.)        │
│          → ML NEVER touches this layer                      │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Secondary Parameter Optimization (ML ALLOWED)      │
│          - Delta/strike selection                          │
│          - DTE optimization                                │
│          - Position sizing adjustments                      │
│          → ML may optimize AFTER Layer 1 validates         │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Regime Context (ML ALLOWED with caution)          │
│          - Broad market regime identification              │
│          - IV environment classification                    │
│          → ML for context, not for signal generation       │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 The Overfitting Reality

**Academic evidence (Harvey, Liu, Zhu 2016):** Of 316 factors identified in academic literature, most fail replication due to data mining.

**Practical implication:** Any improvement ML shows in backtesting should be haircut by 50% for live expectations.

| Backtest Improvement | Expected Live Improvement |
|---------------------|---------------------------|
| +20% Sharpe | +10% Sharpe |
| +30% win rate | +15% win rate |
| +$500/trade | +$250/trade |
| +15% CAGR | +7.5% CAGR |

---

## 2. Gate System: Prerequisites for ML

### 2.1 Gate Overview

ML implementation requires passing ALL gates sequentially. No gate may be skipped.

```
GATE SYSTEM:
┌──────────────────────────────────────────────────────────────┐
│ GATE 0: Base Edge Validation                                 │
│ Requirement: Rule-based strategy shows positive expectancy   │
│ Minimum: 100 trades, positive Sharpe, clear economic logic  │
│ Status: [  ] NOT PASSED  [ ] PASSED                         │
├──────────────────────────────────────────────────────────────┤
│ GATE 1: Sample Size Threshold                                │
│ Requirement: Sufficient trades for ML validation splits      │
│ Minimum: 200 trades (simple ML) or 500 trades (complex ML)  │
│ Status: [  ] NOT PASSED  [ ] PASSED                         │
├──────────────────────────────────────────────────────────────┤
│ GATE 2: Feature Stability Analysis                           │
│ Requirement: Features show consistent relationships          │
│ Minimum: Rolling correlation stability > 0.6                │
│ Status: [  ] NOT PASSED  [ ] PASSED                         │
├──────────────────────────────────────────────────────────────┤
│ GATE 3: Train/Validation/Test Split                          │
│ Requirement: Proper temporal splits, no leakage             │
│ Minimum: 60/20/20 split, test set NEVER touched            │
│ Status: [  ] NOT PASSED  [ ] PASSED                         │
├──────────────────────────────────────────────────────────────┤
│ GATE 4: Out-of-Sample Validation                             │
│ Requirement: Model performs on validation set               │
│ Minimum: >80% of training performance retained              │
│ Status: [  ] NOT PASSED  [ ] PASSED                         │
├──────────────────────────────────────────────────────────────┤
│ GATE 5: Paper Trading Validation                             │
│ Requirement: Live paper trading confirms ML benefit         │
│ Minimum: 6 months, >50 trades, improvement over rule-based  │
│ Status: [  ] NOT PASSED  [ ] PASSED                         │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Gate 0: Base Edge Validation (CRITICAL)

**This gate must be passed BEFORE any ML consideration.**

```python
# Gate 0 Validation Criteria
def validate_gate_0(trade_results: pd.DataFrame) -> dict:
    """
    Validate that rule-based strategy has positive expectancy.
    
    Requirements:
    - Minimum 100 completed trades
    - Positive Sharpe ratio (annualized)
    - Win rate > 40% OR average win > 2x average loss
    - Positive total P&L
    - Tested across minimum 3 symbols
    - Tested across minimum 2 market regimes
    """
    
    validation = {
        'trade_count': len(trade_results) >= 100,
        'sharpe_positive': calculate_sharpe(trade_results) > 0,
        'expectancy_positive': calculate_expectancy(trade_results) > 0,
        'multi_symbol': trade_results['symbol'].nunique() >= 3,
        'multi_regime': covers_multiple_regimes(trade_results),
        'economic_logic': True  # Must be manually confirmed
    }
    
    validation['gate_passed'] = all(validation.values())
    return validation
```

**Economic Logic Requirement (Manual Confirmation):**

Before proceeding past Gate 0, document answers to:

1. **Why does this pattern work?** (Must have structural explanation)
2. **Who is on the other side of this trade?** (Identify counterparty)
3. **Why hasn't this been arbitraged away?** (Barriers to entry)
4. **What would cause this edge to disappear?** (Risk factors)

### 2.3 Gate 1: Sample Size Threshold

| ML Application Type | Minimum Trades | Rationale |
|---------------------|---------------|-----------|
| Simple statistics (base rates) | 50-100 | Central limit theorem |
| Linear regression | 200 | Need variance in features |
| Gradient boosting (XGBoost, LightGBM) | 300-500 | Avoid leaf overfitting |
| Random forest | 300-500 | Need depth for splits |
| Neural networks | 5,000+ | NOT RECOMMENDED for this system |
| Deep learning | 50,000+ | PROHIBITED for this system |

**Current Status Check:**
```python
def check_gate_1(trade_count: int, ml_type: str) -> bool:
    """Check if sample size is sufficient for ML type."""
    
    thresholds = {
        'simple_stats': 100,
        'linear_regression': 200,
        'logistic_regression': 200,
        'random_forest': 500,
        'gradient_boosting': 500,
        'xgboost': 500,
        'lightgbm': 500,
        'neural_network': float('inf'),  # Not recommended
        'deep_learning': float('inf')     # Prohibited
    }
    
    required = thresholds.get(ml_type, float('inf'))
    
    if trade_count < required:
        print(f"GATE 1 FAILED: {trade_count} trades < {required} required for {ml_type}")
        return False
    
    return True
```

### 2.4 Gate 2: Feature Stability Analysis

Features must show **stable relationships** with outcomes across time.

```python
def check_gate_2(features: pd.DataFrame, target: pd.Series, 
                 window_size: int = 50, min_stability: float = 0.6) -> dict:
    """
    Check feature stability using rolling correlation analysis.
    
    A feature is stable if its correlation with target doesn't flip signs
    or vary wildly across rolling windows.
    """
    
    stability_results = {}
    
    for col in features.columns:
        rolling_corr = features[col].rolling(window_size).corr(target)
        
        # Stability metrics
        sign_consistency = (rolling_corr > 0).mean()  # % positive
        sign_consistency = max(sign_consistency, 1 - sign_consistency)  # Consistency either direction
        
        std_of_corr = rolling_corr.std()
        mean_abs_corr = rolling_corr.abs().mean()
        
        stability_score = sign_consistency * (1 - std_of_corr) * mean_abs_corr
        
        stability_results[col] = {
            'stability_score': stability_score,
            'sign_consistency': sign_consistency,
            'correlation_std': std_of_corr,
            'mean_abs_correlation': mean_abs_corr,
            'stable': stability_score >= min_stability
        }
    
    # Gate passes if >50% of features are stable
    stable_features = sum(1 for r in stability_results.values() if r['stable'])
    gate_passed = stable_features / len(stability_results) > 0.5
    
    return {
        'feature_stability': stability_results,
        'stable_feature_count': stable_features,
        'total_features': len(stability_results),
        'gate_passed': gate_passed
    }
```

**Unstable Feature Warning Signs:**
- Correlation flips sign across periods
- High variance in rolling correlation
- Strong correlation in one period, near-zero in another
- Correlation only appears in specific market regimes

### 2.5 Gate 3: Train/Validation/Test Split

**CRITICAL: Temporal splits only. No random shuffling.**

```python
def create_temporal_splits(trades: pd.DataFrame) -> dict:
    """
    Create proper temporal train/validation/test splits.
    
    NEVER use random splits for time series data.
    NEVER look at test set during development.
    """
    
    trades = trades.sort_values('entry_date')
    n = len(trades)
    
    # 60/20/20 split
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    splits = {
        'train': trades.iloc[:train_end].copy(),
        'validation': trades.iloc[train_end:val_end].copy(),
        'test': trades.iloc[val_end:].copy(),  # SACRED - DO NOT TOUCH
        'train_period': f"{trades.iloc[0]['entry_date']} to {trades.iloc[train_end-1]['entry_date']}",
        'val_period': f"{trades.iloc[train_end]['entry_date']} to {trades.iloc[val_end-1]['entry_date']}",
        'test_period': f"{trades.iloc[val_end]['entry_date']} to {trades.iloc[-1]['entry_date']}"
    }
    
    print(f"Train: {len(splits['train'])} trades ({splits['train_period']})")
    print(f"Validation: {len(splits['validation'])} trades ({splits['val_period']})")
    print(f"Test: {len(splits['test'])} trades ({splits['test_period']}) - DO NOT USE UNTIL FINAL EVAL")
    
    return splits
```

**Test Set Rules:**
1. Test set is created at the start and NEVER examined
2. No feature engineering decisions based on test set performance
3. No hyperparameter tuning using test set
4. Test set is used ONCE for final evaluation
5. If test set is "peeked," the entire analysis is contaminated

### 2.6 Gate 4: Out-of-Sample Validation

```python
def check_gate_4(train_metrics: dict, val_metrics: dict, 
                 retention_threshold: float = 0.80) -> dict:
    """
    Check if model performance is retained out-of-sample.
    
    Model must retain >80% of training performance on validation set.
    """
    
    metrics_to_check = ['sharpe', 'win_rate', 'profit_factor', 'expectancy']
    
    retention_results = {}
    for metric in metrics_to_check:
        if metric in train_metrics and metric in val_metrics:
            if train_metrics[metric] != 0:
                retention = val_metrics[metric] / train_metrics[metric]
            else:
                retention = 1.0 if val_metrics[metric] >= 0 else 0.0
            
            retention_results[metric] = {
                'train': train_metrics[metric],
                'validation': val_metrics[metric],
                'retention': retention,
                'passed': retention >= retention_threshold
            }
    
    gate_passed = all(r['passed'] for r in retention_results.values())
    
    return {
        'retention_results': retention_results,
        'gate_passed': gate_passed,
        'recommendation': 'PROCEED' if gate_passed else 'DO NOT PROCEED - Overfitting detected'
    }
```

**Red Flags (Gate 4 Failure Indicators):**
- Sharpe drops >30% from train to validation
- Win rate drops >15 percentage points
- Profit factor drops below 1.0 on validation
- Best performing features on train are worst on validation

### 2.7 Gate 5: Paper Trading Validation

**Minimum Requirements:**
- Duration: 6 months minimum
- Trade count: 50+ trades with ML optimization
- Comparison: ML-optimized vs rule-based baseline (A/B test)
- Environment: Realistic execution (Alpaca paper trading)

```python
def check_gate_5(ml_paper_results: pd.DataFrame, 
                 baseline_paper_results: pd.DataFrame,
                 min_improvement: float = 0.10) -> dict:
    """
    Compare ML-optimized approach to rule-based baseline in paper trading.
    
    ML approach must show >10% improvement (after 50% haircut already applied).
    """
    
    ml_sharpe = calculate_sharpe(ml_paper_results)
    baseline_sharpe = calculate_sharpe(baseline_paper_results)
    
    ml_expectancy = calculate_expectancy(ml_paper_results)
    baseline_expectancy = calculate_expectancy(baseline_paper_results)
    
    sharpe_improvement = (ml_sharpe - baseline_sharpe) / baseline_sharpe if baseline_sharpe > 0 else 0
    expectancy_improvement = (ml_expectancy - baseline_expectancy) / baseline_expectancy if baseline_expectancy > 0 else 0
    
    gate_passed = (
        sharpe_improvement >= min_improvement and 
        expectancy_improvement >= min_improvement and
        len(ml_paper_results) >= 50
    )
    
    return {
        'ml_sharpe': ml_sharpe,
        'baseline_sharpe': baseline_sharpe,
        'sharpe_improvement': sharpe_improvement,
        'ml_expectancy': ml_expectancy,
        'baseline_expectancy': baseline_expectancy,
        'expectancy_improvement': expectancy_improvement,
        'trade_count': len(ml_paper_results),
        'gate_passed': gate_passed,
        'recommendation': 'DEPLOY TO LIVE' if gate_passed else 'CONTINUE PAPER TRADING'
    }
```

---

## 3. Legitimate vs Illegitimate ML Use Cases

### 3.1 Legitimate Use Cases (APPROVED)

| Use Case | Description | Why It's Safe |
|----------|-------------|---------------|
| **Strike/Delta Optimization** | Given a validated signal, predict optimal delta | Not generating signals; optimizing secondary parameter |
| **DTE Selection** | Predict time-to-magnitude for DTE selection | Wrong prediction affects cost, not entry/exit |
| **Position Sizing** | Adjust size based on setup quality score | Reduces risk on lower-quality setups |
| **IV Environment Classification** | Identify high/low IV regimes for premium decisions | Context layer, not signal layer |
| **Magnitude Estimation** | Predict expected move size (not direction) | Informs strike distance, not trade direction |

### 3.2 Illegitimate Use Cases (PROHIBITED)

| Use Case | Description | Why It's Dangerous |
|----------|-------------|-------------------|
| **Signal Generation** | ML decides when to enter trades | Replaces economic logic with statistical correlation |
| **Pattern Classification** | ML classifies bar types or patterns | STRAT bar classification is deterministic, not probabilistic |
| **Direction Prediction** | ML predicts if price goes up/down | Massive overfitting risk, no structural edge |
| **Timeframe Continuity** | ML determines if continuity exists | Binary rule-based check, not probabilistic |
| **Entry/Exit Timing** | ML decides exact entry/exit points | Transaction costs and slippage overwhelm any "edge" |
| **Feature Discovery** | ML finds new features automatically | Data mining without economic reasoning |

### 3.3 Decision Framework

```
When considering ML for a new application, ask:

1. Is the base edge already validated WITHOUT ML?
   - NO  → STOP. Validate base edge first.
   - YES → Continue to question 2.

2. Does this application generate trading signals?
   - YES → PROHIBITED. ML cannot generate signals.
   - NO  → Continue to question 3.

3. Does this application have a clear economic rationale?
   - NO  → PROHIBITED. No black-box optimization.
   - YES → Continue to question 4.

4. What is the cost of ML being wrong?
   - Trade fails completely → PROHIBITED. Too much risk.
   - Trade is suboptimal but still profitable → ALLOWED with validation.
   - Minor efficiency loss → ALLOWED with validation.

5. Do we have sufficient sample size? (See Gate 1)
   - NO  → WAIT. Collect more data.
   - YES → PROCEED with full validation protocol.
```

---

## 4. Sample Size Requirements

### 4.1 Minimum Samples by ML Type

| ML Approach | Minimum Trades | Features Allowed | Use Case |
|-------------|---------------|------------------|----------|
| Frequency estimation | 50-100 | N/A | Base rates by category |
| Simple linear regression | 200 | 3-5 | Magnitude estimation |
| Logistic regression | 200 | 3-5 | Binary classification |
| Ridge/Lasso regression | 300 | 5-10 | Regularized estimation |
| Random Forest | 500 | 10-20 | Setup quality scoring |
| Gradient Boosting | 500 | 10-20 | Delta optimization |
| XGBoost/LightGBM | 500 | 10-20 | Multi-parameter optimization |
| Neural Networks | 5,000+ | Many | NOT RECOMMENDED |
| Deep Learning | 50,000+ | Many | PROHIBITED |

### 4.2 Sample Size Formula

For classification problems:
```
Minimum samples ≥ 10 × (number of features) × (number of classes)
```

For regression problems:
```
Minimum samples ≥ 20 × (number of features)
```

**Example:**
- Delta optimization with 8 features
- Minimum: 20 × 8 = 160 trades per delta bucket
- With 5 delta buckets: 160 × 5 = 800 total trades recommended

### 4.3 Feature-to-Sample Ratio

**Rule of Thumb:** Never use more than `n_samples / 20` features.

```python
def calculate_max_features(n_samples: int, ml_type: str) -> int:
    """Calculate maximum features allowed for sample size."""
    
    ratios = {
        'linear_regression': 20,
        'logistic_regression': 20,
        'ridge_lasso': 15,
        'random_forest': 25,
        'gradient_boosting': 25,
        'xgboost': 25
    }
    
    ratio = ratios.get(ml_type, 20)
    max_features = n_samples // ratio
    
    return max(1, max_features)  # At least 1 feature
```

---

## 5. Validation Protocol

### 5.1 Validation Hierarchy

```
Level 1: In-Sample (Training)
    → Baseline performance, used for model fitting
    → NEVER use for final evaluation
    
Level 2: Out-of-Sample (Validation)
    → Hyperparameter tuning
    → Feature selection
    → Model comparison
    
Level 3: Holdout (Test)
    → Final evaluation ONLY
    → Used exactly ONCE
    → If peeked, analysis is contaminated
    
Level 4: Paper Trading
    → Real-time validation
    → 6 months minimum
    → A/B comparison with baseline
    
Level 5: Live Trading (Gradual)
    → Small position sizes initially
    → Scale up only after confirmed performance
```

### 5.2 Cross-Validation for Time Series

**NEVER use standard k-fold cross-validation for trading data.**

Use **Walk-Forward Validation** instead:

```python
def walk_forward_validation(trades: pd.DataFrame, 
                            train_window: int = 100,
                            test_window: int = 20,
                            step_size: int = 20) -> list:
    """
    Walk-forward validation for time series trading data.
    
    Trains on expanding window, tests on next period.
    """
    
    trades = trades.sort_values('entry_date').reset_index(drop=True)
    n = len(trades)
    
    results = []
    
    start_idx = 0
    while start_idx + train_window + test_window <= n:
        # Training set: trades 0 to start_idx + train_window
        train = trades.iloc[:start_idx + train_window]
        
        # Test set: next test_window trades
        test_start = start_idx + train_window
        test_end = test_start + test_window
        test = trades.iloc[test_start:test_end]
        
        # Train model on train, evaluate on test
        model = train_model(train)
        metrics = evaluate_model(model, test)
        
        results.append({
            'train_end_date': train.iloc[-1]['entry_date'],
            'test_start_date': test.iloc[0]['entry_date'],
            'test_end_date': test.iloc[-1]['entry_date'],
            'train_size': len(train),
            'test_size': len(test),
            'metrics': metrics
        })
        
        start_idx += step_size
    
    return results
```

### 5.3 Purging and Embargo

To prevent leakage between train and test:

```python
def create_purged_splits(trades: pd.DataFrame, 
                         purge_days: int = 5,
                         embargo_days: int = 2) -> dict:
    """
    Create splits with purging and embargo to prevent leakage.
    
    Purge: Remove trades within purge_days of split boundary (prevent lookahead)
    Embargo: Don't use trades within embargo_days after split (prevent momentum leakage)
    """
    
    trades = trades.sort_values('entry_date')
    n = len(trades)
    
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    # Get boundary dates
    train_end_date = trades.iloc[train_end]['entry_date']
    val_end_date = trades.iloc[val_end]['entry_date']
    
    # Apply purging: remove trades too close to boundary
    purge_mask_train = trades['exit_date'] < (train_end_date - pd.Timedelta(days=purge_days))
    purge_mask_val = (
        (trades['entry_date'] > train_end_date + pd.Timedelta(days=embargo_days)) &
        (trades['exit_date'] < val_end_date - pd.Timedelta(days=purge_days))
    )
    purge_mask_test = trades['entry_date'] > val_end_date + pd.Timedelta(days=embargo_days)
    
    return {
        'train': trades[purge_mask_train].copy(),
        'validation': trades[purge_mask_val].copy(),
        'test': trades[purge_mask_test].copy()
    }
```

---

## 6. Overfitting Prevention

### 6.1 The 50% Haircut Rule

**Any ML improvement must be haircut by 50% for deployment expectations.**

```python
def apply_haircut(backtest_improvement: float, haircut: float = 0.50) -> float:
    """
    Apply overfitting haircut to backtest improvement.
    
    If backtest shows +20% improvement, expect +10% live.
    """
    return backtest_improvement * (1 - haircut)


# Example
backtest_sharpe_improvement = 0.30  # 30% improvement in backtest
expected_live_improvement = apply_haircut(backtest_sharpe_improvement)  # 15% expected
```

### 6.2 Regularization Requirements

All ML models MUST use regularization:

| Model Type | Required Regularization |
|------------|------------------------|
| Linear Regression | Ridge (L2) or Lasso (L1) |
| Logistic Regression | L2 penalty, C < 1.0 |
| Random Forest | max_depth ≤ 5, min_samples_leaf ≥ 20 |
| Gradient Boosting | learning_rate ≤ 0.1, max_depth ≤ 3 |
| XGBoost | reg_alpha > 0, reg_lambda > 0 |

```python
# CORRECT: Regularized model
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # Regularization on

# WRONG: Unregularized model
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # No regularization - PROHIBITED
```

### 6.3 Feature Selection Discipline

**Maximum features = min(n_samples / 20, 10)**

```python
def select_features_disciplined(X: pd.DataFrame, y: pd.Series, 
                                 max_features: int = 10) -> list:
    """
    Disciplined feature selection with economic reasoning requirement.
    
    Each selected feature must have documented economic rationale.
    """
    
    n_samples = len(X)
    feature_limit = min(n_samples // 20, max_features)
    
    # Calculate feature importance
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    # Select top features up to limit
    selected = importances.head(feature_limit).index.tolist()
    
    print(f"Sample size: {n_samples}")
    print(f"Feature limit: {feature_limit}")
    print(f"Selected features: {selected}")
    print("\nREQUIRED: Document economic rationale for each selected feature before proceeding.")
    
    return selected
```

### 6.4 Complexity Budget

**Total model complexity must stay within budget:**

```
Complexity Score = (n_features × feature_weight) + (n_hyperparameters × hp_weight) + (model_depth × depth_weight)

Budget Limits:
- Simple optimization: Score ≤ 20
- Moderate optimization: Score ≤ 50
- Complex optimization: Score ≤ 100 (requires additional validation)
```

```python
def calculate_complexity_score(n_features: int, n_hyperparameters: int, 
                                model_depth: int) -> dict:
    """Calculate complexity score and check against budget."""
    
    feature_weight = 2
    hp_weight = 1
    depth_weight = 3
    
    score = (n_features * feature_weight) + \
            (n_hyperparameters * hp_weight) + \
            (model_depth * depth_weight)
    
    if score <= 20:
        category = 'SIMPLE'
        recommendation = 'Standard validation sufficient'
    elif score <= 50:
        category = 'MODERATE'
        recommendation = 'Extended validation recommended'
    elif score <= 100:
        category = 'COMPLEX'
        recommendation = 'Additional validation required, increased haircut (60%)'
    else:
        category = 'EXCESSIVE'
        recommendation = 'REDUCE COMPLEXITY - Too high risk of overfitting'
    
    return {
        'score': score,
        'category': category,
        'recommendation': recommendation,
        'proceed': score <= 100
    }
```

---

## 7. Approved ML Applications

### 7.1 Application 1: Delta/Strike Optimization

**Objective:** Given a validated STRAT signal, predict optimal delta for options positioning.

**Inputs (Features):**
- IV percentile (current vs 30-day range)
- IV rank (current vs 52-week range)
- Days to expiration
- ATR percentile
- Bid-ask spread at different strikes
- Historical time-to-magnitude for this pattern

**Output:** Recommended delta bucket (0.40-0.50, 0.50-0.65, 0.65-0.80)

**Model Type:** Gradient boosting classifier or ordinal regression

**Validation Requirement:** 
- 500+ trades minimum
- Walk-forward validation
- Must beat naive "always use 0.65 delta" baseline

```python
class DeltaOptimizer:
    """
    ML-based delta optimization for STRAT options trades.
    
    ONLY to be used after base STRAT edge is validated.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'iv_percentile_30d',
            'iv_rank_52w', 
            'dte',
            'atr_percentile',
            'pattern_type_encoded',
            'continuity_strength',
            'historical_ttm_median'
        ]
        self.gate_status = {
            'gate_0': False,
            'gate_1': False,
            'gate_2': False,
            'gate_3': False,
            'gate_4': False,
            'gate_5': False
        }
    
    def check_gates(self, trade_data: pd.DataFrame) -> bool:
        """Verify all gates are passed before training."""
        # Implementation checks all gates
        # Returns False if any gate fails
        pass
    
    def train(self, trade_data: pd.DataFrame) -> None:
        """Train delta optimization model."""
        if not self.check_gates(trade_data):
            raise ValueError("Gates not passed. Cannot train ML model.")
        
        # Training implementation
        pass
    
    def predict(self, features: pd.DataFrame) -> str:
        """Predict optimal delta bucket."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return self.model.predict(features)
```

### 7.2 Application 2: DTE Selection

**Objective:** Predict time-to-magnitude to optimize DTE selection.

**Inputs (Features):**
- Pattern type (3-1-2, 2-1-2, 2-2)
- Continuity strength (3/5, 4/5, 5/5)
- ATR as % of price
- Day of week
- Time of day (for intraday patterns)
- Distance from key level (monthly open, etc.)

**Output:** Expected days to magnitude (regression) or DTE bucket (classification)

**Model Type:** Random forest regressor or gradient boosting regressor

**Validation Requirement:**
- 300+ trades minimum
- MAE < 2 days on validation set
- Must improve DTE efficiency vs static selection

### 7.3 Application 3: Position Sizing

**Objective:** Adjust position size based on setup quality score.

**Inputs (Features):**
- Continuity strength
- Pattern type strength ranking
- Volume confirmation level
- ATR environment
- IV environment
- Day of week/month patterns

**Output:** Position size multiplier (0.5x, 1.0x, 1.5x)

**Model Type:** Ordinal regression or random forest classifier

**Validation Requirement:**
- 500+ trades minimum
- Risk-adjusted returns must improve
- Drawdown must not increase significantly

---

## 8. Prohibited ML Applications

### 8.1 Signal Generation (PROHIBITED)

```python
# PROHIBITED - DO NOT IMPLEMENT
class SignalGenerator:
    """
    ML-based signal generation.
    
    THIS IS PROHIBITED. STRAT signals must be rule-based.
    """
    
    def predict_entry(self, market_data):
        raise NotImplementedError(
            "ML signal generation is PROHIBITED. "
            "STRAT edge comes from economic reasoning, not statistical patterns. "
            "Use rule-based signal generation instead."
        )
```

### 8.2 Direction Prediction (PROHIBITED)

```python
# PROHIBITED - DO NOT IMPLEMENT
class DirectionPredictor:
    """
    ML-based direction prediction.
    
    THIS IS PROHIBITED. Direction comes from STRAT continuity analysis.
    """
    
    def predict_direction(self, features):
        raise NotImplementedError(
            "ML direction prediction is PROHIBITED. "
            "Direction is determined by timeframe continuity analysis. "
            "This is a rule-based check, not a probabilistic prediction."
        )
```

### 8.3 Feature Discovery (PROHIBITED)

```python
# PROHIBITED - DO NOT IMPLEMENT
class AutoFeatureDiscovery:
    """
    Automated feature discovery/engineering.
    
    THIS IS PROHIBITED. All features must have economic rationale.
    """
    
    def discover_features(self, data):
        raise NotImplementedError(
            "Automated feature discovery is PROHIBITED. "
            "All features must have documented economic rationale. "
            "Data mining without theory leads to spurious correlations."
        )
```

---

## 9. Implementation Checklist

### 9.1 Pre-Implementation Checklist

Before writing ANY ML code, verify:

```
[ ] Gate 0 passed: Base edge validated without ML
[ ] Gate 1 passed: Sufficient sample size for ML type
[ ] Use case is in APPROVED list (Section 7)
[ ] Use case is NOT in PROHIBITED list (Section 8)
[ ] Economic rationale documented for all features
[ ] Train/validation/test splits defined (temporal, not random)
[ ] Test set quarantined and NOT examined
[ ] Regularization method selected
[ ] Complexity budget calculated and within limits
[ ] Baseline model defined for comparison
[ ] Success criteria defined before training
```

### 9.2 During-Implementation Checklist

During model development:

```
[ ] Using only approved features (documented rationale)
[ ] Regularization applied to all models
[ ] Walk-forward validation implemented
[ ] Purging and embargo applied to prevent leakage
[ ] Tracking train vs validation performance for overfit detection
[ ] Model complexity within budget
[ ] NO peeking at test set
[ ] Comparing to baseline at each iteration
```

### 9.3 Post-Implementation Checklist

After model development:

```
[ ] Gate 2 passed: Feature stability confirmed
[ ] Gate 3 passed: Proper splits verified
[ ] Gate 4 passed: Out-of-sample retention >80%
[ ] Test set evaluation (ONCE ONLY)
[ ] 50% haircut applied to expected improvement
[ ] Paper trading plan defined (6 months, 50+ trades)
[ ] A/B comparison with baseline planned
[ ] Monitoring and maintenance plan documented
[ ] Rollback procedure defined
```

---

## 10. Model Selection Guidelines

### 10.1 Model Hierarchy (Prefer Simpler)

```
PREFERENCE ORDER (most preferred first):

1. Linear/Logistic Regression with regularization
   - Most interpretable
   - Least overfit risk
   - Use when relationships are roughly linear

2. Ridge/Lasso Regression
   - Built-in feature selection (Lasso)
   - Good for moderate feature counts
   - Interpretable coefficients

3. Random Forest (shallow)
   - max_depth ≤ 5
   - min_samples_leaf ≥ 20
   - Good for non-linear relationships
   - Feature importance available

4. Gradient Boosting (conservative)
   - learning_rate ≤ 0.1
   - max_depth ≤ 3
   - n_estimators ≤ 100
   - Best predictive performance (with overfit risk)

5. XGBoost/LightGBM (heavily regularized)
   - Only if simpler models inadequate
   - Requires extensive regularization
   - Highest overfit risk

PROHIBITED:
- Neural Networks (insufficient data)
- Deep Learning (insufficient data)
- Unregularized models
- Ensemble stacking (too complex)
```

### 10.2 Hyperparameter Constraints

```python
# Mandatory constraints for each model type

HYPERPARAMETER_CONSTRAINTS = {
    'ridge': {
        'alpha': {'min': 0.1, 'max': 10.0}  # Regularization strength
    },
    'lasso': {
        'alpha': {'min': 0.01, 'max': 1.0}
    },
    'logistic': {
        'C': {'min': 0.1, 'max': 1.0},  # Inverse regularization
        'penalty': 'l2'
    },
    'random_forest': {
        'max_depth': {'min': 2, 'max': 5},
        'min_samples_leaf': {'min': 20, 'max': 50},
        'n_estimators': {'min': 50, 'max': 200}
    },
    'gradient_boosting': {
        'learning_rate': {'min': 0.01, 'max': 0.1},
        'max_depth': {'min': 2, 'max': 3},
        'n_estimators': {'min': 50, 'max': 100},
        'min_samples_leaf': {'min': 20, 'max': 50}
    },
    'xgboost': {
        'learning_rate': {'min': 0.01, 'max': 0.1},
        'max_depth': {'min': 2, 'max': 3},
        'reg_alpha': {'min': 0.1, 'max': 1.0},  # L1
        'reg_lambda': {'min': 1.0, 'max': 10.0}  # L2
    }
}
```

---

## 11. Feature Engineering Rules

### 11.1 Approved Feature Categories

| Category | Examples | Rationale |
|----------|----------|-----------|
| **Volatility** | IV percentile, IV rank, ATR percentile | Affects options pricing and magnitude |
| **Time** | DTE, day of week, time of month | Market structure patterns |
| **Pattern Quality** | Continuity strength, pattern type | Setup quality indicators |
| **Liquidity** | Bid-ask spread, volume ratio | Execution quality |
| **Market Context** | VIX level, regime state | Broad environment |

### 11.2 Feature Documentation Template

**Every feature must have this documentation before use:**

```python
FEATURE_DOCUMENTATION = {
    'iv_percentile_30d': {
        'description': 'Current IV as percentile of 30-day IV range',
        'economic_rationale': 'High IV = expensive options = lower reward/risk. '
                              'May favor ITM strikes. Low IV = cheap options = '
                              'may favor OTM strikes.',
        'expected_relationship': 'Higher IV → prefer higher delta (ITM)',
        'data_source': 'ThetaData options chain',
        'calculation': '(current_iv - min_iv_30d) / (max_iv_30d - min_iv_30d)',
        'potential_issues': 'IV surface complexity not captured by single number'
    },
    # ... more features
}
```

### 11.3 Prohibited Feature Types

| Feature Type | Why Prohibited |
|--------------|----------------|
| Price levels (raw) | Non-stationary, causes lookahead |
| Technical indicators without lag | Lookahead bias |
| Future-derived features | Obvious lookahead |
| Unnamed engineered features | No economic rationale |
| Features from test period | Data leakage |

---

## 12. Performance Evaluation

### 12.1 Required Metrics

```python
def evaluate_ml_model(predictions: pd.Series, actuals: pd.Series,
                      trade_results: pd.DataFrame) -> dict:
    """
    Complete ML model evaluation for trading application.
    
    Must evaluate both ML metrics AND trading metrics.
    """
    
    # ML Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    ml_metrics = {
        'accuracy': accuracy_score(actuals, predictions),
        'precision': precision_score(actuals, predictions, average='weighted'),
        'recall': recall_score(actuals, predictions, average='weighted'),
        'f1': f1_score(actuals, predictions, average='weighted')
    }
    
    # Trading Metrics (what actually matters)
    trading_metrics = {
        'total_pnl': trade_results['pnl'].sum(),
        'sharpe_ratio': calculate_sharpe(trade_results['pnl']),
        'sortino_ratio': calculate_sortino(trade_results['pnl']),
        'win_rate': (trade_results['pnl'] > 0).mean(),
        'profit_factor': calculate_profit_factor(trade_results['pnl']),
        'max_drawdown': calculate_max_drawdown(trade_results['pnl'].cumsum()),
        'expectancy': calculate_expectancy(trade_results)
    }
    
    # Comparison to baseline
    comparison = {
        'vs_baseline_sharpe': trading_metrics['sharpe_ratio'] - baseline_sharpe,
        'vs_baseline_expectancy': trading_metrics['expectancy'] - baseline_expectancy,
        'improvement_after_haircut': apply_haircut(trading_metrics['sharpe_ratio'] - baseline_sharpe)
    }
    
    return {
        'ml_metrics': ml_metrics,
        'trading_metrics': trading_metrics,
        'comparison': comparison,
        'recommendation': 'DEPLOY' if comparison['improvement_after_haircut'] > 0.10 else 'REJECT'
    }
```

### 12.2 Minimum Performance Thresholds

| Metric | Minimum Threshold | Rationale |
|--------|-------------------|-----------|
| Out-of-sample retention | >80% of train performance | Overfit detection |
| Sharpe improvement (after haircut) | >0.10 | Meaningful improvement |
| Win rate change | Not worse than baseline | Don't sacrifice wins |
| Max drawdown change | Not >10% worse | Risk control |
| Trade count impact | >80% of baseline trades | Don't filter too aggressively |

---

## 13. Deployment Protocol

### 13.1 Staged Deployment

```
Stage 1: Shadow Mode (2 weeks)
    - ML model runs but doesn't affect trades
    - Log predictions vs rule-based decisions
    - Monitor for anomalies
    
Stage 2: Paper Trading (6 months)
    - ML model affects paper trades
    - A/B comparison with baseline
    - Full performance tracking
    
Stage 3: Limited Live (3 months)
    - 25% of live capital uses ML optimization
    - 75% uses baseline
    - Compare real-money performance
    
Stage 4: Full Deployment (if Stage 3 passes)
    - ML optimization on all trades
    - Continue monitoring
    - Rollback plan ready
```

### 13.2 Rollback Triggers

**Immediately rollback to baseline if:**

```python
ROLLBACK_TRIGGERS = {
    'sharpe_degradation': -0.20,      # Sharpe drops >20% vs baseline
    'drawdown_increase': 0.15,         # Max DD increases >15 percentage points
    'win_rate_drop': -0.10,            # Win rate drops >10 percentage points
    'consecutive_losses': 5,           # 5+ consecutive losses
    'model_prediction_drift': 0.30,    # Predictions shift >30% from training distribution
    'feature_distribution_shift': 0.25 # Input features shift >25% from training
}

def check_rollback_triggers(current_metrics: dict, baseline_metrics: dict,
                            model_diagnostics: dict) -> bool:
    """Check if rollback should be triggered."""
    
    triggers_hit = []
    
    # Performance degradation
    sharpe_change = current_metrics['sharpe'] - baseline_metrics['sharpe']
    if sharpe_change < ROLLBACK_TRIGGERS['sharpe_degradation']:
        triggers_hit.append(f"Sharpe degradation: {sharpe_change:.2%}")
    
    # Continue checking other triggers...
    
    if triggers_hit:
        print("ROLLBACK TRIGGERED:")
        for trigger in triggers_hit:
            print(f"  - {trigger}")
        return True
    
    return False
```

---

## 14. Monitoring and Maintenance

### 14.1 Daily Monitoring

```python
def daily_ml_monitoring(model, recent_trades: pd.DataFrame) -> dict:
    """Daily monitoring checklist for ML model."""
    
    monitoring_results = {
        'date': pd.Timestamp.now().date(),
        'checks': {}
    }
    
    # 1. Prediction distribution
    recent_predictions = model.predict(recent_trades[model.feature_names])
    monitoring_results['checks']['prediction_distribution'] = {
        'mean': recent_predictions.mean(),
        'std': recent_predictions.std(),
        'drift_from_training': calculate_distribution_drift(recent_predictions, training_predictions)
    }
    
    # 2. Feature distributions
    for feature in model.feature_names:
        drift = calculate_feature_drift(
            recent_trades[feature], 
            training_data[feature]
        )
        monitoring_results['checks'][f'feature_drift_{feature}'] = drift
    
    # 3. Performance metrics (rolling 20 trades)
    if len(recent_trades) >= 20:
        rolling_sharpe = calculate_sharpe(recent_trades.tail(20)['pnl'])
        monitoring_results['checks']['rolling_sharpe'] = rolling_sharpe
    
    # 4. Error rate
    if 'actual_outcome' in recent_trades.columns:
        accuracy = (recent_trades['prediction'] == recent_trades['actual_outcome']).mean()
        monitoring_results['checks']['recent_accuracy'] = accuracy
    
    return monitoring_results
```

### 14.2 Weekly Review

```
WEEKLY ML REVIEW CHECKLIST:

[ ] Performance vs baseline (cumulative since deployment)
[ ] Feature importance stability (compare to training)
[ ] Prediction confidence distribution
[ ] Edge cases and failures analysis
[ ] Market regime changes assessment
[ ] Data quality issues
[ ] Any rollback triggers approached (even if not hit)
```

### 14.3 Monthly Recalibration Assessment

```python
def monthly_recalibration_assessment(model, monthly_trades: pd.DataFrame,
                                      training_data: pd.DataFrame) -> dict:
    """Assess whether model recalibration is needed."""
    
    assessment = {
        'month': pd.Timestamp.now().strftime('%Y-%m'),
        'recalibration_recommended': False,
        'reasons': []
    }
    
    # 1. Performance decay
    monthly_sharpe = calculate_sharpe(monthly_trades['pnl'])
    training_sharpe = calculate_sharpe(training_data['pnl'])
    
    if monthly_sharpe < training_sharpe * 0.70:  # >30% decay
        assessment['recalibration_recommended'] = True
        assessment['reasons'].append(f"Performance decay: {monthly_sharpe:.2f} vs {training_sharpe:.2f}")
    
    # 2. Feature drift
    for feature in model.feature_names:
        drift = calculate_feature_drift(monthly_trades[feature], training_data[feature])
        if drift > 0.25:  # >25% drift
            assessment['recalibration_recommended'] = True
            assessment['reasons'].append(f"Feature drift ({feature}): {drift:.2%}")
    
    # 3. Time since last calibration
    months_since_calibration = (pd.Timestamp.now() - model.last_calibration_date).days / 30
    if months_since_calibration > 6:
        assessment['recalibration_recommended'] = True
        assessment['reasons'].append(f"Time since calibration: {months_since_calibration:.1f} months")
    
    return assessment
```

### 14.4 Recalibration Protocol

**When recalibration is needed:**

1. **DO NOT** simply retrain on all available data
2. **DO** add new data to training set (expanding window)
3. **DO** re-run full validation protocol (Gates 2-4)
4. **DO** compare new model to current model on recent holdout
5. **DO** deploy via staged protocol (shadow → paper → limited → full)

```python
def recalibrate_model(model, new_data: pd.DataFrame) -> dict:
    """
    Recalibrate model with new data following full protocol.
    """
    
    # 1. Expand training data
    expanded_training = pd.concat([model.original_training_data, new_data])
    
    # 2. Create new splits
    splits = create_temporal_splits(expanded_training)
    
    # 3. Train new model
    new_model = model.__class__()  # New instance
    new_model.train(splits['train'])
    
    # 4. Validate (Gates 2-4)
    gate_2_result = check_gate_2(splits['train'][model.feature_names], splits['train']['target'])
    gate_4_result = check_gate_4(
        evaluate_model(new_model, splits['train']),
        evaluate_model(new_model, splits['validation'])
    )
    
    # 5. Compare to current model
    current_val_metrics = evaluate_model(model, splits['validation'])
    new_val_metrics = evaluate_model(new_model, splits['validation'])
    
    improvement = new_val_metrics['sharpe'] - current_val_metrics['sharpe']
    
    return {
        'new_model': new_model if improvement > 0 else None,
        'improvement': improvement,
        'gate_2_passed': gate_2_result['gate_passed'],
        'gate_4_passed': gate_4_result['gate_passed'],
        'recommendation': 'DEPLOY_NEW' if improvement > 0 and gate_4_result['gate_passed'] else 'KEEP_CURRENT'
    }
```

---

## Appendix A: Code Templates

### A.1 Complete ML Pipeline Template

```python
"""
ML Pipeline Template for STRAT Options Optimization

Usage:
    1. Verify all gates passed
    2. Instantiate pipeline with trade data
    3. Run training
    4. Validate
    5. Deploy via staged protocol
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional


class StratMLPipeline:
    """
    Complete ML pipeline for STRAT options optimization.
    
    Follows all guidelines from ML_IMPLEMENTATION_GUIDE_STRAT.md
    """
    
    def __init__(self, application: str = 'delta_optimization'):
        """
        Initialize pipeline.
        
        Args:
            application: One of 'delta_optimization', 'dte_selection', 'position_sizing'
        """
        self.application = application
        self.model = None
        self.feature_names = []
        self.gate_results = {}
        self.training_metrics = {}
        self.validation_metrics = {}
        self.is_deployed = False
        
        # Validate application is approved
        approved_applications = ['delta_optimization', 'dte_selection', 'position_sizing']
        if application not in approved_applications:
            raise ValueError(f"Application '{application}' not in approved list: {approved_applications}")
    
    def check_all_gates(self, trade_data: pd.DataFrame) -> Dict:
        """Check all gates before training."""
        
        results = {
            'gate_0': self._check_gate_0(trade_data),
            'gate_1': self._check_gate_1(trade_data),
            'gate_2': None,  # Checked after feature engineering
            'gate_3': None,  # Checked after splits
            'gate_4': None,  # Checked after training
            'gate_5': None   # Checked after paper trading
        }
        
        # Can only proceed if Gates 0 and 1 pass
        results['can_proceed'] = results['gate_0']['passed'] and results['gate_1']['passed']
        
        self.gate_results = results
        return results
    
    def _check_gate_0(self, trade_data: pd.DataFrame) -> Dict:
        """Gate 0: Base edge validation."""
        
        sharpe = self._calculate_sharpe(trade_data['pnl'])
        expectancy = self._calculate_expectancy(trade_data)
        win_rate = (trade_data['pnl'] > 0).mean()
        
        passed = (
            len(trade_data) >= 100 and
            sharpe > 0 and
            expectancy > 0 and
            trade_data['symbol'].nunique() >= 3
        )
        
        return {
            'passed': passed,
            'trade_count': len(trade_data),
            'sharpe': sharpe,
            'expectancy': expectancy,
            'win_rate': win_rate,
            'symbols': trade_data['symbol'].nunique()
        }
    
    def _check_gate_1(self, trade_data: pd.DataFrame) -> Dict:
        """Gate 1: Sample size threshold."""
        
        thresholds = {
            'delta_optimization': 500,
            'dte_selection': 300,
            'position_sizing': 500
        }
        
        required = thresholds.get(self.application, 500)
        passed = len(trade_data) >= required
        
        return {
            'passed': passed,
            'trade_count': len(trade_data),
            'required': required
        }
    
    def train(self, trade_data: pd.DataFrame, features: List[str]) -> Dict:
        """
        Train ML model following full protocol.
        
        Args:
            trade_data: DataFrame with trades and features
            features: List of feature column names (must have documented rationale)
        """
        
        # Verify gates
        if not self.gate_results.get('can_proceed', False):
            raise ValueError("Gates 0 and 1 must pass before training. Run check_all_gates() first.")
        
        # Verify feature count within limits
        max_features = min(len(trade_data) // 20, 10)
        if len(features) > max_features:
            raise ValueError(f"Too many features: {len(features)} > {max_features} allowed")
        
        self.feature_names = features
        
        # Create temporal splits
        splits = self._create_temporal_splits(trade_data)
        
        # Gate 3: Verify splits
        self.gate_results['gate_3'] = {
            'passed': True,
            'train_size': len(splits['train']),
            'val_size': len(splits['validation']),
            'test_size': len(splits['test'])
        }
        
        # Check Gate 2: Feature stability
        self.gate_results['gate_2'] = self._check_feature_stability(
            splits['train'][features], 
            splits['train']['target']
        )
        
        if not self.gate_results['gate_2']['passed']:
            print("WARNING: Gate 2 (feature stability) failed. Proceeding with caution.")
        
        # Train model with regularization
        X_train = splits['train'][features]
        y_train = splits['train']['target']
        
        self.model = GradientBoostingClassifier(
            learning_rate=0.05,  # Conservative
            max_depth=3,         # Shallow
            n_estimators=100,
            min_samples_leaf=20,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on train and validation
        self.training_metrics = self._evaluate(splits['train'], features)
        self.validation_metrics = self._evaluate(splits['validation'], features)
        
        # Gate 4: Out-of-sample retention
        retention = self.validation_metrics['sharpe'] / self.training_metrics['sharpe'] \
                    if self.training_metrics['sharpe'] > 0 else 0
        
        self.gate_results['gate_4'] = {
            'passed': retention >= 0.80,
            'retention': retention,
            'train_sharpe': self.training_metrics['sharpe'],
            'val_sharpe': self.validation_metrics['sharpe']
        }
        
        return {
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'gate_results': self.gate_results,
            'can_deploy_to_paper': self.gate_results['gate_4']['passed']
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        
        if self.model is None:
            raise ValueError("Model not trained. Run train() first.")
        
        return self.model.predict(features[self.feature_names])
    
    def _create_temporal_splits(self, data: pd.DataFrame) -> Dict:
        """Create temporal train/val/test splits."""
        
        data = data.sort_values('entry_date')
        n = len(data)
        
        train_end = int(n * 0.60)
        val_end = int(n * 0.80)
        
        return {
            'train': data.iloc[:train_end].copy(),
            'validation': data.iloc[train_end:val_end].copy(),
            'test': data.iloc[val_end:].copy()
        }
    
    def _check_feature_stability(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Check feature stability via rolling correlation."""
        
        stable_count = 0
        for col in X.columns:
            rolling_corr = X[col].rolling(50).corr(y)
            sign_consistency = max((rolling_corr > 0).mean(), (rolling_corr < 0).mean())
            if sign_consistency > 0.6:
                stable_count += 1
        
        return {
            'passed': stable_count / len(X.columns) > 0.5,
            'stable_features': stable_count,
            'total_features': len(X.columns)
        }
    
    def _evaluate(self, data: pd.DataFrame, features: List[str]) -> Dict:
        """Evaluate model on data."""
        
        predictions = self.model.predict(data[features])
        # Implementation depends on specific application
        # Return sharpe, win_rate, expectancy, etc.
        
        return {
            'sharpe': 0.0,  # Calculate actual
            'win_rate': 0.0,
            'expectancy': 0.0
        }
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_expectancy(self, trades: pd.DataFrame) -> float:
        """Calculate expectancy per trade."""
        return trades['pnl'].mean()
```

---

## Appendix B: Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║                    ML IMPLEMENTATION QUICK REFERENCE              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  GATES (ALL MUST PASS):                                          ║
║  ├─ Gate 0: Base edge validated (100+ trades, Sharpe > 0)        ║
║  ├─ Gate 1: Sample size (200-500+ depending on ML type)          ║
║  ├─ Gate 2: Feature stability (>50% features stable)             ║
║  ├─ Gate 3: Proper temporal splits (60/20/20)                    ║
║  ├─ Gate 4: OOS retention (>80% of train performance)            ║
║  └─ Gate 5: Paper trading (6 months, 50+ trades)                 ║
║                                                                  ║
║  APPROVED USES:                                                  ║
║  ✓ Delta/strike optimization                                     ║
║  ✓ DTE selection                                                 ║
║  ✓ Position sizing                                               ║
║  ✓ Magnitude estimation (not direction)                          ║
║                                                                  ║
║  PROHIBITED USES:                                                ║
║  ✗ Signal generation                                             ║
║  ✗ Direction prediction                                          ║
║  ✗ Entry/exit timing                                             ║
║  ✗ Feature discovery                                             ║
║  ✗ Neural networks / deep learning                               ║
║                                                                  ║
║  KEY RULES:                                                      ║
║  • 50% haircut on all backtest improvements                      ║
║  • Max features = min(n_samples/20, 10)                          ║
║  • All features need documented economic rationale               ║
║  • Test set touched ONCE ONLY at final evaluation                ║
║  • Regularization MANDATORY for all models                       ║
║                                                                  ║
║  SAMPLE SIZE MINIMUMS:                                           ║
║  • Simple stats: 100                                             ║
║  • Regression: 200                                               ║
║  • Random Forest: 500                                            ║
║  • Gradient Boosting: 500                                        ║
║  • Neural Networks: NOT RECOMMENDED                              ║
║                                                                  ║
║  ROLLBACK TRIGGERS:                                              ║
║  • Sharpe drops >20% vs baseline                                 ║
║  • Max DD increases >15 percentage points                        ║
║  • Win rate drops >10 percentage points                          ║
║  • 5+ consecutive losses                                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-02 | Claude | Initial creation |

**Review Schedule:** Quarterly or after significant system changes

**Related Documents:**
- ATLAS_SYSTEM_STATUS_COMPREHENSIVE.md
- STRAT_LAB_OPTIMIZATION_INSIGHTS.md
- PROPOSED_SYSTEM_ARCHITECTURE_v2_CORRECTED.md
- 4_BACKTESTING_REQUIREMENTS_AND_ADDITIONAL_CONSIDERATIONS.md

---

*End of Document*
