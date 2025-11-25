# Session 73 Start Prompt

**Date:** November 24, 2025
**Priority:** Data-Driven Strike Selection Implementation

## Current State

Session 72 completed all options module bug fixes and validation:
- 6 critical bugs fixed in options_module.py and greeks.py
- 59 tests passing (test_greeks_validation.py, test_options_pnl.py, test_options_integration.py)
- 50-stock historical validation complete (670 trades, 2020-2025)
- P/L calculations now correct (26% win rate, -$1,668 avg P/L - realistic for stops)

**CRITICAL FINDING:** The 0.3x strike selection formula is purely geometric and does NOT consider:
1. Theta decay cost over expected holding period
2. Average time-to-magnitude from pattern testing
3. Empirical probability of reaching target

## Immediate Task: Data-Driven Strike Selection

Implement strike selection that optimizes for net expected P/L, not just geometric positioning.

### Required Data Sources

1. **Pattern time-to-magnitude statistics** - From Session 67/68/69 validation:
   - Weekly patterns: 3-7 days typical to magnitude
   - Daily patterns: 1-3 days typical
   - Monthly patterns: 5-15 days typical

2. **Pattern win rates by type** - From comprehensive_pattern_analysis.py:
   - 2-1-2 Up @ 1W: 84.6% win rate
   - 2-2 Up @ 1W: 86.2% win rate
   - 3-1-2 patterns: Lower frequency but higher conviction

### Implementation Approach

```python
def calculate_optimal_strike(
    pattern_type: PatternType,
    entry_price: float,
    target_price: float,
    stop_price: float,
    underlying_price: float,
    expected_iv: float = 0.20
) -> dict:
    """
    Calculate optimal strike based on expected net P/L.

    Returns:
        dict with: strike, delta, expected_theta_cost, expected_profit, net_expected_pnl
    """
    # Get historical data for pattern type
    expected_days = PATTERN_AVG_DAYS[pattern_type]
    prob_hit = PATTERN_WIN_RATES[pattern_type]

    # Test multiple strike candidates
    candidates = []
    for strike_offset_pct in [0.2, 0.3, 0.4, 0.5, 0.6]:
        strike = calculate_strike_at_offset(entry_price, target_price, strike_offset_pct)
        greeks = calculate_greeks(underlying_price, strike, expected_days/365, 0.05, expected_iv, option_type)

        theta_cost = abs(greeks.theta) * expected_days * 100
        delta_pnl_if_hit = greeks.delta * (target_price - entry_price) * 100
        net_expected = delta_pnl_if_hit * prob_hit - theta_cost * (1 - prob_hit)

        candidates.append({
            'strike': strike,
            'delta': greeks.delta,
            'theta_cost': theta_cost,
            'expected_profit': delta_pnl_if_hit,
            'net_expected_pnl': net_expected
        })

    # Return candidate with highest net expected P/L
    return max(candidates, key=lambda x: x['net_expected_pnl'])
```

### Files to Modify

1. **strat/options_module.py** - `OptionsExecutor._calculate_strike()`:
   - Add pattern statistics dictionary
   - Implement net expected P/L optimization
   - Keep 0.3x as fallback for unknown patterns

2. **strat/greeks.py** - Add helper functions:
   - `estimate_theta_cost(delta, dte, expected_days)`
   - `calculate_net_expected_pnl(greeks, target_move, prob_hit, expected_days)`

3. **scripts/validate_options_greeks.py** - Add metrics:
   - Track which strike offset was selected
   - Compare net expected P/L to actual P/L

### Validation Criteria

Before declaring GO for paper trading:
1. Delta accuracy > 50% in optimal range (0.50-0.80)
2. Net expected P/L positive for majority of trades
3. Actual P/L closer to expected than with geometric-only selection

## Files to Read First

1. `docs/HANDOFF.md` - Session 72 complete details
2. `strat/options_module.py` - Current strike selection (lines ~247-298)
3. `scripts/comprehensive_pattern_analysis.py` - Pattern statistics source
4. `scripts/pattern_ranking_by_expectancy.csv` - Win rates and expectancy by pattern

## Key Reference Data

From Session 67 analysis (pattern_ranking_by_expectancy.csv):

| Pattern | Timeframe | Win Rate | Avg Days | P/L Expectancy |
|---------|-----------|----------|----------|----------------|
| 2-1-2 Up | Weekly | 84.6% | 3-7 days | 6.82% |
| 2-2 Up | Weekly | 86.2% | 3-5 days | 4.17% |
| 2-1-2 Up | Monthly | 92.9% | 5-15 days | 18.11% |
| 3-1-2 | Weekly | 72.7% | 5-10 days | 4.63% |

## User Philosophy

"Accuracy over speed. No rushing to conclusions. No time limit."

Options leverage means even $1-2 movement = large profit. Must optimize strike selection to maximize probability-weighted net P/L, not just geometric positioning.
