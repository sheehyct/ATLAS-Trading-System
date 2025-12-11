# Strategy Research

This directory contains research and analysis of external trading strategies being evaluated for potential integration into the ATLAS system.

## Credit Spread Leveraged ETF Strategy

**Status**: Under Evaluation (Not Yet Integrated)

**Location**: `credit_spread/`

**Summary**:
- Uses credit spreads (FRED: BAMLH0A0HYM2) to time leveraged ETF entries (SSO)
- Achieved 9.7x return (2006-2025) vs video claim of 16.3x
- 83% win rate, 7 closed trades
- **Recommendation**: Requires signal algorithm refinement before ATLAS integration

**Key Files**:
- `docs/CREDIT_SPREAD_STRATEGY_SUMMARY.md` - Complete analysis report
- `docs/VISUALIZATION_TOOLS_COMPARISON.md` - Visualization tools research
- `reports/quantstats_credit_spread_tearsheet.html` - Professional tearsheet
- `scripts/credit_spread_backtest.py` - Main backtest implementation

**Date Analyzed**: November 8, 2025
