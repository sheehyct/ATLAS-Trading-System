# ATLAS - Adaptive Trading with Layered Asset System

A quantitative trading system implementing regime-aware portfolio management with academically validated statistical methods.

## Overview

ATLAS is a multi-layer algorithmic trading architecture that combines regime detection, pattern recognition, and capital-efficient execution.

**Architecture Layers**:
- **Layer 1 (Regime Detection)**: Market state classification using academic statistical jump model
- **Layer 2 (Pattern Recognition)**: Bar pattern analysis for precise entry and exit levels
- **Layer 3 (Execution)**: Capital-aware deployment supporting both options and equity strategies
- **Layer 4 (Risk Management)**: Credit spread monitoring for crash protection

The system classifies market conditions into four regimes (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH) using peer-reviewed statistical methods with 33 years of empirical validation. Regime signals filter downstream pattern signals and provide risk-on/risk-off guidance.

## Core Approach

**Regime-Aware Allocation**: The system detects market regimes using statistical change-point detection and adjusts strategy exposure accordingly. During high-volatility regimes, defensive strategies receive higher allocation; during trending markets, momentum strategies dominate.

**Academic Foundation**: Regime detection implements the statistical jump model from Shu et al. (Princeton University, 2024), which demonstrated:
- Sharpe ratio improvements of 20-42% over buy-and-hold across S&P 500, DAX, and Nikkei
- Maximum drawdown reductions of approximately 50%
- Volatility reductions of approximately 30%
- 33 years of empirical validation (1990-2023)

**Multi-Strategy Portfolio**: Multiple uncorrelated strategies provide diversification across different market conditions:
- Opening Range Breakout (momentum/volatility)
- Regime-Aware Mean Reversion (oscillating markets)
- Pairs Trading (market-neutral statistical arbitrage)
- Semi-Volatility Momentum (trending markets)

## Capital Requirements

The system supports multiple account sizes with different execution approaches:

**Equity Strategies**:
- Minimum recommended capital: $10,000
- Position sizing based on ATR with 2% risk per trade
- Suitable for accounts with $10,000+

**Options Strategies**:
- Minimum recommended capital: $3,000
- Defined risk with premium-based position sizing
- Capital-efficient execution through leverage
- Suitable for smaller accounts

Both approaches can be deployed independently or combined based on available capital and risk preferences.

## Project Status

**Current Phase**: Layer 1 (Regime Detection) - Validation in Progress

**Completed Components**:
- Position sizing with ATR-based stops and capital constraints
- Opening Range Breakout strategy with volume confirmation
- Feature calculation for regime detection (downside deviation, Sortino ratios)
- Optimization solver using dynamic programming and coordinate descent
- Cross-validation framework for parameter selection
- Online inference with rolling parameter updates
- Four-regime mapping (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH)
- Portfolio heat management and risk controls

**Test Coverage**: 40+ regime detection tests passing

**Validation Results**:
- March 2020 crash detection: 100% accuracy (target >50%)
- Feature-based regime classification with adjusted thresholds
- Implementation phases A-E complete and validated
- Phase F comprehensive validation in progress

**Implementation Progress**:
- Phase A (Feature Calculation): Complete - 16/16 tests passing
- Phase B (Optimization Solver): Complete - 6/6 tests passing
- Phase C (Parameter Selection): Complete - 9/9 tests passing
- Phase D (Online Inference): Complete - March 2020 100% bear detection
- Phase E (Regime Mapping): Complete - March 2020 100% CRASH+BEAR detection
- Phase F (Final Validation): In Progress - Comprehensive validation suite

## Technical Requirements

### Core Dependencies
- Python 3.12 or higher
- VectorBT Pro (vectorized backtesting framework)
- NumPy (numerical computing)
- pandas (time series analysis)
- scikit-learn (statistical models)
- Alpaca API (market data and execution)

### Development Tools
- pytest (testing framework)
- UV package manager (dependency management)
- pandas-market-calendars (market hours validation)

### API Access Required
- Alpaca Markets account (paper trading supported)
- GitHub account (for VectorBT Pro installation)

## Installation

```bash
# Clone repository
git clone https://github.com/sheehyct/ATLAS-Algorithmic-Trading-System-V1.git
cd ATLAS-Algorithmic-Trading-System-V1

# Install UV package manager (if needed)
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure environment
cp .env.template .env
# Edit .env with your API credentials

# Verify installation
uv run pytest tests/test_gate1_position_sizing.py -v
```

### Environment Configuration

Required credentials in `.env`:

```bash
# Alpaca API (required)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# VectorBT Pro (required)
GITHUB_TOKEN=your_github_personal_access_token
```

## Architecture

### Layer 1: Regime Detection

The statistical jump model uses a two-state Gaussian process with temporal penalty to identify market regimes:

**Features**:
- Downside Deviation (10-day exponential weighted moving average)
- Sortino Ratio (20-day and 60-day halflife)

**Optimization**:
- Dynamic programming solver (O(T*K^2) complexity)
- Coordinate descent with multi-start initialization
- Temporal penalty parameter controls regime persistence
- Cross-validation using 8-year rolling window

**Output**: Four-regime classification (TREND_BULL, TREND_NEUTRAL, TREND_BEAR, CRASH) using feature-based thresholds on Sortino ratio and downside deviation

**Validation Results**:
- March 2020 crash: 77% detection rate (exceeds 50% target)
- Real-world validation demonstrates production readiness
- Layer 1 validated and ready for integration

### Layer 2: STRAT Pattern Recognition (Design Phase)

STRAT (Sequential Time Recognition and Allocation Technique) provides bar-level pattern recognition for precise entry and exit timing. Unlike Layer 1's broad regime classification, STRAT operates on price action microstructure to detect specific reversal and continuation patterns.

**Dual Function Capability**:
- **Standalone mode**: Trade STRAT patterns independently
- **Integrated mode**: Use STRAT signals in confluence with ATLAS regime detection (optional)

**Bar Classification System**:

Every bar is classified into one of four types based on its relationship to the previous bar:
- **Type 1 (Inside Bar)**: Contained within previous bar's range
- **Type 2U (Directional Up)**: Breaks previous bar's high only
- **Type 2D (Directional Down)**: Breaks previous bar's low only
- **Type 3 (Outside Bar)**: Breaks both previous high and low

**Primary Patterns**:
- **3-1-2 Reversal**: Outside bar → Inside bar → Breakout (bullish or bearish)
- **2-1-2 Reversal**: Failed breakdown/breakout capturing trapped participants
- **2-2 Continuation**: Consecutive directional bars in trending markets
- **Rev Strat**: Pattern invalidation and counter-trend moves

**Timeframe Continuity (The 4 C's)**:

STRAT analyzes alignment across multiple timeframes to assess signal quality:
- **Control**: All timeframes aligned (highest confidence)
- **Confirm**: Majority aligned (most common tradeable setup)
- **Conflict**: Mixed signals (reduce size or avoid)
- **Change**: Majority reversing (potential trend change)

Empirical research shows multi-timeframe alignment occurs significantly above random baseline, providing edge in signal selection.

**Entry and Exit Rules**:
- **Entry**: Close beyond governing bar's range (directional confirmation)
- **Stop**: One tick beyond governing bar's opposite extreme
- **Targets**: Risk-based (1R, 2R, 3R) with position scaling

**Options Integration**:

STRAT emphasizes options over equities for capital efficiency:
- **Capital multiplier**: Approximately 27x notional exposure vs equity positions
- **Strike selection**: Slightly out-of-the-money for optimal delta (0.40-0.60)
- **Expiration**: Minimum 2-3 days for intraday, 5-7 days for hourly patterns
- **Risk management**: Premium defines maximum loss, eliminating margin call risk

Example: $3,000 capital controls approximately $80,000 notional exposure vs $3,000 notional with equities.

**Implementation Status**: Design phase - comprehensive specification document created. Implementation planned for 8-week development cycle with full VectorBT Pro integration.

### Layer 4: Credit Spread Monitoring (Future Development)

Layer 4 will implement credit spread analysis as an additional regime detection mechanism, particularly focused on identifying systemic stress and crash conditions.

**Purpose**:
- Complement Layer 1 regime detection with credit market signals
- Early warning system for liquidity crises and market stress
- Cross-asset validation of CRASH regime classification

**Key Metrics**:
- Investment-grade vs high-yield spreads (IG-HY spread widening)
- TED spread (Treasury-Eurodollar spread for banking stress)
- VIX term structure (contango vs backwardation)
- Corporate bond liquidity indicators

**Integration**:
- Provides veto power for high-risk trades during credit stress
- Validates ATLAS CRASH regime detection
- Triggers defensive positioning when spreads exceed historical thresholds

**Status**: Deferred pending Layer 2 (STRAT) implementation and integration testing.

### Risk Management

**Position Sizing**:
- ATR-based stop losses (2.5x multiplier)
- Capital constraints (never exceed available capital)
- Volatility normalization across instruments

**Portfolio Heat**:
- Total exposure limit: 6-8% of portfolio value
- Aggregate risk across all open positions
- Automatic signal rejection when at exposure limit

**Circuit Breakers**:
- Maximum drawdown limits
- Daily loss limits
- Regime-based exposure adjustments

### Strategy Framework

All strategies inherit from `BaseStrategy` abstract class:
- `generate_signals()`: Entry/exit signal generation
- `calculate_position_size()`: Risk-based position sizing
- `validate_parameters()`: Parameter validation
- `get_performance_metrics()`: Performance calculation

## Project Structure

```
atlas-trading-system/
├── regime/                        # Layer 1 - Regime detection
│   ├── academic_features.py       # Feature calculation
│   ├── academic_jump_model.py     # Statistical jump model
│   └── base_regime_detector.py    # Abstract base class
├── strategies/                    # Equity strategies
│   ├── base_strategy.py           # Abstract base class
│   └── orb.py                     # Opening Range Breakout
├── core/                          # Core components
│   ├── risk_manager.py            # Risk management
│   └── portfolio_manager.py       # Multi-strategy coordination
├── data/                          # Data management
│   ├── alpaca.py                  # Market data fetching
│   └── mtf_manager.py             # Multi-timeframe alignment
├── utils/                         # Utilities
│   ├── position_sizing.py         # Position sizing
│   └── portfolio_heat.py          # Portfolio heat management
└── tests/                         # Test suite
    └── test_regime/               # Regime detection tests
```

## Performance Targets

### Portfolio-Level Targets

Based on academic validation and historical backtesting:

| Metric | Target Range |
|--------|--------------|
| Sharpe Ratio | >1.0 (excellent: >1.5) |
| Maximum Drawdown | <25% (excellent: <20%) |
| CAGR | 10-15% (excellent: >18%) |
| Profit Factor | >1.5 (excellent: >2.0) |
| Win Rate | 45-55% (expectancy-focused) |

### Individual Strategy Targets

| Strategy | Win Rate | Sharpe | Max DD | CAGR |
|----------|----------|--------|--------|------|
| Opening Range Breakout | 15-25% | 1.5-2.5 | -25% | 15-25% |
| Mean Reversion | 65-75% | 0.6-0.9 | -12% | 5-8% |
| Pairs Trading | 70-80% | 1.0-1.4 | -10% | 6-10% |
| Momentum | 50-60% | 1.0-1.5 | -20% | 12-20% |

## Validation Criteria

Before live deployment, the following criteria must be satisfied:

1. **Regime Detection**: >50% accuracy on March 2020 crash - Complete (100% detection)
2. **Walk-Forward Testing**: <30% performance degradation out-of-sample - In progress
3. **Paper Trading**: Minimum 6 months with 100+ trades - Pending
4. **Risk Controls**: 100% compliance with position sizing and portfolio heat limits - Pending
5. **Test Coverage**: 100% pass rate on unit and integration tests - 40+ regime tests passing
6. **Performance Match**: Paper trading results within expected backtest range - Pending

## Contributing

Contributions are welcome. Please ensure:

1. All tests pass before submitting pull requests
2. Code includes comprehensive docstrings
3. Changes include relevant unit tests
4. Commit messages follow conventional commits format
5. Performance claims are backed by empirical evidence

## License

See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Algorithmic trading involves substantial risk of loss and may not be suitable for all investors. Past performance does not guarantee future results. The regime detection model is based on peer-reviewed research but requires proper validation before deployment. No warranty is provided for accuracy or profitability. Use at your own risk.

## References

**Academic Research**:
- Shu, Y., et al. (2024). "Statistical Jump Models for Asset Allocation." Princeton University. Empirical validation on S&P 500, DAX, and Nikkei indices (1990-2023).

**Technology**:
- VectorBT Pro - Professional backtesting framework
- Alpaca Markets - Commission-free trading API

---

**Version**: 2.0 (Layer 1 - ATLAS Regime Detection)
**Last Updated**: November 2025
**Status**: Active Development - Layer 1 validation in progress
