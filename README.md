# ATLAS - Adaptive Trading with Layered Asset System (Layer 1)

A quantitative trading system implementing regime-aware portfolio management with academically validated statistical methods. ATLAS serves as Layer 1 (regime detection) in a multi-layer trading architecture.

## Overview

**Multi-Layer Architecture (Session 20, updated Session 21)**:
- **Layer 1 (ATLAS)**: Regime detection using academic statistical jump model (Phase F validation next)
- **Layer 2 (STRAT)**: Bar pattern recognition for precise entry/exit levels (PENDING - Sessions 22-27)
- **Layer 3 (Execution)**: Capital-aware deployment - options ($3k optimal) OR equities ($10k+ optimal)
- **Layer 4 (Credit Spreads)**: Crash protection using forward-looking credit spread signals (ACCEPTED - deferred to future)

ATLAS is the regime detection foundation layer that classifies market conditions (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH) using peer-reviewed statistical methods with demonstrated 33-year empirical validation. The regime signal filters downstream STRAT pattern signals and provides risk-on/risk-off guidance for capital-efficient execution.

**Integration Status**: ATLAS Phase F validation next (Session 21), then STRAT integration begins (Sessions 22-27).

## Core Approach

**Regime-Aware Allocation**: The system detects market regimes (bull, bear, neutral, crash) using statistical change-point detection and adjusts strategy exposure accordingly. During high-volatility regimes, defensive strategies receive higher allocation; during trending markets, momentum strategies dominate.

**Academic Foundation**: Regime detection implements the statistical jump model from Shu et al. (Princeton University, 2024), which demonstrated:
- Sharpe ratio improvements of 20-42% over buy-and-hold across S&P 500, DAX, and Nikkei
- Maximum drawdown reductions of approximately 50%
- Volatility reductions of approximately 30%
- 33 years of empirical validation (1990-2023)

**Multi-Strategy Portfolio**: Five uncorrelated strategies provide diversification across different market conditions:
- Opening Range Breakout (momentum/volatility)
- Regime-Aware Mean Reversion (oscillating markets)
- Pairs Trading (market-neutral statistical arbitrage)
- Semi-Volatility Momentum (trending markets)

## Capital Requirements and Deployment Strategy

**CRITICAL CONSTRAINT IDENTIFIED (Session 20)**:

ATLAS equity strategies require **MINIMUM $10,000** for proper position sizing:
- **With $10,000+**: Full position sizing capability (2% risk per trade achievable)
- **With $3,000**: CAPITAL CONSTRAINED (0.06% actual risk vs 2% target) - sub-optimal performance
- **Root Cause**: ATR-based position sizing uses `min(risk_based_size, capital_constrained_size)`

**Deployment Recommendations**:
1. **With $3,000 capital** (User's current situation):
   - **Paper trade ATLAS** with $10,000 simulated capital (validate regime detection)
   - **Live trade STRAT + Options** with $3,000 real capital (27x capital efficiency vs equities)
   - **Rationale**: Options optimal for small capital, ATLAS provides regime filter signal

2. **With $10,000+ capital**:
   - **Deploy ATLAS equities** (optimal for this capital level)
   - **OR deploy STRAT options** (both viable, choose based on preference)

3. **Mixed deployment** (recommended):
   - Paper trade ATLAS regime detection ($10k simulated, validate model)
   - Live trade STRAT options ($3k real, capital-efficient execution)
   - ATLAS regime filters STRAT signals (confluence model)

**Timeline to Paper Trading**: 3-4 weeks (finish ATLAS Phase F + implement STRAT Layers 2-3)

See `docs/HANDOFF.md` "Capital Requirements Analysis" section for detailed position sizing math and capital efficiency comparison.

## Project Status

**Current Phase**: ATLAS Layer 1 regime detection - Phase E COMPLETE, Phase F validation next (Session 21)

**Completed Components**:
- Position sizing with ATR-based stops and capital constraints
- Opening Range Breakout strategy with volume confirmation
- Feature calculation for regime detection (downside deviation, Sortino ratios)
- Optimization solver using dynamic programming and coordinate descent
- Cross-validation framework for parameter selection
- Online inference with rolling parameter updates
- 4-regime mapping (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH)
- Portfolio heat management and risk controls

**Test Coverage**: 40+ regime detection tests passing (100% on completed phases)

**Validation Results**:
- March 2020 crash detection: 100% CRASH+BEAR accuracy (target >50%) - EXCEEDED
- Regime mapping: Feature-based classification with adjusted thresholds
- Implementation: All 5 phases (A-E) complete and validated

**Implementation Progress**:
- Phase A (Feature Calculation): COMPLETE - 16/16 tests passing
- Phase B (Optimization Solver): COMPLETE - 6/6 tests passing
- Phase C (Parameter Selection): COMPLETE - 9/9 tests passing
- Phase D (Online Inference): COMPLETE - March 2020 100% bear detection
- Phase E (Regime Mapping): COMPLETE - March 2020 100% CRASH+BEAR detection
- Phase F (Final Validation): Next - Comprehensive validation suite

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

### Regime Detection

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

**NOTE (Session 20)**: This structure represents Layer 1 (ATLAS) components only. Structure will expand significantly when STRAT (Layer 2) integration begins in Sessions 22-27. Expected additions: `strat/` directory for bar classification and pattern detection components.

```
atlas-trading-system/
├── regime/                        # ATLAS Layer 1 - Regime detection
│   ├── academic_features.py       # Feature calculation (COMPLETE)
│   ├── academic_jump_model.py     # Statistical jump model (Phases A-E COMPLETE)
│   └── base_regime_detector.py    # Abstract base class
├── strategies/                    # ATLAS equity strategies (PENDING - optional)
│   ├── base_strategy.py           # Abstract base class
│   └── orb.py                     # Opening Range Breakout
├── core/                          # Core components
│   ├── risk_manager.py            # Risk management
│   └── portfolio_manager.py       # Multi-strategy coordination
├── data/                          # Data management
│   ├── alpaca.py                  # Market data fetching
│   └── mtf_manager.py             # Multi-timeframe alignment
├── utils/                         # Utilities
│   ├── position_sizing.py         # Position sizing (capital constraints identified)
│   └── portfolio_heat.py          # Portfolio heat management
├── tests/                         # Test suite (46 tests for regime detection)
│   ├── test_regime/               # Regime detection tests (Phases A-E COMPLETE)
│   └── ...
└── docs/                          # Documentation
    ├── HANDOFF.md                 # Session continuity (UPDATED Session 20)
    ├── CLAUDE.md                  # Development rules (UPDATED Session 20)
    ├── System_Architecture_Reference.md  # Architecture (Layer 1 context added)
    └── SYSTEM_ARCHITECTURE/       # 4-part architecture docs (UPDATED Session 20)
```

**Upcoming Directory Expansion (Sessions 22-27)**:
```
├── strat/                         # Layer 2 - Pattern recognition (PENDING)
│   ├── bar_classifier.py          # STRAT bar classification (1, 2U, 2D, 3)
│   ├── pattern_detector.py        # Pattern detection (3-1-2, 2-1-2, etc.)
│   └── timeframe_continuity.py    # Multi-timeframe alignment checker
├── execution/                     # Layer 3 - Capital-aware execution (PENDING)
│   ├── options_execution.py       # Options execution ($3k optimal)
│   └── unified_signals.py         # ATLAS + STRAT signal integration
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

1. **Regime Detection**: >50% accuracy on March 2020 crash - COMPLETE (100% CRASH+BEAR detection)
2. **Walk-Forward Testing**: <30% performance degradation out-of-sample - Pending Phase F
3. **Paper Trading**: Minimum 6 months with 100+ trades - Pending
4. **Risk Controls**: 100% compliance with position sizing and portfolio heat limits - Pending integration
5. **Test Coverage**: 100% pass rate on unit and integration tests - 40+ regime tests passing
6. **Performance Match**: Paper trading results within expected backtest range - Pending

## Documentation

- `docs/System_Architecture_Reference.md` - Complete system architecture
- `docs/research/` - Academic research and validation studies
- Individual strategy documentation in respective modules

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
**Last Updated**: November 2025 (Session 20 - Multi-Layer Architecture Context Added)
**Status**: Active Development
- **Layer 1 (ATLAS)**: Phase E COMPLETE, Phase F validation next (Session 21)
- **Layer 2 (STRAT)**: PENDING (Sessions 22-27)
- **Layer 3 (Execution)**: PENDING (Sessions 28-30)
- **Integrated System**: 3-4 weeks to paper trading deployment
