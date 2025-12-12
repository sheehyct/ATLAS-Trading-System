# ATLAS - Adaptive Trading with Layered Asset System

A quantitative trading system implementing regime-aware portfolio management with academically validated statistical methods and autonomous signal automation.

## Overview

ATLAS is a multi-layer algorithmic trading architecture that combines regime detection, pattern recognition, options execution, and autonomous signal monitoring.

**Architecture Layers**:
- **Layer 1 (Regime Detection)**: Market state classification using academic statistical jump model
- **Layer 2 (Pattern Recognition)**: STRAT bar pattern analysis for precise entry and exit levels
- **Layer 3 (Execution)**: Capital-aware options and equity execution with delta targeting
- **Layer 4 (Risk Management)**: Position monitoring with automated exit logic

**Signal Automation System** (Phase 4 Complete):
- **Signal Detection**: Autonomous pattern scanning across multiple timeframes (1H, 1D, 1W, 1M)
- **Alert Delivery**: Discord webhooks and structured logging for real-time notifications
- **Options Execution**: Automated paper trading to Alpaca with strike/DTE optimization
- **Position Monitoring**: Real-time target/stop/DTE exit condition detection

The system classifies market conditions into four regimes (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH) using peer-reviewed statistical methods with 33 years of empirical validation.

## Core Approach

**Regime-Aware Allocation**: The system detects market regimes using statistical change-point detection and adjusts strategy exposure accordingly:
- TREND_BULL: 100% deployed
- TREND_NEUTRAL: 70% deployed
- TREND_BEAR: 30% deployed
- CRASH: 0% deployed (full cash)

**Academic Foundation**: Regime detection implements the statistical jump model from Shu et al. (Princeton University, 2024):
- Sharpe ratio improvements of 20-42% over buy-and-hold across S&P 500, DAX, and Nikkei
- Maximum drawdown reductions of approximately 50%
- Volatility reductions of approximately 30%
- 33 years of empirical validation (1990-2023)

## Project Status

**Current Phase**: Phase 4 Full Orchestration COMPLETE

### Signal Automation (5-Phase Deployment)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | COMPLETE | Signal Detection + Discord/Logging Alerts |
| Phase 2 | COMPLETE | Options Execution with Delta/DTE Targeting |
| Phase 3 | COMPLETE | Position Monitoring with Auto-Exit Logic |
| Phase 4 | COMPLETE | Full Orchestration + Market Data Integration |
| Phase 5 | DEPLOYED | VPS Deployment for 24/7 Operation (Dec 11, 2025) |

### Test Coverage

**913 tests passing** across all layers:
- Regime Detection: 31 tests
- STRAT Patterns: 56 tests
- ThetaData Integration: 80 tests
- Signal Automation E2E: 14 tests
- Additional validation tests: 732 tests

### Validated Strategies

**52-Week High Momentum Strategy** (Gate 1 Validated):
- Technology sector multi-asset portfolio (30 stocks, top 10 selection)
- ATLAS regime integration for allocation adjustment
- Sharpe Ratio: 0.99 (target: 0.8 minimum) - PASS
- CAGR: 11.80% (target: 10% minimum) - PASS
- Maximum Drawdown: -19.06% (46.7% improvement vs baseline)
- March 2020 CRASH detection: 81.8% accuracy

## Capital Requirements

| Account Size | Recommended Approach | Execution Method |
|--------------|---------------------|------------------|
| $3,000 | STRAT + Options | ~$80,000 notional exposure via leverage |
| $10,000+ | Full equity strategies | Multi-stock portfolios |

Options provide approximately 27x capital efficiency vs equity positions.

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
uv run pytest tests/ -q --tb=no
```

### Environment Configuration

Required credentials in `.env`:

```bash
# Tiingo API (required for historical data)
TIINGO_API_KEY=your_tiingo_api_token

# Alpaca API (required for paper trading)
ALPACA_API_KEY_SMALL=your_alpaca_api_key
ALPACA_SECRET_KEY_SMALL=your_alpaca_secret_key

# Discord Webhook (optional, for alerts)
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# VectorBT Pro (required)
GITHUB_TOKEN=your_github_personal_access_token
```

## Signal Daemon Usage

The signal daemon provides autonomous signal detection and execution:

```bash
# Start daemon (runs continuously until Ctrl+C)
uv run python scripts/signal_daemon.py start

# Start with execution enabled (paper trading)
uv run python scripts/signal_daemon.py start --execute

# Run a single scan
uv run python scripts/signal_daemon.py scan --timeframe 1D

# Run all timeframe scans
uv run python scripts/signal_daemon.py scan-all

# Execute pending signals manually
uv run python scripts/signal_daemon.py execute

# Show current option positions
uv run python scripts/signal_daemon.py positions

# Check positions for exit conditions
uv run python scripts/signal_daemon.py monitor

# Execute exits automatically
uv run python scripts/signal_daemon.py monitor --execute

# Show monitoring statistics
uv run python scripts/signal_daemon.py monitor-stats

# Test alerter connections
uv run python scripts/signal_daemon.py test

# Show signal store status
uv run python scripts/signal_daemon.py status

# Close a specific position
uv run python scripts/signal_daemon.py close AAPL250117C00200000
```

### Environment Variables for Signal Automation

```bash
# Core settings
SIGNAL_SYMBOLS=SPY,QQQ,IWM,DIA,AAPL
SIGNAL_TIMEFRAMES=1H,1D,1W,1M
SIGNAL_LOG_LEVEL=INFO
SIGNAL_STORE_PATH=data/signals

# Execution settings
SIGNAL_EXECUTION_ENABLED=false
SIGNAL_EXECUTION_ACCOUNT=SMALL
SIGNAL_MAX_CAPITAL_PER_TRADE=300
SIGNAL_MAX_CONCURRENT_POSITIONS=5

# Monitoring settings
SIGNAL_MONITORING_ENABLED=true
SIGNAL_MONITOR_INTERVAL=60
SIGNAL_EXIT_DTE=3
SIGNAL_MAX_LOSS_PCT=0.50
SIGNAL_MAX_PROFIT_PCT=1.00
```

## Architecture

### Layer 1: Regime Detection

The statistical jump model uses a two-state Gaussian process with temporal penalty:

**Features**:
- Downside Deviation (10-day EWMA)
- Sortino Ratio (20-day and 60-day halflife)
- VIX Acceleration (flash crash detection)

**Output**: Four-regime classification with feature-based thresholds

**Validation**: March 2020 crash 77% detection rate

### Layer 2: STRAT Pattern Recognition

Bar-level pattern recognition for precise entry and exit timing:

**Bar Classification**:
- Type 1 (Inside Bar): Contained within previous bar's range
- Type 2U/2D (Directional): Breaks previous high or low only
- Type 3 (Outside Bar): Breaks both previous high and low

**Primary Patterns**:
- 3-1-2 Reversal: Outside → Inside → Breakout
- 2-1-2 Reversal: Failed breakdown/breakout
- 2-2 Continuation: Consecutive directional bars
- Rev Strat: Pattern invalidation

**Timeframe Continuity**:
- Control: All timeframes aligned (highest confidence)
- Confirm: Majority aligned
- Conflict: Mixed signals
- Change: Majority reversing

### Layer 3: Options Execution

Capital-efficient execution through options:

**Strike Selection**:
- Target delta: 0.40-0.55 range
- Slightly OTM for optimal risk/reward

**DTE Optimization**:
- Minimum: 7 days
- Target: 14 days
- Maximum: 21 days

**Order Types**:
- Market orders (default)
- Limit orders with configurable buffer

### Layer 4: Position Monitoring

Automated exit condition detection:

**Exit Conditions** (Priority Order):
1. DTE Exit: Close when DTE ≤ threshold (theta decay)
2. Stop Hit: Underlying reaches stop price
3. Max Loss: Unrealized loss exceeds threshold
4. Target Hit: Underlying reaches target price
5. Max Profit: Unrealized gain exceeds threshold

## Project Structure

```
atlas-trading-system/
├── regime/                        # Layer 1 - Regime detection
│   ├── academic_features.py       # Feature calculation
│   ├── academic_jump_model.py     # Statistical jump model
│   └── vix_acceleration.py        # Flash crash detection
├── strat/                         # Layer 2 - STRAT patterns
│   ├── bar_classifier.py          # Bar type classification
│   ├── pattern_detector.py        # Pattern detection
│   ├── options_module.py          # Options pricing/selection
│   └── signal_automation/         # Autonomous trading
│       ├── config.py              # Configuration management
│       ├── signal_store.py        # Signal persistence
│       ├── executor.py            # Options order execution
│       ├── position_monitor.py    # Position monitoring
│       ├── scheduler.py           # APScheduler integration
│       ├── daemon.py              # Main orchestrator
│       └── alerters/              # Alert delivery
│           ├── discord_alerter.py # Discord webhooks
│           └── logging_alerter.py # Structured logging
├── strategies/                    # Strategy implementations
│   ├── base_strategy.py           # Abstract base class
│   ├── high_momentum_52w.py       # 52-week high momentum
│   └── orb.py                     # Opening Range Breakout
├── integrations/                  # External integrations
│   └── alpaca_trading_client.py   # Alpaca paper/live trading
├── scripts/                       # CLI entry points
│   └── signal_daemon.py           # Signal automation CLI
└── tests/                         # Test suite (913 tests)
    ├── test_regime/               # Regime detection tests
    ├── test_strat/                # STRAT pattern tests
    ├── test_integrations/         # Integration tests
    └── test_signal_automation/    # E2E automation tests
```

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Sharpe Ratio | >0.8 | 0.99 |
| CAGR | >10% | 11.80% |
| Maximum Drawdown | <25% | -19.06% |
| Win Rate | 45-55% | TBD (paper trading) |
| Test Coverage | 100% pass | 886/894 (8 known failures) |

## Technical Requirements

### Core Dependencies
- Python 3.12+
- VectorBT Pro (vectorized backtesting)
- NumPy, pandas (numerical computing)
- APScheduler (job scheduling)
- alpaca-py (trading API)
- tiingo (historical data)

### API Access Required
- Tiingo account (free tier, 30+ years historical data)
- Alpaca Markets account (paper trading, options enabled)
- Discord webhook (optional, for alerts)

## Data Sources

| Source | Purpose | Coverage |
|--------|---------|----------|
| Tiingo | Historical OHLCV | 30+ years (SPY from 1993) |
| Alpaca | Real-time quotes, orders | Live market data |
| ThetaData | Options chains | Greeks, IV, historical |
| Yahoo Finance | VIX data only | VIX index (prohibited for equities) |

## Validation Criteria

Before live deployment:

| Criterion | Status |
|-----------|--------|
| Regime Detection >50% crash accuracy | PASS (77-82%) |
| Walk-Forward <30% degradation | In Progress |
| Paper Trading 100+ trades | In Progress |
| Test Coverage 100% pass | 913 passing (10 known regime failures) |
| Risk Controls 100% compliance | Implemented |

## Contributing

Contributions welcome. Please ensure:

1. All tests pass (`uv run pytest tests/`)
2. Code includes comprehensive docstrings
3. Changes include relevant unit tests
4. Commit messages follow conventional commits
5. Performance claims backed by empirical evidence

## Disclaimer

This software is for educational and research purposes only. Algorithmic trading involves substantial risk of loss and may not be suitable for all investors. Past performance does not guarantee future results. The regime detection model is based on peer-reviewed research but requires proper validation before deployment. No warranty is provided for accuracy or profitability. Use at your own risk.

## References

**Academic Research**:
- Shu, Y., et al. (2024). "Statistical Jump Models for Asset Allocation." Princeton University. Empirical validation on S&P 500, DAX, and Nikkei indices (1990-2023).

**Technology**:
- VectorBT Pro - Professional backtesting framework
- Alpaca Markets - Commission-free trading API
- ThetaData - Options data provider
- APScheduler - Python job scheduling

---

**Version**: 3.0 (Phase 4 Full Orchestration)
**Last Updated**: December 2025
**Status**: Active Development - Phase 5 VPS Deployed (Hetzner, Dec 11, 2025)
