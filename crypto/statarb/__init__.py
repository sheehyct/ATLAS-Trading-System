"""
Statistical Arbitrage module for crypto derivatives.

Implements cointegration-based pairs trading strategies for Coinbase CFM.

Core exports (always available):
- StatArbSignalGenerator, StatArbSignal, StatArbConfig - for live trading

Research exports (optional, for backtesting/analysis):
- cointegration tests, spread calculations, backtest runner
"""

# Core signal generator - always available (Session EQUITY-91)
from .signal_generator import (
    StatArbSignalGenerator,
    StatArbSignal,
    StatArbSignalType,
    StatArbConfig,
    StatArbPosition,
)

__all__ = [
    # Signal generation (Session EQUITY-91) - Core exports
    "StatArbSignalGenerator",
    "StatArbSignal",
    "StatArbSignalType",
    "StatArbConfig",
    "StatArbPosition",
]

# Optional research modules - graceful import (Session EQUITY-92)
try:
    from .cointegration import (
        engle_granger_test,
        johansen_test,
        calculate_half_life,
        test_all_pairs,
    )
    __all__.extend([
        "engle_granger_test",
        "johansen_test",
        "calculate_half_life",
        "test_all_pairs",
    ])
except ImportError:
    pass  # Research modules not available

try:
    from .spread import (
        calculate_spread,
        calculate_zscore,
        calculate_hedge_ratio,
    )
    __all__.extend([
        "calculate_spread",
        "calculate_zscore",
        "calculate_hedge_ratio",
    ])
except ImportError:
    pass  # Research modules not available

try:
    from .backtest import (
        run_pairs_backtest,
        PairsBacktestResult,
    )
    __all__.extend([
        "run_pairs_backtest",
        "PairsBacktestResult",
    ])
except ImportError:
    pass  # Research modules not available
