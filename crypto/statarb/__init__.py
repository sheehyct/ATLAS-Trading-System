"""
Statistical Arbitrage module for crypto derivatives.

Implements cointegration-based pairs trading strategies for Coinbase CFM.
"""

from .cointegration import (
    engle_granger_test,
    johansen_test,
    calculate_half_life,
    test_all_pairs,
)
from .spread import (
    calculate_spread,
    calculate_zscore,
    calculate_hedge_ratio,
)
from .backtest import (
    run_pairs_backtest,
    PairsBacktestResult,
)
from .signal_generator import (
    StatArbSignalGenerator,
    StatArbSignal,
    StatArbSignalType,
    StatArbConfig,
    StatArbPosition,
)

__all__ = [
    "engle_granger_test",
    "johansen_test",
    "calculate_half_life",
    "test_all_pairs",
    "calculate_spread",
    "calculate_zscore",
    "calculate_hedge_ratio",
    "run_pairs_backtest",
    "PairsBacktestResult",
    # Signal generation (Session EQUITY-91)
    "StatArbSignalGenerator",
    "StatArbSignal",
    "StatArbSignalType",
    "StatArbConfig",
    "StatArbPosition",
]
