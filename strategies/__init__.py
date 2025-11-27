"""
ATLAS Trading System - Strategy Implementations

This module contains strategy implementations that conform to StrategyProtocol
for use with ValidationRunner.

Session 83K: ATLAS Validation Run for STRAT Strategies
"""

from strategies.strat_options_strategy import (
    STRATOptionsStrategy,
    STRATOptionsConfig,
)

__all__ = [
    'STRATOptionsStrategy',
    'STRATOptionsConfig',
]
