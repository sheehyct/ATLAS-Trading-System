"""
STRAT Backtesting Pipeline - Mirrors Live Trading Logic

Replicates the live signal automation daemon's exact behavior for
accurate historical performance analysis. Includes all 9 exit conditions,
trailing stops, partial exits, capital constraints, and TFC re-evaluation.

Module Structure:
    config          - BacktestConfig (mirrors SignalAutomationConfig)
    engine          - BacktestEngine orchestrator
    data_providers  - OptionsPriceProvider protocol + implementations
    simulation      - Bar simulator, position tracker, capital simulator
    signals         - Signal generation and entry simulation
    exits           - All 9 exit conditions, trailing stops, partial exits
    analytics       - MFE/MAE tracking, trade recording, results formatting
    runners         - CLI entry point
"""

__version__ = '0.1.0'
