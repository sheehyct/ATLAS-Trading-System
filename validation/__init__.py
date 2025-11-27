"""
ATLAS Production Readiness Validation Module

Provides comprehensive validation tools for algorithmic trading strategies
per ATLAS_PRODUCTION_READINESS_CHECKLIST.md.

Components:
- Walk-Forward Validation (Section 1.6)
- Monte Carlo Simulation (Section 1.7)
- Bias Detection (Section 1.4)
- Pattern Metrics Analysis (STRAT-specific)
- Transaction Cost Modeling (Section 1.1)

Session 83C: Foundation module with protocols, configs, and result dataclasses.
Sessions 83D-83K: Validators, runner, and integration.

Usage:
    from validation import ValidationConfig, BacktestResult
    from validation import WalkForwardResults, MonteCarloResults

    # Configure validation
    config = ValidationConfig()
    config.monte_carlo.n_simulations = 10000

    # For options strategies
    options_config = config.for_options()
"""

# Protocol and base classes
from validation.protocols import (
    StrategyProtocol,
    ValidatorProtocol,
    BacktestResult,
    ParameterDict,
    ParameterGrid,
    TradesDataFrame,
    EquityCurve,
)

# Configuration dataclasses
from validation.config import (
    ValidationConfig,
    WalkForwardConfig,
    WalkForwardConfigOptions,
    MonteCarloConfig,
    MonteCarloConfigOptions,
    BiasDetectionConfig,
    PatternMetricsConfig,
    TransactionCostConfig,
    AcceptanceThresholds,
)

# Result dataclasses
from validation.results import (
    FoldResult,
    WalkForwardResults,
    MonteCarloResults,
    BiasCheckResult,
    BiasReport,
    PatternStats,
    OptionsAccuracyMetrics,
    PatternMetricsResults,
    ValidationSummary,
    ValidationReport,
)

__all__ = [
    # Protocols
    'StrategyProtocol',
    'ValidatorProtocol',
    'BacktestResult',
    'ParameterDict',
    'ParameterGrid',
    'TradesDataFrame',
    'EquityCurve',

    # Configs
    'ValidationConfig',
    'WalkForwardConfig',
    'WalkForwardConfigOptions',
    'MonteCarloConfig',
    'MonteCarloConfigOptions',
    'BiasDetectionConfig',
    'PatternMetricsConfig',
    'TransactionCostConfig',
    'AcceptanceThresholds',

    # Results
    'FoldResult',
    'WalkForwardResults',
    'MonteCarloResults',
    'BiasCheckResult',
    'BiasReport',
    'PatternStats',
    'OptionsAccuracyMetrics',
    'PatternMetricsResults',
    'ValidationSummary',
    'ValidationReport',
]

__version__ = '0.1.0'
__session__ = '83C'
