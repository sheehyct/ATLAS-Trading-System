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
Session 83D: Walk-forward validation implementation.
Sessions 83E-83K: Monte Carlo, bias detection, runner, and integration.

Usage:
    from validation import ValidationConfig, BacktestResult
    from validation import WalkForwardValidator, WalkForwardResults

    # Configure validation
    config = ValidationConfig()

    # Run walk-forward validation
    validator = WalkForwardValidator(config.walk_forward)
    results = validator.validate(strategy, data)

    if results.passes_validation:
        print("Strategy passes walk-forward validation")

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

# Validators
from validation.walk_forward import (
    WalkForwardValidator,
    FoldWindow,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

from validation.monte_carlo import (
    MonteCarloValidator,
    TradeRecord,
    generate_synthetic_trades,
)

from validation.bias_detection import (
    BiasDetector,
    detect_look_ahead_bias,
    validate_entry_prices,
)

from validation.pattern_metrics import (
    PatternMetricsAnalyzer,
    analyze_pattern_metrics,
    get_best_patterns_by_metric,
    get_regime_pattern_compatibility,
    generate_pattern_report,
)

# Session 83I: Validation runner for orchestrating all validators
from validation.validation_runner import (
    ValidationRunner,
    run_validation,
)

# Session 83K: STRAT validator with ThetaData integration
from validation.strat_validator import (
    ATLASSTRATValidator,
    ValidationRunConfig,
    ValidationRunResult,
    BatchResult,
    DataSourceMetrics,
    ThetaDataStatus,
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

    # Validators - Walk-Forward
    'WalkForwardValidator',
    'FoldWindow',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',

    # Validators - Monte Carlo
    'MonteCarloValidator',
    'TradeRecord',
    'generate_synthetic_trades',

    # Validators - Bias Detection
    'BiasDetector',
    'detect_look_ahead_bias',
    'validate_entry_prices',

    # Validators - Pattern Metrics
    'PatternMetricsAnalyzer',
    'analyze_pattern_metrics',
    'get_best_patterns_by_metric',
    'get_regime_pattern_compatibility',
    'generate_pattern_report',

    # Validators - Validation Runner (Session 83I)
    'ValidationRunner',
    'run_validation',

    # STRAT Validator with ThetaData (Session 83K)
    'ATLASSTRATValidator',
    'ValidationRunConfig',
    'ValidationRunResult',
    'BatchResult',
    'DataSourceMetrics',
    'ThetaDataStatus',
]

__version__ = '0.7.0'
__session__ = '83K'
