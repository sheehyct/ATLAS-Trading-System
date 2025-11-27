"""
ATLAS Production Readiness Validation - Configuration Module

Defines configuration dataclasses for all validators.
Thresholds based on ATLAS_PRODUCTION_READINESS_CHECKLIST.md.

Session 83C: Foundation for ATLAS compliance validation.

Usage:
    from validation.config import ValidationConfig, WalkForwardConfig

    config = ValidationConfig()
    config.walk_forward.train_period = 252  # 1 year
    config.monte_carlo.n_simulations = 10000
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.

    Per ATLAS Checklist Section 1.6:
    - Train period: 252 days (1 year)
    - Test period: 63 days (3 months)
    - Acceptance: OOS Sharpe degradation < 30%, OOS Sharpe > 0.5

    Attributes:
        train_period: Number of bars for training (default 252 = 1 year daily)
        test_period: Number of bars for testing (default 63 = 3 months daily)
        step_size: Bars to advance between folds (default = test_period)
        min_trades_per_fold: Minimum trades required per fold (default 10)
        max_sharpe_degradation: Maximum allowed Sharpe degradation (0.30 = 30%)
        min_oos_sharpe: Minimum out-of-sample Sharpe ratio
        min_param_stability: Maximum coefficient of variation for parameters
        min_profitable_folds: Minimum percentage of profitable folds (0.60 = 60%)
    """
    train_period: int = 252
    test_period: int = 63
    step_size: Optional[int] = None  # Defaults to test_period if None
    min_trades_per_fold: int = 10
    max_sharpe_degradation: float = 0.30
    min_oos_sharpe: float = 0.5
    min_param_stability: float = 0.20
    min_profitable_folds: float = 0.60

    def __post_init__(self):
        """Set step_size to test_period if not specified."""
        if self.step_size is None:
            self.step_size = self.test_period


@dataclass
class WalkForwardConfigOptions(WalkForwardConfig):
    """
    Walk-forward configuration for options strategies.

    Options have higher variance and wider acceptance thresholds.
    Per ATLAS Checklist Section 9.5.
    """
    max_sharpe_degradation: float = 0.40  # 40% degradation allowed
    min_oos_sharpe: float = 0.3  # Lower Sharpe acceptable
    min_trades_per_fold: int = 5  # Fewer trades typical for options


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulation.

    Per ATLAS Checklist Section 1.7:
    - 10,000 iterations (production standard)
    - Bootstrap resampling with replacement
    - Acceptance: 95% CI excludes 0, P(Loss) < 20%, P(Ruin) < 5%

    Attributes:
        n_simulations: Number of Monte Carlo iterations (default 10000)
        confidence_level: Confidence level for intervals (default 0.95)
        max_probability_of_loss: Maximum acceptable P(Loss) (0.20 = 20%)
        max_probability_of_ruin: Maximum acceptable P(Ruin >50% DD) (0.05 = 5%)
        ruin_threshold: Drawdown threshold for "ruin" (0.50 = 50%)
        seed: Random seed for reproducibility (None = random)
    """
    n_simulations: int = 10000
    confidence_level: float = 0.95
    max_probability_of_loss: float = 0.20
    max_probability_of_ruin: float = 0.05
    ruin_threshold: float = 0.50
    seed: Optional[int] = None


@dataclass
class MonteCarloConfigOptions(MonteCarloConfig):
    """
    Monte Carlo configuration for options strategies.

    Options have higher variance and additional shock modeling.
    Per ATLAS Checklist Section 9.5.2.

    Attributes:
        iv_shock_std: Standard deviation for IV shock (+/- 20% default)
        theta_shock_max: Maximum theta shock multiplier on losers (1.5 = 50% worse)
        apply_options_shocks: Whether to apply IV and theta shocks
    """
    max_probability_of_loss: float = 0.30  # 30% for options
    max_probability_of_ruin: float = 0.10  # 10% for options
    iv_shock_std: float = 0.20
    theta_shock_max: float = 1.50
    apply_options_shocks: bool = True


@dataclass
class BiasDetectionConfig:
    """
    Configuration for look-ahead bias detection.

    Per ATLAS Checklist Section 1.4.

    Attributes:
        check_signal_timing: Verify signals use only past data
        check_entry_achievability: Verify entries within bar range
        check_indicator_shift: Verify indicators properly shifted
        correlation_threshold: Signal-return correlation threshold for bias
    """
    check_signal_timing: bool = True
    check_entry_achievability: bool = True
    check_indicator_shift: bool = True
    correlation_threshold: float = 0.50  # Correlation > 0.5 suggests look-ahead


@dataclass
class PatternMetricsConfig:
    """
    Configuration for STRAT pattern metrics analysis.

    Session 83C: Per-pattern, per-timeframe, per-regime breakdown.

    Attributes:
        patterns_to_track: List of pattern types to analyze
        timeframes_to_track: List of timeframes to analyze
        regimes_to_track: List of ATLAS regimes to analyze
        min_pattern_trades: Minimum trades to report pattern metrics
        optimal_delta_range: Delta range for options accuracy
    """
    patterns_to_track: List[str] = field(default_factory=lambda: [
        '2-1-2U', '2-1-2D',  # Directional continuation
        '3-1-2U', '3-1-2D',  # Outside-inside reversal
        '2D-2U', '2U-2D',    # 2-2 reversal
        '3-2U', '3-2D',      # Outside-directional
        '3-2D-2U', '3-2U-2D' # 3-2-2 reversal
    ])
    timeframes_to_track: List[str] = field(default_factory=lambda: [
        '1D', '1W', '1M'  # Daily, Weekly, Monthly
    ])
    regimes_to_track: List[str] = field(default_factory=lambda: [
        'TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL', 'CRASH'
    ])
    min_pattern_trades: int = 5
    optimal_delta_range: tuple = (0.50, 0.80)


@dataclass
class TransactionCostConfig:
    """
    Configuration for transaction cost modeling.

    Per ATLAS Checklist Section 1.1.

    Attributes:
        fees: Base fee as decimal (0.001 = 0.1%)
        slippage: Base slippage as decimal (0.001 = 0.1%)
        commission_per_contract: Options commission per contract ($0.65 typical)
        spread_cost_pct: Bid-ask spread cost (0.03 = 3% for options)
    """
    # Equity defaults
    fees: float = 0.001
    slippage: float = 0.001

    # Options-specific
    commission_per_contract: float = 0.65
    spread_cost_pct: float = 0.03


@dataclass
class AcceptanceThresholds:
    """
    Acceptance thresholds for validation.

    Two sets: equities (stricter) and options (looser).
    Per ATLAS Checklist Section 2.3.
    """
    # Equities thresholds
    equity_min_oos_sharpe: float = 0.5
    equity_max_sharpe_degradation: float = 0.30
    equity_max_probability_of_loss: float = 0.20
    equity_max_probability_of_ruin: float = 0.05
    equity_max_expected_drawdown: float = 0.25
    equity_min_trades: int = 100

    # Options thresholds (looser)
    options_min_oos_sharpe: float = 0.3
    options_max_sharpe_degradation: float = 0.40
    options_max_probability_of_loss: float = 0.30
    options_max_probability_of_ruin: float = 0.10
    options_max_expected_drawdown: float = 0.40
    options_min_trades: int = 50

    def get_thresholds(self, is_options: bool = False) -> Dict[str, float]:
        """Get appropriate thresholds based on strategy type."""
        if is_options:
            return {
                'min_oos_sharpe': self.options_min_oos_sharpe,
                'max_sharpe_degradation': self.options_max_sharpe_degradation,
                'max_probability_of_loss': self.options_max_probability_of_loss,
                'max_probability_of_ruin': self.options_max_probability_of_ruin,
                'max_expected_drawdown': self.options_max_expected_drawdown,
                'min_trades': self.options_min_trades,
            }
        else:
            return {
                'min_oos_sharpe': self.equity_min_oos_sharpe,
                'max_sharpe_degradation': self.equity_max_sharpe_degradation,
                'max_probability_of_loss': self.equity_max_probability_of_loss,
                'max_probability_of_ruin': self.equity_max_probability_of_ruin,
                'max_expected_drawdown': self.equity_max_expected_drawdown,
                'min_trades': self.equity_min_trades,
            }


@dataclass
class ValidationConfig:
    """
    Master configuration combining all validators.

    Usage:
        config = ValidationConfig()
        config.walk_forward.train_period = 252
        config.monte_carlo.n_simulations = 10000

        runner = ValidationRunner(config)
    """
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    bias_detection: BiasDetectionConfig = field(default_factory=BiasDetectionConfig)
    pattern_metrics: PatternMetricsConfig = field(default_factory=PatternMetricsConfig)
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)

    def for_options(self) -> 'ValidationConfig':
        """
        Return a copy configured for options strategies.

        Applies looser thresholds and enables options-specific features.
        """
        config = ValidationConfig(
            walk_forward=WalkForwardConfigOptions(),
            monte_carlo=MonteCarloConfigOptions(),
            bias_detection=BiasDetectionConfig(),
            pattern_metrics=self.pattern_metrics,
            transaction_costs=self.transaction_costs,
            thresholds=self.thresholds,
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'walk_forward': {
                'train_period': self.walk_forward.train_period,
                'test_period': self.walk_forward.test_period,
                'step_size': self.walk_forward.step_size,
                'min_trades_per_fold': self.walk_forward.min_trades_per_fold,
                'max_sharpe_degradation': self.walk_forward.max_sharpe_degradation,
                'min_oos_sharpe': self.walk_forward.min_oos_sharpe,
            },
            'monte_carlo': {
                'n_simulations': self.monte_carlo.n_simulations,
                'confidence_level': self.monte_carlo.confidence_level,
                'max_probability_of_loss': self.monte_carlo.max_probability_of_loss,
                'max_probability_of_ruin': self.monte_carlo.max_probability_of_ruin,
            },
            'bias_detection': {
                'check_signal_timing': self.bias_detection.check_signal_timing,
                'check_entry_achievability': self.bias_detection.check_entry_achievability,
                'check_indicator_shift': self.bias_detection.check_indicator_shift,
            },
        }
