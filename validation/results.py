"""
ATLAS Production Readiness Validation - Results Module

Defines result dataclasses for all validators.
Each validator returns a specific results type with pass/fail logic.

Session 83C: Foundation for ATLAS compliance validation.

Usage:
    from validation.results import WalkForwardResults, MonteCarloResults

    # Check if validation passes
    if wf_results.passes_validation:
        print("Walk-forward validation PASSED")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import json
from datetime import datetime


@dataclass
class FoldResult:
    """
    Results for a single walk-forward fold.

    Attributes:
        fold_number: Sequential fold identifier
        train_start: Training period start date
        train_end: Training period end date
        test_start: Test period start date
        test_end: Test period end date
        is_sharpe: In-sample Sharpe ratio
        oos_sharpe: Out-of-sample Sharpe ratio
        is_return: In-sample total return
        oos_return: Out-of-sample total return
        is_trades: In-sample trade count
        oos_trades: Out-of-sample trade count
        parameters: Best parameters from training
        is_profitable: Whether OOS return > 0
    """
    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    is_sharpe: float
    oos_sharpe: float
    is_return: float
    oos_return: float
    is_trades: int
    oos_trades: int
    parameters: Dict[str, Any]
    is_profitable: bool = False

    def __post_init__(self):
        """Calculate is_profitable if not set."""
        self.is_profitable = self.oos_return > 0

    @property
    def sharpe_degradation(self) -> float:
        """Calculate Sharpe degradation for this fold."""
        if self.is_sharpe == 0:
            return 1.0
        return 1.0 - (self.oos_sharpe / self.is_sharpe)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fold_number': self.fold_number,
            'train_start': self.train_start.isoformat() if self.train_start else None,
            'train_end': self.train_end.isoformat() if self.train_end else None,
            'test_start': self.test_start.isoformat() if self.test_start else None,
            'test_end': self.test_end.isoformat() if self.test_end else None,
            'is_sharpe': self.is_sharpe,
            'oos_sharpe': self.oos_sharpe,
            'is_return': self.is_return,
            'oos_return': self.oos_return,
            'is_trades': self.is_trades,
            'oos_trades': self.oos_trades,
            'sharpe_degradation': self.sharpe_degradation,
            'is_profitable': self.is_profitable,
            'parameters': self.parameters,
        }


@dataclass
class WalkForwardResults:
    """
    Results from walk-forward validation.

    Per ATLAS Checklist Section 1.6:
    - OOS Sharpe degradation < 30% (equities) / 40% (options)
    - OOS Sharpe > 0.5 (equities) / 0.3 (options)
    - > 60% of folds profitable
    - Parameter stability CV < 20%
    """
    folds: List[FoldResult]
    avg_is_sharpe: float
    avg_oos_sharpe: float
    sharpe_degradation: float
    param_stability: Dict[str, float]  # CV per parameter
    profitable_folds_pct: float
    total_folds: int
    passes_validation: bool
    failure_reasons: List[str] = field(default_factory=list)

    # Thresholds used (for reference)
    max_sharpe_degradation: float = 0.30
    min_oos_sharpe: float = 0.5
    min_profitable_folds: float = 0.60
    max_param_cv: float = 0.20

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'avg_is_sharpe': self.avg_is_sharpe,
            'avg_oos_sharpe': self.avg_oos_sharpe,
            'sharpe_degradation': self.sharpe_degradation,
            'param_stability': self.param_stability,
            'profitable_folds_pct': self.profitable_folds_pct,
            'total_folds': self.total_folds,
            'passes_validation': self.passes_validation,
            'failure_reasons': self.failure_reasons,
            'folds': [f.to_dict() for f in self.folds],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passes_validation else "FAILED"
        lines = [
            "=" * 60,
            f"WALK-FORWARD VALIDATION: {status}",
            "=" * 60,
            f"Total Folds:           {self.total_folds}",
            f"Avg IS Sharpe:         {self.avg_is_sharpe:.2f}",
            f"Avg OOS Sharpe:        {self.avg_oos_sharpe:.2f}",
            f"Sharpe Degradation:    {self.sharpe_degradation:.1%} (max: {self.max_sharpe_degradation:.1%})",
            f"Profitable Folds:      {self.profitable_folds_pct:.1%} (min: {self.min_profitable_folds:.1%})",
            "",
            "Parameter Stability (CV):",
        ]
        for param, cv in self.param_stability.items():
            status_char = "PASS" if cv <= self.max_param_cv else "FAIL"
            lines.append(f"  {param}: {cv:.2%} [{status_char}]")

        if self.failure_reasons:
            lines.append("")
            lines.append("Failure Reasons:")
            for reason in self.failure_reasons:
                lines.append(f"  - {reason}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class MonteCarloResults:
    """
    Results from Monte Carlo simulation.

    Per ATLAS Checklist Section 1.7:
    - 95% CI for Sharpe does not include 0
    - Probability of loss < 20% (equities) / 30% (options)
    - Probability of ruin (>50% DD) < 5% (equities) / 10% (options)
    """
    original_sharpe: float
    simulated_sharpe_mean: float
    simulated_sharpe_std: float
    sharpe_95_ci: Tuple[float, float]
    original_max_dd: float
    simulated_max_dd_95: float
    max_dd_95_ci: Tuple[float, float]
    return_95_ci: Tuple[float, float]
    probability_of_loss: float
    probability_of_ruin: float
    n_simulations: int
    passes_validation: bool
    failure_reasons: List[str] = field(default_factory=list)

    # Simulated distributions (for advanced analysis)
    simulated_sharpes: Optional[np.ndarray] = None
    simulated_max_dds: Optional[np.ndarray] = None
    simulated_returns: Optional[np.ndarray] = None

    # Thresholds used
    max_probability_of_loss: float = 0.20
    max_probability_of_ruin: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes large arrays)."""
        return {
            'original_sharpe': self.original_sharpe,
            'simulated_sharpe_mean': self.simulated_sharpe_mean,
            'simulated_sharpe_std': self.simulated_sharpe_std,
            'sharpe_95_ci': list(self.sharpe_95_ci),
            'original_max_dd': self.original_max_dd,
            'simulated_max_dd_95': self.simulated_max_dd_95,
            'max_dd_95_ci': list(self.max_dd_95_ci),
            'return_95_ci': list(self.return_95_ci),
            'probability_of_loss': self.probability_of_loss,
            'probability_of_ruin': self.probability_of_ruin,
            'n_simulations': self.n_simulations,
            'passes_validation': self.passes_validation,
            'failure_reasons': self.failure_reasons,
        }

    @property
    def sharpe_ci_excludes_zero(self) -> bool:
        """Check if 95% CI for Sharpe excludes zero."""
        return self.sharpe_95_ci[0] > 0

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passes_validation else "FAILED"
        lines = [
            "=" * 60,
            f"MONTE CARLO SIMULATION: {status}",
            "=" * 60,
            f"Simulations:           {self.n_simulations:,}",
            f"Original Sharpe:       {self.original_sharpe:.2f}",
            f"Simulated Sharpe:      {self.simulated_sharpe_mean:.2f} +/- {self.simulated_sharpe_std:.2f}",
            f"Sharpe 95% CI:         [{self.sharpe_95_ci[0]:.2f}, {self.sharpe_95_ci[1]:.2f}]",
            f"CI Excludes Zero:      {'YES' if self.sharpe_ci_excludes_zero else 'NO'}",
            "",
            f"Original Max DD:       {self.original_max_dd:.1%}",
            f"95th Percentile DD:    {self.simulated_max_dd_95:.1%}",
            "",
            f"P(Loss):               {self.probability_of_loss:.1%} (max: {self.max_probability_of_loss:.1%})",
            f"P(Ruin):               {self.probability_of_ruin:.1%} (max: {self.max_probability_of_ruin:.1%})",
        ]

        if self.failure_reasons:
            lines.append("")
            lines.append("Failure Reasons:")
            for reason in self.failure_reasons:
                lines.append(f"  - {reason}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class BiasCheckResult:
    """
    Result for a single bias check.
    """
    check_name: str
    passed: bool
    details: str
    severity: str = 'warning'  # 'warning', 'error', 'critical'
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class BiasReport:
    """
    Results from look-ahead bias detection.

    Per ATLAS Checklist Section 1.4.
    """
    checks: List[BiasCheckResult]
    passes_validation: bool
    bias_detected: bool
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passes_validation': self.passes_validation,
            'bias_detected': self.bias_detected,
            'checks': [
                {
                    'check_name': c.check_name,
                    'passed': c.passed,
                    'details': c.details,
                    'severity': c.severity,
                    'metric_value': c.metric_value,
                    'threshold': c.threshold,
                }
                for c in self.checks
            ],
            'failure_reasons': self.failure_reasons,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passes_validation else "FAILED"
        lines = [
            "=" * 60,
            f"BIAS DETECTION: {status}",
            "=" * 60,
        ]

        for check in self.checks:
            check_status = "PASS" if check.passed else check.severity.upper()
            lines.append(f"[{check_status}] {check.check_name}: {check.details}")

        if self.failure_reasons:
            lines.append("")
            lines.append("Failure Reasons:")
            for reason in self.failure_reasons:
                lines.append(f"  - {reason}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class PatternStats:
    """
    Statistics for a single pattern type.
    """
    pattern_type: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_winner_pnl: float
    avg_loser_pnl: float
    profit_factor: float
    expectancy: float
    avg_risk_reward: float
    avg_days_held: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_type': self.pattern_type,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl': self.avg_pnl,
            'avg_winner_pnl': self.avg_winner_pnl,
            'avg_loser_pnl': self.avg_loser_pnl,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_risk_reward': self.avg_risk_reward,
            'avg_days_held': self.avg_days_held,
        }


@dataclass
class OptionsAccuracyMetrics:
    """
    Options-specific accuracy metrics.
    """
    thetadata_trades: int
    black_scholes_trades: int
    mixed_trades: int
    thetadata_coverage_pct: float
    price_mae: float
    price_mape: float
    price_rmse: float
    delta_mae: float
    avg_entry_delta: float
    avg_exit_delta: float
    delta_in_optimal_range_pct: float
    avg_theta_cost: float
    theta_cost_as_pct_of_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'thetadata_trades': self.thetadata_trades,
            'black_scholes_trades': self.black_scholes_trades,
            'mixed_trades': self.mixed_trades,
            'thetadata_coverage_pct': self.thetadata_coverage_pct,
            'price_mae': self.price_mae,
            'price_mape': self.price_mape,
            'price_rmse': self.price_rmse,
            'delta_mae': self.delta_mae,
            'avg_entry_delta': self.avg_entry_delta,
            'avg_exit_delta': self.avg_exit_delta,
            'delta_in_optimal_range_pct': self.delta_in_optimal_range_pct,
            'avg_theta_cost': self.avg_theta_cost,
            'theta_cost_as_pct_of_pnl': self.theta_cost_as_pct_of_pnl,
        }


@dataclass
class PatternMetricsResults:
    """
    Results from pattern metrics analysis.
    """
    by_pattern: Dict[str, PatternStats]
    by_timeframe: Dict[str, Dict[str, PatternStats]]
    by_regime: Dict[str, Dict[str, PatternStats]]
    options_accuracy: Optional[OptionsAccuracyMetrics]
    total_trades: int
    overall_win_rate: float
    overall_expectancy: float
    best_pattern: Optional[str]
    worst_pattern: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'by_pattern': {k: v.to_dict() for k, v in self.by_pattern.items()},
            'by_timeframe': {
                tf: {k: v.to_dict() for k, v in patterns.items()}
                for tf, patterns in self.by_timeframe.items()
            },
            'by_regime': {
                regime: {k: v.to_dict() for k, v in patterns.items()}
                for regime, patterns in self.by_regime.items()
            },
            'options_accuracy': self.options_accuracy.to_dict() if self.options_accuracy else None,
            'total_trades': self.total_trades,
            'overall_win_rate': self.overall_win_rate,
            'overall_expectancy': self.overall_expectancy,
            'best_pattern': self.best_pattern,
            'worst_pattern': self.worst_pattern,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 70,
            "PATTERN METRICS ANALYSIS",
            "=" * 70,
            f"Total Trades:      {self.total_trades}",
            f"Overall Win Rate:  {self.overall_win_rate:.1%}",
            f"Overall Expectancy: ${self.overall_expectancy:.2f}",
            "",
            "--- BY PATTERN TYPE ---",
            f"{'Pattern':<12} {'Trades':>8} {'Win Rate':>10} {'Avg P/L':>10} {'Expectancy':>12}",
            "-" * 60,
        ]

        for pattern_type, stats in self.by_pattern.items():
            lines.append(
                f"{stats.pattern_type:<12} {stats.trade_count:>8} "
                f"{stats.win_rate:>10.1%} ${stats.avg_pnl:>9.2f} "
                f"${stats.expectancy:>11.2f}"
            )

        if self.best_pattern:
            lines.append(f"\nBest Pattern:  {self.best_pattern}")
        if self.worst_pattern:
            lines.append(f"Worst Pattern: {self.worst_pattern}")

        if self.options_accuracy:
            oa = self.options_accuracy
            lines.extend([
                "",
                "--- OPTIONS ACCURACY ---",
                f"ThetaData Coverage:  {oa.thetadata_coverage_pct:.1%}",
                f"Price MAPE:          {oa.price_mape:.1%}",
                f"Delta in Opt Range:  {oa.delta_in_optimal_range_pct:.1%}",
            ])

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class ValidationSummary:
    """
    Overall validation summary combining all validators.
    """
    passes_all: bool
    walk_forward_passed: bool
    monte_carlo_passed: bool
    bias_detection_passed: bool
    pattern_metrics_generated: bool
    critical_issues: List[str]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passes_all': self.passes_all,
            'walk_forward_passed': self.walk_forward_passed,
            'monte_carlo_passed': self.monte_carlo_passed,
            'bias_detection_passed': self.bias_detection_passed,
            'pattern_metrics_generated': self.pattern_metrics_generated,
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class ValidationReport:
    """
    Complete validation report containing all results.
    """
    strategy_name: str
    is_options: bool
    summary: ValidationSummary
    walk_forward: Optional[WalkForwardResults]
    monte_carlo: Optional[MonteCarloResults]
    bias_detection: Optional[BiasReport]
    pattern_metrics: Optional[PatternMetricsResults]
    config_used: Dict[str, Any]
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'is_options': self.is_options,
            'summary': self.summary.to_dict(),
            'walk_forward': self.walk_forward.to_dict() if self.walk_forward else None,
            'monte_carlo': self.monte_carlo.to_dict() if self.monte_carlo else None,
            'bias_detection': self.bias_detection.to_dict() if self.bias_detection else None,
            'pattern_metrics': self.pattern_metrics.to_dict() if self.pattern_metrics else None,
            'config_used': self.config_used,
            'execution_time_seconds': self.execution_time_seconds,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self) -> str:
        """Print human-readable summary."""
        status = "PRODUCTION READY" if self.summary.passes_all else "NOT READY"
        strategy_type = "Options" if self.is_options else "Equity"

        lines = [
            "",
            "=" * 70,
            f"ATLAS PRODUCTION READINESS VALIDATION REPORT",
            "=" * 70,
            f"Strategy:     {self.strategy_name}",
            f"Type:         {strategy_type}",
            f"Status:       {status}",
            f"Timestamp:    {self.summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Exec Time:    {self.execution_time_seconds:.1f}s",
            "",
            "--- VALIDATION RESULTS ---",
            f"Walk-Forward:    {'PASS' if self.summary.walk_forward_passed else 'FAIL'}",
            f"Monte Carlo:     {'PASS' if self.summary.monte_carlo_passed else 'FAIL'}",
            f"Bias Detection:  {'PASS' if self.summary.bias_detection_passed else 'FAIL'}",
            f"Pattern Metrics: {'Generated' if self.summary.pattern_metrics_generated else 'N/A'}",
        ]

        if self.summary.critical_issues:
            lines.append("")
            lines.append("CRITICAL ISSUES:")
            for issue in self.summary.critical_issues:
                lines.append(f"  [X] {issue}")

        if self.summary.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in self.summary.warnings:
                lines.append(f"  [!] {warning}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
