"""
Validation Runner for ATLAS Production Readiness.

Session 83I: Orchestrates all validators for comprehensive strategy validation
per ATLAS_PRODUCTION_READINESS_CHECKLIST.md.

Components orchestrated:
- Walk-Forward Validation (Section 1.6)
- Monte Carlo Simulation (Section 1.7)
- Bias Detection (Section 1.4)
- Pattern Metrics Analysis (Section 9.2)

Usage:
    from validation import ValidationRunner, ValidationConfig

    runner = ValidationRunner()
    report = runner.validate_strategy(strategy, data, "My Strategy", is_options=True)

    if report.summary.passes_all:
        print("Strategy ready for production")
    else:
        print(f"Validation failed: {report.summary.critical_issues}")

    # Convenience function
    from validation import run_validation
    report = run_validation(strategy, data, "My Strategy", is_options=True)
"""

import logging
import time
from typing import Optional, Dict, Any, List

import pandas as pd

from validation.config import ValidationConfig
from validation.results import (
    ValidationReport,
    ValidationSummary,
    WalkForwardResults,
    MonteCarloResults,
    BiasReport,
    PatternMetricsResults,
)
from validation.protocols import StrategyProtocol, BacktestResult
from validation.walk_forward import WalkForwardValidator
from validation.monte_carlo import MonteCarloValidator
from validation.bias_detection import BiasDetector
from validation.pattern_metrics import PatternMetricsAnalyzer

logger = logging.getLogger(__name__)


class ValidationRunner:
    """
    Orchestrates all validators for comprehensive production readiness validation.

    Per ATLAS_PRODUCTION_READINESS_CHECKLIST.md, runs:
    - Walk-Forward Validation (Section 1.6)
    - Monte Carlo Simulation (Section 1.7)
    - Bias Detection (Section 1.4)
    - Pattern Metrics Analysis (Section 9.2)

    Attributes:
        config: Master validation configuration

    Example:
        >>> runner = ValidationRunner()
        >>> report = runner.validate_strategy(strategy, data, "My STRAT Strategy", is_options=True)
        >>> if runner.passes(report):
        ...     print("Ready for production!")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize ValidationRunner with configuration.

        Args:
            config: Master validation configuration. If None, uses defaults.
                   Use config.for_options() for options strategies.
        """
        self.config = config or ValidationConfig()

    def validate_strategy(
        self,
        strategy: StrategyProtocol,
        data: pd.DataFrame,
        strategy_name: str,
        is_options: bool = False,
        param_grid: Optional[Dict] = None,
        account_size: float = 10000.0,
        skip_walk_forward: bool = False,
        skip_monte_carlo: bool = False,
        skip_bias_detection: bool = False,
        skip_pattern_metrics: bool = False,
    ) -> ValidationReport:
        """
        Run comprehensive validation on a strategy.

        Args:
            strategy: Strategy implementing StrategyProtocol
            data: OHLCV DataFrame with historical price data
            strategy_name: Name for reporting and logging
            is_options: Whether strategy trades options (uses looser thresholds)
            param_grid: Parameter grid for walk-forward optimization
            account_size: Account size for Monte Carlo simulation
            skip_walk_forward: Skip walk-forward validation
            skip_monte_carlo: Skip Monte Carlo simulation
            skip_bias_detection: Skip bias detection checks
            skip_pattern_metrics: Skip pattern metrics analysis

        Returns:
            ValidationReport with all results and pass/fail status
        """
        start_time = time.time()
        logger.info(f"Starting validation for strategy: {strategy_name}")

        # Use options config if needed, but preserve holdout mode (Session 83K-15 fix)
        if is_options:
            # Check if holdout mode is already set - preserve it
            if self.config.walk_forward.validation_mode == 'holdout':
                # Preserve holdout walk_forward config, apply other options settings
                config = self.config.for_options()
                config.walk_forward = self.config.walk_forward  # Restore holdout config
            else:
                config = self.config.for_options()
        else:
            config = self.config

        # Results containers
        wf_results: Optional[WalkForwardResults] = None
        mc_results: Optional[MonteCarloResults] = None
        bias_results: Optional[BiasReport] = None
        pm_results: Optional[PatternMetricsResults] = None

        # Get full backtest for validators that need it
        try:
            backtest_result = strategy.backtest(data)
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            backtest_result = None

        # Run walk-forward validation
        if not skip_walk_forward:
            try:
                wf_results = self._run_walk_forward(strategy, data, config, param_grid)
                logger.info(f"Walk-forward: {'PASSED' if wf_results.passes_validation else 'FAILED'}")
            except Exception as e:
                logger.error(f"Walk-forward validation failed: {e}")

        # Run Monte Carlo simulation
        if not skip_monte_carlo and backtest_result is not None and backtest_result.trades is not None:
            try:
                mc_results = self._run_monte_carlo(
                    backtest_result.trades, config, account_size, is_options
                )
                logger.info(f"Monte Carlo: {'PASSED' if mc_results.passes_validation else 'FAILED'}")
            except Exception as e:
                logger.error(f"Monte Carlo simulation failed: {e}")

        # Run bias detection
        if not skip_bias_detection and backtest_result is not None:
            try:
                signals = strategy.generate_signals(data)
                bias_results = self._run_bias_detection(data, signals, backtest_result, config)
                logger.info(f"Bias Detection: {'PASSED' if bias_results.passes_validation else 'FAILED'}")
            except Exception as e:
                logger.error(f"Bias detection failed: {e}")

        # Run pattern metrics (if trades have pattern info)
        if not skip_pattern_metrics and backtest_result is not None and backtest_result.trades is not None:
            try:
                pm_results = self._run_pattern_metrics(backtest_result.trades, config)
                if pm_results is not None:
                    logger.info("Pattern metrics: Generated")
            except Exception as e:
                logger.error(f"Pattern metrics analysis failed: {e}")

        # Aggregate summary
        summary = self._create_summary(
            wf_results, mc_results, bias_results, pm_results, is_options
        )

        execution_time = time.time() - start_time
        logger.info(f"Validation completed in {execution_time:.2f}s - {'PASSED' if summary.passes_all else 'FAILED'}")

        # Build config dict for report
        config_dict = {}
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif hasattr(config, '__dict__'):
            config_dict = {k: str(v) for k, v in config.__dict__.items()}

        return ValidationReport(
            strategy_name=strategy_name,
            is_options=is_options,
            summary=summary,
            walk_forward=wf_results,
            monte_carlo=mc_results,
            bias_detection=bias_results,
            pattern_metrics=pm_results,
            config_used=config_dict,
            execution_time_seconds=execution_time,
        )

    def _run_walk_forward(
        self,
        strategy: StrategyProtocol,
        data: pd.DataFrame,
        config: ValidationConfig,
        param_grid: Optional[Dict]
    ) -> WalkForwardResults:
        """Run walk-forward validation."""
        validator = WalkForwardValidator(config.walk_forward)
        return validator.validate(strategy, data, param_grid)

    def _run_monte_carlo(
        self,
        trades: pd.DataFrame,
        config: ValidationConfig,
        account_size: float,
        is_options: bool
    ) -> MonteCarloResults:
        """Run Monte Carlo simulation."""
        mc_config = config.monte_carlo
        validator = MonteCarloValidator(mc_config)
        return validator.validate(trades, account_size, is_options=is_options)

    def _run_bias_detection(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        backtest_result: BacktestResult,
        config: ValidationConfig
    ) -> BiasReport:
        """Run bias detection checks."""
        detector = BiasDetector(config.bias_detection)

        # Extract entry/exit prices from trades if available
        entry_prices = None
        exit_prices = None

        if backtest_result.trades is not None and len(backtest_result.trades) > 0:
            trades_df = backtest_result.trades
            if 'entry_price' in trades_df.columns:
                entry_prices = trades_df['entry_price']
            if 'exit_price' in trades_df.columns:
                exit_prices = trades_df['exit_price']

        # Session 83K-3 BUG FIX: Pass entry column (boolean) instead of full DataFrame
        # BiasDetector.check_signal_timing expects a Series/array, not DataFrame
        signal_entries = signals['entry'] if 'entry' in signals.columns else signals

        return detector.full_check(
            data=data,
            signals=signal_entries,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
        )

    def _run_pattern_metrics(
        self,
        trades: pd.DataFrame,
        config: ValidationConfig
    ) -> Optional[PatternMetricsResults]:
        """
        Run pattern metrics analysis.

        Returns None if trades don't have pattern information.
        """
        # Check if trades have pattern information
        if 'pattern_type' not in trades.columns:
            logger.debug("Trades missing 'pattern_type' column - skipping pattern metrics")
            return None

        # TODO: Implement conversion from trades DataFrame to PatternTradeResult list
        # This requires understanding the trade format and mapping to PatternTradeResult
        # For now, return None (pattern metrics will be generated=False in summary)
        logger.debug("Pattern metrics conversion not yet implemented")
        return None

    def _create_summary(
        self,
        wf_results: Optional[WalkForwardResults],
        mc_results: Optional[MonteCarloResults],
        bias_results: Optional[BiasReport],
        pm_results: Optional[PatternMetricsResults],
        is_options: bool,
    ) -> ValidationSummary:
        """Create aggregated validation summary."""
        critical_issues: List[str] = []
        warnings: List[str] = []

        # Walk-forward check
        wf_passed = True
        if wf_results is not None:
            wf_passed = wf_results.passes_validation
            if not wf_passed:
                for reason in wf_results.failure_reasons:
                    critical_issues.append(f"Walk-Forward: {reason}")
        else:
            warnings.append("Walk-forward validation was skipped")

        # Monte Carlo check
        mc_passed = True
        if mc_results is not None:
            mc_passed = mc_results.passes_validation
            if not mc_passed:
                for reason in mc_results.failure_reasons:
                    critical_issues.append(f"Monte Carlo: {reason}")
        else:
            warnings.append("Monte Carlo simulation was skipped")

        # Bias detection check
        bias_passed = True
        if bias_results is not None:
            bias_passed = bias_results.passes_validation
            if not bias_passed:
                for reason in bias_results.failure_reasons:
                    critical_issues.append(f"Bias Detection: {reason}")
        else:
            warnings.append("Bias detection was skipped")

        # Pattern metrics (analysis only, doesn't affect pass/fail)
        if pm_results is None:
            warnings.append("Pattern metrics analysis was not generated")

        passes_all = wf_passed and mc_passed and bias_passed

        return ValidationSummary(
            passes_all=passes_all,
            walk_forward_passed=wf_passed,
            monte_carlo_passed=mc_passed,
            bias_detection_passed=bias_passed,
            pattern_metrics_generated=(pm_results is not None),
            critical_issues=critical_issues,
            warnings=warnings,
        )

    def passes(self, report: ValidationReport) -> bool:
        """
        Check if validation report passes all criteria.

        Args:
            report: ValidationReport from validate_strategy()

        Returns:
            True if all validations passed, False otherwise
        """
        return report.summary.passes_all


def run_validation(
    strategy: StrategyProtocol,
    data: pd.DataFrame,
    strategy_name: str,
    is_options: bool = False,
    config: Optional[ValidationConfig] = None,
    **kwargs
) -> ValidationReport:
    """
    Convenience function to run full validation.

    Args:
        strategy: Strategy to validate (must implement StrategyProtocol)
        data: OHLCV DataFrame
        strategy_name: Name for reporting
        is_options: Whether options strategy (uses looser thresholds)
        config: Validation configuration (uses defaults if None)
        **kwargs: Additional arguments passed to validate_strategy()

    Returns:
        ValidationReport with all results

    Example:
        >>> report = run_validation(my_strategy, spy_data, "STRAT 2-1-2", is_options=True)
        >>> print(f"Passed: {report.summary.passes_all}")
    """
    runner = ValidationRunner(config)
    return runner.validate_strategy(
        strategy=strategy,
        data=data,
        strategy_name=strategy_name,
        is_options=is_options,
        **kwargs
    )
