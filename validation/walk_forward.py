"""
ATLAS Production Readiness Validation - Walk-Forward Validator

Implements rolling window walk-forward validation per ATLAS Checklist Section 1.6.

Key Features:
- Train period: 252 days (1 year) default
- Test period: 63 days (3 months) default
- Rolling window with configurable step size
- Sharpe degradation calculation (IS vs OOS)
- Parameter stability via coefficient of variation
- Fold-by-fold performance tracking

Session 83D: Walk-forward validation implementation.

Usage:
    from validation.walk_forward import WalkForwardValidator
    from validation import WalkForwardConfig

    validator = WalkForwardValidator(WalkForwardConfig())
    results = validator.validate(strategy, data)

    if results.passes_validation:
        print("Strategy passes walk-forward validation")
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from validation.config import WalkForwardConfig
from validation.results import FoldResult, WalkForwardResults
from validation.protocols import StrategyProtocol, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class FoldWindow:
    """
    Defines train/test window indices for a single fold.

    Attributes:
        fold_number: Sequential fold identifier (1-indexed)
        train_start_idx: Index of first training bar
        train_end_idx: Index of last training bar (exclusive)
        test_start_idx: Index of first test bar
        test_end_idx: Index of last test bar (exclusive)
    """
    fold_number: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int

    @property
    def train_size(self) -> int:
        """Number of bars in training window."""
        return self.train_end_idx - self.train_start_idx

    @property
    def test_size(self) -> int:
        """Number of bars in test window."""
        return self.test_end_idx - self.test_start_idx


class WalkForwardValidator:
    """
    Walk-forward validation for strategy robustness testing.

    Implements rolling window out-of-sample testing per ATLAS Checklist Section 1.6:
    - Train on historical data, test on subsequent period
    - Roll forward by test_period to create multiple folds
    - Track Sharpe degradation between in-sample and out-of-sample
    - Measure parameter stability across folds

    Acceptance Criteria (equities):
    - OOS Sharpe degradation < 30%
    - OOS Sharpe > 0.5 average
    - > 60% of folds profitable
    - Parameter stability CV < 20%

    For options strategies, use WalkForwardConfigOptions with looser thresholds.

    Example:
        config = WalkForwardConfig(
            train_period=252,  # 1 year
            test_period=63,    # 3 months
        )
        validator = WalkForwardValidator(config)

        results = validator.validate(strategy, data)
        print(results.summary())

        if results.passes_validation:
            print("Strategy is robust to out-of-sample testing")
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize walk-forward validator.

        Args:
            config: Walk-forward configuration. Uses defaults if None.
        """
        self.config = config or WalkForwardConfig()

    def validate(
        self,
        strategy: StrategyProtocol,
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> WalkForwardResults:
        """
        Run walk-forward validation on strategy.

        Args:
            strategy: Strategy implementing StrategyProtocol (backtest, optimize methods)
            data: OHLCV DataFrame with DatetimeIndex
            param_grid: Optional parameter grid for optimization

        Returns:
            WalkForwardResults with per-fold metrics and pass/fail status
        """
        logger.info(f"Starting walk-forward validation: {len(data)} bars")

        # Generate fold windows
        windows = self._generate_fold_windows(len(data))

        if len(windows) == 0:
            logger.warning("Insufficient data for walk-forward validation")
            return self._create_empty_results("Insufficient data for any folds")

        logger.info(f"Generated {len(windows)} folds")

        # Run validation for each fold
        fold_results: List[FoldResult] = []
        all_parameters: List[Dict[str, Any]] = []

        for window in windows:
            fold_result = self._validate_fold(
                strategy=strategy,
                data=data,
                window=window,
                param_grid=param_grid
            )
            fold_results.append(fold_result)
            all_parameters.append(fold_result.parameters)

            logger.info(
                f"Fold {window.fold_number}: "
                f"IS Sharpe={fold_result.is_sharpe:.2f}, "
                f"OOS Sharpe={fold_result.oos_sharpe:.2f}, "
                f"Degradation={fold_result.sharpe_degradation:.1%}"
            )

        # Calculate aggregate metrics
        results = self._aggregate_results(fold_results, all_parameters)

        logger.info(
            f"Walk-forward complete: {'PASSED' if results.passes_validation else 'FAILED'}"
        )

        return results

    def _generate_fold_windows(self, total_bars: int) -> List[FoldWindow]:
        """
        Generate fold window definitions based on validation mode.

        Session 83K-14: Added holdout mode for sparse pattern strategies.

        Modes:
        - walk_forward: Rolling windows with multiple folds (default)
        - holdout: Single 70/30 train/test split

        Args:
            total_bars: Total number of bars in dataset

        Returns:
            List of FoldWindow objects defining each fold
        """
        # Session 83K-14: Holdout mode - single 70/30 split
        if self.config.validation_mode == 'holdout':
            split_idx = int(total_bars * self.config.holdout_train_pct)
            logger.info(
                f"Holdout mode: {self.config.holdout_train_pct:.0%} train "
                f"({split_idx} bars), {1-self.config.holdout_train_pct:.0%} test "
                f"({total_bars - split_idx} bars)"
            )
            return [FoldWindow(
                fold_number=1,
                train_start_idx=0,
                train_end_idx=split_idx,
                test_start_idx=split_idx,
                test_end_idx=total_bars
            )]

        # Standard walk-forward: rolling windows
        windows = []
        train_period = self.config.train_period
        test_period = self.config.test_period
        step_size = self.config.step_size

        # First fold starts at index 0 for training
        current_idx = train_period
        fold_num = 1

        while current_idx + test_period <= total_bars:
            window = FoldWindow(
                fold_number=fold_num,
                train_start_idx=current_idx - train_period,
                train_end_idx=current_idx,
                test_start_idx=current_idx,
                test_end_idx=current_idx + test_period
            )
            windows.append(window)

            # Advance by step size
            current_idx += step_size
            fold_num += 1

        return windows

    def _validate_fold(
        self,
        strategy: StrategyProtocol,
        data: pd.DataFrame,
        window: FoldWindow,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> FoldResult:
        """
        Validate a single fold: optimize on train, test on OOS.

        Args:
            strategy: Strategy to validate
            data: Full dataset
            window: Fold window definition
            param_grid: Optional parameter grid

        Returns:
            FoldResult with IS and OOS metrics
        """
        # Split data
        train_data = data.iloc[window.train_start_idx:window.train_end_idx].copy()
        test_data = data.iloc[window.test_start_idx:window.test_end_idx].copy()

        # Get dates for reporting
        train_start = train_data.index[0] if hasattr(train_data.index[0], 'to_pydatetime') else train_data.index[0]
        train_end = train_data.index[-1] if hasattr(train_data.index[-1], 'to_pydatetime') else train_data.index[-1]
        test_start = test_data.index[0] if hasattr(test_data.index[0], 'to_pydatetime') else test_data.index[0]
        test_end = test_data.index[-1] if hasattr(test_data.index[-1], 'to_pydatetime') else test_data.index[-1]

        # Convert to datetime if needed
        if isinstance(train_start, pd.Timestamp):
            train_start = train_start.to_pydatetime()
        if isinstance(train_end, pd.Timestamp):
            train_end = train_end.to_pydatetime()
        if isinstance(test_start, pd.Timestamp):
            test_start = test_start.to_pydatetime()
        if isinstance(test_end, pd.Timestamp):
            test_end = test_end.to_pydatetime()

        # Phase 1: Optimize on training data
        try:
            best_params, is_result = strategy.optimize(train_data, param_grid)
        except Exception as e:
            logger.warning(f"Fold {window.fold_number} optimization failed: {e}")
            # Fallback to default parameters
            best_params = {}
            is_result = strategy.backtest(train_data)

        # Phase 2: Test on out-of-sample data (NO REOPTIMIZATION)
        try:
            oos_result = strategy.backtest(test_data, params=best_params)
        except Exception as e:
            logger.warning(f"Fold {window.fold_number} OOS backtest failed: {e}")
            # Create empty result
            oos_result = BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                trades=pd.DataFrame(),
                equity_curve=pd.Series(dtype=float),
                trade_count=0,
                parameters=best_params
            )

        # Session 83K-16: Verify OOS trades fall within test period
        oos_trade_count = oos_result.trade_count
        if oos_result.trades is not None and not oos_result.trades.empty:
            if 'entry_date' in oos_result.trades.columns:
                # Convert test dates to comparable format
                test_start_ts = pd.Timestamp(test_start)
                test_end_ts = pd.Timestamp(test_end)

                # Check each trade's entry date
                entry_dates = pd.to_datetime(oos_result.trades['entry_date'])
                trades_in_period = (entry_dates >= test_start_ts) & (entry_dates <= test_end_ts)
                trades_in_period_count = trades_in_period.sum()

                if trades_in_period_count != len(oos_result.trades):
                    trades_outside = len(oos_result.trades) - trades_in_period_count
                    logger.warning(
                        f"Fold {window.fold_number}: {trades_outside} of {len(oos_result.trades)} "
                        f"OOS trades have entry dates outside test period "
                        f"({test_start_ts.date()} to {test_end_ts.date()}). "
                        f"This may indicate a data leakage issue in the strategy."
                    )
                    # Update trade count to only include trades in period
                    oos_trade_count = trades_in_period_count

        # Create fold result
        # Session 83K-16: Use oos_trade_count which may be filtered to test period
        return FoldResult(
            fold_number=window.fold_number,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            is_sharpe=is_result.sharpe_ratio,
            oos_sharpe=oos_result.sharpe_ratio,
            is_return=is_result.total_return,
            oos_return=oos_result.total_return,
            is_trades=is_result.trade_count,
            oos_trades=oos_trade_count,  # May be filtered count if trades outside period
            parameters=best_params,
            is_profitable=(oos_result.total_return > 0)
        )

    def _aggregate_results(
        self,
        fold_results: List[FoldResult],
        all_parameters: List[Dict[str, Any]]
    ) -> WalkForwardResults:
        """
        Aggregate fold results into overall validation results.

        Calculates:
        - Average IS/OOS Sharpe ratios
        - Overall Sharpe degradation
        - Parameter stability (CV per parameter)
        - Percentage of profitable folds
        - Pass/fail determination with reasons

        Args:
            fold_results: List of FoldResult objects
            all_parameters: List of parameter dicts from each fold

        Returns:
            WalkForwardResults with aggregate metrics
        """
        # Extract metrics
        is_sharpes = [f.is_sharpe for f in fold_results]
        oos_sharpes = [f.oos_sharpe for f in fold_results]

        # Calculate averages
        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0

        # Calculate Sharpe degradation
        # Session 83K-16: Now returns tuple (degradation, is_sign_reversal)
        sharpe_degradation, has_sign_reversal = self._calculate_sharpe_degradation(avg_is_sharpe, avg_oos_sharpe)

        # Session 83K-16: Generate sign reversal warning message if applicable
        sign_reversal_warning = None
        if has_sign_reversal:
            sign_reversal_warning = (
                f"IS/OOS Sharpe sign reversal detected (IS={avg_is_sharpe:.2f}, OOS={avg_oos_sharpe:.2f}). "
                f"This may indicate regime change, data issues, or luck. Manual review recommended."
            )
            logger.warning(sign_reversal_warning)

        # Calculate parameter stability
        param_stability = self._calculate_parameter_stability(all_parameters)

        # Calculate profitable folds percentage
        profitable_count = sum(1 for f in fold_results if f.is_profitable)
        profitable_folds_pct = profitable_count / len(fold_results) if fold_results else 0.0

        # Determine pass/fail with reasons
        failure_reasons = []
        is_holdout = self.config.validation_mode == 'holdout'

        # Check OOS Sharpe - applies to both modes
        if avg_oos_sharpe < self.config.min_oos_sharpe:
            failure_reasons.append(
                f"OOS Sharpe {avg_oos_sharpe:.2f} below minimum {self.config.min_oos_sharpe:.2f}"
            )

        # Check Sharpe degradation - applies to both modes
        if sharpe_degradation > self.config.max_sharpe_degradation:
            failure_reasons.append(
                f"Sharpe degradation {sharpe_degradation:.1%} exceeds maximum {self.config.max_sharpe_degradation:.1%}"
            )

        # Check profitable folds - SKIP for holdout mode (meaningless with 1 fold)
        # Session 83K-14: Holdout has only 1 fold, so this is always 0% or 100%
        if not is_holdout:
            if profitable_folds_pct < self.config.min_profitable_folds:
                failure_reasons.append(
                    f"Profitable folds {profitable_folds_pct:.1%} below minimum {self.config.min_profitable_folds:.1%}"
                )

        # Check parameter stability - SKIP for holdout mode (single parameter set)
        # Session 83K-14: CV cannot be calculated with only 1 fold
        if not is_holdout:
            for param_name, cv in param_stability.items():
                if cv > self.config.min_param_stability:
                    failure_reasons.append(
                        f"Parameter '{param_name}' CV {cv:.1%} exceeds maximum {self.config.min_param_stability:.1%}"
                    )

        # Check minimum trades per fold - applies to both modes
        low_trade_folds = [
            f for f in fold_results
            if f.oos_trades < self.config.min_trades_per_fold
        ]
        if low_trade_folds:
            failure_reasons.append(
                f"{len(low_trade_folds)} folds have fewer than {self.config.min_trades_per_fold} trades"
            )

        passes_validation = len(failure_reasons) == 0

        return WalkForwardResults(
            folds=fold_results,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            sharpe_degradation=sharpe_degradation,
            param_stability=param_stability,
            profitable_folds_pct=profitable_folds_pct,
            total_folds=len(fold_results),
            passes_validation=passes_validation,
            failure_reasons=failure_reasons,
            max_sharpe_degradation=self.config.max_sharpe_degradation,
            min_oos_sharpe=self.config.min_oos_sharpe,
            min_profitable_folds=self.config.min_profitable_folds,
            max_param_cv=self.config.min_param_stability,
            # Session 83K-16: Sign reversal warning
            has_sign_reversal_warning=has_sign_reversal,
            sign_reversal_warning=sign_reversal_warning,
        )

    def _calculate_sharpe_degradation(
        self,
        is_sharpe: float,
        oos_sharpe: float
    ) -> Tuple[float, bool]:
        """
        Calculate Sharpe ratio degradation from IS to OOS.

        Degradation = 1 - (OOS Sharpe / IS Sharpe)

        - 0% degradation = OOS same as IS (perfect)
        - 30% degradation = OOS is 70% of IS
        - 100% degradation = OOS is 0
        - Negative = OOS better than IS (unusual, may indicate luck)

        Session 83K-16: Added sign reversal detection. When IS and OOS Sharpe have
        opposite signs, this indicates a suspicious pattern that should be flagged
        for manual review. Could indicate regime change or data issues.

        Args:
            is_sharpe: In-sample Sharpe ratio
            oos_sharpe: Out-of-sample Sharpe ratio

        Returns:
            Tuple of (degradation, is_sign_reversal):
            - degradation: Decimal (0.30 = 30% degradation)
            - is_sign_reversal: True if IS and OOS Sharpe have opposite signs
        """
        # Session 83K-16: Detect sign reversal (IS and OOS have opposite signs)
        is_sign_reversal = (is_sharpe < 0 and oos_sharpe > 0) or (is_sharpe > 0 and oos_sharpe < 0)

        if is_sharpe <= 0:
            # Cannot calculate meaningful degradation if IS Sharpe is zero or negative
            if oos_sharpe > 0:
                # Negative IS, positive OOS - flag as sign reversal
                return (0.0, is_sign_reversal)
            return (1.0, False)  # Both bad - no sign reversal

        degradation = 1.0 - (oos_sharpe / is_sharpe)

        # Cap at reasonable range (-1 to 1)
        return (max(-1.0, min(1.0, degradation)), is_sign_reversal)

    def _calculate_parameter_stability(
        self,
        all_parameters: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate coefficient of variation (CV) for each parameter across folds.

        CV = std / mean (expressed as decimal)

        Lower CV indicates more stable parameters across different time periods.
        High CV (>20%) suggests overfitting - parameters vary wildly per fold.

        Args:
            all_parameters: List of parameter dicts from each fold

        Returns:
            Dict mapping parameter names to their CV values
        """
        if not all_parameters:
            return {}

        # Collect values per parameter
        param_values: Dict[str, List[float]] = {}

        for params in all_parameters:
            for name, value in params.items():
                # Only track numeric parameters
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if name not in param_values:
                        param_values[name] = []
                    param_values[name].append(float(value))

        # Calculate CV for each parameter
        stability = {}

        for name, values in param_values.items():
            if len(values) < 2:
                stability[name] = 0.0
                continue

            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample std

            if mean_val == 0:
                # Avoid division by zero - use std as measure if mean is 0
                stability[name] = std_val if std_val > 0 else 0.0
            else:
                stability[name] = abs(std_val / mean_val)

        return stability

    def _create_empty_results(self, reason: str) -> WalkForwardResults:
        """Create empty results for failed validation."""
        return WalkForwardResults(
            folds=[],
            avg_is_sharpe=0.0,
            avg_oos_sharpe=0.0,
            sharpe_degradation=1.0,
            param_stability={},
            profitable_folds_pct=0.0,
            total_folds=0,
            passes_validation=False,
            failure_reasons=[reason],
            max_sharpe_degradation=self.config.max_sharpe_degradation,
            min_oos_sharpe=self.config.min_oos_sharpe,
            min_profitable_folds=self.config.min_profitable_folds,
            max_param_cv=self.config.min_param_stability,
            # Session 83K-16: No sign reversal warning for empty results
            has_sign_reversal_warning=False,
            sign_reversal_warning=None,
        )

    def passes(self, results: WalkForwardResults) -> bool:
        """
        Check if results meet acceptance criteria.

        Convenience method matching ValidatorProtocol.

        Args:
            results: Results from validate() method

        Returns:
            True if validation passes
        """
        return results.passes_validation


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio from returns series.

    Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(periods)

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if returns is None or len(returns) < 2:
        return 0.0

    # Convert annual risk-free to periodic
    rf_per_period = risk_free_rate / periods_per_year

    excess_returns = returns - rf_per_period
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

    return sharpe if not np.isnan(sharpe) else 0.0


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Returns drawdown as positive decimal (0.25 = 25% drawdown).

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Maximum drawdown as positive decimal
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown from peak
    drawdown = (running_max - equity_curve) / running_max

    max_dd = drawdown.max()

    return max_dd if not np.isnan(max_dd) else 0.0
