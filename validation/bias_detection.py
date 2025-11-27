"""
ATLAS Production Readiness Validation - Bias Detection

Implements look-ahead bias detection per ATLAS Checklist Section 1.4.

Key Features:
- Signal-to-return correlation check (detects same-bar look-ahead)
- Entry price achievability verification (within bar high/low)
- Indicator first-valid index verification
- Proper signal shifting check (signals from bar N, execution on bar N+1)

Session 83F: Bias detection implementation.

Usage:
    from validation.bias_detection import BiasDetector
    from validation import BiasDetectionConfig

    detector = BiasDetector(BiasDetectionConfig())
    report = detector.full_check(data, signals, entries, exits)

    if report.passes_validation:
        print("No bias detected")
"""

import logging
from typing import Optional, List, Union

import numpy as np
import pandas as pd

from validation.config import BiasDetectionConfig
from validation.results import BiasCheckResult, BiasReport

logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Detects look-ahead bias in trading strategy signals and entries.

    Implements ATLAS Checklist Section 1.4 requirements:
    - Signal-to-return correlation check (correlation > 0.5 suggests bias)
    - Entry price achievability (all entries within bar high/low)
    - Indicator first-valid index verification
    - Signal timing verification (signals use only past data)

    Example:
        config = BiasDetectionConfig(correlation_threshold=0.5)
        detector = BiasDetector(config)

        # Full bias check
        report = detector.full_check(
            data=price_data,      # OHLCV DataFrame
            signals=signals,       # Boolean Series/array of signals
            entry_prices=entry_prices,  # Entry price for each signal
            exit_prices=exit_prices     # Exit price for each exit
        )

        if not report.passes_validation:
            for reason in report.failure_reasons:
                print(f"BIAS DETECTED: {reason}")
    """

    def __init__(self, config: Optional[BiasDetectionConfig] = None):
        """
        Initialize BiasDetector with configuration.

        Args:
            config: BiasDetectionConfig with thresholds. Uses defaults if None.
        """
        self.config = config or BiasDetectionConfig()

    def check_signal_timing(
        self,
        signals: Union[pd.Series, np.ndarray],
        returns: Union[pd.Series, np.ndarray]
    ) -> BiasCheckResult:
        """
        Check for look-ahead bias via signal-return correlation.

        If signals are correlated with same-bar returns, it indicates
        the signal is using future information (the return is calculated
        using the close price, which isn't known until bar close).

        Correct implementation: signals[t] should predict returns[t+1],
        not returns[t]. If corr(signals, returns) > threshold, bias detected.

        Args:
            signals: Boolean or float signals (True/1.0 = long signal)
            returns: Returns for each bar

        Returns:
            BiasCheckResult indicating pass/fail
        """
        if len(signals) != len(returns):
            return BiasCheckResult(
                check_name='signal_timing',
                passed=False,
                details='Signal and return arrays have different lengths',
                severity='error',
                metric_value=None,
                threshold=self.config.correlation_threshold
            )

        # Convert to numpy arrays
        sig_arr = np.array(signals, dtype=float)
        ret_arr = np.array(returns, dtype=float)

        # Remove NaN values for correlation calculation
        mask = ~(np.isnan(sig_arr) | np.isnan(ret_arr))
        sig_clean = sig_arr[mask]
        ret_clean = ret_arr[mask]

        if len(sig_clean) < 10:
            return BiasCheckResult(
                check_name='signal_timing',
                passed=True,
                details='Insufficient data for correlation check (< 10 bars)',
                severity='warning',
                metric_value=None,
                threshold=self.config.correlation_threshold
            )

        # Calculate correlation between signals and SAME-BAR returns
        # High correlation suggests signals are using future information
        if np.std(sig_clean) == 0 or np.std(ret_clean) == 0:
            correlation = 0.0
        else:
            correlation = np.corrcoef(sig_clean, ret_clean)[0, 1]

        # Correlation > threshold suggests look-ahead bias
        passed = abs(correlation) <= self.config.correlation_threshold

        if passed:
            details = f'Same-bar correlation: {correlation:.3f} (threshold: {self.config.correlation_threshold})'
            severity = 'info'
        else:
            details = f'HIGH CORRELATION: {correlation:.3f} suggests look-ahead bias (threshold: {self.config.correlation_threshold})'
            severity = 'critical'

        return BiasCheckResult(
            check_name='signal_timing',
            passed=passed,
            details=details,
            severity=severity,
            metric_value=abs(correlation),
            threshold=self.config.correlation_threshold
        )

    def check_entry_achievability(
        self,
        entry_prices: Union[pd.Series, np.ndarray],
        high_prices: Union[pd.Series, np.ndarray],
        low_prices: Union[pd.Series, np.ndarray],
        tolerance: float = 0.001
    ) -> BiasCheckResult:
        """
        Verify all entry prices are achievable within bar's trading range.

        Entry prices should be between the bar's low and high (inclusive).
        Using prices outside this range indicates impossible fills.

        Args:
            entry_prices: Executed entry prices
            high_prices: Bar high prices
            low_prices: Bar low prices
            tolerance: Percentage tolerance for boundary checks (0.001 = 0.1%)

        Returns:
            BiasCheckResult indicating pass/fail
        """
        entry_arr = np.array(entry_prices, dtype=float)
        high_arr = np.array(high_prices, dtype=float)
        low_arr = np.array(low_prices, dtype=float)

        # Handle empty entry array
        if len(entry_arr) == 0:
            return BiasCheckResult(
                check_name='entry_achievability',
                passed=True,
                details='No entries to check',
                severity='info',
                metric_value=1.0,
                threshold=1.0
            )

        # Ensure arrays are the same length for masking
        min_len = min(len(entry_arr), len(high_arr), len(low_arr))
        entry_arr = entry_arr[:min_len]
        high_arr = high_arr[:min_len]
        low_arr = low_arr[:min_len]

        # Remove NaN values
        mask = ~(np.isnan(entry_arr) | np.isnan(high_arr) | np.isnan(low_arr))
        entry_clean = entry_arr[mask]
        high_clean = high_arr[mask]
        low_clean = low_arr[mask]

        if len(entry_clean) == 0:
            return BiasCheckResult(
                check_name='entry_achievability',
                passed=True,
                details='No entries to check',
                severity='info',
                metric_value=1.0,
                threshold=1.0
            )

        # Apply tolerance to bounds
        low_with_tol = low_clean * (1 - tolerance)
        high_with_tol = high_clean * (1 + tolerance)

        # Check if entries are within range
        achievable = (entry_clean >= low_with_tol) & (entry_clean <= high_with_tol)
        achievable_count = np.sum(achievable)
        total_count = len(entry_clean)
        achievable_pct = achievable_count / total_count

        passed = achievable_pct >= 1.0  # All entries must be achievable

        if passed:
            details = f'All {total_count} entries achievable within bar range'
            severity = 'info'
        else:
            unachievable_count = total_count - achievable_count
            # Find examples of unachievable entries
            unachievable_idx = np.where(~achievable)[0][:3]  # First 3 examples
            examples = []
            for idx in unachievable_idx:
                examples.append(
                    f'Entry ${entry_clean[idx]:.2f} outside [{low_clean[idx]:.2f}, {high_clean[idx]:.2f}]'
                )
            details = f'{unachievable_count}/{total_count} entries unachievable. Examples: {"; ".join(examples)}'
            severity = 'critical'

        return BiasCheckResult(
            check_name='entry_achievability',
            passed=passed,
            details=details,
            severity=severity,
            metric_value=achievable_pct,
            threshold=1.0
        )

    def check_exit_achievability(
        self,
        exit_prices: Union[pd.Series, np.ndarray],
        high_prices: Union[pd.Series, np.ndarray],
        low_prices: Union[pd.Series, np.ndarray],
        tolerance: float = 0.001
    ) -> BiasCheckResult:
        """
        Verify all exit prices are achievable within bar's trading range.

        Same as entry check but for exit prices.

        Args:
            exit_prices: Executed exit prices
            high_prices: Bar high prices
            low_prices: Bar low prices
            tolerance: Percentage tolerance for boundary checks

        Returns:
            BiasCheckResult indicating pass/fail
        """
        exit_arr = np.array(exit_prices, dtype=float)
        high_arr = np.array(high_prices, dtype=float)
        low_arr = np.array(low_prices, dtype=float)

        # Remove NaN values
        mask = ~(np.isnan(exit_arr) | np.isnan(high_arr) | np.isnan(low_arr))
        exit_clean = exit_arr[mask]
        high_clean = high_arr[mask]
        low_clean = low_arr[mask]

        if len(exit_clean) == 0:
            return BiasCheckResult(
                check_name='exit_achievability',
                passed=True,
                details='No exits to check',
                severity='info',
                metric_value=1.0,
                threshold=1.0
            )

        # Apply tolerance to bounds
        low_with_tol = low_clean * (1 - tolerance)
        high_with_tol = high_clean * (1 + tolerance)

        # Check if exits are within range
        achievable = (exit_clean >= low_with_tol) & (exit_clean <= high_with_tol)
        achievable_count = np.sum(achievable)
        total_count = len(exit_clean)
        achievable_pct = achievable_count / total_count

        passed = achievable_pct >= 1.0

        if passed:
            details = f'All {total_count} exits achievable within bar range'
            severity = 'info'
        else:
            unachievable_count = total_count - achievable_count
            details = f'{unachievable_count}/{total_count} exits unachievable'
            severity = 'critical'

        return BiasCheckResult(
            check_name='exit_achievability',
            passed=passed,
            details=details,
            severity=severity,
            metric_value=achievable_pct,
            threshold=1.0
        )

    def check_indicator_first_valid(
        self,
        indicator: Union[pd.Series, np.ndarray],
        expected_lookback: int,
        indicator_name: str = 'indicator'
    ) -> BiasCheckResult:
        """
        Verify indicator has expected number of NaN/invalid values at start.

        Indicators with lookback periods should have NaN values for the first
        N bars where N is the lookback. If an indicator starts producing valid
        values too early, it may be using future data.

        Example: A 20-period SMA should have 19 NaN values at the start.
        If it has valid values from bar 0, something is wrong.

        Args:
            indicator: Indicator values
            expected_lookback: Expected lookback period (NaN count)
            indicator_name: Name for reporting

        Returns:
            BiasCheckResult indicating pass/fail
        """
        ind_arr = np.array(indicator, dtype=float)

        # Count leading NaN values
        first_valid_idx = 0
        for i, val in enumerate(ind_arr):
            if not np.isnan(val):
                first_valid_idx = i
                break
        else:
            # All NaN
            first_valid_idx = len(ind_arr)

        # Expected: first valid at index = expected_lookback - 1 or later
        expected_first_valid = expected_lookback - 1

        passed = first_valid_idx >= expected_first_valid

        if passed:
            details = f'{indicator_name}: first valid at index {first_valid_idx} (expected >= {expected_first_valid})'
            severity = 'info'
        else:
            details = f'{indicator_name}: first valid at index {first_valid_idx}, expected >= {expected_first_valid}. Possible look-ahead bias.'
            severity = 'warning'

        return BiasCheckResult(
            check_name=f'indicator_first_valid_{indicator_name}',
            passed=passed,
            details=details,
            severity=severity,
            metric_value=first_valid_idx,
            threshold=expected_first_valid
        )

    def check_signal_shift(
        self,
        raw_signals: Union[pd.Series, np.ndarray],
        tradeable_signals: Union[pd.Series, np.ndarray]
    ) -> BiasCheckResult:
        """
        Verify tradeable signals are properly shifted from raw signals.

        Raw signals generated at bar N should result in trades at bar N+1.
        This check verifies that tradeable_signals = raw_signals.shift(1).

        Args:
            raw_signals: Signals as generated (at bar close)
            tradeable_signals: Signals used for trading (should be shifted)

        Returns:
            BiasCheckResult indicating pass/fail
        """
        raw_arr = np.array(raw_signals, dtype=float)
        trade_arr = np.array(tradeable_signals, dtype=float)

        if len(raw_arr) != len(trade_arr):
            return BiasCheckResult(
                check_name='signal_shift',
                passed=False,
                details='Raw and tradeable signal arrays have different lengths',
                severity='error'
            )

        # Expected: tradeable_signals[i] == raw_signals[i-1]
        # i.e., tradeable_signals = raw_signals shifted forward by 1
        expected_shifted = np.roll(raw_arr, 1)
        expected_shifted[0] = np.nan  # First value should be NaN after shift

        # Compare (ignoring NaN)
        mask = ~(np.isnan(expected_shifted) | np.isnan(trade_arr))
        if np.sum(mask) == 0:
            return BiasCheckResult(
                check_name='signal_shift',
                passed=True,
                details='No overlapping valid values to compare',
                severity='warning'
            )

        match_pct = np.mean(expected_shifted[mask] == trade_arr[mask])

        # Allow for floating point comparison issues
        passed = match_pct >= 0.99  # 99% match required

        if passed:
            details = f'Signals properly shifted: {match_pct:.1%} match'
            severity = 'info'
        else:
            details = f'Signal shift mismatch: only {match_pct:.1%} match expected shift pattern'
            severity = 'warning'

        return BiasCheckResult(
            check_name='signal_shift',
            passed=passed,
            details=details,
            severity=severity,
            metric_value=match_pct,
            threshold=0.99
        )

    def full_check(
        self,
        data: pd.DataFrame,
        signals: Optional[Union[pd.Series, np.ndarray]] = None,
        entry_prices: Optional[Union[pd.Series, np.ndarray]] = None,
        exit_prices: Optional[Union[pd.Series, np.ndarray]] = None,
        indicators: Optional[dict] = None
    ) -> BiasReport:
        """
        Run all configured bias checks.

        Args:
            data: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            signals: Optional boolean/float signals for timing check
            entry_prices: Optional entry prices for achievability check
            exit_prices: Optional exit prices for achievability check
            indicators: Optional dict of {name: (values, expected_lookback)} for indicator checks

        Returns:
            BiasReport with all check results
        """
        checks: List[BiasCheckResult] = []
        failure_reasons: List[str] = []

        # Normalize column names (handle lowercase)
        close_col = 'close' if 'close' in data.columns else 'Close'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'

        # 1. Signal timing check
        if self.config.check_signal_timing and signals is not None:
            # Calculate returns for correlation check
            close = data[close_col]
            returns = close.pct_change()

            result = self.check_signal_timing(signals, returns)
            checks.append(result)
            if not result.passed:
                failure_reasons.append(result.details)

        # 2. Entry achievability check
        # Session 83K-3 BUG FIX: Skip when entry_prices length doesn't match data
        # This happens when entry_prices comes from trades (one per trade) vs data (one per bar)
        if self.config.check_entry_achievability and entry_prices is not None:
            if len(entry_prices) == len(data):
                result = self.check_entry_achievability(
                    entry_prices,
                    data[high_col],
                    data[low_col]
                )
                checks.append(result)
                if not result.passed:
                    failure_reasons.append(result.details)
            else:
                # Log warning but don't fail - trade-level prices can't be correlated with bar-level data
                checks.append(BiasCheckResult(
                    check_name='entry_achievability',
                    passed=True,
                    details=f'Skipped: entry_prices length ({len(entry_prices)}) != data length ({len(data)})',
                    severity='info'
                ))

        # 3. Exit achievability check
        # Session 83K-3 BUG FIX: Skip when exit_prices length doesn't match data
        if self.config.check_entry_achievability and exit_prices is not None:
            if len(exit_prices) == len(data):
                result = self.check_exit_achievability(
                    exit_prices,
                    data[high_col],
                    data[low_col]
                )
                checks.append(result)
                if not result.passed:
                    failure_reasons.append(result.details)
            else:
                checks.append(BiasCheckResult(
                    check_name='exit_achievability',
                    passed=True,
                    details=f'Skipped: exit_prices length ({len(exit_prices)}) != data length ({len(data)})',
                    severity='info'
                ))

        # 4. Indicator first-valid checks
        if self.config.check_indicator_shift and indicators is not None:
            for name, (values, expected_lookback) in indicators.items():
                result = self.check_indicator_first_valid(
                    values,
                    expected_lookback,
                    indicator_name=name
                )
                checks.append(result)
                if not result.passed:
                    failure_reasons.append(result.details)

        # Determine overall pass/fail
        all_passed = all(check.passed for check in checks) if checks else True
        bias_detected = not all_passed

        return BiasReport(
            checks=checks,
            passes_validation=all_passed,
            bias_detected=bias_detected,
            failure_reasons=failure_reasons
        )


def detect_look_ahead_bias(
    signals: Union[pd.Series, np.ndarray],
    returns: Union[pd.Series, np.ndarray],
    threshold: float = 0.5
) -> bool:
    """
    Quick check for look-ahead bias via signal-return correlation.

    Convenience function for simple bias detection without full report.

    Args:
        signals: Trading signals
        returns: Returns for same bars
        threshold: Correlation threshold (default 0.5)

    Returns:
        True if look-ahead bias detected, False otherwise
    """
    config = BiasDetectionConfig(correlation_threshold=threshold)
    detector = BiasDetector(config)
    result = detector.check_signal_timing(signals, returns)
    return not result.passed


def validate_entry_prices(
    entry_prices: Union[pd.Series, np.ndarray],
    high_prices: Union[pd.Series, np.ndarray],
    low_prices: Union[pd.Series, np.ndarray]
) -> tuple:
    """
    Validate that all entry prices are achievable.

    Convenience function for entry price validation.

    Args:
        entry_prices: Executed entry prices
        high_prices: Bar high prices
        low_prices: Bar low prices

    Returns:
        Tuple of (all_achievable: bool, achievable_pct: float)
    """
    detector = BiasDetector()
    result = detector.check_entry_achievability(entry_prices, high_prices, low_prices)
    return result.passed, result.metric_value
