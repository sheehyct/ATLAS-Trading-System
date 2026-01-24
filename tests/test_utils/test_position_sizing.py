"""
Tests for utils/position_sizing.py - ATR-based position sizing.

EQUITY-83: Phase 3 test coverage for position sizing module.

Tests cover:
- calculate_position_size_atr with scalar inputs
- calculate_position_size_atr with pandas Series (VBT Pro compatibility)
- Capital constraint enforcement
- Edge cases (zero ATR, NaN handling)
- validate_position_size function
"""

import numpy as np
import pandas as pd
import pytest

from utils.position_sizing import (
    calculate_position_size_atr,
    validate_position_size,
)


# =============================================================================
# calculate_position_size_atr Scalar Tests
# =============================================================================

class TestPositionSizeATRScalar:
    """Tests for calculate_position_size_atr with scalar inputs."""

    def test_basic_position_size(self):
        """Test basic position sizing calculation."""
        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=100,
            atr=4.0,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Stop distance = 4.0 * 2.5 = 10
        # Risk amount = 10000 * 0.02 = 200
        # Position size = 200 / 10 = 20 shares
        assert size == 20
        assert risk == pytest.approx(0.02)
        assert constrained == False  # noqa: E712 (numpy bool)

    def test_high_atr_reduces_size(self):
        """Test that higher ATR reduces position size."""
        size_low_atr, _, _ = calculate_position_size_atr(
            init_cash=10000, close=100, atr=2.0, atr_multiplier=2.5, risk_pct=0.02
        )

        size_high_atr, _, _ = calculate_position_size_atr(
            init_cash=10000, close=100, atr=8.0, atr_multiplier=2.5, risk_pct=0.02
        )

        assert size_high_atr < size_low_atr

    def test_higher_risk_pct_increases_size(self):
        """Test that higher risk percentage increases position size."""
        size_low_risk, _, _ = calculate_position_size_atr(
            init_cash=10000, close=100, atr=4.0, atr_multiplier=2.5, risk_pct=0.01
        )

        size_high_risk, _, _ = calculate_position_size_atr(
            init_cash=10000, close=100, atr=4.0, atr_multiplier=2.5, risk_pct=0.02
        )

        assert size_high_risk > size_low_risk

    def test_capital_constraint_applied(self):
        """Test that capital constraint limits position size."""
        # Very low ATR would suggest large position, but capital constrains it
        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=480,
            atr=1.0,  # Very small ATR
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Max affordable = 10000 / 480 = 20.83 shares
        # Risk-based = (10000 * 0.02) / 2.5 = 80 shares
        # Capital constraint should apply
        assert size == pytest.approx(10000 / 480, rel=0.01)
        assert constrained == True  # noqa: E712 (numpy bool)
        assert risk < 0.02  # Actual risk less than target

    def test_zero_atr_handled(self):
        """Test zero ATR is handled gracefully."""
        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=100,
            atr=0.0,  # Zero ATR
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Should use fallback ATR value
        assert size > 0
        assert np.isfinite(size)

    def test_negative_atr_handled(self):
        """Test negative ATR is handled gracefully."""
        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=100,
            atr=-5.0,  # Negative ATR
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Should use fallback ATR value
        assert size >= 0
        assert np.isfinite(size)


# =============================================================================
# calculate_position_size_atr Series Tests (VBT Pro Compatibility)
# =============================================================================

class TestPositionSizeATRSeries:
    """Tests for calculate_position_size_atr with pandas Series (VBT Pro)."""

    def test_series_input_output(self):
        """Test that Series input returns Series output."""
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        close = pd.Series([100, 101, 102, 103, 104], index=dates)
        atr = pd.Series([4.0, 4.1, 4.2, 4.3, 4.4], index=dates)

        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=close,
            atr=atr,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        assert isinstance(size, pd.Series)
        assert isinstance(risk, pd.Series)
        assert isinstance(constrained, pd.Series)
        assert len(size) == 5
        assert size.index.equals(close.index)

    def test_series_values_correct(self):
        """Test Series calculations are correct."""
        dates = pd.date_range('2025-01-01', periods=3, freq='D')
        close = pd.Series([100, 100, 100], index=dates)
        atr = pd.Series([4.0, 8.0, 2.0], index=dates)

        size, _, _ = calculate_position_size_atr(
            init_cash=10000,
            close=close,
            atr=atr,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Higher ATR should give smaller position
        assert size.iloc[1] < size.iloc[0]
        # Lower ATR should give larger position
        assert size.iloc[2] > size.iloc[0]

    def test_series_with_nan_atr(self):
        """Test Series handles NaN ATR values."""
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        close = pd.Series([100, 100, 100, 100, 100], index=dates)
        atr = pd.Series([4.0, np.nan, 4.0, np.nan, 4.0], index=dates)

        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=close,
            atr=atr,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Should not have NaN in output
        assert not size.isna().any()
        assert not risk.isna().any()

    def test_series_with_zero_atr(self):
        """Test Series handles zero ATR values."""
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        close = pd.Series([100, 100, 100, 100, 100], index=dates)
        atr = pd.Series([4.0, 0.0, 4.0, 0.0, 4.0], index=dates)

        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=close,
            atr=atr,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Should fill zeros with valid values
        assert not (size == 0).all()  # Not all zeros
        assert not np.isinf(size).any()

    def test_series_capital_constraint(self):
        """Test capital constraint applies element-wise."""
        dates = pd.date_range('2025-01-01', periods=3, freq='D')
        close = pd.Series([100, 500, 100], index=dates)  # Middle price very high
        atr = pd.Series([1.0, 1.0, 1.0], index=dates)  # Low ATR suggests large position

        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=close,
            atr=atr,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # All should be constrained with such low ATR
        assert constrained.any()

    def test_series_no_inf_values(self):
        """Test Series output has no infinite values."""
        dates = pd.date_range('2025-01-01', periods=5, freq='D')
        close = pd.Series([100, 0.01, 100, 1000, 100], index=dates)
        atr = pd.Series([4.0, 0.001, 4.0, 4.0, 4.0], index=dates)

        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=close,
            atr=atr,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        assert not np.isinf(size).any()
        assert not np.isinf(risk).any()


# =============================================================================
# validate_position_size Tests
# =============================================================================

class TestValidatePositionSize:
    """Tests for validate_position_size function."""

    def test_valid_scalar(self):
        """Test validation passes for valid scalar."""
        is_valid, msg = validate_position_size(
            position_size=20,
            init_cash=10000,
            close=100,
            max_pct=1.0
        )

        assert is_valid is True
        assert msg == "Valid"

    def test_valid_series(self):
        """Test validation passes for valid Series."""
        is_valid, msg = validate_position_size(
            position_size=pd.Series([10, 20, 15]),
            init_cash=10000,
            close=pd.Series([100, 100, 100]),
            max_pct=1.0
        )

        assert is_valid is True
        assert msg == "Valid"

    def test_negative_size_scalar(self):
        """Test negative position size fails validation."""
        is_valid, msg = validate_position_size(
            position_size=-5,
            init_cash=10000,
            close=100,
            max_pct=1.0
        )

        assert is_valid is False
        assert "Negative" in msg

    def test_negative_size_series(self):
        """Test negative position size in Series fails."""
        is_valid, msg = validate_position_size(
            position_size=pd.Series([10, -5, 15]),
            init_cash=10000,
            close=pd.Series([100, 100, 100]),
            max_pct=1.0
        )

        assert is_valid is False
        assert "Negative" in msg

    def test_nan_size_scalar(self):
        """Test NaN position size fails validation."""
        is_valid, msg = validate_position_size(
            position_size=np.nan,
            init_cash=10000,
            close=100,
            max_pct=1.0
        )

        assert is_valid is False
        assert "NaN" in msg or "Inf" in msg

    def test_inf_size_scalar(self):
        """Test infinite position size fails validation."""
        is_valid, msg = validate_position_size(
            position_size=np.inf,
            init_cash=10000,
            close=100,
            max_pct=1.0
        )

        assert is_valid is False
        assert "NaN" in msg or "Inf" in msg

    def test_nan_size_series(self):
        """Test NaN in Series fails validation."""
        is_valid, msg = validate_position_size(
            position_size=pd.Series([10, np.nan, 15]),
            init_cash=10000,
            close=pd.Series([100, 100, 100]),
            max_pct=1.0
        )

        assert is_valid is False

    def test_exceeds_capital_scalar(self):
        """Test position exceeding capital fails."""
        is_valid, msg = validate_position_size(
            position_size=200,  # 200 * 100 = $20,000 > $10,000
            init_cash=10000,
            close=100,
            max_pct=1.0
        )

        assert is_valid is False
        assert "exceeds" in msg.lower()

    def test_exceeds_capital_series(self):
        """Test position exceeding capital in Series fails."""
        is_valid, msg = validate_position_size(
            position_size=pd.Series([10, 200, 15]),  # Middle exceeds
            init_cash=10000,
            close=pd.Series([100, 100, 100]),
            max_pct=1.0
        )

        assert is_valid is False
        assert "exceeds" in msg.lower()

    def test_custom_max_pct(self):
        """Test custom max_pct is respected."""
        # 50 shares at $100 = $5,000 = 50% of capital
        is_valid, msg = validate_position_size(
            position_size=50,
            init_cash=10000,
            close=100,
            max_pct=0.25  # Only allow 25%
        )

        assert is_valid is False
        assert "exceeds" in msg.lower()


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_very_small_capital(self):
        """Test with very small capital amount."""
        size, risk, constrained = calculate_position_size_atr(
            init_cash=100,
            close=50,
            atr=2.0,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        assert size >= 0
        assert np.isfinite(size)

    def test_very_large_capital(self):
        """Test with very large capital amount."""
        size, risk, constrained = calculate_position_size_atr(
            init_cash=10_000_000,
            close=100,
            atr=4.0,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        assert size > 0
        assert np.isfinite(size)

    def test_penny_stock(self):
        """Test with very low price stock."""
        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=0.50,
            atr=0.05,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        assert size >= 0
        assert np.isfinite(size)

    def test_expensive_stock(self):
        """Test with very high price stock."""
        size, risk, constrained = calculate_position_size_atr(
            init_cash=10000,
            close=5000,  # Like BRK.A
            atr=100,
            atr_multiplier=2.5,
            risk_pct=0.02
        )

        # Can afford at most 2 shares
        assert size <= 2
        assert np.isfinite(size)

    def test_different_atr_multipliers(self):
        """Test various ATR multiplier values."""
        for multiplier in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]:
            size, risk, constrained = calculate_position_size_atr(
                init_cash=10000,
                close=100,
                atr=4.0,
                atr_multiplier=multiplier,
                risk_pct=0.02
            )

            assert size >= 0
            assert np.isfinite(size)

    def test_actual_risk_bounded(self):
        """Test actual risk is always between 0 and 100%."""
        for _ in range(10):
            close = np.random.uniform(1, 1000)
            atr = np.random.uniform(0.1, 50)

            size, risk, constrained = calculate_position_size_atr(
                init_cash=10000,
                close=close,
                atr=atr,
                atr_multiplier=2.5,
                risk_pct=0.02
            )

            assert 0 <= risk <= 1.0
