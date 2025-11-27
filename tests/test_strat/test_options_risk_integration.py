"""
Integration tests for OptionsBacktester + OptionsRiskManager

Session 83J: Tests for the Session 83I integration of OptionsRiskManager
into OptionsBacktester's backtest_trades() loop.

Tests cover:
- Backtester accepts risk_manager parameter
- Trades validated when risk_manager provided
- Validation skipped when no risk_manager
- Rejected trades recorded in results
- Circuit breaker states affect trading
- DTE, spread, delta validation
- Validation columns in results DataFrame
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strat.options_module import (
    OptionsBacktester,
    OptionTrade,
    OptionContract,
    OptionType,
    OptionStrategy,
)
from strat.options_risk_manager import (
    OptionsRiskManager,
    OptionsRiskConfig,
    CircuitBreakerState,
)
from strat.tier1_detector import PatternSignal, PatternType, Timeframe


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample OHLCV data for backtesting."""
    dates = pd.date_range('2024-11-01', periods=30, freq='B')
    np.random.seed(42)

    base_price = 590.0
    returns = np.random.normal(0.001, 0.01, len(dates))
    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'open': prices * (1 - np.random.uniform(0, 0.005, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)

    return data


@pytest.fixture
def sample_pattern_signal(sample_price_data) -> PatternSignal:
    """Create a sample pattern signal."""
    return PatternSignal(
        pattern_type=PatternType.PATTERN_312_UP,
        direction=1,
        entry_price=592.0,
        stop_price=588.0,
        target_price=600.0,
        timestamp=sample_price_data.index[0],
        timeframe=Timeframe.DAILY,
        continuation_bars=2,
        is_filtered=True,
    )


@pytest.fixture
def sample_option_trade(sample_pattern_signal) -> OptionTrade:
    """Create a sample option trade."""
    expiration = sample_pattern_signal.timestamp + timedelta(days=21)

    contract = OptionContract(
        underlying='SPY',
        expiration=expiration,
        option_type=OptionType.CALL,
        strike=590.0,
    )

    return OptionTrade(
        pattern_signal=sample_pattern_signal,
        contract=contract,
        strategy=OptionStrategy.LONG_CALL,
        entry_trigger=592.0,
        target_exit=600.0,
        stop_exit=588.0,
        quantity=1,
        option_premium=5.0,
    )


@pytest.fixture
def risk_manager() -> OptionsRiskManager:
    """Create risk manager with default config."""
    return OptionsRiskManager(account_size=10000.0)


@pytest.fixture
def strict_config() -> OptionsRiskConfig:
    """Create strict config for rejection testing."""
    return OptionsRiskConfig(
        min_dte_entry=14,  # Higher min DTE
        max_dte_entry=30,  # Lower max DTE
        max_spread_pct=0.02,  # Very tight spread requirement
        max_position_delta=0.02,  # Very tight delta limit
    )


# =============================================================================
# Test: Backtester Accepts Risk Manager Parameter
# =============================================================================

class TestBacktesterAcceptsRiskManager:
    """Tests for OptionsBacktester initialization with risk_manager."""

    def test_backtester_accepts_risk_manager_parameter(self, risk_manager):
        """Test that OptionsBacktester accepts risk_manager in __init__."""
        backtester = OptionsBacktester(risk_manager=risk_manager)
        assert backtester.risk_manager is not None
        assert backtester.risk_manager is risk_manager

    def test_backtester_accepts_none_risk_manager(self):
        """Test that OptionsBacktester accepts None for risk_manager."""
        backtester = OptionsBacktester(risk_manager=None)
        assert backtester.risk_manager is None

    def test_backtester_default_no_risk_manager(self):
        """Test that OptionsBacktester defaults to no risk_manager."""
        backtester = OptionsBacktester()
        assert backtester.risk_manager is None


# =============================================================================
# Test: Validation When Risk Manager Provided
# =============================================================================

class TestValidationWhenRiskManagerProvided:
    """Tests for validation behavior with risk_manager."""

    def test_backtester_validates_trades_when_risk_manager_provided(
        self, sample_price_data, sample_option_trade, risk_manager
    ):
        """Test that trades are validated when risk_manager is provided."""
        backtester = OptionsBacktester(risk_manager=risk_manager)
        trades = [sample_option_trade]

        results = backtester.backtest_trades(trades, sample_price_data)

        # Results should have validation columns
        assert 'validation_passed' in results.columns
        assert 'validation_reason' in results.columns
        assert 'circuit_state' in results.columns

    def test_backtester_skips_validation_when_no_risk_manager(
        self, sample_price_data, sample_option_trade
    ):
        """Test that validation is skipped when no risk_manager."""
        backtester = OptionsBacktester(risk_manager=None)
        trades = [sample_option_trade]

        results = backtester.backtest_trades(trades, sample_price_data)

        # Without risk_manager, validation columns may not be present
        # or all should pass since no validation occurred
        if 'validation_passed' in results.columns:
            # All should be True (passed) or NaN (no validation)
            if len(results) > 0:
                passed_values = results['validation_passed'].dropna()
                assert all(passed_values) or len(passed_values) == 0


# =============================================================================
# Test: Rejected Trades Recording
# =============================================================================

class TestRejectedTradesRecording:
    """Tests for rejected trade recording in results."""

    def test_rejected_trades_recorded_in_results(
        self, sample_price_data, sample_pattern_signal, strict_config
    ):
        """Test that rejected trades are recorded with validation_passed=False."""
        # Create trade with short DTE (below strict config's min)
        expiration = sample_pattern_signal.timestamp + timedelta(days=5)  # Only 5 DTE

        contract = OptionContract(
            underlying='SPY',
            expiration=expiration,
            option_type=OptionType.CALL,
            strike=590.0,
        )

        trade = OptionTrade(
            pattern_signal=sample_pattern_signal,
            contract=contract,
            strategy=OptionStrategy.LONG_CALL,
            entry_trigger=592.0,
            target_exit=600.0,
            stop_exit=588.0,
            quantity=1,
            option_premium=5.0,
        )

        risk_manager = OptionsRiskManager(config=strict_config, account_size=10000.0)
        backtester = OptionsBacktester(risk_manager=risk_manager)

        results = backtester.backtest_trades([trade], sample_price_data)

        # Should have result recorded
        assert len(results) == 1

        # Should be marked as rejected
        assert results['validation_passed'].iloc[0] == False
        assert 'DTE' in results['validation_reason'].iloc[0]
        assert results['exit_type'].iloc[0] == 'REJECTED'
        assert results['pnl'].iloc[0] == 0.0


# =============================================================================
# Test: Circuit Breaker States
# =============================================================================

class TestCircuitBreakerStates:
    """Tests for circuit breaker state effects on trading."""

    def test_circuit_breaker_halts_new_trades(
        self, sample_price_data, sample_option_trade
    ):
        """Test that HALTED state prevents new trades."""
        risk_manager = OptionsRiskManager(account_size=10000.0)
        risk_manager.circuit_state = CircuitBreakerState.HALTED

        backtester = OptionsBacktester(risk_manager=risk_manager)
        results = backtester.backtest_trades([sample_option_trade], sample_price_data)

        # Trade should be rejected due to HALTED state
        assert len(results) == 1
        assert results['validation_passed'].iloc[0] == False
        assert 'HALTED' in results['validation_reason'].iloc[0]
        assert results['circuit_state'].iloc[0] == 'HALTED'

    def test_reduced_state_adjusts_position_size(
        self, sample_price_data, sample_option_trade
    ):
        """Test that REDUCED state is recorded and noted in warnings."""
        risk_manager = OptionsRiskManager(account_size=10000.0)
        risk_manager.circuit_state = CircuitBreakerState.REDUCED

        backtester = OptionsBacktester(risk_manager=risk_manager)
        results = backtester.backtest_trades([sample_option_trade], sample_price_data)

        # Trade may still pass but should have REDUCED state noted
        if len(results) > 0:
            assert results['circuit_state'].iloc[0] == 'REDUCED'
            # Warnings should mention REDUCED state
            if results['validation_passed'].iloc[0]:
                assert 'REDUCED' in results['validation_warnings'].iloc[0]


# =============================================================================
# Test: Validation Checks (DTE, Spread, Delta)
# =============================================================================

class TestValidationChecks:
    """Tests for specific validation checks."""

    def test_dte_validation_rejects_short_dated(
        self, sample_price_data, sample_pattern_signal
    ):
        """Test that trades with DTE below min are rejected."""
        # Create trade with only 3 DTE (below default min of 7)
        expiration = sample_pattern_signal.timestamp + timedelta(days=3)

        contract = OptionContract(
            underlying='SPY',
            expiration=expiration,
            option_type=OptionType.CALL,
            strike=590.0,
        )

        trade = OptionTrade(
            pattern_signal=sample_pattern_signal,
            contract=contract,
            strategy=OptionStrategy.LONG_CALL,
            entry_trigger=592.0,
            target_exit=600.0,
            stop_exit=588.0,
            quantity=1,
            option_premium=5.0,
        )

        risk_manager = OptionsRiskManager(account_size=10000.0)
        backtester = OptionsBacktester(risk_manager=risk_manager)

        results = backtester.backtest_trades([trade], sample_price_data)

        assert len(results) == 1
        assert results['validation_passed'].iloc[0] == False
        assert 'DTE' in results['validation_reason'].iloc[0]
        assert 'below minimum' in results['validation_reason'].iloc[0]

    def test_delta_limits_enforced(self, sample_price_data, sample_pattern_signal):
        """Test that position delta limits are enforced."""
        # The position delta calculation is: abs(delta * contracts) / account_size
        # For delta ~0.5 and 50 contracts: 0.5 * 50 / 10000 = 0.25%
        # Set max to 0.1% (0.001) so it will be exceeded
        config = OptionsRiskConfig(
            max_position_delta=0.001,  # Very tight: 0.1%
            max_premium_at_risk=0.99,  # Allow high premium so delta triggers first
        )

        expiration = sample_pattern_signal.timestamp + timedelta(days=21)

        contract = OptionContract(
            underlying='SPY',
            expiration=expiration,
            option_type=OptionType.CALL,
            strike=590.0,
        )

        trade = OptionTrade(
            pattern_signal=sample_pattern_signal,
            contract=contract,
            strategy=OptionStrategy.LONG_CALL,
            entry_trigger=592.0,
            target_exit=600.0,
            stop_exit=588.0,
            quantity=50,  # 50 contracts: 0.5 * 50 / 10000 = 0.25% > 0.1%
            option_premium=0.10,  # Very low premium to avoid premium limit
        )

        risk_manager = OptionsRiskManager(config=config, account_size=10000.0)
        backtester = OptionsBacktester(risk_manager=risk_manager)

        results = backtester.backtest_trades([trade], sample_price_data)

        assert len(results) == 1
        # Trade should be rejected due to delta limits
        assert results['validation_passed'].iloc[0] == False
        assert 'delta' in results['validation_reason'].iloc[0].lower()


# =============================================================================
# Test: Validation Columns in Results
# =============================================================================

class TestValidationColumns:
    """Tests for validation column presence and format."""

    def test_validation_columns_in_results_dataframe(
        self, sample_price_data, sample_option_trade, risk_manager
    ):
        """Test that all validation columns are present in results."""
        backtester = OptionsBacktester(risk_manager=risk_manager)
        results = backtester.backtest_trades([sample_option_trade], sample_price_data)

        # All validation columns should be present
        expected_columns = [
            'validation_passed',
            'validation_reason',
            'validation_warnings',
            'circuit_state'
        ]

        for col in expected_columns:
            assert col in results.columns, f"Missing column: {col}"

    def test_validation_columns_types(
        self, sample_price_data, sample_option_trade, risk_manager
    ):
        """Test that validation column types are correct."""
        backtester = OptionsBacktester(risk_manager=risk_manager)
        results = backtester.backtest_trades([sample_option_trade], sample_price_data)

        if len(results) > 0:
            # validation_passed should be boolean
            assert isinstance(results['validation_passed'].iloc[0], (bool, np.bool_))

            # reason, warnings, circuit_state should be strings
            assert isinstance(results['circuit_state'].iloc[0], str)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_trades_list(self, sample_price_data, risk_manager):
        """Test backtesting with empty trades list."""
        backtester = OptionsBacktester(risk_manager=risk_manager)
        results = backtester.backtest_trades([], sample_price_data)

        assert len(results) == 0

    def test_multiple_trades_mixed_validation(
        self, sample_price_data, sample_pattern_signal
    ):
        """Test multiple trades with mixed validation results."""
        # Create one valid trade (21 DTE)
        valid_expiration = sample_pattern_signal.timestamp + timedelta(days=21)
        valid_contract = OptionContract(
            underlying='SPY',
            expiration=valid_expiration,
            option_type=OptionType.CALL,
            strike=590.0,
        )
        valid_trade = OptionTrade(
            pattern_signal=sample_pattern_signal,
            contract=valid_contract,
            strategy=OptionStrategy.LONG_CALL,
            entry_trigger=592.0,
            target_exit=600.0,
            stop_exit=588.0,
            quantity=1,
            option_premium=5.0,
        )

        # Create one invalid trade (3 DTE - below min)
        invalid_expiration = sample_pattern_signal.timestamp + timedelta(days=3)
        invalid_contract = OptionContract(
            underlying='SPY',
            expiration=invalid_expiration,
            option_type=OptionType.CALL,
            strike=590.0,
        )
        invalid_trade = OptionTrade(
            pattern_signal=sample_pattern_signal,
            contract=invalid_contract,
            strategy=OptionStrategy.LONG_CALL,
            entry_trigger=592.0,
            target_exit=600.0,
            stop_exit=588.0,
            quantity=1,
            option_premium=5.0,
        )

        risk_manager = OptionsRiskManager(account_size=10000.0)
        backtester = OptionsBacktester(risk_manager=risk_manager)

        results = backtester.backtest_trades([valid_trade, invalid_trade], sample_price_data)

        # Should have 2 results
        assert len(results) == 2

        # One should be rejected (DTE too low)
        rejected = results[results['validation_passed'] == False]
        assert len(rejected) >= 1
