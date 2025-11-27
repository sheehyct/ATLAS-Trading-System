"""
Tests for Options Risk Manager

Session 83H: Options risk manager implementation per ATLAS Checklist Section 9.3.

Tests cover:
- OptionsRiskConfig defaults and validation
- CircuitBreakerState enum
- ValidationResult dataclass
- PositionRisk dataclass
- OptionsRiskManager initialization
- Pre-trade validation (validate_new_position)
- Circuit breaker state machine
- Portfolio Greeks aggregation
- Force exit signals
- Edge cases and boundary conditions
"""

import pytest
from datetime import datetime, timedelta

from strat.options_risk_manager import (
    OptionsRiskManager,
    OptionsRiskConfig,
    CircuitBreakerState,
    ValidationResult,
    PortfolioGreeks,
    PositionRisk,
    ForceExitSignal,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> OptionsRiskConfig:
    """Create default configuration."""
    return OptionsRiskConfig()


@pytest.fixture
def risk_manager() -> OptionsRiskManager:
    """Create risk manager with $10,000 account."""
    return OptionsRiskManager(account_size=10000.0)


@pytest.fixture
def risk_manager_small_account() -> OptionsRiskManager:
    """Create risk manager with $3,000 account (typical small account)."""
    return OptionsRiskManager(account_size=3000.0)


@pytest.fixture
def sample_position() -> PositionRisk:
    """Create a sample options position."""
    return PositionRisk(
        symbol='SPY',
        option_symbol='SPY241220C00590000',
        delta=0.55,
        gamma=0.02,
        theta=-0.15,
        vega=0.08,
        premium=250.0,
        current_value=275.0,
        dte=21,
        entry_date=datetime(2024, 11, 29),
        contracts=1
    )


# =============================================================================
# Test: OptionsRiskConfig
# =============================================================================

class TestOptionsRiskConfig:
    """Tests for OptionsRiskConfig dataclass."""

    def test_default_values(self, default_config):
        """Test default configuration values match ATLAS specs."""
        assert default_config.max_portfolio_delta == 0.30
        assert default_config.max_portfolio_gamma == 0.05
        assert default_config.max_portfolio_theta == -0.02
        assert default_config.max_portfolio_vega == 0.10
        assert default_config.max_position_delta == 0.10
        assert default_config.min_dte_entry == 7
        assert default_config.max_dte_entry == 45
        assert default_config.forced_exit_dte == 3
        assert default_config.max_spread_pct == 0.10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptionsRiskConfig(
            max_portfolio_delta=0.50,
            min_dte_entry=14,
            max_dte_entry=30
        )
        assert config.max_portfolio_delta == 0.50
        assert config.min_dte_entry == 14
        assert config.max_dte_entry == 30
        # Other values should be defaults
        assert config.max_portfolio_gamma == 0.05

    def test_to_dict(self, default_config):
        """Test configuration serialization."""
        config_dict = default_config.to_dict()
        assert 'max_portfolio_delta' in config_dict
        assert 'min_dte_entry' in config_dict
        assert config_dict['max_portfolio_delta'] == 0.30
        assert config_dict['min_dte_entry'] == 7


# =============================================================================
# Test: CircuitBreakerState
# =============================================================================

class TestCircuitBreakerState:
    """Tests for CircuitBreakerState enum."""

    def test_all_states_exist(self):
        """Test all expected states are defined."""
        assert CircuitBreakerState.NORMAL.value == "NORMAL"
        assert CircuitBreakerState.CAUTION.value == "CAUTION"
        assert CircuitBreakerState.REDUCED.value == "REDUCED"
        assert CircuitBreakerState.HALTED.value == "HALTED"
        assert CircuitBreakerState.EMERGENCY.value == "EMERGENCY"

    def test_state_count(self):
        """Test we have exactly 5 states."""
        states = list(CircuitBreakerState)
        assert len(states) == 5


# =============================================================================
# Test: PositionRisk
# =============================================================================

class TestPositionRisk:
    """Tests for PositionRisk dataclass."""

    def test_pnl_calculation(self, sample_position):
        """Test P/L percentage calculation."""
        # premium=250, current_value=275 -> 10% gain
        assert sample_position.pnl_pct == pytest.approx(0.10, abs=0.01)

    def test_pnl_loss(self):
        """Test P/L calculation for losing position."""
        position = PositionRisk(
            symbol='SPY',
            option_symbol='SPY241220P00580000',
            delta=-0.45,
            gamma=0.02,
            theta=-0.12,
            vega=0.06,
            premium=200.0,
            current_value=100.0,  # 50% loss
            dte=14,
            entry_date=datetime(2024, 12, 1),
            contracts=1
        )
        assert position.pnl_pct == pytest.approx(-0.50, abs=0.01)

    def test_zero_premium_edge_case(self):
        """Test handling of zero premium."""
        position = PositionRisk(
            symbol='SPY',
            option_symbol='SPY241220C00600000',
            delta=0.25,
            gamma=0.01,
            theta=-0.05,
            vega=0.03,
            premium=0.0,  # Edge case
            current_value=50.0,
            dte=7,
            entry_date=datetime(2024, 12, 10),
            contracts=1
        )
        # Should not raise, pnl_pct will be 0 or handled
        assert position.pnl_pct == 0.0


# =============================================================================
# Test: OptionsRiskManager Initialization
# =============================================================================

class TestOptionsRiskManagerInit:
    """Tests for OptionsRiskManager initialization."""

    def test_default_initialization(self, risk_manager):
        """Test default initialization."""
        assert risk_manager.account_size == 10000.0
        assert risk_manager.circuit_state == CircuitBreakerState.NORMAL
        assert len(risk_manager.positions) == 0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = OptionsRiskConfig(max_portfolio_delta=0.50)
        rm = OptionsRiskManager(config=config, account_size=5000.0)
        assert rm.config.max_portfolio_delta == 0.50
        assert rm.account_size == 5000.0

    def test_invalid_account_size_zero(self):
        """Test rejection of zero account size."""
        with pytest.raises(ValueError, match="must be positive"):
            OptionsRiskManager(account_size=0.0)

    def test_invalid_account_size_negative(self):
        """Test rejection of negative account size."""
        with pytest.raises(ValueError, match="must be positive"):
            OptionsRiskManager(account_size=-1000.0)

    def test_initial_state_logged(self, risk_manager):
        """Test initial state is logged."""
        assert len(risk_manager.state_history) == 1
        assert risk_manager.state_history[0][1] == CircuitBreakerState.NORMAL


# =============================================================================
# Test: Pre-Trade Validation
# =============================================================================

class TestValidateNewPosition:
    """Tests for validate_new_position method."""

    def test_valid_position_passes(self, risk_manager):
        """Test valid position passes validation."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,  # 2.5% of account
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is True
        assert len(result.reasons) == 0
        assert result.circuit_state == CircuitBreakerState.NORMAL

    def test_dte_below_minimum_fails(self, risk_manager):
        """Test DTE below minimum fails."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=5,  # Below min of 7
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is False
        assert any("DTE 5 below minimum 7" in r for r in result.reasons)

    def test_dte_above_maximum_fails(self, risk_manager):
        """Test DTE above maximum fails."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=60,  # Above max of 45
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is False
        assert any("DTE 60 above maximum 45" in r for r in result.reasons)

    def test_spread_too_wide_fails(self, risk_manager):
        """Test wide spread fails validation."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=21,
            spread_pct=0.15,  # Above max of 10%
            contracts=1
        )
        assert result.passed is False
        assert any("Spread" in r and "exceeds" in r for r in result.reasons)

    def test_premium_too_large_fails(self, risk_manager):
        """Test premium exceeding limit fails."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=1000.0,  # 10% of $10k account, exceeds 5% limit
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is False
        assert any("Premium" in r and "exceeds" in r for r in result.reasons)

    def test_halted_state_blocks_all(self, risk_manager):
        """Test HALTED state blocks all new positions."""
        risk_manager.circuit_state = CircuitBreakerState.HALTED

        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=100.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is False
        assert "HALTED" in result.reasons[0]

    def test_reduced_state_suggests_smaller_size(self, risk_manager):
        """Test REDUCED state suggests reduced position size."""
        risk_manager.circuit_state = CircuitBreakerState.REDUCED

        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        # Should pass but with adjusted size
        assert result.adjusted_size == 0.5
        assert any("REDUCED" in w for w in result.warnings)

    def test_multiple_validation_failures(self, risk_manager):
        """Test multiple validation failures are reported."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=1000.0,  # Too large
            dte=5,  # Too low
            spread_pct=0.15,  # Too wide
            contracts=1
        )
        assert result.passed is False
        assert len(result.reasons) >= 3

    def test_portfolio_delta_limit(self, risk_manager):
        """Test portfolio delta limit enforcement."""
        # Add existing position with large delta
        position = PositionRisk(
            symbol='SPY',
            option_symbol='SPY241220C00590000',
            delta=2500.0,  # Large delta
            gamma=50.0,
            theta=-10.0,
            vega=20.0,
            premium=250.0,
            current_value=250.0,
            dte=21,
            entry_date=datetime.now(),
            contracts=1
        )
        risk_manager.add_position(position)

        # Try to add more - should fail due to portfolio delta limit
        result = risk_manager.validate_new_position(
            delta=1000.0,  # Adding more delta
            gamma=10.0,
            theta=-2.0,
            vega=5.0,
            premium=250.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is False
        assert any("Portfolio delta" in r for r in result.reasons)


# =============================================================================
# Test: Position Management
# =============================================================================

class TestPositionManagement:
    """Tests for position management methods."""

    def test_add_position(self, risk_manager, sample_position):
        """Test adding a position."""
        risk_manager.add_position(sample_position)
        assert len(risk_manager.positions) == 1
        assert sample_position.option_symbol in risk_manager.positions

    def test_remove_position(self, risk_manager, sample_position):
        """Test removing a position."""
        risk_manager.add_position(sample_position)
        risk_manager.remove_position(sample_position.option_symbol)
        assert len(risk_manager.positions) == 0

    def test_update_position(self, risk_manager, sample_position):
        """Test updating position values."""
        risk_manager.add_position(sample_position)
        risk_manager.update_position(
            sample_position.option_symbol,
            current_value=300.0,
            dte=18
        )
        updated = risk_manager.positions[sample_position.option_symbol]
        assert updated.current_value == 300.0
        assert updated.dte == 18


# =============================================================================
# Test: Portfolio Greeks Aggregation
# =============================================================================

class TestPortfolioGreeks:
    """Tests for portfolio Greeks aggregation."""

    def test_empty_portfolio(self, risk_manager):
        """Test aggregation with no positions."""
        greeks = risk_manager.aggregate_portfolio_greeks()
        assert greeks.net_delta == 0.0
        assert greeks.total_gamma == 0.0
        assert greeks.total_theta == 0.0
        assert greeks.total_vega == 0.0
        assert greeks.position_count == 0

    def test_single_position(self, risk_manager, sample_position):
        """Test aggregation with single position."""
        risk_manager.add_position(sample_position)
        greeks = risk_manager.aggregate_portfolio_greeks()
        assert greeks.net_delta == 0.55
        assert greeks.total_theta == -0.15
        assert greeks.position_count == 1

    def test_multiple_positions(self, risk_manager):
        """Test aggregation with multiple positions."""
        pos1 = PositionRisk(
            symbol='SPY',
            option_symbol='SPY241220C00590000',
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            current_value=250.0,
            dte=21,
            entry_date=datetime.now(),
            contracts=2
        )
        pos2 = PositionRisk(
            symbol='QQQ',
            option_symbol='QQQ241220P00400000',
            delta=-0.45,  # Put has negative delta
            gamma=0.015,
            theta=-0.10,
            vega=0.05,
            premium=150.0,
            current_value=150.0,
            dte=21,
            entry_date=datetime.now(),
            contracts=1
        )
        risk_manager.add_position(pos1)
        risk_manager.add_position(pos2)

        greeks = risk_manager.aggregate_portfolio_greeks()
        # 0.55 * 2 + (-0.45) * 1 = 1.10 - 0.45 = 0.65
        assert greeks.net_delta == pytest.approx(0.65, abs=0.01)
        assert greeks.position_count == 2
        assert greeks.total_premium_at_risk == 400.0


# =============================================================================
# Test: Circuit Breakers
# =============================================================================

class TestCircuitBreakers:
    """Tests for circuit breaker functionality."""

    def test_vix_threshold_triggers_halt(self, risk_manager):
        """Test VIX above threshold triggers HALTED state."""
        force_exits = risk_manager.check_circuit_breakers(current_vix=30.0)
        assert risk_manager.circuit_state == CircuitBreakerState.HALTED

    def test_vix_spike_triggers_halt(self, risk_manager):
        """Test VIX spike triggers HALTED state."""
        # Set initial VIX
        risk_manager.check_circuit_breakers(current_vix=15.0)
        # Large spike
        force_exits = risk_manager.check_circuit_breakers(current_vix=20.0)  # 33% spike
        assert risk_manager.circuit_state == CircuitBreakerState.HALTED

    def test_normal_vix_stays_normal(self, risk_manager):
        """Test normal VIX keeps NORMAL state."""
        force_exits = risk_manager.check_circuit_breakers(current_vix=15.0)
        assert risk_manager.circuit_state == CircuitBreakerState.NORMAL

    def test_force_exit_dte(self, risk_manager):
        """Test DTE-based forced exit."""
        position = PositionRisk(
            symbol='SPY',
            option_symbol='SPY241220C00590000',
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            current_value=250.0,
            dte=2,  # Below forced_exit_dte of 3
            entry_date=datetime.now(),
            contracts=1
        )
        risk_manager.add_position(position)
        force_exits = risk_manager.check_circuit_breakers()

        assert len(force_exits) == 1
        assert "DTE" in force_exits[0].reason

    def test_force_exit_loss(self, risk_manager):
        """Test loss-based forced exit."""
        position = PositionRisk(
            symbol='SPY',
            option_symbol='SPY241220C00590000',
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            current_value=100.0,  # 60% loss
            dte=21,
            entry_date=datetime.now(),
            contracts=1
        )
        risk_manager.add_position(position)
        force_exits = risk_manager.check_circuit_breakers()

        assert len(force_exits) == 1
        assert "loss" in force_exits[0].reason.lower()

    def test_state_progression(self, risk_manager):
        """Test circuit breaker state progression with Greeks."""
        # Add positions that approach limits
        # This tests the state machine logic
        risk_manager.circuit_state = CircuitBreakerState.NORMAL

        # Manually trigger state change
        risk_manager._log_state_change(CircuitBreakerState.CAUTION, "Test")
        risk_manager.circuit_state = CircuitBreakerState.CAUTION

        assert risk_manager.circuit_state == CircuitBreakerState.CAUTION
        assert len(risk_manager.state_history) >= 2


# =============================================================================
# Test: Utility Methods
# =============================================================================

class TestUtilityMethods:
    """Tests for utility methods."""

    def test_can_trade_normal(self, risk_manager):
        """Test can_trade returns True in NORMAL state."""
        can_trade, reason = risk_manager.can_trade()
        assert can_trade is True
        assert reason == "OK"

    def test_can_trade_halted(self, risk_manager):
        """Test can_trade returns False in HALTED state."""
        risk_manager.circuit_state = CircuitBreakerState.HALTED
        can_trade, reason = risk_manager.can_trade()
        assert can_trade is False
        assert "HALTED" in reason

    def test_position_size_multiplier(self, risk_manager):
        """Test position size multiplier varies by state."""
        assert risk_manager.get_position_size_multiplier() == 1.0

        risk_manager.circuit_state = CircuitBreakerState.REDUCED
        assert risk_manager.get_position_size_multiplier() == 0.5

        risk_manager.circuit_state = CircuitBreakerState.HALTED
        assert risk_manager.get_position_size_multiplier() == 0.0

    def test_reset_to_normal(self, risk_manager):
        """Test manual reset to NORMAL state."""
        risk_manager.circuit_state = CircuitBreakerState.HALTED
        risk_manager.reset_to_normal("Market stabilized")

        assert risk_manager.circuit_state == CircuitBreakerState.NORMAL
        assert any("Market stabilized" in h[2] for h in risk_manager.state_history)

    def test_update_account_size(self, risk_manager):
        """Test updating account size."""
        risk_manager.update_account_size(15000.0)
        assert risk_manager.account_size == 15000.0

    def test_update_account_size_invalid(self, risk_manager):
        """Test invalid account size update."""
        with pytest.raises(ValueError):
            risk_manager.update_account_size(-1000.0)

    def test_portfolio_risk_summary(self, risk_manager, sample_position):
        """Test portfolio risk summary generation."""
        risk_manager.add_position(sample_position)
        summary = risk_manager.get_portfolio_risk_summary()

        assert 'circuit_state' in summary
        assert 'account_size' in summary
        assert 'greeks' in summary
        assert 'utilization' in summary
        assert summary['position_count'] == 1

    def test_print_summary(self, risk_manager, sample_position):
        """Test human-readable summary generation."""
        risk_manager.add_position(sample_position)
        summary = risk_manager.print_summary()

        assert "OPTIONS RISK MANAGER SUMMARY" in summary
        assert "Circuit State" in summary
        assert "Net Delta" in summary


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_dte_exactly_at_minimum(self, risk_manager):
        """Test DTE exactly at minimum boundary."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=7,  # Exactly at minimum
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is True

    def test_dte_exactly_at_maximum(self, risk_manager):
        """Test DTE exactly at maximum boundary."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=45,  # Exactly at maximum
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is True

    def test_spread_exactly_at_limit(self, risk_manager):
        """Test spread exactly at limit boundary."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=21,
            spread_pct=0.10,  # Exactly at limit
            contracts=1
        )
        assert result.passed is True

    def test_premium_exactly_at_limit(self, risk_manager):
        """Test premium exactly at 5% limit."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=500.0,  # Exactly 5% of $10k
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is True

    def test_small_account_constraints(self, risk_manager_small_account):
        """Test constraints are tighter for small accounts."""
        # $150 is 5% of $3k - should pass
        result = risk_manager_small_account.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=150.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is True

        # $200 is 6.67% of $3k - should fail
        result = risk_manager_small_account.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=200.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is False

    def test_emergency_state_forces_all_exits(self, risk_manager, sample_position):
        """Test EMERGENCY state forces exit of all positions."""
        risk_manager.add_position(sample_position)
        risk_manager.circuit_state = CircuitBreakerState.EMERGENCY

        force_exits = risk_manager.check_circuit_breakers()

        assert len(force_exits) >= 1
        assert any("EMERGENCY" in fe.reason for fe in force_exits)

    def test_none_vix_allowed(self, risk_manager):
        """Test circuit breaker check with None VIX."""
        force_exits = risk_manager.check_circuit_breakers(current_vix=None)
        assert risk_manager.circuit_state == CircuitBreakerState.NORMAL


# =============================================================================
# Test: Serialization
# =============================================================================

class TestSerialization:
    """Tests for serialization methods."""

    def test_validation_result_to_dict(self, risk_manager):
        """Test ValidationResult serialization."""
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        result_dict = result.to_dict()

        assert 'passed' in result_dict
        assert 'reasons' in result_dict
        assert 'circuit_state' in result_dict
        assert result_dict['passed'] is True

    def test_portfolio_greeks_to_dict(self, risk_manager, sample_position):
        """Test PortfolioGreeks serialization."""
        risk_manager.add_position(sample_position)
        greeks = risk_manager.aggregate_portfolio_greeks()
        greeks_dict = greeks.to_dict()

        assert 'net_delta' in greeks_dict
        assert 'total_theta' in greeks_dict
        assert 'position_count' in greeks_dict

    def test_config_to_dict(self, default_config):
        """Test OptionsRiskConfig serialization."""
        config_dict = default_config.to_dict()

        assert 'max_portfolio_delta' in config_dict
        assert 'min_dte_entry' in config_dict
        assert 'forced_exit_dte' in config_dict


# =============================================================================
# Test: Integration Scenarios
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests simulating real trading scenarios."""

    def test_full_trading_workflow(self, risk_manager):
        """Test complete trading workflow."""
        # 1. Check can trade
        can_trade, _ = risk_manager.can_trade()
        assert can_trade

        # 2. Validate new position
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed

        # 3. Add position
        position = PositionRisk(
            symbol='SPY',
            option_symbol='SPY241220C00590000',
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=250.0,
            current_value=250.0,
            dte=21,
            entry_date=datetime.now(),
            contracts=1
        )
        risk_manager.add_position(position)

        # 4. Check Greeks
        greeks = risk_manager.aggregate_portfolio_greeks()
        assert greeks.position_count == 1

        # 5. Check circuit breakers
        force_exits = risk_manager.check_circuit_breakers(current_vix=15.0)
        assert len(force_exits) == 0

        # 6. Position approaches DTE limit - update
        risk_manager.update_position('SPY241220C00590000', current_value=280.0, dte=3)

        # 7. Check circuit breakers - should signal exit
        force_exits = risk_manager.check_circuit_breakers()
        assert len(force_exits) == 1

        # 8. Remove position
        risk_manager.remove_position('SPY241220C00590000')
        assert len(risk_manager.positions) == 0

    def test_march_2020_crash_scenario(self, risk_manager, sample_position):
        """Simulate March 2020 crash with VIX spike."""
        risk_manager.add_position(sample_position)

        # Initial VIX (pre-crash)
        risk_manager.check_circuit_breakers(current_vix=15.0)
        assert risk_manager.circuit_state == CircuitBreakerState.NORMAL

        # VIX starts rising (March 9, 2020)
        risk_manager.check_circuit_breakers(current_vix=20.0)
        # 33% spike should trigger HALTED
        assert risk_manager.circuit_state == CircuitBreakerState.HALTED

        # Trading should be blocked
        can_trade, reason = risk_manager.can_trade()
        assert can_trade is False

        # New positions should be rejected
        result = risk_manager.validate_new_position(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.08,
            premium=100.0,
            dte=21,
            spread_pct=0.03,
            contracts=1
        )
        assert result.passed is False
