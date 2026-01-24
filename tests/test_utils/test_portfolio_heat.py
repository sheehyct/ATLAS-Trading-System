"""
Tests for utils/portfolio_heat.py - Portfolio heat management.

EQUITY-83: Phase 3 test coverage for portfolio heat module.

Tests cover:
- PortfolioHeatManager initialization
- Heat calculation
- Trade acceptance/rejection
- Position management
- Risk tracking
"""

import pytest
from io import StringIO
import sys

from utils.portfolio_heat import PortfolioHeatManager


# =============================================================================
# Initialization Tests
# =============================================================================

class TestPortfolioHeatInit:
    """Tests for PortfolioHeatManager initialization."""

    def test_default_max_heat(self):
        """Test default max_heat is 8%."""
        manager = PortfolioHeatManager()
        assert manager.max_heat == 0.08

    def test_custom_max_heat(self):
        """Test custom max_heat is accepted."""
        manager = PortfolioHeatManager(max_heat=0.06)
        assert manager.max_heat == 0.06

    def test_max_heat_lower_bound(self):
        """Test max_heat below 6% raises error."""
        with pytest.raises(ValueError) as exc_info:
            PortfolioHeatManager(max_heat=0.05)

        assert "0.06" in str(exc_info.value)

    def test_max_heat_upper_bound(self):
        """Test max_heat above 10% raises error."""
        with pytest.raises(ValueError) as exc_info:
            PortfolioHeatManager(max_heat=0.15)

        assert "0.10" in str(exc_info.value)

    def test_empty_positions_on_init(self):
        """Test positions dict is empty on init."""
        manager = PortfolioHeatManager()
        assert manager.active_positions == {}

    def test_boundary_max_heat_6pct(self):
        """Test 6% max_heat is accepted."""
        manager = PortfolioHeatManager(max_heat=0.06)
        assert manager.max_heat == 0.06

    def test_boundary_max_heat_10pct(self):
        """Test 10% max_heat is accepted."""
        manager = PortfolioHeatManager(max_heat=0.10)
        assert manager.max_heat == 0.10


# =============================================================================
# Heat Calculation Tests
# =============================================================================

class TestHeatCalculation:
    """Tests for calculate_current_heat method."""

    def test_empty_positions_zero_heat(self):
        """Test empty positions gives zero heat."""
        manager = PortfolioHeatManager()
        heat = manager.calculate_current_heat(100000)
        assert heat == 0.0

    def test_single_position_heat(self):
        """Test heat calculation with single position."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        heat = manager.calculate_current_heat(100000)
        assert heat == pytest.approx(0.02)  # 2% heat

    def test_multiple_positions_heat(self):
        """Test heat calculation with multiple positions."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)
        manager.add_position('QQQ', 2500)
        manager.add_position('AAPL', 1500)

        heat = manager.calculate_current_heat(100000)
        # (2000 + 2500 + 1500) / 100000 = 6%
        assert heat == pytest.approx(0.06)

    def test_zero_capital_returns_zero(self):
        """Test zero capital returns zero heat."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        heat = manager.calculate_current_heat(0)
        assert heat == 0.0

    def test_negative_capital_returns_zero(self):
        """Test negative capital returns zero heat."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        heat = manager.calculate_current_heat(-10000)
        assert heat == 0.0


# =============================================================================
# Trade Acceptance Tests
# =============================================================================

class TestCanAcceptTrade:
    """Tests for can_accept_trade method."""

    def test_accept_first_trade(self):
        """Test first trade is accepted when under limit."""
        manager = PortfolioHeatManager(max_heat=0.08)

        accepted = manager.can_accept_trade('SPY', 2000, 100000)
        assert accepted is True

    def test_accept_trade_under_limit(self):
        """Test trade accepted when total heat under limit."""
        manager = PortfolioHeatManager(max_heat=0.08)
        manager.add_position('SPY', 2000)  # 2% heat

        # Adding 3% heat (total 5% < 8%)
        accepted = manager.can_accept_trade('QQQ', 3000, 100000)
        assert accepted is True

    def test_reject_trade_over_limit(self):
        """Test trade rejected when it would exceed limit."""
        manager = PortfolioHeatManager(max_heat=0.08)
        manager.add_position('SPY', 2000)
        manager.add_position('QQQ', 3000)
        manager.add_position('AAPL', 2000)  # 7% heat

        # Adding 2% heat would push to 9% > 8%
        # Capture stdout to verify rejection message
        captured = StringIO()
        sys.stdout = captured
        accepted = manager.can_accept_trade('MSFT', 2000, 100000)
        sys.stdout = sys.__stdout__

        assert accepted is False
        assert "REJECTED" in captured.getvalue()

    def test_reject_at_exact_limit(self):
        """Test trade rejected when it would exactly exceed limit."""
        manager = PortfolioHeatManager(max_heat=0.08)
        manager.add_position('SPY', 4000)  # 4% heat

        # Adding exactly 4% + epsilon would exceed 8%
        captured = StringIO()
        sys.stdout = captured
        accepted = manager.can_accept_trade('QQQ', 4001, 100000)
        sys.stdout = sys.__stdout__

        assert accepted is False

    def test_accept_at_exact_limit(self):
        """Test trade accepted when it exactly meets limit."""
        manager = PortfolioHeatManager(max_heat=0.08)
        manager.add_position('SPY', 4000)  # 4% heat

        # Adding exactly 4% = 8% (at limit, not over)
        accepted = manager.can_accept_trade('QQQ', 4000, 100000)
        assert accepted is True


# =============================================================================
# Position Management Tests
# =============================================================================

class TestPositionManagement:
    """Tests for add/remove/update position methods."""

    def test_add_position(self):
        """Test adding a position."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        assert 'SPY' in manager.active_positions
        assert manager.active_positions['SPY'] == 2000

    def test_add_duplicate_raises_error(self):
        """Test adding duplicate position raises error."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        with pytest.raises(ValueError) as exc_info:
            manager.add_position('SPY', 3000)

        assert "already exists" in str(exc_info.value)

    def test_add_negative_risk_raises_error(self):
        """Test adding negative risk raises error."""
        manager = PortfolioHeatManager()

        with pytest.raises(ValueError) as exc_info:
            manager.add_position('SPY', -1000)

        assert "non-negative" in str(exc_info.value)

    def test_remove_position(self):
        """Test removing a position."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)
        manager.add_position('QQQ', 1500)

        manager.remove_position('SPY')

        assert 'SPY' not in manager.active_positions
        assert 'QQQ' in manager.active_positions

    def test_remove_nonexistent_silent(self):
        """Test removing nonexistent position is silent."""
        manager = PortfolioHeatManager()

        # Should not raise
        manager.remove_position('NONEXISTENT')
        assert manager.get_position_count() == 0

    def test_update_position_risk(self):
        """Test updating position risk."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        manager.update_position_risk('SPY', 1500)

        assert manager.active_positions['SPY'] == 1500

    def test_update_nonexistent_raises_error(self):
        """Test updating nonexistent position raises error."""
        manager = PortfolioHeatManager()

        with pytest.raises(ValueError) as exc_info:
            manager.update_position_risk('SPY', 1500)

        assert "not found" in str(exc_info.value)

    def test_update_negative_risk_raises_error(self):
        """Test updating to negative risk raises error."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        with pytest.raises(ValueError) as exc_info:
            manager.update_position_risk('SPY', -500)

        assert "non-negative" in str(exc_info.value)


# =============================================================================
# Query Methods Tests
# =============================================================================

class TestQueryMethods:
    """Tests for query methods."""

    def test_get_active_positions(self):
        """Test getting active positions."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)
        manager.add_position('QQQ', 1500)

        positions = manager.get_active_positions()

        assert len(positions) == 2
        assert positions['SPY'] == 2000
        assert positions['QQQ'] == 1500

    def test_get_active_positions_returns_copy(self):
        """Test get_active_positions returns a copy."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)

        positions = manager.get_active_positions()
        positions['NEW'] = 5000  # Modify the returned dict

        # Original should be unchanged
        assert 'NEW' not in manager.active_positions

    def test_get_position_count(self):
        """Test getting position count."""
        manager = PortfolioHeatManager()
        assert manager.get_position_count() == 0

        manager.add_position('SPY', 2000)
        assert manager.get_position_count() == 1

        manager.add_position('QQQ', 1500)
        assert manager.get_position_count() == 2

        manager.remove_position('SPY')
        assert manager.get_position_count() == 1


# =============================================================================
# Reset Tests
# =============================================================================

class TestReset:
    """Tests for reset method."""

    def test_reset_clears_positions(self):
        """Test reset clears all positions."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 2000)
        manager.add_position('QQQ', 1500)
        manager.add_position('AAPL', 1000)

        manager.reset()

        assert manager.get_position_count() == 0
        assert manager.active_positions == {}

    def test_reset_resets_heat(self):
        """Test reset brings heat to zero."""
        manager = PortfolioHeatManager()
        manager.add_position('SPY', 5000)

        manager.reset()

        heat = manager.calculate_current_heat(100000)
        assert heat == 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for realistic scenarios."""

    def test_typical_trading_session(self):
        """Test typical trading session workflow."""
        manager = PortfolioHeatManager(max_heat=0.08)
        capital = 100000

        # Morning: Add first position
        assert manager.can_accept_trade('SPY', 2000, capital)
        manager.add_position('SPY', 2000)
        assert manager.calculate_current_heat(capital) == pytest.approx(0.02)

        # Add second position
        assert manager.can_accept_trade('QQQ', 2500, capital)
        manager.add_position('QQQ', 2500)
        assert manager.calculate_current_heat(capital) == pytest.approx(0.045)

        # Add third position
        assert manager.can_accept_trade('AAPL', 2000, capital)
        manager.add_position('AAPL', 2000)
        assert manager.calculate_current_heat(capital) == pytest.approx(0.065)

        # Try to add fourth - would exceed limit
        captured = StringIO()
        sys.stdout = captured
        assert not manager.can_accept_trade('MSFT', 2000, capital)
        sys.stdout = sys.__stdout__

        # Close one position
        manager.remove_position('SPY')
        assert manager.calculate_current_heat(capital) == pytest.approx(0.045)

        # Now we can add MSFT
        assert manager.can_accept_trade('MSFT', 2000, capital)

    def test_trailing_stop_reduces_heat(self):
        """Test trailing stops reduce heat over time."""
        manager = PortfolioHeatManager(max_heat=0.08)
        capital = 100000

        # Enter with 2% risk
        manager.add_position('SPY', 2000)
        initial_heat = manager.calculate_current_heat(capital)
        assert initial_heat == pytest.approx(0.02)

        # Trail stop - reduce risk to 1.5%
        manager.update_position_risk('SPY', 1500)
        reduced_heat = manager.calculate_current_heat(capital)
        assert reduced_heat == pytest.approx(0.015)

        # Trail stop further - reduce risk to 1%
        manager.update_position_risk('SPY', 1000)
        further_reduced = manager.calculate_current_heat(capital)
        assert further_reduced == pytest.approx(0.01)

    def test_zero_risk_position(self):
        """Test position with zero risk (break-even stop)."""
        manager = PortfolioHeatManager()
        capital = 100000

        manager.add_position('SPY', 2000)

        # Move stop to break-even (zero risk)
        manager.update_position_risk('SPY', 0)

        assert manager.active_positions['SPY'] == 0
        assert manager.calculate_current_heat(capital) == 0.0
