"""
Tests for crypto/data/state.py

Comprehensive tests for CryptoSystemState class covering:
- Bar classification updates
- OHLCV data management
- Account state updates
- Pattern management
- Continuity scoring
- Veto checking
- Signal tracking
- Expiration handling

Session EQUITY-70: Phase 3 Test Coverage.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from crypto.data.state import CryptoSystemState


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestCryptoSystemStateInit:
    """Tests for CryptoSystemState initialization."""

    def test_default_initialization(self):
        """State initializes with correct defaults."""
        state = CryptoSystemState()

        assert state.bar_classifications == {}
        assert state.current_bars == {}
        assert state.active_patterns == []
        assert state.account_equity == 0.0
        assert state.available_margin == 0.0
        assert state.current_position is None
        assert state.last_update_time is None
        assert state.is_healthy is True
        assert state.last_error is None

    def test_default_symbols(self):
        """State has default symbols."""
        state = CryptoSystemState()
        assert "BTC-USD" in state.active_symbols

    def test_custom_initialization(self):
        """State can be initialized with custom values."""
        state = CryptoSystemState(
            account_equity=10000.0,
            is_healthy=False,
            active_symbols=["ETH-USD"]
        )

        assert state.account_equity == 10000.0
        assert state.is_healthy is False
        assert state.active_symbols == ["ETH-USD"]


# =============================================================================
# BAR CLASSIFICATION TESTS
# =============================================================================


class TestBarClassifications:
    """Tests for bar classification management."""

    def test_update_classification(self):
        """Classification update stores value correctly."""
        state = CryptoSystemState()
        state.update_classification("1h", 2)

        assert state.bar_classifications["1h"] == 2

    def test_update_classification_updates_timestamp(self):
        """Classification update sets last_update_time."""
        state = CryptoSystemState()
        assert state.last_update_time is None

        state.update_classification("1h", 2)
        assert state.last_update_time is not None
        assert isinstance(state.last_update_time, datetime)

    def test_update_multiple_timeframes(self):
        """Can update classifications for multiple timeframes."""
        state = CryptoSystemState()
        state.update_classification("1w", 3)
        state.update_classification("1d", 2)
        state.update_classification("4h", -2)
        state.update_classification("1h", 1)

        assert state.bar_classifications["1w"] == 3
        assert state.bar_classifications["1d"] == 2
        assert state.bar_classifications["4h"] == -2
        assert state.bar_classifications["1h"] == 1

    def test_update_overwrites_previous(self):
        """New classification overwrites previous."""
        state = CryptoSystemState()
        state.update_classification("1h", 2)
        state.update_classification("1h", -2)

        assert state.bar_classifications["1h"] == -2


# =============================================================================
# BAR DATA TESTS
# =============================================================================


class TestBarData:
    """Tests for OHLCV data management."""

    def test_update_bar_data(self):
        """Bar data update stores DataFrame correctly."""
        state = CryptoSystemState()
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [99, 100],
            "close": [102, 103],
            "volume": [1000, 1100]
        })

        state.update_bar_data("1h", df)

        assert "1h" in state.current_bars
        assert len(state.current_bars["1h"]) == 2

    def test_update_bar_data_copies_dataframe(self):
        """Bar data update makes a copy of the DataFrame."""
        state = CryptoSystemState()
        df = pd.DataFrame({"close": [100, 101]})

        state.update_bar_data("1h", df)

        # Modify original
        df.iloc[0] = 999

        # State should not be affected
        assert state.current_bars["1h"].iloc[0]["close"] != 999

    def test_update_bar_data_updates_timestamp(self):
        """Bar data update sets last_update_time."""
        state = CryptoSystemState()
        df = pd.DataFrame({"close": [100]})

        state.update_bar_data("1h", df)
        assert state.last_update_time is not None


# =============================================================================
# ACCOUNT STATE TESTS
# =============================================================================


class TestAccountState:
    """Tests for account state management."""

    def test_update_account(self):
        """Account update stores values correctly."""
        state = CryptoSystemState()
        state.update_account(equity=10000, margin=8000)

        assert state.account_equity == 10000
        assert state.available_margin == 8000
        assert state.current_position is None

    def test_update_account_with_position(self):
        """Account update stores position dict."""
        state = CryptoSystemState()
        position = {"symbol": "BTC-USD", "size": 0.1, "side": "LONG"}
        state.update_account(equity=10000, margin=8000, position=position)

        assert state.current_position == position

    def test_update_account_updates_timestamp(self):
        """Account update sets last_update_time."""
        state = CryptoSystemState()
        state.update_account(equity=10000, margin=8000)

        assert state.last_update_time is not None


# =============================================================================
# PATTERN MANAGEMENT TESTS
# =============================================================================


class TestPatternManagement:
    """Tests for pattern tracking."""

    def test_add_pattern(self):
        """Add pattern stores it correctly."""
        state = CryptoSystemState()
        pattern = {"type": "3-1-2U", "direction": "LONG", "entry_price": 100000}

        state.add_pattern(pattern)

        assert len(state.active_patterns) == 1
        assert state.active_patterns[0]["type"] == "3-1-2U"

    def test_add_pattern_adds_timestamp(self):
        """Add pattern adds detected_at timestamp."""
        state = CryptoSystemState()
        pattern = {"type": "3-1-2U"}

        state.add_pattern(pattern)

        assert "detected_at" in state.active_patterns[0]

    def test_add_multiple_patterns(self):
        """Can add multiple patterns."""
        state = CryptoSystemState()
        state.add_pattern({"type": "3-1-2U"})
        state.add_pattern({"type": "2D-2U"})
        state.add_pattern({"type": "3-2D"})

        assert len(state.active_patterns) == 3

    def test_clear_expired_patterns(self):
        """Expired patterns are removed correctly."""
        state = CryptoSystemState()

        # Add old pattern (manually set timestamp)
        old_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        state.active_patterns.append({"type": "old", "detected_at": old_time})

        # Add recent pattern
        state.add_pattern({"type": "recent"})

        # Clear patterns older than 60 minutes
        removed = state.clear_expired_patterns(max_age_minutes=60)

        assert removed == 1
        assert len(state.active_patterns) == 1
        assert state.active_patterns[0]["type"] == "recent"

    def test_clear_expired_patterns_returns_count(self):
        """clear_expired_patterns returns correct removal count."""
        state = CryptoSystemState()

        old_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        state.active_patterns = [
            {"type": "old1", "detected_at": old_time},
            {"type": "old2", "detected_at": old_time},
            {"type": "old3", "detected_at": old_time},
        ]

        removed = state.clear_expired_patterns(max_age_minutes=60)
        assert removed == 3


# =============================================================================
# PRICE RETRIEVAL TESTS
# =============================================================================


class TestPriceRetrieval:
    """Tests for price retrieval."""

    def test_get_latest_price(self):
        """Get latest price returns correct value."""
        state = CryptoSystemState()
        df = pd.DataFrame({"close": [100, 101, 102]})
        state.update_bar_data("15m", df)

        price = state.get_latest_price("15m")
        assert price == 102.0

    def test_get_latest_price_no_data(self):
        """Get latest price returns None if no data."""
        state = CryptoSystemState()
        price = state.get_latest_price("15m")
        assert price is None

    def test_get_latest_price_empty_df(self):
        """Get latest price returns None for empty DataFrame."""
        state = CryptoSystemState()
        state.current_bars["15m"] = pd.DataFrame()

        price = state.get_latest_price("15m")
        assert price is None

    def test_get_latest_price_default_timeframe(self):
        """Default timeframe is 15m."""
        state = CryptoSystemState()
        df = pd.DataFrame({"close": [100]})
        state.update_bar_data("15m", df)

        price = state.get_latest_price()  # No argument
        assert price == 100.0


# =============================================================================
# CONTINUITY SCORE TESTS
# =============================================================================


class TestContinuityScore:
    """Tests for FTFC score calculation."""

    def test_continuity_score_all_bullish(self):
        """All bullish timeframes gives score of 4."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 2, "1d": 2, "4h": 2, "1h": 2}

        score = state.get_continuity_score(direction=1)  # Bullish
        assert score == 4

    def test_continuity_score_all_bearish(self):
        """All bearish timeframes gives score of 4 for bearish direction."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": -2, "1d": -2, "4h": -2, "1h": -2}

        score = state.get_continuity_score(direction=-1)  # Bearish
        assert score == 4

    def test_continuity_score_mixed(self):
        """Mixed timeframes gives partial score."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 2, "1d": 2, "4h": -2, "1h": 1}

        score = state.get_continuity_score(direction=1)  # Bullish
        assert score == 2  # Only 1w and 1d are bullish

    def test_continuity_score_no_classifications(self):
        """No classifications gives score of 0."""
        state = CryptoSystemState()
        score = state.get_continuity_score(direction=1)
        assert score == 0

    def test_continuity_score_opposite_direction(self):
        """Opposite direction gives score of 0."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 2, "1d": 2, "4h": 2, "1h": 2}

        score = state.get_continuity_score(direction=-1)  # Looking for bearish
        assert score == 0


# =============================================================================
# VETO CHECKING TESTS
# =============================================================================


class TestVetoChecking:
    """Tests for veto logic."""

    def test_no_vetoes_when_no_inside_bars(self):
        """No vetoes when 1W and 1D are not inside bars."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 2, "1d": 2}

        can_trade, reason = state.check_vetoes()
        assert can_trade is True
        assert reason == "No vetoes active"

    def test_veto_weekly_inside(self):
        """Weekly inside bar triggers veto."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 1, "1d": 2}

        can_trade, reason = state.check_vetoes()
        assert can_trade is False
        assert "Weekly" in reason
        assert "Scenario 1" in reason

    def test_veto_daily_inside(self):
        """Daily inside bar triggers veto."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 2, "1d": 1}

        can_trade, reason = state.check_vetoes()
        assert can_trade is False
        assert "Daily" in reason
        assert "Scenario 1" in reason

    def test_veto_both_inside(self):
        """Both inside bars - weekly veto takes precedence."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 1, "1d": 1}

        can_trade, reason = state.check_vetoes()
        assert can_trade is False
        assert "Weekly" in reason  # Weekly checked first

    def test_veto_empty_classifications(self):
        """No veto when classifications are empty."""
        state = CryptoSystemState()

        can_trade, reason = state.check_vetoes()
        assert can_trade is True


# =============================================================================
# STATUS SUMMARY TESTS
# =============================================================================


class TestStatusSummary:
    """Tests for status summary generation."""

    def test_status_summary_fields(self):
        """Status summary contains expected fields."""
        state = CryptoSystemState()
        summary = state.get_status_summary()

        assert "symbols" in summary
        assert "last_update" in summary
        assert "is_healthy" in summary
        assert "bar_classifications" in summary
        assert "can_trade" in summary
        assert "account_equity" in summary
        assert "has_position" in summary
        assert "active_patterns_count" in summary

    def test_status_summary_values(self):
        """Status summary reflects current state."""
        state = CryptoSystemState()
        state.account_equity = 5000
        state.add_pattern({"type": "test"})
        state.bar_classifications = {"1w": 2, "1d": 2}

        summary = state.get_status_summary()

        assert summary["account_equity"] == 5000
        assert summary["active_patterns_count"] == 1
        assert summary["bar_classifications"]["1w"] == 2
        assert summary["can_trade"] is True

    def test_status_summary_with_veto(self):
        """Status summary includes veto reason when present."""
        state = CryptoSystemState()
        state.bar_classifications = {"1w": 1}

        summary = state.get_status_summary()

        assert summary["can_trade"] is False
        assert summary["veto_reason"] is not None


# =============================================================================
# RESET TESTS
# =============================================================================


class TestReset:
    """Tests for state reset."""

    def test_reset_clears_all_state(self):
        """Reset clears all state to defaults."""
        state = CryptoSystemState()
        state.bar_classifications = {"1h": 2}
        state.account_equity = 10000
        state.add_pattern({"type": "test"})
        state.is_healthy = False
        state.last_error = "some error"

        state.reset()

        assert state.bar_classifications == {}
        assert state.current_bars == {}
        assert state.active_patterns == []
        assert state.account_equity == 0.0
        assert state.available_margin == 0.0
        assert state.current_position is None
        assert state.last_update_time is None
        assert state.is_healthy is True
        assert state.last_error is None


# =============================================================================
# SIGNAL TRACKING TESTS
# =============================================================================


class TestSignalTracking:
    """Tests for signal tracking functionality."""

    def test_get_pending_setups(self):
        """Get pending setups filters correctly."""
        state = CryptoSystemState()
        state.active_patterns = [
            {"type": "3-1-2U", "signal_type": "SETUP", "detected_at": datetime.utcnow().isoformat()},
            {"type": "2D-2U", "signal_type": "COMPLETED", "detected_at": datetime.utcnow().isoformat()},
            {"type": "3-2D", "signal_type": "SETUP", "detected_at": datetime.utcnow().isoformat()},
        ]

        setups = state.get_pending_setups()
        assert len(setups) == 2
        assert all(s["signal_type"] == "SETUP" for s in setups)

    def test_get_pending_setups_empty(self):
        """Get pending setups returns empty list when none exist."""
        state = CryptoSystemState()
        setups = state.get_pending_setups()
        assert setups == []

    def test_check_signal_triggers_long(self):
        """Check signal triggers detects LONG trigger."""
        state = CryptoSystemState()
        state.active_patterns = [
            {
                "type": "3-1-2U",
                "signal_type": "SETUP",
                "direction": "LONG",
                "symbol": "BTC-USD",
                "entry_price": 100000,
                "detected_at": datetime.utcnow().isoformat(),
            }
        ]

        # Price at entry
        triggered = state.check_signal_triggers({"BTC-USD": 100000})
        assert len(triggered) == 1
        assert "triggered_at" in triggered[0]
        assert triggered[0]["triggered_price"] == 100000

    def test_check_signal_triggers_short(self):
        """Check signal triggers detects SHORT trigger."""
        state = CryptoSystemState()
        state.active_patterns = [
            {
                "type": "3-1-2D",
                "signal_type": "SETUP",
                "direction": "SHORT",
                "symbol": "BTC-USD",
                "entry_price": 100000,
                "detected_at": datetime.utcnow().isoformat(),
            }
        ]

        # Price at entry
        triggered = state.check_signal_triggers({"BTC-USD": 100000})
        assert len(triggered) == 1

    def test_check_signal_triggers_not_triggered(self):
        """Check signal triggers returns empty when not triggered."""
        state = CryptoSystemState()
        state.active_patterns = [
            {
                "type": "3-1-2U",
                "signal_type": "SETUP",
                "direction": "LONG",
                "symbol": "BTC-USD",
                "entry_price": 100000,
                "detected_at": datetime.utcnow().isoformat(),
            }
        ]

        # Price below entry for LONG
        triggered = state.check_signal_triggers({"BTC-USD": 99000})
        assert len(triggered) == 0

    def test_check_signal_triggers_missing_symbol(self):
        """Check signal triggers skips missing symbols."""
        state = CryptoSystemState()
        state.active_patterns = [
            {
                "type": "3-1-2U",
                "signal_type": "SETUP",
                "direction": "LONG",
                "symbol": "BTC-USD",
                "entry_price": 100000,
                "detected_at": datetime.utcnow().isoformat(),
            }
        ]

        # Different symbol in prices
        triggered = state.check_signal_triggers({"ETH-USD": 3000})
        assert len(triggered) == 0


# =============================================================================
# SIGNAL EXPIRATION TESTS
# =============================================================================


class TestSignalExpiration:
    """Tests for signal expiration handling."""

    def test_remove_expired_signals(self):
        """Expired signals are removed correctly."""
        state = CryptoSystemState()

        old_time = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        recent_time = datetime.utcnow().isoformat()

        state.active_patterns = [
            {"type": "old", "detected_at": old_time},
            {"type": "recent", "detected_at": recent_time},
        ]

        removed = state.remove_expired_signals(max_age_hours=24)

        assert removed == 1
        assert len(state.active_patterns) == 1
        assert state.active_patterns[0]["type"] == "recent"

    def test_remove_expired_signals_none_expired(self):
        """No signals removed when none expired."""
        state = CryptoSystemState()
        state.add_pattern({"type": "recent"})

        removed = state.remove_expired_signals(max_age_hours=24)

        assert removed == 0
        assert len(state.active_patterns) == 1


# =============================================================================
# SIGNAL QUERY TESTS
# =============================================================================


class TestSignalQueries:
    """Tests for signal query methods."""

    def test_get_signals_by_symbol(self):
        """Get signals by symbol filters correctly."""
        state = CryptoSystemState()
        state.active_patterns = [
            {"type": "A", "symbol": "BTC-USD", "detected_at": datetime.utcnow().isoformat()},
            {"type": "B", "symbol": "ETH-USD", "detected_at": datetime.utcnow().isoformat()},
            {"type": "C", "symbol": "BTC-USD", "detected_at": datetime.utcnow().isoformat()},
        ]

        btc_signals = state.get_signals_by_symbol("BTC-USD")
        assert len(btc_signals) == 2
        assert all(s["symbol"] == "BTC-USD" for s in btc_signals)

    def test_get_signals_by_symbol_empty(self):
        """Get signals by symbol returns empty for unknown symbol."""
        state = CryptoSystemState()
        signals = state.get_signals_by_symbol("UNKNOWN")
        assert signals == []

    def test_get_signals_by_timeframe(self):
        """Get signals by timeframe filters correctly."""
        state = CryptoSystemState()
        state.active_patterns = [
            {"type": "A", "timeframe": "1h", "detected_at": datetime.utcnow().isoformat()},
            {"type": "B", "timeframe": "4h", "detected_at": datetime.utcnow().isoformat()},
            {"type": "C", "timeframe": "1h", "detected_at": datetime.utcnow().isoformat()},
        ]

        hourly_signals = state.get_signals_by_timeframe("1h")
        assert len(hourly_signals) == 2
        assert all(s["timeframe"] == "1h" for s in hourly_signals)

    def test_get_signals_by_timeframe_empty(self):
        """Get signals by timeframe returns empty for unknown timeframe."""
        state = CryptoSystemState()
        signals = state.get_signals_by_timeframe("1w")
        assert signals == []
