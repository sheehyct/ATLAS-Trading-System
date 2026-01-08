"""
TFC Re-evaluation at Entry Tests - Session EQUITY-49

Tests the TFC re-evaluation logic that checks timeframe continuity alignment
at entry time, not just at pattern detection time.

Background:
- TFC is evaluated once at pattern detection (could be hours/days earlier)
- By entry trigger time, TFC alignment may have changed
- Entry should only proceed if TFC still supports the trade direction

Test scenarios:
1. TFC unchanged - entry should proceed
2. TFC improved - entry should proceed
3. TFC degraded but above threshold - entry should proceed
4. TFC degraded below threshold - entry should be blocked
5. TFC direction flipped - entry should be blocked
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pytz

from strat.signal_automation.signal_store import StoredSignal, SignalType, SignalStatus
from strat.signal_automation.daemon import SignalDaemon
from strat.signal_automation.config import SignalAutomationConfig, ExecutionConfig
from strat.timeframe_continuity_adapter import ContinuityAssessment


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config(tmp_path):
    """Create minimal config for daemon with TFC re-eval enabled."""
    exec_config = ExecutionConfig(
        enabled=False,  # Don't need actual execution
        tfc_reeval_enabled=True,
        tfc_reeval_min_strength=2,
        tfc_reeval_block_on_flip=True,
        tfc_reeval_log_always=True,
    )
    return SignalAutomationConfig(
        store_path=str(tmp_path / 'signals'),
        execution=exec_config,
    )


@pytest.fixture
def mock_config_disabled(tmp_path):
    """Create config with TFC re-eval disabled."""
    exec_config = ExecutionConfig(
        enabled=False,
        tfc_reeval_enabled=False,
    )
    return SignalAutomationConfig(
        store_path=str(tmp_path / 'signals'),
        execution=exec_config,
    )


@pytest.fixture
def daemon(mock_config):
    """Create daemon instance for testing."""
    daemon = SignalDaemon(config=mock_config)
    # Mock the scanner's evaluate_tfc method
    daemon.scanner = Mock()
    return daemon


@pytest.fixture
def daemon_disabled(mock_config_disabled):
    """Create daemon with TFC re-eval disabled."""
    daemon = SignalDaemon(config=mock_config_disabled)
    daemon.scanner = Mock()
    return daemon


def create_signal_with_tfc(
    tfc_score: int = 3,
    tfc_alignment: str = "3/4 BULLISH",
    passes_flexible: bool = True,
    direction: str = 'CALL',
    timeframe: str = '1D',
) -> StoredSignal:
    """Helper to create test signals with TFC data."""
    et = pytz.timezone('America/New_York')
    now = datetime.now(et)
    return StoredSignal(
        signal_key=f'TEST_{timeframe}_3-2U_{direction}_{now.strftime("%Y%m%d%H%M")}',
        pattern_type='3-2U',
        direction=direction,
        symbol='TEST',
        timeframe=timeframe,
        detected_time=now,
        entry_trigger=100.0,
        stop_price=95.0,
        target_price=105.0,
        magnitude_pct=5.0,
        risk_reward=1.0,
        tfc_score=tfc_score,
        tfc_alignment=tfc_alignment,
        passes_flexible=passes_flexible,
        signal_type=SignalType.SETUP.value,
    )


def create_mock_tfc_assessment(
    strength: int = 3,
    direction: str = "bullish",
    passes_flexible: bool = True,
) -> ContinuityAssessment:
    """Create mock TFC assessment for testing."""
    return ContinuityAssessment(
        strength=strength,
        passes_flexible=passes_flexible,
        aligned_timeframes=["1D", "1W", "1M"][:strength],
        required_timeframes=["1H", "1D", "1W", "1M"],
        direction=direction,
        detection_timeframe="1D",
        risk_multiplier=1.0 if strength >= 4 else 0.5 if strength >= 3 else 0.0,
        priority_rank=1 if strength >= 4 else 2 if strength >= 3 else 4,
    )


# =============================================================================
# TFC UNCHANGED/IMPROVED TESTS
# =============================================================================


class TestTFCReevalNoChange:
    """Tests where TFC is unchanged or improved - entry should proceed."""

    def test_tfc_unchanged_allows_entry(self, daemon):
        """Entry should proceed when TFC is unchanged."""
        # Original: 3/4 BULLISH
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction='CALL'
        )

        # Current: 3/4 bullish (same)
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_tfc_improved_allows_entry(self, daemon):
        """Entry should proceed when TFC has improved."""
        # Original: 2/4 BULLISH
        signal = create_signal_with_tfc(
            tfc_score=2,
            tfc_alignment="2/4 BULLISH",
            direction='CALL'
        )

        # Current: 4/4 bullish (improved)
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=4,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""


# =============================================================================
# TFC DEGRADED BUT ABOVE THRESHOLD TESTS
# =============================================================================


class TestTFCReevalDegradedAboveThreshold:
    """Tests where TFC degraded but still above minimum threshold."""

    def test_tfc_degraded_above_threshold_allows_entry(self, daemon):
        """Entry should proceed when TFC degraded but above min_strength (2)."""
        # Original: 4/4 BULLISH
        signal = create_signal_with_tfc(
            tfc_score=4,
            tfc_alignment="4/4 BULLISH",
            direction='CALL'
        )

        # Current: 2/4 bullish (degraded but >= min_strength of 2)
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=2,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""


# =============================================================================
# TFC DEGRADED BELOW THRESHOLD TESTS
# =============================================================================


class TestTFCReevalDegradedBelowThreshold:
    """Tests where TFC degraded below minimum threshold - should block."""

    def test_tfc_degraded_below_threshold_blocks_entry(self, daemon):
        """Entry should be blocked when TFC drops below min_strength."""
        # Original: 3/4 BULLISH
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction='CALL'
        )

        # Current: 1/4 bullish (below min_strength of 2)
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=1,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC strength 1 < min threshold 2" in reason

    def test_tfc_zero_blocks_entry(self, daemon):
        """Entry should be blocked when TFC drops to zero."""
        # Original: 3/4 BULLISH
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction='CALL'
        )

        # Current: 0/4 (total loss of alignment)
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=0,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC strength 0 < min threshold 2" in reason


# =============================================================================
# TFC DIRECTION FLIP TESTS
# =============================================================================


class TestTFCReevalDirectionFlip:
    """Tests where TFC direction has flipped - should block."""

    def test_bullish_to_bearish_flip_blocks_entry(self, daemon):
        """Entry should be blocked when TFC flips from bullish to bearish."""
        # Original: 3/4 BULLISH, direction CALL
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction='CALL'
        )

        # Current: 3/4 bearish (same strength, but direction flipped)
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bearish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC direction flipped" in reason
        assert "bullish" in reason and "bearish" in reason

    def test_bearish_to_bullish_flip_blocks_entry(self, daemon):
        """Entry should be blocked when TFC flips from bearish to bullish."""
        # Original: 3/4 BEARISH, direction PUT
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BEARISH",
            direction='PUT'
        )

        # Current: 3/4 bullish (direction flipped)
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC direction flipped" in reason


# =============================================================================
# DISABLED/EDGE CASE TESTS
# =============================================================================


class TestTFCReevalDisabled:
    """Tests when TFC re-evaluation is disabled."""

    def test_disabled_allows_all_entries(self, daemon_disabled):
        """When disabled, all entries should proceed regardless of TFC change."""
        # Even with massive degradation
        signal = create_signal_with_tfc(
            tfc_score=4,
            tfc_alignment="4/4 BULLISH",
            direction='CALL'
        )

        # Would be blocked if enabled (direction flip + strength drop)
        daemon_disabled.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=0,
            direction="bearish"
        )

        should_block, reason = daemon_disabled._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""


class TestTFCReevalEdgeCases:
    """Edge case tests for TFC re-evaluation."""

    def test_scanner_error_allows_entry(self, daemon):
        """If TFC evaluation fails, entry should proceed (don't block on errors)."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction='CALL'
        )

        # Scanner throws exception
        daemon.scanner.evaluate_tfc.side_effect = Exception("Data fetch failed")

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Should not block due to error (entry proceeds with warning logged)
        assert should_block is False
        assert reason == ""

    def test_empty_original_alignment_handles_gracefully(self, daemon):
        """Handles signals with empty original TFC alignment."""
        signal = create_signal_with_tfc(
            tfc_score=0,
            tfc_alignment="",  # No original alignment
            direction='CALL'
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        # Should not crash, and should evaluate based on current TFC
        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # With strength 3 >= min_strength 2, should allow
        assert should_block is False

    def test_put_direction_evaluates_bearish(self, daemon):
        """PUT signals should evaluate TFC in bearish direction."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BEARISH",
            direction='PUT'
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bearish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Verify evaluate_tfc was called with direction=-1 (bearish)
        daemon.scanner.evaluate_tfc.assert_called_with(
            symbol='TEST',
            detection_timeframe='1D',
            direction=-1  # PUT = bearish
        )
        assert should_block is False


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestTFCReevalIntegration:
    """Integration-style tests covering real-world scenarios."""

    def test_daily_setup_triggered_days_later(self, daemon):
        """Daily setup detected Monday, triggered Wednesday with degraded TFC."""
        # Scenario: Setup detected on Monday with 4/4 TFC
        # By Wednesday when price hits trigger, TFC dropped to 2/4
        signal = create_signal_with_tfc(
            tfc_score=4,
            tfc_alignment="4/4 BULLISH",
            direction='CALL',
            timeframe='1D'
        )

        # 2 days later, TFC degraded but still >= min_strength
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=2,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Should proceed despite degradation (2 >= min_strength of 2)
        assert should_block is False

    def test_hourly_setup_same_session_improved(self, daemon):
        """Hourly setup detected 10:30, triggered 14:00 with improved TFC."""
        signal = create_signal_with_tfc(
            tfc_score=2,
            tfc_alignment="2/4 BULLISH",
            direction='CALL',
            timeframe='1H'
        )

        # Later in session, TFC improved
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=4,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False

    def test_weekly_setup_market_reversal(self, daemon):
        """Weekly setup invalidated by market reversal flipping TFC."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction='CALL',
            timeframe='1W'
        )

        # Market reversed, TFC now bearish
        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bearish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Should block due to direction flip
        assert should_block is True
        assert "direction flipped" in reason
