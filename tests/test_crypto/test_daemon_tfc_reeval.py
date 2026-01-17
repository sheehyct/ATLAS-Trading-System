"""
TFC Re-evaluation at Entry Tests for Crypto Daemon - Session EQUITY-68

Tests the TFC re-evaluation logic ported from equity daemon (EQUITY-49) to crypto.

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
6. Error handling - should fail-open (proceed with warning)
7. Config control - respect enabled/disabled settings
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext
from crypto.scanning.daemon import CryptoSignalDaemon, CryptoDaemonConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_daemon_config():
    """Create minimal config for crypto daemon with TFC re-eval enabled."""
    return CryptoDaemonConfig(
        symbols=["BTC-PERP-INTX"],
        scan_interval=900,  # 15 min
        tfc_reeval_enabled=True,
        tfc_reeval_min_strength=3,
        tfc_reeval_block_on_flip=True,
        tfc_reeval_log_always=True,
        api_enabled=False,  # Don't start API server in tests
    )


@pytest.fixture
def mock_daemon_config_disabled():
    """Create config with TFC re-eval disabled."""
    return CryptoDaemonConfig(
        symbols=["BTC-PERP-INTX"],
        scan_interval=900,
        tfc_reeval_enabled=False,
        api_enabled=False,
    )


@pytest.fixture
def mock_daemon_config_no_flip_block():
    """Create config with block_on_flip disabled."""
    return CryptoDaemonConfig(
        symbols=["BTC-PERP-INTX"],
        scan_interval=900,
        tfc_reeval_enabled=True,
        tfc_reeval_min_strength=3,
        tfc_reeval_block_on_flip=False,  # Allow direction flips
        tfc_reeval_log_always=True,
        api_enabled=False,
    )


@pytest.fixture
def mock_scanner():
    """Create mock CryptoSignalScanner."""
    scanner = Mock()
    return scanner


@pytest.fixture
def daemon(mock_daemon_config, mock_scanner):
    """Create daemon instance for testing with mocked scanner."""
    with patch('crypto.scanning.daemon.CoinbaseClient'):
        with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
            daemon = CryptoSignalDaemon(config=mock_daemon_config)
            daemon.scanner = mock_scanner
            return daemon


@pytest.fixture
def daemon_disabled(mock_daemon_config_disabled, mock_scanner):
    """Create daemon with TFC re-eval disabled."""
    with patch('crypto.scanning.daemon.CoinbaseClient'):
        with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
            daemon = CryptoSignalDaemon(config=mock_daemon_config_disabled)
            daemon.scanner = mock_scanner
            return daemon


@pytest.fixture
def daemon_no_flip_block(mock_daemon_config_no_flip_block, mock_scanner):
    """Create daemon that allows direction flips."""
    with patch('crypto.scanning.daemon.CoinbaseClient'):
        with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
            daemon = CryptoSignalDaemon(config=mock_daemon_config_no_flip_block)
            daemon.scanner = mock_scanner
            return daemon


def create_signal_with_tfc(
    tfc_score: int = 3,
    tfc_alignment: str = "3/4 BULLISH",
    tfc_passes: bool = True,
    direction: str = "LONG",
    timeframe: str = "1h",
    symbol: str = "BTC-PERP-INTX",
) -> CryptoDetectedSignal:
    """Helper to create test signals with TFC data."""
    context = CryptoSignalContext(
        tfc_score=tfc_score,
        tfc_alignment=tfc_alignment,
        tfc_passes=tfc_passes,
        risk_multiplier=1.0 if tfc_score >= 3 else 0.5,
        priority_rank=1 if tfc_score >= 4 else 2,
    )

    return CryptoDetectedSignal(
        pattern_type="3-2U" if direction == "LONG" else "3-2D",
        direction=direction,
        symbol=symbol,
        timeframe=timeframe,
        detected_time=datetime.now(timezone.utc),
        entry_trigger=50000.0,
        stop_price=49000.0 if direction == "LONG" else 51000.0,
        target_price=52000.0 if direction == "LONG" else 48000.0,
        magnitude_pct=4.0,
        risk_reward=2.0,
        context=context,
        signal_type="SETUP",
        setup_bar_high=50500.0,
        setup_bar_low=49500.0,
    )


def create_mock_tfc_assessment(
    strength: int = 3,
    direction: str = "bullish",
    passes_flexible: bool = True,
):
    """Create mock TFC assessment for testing."""
    assessment = Mock()
    assessment.strength = strength
    assessment.direction = direction
    assessment.passes_flexible = passes_flexible
    assessment.alignment_label = Mock(return_value=f"{strength}/4 {direction.upper()}")
    return assessment


# =============================================================================
# TFC UNCHANGED/IMPROVED TESTS
# =============================================================================


class TestTFCReevalNoChange:
    """Tests where TFC is unchanged or improved - entry should proceed."""

    def test_tfc_unchanged_allows_entry(self, daemon):
        """Entry should proceed when TFC is unchanged."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_tfc_improved_allows_entry(self, daemon):
        """Entry should proceed when TFC has improved."""
        signal = create_signal_with_tfc(
            tfc_score=2,
            tfc_alignment="2/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=4,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""


# =============================================================================
# TFC DEGRADED TESTS
# =============================================================================


class TestTFCReevalDegraded:
    """Tests where TFC has degraded."""

    def test_tfc_degraded_above_threshold_allows_entry(self, daemon):
        """Entry should proceed when TFC degraded but >= min_strength (3)."""
        signal = create_signal_with_tfc(
            tfc_score=4,
            tfc_alignment="4/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,  # Degraded but still >= min (3)
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_tfc_degraded_below_threshold_blocks_entry(self, daemon):
        """Entry should be blocked when TFC drops below min_strength (3)."""
        signal = create_signal_with_tfc(
            tfc_score=4,
            tfc_alignment="4/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=2,  # Below min threshold of 3
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC strength 2 < min threshold 3" in reason

    def test_tfc_dropped_to_zero_blocks_entry(self, daemon):
        """Entry should be blocked when TFC drops to 0."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=0,
            direction=""  # No direction at 0
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC strength 0 < min threshold 3" in reason


# =============================================================================
# DIRECTION FLIP TESTS
# =============================================================================


class TestTFCReevalDirectionFlip:
    """Tests for TFC direction flip detection."""

    def test_bullish_to_bearish_flip_blocks_entry(self, daemon):
        """Entry should be blocked when TFC flips from bullish to bearish."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,  # Same strength, but flipped direction
            direction="bearish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC direction flipped from bullish to bearish" in reason

    def test_bearish_to_bullish_flip_blocks_entry(self, daemon):
        """Entry should be blocked when TFC flips from bearish to bullish."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BEARISH",
            direction="SHORT"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC direction flipped from bearish to bullish" in reason

    def test_direction_unchanged_allows_entry(self, daemon):
        """Entry should proceed when TFC direction is unchanged."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BEARISH",
            direction="SHORT"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bearish"  # Same direction
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_block_on_flip_disabled_allows_flip(self, daemon_no_flip_block):
        """Entry should proceed on flip when block_on_flip is False."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction="LONG"
        )

        daemon_no_flip_block.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,  # Above threshold
            direction="bearish"  # Flipped
        )

        should_block, reason = daemon_no_flip_block._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""


# =============================================================================
# ERROR HANDLING TESTS (FAIL-OPEN)
# =============================================================================


class TestTFCReevalErrorHandling:
    """Tests for error handling - should fail-open."""

    def test_connection_error_allows_entry(self, daemon):
        """Entry should proceed (with warning) on connection error."""
        signal = create_signal_with_tfc(tfc_score=3, direction="LONG")

        daemon.scanner.evaluate_tfc.side_effect = ConnectionError("Network error")

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_timeout_error_allows_entry(self, daemon):
        """Entry should proceed (with warning) on timeout error."""
        signal = create_signal_with_tfc(tfc_score=3, direction="LONG")

        daemon.scanner.evaluate_tfc.side_effect = TimeoutError("Request timed out")

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_value_error_allows_entry(self, daemon):
        """Entry should proceed (with warning) on value error."""
        signal = create_signal_with_tfc(tfc_score=3, direction="LONG")

        daemon.scanner.evaluate_tfc.side_effect = ValueError("Invalid data")

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_unexpected_error_allows_entry(self, daemon):
        """Entry should proceed (with error log) on unexpected exception."""
        signal = create_signal_with_tfc(tfc_score=3, direction="LONG")

        daemon.scanner.evaluate_tfc.side_effect = RuntimeError("Unexpected error")

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_none_assessment_allows_entry(self, daemon):
        """Entry should proceed when evaluate_tfc returns None."""
        signal = create_signal_with_tfc(tfc_score=3, direction="LONG")

        daemon.scanner.evaluate_tfc.return_value = None

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_invalid_assessment_allows_entry(self, daemon):
        """Entry should proceed when assessment has no strength attribute."""
        signal = create_signal_with_tfc(tfc_score=3, direction="LONG")

        invalid_assessment = Mock(spec=[])  # No attributes
        daemon.scanner.evaluate_tfc.return_value = invalid_assessment

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""


# =============================================================================
# CONFIG CONTROL TESTS
# =============================================================================


class TestTFCReevalConfigControl:
    """Tests for configuration control."""

    def test_disabled_skips_check(self, daemon_disabled):
        """TFC re-eval should be skipped when disabled."""
        signal = create_signal_with_tfc(
            tfc_score=4,
            tfc_alignment="4/4 BULLISH",
            direction="LONG"
        )

        # This would normally block (flip + degradation)
        daemon_disabled.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=0,
            direction="bearish"
        )

        should_block, reason = daemon_disabled._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""
        # Scanner should not be called when disabled
        daemon_disabled.scanner.evaluate_tfc.assert_not_called()


# =============================================================================
# EDGE CASES
# =============================================================================


class TestTFCReevalEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_missing_original_tfc_data_proceeds(self, daemon):
        """Entry should proceed when signal has no original TFC data."""
        signal = CryptoDetectedSignal(
            pattern_type="3-2U",
            direction="LONG",
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=50000.0,
            stop_price=49000.0,
            target_price=52000.0,
            magnitude_pct=4.0,
            risk_reward=2.0,
            context=CryptoSignalContext(),  # Empty context
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Should not block - direction flip detection skipped without original direction
        assert should_block is False

    def test_empty_original_alignment_skips_flip_detection(self, daemon):
        """Direction flip detection should be skipped with empty alignment."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="",  # Empty alignment
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bearish"  # This would be a flip, but can't detect
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Should not block due to flip (can't detect), but could block for other reasons
        assert should_block is False

    def test_no_context_proceeds(self, daemon):
        """Entry should proceed when signal.context is None."""
        signal = CryptoDetectedSignal(
            pattern_type="3-2U",
            direction="LONG",
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=50000.0,
            stop_price=49000.0,
            target_price=52000.0,
            magnitude_pct=4.0,
            risk_reward=2.0,
        )
        signal.context = None  # Explicitly None

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        # Should not raise exception
        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)
        assert should_block is False

    def test_strength_exactly_at_threshold_allows_entry(self, daemon):
        """Entry should proceed when TFC strength equals min_strength exactly."""
        signal = create_signal_with_tfc(
            tfc_score=4,
            tfc_alignment="4/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,  # Exactly at threshold (min_strength=3)
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is False
        assert reason == ""

    def test_short_direction_evaluates_correctly(self, daemon):
        """SHORT direction should be passed correctly to evaluate_tfc."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BEARISH",
            direction="SHORT"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bearish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Verify evaluate_tfc was called with direction=-1 for SHORT
        daemon.scanner.evaluate_tfc.assert_called_once()
        call_kwargs = daemon.scanner.evaluate_tfc.call_args[1]
        assert call_kwargs['direction'] == -1

    def test_long_direction_evaluates_correctly(self, daemon):
        """LONG direction should be passed correctly to evaluate_tfc."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=3,
            direction="bullish"
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        # Verify evaluate_tfc was called with direction=1 for LONG
        daemon.scanner.evaluate_tfc.assert_called_once()
        call_kwargs = daemon.scanner.evaluate_tfc.call_args[1]
        assert call_kwargs['direction'] == 1


# =============================================================================
# FLIP PRIORITY TESTS
# =============================================================================


class TestTFCReevalFlipPriority:
    """Tests for flip blocking priority over strength threshold."""

    def test_flip_blocks_even_with_good_strength(self, daemon):
        """Direction flip should block even if strength is above threshold."""
        signal = create_signal_with_tfc(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            direction="LONG"
        )

        daemon.scanner.evaluate_tfc.return_value = create_mock_tfc_assessment(
            strength=4,  # Excellent strength (above threshold)
            direction="bearish"  # But direction flipped
        )

        should_block, reason = daemon._reevaluate_tfc_at_entry(signal)

        assert should_block is True
        assert "TFC direction flipped" in reason
