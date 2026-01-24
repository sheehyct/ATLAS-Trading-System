"""
EQUITY-87: Tests for FilterManager coordinator.

Tests signal quality filtering logic extracted from SignalDaemon._passes_filters().
Covers magnitude, R:R, pattern, and TFC filters.
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Optional, List

from strat.signal_automation.coordinators.filter_manager import (
    FilterManager,
    FilterConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockSignalContext:
    """Mock signal context for testing TFC."""
    tfc_score: int = 3
    tfc_alignment: str = "3/4 BULLISH"
    aligned_timeframes: List[str] = None

    def __post_init__(self):
        if self.aligned_timeframes is None:
            self.aligned_timeframes = ['1M', '1W', '1D']


@dataclass
class MockDetectedSignal:
    """Mock detected signal for testing filters."""
    symbol: str = "SPY"
    timeframe: str = "1H"
    pattern_type: str = "3-1-2U"
    direction: str = "CALL"
    magnitude_pct: float = 1.5
    risk_reward: float = 2.0
    signal_type: str = "COMPLETED"
    context: Optional[MockSignalContext] = None

    def __post_init__(self):
        if self.context is None:
            self.context = MockSignalContext()


@pytest.fixture
def filter_manager():
    """Create FilterManager with default config."""
    return FilterManager()


@pytest.fixture
def filter_manager_with_scan_config():
    """Create FilterManager with scan config."""
    scan_config = Mock()
    scan_config.min_magnitude_pct = 0.5
    scan_config.min_risk_reward = 1.0
    scan_config.patterns = ['2-2', '3-2', '3-1-2', '2-1-2']
    return FilterManager(scan_config=scan_config)


@pytest.fixture
def basic_signal():
    """Create a basic signal that passes all filters."""
    return MockDetectedSignal(
        symbol="SPY",
        timeframe="1H",
        pattern_type="3-1-2U",
        direction="CALL",
        magnitude_pct=1.5,
        risk_reward=2.0,
        signal_type="COMPLETED",
        context=MockSignalContext(
            tfc_score=3,
            tfc_alignment="3/4 BULLISH",
            aligned_timeframes=['1M', '1W', '1D'],
        ),
    )


# =============================================================================
# FilterConfig Tests
# =============================================================================


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FilterConfig()
        assert config.setup_min_magnitude == 0.1
        assert config.setup_min_rr == 0.3
        assert config.completed_min_magnitude == 0.5
        assert config.completed_min_rr == 1.0
        assert config.tfc_enabled is True
        assert '1H' in config.tfc_minimums
        assert config.tfc_minimums['1H'] == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FilterConfig(
            setup_min_magnitude=0.2,
            setup_min_rr=0.5,
            completed_min_magnitude=1.0,
            completed_min_rr=2.0,
            tfc_enabled=False,
        )
        assert config.setup_min_magnitude == 0.2
        assert config.setup_min_rr == 0.5
        assert config.completed_min_magnitude == 1.0
        assert config.completed_min_rr == 2.0
        assert config.tfc_enabled is False

    def test_tfc_minimums_defaults(self):
        """Test TFC minimum defaults per timeframe."""
        config = FilterConfig()
        assert config.tfc_minimums['1H'] == 3
        assert config.tfc_minimums['4H'] == 2
        assert config.tfc_minimums['1D'] == 2
        assert config.tfc_minimums['1W'] == 1
        assert config.tfc_minimums['1M'] == 1

    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        with patch.dict('os.environ', {}, clear=True):
            config = FilterConfig.from_env()
            assert config.setup_min_magnitude == 0.1
            assert config.setup_min_rr == 0.3
            assert config.tfc_enabled is True

    def test_from_env_custom_values(self):
        """Test from_env with custom environment variables."""
        env = {
            'SIGNAL_SETUP_MIN_MAGNITUDE': '0.2',
            'SIGNAL_SETUP_MIN_RR': '0.5',
            'SIGNAL_TFC_FILTER_ENABLED': 'false',
        }
        with patch.dict('os.environ', env, clear=True):
            config = FilterConfig.from_env()
            assert config.setup_min_magnitude == 0.2
            assert config.setup_min_rr == 0.5
            assert config.tfc_enabled is False

    def test_allowed_patterns_default_empty(self):
        """Test allowed_patterns defaults to empty list."""
        config = FilterConfig()
        assert config.allowed_patterns == []


# =============================================================================
# FilterManager Initialization Tests
# =============================================================================


class TestFilterManagerInit:
    """Tests for FilterManager initialization."""

    def test_default_init(self):
        """Test initialization with defaults."""
        manager = FilterManager()
        assert manager.config is not None
        assert isinstance(manager.config, FilterConfig)

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = FilterConfig(setup_min_magnitude=0.5)
        manager = FilterManager(config=config)
        assert manager.config.setup_min_magnitude == 0.5

    def test_init_with_scan_config(self):
        """Test initialization with scan config."""
        scan_config = Mock()
        scan_config.min_magnitude_pct = 0.7
        manager = FilterManager(scan_config=scan_config)
        assert manager._scan_config.min_magnitude_pct == 0.7

    def test_update_config(self):
        """Test config update method."""
        manager = FilterManager()
        new_config = FilterConfig(setup_min_magnitude=0.3)
        manager.update_config(new_config)
        assert manager.config.setup_min_magnitude == 0.3


# =============================================================================
# Magnitude Filter Tests
# =============================================================================


class TestMagnitudeFilter:
    """Tests for magnitude threshold filtering."""

    def test_setup_signal_passes_magnitude(self, filter_manager):
        """Test SETUP signal passes with sufficient magnitude."""
        signal = MockDetectedSignal(
            signal_type="SETUP",
            magnitude_pct=0.15,  # > 0.1 default
        )
        assert filter_manager._check_magnitude(signal, "test_key") is True

    def test_setup_signal_fails_magnitude(self, filter_manager):
        """Test SETUP signal fails with low magnitude."""
        signal = MockDetectedSignal(
            signal_type="SETUP",
            magnitude_pct=0.05,  # < 0.1 default
        )
        assert filter_manager._check_magnitude(signal, "test_key") is False

    def test_completed_signal_passes_magnitude(self, filter_manager_with_scan_config):
        """Test COMPLETED signal passes with sufficient magnitude."""
        signal = MockDetectedSignal(
            signal_type="COMPLETED",
            magnitude_pct=0.8,  # > 0.5 from scan_config
        )
        assert filter_manager_with_scan_config._check_magnitude(signal, "test_key") is True

    def test_completed_signal_fails_magnitude(self, filter_manager_with_scan_config):
        """Test COMPLETED signal fails with low magnitude."""
        signal = MockDetectedSignal(
            signal_type="COMPLETED",
            magnitude_pct=0.3,  # < 0.5 from scan_config
        )
        assert filter_manager_with_scan_config._check_magnitude(signal, "test_key") is False

    def test_setup_uses_relaxed_threshold(self, filter_manager):
        """Test SETUP signals use relaxed 0.1% threshold vs COMPLETED 0.5%."""
        # SETUP at 0.2% passes
        setup_signal = MockDetectedSignal(signal_type="SETUP", magnitude_pct=0.2)
        assert filter_manager._check_magnitude(setup_signal, "test_key") is True

        # COMPLETED at 0.2% fails (below 0.5%)
        completed_signal = MockDetectedSignal(signal_type="COMPLETED", magnitude_pct=0.2)
        # Without scan_config, uses default completed_min_magnitude=0.5
        assert filter_manager._check_magnitude(completed_signal, "test_key") is False

    def test_magnitude_boundary_values(self, filter_manager):
        """Test magnitude at exact boundary values."""
        # Exactly at threshold - should pass
        signal = MockDetectedSignal(signal_type="SETUP", magnitude_pct=0.1)
        assert filter_manager._check_magnitude(signal, "test_key") is True

        # Just below threshold - should fail
        signal_below = MockDetectedSignal(signal_type="SETUP", magnitude_pct=0.099)
        assert filter_manager._check_magnitude(signal_below, "test_key") is False


# =============================================================================
# Risk/Reward Filter Tests
# =============================================================================


class TestRiskRewardFilter:
    """Tests for R:R ratio filtering."""

    def test_setup_signal_passes_rr(self, filter_manager):
        """Test SETUP signal passes with sufficient R:R."""
        signal = MockDetectedSignal(
            signal_type="SETUP",
            risk_reward=0.5,  # > 0.3 default
        )
        assert filter_manager._check_rr(signal, "test_key") is True

    def test_setup_signal_fails_rr(self, filter_manager):
        """Test SETUP signal fails with low R:R."""
        signal = MockDetectedSignal(
            signal_type="SETUP",
            risk_reward=0.2,  # < 0.3 default
        )
        assert filter_manager._check_rr(signal, "test_key") is False

    def test_completed_signal_passes_rr(self, filter_manager_with_scan_config):
        """Test COMPLETED signal passes with sufficient R:R."""
        signal = MockDetectedSignal(
            signal_type="COMPLETED",
            risk_reward=1.5,  # > 1.0 from scan_config
        )
        assert filter_manager_with_scan_config._check_rr(signal, "test_key") is True

    def test_completed_signal_fails_rr(self, filter_manager_with_scan_config):
        """Test COMPLETED signal fails with low R:R."""
        signal = MockDetectedSignal(
            signal_type="COMPLETED",
            risk_reward=0.8,  # < 1.0 from scan_config
        )
        assert filter_manager_with_scan_config._check_rr(signal, "test_key") is False

    def test_rr_boundary_values(self, filter_manager):
        """Test R:R at exact boundary values."""
        # Exactly at threshold - should pass
        signal = MockDetectedSignal(signal_type="SETUP", risk_reward=0.3)
        assert filter_manager._check_rr(signal, "test_key") is True

        # Just below threshold - should fail
        signal_below = MockDetectedSignal(signal_type="SETUP", risk_reward=0.29)
        assert filter_manager._check_rr(signal_below, "test_key") is False


# =============================================================================
# Pattern Filter Tests
# =============================================================================


class TestPatternFilter:
    """Tests for pattern type filtering."""

    def test_no_pattern_filter_allows_all(self, filter_manager):
        """Test all patterns pass when no filter configured."""
        signal = MockDetectedSignal(pattern_type="3-1-2U")
        assert filter_manager._check_pattern(signal, "test_key") is True

    def test_pattern_in_allowed_list_passes(self, filter_manager_with_scan_config):
        """Test pattern in allowed list passes."""
        signal = MockDetectedSignal(pattern_type="3-1-2U")  # normalizes to 3-1-2
        assert filter_manager_with_scan_config._check_pattern(signal, "test_key") is True

    def test_pattern_not_in_allowed_list_fails(self, filter_manager_with_scan_config):
        """Test pattern not in allowed list fails."""
        signal = MockDetectedSignal(pattern_type="3-2-2U")  # normalizes to 3-2-2, not in list
        assert filter_manager_with_scan_config._check_pattern(signal, "test_key") is False

    def test_normalize_pattern_2u_2d(self, filter_manager):
        """Test pattern normalization removes 2U/2D."""
        assert filter_manager._normalize_pattern("2U-2D") == "2-2"
        assert filter_manager._normalize_pattern("2D-2U") == "2-2"
        assert filter_manager._normalize_pattern("3-2U") == "3-2"
        assert filter_manager._normalize_pattern("3-2D") == "3-2"

    def test_normalize_pattern_setup_question_mark(self, filter_manager):
        """Test pattern normalization handles SETUP -? patterns."""
        assert filter_manager._normalize_pattern("3-1-?") == "3-1-2"
        assert filter_manager._normalize_pattern("2D-1-?") == "2-1-2"

    def test_normalize_complex_patterns(self, filter_manager):
        """Test normalization of complex patterns."""
        assert filter_manager._normalize_pattern("2U-1-2D") == "2-1-2"
        assert filter_manager._normalize_pattern("3-2U-2D") == "3-2-2"

    def test_allowed_patterns_from_config(self):
        """Test allowed patterns from FilterConfig."""
        config = FilterConfig(allowed_patterns=['2-2', '3-2'])
        manager = FilterManager(config=config)

        signal_pass = MockDetectedSignal(pattern_type="3-2U")  # normalizes to 3-2
        signal_fail = MockDetectedSignal(pattern_type="3-1-2U")  # normalizes to 3-1-2

        assert manager._check_pattern(signal_pass, "test_key") is True
        assert manager._check_pattern(signal_fail, "test_key") is False


# =============================================================================
# TFC Filter Tests
# =============================================================================


class TestTFCFilter:
    """Tests for timeframe continuity filtering."""

    def test_tfc_disabled_passes_all(self):
        """Test all signals pass when TFC disabled."""
        config = FilterConfig(tfc_enabled=False)
        manager = FilterManager(config=config)

        signal = MockDetectedSignal(
            context=MockSignalContext(tfc_score=0),  # Would fail if enabled
        )
        assert manager._check_tfc(signal, "test_key") is True

    def test_1h_tfc_3_passes(self, filter_manager):
        """Test 1H signal with TFC 3/4 passes."""
        signal = MockDetectedSignal(
            timeframe="1H",
            context=MockSignalContext(tfc_score=3, aligned_timeframes=['1M', '1W', '1D']),
        )
        assert filter_manager._check_tfc(signal, "test_key") is True

    def test_1h_tfc_4_passes(self, filter_manager):
        """Test 1H signal with TFC 4/4 passes."""
        signal = MockDetectedSignal(
            timeframe="1H",
            context=MockSignalContext(tfc_score=4, aligned_timeframes=['1M', '1W', '1D', '1H']),
        )
        assert filter_manager._check_tfc(signal, "test_key") is True

    def test_1h_tfc_2_with_1d_aligned_passes(self, filter_manager):
        """Test 1H signal with TFC 2/4 passes when 1D is aligned."""
        signal = MockDetectedSignal(
            timeframe="1H",
            context=MockSignalContext(tfc_score=2, aligned_timeframes=['1D', '1W']),
        )
        assert filter_manager._check_tfc(signal, "test_key") is True

    def test_1h_tfc_2_without_1d_fails(self, filter_manager):
        """Test 1H signal with TFC 2/4 fails when 1D not aligned."""
        signal = MockDetectedSignal(
            timeframe="1H",
            context=MockSignalContext(tfc_score=2, aligned_timeframes=['1M', '1W']),
        )
        assert filter_manager._check_tfc(signal, "test_key") is False

    def test_1h_tfc_1_fails(self, filter_manager):
        """Test 1H signal with TFC 1/4 fails."""
        signal = MockDetectedSignal(
            timeframe="1H",
            context=MockSignalContext(tfc_score=1, aligned_timeframes=['1D']),
        )
        assert filter_manager._check_tfc(signal, "test_key") is False

    def test_1h_tfc_0_fails(self, filter_manager):
        """Test 1H signal with TFC 0/4 fails."""
        signal = MockDetectedSignal(
            timeframe="1H",
            context=MockSignalContext(tfc_score=0, aligned_timeframes=[]),
        )
        assert filter_manager._check_tfc(signal, "test_key") is False

    def test_4h_tfc_minimum_2(self, filter_manager):
        """Test 4H signal requires minimum TFC 2."""
        # Passes with 2
        signal_pass = MockDetectedSignal(
            timeframe="4H",
            context=MockSignalContext(tfc_score=2),
        )
        assert filter_manager._check_tfc(signal_pass, "test_key") is True

        # Fails with 1
        signal_fail = MockDetectedSignal(
            timeframe="4H",
            context=MockSignalContext(tfc_score=1),
        )
        assert filter_manager._check_tfc(signal_fail, "test_key") is False

    def test_1d_tfc_minimum_2(self, filter_manager):
        """Test 1D signal requires minimum TFC 2."""
        # Passes with 2
        signal_pass = MockDetectedSignal(
            timeframe="1D",
            context=MockSignalContext(tfc_score=2),
        )
        assert filter_manager._check_tfc(signal_pass, "test_key") is True

        # Fails with 1
        signal_fail = MockDetectedSignal(
            timeframe="1D",
            context=MockSignalContext(tfc_score=1),
        )
        assert filter_manager._check_tfc(signal_fail, "test_key") is False

    def test_1w_tfc_minimum_1(self, filter_manager):
        """Test 1W signal requires minimum TFC 1."""
        # Passes with 1
        signal_pass = MockDetectedSignal(
            timeframe="1W",
            context=MockSignalContext(tfc_score=1),
        )
        assert filter_manager._check_tfc(signal_pass, "test_key") is True

        # Fails with 0
        signal_fail = MockDetectedSignal(
            timeframe="1W",
            context=MockSignalContext(tfc_score=0),
        )
        assert filter_manager._check_tfc(signal_fail, "test_key") is False

    def test_1m_tfc_minimum_1(self, filter_manager):
        """Test 1M signal requires minimum TFC 1."""
        signal = MockDetectedSignal(
            timeframe="1M",
            context=MockSignalContext(tfc_score=1),
        )
        assert filter_manager._check_tfc(signal, "test_key") is True

    def test_unknown_timeframe_uses_default_2(self, filter_manager):
        """Test unknown timeframe uses default minimum of 2."""
        signal_pass = MockDetectedSignal(
            timeframe="30m",  # Not in TFC minimums
            context=MockSignalContext(tfc_score=2),
        )
        assert filter_manager._check_tfc(signal_pass, "test_key") is True

        signal_fail = MockDetectedSignal(
            timeframe="30m",
            context=MockSignalContext(tfc_score=1),
        )
        assert filter_manager._check_tfc(signal_fail, "test_key") is False

    def test_missing_context_uses_zero_tfc(self, filter_manager):
        """Test signal with no context uses TFC score of 0."""
        signal = MockDetectedSignal(timeframe="1D")
        # Explicitly set context to None after creation to bypass __post_init__
        object.__setattr__(signal, 'context', None)
        assert filter_manager._check_tfc(signal, "test_key") is False


# =============================================================================
# Integration Tests (passes_filters)
# =============================================================================


class TestPassesFilters:
    """Integration tests for complete filter chain."""

    def test_valid_signal_passes_all_filters(self, filter_manager_with_scan_config, basic_signal):
        """Test signal that passes all filters."""
        assert filter_manager_with_scan_config.passes_filters(basic_signal) is True

    def test_low_magnitude_fails(self, filter_manager_with_scan_config, basic_signal):
        """Test signal fails on magnitude."""
        basic_signal.magnitude_pct = 0.2  # Below 0.5
        assert filter_manager_with_scan_config.passes_filters(basic_signal) is False

    def test_low_rr_fails(self, filter_manager_with_scan_config, basic_signal):
        """Test signal fails on R:R."""
        basic_signal.risk_reward = 0.5  # Below 1.0
        assert filter_manager_with_scan_config.passes_filters(basic_signal) is False

    def test_invalid_pattern_fails(self, filter_manager_with_scan_config, basic_signal):
        """Test signal fails on pattern filter."""
        basic_signal.pattern_type = "3-2-2U"  # Not in allowed patterns
        assert filter_manager_with_scan_config.passes_filters(basic_signal) is False

    def test_low_tfc_fails(self, filter_manager_with_scan_config, basic_signal):
        """Test signal fails on TFC filter."""
        basic_signal.context.tfc_score = 1
        basic_signal.context.aligned_timeframes = ['1D']
        assert filter_manager_with_scan_config.passes_filters(basic_signal) is False

    def test_filter_order_magnitude_first(self, filter_manager_with_scan_config):
        """Test filters are checked in order: magnitude, R:R, pattern, TFC."""
        # Create signal that fails on magnitude - should not check other filters
        signal = MockDetectedSignal(
            magnitude_pct=0.1,  # Fails
            risk_reward=0.5,  # Would also fail
            pattern_type="invalid-pattern",  # Would also fail
            signal_type="COMPLETED",
            context=MockSignalContext(tfc_score=0),  # Would also fail
        )
        # All filters fail, but magnitude is checked first
        assert filter_manager_with_scan_config.passes_filters(signal) is False

    def test_setup_signal_uses_relaxed_thresholds(self, filter_manager_with_scan_config):
        """Test SETUP signals use relaxed thresholds."""
        signal = MockDetectedSignal(
            signal_type="SETUP",
            magnitude_pct=0.15,  # Above 0.1 SETUP threshold, below 0.5 COMPLETED
            risk_reward=0.35,  # Above 0.3 SETUP threshold, below 1.0 COMPLETED
            pattern_type="3-1-?",  # SETUP pattern
            context=MockSignalContext(tfc_score=3),
        )
        assert filter_manager_with_scan_config.passes_filters(signal) is True

    def test_completed_signal_uses_strict_thresholds(self, filter_manager_with_scan_config):
        """Test COMPLETED signals use strict thresholds."""
        signal = MockDetectedSignal(
            signal_type="COMPLETED",
            magnitude_pct=0.3,  # Above SETUP 0.1, below COMPLETED 0.5
            risk_reward=1.5,
            pattern_type="3-1-2U",
            context=MockSignalContext(tfc_score=3),
        )
        # Fails due to strict magnitude threshold for COMPLETED
        assert filter_manager_with_scan_config.passes_filters(signal) is False


# =============================================================================
# is_setup_signal Tests
# =============================================================================


class TestIsSetupSignal:
    """Tests for _is_setup_signal helper."""

    def test_setup_signal_type(self, filter_manager):
        """Test signal with SETUP type returns True."""
        signal = MockDetectedSignal(signal_type="SETUP")
        assert filter_manager._is_setup_signal(signal) is True

    def test_completed_signal_type(self, filter_manager):
        """Test signal with COMPLETED type returns False."""
        signal = MockDetectedSignal(signal_type="COMPLETED")
        assert filter_manager._is_setup_signal(signal) is False

    def test_missing_signal_type_defaults_completed(self, filter_manager):
        """Test signal without signal_type defaults to COMPLETED (False)."""
        signal = Mock()
        del signal.signal_type  # Remove attribute
        # getattr with default should return 'COMPLETED'
        assert filter_manager._is_setup_signal(signal) is False


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_magnitude(self, filter_manager):
        """Test signal with negative magnitude fails."""
        signal = MockDetectedSignal(signal_type="SETUP", magnitude_pct=-0.5)
        assert filter_manager._check_magnitude(signal, "test_key") is False

    def test_zero_magnitude(self, filter_manager):
        """Test signal with zero magnitude fails."""
        signal = MockDetectedSignal(signal_type="SETUP", magnitude_pct=0.0)
        assert filter_manager._check_magnitude(signal, "test_key") is False

    def test_negative_rr(self, filter_manager):
        """Test signal with negative R:R fails."""
        signal = MockDetectedSignal(signal_type="SETUP", risk_reward=-1.0)
        assert filter_manager._check_rr(signal, "test_key") is False

    def test_zero_rr(self, filter_manager):
        """Test signal with zero R:R fails."""
        signal = MockDetectedSignal(signal_type="SETUP", risk_reward=0.0)
        assert filter_manager._check_rr(signal, "test_key") is False

    def test_empty_pattern_type(self, filter_manager):
        """Test signal with empty pattern type."""
        signal = MockDetectedSignal(pattern_type="")
        # With no pattern filter configured, should pass
        assert filter_manager._check_pattern(signal, "test_key") is True

    def test_context_with_missing_attributes(self, filter_manager):
        """Test signal context with missing TFC attributes."""
        # Create context without tfc_score
        context = Mock()
        del context.tfc_score  # Remove attribute to test getattr default
        signal = MockDetectedSignal(timeframe="1D")
        signal.context = context
        # Should use default TFC score of 0 and fail
        assert filter_manager._check_tfc(signal, "test_key") is False

    def test_high_values_pass(self, filter_manager_with_scan_config):
        """Test signal with very high values passes."""
        signal = MockDetectedSignal(
            magnitude_pct=100.0,
            risk_reward=10.0,
            context=MockSignalContext(tfc_score=4),
        )
        assert filter_manager_with_scan_config.passes_filters(signal) is True
