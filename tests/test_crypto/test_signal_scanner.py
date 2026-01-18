"""
Tests for crypto/scanning/signal_scanner.py

Comprehensive tests for CryptoSignalScanner covering:
- Maintenance window handling
- ATR and volume calculations
- Bar sequence formatting
- Pattern detection
- Setup detection
- TFC evaluation
- Public scanning methods
- Output formatting
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from crypto.scanning.signal_scanner import CryptoSignalScanner, scan_crypto
from crypto.scanning.models import CryptoSignalContext, CryptoDetectedSignal


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_coinbase_client():
    """Create a mock CoinbaseClient for testing without API calls."""
    client = Mock()
    client.get_historical_ohlcv = Mock(return_value=None)
    return client


@pytest.fixture
def scanner(mock_coinbase_client):
    """Create a CryptoSignalScanner with mocked client."""
    return CryptoSignalScanner(client=mock_coinbase_client)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2025-01-01", periods=50, freq="1h", tz="UTC")
    np.random.seed(42)

    # Create realistic price data
    base_price = 100000.0
    returns = np.random.randn(50) * 0.01
    closes = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": closes * (1 + np.random.randn(50) * 0.002),
        "High": closes * (1 + abs(np.random.randn(50) * 0.005)),
        "Low": closes * (1 - abs(np.random.randn(50) * 0.005)),
        "Close": closes,
        "Volume": np.random.randint(100, 1000, 50).astype(float),
    }, index=dates)

    # Add maintenance gap column (required by scanner)
    df["is_maintenance_gap"] = False

    return df


@pytest.fixture
def sample_ohlcv_with_patterns():
    """Create OHLCV data with known STRAT patterns."""
    dates = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")

    # Create a 2-1-2 bullish pattern at index 4-6
    # Bar 4: Type 2U (higher high, higher low than bar 3)
    # Bar 5: Type 1 (inside bar)
    # Bar 6: Type 2U (break above bar 5 high)

    df = pd.DataFrame({
        "Open":  [100, 101, 102, 103, 104, 104.5, 105.5, 106, 107, 108],
        "High":  [101, 102, 103, 105, 106, 105.5, 107,   108, 109, 110],
        "Low":   [99,  100, 101, 102, 103, 104,   105,   105, 106, 107],
        "Close": [100, 101, 102, 104, 105, 105,   106,   107, 108, 109],
        "Volume": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    }, index=dates)

    df["is_maintenance_gap"] = False

    return df


# =============================================================================
# MAINTENANCE WINDOW TESTS
# =============================================================================


class TestMaintenanceWindow:
    """Tests for maintenance window detection."""

    def test_maintenance_window_friday_during_window(self, scanner):
        """Friday 22:00-23:00 UTC should be maintenance window."""
        # Friday 22:30 UTC
        dt = datetime(2025, 1, 17, 22, 30, tzinfo=timezone.utc)  # Friday
        assert dt.weekday() == 4  # Verify it's Friday
        assert scanner.is_maintenance_window(dt) is True

    def test_maintenance_window_friday_before_window(self, scanner):
        """Friday before 22:00 UTC is not maintenance."""
        dt = datetime(2025, 1, 17, 21, 59, tzinfo=timezone.utc)  # Friday
        assert scanner.is_maintenance_window(dt) is False

    def test_maintenance_window_friday_after_window(self, scanner):
        """Friday after 23:00 UTC is not maintenance."""
        dt = datetime(2025, 1, 17, 23, 1, tzinfo=timezone.utc)  # Friday
        assert scanner.is_maintenance_window(dt) is False

    def test_maintenance_window_not_friday(self, scanner):
        """Non-Friday days are never maintenance."""
        # Monday 22:30 UTC
        dt = datetime(2025, 1, 13, 22, 30, tzinfo=timezone.utc)  # Monday
        assert dt.weekday() == 0
        assert scanner.is_maintenance_window(dt) is False

        # Saturday 22:30 UTC
        dt = datetime(2025, 1, 18, 22, 30, tzinfo=timezone.utc)  # Saturday
        assert dt.weekday() == 5
        assert scanner.is_maintenance_window(dt) is False

    def test_maintenance_window_naive_datetime(self, scanner):
        """Naive datetime should be treated as UTC."""
        # Friday 22:30 (naive)
        dt = datetime(2025, 1, 17, 22, 30)
        assert scanner.is_maintenance_window(dt) is True

    def test_maintenance_window_none_uses_now(self, scanner):
        """None argument should use current time."""
        # This test verifies the method doesn't crash with None
        result = scanner.is_maintenance_window(None)
        assert isinstance(result, bool)

    @patch("crypto.config.MAINTENANCE_WINDOW_ENABLED", False)
    def test_maintenance_window_disabled(self, mock_coinbase_client):
        """When disabled, maintenance window should never trigger."""
        scanner = CryptoSignalScanner(client=mock_coinbase_client)
        dt = datetime(2025, 1, 17, 22, 30, tzinfo=timezone.utc)  # Friday
        assert scanner.is_maintenance_window(dt) is False


class TestOverlapsMaintenance:
    """Tests for bar overlap with maintenance window."""

    def test_overlaps_15m_bar_in_window(self, scanner):
        """15m bar starting in maintenance window overlaps."""
        bar_start = datetime(2025, 1, 17, 22, 30, tzinfo=timezone.utc)  # Friday
        assert scanner._overlaps_maintenance(bar_start, "15m") is True

    def test_overlaps_1h_bar_starting_before(self, scanner):
        """1h bar starting before window but ending in window overlaps."""
        bar_start = datetime(2025, 1, 17, 21, 30, tzinfo=timezone.utc)  # Friday
        # Bar ends at 22:30 UTC which is in window
        assert scanner._overlaps_maintenance(bar_start, "1h") is True

    def test_overlaps_4h_bar_starting_during_window(self, scanner):
        """4h bar starting during maintenance window overlaps."""
        bar_start = datetime(2025, 1, 17, 22, 0, tzinfo=timezone.utc)  # Friday
        # Bar starts at 22:00 which is in the window
        assert scanner._overlaps_maintenance(bar_start, "4h") is True

    def test_no_overlap_before_window(self, scanner):
        """Bar ending before maintenance window doesn't overlap."""
        bar_start = datetime(2025, 1, 17, 19, 0, tzinfo=timezone.utc)  # Friday
        # 1h bar ends at 20:00, before 22:00 window start
        assert scanner._overlaps_maintenance(bar_start, "1h") is False

    def test_no_overlap_after_window(self, scanner):
        """Bar starting after maintenance window doesn't overlap."""
        bar_start = datetime(2025, 1, 17, 23, 30, tzinfo=timezone.utc)  # Friday
        assert scanner._overlaps_maintenance(bar_start, "15m") is False

    def test_no_overlap_different_day(self, scanner):
        """Bar on non-Friday doesn't overlap."""
        bar_start = datetime(2025, 1, 16, 22, 30, tzinfo=timezone.utc)  # Thursday
        assert scanner._overlaps_maintenance(bar_start, "1h") is False


# =============================================================================
# ATR AND VOLUME CALCULATION TESTS
# =============================================================================


class TestATRCalculation:
    """Tests for _calculate_atr method."""

    def test_atr_with_valid_data(self, scanner, sample_ohlcv_df):
        """ATR calculation with valid OHLCV data."""
        atr = scanner._calculate_atr(sample_ohlcv_df, period=14)
        assert atr > 0
        assert isinstance(atr, float)

    def test_atr_with_insufficient_data(self, scanner):
        """ATR returns 0 with insufficient bars."""
        dates = pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103],
            "Close": [101, 102, 103, 104, 105],
        }, index=dates)

        atr = scanner._calculate_atr(df, period=14)
        assert atr == 0.0

    def test_atr_calculation_accuracy(self, scanner):
        """Verify ATR calculation is correct."""
        # Create predictable data
        dates = pd.date_range("2025-01-01", periods=20, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100] * 20,
            "High": [105] * 20,  # Range of 5
            "Low": [95] * 20,
            "Close": [100] * 20,
        }, index=dates)

        atr = scanner._calculate_atr(df, period=14)
        # True range = High - Low = 10 for all bars
        assert abs(atr - 10.0) < 0.1


class TestVolumeRatioCalculation:
    """Tests for _calculate_volume_ratio method."""

    def test_volume_ratio_with_valid_data(self, scanner, sample_ohlcv_df):
        """Volume ratio calculation with valid data."""
        ratio = scanner._calculate_volume_ratio(sample_ohlcv_df)
        assert ratio > 0
        assert isinstance(ratio, float)

    def test_volume_ratio_insufficient_data(self, scanner):
        """Volume ratio returns 1.0 with insufficient bars."""
        dates = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100] * 10,
            "High": [105] * 10,
            "Low": [95] * 10,
            "Close": [100] * 10,
            "Volume": [100] * 10,
        }, index=dates)

        ratio = scanner._calculate_volume_ratio(df)
        assert ratio == 1.0

    def test_volume_ratio_double_average(self, scanner):
        """Current volume double the average should give ratio ~2."""
        dates = pd.date_range("2025-01-01", periods=25, freq="1h", tz="UTC")
        volumes = [100] * 24 + [200]  # Last bar has double volume
        df = pd.DataFrame({
            "Open": [100] * 25,
            "High": [105] * 25,
            "Low": [95] * 25,
            "Close": [100] * 25,
            "Volume": volumes,
        }, index=dates)

        ratio = scanner._calculate_volume_ratio(df)
        assert abs(ratio - 2.0) < 0.1

    def test_volume_ratio_zero_average(self, scanner):
        """Zero average volume should return 1.0."""
        dates = pd.date_range("2025-01-01", periods=25, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100] * 25,
            "High": [105] * 25,
            "Low": [95] * 25,
            "Close": [100] * 25,
            "Volume": [0] * 24 + [100],  # All zeros except last
        }, index=dates)

        ratio = scanner._calculate_volume_ratio(df)
        assert ratio == 1.0

    def test_volume_ratio_no_volume_column(self, scanner):
        """Missing Volume column should return 1.0."""
        dates = pd.date_range("2025-01-01", periods=25, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100] * 25,
            "High": [105] * 25,
            "Low": [95] * 25,
            "Close": [100] * 25,
        }, index=dates)

        ratio = scanner._calculate_volume_ratio(df)
        assert ratio == 1.0


class TestGetMarketContext:
    """Tests for _get_market_context method."""

    def test_market_context_returns_crypto_signal_context(self, scanner, sample_ohlcv_df):
        """Market context should return CryptoSignalContext."""
        context = scanner._get_market_context(sample_ohlcv_df)
        assert isinstance(context, CryptoSignalContext)

    def test_market_context_fields_populated(self, scanner, sample_ohlcv_df):
        """Market context should populate ATR and volume fields."""
        context = scanner._get_market_context(sample_ohlcv_df)
        assert context.atr_14 > 0
        assert context.atr_percent > 0
        assert context.volume_ratio > 0

    def test_market_context_atr_percent_calculation(self, scanner):
        """ATR percent should be ATR / price * 100."""
        dates = pd.date_range("2025-01-01", periods=20, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100] * 20,
            "High": [105] * 20,
            "Low": [95] * 20,
            "Close": [100] * 20,
            "Volume": [100] * 20,
        }, index=dates)
        df["is_maintenance_gap"] = False

        context = scanner._get_market_context(df)
        expected_atr_pct = context.atr_14 / 100 * 100  # ATR / Close * 100
        assert abs(context.atr_percent - expected_atr_pct) < 0.01

    def test_market_context_empty_df(self, scanner):
        """Empty DataFrame should return context with zero values."""
        df = pd.DataFrame()
        context = scanner._get_market_context(df)
        assert context.atr_14 == 0.0


# =============================================================================
# BAR SEQUENCE FORMATTING TESTS
# =============================================================================


class TestGetBarSequence:
    """Tests for _get_bar_sequence method."""

    def test_bar_sequence_22_pattern(self, scanner):
        """2-2 pattern bar sequence formatting."""
        # Classifications: Type 1=1, Type 2U=2, Type 2D=-2, Type 3=3
        classifications = np.array([1, 2, -2, 2, 1])

        # 2U-2D at index 2
        seq = scanner._get_bar_sequence("2-2", classifications, 2, -1)
        assert seq == "2U-2D"

        # 2D-2U at index 3
        seq = scanner._get_bar_sequence("2-2", classifications, 3, 1)
        assert seq == "2D-2U"

    def test_bar_sequence_32_pattern(self, scanner):
        """3-2 pattern bar sequence formatting."""
        classifications = np.array([1, 3, 2, 1, -2])

        # 3-2U at index 2
        seq = scanner._get_bar_sequence("3-2", classifications, 2, 1)
        assert seq == "3-2U"

        # 3-2D at index 4 (if bar 3 was outside)
        classifications2 = np.array([1, 1, 1, 3, -2])
        seq = scanner._get_bar_sequence("3-2", classifications2, 4, -1)
        assert seq == "3-2D"

    def test_bar_sequence_322_pattern(self, scanner):
        """3-2-2 pattern bar sequence formatting."""
        classifications = np.array([1, 3, 2, -2, 1])

        # 3-2U-2D at index 3
        seq = scanner._get_bar_sequence("3-2-2", classifications, 3, -1)
        assert seq == "3-2U-2D"

    def test_bar_sequence_212_pattern(self, scanner):
        """2-1-2 pattern bar sequence formatting."""
        classifications = np.array([1, 2, 1, 2, 1])

        # 2U-1-2U at index 3
        seq = scanner._get_bar_sequence("2-1-2", classifications, 3, 1)
        assert seq == "2U-1-2U"

    def test_bar_sequence_312_pattern(self, scanner):
        """3-1-2 pattern bar sequence formatting."""
        classifications = np.array([1, 3, 1, 2, 1])

        # 3-1-2U at index 3
        seq = scanner._get_bar_sequence("3-1-2", classifications, 3, 1)
        assert seq == "3-1-2U"

    def test_bar_sequence_insufficient_index(self, scanner):
        """Insufficient index should return simplified sequence."""
        classifications = np.array([2, 1])

        # Index 1, need 3 bars for 2-1-2
        seq = scanner._get_bar_sequence("2-1-2", classifications, 1, 1)
        assert seq == "2-1-2U"  # Fallback format

    def test_bar_sequence_unknown_pattern(self, scanner):
        """Unknown pattern type should return basic format."""
        classifications = np.array([1, 2, 3])
        seq = scanner._get_bar_sequence("unknown", classifications, 2, 1)
        assert seq == "unknownU"


# =============================================================================
# PATTERN DETECTION TESTS
# =============================================================================


class TestDetectPatterns:
    """Tests for _detect_patterns method."""

    def test_detect_patterns_empty_df(self, scanner):
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame()
        patterns = scanner._detect_patterns(df, "2-2")
        assert patterns == []

    def test_detect_patterns_insufficient_bars(self, scanner):
        """Fewer than 5 bars returns empty list."""
        dates = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [102, 103, 104],
            "Low": [99, 100, 101],
            "Close": [101, 102, 103],
            "Volume": [100, 100, 100],
        }, index=dates)
        df["is_maintenance_gap"] = False

        patterns = scanner._detect_patterns(df, "2-2")
        assert patterns == []

    def test_detect_patterns_with_maintenance_gap(self, scanner):
        """Patterns near maintenance gaps should be skipped."""
        dates = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100] * 10,
            "High": [105] * 10,
            "Low": [95] * 10,
            "Close": [100] * 10,
            "Volume": [100] * 10,
        }, index=dates)
        # Set last 3 bars as maintenance gaps
        df["is_maintenance_gap"] = [False] * 7 + [True] * 3

        patterns = scanner._detect_patterns(df, "2-2")
        # Should return empty due to recent maintenance gap
        assert patterns == []

    def test_detect_patterns_returns_dict_list(self, scanner, sample_ohlcv_df):
        """Detected patterns should be list of dicts."""
        patterns = scanner._detect_patterns(sample_ohlcv_df, "2-2")
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, dict)

    def test_detect_patterns_dict_has_required_fields(self, scanner, sample_ohlcv_df):
        """Each pattern dict should have required fields."""
        patterns = scanner._detect_patterns(sample_ohlcv_df, "2-2")
        required_fields = [
            "index", "timestamp", "signal", "direction",
            "entry", "stop", "target", "magnitude_pct",
            "risk_reward", "bar_sequence", "signal_type",
            "setup_bar_high", "setup_bar_low", "setup_bar_timestamp",
            "has_maintenance_gap"
        ]

        for p in patterns:
            for field in required_fields:
                assert field in p, f"Missing field: {field}"

    def test_detect_patterns_invalid_type(self, scanner, sample_ohlcv_df):
        """Invalid pattern type returns empty list."""
        patterns = scanner._detect_patterns(sample_ohlcv_df, "invalid")
        assert patterns == []

    def test_detect_patterns_all_types(self, scanner, sample_ohlcv_df):
        """All pattern types should be detectable."""
        for pattern_type in ["2-2", "3-2", "3-2-2", "2-1-2", "3-1-2"]:
            patterns = scanner._detect_patterns(sample_ohlcv_df, pattern_type)
            assert isinstance(patterns, list)


# =============================================================================
# SETUP DETECTION TESTS
# =============================================================================


class TestDetectSetups:
    """Tests for _detect_setups method."""

    def test_detect_setups_empty_df(self, scanner):
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame()
        setups = scanner._detect_setups(df)
        assert setups == []

    def test_detect_setups_insufficient_bars(self, scanner):
        """Fewer than 3 bars returns empty list."""
        dates = pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [100, 100],
        }, index=dates)
        df["is_maintenance_gap"] = False

        setups = scanner._detect_setups(df)
        assert setups == []

    def test_detect_setups_returns_dict_list(self, scanner, sample_ohlcv_df):
        """Detected setups should be list of dicts."""
        setups = scanner._detect_setups(sample_ohlcv_df)
        assert isinstance(setups, list)
        for s in setups:
            assert isinstance(s, dict)

    def test_detect_setups_has_required_fields(self, scanner, sample_ohlcv_df):
        """Each setup dict should have required fields."""
        setups = scanner._detect_setups(sample_ohlcv_df)
        required_fields = [
            "index", "timestamp", "signal", "direction",
            "entry", "stop", "target", "magnitude_pct",
            "risk_reward", "bar_sequence", "signal_type",
            "setup_bar_high", "setup_bar_low", "setup_bar_timestamp",
            "setup_pattern", "has_maintenance_gap",
            "prior_bar_type", "prior_bar_high", "prior_bar_low"
        ]

        for s in setups:
            for field in required_fields:
                assert field in s, f"Missing field: {field}"

    def test_detect_setups_signal_type_is_setup(self, scanner, sample_ohlcv_df):
        """All detected setups should have signal_type='SETUP'."""
        setups = scanner._detect_setups(sample_ohlcv_df)
        for s in setups:
            assert s["signal_type"] == "SETUP"

    def test_detect_setups_excludes_last_bar_for_inside(self, scanner):
        """Last bar should not be detected as inside bar setup."""
        # Create data where last bar is Type 1 (inside)
        dates = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            # Bar 8: Type 3 (outside), Bar 9: Type 1 (inside - last bar)
            "Open":  [100, 100, 100, 100, 100, 100, 100, 100, 100, 100.5],
            "High":  [101, 101, 101, 101, 101, 101, 101, 101, 105, 102],  # Bar 9 inside bar 8
            "Low":   [99,  99,  99,  99,  99,  99,  99,  99,  96,  99],
            "Close": [100, 100, 100, 100, 100, 100, 100, 100, 100, 101],
            "Volume": [100] * 10,
        }, index=dates)
        df["is_maintenance_gap"] = False

        setups = scanner._detect_setups(df)
        # Should not find a 3-1-? setup at the last bar index (9)
        for s in setups:
            if s["setup_pattern"] in ["3-1-2", "2-1-2"]:
                assert s["index"] != 9, "Last bar should not be detected as inside bar setup"


# =============================================================================
# PUBLIC SCANNING METHOD TESTS
# =============================================================================


class TestScanSymbolTimeframe:
    """Tests for scan_symbol_timeframe method."""

    def test_scan_returns_list(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """scan_symbol_timeframe should return a list."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        signals = scanner.scan_symbol_timeframe("BTC-PERP-INTX", "1h")
        assert isinstance(signals, list)

    def test_scan_returns_crypto_detected_signals(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """scan_symbol_timeframe should return CryptoDetectedSignal objects."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        signals = scanner.scan_symbol_timeframe("BTC-PERP-INTX", "1h")
        for s in signals:
            assert isinstance(s, CryptoDetectedSignal)

    def test_scan_empty_data_returns_empty_list(self, scanner, mock_coinbase_client):
        """Empty data should return empty list."""
        mock_coinbase_client.get_historical_ohlcv.return_value = pd.DataFrame()
        signals = scanner.scan_symbol_timeframe("BTC-PERP-INTX", "1h")
        assert signals == []

    def test_scan_none_data_returns_empty_list(self, scanner, mock_coinbase_client):
        """None data should return empty list."""
        mock_coinbase_client.get_historical_ohlcv.return_value = None
        signals = scanner.scan_symbol_timeframe("BTC-PERP-INTX", "1h")
        assert signals == []

    def test_scan_signal_has_tfc_context(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """Signals should have TFC context populated."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        signals = scanner.scan_symbol_timeframe("BTC-PERP-INTX", "1h")
        for s in signals:
            assert hasattr(s.context, "tfc_score")
            assert hasattr(s.context, "tfc_alignment")
            assert hasattr(s.context, "tfc_passes")


class TestScanAllTimeframes:
    """Tests for scan_all_timeframes method."""

    def test_scan_all_timeframes_returns_list(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """scan_all_timeframes should return a list."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        signals = scanner.scan_all_timeframes("BTC-PERP-INTX")
        assert isinstance(signals, list)

    def test_scan_all_timeframes_calls_each_tf(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """scan_all_timeframes should scan each default timeframe."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner.scan_all_timeframes("BTC-PERP-INTX")

        # Should be called once per timeframe (4 timeframes: 1w, 1d, 4h, 1h)
        # Plus TFC evaluation calls for each detected signal
        assert mock_coinbase_client.get_historical_ohlcv.call_count >= 4


class TestScanAllSymbols:
    """Tests for scan_all_symbols method."""

    def test_scan_all_symbols_returns_dict(self, scanner, mock_coinbase_client, sample_ohlcv_df, capsys):
        """scan_all_symbols should return a dict."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        results = scanner.scan_all_symbols()
        assert isinstance(results, dict)

    def test_scan_all_symbols_keys_are_symbols(self, scanner, mock_coinbase_client, sample_ohlcv_df, capsys):
        """Dict keys should be trading symbols."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        results = scanner.scan_all_symbols()
        for key in results:
            assert isinstance(key, str)


# =============================================================================
# TFC EVALUATION TESTS
# =============================================================================


class TestEvaluateTFC:
    """Tests for evaluate_tfc method."""

    def test_evaluate_tfc_returns_assessment(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """evaluate_tfc should return TFC assessment."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        assessment = scanner.evaluate_tfc("BTC-PERP-INTX", "1h", 1)

        # Should have standard assessment attributes
        assert hasattr(assessment, "strength")
        assert hasattr(assessment, "passes_flexible")

    def test_evaluate_tfc_bullish_direction(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """evaluate_tfc with direction=1 should use 'bullish'."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        assessment = scanner.evaluate_tfc("BTC-PERP-INTX", "1h", 1)
        assert isinstance(assessment.strength, int)

    def test_evaluate_tfc_bearish_direction(self, scanner, mock_coinbase_client, sample_ohlcv_df):
        """evaluate_tfc with direction=-1 should use 'bearish'."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        assessment = scanner.evaluate_tfc("BTC-PERP-INTX", "1h", -1)
        assert isinstance(assessment.strength, int)


# =============================================================================
# OUTPUT METHOD TESTS
# =============================================================================


class TestPrintSignals:
    """Tests for print_signals method."""

    def test_print_no_signals(self, scanner, capsys):
        """Empty signals should print 'No signals detected'."""
        scanner.print_signals([])
        captured = capsys.readouterr()
        assert "No signals detected" in captured.out

    def test_print_signals_header(self, scanner, capsys):
        """Print should include header."""
        signal = CryptoDetectedSignal(
            pattern_type="3-1-2U",
            direction="LONG",
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=100000.0,
            stop_price=99000.0,
            target_price=101500.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
        )
        scanner.print_signals([signal])
        captured = capsys.readouterr()
        assert "CRYPTO STRAT SIGNALS" in captured.out

    def test_print_signals_shows_pattern_info(self, scanner, capsys):
        """Print should show pattern details."""
        signal = CryptoDetectedSignal(
            pattern_type="3-1-2U",
            direction="LONG",
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=100000.0,
            stop_price=99000.0,
            target_price=101500.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
        )
        scanner.print_signals([signal])
        captured = capsys.readouterr()
        assert "3-1-2U" in captured.out
        assert "LONG" in captured.out
        assert "BTC-PERP-INTX" in captured.out

    def test_print_signals_maintenance_gap_warning(self, scanner, capsys):
        """Print should show maintenance gap warning."""
        signal = CryptoDetectedSignal(
            pattern_type="3-1-2U",
            direction="LONG",
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=100000.0,
            stop_price=99000.0,
            target_price=101500.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
            has_maintenance_gap=True,
        )
        scanner.print_signals([signal])
        captured = capsys.readouterr()
        assert "MAINT GAP" in captured.out


class TestToDataFrame:
    """Tests for to_dataframe method."""

    def test_to_dataframe_empty_signals(self, scanner):
        """Empty signals should return empty DataFrame."""
        df = scanner.to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_to_dataframe_returns_dataframe(self, scanner):
        """to_dataframe should return pandas DataFrame."""
        signal = CryptoDetectedSignal(
            pattern_type="3-1-2U",
            direction="LONG",
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=100000.0,
            stop_price=99000.0,
            target_price=101500.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
        )
        df = scanner.to_dataframe([signal])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_to_dataframe_has_required_columns(self, scanner):
        """DataFrame should have required columns."""
        signal = CryptoDetectedSignal(
            pattern_type="3-1-2U",
            direction="LONG",
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=100000.0,
            stop_price=99000.0,
            target_price=101500.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
        )
        df = scanner.to_dataframe([signal])

        expected_columns = [
            "pattern_type", "direction", "symbol", "timeframe",
            "signal_type", "detected_time", "entry_trigger",
            "stop_price", "target_price", "magnitude_pct",
            "risk_reward", "atr_14", "volume_ratio", "has_maintenance_gap"
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_to_dataframe_multiple_signals(self, scanner):
        """Multiple signals should create multiple rows."""
        signals = [
            CryptoDetectedSignal(
                pattern_type="3-1-2U",
                direction="LONG",
                symbol="BTC-PERP-INTX",
                timeframe="1h",
                detected_time=datetime.now(timezone.utc),
                entry_trigger=100000.0,
                stop_price=99000.0,
                target_price=101500.0,
                magnitude_pct=1.5,
                risk_reward=1.5,
            ),
            CryptoDetectedSignal(
                pattern_type="2D-2U",
                direction="LONG",
                symbol="ETH-PERP-INTX",
                timeframe="4h",
                detected_time=datetime.now(timezone.utc),
                entry_trigger=3500.0,
                stop_price=3400.0,
                target_price=3650.0,
                magnitude_pct=4.3,
                risk_reward=1.5,
            ),
        ]
        df = scanner.to_dataframe(signals)
        assert len(df) == 2


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestScanCryptoFunction:
    """Tests for scan_crypto convenience function."""

    @patch.object(CryptoSignalScanner, "scan_symbol_timeframe")
    def test_scan_crypto_default_params(self, mock_scan):
        """scan_crypto with defaults should scan all symbols and timeframes."""
        mock_scan.return_value = []
        signals = scan_crypto()
        assert isinstance(signals, list)

    @patch.object(CryptoSignalScanner, "scan_symbol_timeframe")
    def test_scan_crypto_custom_symbols(self, mock_scan):
        """scan_crypto should accept custom symbols."""
        mock_scan.return_value = []
        signals = scan_crypto(symbols=["BTC-PERP-INTX"])
        assert isinstance(signals, list)

    @patch.object(CryptoSignalScanner, "scan_symbol_timeframe")
    def test_scan_crypto_custom_timeframes(self, mock_scan):
        """scan_crypto should accept custom timeframes."""
        mock_scan.return_value = []
        signals = scan_crypto(timeframes=["1h", "4h"])
        assert isinstance(signals, list)


# =============================================================================
# DATA FETCHING TESTS
# =============================================================================


class TestFetchData:
    """Tests for _fetch_data method."""

    def test_fetch_data_returns_none_on_empty(self, scanner, mock_coinbase_client):
        """Empty data should return None."""
        mock_coinbase_client.get_historical_ohlcv.return_value = pd.DataFrame()
        result = scanner._fetch_data("BTC-PERP-INTX", "1h")
        assert result is None

    def test_fetch_data_returns_none_on_none(self, scanner, mock_coinbase_client):
        """None data should return None."""
        mock_coinbase_client.get_historical_ohlcv.return_value = None
        result = scanner._fetch_data("BTC-PERP-INTX", "1h")
        assert result is None

    def test_fetch_data_standardizes_columns(self, scanner, mock_coinbase_client):
        """Column names should be standardized to capitalized."""
        dates = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": [100] * 10,
            "high": [105] * 10,
            "low": [95] * 10,
            "close": [100] * 10,
            "volume": [100] * 10,
        }, index=dates)
        mock_coinbase_client.get_historical_ohlcv.return_value = df

        result = scanner._fetch_data("BTC-PERP-INTX", "1h")
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns

    def test_fetch_data_adds_maintenance_gap_column(self, scanner, mock_coinbase_client):
        """Result should have is_maintenance_gap column."""
        dates = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "Open": [100] * 10,
            "High": [105] * 10,
            "Low": [95] * 10,
            "Close": [100] * 10,
            "Volume": [100] * 10,
        }, index=dates)
        mock_coinbase_client.get_historical_ohlcv.return_value = df

        result = scanner._fetch_data("BTC-PERP-INTX", "1h")
        assert "is_maintenance_gap" in result.columns

    def test_fetch_data_handles_exception(self, scanner, mock_coinbase_client):
        """Exception during fetch should return None."""
        mock_coinbase_client.get_historical_ohlcv.side_effect = Exception("API Error")
        result = scanner._fetch_data("BTC-PERP-INTX", "1h")
        assert result is None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_scanner_initialization_with_none_client(self):
        """Scanner should create client if None provided."""
        with patch("crypto.scanning.signal_scanner.CoinbaseClient") as mock_client_class:
            mock_client_class.return_value = Mock()
            scanner = CryptoSignalScanner(client=None)
            mock_client_class.assert_called_once_with(simulation_mode=True)

    def test_scanner_has_tfc_adapter(self, scanner):
        """Scanner should have TFC adapter initialized."""
        assert hasattr(scanner, "tfc_adapter")
        assert scanner.tfc_adapter is not None

    def test_scanner_default_timeframes(self, scanner):
        """Scanner should have default timeframes."""
        assert len(scanner.DEFAULT_TIMEFRAMES) > 0
        assert "1h" in scanner.DEFAULT_TIMEFRAMES

    def test_scanner_default_symbols(self, scanner):
        """Scanner should have default symbols."""
        assert len(scanner.DEFAULT_SYMBOLS) > 0

    def test_scanner_all_patterns(self, scanner):
        """Scanner should have all pattern types."""
        expected_patterns = ["2-2", "3-2", "3-2-2", "2-1-2", "3-1-2"]
        for pattern in expected_patterns:
            assert pattern in scanner.ALL_PATTERNS
