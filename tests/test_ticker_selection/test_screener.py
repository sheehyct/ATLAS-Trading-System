"""Unit tests for SnapshotScreener."""

import pytest
from unittest.mock import MagicMock, patch
from strat.ticker_selection.screener import SnapshotScreener, TickerSelectionConfig


def _make_snapshot(close=100.0, high=102.0, low=98.0, volume=500000, prev_close=99.0):
    """Create a mock Alpaca snapshot object."""
    snap = MagicMock()
    snap.daily_bar.close = close
    snap.daily_bar.high = high
    snap.daily_bar.low = low
    snap.daily_bar.open = 99.5
    snap.daily_bar.volume = volume
    snap.previous_daily_bar.close = prev_close
    return snap


class TestScreenerFilters:
    def setup_method(self):
        self.config = TickerSelectionConfig(
            min_price=5.0,
            max_price=500.0,
            min_dollar_volume=10_000_000.0,
            min_atr_percent=1.5,
        )
        self.screener = SnapshotScreener(self.config)

    def test_passes_all_filters(self):
        """Stock with good price, volume, and ATR passes."""
        snap = _make_snapshot(close=100.0, high=104.0, low=97.0, volume=200000)
        result = self.screener._evaluate('TEST', snap)
        assert result is not None
        assert result.symbol == 'TEST'
        assert result.price == 100.0

    def test_rejected_price_too_low(self):
        snap = _make_snapshot(close=2.0, volume=10000000)
        result = self.screener._evaluate('CHEAP', snap)
        assert result is None

    def test_rejected_price_too_high(self):
        snap = _make_snapshot(close=600.0, high=610.0, low=590.0, volume=100000)
        result = self.screener._evaluate('PRICEY', snap)
        assert result is None

    def test_rejected_low_volume(self):
        snap = _make_snapshot(close=50.0, high=52.0, low=48.0, volume=1000)
        result = self.screener._evaluate('ILLIQUID', snap)
        assert result is None

    def test_rejected_low_atr(self):
        """ATR% below threshold gets filtered."""
        snap = _make_snapshot(close=100.0, high=100.3, low=99.8, volume=200000, prev_close=100.0)
        result = self.screener._evaluate('FLAT', snap)
        assert result is None

    def test_atr_percent_calculated(self):
        snap = _make_snapshot(close=100.0, high=103.0, low=97.0, volume=200000, prev_close=100.0)
        result = self.screener._evaluate('VOL', snap)
        assert result is not None
        # True range = max(6, 3, 3) = 6 -> 6% ATR
        assert result.atr_percent >= 1.5


class TestScreenerBatching:
    @patch.object(SnapshotScreener, '_get_client')
    @patch.object(SnapshotScreener, '_fetch_snapshots')
    def test_batches_correctly(self, mock_fetch, mock_client):
        """Verify symbols are batched at 1000 per request."""
        config = TickerSelectionConfig(max_screened=500)
        screener = SnapshotScreener(config)

        # 2500 symbols should result in 3 batches
        symbols = [f'SYM{i}' for i in range(2500)]
        mock_fetch.return_value = {}  # No snapshots returned

        screener.screen(symbols)

        assert mock_fetch.call_count == 3

    @patch.object(SnapshotScreener, '_get_client')
    @patch.object(SnapshotScreener, '_fetch_snapshots')
    def test_caps_at_max_screened(self, mock_fetch, mock_client):
        """Results capped at max_screened."""
        config = TickerSelectionConfig(
            max_screened=2,
            min_price=1.0,
            min_dollar_volume=0,
            min_atr_percent=0,
        )
        screener = SnapshotScreener(config)

        snapshots = {
            'A': _make_snapshot(close=50.0, high=53.0, low=47.0, volume=1000000),
            'B': _make_snapshot(close=60.0, high=64.0, low=56.0, volume=900000),
            'C': _make_snapshot(close=70.0, high=75.0, low=65.0, volume=800000),
        }
        mock_fetch.return_value = snapshots

        result = screener.screen(['A', 'B', 'C'])
        assert len(result) <= 2
