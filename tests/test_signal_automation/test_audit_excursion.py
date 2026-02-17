"""
Tests for EQUITY-112: MFE/MAE excursion stats in daily trade audit.

Tests the _get_today_excursion_stats() method in daemon.py and the
MFE/MAE + Capital fields in discord_alerter.send_daily_audit().
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from core.trade_analytics.models import ExcursionData, EnrichedTradeRecord


# =========================================================================
# Fixtures
# =========================================================================

def _make_trade(
    pnl: float,
    mfe_pnl: float,
    mae_pnl: float,
    exit_efficiency: float = 0.0,
    went_green: bool = False,
    exit_time: datetime = None,
) -> EnrichedTradeRecord:
    """Create a minimal EnrichedTradeRecord for testing."""
    trade = EnrichedTradeRecord(
        trade_id=f"test_{abs(hash((pnl, mfe_pnl)))}" ,
        symbol="SPY",
        pnl=pnl,
        exit_time=exit_time or datetime.now(),
        excursion=ExcursionData(
            mfe_pnl=mfe_pnl,
            mae_pnl=mae_pnl,
            exit_efficiency=exit_efficiency,
            went_green_before_loss=went_green,
        ),
    )
    return trade


@pytest.fixture
def mock_daemon():
    """Create a daemon with mocked position_monitor and trade store."""
    from strat.signal_automation.daemon import SignalDaemon

    daemon = MagicMock(spec=SignalDaemon)
    daemon._get_today_excursion_stats = SignalDaemon._get_today_excursion_stats.__get__(daemon)

    # Mock position_monitor with get_trade_store()
    mock_store = MagicMock()
    daemon.position_monitor = MagicMock()
    daemon.position_monitor.get_trade_store.return_value = mock_store

    return daemon, mock_store


# =========================================================================
# Tests: _get_today_excursion_stats
# =========================================================================

class TestGetTodayExcursionStats:
    """Tests for daemon._get_today_excursion_stats()."""

    def test_basic_stats_with_trades(self, mock_daemon):
        daemon, store = mock_daemon
        today = date.today()

        trades = [
            _make_trade(pnl=50, mfe_pnl=100, mae_pnl=-30, exit_efficiency=0.5),
            _make_trade(pnl=-25, mfe_pnl=20, mae_pnl=-50, exit_efficiency=0.0, went_green=True),
            _make_trade(pnl=75, mfe_pnl=80, mae_pnl=-10, exit_efficiency=0.94),
        ]
        store.get_trades.return_value = trades

        result = daemon._get_today_excursion_stats(today)

        assert result is not None
        assert result['trades_with_excursion'] == 3
        assert result['avg_mfe'] == pytest.approx((100 + 20 + 80) / 3)
        assert result['avg_mae'] == pytest.approx((-30 + -50 + -10) / 3)
        # exit_efficiency average: all 3 have mfe > 0
        assert result['avg_exit_efficiency'] == pytest.approx((0.5 + 0.0 + 0.94) / 3)

    def test_losers_went_green(self, mock_daemon):
        daemon, store = mock_daemon
        today = date.today()

        trades = [
            _make_trade(pnl=-50, mfe_pnl=30, mae_pnl=-80, went_green=True),
            _make_trade(pnl=-20, mfe_pnl=5, mae_pnl=-40, went_green=False),
            _make_trade(pnl=100, mfe_pnl=120, mae_pnl=-10, exit_efficiency=0.83),
        ]
        store.get_trades.return_value = trades

        result = daemon._get_today_excursion_stats(today)

        assert result['losers_went_green'] == 1
        assert result['total_losers'] == 2

    def test_no_trades_returns_none(self, mock_daemon):
        daemon, store = mock_daemon
        store.get_trades.return_value = []

        result = daemon._get_today_excursion_stats(date.today())
        assert result is None

    def test_no_position_monitor_returns_none(self, mock_daemon):
        daemon, _ = mock_daemon
        daemon.position_monitor = None

        result = daemon._get_today_excursion_stats(date.today())
        assert result is None

    def test_no_trade_store_returns_none(self, mock_daemon):
        daemon, _ = mock_daemon
        daemon.position_monitor.get_trade_store.return_value = None

        result = daemon._get_today_excursion_stats(date.today())
        assert result is None

    def test_trades_without_excursion_data_skipped(self, mock_daemon):
        daemon, store = mock_daemon

        trade_with = _make_trade(pnl=50, mfe_pnl=100, mae_pnl=-20, exit_efficiency=0.5)
        trade_without = EnrichedTradeRecord(
            trade_id="no_excursion",
            symbol="QQQ",
            pnl=30,
            exit_time=datetime.now(),
            excursion=ExcursionData(),  # Default: mfe_pnl=0.0 (not None)
        )
        # Default mfe_pnl=0.0 is not None, so it passes the filter.
        # To test the None case, set mfe_pnl to None explicitly.
        trade_none = EnrichedTradeRecord(
            trade_id="none_excursion",
            symbol="IWM",
            pnl=10,
            exit_time=datetime.now(),
        )
        trade_none.excursion.mfe_pnl = None

        store.get_trades.return_value = [trade_with, trade_without, trade_none]

        result = daemon._get_today_excursion_stats(date.today())
        # trade_none is filtered out (mfe_pnl is None)
        # trade_with and trade_without both have mfe_pnl not None
        assert result['trades_with_excursion'] == 2

    def test_exit_efficiency_only_for_winners(self, mock_daemon):
        """Exit efficiency average should only include trades with MFE > 0."""
        daemon, store = mock_daemon

        trades = [
            _make_trade(pnl=50, mfe_pnl=100, mae_pnl=-20, exit_efficiency=0.5),
            _make_trade(pnl=-30, mfe_pnl=0.0, mae_pnl=-50),  # MFE=0, no efficiency
        ]
        store.get_trades.return_value = trades

        result = daemon._get_today_excursion_stats(date.today())
        # Only the first trade has mfe > 0
        assert result['avg_exit_efficiency'] == pytest.approx(0.5)

    def test_store_exception_returns_none(self, mock_daemon):
        daemon, store = mock_daemon
        store.get_trades.side_effect = Exception("DB error")

        result = daemon._get_today_excursion_stats(date.today())
        assert result is None


# =========================================================================
# Tests: Discord embed rendering
# =========================================================================

class TestDiscordAuditExcursionRendering:
    """Tests for MFE/MAE and Capital fields in send_daily_audit()."""

    @pytest.fixture
    def alerter(self):
        """Create a DiscordAlerter with mocked webhook."""
        from strat.signal_automation.alerters.discord_alerter import DiscordAlerter

        alerter = DiscordAlerter(
            webhook_url="https://discord.com/api/webhooks/123/abc",
        )
        alerter._send_webhook = MagicMock(return_value=True)
        return alerter

    def _base_audit_data(self, **overrides):
        data = {
            'date': '2026-02-17',
            'trades_today': 3,
            'wins': 2,
            'losses': 1,
            'total_pnl': 125.50,
            'profit_factor': 2.5,
            'open_positions': [],
            'anomalies': [],
        }
        data.update(overrides)
        return data

    def test_embed_includes_excursion_field(self, alerter):
        audit_data = self._base_audit_data(
            excursion={
                'trades_with_excursion': 3,
                'avg_mfe': 85.0,
                'avg_mae': -30.0,
                'avg_exit_efficiency': 0.72,
                'losers_went_green': 1,
                'total_losers': 1,
            }
        )

        alerter.send_daily_audit(audit_data)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]
        assert any('MFE/MAE' in name for name in field_names)

        # Verify field content
        exc_field = next(f for f in embed['fields'] if 'MFE/MAE' in f['name'])
        assert '$+85.00' in exc_field['value']
        assert '$-30.00' in exc_field['value']
        assert '72%' in exc_field['value']
        assert '1/1' in exc_field['value']

    def test_embed_without_excursion_is_backward_compatible(self, alerter):
        """Audit without excursion data should render without error."""
        audit_data = self._base_audit_data()

        result = alerter.send_daily_audit(audit_data)
        assert result is True

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]
        assert not any('MFE/MAE' in name for name in field_names)

    def test_embed_includes_capital_field(self, alerter):
        audit_data = self._base_audit_data(
            capital={
                'available_capital': 95000.0,
                'deployed_capital': 5000.0,
                'portfolio_heat_pct': 5.0,
            }
        )

        alerter.send_daily_audit(audit_data)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]
        assert 'Capital Status' in field_names

        cap_field = next(f for f in embed['fields'] if f['name'] == 'Capital Status')
        assert '$95,000' in cap_field['value']
        assert '$5,000' in cap_field['value']
        assert '5.0%' in cap_field['value']

    def test_embed_without_capital_is_backward_compatible(self, alerter):
        audit_data = self._base_audit_data()

        result = alerter.send_daily_audit(audit_data)
        assert result is True

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]
        assert 'Capital Status' not in field_names

    def test_excursion_field_with_no_losers(self, alerter):
        """When all trades are winners, losers section omitted."""
        audit_data = self._base_audit_data(
            excursion={
                'trades_with_excursion': 2,
                'avg_mfe': 100.0,
                'avg_mae': -15.0,
                'avg_exit_efficiency': 0.85,
                'losers_went_green': 0,
                'total_losers': 0,
            }
        )

        alerter.send_daily_audit(audit_data)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        exc_field = next(f for f in embed['fields'] if 'MFE/MAE' in f['name'])
        # No "Losers went green" line when total_losers=0
        assert 'Losers' not in exc_field['value']
