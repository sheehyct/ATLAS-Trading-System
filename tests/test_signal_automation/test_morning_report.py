"""
Tests for EQUITY-112: Pre-market morning report coordinator + Discord embed.

Tests the MorningReportGenerator and discord_alerter.send_morning_report().
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from strat.signal_automation.config import MorningReportConfig
from strat.signal_automation.coordinators.morning_report import MorningReportGenerator


# =========================================================================
# Fixtures
# =========================================================================

def _make_candidate(**overrides):
    """Create a mock candidate dict matching pipeline output."""
    base = {
        'symbol': 'SPY',
        'composite_score': 75.0,
        'rank': 1,
        'pattern': {
            'type': '2U-1-2U',
            'base_type': '2-1-2',
            'signal_type': 'SETUP',
            'direction': 'CALL',
            'timeframe': '1D',
            'is_bidirectional': True,
        },
        'levels': {
            'entry_trigger': 450.00,
            'stop_price': 445.00,
            'target_price': 460.00,
            'current_price': 448.00,
            'distance_to_trigger_pct': 0.45,
        },
        'tfc': {
            'score': 4,
            'alignment': '4/5 Bullish',
            'direction': 'bullish',
            'passes_flexible': True,
            'risk_multiplier': 1.0,
            'priority_rank': 1,
        },
        'metrics': {
            'atr_percent': 2.5,
            'dollar_volume': 5000000000,
            'risk_reward': 2.0,
        },
    }
    base.update(overrides)
    return base


@pytest.fixture
def config():
    return MorningReportConfig(
        enabled=True,
        hour=6,
        minute=0,
        max_candidates=8,
        min_gap_pct=0.5,
    )


@pytest.fixture
def mock_position_monitor():
    pm = MagicMock()
    pm.get_tracked_positions.return_value = []
    return pm


@pytest.fixture
def mock_trading_client():
    tc = MagicMock()
    tc.get_closed_trades.return_value = []
    return tc


@pytest.fixture
def mock_capital_tracker():
    ct = MagicMock()
    ct.get_summary.return_value = {
        'available_capital': 95000.0,
        'deployed_capital': 5000.0,
        'portfolio_heat_pct': 5.0,
    }
    return ct


@pytest.fixture
def mock_alerter():
    alerter = MagicMock()
    alerter.name = 'discord'
    alerter.send_morning_report = MagicMock(return_value=True)
    return alerter


@pytest.fixture
def generator(config, mock_position_monitor, mock_trading_client,
              mock_capital_tracker, mock_alerter):
    return MorningReportGenerator(
        alerters=[mock_alerter],
        position_monitor=mock_position_monitor,
        capital_tracker=mock_capital_tracker,
        trading_client=mock_trading_client,
        config=config,
    )


# =========================================================================
# Tests: MorningReportGenerator
# =========================================================================

class TestMorningReportGenerator:

    @patch('strat.signal_automation.coordinators.morning_report.MorningReportGenerator._run_pipeline')
    @patch('strat.signal_automation.coordinators.morning_report.MorningReportGenerator._get_gap_analysis')
    def test_generate_report_all_sections(self, mock_gaps, mock_pipeline, generator):
        """Full report should have all 5 sections populated."""
        candidates = [_make_candidate(symbol='NVDA'), _make_candidate(symbol='TSLA')]
        mock_pipeline.return_value = (candidates, {'final_candidates': 2})
        mock_gaps.return_value = [
            {'symbol': 'NVDA', 'gap_pct': 2.3, 'prev_close': 140, 'premarket_price': 143.22}
        ]

        report = generator.generate_report()

        assert report['date'] == date.today().isoformat()
        assert len(report['setups']) == 2
        assert len(report['gaps']) == 1
        assert 'duration_seconds' in report
        assert isinstance(report['open_positions'], list)
        assert isinstance(report['yesterday'], dict)
        assert isinstance(report['capital'], dict)

    @patch('strat.signal_automation.coordinators.morning_report.MorningReportGenerator._run_pipeline')
    def test_pipeline_failure_graceful(self, mock_pipeline, generator):
        """If pipeline fails, report still generates with empty setups."""
        mock_pipeline.return_value = ([], {'error': 'ImportError'})

        report = generator.generate_report()

        assert report['setups'] == []
        assert 'error' in report['pipeline_stats']

    @patch('strat.signal_automation.coordinators.morning_report.MorningReportGenerator._run_pipeline')
    def test_max_candidates_respected(self, mock_pipeline, generator):
        """Report should cap setups to max_candidates."""
        candidates = [_make_candidate(symbol=f'SYM{i}') for i in range(15)]
        mock_pipeline.return_value = (candidates, {'final_candidates': 15})

        report = generator.generate_report()

        assert len(report['setups']) == 8  # config.max_candidates

    def test_open_positions(self, generator, mock_position_monitor):
        """Open positions from position monitor."""
        pos = MagicMock()
        pos.symbol = 'AAPL260220C00190000'
        pos.pattern_type = '3-2U'
        pos.timeframe = '1D'
        pos.unrealized_pnl = 45.0
        pos.unrealized_pct = 0.15
        mock_position_monitor.get_tracked_positions.return_value = [pos]

        result = generator._get_open_positions()

        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL260220C00190000'
        assert result[0]['unrealized_pnl'] == 45.0

    def test_open_positions_no_monitor(self, config, mock_alerter):
        """No position monitor returns empty list."""
        gen = MorningReportGenerator(
            alerters=[mock_alerter],
            position_monitor=None,
            config=config,
        )
        assert gen._get_open_positions() == []

    def test_yesterday_recap_with_trades(self, generator, mock_trading_client):
        """Yesterday recap calculates wins, losses, P&L."""
        yesterday = date.today() - timedelta(days=1)
        mock_trading_client.get_closed_trades.return_value = [
            {
                'sell_time_dt': datetime.combine(yesterday, datetime.min.time()).replace(
                    hour=15, minute=55, tzinfo=timezone.utc
                ),
                'realized_pnl': 50.0,
            },
            {
                'sell_time_dt': datetime.combine(yesterday, datetime.min.time()).replace(
                    hour=15, minute=58, tzinfo=timezone.utc
                ),
                'realized_pnl': -30.0,
            },
        ]

        result = generator._get_yesterday_recap()

        assert result['trades'] == 2
        assert result['wins'] == 1
        assert result['losses'] == 1
        assert result['total_pnl'] == pytest.approx(20.0)
        assert result['win_rate'] == pytest.approx(50.0)

    def test_yesterday_recap_no_client(self, config, mock_alerter):
        """No trading client returns empty dict."""
        gen = MorningReportGenerator(
            alerters=[mock_alerter],
            trading_client=None,
            config=config,
        )
        assert gen._get_yesterday_recap() == {}

    def test_capital_status(self, generator):
        result = generator._get_capital_status()
        assert result['available_capital'] == 95000.0
        assert result['portfolio_heat_pct'] == 5.0

    def test_capital_status_no_tracker(self, config, mock_alerter):
        gen = MorningReportGenerator(
            alerters=[mock_alerter],
            capital_tracker=None,
            config=config,
        )
        assert gen._get_capital_status() == {}

    @patch('strat.signal_automation.coordinators.morning_report.MorningReportGenerator._run_pipeline')
    def test_run_sends_to_alerters(self, mock_pipeline, generator, mock_alerter):
        """run() should call send_morning_report on all capable alerters."""
        mock_pipeline.return_value = ([], {})

        generator.run()

        mock_alerter.send_morning_report.assert_called_once()
        report_data = mock_alerter.send_morning_report.call_args[0][0]
        assert 'date' in report_data
        assert 'setups' in report_data


# =========================================================================
# Tests: Gap Analysis
# =========================================================================

class TestGapAnalysis:

    @patch('strat.signal_automation.coordinators.morning_report.get_alpaca_credentials')
    @patch('strat.signal_automation.coordinators.morning_report.StockHistoricalDataClient')
    def test_gap_calculation(self, MockClient, mock_creds, generator):
        """Gap percentage = (premarket - prev_close) / prev_close * 100."""
        mock_creds.return_value = {'api_key': 'k', 'secret_key': 's'}

        # Build mock snapshot
        snap = MagicMock()
        snap.previous_daily_bar.close = 100.0
        snap.latest_trade.price = 103.0

        mock_instance = MagicMock()
        mock_instance.get_stock_snapshot.return_value = {'NVDA': snap}
        MockClient.return_value = mock_instance

        candidates = [_make_candidate(symbol='NVDA')]
        result = generator._get_gap_analysis(candidates)

        assert len(result) == 1
        assert result[0]['symbol'] == 'NVDA'
        assert result[0]['gap_pct'] == pytest.approx(3.0)
        assert result[0]['prev_close'] == 100.0
        assert result[0]['premarket_price'] == 103.0

    @patch('strat.signal_automation.coordinators.morning_report.get_alpaca_credentials')
    @patch('strat.signal_automation.coordinators.morning_report.StockHistoricalDataClient')
    def test_gap_below_threshold_filtered(self, MockClient, mock_creds, generator):
        """Gaps below min_gap_pct (0.5%) should be filtered out."""
        mock_creds.return_value = {'api_key': 'k', 'secret_key': 's'}

        snap = MagicMock()
        snap.previous_daily_bar.close = 100.0
        snap.latest_trade.price = 100.3  # 0.3% gap, below 0.5% threshold

        mock_instance = MagicMock()
        mock_instance.get_stock_snapshot.return_value = {'SPY': snap}
        MockClient.return_value = mock_instance

        candidates = [_make_candidate(symbol='SPY')]
        result = generator._get_gap_analysis(candidates)

        assert len(result) == 0

    def test_gap_analysis_empty_candidates(self, generator):
        """Empty candidates list returns empty gaps."""
        result = generator._get_gap_analysis([])
        assert result == []

    @patch('strat.signal_automation.coordinators.morning_report.get_alpaca_credentials',
           side_effect=Exception("No credentials"))
    def test_gap_analysis_failure_graceful(self, mock_creds, generator):
        """Alpaca failure returns empty list, no crash."""
        candidates = [_make_candidate(symbol='SPY')]
        result = generator._get_gap_analysis(candidates)
        assert result == []


# =========================================================================
# Tests: Discord Morning Report Embed
# =========================================================================

class TestDiscordMorningReportEmbed:

    @pytest.fixture
    def alerter(self):
        from strat.signal_automation.alerters.discord_alerter import DiscordAlerter

        alerter = DiscordAlerter(
            webhook_url="https://discord.com/api/webhooks/123/abc",
        )
        alerter._send_webhook = MagicMock(return_value=True)
        return alerter

    def _base_report(self, **overrides):
        data = {
            'date': '2026-02-17',
            'setups': [_make_candidate(symbol='NVDA'), _make_candidate(symbol='TSLA')],
            'gaps': [
                {'symbol': 'NVDA', 'gap_pct': 2.3, 'prev_close': 140, 'premarket_price': 143.22},
            ],
            'open_positions': [
                {'symbol': 'AAPL', 'pattern_type': '3-2U', 'timeframe': '1D',
                 'unrealized_pnl': 45.0, 'unrealized_pct': 0.15},
            ],
            'yesterday': {'trades': 3, 'wins': 2, 'losses': 1,
                         'total_pnl': 125.0, 'win_rate': 66.7, 'profit_factor': 2.5},
            'capital': {'available_capital': 95000, 'portfolio_heat_pct': 5.0},
            'pipeline_stats': {'final_candidates': 2},
            'duration_seconds': 312.5,
        }
        data.update(overrides)
        return data

    def test_embed_structure(self, alerter):
        """Verify embed has correct title, color, and all fields."""
        report = self._base_report()

        result = alerter.send_morning_report(report)
        assert result is True

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]

        assert 'PRE-MARKET MORNING REPORT' in embed['title']
        assert '2026-02-17' in embed['title']
        assert embed['color'] == 0x0099FF  # Blue info color

        field_names = [f['name'] for f in embed['fields']]
        assert any('STRAT Setups' in n for n in field_names)
        assert any('Pre-Market Gaps' in n for n in field_names)
        assert any('Open Positions' in n for n in field_names)
        assert 'Yesterday' in field_names
        assert 'Capital' in field_names

    def test_setups_field_content(self, alerter):
        report = self._base_report()

        alerter.send_morning_report(report)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        setup_field = next(f for f in embed['fields'] if 'STRAT' in f['name'])

        assert '[CALL]' in setup_field['value']
        assert 'NVDA' in setup_field['value']
        assert 'TFC:' in setup_field['value']

    def test_gaps_field_content(self, alerter):
        report = self._base_report()

        alerter.send_morning_report(report)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        gap_field = next(f for f in embed['fields'] if 'Gap' in f['name'])

        assert 'NVDA' in gap_field['value']
        assert '+2.3%' in gap_field['value']

    def test_empty_report(self, alerter):
        """Empty report still sends without error."""
        report = {
            'date': '2026-02-17',
            'setups': [],
            'gaps': [],
            'open_positions': [],
            'yesterday': {},
            'capital': {},
            'pipeline_stats': {},
            'duration_seconds': 1.0,
        }

        result = alerter.send_morning_report(report)
        assert result is True

    def test_footer_includes_duration(self, alerter):
        report = self._base_report()

        alerter.send_morning_report(report)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        assert '312s' in embed['footer']['text']


# =========================================================================
# Tests: Config
# =========================================================================

class TestMorningReportConfig:

    def test_default_config(self):
        config = MorningReportConfig()
        assert config.enabled is True
        assert config.hour == 6
        assert config.minute == 0
        assert config.max_candidates == 8
        assert config.min_gap_pct == 0.5

    def test_config_from_env(self):
        """Config loads from environment variables."""
        with patch.dict('os.environ', {
            'MORNING_REPORT_ENABLED': 'false',
            'MORNING_REPORT_HOUR': '7',
            'MORNING_REPORT_MINUTE': '30',
        }):
            from strat.signal_automation.config import SignalAutomationConfig
            config = SignalAutomationConfig.from_env()
            assert config.morning_report.enabled is False
            assert config.morning_report.hour == 7
            assert config.morning_report.minute == 30

    def test_config_in_signal_automation_config(self):
        """MorningReportConfig is part of SignalAutomationConfig."""
        from strat.signal_automation.config import SignalAutomationConfig
        config = SignalAutomationConfig()
        assert hasattr(config, 'morning_report')
        assert isinstance(config.morning_report, MorningReportConfig)
