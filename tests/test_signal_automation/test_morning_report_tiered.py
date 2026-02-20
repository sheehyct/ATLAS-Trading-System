"""
Tests for EQUITY-116: Tiered morning report output.

Tests the tier grouping in MorningReportGenerator and the tiered
Discord embed formatting in DiscordAlerter.send_morning_report().
"""

import pytest
from unittest.mock import MagicMock, patch

from strat.signal_automation.config import MorningReportConfig
from strat.signal_automation.coordinators.morning_report import MorningReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(symbol='SPY', tier=2, convergence=None, **overrides):
    """Create a mock candidate dict matching v2.0 pipeline output."""
    base = {
        'symbol': symbol,
        'composite_score': 75.0,
        'rank': 1,
        'tier': tier,
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
        },
        'metrics': {
            'atr_percent': 2.5,
            'dollar_volume': 5_000_000_000,
            'risk_reward': 2.0,
        },
    }
    if convergence is not None:
        base['convergence'] = convergence
    base.update(overrides)
    return base


def _make_convergence_dict(inside_count=2, inside_tfs=None, score=79.0,
                           bullish_trigger=400.0, bearish_trigger=370.0,
                           spread_pct=1.8, alignment='bearish'):
    """Create a convergence dict matching ConvergenceMetadata.to_dict() output."""
    return {
        'inside_bar_count': inside_count,
        'inside_bar_timeframes': inside_tfs or ['1M', '1W'],
        'convergence_score': score,
        'bullish_trigger': bullish_trigger,
        'bearish_trigger': bearish_trigger,
        'trigger_spread_pct': spread_pct,
        'prior_direction_alignment': alignment,
        'is_convergence': inside_count >= 2,
        'trigger_levels': {},
    }


# ---------------------------------------------------------------------------
# Morning Report Tier Grouping
# ---------------------------------------------------------------------------

class TestMorningReportTierGrouping:
    """Test that generate_report() groups candidates by tier."""

    @pytest.fixture
    def generator(self):
        config = MorningReportConfig(enabled=True, max_candidates=8)
        return MorningReportGenerator(
            alerters=[MagicMock()],
            config=config,
        )

    @patch.object(MorningReportGenerator, '_run_pipeline')
    @patch.object(MorningReportGenerator, '_get_gap_analysis', return_value=[])
    def test_tier_grouping(self, _mock_gaps, mock_pipeline, generator):
        """Candidates are grouped into tier1, tier2, tier3."""
        candidates = [
            _make_candidate('CRWD', tier=1, convergence=_make_convergence_dict()),
            _make_candidate('MSFT', tier=2),
            _make_candidate('AAPL', tier=3, pattern={
                'type': '2D-2D', 'base_type': '2-2', 'signal_type': 'SETUP',
                'direction': 'PUT', 'timeframe': '1W', 'is_bidirectional': False,
            }),
            _make_candidate('GOOGL', tier=2),
        ]
        mock_pipeline.return_value = (candidates, {'final_candidates': 4})

        report = generator.generate_report()

        assert len(report['tier1_setups']) == 1
        assert report['tier1_setups'][0]['symbol'] == 'CRWD'
        assert len(report['tier2_setups']) == 2
        assert len(report['tier3_context']) == 1
        assert report['tier3_context'][0]['symbol'] == 'AAPL'

    @patch.object(MorningReportGenerator, '_run_pipeline')
    @patch.object(MorningReportGenerator, '_get_gap_analysis', return_value=[])
    def test_setups_backward_compat(self, _mock_gaps, mock_pipeline, generator):
        """The 'setups' key contains T1 + T2 (not T3)."""
        candidates = [
            _make_candidate('CRWD', tier=1),
            _make_candidate('MSFT', tier=2),
            _make_candidate('AAPL', tier=3),
        ]
        mock_pipeline.return_value = (candidates, {})

        report = generator.generate_report()

        setup_symbols = [c['symbol'] for c in report['setups']]
        assert 'CRWD' in setup_symbols  # T1 included
        assert 'MSFT' in setup_symbols  # T2 included
        assert 'AAPL' not in setup_symbols  # T3 excluded

    @patch.object(MorningReportGenerator, '_run_pipeline')
    @patch.object(MorningReportGenerator, '_get_gap_analysis', return_value=[])
    def test_no_tier_key_defaults_to_tier2(self, _mock_gaps, mock_pipeline, generator):
        """v1.0 candidates without 'tier' key default to tier 2."""
        candidates = [
            {'symbol': 'SPY', 'composite_score': 80.0, 'pattern': {}, 'levels': {},
             'tfc': {}, 'metrics': {}},
        ]
        mock_pipeline.return_value = (candidates, {})

        report = generator.generate_report()

        assert len(report['tier2_setups']) == 1
        assert len(report['tier1_setups']) == 0
        assert len(report['tier3_context']) == 0


# ---------------------------------------------------------------------------
# Discord Tiered Embed Formatting
# ---------------------------------------------------------------------------

class TestDiscordTieredEmbed:
    """Test tiered Discord embed output."""

    @pytest.fixture
    def alerter(self):
        from strat.signal_automation.alerters.discord_alerter import DiscordAlerter
        a = DiscordAlerter(webhook_url="https://discord.com/api/webhooks/123/abc")
        a._send_webhook = MagicMock(return_value=True)
        return a

    def _base_report(self, **overrides):
        data = {
            'date': '2026-02-20',
            'setups': [],
            'tier1_setups': [],
            'tier2_setups': [],
            'tier3_context': [],
            'gaps': [],
            'open_positions': [],
            'yesterday': {},
            'capital': {},
            'pipeline_stats': {'final_candidates': 0},
            'duration_seconds': 45.0,
        }
        data.update(overrides)
        return data

    def test_tier1_field_present(self, alerter):
        """When T1 setups exist, TIER 1: CONVERGENCE field appears."""
        report = self._base_report(
            tier1_setups=[
                _make_candidate('CRWD', tier=1, convergence=_make_convergence_dict(
                    inside_count=3, inside_tfs=['1M', '1W', '1D'],
                    score=87.0, bearish_trigger=371.50, spread_pct=1.8,
                ), pattern={
                    'type': '3-1-2D', 'base_type': '3-1-2', 'signal_type': 'SETUP',
                    'direction': 'PUT', 'timeframe': '1D', 'is_bidirectional': True,
                }),
            ],
        )
        report['setups'] = report['tier1_setups']

        alerter.send_morning_report(report)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]

        assert any('TIER 1' in n and 'CONVERGENCE' in n for n in field_names)
        t1_field = next(f for f in embed['fields'] if 'TIER 1' in f['name'])
        assert 'CRWD' in t1_field['value']
        assert '3-inside' in t1_field['value']
        assert '1M/1W/1D' in t1_field['value']
        assert '$371.50' in t1_field['value']
        assert 'cascades' in t1_field['value']

    def test_tier2_field_present(self, alerter):
        """When T2 setups exist, TIER 2: STANDARD field appears."""
        report = self._base_report(
            tier1_setups=[_make_candidate('CRWD', tier=1, convergence=_make_convergence_dict())],
            tier2_setups=[_make_candidate('MSFT', tier=2)],
        )
        report['setups'] = report['tier1_setups'] + report['tier2_setups']

        alerter.send_morning_report(report)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]

        assert any('TIER 2' in n and 'STANDARD' in n for n in field_names)
        t2_field = next(f for f in embed['fields'] if 'TIER 2' in f['name'])
        assert 'MSFT' in t2_field['value']
        assert '[CALL]' in t2_field['value']

    def test_tier3_context_field(self, alerter):
        """When T3 context exists, DIRECTIONAL CONTEXT field appears."""
        report = self._base_report(
            tier3_context=[
                _make_candidate('AAPL', tier=3, pattern={
                    'type': '2D-2D', 'base_type': '2-2', 'signal_type': 'SETUP',
                    'direction': 'PUT', 'timeframe': '1W', 'is_bidirectional': False,
                }),
                _make_candidate('META', tier=3, pattern={
                    'type': '2U-2U', 'base_type': '2-2', 'signal_type': 'SETUP',
                    'direction': 'CALL', 'timeframe': '1W', 'is_bidirectional': False,
                }),
            ],
        )

        alerter.send_morning_report(report)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]

        assert 'DIRECTIONAL CONTEXT' in field_names
        ctx_field = next(f for f in embed['fields'] if 'CONTEXT' in f['name'])
        assert 'Bearish: AAPL(1W 2D-2D)' in ctx_field['value']
        assert 'Bullish: META(1W 2U-2U)' in ctx_field['value']

    def test_empty_tiers_no_crash(self, alerter):
        """Empty tier lists produce no setup fields, no crash."""
        report = self._base_report()

        result = alerter.send_morning_report(report)
        assert result is True

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]
        assert not any('TIER' in n for n in field_names)
        assert not any('CONTEXT' in n for n in field_names)

    def test_legacy_v1_fallback(self, alerter):
        """v1.0 report data (no tier keys) falls back to flat STRAT Setups field."""
        report = {
            'date': '2026-02-20',
            'setups': [_make_candidate('SPY', tier=2)],
            'gaps': [],
            'open_positions': [],
            'yesterday': {},
            'capital': {},
            'pipeline_stats': {'final_candidates': 1},
            'duration_seconds': 10.0,
            # No tier1_setups, tier2_setups, or tier3_context keys
        }

        alerter.send_morning_report(report)

        payload = alerter._send_webhook.call_args[0][0]
        embed = payload['embeds'][0]
        field_names = [f['name'] for f in embed['fields']]

        # Should use legacy "STRAT Setups" name, not tiered
        assert any('STRAT Setups' in n for n in field_names)
        assert not any('TIER' in n for n in field_names)


# ---------------------------------------------------------------------------
# Format helpers (unit tests)
# ---------------------------------------------------------------------------

class TestFormatHelpers:

    @pytest.fixture
    def alerter(self):
        from strat.signal_automation.alerters.discord_alerter import DiscordAlerter
        a = DiscordAlerter(webhook_url="https://discord.com/api/webhooks/123/abc")
        a._send_webhook = MagicMock(return_value=True)
        return a

    def test_format_setup_line(self, alerter):
        c = _make_candidate('NVDA', tier=2)
        line = alerter._format_setup_line(c)
        assert '[CALL]' in line
        assert 'NVDA' in line
        assert 'TFC:' in line
        assert 'E: $' in line

    def test_format_tier1_field_structure(self, alerter):
        setups = [_make_candidate('CRWD', tier=1, convergence=_make_convergence_dict(
            inside_count=2, inside_tfs=['1W', '1D'], score=75.0,
            bullish_trigger=395.0, bearish_trigger=378.0, spread_pct=4.4,
        ))]
        field = alerter._format_tier1_field(setups)

        assert field['name'] == 'TIER 1: CONVERGENCE (1)'
        assert field['inline'] is False
        assert '**CRWD**' in field['value']
        assert '2-inside' in field['value']
        assert '1W/1D' in field['value']

    def test_format_tier3_field_groups_by_direction(self, alerter):
        context = [
            _make_candidate('AAPL', tier=3, pattern={
                'type': '2D-2D', 'base_type': '2-2', 'signal_type': 'SETUP',
                'direction': 'PUT', 'timeframe': '1W', 'is_bidirectional': False,
            }),
            _make_candidate('NFLX', tier=3, pattern={
                'type': '2D-2D', 'base_type': '2-2', 'signal_type': 'SETUP',
                'direction': 'PUT', 'timeframe': '1D', 'is_bidirectional': False,
            }),
            _make_candidate('META', tier=3, pattern={
                'type': '2U-2U', 'base_type': '2-2', 'signal_type': 'SETUP',
                'direction': 'CALL', 'timeframe': '1W', 'is_bidirectional': False,
            }),
        ]
        field = alerter._format_tier3_field(context)

        assert field['name'] == 'DIRECTIONAL CONTEXT'
        # Bearish line should have both AAPL and NFLX
        assert 'Bearish:' in field['value']
        assert 'AAPL' in field['value']
        assert 'NFLX' in field['value']
        # Bullish line should have META
        assert 'Bullish:' in field['value']
        assert 'META' in field['value']

    def test_format_tier3_empty(self, alerter):
        """Empty tier 3 list produces 'None' value."""
        field = alerter._format_tier3_field([])
        assert field['value'] == 'None'
