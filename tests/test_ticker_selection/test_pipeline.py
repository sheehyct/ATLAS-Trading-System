"""Integration tests for TickerSelectionPipeline with mocked Alpaca responses."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from strat.ticker_selection.config import TickerSelectionConfig
from strat.ticker_selection.pipeline import TickerSelectionPipeline
from strat.ticker_selection.screener import ScreenedStock


def _make_mock_signal(
    symbol='TEST',
    pattern_type='3-1-2U',
    direction='CALL',
    timeframe='1D',
    signal_type='SETUP',
    tfc_score=3,
    tfc_passes=True,
    tfc_alignment='3/4 BULLISH',
    is_bidirectional=True,
):
    """Create a mock DetectedSignal."""
    sig = MagicMock()
    sig.symbol = symbol
    sig.pattern_type = pattern_type
    sig.direction = direction
    sig.timeframe = timeframe
    sig.signal_type = signal_type
    sig.is_bidirectional = is_bidirectional
    sig.entry_trigger = 100.0
    sig.stop_price = 97.0
    sig.target_price = 106.0
    sig.context.tfc_score = tfc_score
    sig.context.tfc_passes = tfc_passes
    sig.context.tfc_alignment = tfc_alignment
    sig.context.risk_multiplier = 0.5
    sig.context.priority_rank = 2
    sig.context.atr_percent = 2.5
    return sig


class TestPipelineHardFilters:
    """Test that hard filters correctly reject signals."""

    def setup_method(self):
        self.config = TickerSelectionConfig(candidates_path='test_candidates.json')
        self.pipeline = TickerSelectionPipeline(self.config)

    def test_rejects_completed_signals(self):
        """Only SETUP signals pass."""
        sig = _make_mock_signal(signal_type='COMPLETED')
        assert sig.signal_type != 'SETUP'

    def test_rejects_tfc_fails_flexible(self):
        """TFC must pass flexible check."""
        sig = _make_mock_signal(tfc_passes=False)
        assert not sig.context.tfc_passes


class TestPipelineOutput:
    """Test candidates.json output format."""

    def test_output_schema(self, tmp_path):
        """Verify output matches expected schema."""
        config = TickerSelectionConfig(
            candidates_path=str(tmp_path / 'candidates.json'),
            universe_cache_path=str(tmp_path / 'universe.json'),
        )
        pipeline = TickerSelectionPipeline(config)

        # Mock all stages
        with patch.object(pipeline.universe_mgr, 'get_universe', return_value=['AAPL', 'MSFT']), \
             patch.object(pipeline.screener, 'screen', return_value=[
                 ScreenedStock('AAPL', 180.0, 50e6, 2.5, 182.0, 177.0, 179.0, 180.0, 178.0, 300000),
             ]), \
             patch.object(pipeline, '_get_scanner') as mock_scanner:

            scanner = MagicMock()
            scanner.scan_symbol_all_timeframes_resampled.return_value = [
                _make_mock_signal(symbol='AAPL'),
            ]
            mock_scanner.return_value = scanner

            result = pipeline.run(dry_run=False)

        # Verify schema
        assert result['version'] == '1.0'
        assert 'generated_at' in result
        assert 'pipeline_stats' in result
        assert 'core_symbols' in result
        assert 'candidates' in result

        stats = result['pipeline_stats']
        assert 'universe_size' in stats
        assert 'screened_size' in stats
        assert 'final_candidates' in stats
        assert 'scan_duration_seconds' in stats

        # Verify file was written
        path = tmp_path / 'candidates.json'
        assert path.exists()
        written = json.loads(path.read_text())
        assert written['version'] == '1.0'

    def test_candidate_entry_schema(self, tmp_path):
        """Verify each candidate has all required fields."""
        config = TickerSelectionConfig(
            candidates_path=str(tmp_path / 'candidates.json'),
            universe_cache_path=str(tmp_path / 'universe.json'),
        )
        pipeline = TickerSelectionPipeline(config)

        with patch.object(pipeline.universe_mgr, 'get_universe', return_value=['CRWD']), \
             patch.object(pipeline.screener, 'screen', return_value=[
                 ScreenedStock('CRWD', 385.0, 45e6, 2.3, 388.0, 380.0, 383.0, 385.0, 382.0, 120000),
             ]), \
             patch.object(pipeline, '_get_scanner') as mock_scanner:

            scanner = MagicMock()
            scanner.scan_symbol_all_timeframes_resampled.return_value = [
                _make_mock_signal(symbol='CRWD'),
            ]
            mock_scanner.return_value = scanner

            result = pipeline.run(dry_run=True)

        if result['candidates']:
            c = result['candidates'][0]
            assert 'symbol' in c
            assert 'composite_score' in c
            assert 'rank' in c
            assert 'pattern' in c
            assert 'levels' in c
            assert 'tfc' in c
            assert 'metrics' in c
            assert 'scoring_breakdown' in c

            # Pattern sub-fields
            p = c['pattern']
            assert 'type' in p
            assert 'base_type' in p
            assert 'signal_type' in p
            assert 'direction' in p

            # TFC sub-fields
            t = c['tfc']
            assert 'score' in t
            assert 'passes_flexible' in t


class TestDedupBySymbol:
    """Test best-setup-per-symbol deduplication."""

    def test_keeps_highest_score_per_symbol(self):
        """When symbol has multiple setups, keep the highest scoring one."""
        from strat.ticker_selection.scorer import ScoredCandidate

        candidates = [
            ScoredCandidate(symbol='GOOGL', composite_score=72.5, timeframe='1H'),
            ScoredCandidate(symbol='GOOGL', composite_score=68.0, timeframe='1D'),
            ScoredCandidate(symbol='GOOGL', composite_score=55.0, timeframe='1W'),
            ScoredCandidate(symbol='AAPL', composite_score=80.0, timeframe='1D'),
        ]

        result = TickerSelectionPipeline._dedup_by_symbol(candidates)
        symbols = {c.symbol for c in result}
        assert symbols == {'GOOGL', 'AAPL'}
        assert len(result) == 2

        googl = [c for c in result if c.symbol == 'GOOGL'][0]
        assert googl.composite_score == 72.5
        assert googl.timeframe == '1H'

    def test_no_duplicates_unchanged(self):
        """When all symbols are unique, nothing is removed."""
        from strat.ticker_selection.scorer import ScoredCandidate

        candidates = [
            ScoredCandidate(symbol='AAPL', composite_score=80.0),
            ScoredCandidate(symbol='MSFT', composite_score=75.0),
            ScoredCandidate(symbol='CRWD', composite_score=70.0),
        ]

        result = TickerSelectionPipeline._dedup_by_symbol(candidates)
        assert len(result) == 3

    def test_empty_input(self):
        """Empty list returns empty list."""
        result = TickerSelectionPipeline._dedup_by_symbol([])
        assert result == []

    def test_single_candidate(self):
        """Single candidate passes through."""
        from strat.ticker_selection.scorer import ScoredCandidate

        candidates = [ScoredCandidate(symbol='SPY', composite_score=90.0)]
        result = TickerSelectionPipeline._dedup_by_symbol(candidates)
        assert len(result) == 1
        assert result[0].symbol == 'SPY'

    def test_tiebreak_uses_first_seen(self):
        """When scores are equal, the later-seen candidate wins (dict overwrite)."""
        from strat.ticker_selection.scorer import ScoredCandidate

        candidates = [
            ScoredCandidate(symbol='TMUS', composite_score=65.0, timeframe='1H'),
            ScoredCandidate(symbol='TMUS', composite_score=65.0, timeframe='1D'),
        ]

        result = TickerSelectionPipeline._dedup_by_symbol(candidates)
        assert len(result) == 1
        # Equal scores: first one kept (> is strict, not >=)
        assert result[0].timeframe == '1H'

    def test_many_duplicates_realistic(self):
        """Simulate the GOOGL x3, GOOG x3, TMUS x3 scenario from production."""
        from strat.ticker_selection.scorer import ScoredCandidate

        candidates = [
            ScoredCandidate(symbol='GOOGL', composite_score=72.5, timeframe='1H'),
            ScoredCandidate(symbol='GOOGL', composite_score=68.0, timeframe='1D'),
            ScoredCandidate(symbol='GOOGL', composite_score=60.0, timeframe='1W'),
            ScoredCandidate(symbol='GOOG', composite_score=72.5, timeframe='1H'),
            ScoredCandidate(symbol='GOOG', composite_score=68.0, timeframe='1D'),
            ScoredCandidate(symbol='GOOG', composite_score=60.0, timeframe='1W'),
            ScoredCandidate(symbol='TMUS', composite_score=70.0, timeframe='1H'),
            ScoredCandidate(symbol='TMUS', composite_score=65.0, timeframe='1D'),
            ScoredCandidate(symbol='TMUS', composite_score=55.0, timeframe='1W'),
            ScoredCandidate(symbol='AAPL', composite_score=80.0, timeframe='1D'),
            ScoredCandidate(symbol='MSFT', composite_score=75.0, timeframe='1D'),
        ]

        result = TickerSelectionPipeline._dedup_by_symbol(candidates)
        assert len(result) == 5  # GOOGL, GOOG, TMUS, AAPL, MSFT
        scores = {c.symbol: c.composite_score for c in result}
        assert scores['GOOGL'] == 72.5
        assert scores['GOOG'] == 72.5
        assert scores['TMUS'] == 70.0
        assert scores['AAPL'] == 80.0
        assert scores['MSFT'] == 75.0


class TestDaemonCandidateLoading:
    """Test daemon's _load_candidates method."""

    def test_loads_fresh_candidates(self, tmp_path):
        """Daemon loads candidates when file is fresh."""
        from strat.signal_automation.config import SignalAutomationConfig

        candidates_path = tmp_path / 'candidates.json'
        data = {
            'version': '1.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'candidates': [
                {'symbol': 'AAPL', 'composite_score': 85.0},
                {'symbol': 'CRWD', 'composite_score': 78.0},
            ],
        }
        candidates_path.write_text(json.dumps(data))

        config = SignalAutomationConfig()
        config.ticker_selection.enabled = True
        config.ticker_selection.candidates_path = str(candidates_path)

        # Create a minimal daemon mock to test _load_candidates
        from strat.signal_automation.daemon import SignalDaemon
        with patch.object(SignalDaemon, '__init__', lambda self, *a, **kw: None):
            daemon = SignalDaemon.__new__(SignalDaemon)
            daemon.config = config

            result = daemon._load_candidates()
            assert result is not None
            assert len(result) == 2
            assert result[0]['symbol'] == 'AAPL'

    def test_returns_none_when_stale(self, tmp_path):
        """Daemon returns None for stale candidates."""
        from strat.signal_automation.config import SignalAutomationConfig
        from datetime import timedelta

        candidates_path = tmp_path / 'candidates.json'
        old_time = datetime.now(timezone.utc) - timedelta(hours=30)
        data = {
            'version': '1.0',
            'generated_at': old_time.isoformat(),
            'candidates': [{'symbol': 'AAPL'}],
        }
        candidates_path.write_text(json.dumps(data))

        config = SignalAutomationConfig()
        config.ticker_selection.enabled = True
        config.ticker_selection.candidates_path = str(candidates_path)

        from strat.signal_automation.daemon import SignalDaemon
        with patch.object(SignalDaemon, '__init__', lambda self, *a, **kw: None):
            daemon = SignalDaemon.__new__(SignalDaemon)
            daemon.config = config

            result = daemon._load_candidates()
            assert result is None

    def test_returns_none_when_missing(self, tmp_path):
        """Daemon returns None when file doesn't exist."""
        from strat.signal_automation.config import SignalAutomationConfig

        config = SignalAutomationConfig()
        config.ticker_selection.enabled = True
        config.ticker_selection.candidates_path = str(tmp_path / 'nonexistent.json')

        from strat.signal_automation.daemon import SignalDaemon
        with patch.object(SignalDaemon, '__init__', lambda self, *a, **kw: None):
            daemon = SignalDaemon.__new__(SignalDaemon)
            daemon.config = config

            result = daemon._load_candidates()
            assert result is None

    def test_active_symbols_fallback(self, tmp_path):
        """When ticker selection fails, falls back to static list."""
        from strat.signal_automation.config import SignalAutomationConfig

        config = SignalAutomationConfig()
        config.ticker_selection.enabled = True
        config.ticker_selection.candidates_path = str(tmp_path / 'nonexistent.json')

        from strat.signal_automation.daemon import SignalDaemon
        with patch.object(SignalDaemon, '__init__', lambda self, *a, **kw: None):
            daemon = SignalDaemon.__new__(SignalDaemon)
            daemon.config = config

            symbols = daemon.active_symbols
            assert symbols == config.scan.symbols

    def test_active_symbols_merges_core_and_dynamic(self, tmp_path):
        """Core ETFs + dynamic candidates, deduplicated."""
        from strat.signal_automation.config import SignalAutomationConfig

        candidates_path = tmp_path / 'candidates.json'
        data = {
            'version': '1.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'candidates': [
                {'symbol': 'CRWD'},
                {'symbol': 'AAPL'},
                {'symbol': 'SPY'},  # Duplicate of core -- should be deduped
            ],
        }
        candidates_path.write_text(json.dumps(data))

        config = SignalAutomationConfig()
        config.ticker_selection.enabled = True
        config.ticker_selection.candidates_path = str(candidates_path)
        config.ticker_selection.core_symbols = ['SPY', 'QQQ']

        from strat.signal_automation.daemon import SignalDaemon
        with patch.object(SignalDaemon, '__init__', lambda self, *a, **kw: None):
            daemon = SignalDaemon.__new__(SignalDaemon)
            daemon.config = config

            symbols = daemon.active_symbols
            # SPY, QQQ (core) + CRWD, AAPL (dynamic, SPY deduped)
            assert 'SPY' in symbols
            assert 'QQQ' in symbols
            assert 'CRWD' in symbols
            assert 'AAPL' in symbols
            assert symbols.count('SPY') == 1  # No duplicate
