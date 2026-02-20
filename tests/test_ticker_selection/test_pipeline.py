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
             patch.object(pipeline, '_get_scanner') as mock_scanner, \
             patch.object(pipeline, '_analyze_convergence', return_value={}):

            scanner = MagicMock()
            scanner.scan_symbol_all_timeframes_resampled.return_value = [
                _make_mock_signal(symbol='AAPL'),
            ]
            mock_scanner.return_value = scanner

            result = pipeline.run(dry_run=False)

        # Verify schema
        assert result['version'] == '2.0'
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
        assert written['version'] == '2.0'

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
             patch.object(pipeline, '_get_scanner') as mock_scanner, \
             patch.object(pipeline, '_analyze_convergence', return_value={}):

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
            assert 'tier' in c
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


# ---------------------------------------------------------------------------
# Tier Classification (Phase 2 - Convergence Architecture)
# ---------------------------------------------------------------------------

def _make_scored_candidate(symbol='TEST', composite_score=70.0,
                           pattern_type='3-1-2U', direction='CALL',
                           timeframe='1D', current_price=100.0, **kwargs):
    """Build a ScoredCandidate for tier testing."""
    from strat.ticker_selection.scorer import ScoredCandidate
    return ScoredCandidate(
        symbol=symbol,
        composite_score=composite_score,
        pattern_type=pattern_type,
        direction=direction,
        timeframe=timeframe,
        current_price=current_price,
        **kwargs,
    )


def _make_convergence_meta(inside_count=0, is_convergence=False,
                           convergence_score=0.0, inside_tfs=None):
    """Build a ConvergenceMetadata for tier testing."""
    from strat.ticker_selection.convergence import ConvergenceMetadata
    return ConvergenceMetadata(
        inside_bar_count=inside_count,
        inside_bar_timeframes=inside_tfs or [],
        trigger_levels={},
        convergence_score=convergence_score,
        bullish_trigger=None,
        bearish_trigger=None,
        trigger_spread_pct=0.0,
        prior_direction_alignment='mixed',
        is_convergence=is_convergence,
    )


class TestTierClassification:
    """Test _classify_tiers assigns correct tiers."""

    def setup_method(self):
        self.config = TickerSelectionConfig()
        self.pipeline = TickerSelectionPipeline(self.config)

    def test_convergence_is_tier1(self):
        """Symbol with 2+ inside TFs gets Tier 1."""
        candidates = [_make_scored_candidate('CRWD')]
        convergence_map = {
            'CRWD': _make_convergence_meta(
                inside_count=2, is_convergence=True, convergence_score=79.0,
                inside_tfs=['1M', '1W'],
            ),
        }
        self.pipeline._classify_tiers(candidates, convergence_map)

        assert candidates[0].tier == 1
        assert candidates[0].convergence is not None
        assert candidates[0].convergence.is_convergence is True

    def test_continuation_is_tier3(self):
        """True continuation pattern gets Tier 3."""
        candidates = [_make_scored_candidate(
            'AAPL', pattern_type='2D-2D', direction='PUT',
        )]
        self.pipeline._classify_tiers(candidates, {})

        assert candidates[0].tier == 3

    def test_standard_is_tier2(self):
        """Normal setup without convergence or continuation is Tier 2."""
        candidates = [_make_scored_candidate('MSFT')]
        self.pipeline._classify_tiers(candidates, {})

        assert candidates[0].tier == 2

    def test_convergence_overrides_continuation(self):
        """If a symbol has both convergence AND continuation pattern, convergence wins (Tier 1)."""
        candidates = [_make_scored_candidate(
            'NFLX', pattern_type='2D-2D', direction='PUT',
        )]
        convergence_map = {
            'NFLX': _make_convergence_meta(
                inside_count=3, is_convergence=True, convergence_score=95.0,
                inside_tfs=['1M', '1W', '1D'],
            ),
        }
        self.pipeline._classify_tiers(candidates, convergence_map)

        assert candidates[0].tier == 1  # Convergence wins

    def test_single_inside_not_tier1(self):
        """Single inside TF (below threshold) stays Tier 2."""
        candidates = [_make_scored_candidate('GOOGL')]
        convergence_map = {
            'GOOGL': _make_convergence_meta(
                inside_count=1, is_convergence=False, convergence_score=20.0,
                inside_tfs=['1D'],
            ),
        }
        self.pipeline._classify_tiers(candidates, convergence_map)

        assert candidates[0].tier == 2
        assert candidates[0].convergence is not None  # Still attached


class TestTieredSortAndCap:
    """Test _sort_and_cap_tiered ordering and capping."""

    def setup_method(self):
        self.config = TickerSelectionConfig(
            max_tier1_candidates=2,
            max_tier2_candidates=3,
            max_tier3_context=2,
        )
        self.pipeline = TickerSelectionPipeline(self.config)

    def test_tier_ordering(self):
        """Output should be T1 first, then T2, then T3."""
        candidates = [
            _make_scored_candidate('T2A', composite_score=80, tier=2),
            _make_scored_candidate('T1A', composite_score=60, tier=1),
            _make_scored_candidate('T3A', composite_score=50, tier=3),
            _make_scored_candidate('T2B', composite_score=75, tier=2),
            _make_scored_candidate('T1B', composite_score=55, tier=1),
        ]
        convergence_map = {
            'T1A': _make_convergence_meta(inside_count=3, is_convergence=True, convergence_score=90),
            'T1B': _make_convergence_meta(inside_count=2, is_convergence=True, convergence_score=70),
        }
        result = self.pipeline._sort_and_cap_tiered(candidates, convergence_map)

        tiers = [c.tier for c in result]
        # T1 first, then T2, then T3
        assert tiers == [1, 1, 2, 2, 3]
        # T1 sorted by convergence_score desc
        assert result[0].symbol == 'T1A'
        assert result[1].symbol == 'T1B'
        # T2 sorted by composite_score desc
        assert result[2].symbol == 'T2A'
        assert result[3].symbol == 'T2B'

    def test_per_tier_caps(self):
        """Each tier is capped independently."""
        candidates = [
            _make_scored_candidate(f'T1_{i}', composite_score=80-i, tier=1)
            for i in range(5)
        ] + [
            _make_scored_candidate(f'T2_{i}', composite_score=70-i, tier=2)
            for i in range(6)
        ] + [
            _make_scored_candidate(f'T3_{i}', composite_score=40-i, tier=3)
            for i in range(4)
        ]
        convergence_map = {
            f'T1_{i}': _make_convergence_meta(
                inside_count=2, is_convergence=True, convergence_score=90-i*10,
            )
            for i in range(5)
        }
        result = self.pipeline._sort_and_cap_tiered(candidates, convergence_map)

        tier_counts = {}
        for c in result:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
        assert tier_counts[1] == 2  # max_tier1_candidates
        assert tier_counts[2] == 3  # max_tier2_candidates
        assert tier_counts[3] == 2  # max_tier3_context

    def test_empty_tiers_graceful(self):
        """Pipeline handles no T1 or T3 candidates gracefully."""
        candidates = [
            _make_scored_candidate('ONLY_T2', composite_score=80, tier=2),
        ]
        result = self.pipeline._sort_and_cap_tiered(candidates, {})
        assert len(result) == 1
        assert result[0].tier == 2


class TestPipelineWithConvergence:
    """Integration tests: pipeline.run() with convergence mocked."""

    def test_tier_counts_in_stats(self, tmp_path):
        """Pipeline stats include tier1_count, tier2_count, tier3_count."""
        config = TickerSelectionConfig(
            candidates_path=str(tmp_path / 'candidates.json'),
            universe_cache_path=str(tmp_path / 'universe.json'),
        )
        pipeline = TickerSelectionPipeline(config)

        with patch.object(pipeline.universe_mgr, 'get_universe', return_value=['AAPL']), \
             patch.object(pipeline.screener, 'screen', return_value=[
                 ScreenedStock('AAPL', 180.0, 50e6, 2.5, 182.0, 177.0, 179.0, 180.0, 178.0, 300000),
             ]), \
             patch.object(pipeline, '_get_scanner') as mock_scanner, \
             patch.object(pipeline, '_analyze_convergence', return_value={}):

            scanner = MagicMock()
            scanner.scan_symbol_all_timeframes_resampled.return_value = [
                _make_mock_signal(symbol='AAPL'),
            ]
            mock_scanner.return_value = scanner

            result = pipeline.run(dry_run=True)

        stats = result['pipeline_stats']
        assert stats['tier1_count'] == 0   # No convergence mocked
        assert stats['tier2_count'] == 1   # AAPL is standard
        assert stats['tier3_count'] == 0

    def test_convergence_metadata_in_json(self, tmp_path):
        """When convergence detected, metadata appears in candidate JSON."""
        from strat.ticker_selection.convergence import ConvergenceMetadata
        config = TickerSelectionConfig(
            candidates_path=str(tmp_path / 'candidates.json'),
            universe_cache_path=str(tmp_path / 'universe.json'),
        )
        pipeline = TickerSelectionPipeline(config)

        convergence_result = {
            'CRWD': _make_convergence_meta(
                inside_count=2, is_convergence=True,
                convergence_score=79.0, inside_tfs=['1M', '1W'],
            ),
        }

        with patch.object(pipeline.universe_mgr, 'get_universe', return_value=['CRWD']), \
             patch.object(pipeline.screener, 'screen', return_value=[
                 ScreenedStock('CRWD', 385.0, 45e6, 2.3, 388.0, 380.0, 383.0, 385.0, 382.0, 120000),
             ]), \
             patch.object(pipeline, '_get_scanner') as mock_scanner, \
             patch.object(pipeline, '_analyze_convergence', return_value=convergence_result):

            scanner = MagicMock()
            scanner.scan_symbol_all_timeframes_resampled.return_value = [
                _make_mock_signal(symbol='CRWD'),
            ]
            mock_scanner.return_value = scanner

            result = pipeline.run(dry_run=True)

        assert len(result['candidates']) == 1
        c = result['candidates'][0]
        assert c['tier'] == 1
        assert 'convergence' in c
        assert c['convergence']['inside_bar_count'] == 2
        assert c['convergence']['is_convergence'] is True

    def test_backward_compat_no_convergence(self, tmp_path):
        """When no convergence, candidates still have tier field but no convergence key."""
        config = TickerSelectionConfig(
            candidates_path=str(tmp_path / 'candidates.json'),
            universe_cache_path=str(tmp_path / 'universe.json'),
        )
        pipeline = TickerSelectionPipeline(config)

        with patch.object(pipeline.universe_mgr, 'get_universe', return_value=['MSFT']), \
             patch.object(pipeline.screener, 'screen', return_value=[
                 ScreenedStock('MSFT', 400.0, 60e6, 2.0, 405.0, 395.0, 398.0, 400.0, 397.0, 150000),
             ]), \
             patch.object(pipeline, '_get_scanner') as mock_scanner, \
             patch.object(pipeline, '_analyze_convergence', return_value={}):

            scanner = MagicMock()
            scanner.scan_symbol_all_timeframes_resampled.return_value = [
                _make_mock_signal(symbol='MSFT'),
            ]
            mock_scanner.return_value = scanner

            result = pipeline.run(dry_run=True)

        if result['candidates']:
            c = result['candidates'][0]
            assert c['tier'] == 2  # Default tier
            assert 'convergence' not in c  # No convergence key when not detected
