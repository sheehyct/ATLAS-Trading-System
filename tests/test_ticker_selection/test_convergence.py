"""Unit tests for multi-TF convergence detection engine."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from strat.ticker_selection.convergence import (
    ConvergenceAnalyzer,
    ConvergenceMetadata,
    InsideBarInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inside_bar_info(timeframe='1D', is_inside=True,
                          prior_high=100.0, prior_low=90.0,
                          prior_dir='U'):
    """Build an InsideBarInfo for testing."""
    return InsideBarInfo(
        timeframe=timeframe,
        is_inside=is_inside,
        prior_bar_high=prior_high,
        prior_bar_low=prior_low,
        prior_bar_direction=prior_dir,
    )


def _make_bars_df(highs, lows, opens=None, closes=None):
    """Build a minimal OHLC DataFrame matching Alpaca bar format."""
    n = len(highs)
    if opens is None:
        opens = lows  # simplification
    if closes is None:
        closes = highs  # simplification
    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': [1_000_000] * n,
    })


@pytest.fixture
def analyzer():
    """Create a ConvergenceAnalyzer with mocked Alpaca client."""
    a = ConvergenceAnalyzer.__new__(ConvergenceAnalyzer)
    a._alpaca_account = 'SMALL'
    a._client = MagicMock()
    return a


# ---------------------------------------------------------------------------
# InsideBarInfo dataclass
# ---------------------------------------------------------------------------

class TestInsideBarInfo:
    def test_fields(self):
        info = _make_inside_bar_info()
        assert info.timeframe == '1D'
        assert info.is_inside is True
        assert info.prior_bar_high == 100.0
        assert info.prior_bar_low == 90.0
        assert info.prior_bar_direction == 'U'


# ---------------------------------------------------------------------------
# ConvergenceMetadata
# ---------------------------------------------------------------------------

class TestConvergenceMetadata:
    def test_to_dict_keys(self):
        meta = ConvergenceMetadata(
            inside_bar_count=2,
            inside_bar_timeframes=['1M', '1W'],
            trigger_levels={'1M': {'high': 400.0, 'low': 380.0}},
            convergence_score=75.0,
            bullish_trigger=400.0,
            bearish_trigger=380.0,
            trigger_spread_pct=5.13,
            prior_direction_alignment='bearish',
            is_convergence=True,
        )
        d = meta.to_dict()
        assert set(d.keys()) == {
            'inside_bar_count', 'inside_bar_timeframes', 'trigger_levels',
            'convergence_score', 'bullish_trigger', 'bearish_trigger',
            'trigger_spread_pct', 'prior_direction_alignment', 'is_convergence',
        }
        # bar_states and tier should NOT be in serialized output
        assert 'bar_states' not in d
        assert 'tier' not in d

    def test_to_dict_rounding(self):
        meta = ConvergenceMetadata(
            inside_bar_count=1,
            inside_bar_timeframes=['1D'],
            trigger_levels={},
            convergence_score=33.333,
            bullish_trigger=100.0,
            bearish_trigger=90.0,
            trigger_spread_pct=1.2345,
            prior_direction_alignment='mixed',
            is_convergence=False,
        )
        d = meta.to_dict()
        assert d['convergence_score'] == 33.3
        assert d['trigger_spread_pct'] == 1.23


# ---------------------------------------------------------------------------
# _build_metadata -- core logic tests (no API calls)
# ---------------------------------------------------------------------------

class TestBuildMetadata:
    """Test _build_metadata directly by supplying bar states."""

    def test_no_inside_bars(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=False),
            _make_inside_bar_info('1W', is_inside=False),
            _make_inside_bar_info('1D', is_inside=False),
        ]
        meta = analyzer._build_metadata(states, current_price=100.0)

        assert meta.inside_bar_count == 0
        assert meta.inside_bar_timeframes == []
        assert meta.is_convergence is False
        assert meta.bullish_trigger is None
        assert meta.bearish_trigger is None
        assert meta.convergence_score == 0.0

    def test_single_inside_bar(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=False),
            _make_inside_bar_info('1W', is_inside=False),
            _make_inside_bar_info('1D', is_inside=True, prior_high=102.0, prior_low=97.0),
        ]
        meta = analyzer._build_metadata(states, current_price=100.0)

        assert meta.inside_bar_count == 1
        assert meta.inside_bar_timeframes == ['1D']
        assert meta.is_convergence is False
        assert meta.bullish_trigger == 102.0
        assert meta.bearish_trigger == 97.0

    def test_double_inside_convergence(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=False),
            _make_inside_bar_info('1W', is_inside=True, prior_high=395.0, prior_low=378.0, prior_dir='D'),
            _make_inside_bar_info('1D', is_inside=True, prior_high=388.0, prior_low=382.0, prior_dir='D'),
        ]
        meta = analyzer._build_metadata(states, current_price=385.0)

        assert meta.inside_bar_count == 2
        assert meta.inside_bar_timeframes == ['1W', '1D']
        assert meta.is_convergence is True
        # Bullish trigger = highest high across inside TFs
        assert meta.bullish_trigger == 395.0
        # Bearish trigger = lowest low across inside TFs
        assert meta.bearish_trigger == 378.0

    def test_triple_inside_max_convergence(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=True, prior_high=400.0, prior_low=370.0, prior_dir='D'),
            _make_inside_bar_info('1W', is_inside=True, prior_high=395.0, prior_low=378.0, prior_dir='D'),
            _make_inside_bar_info('1D', is_inside=True, prior_high=388.0, prior_low=382.0, prior_dir='D'),
        ]
        meta = analyzer._build_metadata(states, current_price=385.0)

        assert meta.inside_bar_count == 3
        assert meta.is_convergence is True
        assert meta.bullish_trigger == 400.0
        assert meta.bearish_trigger == 370.0

    def test_trigger_levels_per_tf(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=True, prior_high=400.0, prior_low=370.0),
            _make_inside_bar_info('1W', is_inside=True, prior_high=395.0, prior_low=378.0),
            _make_inside_bar_info('1D', is_inside=False),
        ]
        meta = analyzer._build_metadata(states, current_price=385.0)

        assert '1M' in meta.trigger_levels
        assert meta.trigger_levels['1M'] == {'high': 400.0, 'low': 370.0}
        assert '1W' in meta.trigger_levels
        assert meta.trigger_levels['1W'] == {'high': 395.0, 'low': 378.0}
        assert '1D' not in meta.trigger_levels

    def test_trigger_spread_pct(self, analyzer):
        states = [
            _make_inside_bar_info('1W', is_inside=True, prior_high=105.0, prior_low=95.0),
            _make_inside_bar_info('1D', is_inside=True, prior_high=103.0, prior_low=97.0),
            _make_inside_bar_info('1M', is_inside=False),
        ]
        # bullish=105, bearish=95, spread=10, price=100 -> 10%
        meta = analyzer._build_metadata(states, current_price=100.0)
        assert meta.trigger_spread_pct == pytest.approx(10.0, abs=0.01)

    def test_trigger_spread_zero_price(self, analyzer):
        """When current_price=0, spread should be 0 (avoid division by zero)."""
        states = [
            _make_inside_bar_info('1D', is_inside=True, prior_high=100.0, prior_low=90.0),
            _make_inside_bar_info('1W', is_inside=False),
            _make_inside_bar_info('1M', is_inside=False),
        ]
        meta = analyzer._build_metadata(states, current_price=0.0)
        assert meta.trigger_spread_pct == 0.0


# ---------------------------------------------------------------------------
# Direction alignment
# ---------------------------------------------------------------------------

class TestDirectionAlignment:
    def test_all_bullish(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=True, prior_dir='U'),
            _make_inside_bar_info('1W', is_inside=True, prior_dir='U'),
            _make_inside_bar_info('1D', is_inside=True, prior_dir='U'),
        ]
        assert analyzer._assess_direction_alignment(states) == 'bullish'

    def test_all_bearish(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=True, prior_dir='D'),
            _make_inside_bar_info('1W', is_inside=True, prior_dir='D'),
        ]
        assert analyzer._assess_direction_alignment(states) == 'bearish'

    def test_mixed(self, analyzer):
        states = [
            _make_inside_bar_info('1M', is_inside=True, prior_dir='U'),
            _make_inside_bar_info('1W', is_inside=True, prior_dir='D'),
        ]
        assert analyzer._assess_direction_alignment(states) == 'mixed'

    def test_neutral_prior_bars(self, analyzer):
        """If all prior bars are outside/reference (N), alignment is mixed."""
        states = [
            _make_inside_bar_info('1M', is_inside=True, prior_dir='N'),
            _make_inside_bar_info('1W', is_inside=True, prior_dir='N'),
        ]
        assert analyzer._assess_direction_alignment(states) == 'mixed'

    def test_no_inside_bars_is_mixed(self, analyzer):
        states = [
            _make_inside_bar_info('1D', is_inside=False, prior_dir='U'),
        ]
        assert analyzer._assess_direction_alignment(states) == 'mixed'

    def test_partial_neutral_still_aligned(self, analyzer):
        """One U + one N = only 'U' is directional -> bullish."""
        states = [
            _make_inside_bar_info('1M', is_inside=True, prior_dir='U'),
            _make_inside_bar_info('1W', is_inside=True, prior_dir='N'),
        ]
        assert analyzer._assess_direction_alignment(states) == 'bullish'


# ---------------------------------------------------------------------------
# Scoring formula
# ---------------------------------------------------------------------------

class TestConvergenceScoring:
    def test_zero_inside_bars_score_zero(self, analyzer):
        score = analyzer._calculate_score(0, 0.0, 'mixed')
        assert score == 0.0

    def test_single_inside_low_score(self, analyzer):
        # 1 inside = 20 * 0.50 = 10
        # spread 5% -> proximity 20 * 0.30 = 6
        # mixed -> 30 * 0.20 = 6
        # total = 22
        score = analyzer._calculate_score(1, 5.0, 'mixed')
        assert score == pytest.approx(22.0, abs=0.1)

    def test_double_inside_moderate_score(self, analyzer):
        # 2 inside = 70 * 0.50 = 35
        # spread 1.5% -> proximity 80 * 0.30 = 24
        # bearish aligned -> 100 * 0.20 = 20
        # total = 79
        score = analyzer._calculate_score(2, 1.5, 'bearish')
        assert score == pytest.approx(79.0, abs=0.1)

    def test_triple_inside_tight_aligned_max_score(self, analyzer):
        # 3 inside = 100 * 0.50 = 50
        # spread 0.5% -> proximity 100 * 0.30 = 30
        # bullish aligned -> 100 * 0.20 = 20
        # total = 100
        score = analyzer._calculate_score(3, 0.5, 'bullish')
        assert score == pytest.approx(100.0, abs=0.1)

    def test_score_always_in_range(self, analyzer):
        """Score should always be 0-100 regardless of inputs."""
        for count in range(4):
            for spread in [0.0, 0.5, 1.5, 3.0, 6.0]:
                for alignment in ['bullish', 'bearish', 'mixed']:
                    score = analyzer._calculate_score(count, spread, alignment)
                    assert 0 <= score <= 100, (
                        f"Out of range: count={count}, spread={spread}, "
                        f"alignment={alignment}, score={score}"
                    )


# ---------------------------------------------------------------------------
# _get_bar_state with mocked bar data
# ---------------------------------------------------------------------------

class TestGetBarState:
    def test_inside_bar_detected(self, analyzer):
        """Last bar is inside (H <= prev_H and L >= prev_L)."""
        # Bar 0: H=100 L=90 (ref)
        # Bar 1: H=105 L=88 (3/outside)
        # Bar 2: H=103 L=89 (1/inside relative to bar 1)
        bars_df = _make_bars_df(
            highs=[100, 105, 103],
            lows=[90, 88, 89],
        )
        with patch.object(analyzer, '_fetch_bars', return_value=bars_df):
            state = analyzer._get_bar_state('TEST', '1D')

        assert state.is_inside is True
        assert state.timeframe == '1D'
        # Prior bar (bar 1) H/L define trigger levels
        assert state.prior_bar_high == 105.0
        assert state.prior_bar_low == 88.0

    def test_non_inside_bar(self, analyzer):
        """Last bar breaks out (2U)."""
        # Bar 0: H=100 L=90 (ref)
        # Bar 1: H=95  L=92 (1/inside)
        # Bar 2: H=101 L=93 (2U, breaks bar 1 high)
        bars_df = _make_bars_df(
            highs=[100, 95, 101],
            lows=[90, 92, 93],
        )
        with patch.object(analyzer, '_fetch_bars', return_value=bars_df):
            state = analyzer._get_bar_state('TEST', '1D')

        assert state.is_inside is False

    def test_prior_bar_direction_2u(self, analyzer):
        """Prior bar classified as 2U -> direction 'U'."""
        # Bar 0: H=100 L=90 (ref)
        # Bar 1: H=105 L=92 (2U - breaks bar 0 high only)
        # Bar 2: H=103 L=93 (inside - within bar 1)
        bars_df = _make_bars_df(
            highs=[100, 105, 103],
            lows=[90, 92, 93],
        )
        with patch.object(analyzer, '_fetch_bars', return_value=bars_df):
            state = analyzer._get_bar_state('TEST', '1D')

        assert state.is_inside is True
        assert state.prior_bar_direction == 'U'

    def test_prior_bar_direction_2d(self, analyzer):
        """Prior bar classified as 2D -> direction 'D'."""
        # Bar 0: H=100 L=90 (ref)
        # Bar 1: H=99  L=87 (2D - breaks bar 0 low only)
        # Bar 2: H=98  L=88 (inside - within bar 1)
        bars_df = _make_bars_df(
            highs=[100, 99, 98],
            lows=[90, 87, 88],
        )
        with patch.object(analyzer, '_fetch_bars', return_value=bars_df):
            state = analyzer._get_bar_state('TEST', '1D')

        assert state.is_inside is True
        assert state.prior_bar_direction == 'D'

    def test_insufficient_bars(self, analyzer):
        """With fewer than 3 bars, return non-inside default."""
        bars_df = _make_bars_df(highs=[100, 105], lows=[90, 92])
        with patch.object(analyzer, '_fetch_bars', return_value=bars_df):
            state = analyzer._get_bar_state('TEST', '1D')

        assert state.is_inside is False
        assert state.prior_bar_direction == 'N'

    def test_none_bars_returns_default(self, analyzer):
        """If fetch returns None, return non-inside default."""
        with patch.object(analyzer, '_fetch_bars', return_value=None):
            state = analyzer._get_bar_state('TEST', '1M')

        assert state.is_inside is False


# ---------------------------------------------------------------------------
# analyze_symbol with mocked _get_bar_state
# ---------------------------------------------------------------------------

class TestAnalyzeSymbol:
    def test_full_convergence_analysis(self, analyzer):
        """End-to-end with mocked bar states."""
        def mock_bar_state(symbol, tf):
            states = {
                '1M': _make_inside_bar_info('1M', is_inside=True,
                                            prior_high=400, prior_low=370, prior_dir='D'),
                '1W': _make_inside_bar_info('1W', is_inside=True,
                                            prior_high=395, prior_low=378, prior_dir='D'),
                '1D': _make_inside_bar_info('1D', is_inside=False,
                                            prior_high=388, prior_low=382, prior_dir='U'),
            }
            return states[tf]

        with patch.object(analyzer, '_get_bar_state', side_effect=mock_bar_state):
            meta = analyzer.analyze_symbol('CRWD', current_price=385.0)

        assert meta.inside_bar_count == 2
        assert meta.is_convergence is True
        assert meta.bullish_trigger == 400.0
        assert meta.bearish_trigger == 370.0
        assert meta.prior_direction_alignment == 'bearish'
        assert meta.convergence_score > 0

    def test_bar_state_error_handled(self, analyzer):
        """If one TF fails, it should still analyze the others."""
        call_count = 0

        def mock_bar_state(symbol, tf):
            nonlocal call_count
            call_count += 1
            if tf == '1M':
                raise RuntimeError("API error")
            return _make_inside_bar_info(tf, is_inside=True,
                                        prior_high=100, prior_low=90, prior_dir='U')

        with patch.object(analyzer, '_get_bar_state', side_effect=mock_bar_state):
            meta = analyzer.analyze_symbol('TEST', current_price=95.0)

        assert call_count == 3  # all 3 TFs attempted
        assert meta.inside_bar_count == 2  # 1W and 1D still detected
        assert meta.is_convergence is True


# ---------------------------------------------------------------------------
# analyze_batch
# ---------------------------------------------------------------------------

class TestAnalyzeBatch:
    def test_batch_analysis(self, analyzer):
        """Batch wrapper calls analyze_symbol for each."""
        def mock_analyze(symbol, current_price=0.0):
            return ConvergenceMetadata(
                inside_bar_count=1 if symbol == 'AAPL' else 0,
                inside_bar_timeframes=['1D'] if symbol == 'AAPL' else [],
                trigger_levels={},
                convergence_score=20.0 if symbol == 'AAPL' else 0.0,
                bullish_trigger=None,
                bearish_trigger=None,
                trigger_spread_pct=0.0,
                prior_direction_alignment='mixed',
                is_convergence=False,
            )

        with patch.object(analyzer, 'analyze_symbol', side_effect=mock_analyze):
            results = analyzer.analyze_batch(
                ['AAPL', 'MSFT'],
                prices={'AAPL': 180.0, 'MSFT': 400.0},
            )

        assert 'AAPL' in results
        assert 'MSFT' in results
        assert results['AAPL'].inside_bar_count == 1

    def test_batch_error_does_not_stop_others(self, analyzer):
        """If one symbol fails, others should still be analyzed."""
        call_count = 0

        def mock_analyze(symbol, current_price=0.0):
            nonlocal call_count
            call_count += 1
            if symbol == 'BAD':
                raise RuntimeError("Data error")
            return ConvergenceMetadata(
                inside_bar_count=0, inside_bar_timeframes=[],
                trigger_levels={}, convergence_score=0.0,
                bullish_trigger=None, bearish_trigger=None,
                trigger_spread_pct=0.0, prior_direction_alignment='mixed',
                is_convergence=False,
            )

        with patch.object(analyzer, 'analyze_symbol', side_effect=mock_analyze):
            results = analyzer.analyze_batch(['GOOD', 'BAD', 'ALSO_GOOD'])

        assert call_count == 3
        assert 'GOOD' in results
        assert 'BAD' not in results  # failed, excluded
        assert 'ALSO_GOOD' in results
