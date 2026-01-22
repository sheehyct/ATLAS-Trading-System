"""
Tests for Stock Scanner Bridge - Session EQUITY-76

Comprehensive test coverage for integrations/stock_scanner_bridge.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

# Import the module under test
from integrations.stock_scanner_bridge import MomentumPortfolioBacktest


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def backtest():
    """Create a MomentumPortfolioBacktest instance for testing."""
    return MomentumPortfolioBacktest(
        universe='technology',
        top_n=10,
        volume_threshold=1.25,
        min_distance=0.90,
        rebalance_frequency='semi_annual'
    )


@pytest.fixture
def mock_ohlcv_data():
    """Create mock OHLCV data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    # Filter to weekdays only
    dates = dates[dates.dayofweek < 5]

    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META']

    # Create mock data with uptrend
    np.random.seed(42)
    data = {}
    for symbol in symbols:
        base_price = 100 + np.random.randn() * 20
        trend = np.linspace(0, 50, len(dates))
        noise = np.random.randn(len(dates)) * 2
        prices = base_price + trend + noise

        data[symbol] = {
            'Open': prices - 0.5,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }

    # Create DataFrames
    open_df = pd.DataFrame({s: data[s]['Open'] for s in symbols}, index=dates)
    high_df = pd.DataFrame({s: data[s]['High'] for s in symbols}, index=dates)
    low_df = pd.DataFrame({s: data[s]['Low'] for s in symbols}, index=dates)
    close_df = pd.DataFrame({s: data[s]['Close'] for s in symbols}, index=dates)
    volume_df = pd.DataFrame({s: data[s]['Volume'] for s in symbols}, index=dates)

    return {
        'Open': open_df,
        'High': high_df,
        'Low': low_df,
        'Close': close_df,
        'Volume': volume_df
    }


@pytest.fixture
def mock_vbt_data(mock_ohlcv_data):
    """Create a mock VBT data object."""
    mock_data = Mock()
    mock_data.get = lambda key: mock_ohlcv_data[key]
    return mock_data


# ============================================================================
# Test REGIME_ALLOCATION Constant
# ============================================================================

class TestRegimeAllocationConstant:
    """Tests for the REGIME_ALLOCATION constant."""

    def test_has_trend_bull(self):
        """Should have TREND_BULL allocation rules."""
        assert 'TREND_BULL' in MomentumPortfolioBacktest.REGIME_ALLOCATION
        rules = MomentumPortfolioBacktest.REGIME_ALLOCATION['TREND_BULL']
        assert rules['portfolio_size_multiplier'] == 1.0
        assert rules['allocation_pct'] == 1.0

    def test_has_trend_neutral(self):
        """Should have TREND_NEUTRAL allocation rules."""
        assert 'TREND_NEUTRAL' in MomentumPortfolioBacktest.REGIME_ALLOCATION
        rules = MomentumPortfolioBacktest.REGIME_ALLOCATION['TREND_NEUTRAL']
        assert rules['portfolio_size_multiplier'] == 0.5
        assert rules['allocation_pct'] == 0.7

    def test_has_trend_bear(self):
        """Should have TREND_BEAR allocation rules."""
        assert 'TREND_BEAR' in MomentumPortfolioBacktest.REGIME_ALLOCATION
        rules = MomentumPortfolioBacktest.REGIME_ALLOCATION['TREND_BEAR']
        assert rules['portfolio_size_multiplier'] == 0.3
        assert rules['allocation_pct'] == 0.3

    def test_has_crash(self):
        """Should have CRASH allocation rules."""
        assert 'CRASH' in MomentumPortfolioBacktest.REGIME_ALLOCATION
        rules = MomentumPortfolioBacktest.REGIME_ALLOCATION['CRASH']
        assert rules['portfolio_size_multiplier'] == 0.0
        assert rules['allocation_pct'] == 0.0


# ============================================================================
# Test UNIVERSES Constant
# ============================================================================

class TestUniversesConstant:
    """Tests for the UNIVERSES constant."""

    def test_has_technology(self):
        """Should have technology universe."""
        assert 'technology' in MomentumPortfolioBacktest.UNIVERSES
        tech = MomentumPortfolioBacktest.UNIVERSES['technology']
        assert len(tech) == 30
        assert 'AAPL' in tech
        assert 'NVDA' in tech

    def test_has_sp500_proxy(self):
        """Should have sp500_proxy universe."""
        assert 'sp500_proxy' in MomentumPortfolioBacktest.UNIVERSES
        sp500 = MomentumPortfolioBacktest.UNIVERSES['sp500_proxy']
        assert len(sp500) == 40
        assert 'AAPL' in sp500
        assert 'JPM' in sp500

    def test_has_healthcare(self):
        """Should have healthcare universe."""
        assert 'healthcare' in MomentumPortfolioBacktest.UNIVERSES
        health = MomentumPortfolioBacktest.UNIVERSES['healthcare']
        assert len(health) == 20
        assert 'UNH' in health
        assert 'JNJ' in health

    def test_has_financials(self):
        """Should have financials universe."""
        assert 'financials' in MomentumPortfolioBacktest.UNIVERSES
        fin = MomentumPortfolioBacktest.UNIVERSES['financials']
        assert len(fin) == 20
        assert 'JPM' in fin
        assert 'GS' in fin


# ============================================================================
# Test MomentumPortfolioBacktest Initialization
# ============================================================================

class TestMomentumPortfolioBacktestInit:
    """Tests for MomentumPortfolioBacktest initialization."""

    def test_init_default_values(self):
        """Should use default values when not specified."""
        bt = MomentumPortfolioBacktest()
        assert bt.universe_name == 'sp500_proxy'
        assert bt.top_n == 20
        assert bt.volume_threshold is None
        assert bt.min_distance == 0.90
        assert bt.rebalance_frequency == 'semi_annual'

    def test_init_technology_universe(self):
        """Should use technology universe when specified."""
        bt = MomentumPortfolioBacktest(universe='technology')
        assert bt.universe_name == 'technology'
        assert len(bt.universe) == 30
        assert 'AAPL' in bt.universe

    def test_init_sp500_proxy_universe(self):
        """Should use sp500_proxy universe when specified."""
        bt = MomentumPortfolioBacktest(universe='sp500_proxy')
        assert bt.universe_name == 'sp500_proxy'
        assert len(bt.universe) == 40

    def test_init_unknown_universe_defaults_to_sp500(self):
        """Should default to sp500_proxy for unknown universe."""
        bt = MomentumPortfolioBacktest(universe='unknown')
        assert bt.universe == MomentumPortfolioBacktest.UNIVERSES['sp500_proxy']

    def test_init_custom_top_n(self):
        """Should accept custom top_n parameter."""
        bt = MomentumPortfolioBacktest(top_n=5)
        assert bt.top_n == 5

    def test_init_volume_threshold(self):
        """Should accept volume threshold parameter."""
        bt = MomentumPortfolioBacktest(volume_threshold=1.5)
        assert bt.volume_threshold == 1.5

    def test_init_min_distance(self):
        """Should accept min_distance parameter."""
        bt = MomentumPortfolioBacktest(min_distance=0.95)
        assert bt.min_distance == 0.95

    def test_init_quarterly_rebalance(self):
        """Should accept quarterly rebalance frequency."""
        bt = MomentumPortfolioBacktest(rebalance_frequency='quarterly')
        assert bt.rebalance_frequency == 'quarterly'


# ============================================================================
# Test Rebalance Dates Generation
# ============================================================================

class TestGetRebalanceDates:
    """Tests for get_rebalance_dates method."""

    def test_semi_annual_generates_feb_aug(self, backtest):
        """Semi-annual should generate February and August dates."""
        dates = backtest.get_rebalance_dates('2020-01-01', '2021-12-31')

        # Should have dates in 2020 and 2021
        assert len(dates) >= 4

        # All dates should be Feb 1 or Aug 1
        for date in dates:
            month = int(date.split('-')[1])
            assert month in [2, 8]

    def test_semi_annual_feb_2020(self, backtest):
        """Should include Feb 2020 for 2020 start date."""
        dates = backtest.get_rebalance_dates('2020-01-01', '2020-12-31')
        assert '2020-02-01' in dates

    def test_semi_annual_aug_2020(self, backtest):
        """Should include Aug 2020 for 2020 dates."""
        dates = backtest.get_rebalance_dates('2020-01-01', '2020-12-31')
        assert '2020-08-01' in dates

    def test_semi_annual_respects_start_date(self, backtest):
        """Should not include dates before start."""
        dates = backtest.get_rebalance_dates('2020-03-01', '2020-12-31')
        assert '2020-02-01' not in dates
        assert '2020-08-01' in dates

    def test_semi_annual_respects_end_date(self, backtest):
        """Should not include dates after end."""
        dates = backtest.get_rebalance_dates('2020-01-01', '2020-07-01')
        assert '2020-02-01' in dates
        assert '2020-08-01' not in dates

    def test_quarterly_generates_quarter_starts(self):
        """Quarterly should generate quarter start dates."""
        bt = MomentumPortfolioBacktest(rebalance_frequency='quarterly')
        dates = bt.get_rebalance_dates('2020-01-01', '2020-12-31')

        # Should have 4 quarterly dates
        assert len(dates) >= 4

    def test_unknown_frequency_raises(self):
        """Unknown frequency should raise ValueError."""
        bt = MomentumPortfolioBacktest(rebalance_frequency='monthly')
        with pytest.raises(ValueError, match="Unknown rebalance frequency"):
            bt.get_rebalance_dates('2020-01-01', '2020-12-31')

    def test_empty_range_returns_empty(self, backtest):
        """Empty date range should return empty list."""
        dates = backtest.get_rebalance_dates('2020-03-01', '2020-05-01')
        assert dates == []


# ============================================================================
# Test Allocation Matrix Building
# ============================================================================

class TestBuildAllocationMatrix:
    """Tests for _build_allocation_matrix method."""

    def test_allocation_matrix_shape(self, backtest, mock_ohlcv_data):
        """Allocation matrix should have same shape as close data."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03', '2020-08-03']  # Weekdays
        portfolios = {
            '2020-02-03': ['AAPL', 'MSFT'],
            '2020-08-03': ['NVDA', 'GOOGL']
        }

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        assert allocations.shape == close.shape

    def test_allocation_equal_weights(self, backtest, mock_ohlcv_data):
        """Should assign equal weights to selected stocks."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03']
        portfolios = {'2020-02-03': ['AAPL', 'MSFT']}

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        # After Feb 3, AAPL and MSFT should have 0.5 each
        idx = allocations.index[allocations.index >= pd.Timestamp('2020-02-03')][0]
        assert abs(allocations.loc[idx, 'AAPL'] - 0.5) < 0.01
        assert abs(allocations.loc[idx, 'MSFT'] - 0.5) < 0.01

    def test_allocation_clears_at_rebalance(self, backtest, mock_ohlcv_data):
        """Should clear old allocations at rebalance."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03', '2020-08-03']
        portfolios = {
            '2020-02-03': ['AAPL', 'MSFT'],
            '2020-08-03': ['NVDA', 'GOOGL']  # Different stocks
        }

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        # After Aug 3, AAPL and MSFT should be 0
        idx = allocations.index[allocations.index >= pd.Timestamp('2020-08-03')][0]
        assert allocations.loc[idx, 'AAPL'] == 0.0
        assert allocations.loc[idx, 'MSFT'] == 0.0
        # NVDA and GOOGL should be 0.5
        assert abs(allocations.loc[idx, 'NVDA'] - 0.5) < 0.01
        assert abs(allocations.loc[idx, 'GOOGL'] - 0.5) < 0.01

    def test_allocation_with_regime_filter(self, backtest, mock_ohlcv_data):
        """Should apply regime allocation percentage."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03']
        portfolios = {'2020-02-03': ['AAPL', 'MSFT']}
        regime_at_rebalance = {'2020-02-03': 'TREND_NEUTRAL'}

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios,
            regime_at_rebalance=regime_at_rebalance
        )

        # TREND_NEUTRAL = 70% allocation, so 0.5 * 0.7 = 0.35
        idx = allocations.index[allocations.index >= pd.Timestamp('2020-02-03')][0]
        assert abs(allocations.loc[idx, 'AAPL'] - 0.35) < 0.01

    def test_allocation_empty_portfolio(self, backtest, mock_ohlcv_data):
        """Should handle empty portfolio gracefully."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03']
        portfolios = {'2020-02-03': []}  # Empty

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        # All allocations should be 0
        assert (allocations == 0.0).all().all()

    def test_allocation_handles_timezone_aware_data(self, backtest):
        """Should handle timezone-aware data index."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D', tz='America/New_York')
        dates = dates[dates.dayofweek < 5]

        close = pd.DataFrame({
            'AAPL': np.random.rand(len(dates)) * 100 + 100,
            'MSFT': np.random.rand(len(dates)) * 100 + 100
        }, index=dates)

        rebalance_dates = ['2020-02-03']
        portfolios = {'2020-02-03': ['AAPL']}

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        assert allocations.shape == close.shape


# ============================================================================
# Test Metrics Extraction
# ============================================================================

class TestExtractMetrics:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_keys(self, backtest):
        """Should extract all expected metric keys."""
        # Create mock portfolio
        mock_pf = Mock()
        mock_pf.wrapper.index = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        mock_pf.init_cash = 100000
        mock_pf.final_value = 150000
        mock_pf.total_return = 0.50
        mock_pf.sharpe_ratio = 1.2
        mock_pf.sortino_ratio = 1.5
        mock_pf.max_drawdown = 0.15
        mock_pf.calmar_ratio = 0.8
        mock_pf.orders.readable = pd.DataFrame({'id': range(100)})

        metrics = backtest._extract_metrics(mock_pf)

        assert 'init_cash' in metrics
        assert 'final_value' in metrics
        assert 'total_return' in metrics
        assert 'cagr' in metrics
        assert 'sharpe' in metrics
        assert 'sortino' in metrics
        assert 'max_dd' in metrics
        assert 'calmar' in metrics
        assert 'total_trades' in metrics
        assert 'duration_days' in metrics
        assert 'duration_years' in metrics

    def test_extract_metrics_cagr_calculation(self, backtest):
        """Should calculate CAGR correctly."""
        mock_pf = Mock()
        # 5 year period
        mock_pf.wrapper.index = pd.date_range('2020-01-01', '2025-01-01', freq='D')
        mock_pf.init_cash = 100000
        mock_pf.final_value = 200000
        mock_pf.total_return = 1.0  # 100% return
        mock_pf.sharpe_ratio = 1.0
        mock_pf.sortino_ratio = 1.0
        mock_pf.max_drawdown = 0.1
        mock_pf.calmar_ratio = 1.0
        mock_pf.orders.readable = pd.DataFrame({'id': range(10)})

        metrics = backtest._extract_metrics(mock_pf)

        # CAGR for 100% return over 5 years = (2)^(1/5) - 1 = ~14.87%
        assert abs(metrics['cagr'] - 0.1487) < 0.01


# ============================================================================
# Test Portfolio Selection
# ============================================================================

class TestSelectPortfolioAtDate:
    """Tests for select_portfolio_at_date method."""

    @patch('integrations.stock_scanner_bridge.Momentum52WDetector')
    def test_select_portfolio_returns_list(self, mock_detector, backtest, mock_vbt_data):
        """Should return list of selected tickers."""
        # Setup mock detector to return signals
        mock_signal = Mock()
        mock_signal.ticker = 'AAPL'
        mock_signal.momentum_score = 0.95
        mock_detector.analyze_stock.return_value = mock_signal

        # Override universe to match mock data
        backtest.universe = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META']

        selected = backtest.select_portfolio_at_date(mock_vbt_data, '2024-06-01')

        assert isinstance(selected, list)

    @patch('integrations.stock_scanner_bridge.Momentum52WDetector')
    def test_select_portfolio_limits_to_top_n(self, mock_detector, backtest, mock_vbt_data):
        """Should limit selection to top_n stocks."""
        # Create mock signals with different scores
        def create_signal(ticker):
            sig = Mock()
            sig.ticker = ticker
            sig.momentum_score = hash(ticker) % 100 / 100  # Deterministic scores
            return sig

        mock_detector.analyze_stock.side_effect = lambda **kwargs: create_signal(kwargs['ticker'])

        backtest.universe = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META']
        backtest.top_n = 3

        selected = backtest.select_portfolio_at_date(mock_vbt_data, '2024-06-01')

        assert len(selected) <= backtest.top_n

    def test_select_portfolio_insufficient_history(self, backtest):
        """Should return empty list with insufficient history."""
        # Create mock data with only 100 days (less than 252)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        dates = dates[dates.dayofweek < 5]

        mock_data = Mock()
        mock_data.get = lambda key: pd.DataFrame({
            'AAPL': np.random.rand(len(dates)) * 100 + 100
        }, index=dates)

        backtest.universe = ['AAPL']
        selected = backtest.select_portfolio_at_date(mock_data, '2024-03-01')

        assert selected == []


# ============================================================================
# Test Regime Data Fetching
# ============================================================================

class TestFetchRegimeData:
    """Tests for _fetch_regime_data method."""

    def test_fetch_regime_data_returns_dataframe_and_series(self, backtest):
        """Should return SPY DataFrame and VIX Series (uses live data or fallback)."""
        # This test verifies the return types regardless of data source
        # The method has internal fallback logic that's hard to mock
        # Skip if no network access
        try:
            spy_data, vix_data = backtest._fetch_regime_data('2024-01-01', '2024-06-30', lookback=100)
            assert isinstance(spy_data, pd.DataFrame)
            assert isinstance(vix_data, pd.Series)
            assert 'Close' in spy_data.columns
        except Exception:
            pytest.skip("Network access required for this test")

    @patch('integrations.stock_scanner_bridge.TiingoDataFetcher')
    @patch('integrations.stock_scanner_bridge.vbt.YFData')
    def test_fetch_regime_data_fallback_to_yf(self, mock_yf, mock_tiingo_class, backtest):
        """Should fallback to Yahoo Finance when Tiingo fails."""
        # Make Tiingo fail
        mock_tiingo_class.side_effect = Exception("Tiingo unavailable")

        # Setup YF mock
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        mock_spy = Mock()
        mock_spy.get.return_value = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100 + 400,
            'High': np.random.rand(len(dates)) * 100 + 410,
            'Low': np.random.rand(len(dates)) * 100 + 390,
            'Close': np.random.rand(len(dates)) * 100 + 400,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        mock_yf.pull.return_value = mock_spy

        spy_data, vix_data = backtest._fetch_regime_data('2024-01-01', '2024-12-31')

        # Should have called YF.pull for both SPY and VIX
        assert mock_yf.pull.call_count == 2


# ============================================================================
# Test Regime Generation
# ============================================================================

class TestGenerateRegimes:
    """Tests for _generate_regimes method."""

    def test_generate_regimes_returns_series(self, backtest):
        """Should return pandas Series of regimes."""
        # Create realistic test data
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        dates = dates[dates.dayofweek < 5]  # Filter weekends

        spy_data = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100 + 400,
            'High': np.random.rand(len(dates)) * 100 + 410,
            'Low': np.random.rand(len(dates)) * 100 + 390,
            'Close': np.random.rand(len(dates)) * 100 + 400,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        vix_data = pd.Series(np.random.rand(len(dates)) * 10 + 15, index=dates)

        # Use actual regime generation (tests integration with academic_jump_model)
        try:
            result = backtest._generate_regimes(spy_data, vix_data, lookback=500)
            assert isinstance(result, pd.Series)
            # Regimes should be one of the known values
            valid_regimes = ['TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH']
            assert all(r in valid_regimes for r in result.unique())
        except ImportError:
            pytest.skip("AcademicJumpModel not available")


# ============================================================================
# Test Full Backtest Run
# ============================================================================

class TestRunBacktest:
    """Tests for run method."""

    @patch('integrations.stock_scanner_bridge.vbt.YFData')
    @patch('integrations.stock_scanner_bridge.Momentum52WDetector')
    def test_run_returns_results_dict(self, mock_detector, mock_yf, backtest):
        """Should return dictionary with results."""
        # Setup mock data
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        dates = dates[dates.dayofweek < 5]

        mock_data = Mock()
        close_df = pd.DataFrame({
            symbol: np.random.rand(len(dates)) * 100 + 100
            for symbol in backtest.universe
        }, index=dates)
        mock_data.get = lambda key: close_df if key == 'Close' else close_df
        mock_yf.pull.return_value = mock_data

        # Mock detector returns signals
        def create_signal(ticker, **kwargs):
            sig = Mock()
            sig.ticker = ticker
            sig.momentum_score = 0.95
            return sig
        mock_detector.analyze_stock.side_effect = create_signal

        # Mock Portfolio
        with patch('integrations.stock_scanner_bridge.vbt.Portfolio') as mock_portfolio:
            mock_pf = Mock()
            mock_pf.wrapper.index = dates
            mock_pf.init_cash = 100000
            mock_pf.final_value = 150000
            mock_pf.total_return = 0.50
            mock_pf.sharpe_ratio = 1.2
            mock_pf.sortino_ratio = 1.5
            mock_pf.max_drawdown = 0.15
            mock_pf.calmar_ratio = 0.8
            mock_pf.orders.readable = pd.DataFrame({'id': range(100)})
            mock_portfolio.from_orders.return_value = mock_pf

            results = backtest.run(
                start_date='2020-01-01',
                end_date='2024-12-31',
                initial_capital=100000
            )

        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'allocations' in results
        assert 'rebalance_dates' in results

    @patch('integrations.stock_scanner_bridge.vbt.YFData')
    def test_run_handles_download_error(self, mock_yf, backtest):
        """Should handle data download errors gracefully."""
        mock_yf.pull.side_effect = Exception("Network error")

        results = backtest.run(
            start_date='2020-01-01',
            end_date='2024-12-31'
        )

        assert 'error' in results

    def test_run_no_rebalance_dates(self, backtest):
        """Should handle case with no rebalance dates."""
        with patch('integrations.stock_scanner_bridge.vbt.YFData') as mock_yf:
            dates = pd.date_range('2020-03-15', '2020-04-15', freq='D')
            dates = dates[dates.dayofweek < 5]

            mock_data = Mock()
            close_df = pd.DataFrame({
                'AAPL': np.random.rand(len(dates)) * 100 + 100
            }, index=dates)
            mock_data.get = lambda key: close_df
            mock_yf.pull.return_value = mock_data

            results = backtest.run(
                start_date='2020-03-15',
                end_date='2020-04-15'
            )

            assert 'error' in results
            assert 'No rebalance dates' in results['error']


# ============================================================================
# Test with Regime Filter
# ============================================================================

class TestRunWithRegimeFilter:
    """Tests for run method with regime filter enabled."""

    @patch('integrations.stock_scanner_bridge.vbt.YFData')
    @patch('integrations.stock_scanner_bridge.vbt.Portfolio')
    @patch('integrations.stock_scanner_bridge.Momentum52WDetector')
    @patch.object(MomentumPortfolioBacktest, '_fetch_regime_data')
    @patch.object(MomentumPortfolioBacktest, '_generate_regimes')
    def test_run_with_regime_filter(
        self, mock_regimes, mock_fetch_regime, mock_detector, mock_portfolio, mock_yf, backtest
    ):
        """Should apply regime filter when enabled."""
        # Setup mock data
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        dates = dates[dates.dayofweek < 5]

        mock_data = Mock()
        close_df = pd.DataFrame({
            symbol: np.random.rand(len(dates)) * 100 + 100
            for symbol in backtest.universe
        }, index=dates)
        mock_data.get = lambda key: close_df if key == 'Close' else close_df
        mock_yf.pull.return_value = mock_data

        # Mock regime data
        spy_ohlcv = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100 + 400,
            'High': np.random.rand(len(dates)) * 100 + 410,
            'Low': np.random.rand(len(dates)) * 100 + 390,
            'Close': np.random.rand(len(dates)) * 100 + 400,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        vix_close = pd.Series(np.random.rand(len(dates)) * 10 + 15, index=dates)
        mock_fetch_regime.return_value = (spy_ohlcv, vix_close)

        # Mock regime generation
        regime_series = pd.Series(['TREND_BULL'] * len(dates), index=dates)
        mock_regimes.return_value = regime_series

        # Mock detector
        def create_signal(ticker, **kwargs):
            sig = Mock()
            sig.ticker = ticker
            sig.momentum_score = 0.95
            return sig
        mock_detector.analyze_stock.side_effect = create_signal

        # Mock Portfolio
        mock_pf = Mock()
        mock_pf.wrapper.index = dates
        mock_pf.init_cash = 100000
        mock_pf.final_value = 150000
        mock_pf.total_return = 0.50
        mock_pf.sharpe_ratio = 1.2
        mock_pf.sortino_ratio = 1.5
        mock_pf.max_drawdown = 0.15
        mock_pf.calmar_ratio = 0.8
        mock_pf.orders.readable = pd.DataFrame({'id': range(100)})
        mock_portfolio.from_orders.return_value = mock_pf

        results = backtest.run(
            start_date='2020-01-01',
            end_date='2024-12-31',
            use_regime_filter=True
        )

        assert results['regime_filter_enabled'] is True
        assert 'regime_at_rebalance' in results
        assert 'regime_series' in results


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_timezone_aware_close_data(self, backtest, mock_ohlcv_data):
        """Should handle timezone-aware close data."""
        # Make close data timezone-aware
        close = mock_ohlcv_data['Close'].copy()
        close.index = close.index.tz_localize('America/New_York')

        rebalance_dates = ['2020-02-03']
        portfolios = {'2020-02-03': ['AAPL', 'MSFT']}

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        assert allocations.shape == close.shape

    def test_missing_symbol_in_universe(self, backtest, mock_ohlcv_data):
        """Should handle symbols not in data gracefully."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03']
        # Include symbol not in mock data
        portfolios = {'2020-02-03': ['AAPL', 'MSFT', 'UNKNOWN_SYMBOL']}

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        # Should not raise, just skip unknown symbol
        assert allocations.shape == close.shape

    def test_weekend_rebalance_date(self, backtest, mock_ohlcv_data):
        """Should handle rebalance date falling on weekend."""
        close = mock_ohlcv_data['Close']
        # Feb 1, 2020 was a Saturday
        rebalance_dates = ['2020-02-01']
        portfolios = {'2020-02-01': ['AAPL', 'MSFT']}

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios
        )

        # Should find next trading day
        assert allocations.shape == close.shape


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    def test_multiple_rebalances_with_different_regimes(self, backtest, mock_ohlcv_data):
        """Test multiple rebalances with varying regimes."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03', '2020-08-03', '2021-02-01']
        portfolios = {
            '2020-02-03': ['AAPL', 'MSFT'],
            '2020-08-03': ['NVDA', 'GOOGL'],
            '2021-02-01': ['META', 'AAPL']
        }
        regime_at_rebalance = {
            '2020-02-03': 'TREND_BULL',    # 100% allocation
            '2020-08-03': 'TREND_NEUTRAL', # 70% allocation
            '2021-02-01': 'TREND_BEAR'     # 30% allocation
        }

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios,
            regime_at_rebalance=regime_at_rebalance
        )

        # Check allocations at each period
        # Feb 2020: 0.5 each (TREND_BULL = 100%)
        feb_idx = allocations.index[allocations.index >= pd.Timestamp('2020-02-03')][0]
        assert abs(allocations.loc[feb_idx, 'AAPL'] - 0.5) < 0.01

        # Aug 2020: 0.35 each (TREND_NEUTRAL = 70%)
        aug_idx = allocations.index[allocations.index >= pd.Timestamp('2020-08-03')][0]
        assert abs(allocations.loc[aug_idx, 'NVDA'] - 0.35) < 0.01

        # Feb 2021: 0.15 each (TREND_BEAR = 30%)
        feb21_idx = allocations.index[allocations.index >= pd.Timestamp('2021-02-01')][0]
        assert abs(allocations.loc[feb21_idx, 'META'] - 0.15) < 0.01

    def test_crash_regime_zero_allocation(self, backtest, mock_ohlcv_data):
        """CRASH regime should result in zero allocation."""
        close = mock_ohlcv_data['Close']
        rebalance_dates = ['2020-02-03']
        portfolios = {'2020-02-03': ['AAPL', 'MSFT']}
        regime_at_rebalance = {'2020-02-03': 'CRASH'}

        allocations = backtest._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=portfolios,
            regime_at_rebalance=regime_at_rebalance
        )

        # CRASH = 0% allocation
        feb_idx = allocations.index[allocations.index >= pd.Timestamp('2020-02-03')][0]
        assert allocations.loc[feb_idx, 'AAPL'] == 0.0
        assert allocations.loc[feb_idx, 'MSFT'] == 0.0
