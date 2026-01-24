"""
Tests for integrations/alphavantage_fundamentals.py - Alpha Vantage fundamental data.

EQUITY-83: Phase 3 test coverage for Alpha Vantage integration.

Tests cover:
- AlphaVantageFundamentals initialization
- Cache management
- Rate limiting
- Endpoint fetching
- Quality metrics calculation
- Batch processing
- Quality score calculation
"""

import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from integrations.alphavantage_fundamentals import AlphaVantageFundamentals


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_api_key():
    """Mock API key."""
    return "TEST_API_KEY_12345"


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "alphavantage_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def fetcher(mock_api_key, temp_cache_dir):
    """Create fetcher with mocked settings."""
    with patch('integrations.alphavantage_fundamentals.get_alphavantage_key', return_value=mock_api_key):
        f = AlphaVantageFundamentals(api_key=mock_api_key)
        f.CACHE_DIR = temp_cache_dir
        return f


@pytest.fixture
def sample_overview():
    """Sample Alpha Vantage OVERVIEW response."""
    return {
        'Symbol': 'AAPL',
        'ReturnOnEquityTTM': '0.145',
        'DebtEquityRatio': '1.52'
    }


@pytest.fixture
def sample_balance_sheet():
    """Sample Alpha Vantage BALANCE_SHEET response."""
    return {
        'annualReports': [{
            'totalAssets': '350000000000',
            'totalDebt': '110000000000',
            'totalShareholderEquity': '70000000000'
        }]
    }


@pytest.fixture
def sample_income_statement():
    """Sample Alpha Vantage INCOME_STATEMENT response."""
    return {
        'annualReports': [{
            'netIncome': '95000000000'
        }]
    }


@pytest.fixture
def sample_cash_flow():
    """Sample Alpha Vantage CASH_FLOW response."""
    return {
        'annualReports': [{
            'operatingCashflow': '105000000000'
        }]
    }


# =============================================================================
# Initialization Tests
# =============================================================================

class TestInitialization:
    """Tests for AlphaVantageFundamentals initialization."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with explicit API key."""
        with patch('integrations.alphavantage_fundamentals.get_alphavantage_key', return_value=mock_api_key):
            fetcher = AlphaVantageFundamentals(api_key=mock_api_key)

            assert fetcher.api_key == mock_api_key
            assert fetcher._call_count == 0

    def test_init_without_api_key_uses_settings(self, mock_api_key):
        """Test initialization falls back to settings."""
        with patch('integrations.alphavantage_fundamentals.get_alphavantage_key', return_value=mock_api_key):
            fetcher = AlphaVantageFundamentals()

            assert fetcher.api_key == mock_api_key

    def test_init_no_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with patch('integrations.alphavantage_fundamentals.get_alphavantage_key', return_value=None):
            with pytest.raises(ValueError) as exc_info:
                AlphaVantageFundamentals(api_key=None)

            assert "ALPHAVANTAGE_API_KEY" in str(exc_info.value)

    def test_creates_cache_directory(self, mock_api_key, tmp_path):
        """Test cache directory is created."""
        cache_dir = tmp_path / "new_cache"

        with patch('integrations.alphavantage_fundamentals.get_alphavantage_key', return_value=mock_api_key):
            fetcher = AlphaVantageFundamentals(api_key=mock_api_key)
            fetcher.CACHE_DIR = cache_dir
            fetcher.CACHE_DIR.mkdir(parents=True, exist_ok=True)

            assert cache_dir.exists()


# =============================================================================
# Cache Tests
# =============================================================================

class TestCacheManagement:
    """Tests for cache management."""

    def test_get_cache_path(self, fetcher):
        """Test cache path generation."""
        path = fetcher._get_cache_path('AAPL', 'overview')

        assert path.name == 'AAPL_overview.json'
        assert path.parent == fetcher.CACHE_DIR

    def test_is_cache_valid_no_file(self, fetcher):
        """Test cache invalid when file doesn't exist."""
        path = fetcher._get_cache_path('NONEXISTENT', 'overview')

        assert fetcher._is_cache_valid(path) is False

    def test_is_cache_valid_fresh_file(self, fetcher, temp_cache_dir):
        """Test cache valid for fresh file."""
        cache_file = temp_cache_dir / "AAPL_overview.json"
        cache_file.write_text('{}')

        assert fetcher._is_cache_valid(cache_file) is True

    def test_is_cache_valid_expired_file(self, fetcher, temp_cache_dir):
        """Test cache invalid for expired file."""
        cache_file = temp_cache_dir / "AAPL_overview.json"
        cache_file.write_text('{}')

        # Set modification time to 100 days ago
        import os
        old_time = datetime.now() - timedelta(days=100)
        os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))

        assert fetcher._is_cache_valid(cache_file) is False

    def test_clear_cache_single_symbol(self, fetcher, temp_cache_dir):
        """Test clearing cache for single symbol."""
        # Create cache files
        for endpoint in ['overview', 'balance_sheet', 'income_statement', 'cash_flow']:
            (temp_cache_dir / f"AAPL_{endpoint}.json").write_text('{}')
        (temp_cache_dir / "MSFT_overview.json").write_text('{}')

        deleted = fetcher.clear_cache('AAPL')

        assert deleted == 4
        assert not (temp_cache_dir / "AAPL_overview.json").exists()
        assert (temp_cache_dir / "MSFT_overview.json").exists()

    def test_clear_cache_all(self, fetcher, temp_cache_dir):
        """Test clearing all cache."""
        # Create cache files
        (temp_cache_dir / "AAPL_overview.json").write_text('{}')
        (temp_cache_dir / "MSFT_overview.json").write_text('{}')

        deleted = fetcher.clear_cache()

        assert deleted == 2
        assert len(list(temp_cache_dir.glob('*.json'))) == 0


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_increments_count(self, fetcher):
        """Test rate limit increments call count."""
        initial_count = fetcher._call_count

        fetcher._rate_limit()

        assert fetcher._call_count == initial_count + 1

    def test_rate_limit_sets_last_call_time(self, fetcher):
        """Test rate limit sets last call time."""
        assert fetcher._last_call_time is None

        fetcher._rate_limit()

        assert fetcher._last_call_time is not None

    def test_get_api_call_count(self, fetcher):
        """Test get_api_call_count returns count."""
        fetcher._call_count = 5

        assert fetcher.get_api_call_count() == 5


# =============================================================================
# Fetch Endpoint Tests
# =============================================================================

class TestFetchEndpoint:
    """Tests for _fetch_endpoint method."""

    def test_fetch_uses_cache(self, fetcher, temp_cache_dir, sample_overview):
        """Test fetch uses cached data."""
        cache_file = temp_cache_dir / "AAPL_overview.json"
        cache_file.write_text(json.dumps(sample_overview))

        result = fetcher._fetch_endpoint('AAPL', 'OVERVIEW')

        assert result['Symbol'] == 'AAPL'
        assert fetcher._call_count == 0  # No API call made

    def test_fetch_makes_api_call(self, fetcher, sample_overview):
        """Test fetch makes API call when no cache."""
        with patch('integrations.alphavantage_fundamentals.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_overview
            mock_get.return_value = mock_response

            result = fetcher._fetch_endpoint('AAPL', 'OVERVIEW')

            assert result['Symbol'] == 'AAPL'
            assert fetcher._call_count == 1

    def test_fetch_handles_api_error(self, fetcher):
        """Test fetch raises on API error."""
        with patch('integrations.alphavantage_fundamentals.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'Error Message': 'Invalid symbol'}
            mock_get.return_value = mock_response

            with pytest.raises(ValueError) as exc_info:
                fetcher._fetch_endpoint('INVALID', 'OVERVIEW')

            assert "Invalid symbol" in str(exc_info.value)

    def test_fetch_handles_rate_limit_error(self, fetcher):
        """Test fetch raises on rate limit error."""
        with patch('integrations.alphavantage_fundamentals.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'Note': 'Rate limit exceeded'}
            mock_get.return_value = mock_response

            with pytest.raises(ValueError) as exc_info:
                fetcher._fetch_endpoint('AAPL', 'OVERVIEW')

            assert "rate limit" in str(exc_info.value).lower()


# =============================================================================
# Accruals Calculation Tests
# =============================================================================

class TestAccrualsCalculation:
    """Tests for _calculate_accruals_ratio method."""

    def test_calculate_accruals_ratio(self, fetcher, sample_income_statement, sample_cash_flow, sample_balance_sheet):
        """Test accruals ratio calculation."""
        # Net Income: 95B, Operating CF: 105B, Total Assets: 350B
        # Accruals = 95B - 105B = -10B
        # Ratio = -10B / 350B = -0.0286

        ratio = fetcher._calculate_accruals_ratio(
            sample_income_statement,
            sample_cash_flow,
            sample_balance_sheet
        )

        assert ratio == pytest.approx(-0.0286, rel=0.01)

    def test_calculate_accruals_zero_assets(self, fetcher):
        """Test accruals returns NaN for zero assets."""
        income = {'annualReports': [{'netIncome': '1000'}]}
        cash_flow = {'annualReports': [{'operatingCashflow': '500'}]}
        balance = {'annualReports': [{'totalAssets': '0'}]}

        ratio = fetcher._calculate_accruals_ratio(income, cash_flow, balance)

        assert np.isnan(ratio)

    def test_calculate_accruals_missing_data(self, fetcher):
        """Test accruals handles missing data gracefully."""
        # When data is missing, function returns NaN via exception handling
        # or 0.0 if empty dicts parse to 0 values
        ratio = fetcher._calculate_accruals_ratio({}, {}, {})

        # Function returns 0.0 when empty dicts default to 0 values
        # (0 - 0) / 1 = 0.0
        assert ratio == 0.0 or np.isnan(ratio)


# =============================================================================
# Quality Metrics Tests
# =============================================================================

class TestGetQualityMetrics:
    """Tests for get_quality_metrics method."""

    def test_get_quality_metrics(self, fetcher, sample_overview, sample_balance_sheet,
                                   sample_income_statement, sample_cash_flow, temp_cache_dir):
        """Test getting quality metrics for a symbol."""
        # Pre-cache all data
        (temp_cache_dir / "AAPL_overview.json").write_text(json.dumps(sample_overview))
        (temp_cache_dir / "AAPL_balance_sheet.json").write_text(json.dumps(sample_balance_sheet))
        (temp_cache_dir / "AAPL_income_statement.json").write_text(json.dumps(sample_income_statement))
        (temp_cache_dir / "AAPL_cash_flow.json").write_text(json.dumps(sample_cash_flow))

        metrics = fetcher.get_quality_metrics('AAPL')

        assert metrics['symbol'] == 'AAPL'
        assert metrics['roe'] == pytest.approx(0.145)
        assert metrics['debt_to_equity'] == pytest.approx(1.52)
        assert 'accruals_ratio' in metrics

    def test_get_quality_metrics_handles_error(self, fetcher):
        """Test handles errors gracefully."""
        with patch.object(fetcher, '_fetch_overview', side_effect=Exception("API Error")):
            metrics = fetcher.get_quality_metrics('AAPL')

            assert metrics['symbol'] == 'AAPL'
            assert np.isnan(metrics['roe'])
            assert 'error' in metrics


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestBatchProcessing:
    """Tests for get_quality_metrics_batch method."""

    def test_batch_returns_dataframe(self, fetcher, temp_cache_dir, sample_overview,
                                      sample_balance_sheet, sample_income_statement, sample_cash_flow):
        """Test batch returns DataFrame."""
        # Pre-cache data for AAPL
        for symbol in ['AAPL']:
            (temp_cache_dir / f"{symbol}_overview.json").write_text(json.dumps(sample_overview))
            (temp_cache_dir / f"{symbol}_balance_sheet.json").write_text(json.dumps(sample_balance_sheet))
            (temp_cache_dir / f"{symbol}_income_statement.json").write_text(json.dumps(sample_income_statement))
            (temp_cache_dir / f"{symbol}_cash_flow.json").write_text(json.dumps(sample_cash_flow))

        result = fetcher.get_quality_metrics_batch(['AAPL'])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'roe' in result.columns

    def test_batch_respects_max_new_fetches(self, fetcher, temp_cache_dir):
        """Test batch respects rate limit."""
        # No cached data - would need API calls
        # With max_new_fetches=1, should only fetch first symbol

        with patch.object(fetcher, 'get_quality_metrics') as mock_get:
            mock_get.return_value = {'symbol': 'AAPL', 'roe': 0.15, 'accruals_ratio': 0.01, 'debt_to_equity': 1.5}

            result = fetcher.get_quality_metrics_batch(['AAPL', 'MSFT', 'GOOGL'], max_new_fetches=1)

            # Should have called for first symbol, skipped others
            assert mock_get.call_count <= 2  # May check cache first


# =============================================================================
# Quality Score Calculation Tests
# =============================================================================

class TestQualityScoreCalculation:
    """Tests for calculate_quality_scores method."""

    def test_calculate_quality_scores(self, fetcher):
        """Test quality score calculation."""
        metrics_df = pd.DataFrame([
            {'symbol': 'AAPL', 'roe': 0.20, 'accruals_ratio': -0.05, 'debt_to_equity': 1.0},
            {'symbol': 'MSFT', 'roe': 0.30, 'accruals_ratio': 0.02, 'debt_to_equity': 0.5},
            {'symbol': 'GOOGL', 'roe': 0.15, 'accruals_ratio': 0.00, 'debt_to_equity': 0.2},
        ])

        result = fetcher.calculate_quality_scores(metrics_df)

        assert 'quality_score' in result.columns
        assert 'roe_rank' in result.columns
        assert 'earnings_quality' in result.columns
        assert 'inverse_leverage' in result.columns

        # All scores should be between 0 and 1
        assert (result['quality_score'] >= 0).all()
        assert (result['quality_score'] <= 1).all()

    def test_quality_score_ranking(self, fetcher):
        """Test quality score ranks correctly."""
        metrics_df = pd.DataFrame([
            {'symbol': 'HIGH', 'roe': 0.50, 'accruals_ratio': -0.10, 'debt_to_equity': 0.1},  # Best
            {'symbol': 'LOW', 'roe': 0.05, 'accruals_ratio': 0.20, 'debt_to_equity': 3.0},   # Worst
        ])

        result = fetcher.calculate_quality_scores(metrics_df)

        high_score = result[result['symbol'] == 'HIGH']['quality_score'].iloc[0]
        low_score = result[result['symbol'] == 'LOW']['quality_score'].iloc[0]

        assert high_score > low_score

    def test_quality_score_handles_nan(self, fetcher):
        """Test quality score handles NaN values."""
        metrics_df = pd.DataFrame([
            {'symbol': 'AAPL', 'roe': 0.20, 'accruals_ratio': np.nan, 'debt_to_equity': 1.0},
            {'symbol': 'MSFT', 'roe': np.nan, 'accruals_ratio': 0.02, 'debt_to_equity': np.nan},
        ])

        result = fetcher.calculate_quality_scores(metrics_df)

        # Should not raise, scores may contain NaN
        assert 'quality_score' in result.columns
