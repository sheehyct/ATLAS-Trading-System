"""Tests for Finviz enrichment module."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from strat.ticker_selection.enrichment import FinvizEnrichment, FinvizEnricher


# ---------------------------------------------------------------------------
# FinvizEnrichment dataclass
# ---------------------------------------------------------------------------

class TestFinvizEnrichmentDataclass:
    def test_defaults(self):
        e = FinvizEnrichment()
        assert e.symbol == ''
        assert e.sector == ''
        assert e.news_headlines == []
        assert e.fetch_error == ''
        assert e.cached is False
        assert e.target_price is None

    def test_to_dict(self):
        e = FinvizEnrichment(
            symbol='AAPL',
            sector='Technology',
            industry='Consumer Electronics',
            earnings_date='Apr 22 AMC',
            analyst_recommendation='Buy',
            target_price=475.0,
            news_headlines=['H1', 'H2', 'H3'],
        )
        d = e.to_dict()
        assert d['sector'] == 'Technology'
        assert d['industry'] == 'Consumer Electronics'
        assert d['earnings_date'] == 'Apr 22 AMC'
        assert d['analyst_recommendation'] == 'Buy'
        assert d['target_price'] == 475.0
        assert d['news_headlines'] == ['H1', 'H2', 'H3']
        assert 'fetch_error' not in d

    def test_to_dict_with_error(self):
        e = FinvizEnrichment(symbol='BAD', fetch_error='timeout')
        d = e.to_dict()
        assert d['fetch_error'] == 'timeout'

    def test_to_dict_truncates_news(self):
        e = FinvizEnrichment(
            symbol='AAPL',
            news_headlines=['H1', 'H2', 'H3', 'H4', 'H5'],
        )
        d = e.to_dict()
        assert len(d['news_headlines']) == 3
        assert d['news_headlines'] == ['H1', 'H2', 'H3']


# ---------------------------------------------------------------------------
# Recommendation mapping
# ---------------------------------------------------------------------------

class TestRecommendationMapping:
    def test_strong_buy(self):
        assert FinvizEnricher._map_recommendation('1.0') == 'Strong Buy'
        assert FinvizEnricher._map_recommendation('1.5') == 'Strong Buy'

    def test_buy(self):
        assert FinvizEnricher._map_recommendation('1.6') == 'Buy'
        assert FinvizEnricher._map_recommendation('2.0') == 'Buy'
        assert FinvizEnricher._map_recommendation('2.5') == 'Buy'

    def test_hold(self):
        assert FinvizEnricher._map_recommendation('2.6') == 'Hold'
        assert FinvizEnricher._map_recommendation('3.0') == 'Hold'
        assert FinvizEnricher._map_recommendation('3.5') == 'Hold'

    def test_underperform(self):
        assert FinvizEnricher._map_recommendation('3.6') == 'Underperform'
        assert FinvizEnricher._map_recommendation('4.0') == 'Underperform'
        assert FinvizEnricher._map_recommendation('4.5') == 'Underperform'

    def test_sell(self):
        assert FinvizEnricher._map_recommendation('4.6') == 'Sell'
        assert FinvizEnricher._map_recommendation('5.0') == 'Sell'

    def test_empty(self):
        assert FinvizEnricher._map_recommendation('') == ''
        assert FinvizEnricher._map_recommendation(None) == ''

    def test_non_numeric(self):
        assert FinvizEnricher._map_recommendation('N/A') == ''


# ---------------------------------------------------------------------------
# Earnings date parsing
# ---------------------------------------------------------------------------

class TestEarningsDateParsing:
    def test_amc(self):
        assert FinvizEnricher._parse_earnings_date('Apr 22 AMC') == 'Apr 22 AMC'

    def test_bmo(self):
        assert FinvizEnricher._parse_earnings_date('Jul 15 BMO') == 'Jul 15 BMO'

    def test_dash(self):
        assert FinvizEnricher._parse_earnings_date('-') == ''

    def test_empty(self):
        assert FinvizEnricher._parse_earnings_date('') == ''
        assert FinvizEnricher._parse_earnings_date(None) == ''

    def test_whitespace(self):
        assert FinvizEnricher._parse_earnings_date('  Apr 22 AMC  ') == 'Apr 22 AMC'


# ---------------------------------------------------------------------------
# Safe float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_normal(self):
        assert FinvizEnricher._safe_float('123.45') == 123.45

    def test_with_comma(self):
        assert FinvizEnricher._safe_float('1,234.56') == 1234.56

    def test_none(self):
        assert FinvizEnricher._safe_float(None) is None

    def test_invalid(self):
        assert FinvizEnricher._safe_float('N/A') is None
        assert FinvizEnricher._safe_float('') is None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_write_and_read(self, tmp_path):
        enricher = FinvizEnricher(cache_dir=str(tmp_path), cache_ttl=3600)
        original = FinvizEnrichment(
            symbol='AAPL',
            sector='Technology',
            industry='Consumer Electronics',
            earnings_date='Apr 22 AMC',
            analyst_recommendation='Buy',
            target_price=475.0,
            news_headlines=['H1', 'H2'],
        )
        enricher._save_cache(original)

        loaded = enricher._load_cache('AAPL')
        assert loaded is not None
        assert loaded.symbol == 'AAPL'
        assert loaded.sector == 'Technology'
        assert loaded.industry == 'Consumer Electronics'
        assert loaded.target_price == 475.0
        assert loaded.news_headlines == ['H1', 'H2']

    def test_expired_cache(self, tmp_path):
        enricher = FinvizEnricher(cache_dir=str(tmp_path), cache_ttl=1)
        enricher._save_cache(FinvizEnrichment(symbol='AAPL', sector='Tech'))

        # Force file to look old
        cache_file = tmp_path / 'AAPL_finviz.json'
        old_time = time.time() - 10
        import os
        os.utime(str(cache_file), (old_time, old_time))

        assert enricher._load_cache('AAPL') is None

    def test_cache_miss(self, tmp_path):
        enricher = FinvizEnricher(cache_dir=str(tmp_path))
        assert enricher._load_cache('NVDA') is None


# ---------------------------------------------------------------------------
# enrich_candidates integration
# ---------------------------------------------------------------------------

class TestEnrichCandidates:
    @patch.object(FinvizEnricher, '_scrape_finviz')
    def test_returns_all_symbols(self, mock_scrape, tmp_path):
        mock_scrape.side_effect = lambda sym: FinvizEnrichment(
            symbol=sym, sector='Tech'
        )
        enricher = FinvizEnricher(cache_dir=str(tmp_path))
        result = enricher.enrich_candidates(['AAPL', 'MSFT', 'GOOG'])
        assert set(result.keys()) == {'AAPL', 'MSFT', 'GOOG'}
        assert all(r.sector == 'Tech' for r in result.values())

    @patch.object(FinvizEnricher, '_scrape_finviz')
    def test_graceful_failure(self, mock_scrape, tmp_path):
        mock_scrape.side_effect = Exception('network error')
        enricher = FinvizEnricher(cache_dir=str(tmp_path))
        result = enricher.enrich_candidates(['AAPL', 'MSFT'])
        assert set(result.keys()) == {'AAPL', 'MSFT'}
        assert all(r.fetch_error != '' for r in result.values())

    @patch.object(FinvizEnricher, '_scrape_finviz')
    def test_deduplicates_symbols(self, mock_scrape, tmp_path):
        mock_scrape.side_effect = lambda sym: FinvizEnrichment(
            symbol=sym, sector='Tech'
        )
        enricher = FinvizEnricher(cache_dir=str(tmp_path))
        result = enricher.enrich_candidates(['AAPL', 'AAPL', 'MSFT'])
        assert mock_scrape.call_count == 2  # AAPL only scraped once

    def test_empty_list(self, tmp_path):
        enricher = FinvizEnricher(cache_dir=str(tmp_path))
        assert enricher.enrich_candidates([]) == {}
