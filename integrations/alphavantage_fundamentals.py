"""
Alpha Vantage Fundamental Data Fetcher for Quality-Momentum Strategy.

Provides cached access to fundamental data including:
- ROE (Return on Equity)
- Accruals Ratio (for earnings quality)
- Debt-to-Equity Ratio (leverage)

Rate Limit: 25 calls/day (free tier)
Cache: 90-day expiration (aligns with quarterly earnings)

Session EQUITY-35: Created for Quality-Momentum strategy implementation.
"""

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import json
import time
from typing import Dict, Optional, List

from config.settings import get_alphavantage_key


class AlphaVantageFundamentals:
    """
    Fetches and caches fundamental data from Alpha Vantage.

    Provides quality metrics for the Quality-Momentum strategy:
    - ROE (Return on Equity): Higher is better
    - Accruals Ratio: Lower is better (higher earnings quality)
    - Debt-to-Equity: Lower is better (less leverage)

    Usage:
        fetcher = AlphaVantageFundamentals()
        metrics = fetcher.get_quality_metrics('AAPL')
        print(f"ROE: {metrics['roe']}, D/E: {metrics['debt_to_equity']}")

    Rate Limiting:
        Free tier allows 25 calls/day. This class enforces a 12.5 second
        delay between API calls and uses aggressive caching (90 days).
    """

    BASE_URL = "https://www.alphavantage.co/query"
    CACHE_DIR = Path("data/alphavantage_cache")
    CACHE_EXPIRY_DAYS = 90  # Quarterly earnings cycle
    RATE_LIMIT_DELAY = 12.5  # 25 calls/day = 1 call per ~3.5 minutes, but be conservative

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage fundamentals fetcher.

        Args:
            api_key: Alpha Vantage API key. If None, uses ALPHAVANTAGE_API_KEY
                    from environment via config.settings.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or get_alphavantage_key()
        if not self.api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY not set. Add it to your .env file."
            )

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._call_count = 0
        self._last_call_time = None

    def _rate_limit(self) -> None:
        """Enforce rate limit with delay between API calls."""
        if self._last_call_time:
            elapsed = (datetime.now() - self._last_call_time).total_seconds()
            if elapsed < self.RATE_LIMIT_DELAY:
                sleep_time = self.RATE_LIMIT_DELAY - elapsed
                print(f"  Rate limiting: sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        self._last_call_time = datetime.now()
        self._call_count += 1

    def _get_cache_path(self, symbol: str, endpoint: str) -> Path:
        """Get cache file path for symbol/endpoint combination."""
        return self.CACHE_DIR / f"{symbol}_{endpoint}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(days=self.CACHE_EXPIRY_DAYS)

    def _fetch_endpoint(self, symbol: str, function: str) -> Dict:
        """
        Fetch data from Alpha Vantage API with caching.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            function: Alpha Vantage function (OVERVIEW, BALANCE_SHEET, etc.)

        Returns:
            API response as dictionary.
        """
        cache_path = self._get_cache_path(symbol, function.lower())

        if self._is_cache_valid(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)

        self._rate_limit()
        print(f"  Fetching {function} for {symbol}...")

        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key
        }

        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if 'Note' in data:
            # Rate limit exceeded
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

        # Cache the response
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)

        return data

    def _fetch_overview(self, symbol: str) -> Dict:
        """Fetch company overview (contains ROE, D/E ratio)."""
        return self._fetch_endpoint(symbol, 'OVERVIEW')

    def _fetch_balance_sheet(self, symbol: str) -> Dict:
        """Fetch balance sheet for debt/equity and accruals calculation."""
        return self._fetch_endpoint(symbol, 'BALANCE_SHEET')

    def _fetch_income_statement(self, symbol: str) -> Dict:
        """Fetch income statement for net income."""
        return self._fetch_endpoint(symbol, 'INCOME_STATEMENT')

    def _fetch_cash_flow(self, symbol: str) -> Dict:
        """Fetch cash flow statement for operating cash flow."""
        return self._fetch_endpoint(symbol, 'CASH_FLOW')

    def _calculate_accruals_ratio(
        self,
        income: Dict,
        cash_flow: Dict,
        balance: Dict
    ) -> float:
        """
        Calculate accruals ratio for earnings quality.

        Accruals Ratio = (Net Income - Operating Cash Flow) / Total Assets

        Lower values indicate higher earnings quality (more cash-based earnings).

        Args:
            income: Income statement data from Alpha Vantage
            cash_flow: Cash flow statement data
            balance: Balance sheet data

        Returns:
            Accruals ratio as float, or NaN if calculation fails.
        """
        try:
            # Get most recent annual reports
            annual_income = income.get('annualReports', [{}])[0]
            annual_cf = cash_flow.get('annualReports', [{}])[0]
            annual_balance = balance.get('annualReports', [{}])[0]

            net_income = float(annual_income.get('netIncome', 0) or 0)
            operating_cf = float(annual_cf.get('operatingCashflow', 0) or 0)
            total_assets = float(annual_balance.get('totalAssets', 1) or 1)

            if total_assets == 0:
                return np.nan

            accruals = net_income - operating_cf
            accruals_ratio = accruals / total_assets

            return accruals_ratio

        except (KeyError, ValueError, TypeError, IndexError):
            return np.nan

    def get_quality_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Get quality metrics for a single symbol.

        Fetches data from Alpha Vantage (with caching) and calculates:
        - ROE: Return on Equity (higher is better)
        - Accruals Ratio: (Net Income - OCF) / Assets (lower is better)
        - Debt-to-Equity: Total Debt / Equity (lower is better)

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with keys: symbol, roe, accruals_ratio, debt_to_equity.
            Values are NaN if data unavailable.

        Note:
            This method may make up to 4 API calls per symbol if not cached.
            With 25 calls/day limit, you can fetch ~6 new symbols per day.
        """
        try:
            print(f"Fetching quality metrics for {symbol}...")

            overview = self._fetch_overview(symbol)
            balance = self._fetch_balance_sheet(symbol)
            income = self._fetch_income_statement(symbol)
            cash_flow = self._fetch_cash_flow(symbol)

            # ROE from OVERVIEW (already calculated by Alpha Vantage)
            roe_str = overview.get('ReturnOnEquityTTM', '')
            roe = float(roe_str) if roe_str and roe_str != 'None' else np.nan

            # Debt-to-Equity from balance sheet
            # Alpha Vantage OVERVIEW may have this, otherwise calculate
            de_str = overview.get('DebtEquityRatio', '')
            if de_str and de_str != 'None':
                debt_to_equity = float(de_str)
            else:
                # Calculate from balance sheet
                try:
                    annual_balance = balance.get('annualReports', [{}])[0]
                    total_debt = float(annual_balance.get('totalDebt', 0) or 0)
                    equity = float(annual_balance.get('totalShareholderEquity', 1) or 1)
                    debt_to_equity = total_debt / equity if equity != 0 else np.nan
                except (KeyError, ValueError, TypeError, IndexError):
                    debt_to_equity = np.nan

            # Calculate Accruals Ratio for earnings quality
            accruals_ratio = self._calculate_accruals_ratio(income, cash_flow, balance)

            return {
                'symbol': symbol,
                'roe': roe,
                'accruals_ratio': accruals_ratio,
                'debt_to_equity': debt_to_equity
            }

        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            return {
                'symbol': symbol,
                'roe': np.nan,
                'accruals_ratio': np.nan,
                'debt_to_equity': np.nan,
                'error': str(e)
            }

    def get_quality_metrics_batch(
        self,
        symbols: List[str],
        max_new_fetches: int = 6
    ) -> pd.DataFrame:
        """
        Get quality metrics for multiple symbols.

        Processes cached symbols first, then fetches new data up to the
        daily limit.

        Args:
            symbols: List of stock symbols
            max_new_fetches: Maximum new symbols to fetch (rate limit aware).
                            Each symbol requires 4 API calls.
                            With 25 calls/day, max is ~6 symbols.

        Returns:
            DataFrame with columns: symbol, roe, accruals_ratio, debt_to_equity
        """
        metrics_list = []
        new_fetch_count = 0

        for symbol in symbols:
            # Check if all endpoints are cached
            cache_valid = all(
                self._is_cache_valid(self._get_cache_path(symbol, ep))
                for ep in ['overview', 'balance_sheet', 'income_statement', 'cash_flow']
            )

            if not cache_valid:
                new_fetch_count += 1
                if new_fetch_count > max_new_fetches:
                    print(f"  Skipping {symbol}: rate limit reached ({max_new_fetches} new fetches)")
                    continue

            metrics = self.get_quality_metrics(symbol)
            metrics_list.append(metrics)

        return pd.DataFrame(metrics_list)

    def calculate_quality_scores(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite quality scores from raw metrics.

        Quality Score = 0.40 * ROE_rank + 0.30 * Earnings_quality + 0.30 * Inverse_leverage

        Where:
        - ROE_rank: Percentile rank of ROE (higher ROE = higher rank)
        - Earnings_quality: 1 - accruals_ratio_rank (lower accruals = higher quality)
        - Inverse_leverage: 1 - debt_equity_rank (lower D/E = higher rank)

        Args:
            metrics_df: DataFrame from get_quality_metrics_batch()

        Returns:
            DataFrame with additional columns: roe_rank, earnings_quality,
            inverse_leverage, quality_score
        """
        df = metrics_df.copy()

        # ROE rank: higher ROE = higher rank (0-1 percentile)
        df['roe_rank'] = df['roe'].rank(pct=True, na_option='bottom')

        # Earnings quality: lower accruals ratio = higher quality
        # Rank accruals (higher = worse), then invert
        df['accruals_rank'] = df['accruals_ratio'].rank(pct=True, na_option='top')
        df['earnings_quality'] = 1 - df['accruals_rank']

        # Leverage rank: lower debt/equity = higher rank
        df['leverage_rank'] = df['debt_to_equity'].rank(pct=True, na_option='top')
        df['inverse_leverage'] = 1 - df['leverage_rank']

        # Composite quality score (per architecture spec)
        df['quality_score'] = (
            0.40 * df['roe_rank'] +
            0.30 * df['earnings_quality'] +
            0.30 * df['inverse_leverage']
        )

        return df

    def get_api_call_count(self) -> int:
        """Return number of API calls made this session."""
        return self._call_count

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            symbol: If provided, clear only this symbol's cache.
                   If None, clear all cached data.

        Returns:
            Number of cache files deleted.
        """
        deleted = 0

        if symbol:
            for endpoint in ['overview', 'balance_sheet', 'income_statement', 'cash_flow']:
                cache_path = self._get_cache_path(symbol, endpoint)
                if cache_path.exists():
                    cache_path.unlink()
                    deleted += 1
        else:
            for cache_file in self.CACHE_DIR.glob('*.json'):
                cache_file.unlink()
                deleted += 1

        return deleted
