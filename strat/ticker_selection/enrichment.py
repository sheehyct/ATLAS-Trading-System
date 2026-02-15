"""
Finviz enrichment for ticker selection candidates.

Adds human-readable context (sector, earnings, analyst consensus, news)
to scored candidates. Informational only -- does NOT affect scoring.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FinvizEnrichment:
    """Enrichment data for a single symbol from Finviz."""

    symbol: str = ''
    sector: str = ''
    industry: str = ''
    earnings_date: str = ''
    analyst_recommendation: str = ''
    target_price: Optional[float] = None
    news_headlines: List[str] = field(default_factory=list)
    fetch_error: str = ''
    cached: bool = False

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON output (truncates news to 3)."""
        d = {
            'sector': self.sector,
            'industry': self.industry,
            'earnings_date': self.earnings_date,
            'analyst_recommendation': self.analyst_recommendation,
            'target_price': self.target_price,
            'news_headlines': self.news_headlines[:3],
        }
        if self.fetch_error:
            d['fetch_error'] = self.fetch_error
        return d


class FinvizEnricher:
    """
    Enriches ticker candidates with Finviz fundamental data.

    Uses file-based caching with configurable TTL to avoid
    re-scraping. All errors are caught -- pipeline never blocks.
    """

    def __init__(
        self,
        cache_dir: str = 'data/finviz_cache/',
        cache_ttl: int = 6 * 3600,
        max_workers: int = 4,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers

    def enrich_candidates(self, symbols: List[str]) -> Dict[str, FinvizEnrichment]:
        """
        Fetch enrichment data for all symbols in parallel.

        Returns a dict keyed by symbol for ALL input symbols,
        even on failure (with fetch_error populated).
        """
        if not symbols:
            return {}

        unique = list(dict.fromkeys(symbols))
        results: Dict[str, FinvizEnrichment] = {}
        workers = min(self.max_workers, len(unique))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._fetch_single, sym): sym
                for sym in unique
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    results[sym] = future.result()
                except Exception as e:
                    logger.warning(f"Finviz enrichment future failed for {sym}: {e}")
                    results[sym] = FinvizEnrichment(
                        symbol=sym, fetch_error=str(e)
                    )

        return results

    def _fetch_single(self, symbol: str) -> FinvizEnrichment:
        """Check cache first, scrape on miss. Catches all errors."""
        try:
            cached = self._load_cache(symbol)
            if cached is not None:
                cached.cached = True
                return cached

            enrichment = self._scrape_finviz(symbol)
            self._save_cache(enrichment)
            return enrichment

        except Exception as e:
            logger.debug(f"Finviz fetch error for {symbol}: {e}")
            return FinvizEnrichment(symbol=symbol, fetch_error=str(e))

    def _scrape_finviz(self, symbol: str) -> FinvizEnrichment:
        """Scrape Finviz for fundamental data and news."""
        from finvizfinance.quote import finvizfinance

        ticker = finvizfinance(symbol)
        fundament = ticker.ticker_fundament()
        news_df = ticker.ticker_news()

        headlines = []
        if news_df is not None and not news_df.empty:
            title_col = 'Title' if 'Title' in news_df.columns else news_df.columns[0]
            headlines = news_df[title_col].head(3).tolist()

        return FinvizEnrichment(
            symbol=symbol,
            sector=fundament.get('Sector', ''),
            industry=fundament.get('Industry', ''),
            earnings_date=self._parse_earnings_date(
                fundament.get('Earnings', '')
            ),
            analyst_recommendation=self._map_recommendation(
                fundament.get('Recom', '')
            ),
            target_price=self._safe_float(fundament.get('Target Price', '')),
            news_headlines=headlines,
        )

    def _load_cache(self, symbol: str) -> Optional[FinvizEnrichment]:
        """Load cached enrichment if present and fresh."""
        path = self.cache_dir / f"{symbol}_finviz.json"
        if not path.exists():
            return None

        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl:
            return None

        try:
            data = json.loads(path.read_text())
            return FinvizEnrichment(
                symbol=data.get('symbol', symbol),
                sector=data.get('sector', ''),
                industry=data.get('industry', ''),
                earnings_date=data.get('earnings_date', ''),
                analyst_recommendation=data.get('analyst_recommendation', ''),
                target_price=data.get('target_price'),
                news_headlines=data.get('news_headlines', []),
            )
        except Exception as e:
            logger.debug(f"Cache read error for {symbol}: {e}")
            return None

    def _save_cache(self, enrichment: FinvizEnrichment) -> None:
        """Save enrichment to file-based cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{enrichment.symbol}_finviz.json"
            data = {
                'symbol': enrichment.symbol,
                'sector': enrichment.sector,
                'industry': enrichment.industry,
                'earnings_date': enrichment.earnings_date,
                'analyst_recommendation': enrichment.analyst_recommendation,
                'target_price': enrichment.target_price,
                'news_headlines': enrichment.news_headlines,
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Cache write error for {enrichment.symbol}: {e}")

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safe float parse, returns None on failure."""
        if value is None:
            return None
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _map_recommendation(raw) -> str:
        """Convert Finviz numeric recommendation (1.0-5.0) to text."""
        value = FinvizEnricher._safe_float(raw)
        if value is None:
            return ''
        if value <= 1.5:
            return 'Strong Buy'
        if value <= 2.5:
            return 'Buy'
        if value <= 3.5:
            return 'Hold'
        if value <= 4.5:
            return 'Underperform'
        return 'Sell'

    @staticmethod
    def _parse_earnings_date(raw) -> str:
        """Clean Finviz earnings string. Returns '' for '-' or empty."""
        if not raw or str(raw).strip() in ('-', ''):
            return ''
        return str(raw).strip()
