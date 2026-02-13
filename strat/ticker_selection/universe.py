"""
Universe discovery via Alpaca ``get_all_assets()``.

Fetches the full US equity universe, filters to tradable NYSE/NASDAQ/AMEX
symbols with clean tickers, and caches the result for 24 hours.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from strat.ticker_selection.config import TickerSelectionConfig

logger = logging.getLogger(__name__)

# Only uppercase letters, 1-5 chars (excludes warrants, units, preferred, etc.)
_CLEAN_SYMBOL_RE = re.compile(r'^[A-Z]{1,5}$')

_VALID_EXCHANGES = {'NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS', 'NYSEARCA'}

# 24-hour cache lifetime
_CACHE_MAX_AGE_SECONDS = 86400


class UniverseManager:
    """Discovers and caches the tradable US equity universe."""

    def __init__(self, config: Optional[TickerSelectionConfig] = None):
        self.config = config or TickerSelectionConfig()
        self._cache_path = Path(self.config.universe_cache_path)

    def get_universe(self) -> List[str]:
        """
        Return the filtered universe of tradable US equity symbols.

        Uses a 24-hour file cache to avoid redundant API calls.
        """
        cached = self._load_cache()
        if cached is not None:
            logger.info(f"Universe loaded from cache: {len(cached)} symbols")
            return cached

        symbols = self._fetch_from_alpaca()
        self._save_cache(symbols)
        return symbols

    def _fetch_from_alpaca(self) -> List[str]:
        """Fetch all assets from Alpaca and apply filters."""
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        from config.settings import get_alpaca_credentials

        creds = get_alpaca_credentials(self.config.alpaca_account)
        client = TradingClient(
            api_key=creds['api_key'],
            secret_key=creds['secret_key'],
            paper=True,
        )

        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = client.get_all_assets(request)

        symbols = []
        for asset in assets:
            # Must be tradable
            if not asset.tradable:
                continue

            # Must be on a major exchange
            exchange = getattr(asset, 'exchange', None)
            if exchange is not None:
                # Alpaca returns AssetExchange enum; use .name for comparison
                ex_name = getattr(exchange, 'name', str(exchange)).upper()
                if ex_name not in _VALID_EXCHANGES:
                    continue

            # Clean symbol format (no warrants, units, preferred shares)
            sym = asset.symbol
            if not _CLEAN_SYMBOL_RE.match(sym):
                continue

            # Exclude leveraged/inverse ETFs and penny-stock-like tickers
            name = getattr(asset, 'name', '') or ''
            name_lower = name.lower()
            if any(kw in name_lower for kw in ['leveraged', 'inverse', 'proshares ultra', 'direxion daily']):
                continue

            symbols.append(sym)

        symbols.sort()
        logger.info(
            f"Alpaca universe: {len(assets)} total assets -> "
            f"{len(symbols)} after filters"
        )
        return symbols

    def _load_cache(self) -> Optional[List[str]]:
        """Load cached universe if fresh."""
        if not self._cache_path.exists():
            return None

        try:
            age = time.time() - self._cache_path.stat().st_mtime
            if age > _CACHE_MAX_AGE_SECONDS:
                logger.debug("Universe cache expired")
                return None

            data = json.loads(self._cache_path.read_text())
            symbols = data.get('symbols', [])
            if not symbols:
                return None
            return symbols
        except Exception as e:
            logger.warning(f"Failed to read universe cache: {e}")
            return None

    def _save_cache(self, symbols: List[str]) -> None:
        """Persist universe to cache file."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'generated_at': time.time(),
                'count': len(symbols),
                'symbols': symbols,
            }
            self._cache_path.write_text(json.dumps(data, indent=2))
            logger.info(f"Universe cached: {len(symbols)} symbols -> {self._cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write universe cache: {e}")
