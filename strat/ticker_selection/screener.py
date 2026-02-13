"""
Bulk snapshot screener using Alpaca's ``get_stock_snapshot()`` API.

Filters the universe down to ~300-500 liquid, volatile stocks suitable
for STRAT pattern scanning.  Uses batched requests (1,000 symbols per
call) to minimize API usage.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from strat.ticker_selection.config import TickerSelectionConfig

logger = logging.getLogger(__name__)

# Alpaca snapshots batch limit
_BATCH_SIZE = 1000


@dataclass
class ScreenedStock:
    """A stock that passed the snapshot screen."""
    symbol: str
    price: float
    dollar_volume: float
    atr_percent: float
    day_high: float
    day_low: float
    day_open: float
    day_close: float
    prev_close: float
    volume: int


class SnapshotScreener:
    """
    Screen stocks using Alpaca's snapshot endpoint.

    For ~4,000 symbols this requires only 4-5 API calls,
    completing in under 10 seconds on Algo Trader Plus.
    """

    def __init__(self, config: Optional[TickerSelectionConfig] = None):
        self.config = config or TickerSelectionConfig()
        self._client = None

    def _get_client(self):
        """Lazy-init the Alpaca data client."""
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            from config.settings import get_alpaca_credentials

            creds = get_alpaca_credentials(self.config.alpaca_account)
            self._client = StockHistoricalDataClient(
                api_key=creds['api_key'],
                secret_key=creds['secret_key'],
            )
        return self._client

    def screen(self, symbols: List[str]) -> List[ScreenedStock]:
        """
        Screen a list of symbols and return those passing all filters.

        Args:
            symbols: Universe of symbols to screen.

        Returns:
            Sorted list of ScreenedStock (descending by dollar_volume),
            capped at ``config.max_screened``.
        """
        client = self._get_client()
        passed: List[ScreenedStock] = []

        # Process in batches
        total_batches = math.ceil(len(symbols) / _BATCH_SIZE)
        for batch_idx in range(total_batches):
            start = batch_idx * _BATCH_SIZE
            batch = symbols[start:start + _BATCH_SIZE]

            try:
                snapshots = self._fetch_snapshots(client, batch)
            except Exception as e:
                logger.error(f"Snapshot batch {batch_idx + 1}/{total_batches} failed: {e}")
                continue

            for sym, snap in snapshots.items():
                stock = self._evaluate(sym, snap)
                if stock is not None:
                    passed.append(stock)

            logger.debug(
                f"Snapshot batch {batch_idx + 1}/{total_batches}: "
                f"{len(batch)} symbols -> {len(passed)} passed so far"
            )

        # Sort by dollar volume descending, cap at max_screened
        passed.sort(key=lambda s: s.dollar_volume, reverse=True)
        capped = passed[:self.config.max_screened]

        logger.info(
            f"Screener: {len(symbols)} symbols -> "
            f"{len(passed)} passed filters -> "
            f"{len(capped)} after cap"
        )
        return capped

    def _fetch_snapshots(self, client, symbols: List[str]) -> Dict:
        """Fetch snapshots for a batch of symbols."""
        from alpaca.data.requests import StockSnapshotRequest

        request = StockSnapshotRequest(symbol_or_symbols=symbols)
        return client.get_stock_snapshot(request)

    def _evaluate(self, symbol: str, snap) -> Optional[ScreenedStock]:
        """
        Evaluate a single snapshot against screening criteria.

        Returns a ScreenedStock if it passes, else None.
        """
        try:
            # Extract daily bar
            daily_bar = snap.daily_bar
            if daily_bar is None:
                return None

            price = float(daily_bar.close)
            volume = int(daily_bar.volume)

            # Price filter
            if price < self.config.min_price or price > self.config.max_price:
                return None

            # Dollar volume filter
            dollar_volume = price * volume
            if dollar_volume < self.config.min_dollar_volume:
                return None

            # ATR% estimate from single daily bar (high - low) / close
            day_high = float(daily_bar.high)
            day_low = float(daily_bar.low)
            day_range = day_high - day_low
            atr_pct = (day_range / price * 100) if price > 0 else 0.0

            # Use previous close if available for better ATR estimate
            prev_close = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else price
            true_range = max(
                day_high - day_low,
                abs(day_high - prev_close),
                abs(day_low - prev_close),
            )
            atr_pct = (true_range / price * 100) if price > 0 else 0.0

            if atr_pct < self.config.min_atr_percent:
                return None

            return ScreenedStock(
                symbol=symbol,
                price=price,
                dollar_volume=dollar_volume,
                atr_percent=round(atr_pct, 2),
                day_high=day_high,
                day_low=day_low,
                day_open=float(daily_bar.open),
                day_close=price,
                prev_close=prev_close,
                volume=volume,
            )

        except Exception as e:
            logger.debug(f"Screener skip {symbol}: {e}")
            return None
