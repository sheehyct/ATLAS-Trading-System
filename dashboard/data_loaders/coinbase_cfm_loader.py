"""
Coinbase CFM Data Loader for Dashboard.

Fetches real Coinbase CFM derivatives trading data using read-only API access.
Provides methods matching the dashboard data loader interface pattern.

Configuration:
    Set COINBASE_READONLY_API_KEY and COINBASE_READONLY_API_SECRET environment
    variables for read-only API access to your Coinbase account.

Supported Products:
    - Crypto Perpetuals: BIP (Bitcoin), ETP (Ether), SOP (Solana), ADP (Cardano), XRP
    - Commodity Futures: SLRH (Silver), GOLJ (Gold)

Usage:
    from dashboard.data_loaders.coinbase_cfm_loader import CoinbaseCFMLoader

    loader = CoinbaseCFMLoader()
    if loader._connected:
        positions = loader.get_open_positions()
        closed = loader.get_closed_trades()
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import Coinbase client and calculator
try:
    from crypto.exchange.coinbase_client import CoinbaseClient
    from crypto.analytics.coinbase_cfm_calculator import (
        CoinbaseCFMCalculator,
        CFMOpenPosition,
        CFM_SYMBOL_MAP,
        CRYPTO_PERPS,
        COMMODITY_FUTURES,
    )
    _IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CFM imports not available: {e}")
    _IMPORTS_AVAILABLE = False


# Cache TTL in seconds (5 minutes)
DEFAULT_CACHE_TTL = 300


class CoinbaseCFMLoader:
    """
    Load Coinbase CFM trading data for dashboard display.

    Uses read-only API access (COINBASE_READONLY_API_KEY) to fetch:
    - Filled orders for P/L calculation
    - Open positions
    - Funding payments (perpetuals only)

    Implements the dashboard data loader interface pattern:
    - _connected flag for connection status
    - init_error for error tracking
    - Methods return List[Dict] or Dict (never raw objects)
    - Graceful degradation with empty defaults
    - TTL-based caching to respect API rate limits
    """

    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL):
        """
        Initialize the CFM data loader.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 minutes)
        """
        self._connected = False
        self.init_error: Optional[str] = None
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # key -> (timestamp, data)

        if not _IMPORTS_AVAILABLE:
            self.init_error = "CFM modules not available"
            logger.warning("CoinbaseCFMLoader: imports not available")
            return

        # Check if readonly credentials are configured
        readonly_key = os.getenv("COINBASE_READONLY_API_KEY")
        readonly_secret = os.getenv("COINBASE_READONLY_API_SECRET")

        if not readonly_key or not readonly_secret:
            self.init_error = "COINBASE_READONLY_API_KEY/SECRET not configured"
            logger.warning(f"CoinbaseCFMLoader: {self.init_error}")
            return

        # Initialize client and calculator
        try:
            self._client = CoinbaseClient()
            self._calculator = CoinbaseCFMCalculator()

            # Test connection
            if self._client.is_readonly_available():
                self._connected = True
                logger.info("CoinbaseCFMLoader connected with readonly API access")
            else:
                self.init_error = "Readonly client initialization failed"
                logger.warning(f"CoinbaseCFMLoader: {self.init_error}")

        except Exception as e:
            self.init_error = str(e)
            logger.error(f"CoinbaseCFMLoader init error: {e}")

    # =========================================================================
    # CACHING
    # =========================================================================

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        if cache_key in self._cache:
            cached_at, data = self._cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_at).total_seconds()
            if age < self._cache_ttl:
                logger.debug(f"Cache hit for {cache_key} (age: {age:.0f}s)")
                return data
        return None

    def _set_cached(self, cache_key: str, data: Any) -> None:
        """Store data in cache."""
        self._cache[cache_key] = (datetime.now(timezone.utc), data)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache = {}
        logger.debug("CoinbaseCFMLoader cache cleared")

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_and_process_fills(self, days: int = 90, force_refresh: bool = False) -> bool:
        """
        Fetch fills from Coinbase and process with calculator.

        Args:
            days: Number of days of history to fetch
            force_refresh: If True, bypass cache

        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            return False

        cache_key = f"fills_{days}"

        if not force_refresh:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return True

        try:
            # Fetch fills
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            fills = self._client.get_fills_live(
                start_date=start_date,
                end_date=end_date,
                limit=1000,
            )

            if fills:
                self._calculator.process_fills(fills)
                self._set_cached(cache_key, True)
                logger.info(f"Processed {len(fills)} CFM fills from Coinbase")
                return True
            else:
                logger.warning("No CFM fills returned from Coinbase")
                return False

        except Exception as e:
            logger.error(f"Error fetching CFM fills: {e}")
            return False

    # =========================================================================
    # ACCOUNT METHODS
    # =========================================================================

    def get_account_summary(self) -> Dict:
        """
        Get CFM account summary.

        Returns:
            Dict with balance info, realized P/L, positions summary.
            Empty dict if not connected.
        """
        if not self._connected:
            return {}

        cache_key = "account_summary"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # Fetch portfolio summary from Coinbase
            portfolio = self._client.get_cfm_portfolio_summary()

            # Ensure fills are processed
            self._fetch_and_process_fills()

            # Get P/L totals
            pnl_totals = self._calculator.get_realized_pnl_total()
            positions = self.get_open_positions()

            # Calculate unrealized P/L from positions
            unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)

            summary = {
                "portfolio": portfolio,
                "realized_pnl": pnl_totals.get("net_pnl", 0),
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": pnl_totals.get("net_pnl", 0) + unrealized_pnl,
                "total_fees": pnl_totals.get("total_fees", 0),
                "trade_count": pnl_totals.get("trade_count", 0),
                "open_positions_count": len(positions),
            }

            self._set_cached(cache_key, summary)
            return summary

        except Exception as e:
            logger.error(f"Error getting CFM account summary: {e}")
            return {}

    def get_performance_metrics(self) -> Dict:
        """
        Get trading performance metrics.

        Returns:
            Dict with win_rate, profit_factor, expectancy, etc.
            Empty dict if not connected.
        """
        if not self._connected:
            return {}

        cache_key = "performance_metrics"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # Ensure fills are processed
            self._fetch_and_process_fills()

            metrics = self._calculator.get_performance_metrics()
            self._set_cached(cache_key, metrics)
            return metrics

        except Exception as e:
            logger.error(f"Error getting CFM performance metrics: {e}")
            return {}

    # =========================================================================
    # POSITION METHODS
    # =========================================================================

    def get_open_positions(self) -> List[Dict]:
        """
        Get open CFM positions.

        Returns:
            List of position dicts with product_id, side, quantity,
            avg_entry_price, unrealized_pnl, etc.
            Empty list if not connected.
        """
        if not self._connected:
            return []

        cache_key = "open_positions"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # Try to get positions directly from Coinbase
            cfm_positions = self._client.get_cfm_positions()

            if cfm_positions:
                # Use live positions from Coinbase
                positions = []
                for pos in cfm_positions:
                    positions.append({
                        "product_id": pos.get("product_id", ""),
                        "base_symbol": pos.get("symbol", "").split("-")[0] if pos.get("symbol") else "",
                        "side": pos.get("side") or pos.get("position_side", ""),
                        "quantity": float(pos.get("number_of_contracts") or pos.get("net_size") or 0),
                        "avg_entry_price": float(pos.get("avg_entry_price") or 0),
                        "current_price": float(pos.get("current_price") or pos.get("mark_price") or 0),
                        "unrealized_pnl": float(pos.get("unrealized_pnl") or pos.get("aggregated_pnl") or 0),
                        "leverage": pos.get("leverage"),
                        "liquidation_price": pos.get("liquidation_price"),
                        "product_type": self._classify_product(pos.get("product_id", "")),
                    })
                self._set_cached(cache_key, positions)
                return positions

            # Fall back to calculated positions from fills
            self._fetch_and_process_fills()
            calc_positions = self._calculator.get_open_positions()
            positions = [p.to_dict() for p in calc_positions]
            self._set_cached(cache_key, positions)
            return positions

        except Exception as e:
            logger.error(f"Error getting CFM open positions: {e}")
            return []

    def _classify_product(self, product_id: str) -> str:
        """Classify product type from product_id."""
        if not product_id:
            return "unknown"
        base = product_id.split("-")[0].upper()
        if base in CRYPTO_PERPS:
            return "crypto_perp"
        elif base in COMMODITY_FUTURES:
            return "commodity_future"
        return "unknown"

    # =========================================================================
    # TRADE HISTORY
    # =========================================================================

    def get_closed_trades(self, limit: int = 50) -> List[Dict]:
        """
        Get closed trades with P/L.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts sorted by exit_time descending.
            Empty list if not connected.
        """
        if not self._connected:
            return []

        cache_key = f"closed_trades_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            self._fetch_and_process_fills()
            trades = self._calculator.get_closed_trades(limit=limit)
            self._set_cached(cache_key, trades)
            return trades

        except Exception as e:
            logger.error(f"Error getting CFM closed trades: {e}")
            return []

    def get_realized_pnl(self, days: int = 90) -> Dict:
        """
        Get realized P/L summary.

        Args:
            days: Number of days of history

        Returns:
            Dict with gross_pnl, total_fees, net_pnl, trade_count.
            Empty dict if not connected.
        """
        if not self._connected:
            return {}

        try:
            self._fetch_and_process_fills(days=days)
            return self._calculator.get_realized_pnl_total()
        except Exception as e:
            logger.error(f"Error getting CFM realized P/L: {e}")
            return {}

    # =========================================================================
    # P/L BREAKDOWN
    # =========================================================================

    def get_pnl_by_product(self) -> Dict[str, Dict]:
        """
        Get P/L breakdown by product/symbol.

        Returns:
            Dict mapping symbol to P/L metrics (BIP, ETP, etc.)
        """
        if not self._connected:
            return {}

        cache_key = "pnl_by_product"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            self._fetch_and_process_fills()
            pnl = self._calculator.get_pnl_by_product()
            self._set_cached(cache_key, pnl)
            return pnl

        except Exception as e:
            logger.error(f"Error getting P/L by product: {e}")
            return {}

    def get_pnl_by_product_type(self) -> Dict[str, Dict]:
        """
        Get P/L breakdown by product type (crypto_perp vs commodity_future).

        Returns:
            Dict with 'crypto_perp' and 'commodity_future' keys
        """
        if not self._connected:
            return {}

        cache_key = "pnl_by_type"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            self._fetch_and_process_fills()
            pnl = self._calculator.get_pnl_by_product_type()
            self._set_cached(cache_key, pnl)
            return pnl

        except Exception as e:
            logger.error(f"Error getting P/L by product type: {e}")
            return {}

    # =========================================================================
    # FUNDING PAYMENTS (PERPETUALS ONLY)
    # =========================================================================

    def get_funding_payments(self, days: int = 30) -> List[Dict]:
        """
        Get funding rate payments for perpetuals.

        Perpetuals (BIP, ETP, SOP, ADP, XRP) have 8-hour funding intervals.
        Commodity futures (SLRH, GOLJ) do not have funding.

        Args:
            days: Number of days of history

        Returns:
            List of funding payment dicts.
            Empty list if not connected.
        """
        if not self._connected:
            return []

        cache_key = f"funding_{days}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            payments = self._client.get_cfm_funding_payments(days=days)
            self._set_cached(cache_key, payments)
            return payments

        except Exception as e:
            logger.error(f"Error getting CFM funding payments: {e}")
            return []

    def get_funding_summary(self) -> Dict:
        """
        Get funding payment summary.

        Returns:
            Dict with total_paid, total_received, net_funding.
        """
        if not self._connected:
            return {"total_paid": 0, "total_received": 0, "net_funding": 0, "payment_count": 0}

        try:
            payments = self.get_funding_payments()
            self._calculator.add_funding_payments(payments)
            return self._calculator.get_funding_summary()

        except Exception as e:
            logger.error(f"Error getting funding summary: {e}")
            return {"total_paid": 0, "total_received": 0, "net_funding": 0, "payment_count": 0}

    # =========================================================================
    # COMBINED SUMMARY
    # =========================================================================

    def get_combined_summary(self) -> Dict:
        """
        Get combined summary for dashboard header display.

        Returns:
            Dict with sections for crypto_perps, commodity_futures, and combined totals.
        """
        if not self._connected:
            return {
                "connected": False,
                "error": self.init_error,
                "crypto_perps": {},
                "commodity_futures": {},
                "combined": {},
            }

        try:
            pnl_by_type = self.get_pnl_by_product_type()
            positions = self.get_open_positions()
            funding = self.get_funding_summary()

            # Separate positions by type
            crypto_positions = [p for p in positions if p.get("product_type") == "crypto_perp"]
            commodity_positions = [p for p in positions if p.get("product_type") == "commodity_future"]

            crypto_unrealized = sum(p.get("unrealized_pnl", 0) for p in crypto_positions)
            commodity_unrealized = sum(p.get("unrealized_pnl", 0) for p in commodity_positions)

            crypto_pnl = pnl_by_type.get("crypto_perp", {})
            commodity_pnl = pnl_by_type.get("commodity_future", {})

            return {
                "connected": True,
                "crypto_perps": {
                    "realized_pnl": crypto_pnl.get("net_pnl", 0),
                    "unrealized_pnl": crypto_unrealized,
                    "total_fees": crypto_pnl.get("total_fees", 0),
                    "trade_count": crypto_pnl.get("trade_count", 0),
                    "win_rate": crypto_pnl.get("win_rate", 0),
                    "funding_paid": funding.get("total_paid", 0),
                    "funding_received": funding.get("total_received", 0),
                    "net_funding": funding.get("net_funding", 0),
                    "open_positions": len(crypto_positions),
                    "products": CRYPTO_PERPS,
                },
                "commodity_futures": {
                    "realized_pnl": commodity_pnl.get("net_pnl", 0),
                    "unrealized_pnl": commodity_unrealized,
                    "total_fees": commodity_pnl.get("total_fees", 0),
                    "trade_count": commodity_pnl.get("trade_count", 0),
                    "win_rate": commodity_pnl.get("win_rate", 0),
                    "open_positions": len(commodity_positions),
                    "products": COMMODITY_FUTURES,
                },
                "combined": {
                    "realized_pnl": crypto_pnl.get("net_pnl", 0) + commodity_pnl.get("net_pnl", 0),
                    "unrealized_pnl": crypto_unrealized + commodity_unrealized,
                    "total_pnl": (
                        crypto_pnl.get("net_pnl", 0) + commodity_pnl.get("net_pnl", 0) +
                        crypto_unrealized + commodity_unrealized
                    ),
                    "total_fees": crypto_pnl.get("total_fees", 0) + commodity_pnl.get("total_fees", 0),
                    "trade_count": crypto_pnl.get("trade_count", 0) + commodity_pnl.get("trade_count", 0),
                    "open_positions": len(positions),
                },
            }

        except Exception as e:
            logger.error(f"Error getting combined summary: {e}")
            return {
                "connected": True,
                "error": str(e),
                "crypto_perps": {},
                "commodity_futures": {},
                "combined": {},
            }

    # =========================================================================
    # STATUS METHODS
    # =========================================================================

    def is_connected(self) -> bool:
        """Check if loader is connected to Coinbase."""
        return self._connected

    def get_status(self) -> Dict:
        """
        Get loader status for dashboard display.

        Returns:
            Dict with connected, error, cache_size, etc.
        """
        return {
            "connected": self._connected,
            "error": self.init_error,
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl,
            "products": {
                "crypto_perps": CRYPTO_PERPS if _IMPORTS_AVAILABLE else [],
                "commodity_futures": COMMODITY_FUTURES if _IMPORTS_AVAILABLE else [],
            },
        }
