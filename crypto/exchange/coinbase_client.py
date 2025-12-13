"""
Coinbase Advanced Trade API client for crypto derivatives trading.

Provides:
- Historical OHLCV data fetching with resampling (4h, 1w)
- Order execution (market, limit, stop)
- Position management for futures
- Simulation mode for paper trading (Coinbase has no native paper trading)
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from coinbase.rest import RESTClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Public API base URL (no auth required)
COINBASE_PUBLIC_API = "https://api.exchange.coinbase.com"


class CoinbaseClient:
    """
    Client for Coinbase Advanced Trade API.

    Supports both live trading and simulation mode for paper trading.
    Simulation mode maintains mock orders, positions, and balances.
    """

    # Granularity mapping for Coinbase API
    GRANULARITY_MAP = {
        "1m": "ONE_MINUTE",
        "5m": "FIVE_MINUTE",
        "15m": "FIFTEEN_MINUTE",
        "30m": "THIRTY_MINUTE",
        "1h": "ONE_HOUR",
        "2h": "TWO_HOUR",
        "6h": "SIX_HOUR",
        "1d": "ONE_DAY",
    }

    # Seconds per granularity for calculating time ranges
    GRANULARITY_SECONDS = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800,
        "ONE_HOUR": 3600,
        "TWO_HOUR": 7200,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        simulation_mode: bool = True,
    ) -> None:
        """
        Initialize Coinbase Advanced Trade client.

        Args:
            api_key: Coinbase API key (defaults to COINBASE_API_KEY env var)
            api_secret: Coinbase private key (defaults to COINBASE_API_SECRET env var)
            simulation_mode: If True, all orders are simulated (default: True)
        """
        self.simulation_mode = simulation_mode
        load_dotenv()

        # Simulation state
        self._mock_orders: List[Dict[str, Any]] = []
        self._mock_position: Optional[Dict[str, Any]] = None
        self._mock_balance: Dict[str, float] = {"USDC": 1000.0}
        self._trade_history: List[Dict[str, Any]] = []

        # Load credentials
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")

        if not self.api_key or not self.api_secret:
            logger.warning(
                "Coinbase API credentials not found. Client will be unauthenticated."
            )
            self.client = None
        else:
            try:
                # Handle private key formatting (env vars may have literal \n)
                if self.api_secret:
                    self.api_secret = self.api_secret.replace("\\n", "\n").strip()
                    # Add headers if missing
                    if not self.api_secret.startswith("-----BEGIN"):
                        self.api_secret = (
                            f"-----BEGIN EC PRIVATE KEY-----\n"
                            f"{self.api_secret}\n"
                            f"-----END EC PRIVATE KEY-----"
                        )

                self.client = RESTClient(
                    api_key=self.api_key, api_secret=self.api_secret
                )
                logger.info(
                    "Coinbase client initialized (simulation_mode=%s)",
                    self.simulation_mode,
                )
            except Exception as e:
                logger.error("Failed to initialize Coinbase client: %s", e)
                self.client = None

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_historical_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 300,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Coinbase.

        Automatically handles resampling for intervals not natively supported
        (4h resampled from 1h, 1w resampled from 1d).

        Falls back to public API if authenticated client fails.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'ETH-USD')
            interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d', '1w')
            limit: Number of bars to fetch (default: 300)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime (UTC)
        """
        # Try authenticated client first, then fall back to public API
        if self.client:
            try:
                df = self._fetch_via_sdk(symbol, interval, limit, start_time, end_time)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning("SDK fetch failed, trying public API: %s", e)

        # Fall back to public API (no auth required)
        return self._fetch_via_public_api(symbol, interval, limit, start_time, end_time)

    def _fetch_via_sdk(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: Optional[int],
        end_time: Optional[int],
    ) -> pd.DataFrame:
        """Fetch OHLCV data via authenticated SDK."""
        granularity, resample_rule = self._resolve_granularity(interval)
        limit_bars = 300 if resample_rule else limit

        if end_time is None:
            end = int(datetime.now().timestamp())
        else:
            end = int(end_time / 1000)

        if start_time is None:
            gran_seconds = self.GRANULARITY_SECONDS.get(granularity, 3600)
            start = end - (limit_bars * gran_seconds)
        else:
            start = int(start_time / 1000)

        candles = self.client.get_candles(
            product_id=symbol,
            start=str(start),
            end=str(end),
            granularity=granularity,
        )

        data = self._parse_candles_response(candles)
        if not data:
            return pd.DataFrame()

        df = self._build_ohlcv_dataframe(data)

        if resample_rule:
            df = self._resample_ohlcv(df, resample_rule)

        return df.tail(limit)

    def _fetch_via_public_api(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: Optional[int],
        end_time: Optional[int],
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data via public API (no auth required).

        Public API uses different granularity format (seconds instead of enum).
        """
        try:
            # Map interval to seconds for public API
            interval_seconds = {
                "1m": 60,
                "5m": 300,
                "15m": 900,
                "1h": 3600,
                "6h": 21600,
                "1d": 86400,
            }

            # Handle 4h and 1w via resampling
            # Public API returns max ~300 candles, so limit source bars
            resample_rule = None
            if interval == "4h":
                granularity = 3600  # Fetch 1h, resample to 4h
                resample_rule = "4h"
                limit_bars = min(limit * 4, 300)  # Max 300 from API
            elif interval == "1w":
                granularity = 86400  # Fetch 1d, resample to 1w
                resample_rule = "1W"
                limit_bars = min(limit * 7, 300)  # Max 300 from API
            else:
                granularity = interval_seconds.get(interval, 3600)
                limit_bars = min(limit, 300)

            if end_time is None:
                end = int(datetime.now().timestamp())
            else:
                end = int(end_time / 1000)

            if start_time is None:
                start = end - (limit_bars * granularity)
            else:
                start = int(start_time / 1000)

            url = f"{COINBASE_PUBLIC_API}/products/{symbol}/candles"
            params = {
                "start": start,
                "end": end,
                "granularity": granularity,
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if not data:
                logger.warning("No candles returned from public API for %s", symbol)
                return pd.DataFrame()

            # Public API format: [time, low, high, open, close, volume]
            df = pd.DataFrame(
                data, columns=["timestamp", "low", "high", "open", "close", "volume"]
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df = df.set_index("datetime").sort_index()
            df = df[["open", "high", "low", "close", "volume"]]

            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            if resample_rule:
                df = self._resample_ohlcv(df, resample_rule)

            return df.tail(limit)

        except Exception as e:
            logger.error("Error fetching from public API for %s: %s", symbol, e)
            return pd.DataFrame()

    def _resolve_granularity(self, interval: str) -> Tuple[str, Optional[str]]:
        """
        Resolve interval to Coinbase granularity and resampling rule.

        Args:
            interval: Requested interval

        Returns:
            Tuple of (coinbase_granularity, resample_rule or None)
        """
        if interval == "4h":
            return "ONE_HOUR", "4h"
        elif interval == "1w":
            return "ONE_DAY", "1W"
        else:
            return self.GRANULARITY_MAP.get(interval, "ONE_HOUR"), None

    def _parse_candles_response(self, candles: Any) -> List[Dict[str, Any]]:
        """Parse candles from Coinbase API response."""
        if hasattr(candles, "candles"):
            raw_data = candles.candles
        else:
            raw_data = candles.get("candles", [])

        result = []
        for c in raw_data:
            if hasattr(c, "start"):
                result.append(
                    {
                        "timestamp": c.start,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                )
            else:
                result.append(
                    {
                        "timestamp": c.get("start"),
                        "open": c.get("open"),
                        "high": c.get("high"),
                        "low": c.get("low"),
                        "close": c.get("close"),
                        "volume": c.get("volume"),
                    }
                )
        return result

    def _build_ohlcv_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build OHLCV DataFrame from parsed candle data."""
        df = pd.DataFrame(data)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("datetime").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV data to larger timeframe."""
        resampled = df.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        return resampled.dropna()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Current price or None if unavailable
        """
        # Try SDK first
        if self.client:
            try:
                ticker = self.client.get_product(product_id=symbol)
                if hasattr(ticker, "price"):
                    return float(ticker.price)
                return float(ticker.get("price", 0))
            except Exception as e:
                logger.warning("SDK price fetch failed, trying public API: %s", e)

        # Fall back to public API
        return self._get_price_via_public_api(symbol)

    def _get_price_via_public_api(self, symbol: str) -> Optional[float]:
        """Get current price via public API."""
        try:
            url = f"{COINBASE_PUBLIC_API}/products/{symbol}/ticker"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data.get("price", 0))
        except Exception as e:
            logger.error("Error fetching price from public API for %s: %s", symbol, e)
            return None

    # =========================================================================
    # ACCOUNT
    # =========================================================================

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dict with account info including balances
        """
        if self.simulation_mode:
            return {
                "accounts": [
                    {
                        "currency": "USDC",
                        "available_balance": {"value": self._mock_balance["USDC"]},
                    }
                ]
            }

        if not self.client:
            return {}

        try:
            response = self.client.get_accounts()
            if hasattr(response, "accounts"):
                return {
                    "accounts": [self._account_to_dict(a) for a in response.accounts]
                }
            return response
        except Exception as e:
            logger.error("Error fetching accounts: %s", e)
            return {}

    def _account_to_dict(self, account: Any) -> Dict[str, Any]:
        """Convert account object to dict."""
        if isinstance(account, dict):
            return account
        return {
            "uuid": getattr(account, "uuid", None),
            "name": getattr(account, "name", None),
            "currency": getattr(account, "currency", None),
            "available_balance": getattr(account, "available_balance", {}),
            "default": getattr(account, "default", False),
            "active": getattr(account, "active", True),
            "type": getattr(account, "type", None),
            "hold": getattr(account, "hold", {}),
        }

    # =========================================================================
    # ORDERS
    # =========================================================================

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new order.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: 'BUY' or 'SELL'
            order_type: 'MARKET', 'LIMIT', or 'STOP'
            quantity: Order quantity in base currency
            price: Limit price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)
            client_order_id: Optional custom order ID

        Returns:
            Dict with order details and status
        """
        if not client_order_id:
            client_order_id = str(uuid.uuid4())

        if self.simulation_mode:
            return self._create_mock_order(
                symbol, side, order_type, quantity, price, stop_price, client_order_id
            )

        if not self.client:
            raise ValueError("Client not initialized")

        return self._create_live_order(
            symbol, side, order_type, quantity, price, stop_price, client_order_id
        )

    def _create_mock_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        stop_price: Optional[float],
        client_order_id: str,
    ) -> Dict[str, Any]:
        """Create a simulated order for paper trading."""
        logger.info(
            "[SIMULATION] Creating order: %s %s %s @ %s",
            side,
            quantity,
            symbol,
            price or "MARKET",
        )

        mock_order = {
            "order_id": str(uuid.uuid4()),
            "client_order_id": client_order_id,
            "product_id": symbol,
            "side": side,
            "status": "OPEN",
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Market orders fill immediately
        if order_type == "MARKET":
            mock_order["status"] = "FILLED"
            fill_price = price or self.get_current_price(symbol) or 0
            self._update_mock_position(symbol, side, quantity, fill_price)
            self._record_trade(mock_order, fill_price)

        self._mock_orders.append(mock_order)

        return {
            "success": True,
            "order_id": mock_order["order_id"],
            "response": mock_order,
        }

    def _create_live_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        stop_price: Optional[float],
        client_order_id: str,
    ) -> Dict[str, Any]:
        """Create a live order on Coinbase."""
        try:
            order_config = self._build_order_config(
                order_type, quantity, price, stop_price, side
            )
            response = self.client.create_order(
                client_order_id=client_order_id,
                product_id=symbol,
                side=side,
                order_configuration=order_config,
            )
            return response
        except Exception as e:
            logger.error("Error creating order: %s", e)
            raise

    def _build_order_config(
        self,
        order_type: str,
        quantity: float,
        price: Optional[float],
        stop_price: Optional[float],
        side: str,
    ) -> Dict[str, Any]:
        """Build order configuration for Coinbase API."""
        if order_type == "MARKET":
            return {"market_market_ioc": {"base_size": str(quantity)}}
        elif order_type == "LIMIT":
            if not price:
                raise ValueError("Price required for LIMIT order")
            return {
                "limit_limit_gtc": {
                    "base_size": str(quantity),
                    "limit_price": str(price),
                    "post_only": False,
                }
            }
        elif order_type == "STOP":
            if not stop_price:
                raise ValueError("Stop price required for STOP order")
            stop_direction = (
                "STOP_DIRECTION_STOP_UP" if side == "BUY" else "STOP_DIRECTION_STOP_DOWN"
            )
            return {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(quantity),
                    "limit_price": str(price) if price else str(stop_price),
                    "stop_price": str(stop_price),
                    "stop_direction": stop_direction,
                }
            }
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def _update_mock_position(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> None:
        """Update simulated position state."""
        if not self._mock_position:
            self._mock_position = {
                "product_id": symbol,
                "side": side,
                "quantity": quantity,
                "avg_entry_price": price,
                "unrealized_pnl": 0.0,
            }
        else:
            curr_side = self._mock_position["side"]
            curr_qty = float(self._mock_position["quantity"])
            curr_price = float(self._mock_position["avg_entry_price"])

            if side == curr_side:
                # Adding to position
                new_qty = curr_qty + quantity
                total_cost = (curr_qty * curr_price) + (quantity * price)
                new_avg = total_cost / new_qty
                self._mock_position["quantity"] = new_qty
                self._mock_position["avg_entry_price"] = new_avg
            else:
                # Closing/flipping position
                if quantity >= curr_qty:
                    remaining = quantity - curr_qty
                    if remaining > 0:
                        self._mock_position = {
                            "product_id": symbol,
                            "side": side,
                            "quantity": remaining,
                            "avg_entry_price": price,
                            "unrealized_pnl": 0.0,
                        }
                    else:
                        self._mock_position = None
                else:
                    self._mock_position["quantity"] = curr_qty - quantity

    def _record_trade(self, order: Dict[str, Any], fill_price: float) -> None:
        """Record trade in history for paper trading analysis."""
        self._trade_history.append(
            {
                "order_id": order["order_id"],
                "symbol": order["product_id"],
                "side": order["side"],
                "quantity": order["quantity"],
                "price": fill_price,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order by ID.

        Args:
            order_id: Order ID to cancel

        Returns:
            Dict with cancellation status
        """
        if self.simulation_mode:
            logger.info("[SIMULATION] Canceling order: %s", order_id)
            self._mock_orders = [
                o for o in self._mock_orders if o["order_id"] != order_id
            ]
            return {"success": True}

        if not self.client:
            raise ValueError("Client not initialized")

        try:
            return self.client.cancel_orders(order_ids=[order_id])
        except Exception as e:
            logger.error("Error canceling order: %s", e)
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of open orders
        """
        if self.simulation_mode:
            orders = [o for o in self._mock_orders if o["status"] == "OPEN"]
            if symbol:
                orders = [o for o in orders if o["product_id"] == symbol]
            return orders

        if not self.client:
            return []

        try:
            orders = self.client.list_orders(product_id=symbol, order_status=["OPEN"])
            if hasattr(orders, "orders"):
                return [self._order_to_dict(o) for o in orders.orders]
            return orders.get("orders", [])
        except Exception as e:
            logger.error("Error fetching open orders: %s", e)
            return []

    def _order_to_dict(self, order: Any) -> Dict[str, Any]:
        """Convert order object to dict."""
        if isinstance(order, dict):
            return order
        return {
            "order_id": getattr(order, "order_id", None),
            "product_id": getattr(order, "product_id", None),
            "side": getattr(order, "side", None),
            "status": getattr(order, "status", None),
            "order_configuration": getattr(order, "order_configuration", None),
        }

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position dict or None if no position
        """
        if self.simulation_mode:
            if self._mock_position and self._mock_position.get("product_id") == symbol:
                return self._mock_position
            return None

        if not self.client:
            return None

        try:
            if hasattr(self.client, "get_futures_positions"):
                positions = self.client.get_futures_positions()
                if hasattr(positions, "positions"):
                    for p in positions.positions:
                        p_id = getattr(p, "product_id", None) or p.get("product_id")
                        if p_id == symbol:
                            return self._position_to_dict(p)
            return None
        except Exception as e:
            logger.error("Error fetching position: %s", e)
            return None

    def _position_to_dict(self, position: Any) -> Dict[str, Any]:
        """Convert position object to dict."""
        if isinstance(position, dict):
            return position
        return {
            "product_id": getattr(position, "product_id", None),
            "side": getattr(position, "side", None),
            "quantity": getattr(position, "number_of_contracts", None),
            "avg_entry_price": getattr(position, "avg_entry_price", None),
            "unrealized_pnl": getattr(position, "unrealized_pnl", None),
        }

    # =========================================================================
    # PAPER TRADING UTILITIES
    # =========================================================================

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get paper trading trade history."""
        return self._trade_history.copy()

    def reset_simulation(self, starting_balance: float = 1000.0) -> None:
        """
        Reset simulation state for fresh paper trading.

        Args:
            starting_balance: Starting USDC balance
        """
        self._mock_orders = []
        self._mock_position = None
        self._mock_balance = {"USDC": starting_balance}
        self._trade_history = []
        logger.info("Simulation reset with balance: $%.2f", starting_balance)

    def set_mock_balance(self, currency: str, amount: float) -> None:
        """
        Set mock balance for paper trading.

        Args:
            currency: Currency code (e.g., 'USDC')
            amount: Balance amount
        """
        self._mock_balance[currency] = amount
