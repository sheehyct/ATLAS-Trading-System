"""
Alpaca Trading Client - Order Execution Interface

Provides wrapper around alpaca-py SDK with:
- Retry logic with exponential backoff (3 attempts)
- Order submission interface (market, limit, stop orders)
- Account status monitoring
- Paper vs live environment switching
- Rate limit handling (200 req/min for paper trading)
- Connection validation and health checks

Account Configuration (via centralized config.settings):
- LARGE: $10,000 paper trading account (primary deployment)
- MID: $5,000 paper trading account
- SMALL: $3,000 paper trading account (future options deployment)

Session 70: Updated to use centralized config.settings for all credentials.
All environment variables are loaded from root .env via config.settings.

Usage:
    client = AlpacaTradingClient(account='LARGE')
    if client.connect():
        order = client.submit_market_order('SPY', 10, 'buy')
        print(f"Order submitted: {order['id']}")
"""

import time
import logging
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime

# Use centralized config (loads from root .env with all credentials)
from config.settings import get_alpaca_credentials

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    GetOrdersRequest,
    GetOptionContractsRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderStatus,
    AssetStatus,
    ContractType,
)
from alpaca.common.exceptions import APIError

# Market data imports for stock quotes (Session 83K-50)
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestBarRequest


class AlpacaTradingClient:
    """
    Production-grade wrapper for Alpaca Trading API.

    Features:
    - Automatic retry with exponential backoff
    - Environment-based paper/live switching
    - Rate limit handling
    - Comprehensive error logging
    - Order lifecycle tracking
    """

    ACCOUNT_CONFIGS = {
        'LARGE': {
            'capital': 10000,
            'description': 'Primary paper trading account for equity strategies'
        },
        'MID': {
            'capital': 5000,
            'description': 'Mid-tier paper trading account'
        },
        'SMALL': {
            'capital': 3000,
            'description': 'Future options deployment account'
        }
    }

    def __init__(
        self,
        account: str = 'LARGE',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Alpaca trading client.

        Args:
            account: Account identifier ('LARGE', 'MID', or 'SMALL')
            logger: Optional logger instance (creates default if None)
        """
        if account not in self.ACCOUNT_CONFIGS:
            raise ValueError(
                f"Invalid account '{account}'. "
                f"Must be one of: {list(self.ACCOUNT_CONFIGS.keys())}"
            )

        self.account = account
        self.account_config = self.ACCOUNT_CONFIGS[account]

        # Set up logging
        self.logger = logger or self._create_default_logger()

        # Get credentials from centralized config
        # This ensures .env is loaded from root with ALL credentials
        creds = get_alpaca_credentials(account)
        self.api_key = creds['api_key']
        self.secret_key = creds['secret_key']
        self.base_url = creds['base_url']

        # Validate credentials
        if not self.api_key or not self.secret_key:
            raise ValueError(
                f"Missing Alpaca API credentials for account '{account}'. "
                f"Check your root .env file for ALPACA_{account}_KEY and "
                f"ALPACA_{account}_SECRET environment variables."
            )

        # Trading client (initialized in connect())
        self.client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self.connected = False

        # Rate limiting (200 requests per minute for paper trading)
        self.max_requests_per_minute = 200
        self.request_timestamps: List[float] = []

        self.logger.info(
            f"AlpacaTradingClient initialized: "
            f"account={account}, "
            f"capital=${self.account_config['capital']:,}, "
            f"base_url={self.base_url}"
        )

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger if none provided."""
        logger = logging.getLogger('alpaca_trading_client')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def connect(self) -> bool:
        """
        Connect to Alpaca Trading API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=True  # Always use paper trading for safety
            )

            # Initialize market data client (Session 83K-50)
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )

            # Test connection by fetching account info
            account = self._retry_api_call(self.client.get_account)

            self.connected = True
            self.logger.info(
                f"Connected to Alpaca (paper trading): "
                f"account_id={account.id}, "
                f"equity=${float(account.equity):,.2f}, "
                f"buying_power=${float(account.buying_power):,.2f}"
            )

            return True

        except Exception as e:
            self.connected = False
            self.logger.error(f"Failed to connect to Alpaca: {str(e)}")
            return False

    def get_account(self) -> Dict[str, Any]:
        """
        Get current account information.

        Returns:
            Dict with account details (equity, buying_power, positions, etc.)

        Raises:
            RuntimeError: If not connected
        """
        self._ensure_connected()

        account = self._retry_api_call(self.client.get_account)

        return {
            'account_id': account.id,
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'account_blocked': account.account_blocked,
            'timestamp': datetime.now().isoformat()
        }

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str
    ) -> Dict[str, Any]:
        """
        Submit market order.

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            qty: Number of shares (positive integer)
            side: 'buy' or 'sell'

        Returns:
            Dict with order details (id, status, submitted_at, etc.)

        Raises:
            RuntimeError: If not connected
            ValueError: If invalid parameters
            APIError: If order submission fails
        """
        self._ensure_connected()
        self._validate_order_params(symbol, qty, side)

        # Convert side to OrderSide enum
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        # Create market order request
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )

        self.logger.info(
            f"Submitting market order: {side.upper()} {qty} {symbol}"
        )

        # Submit order with retry logic
        order = self._retry_api_call(
            self.client.submit_order,
            order_request
        )

        order_dict = self._order_to_dict(order)

        self.logger.info(
            f"Order submitted successfully: "
            f"id={order.id}, "
            f"status={order.status}"
        )

        return order_dict

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float
    ) -> Dict[str, Any]:
        """
        Submit limit order.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price (must be positive)

        Returns:
            Dict with order details

        Raises:
            RuntimeError: If not connected
            ValueError: If invalid parameters
        """
        self._ensure_connected()
        self._validate_order_params(symbol, qty, side)

        if limit_price <= 0:
            raise ValueError(f"Limit price must be positive: {limit_price}")

        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )

        self.logger.info(
            f"Submitting limit order: "
            f"{side.upper()} {qty} {symbol} @ ${limit_price:.2f}"
        )

        order = self._retry_api_call(
            self.client.submit_order,
            order_request
        )

        order_dict = self._order_to_dict(order)

        self.logger.info(f"Limit order submitted: id={order.id}")

        return order_dict

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with position details or None if no position
        """
        self._ensure_connected()

        try:
            position = self._retry_api_call(
                self.client.get_open_position,
                symbol
            )

            return {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'side': 'long' if int(position.qty) > 0 else 'short',
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'current_price': float(position.current_price)
            }

        except APIError as e:
            if 'position does not exist' in str(e).lower():
                return None
            raise

    def list_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions.

        Returns:
            List of position dicts
        """
        self._ensure_connected()

        positions = self._retry_api_call(self.client.get_all_positions)

        return [
            {
                'symbol': pos.symbol,
                'qty': int(pos.qty),
                'side': 'long' if int(pos.qty) > 0 else 'short',
                'avg_entry_price': float(pos.avg_entry_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price)
            }
            for pos in positions
        ]

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            Dict with order details
        """
        self._ensure_connected()

        order = self._retry_api_call(self.client.get_order_by_id, order_id)

        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if canceled successfully, False otherwise
        """
        self._ensure_connected()

        try:
            self._retry_api_call(self.client.cancel_order_by_id, order_id)
            self.logger.info(f"Order canceled: {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False

    def get_filled_orders(self, after: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get filled orders.

        Args:
            after: Only return orders filled after this timestamp

        Returns:
            List of filled order dicts
        """
        self._ensure_connected()

        request = GetOrdersRequest(
            status=OrderStatus.FILLED,
            after=after
        )

        orders = self._retry_api_call(self.client.get_orders, request)

        return [self._order_to_dict(order) for order in orders]

    def _retry_api_call(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """
        Execute API call with retry logic and exponential backoff.

        Args:
            func: API function to call
            *args: Positional arguments for func
            max_retries: Maximum retry attempts
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)

        Raises:
            Exception: If all retries fail
        """
        # Rate limiting check
        self._check_rate_limit()

        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)

                # Track request timestamp for rate limiting
                self.request_timestamps.append(time.time())

                return result

            except APIError as e:
                # Check if retryable error
                is_retryable = any(
                    keyword in str(e).lower()
                    for keyword in ['timeout', 'rate limit', 'connection']
                )

                if not is_retryable or attempt == max_retries - 1:
                    self.logger.error(
                        f"API call failed (attempt {attempt + 1}/{max_retries}): "
                        f"{str(e)}"
                    )
                    raise

                # Exponential backoff
                wait_time = 2 ** attempt
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)

        raise RuntimeError(f"API call failed after {max_retries} attempts")

    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        now = time.time()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if now - ts < 60
        ]

        # If at limit, wait until oldest request is 1 minute old
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest_request = self.request_timestamps[0]
            wait_time = 60 - (now - oldest_request)

            if wait_time > 0:
                self.logger.warning(
                    f"Rate limit reached, waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)

    def _ensure_connected(self):
        """Raise error if not connected."""
        if not self.connected or self.client is None:
            raise RuntimeError(
                "Not connected to Alpaca. Call connect() first."
            )

    def _validate_order_params(self, symbol: str, qty: int, side: str):
        """Validate order parameters."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")

        if qty <= 0 or not isinstance(qty, int):
            raise ValueError(f"Quantity must be positive integer: {qty}")

        if side.lower() not in ['buy', 'sell']:
            raise ValueError(f"Side must be 'buy' or 'sell': {side}")

    # =========================================================================
    # OPTIONS TRADING METHODS (Session 83K-47)
    # =========================================================================

    def get_option_contracts(
        self,
        underlying: str,
        expiration_date: Optional[str] = None,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        contract_type: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get available option contracts for an underlying symbol.

        Args:
            underlying: Underlying symbol (e.g., 'SPY', 'AAPL')
            expiration_date: Filter by exact expiration (YYYY-MM-DD format)
            expiration_date_gte: Minimum expiration date (YYYY-MM-DD format)
            expiration_date_lte: Maximum expiration date (YYYY-MM-DD format)
            contract_type: 'call' or 'put'
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price

        Returns:
            List of option contract dicts with symbol, strike, expiration, type
        """
        self._ensure_connected()

        # Build request
        request_params = {
            'underlying_symbols': [underlying.upper()],
            'status': AssetStatus.ACTIVE,
        }

        if expiration_date:
            request_params['expiration_date'] = expiration_date

        if expiration_date_gte:
            request_params['expiration_date_gte'] = expiration_date_gte

        if expiration_date_lte:
            request_params['expiration_date_lte'] = expiration_date_lte

        if contract_type:
            if contract_type.lower() == 'call':
                request_params['type'] = ContractType.CALL
            elif contract_type.lower() == 'put':
                request_params['type'] = ContractType.PUT

        if strike_price_gte is not None:
            request_params['strike_price_gte'] = str(strike_price_gte)

        if strike_price_lte is not None:
            request_params['strike_price_lte'] = str(strike_price_lte)

        request = GetOptionContractsRequest(**request_params)

        response = self._retry_api_call(
            self.client.get_option_contracts,
            request
        )

        contracts = []
        if response and hasattr(response, 'option_contracts'):
            for contract in response.option_contracts:
                contracts.append({
                    'symbol': contract.symbol,
                    'underlying': contract.underlying_symbol,
                    'strike': float(contract.strike_price),
                    'expiration': contract.expiration_date.isoformat() if contract.expiration_date else None,
                    'type': contract.type.value if contract.type else None,
                    'status': contract.status.value if contract.status else None,
                    'tradable': contract.tradable,
                })

        self.logger.info(
            f"Found {len(contracts)} option contracts for {underlying}"
        )

        return contracts

    def submit_option_market_order(
        self,
        symbol: str,
        qty: int,
        side: str
    ) -> Dict[str, Any]:
        """
        Submit market order for options.

        Args:
            symbol: OCC-format option symbol (e.g., 'SPY241220C00450000')
            qty: Number of contracts (positive integer)
            side: 'buy' or 'sell'

        Returns:
            Dict with order details

        Raises:
            RuntimeError: If not connected
            ValueError: If invalid parameters
        """
        self._ensure_connected()
        self._validate_option_order_params(symbol, qty, side)

        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY  # Options must use DAY
        )

        self.logger.info(
            f"Submitting option market order: {side.upper()} {qty} {symbol}"
        )

        order = self._retry_api_call(
            self.client.submit_order,
            order_request
        )

        order_dict = self._order_to_dict(order)
        order_dict['asset_class'] = 'option'

        self.logger.info(
            f"Option order submitted: id={order.id}, status={order.status}"
        )

        return order_dict

    def submit_option_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float
    ) -> Dict[str, Any]:
        """
        Submit limit order for options.

        Args:
            symbol: OCC-format option symbol (e.g., 'SPY241220C00450000')
            qty: Number of contracts
            side: 'buy' or 'sell'
            limit_price: Limit price per contract ($/share, not total)

        Returns:
            Dict with order details

        Raises:
            RuntimeError: If not connected
            ValueError: If invalid parameters
        """
        self._ensure_connected()
        self._validate_option_order_params(symbol, qty, side)

        if limit_price <= 0:
            raise ValueError(f"Limit price must be positive: {limit_price}")

        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,  # Options must use DAY
            limit_price=limit_price
        )

        self.logger.info(
            f"Submitting option limit order: "
            f"{side.upper()} {qty} {symbol} @ ${limit_price:.2f}"
        )

        order = self._retry_api_call(
            self.client.submit_order,
            order_request
        )

        order_dict = self._order_to_dict(order)
        order_dict['asset_class'] = 'option'

        self.logger.info(f"Option limit order submitted: id={order.id}")

        return order_dict

    def get_option_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for an option contract.

        Args:
            symbol: OCC-format option symbol

        Returns:
            Dict with position details or None if no position
        """
        self._ensure_connected()

        try:
            position = self._retry_api_call(
                self.client.get_open_position,
                symbol
            )

            return {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'side': 'long' if int(position.qty) > 0 else 'short',
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'current_price': float(position.current_price),
                'asset_class': 'option',
            }

        except APIError as e:
            if 'position does not exist' in str(e).lower():
                return None
            raise

    def list_option_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current option positions.

        Returns:
            List of option position dicts
        """
        self._ensure_connected()

        all_positions = self._retry_api_call(self.client.get_all_positions)

        option_positions = []
        for pos in all_positions:
            # Filter for options (OCC symbols are typically longer)
            # OCC format: underlying + YYMMDD + C/P + strike (8 digits)
            if len(pos.symbol) > 10 and ('C' in pos.symbol[-9:] or 'P' in pos.symbol[-9:]):
                option_positions.append({
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'side': 'long' if int(pos.qty) > 0 else 'short',
                    'avg_entry_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'asset_class': 'option',
                })

        return option_positions

    # =========================================================================
    # CLOSED TRADES / REALIZED P&L (Session 83K-81)
    # =========================================================================

    def get_fill_activities(
        self,
        after: Optional[datetime] = None,
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch all FILL activities from Alpaca for realized P&L calculation.

        Uses /v2/account/activities/FILL endpoint with pagination.

        Args:
            after: Only get fills after this datetime (default: 30 days ago)
            page_size: Number of fills per page (max 100)

        Returns:
            List of normalized fill dicts sorted by time ascending
        """
        self._ensure_connected()

        # Default to 30 days ago if not specified
        if after is None:
            from datetime import timedelta
            after = datetime.now() - timedelta(days=30)

        fills = []
        page_token = None

        while True:
            try:
                # Use the trading client's get_activities method
                activities = self._retry_api_call(
                    self.client.get_activities,
                    activity_types='FILL',
                    after=after.isoformat() if after else None,
                    page_size=page_size,
                    page_token=page_token
                )

                if not activities:
                    break

                for activity in activities:
                    # Normalize the activity data
                    symbol = getattr(activity, 'symbol', None)
                    side = getattr(activity, 'side', None)
                    price = getattr(activity, 'price', None)
                    qty = getattr(activity, 'qty', None)
                    transaction_time = getattr(activity, 'transaction_time', None)
                    activity_id = getattr(activity, 'id', None)

                    if not all([symbol, side, price, qty]):
                        continue

                    fills.append({
                        'id': str(activity_id) if activity_id else '',
                        'symbol': symbol,
                        'side': str(side).lower(),
                        'price': float(price),
                        'qty': float(qty),
                        'time': transaction_time.isoformat() if transaction_time else '',
                        'time_dt': transaction_time if transaction_time else None
                    })

                # Check for next page
                # Alpaca uses the last activity ID as page token
                if len(activities) < page_size:
                    break

                page_token = str(activities[-1].id) if activities else None
                if not page_token:
                    break

                time.sleep(0.1)  # Rate limit protection

            except Exception as e:
                self.logger.error(f"Error fetching fill activities: {e}")
                break

        # Sort by time ascending
        fills.sort(key=lambda x: (x['time_dt'] or datetime.min, x['id']))
        return fills

    def get_closed_trades(
        self,
        after: Optional[datetime] = None,
        options_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get realized trades using FIFO matching of fills.

        Matches sell orders to earlier buy orders on a first-in-first-out basis
        to calculate realized P&L for closed positions.

        Args:
            after: Only get trades closed after this datetime
            options_only: If True, only return option trades (default: True)

        Returns:
            List of closed trade dicts with realized P&L
        """
        fills = self.get_fill_activities(after=after)

        if not fills:
            return []

        # Filter for options if requested
        if options_only:
            fills = [
                f for f in fills
                if len(f['symbol']) > 10 and ('C' in f['symbol'][-9:] or 'P' in f['symbol'][-9:])
            ]

        # Group fills by symbol
        by_symbol = {}
        for f in fills:
            by_symbol.setdefault(f['symbol'], []).append(f)

        closed_trades = []

        for symbol, symbol_fills in by_symbol.items():
            buy_queue = []

            for fill in symbol_fills:
                side = fill['side']
                qty = float(fill['qty'])
                price = float(fill['price'])

                if side == 'buy':
                    buy_queue.append({
                        'qty_rem': qty,
                        'price': price,
                        'time': fill['time'],
                        'time_dt': fill['time_dt'],
                        'id': fill['id']
                    })
                elif side == 'sell':
                    qty_to_match = qty

                    while qty_to_match > 0 and buy_queue:
                        lot = buy_queue[0]
                        take = min(lot['qty_rem'], qty_to_match)

                        # For options, multiply by 100 for contract value
                        multiplier = 100 if len(symbol) > 10 else 1
                        cost_basis = take * lot['price'] * multiplier
                        proceeds = take * price * multiplier
                        pnl = proceeds - cost_basis
                        roi = (pnl / cost_basis * 100) if cost_basis != 0 else 0

                        # Calculate duration
                        duration_str = ''
                        if lot['time_dt'] and fill['time_dt']:
                            duration = fill['time_dt'] - lot['time_dt']
                            days = duration.days
                            hours, rem = divmod(duration.seconds, 3600)
                            minutes = rem // 60
                            if days > 0:
                                duration_str = f"{days}d {hours}h {minutes}m"
                            elif hours > 0:
                                duration_str = f"{hours}h {minutes}m"
                            else:
                                duration_str = f"{minutes}m"

                        closed_trades.append({
                            'symbol': symbol,
                            'qty': int(take),
                            'buy_price': round(lot['price'], 2),
                            'sell_price': round(price, 2),
                            'cost_basis': round(cost_basis, 2),
                            'proceeds': round(proceeds, 2),
                            'realized_pnl': round(pnl, 2),
                            'roi_percent': round(roi, 2),
                            'buy_time': lot['time'],
                            'sell_time': fill['time'],
                            'buy_time_dt': lot['time_dt'],
                            'sell_time_dt': fill['time_dt'],
                            'duration': duration_str,
                            'buy_fill_id': lot['id'],
                            'sell_fill_id': fill['id'],
                        })

                        lot['qty_rem'] -= take
                        qty_to_match -= take

                        if lot['qty_rem'] <= 0:
                            buy_queue.pop(0)

        # Sort by sell time descending (newest first)
        closed_trades.sort(key=lambda x: x['sell_time_dt'] or datetime.min, reverse=True)

        return closed_trades

    def close_option_position(
        self,
        symbol: str,
        qty: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Close an option position.

        Args:
            symbol: OCC-format option symbol
            qty: Number of contracts to close (None = close all)

        Returns:
            Dict with close order details

        Raises:
            RuntimeError: If not connected or no position exists
        """
        self._ensure_connected()

        # Get current position first
        position = self.get_option_position(symbol)
        if not position:
            raise RuntimeError(f"No open position for {symbol}")

        close_qty = qty if qty is not None else abs(position['qty'])

        # Determine side (opposite of current position)
        close_side = 'sell' if position['qty'] > 0 else 'buy'

        self.logger.info(
            f"Closing option position: {close_side.upper()} {close_qty} {symbol}"
        )

        return self.submit_option_market_order(symbol, close_qty, close_side)

    def _validate_option_order_params(self, symbol: str, qty: int, side: str):
        """Validate option order parameters."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid option symbol: {symbol}")

        # Basic OCC format validation (minimum length check)
        if len(symbol) < 15:
            raise ValueError(
                f"Invalid OCC option symbol format: {symbol}. "
                f"Expected format like 'SPY241220C00450000'"
            )

        if qty <= 0 or not isinstance(qty, int):
            raise ValueError(f"Quantity must be positive integer: {qty}")

        if side.lower() not in ['buy', 'sell']:
            raise ValueError(f"Side must be 'buy' or 'sell': {side}")

    def _order_to_dict(self, order) -> Dict[str, Any]:
        """Convert Alpaca order object to dict."""
        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': int(order.qty),
            'side': order.side.value,
            'type': order.type.value,
            'status': order.status.value,
            'time_in_force': order.time_in_force.value,
            'limit_price': float(order.limit_price) if order.limit_price else None,
            'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
            'filled_at': order.filled_at.isoformat() if order.filled_at else None,
        }

    # =========================================================================
    # MARKET DATA METHODS (Session 83K-50)
    # =========================================================================

    def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for a stock symbol.

        Uses Alpaca's market data API to get real-time bid/ask quotes.
        This works even if you don't hold a position in the stock.

        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')

        Returns:
            Dict with quote data (bid, ask, mid, timestamp) or None if unavailable
        """
        self._ensure_connected()

        if self.data_client is None:
            self.logger.warning("Market data client not initialized")
            return None

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol.upper()])
            quotes = self.data_client.get_stock_latest_quote(request)

            quote = quotes.get(symbol.upper())
            if quote:
                return {
                    'symbol': symbol.upper(),
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'mid': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                    'bid_size': int(quote.bid_size),
                    'ask_size': int(quote.ask_size),
                    'timestamp': quote.timestamp.isoformat() if quote.timestamp else None,
                }

        except Exception as e:
            self.logger.warning(f"Error fetching quote for {symbol}: {e}")

        return None

    def get_stock_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get latest quotes for multiple stock symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to quote data
        """
        self._ensure_connected()

        if self.data_client is None:
            self.logger.warning("Market data client not initialized")
            return {}

        result = {}
        try:
            upper_symbols = [s.upper() for s in symbols]
            request = StockLatestQuoteRequest(symbol_or_symbols=upper_symbols)
            quotes = self.data_client.get_stock_latest_quote(request)

            for symbol, quote in quotes.items():
                result[symbol] = {
                    'symbol': symbol,
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'mid': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                    'bid_size': int(quote.bid_size),
                    'ask_size': int(quote.ask_size),
                    'timestamp': quote.timestamp.isoformat() if quote.timestamp else None,
                }

        except Exception as e:
            self.logger.warning(f"Error fetching quotes: {e}")

        return result

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price (mid-point) for a stock symbol.

        Convenience method that returns just the mid price.

        Args:
            symbol: Stock symbol

        Returns:
            Mid price (average of bid/ask) or None if unavailable
        """
        quote = self.get_stock_quote(symbol)
        if quote:
            return quote.get('mid')
        return None
