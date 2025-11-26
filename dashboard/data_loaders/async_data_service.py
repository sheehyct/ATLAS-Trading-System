"""
Async Data Service for ATLAS Dashboard

Provides high-performance asynchronous data fetching using httpx for:
- Alpaca API (positions, account, orders)
- Market data (Yahoo Finance, Tiingo)
- ThetaData options data (when integrated)

Benefits over synchronous approach:
- Concurrent API calls (fetch multiple endpoints simultaneously)
- Non-blocking I/O (dashboard stays responsive during data fetches)
- Connection pooling with HTTP/2 support
- Automatic retries with exponential backoff
- Request timeout management

Architecture:
- Uses httpx.AsyncClient for connection pooling
- Quart/FastAPI compatible for async Dash patterns
- Falls back gracefully if async not available

Usage with Dash:
    # In callbacks, use run_async() wrapper
    from dashboard.data_loaders.async_data_service import AsyncDataService, run_async

    service = AsyncDataService()

    @app.callback(Output('data', 'children'), Input('refresh', 'n_clicks'))
    def update_data(n):
        # Run async code in sync callback
        data = run_async(service.fetch_all_portfolio_data())
        return format_data(data)

Session 76: Initial async implementation for dashboard performance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class AsyncConfig:
    """Configuration for async HTTP client."""
    timeout: float = 30.0  # Request timeout in seconds
    max_connections: int = 100  # Connection pool size
    max_keepalive_connections: int = 20  # Keep-alive connections
    retries: int = 3  # Max retry attempts
    retry_delay: float = 1.0  # Initial retry delay (exponential backoff)
    http2: bool = True  # Enable HTTP/2 for multiplexing


DEFAULT_CONFIG = AsyncConfig()


# ============================================
# ASYNC HTTP CLIENT WRAPPER
# ============================================

class AsyncHTTPClient:
    """
    Unified async HTTP client supporting both httpx and aiohttp.

    Prefers httpx for HTTP/2 support and better async patterns.
    Falls back to aiohttp if httpx unavailable.
    """

    def __init__(self, config: AsyncConfig = DEFAULT_CONFIG):
        self.config = config
        self._client: Optional[Any] = None
        self._use_httpx = HTTPX_AVAILABLE

        if not HTTPX_AVAILABLE and not AIOHTTP_AVAILABLE:
            logger.warning(
                "Neither httpx nor aiohttp available. "
                "Install with: pip install httpx or pip install aiohttp"
            )

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Initialize connection pool."""
        if self._use_httpx and HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_keepalive_connections
                ),
                http2=self.config.http2
            )
            logger.info("AsyncHTTPClient initialized with httpx (HTTP/2 enabled)")
        elif AIOHTTP_AVAILABLE:
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                keepalive_timeout=30
            )
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._client = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            self._use_httpx = False
            logger.info("AsyncHTTPClient initialized with aiohttp")
        else:
            raise RuntimeError("No async HTTP library available")

    async def close(self):
        """Close connection pool."""
        if self._client:
            await self._client.aclose() if self._use_httpx else await self._client.close()
            self._client = None

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make async GET request with retry logic.

        Args:
            url: Request URL
            headers: Optional headers
            params: Optional query parameters

        Returns:
            JSON response as dict
        """
        return await self._request("GET", url, headers=headers, params=params)

    async def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make async POST request."""
        return await self._request("POST", url, headers=headers, json_data=json_data)

    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute request with retry logic."""
        if not self._client:
            await self.connect()

        last_exception = None

        for attempt in range(self.config.retries):
            try:
                if self._use_httpx:
                    response = await self._client.request(
                        method,
                        url,
                        headers=headers,
                        params=params,
                        json=json_data
                    )
                    response.raise_for_status()
                    return response.json()
                else:
                    async with self._client.request(
                        method,
                        url,
                        headers=headers,
                        params=params,
                        json=json_data
                    ) as response:
                        response.raise_for_status()
                        return await response.json()

            except Exception as e:
                last_exception = e
                if attempt < self.config.retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.config.retries}), "
                        f"retrying in {delay}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"Request failed after {self.config.retries} attempts: {last_exception}")
        raise last_exception


# ============================================
# ALPACA ASYNC CLIENT
# ============================================

class AlpacaAsyncClient:
    """
    Async client for Alpaca Trading API.

    Provides concurrent fetching of:
    - Account status
    - Positions
    - Orders
    - Market data
    """

    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL = "https://api.alpaca.markets"
    DATA_BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        config: AsyncConfig = DEFAULT_CONFIG
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = self.PAPER_BASE_URL if paper else self.LIVE_BASE_URL
        self.http_client = AsyncHTTPClient(config)

        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json"
        }

    async def __aenter__(self):
        await self.http_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.close()

    async def get_account(self) -> Dict[str, Any]:
        """Fetch account information."""
        url = f"{self.base_url}/v2/account"
        data = await self.http_client.get(url, headers=self._headers)

        return {
            'account_id': data.get('id'),
            'equity': float(data.get('equity', 0)),
            'cash': float(data.get('cash', 0)),
            'buying_power': float(data.get('buying_power', 0)),
            'portfolio_value': float(data.get('portfolio_value', 0)),
            'last_equity': float(data.get('last_equity', 0)),
            'pattern_day_trader': data.get('pattern_day_trader', False),
            'trading_blocked': data.get('trading_blocked', False),
            'timestamp': datetime.now().isoformat()
        }

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all open positions."""
        url = f"{self.base_url}/v2/positions"
        positions = await self.http_client.get(url, headers=self._headers)

        return [
            {
                'symbol': pos.get('symbol'),
                'qty': int(pos.get('qty', 0)),
                'side': 'long' if int(pos.get('qty', 0)) > 0 else 'short',
                'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                'market_value': float(pos.get('market_value', 0)),
                'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                'unrealized_plpc': float(pos.get('unrealized_plpc', 0)),
                'current_price': float(pos.get('current_price', 0))
            }
            for pos in positions
        ]

    async def get_orders(
        self,
        status: str = 'all',
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Fetch orders with optional status filter."""
        url = f"{self.base_url}/v2/orders"
        params = {'status': status, 'limit': limit}
        orders = await self.http_client.get(url, headers=self._headers, params=params)

        return [
            {
                'id': order.get('id'),
                'symbol': order.get('symbol'),
                'qty': int(order.get('qty', 0)),
                'side': order.get('side'),
                'type': order.get('type'),
                'status': order.get('status'),
                'filled_qty': int(order.get('filled_qty', 0)) if order.get('filled_qty') else 0,
                'filled_avg_price': float(order.get('filled_avg_price', 0)) if order.get('filled_avg_price') else None,
                'submitted_at': order.get('submitted_at'),
                'filled_at': order.get('filled_at')
            }
            for order in orders
        ]

    async def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch latest quotes for multiple symbols concurrently."""
        url = f"{self.DATA_BASE_URL}/v2/stocks/quotes/latest"
        params = {'symbols': ','.join(symbols)}

        data = await self.http_client.get(url, headers=self._headers, params=params)

        quotes = {}
        for symbol, quote in data.get('quotes', {}).items():
            quotes[symbol] = {
                'bid': float(quote.get('bp', 0)),
                'ask': float(quote.get('ap', 0)),
                'bid_size': int(quote.get('bs', 0)),
                'ask_size': int(quote.get('as', 0)),
                'timestamp': quote.get('t')
            }

        return quotes

    async def fetch_all_portfolio_data(self) -> Dict[str, Any]:
        """
        Fetch all portfolio data concurrently.

        This is the KEY PERFORMANCE IMPROVEMENT over synchronous fetching.
        Instead of 3 sequential API calls (~300-500ms each), we make
        all calls concurrently (~300-500ms total).

        Returns:
            Dict with account, positions, and orders
        """
        # Execute all API calls concurrently
        account_task = asyncio.create_task(self.get_account())
        positions_task = asyncio.create_task(self.get_positions())
        orders_task = asyncio.create_task(self.get_orders(status='open'))

        # Wait for all tasks to complete
        account, positions, orders = await asyncio.gather(
            account_task,
            positions_task,
            orders_task,
            return_exceptions=True
        )

        # Handle any exceptions
        result = {}

        if isinstance(account, Exception):
            logger.error(f"Failed to fetch account: {account}")
            result['account'] = {}
        else:
            result['account'] = account

        if isinstance(positions, Exception):
            logger.error(f"Failed to fetch positions: {positions}")
            result['positions'] = []
        else:
            result['positions'] = positions

        if isinstance(orders, Exception):
            logger.error(f"Failed to fetch orders: {orders}")
            result['orders'] = []
        else:
            result['orders'] = orders

        result['fetch_timestamp'] = datetime.now().isoformat()

        return result


# ============================================
# THETADATA ASYNC CLIENT (FUTURE INTEGRATION)
# ============================================

class ThetaDataAsyncClient:
    """
    Async client for ThetaData options API.

    ThetaData provides:
    - Historical options data
    - Real-time options quotes
    - Greeks calculations
    - Unusual options activity

    NOTE: This is a placeholder for future integration.
    Requires ThetaData subscription and API credentials.
    """

    BASE_URL = "http://127.0.0.1:25510"  # Local ThetaData Terminal

    def __init__(self, config: AsyncConfig = DEFAULT_CONFIG):
        self.http_client = AsyncHTTPClient(config)
        self._connected = False

    async def __aenter__(self):
        await self.http_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.close()

    async def get_option_chain(
        self,
        symbol: str,
        expiration: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch option chain for symbol and expiration.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            expiration: Expiration date (YYYYMMDD)

        Returns:
            List of option contracts with quotes
        """
        url = f"{self.BASE_URL}/v2/list/contracts"
        params = {
            'root': symbol,
            'exp': expiration
        }

        try:
            data = await self.http_client.get(url, params=params)
            return data.get('response', [])
        except Exception as e:
            logger.error(f"ThetaData option chain fetch failed: {e}")
            return []

    async def get_option_quote(
        self,
        contract: str
    ) -> Dict[str, Any]:
        """
        Fetch real-time quote for option contract.

        Args:
            contract: OSI contract symbol

        Returns:
            Quote with bid/ask and Greeks
        """
        url = f"{self.BASE_URL}/v2/snapshot/option/quote"
        params = {'contract': contract}

        try:
            data = await self.http_client.get(url, params=params)
            return data.get('response', {})
        except Exception as e:
            logger.error(f"ThetaData quote fetch failed: {e}")
            return {}

    async def get_historical_options(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        right: str = 'C'  # C=Call, P=Put
    ) -> pd.DataFrame:
        """
        Fetch historical options data.

        Args:
            symbol: Underlying symbol
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            right: Option type (C=Call, P=Put)

        Returns:
            DataFrame with historical options data
        """
        url = f"{self.BASE_URL}/v2/hist/option/eod"
        params = {
            'root': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'right': right
        }

        try:
            data = await self.http_client.get(url, params=params)
            if data.get('response'):
                return pd.DataFrame(data['response'])
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"ThetaData historical fetch failed: {e}")
            return pd.DataFrame()


# ============================================
# ASYNC DATA SERVICE (UNIFIED INTERFACE)
# ============================================

class AsyncDataService:
    """
    Unified async data service for ATLAS Dashboard.

    Manages:
    - Connection lifecycle
    - Concurrent data fetching
    - Caching with TTL
    - Error handling and fallbacks

    Usage:
        async with AsyncDataService() as service:
            data = await service.fetch_dashboard_data()
    """

    def __init__(
        self,
        alpaca_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
        paper: bool = True,
        config: AsyncConfig = DEFAULT_CONFIG
    ):
        self.config = config

        # Initialize clients (credentials loaded lazily)
        self._alpaca_client: Optional[AlpacaAsyncClient] = None
        self._theta_client: Optional[ThetaDataAsyncClient] = None

        # Credential storage
        self._alpaca_key = alpaca_key
        self._alpaca_secret = alpaca_secret
        self._paper = paper

        # Cache with TTL
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(seconds=30)

    async def __aenter__(self):
        await self._init_clients()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_clients()

    async def _init_clients(self):
        """Initialize API clients with lazy credential loading."""
        # Try to load credentials from config if not provided
        if not self._alpaca_key or not self._alpaca_secret:
            try:
                from config.settings import get_alpaca_credentials
                creds = get_alpaca_credentials('LARGE')
                self._alpaca_key = creds['api_key']
                self._alpaca_secret = creds['secret_key']
            except Exception as e:
                logger.warning(f"Could not load Alpaca credentials: {e}")

        # Initialize Alpaca client if credentials available
        if self._alpaca_key and self._alpaca_secret:
            self._alpaca_client = AlpacaAsyncClient(
                self._alpaca_key,
                self._alpaca_secret,
                paper=self._paper,
                config=self.config
            )
            await self._alpaca_client.http_client.connect()

        # ThetaData client (no auth required for local terminal)
        self._theta_client = ThetaDataAsyncClient(config=self.config)

    async def _close_clients(self):
        """Close all API client connections."""
        if self._alpaca_client:
            await self._alpaca_client.http_client.close()
        if self._theta_client:
            await self._theta_client.http_client.close()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                return value
        return None

    def _set_cached(self, key: str, value: Any):
        """Set value in cache with current timestamp."""
        self._cache[key] = (value, datetime.now())

    async def fetch_portfolio_data(self) -> Dict[str, Any]:
        """
        Fetch complete portfolio data with caching.

        Returns:
            Dict with account, positions, orders
        """
        cached = self._get_cached('portfolio')
        if cached:
            logger.debug("Returning cached portfolio data")
            return cached

        if not self._alpaca_client:
            logger.warning("Alpaca client not initialized")
            return {'account': {}, 'positions': [], 'orders': []}

        data = await self._alpaca_client.fetch_all_portfolio_data()
        self._set_cached('portfolio', data)

        return data

    async def fetch_dashboard_data(
        self,
        include_options: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch all data needed for dashboard refresh.

        This is the main entry point for dashboard callbacks.
        Fetches all data concurrently for maximum performance.

        Args:
            include_options: Whether to include ThetaData options data

        Returns:
            Complete dashboard data package
        """
        tasks = []
        task_names = []

        # Always fetch portfolio data
        tasks.append(self.fetch_portfolio_data())
        task_names.append('portfolio')

        # Optionally fetch options data
        if include_options and self._theta_client:
            # Example: fetch SPY option chain for nearest expiration
            tasks.append(self._theta_client.get_option_chain('SPY', '20241220'))
            task_names.append('options')

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Package results
        dashboard_data = {}
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {name}: {result}")
                dashboard_data[name] = {} if name == 'portfolio' else []
            else:
                dashboard_data[name] = result

        dashboard_data['timestamp'] = datetime.now().isoformat()

        return dashboard_data


# ============================================
# SYNC WRAPPER FOR DASH CALLBACKS
# ============================================

def run_async(coro):
    """
    Run async coroutine in synchronous context.

    Use this in Dash callbacks to execute async code:

        @app.callback(...)
        def update_data(n):
            data = run_async(service.fetch_portfolio_data())
            return format_data(data)

    Args:
        coro: Async coroutine to execute

    Returns:
        Result of coroutine
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create new one
        return asyncio.run(coro)


# ============================================
# EXAMPLE USAGE
# ============================================

async def example_usage():
    """Example demonstrating async data fetching."""

    print("Initializing AsyncDataService...")

    async with AsyncDataService(paper=True) as service:
        # Fetch all portfolio data concurrently
        print("\nFetching portfolio data...")
        start = datetime.now()

        data = await service.fetch_portfolio_data()

        elapsed = (datetime.now() - start).total_seconds()
        print(f"Fetched in {elapsed:.3f}s")

        # Display results
        account = data.get('account', {})
        positions = data.get('positions', [])

        print(f"\nAccount Equity: ${account.get('equity', 0):,.2f}")
        print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"Open Positions: {len(positions)}")

        for pos in positions:
            print(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
