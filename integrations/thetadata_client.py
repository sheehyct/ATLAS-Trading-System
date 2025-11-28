"""
ThetaData REST API Client for ATLAS Trading System.

Session 79: Provides historical options data for backtesting, replacing
synthetic Black-Scholes pricing with real market data.

Session 81: Updated for ThetaData API v3 (November 2025 upgrade).
    - Port changed from 25510 to 25503
    - Endpoint paths changed to /v3/option/* format
    - Strike format: dollars (not * 1000)
    - Right format: 'call'/'put' (not 'C'/'P')
    - Response format: nested JSON with 'contract' and 'data' arrays

Architecture:
    ThetaData uses a LOCAL terminal architecture:
    [Python Code] <--REST--> [Theta Terminal localhost:25503] <--Compressed--> [ThetaData Servers]

    The terminal handles authentication via its creds.txt file.
    No API key is needed in REST calls.

Key Endpoints (v3 Standard Tier):
    /v3/option/list/symbols - Available option roots
    /v3/option/list/expirations - Available expirations
    /v3/option/list/strikes - Available strikes
    /v3/option/history/quote - Historical NBBO quotes
    /v3/option/snapshot/greeks/all - Real-time Greeks snapshot

Data Formats (v3):
    - Strike: dollars (e.g., 450.00)
    - Expiration: YYYYMMDD or YYYY-MM-DD
    - Option Type: 'call' or 'put'

Usage:
    from integrations.thetadata_client import ThetaDataRESTClient

    client = ThetaDataRESTClient()
    if client.connect():
        quote = client.get_quote('SPY', datetime(2024,12,20), 450.0, 'C', as_of=datetime(2024,11,15))
        print(f"Bid: {quote.bid}, Ask: {quote.ask}, IV: {quote.iv}")
"""

import time
import logging
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import pandas as pd

from config.settings import get_thetadata_config


@dataclass
class OptionsQuote:
    """
    Standardized options quote data structure.

    Provides a consistent interface regardless of data source (REST or WebSocket).
    All fields use standard Python types for easy integration with options_module.

    Attributes:
        symbol: OSI symbol (e.g., 'SPY241220C00450000')
        underlying: Underlying symbol (e.g., 'SPY')
        strike: Strike price in dollars (e.g., 450.0)
        expiration: Expiration date
        option_type: 'C' for call, 'P' for put
        timestamp: Quote timestamp
        bid: Best bid price
        ask: Best ask price
        mid: Midpoint price ((bid + ask) / 2)
        bid_size: Bid size in contracts
        ask_size: Ask size in contracts
        volume: Trading volume (if available)
        open_interest: Open interest (if available)
        iv: Implied volatility (if available)
        delta: Delta (if available)
        gamma: Gamma (if available)
        theta: Theta per day (if available)
        vega: Vega per 1% IV change (if available)
        underlying_price: Underlying price at quote time (if available)
    """
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # 'C' or 'P'
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    open_interest: int = 0
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    underlying_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'strike': self.strike,
            'expiration': self.expiration,
            'option_type': self.option_type,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'mid': self.mid,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'iv': self.iv,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'underlying_price': self.underlying_price,
        }


class ThetaDataProviderBase(ABC):
    """
    Abstract base class for ThetaData providers.

    Enables REST (historical) and WebSocket (streaming) implementations
    to share a common interface for options_module consumption.

    This abstraction allows:
    1. Easy mocking for unit tests
    2. Future WebSocket streaming as a drop-in replacement
    3. Clean separation between API specifics and business logic
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to ThetaData terminal.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection gracefully."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check connection status.

        Returns:
            True if connected and ready for requests
        """
        pass

    @abstractmethod
    def get_quote(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionsQuote]:
        """
        Get options quote (current or historical).

        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' for call, 'P' for put
            as_of: Historical date for quote (None = latest)

        Returns:
            OptionsQuote object or None if not found
        """
        pass

    @abstractmethod
    def get_greeks(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: Optional[datetime] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get Greeks for an option from ThetaData.

        Args:
            underlying: Underlying symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' or 'P'
            as_of: Historical date (None = latest)

        Returns:
            Dict with 'delta', 'gamma', 'theta', 'vega', 'iv' or None
        """
        pass

    @abstractmethod
    def get_expirations(
        self,
        underlying: str,
        min_dte: int = 0,
        max_dte: int = 365
    ) -> List[datetime]:
        """
        Get available expiration dates for underlying.

        Args:
            underlying: Underlying symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration

        Returns:
            List of available expiration dates
        """
        pass

    @abstractmethod
    def get_strikes(
        self,
        underlying: str,
        expiration: datetime
    ) -> List[float]:
        """
        Get available strikes for a specific expiration.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date

        Returns:
            List of available strike prices
        """
        pass


class ThetaDataRESTClient(ThetaDataProviderBase):
    """
    REST API v3 implementation for historical options data.

    Connects to local ThetaData terminal at localhost:25503.
    Implements retry logic with exponential backoff for reliability.

    Session 81: Updated for v3 API format.

    Attributes:
        host: Terminal host (default: 127.0.0.1)
        port: Terminal port (default: 25503)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 25503

    def __init__(
        self,
        host: str = None,
        port: int = None,
        timeout: int = 30,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ThetaData REST client.

        Args:
            host: Terminal host (default from config or 127.0.0.1)
            port: Terminal port (default from config or 25503)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            logger: Optional logger instance
        """
        # Get config (may override defaults)
        config = get_thetadata_config()

        self.host = host or config.get('host', self.DEFAULT_HOST)
        self.port = port or config.get('port', self.DEFAULT_PORT)
        self.timeout = timeout or config.get('timeout', 30)
        self.max_retries = max_retries

        # Session 81: Updated to v3 API
        self.base_url = f"http://{self.host}:{self.port}/v3"
        self.logger = logger or self._create_default_logger()

        self._session = requests.Session()
        self._connected = False

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger if none provided."""
        logger = logging.getLogger('thetadata_client')
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
        Test connection to ThetaData terminal.

        Validates terminal is running by fetching available option symbols.
        Session 81: Updated for v3 API endpoint.

        Returns:
            True if terminal responds, False otherwise
        """
        try:
            # v3 API uses /v3/option/list/symbols instead of /v2/list/roots
            response = self._session.get(
                f"{self.base_url}/option/list/symbols",
                params={'format': 'json'},
                timeout=self.timeout
            )
            response.raise_for_status()

            self._connected = True
            self.logger.info(
                f"Connected to ThetaData terminal v3 at {self.host}:{self.port}"
            )
            return True

        except requests.exceptions.ConnectionError:
            self._connected = False
            self.logger.error(
                f"ThetaData terminal not running at {self.host}:{self.port}. "
                "Start Theta Terminal before making requests."
            )
            return False

        except Exception as e:
            self._connected = False
            self.logger.error(f"Failed to connect to ThetaData: {str(e)}")
            return False

    def disconnect(self) -> None:
        """Close session gracefully."""
        self._session.close()
        self._connected = False
        self.logger.info("Disconnected from ThetaData terminal")

    def is_connected(self) -> bool:
        """Check if connected to terminal."""
        return self._connected

    # =========================================================================
    # Data Format Helpers
    # =========================================================================

    def _format_strike(self, strike: float) -> float:
        """
        Format strike for ThetaData v3 API.

        Session 81: v3 API uses dollars directly (not * 1000 like v2).

        Args:
            strike: Strike price in dollars (e.g., 450.50)

        Returns:
            Strike in dollars (same value for v3)
        """
        return strike

    def _format_right(self, option_type: str) -> str:
        """
        Convert option type to v3 format.

        Session 81: v3 API uses 'call'/'put' instead of 'C'/'P'.

        Args:
            option_type: 'C' or 'P' (or 'call'/'put')

        Returns:
            'call' or 'put' for v3 API
        """
        opt_upper = option_type.upper()
        if opt_upper in ('C', 'CALL'):
            return 'call'
        elif opt_upper in ('P', 'PUT'):
            return 'put'
        else:
            raise ValueError(f"Invalid option type: {option_type}. Use 'C' or 'P'.")

    def _format_expiration(self, expiration: datetime) -> str:
        """
        Convert expiration to ThetaData format (YYYYMMDD).

        Args:
            expiration: Expiration date

        Returns:
            Date string in YYYYMMDD format
        """
        return expiration.strftime('%Y%m%d')

    def _format_date(self, date: datetime) -> str:
        """
        Convert date to ThetaData format (YYYYMMDD).

        Args:
            date: Date to format

        Returns:
            Date string in YYYYMMDD format
        """
        return date.strftime('%Y%m%d')

    def _parse_date(self, date_int: int) -> datetime:
        """
        Parse ThetaData date format (YYYYMMDD integer) to datetime.

        Args:
            date_int: Date as integer (e.g., 20241220)

        Returns:
            datetime object
        """
        date_str = str(date_int)
        return datetime.strptime(date_str, '%Y%m%d')

    def _parse_strike(self, strike_value) -> float:
        """
        Parse ThetaData strike format to dollars.

        Session 81: v3 API returns dollars directly, no conversion needed.

        Args:
            strike_value: Strike from API (v3: dollars, v2: * 1000)

        Returns:
            Strike in dollars
        """
        # v3 returns float dollars directly
        return float(strike_value)

    def _safe_get_series(self, row: pd.Series, key: str, default: float = 0.0) -> float:
        """
        Safely extract value from Pandas Series with fallback.

        Session 80 BUG FIX: Series.get() behavior varies by Pandas version.
        This method provides consistent cross-version behavior.

        Args:
            row: Pandas Series (typically a DataFrame row)
            key: Column name to extract
            default: Default value if key not found

        Returns:
            Float value from Series or default
        """
        try:
            if key in row.index:
                return float(row[key])
            return default
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value, default: float = 0.0) -> float:
        """
        Safely convert value to float, handling N/A, null, and empty strings.

        Session 83 BUG FIX: ThetaData API can return "N/A", null, or empty
        strings instead of numeric values. float() crashes on these.

        Args:
            value: Value to convert (int, float, str, or None)
            default: Default value if conversion fails

        Returns:
            Float value or default
        """
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if not value or value.upper() in ('N/A', 'NA', 'NULL', 'NONE', ''):
                return default
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _generate_osi_symbol(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str
    ) -> str:
        """
        Generate OSI (Options Symbology Initiative) format symbol.

        Format: SYMBOL + YYMMDD + C/P + STRIKE*1000 (8 digits)
        Example: SPY241220C00450000 = SPY Dec 20 2024 $450 Call

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price in dollars
            option_type: 'C' or 'P'

        Returns:
            OSI format symbol
        """
        symbol_part = underlying.upper()
        date_part = expiration.strftime('%y%m%d')
        type_part = option_type.upper()
        strike_part = str(int(strike * 1000)).zfill(8)
        return f"{symbol_part}{date_part}{type_part}{strike_part}"

    # =========================================================================
    # API Request Methods
    # =========================================================================

    def _make_request_v3(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make v3 REST API request with retry logic.

        Session 81: New method for v3 JSON responses.

        Args:
            endpoint: API endpoint (e.g., 'option/history/quote')
            params: Request parameters

        Returns:
            JSON response dictionary

        Raises:
            requests.exceptions.RequestException: If all retries fail
        """
        # Always use JSON format for v3
        params['format'] = 'json'
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Session 83 BUG FIX: Check for text error responses more carefully
                # Only match if response STARTS with error indicators to avoid
                # false positives on valid JSON containing "error" field names
                response_lower = response.text.strip().lower()

                if response_lower.startswith('no data'):
                    self.logger.debug(f"No data found from {endpoint}")
                    return {'response': []}

                if response_lower.startswith('error') or response_lower.startswith('invalid'):
                    self.logger.error(f"API error from {endpoint}: {response.text[:200]}")
                    return {'response': []}

                return response.json()

            except requests.exceptions.RequestException as e:
                is_retryable = any(
                    keyword in str(e).lower()
                    for keyword in ['timeout', 'connection', '429', '503']
                )

                if not is_retryable or attempt == self.max_retries - 1:
                    self.logger.error(
                        f"API request failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{endpoint} - {str(e)}"
                    )
                    raise

                wait_time = 2 ** attempt
                self.logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)

        raise RuntimeError(f"API request failed after {self.max_retries} attempts")

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        use_csv: bool = True
    ) -> pd.DataFrame:
        """
        Make REST API request with retry logic (legacy v2 compatibility).

        Note: For v3 API, prefer _make_request_v3() which returns JSON directly.

        Args:
            endpoint: API endpoint (e.g., 'hist/option/quote')
            params: Request parameters
            use_csv: If True, request CSV format for DataFrame conversion

        Returns:
            DataFrame with response data

        Raises:
            requests.exceptions.RequestException: If all retries fail
        """
        if use_csv:
            params['use_csv'] = 'true'

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Parse CSV response to DataFrame
                # Session 80 BUG FIX: Validate response before parsing
                if use_csv:
                    # Check for empty or error responses
                    if not response.text or response.text.strip() == '':
                        self.logger.warning(f"Empty response from {endpoint}")
                        return pd.DataFrame()

                    # Check for error message responses (ThetaData returns text errors)
                    response_start = response.text.lower()[:100] if len(response.text) >= 100 else response.text.lower()
                    if response.text.startswith('Error') or 'error' in response_start:
                        self.logger.error(f"API error from {endpoint}: {response.text[:200]}")
                        return pd.DataFrame()

                    try:
                        df = pd.read_csv(io.StringIO(response.text))
                        return df
                    except pd.errors.EmptyDataError:
                        self.logger.warning(f"Empty CSV data from {endpoint}")
                        return pd.DataFrame()
                    except pd.errors.ParserError as e:
                        self.logger.error(f"CSV parse error from {endpoint}: {e}")
                        return pd.DataFrame()
                else:
                    return pd.DataFrame(response.json().get('response', []))

            except requests.exceptions.RequestException as e:
                # Check if retryable
                is_retryable = any(
                    keyword in str(e).lower()
                    for keyword in ['timeout', 'connection', '429', '503']
                )

                if not is_retryable or attempt == self.max_retries - 1:
                    self.logger.error(
                        f"API request failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{endpoint} - {str(e)}"
                    )
                    raise

                # Exponential backoff
                wait_time = 2 ** attempt
                self.logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)

        raise RuntimeError(f"API request failed after {self.max_retries} attempts")

    # =========================================================================
    # Public Interface Methods
    # =========================================================================

    def get_quote(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionsQuote]:
        """
        Get historical options quote.

        Session 81: Updated for v3 API format.

        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' for call, 'P' for put
            as_of: Date for historical quote

        Returns:
            OptionsQuote object or None if not found
        """
        if as_of is None:
            as_of = datetime.now() - timedelta(days=1)

        # v3 API parameters
        # Session 83K-7 FIX: Removed interval=1h which caused 472 errors
        # Use tick-level data instead for maximum coverage
        params = {
            'symbol': underlying.upper(),
            'expiration': self._format_expiration(expiration),
            'strike': self._format_strike(strike),
            'right': self._format_right(option_type),
            'date': self._format_date(as_of),
            # No interval parameter = tick-level data (maximum coverage)
        }

        try:
            result = self._make_request_v3('option/history/quote', params)

            response_list = result.get('response', [])
            if not response_list:
                return None

            # v3 returns nested structure: response[0].contract, response[0].data
            contract_data = response_list[0]
            data_list = contract_data.get('data', [])

            if not data_list:
                return None

            # Session 83K-7 FIX: Find last non-zero quote instead of just last entry
            # Many options have bid=0, ask=0 at market close for illiquid strikes
            bid = 0.0
            ask = 0.0
            last_quote = data_list[-1]  # Default to last entry for timestamps

            # Search backwards for valid quote (most recent with non-zero bid or ask)
            for quote in reversed(data_list):
                q_bid = self._safe_float(quote.get('bid'), 0.0)
                q_ask = self._safe_float(quote.get('ask'), 0.0)
                if q_bid > 0 or q_ask > 0:
                    bid = q_bid
                    ask = q_ask
                    last_quote = quote
                    break

            # Normalize option_type to single char for internal consistency
            opt_type_normalized = 'C' if option_type.upper() in ('C', 'CALL') else 'P'

            return OptionsQuote(
                symbol=self._generate_osi_symbol(underlying, expiration, strike, opt_type_normalized),
                underlying=underlying.upper(),
                strike=strike,
                expiration=expiration,
                option_type=opt_type_normalized,
                timestamp=as_of,
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2 if (bid > 0 or ask > 0) else 0,
                bid_size=int(last_quote.get('bid_size', 0)),
                ask_size=int(last_quote.get('ask_size', 0)),
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to get quote for {underlying} {strike} {option_type} "
                f"exp {expiration.date()}: {str(e)}"
            )
            return None

    def get_greeks(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: Optional[datetime] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get Greeks for an option.

        Session 81: Updated for v3 API. Uses snapshot endpoint for real-time
        Greeks. For historical Greeks, falls back to Black-Scholes calculations.

        Note: v3 API primarily supports real-time Greeks snapshots.
        For historical backtesting, the options_module uses Black-Scholes.

        Args:
            underlying: Underlying symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' or 'P'
            as_of: Date for historical Greeks (None = snapshot)

        Returns:
            Dict with delta, gamma, theta, vega, iv or None if not found
        """
        # Try v3 history/greeks endpoint first
        if as_of is None:
            as_of = datetime.now() - timedelta(days=1)

        # v3 parameters for history/greeks
        # Session 83K-7 FIX: Removed interval=1h which caused 472 errors
        params = {
            'symbol': underlying.upper(),
            'expiration': self._format_expiration(expiration),
            'strike': self._format_strike(strike),
            'right': self._format_right(option_type),
            'date': self._format_date(as_of),
        }

        try:
            # Session 83K-3 FIX: Use correct endpoint for first-order Greeks
            # Wrong: /v3/option/history/greeks
            # Correct: /v3/option/history/greeks/first_order
            result = self._make_request_v3('option/history/greeks/first_order', params)

            response_list = result.get('response', [])
            if not response_list:
                self.logger.debug(
                    f"No historical Greeks for {underlying} {strike} {option_type}, "
                    "fallback to Black-Scholes recommended"
                )
                return None

            # v3 returns nested structure
            contract_data = response_list[0]
            data_list = contract_data.get('data', [])

            if not data_list:
                return None

            # Get the last row
            last_data = data_list[-1]

            # Session 83 BUG FIX: Use _safe_float for all numeric conversions
            # v3 field names - handle N/A, null, empty strings gracefully
            iv_implied = self._safe_float(last_data.get('implied_volatility'), 0.0)
            iv_fallback = self._safe_float(last_data.get('iv'), 0.0)
            iv_value = iv_implied if iv_implied > 0 else iv_fallback

            return {
                'delta': self._safe_float(last_data.get('delta'), 0.0),
                'gamma': self._safe_float(last_data.get('gamma'), 0.0),
                'theta': self._safe_float(last_data.get('theta'), 0.0),
                'vega': self._safe_float(last_data.get('vega'), 0.0),
                'iv': iv_value,
                'underlying_price': self._safe_float(last_data.get('underlying_price'), 0.0),
            }

        except Exception as e:
            self.logger.warning(
                f"Failed to get Greeks for {underlying} {strike} {option_type} "
                f"exp {expiration.date()}: {str(e)}"
            )
            return None

    def get_quote_with_greeks(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionsQuote]:
        """
        Get quote with Greeks data combined.

        Fetches both quote and Greeks data and combines into single OptionsQuote.

        Args:
            underlying: Underlying symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' or 'P'
            as_of: Date for historical data

        Returns:
            OptionsQuote with Greeks populated, or None if not found
        """
        quote = self.get_quote(underlying, expiration, strike, option_type, as_of)
        if quote is None:
            return None

        greeks = self.get_greeks(underlying, expiration, strike, option_type, as_of)
        if greeks:
            quote.delta = greeks.get('delta')
            quote.gamma = greeks.get('gamma')
            quote.theta = greeks.get('theta')
            quote.vega = greeks.get('vega')
            quote.iv = greeks.get('iv')
            quote.underlying_price = greeks.get('underlying_price')

        return quote

    def get_expirations(
        self,
        underlying: str,
        min_dte: int = 0,
        max_dte: int = 365
    ) -> List[datetime]:
        """
        Get available expiration dates for underlying.

        Session 81: Updated for v3 API format.

        Args:
            underlying: Underlying symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration

        Returns:
            List of available expiration dates within DTE range
        """
        params = {
            'symbol': underlying.upper(),
        }

        try:
            result = self._make_request_v3('option/list/expirations', params)

            response_list = result.get('response', [])
            if not response_list:
                return []

            # Parse expirations from v3 format
            # v3 returns: [{"symbol": "SPY", "expiration": "2024-12-20"}, ...]
            today = datetime.now().date()
            expirations = []

            for item in response_list:
                exp_str = item.get('expiration', '')
                if not exp_str:
                    continue

                # v3 returns YYYY-MM-DD format
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                except ValueError:
                    # Try YYYYMMDD format as fallback
                    try:
                        exp_date = datetime.strptime(exp_str, '%Y%m%d')
                    except ValueError:
                        continue

                dte = (exp_date.date() - today).days

                if min_dte <= dte <= max_dte:
                    expirations.append(exp_date)

            return sorted(expirations)

        except Exception as e:
            self.logger.warning(
                f"Failed to get expirations for {underlying}: {str(e)}"
            )
            return []

    def get_strikes(
        self,
        underlying: str,
        expiration: datetime
    ) -> List[float]:
        """
        Get available strikes for a specific expiration.

        Session 81: Updated for v3 API format.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date

        Returns:
            List of available strike prices in dollars
        """
        params = {
            'symbol': underlying.upper(),
            'expiration': self._format_expiration(expiration),
        }

        try:
            result = self._make_request_v3('option/list/strikes', params)

            response_list = result.get('response', [])
            if not response_list:
                return []

            # Parse strikes from v3 format
            # v3 returns: [{"symbol": "SPY", "strike": 590.000}, ...]
            strikes = [
                self._parse_strike(item.get('strike', 0))
                for item in response_list
                if item.get('strike') is not None
            ]

            return sorted(strikes)

        except Exception as e:
            self.logger.warning(
                f"Failed to get strikes for {underlying} exp {expiration.date()}: {str(e)}"
            )
            return []

    def get_historical_quotes(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Get historical quote data for a date range.

        Session 81: Updated for v3 API format.

        Args:
            underlying: Underlying symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' or 'P'
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Time interval (v3 format: '1h', '1m', '1s', etc.)

        Returns:
            DataFrame with historical quotes
        """
        # v3 requires fetching day by day for multi-day ranges
        # For now, fetch just the end_date with the specified interval
        params = {
            'symbol': underlying.upper(),
            'expiration': self._format_expiration(expiration),
            'strike': self._format_strike(strike),
            'right': self._format_right(option_type),
            'date': self._format_date(end_date),
            'interval': interval,
        }

        try:
            result = self._make_request_v3('option/history/quote', params)

            response_list = result.get('response', [])
            if not response_list:
                return pd.DataFrame()

            # Extract data from v3 nested response
            contract_data = response_list[0]
            data_list = contract_data.get('data', [])

            if not data_list:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data_list)

            # Add calculated mid price
            if not df.empty and 'bid' in df.columns and 'ask' in df.columns:
                df['mid'] = (df['bid'] + df['ask']) / 2

            return df

        except Exception as e:
            self.logger.error(
                f"Failed to get historical quotes for {underlying} {strike} {option_type}: "
                f"{str(e)}"
            )
            return pd.DataFrame()


# Convenience function for creating client
def create_thetadata_client(
    mode: str = 'rest',
    **kwargs
) -> ThetaDataProviderBase:
    """
    Factory function to create ThetaData provider.

    Args:
        mode: Provider mode ('rest' for historical, 'websocket' for live - future)
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        ThetaDataProviderBase implementation

    Raises:
        NotImplementedError: If mode is 'websocket' (future implementation)
        ValueError: If mode is invalid
    """
    if mode == 'rest':
        return ThetaDataRESTClient(**kwargs)
    elif mode == 'websocket':
        raise NotImplementedError(
            "WebSocket mode planned for future paper trading phase. "
            "Use 'rest' mode for historical backtesting."
        )
    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'rest' or 'websocket'.")


if __name__ == "__main__":
    print("=" * 60)
    print("ThetaData Client Test (v3 API)")
    print("=" * 60)

    # Create client
    client = ThetaDataRESTClient()

    # Test connection
    print("\n[TEST 1] Connecting to ThetaData terminal v3...")
    if client.connect():
        print("  PASS - Connected successfully")

        # Test get expirations
        print("\n[TEST 2] Getting SPY expirations...")
        expirations = client.get_expirations('SPY', min_dte=7, max_dte=60)
        if expirations:
            print(f"  PASS - Found {len(expirations)} expirations")
            print(f"  First 3: {[e.strftime('%Y-%m-%d') for e in expirations[:3]]}")
        else:
            print("  WARN - No expirations found")

        # Test get strikes (if expirations found)
        if expirations:
            print("\n[TEST 3] Getting strikes for first expiration...")
            strikes = client.get_strikes('SPY', expirations[0])
            if strikes:
                print(f"  PASS - Found {len(strikes)} strikes")
                # Find ATM strikes (around current SPY ~590-600)
                atm_strikes = [s for s in strikes if 580 <= s <= 620]
                print(f"  ATM strikes (580-620): {atm_strikes[:10]}")
            else:
                print("  WARN - No strikes found")

        # Test get quote with historical data
        print("\n[TEST 4] Getting historical quote...")
        from datetime import datetime
        test_exp = datetime(2024, 12, 20)
        test_date = datetime(2024, 11, 15)
        quote = client.get_quote('SPY', test_exp, 590.0, 'C', as_of=test_date)
        if quote:
            print(f"  PASS - Quote received")
            print(f"  Symbol: {quote.symbol}")
            print(f"  Bid: ${quote.bid:.2f}, Ask: ${quote.ask:.2f}, Mid: ${quote.mid:.2f}")
        else:
            print("  WARN - No quote data (check if date/strike valid)")

        # Test get greeks
        print("\n[TEST 5] Getting historical Greeks...")
        greeks = client.get_greeks('SPY', test_exp, 590.0, 'C', as_of=test_date)
        if greeks:
            print(f"  PASS - Greeks received")
            print(f"  Delta: {greeks['delta']:.4f}")
            print(f"  Gamma: {greeks['gamma']:.6f}")
            print(f"  Theta: {greeks['theta']:.4f}")
            if greeks['iv'] > 0:
                print(f"  IV: {greeks['iv']:.2%}")
            else:
                print(f"  IV: N/A (use Black-Scholes fallback)")
        else:
            print("  WARN - No Greeks data (v3 may require real-time snapshot)")

        client.disconnect()
    else:
        print("  FAIL - Could not connect to ThetaData terminal")
        print("  Make sure Theta Terminal is running at localhost:25503")

    print("\n" + "=" * 60)
    print("ThetaData Client Test Complete")
    print("=" * 60)
