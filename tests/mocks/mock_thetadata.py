"""
Mock ThetaData Provider for testing.

Session 79: Provides mock implementation of ThetaDataProviderBase
for unit testing without requiring ThetaData terminal.

Usage:
    from tests.mocks.mock_thetadata import MockThetaDataProvider
    from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher

    # Create mock with predetermined data
    mock = MockThetaDataProvider()
    mock.add_mock_quote('SPY', datetime(2024,12,20), 450.0, 'C', bid=5.00, ask=5.20)
    mock.add_mock_greeks('SPY', datetime(2024,12,20), 450.0, 'C',
                         delta=0.55, gamma=0.02, theta=-0.05, vega=0.15, iv=0.22)

    # Use mock in fetcher
    fetcher = ThetaDataOptionsFetcher(provider=mock)
    price = fetcher.get_option_price('SPY', datetime(2024,12,20), 450.0, 'C', datetime(2024,11,15))
    assert price == 5.10  # Mid price
"""

from datetime import datetime
from typing import Dict, List, Optional

from integrations.thetadata_client import ThetaDataProviderBase, OptionsQuote


class MockThetaDataProvider(ThetaDataProviderBase):
    """
    Mock ThetaData provider for unit testing.

    Allows adding predetermined quotes and Greeks data that will be
    returned when requested. Useful for testing options_module without
    requiring ThetaData terminal.

    Attributes:
        _quotes: Dict mapping (underlying, expiration, strike, type) to OptionsQuote
        _greeks: Dict mapping (underlying, expiration, strike, type) to Greeks dict
        _expirations: Dict mapping underlying to list of expiration dates
        _strikes: Dict mapping (underlying, expiration) to list of strikes
        _connected: Simulated connection state
    """

    def __init__(self, auto_connect: bool = True):
        """
        Initialize mock provider.

        Args:
            auto_connect: If True, start in connected state
        """
        self._quotes: Dict[str, OptionsQuote] = {}
        self._greeks: Dict[str, Dict[str, float]] = {}
        self._expirations: Dict[str, List[datetime]] = {}
        self._strikes: Dict[str, List[float]] = {}
        self._connected = auto_connect
        self._call_count = {
            'get_quote': 0,
            'get_greeks': 0,
            'get_expirations': 0,
            'get_strikes': 0,
        }

    def _make_key(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str
    ) -> str:
        """Generate key for quote/greeks lookup."""
        exp_str = expiration.strftime('%Y%m%d')
        return f"{underlying.upper()}_{exp_str}_{strike:.2f}_{option_type.upper()}"

    def _make_strikes_key(self, underlying: str, expiration: datetime) -> str:
        """Generate key for strikes lookup."""
        exp_str = expiration.strftime('%Y%m%d')
        return f"{underlying.upper()}_{exp_str}"

    def _generate_osi_symbol(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str
    ) -> str:
        """Generate OSI symbol."""
        symbol_part = underlying.upper()
        date_part = expiration.strftime('%y%m%d')
        type_part = option_type.upper()
        strike_part = str(int(strike * 1000)).zfill(8)
        return f"{symbol_part}{date_part}{type_part}{strike_part}"

    # =========================================================================
    # Mock Data Setup Methods
    # =========================================================================

    def add_mock_quote(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        bid: float,
        ask: float,
        bid_size: int = 100,
        ask_size: int = 100,
        volume: int = 1000,
        open_interest: int = 5000,
        iv: float = None,
        delta: float = None,
        gamma: float = None,
        theta: float = None,
        vega: float = None,
        underlying_price: float = None
    ) -> None:
        """
        Add mock quote data.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price
            option_type: 'C' or 'P'
            bid: Bid price
            ask: Ask price
            bid_size: Bid size
            ask_size: Ask size
            volume: Trading volume
            open_interest: Open interest
            iv: Implied volatility (optional)
            delta: Delta (optional)
            gamma: Gamma (optional)
            theta: Theta (optional)
            vega: Vega (optional)
            underlying_price: Underlying price (optional)
        """
        key = self._make_key(underlying, expiration, strike, option_type)

        self._quotes[key] = OptionsQuote(
            symbol=self._generate_osi_symbol(underlying, expiration, strike, option_type),
            underlying=underlying.upper(),
            strike=strike,
            expiration=expiration,
            option_type=option_type.upper(),
            timestamp=datetime.now(),
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2,
            bid_size=bid_size,
            ask_size=ask_size,
            volume=volume,
            open_interest=open_interest,
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            underlying_price=underlying_price,
        )

    def add_mock_greeks(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        iv: float,
        underlying_price: float = None
    ) -> None:
        """
        Add mock Greeks data.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price
            option_type: 'C' or 'P'
            delta: Delta value
            gamma: Gamma value
            theta: Theta value (per day)
            vega: Vega value
            iv: Implied volatility
            underlying_price: Underlying price (optional)
        """
        key = self._make_key(underlying, expiration, strike, option_type)

        self._greeks[key] = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'iv': iv,
            'underlying_price': underlying_price,
        }

    def add_mock_expirations(
        self,
        underlying: str,
        expirations: List[datetime]
    ) -> None:
        """
        Add mock expiration dates for underlying.

        Args:
            underlying: Underlying symbol
            expirations: List of expiration dates
        """
        self._expirations[underlying.upper()] = sorted(expirations)

    def add_mock_strikes(
        self,
        underlying: str,
        expiration: datetime,
        strikes: List[float]
    ) -> None:
        """
        Add mock strikes for a specific expiration.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strikes: List of strike prices
        """
        key = self._make_strikes_key(underlying, expiration)
        self._strikes[key] = sorted(strikes)

    def reset(self) -> None:
        """Clear all mock data and reset call counts."""
        self._quotes.clear()
        self._greeks.clear()
        self._expirations.clear()
        self._strikes.clear()
        for key in self._call_count:
            self._call_count[key] = 0

    # =========================================================================
    # Test Inspection Methods
    # =========================================================================

    def get_call_count(self, method: str) -> int:
        """Get number of times a method was called."""
        return self._call_count.get(method, 0)

    def set_connected(self, connected: bool) -> None:
        """Set mock connection state."""
        self._connected = connected

    # =========================================================================
    # ThetaDataProviderBase Implementation
    # =========================================================================

    def connect(self) -> bool:
        """Mock connection (always succeeds unless manually disconnected)."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Mock disconnect."""
        self._connected = False

    def is_connected(self) -> bool:
        """Return mock connection state."""
        return self._connected

    def get_quote(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: Optional[datetime] = None
    ) -> Optional[OptionsQuote]:
        """
        Get mock quote data.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price
            option_type: 'C' or 'P'
            as_of: Ignored (mock returns same data regardless of date)

        Returns:
            OptionsQuote if data exists, None otherwise
        """
        self._call_count['get_quote'] += 1

        if not self._connected:
            return None

        key = self._make_key(underlying, expiration, strike, option_type)
        return self._quotes.get(key)

    def get_greeks(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: Optional[datetime] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get mock Greeks data.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price
            option_type: 'C' or 'P'
            as_of: Ignored (mock returns same data regardless of date)

        Returns:
            Dict with Greeks if data exists, None otherwise
        """
        self._call_count['get_greeks'] += 1

        if not self._connected:
            return None

        key = self._make_key(underlying, expiration, strike, option_type)
        return self._greeks.get(key)

    def get_expirations(
        self,
        underlying: str,
        min_dte: int = 0,
        max_dte: int = 365
    ) -> List[datetime]:
        """
        Get mock expiration dates.

        Args:
            underlying: Underlying symbol
            min_dte: Minimum DTE (ignored in mock)
            max_dte: Maximum DTE (ignored in mock)

        Returns:
            List of mock expiration dates
        """
        self._call_count['get_expirations'] += 1

        if not self._connected:
            return []

        return self._expirations.get(underlying.upper(), [])

    def get_strikes(
        self,
        underlying: str,
        expiration: datetime
    ) -> List[float]:
        """
        Get mock strikes.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date

        Returns:
            List of mock strike prices
        """
        self._call_count['get_strikes'] += 1

        if not self._connected:
            return []

        key = self._make_strikes_key(underlying, expiration)
        return self._strikes.get(key, [])


# Convenience function for creating pre-populated mock
def create_spy_mock_provider() -> MockThetaDataProvider:
    """
    Create a MockThetaDataProvider pre-populated with SPY data.

    Useful for quick testing without manually setting up all mock data.

    Returns:
        MockThetaDataProvider with SPY quotes and Greeks
    """
    mock = MockThetaDataProvider()

    # Add SPY expirations (weekly for next 8 weeks)
    from datetime import timedelta
    today = datetime.now()
    expirations = []
    for i in range(1, 9):
        # Find next Friday
        days_until_friday = (4 - today.weekday()) % 7
        exp = today + timedelta(days=days_until_friday + (i * 7))
        expirations.append(exp.replace(hour=0, minute=0, second=0, microsecond=0))
    mock.add_mock_expirations('SPY', expirations)

    # Add ATM strikes for first expiration
    strikes = list(range(440, 470, 5))  # 440, 445, 450, ..., 465
    mock.add_mock_strikes('SPY', expirations[0], [float(s) for s in strikes])

    # Add quotes and greeks for a few strikes
    for strike in [445.0, 450.0, 455.0]:
        # Call
        mock.add_mock_quote(
            'SPY', expirations[0], strike, 'C',
            bid=5.00 + (450 - strike) * 0.5,
            ask=5.20 + (450 - strike) * 0.5,
            iv=0.22,
            delta=0.55 + (450 - strike) * 0.05,
            underlying_price=450.0
        )
        mock.add_mock_greeks(
            'SPY', expirations[0], strike, 'C',
            delta=0.55 + (450 - strike) * 0.05,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            iv=0.22,
            underlying_price=450.0
        )

        # Put
        mock.add_mock_quote(
            'SPY', expirations[0], strike, 'P',
            bid=4.80 - (450 - strike) * 0.5,
            ask=5.00 - (450 - strike) * 0.5,
            iv=0.22,
            delta=-0.45 + (450 - strike) * 0.05,
            underlying_price=450.0
        )
        mock.add_mock_greeks(
            'SPY', expirations[0], strike, 'P',
            delta=-0.45 + (450 - strike) * 0.05,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            iv=0.22,
            underlying_price=450.0
        )

    return mock


if __name__ == "__main__":
    print("=" * 60)
    print("Mock ThetaData Provider Test")
    print("=" * 60)

    # Create mock
    mock = create_spy_mock_provider()

    # Test connection
    print("\n[TEST 1] Connection...")
    assert mock.is_connected(), "Should be connected"
    print("  PASS - Connected")

    # Test expirations
    print("\n[TEST 2] Expirations...")
    exps = mock.get_expirations('SPY')
    assert len(exps) == 8, f"Expected 8 expirations, got {len(exps)}"
    print(f"  PASS - Got {len(exps)} expirations")

    # Test strikes
    print("\n[TEST 3] Strikes...")
    strikes = mock.get_strikes('SPY', exps[0])
    assert len(strikes) > 0, "Should have strikes"
    print(f"  PASS - Got {len(strikes)} strikes")

    # Test quote
    print("\n[TEST 4] Quote...")
    quote = mock.get_quote('SPY', exps[0], 450.0, 'C')
    assert quote is not None, "Should have quote"
    print(f"  PASS - Quote: bid=${quote.bid:.2f}, ask=${quote.ask:.2f}")

    # Test greeks
    print("\n[TEST 5] Greeks...")
    greeks = mock.get_greeks('SPY', exps[0], 450.0, 'C')
    assert greeks is not None, "Should have greeks"
    print(f"  PASS - Delta={greeks['delta']:.3f}, IV={greeks['iv']:.2%}")

    # Test call counts
    print("\n[TEST 6] Call counts...")
    print(f"  get_quote calls: {mock.get_call_count('get_quote')}")
    print(f"  get_greeks calls: {mock.get_call_count('get_greeks')}")

    # Test disconnect behavior
    print("\n[TEST 7] Disconnect behavior...")
    mock.set_connected(False)
    quote_disconnected = mock.get_quote('SPY', exps[0], 450.0, 'C')
    assert quote_disconnected is None, "Should return None when disconnected"
    print("  PASS - Returns None when disconnected")

    print("\n" + "=" * 60)
    print("Mock ThetaData Provider Test Complete")
    print("=" * 60)
