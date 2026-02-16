"""
Options Price Provider Protocol

Defines the interface for options pricing during backtests.
Implementations include ThetaData (real historical quotes)
and Black-Scholes (offline fallback).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, List


@dataclass
class OptionsQuoteResult:
    """
    Standardized options quote for backtesting.

    Returned by all OptionsPriceProvider implementations.
    """
    # Identification
    symbol: str                  # Underlying symbol
    strike: float
    expiration: str              # YYYY-MM-DD
    option_type: str             # 'C' or 'P'

    # Prices
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0

    # Greeks (if available)
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    # Underlying state
    underlying_price: Optional[float] = None

    # Metadata
    timestamp: Optional[datetime] = None
    data_source: str = 'unknown'  # 'thetadata', 'blackscholes'

    @property
    def is_valid(self) -> bool:
        """Whether this quote has usable pricing data."""
        return self.bid > 0 or self.ask > 0 or self.mid > 0

    @property
    def fill_price_buy(self) -> float:
        """Price to buy at (worst case = ask, fallback to mid)."""
        if self.ask > 0:
            return self.ask
        return self.mid if self.mid > 0 else self.bid

    @property
    def fill_price_sell(self) -> float:
        """Price to sell at (worst case = bid, fallback to mid)."""
        if self.bid > 0:
            return self.bid
        return self.mid if self.mid > 0 else self.ask


class OptionsPriceProvider(Protocol):
    """
    Protocol for options pricing during backtests.

    Two implementations:
    - ThetaDataProvider: Real historical NBBO quotes
    - BlackScholesProvider: Offline fallback using greeks.py
    """

    def get_quote(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        option_type: str,
        as_of: datetime,
    ) -> Optional[OptionsQuoteResult]:
        """
        Get an options quote for a specific contract at a specific date.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            expiration: Expiration date YYYY-MM-DD
            strike: Strike price in dollars
            option_type: 'C' or 'P'
            as_of: Date to price the option

        Returns:
            OptionsQuoteResult if available, None if not
        """
        ...

    def find_expiration(
        self,
        symbol: str,
        target_dte: int,
        as_of: datetime,
        min_dte: int = 7,
        max_dte: int = 21,
    ) -> Optional[str]:
        """
        Find an expiration date closest to target DTE.

        Args:
            symbol: Underlying symbol
            target_dte: Desired days to expiration
            as_of: Reference date
            min_dte: Minimum acceptable DTE
            max_dte: Maximum acceptable DTE

        Returns:
            Expiration date string YYYY-MM-DD, or None if not found
        """
        ...

    def get_strikes(
        self,
        symbol: str,
        expiration: str,
    ) -> List[float]:
        """
        Get available strikes for a symbol/expiration.

        Args:
            symbol: Underlying symbol
            expiration: Expiration date YYYY-MM-DD

        Returns:
            List of available strike prices
        """
        ...
