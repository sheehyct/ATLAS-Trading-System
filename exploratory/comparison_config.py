"""
Session 83B: Configuration for Expanded Comparison Testing

This module provides configuration dataclasses for the synthetic vs real
options P/L comparison script.

Symbols: SPY, QQQ, AAPL, IWM, DIA, NVDA
Date Range: Configurable (default Jan 2024 - Nov 2024)
Strike Intervals: Per-symbol standard option intervals
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


# Standard strike intervals for each symbol (in dollars)
STRIKE_INTERVALS: Dict[str, float] = {
    'SPY': 5.0,    # SPY has $5 strike intervals
    'QQQ': 5.0,    # QQQ has $5 strike intervals
    'AAPL': 2.5,   # AAPL has $2.50 strike intervals
    'IWM': 1.0,    # IWM has $1 strike intervals
    'DIA': 5.0,    # DIA has $5 strike intervals
    'NVDA': 5.0,   # NVDA has $5 strike intervals (high price)
}

# Approximate price ranges for ATM strike estimation (Oct 2024)
# Used as fallback when Tiingo data unavailable
APPROXIMATE_PRICES: Dict[str, float] = {
    'SPY': 575.0,
    'QQQ': 490.0,
    'AAPL': 225.0,
    'IWM': 220.0,
    'DIA': 425.0,
    'NVDA': 140.0,
}


@dataclass
class ComparisonConfig:
    """
    Configuration for synthetic vs real options P/L comparison.

    Attributes:
        symbols: List of underlying symbols to test
        start_date: Start of date range for trade generation
        end_date: End of date range for trade generation
        expiration_offset_days: Days until option expiration (from trade date)
        patterns_per_symbol: Number of pattern trades to generate per symbol
        use_dynamic_strikes: If True, fetch ATM strikes from Tiingo
        validate_data_first: If True, run data availability check before backtest
    """
    symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'AAPL', 'IWM', 'DIA', 'NVDA'
    ])
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 11, 30))
    expiration_offset_days: int = 30
    patterns_per_symbol: int = 3
    use_dynamic_strikes: bool = True
    validate_data_first: bool = True

    def get_strike_interval(self, symbol: str) -> float:
        """Get the standard strike interval for a symbol."""
        return STRIKE_INTERVALS.get(symbol, 5.0)

    def get_approximate_price(self, symbol: str) -> float:
        """Get approximate price for ATM strike fallback."""
        return APPROXIMATE_PRICES.get(symbol, 100.0)

    def round_to_strike(self, price: float, symbol: str) -> float:
        """Round a price to the nearest valid strike for a symbol."""
        interval = self.get_strike_interval(symbol)
        return round(price / interval) * interval


@dataclass
class SymbolMetrics:
    """
    Per-symbol metrics from comparison testing.

    Attributes:
        symbol: Underlying symbol
        trade_count: Number of trades processed
        thetadata_pct: Percentage of trades using ThetaData pricing
        black_scholes_pct: Percentage of trades using Black-Scholes fallback
        mixed_pct: Percentage of trades with mixed data sources
        price_mae: Mean Absolute Error of option prices (per share)
        price_mape: Mean Absolute Percentage Error of option prices
        pnl_mae: Mean Absolute Error of P/L
        pnl_rmse: Root Mean Squared Error of P/L
        data_available: Whether data was available for this symbol
    """
    symbol: str
    trade_count: int = 0
    thetadata_pct: float = 0.0
    black_scholes_pct: float = 0.0
    mixed_pct: float = 0.0
    price_mae: Optional[float] = None
    price_mape: Optional[float] = None
    pnl_mae: Optional[float] = None
    pnl_rmse: Optional[float] = None
    data_available: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'symbol': self.symbol,
            'trades': self.trade_count,
            'thetadata_pct': self.thetadata_pct,
            'bs_pct': self.black_scholes_pct,
            'price_mae': self.price_mae,
            'price_mape': self.price_mape,
            'pnl_mae': self.pnl_mae,
        }


@dataclass
class DataAvailability:
    """
    Result of data availability validation for a symbol.

    Attributes:
        symbol: Underlying symbol
        tiingo_available: Whether Tiingo has price data
        thetadata_connected: Whether ThetaData terminal is connected
        thetadata_has_expirations: Whether ThetaData has options expirations
        sample_quote_ok: Whether a sample quote was successfully retrieved
        error_message: Error message if any check failed
    """
    symbol: str
    tiingo_available: bool = False
    thetadata_connected: bool = False
    thetadata_has_expirations: bool = False
    sample_quote_ok: bool = False
    error_message: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """Check if symbol is ready for comparison testing."""
        return (
            self.tiingo_available and
            self.thetadata_connected and
            self.thetadata_has_expirations
        )

    def status_str(self) -> str:
        """Return status string for display."""
        if self.is_ready:
            return "READY"
        elif not self.tiingo_available:
            return "NO_TIINGO"
        elif not self.thetadata_connected:
            return "NO_THETADATA"
        elif not self.thetadata_has_expirations:
            return "NO_OPTIONS"
        else:
            return "UNKNOWN"
