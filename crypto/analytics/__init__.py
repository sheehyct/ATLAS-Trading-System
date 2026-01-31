"""
Crypto analytics modules.

Provides P/L calculation and analysis for Coinbase CFM derivatives trading.
"""

from crypto.analytics.coinbase_cfm_calculator import (
    CFMTransaction,
    CFMLot,
    CFMRealizedPL,
    CFMOpenPosition,
    CoinbaseCFMCalculator,
)

__all__ = [
    "CFMTransaction",
    "CFMLot",
    "CFMRealizedPL",
    "CFMOpenPosition",
    "CoinbaseCFMCalculator",
]
