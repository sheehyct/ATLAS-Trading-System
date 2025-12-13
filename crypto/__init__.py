"""
Crypto derivatives trading module for Atlas.

Provides BTC/ETH/SOL perpetual futures trading via Coinbase Advanced Trade API.
Includes paper trading simulation since Coinbase doesn't offer native paper trading.

Modules:
- exchange: Coinbase API client for data and orders
- data: System state management
- trading: Position sizing and derivatives calculations
- simulation: Paper trading simulation
- scanning: STRAT pattern signal scanner (Session CRYPTO-2)
"""

__version__ = "0.2.0"

# Convenience imports for common usage
from crypto.scanning import CryptoSignalScanner, CryptoDetectedSignal

__all__ = [
    "CryptoSignalScanner",
    "CryptoDetectedSignal",
]
