"""
Crypto derivatives trading module for Atlas.

Provides BTC/ETH/SOL perpetual futures trading via Coinbase Advanced Trade API.
Includes paper trading simulation since Coinbase doesn't offer native paper trading.

Modules:
- exchange: Coinbase API client for data and orders
- data: System state management
- trading: Position sizing and derivatives calculations
- simulation: Paper trading simulation
- scanning: STRAT pattern signal scanner, entry monitor, and daemon

Session History:
- CRYPTO-2: Signal scanner
- CRYPTO-3: Entry monitor, daemon orchestrator
"""

__version__ = "0.3.0"

# Convenience imports for common usage
from crypto.scanning import (
    CryptoSignalScanner,
    CryptoDetectedSignal,
    CryptoEntryMonitor,
    CryptoEntryMonitorConfig,
    CryptoTriggerEvent,
    CryptoSignalDaemon,
    CryptoDaemonConfig,
)

__all__ = [
    # Scanner
    "CryptoSignalScanner",
    "CryptoDetectedSignal",
    # Entry Monitor
    "CryptoEntryMonitor",
    "CryptoEntryMonitorConfig",
    "CryptoTriggerEvent",
    # Daemon
    "CryptoSignalDaemon",
    "CryptoDaemonConfig",
]
