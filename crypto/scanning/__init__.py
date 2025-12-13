"""
Crypto STRAT signal scanning module.

Provides pattern detection, entry monitoring, and daemon orchestration
for crypto perpetual futures using STRAT methodology, adapted for 24/7 markets.

Session History:
- CRYPTO-2: Signal scanner
- CRYPTO-3: Entry monitor, daemon
"""

from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext
from crypto.scanning.signal_scanner import CryptoSignalScanner
from crypto.scanning.entry_monitor import (
    CryptoEntryMonitor,
    CryptoEntryMonitorConfig,
    CryptoTriggerEvent,
)
from crypto.scanning.daemon import CryptoSignalDaemon, CryptoDaemonConfig

__all__ = [
    # Scanner
    "CryptoSignalScanner",
    # Models
    "CryptoDetectedSignal",
    "CryptoSignalContext",
    # Entry Monitor
    "CryptoEntryMonitor",
    "CryptoEntryMonitorConfig",
    "CryptoTriggerEvent",
    # Daemon
    "CryptoSignalDaemon",
    "CryptoDaemonConfig",
]
