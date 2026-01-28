"""
Crypto scanning coordinators - extracted from CryptoSignalDaemon (Phase 6.4).

Each coordinator handles a single responsibility:
- CryptoHealthMonitor: Health checks, status reporting, print_status
- CryptoEntryValidator: Stale setup validation, TFC re-evaluation at entry
- CryptoStatArbExecutor: StatArb signal checking and trade execution
- CryptoFilterManager: Signal quality filtering, deduplication, expiry cleanup
- CryptoAlertManager: Discord alerting for signals, triggers, entries, exits
"""

from crypto.scanning.coordinators.health_monitor import CryptoHealthMonitor
from crypto.scanning.coordinators.entry_validator import CryptoEntryValidator
from crypto.scanning.coordinators.statarb_executor import CryptoStatArbExecutor
from crypto.scanning.coordinators.filter_manager import CryptoFilterManager
from crypto.scanning.coordinators.alert_manager import CryptoAlertManager

__all__ = [
    "CryptoHealthMonitor",
    "CryptoEntryValidator",
    "CryptoStatArbExecutor",
    "CryptoFilterManager",
    "CryptoAlertManager",
]
