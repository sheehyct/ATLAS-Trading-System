"""
EQUITY-94: CryptoHealthMonitor - Extracted from CryptoSignalDaemon

Manages health checks, status reporting, and status display for the
crypto signal daemon.

Responsibilities:
- Run background health check loop with status logging
- Assemble full daemon status dictionary from component stats
- Print formatted status to console

Extracted as part of Phase 6.4 coordinator extraction.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from crypto.config import (
    get_current_leverage_tier,
    is_intraday_window,
    time_until_intraday_close_et,
)

logger = logging.getLogger(__name__)


@dataclass
class CryptoDaemonStats:
    """
    Thread-safe daemon statistics container.

    Passed to CryptoHealthMonitor so it can read current stats without
    direct access to daemon internals.
    """

    running: bool = False
    start_time: Optional[datetime] = None
    scan_count: int = 0
    signal_count: int = 0
    trigger_count: int = 0
    execution_count: int = 0
    error_count: int = 0
    signals_in_store: int = 0
    maintenance_window: bool = False
    symbols: List[str] = field(default_factory=list)
    scan_interval: int = 900
    entry_stats: Dict[str, Any] = field(default_factory=dict)
    paper_stats: Dict[str, Any] = field(default_factory=dict)
    statarb_stats: Dict[str, Any] = field(default_factory=dict)
    strat_active_symbols: List[str] = field(default_factory=list)


class CryptoHealthMonitor:
    """
    Health monitoring and status reporting for crypto signal daemon.

    Runs a background health check loop and assembles status from
    daemon stats + leverage tier information.
    """

    def __init__(
        self,
        get_stats: Callable[[], CryptoDaemonStats],
        get_current_time_et: Callable[[], datetime],
        health_check_interval: int = 300,
    ):
        """
        Initialize health monitor.

        Args:
            get_stats: Callback returning current daemon statistics
            get_current_time_et: Callback returning current ET time
            health_check_interval: Seconds between health checks
        """
        self._get_stats = get_stats
        self._get_current_time_et = get_current_time_et
        self._health_check_interval = health_check_interval

    def run_health_loop(self, shutdown_event: threading.Event) -> None:
        """
        Background health check loop for status logging.

        Args:
            shutdown_event: Event to signal shutdown
        """
        while not shutdown_event.is_set():
            try:
                stats = self._get_stats()
                logger.info(
                    f"HEALTH: scans={stats.scan_count}, "
                    f"signals={stats.signal_count}, "
                    f"triggers={stats.trigger_count}, "
                    f"executions={stats.execution_count}, "
                    f"errors={stats.error_count}"
                )
            except Exception as e:
                logger.error(f"Health check error: {e}")

            shutdown_event.wait(timeout=self._health_check_interval)

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status and statistics."""
        stats = self._get_stats()

        uptime = None
        if stats.start_time:
            uptime = (
                datetime.now(timezone.utc) - stats.start_time
            ).total_seconds()

        # Get current leverage tier
        now_et = self._get_current_time_et()
        tier = get_current_leverage_tier(now_et)
        is_intraday = is_intraday_window(now_et)
        time_to_close = None
        if is_intraday:
            time_to_close = time_until_intraday_close_et(now_et).total_seconds()

        return {
            "running": stats.running,
            "start_time": (
                stats.start_time.isoformat() if stats.start_time else None
            ),
            "uptime_seconds": uptime,
            "scan_count": stats.scan_count,
            "signal_count": stats.signal_count,
            "trigger_count": stats.trigger_count,
            "execution_count": stats.execution_count,
            "error_count": stats.error_count,
            "signals_in_store": stats.signals_in_store,
            "maintenance_window": stats.maintenance_window,
            "leverage_tier": tier,
            "intraday_available": is_intraday,
            "intraday_close_seconds": time_to_close,
            "symbols": stats.symbols,
            "scan_interval": stats.scan_interval,
            "entry_monitor": stats.entry_stats,
            "paper_trader": stats.paper_stats,
            "statarb": stats.statarb_stats,
            "strat_active_symbols": stats.strat_active_symbols,
        }

    def print_status(self) -> None:
        """Print current daemon status."""
        status = self.get_status()

        print("\n" + "=" * 70)
        print("CRYPTO SIGNAL DAEMON STATUS")
        print("=" * 70)
        print(f"Running: {status['running']}")
        print(f"Uptime: {status['uptime_seconds']:.0f}s" if status["uptime_seconds"] else "Not started")
        print(f"Maintenance Window: {status['maintenance_window']}")
        print()

        # Leverage tier info
        tier = status.get('leverage_tier', 'swing')
        is_intraday = status.get('intraday_available', False)
        time_to_close = status.get('intraday_close_seconds')
        print(f"Leverage Tier: {tier.upper()} ({'10x' if tier == 'intraday' else '4x'})")
        if is_intraday and time_to_close:
            hours = time_to_close / 3600
            print(f"  Intraday Window: {hours:.1f}h until 4PM ET close")
        elif not is_intraday:
            print(f"  Note: In 4-6PM ET gap (swing only)")
        print()

        print(f"Scans: {status['scan_count']}")
        print(f"Signals Detected: {status['signal_count']}")
        print(f"Triggers Fired: {status['trigger_count']}")
        print(f"Executions: {status['execution_count']}")
        print(f"Errors: {status['error_count']}")
        print()
        print(f"Signals in Store: {status['signals_in_store']}")
        print(f"Symbols: {', '.join(status['symbols'])}")
        print(f"Scan Interval: {status['scan_interval']}s")

        if status.get("entry_monitor"):
            em = status["entry_monitor"]
            print()
            print(f"Entry Monitor:")
            print(f"  Pending Signals: {em.get('pending_signals', 0)}")
            print(f"  Total Triggers: {em.get('trigger_count', 0)}")

        if status.get("paper_trader"):
            pt = status["paper_trader"]
            print()
            print(f"Paper Trader:")
            print(f"  Balance: ${pt.get('current_balance', 0):,.2f}")
            print(f"  Realized P&L: ${pt.get('realized_pnl', 0):,.2f}")
            print(f"  Open Trades: {pt.get('open_trades', 0)}")
            print(f"  Closed Trades: {pt.get('closed_trades', 0)}")

        # StatArb status
        if status.get("statarb"):
            sa = status["statarb"]
            print()
            print("StatArb:")
            print(f"  Pairs: {', '.join(sa.get('pairs', []))}")
            print(f"  Positions: {sa.get('positions', 0)}")
            zscores = sa.get('zscores', {})
            if zscores:
                for pair, z in zscores.items():
                    z_str = f"{z:.2f}" if z is not None else "N/A"
                    print(f"  Z-score ({pair}): {z_str}")
            strat_active = status.get('strat_active_symbols', [])
            if strat_active:
                print(f"  STRAT Active: {', '.join(strat_active)}")

        print("=" * 70)
