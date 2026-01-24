"""
EQUITY-85: HealthMonitor - Extracted from SignalDaemon

Manages health checks and daily trade audits for the signal automation system.

Responsibilities:
- Generate health status with uptime, counts, and component status
- Generate daily trade audits with P&L statistics
- Send audit results to alerters (Discord, logging)
- Log health checks via logging alerter

Extracted as part of Phase 4 god class refactoring (EQUITY-84 plan).
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Protocol

logger = logging.getLogger(__name__)


class SchedulerProtocol(Protocol):
    """Protocol for scheduler status."""
    def get_status(self) -> Dict[str, Any]: ...


class PositionMonitorProtocol(Protocol):
    """Protocol for position monitor methods needed by HealthMonitor."""
    def get_stats(self) -> Dict[str, Any]: ...
    def get_tracked_positions(self) -> List[Any]: ...


@dataclass
class DaemonStats:
    """
    Thread-safe daemon statistics container.

    Passed to HealthMonitor so it can read current stats without
    direct access to daemon internals.
    """
    start_time: Optional[datetime] = None
    is_running: bool = False
    scan_count: int = 0
    signal_count: int = 0
    execution_count: int = 0
    exit_count: int = 0
    error_count: int = 0


class HealthMonitor:
    """
    Monitors daemon health and generates daily trade audits.

    Extracted from SignalDaemon as part of EQUITY-85 Phase 4 refactoring.
    Uses Facade pattern - daemon delegates health/audit to this coordinator.

    Args:
        stats: DaemonStats dataclass with current statistics
        signal_store: SignalStore for counting signals
        alerters: List of alerter instances for health check logging
        scheduler: Scheduler instance for status info
        position_monitor: Optional PositionMonitor for position stats
        paper_trades_path: Path to paper trades JSON file
        on_error: Optional callback for error counting
    """

    def __init__(
        self,
        stats: DaemonStats,
        signal_store: Any,  # SignalStore
        alerters: List[Any],  # List[BaseAlerter]
        scheduler: Optional[SchedulerProtocol] = None,
        position_monitor: Optional[PositionMonitorProtocol] = None,
        paper_trades_path: Optional[Path] = None,
        on_error: Optional[Callable[[], None]] = None,
        executor_enabled: bool = False,
    ):
        self._stats = stats
        self._signal_store = signal_store
        self._alerters = alerters
        self._scheduler = scheduler
        self._position_monitor = position_monitor
        self._paper_trades_path = paper_trades_path or self._default_paper_trades_path()
        self._on_error = on_error or (lambda: None)
        self._executor_enabled = executor_enabled

    @staticmethod
    def _default_paper_trades_path() -> Path:
        """Return default paper trades path relative to package location."""
        return Path(__file__).parent.parent.parent.parent / 'paper_trades' / 'paper_trades.json'

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Health status dictionary with:
            - status: 'healthy' or 'stopped'
            - uptime_seconds: Time since daemon start
            - scan_count, signal_count, execution_count, exit_count, error_count
            - signals_in_store: Current signal store size
            - alerters: List of alerter names
            - scheduler: Scheduler status dict
            - execution_enabled, monitoring_enabled: Feature flags
            - monitoring: Position monitor stats (if available)
        """
        uptime = None
        if self._stats.start_time:
            uptime = (datetime.now() - self._stats.start_time).total_seconds()

        status = {
            'status': 'healthy' if self._stats.is_running else 'stopped',
            'uptime_seconds': uptime,
            'scan_count': self._stats.scan_count,
            'signal_count': self._stats.signal_count,
            'execution_count': self._stats.execution_count,
            'exit_count': self._stats.exit_count,
            'error_count': self._stats.error_count,
            'signals_in_store': len(self._signal_store) if self._signal_store else 0,
            'alerters': [getattr(a, 'name', str(a)) for a in self._alerters],
            'scheduler': self._scheduler.get_status() if self._scheduler else {},
            'execution_enabled': self._executor_enabled,
            'monitoring_enabled': self._position_monitor is not None,
        }

        # Add monitoring stats (Session 83K-49)
        if self._position_monitor is not None:
            try:
                monitor_stats = self._position_monitor.get_stats()
                status['monitoring'] = monitor_stats
            except Exception as e:
                logger.warning(f"Failed to get position monitor stats: {e}")
                status['monitoring'] = {'error': str(e)}

        # Log health check via logging alerters
        self._log_health_check(status)

        return status

    def _log_health_check(self, status: Dict[str, Any]) -> None:
        """Log health check via logging alerters."""
        # Import here to avoid circular imports
        from strat.signal_automation.alerters import LoggingAlerter

        for alerter in self._alerters:
            if isinstance(alerter, LoggingAlerter):
                try:
                    alerter.log_health_check(status)
                except Exception as e:
                    logger.warning(f"Failed to log health check: {e}")

    def generate_daily_audit(self) -> Dict[str, Any]:
        """
        EQUITY-52: Generate daily trade audit statistics.

        Collects today's trading activity including:
        - Trades executed today
        - Win/loss counts and P&L
        - Open positions summary
        - Anomaly detection

        Returns:
            Dictionary with audit data for Discord reporting
        """
        today = date.today().isoformat()

        # Initialize audit data
        audit_data = {
            'date': today,
            'trades_today': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'profit_factor': 0.0,
            'open_positions': [],
            'anomalies': [],
        }

        # Load paper trades
        self._load_paper_trades_audit(audit_data, today)

        # Get open positions from position monitor
        self._add_open_positions_to_audit(audit_data)

        logger.info(
            f"Daily audit generated: {audit_data['trades_today']} trades, "
            f"P/L: ${audit_data['total_pnl']:.2f}, "
            f"Open: {len(audit_data['open_positions'])}"
        )

        return audit_data

    def _load_paper_trades_audit(self, audit_data: Dict[str, Any], today: str) -> None:
        """Load and process paper trades for audit."""
        if not self._paper_trades_path.exists():
            logger.warning(f"Paper trades file not found at {self._paper_trades_path.absolute()}")
            audit_data['anomalies'].append(f"Paper trades file not found: {self._paper_trades_path}")
            return

        try:
            with open(self._paper_trades_path, 'r') as f:
                data = json.load(f)
                trades = data if isinstance(data, list) else data.get('trades', [])

            # Filter today's closed trades
            gross_profit = 0.0
            gross_loss = 0.0
            for trade in trades:
                exit_time = trade.get('exit_time', '')
                if exit_time and exit_time.startswith(today):
                    audit_data['trades_today'] += 1
                    pnl = trade.get('pnl_dollars', 0)
                    audit_data['total_pnl'] += pnl
                    if pnl > 0:
                        audit_data['wins'] += 1
                        gross_profit += pnl
                    elif pnl < 0:
                        audit_data['losses'] += 1
                        gross_loss += abs(pnl)

            # Calculate profit factor
            if gross_loss > 0:
                audit_data['profit_factor'] = gross_profit / gross_loss

        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.error(f"Failed to load paper trades for audit: {e}")
            audit_data['anomalies'].append(f"Failed to load trades: {e}")

    def _add_open_positions_to_audit(self, audit_data: Dict[str, Any]) -> None:
        """Add open positions to audit data."""
        if self._position_monitor is None:
            return

        try:
            positions = self._position_monitor.get_tracked_positions()
            for pos in positions:
                pos_summary = {
                    'symbol': getattr(pos, 'symbol', 'Unknown'),
                    'pattern_type': getattr(pos, 'pattern_type', 'Unknown'),
                    'timeframe': getattr(pos, 'timeframe', 'Unknown'),
                    'unrealized_pnl': getattr(pos, 'unrealized_pnl', 0.0),
                    'unrealized_pct': getattr(pos, 'unrealized_pct', 0.0),
                }
                audit_data['open_positions'].append(pos_summary)
        except Exception as e:
            logger.warning(f"Failed to get open positions for audit: {e}")
            audit_data['anomalies'].append(f"Failed to get open positions: {e}")

    def run_daily_audit(self) -> None:
        """
        EQUITY-52: Run daily audit and send to Discord.

        Called by scheduler at 4:30 PM ET.
        Generates audit data and sends via all alerters that support it.
        """
        try:
            audit_data = self.generate_daily_audit()

            # Send to all alerters that have send_daily_audit method
            for alerter in self._alerters:
                if hasattr(alerter, 'send_daily_audit'):
                    try:
                        alerter.send_daily_audit(audit_data)
                    except Exception as e:
                        logger.error(f"Failed to send daily audit via {getattr(alerter, 'name', str(alerter))}: {e}")

            logger.info("Daily audit completed and sent")
        except Exception as e:
            logger.error(f"Failed to run daily audit: {e}")
            self._on_error()

    def update_stats(self, **kwargs) -> None:
        """
        Update stats from daemon.

        Allows daemon to push updated stats without direct access.

        Args:
            **kwargs: Stats to update (scan_count, signal_count, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self._stats, key):
                setattr(self._stats, key, value)

    @property
    def paper_trades_path(self) -> Path:
        """Return configured paper trades path."""
        return self._paper_trades_path
