"""
Logging Alerter - Session 83K-45

Structured JSON logging alerter for signals.
Always enabled for audit trail and debugging.

Log Format:
- JSON structured logs for machine parsing
- Human-readable console output
- Rotating file handler for disk management
"""

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from logging.handlers import RotatingFileHandler

from strat.signal_automation.alerters.base import BaseAlerter
from strat.signal_automation.signal_store import StoredSignal


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }

        # Add signal data if present
        if hasattr(record, 'signal_data'):
            log_entry['signal'] = record.signal_data

        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_entry['extra'] = record.extra

        return json.dumps(log_entry)


class LoggingAlerter(BaseAlerter):
    """
    Structured logging alerter for signals.

    Features:
    - JSON structured logs to file
    - Console output for visibility
    - Rotating file handler (10MB, 5 backups)
    - Always enabled (no external dependencies)

    Usage:
        alerter = LoggingAlerter('logs/signals.log')
        alerter.send_alert(signal)
    """

    def __init__(
        self,
        log_file: str = 'logs/signals.log',
        level: str = 'INFO',
        console_output: bool = True
    ):
        """
        Initialize logging alerter.

        Args:
            log_file: Path to log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_output: Whether to also log to console
        """
        super().__init__('logging')

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create dedicated logger for signals
        self.logger = logging.getLogger('strat.signals')
        self.logger.setLevel(getattr(logging, level.upper()))

        # Prevent propagation to root logger
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers = []

        # File handler with rotation
        file_handler = RotatingFileHandler(
            str(self.log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

        # Console handler (human-readable)
        if console_output:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def send_alert(self, signal: StoredSignal) -> bool:
        """
        Log signal alert.

        Args:
            signal: Signal to log

        Returns:
            True (logging always succeeds)
        """
        # Check throttling
        if self.is_throttled(signal.signal_key):
            self.logger.debug(f"Throttled: {signal.signal_key}")
            return True  # Throttled is not a failure

        # Prepare signal data for JSON
        signal_data = {
            'signal_key': signal.signal_key,
            'symbol': signal.symbol,
            'pattern': signal.pattern_type,
            'direction': signal.direction,
            'timeframe': signal.timeframe,
            'entry': signal.entry_trigger,
            'target': signal.target_price,
            'stop': signal.stop_price,
            'magnitude_pct': signal.magnitude_pct,
            'risk_reward': signal.risk_reward,
            'vix': signal.vix,
            'regime': signal.market_regime,
            'detected_time': signal.detected_time.isoformat(),
        }

        # Create log record with signal data
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            __file__,
            0,
            f"SIGNAL: {signal.symbol} {signal.pattern_type} {signal.direction} @ ${signal.entry_trigger:.2f}",
            (),
            None
        )
        record.signal_data = signal_data

        # Log with custom record
        self.logger.handle(record)

        # Record for throttling
        self.record_alert(signal.signal_key)

        return True

    def send_batch_alert(self, signals: list) -> bool:
        """
        Log multiple signals as a batch.

        Args:
            signals: List of StoredSignal

        Returns:
            True (logging always succeeds)
        """
        if not signals:
            return True

        # Log summary first
        self.logger.info(f"BATCH ALERT: {len(signals)} signals detected")

        # Log each signal
        for signal in signals:
            self.send_alert(signal)

        return True

    def test_connection(self) -> bool:
        """
        Test logging is working.

        Returns:
            True if logging works
        """
        try:
            self.logger.info("Logging alerter connection test")
            return True
        except Exception as e:
            print(f"Logging test failed: {e}")
            return False

    def log_scan_started(self, timeframe: str, symbols: list) -> None:
        """Log that a scan has started."""
        self.logger.info(
            f"SCAN STARTED: {timeframe} timeframe, symbols: {', '.join(symbols)}"
        )

    def log_scan_completed(
        self,
        timeframe: str,
        signals_found: int,
        duration_seconds: float
    ) -> None:
        """Log that a scan has completed."""
        self.logger.info(
            f"SCAN COMPLETED: {timeframe} - {signals_found} signals in {duration_seconds:.2f}s"
        )

    def log_scan_error(self, timeframe: str, error: str) -> None:
        """Log a scan error."""
        self.logger.error(f"SCAN ERROR: {timeframe} - {error}")

    def log_daemon_started(self) -> None:
        """Log daemon startup."""
        self.logger.info("DAEMON STARTED: Signal automation daemon is running")

    def log_daemon_stopped(self, reason: str = 'shutdown') -> None:
        """Log daemon shutdown."""
        self.logger.info(f"DAEMON STOPPED: {reason}")

    def log_health_check(self, status: Dict[str, Any]) -> None:
        """Log health check status."""
        record = self.logger.makeRecord(
            self.logger.name,
            logging.DEBUG,
            __file__,
            0,
            "HEALTH CHECK",
            (),
            None
        )
        record.extra = status
        self.logger.handle(record)

    def log_position_exit(
        self,
        exit_signal,  # ExitSignal from position_monitor
        order_result: Dict[str, Any]
    ) -> None:
        """
        Log a position exit (Session 83K-49).

        Args:
            exit_signal: ExitSignal with exit details
            order_result: Order result from Alpaca
        """
        reason_str = exit_signal.reason.value if hasattr(exit_signal.reason, 'value') else str(exit_signal.reason)
        pnl = exit_signal.unrealized_pnl

        # Prepare exit data for JSON
        exit_data = {
            'osi_symbol': exit_signal.osi_symbol,
            'signal_key': exit_signal.signal_key,
            'exit_reason': reason_str,
            'pnl': pnl,
            'underlying_price': exit_signal.underlying_price,
            'option_price': exit_signal.current_option_price,
            'dte': exit_signal.dte,
            'order_id': order_result.get('id') if order_result else None,
        }

        # Create log record with exit data
        pnl_indicator = "PROFIT" if pnl >= 0 else "LOSS"
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            __file__,
            0,
            f"POSITION EXIT [{pnl_indicator}]: {exit_signal.osi_symbol} - {reason_str} - P&L: ${pnl:+.2f}",
            (),
            None
        )
        record.extra = exit_data

        self.logger.handle(record)
