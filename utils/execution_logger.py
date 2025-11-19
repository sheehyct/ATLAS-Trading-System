"""
Execution Logger - Centralized logging for order execution audit trail

Provides comprehensive logging for all execution events:
- Order submissions, fills, rejections
- Position updates (open, close, adjust)
- Errors and exceptions
- Reconciliation reports

Log Destinations:
1. Console: INFO level and above (real-time monitoring)
2. File: logs/execution_{date}.log (all levels DEBUG-CRITICAL)
3. CSV: logs/trades_{date}.csv (trade events for analysis)
4. Errors: logs/errors_{date}.log (ERROR and CRITICAL only)

Log Rotation:
- Daily rotation (new file each day)
- Keep last 90 days
- Archive older logs to logs/archive/

Usage:
    logger = ExecutionLogger()
    logger.log_order_submission('SPY', 10, 'BUY', 'market', 'order_123')
    logger.log_order_fill('order_123', 450.25, 10, 0.50)
"""

import os
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from logging.handlers import TimedRotatingFileHandler


class ExecutionLogger:
    """
    Production-grade execution logging with multiple destinations.

    Features:
    - Multi-destination logging (console, file, CSV, errors)
    - Daily log rotation with 90-day retention
    - CSV audit trail for trade analysis
    - Comprehensive event coverage
    """

    def __init__(self, log_dir: str = 'logs/'):
        """
        Initialize execution logger.

        Args:
            log_dir: Directory for log files (default: logs/)
        """
        self.log_dir = Path(log_dir)
        self.archive_dir = self.log_dir / 'archive'

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Get today's date for file naming
        self.today = datetime.now().strftime('%Y-%m-%d')

        # Initialize loggers
        self.logger = self._create_logger()
        self.csv_file = self._create_csv_file()

        # Archive old logs (>90 days)
        self._archive_old_logs()

        self.logger.info("ExecutionLogger initialized")

    def _create_logger(self) -> logging.Logger:
        """Create multi-handler logger."""
        logger = logging.getLogger('execution_logger')
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        logger.handlers = []

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Handler 1: Console (INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler 2: Execution file (all levels)
        execution_file = self.log_dir / f'execution_{self.today}.log'
        file_handler = logging.FileHandler(execution_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Handler 3: Error file (ERROR and above)
        error_file = self.log_dir / f'errors_{self.today}.log'
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        return logger

    def _create_csv_file(self) -> Path:
        """Create or open CSV audit trail."""
        csv_path = self.log_dir / f'trades_{self.today}.csv'

        # Create with headers if doesn't exist
        if not csv_path.exists():
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'symbol',
                    'action',
                    'qty',
                    'price',
                    'order_type',
                    'order_id',
                    'status',
                    'error'
                ])

        return csv_path

    def _write_csv(
        self,
        symbol: str,
        action: str,
        qty: int,
        price: Optional[float] = None,
        order_type: str = '',
        order_id: str = '',
        status: str = '',
        error: str = ''
    ):
        """Write trade event to CSV."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                symbol,
                action,
                qty,
                price if price is not None else '',
                order_type,
                order_id,
                status,
                error
            ])

    def _archive_old_logs(self):
        """Archive logs older than 90 days."""
        cutoff_date = datetime.now() - timedelta(days=90)

        for log_file in self.log_dir.glob('*.log'):
            # Extract date from filename
            try:
                file_date_str = log_file.stem.split('_')[-1]
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')

                if file_date < cutoff_date:
                    # Move to archive
                    archive_path = self.archive_dir / log_file.name
                    log_file.rename(archive_path)
                    self.logger.debug(f"Archived old log: {log_file.name}")
            except (ValueError, IndexError):
                # Skip files that don't match expected pattern
                continue

        # Archive old CSV files
        for csv_file in self.log_dir.glob('trades_*.csv'):
            try:
                file_date_str = csv_file.stem.split('_')[-1]
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')

                if file_date < cutoff_date:
                    archive_path = self.archive_dir / csv_file.name
                    csv_file.rename(archive_path)
                    self.logger.debug(f"Archived old CSV: {csv_file.name}")
            except (ValueError, IndexError):
                continue

    def log_order_submission(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str,
        order_id: str
    ):
        """
        Log order submission event.

        Args:
            symbol: Ticker symbol
            qty: Share quantity
            side: BUY or SELL
            order_type: market, limit, stop
            order_id: Alpaca order ID
        """
        self.logger.info(
            f"Order submitted: {side} {qty} {symbol} "
            f"(type={order_type}, id={order_id})"
        )

        self._write_csv(
            symbol=symbol,
            action=side,
            qty=qty,
            order_type=order_type,
            order_id=order_id,
            status='submitted'
        )

    def log_order_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_qty: int,
        commission: float,
        symbol: str = 'UNKNOWN'
    ):
        """
        Log order fill event.

        Args:
            order_id: Alpaca order ID
            fill_price: Execution price
            fill_qty: Filled quantity
            commission: Trading commission
            symbol: Ticker symbol (optional, for CSV)
        """
        self.logger.info(
            f"Order filled: id={order_id}, price=${fill_price:.2f}, "
            f"qty={fill_qty}, commission=${commission:.2f}"
        )

        self._write_csv(
            symbol=symbol,
            action='FILL',
            qty=fill_qty,
            price=fill_price,
            order_id=order_id,
            status='filled'
        )

    def log_order_rejection(
        self,
        order_id: str,
        reason: str,
        symbol: str = 'UNKNOWN',
        qty: int = 0
    ):
        """
        Log order rejection event.

        Args:
            order_id: Alpaca order ID
            reason: Rejection reason
            symbol: Ticker symbol (optional)
            qty: Quantity attempted (optional)
        """
        self.logger.warning(
            f"Order rejected: id={order_id}, reason={reason}"
        )

        self._write_csv(
            symbol=symbol,
            action='REJECT',
            qty=qty,
            order_id=order_id,
            status='rejected',
            error=reason
        )

    def log_position_update(
        self,
        symbol: str,
        action: str,
        qty: int,
        price: float
    ):
        """
        Log position update event.

        Args:
            symbol: Ticker symbol
            action: OPEN, CLOSE, ADJUST
            qty: Position quantity
            price: Current price
        """
        self.logger.info(
            f"Position {action.lower()}: {symbol} qty={qty} @ ${price:.2f}"
        )

        self._write_csv(
            symbol=symbol,
            action=action,
            qty=qty,
            price=price,
            status='position_update'
        )

    def log_error(
        self,
        component: str,
        error_msg: str,
        exc_info: Optional[Exception] = None
    ):
        """
        Log error event.

        Args:
            component: Component name (e.g., 'AlpacaTradingClient')
            error_msg: Error description
            exc_info: Exception object (optional)
        """
        if exc_info:
            self.logger.error(
                f"{component}: {error_msg}",
                exc_info=exc_info
            )
        else:
            self.logger.error(f"{component}: {error_msg}")

    def log_reconciliation(
        self,
        target_positions: Dict[str, int],
        actual_positions: Dict[str, int],
        discrepancies: List[str]
    ):
        """
        Log position reconciliation report.

        Args:
            target_positions: Expected positions {symbol: qty}
            actual_positions: Actual positions {symbol: qty}
            discrepancies: List of discrepancy descriptions
        """
        self.logger.info("=" * 60)
        self.logger.info("POSITION RECONCILIATION")
        self.logger.info("=" * 60)

        if not discrepancies:
            self.logger.info("All positions match target (within tolerance)")
        else:
            self.logger.warning(f"Found {len(discrepancies)} discrepancies:")
            for discrepancy in discrepancies:
                self.logger.warning(f"  - {discrepancy}")

        # Log target vs actual summary
        all_symbols = set(target_positions.keys()) | set(actual_positions.keys())

        self.logger.info("")
        self.logger.info("Position Summary:")
        self.logger.info(f"{'Symbol':<10} {'Target':<10} {'Actual':<10} {'Diff':<10}")
        self.logger.info("-" * 40)

        for symbol in sorted(all_symbols):
            target = target_positions.get(symbol, 0)
            actual = actual_positions.get(symbol, 0)
            diff = actual - target

            status = "MATCH" if diff == 0 else "DIFF"
            self.logger.info(
                f"{symbol:<10} {target:<10} {actual:<10} {diff:<10} [{status}]"
            )

        self.logger.info("=" * 60)
