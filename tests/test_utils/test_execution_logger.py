"""
Tests for ExecutionLogger

Validates:
- Logger initialization and directory creation
- Order submission logging
- Order fill logging
- Order rejection logging
- Position update logging
- Error logging
- Reconciliation reporting
- CSV audit trail format
- Log rotation and archival (90-day retention)
"""

import pytest
import csv
from pathlib import Path
from datetime import datetime, timedelta
import shutil

from utils.execution_logger import ExecutionLogger


class TestExecutionLoggerInitialization:
    """Test logger initialization and directory creation."""

    def test_default_initialization(self, tmp_path):
        """Test logger initializes with default log directory."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        # Verify directories created
        assert log_dir.exists()
        assert (log_dir / 'archive').exists()

        # Verify log files created
        today = datetime.now().strftime('%Y-%m-%d')
        assert (log_dir / f'execution_{today}.log').exists()
        assert (log_dir / f'errors_{today}.log').exists()
        assert (log_dir / f'trades_{today}.csv').exists()

    def test_csv_headers_created(self, tmp_path):
        """Test CSV file created with correct headers."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        # Read headers
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            'timestamp',
            'symbol',
            'action',
            'qty',
            'price',
            'order_type',
            'order_id',
            'status',
            'error'
        ]

        assert headers == expected_headers


class TestOrderSubmissionLogging:
    """Test order submission event logging."""

    def test_log_order_submission(self, tmp_path):
        """Test logging market order submission."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        logger.log_order_submission(
            symbol='SPY',
            qty=10,
            side='BUY',
            order_type='market',
            order_id='order_123'
        )

        # Verify CSV entry
        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        assert row['symbol'] == 'SPY'
        assert row['action'] == 'BUY'
        assert row['qty'] == '10'
        assert row['order_type'] == 'market'
        assert row['order_id'] == 'order_123'
        assert row['status'] == 'submitted'

    def test_log_multiple_submissions(self, tmp_path):
        """Test logging multiple order submissions."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        # Submit 3 orders
        for i in range(3):
            logger.log_order_submission(
                symbol=f'STOCK{i}',
                qty=10 * (i + 1),
                side='BUY' if i % 2 == 0 else 'SELL',
                order_type='market',
                order_id=f'order_{i}'
            )

        # Verify 3 CSV entries
        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3


class TestOrderFillLogging:
    """Test order fill event logging."""

    def test_log_order_fill(self, tmp_path):
        """Test logging order fill."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        logger.log_order_fill(
            order_id='order_123',
            fill_price=450.25,
            fill_qty=10,
            commission=0.50,
            symbol='SPY'
        )

        # Verify CSV entry
        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        assert row['symbol'] == 'SPY'
        assert row['action'] == 'FILL'
        assert row['qty'] == '10'
        assert row['price'] == '450.25'
        assert row['order_id'] == 'order_123'
        assert row['status'] == 'filled'


class TestOrderRejectionLogging:
    """Test order rejection event logging."""

    def test_log_order_rejection(self, tmp_path):
        """Test logging order rejection."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        logger.log_order_rejection(
            order_id='order_456',
            reason='Insufficient buying power',
            symbol='TSLA',
            qty=5
        )

        # Verify CSV entry
        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        assert row['symbol'] == 'TSLA'
        assert row['action'] == 'REJECT'
        assert row['qty'] == '5'
        assert row['order_id'] == 'order_456'
        assert row['status'] == 'rejected'
        assert row['error'] == 'Insufficient buying power'


class TestPositionUpdateLogging:
    """Test position update event logging."""

    def test_log_position_open(self, tmp_path):
        """Test logging position open."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        logger.log_position_update(
            symbol='AAPL',
            action='OPEN',
            qty=20,
            price=175.50
        )

        # Verify CSV entry
        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        assert row['symbol'] == 'AAPL'
        assert row['action'] == 'OPEN'
        assert row['qty'] == '20'
        assert row['price'] == '175.5'
        assert row['status'] == 'position_update'

    def test_log_position_close(self, tmp_path):
        """Test logging position close."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        logger.log_position_update(
            symbol='AAPL',
            action='CLOSE',
            qty=20,
            price=180.25
        )

        # Verify CSV entry
        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]['action'] == 'CLOSE'


class TestErrorLogging:
    """Test error event logging."""

    def test_log_error_without_exception(self, tmp_path):
        """Test logging error message without exception."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        logger.log_error(
            component='AlpacaTradingClient',
            error_msg='Connection timeout'
        )

        # Verify error logged to file
        today = datetime.now().strftime('%Y-%m-%d')
        error_file = log_dir / f'errors_{today}.log'

        with open(error_file, 'r') as f:
            content = f.read()

        assert 'AlpacaTradingClient' in content
        assert 'Connection timeout' in content
        assert 'ERROR' in content

    def test_log_error_with_exception(self, tmp_path):
        """Test logging error with exception traceback."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_error(
                component='OrderValidator',
                error_msg='Validation failed',
                exc_info=e
            )

        # Verify error logged with traceback
        today = datetime.now().strftime('%Y-%m-%d')
        error_file = log_dir / f'errors_{today}.log'

        with open(error_file, 'r') as f:
            content = f.read()

        assert 'OrderValidator' in content
        assert 'Validation failed' in content
        assert 'ValueError' in content
        assert 'Test exception' in content


class TestReconciliationLogging:
    """Test position reconciliation reporting."""

    def test_log_reconciliation_match(self, tmp_path):
        """Test reconciliation when positions match."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        target = {'SPY': 10, 'QQQ': 5}
        actual = {'SPY': 10, 'QQQ': 5}
        discrepancies = []

        logger.log_reconciliation(target, actual, discrepancies)

        # Verify logged to execution file
        today = datetime.now().strftime('%Y-%m-%d')
        exec_file = log_dir / f'execution_{today}.log'

        with open(exec_file, 'r') as f:
            content = f.read()

        assert 'POSITION RECONCILIATION' in content
        assert 'All positions match target' in content
        assert 'SPY' in content
        assert 'QQQ' in content

    def test_log_reconciliation_discrepancies(self, tmp_path):
        """Test reconciliation with discrepancies."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        target = {'SPY': 10, 'QQQ': 5}
        actual = {'SPY': 9, 'QQQ': 5}  # SPY off by 1
        discrepancies = ['SPY: target=10, actual=9, diff=-1']

        logger.log_reconciliation(target, actual, discrepancies)

        # Verify discrepancies logged
        today = datetime.now().strftime('%Y-%m-%d')
        exec_file = log_dir / f'execution_{today}.log'

        with open(exec_file, 'r') as f:
            content = f.read()

        assert 'Found 1 discrepancies' in content
        assert 'SPY: target=10, actual=9, diff=-1' in content


class TestLogRotationAndArchival:
    """Test log rotation and archival (90-day retention)."""

    def test_archive_old_logs(self, tmp_path):
        """Test archival of logs older than 90 days."""
        log_dir = tmp_path / 'logs'
        archive_dir = log_dir / 'archive'
        log_dir.mkdir(parents=True)
        archive_dir.mkdir(parents=True)

        # Create old log file (100 days ago)
        old_date = datetime.now() - timedelta(days=100)
        old_date_str = old_date.strftime('%Y-%m-%d')
        old_log = log_dir / f'execution_{old_date_str}.log'
        old_log.write_text('Old log content')

        # Create recent log (30 days ago)
        recent_date = datetime.now() - timedelta(days=30)
        recent_date_str = recent_date.strftime('%Y-%m-%d')
        recent_log = log_dir / f'execution_{recent_date_str}.log'
        recent_log.write_text('Recent log content')

        # Initialize logger (triggers archival)
        logger = ExecutionLogger(log_dir=str(log_dir))

        # Verify old log archived
        assert not old_log.exists()
        assert (archive_dir / f'execution_{old_date_str}.log').exists()

        # Verify recent log NOT archived
        assert recent_log.exists()
        assert not (archive_dir / f'execution_{recent_date_str}.log').exists()

    def test_archive_old_csv_files(self, tmp_path):
        """Test archival of old CSV files."""
        log_dir = tmp_path / 'logs'
        archive_dir = log_dir / 'archive'
        log_dir.mkdir(parents=True)
        archive_dir.mkdir(parents=True)

        # Create old CSV (100 days ago)
        old_date = datetime.now() - timedelta(days=100)
        old_date_str = old_date.strftime('%Y-%m-%d')
        old_csv = log_dir / f'trades_{old_date_str}.csv'
        old_csv.write_text('timestamp,symbol,action\n')

        # Initialize logger
        logger = ExecutionLogger(log_dir=str(log_dir))

        # Verify old CSV archived
        assert not old_csv.exists()
        assert (archive_dir / f'trades_{old_date_str}.csv').exists()


class TestCSVAuditTrail:
    """Test CSV audit trail format and integrity."""

    def test_csv_format_consistency(self, tmp_path):
        """Test CSV maintains consistent format across events."""
        log_dir = tmp_path / 'logs'
        logger = ExecutionLogger(log_dir=str(log_dir))

        # Log multiple event types
        logger.log_order_submission('SPY', 10, 'BUY', 'market', 'order_1')
        logger.log_order_fill('order_1', 450.25, 10, 0.50, 'SPY')
        logger.log_order_rejection('order_2', 'Insufficient funds', 'QQQ', 5)
        logger.log_position_update('AAPL', 'OPEN', 20, 175.50)

        # Read CSV
        today = datetime.now().strftime('%Y-%m-%d')
        csv_file = log_dir / f'trades_{today}.csv'

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Verify 4 entries with consistent format
        assert len(rows) == 4

        # Verify all rows have 9 columns
        for row in rows:
            assert len(row) == 9

        # Verify timestamps exist and are valid
        for row in rows:
            timestamp_str = row['timestamp']
            datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
