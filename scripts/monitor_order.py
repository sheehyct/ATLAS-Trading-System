"""
Order Monitoring Script - Track Order Fill Status

Monitors an Alpaca paper trading order until it reaches a terminal state (FILLED, CANCELLED, REJECTED, etc.).

Usage:
    # Monitor specific order
    uv run python scripts/monitor_order.py --order-id 6c1ce511-0fb0-4941-96d5-e059e5ec3e88

    # With custom poll interval
    uv run python scripts/monitor_order.py --order-id 6c1ce511-0fb0-4941-96d5-e059e5ec3e88 --poll-interval 10

    # Verbose logging
    uv run python scripts/monitor_order.py --order-id 6c1ce511-0fb0-4941-96d5-e059e5ec3e88 --verbose

Features:
- Polls order status at configurable intervals (default: 30 seconds)
- Logs all status changes
- Displays fill details when order completes
- Handles errors and connection issues
- Exits when order reaches terminal state

Terminal States:
- FILLED: Order fully executed
- CANCELLED: Order cancelled before fill
- EXPIRED: Order expired (for limit/stop orders)
- REJECTED: Order rejected by exchange
- REPLACED: Order replaced by modification
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from integrations.alpaca_trading_client import AlpacaTradingClient
from utils.execution_logger import ExecutionLogger


class OrderMonitor:
    """
    Order status monitor for paper trading orders.

    Polls Alpaca API to track order status changes and logs fill details.
    """

    # Terminal states (order complete, stop polling)
    TERMINAL_STATES = {
        'filled', 'cancelled', 'expired', 'rejected', 'replaced'
    }

    # Active states (order still processing, continue polling)
    ACTIVE_STATES = {
        'new', 'accepted', 'pending_new', 'accepted_for_bidding',
        'stopped', 'pending_cancel', 'pending_replace', 'partially_filled'
    }

    def __init__(
        self,
        order_id: str,
        account: str = 'LARGE',
        poll_interval: int = 30,
        verbose: bool = False
    ):
        """
        Initialize order monitor.

        Args:
            order_id: Alpaca order ID to monitor
            account: Alpaca account (LARGE or SMALL)
            poll_interval: Seconds between status checks (default: 30)
            verbose: Enable verbose logging (default: False)
        """
        self.order_id = order_id
        self.account = account
        self.poll_interval = poll_interval
        self.verbose = verbose

        # Initialize logger
        self.logger = ExecutionLogger()
        if verbose:
            self.logger.logger.setLevel('DEBUG')

        # Initialize trading client
        self.trading_client = AlpacaTradingClient(
            account=account,
            logger=self.logger.logger
        )

        # Tracking state
        self.last_status = None
        self.poll_count = 0

        self.logger.logger.info("=" * 60)
        self.logger.logger.info("ORDER MONITORING STARTED")
        self.logger.logger.info("=" * 60)
        self.logger.logger.info(f"Order ID: {order_id}")
        self.logger.logger.info(f"Account: {account}")
        self.logger.logger.info(f"Poll Interval: {poll_interval}s")
        self.logger.logger.info("")

    def run(self) -> Dict[str, Any]:
        """
        Monitor order until terminal state reached.

        Returns:
            Dict with final order details
        """
        # Connect to Alpaca
        if not self.trading_client.connect():
            self.logger.logger.error("Failed to connect to Alpaca")
            return None

        self.logger.logger.info("Connected to Alpaca paper trading")
        self.logger.logger.info("")

        # Initial order status
        try:
            order = self.trading_client.get_order(self.order_id)
            self._log_order_status(order, initial=True)

        except Exception as e:
            self.logger.logger.error(f"Failed to retrieve order: {str(e)}")
            return None

        # Poll until terminal state
        while True:
            # Check if terminal state reached
            if order['status'].lower() in self.TERMINAL_STATES:
                self.logger.logger.info("")
                self.logger.logger.info("=" * 60)
                self.logger.logger.info(f"ORDER COMPLETE: {order['status'].upper()}")
                self.logger.logger.info("=" * 60)
                self._log_fill_details(order)
                return order

            # Wait for next poll
            self.logger.logger.info(
                f"Order status: {order['status']} - "
                f"Waiting {self.poll_interval}s before next check..."
            )
            time.sleep(self.poll_interval)

            # Poll for status update
            self.poll_count += 1
            try:
                order = self.trading_client.get_order(self.order_id)

                # Log if status changed
                if order['status'] != self.last_status:
                    self.logger.logger.info("")
                    self.logger.logger.info(f"[POLL #{self.poll_count}] Status changed: "
                                          f"{self.last_status} -> {order['status']}")
                    self._log_order_status(order)
                    self.last_status = order['status']
                else:
                    if self.verbose:
                        self.logger.logger.debug(
                            f"[POLL #{self.poll_count}] Status unchanged: {order['status']}"
                        )

            except Exception as e:
                self.logger.logger.error(
                    f"[POLL #{self.poll_count}] Failed to retrieve order: {str(e)}"
                )
                self.logger.logger.info(f"Will retry in {self.poll_interval}s...")

    def _log_order_status(self, order: Dict[str, Any], initial: bool = False):
        """Log current order status."""
        prefix = "Initial" if initial else "Current"

        self.logger.logger.info(f"{prefix} Order Status:")
        self.logger.logger.info(f"  Symbol: {order['symbol']}")
        self.logger.logger.info(f"  Side: {order['side']}")
        self.logger.logger.info(f"  Quantity: {order['qty']} shares")
        self.logger.logger.info(f"  Type: {order['type']}")
        self.logger.logger.info(f"  Status: {order['status']}")
        self.logger.logger.info(f"  Time in Force: {order['time_in_force']}")

        if order.get('limit_price'):
            self.logger.logger.info(f"  Limit Price: ${order['limit_price']:.2f}")

        if order.get('filled_qty', 0) > 0:
            self.logger.logger.info(f"  Filled Qty: {order['filled_qty']} shares")

        if order.get('filled_avg_price'):
            self.logger.logger.info(f"  Avg Fill Price: ${order['filled_avg_price']:.2f}")

        if order.get('submitted_at'):
            self.logger.logger.info(f"  Submitted: {order['submitted_at']}")

        if self.verbose and order.get('filled_at'):
            self.logger.logger.info(f"  Filled: {order['filled_at']}")

        self.last_status = order['status']

    def _log_fill_details(self, order: Dict[str, Any]):
        """Log detailed fill information for completed order."""
        status = order['status'].lower()

        if status == 'filled':
            self.logger.logger.info("")
            self.logger.logger.info("FILL DETAILS:")
            self.logger.logger.info(f"  Symbol: {order['symbol']}")
            self.logger.logger.info(f"  Side: {order['side']}")
            self.logger.logger.info(f"  Quantity: {order['filled_qty']} shares")
            self.logger.logger.info(f"  Avg Fill Price: ${order['filled_avg_price']:.2f}")
            self.logger.logger.info(f"  Total Value: ${order['filled_qty'] * order['filled_avg_price']:.2f}")
            self.logger.logger.info(f"  Filled At: {order['filled_at']}")
            self.logger.logger.info(f"  Submitted At: {order['submitted_at']}")

            # Calculate time to fill
            if order.get('submitted_at') and order.get('filled_at'):
                try:
                    submitted = datetime.fromisoformat(order['submitted_at'].replace('Z', '+00:00'))
                    filled = datetime.fromisoformat(order['filled_at'].replace('Z', '+00:00'))
                    time_to_fill = (filled - submitted).total_seconds()
                    self.logger.logger.info(f"  Time to Fill: {time_to_fill:.1f} seconds")
                except Exception:
                    pass

            # Log to CSV audit trail
            self.logger.log_order_fill(
                order_id=order['id'],
                fill_price=order['filled_avg_price'],
                fill_qty=order['filled_qty'],
                commission=0.0,  # Paper trading has no commissions
                symbol=order['symbol']
            )

            self.logger.logger.info("")
            self.logger.logger.info("[SUCCESS] Order filled and logged to audit trail")

        elif status == 'cancelled':
            self.logger.logger.info("")
            self.logger.logger.info("Order was cancelled before filling")
            if order.get('filled_qty', 0) > 0:
                self.logger.logger.info(f"  Partially filled: {order['filled_qty']} of {order['qty']} shares")
                self.logger.logger.info(f"  Avg fill price: ${order['filled_avg_price']:.2f}")

        elif status == 'rejected':
            self.logger.logger.info("")
            self.logger.logger.info("[REJECTED] Order rejected by exchange")
            self.logger.logger.info("  Check order parameters and account status")

        elif status == 'expired':
            self.logger.logger.info("")
            self.logger.logger.info("[EXPIRED] Order expired (limit/stop orders only)")
            if order.get('filled_qty', 0) > 0:
                self.logger.logger.info(f"  Partially filled: {order['filled_qty']} of {order['qty']} shares")

        self.logger.logger.info("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor Alpaca paper trading order status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--order-id',
        required=True,
        help='Alpaca order ID to monitor'
    )

    parser.add_argument(
        '--account',
        default='LARGE',
        choices=['LARGE', 'SMALL'],
        help='Alpaca account (default: LARGE)'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=30,
        help='Seconds between status checks (default: 30)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Create monitor
    monitor = OrderMonitor(
        order_id=args.order_id,
        account=args.account,
        poll_interval=args.poll_interval,
        verbose=args.verbose
    )

    # Run monitoring
    try:
        final_order = monitor.run()

        if final_order:
            # Exit with success
            sys.exit(0)
        else:
            # Exit with error
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
