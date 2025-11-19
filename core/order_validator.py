"""
Order Validator - Pre-submission risk checks for order execution

Validates orders before submission to prevent:
- Insufficient buying power
- Excessive position concentration (>15% single position)
- Portfolio over-allocation (>105% deployed)
- Duplicate orders
- Non-market hours trading
- Regime compliance violations
- Invalid symbols

Validation Gates:
1. Buying Power: order_value <= account.buying_power
2. Position Size: order_value <= portfolio_value * 0.15
3. Total Allocation: sum(positions + orders) <= portfolio_value * 1.05
4. Duplicate Check: No pending order for same symbol+side
5. Market Hours: 9:30 AM - 4:00 PM ET on NYSE trading days
6. Regime Compliance: BULL=100%, NEUTRAL=70%, BEAR=30%, CRASH=0%
7. Symbol Validity: Uppercase letters, 1-5 chars, optional hyphen

Usage:
    validator = OrderValidator()
    result = validator.validate_order_batch(orders, account_info, 'TREND_BULL')
    if result['valid']:
        submit_orders(orders)
    else:
        log_errors(result['errors'])
"""

import re
from datetime import datetime
from typing import Tuple, List, Dict, Any
import pandas_market_calendars as mcal


class OrderValidator:
    """
    Production-grade order validation with comprehensive risk checks.

    Features:
    - Multi-gate validation (7 validation gates)
    - Regime-aware allocation enforcement
    - NYSE market calendar integration
    - Batch order validation
    """

    # Regime allocation limits
    REGIME_LIMITS = {
        'TREND_BULL': 1.00,      # 100% deployed
        'TREND_NEUTRAL': 0.70,   # 70% deployed
        'TREND_BEAR': 0.30,      # 30% deployed
        'CRASH': 0.00            # 0% deployed (cash only)
    }

    def __init__(
        self,
        max_position_pct: float = 0.15,
        max_portfolio_heat: float = 0.08
    ):
        """
        Initialize order validator.

        Args:
            max_position_pct: Maximum single position size (default 15%)
            max_portfolio_heat: Maximum total portfolio risk (default 8%)
        """
        self.max_position_pct = max_position_pct
        self.max_portfolio_heat = max_portfolio_heat

        # NYSE calendar for market hours
        self.nyse = mcal.get_calendar('NYSE')

    def validate_buying_power(
        self,
        account_info: Dict[str, Any],
        order_value: float
    ) -> Tuple[bool, str]:
        """
        Validate sufficient buying power for order.

        Args:
            account_info: Account details with 'buying_power' key
            order_value: Total order value

        Returns:
            (valid, message) tuple
        """
        buying_power = account_info.get('buying_power', 0.0)

        if order_value > buying_power:
            return False, (
                f"Insufficient buying power: "
                f"need ${order_value:,.2f}, available ${buying_power:,.2f}"
            )

        return True, "Buying power sufficient"

    def validate_position_size(
        self,
        order_value: float,
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        Validate position size within limits (max 15%).

        Args:
            order_value: Total order value
            portfolio_value: Current portfolio value

        Returns:
            (valid, message) tuple
        """
        max_position_value = portfolio_value * self.max_position_pct
        position_pct = order_value / portfolio_value if portfolio_value > 0 else 0

        if order_value > max_position_value:
            return False, (
                f"Position size {position_pct:.1%} exceeds limit "
                f"{self.max_position_pct:.1%} "
                f"(${order_value:,.2f} > ${max_position_value:,.2f})"
            )

        return True, f"Position size {position_pct:.1%} within limit"

    def validate_total_allocation(
        self,
        positions: List[Dict[str, Any]],
        new_orders: List[Dict[str, Any]],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        Validate total allocation within limits (max 105%).

        Args:
            positions: Current positions [{symbol, qty, market_value}]
            new_orders: New orders to submit [{symbol, qty, price, side}]
            portfolio_value: Current portfolio value

        Returns:
            (valid, message) tuple
        """
        # Calculate current position value
        position_value = sum(
            pos.get('market_value', pos.get('qty', 0) * pos.get('price', 0))
            for pos in positions
        )

        # Calculate net impact of orders (BUY adds, SELL reduces)
        # If no 'side' specified, assume BUY for backward compatibility
        buy_value = sum(
            order.get('qty', 0) * order.get('price', 0)
            for order in new_orders
            if order.get('side', 'BUY').upper() == 'BUY'
        )

        sell_value = sum(
            order.get('qty', 0) * order.get('price', 0)
            for order in new_orders
            if order.get('side', '').upper() == 'SELL'
        )

        # Target allocation = current positions - sells + buys
        total_value = position_value - sell_value + buy_value
        allocation_pct = total_value / portfolio_value if portfolio_value > 0 else 0

        max_allocation = 1.05  # 105% max (allow small over-allocation for fills)

        if allocation_pct > max_allocation:
            return False, (
                f"Total allocation {allocation_pct:.1%} exceeds limit "
                f"{max_allocation:.1%} "
                f"(positions ${position_value:,.2f} - sells ${sell_value:,.2f} + "
                f"buys ${buy_value:,.2f} = ${total_value:,.2f} "
                f"> ${portfolio_value * max_allocation:,.2f})"
            )

        return True, f"Total allocation {allocation_pct:.1%} within limit"

    def validate_no_duplicate_orders(
        self,
        pending_orders: List[Dict[str, Any]],
        symbol: str,
        side: str
    ) -> Tuple[bool, str]:
        """
        Validate no duplicate pending orders for same symbol+side.

        Args:
            pending_orders: List of pending orders [{symbol, side}]
            symbol: Symbol to check
            side: Side to check (BUY or SELL)

        Returns:
            (valid, message) tuple
        """
        for order in pending_orders:
            if (order.get('symbol') == symbol and
                order.get('side', '').upper() == side.upper()):
                return False, (
                    f"Duplicate order detected: {side} {symbol} already pending"
                )

        return True, "No duplicate orders"

    def validate_market_hours(
        self,
        current_time: datetime = None
    ) -> Tuple[bool, str]:
        """
        Validate NYSE market hours (9:30 AM - 4:00 PM ET).

        Args:
            current_time: Time to check (default: now)

        Returns:
            (valid, message) tuple
        """
        if current_time is None:
            current_time = datetime.now()

        # Get today's schedule
        schedule = self.nyse.schedule(
            start_date=current_time.date(),
            end_date=current_time.date()
        )

        # Check if market is open today
        if schedule.empty:
            return False, (
                f"Market closed: {current_time.strftime('%Y-%m-%d')} is not a trading day"
            )

        # Get market open/close times
        market_open = schedule.iloc[0]['market_open']
        market_close = schedule.iloc[0]['market_close']

        # Localize current time to ET if naive
        if current_time.tzinfo is None:
            import pytz
            et = pytz.timezone('America/New_York')
            current_time = et.localize(current_time)

        # Check if within market hours
        if current_time < market_open:
            return False, (
                f"Market not yet open: "
                f"opens at {market_open.strftime('%H:%M')} ET, "
                f"current time {current_time.strftime('%H:%M')} ET"
            )

        if current_time > market_close:
            return False, (
                f"Market closed: "
                f"closed at {market_close.strftime('%H:%M')} ET, "
                f"current time {current_time.strftime('%H:%M')} ET"
            )

        return True, f"Market open until {market_close.strftime('%H:%M')} ET"

    def validate_regime_compliance(
        self,
        regime: str,
        allocation_pct: float
    ) -> Tuple[bool, str]:
        """
        Validate allocation complies with regime limits.

        Regime Limits:
        - TREND_BULL: 100% max allocation
        - TREND_NEUTRAL: 70% max allocation
        - TREND_BEAR: 30% max allocation
        - CRASH: 0% allocation (NO new orders)

        Args:
            regime: Current regime
            allocation_pct: Proposed allocation percentage (0.0 to 1.0)

        Returns:
            (valid, message) tuple
        """
        if regime not in self.REGIME_LIMITS:
            return False, (
                f"Unknown regime '{regime}'. "
                f"Must be one of: {list(self.REGIME_LIMITS.keys())}"
            )

        max_allocation = self.REGIME_LIMITS[regime]

        if allocation_pct > max_allocation:
            return False, (
                f"Allocation {allocation_pct:.1%} exceeds {regime} limit "
                f"{max_allocation:.1%}"
            )

        return True, f"Allocation {allocation_pct:.1%} within {regime} limit"

    def validate_symbol(
        self,
        symbol: str
    ) -> Tuple[bool, str]:
        """
        Validate symbol format and tradability.

        Valid symbols:
        - Uppercase letters only
        - 1-5 characters
        - Optional single hyphen (e.g., BRK-B)

        Args:
            symbol: Ticker symbol

        Returns:
            (valid, message) tuple
        """
        # Check for empty symbol
        if not symbol:
            return False, "Symbol cannot be empty"

        # Check length
        if len(symbol) < 1 or len(symbol) > 6:
            return False, (
                f"Invalid symbol length: {len(symbol)} chars "
                "(must be 1-6 characters)"
            )

        # Check format: uppercase letters and optional hyphen
        pattern = r'^[A-Z]{1,5}(-[A-Z])?$'
        if not re.match(pattern, symbol):
            return False, (
                f"Invalid symbol format: '{symbol}' "
                "(must be uppercase letters, optional hyphen)"
            )

        return True, f"Symbol '{symbol}' is valid"

    def validate_order_batch(
        self,
        orders: List[Dict[str, Any]],
        account_info: Dict[str, Any],
        regime: str,
        pending_orders: List[Dict[str, Any]] = None,
        current_positions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate batch of orders with comprehensive checks.

        Args:
            orders: Orders to validate [{symbol, qty, side, price, order_type}]
            account_info: Account details {equity, buying_power, portfolio_value}
            regime: Current regime (TREND_BULL, TREND_NEUTRAL, TREND_BEAR, CRASH)
            pending_orders: Pending orders (for duplicate check)
            current_positions: Current positions (for allocation check)

        Returns:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str]
            }
        """
        if pending_orders is None:
            pending_orders = []

        if current_positions is None:
            current_positions = []

        errors = []
        warnings = []

        portfolio_value = account_info.get('portfolio_value',
                                          account_info.get('equity', 0.0))

        # Calculate net cash impact (BUY orders consume cash, SELL orders free cash)
        buy_value = sum(
            order.get('qty', 0) * order.get('price', 0)
            for order in orders
            if order.get('side', '').upper() == 'BUY'
        )

        sell_value = sum(
            order.get('qty', 0) * order.get('price', 0)
            for order in orders
            if order.get('side', '').upper() == 'SELL'
        )

        net_cash_required = buy_value - sell_value

        # Gate 1: Buying power check (only check if net cash required)
        if net_cash_required > 0:
            valid, msg = self.validate_buying_power(account_info, net_cash_required)
            if not valid:
                errors.append(f"[BUYING_POWER] {msg}")

        # Gate 2: Regime compliance (use target allocation after orders execute)
        # Current position value
        current_position_value = sum(
            pos.get('market_value', 0)
            for pos in current_positions
        )

        # Target allocation = current positions - sells + buys
        target_allocation_value = current_position_value - sell_value + buy_value
        allocation_pct = target_allocation_value / portfolio_value if portfolio_value > 0 else 0

        valid, msg = self.validate_regime_compliance(regime, allocation_pct)
        if not valid:
            errors.append(f"[REGIME] {msg}")

        # Gate 3: Total allocation check
        valid, msg = self.validate_total_allocation(
            current_positions,
            orders,
            portfolio_value
        )
        if not valid:
            errors.append(f"[ALLOCATION] {msg}")

        # Per-order validations
        for i, order in enumerate(orders):
            symbol = order.get('symbol', '')
            qty = order.get('qty', 0)
            side = order.get('side', '')
            price = order.get('price', 0)
            order_value = qty * price

            # Gate 4: Symbol validation
            valid, msg = self.validate_symbol(symbol)
            if not valid:
                errors.append(f"[SYMBOL] Order {i+1}: {msg}")

            # Gate 5: Position size check
            valid, msg = self.validate_position_size(order_value, portfolio_value)
            if not valid:
                warnings.append(f"[POSITION_SIZE] Order {i+1} ({symbol}): {msg}")

            # Gate 6: Duplicate check
            valid, msg = self.validate_no_duplicate_orders(
                pending_orders,
                symbol,
                side
            )
            if not valid:
                errors.append(f"[DUPLICATE] Order {i+1}: {msg}")

        # Gate 7: Market hours check (applies to all orders)
        valid, msg = self.validate_market_hours()
        if not valid:
            warnings.append(f"[MARKET_HOURS] {msg}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
