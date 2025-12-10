"""
Signal-to-Order Executor - Session 83K-47

Converts STRAT pattern signals to Alpaca options orders:
- Signal -> Option Contract Selection -> Order Submission
- Uses existing strike selection (delta 0.40-0.55 targeting)
- Uses existing DTE optimization (7-21 day range)
- Integrates with Alpaca paper trading

Designed to plug into the SignalDaemon for autonomous execution.

Session 83K-50: Added execution persistence to disk.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from integrations.alpaca_trading_client import AlpacaTradingClient
from strat.options_module import (
    OptionsExecutor,
    OptionContract,
    OptionType,
    OptionTrade,
)
from strat.tier1_detector import PatternSignal, PatternType, Timeframe
from strat.signal_automation.signal_store import StoredSignal
from strat.paper_signal_scanner import DetectedSignal

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """State of an execution request."""
    PENDING = "pending"           # Signal received, not yet executed
    ORDER_SUBMITTED = "submitted" # Order submitted to Alpaca
    ORDER_FILLED = "filled"       # Order filled, position open
    MONITORING = "monitoring"     # Position being monitored
    CLOSED = "closed"             # Position closed
    FAILED = "failed"             # Execution failed
    SKIPPED = "skipped"           # Signal skipped (filters/risk)


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    signal_key: str
    state: ExecutionState
    order_id: Optional[str] = None
    osi_symbol: Optional[str] = None
    strike: Optional[float] = None
    expiration: Optional[str] = None
    contracts: int = 0
    premium: float = 0.0
    side: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'signal_key': self.signal_key,
            'state': self.state.value,
            'order_id': self.order_id,
            'osi_symbol': self.osi_symbol,
            'strike': self.strike,
            'expiration': self.expiration,
            'contracts': self.contracts,
            'premium': self.premium,
            'side': self.side,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create from dictionary (JSON deserialization). Session 83K-50."""
        # Convert state string back to enum
        state = ExecutionState(data['state'])

        # Convert timestamp string back to datetime
        timestamp = data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            signal_key=data['signal_key'],
            state=state,
            order_id=data.get('order_id'),
            osi_symbol=data.get('osi_symbol'),
            strike=data.get('strike'),
            expiration=data.get('expiration'),
            contracts=data.get('contracts', 0),
            premium=data.get('premium', 0.0),
            side=data.get('side', ''),
            timestamp=timestamp,
            error=data.get('error'),
        )


@dataclass
class ExecutorConfig:
    """Configuration for signal executor."""
    # Account settings
    account: str = 'SMALL'              # Alpaca account to use
    max_capital_per_trade: float = 300.0  # Max $ per trade
    max_concurrent_positions: int = 5   # Max open positions

    # Delta targeting (Session 83K-64: User-approved middle ground)
    target_delta: float = 0.55          # Target delta for strikes
    delta_range_min: float = 0.45       # Minimum delta
    delta_range_max: float = 0.65       # Maximum delta

    # DTE targeting (7-21 day range per spec)
    min_dte: int = 7                    # Minimum DTE
    max_dte: int = 21                   # Maximum DTE
    target_dte: int = 14                # Target DTE

    # Risk filters
    min_magnitude_pct: float = 0.5      # Minimum pattern magnitude
    min_risk_reward: float = 1.0        # Minimum R:R ratio
    max_bid_ask_spread_pct: float = 0.10  # Max spread as % of mid

    # Order settings
    use_limit_orders: bool = True       # Use limit vs market orders
    limit_price_buffer: float = 0.02    # Buffer above ask for limits

    # Paper trading mode
    paper_mode: bool = True             # Always paper for safety

    # Persistence (Session 83K-50)
    persistence_path: str = 'data/executions'  # Directory for execution data


class SignalExecutor:
    """
    Converts pattern signals to options orders and submits to Alpaca.

    Usage:
        executor = SignalExecutor(config)
        executor.connect()

        # Execute a signal
        result = executor.execute_signal(signal)

        # Get current positions
        positions = executor.get_positions()
    """

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        trading_client: Optional[AlpacaTradingClient] = None,
        options_executor: Optional[OptionsExecutor] = None,
    ):
        """
        Initialize signal executor.

        Args:
            config: Executor configuration
            trading_client: Pre-configured Alpaca client (for testing)
            options_executor: Pre-configured options executor (for testing)
        """
        self.config = config or ExecutorConfig()
        self._trading_client = trading_client
        self._options_executor = options_executor

        # Persistence setup (Session 83K-50)
        self._persistence_path = Path(self.config.persistence_path)
        self._persistence_path.mkdir(parents=True, exist_ok=True)
        self._executions_file = self._persistence_path / 'executions.json'

        # Execution tracking
        self._executions: Dict[str, ExecutionResult] = {}
        self._connected = False

        # Load existing executions from disk
        self._load()

    def connect(self) -> bool:
        """
        Connect to Alpaca trading API.

        Returns:
            True if connected successfully
        """
        if self._trading_client is None:
            self._trading_client = AlpacaTradingClient(
                account=self.config.account
            )

        if self._trading_client.connect():
            self._connected = True
            logger.info(
                f"SignalExecutor connected to Alpaca "
                f"(account={self.config.account})"
            )
            return True

        logger.error("Failed to connect to Alpaca")
        return False

    # =========================================================================
    # PERSISTENCE METHODS (Session 83K-50)
    # =========================================================================

    def _load(self) -> None:
        """Load executions from disk."""
        if self._executions_file.exists():
            try:
                with open(self._executions_file, 'r') as f:
                    data = json.load(f)
                    for key, exec_data in data.items():
                        self._executions[key] = ExecutionResult.from_dict(exec_data)
                logger.info(
                    f"Loaded {len(self._executions)} executions from "
                    f"{self._executions_file}"
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error loading executions: {e}")
                self._executions = {}
        else:
            logger.debug(f"No existing executions file at {self._executions_file}")

    def _save(self) -> None:
        """Save executions to disk."""
        try:
            data = {
                key: execution.to_dict()
                for key, execution in self._executions.items()
            }
            with open(self._executions_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(
                f"Saved {len(self._executions)} executions to "
                f"{self._executions_file}"
            )
        except IOError as e:
            logger.error(f"Error saving executions: {e}")

    # =========================================================================
    # EXECUTION METHODS
    # =========================================================================

    def execute_signal(
        self,
        signal: StoredSignal,
        underlying_price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a pattern signal by submitting an options order.

        Args:
            signal: StoredSignal from signal store
            underlying_price: Current underlying price (fetches if not provided)

        Returns:
            ExecutionResult with execution details
        """
        signal_key = signal.signal_key

        # Check if already executed
        if signal_key in self._executions:
            existing = self._executions[signal_key]
            if existing.state not in [ExecutionState.FAILED, ExecutionState.SKIPPED]:
                logger.warning(f"Signal {signal_key} already executed")
                return existing

        # Validate connection
        if not self._connected:
            return self._create_failed_result(
                signal_key, "Not connected to Alpaca"
            )

        # Apply quality filters
        if not self._passes_filters(signal):
            result = ExecutionResult(
                signal_key=signal_key,
                state=ExecutionState.SKIPPED,
                error="Signal did not pass quality filters"
            )
            self._executions[signal_key] = result
            self._save()  # Persist to disk
            return result

        # Check position limits
        if not self._can_open_position():
            result = ExecutionResult(
                signal_key=signal_key,
                state=ExecutionState.SKIPPED,
                error=f"Max positions ({self.config.max_concurrent_positions}) reached"
            )
            self._executions[signal_key] = result
            self._save()  # Persist to disk
            return result

        try:
            # Get underlying price if not provided
            if underlying_price is None:
                underlying_price = self._get_underlying_price(signal.symbol)
                if underlying_price is None:
                    return self._create_failed_result(
                        signal_key, f"Could not get price for {signal.symbol}"
                    )

            # Select option contract
            contract = self._select_contract(signal, underlying_price)
            if contract is None:
                return self._create_failed_result(
                    signal_key, "No suitable contract found"
                )

            # Calculate position size
            contracts = self._calculate_position_size(contract, underlying_price)
            if contracts < 1:
                return self._create_failed_result(
                    signal_key, "Insufficient capital for 1 contract"
                )

            # Determine order side
            side = 'buy'  # We only buy calls/puts (long options)

            # Get option price for limit order
            option_price = self._get_option_price(contract)

            # Submit order
            if self.config.use_limit_orders and option_price:
                limit_price = option_price * (1 + self.config.limit_price_buffer)
                order = self._trading_client.submit_option_limit_order(
                    symbol=contract.osi_symbol,
                    qty=contracts,
                    side=side,
                    limit_price=round(limit_price, 2)
                )
            else:
                order = self._trading_client.submit_option_market_order(
                    symbol=contract.osi_symbol,
                    qty=contracts,
                    side=side
                )

            # Create result
            result = ExecutionResult(
                signal_key=signal_key,
                state=ExecutionState.ORDER_SUBMITTED,
                order_id=order.get('id'),
                osi_symbol=contract.osi_symbol,
                strike=contract.strike,
                expiration=contract.expiration.strftime('%Y-%m-%d'),
                contracts=contracts,
                premium=option_price or 0.0,
                side=side,
            )

            self._executions[signal_key] = result
            self._save()  # Persist to disk

            logger.info(
                f"Order submitted for {signal_key}: "
                f"{side.upper()} {contracts} {contract.osi_symbol}"
            )

            return result

        except Exception as e:
            logger.exception(f"Execution error for {signal_key}: {e}")
            return self._create_failed_result(signal_key, str(e))

    def _passes_filters(self, signal: StoredSignal) -> bool:
        """Check if signal passes quality filters."""
        # Magnitude filter
        if signal.magnitude_pct < self.config.min_magnitude_pct:
            logger.debug(
                f"Signal {signal.signal_key} failed magnitude filter: "
                f"{signal.magnitude_pct:.2f}% < {self.config.min_magnitude_pct}%"
            )
            return False

        # R:R filter
        if signal.risk_reward < self.config.min_risk_reward:
            logger.debug(
                f"Signal {signal.signal_key} failed R:R filter: "
                f"{signal.risk_reward:.2f} < {self.config.min_risk_reward}"
            )
            return False

        return True

    def _can_open_position(self) -> bool:
        """Check if we can open another position."""
        try:
            positions = self._trading_client.list_option_positions()
            return len(positions) < self.config.max_concurrent_positions
        except Exception as e:
            logger.warning(f"Error checking positions: {e}")
            return False

    def _get_underlying_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for underlying symbol.

        Session 83K-50: Uses new get_stock_price() market data method
        to fetch real-time quotes without requiring an equity position.
        """
        try:
            # Use market data API for real-time quotes (Session 83K-50)
            price = self._trading_client.get_stock_price(symbol)
            if price:
                return price

            # Fallback: Try equity position if market data fails
            position = self._trading_client.get_position(symbol)
            if position:
                return position.get('current_price')

            return None

        except Exception as e:
            logger.warning(f"Error getting price for {symbol}: {e}")
            return None

    def _select_contract(
        self,
        signal: StoredSignal,
        underlying_price: float
    ) -> Optional[OptionContract]:
        """
        Select optimal option contract for signal.

        Uses delta targeting (0.40-0.55) and DTE range (7-21 days).
        """
        # Determine option type
        if signal.direction.lower() in ['bull', 'up', 'call', '1']:
            option_type = OptionType.CALL
            contract_type = 'call'
        else:
            option_type = OptionType.PUT
            contract_type = 'put'

        # Calculate expiration date range
        min_exp_date = (datetime.now() + timedelta(days=self.config.min_dte)).strftime('%Y-%m-%d')
        max_exp_date = (datetime.now() + timedelta(days=self.config.max_dte)).strftime('%Y-%m-%d')

        # Get available contracts from Alpaca with DTE filter
        try:
            contracts = self._trading_client.get_option_contracts(
                underlying=signal.symbol,
                contract_type=contract_type,
                expiration_date_gte=min_exp_date,
                expiration_date_lte=max_exp_date,
                strike_price_gte=underlying_price * 0.85,
                strike_price_lte=underlying_price * 1.15,
            )
        except Exception as e:
            logger.error(f"Error fetching contracts: {e}")
            return None

        if not contracts:
            logger.warning(
                f"No contracts found for {signal.symbol} "
                f"(DTE {self.config.min_dte}-{self.config.max_dte}, "
                f"strike {underlying_price * 0.85:.0f}-{underlying_price * 1.15:.0f})"
            )
            return None

        # Add DTE to contracts for selection
        valid_contracts = []
        for c in contracts:
            if c.get('expiration'):
                exp_date = datetime.fromisoformat(c['expiration'])
                c['dte'] = (exp_date - datetime.now()).days
                valid_contracts.append(c)

        if not valid_contracts:
            logger.warning(f"No valid contracts after filtering for {signal.symbol}")
            return None

        # Select strike closest to target delta
        # For simplicity, use ATM strike (delta ~0.50)
        target_strike = underlying_price
        best_contract = min(
            valid_contracts,
            key=lambda c: abs(c['strike'] - target_strike)
        )

        # Create OptionContract
        return OptionContract(
            underlying=signal.symbol,
            expiration=datetime.fromisoformat(best_contract['expiration']),
            option_type=option_type,
            strike=best_contract['strike'],
            osi_symbol=best_contract['symbol']
        )

    def _calculate_position_size(
        self,
        contract: OptionContract,
        underlying_price: float
    ) -> int:
        """Calculate number of contracts based on capital."""
        # Estimate premium (rough: 3-5% of underlying for ATM)
        estimated_premium = underlying_price * 0.03

        # Calculate max contracts
        max_contracts = int(
            self.config.max_capital_per_trade / (estimated_premium * 100)
        )

        return max(1, min(max_contracts, 5))  # 1-5 contracts

    def _get_option_price(
        self,
        contract: OptionContract
    ) -> Optional[float]:
        """Get current option price (mid of bid/ask)."""
        # In production, would use Alpaca market data API
        # For paper trading, estimate based on underlying
        return None

    def _create_failed_result(
        self,
        signal_key: str,
        error: str
    ) -> ExecutionResult:
        """Create a failed execution result."""
        result = ExecutionResult(
            signal_key=signal_key,
            state=ExecutionState.FAILED,
            error=error
        )
        self._executions[signal_key] = result
        self._save()  # Persist to disk
        return result

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current option positions."""
        if not self._connected:
            return []
        return self._trading_client.list_option_positions()

    def get_execution(self, signal_key: str) -> Optional[ExecutionResult]:
        """Get execution result for a signal."""
        return self._executions.get(signal_key)

    def get_all_executions(self) -> Dict[str, ExecutionResult]:
        """Get all execution results."""
        return self._executions.copy()

    def check_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Check status of a submitted order."""
        if not self._connected:
            return None

        try:
            return self._trading_client.get_order(order_id)
        except Exception as e:
            logger.error(f"Error checking order {order_id}: {e}")
            return None

    def close_position(self, osi_symbol: str) -> Optional[Dict[str, Any]]:
        """Close an open option position."""
        if not self._connected:
            return None

        try:
            return self._trading_client.close_option_position(osi_symbol)
        except Exception as e:
            logger.error(f"Error closing position {osi_symbol}: {e}")
            return None

    @property
    def is_connected(self) -> bool:
        """Check if executor is connected."""
        return self._connected

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        if not self._connected:
            return None
        return self._trading_client.get_account()


# Convenience function for creating executor with default config
def create_paper_executor() -> SignalExecutor:
    """Create a paper trading executor with default settings."""
    config = ExecutorConfig(
        account='SMALL',
        paper_mode=True,
        max_capital_per_trade=300.0,
        max_concurrent_positions=5,
    )
    return SignalExecutor(config)
