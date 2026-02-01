"""
Crypto STRAT Signal Daemon - Session CRYPTO-3

Main daemon orchestrating signal automation for crypto perpetual futures:
- Pattern scanning via CryptoSignalScanner (every 15 minutes)
- Entry trigger monitoring via CryptoEntryMonitor (every 60 seconds)
- Paper trading execution via PaperTrader (when triggers fire)

Key Differences from Equities Daemon:
- 24/7 operation (no market hours filter)
- Friday maintenance window handling
- Simpler architecture (no APScheduler, threading-based)
- No options, direct futures execution

Coordinators (Phase 6.4 - EQUITY-94):
- CryptoHealthMonitor: Health checks, status reporting
- CryptoEntryValidator: Stale setup detection, TFC re-evaluation
- CryptoStatArbExecutor: StatArb signal checking and execution
- CryptoFilterManager: Signal quality filtering, deduplication
- CryptoAlertManager: Discord alerting

Usage:
    from crypto.scanning.daemon import CryptoSignalDaemon

    daemon = CryptoSignalDaemon()
    daemon.start()  # Blocks until shutdown
    # Or:
    daemon.start(block=False)  # Returns immediately
    ...
    daemon.stop()
"""

import logging
import signal as os_signal
import sys
import threading
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

try:
    import pytz
    ET_TIMEZONE = pytz.timezone("America/New_York")
except ImportError:
    ET_TIMEZONE = None

from crypto import config
from crypto.config import (
    get_current_leverage_tier,
    get_max_leverage_for_symbol,
    time_until_intraday_close_et,
)
from crypto.exchange.coinbase_client import CoinbaseClient
from crypto.scanning.coordinators.alert_manager import CryptoAlertManager
from crypto.scanning.coordinators.entry_validator import CryptoEntryValidator
from crypto.scanning.coordinators.filter_manager import CryptoFilterManager
from crypto.scanning.coordinators.health_monitor import (
    CryptoDaemonStats,
    CryptoHealthMonitor,
)
from crypto.scanning.coordinators.statarb_executor import CryptoStatArbExecutor
from crypto.scanning.entry_monitor import (
    CryptoEntryMonitor,
    CryptoEntryMonitorConfig,
    CryptoTriggerEvent,
)
from crypto.scanning.models import CryptoDetectedSignal
from crypto.scanning.signal_scanner import CryptoSignalScanner
from crypto.simulation.paper_trader import PaperTrader
from crypto.simulation.position_monitor import CryptoPositionMonitor
from crypto.trading.sizing import (
    calculate_position_size,
    calculate_position_size_leverage_first,
    should_skip_trade,
)
from crypto.trading.fees import analyze_fee_impact

logger = logging.getLogger(__name__)


@dataclass
class CryptoDaemonConfig:
    """Configuration for crypto signal daemon."""

    # Symbols to scan
    symbols: List[str] = field(default_factory=lambda: config.CRYPTO_SYMBOLS)

    # Scan interval in seconds (default 15 minutes)
    scan_interval: int = config.SCAN_INTERVAL_SECONDS

    # Entry monitor poll interval
    entry_poll_interval: int = config.ENTRY_MONITOR_POLL_SECONDS

    # Signal expiry in hours
    signal_expiry_hours: int = config.SIGNAL_EXPIRY_HOURS

    # Minimum signal quality filters
    min_magnitude_pct: float = config.MIN_MAGNITUDE_PCT
    min_risk_reward: float = config.MIN_SIGNAL_RISK_REWARD

    # Enable paper trading execution
    enable_execution: bool = True

    # Paper trading initial balance
    paper_balance: float = config.DEFAULT_PAPER_BALANCE

    # Enable maintenance window pause
    maintenance_window_enabled: bool = config.MAINTENANCE_WINDOW_ENABLED

    # Health check interval in seconds
    health_check_interval: int = 300  # 5 minutes

    # Callback when signal triggered (for custom handling)
    on_trigger: Optional[Callable[[CryptoTriggerEvent], None]] = None

    # Discord webhook URL for alerts (Session CRYPTO-5)
    discord_webhook_url: Optional[str] = None

    # Discord alert types (Session CRYPTO-7)
    alert_on_signal_detection: bool = False  # Pattern detection (noisy)
    alert_on_trigger: bool = False  # When SETUP price hit (but trade may not execute)
    alert_on_trade_entry: bool = True  # When trade actually executes
    alert_on_trade_exit: bool = True  # When trade closes with P&L

    # REST API configuration (Session CRYPTO-6)
    api_enabled: bool = True
    api_host: str = '0.0.0.0'
    api_port: int = 8080

    # TFC Re-evaluation at Entry (Session EQUITY-67 port from EQUITY-49)
    tfc_reeval_enabled: bool = True
    tfc_reeval_min_strength: int = 3  # Block entry if TFC drops below this
    tfc_reeval_block_on_flip: bool = True  # Block if TFC direction flipped
    tfc_reeval_log_always: bool = True  # Log comparison even when not blocking

    # StatArb Integration (Session EQUITY-92)
    statarb_enabled: bool = False  # Disabled by default - enable explicitly
    statarb_pairs: List[tuple] = field(default_factory=list)  # e.g., [("ADA-USD", "XRP-USD")]
    statarb_config: Optional[Any] = None  # StatArbConfig instance

    # Spot/Derivative Architecture (Session EQUITY-99)
    # Use spot data for cleaner price action in signal detection
    # Execute trades on derivatives for actual trading
    use_spot_for_signals: bool = config.USE_SPOT_FOR_SIGNALS
    use_spot_for_triggers: bool = config.USE_SPOT_FOR_TRIGGERS


class CryptoSignalDaemon:
    """
    Main daemon for autonomous crypto STRAT signal detection and execution.

    Orchestrates:
    - CryptoSignalScanner for pattern detection (15-min intervals)
    - CryptoEntryMonitor for trigger polling (1-min intervals)
    - PaperTrader for simulated execution
    - Coordinators for health, validation, filtering, alerting, StatArb

    24/7 operation with Friday maintenance window handling.

    Usage:
        daemon = CryptoSignalDaemon()
        daemon.start()  # Blocks until Ctrl+C

        # Or non-blocking:
        daemon.start(block=False)
        ...
        daemon.stop()

        # Manual operations:
        signals = daemon.run_scan()  # Single scan
        daemon.run_scan_and_monitor()  # Scan + add to monitor
    """

    def __init__(
        self,
        config: Optional[CryptoDaemonConfig] = None,
        client: Optional[CoinbaseClient] = None,
        scanner: Optional[CryptoSignalScanner] = None,
        paper_trader: Optional[PaperTrader] = None,
    ):
        """
        Initialize crypto signal daemon.

        Args:
            config: Daemon configuration
            client: CoinbaseClient instance (shared across components)
            scanner: Optional pre-configured scanner
            paper_trader: Optional pre-configured paper trader
        """
        self.config = config or CryptoDaemonConfig()

        # Shared Coinbase client
        self.client = client or CoinbaseClient(simulation_mode=True)

        # Components
        self.scanner = scanner or CryptoSignalScanner(client=self.client)
        self.paper_trader = paper_trader
        self.position_monitor: Optional[CryptoPositionMonitor] = None
        self.entry_monitor: Optional[CryptoEntryMonitor] = None

        # Daemon state
        self._running = False
        self._shutdown_event = threading.Event()
        self._scan_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None

        # Statistics
        self._start_time: Optional[datetime] = None
        self._scan_count = 0
        self._signal_count = 0
        self._trigger_count = 0
        self._execution_count = 0
        self._error_count = 0

        # Initialize components
        self._setup_paper_trader()
        self._setup_entry_monitor()

        # Initialize coordinators (Phase 6.4 - EQUITY-94)
        self._setup_filter_manager()
        self._setup_alert_manager()
        self._setup_entry_validator()
        self._setup_health_monitor()
        self._setup_statarb_executor()

    # =========================================================================
    # COMPONENT SETUP
    # =========================================================================

    def _setup_paper_trader(self) -> None:
        """Initialize paper trader and position monitor if execution enabled."""
        if not self.config.enable_execution:
            logger.info("Execution disabled - signals will be detected only")
            return

        if self.paper_trader is not None:
            logger.info("Using provided paper trader")
        else:
            self.paper_trader = PaperTrader(
                starting_balance=self.config.paper_balance,
                account_name="crypto_daemon",
            )
            logger.info(
                f"Paper trader initialized (balance: ${self.config.paper_balance:,.2f})"
            )

        # Initialize position monitor (Session CRYPTO-4)
        self.position_monitor = CryptoPositionMonitor(
            client=self.client,
            paper_trader=self.paper_trader,
        )
        logger.info("Position monitor initialized")

    def _setup_entry_monitor(self) -> None:
        """Initialize entry trigger monitor."""
        monitor_config = CryptoEntryMonitorConfig(
            poll_interval=self.config.entry_poll_interval,
            maintenance_window_enabled=self.config.maintenance_window_enabled,
            signal_expiry_hours=self.config.signal_expiry_hours,
            on_trigger=self._on_trigger,
            on_poll=self._on_poll,
        )

        self.entry_monitor = CryptoEntryMonitor(
            client=self.client,
            config=monitor_config,
        )
        logger.info(
            f"Entry monitor initialized (poll: {self.config.entry_poll_interval}s)"
        )

    def _on_poll(self) -> None:
        """Callback on each entry monitor poll cycle (60s position checks)."""
        if self.position_monitor:
            closed_count = self.check_positions()
            if closed_count > 0:
                logger.info(f"Poll position check: closed {closed_count} trade(s)")

    def _setup_filter_manager(self) -> None:
        """Initialize filter manager coordinator."""
        self.filter_manager = CryptoFilterManager(
            min_magnitude_pct=self.config.min_magnitude_pct,
            min_risk_reward=self.config.min_risk_reward,
            signal_expiry_hours=self.config.signal_expiry_hours,
        )

    def _setup_alert_manager(self) -> None:
        """Initialize alert manager coordinator."""
        self.alert_manager = CryptoAlertManager(
            webhook_url=self.config.discord_webhook_url,
            alert_on_signal_detection=self.config.alert_on_signal_detection,
            alert_on_trigger=self.config.alert_on_trigger,
            alert_on_trade_entry=self.config.alert_on_trade_entry,
            alert_on_trade_exit=self.config.alert_on_trade_exit,
        )
        # Backward compatibility: expose alerter for API server access
        self.discord_alerter = self.alert_manager.alerter

    def _setup_entry_validator(self) -> None:
        """Initialize entry validator coordinator."""
        self.entry_validator = CryptoEntryValidator(
            tfc_reeval_enabled=self.config.tfc_reeval_enabled,
            tfc_reeval_min_strength=self.config.tfc_reeval_min_strength,
            tfc_reeval_block_on_flip=self.config.tfc_reeval_block_on_flip,
            tfc_reeval_log_always=self.config.tfc_reeval_log_always,
        )
        self.entry_validator.set_tfc_evaluator(self.scanner)
        self.entry_validator.set_error_callback(self._increment_error_count)

    def _setup_health_monitor(self) -> None:
        """Initialize health monitor coordinator."""
        self.health_monitor = CryptoHealthMonitor(
            get_stats=self._get_daemon_stats,
            get_current_time_et=self._get_current_time_et,
            health_check_interval=self.config.health_check_interval,
        )

    def _setup_statarb_executor(self) -> None:
        """Initialize StatArb executor coordinator if enabled."""
        if not self.config.statarb_enabled:
            self.statarb_executor: Optional[CryptoStatArbExecutor] = None
            logger.debug("StatArb integration disabled")
            return

        self.statarb_executor = CryptoStatArbExecutor(
            client=self.client,
            statarb_pairs=self.config.statarb_pairs,
            statarb_config=self.config.statarb_config,
            paper_balance=self.config.paper_balance,
            paper_trader=self.paper_trader,
            get_current_time_et=self._get_current_time_et,
            on_execution=self._increment_execution_count,
            on_error=self._increment_error_count,
        )
        if self.alert_manager.is_configured:
            self.statarb_executor.set_alerter(
                self.alert_manager.alerter,
                alert_on_entry=self.config.alert_on_trade_entry,
                alert_on_exit=self.config.alert_on_trade_exit,
            )

    def _start_api_server(self) -> None:
        """Start REST API server in background thread."""
        try:
            from crypto.api.server import init_api, run_api

            init_api(self)

            self._api_thread = threading.Thread(
                target=run_api,
                kwargs={
                    'host': self.config.api_host,
                    'port': self.config.api_port,
                },
                daemon=True,
                name="CryptoAPIServer",
            )
            self._api_thread.start()
            logger.info(
                f"REST API server started on {self.config.api_host}:{self.config.api_port}"
            )
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")

    # =========================================================================
    # STATISTICS CALLBACKS
    # =========================================================================

    def _increment_execution_count(self) -> None:
        """Callback for coordinators to increment execution count."""
        self._execution_count += 1

    def _increment_error_count(self) -> None:
        """Callback for coordinators to increment error count."""
        self._error_count += 1

    # =========================================================================
    # TRIGGER HANDLING
    # =========================================================================

    def _on_trigger(self, event: CryptoTriggerEvent) -> None:
        """
        Callback when entry trigger fires.

        Args:
            event: Trigger event with signal and prices
        """
        self._trigger_count += 1
        signal = event.signal
        actual_direction = getattr(event, '_actual_direction', signal.direction)

        logger.info(
            f"TRIGGER FIRED: {signal.symbol} {signal.pattern_type} {actual_direction} "
            f"@ ${event.current_price:,.2f} (trigger: ${event.trigger_price:,.2f})"
        )

        # Stale setup check (delegated to entry_validator)
        is_stale, stale_reason = self.entry_validator.is_setup_stale(signal)
        if is_stale:
            logger.warning(f"STALE SETUP REJECTED: {stale_reason}")
            return

        # TFC re-evaluation (delegated to entry_validator)
        should_block, block_reason = self.entry_validator.reevaluate_tfc_at_entry(signal)
        if should_block:
            logger.warning(f"TFC REEVAL BLOCKED: {signal.symbol} {signal.pattern_type} - {block_reason}")
            return

        # Execute via paper trader if available
        if self.paper_trader is not None:
            try:
                self._execute_trade(event)
            except Exception as e:
                logger.error(f"Execution error: {e}")
                self._error_count += 1

        # Fire custom callback if provided
        if self.config.on_trigger:
            try:
                self.config.on_trigger(event)
            except Exception as e:
                logger.error(f"Custom trigger callback error: {e}")

        # Send Discord trigger alert (delegated to alert_manager)
        self.alert_manager.send_trigger_alert(event)

    def _get_current_time_et(self) -> datetime:
        """Get current time in ET timezone."""
        now_utc = datetime.now(timezone.utc)
        if ET_TIMEZONE is not None:
            return now_utc.astimezone(ET_TIMEZONE)
        from datetime import timedelta
        return now_utc - timedelta(hours=5)

    def _check_fee_profitability(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        available_balance: float,
        leverage: float,
    ) -> tuple:
        """
        Check if trade is profitable after accounting for fees.

        Session EQUITY-99: Reject trades where fees consume too much of target profit.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Target price
            available_balance: Available account balance
            leverage: Position leverage

        Returns:
            (should_skip, reason) - True if trade should be skipped due to fees
        """
        if not config.FEE_PROFITABILITY_FILTER_ENABLED:
            return False, ""

        # Calculate stop and target percentages
        stop_pct = abs(entry_price - stop_price) / entry_price
        target_pct = abs(target_price - entry_price) / entry_price

        if target_pct <= 0:
            return True, "Target percentage is zero or negative"

        try:
            fee_analysis = analyze_fee_impact(
                account_value=available_balance,
                leverage=leverage,
                price=entry_price,
                symbol=symbol,
                stop_percent=stop_pct,
                target_percent=target_pct,
            )

            fee_pct_of_target = fee_analysis.get("fee_as_pct_of_target", 0)

            if fee_pct_of_target > config.MAX_FEE_PCT_OF_TARGET:
                return True, (
                    f"Fees {fee_pct_of_target:.1%} of target profit "
                    f"(max: {config.MAX_FEE_PCT_OF_TARGET:.1%})"
                )

            logger.debug(
                f"Fee check passed: {symbol} fees={fee_pct_of_target:.1%} of target, "
                f"net R:R={fee_analysis.get('net_rr_ratio', 0):.2f}"
            )
            return False, ""

        except Exception as e:
            logger.warning(f"Fee analysis failed for {symbol}: {e}")
            return False, ""  # Allow trade on analysis failure

    def _execute_trade(self, event: CryptoTriggerEvent) -> None:
        """
        Execute trade via paper trader with time-based leverage.

        Uses intraday leverage (10x) during 6PM-4PM ET window,
        swing leverage (4x) during 4PM-6PM ET gap.

        Args:
            event: Trigger event
        """
        if self.paper_trader is None:
            return

        signal = event.signal
        direction = getattr(event, '_actual_direction', signal.direction)
        actual_pattern = getattr(event, '_actual_pattern', signal.pattern_type)

        # Recalculate stop/target when direction flips
        stop_price = signal.stop_price
        target_price = signal.target_price

        if direction != signal.direction and signal.setup_bar_high > 0 and signal.setup_bar_low > 0:
            bar_range = signal.setup_bar_high - signal.setup_bar_low
            if direction == "SHORT":
                stop_price = signal.setup_bar_high
                target_price = event.current_price - bar_range
            else:
                stop_price = signal.setup_bar_low
                target_price = event.current_price + bar_range
            logger.info(
                f"DIRECTION FLIPPED: {signal.direction} -> {direction}, "
                f"recalculated stop=${stop_price:,.2f} target=${target_price:,.2f}"
            )

        # Get current leverage tier based on time
        now_et = self._get_current_time_et()
        tier = get_current_leverage_tier(now_et)
        max_leverage = get_max_leverage_for_symbol(signal.symbol, now_et)

        logger.info(
            f"Leverage tier: {tier} ({max_leverage}x) for {signal.symbol}"
        )

        # Validate trade inputs
        if event.current_price <= 0 or stop_price <= 0:
            logger.warning("SKIPPING TRADE: Invalid entry or stop price")
            return
        if abs(event.current_price - stop_price) == 0:
            logger.warning("SKIPPING TRADE: Stop distance is zero")
            return

        # Leverage constraint check (skip when using leverage-first sizing)
        if not config.LEVERAGE_FIRST_SIZING:
            skip, reason = should_skip_trade(
                account_value=self.paper_trader.account.current_balance,
                risk_percent=config.DEFAULT_RISK_PERCENT,
                entry_price=event.current_price,
                stop_price=stop_price,
                max_leverage=max_leverage,
            )
            if skip:
                logger.warning(f"SKIPPING TRADE: {reason}")
                return

        # Timeframe continuity-driven filtering and sizing
        tfc_passes = getattr(signal.context, "tfc_passes", False)
        risk_multiplier = getattr(signal.context, "risk_multiplier", 1.0) or 0.0
        if not tfc_passes:
            logger.info(
                "Skipping trade due to failing timeframe continuity filter"
            )
            return

        # Fee profitability filter (Session EQUITY-99)
        available_balance = self.paper_trader.get_available_balance()
        skip_fee, fee_reason = self._check_fee_profitability(
            symbol=signal.symbol,
            entry_price=event.current_price,
            stop_price=stop_price,
            target_price=target_price,
            available_balance=available_balance,
            leverage=max_leverage,
        )
        if skip_fee:
            logger.info(f"SKIPPING TRADE (FEES): {signal.symbol} - {fee_reason}")
            return

        # Calculate position size
        # Note: available_balance already fetched above for fee check

        if config.LEVERAGE_FIRST_SIZING:
            position_size, implied_leverage, actual_risk = calculate_position_size_leverage_first(
                account_value=available_balance,
                entry_price=event.current_price,
                stop_price=stop_price,
                leverage=max_leverage,
            )
            logger.info(
                f"LEVERAGE-FIRST SIZING: {max_leverage}x leverage, "
                f"notional=${available_balance * max_leverage:,.2f}, "
                f"actual_risk=${actual_risk:,.2f} ({actual_risk/available_balance*100:.1f}% of account)"
            )
        else:
            position_size, implied_leverage, actual_risk = calculate_position_size(
                account_value=available_balance,
                risk_percent=config.DEFAULT_RISK_PERCENT,
                entry_price=event.current_price,
                stop_price=stop_price,
                max_leverage=max_leverage,
            )
            position_size *= risk_multiplier
            actual_risk *= risk_multiplier

        if position_size <= 0:
            logger.warning("Position size is zero or negative - skipping trade")
            return

        side = "BUY" if direction == "LONG" else "SELL"

        # Log intraday window info
        if tier == "intraday":
            time_remaining = time_until_intraday_close_et(now_et)
            logger.info(
                f"INTRADAY TRADE: {time_remaining.total_seconds()/3600:.1f}h "
                f"until 4PM ET close requirement"
            )

        # Execute via paper trader
        entry_bar_type = "2U" if direction == "LONG" else "2D"

        # Session EQUITY-99: Use execution_symbol if set, otherwise fall back to symbol
        # This allows signal detection on spot data while executing on derivatives
        execution_symbol = getattr(signal, 'execution_symbol', None) or signal.symbol

        try:
            trade = self.paper_trader.open_trade(
                symbol=execution_symbol,
                side=side,
                quantity=position_size,
                entry_price=event.current_price,
                stop_price=stop_price,
                target_price=target_price,
                timeframe=signal.timeframe,
                pattern_type=actual_pattern,
                tfc_score=signal.context.tfc_score,
                risk_multiplier=risk_multiplier,
                leverage=implied_leverage,
                entry_bar_type=entry_bar_type,
                entry_bar_high=signal.setup_bar_high,
                entry_bar_low=signal.setup_bar_low,
                leverage_tier=tier,  # Session EQUITY-99: Track for deadline enforcement
            )

            if trade is None:
                logger.warning(
                    f"Trade rejected: insufficient margin for {execution_symbol}"
                )
                return

            self._execution_count += 1
            logger.info(
                f"TRADE OPENED: {trade.trade_id} {execution_symbol} {side} "
                f"qty={position_size:.6f} @ ${event.current_price:,.2f} "
                f"({implied_leverage:.1f}x leverage, ${actual_risk:.2f} risk)"
            )
            logger.info(
                f"  Stop: ${stop_price:,.2f} | Target: ${target_price:,.2f}"
            )

            # Send Discord entry alert (delegated to alert_manager)
            self.alert_manager.send_entry_alert(
                signal=signal,
                entry_price=event.current_price,
                quantity=position_size,
                leverage=implied_leverage,
                pattern_override=actual_pattern,
                direction_override=direction,
                stop_override=stop_price,
                target_override=target_price,
            )

        except Exception as e:
            logger.warning(f"Failed to open trade for {execution_symbol}: {e}")

    # =========================================================================
    # SCANNING
    # =========================================================================

    def run_scan(self) -> List[CryptoDetectedSignal]:
        """
        Run a single scan across all symbols and timeframes.

        Returns:
            List of new (non-duplicate) signals found
        """
        self._scan_count += 1
        start_time = time_module.time()
        new_signals: List[CryptoDetectedSignal] = []

        logger.info(f"Starting scan #{self._scan_count}...")

        for symbol in self.config.symbols:
            try:
                signals = self.scanner.scan_all_timeframes(symbol)

                for signal in signals:
                    # Apply filters (delegated to filter_manager)
                    if not self.filter_manager.passes_filters(signal):
                        continue

                    # Check for duplicate (delegated to filter_manager)
                    if self.filter_manager.is_duplicate(signal):
                        continue

                    # Store signal (delegated to filter_manager)
                    self.filter_manager.store_signal(signal)

                    new_signals.append(signal)
                    self._signal_count += 1

                    logger.info(
                        f"NEW SIGNAL: {signal.symbol} {signal.pattern_type} "
                        f"{signal.direction} ({signal.timeframe}) "
                        f"[{signal.signal_type}]"
                    )

                    # Send Discord signal alert (delegated to alert_manager)
                    self.alert_manager.send_signal_alert(signal)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                self._error_count += 1

        duration = time_module.time() - start_time
        logger.info(
            f"Scan complete: {len(new_signals)} new signals in {duration:.2f}s"
        )

        return new_signals

    def run_scan_and_monitor(self) -> List[CryptoDetectedSignal]:
        """
        Run scan, execute TRIGGERED patterns, and add SETUP signals to entry monitor.

        Returns:
            List of new signals found
        """
        new_signals = self.run_scan()

        # Execute TRIGGERED patterns immediately
        triggered_count = 0
        executed_symbols = set()

        for signal in new_signals:
            if signal.signal_type == "COMPLETED":
                signal_key = f"{signal.symbol}_{signal.timeframe}"

                if signal_key in executed_symbols:
                    logger.debug(
                        f"Skipping duplicate TRIGGERED: {signal.symbol} "
                        f"{signal.pattern_type} ({signal.timeframe})"
                    )
                    continue

                self._execute_triggered_pattern(signal)
                executed_symbols.add(signal_key)
                triggered_count += 1

        if triggered_count > 0:
            logger.info(f"Executed {triggered_count} TRIGGERED patterns")

        # Add SETUP signals to entry monitor
        if self.entry_monitor is not None:
            setup_count = self.entry_monitor.add_signals(new_signals)
            if setup_count > 0:
                logger.info(f"Added {setup_count} SETUP signals to entry monitor")

        return new_signals

    def _execute_triggered_pattern(self, signal: CryptoDetectedSignal) -> None:
        """
        Execute a TRIGGERED pattern immediately at market price.

        Args:
            signal: TRIGGERED signal (signal_type="COMPLETED")
        """
        if self.paper_trader is None:
            return

        # Get current market price
        try:
            current_price = self.client.get_current_price(signal.symbol)
            if current_price is None or current_price <= 0:
                logger.warning(f"Could not get price for {signal.symbol} - skipping")
                return
        except Exception as e:
            logger.warning(f"Error getting price for {signal.symbol}: {e}")
            return

        direction = signal.direction
        stop_price = signal.stop_price
        target_price = signal.target_price

        # Validate entry is still viable (price hasn't blown past target)
        if direction == "LONG":
            if current_price >= target_price:
                logger.info(
                    f"SKIP TRIGGERED: {signal.symbol} {signal.pattern_type} LONG - "
                    f"price ${current_price:,.2f} already at/past target ${target_price:,.2f}"
                )
                return
        else:  # SHORT
            if current_price <= target_price:
                logger.info(
                    f"SKIP TRIGGERED: {signal.symbol} {signal.pattern_type} SHORT - "
                    f"price ${current_price:,.2f} already at/past target ${target_price:,.2f}"
                )
                return

        # Get current leverage tier
        now_et = self._get_current_time_et()
        tier = get_current_leverage_tier(now_et)
        max_leverage = get_max_leverage_for_symbol(signal.symbol, now_et)

        logger.info(
            f"TRIGGERED PATTERN: {signal.symbol} {signal.pattern_type} {direction} "
            f"({signal.timeframe}) - executing at ${current_price:,.2f}"
        )

        # Validate trade inputs
        if current_price <= 0 or stop_price <= 0:
            logger.warning("SKIPPING TRIGGERED: Invalid entry or stop price")
            return
        if abs(current_price - stop_price) == 0:
            logger.warning("SKIPPING TRIGGERED: Stop distance is zero")
            return

        # Leverage constraint check
        if not config.LEVERAGE_FIRST_SIZING:
            skip, reason = should_skip_trade(
                account_value=self.paper_trader.account.current_balance,
                risk_percent=config.DEFAULT_RISK_PERCENT,
                entry_price=current_price,
                stop_price=stop_price,
                max_leverage=max_leverage,
            )
            if skip:
                logger.warning(f"SKIPPING TRIGGERED: {reason}")
                return

        tfc_passes = getattr(signal.context, "tfc_passes", False)
        risk_multiplier = getattr(signal.context, "risk_multiplier", 1.0) or 0.0
        if not tfc_passes:
            logger.info("Skipping triggered trade: timeframe continuity failed")
            return

        # Calculate position size
        available_balance = self.paper_trader.get_available_balance()

        if config.LEVERAGE_FIRST_SIZING:
            position_size, implied_leverage, actual_risk = calculate_position_size_leverage_first(
                account_value=available_balance,
                entry_price=current_price,
                stop_price=stop_price,
                leverage=max_leverage,
            )
            logger.info(
                f"LEVERAGE-FIRST SIZING (TRIGGERED): {max_leverage}x leverage, "
                f"notional=${available_balance * max_leverage:,.2f}, "
                f"actual_risk=${actual_risk:,.2f} ({actual_risk/available_balance*100:.1f}% of account)"
            )
        else:
            position_size, implied_leverage, actual_risk = calculate_position_size(
                account_value=available_balance,
                risk_percent=config.DEFAULT_RISK_PERCENT,
                entry_price=current_price,
                stop_price=stop_price,
                max_leverage=max_leverage,
            )
            position_size *= risk_multiplier
            actual_risk *= risk_multiplier

        if position_size <= 0:
            logger.warning("Position size is zero or negative - skipping")
            return

        side = "BUY" if direction == "LONG" else "SELL"

        # Session EQUITY-99: Use execution_symbol if set, otherwise fall back to symbol
        execution_symbol = getattr(signal, 'execution_symbol', None) or signal.symbol

        # Execute trade
        try:
            trade = self.paper_trader.open_trade(
                symbol=execution_symbol,
                side=side,
                quantity=position_size,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                timeframe=signal.timeframe,
                pattern_type=signal.pattern_type,
                tfc_score=signal.context.tfc_score,
                risk_multiplier=risk_multiplier,
                leverage=implied_leverage,
            )

            if trade is None:
                logger.warning(
                    f"Trade rejected: insufficient margin for {execution_symbol}"
                )
                return

            self._execution_count += 1
            logger.info(
                f"TRADE OPENED (TRIGGERED): {trade.trade_id} {execution_symbol} {side} "
                f"qty={position_size:.6f} @ ${current_price:,.2f} "
                f"({implied_leverage:.1f}x leverage, ${actual_risk:.2f} risk)"
            )
            logger.info(
                f"  Pattern: {signal.pattern_type} | Stop: ${stop_price:,.2f} | "
                f"Target: ${target_price:,.2f}"
            )

            # Send Discord alert (delegated to alert_manager)
            self.alert_manager.send_entry_alert(
                signal=signal,
                entry_price=current_price,
                quantity=position_size,
                leverage=implied_leverage,
            )

        except Exception as e:
            logger.error(f"Failed to execute triggered pattern: {e}")

    # =========================================================================
    # MAINTENANCE WINDOW
    # =========================================================================

    def is_maintenance_window(self) -> bool:
        """Check if currently in maintenance window."""
        if not self.config.maintenance_window_enabled:
            return False

        now = datetime.now(timezone.utc)

        # Friday (weekday 4)
        if now.weekday() != config.MAINTENANCE_DAY:
            return False

        # 22:00-23:00 UTC
        return (
            config.MAINTENANCE_START_HOUR_UTC
            <= now.hour
            < config.MAINTENANCE_END_HOUR_UTC
        )

    # =========================================================================
    # BACKGROUND LOOPS
    # =========================================================================

    def _scan_loop(self) -> None:
        """Background scan loop - runs every scan_interval."""
        logger.info("Scan loop started")

        while not self._shutdown_event.is_set():
            try:
                if self.is_maintenance_window():
                    logger.info("Maintenance window - skipping scan")
                else:
                    # STRAT pattern scanning (primary strategy)
                    self.run_scan_and_monitor()

                    # StatArb signal checking (delegated to statarb_executor)
                    if self.statarb_executor is not None:
                        self.statarb_executor.check_and_execute()

                # Clean up expired signals (delegated to filter_manager)
                self.filter_manager.cleanup_expired_signals()

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                self._error_count += 1

            self._shutdown_event.wait(timeout=self.config.scan_interval)

        logger.info("Scan loop stopped")

    # =========================================================================
    # DAEMON CONTROL
    # =========================================================================

    def _setup_signal_handlers(self) -> None:
        """Setup OS signal handlers for graceful shutdown."""

        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()

        os_signal.signal(os_signal.SIGINT, handle_shutdown)
        os_signal.signal(os_signal.SIGTERM, handle_shutdown)

        if sys.platform == "win32":
            try:
                os_signal.signal(os_signal.SIGBREAK, handle_shutdown)
            except AttributeError:
                pass

    def start(self, block: bool = True) -> None:
        """
        Start the daemon.

        Args:
            block: Block until shutdown signal (default: True)
        """
        if self._running:
            logger.warning("Daemon already running")
            return

        logger.info("Starting crypto signal daemon (24/7 mode)...")
        self._start_time = datetime.now(timezone.utc)
        self._running = True
        self._shutdown_event.clear()

        self._setup_signal_handlers()

        # Start entry monitor
        if self.entry_monitor is not None:
            self.entry_monitor.start()
            logger.info("Entry monitor started")

        # Start scan thread
        self._scan_thread = threading.Thread(
            target=self._scan_loop, daemon=True, name="CryptoScanLoop"
        )
        self._scan_thread.start()
        logger.info(f"Scan loop started (interval: {self.config.scan_interval}s)")

        # Start health check thread (delegated to health_monitor)
        self._health_thread = threading.Thread(
            target=self.health_monitor.run_health_loop,
            args=(self._shutdown_event,),
            daemon=True,
            name="CryptoHealthLoop",
        )
        self._health_thread.start()
        logger.info("Health check loop started")

        # Start REST API server
        if self.config.api_enabled:
            self._start_api_server()

        logger.info("Crypto signal daemon started successfully")

        if block:
            self._run_blocking()

    def _run_blocking(self) -> None:
        """Block until shutdown signal."""
        logger.info("Daemon running (Ctrl+C to stop)")

        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")

        self.stop()

    def stop(self) -> None:
        """Stop the daemon gracefully."""
        if not self._running:
            return

        logger.info("Stopping crypto signal daemon...")

        self._shutdown_event.set()

        if self.entry_monitor is not None:
            self.entry_monitor.stop()

        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=5)
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)

        logger.info(f"Final stats: {self.get_status()}")

        self._running = False
        logger.info("Crypto signal daemon stopped")

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    # =========================================================================
    # STATUS AND STATISTICS
    # =========================================================================

    def _get_daemon_stats(self) -> CryptoDaemonStats:
        """Collect daemon statistics for health monitor."""
        entry_stats = {}
        if self.entry_monitor:
            entry_stats = self.entry_monitor.get_stats()

        paper_stats = {}
        if self.paper_trader:
            paper_stats = self.paper_trader.get_account_summary()

        statarb_stats = {}
        if self.statarb_executor is not None and self.statarb_executor.generator is not None:
            statarb_stats = self.statarb_executor.get_status()

        strat_active = []
        if self.statarb_executor is not None:
            strat_active = list(self.statarb_executor._strat_active_symbols)

        return CryptoDaemonStats(
            running=self._running,
            start_time=self._start_time,
            scan_count=self._scan_count,
            signal_count=self._signal_count,
            trigger_count=self._trigger_count,
            execution_count=self._execution_count,
            error_count=self._error_count,
            signals_in_store=self.filter_manager.signals_in_store,
            maintenance_window=self.is_maintenance_window(),
            symbols=list(self.config.symbols),
            scan_interval=self.config.scan_interval,
            entry_stats=entry_stats,
            paper_stats=paper_stats,
            statarb_stats=statarb_stats,
            strat_active_symbols=strat_active,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status and statistics (delegated to health_monitor)."""
        return self.health_monitor.get_status()

    def get_detected_signals(self) -> List[CryptoDetectedSignal]:
        """Get all detected signals currently in store."""
        return self.filter_manager.get_detected_signals()

    def get_pending_setups(self) -> List[CryptoDetectedSignal]:
        """Get SETUP signals waiting for trigger."""
        if self.entry_monitor is None:
            return []
        return self.entry_monitor.get_pending_signals()

    # =========================================================================
    # POSITION MONITORING
    # =========================================================================

    def check_positions(self) -> int:
        """
        Check open positions for stop/target exits and execute if triggered.

        Returns:
            Number of positions closed
        """
        if self.position_monitor is None:
            logger.debug("Position monitor not initialized")
            return 0

        exit_signals = self.position_monitor.check_exits()

        if not exit_signals:
            return 0

        closed = self.position_monitor.execute_all_exits(exit_signals)

        # Send Discord exit alerts (delegated to alert_manager)
        if closed:
            self.alert_manager.send_exit_alerts(closed)

        return len(closed)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions with current P&L."""
        if self.position_monitor is None:
            return []
        return self.position_monitor.get_open_positions_with_pnl()

    def print_status(self) -> None:
        """Print current daemon status (delegated to health_monitor)."""
        self.health_monitor.print_status()
