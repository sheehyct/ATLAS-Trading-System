"""
STRAT Signal Daemon - Session 83K-46/49

Main daemon class that orchestrates signal automation:
- Pattern scanning via PaperSignalScanner
- Signal storage and deduplication
- Alert delivery (Discord + logging)
- Job scheduling via APScheduler
- Options order execution via SignalExecutor (Session 83K-48)
- Position monitoring and auto-exit via PositionMonitor (Session 83K-49)

Designed for autonomous operation with:
- Graceful startup and shutdown
- Health monitoring
- Error recovery
- Signal lifecycle management
- Optional execution mode for paper trading
- Automatic position monitoring with target/stop exits
"""

import signal
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from threading import Event

from strat.signal_automation.config import SignalAutomationConfig, ExecutionConfig
from strat.signal_automation.signal_store import SignalStore, StoredSignal, SignalType, SignalStatus
from strat.signal_automation.scheduler import SignalScheduler
from strat.signal_automation.alerters import (
    BaseAlerter,
    LoggingAlerter,
    DiscordAlerter,
)
from strat.signal_automation.executor import (
    SignalExecutor,
    ExecutorConfig as ExecutorConfigClass,
    ExecutionResult,
    ExecutionState,
)
from strat.signal_automation.position_monitor import (
    PositionMonitor,
    MonitoringConfig as MonitoringConfigClass,
    ExitSignal,
    ExitReason,
    TrackedPosition,
)
from strat.signal_automation.entry_monitor import (
    EntryMonitor,
    EntryMonitorConfig,
    TriggerEvent,
)
from strat.paper_signal_scanner import PaperSignalScanner, DetectedSignal

logger = logging.getLogger(__name__)


class SignalDaemon:
    """
    Main daemon for autonomous STRAT signal detection, alerting, and execution.

    Orchestrates:
    - PaperSignalScanner for pattern detection
    - SignalStore for persistence and deduplication
    - Alerters for notification delivery
    - SignalScheduler for job timing
    - SignalExecutor for options order execution (Session 83K-48)

    Usage:
        daemon = SignalDaemon.from_config(config)
        daemon.start()  # Blocks until shutdown signal

        # Or run a single scan manually:
        signals = daemon.run_scan('1D')

        # Execute signals manually:
        results = daemon.execute_signals(signals)
    """

    def __init__(
        self,
        config: SignalAutomationConfig,
        scanner: Optional[PaperSignalScanner] = None,
        signal_store: Optional[SignalStore] = None,
        executor: Optional[SignalExecutor] = None,
        position_monitor: Optional[PositionMonitor] = None,
    ):
        """
        Initialize signal daemon.

        Args:
            config: Full automation configuration
            scanner: Optional pre-configured scanner
            signal_store: Optional pre-configured signal store
            executor: Optional pre-configured executor (for testing)
            position_monitor: Optional pre-configured monitor (for testing)
        """
        self.config = config
        self._shutdown_event = Event()
        self._is_running = False

        # Initialize components
        self.scanner = scanner or PaperSignalScanner()
        self.signal_store = signal_store or SignalStore(config.store_path)
        self.scheduler = SignalScheduler(config.schedule)
        self.alerters: List[BaseAlerter] = []
        self.executor: Optional[SignalExecutor] = executor
        self.position_monitor: Optional[PositionMonitor] = position_monitor
        self.entry_monitor: Optional[EntryMonitor] = None

        # Daemon state
        self._start_time: Optional[datetime] = None
        self._scan_count = 0
        self._signal_count = 0
        self._error_count = 0
        self._execution_count = 0
        self._exit_count = 0

        # Initialize components
        self._setup_alerters()
        self._setup_executor()
        self._setup_position_monitor()
        self._setup_entry_monitor()

    def _setup_alerters(self) -> None:
        """Initialize configured alerters."""
        # Logging alerter (always enabled)
        if self.config.alerts.logging_enabled:
            logging_alerter = LoggingAlerter(
                log_file=self.config.alerts.log_file,
                level=self.config.alerts.log_level,
                console_output=True,
            )
            logging_alerter.set_throttle_interval(
                self.config.alerts.min_alert_interval_seconds
            )
            self.alerters.append(logging_alerter)
            logger.info("Logging alerter initialized")

        # Discord alerter (if configured)
        if self.config.alerts.discord_enabled:
            try:
                discord_alerter = DiscordAlerter(
                    webhook_url=self.config.alerts.discord_webhook_url,
                )
                discord_alerter.set_throttle_interval(
                    self.config.alerts.min_alert_interval_seconds
                )
                self.alerters.append(discord_alerter)
                logger.info("Discord alerter initialized")
            except ValueError as e:
                logger.error(f"Failed to initialize Discord alerter: {e}")

    def _setup_executor(self) -> None:
        """Initialize executor if execution is enabled (Session 83K-48)."""
        if not self.config.execution.enabled:
            logger.info("Execution disabled - signals will be alerted only")
            return

        if self.executor is not None:
            # Executor was provided (e.g., for testing)
            logger.info("Using provided executor")
            return

        # Create executor from config
        exec_config = ExecutorConfigClass(
            account=self.config.execution.account,
            max_capital_per_trade=self.config.execution.max_capital_per_trade,
            max_concurrent_positions=self.config.execution.max_concurrent_positions,
            target_delta=self.config.execution.target_delta,
            delta_range_min=self.config.execution.delta_range_min,
            delta_range_max=self.config.execution.delta_range_max,
            min_dte=self.config.execution.min_dte,
            max_dte=self.config.execution.max_dte,
            target_dte=self.config.execution.target_dte,
            use_limit_orders=self.config.execution.use_limit_orders,
            limit_price_buffer=self.config.execution.limit_price_buffer,
            paper_mode=True,  # Always paper for safety
        )

        self.executor = SignalExecutor(exec_config)

        # Connect to Alpaca
        if self.executor.connect():
            logger.info(
                f"Executor initialized and connected "
                f"(account={self.config.execution.account})"
            )
        else:
            logger.error("Failed to connect executor to Alpaca - execution disabled")
            self.executor = None

    def _setup_position_monitor(self) -> None:
        """Initialize position monitor if execution is enabled (Session 83K-49)."""
        if not self.config.monitoring.enabled:
            logger.info("Position monitoring disabled")
            return

        if self.executor is None:
            logger.info("Position monitoring requires executor - skipping")
            return

        if self.position_monitor is not None:
            # Monitor was provided (e.g., for testing)
            logger.info("Using provided position monitor")
            return

        # Create monitor config from settings
        monitor_config = MonitoringConfigClass(
            exit_dte=self.config.monitoring.exit_dte,
            max_loss_pct=self.config.monitoring.max_loss_pct,
            max_profit_pct=self.config.monitoring.max_profit_pct,
            check_interval=self.config.monitoring.check_interval,
            alert_on_exit=self.config.monitoring.alert_on_exit,
        )

        # Create position monitor
        self.position_monitor = PositionMonitor(
            config=monitor_config,
            executor=self.executor,
            trading_client=self.executor._trading_client,
            signal_store=self.signal_store,
            on_exit_callback=self._on_position_exit,
        )

        logger.info("Position monitor initialized")

    def _on_position_exit(
        self,
        exit_signal: ExitSignal,
        order_result: Dict[str, Any]
    ) -> None:
        """Callback when a position is exited (Session 83K-49)."""
        self._exit_count += 1

        # Get signal details for the alert
        signal = self.signal_store.get_signal(exit_signal.signal_key) if exit_signal.signal_key else None

        # Send exit alert (Session 83K-77: simplified Discord alerts)
        for alerter in self.alerters:
            try:
                if isinstance(alerter, DiscordAlerter):
                    # Use simplified exit alert
                    reason_str = exit_signal.reason.value if hasattr(exit_signal.reason, 'value') else str(exit_signal.reason)
                    alerter.send_simple_exit_alert(
                        symbol=signal.symbol if signal else exit_signal.osi_symbol[:3],
                        pattern_type=signal.pattern_type if signal else "Unknown",
                        timeframe=signal.timeframe if signal else "Unknown",
                        direction=signal.direction if signal else "CALL",
                        exit_reason=reason_str,
                        pnl=exit_signal.unrealized_pnl,
                    )
                elif isinstance(alerter, LoggingAlerter):
                    alerter.log_position_exit(exit_signal, order_result)
            except Exception as e:
                logger.error(f"Exit alert error ({alerter.name}): {e}")

    def _setup_entry_monitor(self) -> None:
        """Initialize entry trigger monitor (Session 83K-66)."""
        if self.executor is None:
            logger.info("Entry monitoring requires executor - skipping")
            return

        # Create price fetcher using Alpaca client
        def price_fetcher(symbols: List[str]) -> Dict[str, float]:
            """Fetch current prices (mid price) for symbols via Alpaca."""
            try:
                quotes = self.executor._trading_client.get_stock_quotes(symbols)
                # Extract mid price from quote data
                prices = {}
                for symbol, quote in quotes.items():
                    if isinstance(quote, dict) and 'mid' in quote:
                        prices[symbol] = quote['mid']
                    elif isinstance(quote, (int, float)):
                        prices[symbol] = float(quote)
                return prices
            except Exception as e:
                logger.error(f"Price fetch error: {e}")
                return {}

        # Configure entry monitor
        monitor_config = EntryMonitorConfig(
            poll_interval=60,  # 1 minute
            market_hours_only=True,
            on_trigger=self._on_entry_triggered,
        )

        self.entry_monitor = EntryMonitor(
            signal_store=self.signal_store,
            price_fetcher=price_fetcher,
            config=monitor_config,
        )

        logger.info("Entry monitor initialized (1-minute polling during market hours)")

    def _on_entry_triggered(self, event: TriggerEvent) -> None:
        """
        Callback when an entry trigger is hit (Session 83K-66).

        Executes the signal if executor is available and respects position limits.
        Signals are already sorted by priority (1M > 1W > 1D > 1H).
        """
        signal = event.signal

        # Session CRYPTO-11: Use actual direction from entry monitor (bidirectional monitoring)
        # This handles cases where SETUP (X-1-?) broke in opposite direction
        actual_direction = getattr(event, '_actual_direction', signal.direction)
        original_direction = signal.direction

        # Update signal direction if it changed due to opposite break
        if actual_direction != original_direction:
            signal.direction = actual_direction
            logger.info(
                f"DIRECTION CHANGED: {signal.symbol} {signal.pattern_type} "
                f"{original_direction} -> {actual_direction} (opposite break detected)"
            )

        logger.info(
            f"ENTRY TRIGGERED: {signal.symbol} {signal.pattern_type} {signal.direction} "
            f"@ ${event.current_price:.2f} (trigger: ${event.trigger_price:.2f}, "
            f"priority: {signal.priority})"
        )

        # Update signal status
        self.signal_store.mark_triggered(signal.signal_key)

        # Execute if executor available
        if self.executor is not None:
            try:
                # Session 83K-73: Pass StoredSignal directly to executor
                # The executor expects StoredSignal (has signal_key attribute)
                # Previously was incorrectly converting to DetectedSignal
                result = self.executor.execute_signal(signal)
                self._execution_count += 1

                if result.state == ExecutionState.ORDER_SUBMITTED:
                    logger.info(
                        f"ORDER SUBMITTED: {signal.symbol} {signal.direction} "
                        f"(priority: {signal.priority})"
                    )

                    # Send alert (Session 83K-77: simplified Discord alerts)
                    for alerter in self.alerters:
                        try:
                            if isinstance(alerter, DiscordAlerter):
                                alerter.send_entry_alert(signal, result)
                            elif isinstance(alerter, LoggingAlerter):
                                alerter.log_execution(result)
                        except Exception as e:
                            logger.error(f"Alert error: {e}")

            except Exception as e:
                logger.error(f"Execution error for {signal.symbol}: {e}")

    @classmethod
    def from_config(
        cls,
        config: Optional[SignalAutomationConfig] = None
    ) -> 'SignalDaemon':
        """
        Create daemon from configuration.

        Args:
            config: Configuration (uses env-based config if not provided)

        Returns:
            Configured SignalDaemon instance
        """
        if config is None:
            config = SignalAutomationConfig.from_env()

        # Validate configuration
        issues = config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Config issue: {issue}")

        return cls(config)

    def run_scan(self, timeframe: str) -> List[StoredSignal]:
        """
        Run a single scan for a specific timeframe.

        Args:
            timeframe: Timeframe to scan ('1H', '1D', '1W', '1M')

        Returns:
            List of new (non-duplicate) signals found
        """
        start_time = time.time()
        self._scan_count += 1

        # Notify scan started
        for alerter in self.alerters:
            if isinstance(alerter, LoggingAlerter):
                alerter.log_scan_started(timeframe, self.config.scan.symbols)

        try:
            new_signals: List[StoredSignal] = []

            # Scan each symbol
            for symbol in self.config.scan.symbols:
                try:
                    detected = self._scan_symbol(symbol, timeframe)
                    new_signals.extend(detected)
                except Exception as e:
                    logger.error(f"Error scanning {symbol} {timeframe}: {e}")
                    self._error_count += 1

            # Send alerts for new signals
            if new_signals:
                self._send_alerts(new_signals)

                # Execute signals if execution enabled (Session 83K-48)
                if self.executor is not None:
                    exec_results = self._execute_signals(new_signals)
                    executed_count = sum(
                        1 for r in exec_results
                        if r.state == ExecutionState.ORDER_SUBMITTED
                    )
                    logger.info(
                        f"Executed {executed_count}/{len(new_signals)} signals"
                    )

            # Scan completion
            duration = time.time() - start_time
            for alerter in self.alerters:
                if isinstance(alerter, LoggingAlerter):
                    alerter.log_scan_completed(
                        timeframe,
                        len(new_signals),
                        duration
                    )

            logger.info(
                f"Scan complete: {timeframe} - {len(new_signals)} signals "
                f"in {duration:.2f}s"
            )

            return new_signals

        except Exception as e:
            logger.error(f"Scan error for {timeframe}: {e}")
            self._error_count += 1

            for alerter in self.alerters:
                if isinstance(alerter, LoggingAlerter):
                    alerter.log_scan_error(timeframe, str(e))

            return []

    # =========================================================================
    # Session 83K-80: 15-Minute Base Scan (HTF Resampling Architecture)
    # =========================================================================

    def run_base_scan(self) -> List[StoredSignal]:
        """
        Run unified multi-TF scan using 15-min base resampling.

        Session 83K-80: HTF Scanning Architecture Fix.

        This method:
        1. Runs every 15 minutes during market hours
        2. Fetches 15-min data once per symbol
        3. Resamples to 1H, 1D, 1W, 1M
        4. Detects SETUP patterns on ALL timeframes

        Replaces separate run_scan() calls for each timeframe.

        Returns:
            List of new (non-duplicate) signals found across all timeframes
        """
        start_time = time.time()
        self._scan_count += 1

        # Notify scan started
        for alerter in self.alerters:
            if isinstance(alerter, LoggingAlerter):
                alerter.log_scan_started('ALL (15min resampled)', self.config.scan.symbols)

        try:
            new_signals: List[StoredSignal] = []

            # Scan each symbol using resampling
            for symbol in self.config.scan.symbols:
                try:
                    detected = self._scan_symbol_resampled(symbol)
                    new_signals.extend(detected)
                except Exception as e:
                    logger.error(f"Error scanning {symbol} (resampled): {e}")
                    self._error_count += 1

            # Send alerts for new signals
            if new_signals:
                self._send_alerts(new_signals)

                # Execute signals if execution enabled
                if self.executor is not None:
                    exec_results = self._execute_signals(new_signals)
                    executed_count = sum(
                        1 for r in exec_results
                        if r.state == ExecutionState.ORDER_SUBMITTED
                    )
                    logger.info(
                        f"Executed {executed_count}/{len(new_signals)} signals"
                    )

            # Scan completion
            duration = time.time() - start_time
            for alerter in self.alerters:
                if isinstance(alerter, LoggingAlerter):
                    alerter.log_scan_completed(
                        'ALL (15min resampled)',
                        len(new_signals),
                        duration
                    )

            logger.info(
                f"Base scan complete: ALL TFs - {len(new_signals)} signals "
                f"in {duration:.2f}s"
            )

            return new_signals

        except Exception as e:
            logger.error(f"Base scan error: {e}")
            self._error_count += 1

            for alerter in self.alerters:
                if isinstance(alerter, LoggingAlerter):
                    alerter.log_scan_error('ALL (15min resampled)', str(e))

            return []

    def _scan_symbol_resampled(self, symbol: str) -> List[StoredSignal]:
        """
        Scan a single symbol across all timeframes using 15-min resampling.

        Session 83K-80: Uses scan_symbol_all_timeframes_resampled().

        Args:
            symbol: Symbol to scan

        Returns:
            List of new signals (non-duplicates) across all timeframes
        """
        new_signals: List[StoredSignal] = []

        # Use the new resampling method
        detected_signals = self.scanner.scan_symbol_all_timeframes_resampled(symbol)

        for signal in detected_signals:
            # Apply quality filters
            if not self._passes_filters(signal):
                continue

            # Check for duplicates
            if self.signal_store.is_duplicate(signal):
                continue

            # Store new signal
            stored = self.signal_store.add_signal(signal)

            # Session 83K-68: Mark COMPLETED signals as HISTORICAL_TRIGGERED
            if stored.signal_type == SignalType.COMPLETED.value:
                self.signal_store.mark_historical_triggered(stored.signal_key)
                stored.status = SignalStatus.HISTORICAL_TRIGGERED.value
                logger.info(
                    f"HISTORICAL: {stored.symbol} {stored.pattern_type} {stored.direction} "
                    f"{stored.timeframe} (entry @ ${stored.entry_trigger:.2f} already occurred)"
                )

            new_signals.append(stored)
            self._signal_count += 1

        return new_signals

    def _scan_symbol(
        self,
        symbol: str,
        timeframe: str
    ) -> List[StoredSignal]:
        """
        Scan a single symbol for patterns.

        Args:
            symbol: Symbol to scan
            timeframe: Timeframe to scan

        Returns:
            List of new signals (non-duplicates)
        """
        new_signals: List[StoredSignal] = []

        # Get signals from scanner
        try:
            detected_signals = self.scanner.scan_symbol_timeframe(symbol, timeframe)
        except Exception as e:
            logger.error(f"Scanner error for {symbol} {timeframe}: {e}")
            return []

        for signal in detected_signals:
            # Apply quality filters
            if not self._passes_filters(signal):
                continue

            # Check for duplicate
            if self.signal_store.is_duplicate(signal):
                continue

            # Store new signal
            stored = self.signal_store.add_signal(signal)

            # Session 83K-68: Mark COMPLETED signals as HISTORICAL_TRIGGERED
            # These patterns already closed with the trigger bar, so entry already happened
            if stored.signal_type == SignalType.COMPLETED.value:
                self.signal_store.mark_historical_triggered(stored.signal_key)
                # Update stored object so alerting knows to skip mark_alerted
                stored.status = SignalStatus.HISTORICAL_TRIGGERED.value
                logger.info(
                    f"HISTORICAL: {stored.symbol} {stored.pattern_type} {stored.direction} "
                    f"(entry @ ${stored.entry_trigger:.2f} already occurred)"
                )

            new_signals.append(stored)
            self._signal_count += 1

        return new_signals

    def _passes_filters(self, signal: DetectedSignal) -> bool:
        """
        Check if signal passes quality filters.

        Args:
            signal: Signal to check

        Returns:
            True if passes all filters
        """
        # =================================================================
        # Session 83K-71: SETUP signals use relaxed thresholds
        # SETUP signals (ending in inside bar) have naturally lower magnitude
        # because target is first directional bar's extreme, which is close.
        # We allow them through for live monitoring, then can filter at execution.
        #
        # TO RESTORE STRICT FILTERING FOR SETUPS:
        # Set SIGNAL_SETUP_MIN_MAGNITUDE=0.5 and SIGNAL_SETUP_MIN_RR=1.0
        # in environment, or change the defaults below to match config values.
        # =================================================================
        is_setup = getattr(signal, 'signal_type', 'COMPLETED') == 'SETUP'

        if is_setup:
            # Relaxed thresholds for SETUP signals (paper trading/monitoring)
            # Defaults: magnitude >= 0.1%, R:R >= 0.3
            # These can be overridden via environment variables
            import os
            setup_min_magnitude = float(os.environ.get('SIGNAL_SETUP_MIN_MAGNITUDE', '0.1'))
            setup_min_rr = float(os.environ.get('SIGNAL_SETUP_MIN_RR', '0.3'))

            if signal.magnitude_pct < setup_min_magnitude:
                return False
            if signal.risk_reward < setup_min_rr:
                return False
        else:
            # Standard thresholds for COMPLETED signals (historical)
            if signal.magnitude_pct < self.config.scan.min_magnitude_pct:
                return False
            if signal.risk_reward < self.config.scan.min_risk_reward:
                return False

        # Pattern filter (if specific patterns configured)
        if self.config.scan.patterns:
            # Convert directional pattern name to base pattern for comparison
            # e.g., '2U-2D' -> '2-2', '2D-1-2U' -> '2-1-2', '3-2U' -> '3-2'
            # Also handle SETUP patterns like '3-1-?' and '2D-1-?'
            base_pattern = signal.pattern_type.replace('2U', '2').replace('2D', '2').replace('-?', '-2')
            if base_pattern not in self.config.scan.patterns:
                return False

        return True

    def _send_alerts(self, signals: List[StoredSignal]) -> None:
        """
        Send alerts for new signals.

        Session 83K-77: Discord only receives trade execution alerts (entry/exit),
        not signal detection alerts. Logging alerter still logs all signals.

        Args:
            signals: Signals to alert
        """
        for alerter in self.alerters:
            try:
                # Session 83K-77: Skip Discord for signal detection (only trades)
                if isinstance(alerter, DiscordAlerter):
                    # Mark as alerted without sending Discord message
                    for signal in signals:
                        if signal.status != SignalStatus.HISTORICAL_TRIGGERED.value:
                            self.signal_store.mark_alerted(signal.signal_key)
                    continue

                # Logging alerter: log all signals
                if len(signals) > 1 and hasattr(alerter, 'send_batch_alert'):
                    success = alerter.send_batch_alert(signals)
                else:
                    # Send individual alerts
                    success = True
                    for signal in signals:
                        if not alerter.send_alert(signal):
                            success = False

                if success:
                    # Mark signals as alerted (skip if already HISTORICAL_TRIGGERED)
                    for signal in signals:
                        if signal.status != SignalStatus.HISTORICAL_TRIGGERED.value:
                            self.signal_store.mark_alerted(signal.signal_key)

            except Exception as e:
                logger.error(f"Alert error ({alerter.name}): {e}")
                self._error_count += 1

    def _execute_signals(
        self,
        signals: List[StoredSignal]
    ) -> List[ExecutionResult]:
        """
        Execute signals via the executor (Session 83K-48).

        Args:
            signals: Signals to execute

        Returns:
            List of execution results
        """
        if self.executor is None:
            return []

        results: List[ExecutionResult] = []

        for signal in signals:
            try:
                result = self.executor.execute_signal(signal)
                results.append(result)

                if result.state == ExecutionState.ORDER_SUBMITTED:
                    self._execution_count += 1
                    # Mark signal as triggered in store
                    self.signal_store.mark_triggered(signal.signal_key)
                    logger.info(
                        f"Order submitted for {signal.signal_key}: "
                        f"{result.osi_symbol}"
                    )
                elif result.state == ExecutionState.SKIPPED:
                    logger.debug(
                        f"Signal skipped: {signal.signal_key} - {result.error}"
                    )
                elif result.state == ExecutionState.FAILED:
                    logger.warning(
                        f"Execution failed: {signal.signal_key} - {result.error}"
                    )
                    self._error_count += 1

            except Exception as e:
                logger.exception(f"Execution error for {signal.signal_key}: {e}")
                self._error_count += 1
                results.append(ExecutionResult(
                    signal_key=signal.signal_key,
                    state=ExecutionState.FAILED,
                    error=str(e)
                ))

        return results

    def execute_signals(
        self,
        signals: List[StoredSignal]
    ) -> List[ExecutionResult]:
        """
        Public method to execute signals (Session 83K-48).

        Allows manual execution of signals from CLI.

        Args:
            signals: Signals to execute

        Returns:
            List of execution results
        """
        if self.executor is None:
            logger.error("Execution not enabled - set SIGNAL_EXECUTION_ENABLED=true")
            return []

        return self._execute_signals(signals)

    def _create_scan_callback(self, timeframe: str):
        """Create callback function for scheduled scan."""
        def callback():
            try:
                self.run_scan(timeframe)
            except Exception as e:
                logger.error(f"Scheduled scan error ({timeframe}): {e}")
                self._error_count += 1
        return callback

    def _run_position_check(self) -> None:
        """
        Check positions for exit conditions (Session 83K-49).

        Called periodically by scheduler when monitoring is enabled.
        """
        if self.position_monitor is None:
            return

        try:
            # Check all positions
            exit_signals = self.position_monitor.check_positions()

            if exit_signals:
                logger.info(f"Found {len(exit_signals)} exit signal(s)")

                # Execute exits
                for signal in exit_signals:
                    result = self.position_monitor.execute_exit(signal)
                    if result:
                        self._exit_count += 1
                        logger.info(
                            f"Position closed: {signal.osi_symbol} - "
                            f"{signal.reason.value} - P&L: ${signal.unrealized_pnl:.2f}"
                        )

        except Exception as e:
            logger.error(f"Position check error: {e}")
            self._error_count += 1

    def check_positions_now(self) -> List[ExitSignal]:
        """
        Manually check positions for exit conditions (Session 83K-49).

        Returns:
            List of exit signals found
        """
        if self.position_monitor is None:
            logger.warning("Position monitoring not enabled")
            return []

        return self.position_monitor.check_positions()

    def get_tracked_positions(self) -> List[TrackedPosition]:
        """
        Get all tracked positions (Session 83K-49).

        Returns:
            List of TrackedPosition objects
        """
        if self.position_monitor is None:
            return []
        return self.position_monitor.get_tracked_positions()

    def _health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Health status dictionary
        """
        uptime = None
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        status = {
            'status': 'healthy' if self._is_running else 'stopped',
            'uptime_seconds': uptime,
            'scan_count': self._scan_count,
            'signal_count': self._signal_count,
            'execution_count': self._execution_count,
            'exit_count': self._exit_count,
            'error_count': self._error_count,
            'signals_in_store': len(self.signal_store),
            'alerters': [a.name for a in self.alerters],
            'scheduler': self.scheduler.get_status(),
            'execution_enabled': self.executor is not None,
            'monitoring_enabled': self.position_monitor is not None,
        }

        # Add monitoring stats (Session 83K-49)
        if self.position_monitor is not None:
            monitor_stats = self.position_monitor.get_stats()
            status['monitoring'] = monitor_stats

        # Log health check
        for alerter in self.alerters:
            if isinstance(alerter, LoggingAlerter):
                alerter.log_health_check(status)

        return status

    def _setup_signal_handlers(self) -> None:
        """Setup OS signal handlers for graceful shutdown."""
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Windows-specific handling
        if sys.platform == 'win32':
            try:
                signal.signal(signal.SIGBREAK, handle_shutdown)
            except AttributeError:
                pass

    def start(self, block: bool = True) -> None:
        """
        Start the daemon.

        Args:
            block: Block until shutdown signal (default: True)
        """
        if self._is_running:
            logger.warning("Daemon already running")
            return

        logger.info("Starting signal daemon...")
        self._start_time = datetime.now()
        self._is_running = True

        # Setup signal handlers
        self._setup_signal_handlers()

        # Register scheduled jobs
        # Session 83K-80: Use either new 15-min resampling or legacy separate jobs
        if self.config.schedule.enable_htf_resampling:
            # NEW: Single 15-min job handles all timeframes via resampling
            self.scheduler.add_base_scan_job(self.run_base_scan)
            logger.info(
                "Using 15-min base resampling for ALL timeframes "
                "(HTF scanning architecture fix)"
            )
        else:
            # LEGACY: Separate jobs per timeframe
            self.scheduler.add_hourly_job(
                self._create_scan_callback('1H')
            )
            self.scheduler.add_daily_job(
                self._create_scan_callback('1D')
            )
            self.scheduler.add_weekly_job(
                self._create_scan_callback('1W')
            )
            self.scheduler.add_monthly_job(
                self._create_scan_callback('1M')
            )
            logger.info("Using legacy separate scan jobs per timeframe")

        # Health check job
        self.scheduler.add_health_check_job(
            self._health_check,
            interval_seconds=self.config.health_check_interval,
        )

        # Position monitoring job (Session 83K-49)
        if self.position_monitor is not None:
            self.scheduler.add_interval_job(
                self._run_position_check,
                interval_seconds=self.config.monitoring.check_interval,
                job_id='position_monitor',
            )
            logger.info(
                f"Position monitoring enabled "
                f"(check every {self.config.monitoring.check_interval}s)"
            )

        # Entry trigger monitoring (Session 83K-66)
        if self.entry_monitor is not None:
            self.entry_monitor.start()
            logger.info(
                f"Entry trigger monitoring enabled "
                f"(1-minute polling during market hours)"
            )

        # Start scheduler
        self.scheduler.start()

        # Notify startup (Session 83K-77: Discord only sends trade alerts, not daemon status)
        for alerter in self.alerters:
            if isinstance(alerter, LoggingAlerter):
                alerter.log_daemon_started()

        logger.info("Signal daemon started successfully")

        if block:
            self._run_loop()

    def _run_loop(self) -> None:
        """Main daemon loop - waits for shutdown signal."""
        logger.info("Daemon entering main loop (Ctrl+C to stop)")

        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")

        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the daemon gracefully."""
        if not self._is_running:
            return

        logger.info("Shutting down signal daemon...")

        # Stop entry monitor (Session 83K-66)
        if self.entry_monitor is not None:
            self.entry_monitor.stop()

        # Stop scheduler
        self.scheduler.shutdown(wait=True)

        # Final health status
        status = self._health_check()

        # Notify shutdown
        # Session 83K-77: Discord only sends trade alerts, not daemon status
        for alerter in self.alerters:
            if isinstance(alerter, LoggingAlerter):
                alerter.log_daemon_stopped('graceful_shutdown')

        self._is_running = False
        logger.info("Signal daemon shutdown complete")

    def run_all_scans(self) -> Dict[str, List[StoredSignal]]:
        """
        Run scans for all configured timeframes.

        Returns:
            Dictionary of timeframe -> signals
        """
        results = {}
        for tf in self.config.scan.timeframes:
            results[tf] = self.run_scan(tf)
        return results

    def test_alerters(self) -> Dict[str, bool]:
        """
        Test all configured alerters.

        Returns:
            Dictionary of alerter_name -> test_passed
        """
        results = {}
        for alerter in self.alerters:
            try:
                results[alerter.name] = alerter.test_connection()
            except Exception as e:
                logger.error(f"Alerter test failed ({alerter.name}): {e}")
                results[alerter.name] = False
        return results

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._is_running

    def get_status(self) -> Dict[str, Any]:
        """
        Get daemon status.

        Returns:
            Status dictionary
        """
        return {
            'running': self._is_running,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'scan_count': self._scan_count,
            'signal_count': self._signal_count,
            'execution_count': self._execution_count,
            'error_count': self._error_count,
            'signals_in_store': len(self.signal_store),
            'alerters': [a.name for a in self.alerters],
            'execution_enabled': self.executor is not None,
            'config': {
                'symbols': self.config.scan.symbols,
                'timeframes': self.config.scan.timeframes,
                'patterns': self.config.scan.patterns,
            },
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current option positions from executor (Session 83K-48).

        Returns:
            List of position dictionaries
        """
        if self.executor is None:
            return []
        return self.executor.get_positions()

    def close_position(self, osi_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Close an option position (Session 83K-48).

        Args:
            osi_symbol: OCC symbol to close

        Returns:
            Close order details or None if failed
        """
        if self.executor is None:
            logger.error("Execution not enabled")
            return None
        return self.executor.close_position(osi_symbol)
