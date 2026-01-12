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
from datetime import datetime, time as dt_time, timedelta
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
        # Session EQUITY-29: Reduced from 60s to 15s for faster "on the break" detection
        monitor_config = EntryMonitorConfig(
            poll_interval=15,  # 15 seconds for faster trigger detection
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

        # Session EQUITY-46: Check if setup is stale before executing
        # A setup becomes stale when a new bar closes, potentially changing the pattern structure
        is_stale, stale_reason = self._is_setup_stale(signal)
        if is_stale:
            logger.warning(
                f"STALE SETUP REJECTED: {signal.symbol} {signal.pattern_type} {signal.direction} "
                f"@ ${event.current_price:.2f} - {stale_reason}"
            )
            # Mark as expired instead of triggered
            self.signal_store.mark_expired(signal.signal_key)
            return

        # Session EQUITY-49: TFC Re-evaluation at Entry
        # TFC can change between pattern detection and entry trigger (hours/days later).
        # Re-evaluate TFC and optionally block entry if alignment degraded significantly.
        tfc_blocked, tfc_reason = self._reevaluate_tfc_at_entry(signal)
        if tfc_blocked:
            logger.warning(
                f"TFC REEVAL REJECTED: {signal.symbol} {signal.pattern_type} {signal.direction} "
                f"@ ${event.current_price:.2f} - {tfc_reason}"
            )
            # Mark as expired instead of triggered
            self.signal_store.mark_expired(signal.signal_key)
            return

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
                    # Store OSI symbol for closed trade correlation
                    if result.osi_symbol:
                        self.signal_store.set_executed_osi_symbol(
                            signal.signal_key, result.osi_symbol
                        )
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

                # Session EQUITY-32: Execute TRIGGERED patterns immediately
                # These are COMPLETED patterns where entry bar already formed.
                if self.executor is not None:
                    triggered_count = 0
                    executed_symbols = set()

                    for signal in new_signals:
                        if signal.signal_type == SignalType.COMPLETED.value:
                            signal_key = f"{signal.symbol}_{signal.timeframe}"

                            if signal_key in executed_symbols:
                                continue

                            result = self._execute_triggered_pattern(signal)
                            if result and result.state == ExecutionState.ORDER_SUBMITTED:
                                executed_symbols.add(signal_key)
                                triggered_count += 1

                    if triggered_count > 0:
                        logger.info(f"Executed {triggered_count} TRIGGERED patterns")

                # Execute SETUP signals if any (skipped by _execute_signals per EQUITY-29)
                if self.executor is not None:
                    exec_results = self._execute_signals(new_signals)
                    executed_count = sum(
                        1 for r in exec_results
                        if r.state == ExecutionState.ORDER_SUBMITTED
                    )
                    if executed_count > 0:
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

                # Session EQUITY-32: Execute TRIGGERED patterns immediately
                # These are COMPLETED patterns where entry bar already formed.
                # Previously these were marked HISTORICAL_TRIGGERED and skipped!
                if self.executor is not None:
                    triggered_count = 0
                    executed_symbols = set()  # Track symbol+timeframe to avoid duplicates

                    for signal in new_signals:
                        if signal.signal_type == SignalType.COMPLETED.value:
                            signal_key = f"{signal.symbol}_{signal.timeframe}"

                            # Skip if we already executed for this symbol/timeframe
                            if signal_key in executed_symbols:
                                logger.debug(
                                    f"Skipping duplicate TRIGGERED: {signal.symbol} "
                                    f"{signal.pattern_type} ({signal.timeframe})"
                                )
                                continue

                            result = self._execute_triggered_pattern(signal)
                            if result and result.state == ExecutionState.ORDER_SUBMITTED:
                                executed_symbols.add(signal_key)
                                triggered_count += 1

                    if triggered_count > 0:
                        logger.info(f"Executed {triggered_count} TRIGGERED patterns")

                # Execute SETUP signals if any (will be skipped by _execute_signals per EQUITY-29)
                # This call is kept for logging/alerting purposes
                if self.executor is not None:
                    exec_results = self._execute_signals(new_signals)
                    executed_count = sum(
                        1 for r in exec_results
                        if r.state == ExecutionState.ORDER_SUBMITTED
                    )
                    if executed_count > 0:
                        logger.info(
                            f"Executed {executed_count}/{len(new_signals)} signals via _execute_signals"
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

        Session EQUITY-47: Added logging for filter rejections with actual vs threshold values.

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
        # Human-readable key for filter logging (not the database signal_key which uses timestamp)
        signal_key = f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_{signal.direction}"

        if is_setup:
            # Relaxed thresholds for SETUP signals (paper trading/monitoring)
            # Defaults: magnitude >= 0.1%, R:R >= 0.3
            # These can be overridden via environment variables
            import os
            setup_min_magnitude = float(os.environ.get('SIGNAL_SETUP_MIN_MAGNITUDE', '0.1'))
            setup_min_rr = float(os.environ.get('SIGNAL_SETUP_MIN_RR', '0.3'))

            if signal.magnitude_pct < setup_min_magnitude:
                # Session EQUITY-47: Log filter rejection with actual vs threshold
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"magnitude {signal.magnitude_pct:.3f}% < min {setup_min_magnitude}% (SETUP)"
                )
                return False
            if signal.risk_reward < setup_min_rr:
                # Session EQUITY-47: Log filter rejection with actual vs threshold
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"R:R {signal.risk_reward:.2f} < min {setup_min_rr} (SETUP)"
                )
                return False
        else:
            # Standard thresholds for COMPLETED signals (historical)
            if signal.magnitude_pct < self.config.scan.min_magnitude_pct:
                # Session EQUITY-47: Log filter rejection with actual vs threshold
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"magnitude {signal.magnitude_pct:.3f}% < min {self.config.scan.min_magnitude_pct}%"
                )
                return False
            if signal.risk_reward < self.config.scan.min_risk_reward:
                # Session EQUITY-47: Log filter rejection with actual vs threshold
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"R:R {signal.risk_reward:.2f} < min {self.config.scan.min_risk_reward}"
                )
                return False

        # Pattern filter (if specific patterns configured)
        if self.config.scan.patterns:
            # Convert directional pattern name to base pattern for comparison
            # e.g., '2U-2D' -> '2-2', '2D-1-2U' -> '2-1-2', '3-2U' -> '3-2'
            # Also handle SETUP patterns like '3-1-?' and '2D-1-?'
            base_pattern = signal.pattern_type.replace('2U', '2').replace('2D', '2').replace('-?', '-2')
            if base_pattern not in self.config.scan.patterns:
                # Session EQUITY-47: Log filter rejection for pattern type
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"pattern '{base_pattern}' not in allowed patterns {self.config.scan.patterns}"
                )
                return False

        return True

    def _is_market_hours(self) -> bool:
        """
        Check if current time is within NYSE market hours.

        Session EQUITY-34: Uses pandas_market_calendars for accurate holiday
        and early close handling (e.g., Christmas Eve 1PM, Thanksgiving Friday 1PM).

        Returns:
            True if within market hours, False otherwise
        """
        import pytz
        import pandas_market_calendars as mcal

        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')

        # Get today's schedule
        schedule = nyse.schedule(
            start_date=now.date(),
            end_date=now.date()
        )

        # Check if market is open today (handles holidays)
        if schedule.empty:
            logger.debug(f"Market closed: {now.date()} is not a trading day")
            return False

        # Get market open/close times (handles early closes)
        market_open = schedule.iloc[0]['market_open']
        market_close = schedule.iloc[0]['market_close']

        # Compare timezone-aware datetimes
        if now < market_open:
            logger.debug(f"Market not yet open: opens at {market_open.strftime('%H:%M')} ET")
            return False

        if now > market_close:
            logger.debug(f"Market closed: closed at {market_close.strftime('%H:%M')} ET")
            return False

        return True

    def _is_setup_stale(self, signal: StoredSignal) -> tuple[bool, str]:
        """
        Check if a SETUP signal is stale (new bar has closed since detection).

        Session EQUITY-46: Fix for stale setup bug where setups from N days ago
        triggered when they should have been invalidated by intervening bars.

        A setup is only valid during the "forming bar" period. Once that bar closes:
        - If trigger was hit -> pattern COMPLETED (entry happened)
        - If trigger NOT hit -> pattern EVOLVED (new bar inserted, setup stale)

        Args:
            signal: The stored signal to check

        Returns:
            Tuple of (is_stale: bool, reason: str)
        """
        import pytz

        # Only check SETUP signals (COMPLETED are already validated)
        if signal.signal_type != SignalType.SETUP.value:
            return False, ""

        # Need setup_bar_timestamp to check staleness
        setup_ts = signal.setup_bar_timestamp
        if setup_ts is None:
            logger.warning(
                f"STALE CHECK SKIPPED: {signal.signal_key} has no setup_bar_timestamp"
            )
            return False, ""

        # Ensure timezone-aware comparison
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Make setup_ts timezone-aware if it isn't
        if setup_ts.tzinfo is None:
            setup_ts = et.localize(setup_ts)

        # Calculate staleness based on timeframe
        timeframe = signal.timeframe

        if timeframe == '1H':
            # For hourly: setup is valid for 1 hour after setup_bar_timestamp
            # (the forming bar period)
            valid_until = setup_ts + timedelta(hours=1)
            if now > valid_until:
                return True, f"Hourly setup expired: detected {setup_ts.strftime('%H:%M')}, now {now.strftime('%H:%M')}"

        elif timeframe == '1D':
            # For daily: setup is valid until end of NEXT trading day
            # If setup from Jan 5, valid during Jan 6, stale on Jan 7+
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar('NYSE')

            # Get trading days from setup date to now
            schedule = nyse.schedule(
                start_date=setup_ts.date(),
                end_date=now.date()
            )

            if len(schedule) > 2:
                # More than 2 trading days (setup day + forming day + today)
                return True, f"Daily setup expired: {len(schedule)} trading days since setup ({setup_ts.date()} to {now.date()})"

        elif timeframe == '1W':
            # For weekly: setup is valid until end of NEXT week
            # Calculate weeks difference
            setup_week = setup_ts.isocalendar()[1]
            setup_year = setup_ts.isocalendar()[0]
            now_week = now.isocalendar()[1]
            now_year = now.isocalendar()[0]

            # Simple week difference (handles year boundary)
            weeks_diff = (now_year - setup_year) * 52 + (now_week - setup_week)

            if weeks_diff > 1:
                return True, f"Weekly setup expired: {weeks_diff} weeks since setup"

        elif timeframe == '1M':
            # For monthly: setup is valid until end of NEXT month
            setup_month = setup_ts.month
            setup_year = setup_ts.year
            now_month = now.month
            now_year = now.year

            months_diff = (now_year - setup_year) * 12 + (now_month - setup_month)

            if months_diff > 1:
                return True, f"Monthly setup expired: {months_diff} months since setup"

        return False, ""

    def _reevaluate_tfc_at_entry(self, signal: StoredSignal) -> tuple[bool, str]:
        """
        Re-evaluate TFC alignment at entry time and check if entry should be blocked.

        Session EQUITY-49: TFC Re-evaluation at Entry.

        TFC can change between pattern detection and entry trigger (hours/days later).
        This method:
        1. Re-evaluates TFC using current market data
        2. Compares with original TFC at detection time
        3. Logs the comparison for audit trail
        4. Optionally blocks entry if TFC degraded significantly or flipped direction

        Args:
            signal: The stored signal about to be executed

        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        # Check if TFC re-evaluation is enabled
        if not self.config.execution.tfc_reeval_enabled:
            return False, ""

        # Get original TFC data from signal with defensive validation (Issue #2 fix)
        original_strength = signal.tfc_score if signal.tfc_score is not None else 0
        original_alignment = signal.tfc_alignment or ""  # Handle None
        original_passes = signal.passes_flexible if signal.passes_flexible is not None else True

        # Determine original direction from alignment string
        original_tfc_direction = ""
        if original_alignment and "BULLISH" in original_alignment.upper():
            original_tfc_direction = "bullish"
        elif original_alignment and "BEARISH" in original_alignment.upper():
            original_tfc_direction = "bearish"
        elif not original_alignment:
            logger.debug(
                f"TFC REEVAL: {signal.signal_key} has no original TFC alignment - "
                f"direction flip detection will be skipped"
            )

        # Re-evaluate TFC using current market data
        # Direction: CALL = bullish (1), PUT = bearish (-1)
        direction_int = 1 if signal.direction == 'CALL' else -1

        try:
            current_tfc = self.scanner.evaluate_tfc(
                symbol=signal.symbol,
                detection_timeframe=signal.timeframe,
                direction=direction_int
            )
        except (ConnectionError, TimeoutError, ValueError) as e:
            # Expected errors: network issues, data issues - log and proceed
            logger.warning(
                f"TFC REEVAL ERROR (recoverable): {signal.symbol} {signal.pattern_type} - "
                f"{type(e).__name__}: {e} (proceeding with entry)"
            )
            self._error_count += 1
            return False, ""
        except Exception as e:
            # Unexpected error - log as error, increment counter, but still proceed
            # (fail-open to avoid blocking all entries on system errors)
            logger.error(
                f"TFC REEVAL UNEXPECTED ERROR: {signal.symbol} {signal.pattern_type} - "
                f"{type(e).__name__}: {e} (proceeding with entry)"
            )
            self._error_count += 1
            return False, ""

        # Validate returned assessment (Issue #5 fix)
        if current_tfc is None or not hasattr(current_tfc, 'strength'):
            logger.error(f"TFC REEVAL: Invalid assessment returned for {signal.symbol} - proceeding with entry")
            self._error_count += 1
            return False, ""

        current_strength = current_tfc.strength if current_tfc.strength is not None else 0
        current_alignment = current_tfc.alignment_label() if hasattr(current_tfc, 'alignment_label') else f"{current_strength}/?"
        current_passes = getattr(current_tfc, 'passes_flexible', True)
        current_direction = getattr(current_tfc, 'direction', '') or ""

        # Calculate strength change
        strength_delta = current_strength - original_strength

        # Detect direction flip (e.g., bullish -> bearish) (Issue #4 fix)
        direction_flipped = False
        if original_tfc_direction and current_direction:
            direction_flipped = original_tfc_direction != current_direction
        elif not original_tfc_direction and current_direction:
            # Can't detect flip without original direction - log at debug level
            logger.debug(
                f"TFC REEVAL: {signal.signal_key} - direction flip detection skipped "
                f"(no original TFC direction in signal)"
            )

        # Build comparison log message
        comparison = (
            f"TFC REEVAL: {signal.signal_key} ({signal.symbol} {signal.pattern_type} {signal.direction}) | "
            f"Original: {original_alignment or 'N/A'} (score={original_strength}, passes={original_passes}) | "
            f"Current: {current_alignment} (score={current_strength}, passes={current_passes}) | "
            f"Delta: {strength_delta:+d} | "
            f"Flipped: {direction_flipped}"
        )

        # Always log if configured
        if self.config.execution.tfc_reeval_log_always:
            if strength_delta < 0 or direction_flipped:
                logger.warning(comparison)
            else:
                logger.info(comparison)

        # Determine if entry should be blocked
        should_block = False
        block_reason = ""

        # Block if direction flipped (most severe)
        if direction_flipped and self.config.execution.tfc_reeval_block_on_flip:
            should_block = True
            block_reason = f"TFC direction flipped from {original_tfc_direction} to {current_direction}"

        # Block if strength dropped below minimum threshold
        elif current_strength < self.config.execution.tfc_reeval_min_strength:
            should_block = True
            block_reason = f"TFC strength {current_strength} < min threshold {self.config.execution.tfc_reeval_min_strength}"

        return should_block, block_reason

    def _send_alerts(self, signals: List[StoredSignal]) -> None:
        """
        Send alerts for new signals.

        Session EQUITY-34: Uses explicit config flags for Discord alert control.
        Discord only receives alerts based on alert_on_signal_detection config.
        Trade entry/exit alerts are controlled by alert_on_trade_entry/exit flags.
        Logging alerter still logs all signals.

        Args:
            signals: Signals to alert
        """
        # Session EQUITY-35: Debug logging to understand alert flow
        import pytz
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)
        logger.info(
            f"_send_alerts called: {len(signals)} signals at {now_et.strftime('%H:%M:%S ET')}, "
            f"is_market_hours={self._is_market_hours()}, "
            f"alert_on_signal_detection={self.config.alerts.alert_on_signal_detection}"
        )

        # Session EQUITY-40: Sort signals by priority and continuity strength for deterministic batching
        signals = sorted(
            signals,
            key=lambda s: (
                getattr(s, 'priority', 0),
                getattr(s, 'continuity_strength', 0),
                getattr(s, 'magnitude_pct', 0),
            ),
            reverse=True,
        )

        # Session EQUITY-33: Skip ALL alerting during premarket/afterhours
        if not self._is_market_hours():
            logger.info(
                f"BLOCKED (outside market hours): {len(signals)} signals at {now_et.strftime('%H:%M:%S ET')}"
            )
            # Still mark as alerted for internal tracking
            for signal in signals:
                if signal.status != SignalStatus.HISTORICAL_TRIGGERED.value:
                    self.signal_store.mark_alerted(signal.signal_key)
            return

        for alerter in self.alerters:
            try:
                # Session EQUITY-34: Use explicit config flag for Discord signal detection
                if isinstance(alerter, DiscordAlerter):
                    if not self.config.alerts.alert_on_signal_detection:
                        # Session EQUITY-35: Log when Discord alerts are blocked
                        logger.info(
                            f"BLOCKED Discord pattern alerts: alert_on_signal_detection=False, "
                            f"{len(signals)} signals"
                        )
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

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol using executor's trading client.

        Session EQUITY-32: Helper for triggered pattern execution.

        Args:
            symbol: Stock symbol

        Returns:
            Current mid price or None if unavailable
        """
        if self.executor is None:
            return None
        try:
            quotes = self.executor._trading_client.get_stock_quotes([symbol])
            if symbol in quotes:
                quote = quotes[symbol]
                if isinstance(quote, dict) and 'mid' in quote:
                    return quote['mid']
                elif isinstance(quote, (int, float)):
                    return float(quote)
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
        return None

    def _execute_triggered_pattern(self, signal: StoredSignal) -> Optional[ExecutionResult]:
        """
        Execute a TRIGGERED pattern immediately at market price.

        Session EQUITY-32: Patterns where entry bar has formed should
        execute at current market price, not be discarded.

        These are COMPLETED patterns (e.g., 3-1-2U where the 2U bar has
        already broken the inside bar). STRAT principle: entry on the break.

        Args:
            signal: TRIGGERED signal (signal_type="COMPLETED")

        Returns:
            ExecutionResult if executed, None if skipped
        """
        if self.executor is None:
            return None

        # Get current market price
        current_price = self._get_current_price(signal.symbol)
        if current_price is None or current_price <= 0:
            logger.warning(f"Could not get price for {signal.symbol} - skipping triggered pattern")
            return None

        direction = signal.direction
        stop_price = signal.stop_price
        target_price = signal.target_price

        # Validate entry is still viable (price hasn't blown past target)
        if direction == "CALL":
            if current_price >= target_price:
                logger.info(
                    f"SKIP TRIGGERED: {signal.symbol} {signal.pattern_type} CALL - "
                    f"price ${current_price:.2f} already at/past target ${target_price:.2f}"
                )
                return None
        else:  # PUT
            if current_price <= target_price:
                logger.info(
                    f"SKIP TRIGGERED: {signal.symbol} {signal.pattern_type} PUT - "
                    f"price ${current_price:.2f} already at/past target ${target_price:.2f}"
                )
                return None

        # "Let the Market Breathe" - Skip intraday patterns if too early in session
        if not self._is_intraday_entry_allowed(signal):
            return None

        logger.info(
            f"TRIGGERED PATTERN: {signal.symbol} {signal.pattern_type} {signal.direction} "
            f"({signal.timeframe}) - executing at ${current_price:.2f}"
        )

        try:
            # Execute via SignalExecutor
            # Temporarily set status to ALERTED to bypass HISTORICAL_TRIGGERED skip in executor
            original_status = signal.status
            signal.status = SignalStatus.ALERTED.value

            result = self.executor.execute_signal(signal, underlying_price=current_price)

            signal.status = original_status  # Restore original status

            if result.state == ExecutionState.ORDER_SUBMITTED:
                self._execution_count += 1
                self.signal_store.mark_triggered(signal.signal_key)

                if result.osi_symbol:
                    self.signal_store.set_executed_osi_symbol(
                        signal.signal_key, result.osi_symbol
                    )

                logger.info(
                    f"TRADE OPENED (TRIGGERED): {signal.symbol} {signal.direction} "
                    f"{signal.pattern_type} ({signal.timeframe}) - {result.osi_symbol}"
                )

                # Send Discord entry alert
                for alerter in self.alerters:
                    try:
                        if isinstance(alerter, DiscordAlerter):
                            alerter.send_entry_alert(signal, result)
                        elif isinstance(alerter, LoggingAlerter):
                            alerter.log_execution(result)
                    except Exception as e:
                        logger.error(f"Entry alert error: {e}")

                return result
            else:
                logger.info(
                    f"TRIGGERED execution skipped/failed: {signal.symbol} - {result.error}"
                )
                return result

        except Exception as e:
            logger.error(f"Failed to execute triggered pattern: {e}")
            return None

    def _is_intraday_entry_allowed(self, signal: StoredSignal) -> bool:
        """
        Check if intraday pattern entry is allowed based on "let the market breathe" rules.

        Session EQUITY-18: Extended to support 15m, 30m, and 1H timeframes.

        For intraday patterns, we must wait for sufficient bars to close:

        15m timeframe:
        - 2-bar patterns: Earliest entry at 9:45 AM ET
        - 3-bar patterns: Earliest entry at 10:00 AM ET

        30m timeframe:
        - 2-bar patterns: Earliest entry at 10:00 AM ET
        - 3-bar patterns: Earliest entry at 10:30 AM ET

        1H timeframe:
        - 2-bar patterns: Earliest entry at 10:30 AM ET
        - 3-bar patterns: Earliest entry at 11:30 AM ET

        Daily, Weekly, Monthly patterns have no time restriction because
        larger timeframes carry more significance.

        Args:
            signal: The signal to check

        Returns:
            True if entry is allowed, False if too early
        """
        # Only apply time restriction to intraday patterns
        intraday_timeframes = ('15m', '30m', '1H')
        if signal.timeframe not in intraday_timeframes:
            return True

        # Session EQUITY-57: CRITICAL FIX - Must use Eastern Time, not system local time
        # VPS runs in UTC, so datetime.now().time() returns UTC time, not ET
        # The time thresholds (10:30, 11:30) are in Eastern Time
        import pytz
        eastern = pytz.timezone('America/New_York')
        current_time = datetime.now(eastern).time()

        # Session EQUITY-18: Time thresholds per timeframe
        # Based on "Let the Market Breathe" design from HANDOFF.md
        time_thresholds = {
            '15m': {
                '2bar': dt_time(9, 45),   # After first 15m bar closes
                '3bar': dt_time(10, 0),   # After first two 15m bars close
            },
            '30m': {
                '2bar': dt_time(10, 0),   # After first 30m bar closes
                '3bar': dt_time(10, 30),  # After first two 30m bars close
            },
            '1H': {
                '2bar': dt_time(10, 30),  # After first 1H bar closes
                '3bar': dt_time(11, 30),  # After first two 1H bars close
            },
        }

        thresholds = time_thresholds[signal.timeframe]

        # Determine if this is a 2-bar or 3-bar pattern by counting components
        # 3-bar patterns have 3 components: X-Y-Z (e.g., 3-2D-2U, 2D-1-2U, 2U-1-?)
        # 2-bar patterns have 2 components: X-Y (e.g., 2D-2U, 3-2D)
        pattern = signal.pattern_type
        pattern_parts = pattern.split('-')
        is_3bar_pattern = len(pattern_parts) >= 3

        earliest_time = thresholds['3bar'] if is_3bar_pattern else thresholds['2bar']
        pattern_type_str = '3-bar' if is_3bar_pattern else '2-bar'

        if current_time < earliest_time:
            logger.info(
                f"TIMING FILTER BLOCKED: {signal.symbol} {pattern} ({signal.timeframe}) - "
                f"{pattern_type_str} pattern before {earliest_time.strftime('%H:%M')} "
                f"(current: {current_time.strftime('%H:%M')})"
            )
            return False

        # Session EQUITY-33: Log when patterns pass the filter for verification
        logger.info(
            f"TIMING FILTER PASSED: {signal.symbol} {pattern} ({signal.timeframe}) - "
            f"{pattern_type_str} at {current_time.strftime('%H:%M')} "
            f"(threshold: {earliest_time.strftime('%H:%M')})"
        )
        return True

    def _execute_signals(
        self,
        signals: List[StoredSignal]
    ) -> List[ExecutionResult]:
        """
        Execute signals via the executor (Session 83K-48).

        Includes:
        - "Let the market breathe" filtering for hourly patterns
        - Discord entry alerts on successful order submission

        Args:
            signals: Signals to execute

        Returns:
            List of execution results
        """
        if self.executor is None:
            return []

        results: List[ExecutionResult] = []

        for signal in signals:
            # Session EQUITY-29: SETUP signals should NOT execute immediately
            # They need to wait for entry_monitor to detect trigger break
            # Only COMPLETED signals (entry already happened) execute immediately
            if getattr(signal, 'signal_type', 'COMPLETED') == 'SETUP':
                logger.debug(
                    f"SETUP signal {signal.signal_key} skipped from immediate execution - "
                    f"waiting for entry_monitor trigger"
                )
                continue

            # Session EQUITY-42: COMPLETED signals already executed by _execute_triggered_pattern()
            # Skip them here to prevent duplicate execution and duplicate Discord alerts
            # Bug: run_scan() and run_base_scan() call _execute_triggered_pattern() first,
            # then call _execute_signals() with the same signals, causing double execution.
            if getattr(signal, 'signal_type', 'COMPLETED') == 'COMPLETED':
                logger.debug(
                    f"COMPLETED signal {signal.signal_key} skipped - "
                    f"already executed by _execute_triggered_pattern()"
                )
                continue

            # "Let the Market Breathe" - Skip intraday patterns if too early in session
            if not self._is_intraday_entry_allowed(signal):
                results.append(ExecutionResult(
                    signal_key=signal.signal_key,
                    state=ExecutionState.SKIPPED,
                    error=f"Intraday {signal.timeframe} pattern blocked - too early in session (let market breathe)"
                ))
                continue

            try:
                result = self.executor.execute_signal(signal)
                results.append(result)

                if result.state == ExecutionState.ORDER_SUBMITTED:
                    self._execution_count += 1
                    # Mark signal as triggered in store
                    self.signal_store.mark_triggered(signal.signal_key)
                    # Store OSI symbol for closed trade correlation
                    if result.osi_symbol:
                        self.signal_store.set_executed_osi_symbol(
                            signal.signal_key, result.osi_symbol
                        )
                    logger.info(
                        f"Order submitted for {signal.signal_key}: "
                        f"{result.osi_symbol}"
                    )

                    # Send Discord entry alert (was missing from this code path!)
                    for alerter in self.alerters:
                        try:
                            if isinstance(alerter, DiscordAlerter):
                                alerter.send_entry_alert(signal, result)
                            elif isinstance(alerter, LoggingAlerter):
                                alerter.log_execution(result)
                        except Exception as e:
                            logger.error(f"Entry alert error: {e}")

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

    # =========================================================================
    # EQUITY-52: Daily Trade Audit
    # =========================================================================

    def _generate_daily_audit(self) -> Dict[str, Any]:
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
        import json
        from pathlib import Path
        from datetime import date

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
        paper_trades_path = Path('paper_trades/paper_trades.json')
        if paper_trades_path.exists():
            try:
                with open(paper_trades_path, 'r') as f:
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

        # Get open positions from position monitor
        if self.position_monitor is not None:
            positions = self.position_monitor.get_tracked_positions()
            for pos in positions:
                pos_summary = {
                    'symbol': pos.symbol,
                    'pattern_type': pos.pattern_type,
                    'timeframe': pos.timeframe,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pct': pos.unrealized_pct,
                }
                audit_data['open_positions'].append(pos_summary)

        # Check for anomalies
        # Anomaly 1: Trades with unexpected exit reasons
        # Anomaly 2: Large losses (> $100)
        # These can be expanded in future sessions

        logger.info(
            f"Daily audit generated: {audit_data['trades_today']} trades, "
            f"P/L: ${audit_data['total_pnl']:.2f}, "
            f"Open: {len(audit_data['open_positions'])}"
        )

        return audit_data

    def _run_daily_audit(self) -> None:
        """
        EQUITY-52: Run daily audit and send to Discord.

        Called by scheduler at 4:30 PM ET.
        Generates audit data and sends via all alerters that support it.
        """
        try:
            audit_data = self._generate_daily_audit()

            # Send to all alerters that have send_daily_audit method
            for alerter in self.alerters:
                if hasattr(alerter, 'send_daily_audit'):
                    try:
                        alerter.send_daily_audit(audit_data)
                    except Exception as e:
                        logger.error(f"Failed to send daily audit via {alerter.name}: {e}")

            logger.info("Daily audit completed and sent")
        except Exception as e:
            logger.error(f"Failed to run daily audit: {e}")
            self._error_count += 1

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
            # Session EQUITY-18: Add 15m and 30m scan jobs
            self.scheduler.add_15m_job(
                self._create_scan_callback('15m')
            )
            self.scheduler.add_30m_job(
                self._create_scan_callback('30m')
            )
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
            logger.info("Using separate scan jobs per timeframe (15m, 30m, 1H, 1D, 1W, 1M)")

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

        # EQUITY-52: Daily trade audit at 4:30 PM ET
        # Add as cron job via underlying APScheduler
        try:
            from apscheduler.triggers.cron import CronTrigger
            audit_trigger = CronTrigger(
                hour=16,
                minute=30,
                timezone='America/New_York'
            )
            self.scheduler._scheduler.add_job(
                func=self._run_daily_audit,
                trigger=audit_trigger,
                id='daily_audit',
                name='Daily Trade Audit',
                replace_existing=True
            )
            logger.info("Daily trade audit scheduled for 4:30 PM ET")
        except Exception as e:
            logger.warning(f"Failed to schedule daily audit: {e}")

        # Start scheduler
        self.scheduler.start()

        # Start API server if enabled (Session EQUITY-33)
        if self.config.api.enabled:
            self._start_api_server()

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

    def _start_api_server(self) -> None:
        """
        Start the REST API server in a background thread.

        Session EQUITY-33: Enables remote dashboard access to daemon data.
        """
        import threading

        try:
            from strat.signal_automation.api.server import init_api, run_api

            # Initialize API with this daemon instance
            init_api(self)

            # Start API server in background thread
            api_thread = threading.Thread(
                target=run_api,
                kwargs={
                    'host': self.config.api.host,
                    'port': self.config.api.port,
                    'debug': False,
                },
                daemon=True,  # Thread dies when main process exits
                name='equity-api-server',
            )
            api_thread.start()

            logger.info(
                f"API server started on {self.config.api.host}:{self.config.api.port}"
            )

        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
