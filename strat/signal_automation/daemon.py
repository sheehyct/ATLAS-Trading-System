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
import traceback
from datetime import datetime, timedelta
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
from strat.signal_automation.coordinators import (
    AlertManager,
    FilterManager,
    FilterConfig,
    ExecutionCoordinator,
    StaleSetupValidator,
)
from strat.signal_automation.utils import MarketHoursValidator
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
        self._market_hours_validator = MarketHoursValidator()  # EQUITY-86: Shared utility
        self.alerters: List[BaseAlerter] = []
        self.executor: Optional[SignalExecutor] = executor
        self.position_monitor: Optional[PositionMonitor] = position_monitor
        self.capital_tracker = None  # EQUITY-107: Virtual balance tracker
        self.entry_monitor: Optional[EntryMonitor] = None
        self._morning_report = None  # EQUITY-112: Pre-market morning report

        # Daemon state
        self._start_time: Optional[datetime] = None
        self._scan_count = 0
        self._signal_count = 0
        self._error_count = 0
        self._execution_count = 0
        self._exit_count = 0
        self._entry_alerts_sent: set = set()  # EQUITY-101: Dedup entry alerts
        self._executed_signal_keys: set = set()  # EQUITY-101: Dedup execution
        self._pdt_blocked_symbols: set = set()  # EQUITY-105: Skip PDT-blocked positions in EOD retries

        # Initialize components
        self._setup_alerters()
        self._setup_alert_manager()  # EQUITY-85: Facade pattern
        self._setup_filter_manager()  # EQUITY-87: Facade pattern
        self._setup_stale_validator()  # EQUITY-89: Facade pattern
        self._setup_executor()
        self._setup_capital_tracker()  # EQUITY-107: After executor, before coordinator
        self._setup_execution_coordinator()  # EQUITY-88: Facade pattern
        self._setup_position_monitor()
        self._setup_entry_monitor()
        self._setup_morning_report()  # EQUITY-112: After position_monitor

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

    def _setup_alert_manager(self) -> None:
        """
        Initialize AlertManager coordinator (EQUITY-85: Facade pattern).

        AlertManager delegates alert delivery to configured alerters.
        This enables clean separation of concerns and isolated testing.
        """
        self._alert_manager = AlertManager(
            alerters=self.alerters,
            signal_store=self.signal_store,
            config_alerts=self.config.alerts,
            is_market_hours_fn=self._is_market_hours,
            on_error=self._increment_error_count,
        )
        logger.info("AlertManager coordinator initialized")

    def _increment_error_count(self) -> None:
        """Callback for AlertManager to increment error count."""
        self._error_count += 1

    def _setup_filter_manager(self) -> None:
        """
        Initialize FilterManager coordinator (EQUITY-87: Facade pattern).

        FilterManager handles signal quality filtering with configurable thresholds.
        Extracts filter logic from _passes_filters() for isolated testing.
        """
        # Create filter config from environment variables
        filter_config = FilterConfig.from_env()

        self._filter_manager = FilterManager(
            config=filter_config,
            scan_config=self.config.scan,
        )
        logger.info("FilterManager coordinator initialized")

    def _setup_stale_validator(self) -> None:
        """
        Initialize StaleSetupValidator (EQUITY-89: Facade pattern).

        StaleSetupValidator checks if SETUP signals are still fresh or have
        become stale (new bar closed since detection).
        """
        self._stale_validator = StaleSetupValidator()
        logger.info("StaleSetupValidator initialized")

    def _setup_execution_coordinator(self) -> None:
        """
        Initialize ExecutionCoordinator (EQUITY-88: Facade pattern).

        ExecutionCoordinator handles signal execution, TFC re-evaluation,
        and intraday timing filters. Wired after executor setup so it can
        reference the executor instance.
        """
        self._execution_coordinator = ExecutionCoordinator(
            config=self.config.execution,
            executor=self.executor,
            signal_store=self.signal_store,
            tfc_evaluator=self.scanner,  # Scanner has evaluate_tfc()
            alerters=self.alerters,
            on_execution=self._increment_execution_count,
            on_error=self._increment_error_count,
        )
        logger.info("ExecutionCoordinator initialized")

    def _increment_execution_count(self) -> None:
        """Callback for ExecutionCoordinator to increment execution count."""
        self._execution_count += 1

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
            max_hourly_entries_per_day=self.config.execution.max_hourly_entries_per_day,
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

    def _setup_capital_tracker(self) -> None:
        """Initialize virtual balance tracker if enabled (Session EQUITY-107)."""
        if not hasattr(self.config, 'capital') or not self.config.capital.enabled:
            logger.info("Capital tracking disabled")
            return

        try:
            from strat.signal_automation.capital_tracker import VirtualBalanceTracker

            self.capital_tracker = VirtualBalanceTracker(
                virtual_capital=self.config.capital.virtual_capital,
                sizing_mode=self.config.capital.sizing_mode,
                fixed_dollar_amount=self.config.capital.fixed_dollar_amount,
                pct_of_capital=self.config.capital.pct_of_capital,
                max_portfolio_heat=self.config.capital.max_portfolio_heat,
                settlement_days=self.config.capital.settlement_days,
                state_file=self.config.capital.state_file,
            )
            self.capital_tracker.load()

            # Wire into executor if available
            if self.executor is not None:
                self.executor._capital_tracker = self.capital_tracker

            logger.info(
                f"Capital tracker initialized: "
                f"virtual=${self.capital_tracker._virtual_capital:.2f}, "
                f"available=${self.capital_tracker.available_capital:.2f}, "
                f"mode={self.config.capital.sizing_mode}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize capital tracker: {e}")
            self.capital_tracker = None

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
            capital_tracker=self.capital_tracker,  # EQUITY-107
        )

        logger.info("Position monitor initialized")

    def _on_position_exit(
        self,
        exit_signal: ExitSignal,
        order_result: Dict[str, Any]
    ) -> None:
        """
        Callback when a position is exited (Session 83K-49).

        EQUITY-85: Delegates exit alerts to AlertManager coordinator.
        """
        self._exit_count += 1

        # Get signal details for the alert
        signal = self.signal_store.get_signal(exit_signal.signal_key) if exit_signal.signal_key else None

        # Delegate to AlertManager (EQUITY-85: Facade pattern)
        self._alert_manager.send_exit_alert(exit_signal, order_result, signal)

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

    def _setup_morning_report(self) -> None:
        """Initialize morning report generator (EQUITY-112)."""
        if not self.config.morning_report.enabled:
            logger.info("Morning report disabled")
            return

        try:
            from strat.signal_automation.coordinators.morning_report import (
                MorningReportGenerator,
            )

            # Get trading client from executor if available
            trading_client = None
            if self.executor and hasattr(self.executor, '_trading_client'):
                trading_client = self.executor._trading_client

            self._morning_report = MorningReportGenerator(
                alerters=self.alerters,
                position_monitor=self.position_monitor,
                capital_tracker=self.capital_tracker,
                trading_client=trading_client,
                config=self.config.morning_report,
            )

            logger.info("Morning report generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize morning report: {e}")
            self._morning_report = None

    def _on_entry_triggered(self, event: TriggerEvent) -> None:
        """
        Callback when an entry trigger is hit (Session 83K-66).

        Executes the signal if executor is available and respects position limits.
        Signals are already sorted by priority (1M > 1W > 1D > 1H).
        """
        signal = event.signal

        # EQUITY-113: Save old key before any direction/pattern mutations
        old_signal_key = signal.signal_key

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

        # EQUITY-105: Resolve bidirectional pattern type on entry trigger
        # "3-?" + CALL = "3-2U", "3-?" + PUT = "3-2D"
        # "3-1-?" + CALL = "3-1-2U", "3-1-?" + PUT = "3-1-2D"
        # "2-1-?" + CALL = "2-1-2U", "2-1-?" + PUT = "2-1-2D"
        if '?' in signal.pattern_type:
            old_pattern = signal.pattern_type
            direction_suffix = '2U' if actual_direction == 'CALL' else '2D'
            signal.pattern_type = signal.pattern_type.replace('?', direction_suffix)
            logger.info(
                f"PATTERN RESOLVED: {signal.symbol} {old_pattern} -> "
                f"{signal.pattern_type} ({actual_direction} break)"
            )

        # EQUITY-113: Regenerate signal_key after direction/pattern mutations.
        # The key encodes pattern_type and direction, so it must be rebuilt
        # when either changes. Uses the same timestamp truncation as generate_key().
        dt = signal.detected_time
        if signal.timeframe == '1H':
            truncated = dt.replace(minute=0, second=0, microsecond=0)
        elif signal.timeframe == '1D':
            truncated = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif signal.timeframe == '1W':
            truncated = dt - timedelta(days=dt.weekday())
            truncated = truncated.replace(hour=0, minute=0, second=0, microsecond=0)
        elif signal.timeframe == '1M':
            truncated = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            truncated = dt
        timestamp_str = truncated.strftime('%Y%m%d%H%M')
        new_signal_key = (
            f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_"
            f"{signal.direction}_{timestamp_str}"
        )

        if new_signal_key != old_signal_key:
            self.signal_store.rekey_signal(old_signal_key, new_signal_key)
            signal.signal_key = new_signal_key
            logger.info(f"SIGNAL REKEYED: {old_signal_key} -> {new_signal_key}")

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

        # EQUITY-102: Write re-evaluated TFC back to signal store so
        # trade_metadata and downstream consumers get correct TFC values
        tfc_assessment = self._execution_coordinator.last_tfc_assessment
        if tfc_assessment is not None:
            tfc_score, tfc_alignment = tfc_assessment
            self.signal_store.update_tfc(signal.signal_key, tfc_score, tfc_alignment)

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
            # EQUITY-101: Prevent duplicate execution of same signal
            if signal.signal_key in self._executed_signal_keys:
                logger.info(f"Skipping duplicate execution: {signal.signal_key}")
                return
            self._executed_signal_keys.add(signal.signal_key)

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
                    # EQUITY-101: Dedup - skip if alert already sent for this signal
                    if signal.signal_key not in self._entry_alerts_sent:
                        self._entry_alerts_sent.add(signal.signal_key)
                        for alerter in self.alerters:
                            try:
                                if isinstance(alerter, DiscordAlerter):
                                    alerter.send_entry_alert(signal, result)
                                elif isinstance(alerter, LoggingAlerter):
                                    alerter.log_execution(result)
                            except Exception as e:
                                logger.error(f"Alert error: {e}")
                    else:
                        logger.debug(f"Skipping duplicate entry alert: {signal.signal_key}")

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

            # Scan each symbol (dynamic via ticker selection or static)
            for symbol in self.active_symbols:
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

                            # EQUITY-101: Skip if already executed via another path
                            if signal.signal_key in self._executed_signal_keys:
                                logger.debug(f"Skipping already-executed TRIGGERED: {signal.signal_key}")
                                continue
                            self._executed_signal_keys.add(signal.signal_key)

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

            # Scan each symbol using resampling (dynamic via ticker selection or static)
            for symbol in self.active_symbols:
                try:
                    detected = self._scan_symbol_resampled(symbol)
                    new_signals.extend(detected)
                except Exception as e:
                    logger.error(f"Error scanning {symbol} (resampled): {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
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

                            # EQUITY-101: Skip if already executed via another path
                            if signal.signal_key in self._executed_signal_keys:
                                logger.debug(f"Skipping already-executed TRIGGERED: {signal.signal_key}")
                                continue
                            self._executed_signal_keys.add(signal.signal_key)

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

        Session EQUITY-87: Delegates to FilterManager coordinator.
        Session EQUITY-47: Logging for filter rejections with actual vs threshold values.

        Args:
            signal: Signal to check

        Returns:
            True if passes all filters
        """
        # EQUITY-87: Delegate to FilterManager coordinator
        return self._filter_manager.passes_filters(signal)

    def _is_market_hours(self) -> bool:
        """
        Check if current time is within NYSE market hours.

        Session EQUITY-34: Uses pandas_market_calendars for accurate holiday
        and early close handling (e.g., Christmas Eve 1PM, Thanksgiving Friday 1PM).
        Session EQUITY-86: Delegates to shared MarketHoursValidator utility.

        Returns:
            True if within market hours, False otherwise
        """
        return self._market_hours_validator.is_market_hours()

    def _is_setup_stale(self, signal: StoredSignal) -> tuple[bool, str]:
        """
        Check if a SETUP signal is stale (new bar has closed since detection).

        Session EQUITY-46: Fix for stale setup bug where setups from N days ago
        triggered when they should have been invalidated by intervening bars.
        Session EQUITY-89: Delegates to StaleSetupValidator coordinator.

        Args:
            signal: The stored signal to check

        Returns:
            Tuple of (is_stale: bool, reason: str)
        """
        return self._stale_validator.is_setup_stale(signal)

    def _reevaluate_tfc_at_entry(self, signal: StoredSignal) -> tuple[bool, str]:
        """
        Re-evaluate TFC alignment at entry time and check if entry should be blocked.

        Session EQUITY-49: TFC Re-evaluation at Entry.
        Session EQUITY-88: Delegates to ExecutionCoordinator.

        Args:
            signal: The stored signal about to be executed

        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        return self._execution_coordinator.reevaluate_tfc_at_entry(signal)

    def _send_alerts(self, signals: List[StoredSignal]) -> None:
        """
        Send alerts for new signals.

        EQUITY-85: Delegates to AlertManager coordinator (Facade pattern).
        AlertManager handles market hours blocking, signal sorting,
        Discord config flags, and alerter error handling.

        Args:
            signals: Signals to alert
        """
        self._alert_manager.send_signal_alerts(signals)

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol using executor's trading client.

        Session EQUITY-32: Helper for triggered pattern execution.
        Session EQUITY-88: Delegates to ExecutionCoordinator.

        Args:
            symbol: Stock symbol

        Returns:
            Current mid price or None if unavailable
        """
        return self._execution_coordinator._get_current_price(symbol)

    def _execute_triggered_pattern(self, signal: StoredSignal) -> Optional[ExecutionResult]:
        """
        Execute a TRIGGERED pattern immediately at market price.

        Session EQUITY-32: Patterns where entry bar has formed should
        execute at current market price, not be discarded.
        Session EQUITY-88: Delegates to ExecutionCoordinator.

        Args:
            signal: TRIGGERED signal (signal_type="COMPLETED")

        Returns:
            ExecutionResult if executed, None if skipped
        """
        return self._execution_coordinator.execute_triggered_pattern(signal)

    def _is_intraday_entry_allowed(self, signal: StoredSignal) -> bool:
        """
        Check if intraday pattern entry is allowed based on "let the market breathe" rules.

        Session EQUITY-18: Extended to support 15m, 30m, and 1H timeframes.
        Session EQUITY-88: Delegates to ExecutionCoordinator.

        Args:
            signal: The signal to check

        Returns:
            True if entry is allowed, False if too early
        """
        return self._execution_coordinator.is_intraday_entry_allowed(signal)

    def _execute_signals(
        self,
        signals: List[StoredSignal]
    ) -> List[ExecutionResult]:
        """
        Execute signals via the executor (Session 83K-48).

        Session EQUITY-88: Delegates to ExecutionCoordinator.

        Args:
            signals: Signals to execute

        Returns:
            List of execution results
        """
        return self._execution_coordinator.execute_signals(signals)

    def execute_signals(
        self,
        signals: List[StoredSignal]
    ) -> List[ExecutionResult]:
        """
        Public method to execute signals (Session 83K-48).

        Allows manual execution of signals from CLI.
        Session EQUITY-88: Delegates to ExecutionCoordinator.

        Args:
            signals: Signals to execute

        Returns:
            List of execution results
        """
        if self.executor is None:
            logger.error("Execution not enabled - set SIGNAL_EXECUTION_ENABLED=true")
            return []

        return self._execution_coordinator.execute_signals(signals)

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

        # Add capital tracking status (EQUITY-107)
        if self.capital_tracker is not None:
            status['capital'] = self.capital_tracker.get_summary()

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

        EQUITY-95: Switched from paper_trades.json to Alpaca get_closed_trades()
        for live, accurate trade data.

        Returns:
            Dictionary with audit data for Discord reporting
        """
        from datetime import date, datetime, timezone

        today = date.today()
        today_iso = today.isoformat()

        # Initialize audit data
        audit_data = {
            'date': today_iso,
            'trades_today': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'profit_factor': 0.0,
            'open_positions': [],
            'anomalies': [],
        }

        # EQUITY-95: Get closed trades from Alpaca instead of paper_trades.json
        # This ensures we have accurate, live data from the broker
        if self.executor and hasattr(self.executor, '_trading_client') and self.executor._trading_client:
            try:
                # Get start of today in UTC for the Alpaca query
                start_of_today = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
                closed_trades = self.executor._trading_client.get_closed_trades(
                    after=start_of_today,
                    options_only=True
                )

                # Process today's closed trades
                gross_profit = 0.0
                gross_loss = 0.0
                for trade in closed_trades:
                    # Filter to today's trades (sell_time_dt is the close time)
                    sell_time = trade.get('sell_time_dt')
                    if sell_time and sell_time.date() == today:
                        audit_data['trades_today'] += 1
                        pnl = trade.get('realized_pnl', 0)
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

            except Exception as e:
                logger.error(f"Failed to get closed trades from Alpaca: {e}")
                audit_data['anomalies'].append(f"Failed to get trades from Alpaca: {e}")
        else:
            logger.warning("No trading client available for audit - executor not configured")
            audit_data['anomalies'].append("Trading client not available")

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

        # EQUITY-107: Add capital tracking summary to audit
        if self.capital_tracker is not None:
            audit_data['capital'] = self.capital_tracker.get_summary()

        # EQUITY-112: Add MFE/MAE excursion stats from TradeStore
        excursion_stats = self._get_today_excursion_stats(today)
        if excursion_stats:
            audit_data['excursion'] = excursion_stats

        logger.info(
            f"Daily audit generated: {audit_data['trades_today']} trades, "
            f"P/L: ${audit_data['total_pnl']:.2f}, "
            f"Open: {len(audit_data['open_positions'])}"
        )

        return audit_data

    def _get_today_excursion_stats(self, today) -> Optional[Dict[str, Any]]:
        """
        EQUITY-112: Query TradeStore for today's closed trades and compute
        aggregate MFE/MAE excursion statistics.

        Returns:
            Dict with avg_mfe, avg_mae, avg_exit_efficiency, losers_went_green
            or None if no data available.
        """
        if self.position_monitor is None:
            return None

        store = self.position_monitor.get_trade_store()
        if store is None:
            return None

        try:
            start_of_today = datetime.combine(today, datetime.min.time())
            end_of_today = start_of_today + timedelta(days=1)

            trades = store.get_trades(after=start_of_today, before=end_of_today)
            if not trades:
                return None

            # Filter to trades with excursion data
            with_excursion = [
                t for t in trades
                if t.excursion and t.excursion.mfe_pnl is not None
            ]
            if not with_excursion:
                return None

            total_mfe = sum(t.excursion.mfe_pnl for t in with_excursion)
            total_mae = sum(t.excursion.mae_pnl for t in with_excursion)
            n = len(with_excursion)

            # Exit efficiency only for winners (mfe > 0)
            efficiencies = [
                t.excursion.exit_efficiency
                for t in with_excursion
                if t.excursion.mfe_pnl > 0 and t.excursion.exit_efficiency is not None
            ]
            avg_efficiency = (
                sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
            )

            losers = [t for t in with_excursion if t.pnl < 0]
            losers_went_green = sum(
                1 for t in losers if t.excursion.went_green_before_loss
            )

            return {
                'trades_with_excursion': n,
                'avg_mfe': total_mfe / n,
                'avg_mae': total_mae / n,
                'avg_exit_efficiency': avg_efficiency,
                'losers_went_green': losers_went_green,
                'total_losers': len(losers),
            }
        except Exception as e:
            logger.warning(f"Failed to compute excursion stats: {e}")
            return None

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

    def _run_morning_report(self) -> None:
        """
        EQUITY-112: Run pre-market morning report and send to Discord.

        Called by scheduler at configured time (default 6:00 AM ET).
        """
        if self._morning_report is None:
            return

        try:
            self._morning_report.run()
        except Exception as e:
            logger.error(f"Morning report failed: {e}")
            self._error_count += 1

    def _run_eod_exit_job(self) -> None:
        """
        EQUITY-100: Dedicated EOD exit job for 1H positions.

        Runs at 15:50, 15:53, 15:55, 15:57, 15:59 ET with retry logic.
        Multiple scheduled attempts ensure 1H positions exit before market close.
        """
        if self.position_monitor is None:
            return

        try:
            # Get all tracked positions
            positions = self.position_monitor.get_tracked_positions()
            hourly_positions = [
                p for p in positions
                if p.timeframe and p.timeframe.upper() in ['1H', '60MIN', '60M']
            ]

            if not hourly_positions:
                logger.debug("EOD exit job: No 1H positions to exit")
                return

            logger.warning(
                f"EOD EXIT JOB: Found {len(hourly_positions)} 1H position(s) to exit"
            )

            for pos in hourly_positions:
                # EQUITY-105: Skip positions already blocked by PDT this session
                if pos.osi_symbol in self._pdt_blocked_symbols:
                    logger.warning(
                        f"EOD EXIT SKIPPED: {pos.signal_key} ({pos.osi_symbol}) - "
                        f"PDT blocked earlier today, will exit at 9:31 AM tomorrow"
                    )
                    continue

                # Create exit signal for EOD
                exit_signal = ExitSignal(
                    osi_symbol=pos.osi_symbol,
                    signal_key=pos.signal_key,
                    reason=ExitReason.EOD_EXIT,
                    underlying_price=pos.underlying_price or 0.0,
                    current_option_price=pos.current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    dte=pos.dte,
                    details="Dedicated EOD exit job for 1H trade",
                )

                # Execute with retry
                result = self._execute_eod_exit_with_retry(exit_signal, max_retries=3)

                if result:
                    self._exit_count += 1
                    logger.info(f"EOD EXIT SUCCESS: {pos.signal_key} ({pos.osi_symbol})")
                else:
                    logger.error(
                        f"EOD EXIT FAILED: {pos.signal_key} ({pos.osi_symbol}) - "
                        f"will retry on next scheduled job"
                    )

        except Exception as e:
            logger.error(f"EOD exit job error: {e}")
            self._error_count += 1

    def _execute_eod_exit_with_retry(
        self,
        exit_signal: ExitSignal,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Optional[Dict[str, Any]]:
        """
        EQUITY-100: Execute EOD exit with retry logic and exponential backoff.
        EQUITY-105: Added PDT error detection with immediate Discord alerting.

        Args:
            exit_signal: ExitSignal to execute
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)

        Returns:
            Order result if successful, None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                result = self.position_monitor.execute_exit(exit_signal)
                if result:
                    return result

                # Failed but no exception - wait and retry
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"EOD exit attempt {attempt + 1} failed for {exit_signal.osi_symbol}, "
                        f"retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)

            except Exception as e:
                error_str = str(e).lower()

                # EQUITY-105: PDT rejection from Alpaca (code 40310000)
                # Account-level restriction - retrying won't help, alert and bail.
                # Register symbol to skip on subsequent EOD cron jobs today.
                if 'pattern day trading' in error_str:
                    logger.critical(
                        f"PDT PROTECTION BLOCKED EOD EXIT: {exit_signal.osi_symbol} - "
                        f"Account flagged for pattern day trading. "
                        f"Ensure account equity >= $25K or reset paper account."
                    )
                    self._pdt_blocked_symbols.add(exit_signal.osi_symbol)
                    self._send_pdt_alert(exit_signal)
                    return None

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"EOD exit attempt {attempt + 1} error for {exit_signal.osi_symbol}: {e}, "
                        f"retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"EOD exit failed after {max_retries} attempts for "
                        f"{exit_signal.osi_symbol}: {e}"
                    )

        return None

    def _send_pdt_alert(self, exit_signal: ExitSignal) -> None:
        """
        EQUITY-105: Send critical Discord alert when PDT blocks an EOD exit.
        """
        error_msg = (
            f"**EOD EXIT BLOCKED BY PDT**\n\n"
            f"Position: `{exit_signal.osi_symbol}`\n"
            f"Signal: `{exit_signal.signal_key}`\n"
            f"P&L: ${exit_signal.unrealized_pnl:.2f}\n\n"
            f"Pattern Day Trading protection rejected the exit order. "
            f"This position will carry overnight and exit at 9:31 AM ET tomorrow.\n\n"
            f"**Action Required:** Ensure paper account equity >= $25,000 "
            f"or reset the account to avoid PDT restrictions."
        )
        alert_sent = False
        for alerter in self.alerters:
            if hasattr(alerter, 'send_error_alert'):
                try:
                    alerter.send_error_alert('PDT_BLOCKED', error_msg)
                    alert_sent = True
                except Exception as e:
                    logger.error(f"Failed to send PDT alert via {alerter.name}: {e}")

        if not alert_sent:
            logger.critical(
                f"PDT ALERT COULD NOT BE DELIVERED - no alerter supports send_error_alert. "
                f"Position {exit_signal.osi_symbol} is stuck. Manual intervention required."
            )

    def _run_market_open_stale_check(self) -> None:
        """
        EQUITY-100: Market open stale position check.

        Runs at 9:31 AM ET to detect 1H positions that survived overnight
        and exits them immediately. This is a safety net for positions that
        escaped the EOD exit window.
        """
        # EQUITY-107: Settle pending capital at market open
        if self.capital_tracker is not None:
            try:
                self.capital_tracker.settle_pending()
                logger.info("Capital settlements processed at market open")
            except Exception as e:
                logger.error(f"Capital settlement error: {e}")

        # EQUITY-105: Clear PDT blocked set for new trading day
        if self._pdt_blocked_symbols:
            logger.info(
                f"Clearing {len(self._pdt_blocked_symbols)} PDT-blocked symbol(s) "
                f"from previous day: {self._pdt_blocked_symbols}"
            )
            self._pdt_blocked_symbols.clear()

        if self.position_monitor is None:
            return

        try:
            positions = self.position_monitor.get_tracked_positions()

            for pos in positions:
                # Only check 1H positions
                if not pos.timeframe or pos.timeframe.upper() not in ['1H', '60MIN', '60M']:
                    continue

                # Check if position is stale (entered on previous trading day)
                if self.position_monitor._is_stale_1h_position(pos.entry_time):
                    logger.warning(
                        f"STALE 1H DETECTED at market open: {pos.signal_key} "
                        f"({pos.osi_symbol}) - entered {pos.entry_time}"
                    )

                    # Force exit this stale position
                    exit_signal = ExitSignal(
                        osi_symbol=pos.osi_symbol,
                        signal_key=pos.signal_key,
                        reason=ExitReason.EOD_EXIT,
                        underlying_price=pos.underlying_price or 0.0,
                        current_option_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        dte=pos.dte,
                        details=f"STALE 1H position - entered {pos.entry_time.strftime('%Y-%m-%d')}, exiting at market open",
                    )

                    result = self._execute_eod_exit_with_retry(exit_signal, max_retries=3)

                    if result:
                        self._exit_count += 1
                        logger.info(f"STALE 1H EXITED: {pos.signal_key} ({pos.osi_symbol})")
                    else:
                        logger.error(f"STALE 1H EXIT FAILED: {pos.signal_key} ({pos.osi_symbol})")

        except Exception as e:
            logger.error(f"Market open stale check error: {e}")
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

        # EQUITY-112: Pre-market morning report
        if self._morning_report is not None:
            try:
                from apscheduler.triggers.cron import CronTrigger

                mr_config = self.config.morning_report
                morning_trigger = CronTrigger(
                    hour=mr_config.hour,
                    minute=mr_config.minute,
                    day_of_week='mon-fri',
                    timezone='America/New_York',
                )
                self.scheduler._scheduler.add_job(
                    func=self._run_morning_report,
                    trigger=morning_trigger,
                    id='morning_report',
                    name='Pre-Market Morning Report',
                    replace_existing=True,
                )
                logger.info(
                    f"Morning report scheduled for "
                    f"{mr_config.hour}:{mr_config.minute:02d} AM ET"
                )
            except Exception as e:
                logger.warning(f"Failed to schedule morning report: {e}")

        # EQUITY-100: Dedicated EOD exit jobs for 1H positions
        # Multiple scheduled times ensure exits happen before market close
        if self.position_monitor is not None:
            try:
                from apscheduler.triggers.cron import CronTrigger

                # Schedule EOD exit jobs at 15:50, 15:53, 15:55, 15:57, 15:59 ET
                eod_times = [
                    (15, 50),  # First attempt - 10 min buffer
                    (15, 53),  # Second attempt
                    (15, 55),  # Third attempt (original EOD time)
                    (15, 57),  # Fourth attempt
                    (15, 59),  # Final attempt - 1 min buffer
                ]

                for hour, minute in eod_times:
                    job_id = f'eod_exit_{hour}_{minute:02d}'
                    eod_trigger = CronTrigger(
                        hour=hour,
                        minute=minute,
                        day_of_week='mon-fri',
                        timezone='America/New_York'
                    )
                    self.scheduler._scheduler.add_job(
                        func=self._run_eod_exit_job,
                        trigger=eod_trigger,
                        id=job_id,
                        name=f'EOD Exit {hour}:{minute:02d} ET',
                        replace_existing=True
                    )

                logger.info("EOD exit jobs scheduled at 15:50, 15:53, 15:55, 15:57, 15:59 ET")
            except Exception as e:
                logger.warning(f"Failed to schedule EOD exit jobs: {e}")

        # EQUITY-100: Market open stale 1H position check
        # Safety net to exit positions that survived overnight
        if self.position_monitor is not None:
            try:
                from apscheduler.triggers.cron import CronTrigger

                stale_check_trigger = CronTrigger(
                    hour=9,
                    minute=31,
                    day_of_week='mon-fri',
                    timezone='America/New_York'
                )
                self.scheduler._scheduler.add_job(
                    func=self._run_market_open_stale_check,
                    trigger=stale_check_trigger,
                    id='market_open_stale_check',
                    name='Market Open Stale 1H Check',
                    replace_existing=True
                )

                logger.info("Market open stale 1H check scheduled for 9:31 AM ET")
            except Exception as e:
                logger.warning(f"Failed to schedule market open stale check: {e}")

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

        EQUITY-85: Delegates to AlertManager coordinator.

        Returns:
            Dictionary of alerter_name -> test_passed
        """
        return self._alert_manager.test_alerters()

    @property
    def active_symbols(self) -> List[str]:
        """
        Return the list of symbols to scan this cycle.

        If ticker_selection is enabled, merges dynamic candidates with
        core ETFs.  Falls back to the static config list on any failure.
        """
        if not self.config.ticker_selection.enabled:
            return self.config.scan.symbols

        candidates = self._load_candidates()
        if candidates is None:
            logger.warning(
                "Ticker selection enabled but candidates unavailable - "
                "falling back to static symbol list"
            )
            return self.config.scan.symbols

        # Core ETFs + dynamic candidates (deduplicated, capped)
        core = list(self.config.ticker_selection.core_symbols)
        dynamic_syms = [
            c['symbol'] for c in candidates
            if c.get('symbol') not in core
        ]
        cap = self.config.ticker_selection.max_dynamic_symbols
        symbols = core + dynamic_syms[:cap]

        logger.info(
            f"Active symbols: {len(core)} core + "
            f"{min(len(dynamic_syms), cap)} dynamic = {len(symbols)} total"
        )
        return symbols

    def _load_candidates(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load candidates from candidates.json.

        Returns None if file is missing, stale, or corrupt.
        """
        import json
        from pathlib import Path
        from datetime import datetime, timezone

        path = Path(self.config.ticker_selection.candidates_path)
        if not path.exists():
            logger.debug(f"Candidates file not found: {path}")
            return None

        try:
            data = json.loads(path.read_text())

            # Reject synthetic data (test-only, never trade on it)
            if data.get('pipeline_stats', {}).get('synthetic'):
                logger.warning(
                    "Candidates file contains synthetic data -- ignoring"
                )
                return None

            # Check freshness
            generated_at = data.get('generated_at', '')
            if generated_at:
                gen_time = datetime.fromisoformat(generated_at)
                if gen_time.tzinfo is None:
                    gen_time = gen_time.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - gen_time).total_seconds()
                if age > self.config.ticker_selection.stale_threshold_seconds:
                    logger.warning(
                        f"Candidates stale: {age / 3600:.1f}h old "
                        f"(threshold: {self.config.ticker_selection.stale_threshold_seconds / 3600:.0f}h)"
                    )
                    return None

            candidates = data.get('candidates', [])
            if not candidates:
                logger.debug("Candidates file has no candidates")
                return None

            return candidates

        except Exception as e:
            logger.error(f"Failed to load candidates: {e}")
            return None

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
