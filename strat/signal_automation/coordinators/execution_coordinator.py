"""
EQUITY-88: ExecutionCoordinator - Extracted from SignalDaemon

Manages signal execution, TFC re-evaluation, and intraday timing filters.

Responsibilities:
- Execute TRIGGERED patterns at market price
- Execute SETUP signals after entry monitor triggers
- Re-evaluate TFC alignment at entry time
- Apply "Let the Market Breathe" intraday timing filters
- Send entry alerts via configured alerters
"""

import logging
from datetime import datetime, time as dt_time
from typing import Optional, List, Dict, Any, Callable, Protocol

from strat.signal_automation.config import ExecutionConfig
from strat.signal_automation.signal_store import SignalStore, StoredSignal, SignalType, SignalStatus
from strat.signal_automation.executor import (
    SignalExecutor,
    ExecutionResult,
    ExecutionState,
)
from strat.signal_automation.alerters import DiscordAlerter, LoggingAlerter, BaseAlerter

logger = logging.getLogger(__name__)


class TFCEvaluator(Protocol):
    """Protocol for TFC evaluation - allows scanner injection."""

    def evaluate_tfc(
        self,
        symbol: str,
        detection_timeframe: str,
        direction: int
    ) -> Any:
        """Evaluate TFC for a symbol."""
        ...


class PriceFetcher(Protocol):
    """Protocol for fetching current prices."""

    def get_stock_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get stock quotes for symbols."""
        ...


class ExecutionCoordinator:
    """
    Coordinates signal execution for the signal daemon.

    Extracted from SignalDaemon as part of EQUITY-88 Phase 3 refactoring.
    Uses Facade pattern - daemon delegates execution to this coordinator.

    Args:
        config: ExecutionConfig with TFC re-eval and execution settings
        executor: SignalExecutor for submitting orders
        signal_store: SignalStore for updating signal status
        tfc_evaluator: Scanner or similar for TFC re-evaluation
        alerters: List of alerters for sending entry alerts
        on_execution: Callback when execution count increments
        on_error: Callback when error count increments
    """

    def __init__(
        self,
        config: ExecutionConfig,
        executor: Optional[SignalExecutor] = None,
        signal_store: Optional[SignalStore] = None,
        tfc_evaluator: Optional[TFCEvaluator] = None,
        alerters: Optional[List[BaseAlerter]] = None,
        on_execution: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[], None]] = None,
    ):
        self._config = config
        self._executor = executor
        self._signal_store = signal_store
        self._tfc_evaluator = tfc_evaluator
        self._alerters = alerters or []
        self._on_execution = on_execution or (lambda: None)
        self._on_error = on_error or (lambda: None)

    def execute_triggered_pattern(self, signal: StoredSignal) -> Optional[ExecutionResult]:
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
        if self._executor is None:
            return None

        # Get current market price
        current_price = self._get_current_price(signal.symbol)
        if current_price is None or current_price <= 0:
            logger.warning(f"Could not get price for {signal.symbol} - skipping triggered pattern")
            return None

        direction = signal.direction
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
        if not self.is_intraday_entry_allowed(signal):
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

            result = self._executor.execute_signal(signal, underlying_price=current_price)

            signal.status = original_status  # Restore original status

            if result.state == ExecutionState.ORDER_SUBMITTED:
                self._on_execution()
                if self._signal_store:
                    self._signal_store.mark_triggered(signal.signal_key)

                    if result.osi_symbol:
                        self._signal_store.set_executed_osi_symbol(
                            signal.signal_key, result.osi_symbol
                        )

                logger.info(
                    f"TRADE OPENED (TRIGGERED): {signal.symbol} {signal.direction} "
                    f"{signal.pattern_type} ({signal.timeframe}) - {result.osi_symbol}"
                )

                # Send Discord entry alert
                self._send_entry_alerts(signal, result)

                return result
            else:
                logger.info(
                    f"TRIGGERED execution skipped/failed: {signal.symbol} - {result.error}"
                )
                return result

        except Exception as e:
            logger.error(f"Failed to execute triggered pattern: {e}")
            return None

    def execute_signals(self, signals: List[StoredSignal]) -> List[ExecutionResult]:
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
        if self._executor is None:
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

            # Session EQUITY-42: COMPLETED signals already executed by execute_triggered_pattern()
            # Skip them here to prevent duplicate execution and duplicate Discord alerts
            if getattr(signal, 'signal_type', 'COMPLETED') == 'COMPLETED':
                logger.debug(
                    f"COMPLETED signal {signal.signal_key} skipped - "
                    f"already executed by execute_triggered_pattern()"
                )
                continue

            # "Let the Market Breathe" - Skip intraday patterns if too early in session
            if not self.is_intraday_entry_allowed(signal):
                results.append(ExecutionResult(
                    signal_key=signal.signal_key,
                    state=ExecutionState.SKIPPED,
                    error=f"Intraday {signal.timeframe} pattern blocked - too early in session (let market breathe)"
                ))
                continue

            try:
                result = self._executor.execute_signal(signal)
                results.append(result)

                if result.state == ExecutionState.ORDER_SUBMITTED:
                    self._on_execution()
                    # Mark signal as triggered in store
                    if self._signal_store:
                        self._signal_store.mark_triggered(signal.signal_key)
                        # Store OSI symbol for closed trade correlation
                        if result.osi_symbol:
                            self._signal_store.set_executed_osi_symbol(
                                signal.signal_key, result.osi_symbol
                            )
                    logger.info(
                        f"Order submitted for {signal.signal_key}: "
                        f"{result.osi_symbol}"
                    )

                    # Send Discord entry alert
                    self._send_entry_alerts(signal, result)

                elif result.state == ExecutionState.SKIPPED:
                    logger.debug(
                        f"Signal skipped: {signal.signal_key} - {result.error}"
                    )
                elif result.state == ExecutionState.FAILED:
                    logger.warning(
                        f"Execution failed: {signal.signal_key} - {result.error}"
                    )
                    self._on_error()

            except Exception as e:
                logger.exception(f"Execution error for {signal.signal_key}: {e}")
                self._on_error()
                results.append(ExecutionResult(
                    signal_key=signal.signal_key,
                    state=ExecutionState.FAILED,
                    error=str(e)
                ))

        return results

    def is_intraday_entry_allowed(self, signal: StoredSignal) -> bool:
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

    def reevaluate_tfc_at_entry(self, signal: StoredSignal) -> tuple[bool, str]:
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
        if not self._config.tfc_reeval_enabled:
            return False, ""

        # Need TFC evaluator for re-evaluation
        if self._tfc_evaluator is None:
            logger.debug("TFC REEVAL: No TFC evaluator configured - skipping re-evaluation")
            return False, ""

        # Get original TFC data from signal with defensive validation (Issue #2 fix)
        original_strength = signal.tfc_score if signal.tfc_score is not None else 0
        original_alignment = signal.tfc_alignment or ""  # Handle None
        # Session EQUITY-62: Default to False (fail-closed, not fail-open)
        original_passes = signal.passes_flexible if signal.passes_flexible is not None else False

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
            current_tfc = self._tfc_evaluator.evaluate_tfc(
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
            self._on_error()
            return False, ""
        except Exception as e:
            # Unexpected error - log as error, increment counter, but still proceed
            # (fail-open to avoid blocking all entries on system errors)
            logger.error(
                f"TFC REEVAL UNEXPECTED ERROR: {signal.symbol} {signal.pattern_type} - "
                f"{type(e).__name__}: {e} (proceeding with entry)"
            )
            self._on_error()
            return False, ""

        # Validate returned assessment (Issue #5 fix)
        if current_tfc is None or not hasattr(current_tfc, 'strength'):
            logger.error(f"TFC REEVAL: Invalid assessment returned for {signal.symbol} - proceeding with entry")
            self._on_error()
            return False, ""

        current_strength = current_tfc.strength if current_tfc.strength is not None else 0
        current_alignment = current_tfc.alignment_label() if hasattr(current_tfc, 'alignment_label') else f"{current_strength}/?"
        # Session EQUITY-62: Default to False for safety (fail-closed)
        current_passes = getattr(current_tfc, 'passes_flexible', False)
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
        if self._config.tfc_reeval_log_always:
            if strength_delta < 0 or direction_flipped:
                logger.warning(comparison)
            else:
                logger.info(comparison)

        # Determine if entry should be blocked
        should_block = False
        block_reason = ""

        # Block if direction flipped (most severe)
        if direction_flipped and self._config.tfc_reeval_block_on_flip:
            should_block = True
            block_reason = f"TFC direction flipped from {original_tfc_direction} to {current_direction}"

        # Block if strength dropped below minimum threshold
        elif current_strength < self._config.tfc_reeval_min_strength:
            should_block = True
            block_reason = f"TFC strength {current_strength} < min threshold {self._config.tfc_reeval_min_strength}"

        return should_block, block_reason

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol using executor's trading client.

        Session EQUITY-32: Helper for triggered pattern execution.

        Args:
            symbol: Stock symbol

        Returns:
            Current mid price or None if unavailable
        """
        if self._executor is None:
            return None
        try:
            quotes = self._executor._trading_client.get_stock_quotes([symbol])
            if symbol in quotes:
                quote = quotes[symbol]
                if isinstance(quote, dict) and 'mid' in quote:
                    return quote['mid']
                elif isinstance(quote, (int, float)):
                    return float(quote)
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
        return None

    def _send_entry_alerts(self, signal: StoredSignal, result: ExecutionResult) -> None:
        """
        Send entry alerts via configured alerters.

        Args:
            signal: The executed signal
            result: The execution result
        """
        for alerter in self._alerters:
            try:
                if isinstance(alerter, DiscordAlerter):
                    alerter.send_entry_alert(signal, result)
                elif isinstance(alerter, LoggingAlerter):
                    alerter.log_execution(result)
            except Exception as e:
                logger.error(f"Entry alert error: {e}")

    @property
    def config(self) -> ExecutionConfig:
        """Return current execution configuration."""
        return self._config

    @property
    def executor(self) -> Optional[SignalExecutor]:
        """Return the executor instance."""
        return self._executor

    def set_executor(self, executor: SignalExecutor) -> None:
        """
        Set the executor instance.

        Allows deferred executor setup after coordinator creation.

        Args:
            executor: SignalExecutor to use
        """
        self._executor = executor

    def set_tfc_evaluator(self, evaluator: TFCEvaluator) -> None:
        """
        Set the TFC evaluator instance.

        Allows deferred evaluator setup after coordinator creation.

        Args:
            evaluator: TFC evaluator (typically PaperSignalScanner)
        """
        self._tfc_evaluator = evaluator
