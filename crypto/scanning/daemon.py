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
    # Fallback if pytz not available
    ET_TIMEZONE = None

from crypto import config
from crypto.config import (
    get_current_leverage_tier,
    get_max_leverage_for_symbol,
    is_intraday_window,
    time_until_intraday_close_et,
)
from crypto.exchange.coinbase_client import CoinbaseClient
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

# Optional Discord alerter import - Session CRYPTO-5
try:
    from crypto.alerters.discord_alerter import CryptoDiscordAlerter
except Exception as _discord_import_err:
    CryptoDiscordAlerter = None  # type: ignore
    # Log will be available after logger is configured
    _discord_import_error_msg = str(_discord_import_err)

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


class CryptoSignalDaemon:
    """
    Main daemon for autonomous crypto STRAT signal detection and execution.

    Orchestrates:
    - CryptoSignalScanner for pattern detection (15-min intervals)
    - CryptoEntryMonitor for trigger polling (1-min intervals)
    - PaperTrader for simulated execution

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
        self.discord_alerter: Optional[CryptoDiscordAlerter] = None  # Session CRYPTO-5

        # Signal tracking
        self._detected_signals: Dict[str, CryptoDetectedSignal] = {}
        self._signals_lock = threading.Lock()

        # Daemon state
        self._running = False
        self._shutdown_event = threading.Event()
        self._scan_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None  # Session CRYPTO-6

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
        self._setup_discord_alerter()  # Session CRYPTO-5

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
            on_poll=self._on_poll,  # Session CRYPTO-5: 60s position checks
        )

        self.entry_monitor = CryptoEntryMonitor(
            client=self.client,
            config=monitor_config,
        )
        logger.info(
            f"Entry monitor initialized (poll: {self.config.entry_poll_interval}s)"
        )

    def _on_poll(self) -> None:
        """
        Callback on each entry monitor poll cycle - Session CRYPTO-5.

        Used for more frequent position monitoring (60s instead of 5min health loop).
        """
        if self.position_monitor:
            closed_count = self.check_positions()
            if closed_count > 0:
                logger.info(f"Poll position check: closed {closed_count} trade(s)")

    def _setup_discord_alerter(self) -> None:
        """Initialize Discord alerter if webhook URL provided - Session CRYPTO-5."""
        if not self.config.discord_webhook_url:
            logger.debug("Discord alerter not configured (no webhook URL)")
            return

        if CryptoDiscordAlerter is None:
            error_msg = globals().get('_discord_import_error_msg', 'unknown error')
            logger.warning(f"Discord alerter not available (import failed: {error_msg})")
            return

        try:
            self.discord_alerter = CryptoDiscordAlerter(
                webhook_url=self.config.discord_webhook_url,
                username='ATLAS Crypto Bot',
            )
            logger.info("Discord alerter initialized")
            # Send startup connection message (Session CRYPTO-10)
            self.discord_alerter.test_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Discord alerter: {e}")

    def _start_api_server(self) -> None:
        """Start REST API server in background thread - Session CRYPTO-6."""
        try:
            from crypto.api.server import init_api, run_api

            # Initialize API with daemon reference
            init_api(self)

            # Start API in background thread
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
        # Use actual direction from entry monitor (may differ if setup changed direction)
        actual_direction = getattr(event, '_actual_direction', signal.direction)

        logger.info(
            f"TRIGGER FIRED: {signal.symbol} {signal.pattern_type} {actual_direction} "
            f"@ ${event.current_price:,.2f} (trigger: ${event.trigger_price:,.2f})"
        )

        # Session EQUITY-59: Check if setup is stale before executing
        is_stale, stale_reason = self._is_setup_stale(signal)
        if is_stale:
            logger.warning(f"STALE SETUP REJECTED: {stale_reason}")
            return

        # Session EQUITY-67: TFC re-evaluation at entry (ported from equity daemon)
        should_block, block_reason = self._reevaluate_tfc_at_entry(signal)
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

        # Send Discord trigger alert - Session CRYPTO-5 (controlled by config)
        if self.discord_alerter and self.config.alert_on_trigger:
            try:
                self.discord_alerter.send_trigger_alert(event)
            except Exception as e:
                logger.error(f"Discord trigger alert error: {e}")

    def _get_current_time_et(self) -> datetime:
        """Get current time in ET timezone."""
        now_utc = datetime.now(timezone.utc)
        if ET_TIMEZONE is not None:
            return now_utc.astimezone(ET_TIMEZONE)
        # Fallback: approximate ET as UTC-5 (ignores DST)
        from datetime import timedelta
        return now_utc - timedelta(hours=5)

    def _is_setup_stale(self, signal: CryptoDetectedSignal) -> tuple[bool, str]:
        """
        Check if a SETUP signal is stale (forming bar period has passed).

        Session EQUITY-59: Port of EQUITY-46 stale setup validation for crypto.
        Adapted for 24/7 crypto markets (no NYSE calendar needed).

        A setup is only valid during the "forming bar" period. Once that period ends:
        - If trigger was hit -> pattern COMPLETED (entry happened)
        - If trigger NOT hit -> pattern EVOLVED (new bar inserted, setup stale)

        Staleness windows for crypto 24/7 markets:
        - 1H: 2 hours (1 bar + 1 buffer)
        - 4H: 8 hours (2x bar width)
        - 1D: 48 hours (2 calendar days)
        - 1W: 336 hours (2 weeks)

        Args:
            signal: The detected signal to check

        Returns:
            Tuple of (is_stale: bool, reason: str)
        """
        from datetime import timedelta

        # Only check SETUP signals (COMPLETED are already validated)
        if signal.signal_type == "COMPLETED":
            return False, ""

        # Need setup_bar_timestamp to check staleness
        setup_ts = getattr(signal, 'setup_bar_timestamp', None)
        if setup_ts is None:
            # Try to get from context
            if hasattr(signal, 'context') and signal.context:
                setup_ts = getattr(signal.context, 'setup_bar_timestamp', None)

        if setup_ts is None:
            logger.warning(
                f"STALE CHECK SKIPPED: {signal.symbol} {signal.pattern_type} has no setup_bar_timestamp"
            )
            return False, ""

        # Get current time in UTC for comparison
        now = datetime.now(timezone.utc)

        # Ensure setup_ts is in UTC for comparison
        if setup_ts.tzinfo is None:
            setup_ts = setup_ts.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC if already has timezone (e.g., ET)
            setup_ts = setup_ts.astimezone(timezone.utc)

        # Calculate staleness based on timeframe
        timeframe = signal.timeframe.upper() if signal.timeframe else ""

        # Staleness windows for 24/7 crypto markets
        # Note: Using 2x bar width allows for market volatility during bar formation
        # and accounts for the fact that triggers may fire late in the forming bar.
        # This is intentionally looser than strict 1-bar validity to avoid
        # rejecting valid setups that trigger near bar boundaries.
        staleness_hours = {
            '1H': 2,     # 2 hours (1 bar + buffer for late triggers)
            '4H': 8,     # 8 hours (2x bar width)
            '1D': 48,    # 48 hours (2 calendar days)
            '1W': 336,   # 2 weeks
            '1M': 1440,  # ~2 months (60 days)
        }

        hours = staleness_hours.get(timeframe, 2)  # Default to 2 hours
        valid_until = setup_ts + timedelta(hours=hours)

        if now > valid_until:
            age_hours = (now - setup_ts).total_seconds() / 3600
            return True, (
                f"Setup expired: {signal.symbol} {signal.pattern_type} ({timeframe}) "
                f"detected {age_hours:.1f}h ago, max validity {hours}h"
            )

        return False, ""

    def _reevaluate_tfc_at_entry(
        self, signal: CryptoDetectedSignal
    ) -> tuple[bool, str]:
        """
        Re-evaluate TFC alignment at entry time and check if entry should be blocked.

        Session EQUITY-67: Port of EQUITY-49 TFC re-evaluation for crypto.

        TFC can change between pattern detection and entry trigger (hours/days later).
        This method:
        1. Re-evaluates TFC using current market data
        2. Compares with original TFC at detection time
        3. Logs the comparison for audit trail
        4. Optionally blocks entry if TFC degraded significantly or flipped direction

        Args:
            signal: The detected signal about to be executed

        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        # Check if TFC re-evaluation is enabled
        if not self.config.tfc_reeval_enabled:
            return False, ""

        # Get original TFC data from signal context
        original_strength = 0
        original_alignment = ""
        original_passes = False

        if signal.context:
            original_strength = signal.context.tfc_score or 0
            original_alignment = signal.context.tfc_alignment or ""
            original_passes = signal.context.tfc_passes or False

        # Determine original direction from alignment string
        original_tfc_direction = ""
        if original_alignment and "BULLISH" in original_alignment.upper():
            original_tfc_direction = "bullish"
        elif original_alignment and "BEARISH" in original_alignment.upper():
            original_tfc_direction = "bearish"
        elif not original_alignment:
            logger.debug(
                f"TFC REEVAL: {signal.symbol} {signal.pattern_type} has no original TFC alignment - "
                f"direction flip detection will be skipped"
            )

        # Re-evaluate TFC using current market data
        # LONG = 1 (want 2U bars), SHORT = -1 (want 2D bars)
        direction_int = 1 if signal.direction == "LONG" else -1

        try:
            current_tfc = self.scanner.evaluate_tfc(
                symbol=signal.symbol,
                detection_timeframe=signal.timeframe,
                direction=direction_int,
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

        # Validate returned assessment
        if current_tfc is None or not hasattr(current_tfc, "strength"):
            logger.error(
                f"TFC REEVAL: Invalid assessment returned for {signal.symbol} - proceeding with entry"
            )
            self._error_count += 1
            return False, ""

        current_strength = current_tfc.strength if current_tfc.strength is not None else 0
        current_alignment = (
            current_tfc.alignment_label()
            if hasattr(current_tfc, "alignment_label")
            else f"{current_strength}/?"
        )
        current_passes = getattr(current_tfc, "passes_flexible", False)
        current_direction = getattr(current_tfc, "direction", "") or ""

        # Calculate strength change
        strength_delta = current_strength - original_strength

        # Detect direction flip (e.g., bullish -> bearish)
        direction_flipped = False
        if original_tfc_direction and current_direction:
            direction_flipped = original_tfc_direction != current_direction
        elif not original_tfc_direction and current_direction:
            # Can't detect flip without original direction - log at debug level
            logger.debug(
                f"TFC REEVAL: {signal.symbol} {signal.pattern_type} - direction flip detection skipped "
                f"(no original TFC direction in signal)"
            )

        # Build comparison log message
        comparison = (
            f"TFC REEVAL: {signal.symbol} {signal.pattern_type} {signal.direction} | "
            f"Original: {original_alignment or 'N/A'} (score={original_strength}, passes={original_passes}) | "
            f"Current: {current_alignment} (score={current_strength}, passes={current_passes}) | "
            f"Delta: {strength_delta:+d} | "
            f"Flipped: {direction_flipped}"
        )

        # Always log if configured
        if self.config.tfc_reeval_log_always:
            if strength_delta < 0 or direction_flipped:
                logger.warning(comparison)
            else:
                logger.info(comparison)

        # Determine if entry should be blocked
        should_block = False
        block_reason = ""

        # Block if direction flipped (most severe)
        if direction_flipped and self.config.tfc_reeval_block_on_flip:
            should_block = True
            block_reason = f"TFC direction flipped from {original_tfc_direction} to {current_direction}"

        # Block if strength dropped below minimum threshold
        elif current_strength < self.config.tfc_reeval_min_strength:
            should_block = True
            block_reason = f"TFC strength {current_strength} < min threshold {self.config.tfc_reeval_min_strength}"

        return should_block, block_reason

    def _execute_trade(self, event: CryptoTriggerEvent) -> None:
        """
        Execute trade via paper trader with time-based leverage.

        Uses intraday leverage (10x) during 6PM-4PM ET window,
        swing leverage (4x) during 4PM-6PM ET gap.

        Only one position per symbol allowed - skips if position already exists.

        Args:
            event: Trigger event
        """
        if self.paper_trader is None:
            return

        signal = event.signal
        # Use actual direction from entry monitor if pattern changed (Session CRYPTO-8)
        # This handles cases where SETUP (X-1-?) became 2-bar pattern (X-2D)
        direction = getattr(event, '_actual_direction', signal.direction)
        actual_pattern = getattr(event, '_actual_pattern', signal.pattern_type)

        # Session CRYPTO-MONITOR-2 FIX: Recalculate stop/target when direction flips
        # For bidirectional setups (3-?), the signal has LONG values. If triggering SHORT
        # (or vice versa), we need the correct stop/target for the actual direction.
        stop_price = signal.stop_price
        target_price = signal.target_price

        if direction != signal.direction and signal.setup_bar_high > 0 and signal.setup_bar_low > 0:
            # Direction flipped - recalculate for actual direction
            bar_range = signal.setup_bar_high - signal.setup_bar_low
            if direction == "SHORT":
                # Flipped to SHORT: stop at high, target below entry
                stop_price = signal.setup_bar_high
                target_price = event.current_price - bar_range
            else:
                # Flipped to LONG: stop at low, target above entry
                stop_price = signal.setup_bar_low
                target_price = event.current_price + bar_range
            logger.info(
                f"DIRECTION FLIPPED: {signal.direction} -> {direction}, "
                f"recalculated stop=${stop_price:,.2f} target=${target_price:,.2f}"
            )

        # Session CRYPTO-MONITOR-2: REMOVED position limit
        # Previously only allowed one position per symbol, but this prevented
        # multiple valid signals from executing and limited our ability to
        # analyze trade accuracy. Now allowing multiple positions per symbol.

        # Get current leverage tier based on time
        now_et = self._get_current_time_et()
        tier = get_current_leverage_tier(now_et)
        max_leverage = get_max_leverage_for_symbol(signal.symbol, now_et)

        logger.info(
            f"Leverage tier: {tier} ({max_leverage}x) for {signal.symbol}"
        )

        # Session EQUITY-59: Always validate trade inputs (bug prevention)
        if event.current_price <= 0 or stop_price <= 0:
            logger.warning("SKIPPING TRADE: Invalid entry or stop price")
            return
        if abs(event.current_price - stop_price) == 0:
            logger.warning("SKIPPING TRADE: Stop distance is zero")
            return

        # Session EQUITY-59: Skip leverage-constraint check when using leverage-first sizing
        if not config.LEVERAGE_FIRST_SIZING:
            # Check if trade should be skipped due to leverage constraints
            skip, reason = should_skip_trade(
                account_value=self.paper_trader.account.current_balance,
                risk_percent=config.DEFAULT_RISK_PERCENT,
                entry_price=event.current_price,
                stop_price=stop_price,  # Use corrected stop
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

        # Calculate position size with time-based leverage (Session EQUITY-34: use available balance)
        available_balance = self.paper_trader.get_available_balance()

        # Session EQUITY-59: Use leverage-first sizing for full capital deployment
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
                stop_price=stop_price,  # Use corrected stop
                max_leverage=max_leverage,
            )
            # Apply continuity-based risk multiplier (only for risk-based sizing)
            position_size *= risk_multiplier
            actual_risk *= risk_multiplier

        if position_size <= 0:
            logger.warning("Position size is zero or negative - skipping trade")
            return

        # Map direction to side (BUY for LONG, SELL for SHORT)
        side = "BUY" if direction == "LONG" else "SELL"

        # Log intraday window info
        if tier == "intraday":
            time_remaining = time_until_intraday_close_et(now_et)
            logger.info(
                f"INTRADAY TRADE: {time_remaining.total_seconds()/3600:.1f}h "
                f"until 4PM ET close requirement"
            )

        # Execute via paper trader with stop/target for position monitoring
        # Session EQUITY-67: Determine entry bar type from direction for pattern invalidation
        entry_bar_type = "2U" if direction == "LONG" else "2D"

        try:
            trade = self.paper_trader.open_trade(
                symbol=signal.symbol,
                side=side,
                quantity=position_size,
                entry_price=event.current_price,
                stop_price=stop_price,  # Use corrected stop
                target_price=target_price,  # Use corrected target
                timeframe=signal.timeframe,
                pattern_type=actual_pattern,  # Use resolved pattern
                tfc_score=signal.context.tfc_score,  # Session CRYPTO-9
                risk_multiplier=risk_multiplier,
                leverage=implied_leverage,  # Session EQUITY-34: margin tracking
                # Session EQUITY-67: Pattern invalidation tracking
                entry_bar_type=entry_bar_type,
                entry_bar_high=signal.setup_bar_high,
                entry_bar_low=signal.setup_bar_low,
            )

            # Session EQUITY-34: Check if trade was rejected due to insufficient margin
            if trade is None:
                logger.warning(
                    f"Trade rejected: insufficient margin for {signal.symbol}"
                )
                return

            self._execution_count += 1
            logger.info(
                f"TRADE OPENED: {trade.trade_id} {signal.symbol} {side} "
                f"qty={position_size:.6f} @ ${event.current_price:,.2f} "
                f"({implied_leverage:.1f}x leverage, ${actual_risk:.2f} risk)"
            )
            logger.info(
                f"  Stop: ${stop_price:,.2f} | Target: ${target_price:,.2f}"
            )

            # Send Discord entry alert (Session CRYPTO-MONITOR-2: pass corrected values)
            if self.discord_alerter and self.config.alert_on_trade_entry:
                try:
                    self.discord_alerter.send_entry_alert(
                        signal=signal,
                        entry_price=event.current_price,
                        quantity=position_size,
                        leverage=implied_leverage,
                        pattern_override=actual_pattern,
                        direction_override=direction,
                        stop_override=stop_price,
                        target_override=target_price,
                    )
                except Exception as alert_err:
                    logger.warning(f"Failed to send entry alert: {alert_err}")

        except Exception as e:
            logger.warning(f"Failed to open trade for {signal.symbol}: {e}")

    # =========================================================================
    # SCANNING
    # =========================================================================

    def _passes_filters(self, signal: CryptoDetectedSignal) -> bool:
        """
        Check if signal passes quality filters.

        Args:
            signal: Signal to check

        Returns:
            True if passes all filters
        """
        # Magnitude filter
        if signal.magnitude_pct < self.config.min_magnitude_pct:
            return False

        # Risk:Reward filter
        if signal.risk_reward < self.config.min_risk_reward:
            return False

        # Skip signals with maintenance gaps
        if signal.has_maintenance_gap:
            logger.debug(f"Skipping signal with maintenance gap: {signal.symbol}")
            return False

        return True

    def _generate_signal_id(self, signal: CryptoDetectedSignal) -> str:
        """Generate unique ID for deduplication.

        CRYPTO-MONITOR-1 FIX: Use setup_bar_timestamp instead of detected_time.
        This ensures:
        - Same bar across scans -> same ID -> deduplicated
        - Different bars -> different IDs -> kept as separate setups
        """
        # Use setup_bar_timestamp for deduplication (bar-based, not scan-based)
        bar_ts = signal.setup_bar_timestamp
        if bar_ts is not None and hasattr(bar_ts, 'isoformat'):
            ts_str = bar_ts.isoformat()
        elif bar_ts is not None:
            ts_str = str(bar_ts)
        else:
            # Fallback to detected_time if no setup_bar_timestamp
            ts_str = signal.detected_time.isoformat()

        return (
            f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_"
            f"{signal.direction}_{ts_str}"
        )

    def _is_duplicate(self, signal: CryptoDetectedSignal) -> bool:
        """Check if signal is a duplicate."""
        signal_id = self._generate_signal_id(signal)
        with self._signals_lock:
            return signal_id in self._detected_signals

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
                    # Apply filters
                    if not self._passes_filters(signal):
                        continue

                    # Check for duplicate
                    if self._is_duplicate(signal):
                        continue

                    # Store signal
                    signal_id = self._generate_signal_id(signal)
                    with self._signals_lock:
                        self._detected_signals[signal_id] = signal

                    new_signals.append(signal)
                    self._signal_count += 1

                    logger.info(
                        f"NEW SIGNAL: {signal.symbol} {signal.pattern_type} "
                        f"{signal.direction} ({signal.timeframe}) "
                        f"[{signal.signal_type}]"
                    )

                    # Send Discord signal alert - Session CRYPTO-5 (controlled by config)
                    if self.discord_alerter and self.config.alert_on_signal_detection:
                        try:
                            self.discord_alerter.send_signal_alert(signal)
                        except Exception as e:
                            logger.error(f"Discord signal alert error: {e}")

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

        TRIGGERED patterns (signal_type="COMPLETED") are patterns where the entry
        bar has already formed - these should execute immediately at market price.

        SETUP patterns are waiting for a price break - these go to entry_monitor.

        Returns:
            List of new signals found
        """
        new_signals = self.run_scan()

        # Session CRYPTO-MONITOR-3: Execute TRIGGERED patterns immediately
        # These are patterns where entry condition was already met (e.g., 3-1-2D
        # where the 2D bar broke the inside bar). Previously these were discarded!
        #
        # IMPORTANT: Track executed symbols to prevent:
        # 1. Duplicate trades on same symbol/timeframe
        # 2. Conflicting trades (LONG and SHORT on same symbol)
        triggered_count = 0
        executed_symbols = set()  # Track symbol+timeframe to avoid duplicates

        for signal in new_signals:
            if signal.signal_type == "COMPLETED":
                # Create unique key for this signal
                signal_key = f"{signal.symbol}_{signal.timeframe}"

                # Skip if we already executed a trade for this symbol/timeframe
                if signal_key in executed_symbols:
                    logger.debug(
                        f"Skipping duplicate TRIGGERED: {signal.symbol} "
                        f"{signal.pattern_type} ({signal.timeframe})"
                    )
                    continue

                # Execute and mark as done
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

        Session CRYPTO-MONITOR-3: Patterns where entry bar has formed should
        execute at current market price, not be discarded.

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

        # Session EQUITY-59: Always validate trade inputs (bug prevention)
        if current_price <= 0 or stop_price <= 0:
            logger.warning("SKIPPING TRIGGERED: Invalid entry or stop price")
            return
        if abs(current_price - stop_price) == 0:
            logger.warning("SKIPPING TRIGGERED: Stop distance is zero")
            return

        # Session EQUITY-59: Skip leverage-constraint check when using leverage-first sizing
        if not config.LEVERAGE_FIRST_SIZING:
            # Check if trade should be skipped due to leverage constraints
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

        # Calculate position size using available balance (Session EQUITY-34)
        available_balance = self.paper_trader.get_available_balance()

        # Session EQUITY-59: Use leverage-first sizing for full capital deployment
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
            # Apply continuity-based risk multiplier (only for risk-based sizing)
            position_size *= risk_multiplier
            actual_risk *= risk_multiplier

        if position_size <= 0:
            logger.warning("Position size is zero or negative - skipping")
            return

        side = "BUY" if direction == "LONG" else "SELL"

        # Execute trade
        try:
            trade = self.paper_trader.open_trade(
                symbol=signal.symbol,
                side=side,
                quantity=position_size,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                timeframe=signal.timeframe,
                pattern_type=signal.pattern_type,
                tfc_score=signal.context.tfc_score,
                risk_multiplier=risk_multiplier,
                leverage=implied_leverage,  # Session EQUITY-34: margin tracking
            )

            # Session EQUITY-34: Check if trade was rejected due to insufficient margin
            if trade is None:
                logger.warning(
                    f"Trade rejected: insufficient margin for {signal.symbol}"
                )
                return

            self._execution_count += 1
            logger.info(
                f"TRADE OPENED (TRIGGERED): {trade.trade_id} {signal.symbol} {side} "
                f"qty={position_size:.6f} @ ${current_price:,.2f} "
                f"({implied_leverage:.1f}x leverage, ${actual_risk:.2f} risk)"
            )
            logger.info(
                f"  Pattern: {signal.pattern_type} | Stop: ${stop_price:,.2f} | "
                f"Target: ${target_price:,.2f}"
            )

            # Send Discord alert
            if self.discord_alerter and self.config.alert_on_trade_entry:
                try:
                    self.discord_alerter.send_entry_alert(
                        signal=signal,
                        entry_price=current_price,
                        quantity=position_size,
                        leverage=implied_leverage,
                    )
                except Exception as alert_err:
                    logger.warning(f"Failed to send entry alert: {alert_err}")

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
                # Check maintenance window
                if self.is_maintenance_window():
                    logger.info("Maintenance window - skipping scan")
                else:
                    self.run_scan_and_monitor()

                # Clean up expired signals
                self._cleanup_expired_signals()

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                self._error_count += 1

            # Wait for next scan (interruptible)
            self._shutdown_event.wait(timeout=self.config.scan_interval)

        logger.info("Scan loop stopped")

    def _cleanup_expired_signals(self) -> None:
        """Remove signals older than expiry threshold."""
        now = datetime.now(timezone.utc)
        expired_ids = []

        with self._signals_lock:
            for signal_id, signal in self._detected_signals.items():
                detected_time = signal.detected_time
                if detected_time.tzinfo is None:
                    detected_time = detected_time.replace(tzinfo=timezone.utc)

                age_hours = (now - detected_time).total_seconds() / 3600
                if age_hours > self.config.signal_expiry_hours:
                    expired_ids.append(signal_id)

            for signal_id in expired_ids:
                del self._detected_signals[signal_id]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired signals")

    def _health_loop(self) -> None:
        """Background health check loop for status logging."""
        # Note: Position monitoring moved to entry monitor poll (60s) in Session CRYPTO-5
        while not self._shutdown_event.is_set():
            try:
                status = self.get_status()
                logger.info(
                    f"HEALTH: scans={status['scan_count']}, "
                    f"signals={status['signal_count']}, "
                    f"triggers={status['trigger_count']}, "
                    f"executions={status['execution_count']}, "
                    f"errors={status['error_count']}"
                )
            except Exception as e:
                logger.error(f"Health check error: {e}")

            # Wait for next check (interruptible)
            self._shutdown_event.wait(timeout=self.config.health_check_interval)

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

        # Windows-specific
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

        # Setup signal handlers
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

        # Start health check thread
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="CryptoHealthLoop"
        )
        self._health_thread.start()
        logger.info("Health check loop started")

        # Start REST API server (Session CRYPTO-6)
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

        # Signal shutdown
        self._shutdown_event.set()

        # Stop entry monitor
        if self.entry_monitor is not None:
            self.entry_monitor.stop()

        # Wait for threads
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=5)
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)

        # Final status
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

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status and statistics."""
        uptime = None
        if self._start_time:
            uptime = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()

        with self._signals_lock:
            signals_in_store = len(self._detected_signals)

        entry_stats = {}
        if self.entry_monitor:
            entry_stats = self.entry_monitor.get_stats()

        paper_stats = {}
        if self.paper_trader:
            paper_stats = self.paper_trader.get_account_summary()

        # Get current leverage tier
        now_et = self._get_current_time_et()
        tier = get_current_leverage_tier(now_et)
        is_intraday = is_intraday_window(now_et)
        time_to_close = None
        if is_intraday:
            time_to_close = time_until_intraday_close_et(now_et).total_seconds()

        return {
            "running": self._running,
            "start_time": (
                self._start_time.isoformat() if self._start_time else None
            ),
            "uptime_seconds": uptime,
            "scan_count": self._scan_count,
            "signal_count": self._signal_count,
            "trigger_count": self._trigger_count,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "signals_in_store": signals_in_store,
            "maintenance_window": self.is_maintenance_window(),
            "leverage_tier": tier,
            "intraday_available": is_intraday,
            "intraday_close_seconds": time_to_close,
            "symbols": self.config.symbols,
            "scan_interval": self.config.scan_interval,
            "entry_monitor": entry_stats,
            "paper_trader": paper_stats,
        }

    def get_detected_signals(self) -> List[CryptoDetectedSignal]:
        """Get all detected signals currently in store."""
        with self._signals_lock:
            return list(self._detected_signals.values())

    def get_pending_setups(self) -> List[CryptoDetectedSignal]:
        """Get SETUP signals waiting for trigger."""
        if self.entry_monitor is None:
            return []
        return self.entry_monitor.get_pending_signals()

    # =========================================================================
    # POSITION MONITORING (Session CRYPTO-4)
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

        # Check for exit conditions
        exit_signals = self.position_monitor.check_exits()

        if not exit_signals:
            return 0

        # Execute exits
        closed = self.position_monitor.execute_all_exits(exit_signals)

        # Send Discord exit alerts for each closed trade (Session CRYPTO-7)
        if self.discord_alerter and self.config.alert_on_trade_exit and closed:
            for trade in closed:
                try:
                    # Calculate P&L
                    if trade.side == "BUY":  # LONG
                        pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                        pnl_pct = ((trade.exit_price / trade.entry_price) - 1) * 100
                    else:  # SHORT
                        pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                        pnl_pct = ((trade.entry_price / trade.exit_price) - 1) * 100

                    # Session CRYPTO-MONITOR-3: Include pattern context for traceability
                    self.discord_alerter.send_exit_alert(
                        symbol=trade.symbol,
                        direction="LONG" if trade.side == "BUY" else "SHORT",
                        exit_reason=trade.exit_reason or "Unknown",
                        entry_price=trade.entry_price,
                        exit_price=trade.exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        pattern_type=trade.pattern_type,
                        timeframe=trade.timeframe,
                        entry_time=trade.entry_time,
                        exit_time=trade.exit_time,
                    )
                except Exception as alert_err:
                    logger.warning(f"Failed to send exit alert: {alert_err}")

        return len(closed)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions with current P&L.

        Returns:
            List of position dicts with P&L info
        """
        if self.position_monitor is None:
            return []
        return self.position_monitor.get_open_positions_with_pnl()

    def print_status(self) -> None:
        """Print current daemon status."""
        status = self.get_status()

        print("\n" + "=" * 70)
        print("CRYPTO SIGNAL DAEMON STATUS")
        print("=" * 70)
        print(f"Running: {status['running']}")
        print(f"Uptime: {status['uptime_seconds']:.0f}s" if status["uptime_seconds"] else "Not started")
        print(f"Maintenance Window: {status['maintenance_window']}")
        print()

        # Leverage tier info
        tier = status.get('leverage_tier', 'swing')
        is_intraday = status.get('intraday_available', False)
        time_to_close = status.get('intraday_close_seconds')
        print(f"Leverage Tier: {tier.upper()} ({'10x' if tier == 'intraday' else '4x'})")
        if is_intraday and time_to_close:
            hours = time_to_close / 3600
            print(f"  Intraday Window: {hours:.1f}h until 4PM ET close")
        elif not is_intraday:
            print(f"  Note: In 4-6PM ET gap (swing only)")
        print()

        print(f"Scans: {status['scan_count']}")
        print(f"Signals Detected: {status['signal_count']}")
        print(f"Triggers Fired: {status['trigger_count']}")
        print(f"Executions: {status['execution_count']}")
        print(f"Errors: {status['error_count']}")
        print()
        print(f"Signals in Store: {status['signals_in_store']}")
        print(f"Symbols: {', '.join(status['symbols'])}")
        print(f"Scan Interval: {status['scan_interval']}s")

        if status.get("entry_monitor"):
            em = status["entry_monitor"]
            print()
            print(f"Entry Monitor:")
            print(f"  Pending Signals: {em.get('pending_signals', 0)}")
            print(f"  Total Triggers: {em.get('trigger_count', 0)}")

        if status.get("paper_trader"):
            pt = status["paper_trader"]
            print()
            print(f"Paper Trader:")
            print(f"  Balance: ${pt.get('current_balance', 0):,.2f}")
            print(f"  Realized P&L: ${pt.get('realized_pnl', 0):,.2f}")
            print(f"  Open Trades: {pt.get('open_trades', 0)}")
            print(f"  Closed Trades: {pt.get('closed_trades', 0)}")

        print("=" * 70)
