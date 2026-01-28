"""
EQUITY-94: CryptoEntryValidator - Extracted from CryptoSignalDaemon

Validates entry conditions before trade execution:
- Stale setup detection (forming bar period expired)
- TFC re-evaluation at entry time (direction flip, strength degradation)

Extracted as part of Phase 6.4 coordinator extraction.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from crypto.scanning.models import CryptoDetectedSignal

logger = logging.getLogger(__name__)


class TFCEvaluator(Protocol):
    """Protocol for TFC evaluation capability."""

    def evaluate_tfc(
        self, symbol: str, detection_timeframe: str, direction: int
    ) -> object: ...


class CryptoEntryValidator:
    """
    Validates entry conditions for crypto STRAT signals.

    Checks:
    1. Setup staleness (forming bar period expired)
    2. TFC re-evaluation (alignment changed since detection)
    """

    def __init__(
        self,
        tfc_reeval_enabled: bool = True,
        tfc_reeval_min_strength: int = 3,
        tfc_reeval_block_on_flip: bool = True,
        tfc_reeval_log_always: bool = True,
    ):
        """
        Initialize entry validator.

        Args:
            tfc_reeval_enabled: Enable TFC re-evaluation at entry
            tfc_reeval_min_strength: Block entry if TFC drops below this
            tfc_reeval_block_on_flip: Block if TFC direction flipped
            tfc_reeval_log_always: Log comparison even when not blocking
        """
        self._tfc_reeval_enabled = tfc_reeval_enabled
        self._tfc_reeval_min_strength = tfc_reeval_min_strength
        self._tfc_reeval_block_on_flip = tfc_reeval_block_on_flip
        self._tfc_reeval_log_always = tfc_reeval_log_always
        self._tfc_evaluator: Optional[TFCEvaluator] = None
        self._on_error: Optional[callable] = None

    def set_tfc_evaluator(self, evaluator: TFCEvaluator) -> None:
        """Set TFC evaluator (deferred initialization)."""
        self._tfc_evaluator = evaluator

    def set_error_callback(self, callback: callable) -> None:
        """Set error callback for incrementing daemon error count."""
        self._on_error = callback

    def _increment_errors(self) -> None:
        """Increment error count via callback."""
        if self._on_error:
            self._on_error()

    def is_setup_stale(
        self, signal: CryptoDetectedSignal
    ) -> tuple[bool, str]:
        """
        Check if a SETUP signal is stale (forming bar period has passed).

        Session EQUITY-59: Port of EQUITY-46 stale setup validation for crypto.
        Adapted for 24/7 crypto markets (no NYSE calendar needed).

        A setup is only valid during the "forming bar" period. Once that period ends:
        - If trigger was hit -> pattern COMPLETED (entry happened)
        - If trigger NOT hit -> pattern EVOLVED (new bar inserted, setup stale)

        Staleness windows for crypto 24/7 markets:
        - 1H: 2 hours (1 bar + buffer for late triggers)
        - 4H: 8 hours (2x bar width)
        - 1D: 48 hours (2 calendar days)
        - 1W: 336 hours (2 weeks)
        - 1M: 1440 hours (~2 months, 60 days)

        Args:
            signal: The detected signal to check

        Returns:
            Tuple of (is_stale: bool, reason: str)
        """
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
                f"STALE CHECK SKIPPED: {signal.symbol} {signal.pattern_type} "
                f"has no setup_bar_timestamp"
            )
            return False, ""

        # Get current time in UTC for comparison
        now = datetime.now(timezone.utc)

        # Ensure setup_ts is in UTC for comparison
        if setup_ts.tzinfo is None:
            setup_ts = setup_ts.replace(tzinfo=timezone.utc)
        else:
            setup_ts = setup_ts.astimezone(timezone.utc)

        # Calculate staleness based on timeframe
        timeframe = signal.timeframe.upper() if signal.timeframe else ""

        # Staleness windows for 24/7 crypto markets
        staleness_hours = {
            '1H': 2,
            '4H': 8,
            '1D': 48,
            '1W': 336,
            '1M': 1440,
        }

        hours = staleness_hours.get(timeframe, 2)
        valid_until = setup_ts + timedelta(hours=hours)

        if now > valid_until:
            age_hours = (now - setup_ts).total_seconds() / 3600
            return True, (
                f"Setup expired: {signal.symbol} {signal.pattern_type} ({timeframe}) "
                f"detected {age_hours:.1f}h ago, max validity {hours}h"
            )

        return False, ""

    def reevaluate_tfc_at_entry(
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
        if not self._tfc_reeval_enabled:
            return False, ""

        if self._tfc_evaluator is None:
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
                f"TFC REEVAL: {signal.symbol} {signal.pattern_type} has no original "
                f"TFC alignment - direction flip detection will be skipped"
            )

        # Re-evaluate TFC using current market data
        direction_int = 1 if signal.direction == "LONG" else -1

        try:
            current_tfc = self._tfc_evaluator.evaluate_tfc(
                symbol=signal.symbol,
                detection_timeframe=signal.timeframe,
                direction=direction_int,
            )
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.warning(
                f"TFC REEVAL ERROR (recoverable): {signal.symbol} {signal.pattern_type} - "
                f"{type(e).__name__}: {e} (proceeding with entry)"
            )
            self._increment_errors()
            return False, ""
        except Exception as e:
            logger.error(
                f"TFC REEVAL UNEXPECTED ERROR: {signal.symbol} {signal.pattern_type} - "
                f"{type(e).__name__}: {e} (proceeding with entry)"
            )
            self._increment_errors()
            return False, ""

        # Validate returned assessment
        if current_tfc is None or not hasattr(current_tfc, "strength"):
            logger.error(
                f"TFC REEVAL: Invalid assessment returned for {signal.symbol} "
                f"- proceeding with entry"
            )
            self._increment_errors()
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

        # Detect direction flip
        direction_flipped = False
        if original_tfc_direction and current_direction:
            direction_flipped = original_tfc_direction != current_direction
        elif not original_tfc_direction and current_direction:
            logger.debug(
                f"TFC REEVAL: {signal.symbol} {signal.pattern_type} - direction flip "
                f"detection skipped (no original TFC direction in signal)"
            )

        # Build comparison log message
        comparison = (
            f"TFC REEVAL: {signal.symbol} {signal.pattern_type} {signal.direction} | "
            f"Original: {original_alignment or 'N/A'} (score={original_strength}, "
            f"passes={original_passes}) | "
            f"Current: {current_alignment} (score={current_strength}, "
            f"passes={current_passes}) | "
            f"Delta: {strength_delta:+d} | "
            f"Flipped: {direction_flipped}"
        )

        if self._tfc_reeval_log_always:
            if strength_delta < 0 or direction_flipped:
                logger.warning(comparison)
            else:
                logger.info(comparison)

        # Determine if entry should be blocked
        should_block = False
        block_reason = ""

        if direction_flipped and self._tfc_reeval_block_on_flip:
            should_block = True
            block_reason = (
                f"TFC direction flipped from {original_tfc_direction} to {current_direction}"
            )
        elif current_strength < self._tfc_reeval_min_strength:
            should_block = True
            block_reason = (
                f"TFC strength {current_strength} < min threshold "
                f"{self._tfc_reeval_min_strength}"
            )

        return should_block, block_reason
