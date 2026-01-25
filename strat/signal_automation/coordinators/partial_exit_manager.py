"""
EQUITY-90: PartialExitManager - Extracted from PositionMonitor

Manages partial exit logic for multi-contract option positions.

Partial exits allow taking profits on a portion of a position while
letting the rest run with trailing stop protection.

Strategy (EQUITY-36):
- Trigger: When underlying reaches 1.0x R:R target
- Action: Close 50% of contracts (rounded up)
- Remaining: Let position continue with trailing stop

Session EQUITY-90: Phase 4.3 - God class refactoring.
"""

import logging
from typing import Optional

from strat.signal_automation.position_monitor import (
    TrackedPosition,
    ExitSignal,
    ExitReason,
    MonitoringConfig,
)

logger = logging.getLogger(__name__)


class PartialExitManager:
    """
    Manages partial exit logic for multi-contract positions.

    Extracted from PositionMonitor as part of EQUITY-90 Phase 4.3 refactoring.
    Implements the PartialExitChecker protocol defined in exit_evaluator.

    Partial exits allow locking in profits on a portion of a position while
    letting the remainder run with trailing stop protection. This balances
    the goals of capturing gains and maximizing winners.

    Usage:
        manager = PartialExitManager(config)
        exit_signal = manager.check(position)

    Args:
        config: MonitoringConfig with partial exit settings
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize partial exit manager.

        Args:
            config: MonitoringConfig with partial_exit_pct setting
        """
        self._config = config

    def check(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Check if partial exit conditions are met for a position.

        Conditions:
        1. Position must have more than 1 contract
        2. Partial exit must not have been done already
        3. Underlying must have reached 1.0x R:R target (target_1x)

        Args:
            pos: TrackedPosition to check

        Returns:
            ExitSignal for partial exit if conditions met, None otherwise
        """
        # Skip if only 1 contract (use trailing stop instead)
        if pos.contracts <= 1:
            return None

        # Skip if partial exit already done
        if pos.partial_exit_done:
            return None

        # Check if 1.0x R:R target reached
        is_bullish = pos.direction.upper() in ['CALL', 'BULL', 'UP']

        if is_bullish:
            target_1x_reached = pos.underlying_price >= pos.target_1x
        else:
            target_1x_reached = pos.underlying_price <= pos.target_1x

        if target_1x_reached:
            # Calculate contracts to close (default 50% rounded up)
            contracts_to_close = max(1, int(pos.contracts * self._config.partial_exit_pct + 0.5))

            logger.info(
                f"PARTIAL EXIT for {pos.osi_symbol}: "
                f"Closing {contracts_to_close} of {pos.contracts} contracts at 1.0x R:R "
                f"(underlying ${pos.underlying_price:.2f} hit target_1x ${pos.target_1x:.2f})"
            )

            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.PARTIAL_EXIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Partial exit: {contracts_to_close}/{pos.contracts} contracts at 1.0x R:R (${pos.target_1x:.2f})",
                contracts_to_close=contracts_to_close,
            )

        return None

    @property
    def config(self) -> MonitoringConfig:
        """Return current monitoring configuration."""
        return self._config
