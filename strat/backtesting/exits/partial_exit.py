"""
Partial Exit Evaluator - Backtest Port

Mirrors the live PartialExitManager (EQUITY-36/EQUITY-90).

Strategy:
- Trigger: When underlying reaches 1.0x R:R target (target_1x)
- Action: Close 50% of contracts (rounded up)
- Remaining: Continue with trailing stop protection
- Only for multi-contract positions (contracts > 1)
- Can only execute once per position
"""

import logging
from typing import Optional

from strat.backtesting.config import BacktestConfig
from strat.backtesting.simulation.position_tracker import SimulatedPosition, ExitReason

logger = logging.getLogger(__name__)


class PartialExitEvaluator:
    """
    Evaluates partial exit conditions for multi-contract positions.

    Triggers when underlying reaches 1.0x R:R target, closing
    50% of contracts to lock in profits.
    """

    def __init__(self, config: BacktestConfig):
        self._config = config

    def check(self, pos: SimulatedPosition) -> Optional['ExitEvalResult']:
        """
        Check if partial exit conditions are met.

        Conditions:
        1. More than 1 contract remaining
        2. Partial exit not already done
        3. Underlying reached 1.0x R:R target

        Returns:
            ExitEvalResult for partial exit if triggered, None otherwise
        """
        from strat.backtesting.exits.exit_evaluator import ExitEvalResult

        # Single contract -> use trailing stop only
        if pos.contracts_remaining <= 1:
            return None

        # Already done partial
        if pos.partial_exit_done:
            return None

        # Check 1.0x R:R target
        if pos.target_1x <= 0:
            return None

        if pos.is_bullish:
            target_reached = pos.underlying_price >= pos.target_1x
        else:
            target_reached = pos.underlying_price <= pos.target_1x

        if target_reached:
            # Close 50% of contracts (rounded up)
            contracts_to_close = max(
                1, int(pos.contracts_remaining * self._config.partial_exit_pct + 0.5)
            )

            logger.debug(
                "Partial exit for %s: %d/%d contracts at 1.0x R:R ($%.2f)",
                pos.symbol, contracts_to_close, pos.contracts_remaining, pos.target_1x,
            )

            return ExitEvalResult(
                should_exit=True,
                reason=ExitReason.PARTIAL_EXIT,
                details=f"Partial: {contracts_to_close}/{pos.contracts_remaining} "
                        f"contracts at 1.0x R:R (${pos.target_1x:.2f})",
                contracts_to_close=contracts_to_close,
            )

        return None
