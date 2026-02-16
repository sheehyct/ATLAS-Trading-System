"""
Excursion Tracker - Bar-level MFE/MAE

Tracks Maximum Favorable Excursion (best P&L) and Maximum Adverse
Excursion (worst P&L) during each trade's lifetime.

Key metrics produced:
- MFE: Best unrealized P&L seen (answers "how much profit was available?")
- MAE: Worst unrealized P&L seen (answers "how deep did drawdown go?")
- Exit efficiency: actual P&L / MFE (1.0 = perfect exit)
- Went green before loss: MFE > 0 for losing trades
"""

from datetime import datetime
from typing import Optional

from core.trade_analytics.models import ExcursionData
from strat.backtesting.simulation.position_tracker import SimulatedPosition


class ExcursionTracker:
    """
    Tracks MFE/MAE during a position's lifecycle.

    Called each bar with the current option price estimate to update
    excursion data. Finalized when the trade closes.

    Usage:
        tracker = ExcursionTracker(position)
        # Each bar:
        tracker.update(option_price, underlying_price, bar_time, bars_from_entry)
        # On close:
        excursion = tracker.finalize(final_pnl)
    """

    def __init__(self, pos: SimulatedPosition):
        self._entry_price = pos.entry_price
        self._contracts = pos.contracts
        self._is_bullish = pos.is_bullish

        # Tracking state
        self._mfe_pnl = 0.0
        self._mfe_pct = 0.0
        self._mfe_price = 0.0
        self._mfe_time: Optional[datetime] = None
        self._mfe_bars = 0

        self._mae_pnl = 0.0
        self._mae_pct = 0.0
        self._mae_price = 0.0
        self._mae_time: Optional[datetime] = None
        self._mae_bars = 0

    def update(
        self,
        option_price: float,
        underlying_price: float,
        bar_time: datetime,
        bars_from_entry: int,
    ) -> None:
        """
        Update excursion tracking with current bar data.

        Args:
            option_price: Current option price estimate
            underlying_price: Current underlying price
            bar_time: Current bar timestamp
            bars_from_entry: Bars since entry
        """
        if self._entry_price <= 0:
            return

        pnl = (option_price - self._entry_price) * self._contracts * 100
        pct = (option_price - self._entry_price) / self._entry_price

        # Update MFE (maximum favorable excursion)
        if pnl > self._mfe_pnl:
            self._mfe_pnl = pnl
            self._mfe_pct = pct
            self._mfe_price = underlying_price
            self._mfe_time = bar_time
            self._mfe_bars = bars_from_entry

        # Update MAE (maximum adverse excursion)
        if pnl < self._mae_pnl:
            self._mae_pnl = pnl
            self._mae_pct = pct
            self._mae_price = underlying_price
            self._mae_time = bar_time
            self._mae_bars = bars_from_entry

    def finalize(self, final_pnl: float) -> ExcursionData:
        """
        Produce final ExcursionData when trade closes.

        Args:
            final_pnl: Realized P&L of the trade

        Returns:
            ExcursionData with all metrics computed
        """
        # Exit efficiency: how much of available profit was captured
        exit_efficiency = 0.0
        profit_captured = 0.0
        if self._mfe_pnl > 0:
            exit_efficiency = final_pnl / self._mfe_pnl
            profit_captured = min(100.0, max(0.0, exit_efficiency * 100))

        # Did this loser go green first?
        went_green = final_pnl <= 0 and self._mfe_pnl > 0

        return ExcursionData(
            mfe_pnl=self._mfe_pnl,
            mfe_pct=self._mfe_pct,
            mfe_price=self._mfe_price,
            mfe_time=self._mfe_time,
            mfe_bars_from_entry=self._mfe_bars,
            mae_pnl=self._mae_pnl,
            mae_pct=self._mae_pct,
            mae_price=self._mae_price,
            mae_time=self._mae_time,
            mae_bars_from_entry=self._mae_bars,
            exit_efficiency=exit_efficiency,
            profit_captured_pct=profit_captured,
            went_green_before_loss=went_green,
        )
