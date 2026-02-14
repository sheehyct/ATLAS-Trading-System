"""
Virtual Balance Tracker - Session EQUITY-107

Tracks virtual capital, deployed capital, and pending settlements for
options trading in a cash account. Integrates with PortfolioHeatManager
for aggregate risk gating.

Key features:
- Thread-safe capital reservation/release
- T+1 settlement tracking for cash accounts
- Partial exit support with proportional cost basis
- JSON state persistence with graceful recovery
- Alpaca position sync for reconciliation
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.portfolio_heat import PortfolioHeatManager

logger = logging.getLogger(__name__)


def _next_trading_day() -> str:
    """Return the next NYSE trading day as 'YYYY-MM-DD'.

    Uses pandas_market_calendars when available, otherwise falls back
    to a simple +1 business day heuristic.
    """
    try:
        import pandas as pd
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar('NYSE')
        today = pd.Timestamp.now(tz='America/New_York').normalize()
        # Look ahead up to 10 days to cover holidays
        schedule = nyse.valid_days(
            start_date=today + pd.Timedelta(days=1),
            end_date=today + pd.Timedelta(days=10),
        )
        if len(schedule) > 0:
            return schedule[0].strftime('%Y-%m-%d')
    except ImportError:
        pass

    # Fallback: next weekday
    from datetime import timedelta
    today = datetime.now()
    offset = 1
    nxt = today + timedelta(days=offset)
    while nxt.weekday() >= 5:  # Saturday=5, Sunday=6
        offset += 1
        nxt = today + timedelta(days=offset)
    return nxt.strftime('%Y-%m-%d')


class VirtualBalanceTracker:
    """
    Tracks virtual capital for an options cash account.

    Manages deployed capital, pending settlements, and realized P&L.
    All state-modifying methods are thread-safe and auto-persist.

    Usage:
        tracker = VirtualBalanceTracker(virtual_capital=3000.0)
        tracker.load()

        if tracker.can_open_trade(cost=300.0):
            tracker.reserve_capital('SPY250321C00600000', 300.0)

        tracker.release_capital('SPY250321C00600000', proceeds=450.0)
        tracker.settle_pending()
    """

    def __init__(
        self,
        virtual_capital: float = 3000.0,
        sizing_mode: str = 'fixed_dollar',
        fixed_dollar_amount: float = 300.0,
        pct_of_capital: float = 0.10,
        max_portfolio_heat: float = 0.08,
        settlement_days: int = 1,
        state_file: str = 'data/executions/capital_state.json',
    ):
        self._starting_capital = virtual_capital
        self._sizing_mode = sizing_mode
        self._fixed_dollar_amount = fixed_dollar_amount
        self._pct_of_capital = pct_of_capital
        self._max_portfolio_heat = max_portfolio_heat
        self._settlement_days = settlement_days
        self._state_file = state_file

        # Internal state
        self._virtual_capital: float = virtual_capital
        self._deployed_capital: float = 0.0
        self._positions: Dict[str, float] = {}  # osi_symbol -> cost_basis
        self._pending_settlements: List[Dict] = []
        self._realized_pnl: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Computed property
    # ------------------------------------------------------------------

    @property
    def available_capital(self) -> float:
        """Capital available for new trades (excludes deployed + unsettled)."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        unsettled = sum(
            s['amount'] for s in self._pending_settlements
            if s['settlement_date'] > today_str
        )
        return self._virtual_capital - self._deployed_capital - unsettled

    # ------------------------------------------------------------------
    # Trade gating
    # ------------------------------------------------------------------

    def can_open_trade(self, cost: float) -> bool:
        """Check if a new trade with the given cost is permitted.

        Validates both capital availability and portfolio heat limit.
        """
        if cost <= 0:
            return False

        if self.available_capital < cost:
            logger.info(
                "Insufficient capital: need $%.2f, available $%.2f",
                cost, self.available_capital,
            )
            return False

        # Portfolio heat check
        try:
            heat_mgr = PortfolioHeatManager(max_heat=self._max_portfolio_heat)
            for sym, risk in self._positions.items():
                heat_mgr.add_position(sym, risk)
            if not heat_mgr.can_accept_trade('_new_', cost, self._virtual_capital):
                logger.info("Trade rejected by portfolio heat limit")
                return False
        except ValueError:
            # PortfolioHeatManager raises if max_heat is out of range;
            # fall through and allow the trade on capital basis alone.
            logger.warning("PortfolioHeatManager config error, skipping heat check")

        return True

    def get_trade_budget(self, signal=None) -> float:
        """Return the dollar budget for the next trade.

        Args:
            signal: Optional signal object (reserved for future sizing logic).
        """
        if self._sizing_mode == 'pct_capital':
            raw = self._pct_of_capital * self._virtual_capital
        else:
            raw = self._fixed_dollar_amount
        return min(raw, max(0.0, self.available_capital))

    # ------------------------------------------------------------------
    # Capital reservation / release
    # ------------------------------------------------------------------

    def reserve_capital(self, osi_symbol: str, cost: float) -> None:
        """Reserve capital for a new position (thread-safe)."""
        with self._lock:
            self._positions[osi_symbol] = cost
            self._deployed_capital += cost
            logger.info(
                "Reserved $%.2f for %s (deployed=$%.2f)",
                cost, osi_symbol, self._deployed_capital,
            )
            self.save()

    def release_capital(self, osi_symbol: str, proceeds: float) -> None:
        """Release capital for a fully closed position (thread-safe)."""
        with self._lock:
            original_cost = self._positions.pop(osi_symbol, 0.0)
            self._deployed_capital = max(0.0, self._deployed_capital - original_cost)
            pnl = proceeds - original_cost
            self._virtual_capital += pnl
            self._realized_pnl += pnl
            self._pending_settlements.append({
                'amount': proceeds,
                'settlement_date': _next_trading_day(),
                'osi_symbol': osi_symbol,
            })
            logger.info(
                "Released %s: cost=$%.2f proceeds=$%.2f pnl=$%.2f",
                osi_symbol, original_cost, proceeds, pnl,
            )
            self.save()

    def release_capital_partial(
        self, osi_symbol: str, fraction: float, proceeds: float,
    ) -> None:
        """Release a fraction of a position (thread-safe).

        Args:
            osi_symbol: Option symbol.
            fraction: Fraction to close (0.0 - 1.0).
            proceeds: Cash received for the closed portion.
        """
        with self._lock:
            original_cost = self._positions.get(osi_symbol, 0.0)
            partial_cost = original_cost * fraction
            self._positions[osi_symbol] = original_cost - partial_cost
            self._deployed_capital = max(0.0, self._deployed_capital - partial_cost)
            pnl = proceeds - partial_cost
            self._virtual_capital += pnl
            self._realized_pnl += pnl
            self._pending_settlements.append({
                'amount': proceeds,
                'settlement_date': _next_trading_day(),
                'osi_symbol': osi_symbol,
            })
            logger.info(
                "Partial release %s (%.0f%%): partial_cost=$%.2f proceeds=$%.2f pnl=$%.2f",
                osi_symbol, fraction * 100, partial_cost, proceeds, pnl,
            )
            self.save()

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_pending(self) -> None:
        """Clear settlements that have reached their settlement date (thread-safe)."""
        with self._lock:
            today_str = datetime.now().strftime('%Y-%m-%d')
            settled = [
                s for s in self._pending_settlements
                if s['settlement_date'] <= today_str
            ]
            self._pending_settlements = [
                s for s in self._pending_settlements
                if s['settlement_date'] > today_str
            ]
            for s in settled:
                logger.info(
                    "Settled $%.2f from %s", s['amount'], s['osi_symbol'],
                )
            if settled:
                self.save()

    # ------------------------------------------------------------------
    # Sync with broker
    # ------------------------------------------------------------------

    def sync_with_positions(self, alpaca_positions: list) -> None:
        """Reconcile tracker state with live Alpaca positions.

        Args:
            alpaca_positions: List of dicts with 'symbol', 'avg_entry_price', 'qty'.
        """
        with self._lock:
            alpaca_map = {
                p['symbol']: float(p['avg_entry_price']) * abs(float(p['qty'])) * 100
                for p in alpaca_positions
            }

            # Add positions in Alpaca but not in tracker
            for sym, cost in alpaca_map.items():
                if sym not in self._positions:
                    logger.info("Sync: adding unknown position %s ($%.2f)", sym, cost)
                    self._positions[sym] = cost
                    self._deployed_capital += cost

            # Remove positions in tracker but not in Alpaca (externally closed)
            stale = [s for s in self._positions if s not in alpaca_map]
            for sym in stale:
                cost = self._positions.pop(sym)
                self._deployed_capital = max(0.0, self._deployed_capital - cost)
                logger.info("Sync: removed stale position %s ($%.2f)", sym, cost)

            self.save()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a snapshot of the tracker state."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        unsettled = sum(
            s['amount'] for s in self._pending_settlements
            if s['settlement_date'] > today_str
        )
        return {
            'virtual_capital': self._virtual_capital,
            'deployed_capital': self._deployed_capital,
            'available_capital': self.available_capital,
            'realized_pnl': self._realized_pnl,
            'position_count': len(self._positions),
            'unsettled_amount': unsettled,
            'positions': dict(self._positions),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist state to JSON file."""
        try:
            path = Path(self._state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'virtual_capital': self._virtual_capital,
                'deployed_capital': self._deployed_capital,
                'positions': self._positions,
                'pending_settlements': self._pending_settlements,
                'realized_pnl': self._realized_pnl,
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Capital state saved to %s", self._state_file)
        except IOError as e:
            logger.error("Error saving capital state: %s", e)

    def load(self) -> 'VirtualBalanceTracker':
        """Load state from JSON file. Returns self for chaining."""
        path = Path(self._state_file)
        if not path.exists():
            logger.warning("No capital state file at %s, using defaults", self._state_file)
            return self
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._virtual_capital = float(data.get('virtual_capital', self._starting_capital))
            self._deployed_capital = float(data.get('deployed_capital', 0.0))
            self._positions = data.get('positions', {})
            self._pending_settlements = data.get('pending_settlements', [])
            self._realized_pnl = float(data.get('realized_pnl', 0.0))
            logger.info(
                "Loaded capital state: virtual=$%.2f deployed=$%.2f pnl=$%.2f",
                self._virtual_capital, self._deployed_capital, self._realized_pnl,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning("Corrupt capital state file, using defaults: %s", e)
            self._virtual_capital = self._starting_capital
            self._deployed_capital = 0.0
            self._positions = {}
            self._pending_settlements = []
            self._realized_pnl = 0.0
        return self
