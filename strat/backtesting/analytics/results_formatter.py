"""
Results Formatter - Summary Stats, Equity Curve, DataFrame Output

Produces human-readable backtest results including:
- Win rate, average P&L, Sharpe ratio
- Pattern and timeframe breakdowns
- Exit reason distribution
- Equity curve
- Trades DataFrame for further analysis
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from strat.backtesting.config import BacktestConfig
from strat.backtesting.simulation.position_tracker import SimulatedPosition

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Complete backtest results with summary statistics.

    Produced by ResultsFormatter.format() after simulation.
    """
    # Trade data
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_count: int = 0

    # Summary stats
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # Breakdowns
    by_pattern: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_timeframe: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_exit_reason: Dict[str, int] = field(default_factory=dict)

    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # Capital summary
    capital_summary: Optional[Dict[str, Any]] = None

    # Config used
    config_summary: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Total Trades: {self.trade_count}",
            f"Total P&L:    ${self.total_pnl:,.2f}",
            f"Win Rate:     {self.win_rate:.1%}",
            f"Avg P&L:      ${self.avg_pnl:,.2f}",
            f"Avg Winner:   ${self.avg_winner:,.2f}",
            f"Avg Loser:    ${self.avg_loser:,.2f}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Max Drawdown: ${self.max_drawdown:,.2f}",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            "",
            "Exit Reason Distribution:",
        ]
        for reason, count in sorted(self.by_exit_reason.items(),
                                     key=lambda x: -x[1]):
            pct = count / self.trade_count * 100 if self.trade_count > 0 else 0
            lines.append(f"  {reason:20s} {count:4d} ({pct:5.1f}%)")

        if self.by_timeframe:
            lines.append("")
            lines.append("By Timeframe:")
            for tf, stats in sorted(self.by_timeframe.items()):
                lines.append(
                    f"  {tf}: {stats['count']} trades, "
                    f"{stats['win_rate']:.1%} WR, "
                    f"${stats['total_pnl']:,.2f} P&L"
                )

        if self.by_pattern:
            lines.append("")
            lines.append("By Pattern:")
            for pat, stats in sorted(self.by_pattern.items()):
                lines.append(
                    f"  {pat:12s}: {stats['count']} trades, "
                    f"{stats['win_rate']:.1%} WR, "
                    f"${stats['total_pnl']:,.2f} P&L"
                )

        if self.capital_summary:
            lines.append("")
            lines.append("Capital Summary:")
            cs = self.capital_summary
            lines.append(f"  Starting:  ${cs.get('starting_capital', 0):,.2f}")
            lines.append(f"  Final:     ${cs.get('virtual_capital', 0):,.2f}")
            lines.append(f"  Return:    {cs.get('total_return_pct', 0):.1f}%")

        lines.append("=" * 60)
        return "\n".join(lines)


class ResultsFormatter:
    """Formats raw trade data into BacktestResults."""

    @staticmethod
    def format(
        trades: List[SimulatedPosition],
        config: BacktestConfig,
        capital_summary: Optional[Dict[str, Any]] = None,
    ) -> BacktestResults:
        """
        Format a list of closed trades into BacktestResults.

        Args:
            trades: List of closed SimulatedPosition
            config: BacktestConfig used for the run
            capital_summary: Optional capital state summary

        Returns:
            BacktestResults with all statistics computed
        """
        results = BacktestResults()
        results.capital_summary = capital_summary
        results.trade_count = len(trades)

        if not trades:
            return results

        # Build trades DataFrame
        results.trades_df = ResultsFormatter._build_trades_df(trades)

        # Basic stats
        pnls = [t.realized_pnl or 0.0 for t in trades]
        results.total_pnl = sum(pnls)
        results.avg_pnl = results.total_pnl / len(trades) if trades else 0.0

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        results.win_rate = len(winners) / len(trades) if trades else 0.0
        results.avg_winner = sum(winners) / len(winners) if winners else 0.0
        results.avg_loser = sum(losers) / len(losers) if losers else 0.0

        # Profit factor
        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Equity curve and drawdown
        equity = pd.Series(pnls).cumsum()
        results.equity_curve = equity
        if len(equity) > 0:
            running_max = equity.cummax()
            drawdown = equity - running_max
            results.max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Sharpe ratio (annualized, assuming ~252 trades/year is rough)
        if len(pnls) > 1:
            pnl_series = pd.Series(pnls)
            mean_pnl = pnl_series.mean()
            std_pnl = pnl_series.std()
            if std_pnl > 0:
                results.sharpe_ratio = mean_pnl / std_pnl * np.sqrt(len(pnls))

        # Exit reason distribution
        for t in trades:
            reason = t.exit_reason.value if t.exit_reason else 'UNKNOWN'
            results.by_exit_reason[reason] = results.by_exit_reason.get(reason, 0) + 1

        # Timeframe breakdown
        results.by_timeframe = ResultsFormatter._breakdown_by(trades, 'timeframe')

        # Pattern breakdown
        results.by_pattern = ResultsFormatter._breakdown_by(trades, 'pattern_type')

        # Config summary
        results.config_summary = {
            'symbols': config.symbols,
            'timeframes': config.timeframes,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'capital_tracking': config.capital_tracking_enabled,
            'trailing_stops': config.use_trailing_stop,
            'partial_exits': config.partial_exit_enabled,
            'pattern_invalidation': config.pattern_invalidation_enabled,
        }

        return results

    @staticmethod
    def _build_trades_df(trades: List[SimulatedPosition]) -> pd.DataFrame:
        """Build a DataFrame from trade data."""
        rows = []
        for t in trades:
            rows.append({
                'symbol': t.symbol,
                'timeframe': t.timeframe,
                'pattern_type': t.pattern_type,
                'direction': t.direction,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_trigger': t.entry_trigger,
                'actual_entry': t.actual_entry_underlying,
                'stop_price': t.stop_price,
                'target_price': t.target_price,
                'entry_option_price': t.entry_price,
                'exit_option_price': t.exit_price,
                'contracts': t.contracts,
                'pnl': t.realized_pnl or 0.0,
                'exit_reason': t.exit_reason.value if t.exit_reason else 'UNKNOWN',
                'bars_held': t.bars_held,
                'trailing_stop_active': t.trailing_stop_active,
                'partial_exit_done': t.partial_exit_done,
                'high_water_mark': t.high_water_mark,
                'strike': t.strike,
                'expiration': t.expiration,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _breakdown_by(
        trades: List[SimulatedPosition],
        field: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute breakdown statistics by a given field."""
        groups: Dict[str, List[float]] = {}
        for t in trades:
            key = getattr(t, field, 'unknown')
            groups.setdefault(key, []).append(t.realized_pnl or 0.0)

        breakdown = {}
        for key, pnls in groups.items():
            winners = [p for p in pnls if p > 0]
            breakdown[key] = {
                'count': len(pnls),
                'total_pnl': sum(pnls),
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0.0,
                'win_rate': len(winners) / len(pnls) if pnls else 0.0,
                'avg_winner': sum(winners) / len(winners) if winners else 0.0,
            }
        return breakdown
