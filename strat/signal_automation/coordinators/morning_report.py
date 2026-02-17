"""
EQUITY-112: MorningReportGenerator - Pre-market morning report coordinator.

Generates a 6 AM ET pre-market report including:
1. Fresh STRAT setups from the ticker selection pipeline
2. Pre-market gap analysis (pre-market price vs previous close)
3. Open positions with unrealized P&L
4. Yesterday's closed trade performance recap
5. Capital status summary

Follows the coordinator pattern established in EQUITY-85 (HealthMonitor).
"""

import logging
import time
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Any, Optional, Protocol

from strat.signal_automation.config import MorningReportConfig

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
except ImportError:
    StockHistoricalDataClient = None  # type: ignore[assignment,misc]
    StockSnapshotRequest = None  # type: ignore[assignment,misc]

try:
    from integrations.alpaca_trading_client import get_alpaca_credentials
except ImportError:
    get_alpaca_credentials = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PositionMonitorProtocol(Protocol):
    """Methods needed from PositionMonitor."""
    def get_tracked_positions(self) -> List[Any]: ...


class TradingClientProtocol(Protocol):
    """Methods needed from AlpacaTradingClient."""
    def get_closed_trades(self, **kwargs) -> List[Dict]: ...


class CapitalTrackerProtocol(Protocol):
    """Methods needed from VirtualBalanceTracker."""
    def get_summary(self) -> Dict[str, Any]: ...


class MorningReportGenerator:
    """
    Generates pre-market morning reports for Discord delivery.

    Constructor dependencies match the coordinator pattern:
    - alerters: for sending the Discord embed
    - position_monitor: for open positions
    - capital_tracker: for balance/heat info
    - trading_client: for Alpaca closed trades (yesterday recap)
    - config: MorningReportConfig
    """

    def __init__(
        self,
        alerters: List[Any],
        position_monitor: Optional[PositionMonitorProtocol] = None,
        capital_tracker: Optional[CapitalTrackerProtocol] = None,
        trading_client: Optional[TradingClientProtocol] = None,
        config: Optional[MorningReportConfig] = None,
    ):
        self._alerters = alerters
        self._position_monitor = position_monitor
        self._capital_tracker = capital_tracker
        self._trading_client = trading_client
        self._config = config or MorningReportConfig()

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate the full morning report.

        Returns:
            Report data dict with sections: setups, gaps, positions,
            yesterday, capital, pipeline_stats.
        """
        report_start = time.time()
        today = date.today()

        report = {
            'date': today.isoformat(),
            'setups': [],
            'gaps': [],
            'open_positions': [],
            'yesterday': {},
            'capital': {},
            'pipeline_stats': {},
            'anomalies': [],
        }

        # Section 1: Run ticker selection pipeline for fresh setups
        candidates, pipeline_stats = self._run_pipeline()
        report['setups'] = candidates[:self._config.max_candidates]
        report['pipeline_stats'] = pipeline_stats

        # Section 2: Gap analysis using Alpaca pre-market snapshots
        if candidates:
            report['gaps'] = self._get_gap_analysis(candidates)

        # Section 3: Open positions
        report['open_positions'] = self._get_open_positions()

        # Section 4: Yesterday's performance recap
        report['yesterday'] = self._get_yesterday_recap()

        # Section 5: Capital status
        report['capital'] = self._get_capital_status()

        report['duration_seconds'] = round(time.time() - report_start, 1)

        logger.info(
            f"Morning report generated: {len(report['setups'])} setups, "
            f"{len(report['gaps'])} gaps, "
            f"{len(report['open_positions'])} positions, "
            f"duration: {report['duration_seconds']}s"
        )

        return report

    def _run_pipeline(self) -> tuple:
        """
        Run the ticker selection pipeline and extract candidates.

        Returns:
            (candidates_list, pipeline_stats_dict)
        """
        try:
            from strat.ticker_selection.pipeline import run_selection

            logger.info("Morning report: running ticker selection pipeline...")
            result = run_selection(dry_run=False)

            candidates = result.get('candidates', [])
            stats = result.get('pipeline_stats', {})

            logger.info(
                f"Pipeline complete: {stats.get('final_candidates', 0)} candidates, "
                f"{stats.get('scan_duration_seconds', 0):.1f}s"
            )
            return candidates, stats

        except Exception as e:
            logger.error(f"Ticker selection pipeline failed: {e}")
            return [], {'error': str(e)}

    def _get_gap_analysis(self, candidates: List[Dict]) -> List[Dict]:
        """
        Calculate pre-market gap % for candidate symbols.

        Uses Alpaca StockHistoricalDataClient to fetch snapshots with
        latest trade/quote and previous daily bar close.
        """
        symbols = [c.get('symbol', '') for c in candidates if c.get('symbol')]
        if not symbols:
            return []

        try:
            if StockHistoricalDataClient is None or get_alpaca_credentials is None:
                logger.warning("Alpaca SDK not available for gap analysis")
                return []

            creds = get_alpaca_credentials('SMALL')
            client = StockHistoricalDataClient(
                api_key=creds['api_key'],
                secret_key=creds['secret_key'],
            )

            request = StockSnapshotRequest(symbol_or_symbols=symbols)
            snapshots = client.get_stock_snapshot(request)

            gaps = []
            for symbol, snap in snapshots.items():
                try:
                    prev_close = (
                        float(snap.previous_daily_bar.close)
                        if snap.previous_daily_bar else None
                    )
                    # Use latest trade for pre-market price
                    premarket_price = (
                        float(snap.latest_trade.price)
                        if snap.latest_trade else None
                    )

                    if prev_close and premarket_price and prev_close > 0:
                        gap_pct = (premarket_price - prev_close) / prev_close * 100
                        if abs(gap_pct) >= self._config.min_gap_pct:
                            gaps.append({
                                'symbol': str(symbol),
                                'prev_close': prev_close,
                                'premarket_price': premarket_price,
                                'gap_pct': round(gap_pct, 2),
                            })
                except Exception as e:
                    logger.debug(f"Gap calc skip {symbol}: {e}")

            # Sort by absolute gap magnitude
            gaps.sort(key=lambda g: abs(g['gap_pct']), reverse=True)
            return gaps

        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return []

    def _get_open_positions(self) -> List[Dict]:
        """Get current open positions from position monitor."""
        if self._position_monitor is None:
            return []

        try:
            positions = self._position_monitor.get_tracked_positions()
            return [
                {
                    'symbol': getattr(pos, 'symbol', 'Unknown'),
                    'pattern_type': getattr(pos, 'pattern_type', 'Unknown'),
                    'timeframe': getattr(pos, 'timeframe', 'Unknown'),
                    'unrealized_pnl': getattr(pos, 'unrealized_pnl', 0.0),
                    'unrealized_pct': getattr(pos, 'unrealized_pct', 0.0),
                }
                for pos in positions
            ]
        except Exception as e:
            logger.warning(f"Failed to get open positions: {e}")
            return []

    def _get_yesterday_recap(self) -> Dict[str, Any]:
        """Get yesterday's closed trade performance from Alpaca."""
        if self._trading_client is None:
            return {}

        try:
            today = date.today()
            yesterday_start = datetime.combine(
                today - timedelta(days=1),
                datetime.min.time(),
            ).replace(tzinfo=timezone.utc)
            today_start = datetime.combine(
                today, datetime.min.time()
            ).replace(tzinfo=timezone.utc)

            closed = self._trading_client.get_closed_trades(
                after=yesterday_start,
                options_only=True,
            )

            wins = 0
            losses = 0
            total_pnl = 0.0
            gross_profit = 0.0
            gross_loss = 0.0

            for trade in closed:
                sell_time = trade.get('sell_time_dt')
                if sell_time and sell_time.date() >= (today - timedelta(days=1)):
                    if sell_time.date() < today:
                        pnl = trade.get('realized_pnl', 0)
                        total_pnl += pnl
                        if pnl > 0:
                            wins += 1
                            gross_profit += pnl
                        elif pnl < 0:
                            losses += 1
                            gross_loss += abs(pnl)

            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

            return {
                'trades': total_trades,
                'wins': wins,
                'losses': losses,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
            }

        except Exception as e:
            logger.warning(f"Failed to get yesterday recap: {e}")
            return {}

    def _get_capital_status(self) -> Dict[str, Any]:
        """Get capital tracking summary."""
        if self._capital_tracker is None:
            return {}

        try:
            return self._capital_tracker.get_summary()
        except Exception as e:
            logger.warning(f"Failed to get capital status: {e}")
            return {}

    def run(self) -> None:
        """Generate the morning report and send via all alerters."""
        try:
            report_data = self.generate_report()

            for alerter in self._alerters:
                if hasattr(alerter, 'send_morning_report'):
                    try:
                        alerter.send_morning_report(report_data)
                    except Exception as e:
                        logger.error(
                            f"Failed to send morning report via "
                            f"{getattr(alerter, 'name', 'unknown')}: {e}"
                        )

            logger.info("Morning report completed and sent")

        except Exception as e:
            logger.error(f"Morning report generation failed: {e}")
            raise
