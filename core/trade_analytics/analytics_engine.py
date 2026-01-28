"""
Trade Analytics Engine - Learn From Your Trades

The core analytics engine that answers questions like:
- "What's my win rate on hourly patterns?"
- "Is 1.5% magnitude filter optimal?"
- "What's my optimal TFC threshold?"
- "How much profit am I leaving on table?"
- "Win rate by VIX level?"

Provides:
1. Segmented Win Rate Analysis - win rate by any factor
2. Exit Efficiency Analysis - how well are we capturing profits
3. Parameter Sensitivity Analysis - optimal thresholds
4. Factor Correlation Analysis - what factors predict success

Session: Trade Analytics Implementation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import statistics

from core.trade_analytics.models import EnrichedTradeRecord
from core.trade_analytics.trade_store import TradeStore

logger = logging.getLogger(__name__)


@dataclass
class SegmentStats:
    """Statistics for a segment of trades."""
    segment_name: str
    segment_value: Any
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    avg_mfe: float
    avg_mae: float
    avg_exit_efficiency: float
    losers_went_green: int
    losers_went_green_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment': self.segment_name,
            'value': self.segment_value,
            'trades': self.trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(self.win_rate, 2),
            'total_pnl': round(self.total_pnl, 2),
            'avg_pnl': round(self.avg_pnl, 2),
            'avg_winner': round(self.avg_winner, 2),
            'avg_loser': round(self.avg_loser, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_mfe': round(self.avg_mfe, 2),
            'avg_mae': round(self.avg_mae, 2),
            'avg_exit_efficiency': round(self.avg_exit_efficiency, 2),
            'losers_went_green': self.losers_went_green,
            'losers_went_green_pct': round(self.losers_went_green_pct, 2),
        }


class TradeAnalyticsEngine:
    """
    Comprehensive trade analytics engine.
    
    CORE PHILOSOPHY:
    Every trade is a data point. Learn from wins AND losses.
    The goal is to find patterns that predict success.
    
    Usage:
        store = TradeStore(Path("trades.json"))
        analytics = TradeAnalyticsEngine(store)
        
        # Question: "What's my win rate on hourly patterns?"
        hourly_stats = analytics.win_rate_by_factor("timeframe", filter_value="1H")
        
        # Question: "Is 1.5% magnitude working?"
        mag_analysis = analytics.magnitude_sensitivity()
        
        # Question: "What's optimal TFC threshold?"
        tfc_analysis = analytics.tfc_sensitivity()
        
        # Question: "How much profit am I leaving on table?"
        efficiency = analytics.exit_efficiency_report()
        
        # Question: "Win rate by VIX level?"
        vix_stats = analytics.win_rate_by_factor("vix_level", bins=[15, 20, 25, 30])
    """
    
    def __init__(self, store: TradeStore):
        """
        Initialize analytics engine.
        
        Args:
            store: TradeStore with historical trades
        """
        self.store = store
    
    # =========================================================================
    # SEGMENTED WIN RATE ANALYSIS
    # =========================================================================
    
    def win_rate_by_factor(
        self,
        factor: str,
        bins: Optional[List[float]] = None,
        filter_value: Optional[Any] = None,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> List[SegmentStats]:
        """
        Calculate win rate segmented by any factor.
        
        THE KEY ANALYSIS METHOD - answers questions like:
        - "What's my win rate on hourly patterns?" -> factor="timeframe"
        - "Win rate by VIX level?" -> factor="vix_level", bins=[15, 20, 25, 30]
        - "Win rate by magnitude?" -> factor="magnitude_pct", bins=[1.0, 1.5, 2.0]
        - "Win rate by TFC score?" -> factor="tfc_score"
        - "Win rate by pattern type?" -> factor="pattern_type"
        
        Args:
            factor: Factor to segment by. Supports dot notation for nested fields:
                - "timeframe" -> pattern.timeframe
                - "pattern_type" -> pattern.pattern_type
                - "tfc_score" -> pattern.tfc_score
                - "magnitude_pct" -> pattern.magnitude_pct
                - "vix_level" -> market.vix_level
                - "vix_regime" -> market.vix_regime
                - "atr_percent" -> market.atr_percent
                - "exit_reason" -> exit_reason
                - "day_of_week" -> market.day_of_week
            bins: For numeric factors, bin edges (e.g., [15, 20, 25, 30] for VIX)
            filter_value: If set, only return stats for this specific value
            trades: Optional list of trades to analyze (defaults to all closed)
        
        Returns:
            List of SegmentStats, one per segment
        """
        if trades is None:
            trades = self.store.get_closed_trades()
        
        if not trades:
            return []
        
        # Extract factor values
        factor_values = []
        for trade in trades:
            value = self._get_factor_value(trade, factor)
            factor_values.append((trade, value))
        
        # Group by factor
        if bins is not None:
            # Bin numeric values
            segments = self._bin_trades(factor_values, bins, factor)
        else:
            # Group by discrete values
            segments = self._group_trades(factor_values)
        
        # Calculate stats for each segment
        results = []
        for segment_name, segment_trades in segments.items():
            if filter_value is not None and segment_name != str(filter_value):
                continue
            
            stats = self._calculate_segment_stats(factor, segment_name, segment_trades)
            results.append(stats)
        
        # Sort by segment name/value
        results.sort(key=lambda s: str(s.segment_value))
        
        return results
    
    def _get_factor_value(self, trade: EnrichedTradeRecord, factor: str) -> Any:
        """Extract factor value from trade, supporting nested fields."""
        # Direct fields
        if hasattr(trade, factor):
            return getattr(trade, factor)
        
        # Pattern fields
        if hasattr(trade.pattern, factor):
            return getattr(trade.pattern, factor)
        
        # Market fields
        if hasattr(trade.market, factor):
            return getattr(trade.market, factor)
        
        # Position fields
        if hasattr(trade.position, factor):
            return getattr(trade.position, factor)
        
        # Excursion fields
        if hasattr(trade.excursion, factor):
            return getattr(trade.excursion, factor)
        
        # Common shortcuts
        shortcuts = {
            'timeframe': lambda t: t.pattern.timeframe,
            'pattern_type': lambda t: t.pattern.pattern_type,
            'tfc_score': lambda t: t.pattern.tfc_score,
            'magnitude_pct': lambda t: t.pattern.magnitude_pct,
            'vix_level': lambda t: t.market.vix_level,
            'vix_regime': lambda t: t.market.vix_regime,
            'atr_percent': lambda t: t.market.atr_percent,
            'day_of_week': lambda t: t.market.day_of_week,
            'hour_of_day': lambda t: t.market.hour_of_day,
        }
        
        if factor in shortcuts:
            return shortcuts[factor](trade)
        
        logger.warning(f"Unknown factor: {factor}")
        return None
    
    def _bin_trades(
        self,
        factor_values: List[Tuple[EnrichedTradeRecord, Any]],
        bins: List[float],
        factor: str,
    ) -> Dict[str, List[EnrichedTradeRecord]]:
        """Bin trades by numeric factor value."""
        segments: Dict[str, List[EnrichedTradeRecord]] = {}
        
        # Create bin labels
        bin_labels = []
        for i, edge in enumerate(bins):
            if i == 0:
                bin_labels.append(f"<{edge}")
            else:
                bin_labels.append(f"{bins[i-1]}-{edge}")
        bin_labels.append(f">{bins[-1]}")
        
        # Initialize segments
        for label in bin_labels:
            segments[label] = []
        
        # Assign trades to bins
        for trade, value in factor_values:
            if value is None:
                continue
            
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                continue
            
            # Find appropriate bin
            assigned = False
            for i, edge in enumerate(bins):
                if numeric_value < edge:
                    segments[bin_labels[i]].append(trade)
                    assigned = True
                    break
            
            if not assigned:
                segments[bin_labels[-1]].append(trade)
        
        return segments
    
    def _group_trades(
        self,
        factor_values: List[Tuple[EnrichedTradeRecord, Any]],
    ) -> Dict[str, List[EnrichedTradeRecord]]:
        """Group trades by discrete factor value."""
        segments: Dict[str, List[EnrichedTradeRecord]] = {}
        
        for trade, value in factor_values:
            key = str(value) if value is not None else "Unknown"
            if key not in segments:
                segments[key] = []
            segments[key].append(trade)
        
        return segments
    
    def _calculate_segment_stats(
        self,
        factor: str,
        segment_name: str,
        trades: List[EnrichedTradeRecord],
    ) -> SegmentStats:
        """Calculate comprehensive stats for a segment."""
        if not trades:
            return SegmentStats(
                segment_name=factor,
                segment_value=segment_name,
                trades=0, wins=0, losses=0, win_rate=0,
                total_pnl=0, avg_pnl=0, avg_winner=0, avg_loser=0,
                profit_factor=0, avg_mfe=0, avg_mae=0,
                avg_exit_efficiency=0, losers_went_green=0, losers_went_green_pct=0,
            )
        
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        
        # MFE/MAE stats
        mfe_values = [t.excursion.mfe_pnl for t in trades if t.excursion.mfe_pnl != 0]
        mae_values = [t.excursion.mae_pnl for t in trades if t.excursion.mae_pnl != 0]
        efficiency_values = [t.excursion.exit_efficiency for t in trades if t.excursion.mfe_pnl > 0]
        
        losers_went_green = sum(1 for t in losers if t.excursion.went_green_before_loss)
        
        return SegmentStats(
            segment_name=factor,
            segment_value=segment_name,
            trades=len(trades),
            wins=len(winners),
            losses=len(losers),
            win_rate=len(winners) / len(trades) * 100 if trades else 0,
            total_pnl=total_pnl,
            avg_pnl=total_pnl / len(trades) if trades else 0,
            avg_winner=gross_profit / len(winners) if winners else 0,
            avg_loser=gross_loss / len(losers) if losers else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            avg_mfe=statistics.mean(mfe_values) if mfe_values else 0,
            avg_mae=statistics.mean(mae_values) if mae_values else 0,
            avg_exit_efficiency=statistics.mean(efficiency_values) if efficiency_values else 0,
            losers_went_green=losers_went_green,
            losers_went_green_pct=losers_went_green / len(losers) * 100 if losers else 0,
        )
    
    # =========================================================================
    # EXIT EFFICIENCY ANALYSIS
    # =========================================================================
    
    def exit_efficiency_report(
        self,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze how well we're capturing available profit.
        
        THE KEY INSIGHT: Are we leaving money on the table?
        
        If MFE consistently 2x actual exit -> leaving money on table
        If MAE consistently hitting stop before MFE -> stops too tight
        
        Returns:
            {
                'winners': {
                    'count': int,
                    'avg_exit_efficiency': float,  # exit_pnl / mfe
                    'avg_mfe_pct': float,
                    'avg_actual_profit_pct': float,
                    'profit_left_on_table': float,  # total MFE - total exit
                    'mfe_vs_target': str,  # "MFE typically 1.5x target"
                },
                'losers': {
                    'count': int,
                    'avg_mfe_before_loss': float,  # How much did losers go up?
                    'pct_went_green_first': float,  # % of losers that went positive
                    'avg_mae_pct': float,
                    'avg_actual_loss_pct': float,
                    'mae_vs_stop': str,  # "MAE typically hit stop"
                },
                'insights': [str],  # Actionable recommendations
            }
        """
        if trades is None:
            trades = self.store.get_closed_trades()
        
        if not trades:
            return {'message': 'No closed trades for analysis'}
        
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        # Winner analysis
        winner_stats = self._analyze_winner_exits(winners)
        
        # Loser analysis
        loser_stats = self._analyze_loser_exits(losers)
        
        # Generate insights
        insights = self._generate_exit_insights(winner_stats, loser_stats)
        
        return {
            'winners': winner_stats,
            'losers': loser_stats,
            'insights': insights,
        }
    
    def _analyze_winner_exits(self, winners: List[EnrichedTradeRecord]) -> Dict[str, Any]:
        """Analyze exit efficiency for winning trades."""
        if not winners:
            return {'count': 0}
        
        efficiencies = []
        mfe_pcts = []
        profit_pcts = []
        total_mfe = 0
        total_exit = 0
        
        for w in winners:
            if w.excursion.mfe_pnl > 0:
                efficiencies.append(w.excursion.exit_efficiency)
                total_mfe += w.excursion.mfe_pnl
            if w.excursion.mfe_pct != 0:
                mfe_pcts.append(w.excursion.mfe_pct)
            if w.pnl_pct != 0:
                profit_pcts.append(w.pnl_pct)
            total_exit += w.pnl
        
        profit_left = total_mfe - total_exit
        
        return {
            'count': len(winners),
            'avg_exit_efficiency': statistics.mean(efficiencies) if efficiencies else 0,
            'avg_mfe_pct': statistics.mean(mfe_pcts) if mfe_pcts else 0,
            'avg_actual_profit_pct': statistics.mean(profit_pcts) if profit_pcts else 0,
            'profit_left_on_table': profit_left,
            'total_mfe': total_mfe,
            'total_exit': total_exit,
        }
    
    def _analyze_loser_exits(self, losers: List[EnrichedTradeRecord]) -> Dict[str, Any]:
        """Analyze exit efficiency for losing trades."""
        if not losers:
            return {'count': 0}
        
        mfe_before_loss = []
        mae_pcts = []
        loss_pcts = []
        went_green = 0
        
        for l in losers:
            if l.excursion.mfe_pnl > 0:
                mfe_before_loss.append(l.excursion.mfe_pnl)
            if l.excursion.went_green_before_loss:
                went_green += 1
            if l.excursion.mae_pct != 0:
                mae_pcts.append(l.excursion.mae_pct)
            if l.pnl_pct != 0:
                loss_pcts.append(abs(l.pnl_pct))
        
        return {
            'count': len(losers),
            'avg_mfe_before_loss': statistics.mean(mfe_before_loss) if mfe_before_loss else 0,
            'pct_went_green_first': went_green / len(losers) * 100,
            'losers_went_green': went_green,
            'avg_mae_pct': statistics.mean(mae_pcts) if mae_pcts else 0,
            'avg_actual_loss_pct': statistics.mean(loss_pcts) if loss_pcts else 0,
        }
    
    def _generate_exit_insights(
        self,
        winner_stats: Dict[str, Any],
        loser_stats: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable insights from exit analysis."""
        insights = []
        
        # Winner insights
        if winner_stats.get('avg_exit_efficiency', 0) < 0.5:
            insights.append(
                f"⚠️ LOW EXIT EFFICIENCY: Capturing only "
                f"{winner_stats['avg_exit_efficiency']:.0%} of available profit. "
                f"Consider wider targets or trailing stops."
            )
        
        if winner_stats.get('profit_left_on_table', 0) > 0:
            insights.append(
                f"💰 PROFIT LEFT: ${winner_stats['profit_left_on_table']:.2f} "
                f"additional profit was available but not captured."
            )
        
        # Loser insights
        pct_green = loser_stats.get('pct_went_green_first', 0)
        if pct_green > 50:
            insights.append(
                f"⚠️ {pct_green:.0f}% of losers went GREEN before losing. "
                f"Consider taking partial profits earlier or tighter trailing stops."
            )
        
        avg_mfe_before = loser_stats.get('avg_mfe_before_loss', 0)
        if avg_mfe_before > 50:  # $50 avg profit before reversal
            insights.append(
                f"💡 Losers averaged ${avg_mfe_before:.2f} profit before reversing. "
                f"A trailing stop could have captured some of this."
            )
        
        return insights
    
    # =========================================================================
    # PARAMETER SENSITIVITY ANALYSIS
    # =========================================================================
    
    def magnitude_sensitivity(
        self,
        thresholds: Optional[List[float]] = None,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Answer: "Is 1.5% magnitude filter optimal?"
        
        Simulates different magnitude thresholds to find optimal filter.
        
        Args:
            thresholds: Magnitude % thresholds to test (default: 0.5-3.0%)
            trades: Trades to analyze
        
        Returns:
            List of stats per threshold showing trades_included, win_rate, avg_pnl
        """
        if trades is None:
            trades = self.store.get_closed_trades()
        
        if thresholds is None:
            thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        results = []
        
        for threshold in thresholds:
            # Filter trades that would pass this magnitude threshold
            filtered = [
                t for t in trades
                if t.pattern.magnitude_pct >= threshold
            ]
            
            if not filtered:
                results.append({
                    'threshold': threshold,
                    'trades': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'profit_factor': 0,
                })
                continue
            
            winners = [t for t in filtered if t.is_winner]
            total_pnl = sum(t.pnl for t in filtered)
            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in filtered if not t.is_winner))
            
            results.append({
                'threshold': threshold,
                'trades': len(filtered),
                'trades_excluded': len(trades) - len(filtered),
                'win_rate': len(winners) / len(filtered) * 100,
                'avg_pnl': total_pnl / len(filtered),
                'total_pnl': total_pnl,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            })
        
        return results
    
    def tfc_sensitivity(
        self,
        thresholds: Optional[List[int]] = None,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Answer: "Is TFC 4/5 better than 3/5?"
        
        Simulates different TFC score thresholds.
        
        Args:
            thresholds: TFC scores to test (default: 2, 3, 4, 5)
            trades: Trades to analyze
        
        Returns:
            List of stats per threshold
        """
        if trades is None:
            trades = self.store.get_closed_trades()
        
        if thresholds is None:
            thresholds = [2, 3, 4, 5]
        
        results = []
        
        for threshold in thresholds:
            # Filter trades that would pass this TFC threshold
            filtered = [
                t for t in trades
                if t.pattern.tfc_score >= threshold
            ]
            
            if not filtered:
                results.append({
                    'min_tfc': threshold,
                    'trades': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'profit_factor': 0,
                })
                continue
            
            winners = [t for t in filtered if t.is_winner]
            total_pnl = sum(t.pnl for t in filtered)
            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in filtered if not t.is_winner))
            
            results.append({
                'min_tfc': threshold,
                'trades': len(filtered),
                'trades_excluded': len(trades) - len(filtered),
                'win_rate': len(winners) / len(filtered) * 100,
                'avg_pnl': total_pnl / len(filtered),
                'total_pnl': total_pnl,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            })
        
        return results
    
    def vix_sensitivity(
        self,
        bins: Optional[List[float]] = None,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> List[SegmentStats]:
        """
        Answer: "What VIX level am I most profitable at?"
        
        Analyzes performance by VIX regime.
        
        Args:
            bins: VIX bin edges (default: [15, 20, 25, 30])
            trades: Trades to analyze
        
        Returns:
            SegmentStats per VIX bin
        """
        if bins is None:
            bins = [15, 20, 25, 30]
        
        return self.win_rate_by_factor("vix_level", bins=bins, trades=trades)
    
    # =========================================================================
    # PATTERN PERFORMANCE COMPARISON
    # =========================================================================
    
    def pattern_comparison(
        self,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance across pattern types.
        
        Answer: "Are 2-1-2 patterns better than 3-2 patterns?"
        
        Returns:
            Dict of pattern_type -> stats
        """
        if trades is None:
            trades = self.store.get_closed_trades()
        
        stats = self.win_rate_by_factor("pattern_type", trades=trades)
        
        return {
            s.segment_value: s.to_dict()
            for s in stats
        }
    
    def timeframe_comparison(
        self,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance across timeframes.
        
        Answer: "Are hourly patterns profitable?"
        
        Returns:
            Dict of timeframe -> stats
        """
        if trades is None:
            trades = self.store.get_closed_trades()
        
        stats = self.win_rate_by_factor("timeframe", trades=trades)
        
        return {
            s.segment_value: s.to_dict()
            for s in stats
        }
    
    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================
    
    def generate_report(
        self,
        trades: Optional[List[EnrichedTradeRecord]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report.
        
        This is the "one function to rule them all" - generates a complete
        analysis of trading performance.
        
        Returns:
            {
                'summary': overall stats,
                'by_timeframe': performance by timeframe,
                'by_pattern': performance by pattern type,
                'by_vix': performance by VIX level,
                'exit_efficiency': how well profits captured,
                'magnitude_sensitivity': optimal magnitude filter,
                'tfc_sensitivity': optimal TFC threshold,
                'insights': actionable recommendations,
            }
        """
        if trades is None:
            trades = self.store.get_closed_trades()
        
        if not trades:
            return {'message': 'No closed trades for analysis'}
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_trades': len(trades),
        }
        
        # Overall summary
        report['summary'] = self.store.get_stats()
        
        # By timeframe
        report['by_timeframe'] = self.timeframe_comparison(trades)
        
        # By pattern
        report['by_pattern'] = self.pattern_comparison(trades)
        
        # By VIX
        vix_stats = self.vix_sensitivity(trades=trades)
        report['by_vix'] = {s.segment_value: s.to_dict() for s in vix_stats}
        
        # Exit efficiency
        report['exit_efficiency'] = self.exit_efficiency_report(trades)
        
        # Sensitivity analyses
        report['magnitude_sensitivity'] = self.magnitude_sensitivity(trades=trades)
        report['tfc_sensitivity'] = self.tfc_sensitivity(trades=trades)
        
        # Generate overall insights
        report['insights'] = self._generate_overall_insights(report)
        
        return report
    
    def _generate_overall_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate overall insights from full report."""
        insights = []
        
        # Exit efficiency insights
        if 'exit_efficiency' in report and 'insights' in report['exit_efficiency']:
            insights.extend(report['exit_efficiency']['insights'])
        
        # Timeframe insights
        if 'by_timeframe' in report:
            best_tf = max(
                report['by_timeframe'].items(),
                key=lambda x: x[1].get('win_rate', 0) if x[1].get('trades', 0) >= 5 else 0
            )
            if best_tf[1].get('trades', 0) >= 5:
                insights.append(
                    f"📊 BEST TIMEFRAME: {best_tf[0]} with "
                    f"{best_tf[1]['win_rate']:.0f}% win rate across "
                    f"{best_tf[1]['trades']} trades"
                )
        
        # VIX insights
        if 'by_vix' in report:
            for vix_range, stats in report['by_vix'].items():
                if stats.get('trades', 0) >= 5 and stats.get('win_rate', 0) < 40:
                    insights.append(
                        f"⚠️ LOW WIN RATE in VIX {vix_range}: "
                        f"{stats['win_rate']:.0f}%. Consider reducing position size."
                    )
        
        # Magnitude insights
        if 'magnitude_sensitivity' in report:
            mag_data = report['magnitude_sensitivity']
            if len(mag_data) > 1:
                best_mag = max(
                    mag_data,
                    key=lambda x: x.get('profit_factor', 0) if x.get('trades', 0) >= 5 else 0
                )
                if best_mag.get('trades', 0) >= 5:
                    insights.append(
                        f"💡 OPTIMAL MAGNITUDE: {best_mag['threshold']}% "
                        f"(PF: {best_mag['profit_factor']:.2f}, "
                        f"WR: {best_mag['win_rate']:.0f}%)"
                    )
        
        return insights
