"""
ATLAS Production Readiness Validation - Pattern Metrics Analyzer

Analyzes STRAT pattern performance with per-pattern, per-timeframe,
and per-regime breakdowns. Includes options accuracy metrics.

Session 83G: Pattern metrics implementation per ATLAS Checklist Section 9.2.

Usage:
    from validation.pattern_metrics import PatternMetricsAnalyzer
    from strat.pattern_metrics import PatternTradeResult

    # Create analyzer
    analyzer = PatternMetricsAnalyzer()

    # Analyze trades
    trades = [PatternTradeResult(...), ...]
    results = analyzer.analyze(trades)

    # Print summary
    print(results.summary())

    # Access breakdowns
    print(results.by_pattern['3-1-2U'].win_rate)
    print(results.by_timeframe['1D']['3-1-2U'].expectancy)
    print(results.by_regime['TREND_BULL']['3-1-2U'].profit_factor)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from validation.config import PatternMetricsConfig
from validation.results import (
    PatternStats,
    OptionsAccuracyMetrics,
    PatternMetricsResults,
)

# Import will be deferred to avoid circular imports
# from strat.pattern_metrics import PatternTradeResult


@dataclass
class PatternMetricsAnalyzer:
    """
    Analyzes STRAT pattern performance.

    Provides:
    - Per-pattern breakdown (3-1-2, 2-1-2, 2-2, etc.)
    - Per-timeframe breakdown (1D, 1W, 1M)
    - Per-regime breakdown (TREND_BULL, TREND_BEAR, etc.)
    - Options accuracy metrics (ThetaData coverage)

    Attributes:
        config: PatternMetricsConfig with analysis parameters
    """
    config: PatternMetricsConfig = None

    def __post_init__(self):
        """Initialize default config if not provided."""
        if self.config is None:
            self.config = PatternMetricsConfig()

    def analyze(
        self,
        trades: List['PatternTradeResult'],
    ) -> PatternMetricsResults:
        """
        Perform full pattern metrics analysis.

        Args:
            trades: List of PatternTradeResult instances

        Returns:
            PatternMetricsResults with all breakdowns
        """
        if not trades:
            return self._empty_results()

        # Analyze by different dimensions
        by_pattern = self._analyze_by_pattern(trades)
        by_timeframe = self._analyze_by_timeframe(trades)
        by_regime = self._analyze_by_regime(trades)

        # Check for options trades and calculate accuracy
        options_trades = [t for t in trades if t.is_options_trade]
        options_accuracy = None
        if options_trades:
            options_accuracy = self._calculate_options_accuracy(options_trades)

        # Calculate overall metrics
        total_trades = len(trades)
        winners = [t for t in trades if t.is_winner]
        overall_win_rate = len(winners) / total_trades if total_trades > 0 else 0.0

        total_pnl = sum(t.pnl for t in trades)
        overall_expectancy = total_pnl / total_trades if total_trades > 0 else 0.0

        # Find best and worst patterns
        best_pattern = None
        worst_pattern = None
        if by_pattern:
            sorted_patterns = sorted(
                by_pattern.items(),
                key=lambda x: x[1].expectancy,
                reverse=True
            )
            if sorted_patterns:
                best_pattern = sorted_patterns[0][0]
                worst_pattern = sorted_patterns[-1][0]

        return PatternMetricsResults(
            by_pattern=by_pattern,
            by_timeframe=by_timeframe,
            by_regime=by_regime,
            options_accuracy=options_accuracy,
            total_trades=total_trades,
            overall_win_rate=overall_win_rate,
            overall_expectancy=overall_expectancy,
            best_pattern=best_pattern,
            worst_pattern=worst_pattern,
        )

    def _analyze_by_pattern(
        self,
        trades: List['PatternTradeResult'],
    ) -> Dict[str, PatternStats]:
        """
        Group and analyze trades by pattern type.

        Args:
            trades: List of PatternTradeResult

        Returns:
            Dict mapping pattern_type to PatternStats
        """
        # Group trades by pattern
        pattern_groups: Dict[str, List] = {}
        for trade in trades:
            pattern = trade.pattern_type
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(trade)

        # Calculate stats for each pattern
        results = {}
        for pattern_type, pattern_trades in pattern_groups.items():
            if len(pattern_trades) >= self.config.min_pattern_trades:
                stats = self._calculate_pattern_stats(pattern_type, pattern_trades)
                results[pattern_type] = stats

        return results

    def _analyze_by_timeframe(
        self,
        trades: List['PatternTradeResult'],
    ) -> Dict[str, Dict[str, PatternStats]]:
        """
        Group and analyze trades by timeframe, then by pattern.

        Args:
            trades: List of PatternTradeResult

        Returns:
            Dict mapping timeframe -> pattern_type -> PatternStats
        """
        # Group by timeframe first
        tf_groups: Dict[str, List] = {}
        for trade in trades:
            tf = trade.timeframe
            if tf not in tf_groups:
                tf_groups[tf] = []
            tf_groups[tf].append(trade)

        # For each timeframe, analyze by pattern
        results = {}
        for timeframe, tf_trades in tf_groups.items():
            pattern_stats = self._analyze_by_pattern(tf_trades)
            if pattern_stats:
                results[timeframe] = pattern_stats

        return results

    def _analyze_by_regime(
        self,
        trades: List['PatternTradeResult'],
    ) -> Dict[str, Dict[str, PatternStats]]:
        """
        Group and analyze trades by ATLAS regime, then by pattern.

        Args:
            trades: List of PatternTradeResult

        Returns:
            Dict mapping regime -> pattern_type -> PatternStats
        """
        # Group by regime first
        regime_groups: Dict[str, List] = {}
        for trade in trades:
            regime = trade.regime
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(trade)

        # For each regime, analyze by pattern
        results = {}
        for regime, regime_trades in regime_groups.items():
            pattern_stats = self._analyze_by_pattern(regime_trades)
            if pattern_stats:
                results[regime] = pattern_stats

        return results

    def _calculate_pattern_stats(
        self,
        pattern_type: str,
        trades: List['PatternTradeResult'],
    ) -> PatternStats:
        """
        Calculate comprehensive statistics for a pattern type.

        Args:
            pattern_type: Pattern identifier string
            trades: List of trades for this pattern

        Returns:
            PatternStats dataclass
        """
        if not trades:
            return self._empty_pattern_stats(pattern_type)

        trade_count = len(trades)
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0

        # P/L calculations
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / trade_count if trade_count > 0 else 0.0

        # Average winner/loser
        avg_winner_pnl = (
            sum(t.pnl for t in winners) / win_count if win_count > 0 else 0.0
        )
        avg_loser_pnl = (
            sum(t.pnl for t in losers) / loss_count if loss_count > 0 else 0.0
        )

        # Profit factor
        gross_profit = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy = (Win Rate * Avg Win) + ((1 - Win Rate) * Avg Loss)
        # Note: avg_loser_pnl is already negative for losers
        expectancy = (win_rate * avg_winner_pnl) + ((1 - win_rate) * avg_loser_pnl)

        # Average R:R
        rr_values = [t.planned_risk_reward for t in trades if t.planned_risk_reward > 0]
        avg_risk_reward = sum(rr_values) / len(rr_values) if rr_values else 0.0

        # Average days held
        days_held = [t.days_held for t in trades if t.days_held > 0]
        avg_days_held = sum(days_held) / len(days_held) if days_held else 0.0

        return PatternStats(
            pattern_type=pattern_type,
            trade_count=trade_count,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_winner_pnl=avg_winner_pnl,
            avg_loser_pnl=avg_loser_pnl,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            expectancy=expectancy,
            avg_risk_reward=avg_risk_reward,
            avg_days_held=avg_days_held,
        )

    def _calculate_options_accuracy(
        self,
        trades: List['PatternTradeResult'],
    ) -> OptionsAccuracyMetrics:
        """
        Calculate options-specific accuracy metrics.

        Tracks ThetaData coverage, price accuracy, delta quality,
        and theta cost impact.

        Args:
            trades: List of options trades

        Returns:
            OptionsAccuracyMetrics dataclass
        """
        if not trades:
            return self._empty_options_accuracy()

        total = len(trades)

        # Data source breakdown
        thetadata_trades = sum(1 for t in trades if t.data_source == 'ThetaData')
        bs_trades = sum(1 for t in trades if t.data_source == 'BlackScholes')
        mixed_trades = sum(1 for t in trades if t.data_source == 'Mixed')

        thetadata_coverage_pct = thetadata_trades / total if total > 0 else 0.0

        # Price accuracy - would need reference prices for MAE/MAPE/RMSE
        # For now, we'll use placeholders - actual implementation would
        # compare against benchmark prices
        price_mae = 0.0
        price_mape = 0.0
        price_rmse = 0.0

        # Delta accuracy
        deltas_entry = [t.entry_delta for t in trades if t.entry_delta is not None]
        deltas_exit = [t.exit_delta for t in trades if t.exit_delta is not None]

        avg_entry_delta = sum(deltas_entry) / len(deltas_entry) if deltas_entry else 0.0
        avg_exit_delta = sum(deltas_exit) / len(deltas_exit) if deltas_exit else 0.0

        # Delta MAE (difference from target 0.65 midpoint)
        target_delta = 0.65
        delta_errors = [abs(d - target_delta) for d in deltas_entry]
        delta_mae = sum(delta_errors) / len(delta_errors) if delta_errors else 0.0

        # Delta in optimal range (0.50-0.80 per config)
        min_delta, max_delta = self.config.optimal_delta_range
        in_range = sum(
            1 for d in deltas_entry
            if min_delta <= abs(d) <= max_delta
        )
        delta_in_optimal_range_pct = in_range / len(deltas_entry) if deltas_entry else 0.0

        # Theta cost analysis
        theta_costs = [t.theta_cost for t in trades if t.theta_cost is not None]
        avg_theta_cost = sum(theta_costs) / len(theta_costs) if theta_costs else 0.0

        # Theta cost as percentage of P/L
        total_theta = sum(theta_costs) if theta_costs else 0.0
        total_pnl = sum(abs(t.pnl) for t in trades)
        theta_cost_pct = abs(total_theta / total_pnl) if total_pnl > 0 else 0.0

        return OptionsAccuracyMetrics(
            thetadata_trades=thetadata_trades,
            black_scholes_trades=bs_trades,
            mixed_trades=mixed_trades,
            thetadata_coverage_pct=thetadata_coverage_pct,
            price_mae=price_mae,
            price_mape=price_mape,
            price_rmse=price_rmse,
            delta_mae=delta_mae,
            avg_entry_delta=avg_entry_delta,
            avg_exit_delta=avg_exit_delta,
            delta_in_optimal_range_pct=delta_in_optimal_range_pct,
            avg_theta_cost=avg_theta_cost,
            theta_cost_as_pct_of_pnl=theta_cost_pct,
        )

    def _empty_results(self) -> PatternMetricsResults:
        """Return empty results for no trades."""
        return PatternMetricsResults(
            by_pattern={},
            by_timeframe={},
            by_regime={},
            options_accuracy=None,
            total_trades=0,
            overall_win_rate=0.0,
            overall_expectancy=0.0,
            best_pattern=None,
            worst_pattern=None,
        )

    def _empty_pattern_stats(self, pattern_type: str) -> PatternStats:
        """Return empty PatternStats for a pattern type."""
        return PatternStats(
            pattern_type=pattern_type,
            trade_count=0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            avg_winner_pnl=0.0,
            avg_loser_pnl=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_risk_reward=0.0,
            avg_days_held=0.0,
        )

    def _empty_options_accuracy(self) -> OptionsAccuracyMetrics:
        """Return empty OptionsAccuracyMetrics."""
        return OptionsAccuracyMetrics(
            thetadata_trades=0,
            black_scholes_trades=0,
            mixed_trades=0,
            thetadata_coverage_pct=0.0,
            price_mae=0.0,
            price_mape=0.0,
            price_rmse=0.0,
            delta_mae=0.0,
            avg_entry_delta=0.0,
            avg_exit_delta=0.0,
            delta_in_optimal_range_pct=0.0,
            avg_theta_cost=0.0,
            theta_cost_as_pct_of_pnl=0.0,
        )


def analyze_pattern_metrics(
    trades: List['PatternTradeResult'],
    config: Optional[PatternMetricsConfig] = None,
) -> PatternMetricsResults:
    """
    Convenience function to analyze pattern metrics.

    Args:
        trades: List of PatternTradeResult instances
        config: Optional PatternMetricsConfig

    Returns:
        PatternMetricsResults with all breakdowns
    """
    analyzer = PatternMetricsAnalyzer(config=config)
    return analyzer.analyze(trades)


def get_best_patterns_by_metric(
    results: PatternMetricsResults,
    metric: str = 'expectancy',
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Get the best performing patterns by a specific metric.

    Args:
        results: PatternMetricsResults from analyze()
        metric: Metric to rank by ('expectancy', 'win_rate', 'profit_factor', 'avg_pnl')
        top_n: Number of top patterns to return

    Returns:
        List of (pattern_type, metric_value) tuples, sorted descending
    """
    if not results.by_pattern:
        return []

    pattern_metrics = []
    for pattern_type, stats in results.by_pattern.items():
        value = getattr(stats, metric, 0.0)
        pattern_metrics.append((pattern_type, value))

    # Sort descending
    pattern_metrics.sort(key=lambda x: x[1], reverse=True)

    return pattern_metrics[:top_n]


def get_regime_pattern_compatibility(
    results: PatternMetricsResults,
    min_win_rate: float = 0.50,
    min_expectancy: float = 0.0,
) -> Dict[str, List[str]]:
    """
    Identify which patterns work best in each regime.

    Args:
        results: PatternMetricsResults from analyze()
        min_win_rate: Minimum win rate to consider pattern viable
        min_expectancy: Minimum expectancy to consider pattern viable

    Returns:
        Dict mapping regime -> list of viable pattern types
    """
    compatibility = {}

    for regime, pattern_stats in results.by_regime.items():
        viable_patterns = []
        for pattern_type, stats in pattern_stats.items():
            if stats.win_rate >= min_win_rate and stats.expectancy >= min_expectancy:
                viable_patterns.append(pattern_type)
        compatibility[regime] = viable_patterns

    return compatibility


def generate_pattern_report(
    results: PatternMetricsResults,
    include_details: bool = True,
) -> str:
    """
    Generate a comprehensive pattern metrics report.

    Args:
        results: PatternMetricsResults from analyze()
        include_details: Whether to include per-timeframe and per-regime breakdowns

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "STRAT PATTERN METRICS REPORT",
        "=" * 70,
        "",
        f"Total Trades:       {results.total_trades}",
        f"Overall Win Rate:   {results.overall_win_rate:.1%}",
        f"Overall Expectancy: ${results.overall_expectancy:.2f}",
        "",
    ]

    # Pattern summary
    lines.append("--- BY PATTERN TYPE ---")
    lines.append(
        f"{'Pattern':<12} {'Trades':>8} {'Win%':>8} {'Avg P/L':>10} "
        f"{'Expect':>10} {'PF':>8} {'R:R':>8}"
    )
    lines.append("-" * 70)

    for pattern_type, stats in sorted(results.by_pattern.items()):
        pf_str = f"{stats.profit_factor:.2f}" if stats.profit_factor < 999 else "INF"
        lines.append(
            f"{pattern_type:<12} {stats.trade_count:>8} {stats.win_rate:>7.1%} "
            f"${stats.avg_pnl:>9.2f} ${stats.expectancy:>9.2f} "
            f"{pf_str:>8} {stats.avg_risk_reward:>7.2f}:1"
        )

    if results.best_pattern:
        lines.append("")
        lines.append(f"Best Pattern:  {results.best_pattern}")
    if results.worst_pattern:
        lines.append(f"Worst Pattern: {results.worst_pattern}")

    if include_details and results.by_timeframe:
        lines.append("")
        lines.append("--- BY TIMEFRAME ---")
        for timeframe, patterns in sorted(results.by_timeframe.items()):
            lines.append(f"\n[{timeframe}]")
            for pattern_type, stats in sorted(patterns.items()):
                lines.append(
                    f"  {pattern_type:<12} {stats.trade_count:>5} trades, "
                    f"{stats.win_rate:>5.1%} win, ${stats.expectancy:>8.2f} exp"
                )

    if include_details and results.by_regime:
        lines.append("")
        lines.append("--- BY REGIME ---")
        for regime, patterns in sorted(results.by_regime.items()):
            lines.append(f"\n[{regime}]")
            for pattern_type, stats in sorted(patterns.items()):
                lines.append(
                    f"  {pattern_type:<12} {stats.trade_count:>5} trades, "
                    f"{stats.win_rate:>5.1%} win, ${stats.expectancy:>8.2f} exp"
                )

    if results.options_accuracy:
        oa = results.options_accuracy
        lines.extend([
            "",
            "--- OPTIONS ACCURACY ---",
            f"ThetaData Trades:    {oa.thetadata_trades}",
            f"Black-Scholes:       {oa.black_scholes_trades}",
            f"Mixed Source:        {oa.mixed_trades}",
            f"ThetaData Coverage:  {oa.thetadata_coverage_pct:.1%}",
            f"Avg Entry Delta:     {oa.avg_entry_delta:.3f}",
            f"Delta in Opt Range:  {oa.delta_in_optimal_range_pct:.1%}",
            f"Avg Theta Cost:      ${oa.avg_theta_cost:.2f}",
            f"Theta % of P/L:      {oa.theta_cost_as_pct_of_pnl:.1%}",
        ])

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
