"""
Backtest Configuration - Mirrors SignalAutomationConfig

Provides a single BacktestConfig dataclass that captures all parameters
needed to replicate the live trading pipeline's behavior in backtesting.

Includes factory method from_live_config() to extract defaults from the
live SignalAutomationConfig, ensuring parameter parity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BacktestConfig:
    """
    Master configuration for the backtesting pipeline.

    Mirrors the live SignalAutomationConfig but removes daemon-specific
    settings (scheduling, alerts, API server) and adds backtest-specific
    settings (date range, data source).

    All defaults match the live production system.
    """

    # ── Date Range ──────────────────────────────────────────────────
    start_date: str = '2019-01-01'
    end_date: str = '2025-01-01'

    # ── Symbol & Timeframe Selection ────────────────────────────────
    symbols: List[str] = field(default_factory=lambda: ['SPY'])
    timeframes: List[str] = field(default_factory=lambda: [
        '1H', '1D', '1W', '1M',
    ])
    patterns: List[str] = field(default_factory=lambda: [
        '2-2', '3-2', '3-2-2', '2-1-2', '3-1-2',
    ])

    # ── Pattern Detection (mirrors ScanConfig) ──────────────────────
    lookback_bars: int = 50
    signal_age_bars: int = 3
    min_magnitude_pct: float = 0.5
    min_risk_reward: float = 1.0

    # ── Strike Selection (mirrors ExecutionConfig) ──────────────────
    target_delta: float = 0.55
    delta_range_min: float = 0.45
    delta_range_max: float = 0.65
    min_dte: int = 7
    max_dte: int = 21
    target_dte: int = 14

    # DTE by timeframe (used by existing backtest, maps TF to target DTE)
    dte_by_timeframe: Dict[str, int] = field(default_factory=lambda: {
        '1H': 7, '1D': 21, '1W': 35, '1M': 75,
    })

    # ── Entry Rules (mirrors ExecutionConfig + EntryMonitorConfig) ──
    # Time gates for 1H patterns ("let the market breathe")
    hourly_2bar_earliest_hour: int = 10
    hourly_2bar_earliest_minute: int = 30
    hourly_3bar_earliest_hour: int = 11
    hourly_3bar_earliest_minute: int = 30

    # TFC re-evaluation at entry
    tfc_reeval_enabled: bool = True
    tfc_reeval_min_strength: int = 3
    tfc_reeval_block_on_flip: bool = True

    # Hourly daily entry limit (-1 = unlimited)
    max_hourly_entries_per_day: int = -1

    # ── Exit Rules (mirrors MonitoringConfig) ───────────────────────
    # Target R:R by timeframe (1H forced to 1.0x per EQUITY-36)
    target_rr_by_timeframe: Dict[str, float] = field(default_factory=lambda: {
        '1H': 1.0, '1D': 1.5, '1W': 1.5, '1M': 1.5,
    })

    # Max holding bars by timeframe
    max_holding_bars: Dict[str, int] = field(default_factory=lambda: {
        '1H': 28, '1D': 18, '1W': 4, '1M': 2,
    })

    # DTE exit threshold
    exit_dte: int = 3

    # Max loss % by timeframe (EQUITY-42: wider for longer timeframes)
    max_loss_pct_by_timeframe: Dict[str, float] = field(default_factory=lambda: {
        '1H': 0.40, '1D': 0.50, '1W': 0.65, '1M': 0.75,
    })

    # EOD exit for 1H trades (EQUITY-35)
    eod_exit_hour: int = 15
    eod_exit_minute: int = 55

    # ── Trailing Stop (EQUITY-36 + EQUITY-52) ──────────────────────
    use_trailing_stop: bool = True
    trailing_stop_activation_rr: float = 0.5
    trailing_stop_pct: float = 0.50
    trailing_stop_min_profit_pct: float = 0.0

    # ATR-based trailing for 3-2 patterns (EQUITY-52)
    use_atr_trailing_for_32: bool = True
    atr_trailing_activation_multiple: float = 0.75
    atr_trailing_distance_multiple: float = 1.0

    # ── Partial Exits (EQUITY-36) ──────────────────────────────────
    partial_exit_enabled: bool = True
    partial_exit_rr: float = 1.0
    partial_exit_pct: float = 0.50

    # ── Pattern Invalidation (EQUITY-44/48) ─────────────────────────
    pattern_invalidation_enabled: bool = True

    # ── Capital Simulation (EQUITY-107) ─────────────────────────────
    capital_tracking_enabled: bool = True
    virtual_capital: float = 3000.0
    sizing_mode: str = 'fixed_dollar'
    fixed_dollar_amount: float = 300.0
    pct_of_capital: float = 0.10
    max_portfolio_heat: float = 0.08
    max_concurrent_positions: int = 5
    settlement_days: int = 1

    # ── Data Source ─────────────────────────────────────────────────
    # 'thetadata' or 'blackscholes'
    options_price_source: str = 'thetadata'

    # Use bid/ask for realistic fill modeling
    # Entry: buy at ask, Exit: sell at bid
    use_bid_ask_fill: bool = True

    # ── Output ──────────────────────────────────────────────────────
    output_dir: str = 'data/backtests'

    @classmethod
    def from_live_config(cls, live_config) -> 'BacktestConfig':
        """
        Create BacktestConfig from a live SignalAutomationConfig.

        Extracts all relevant parameters to ensure parity between
        live trading and backtesting behavior.

        Args:
            live_config: SignalAutomationConfig instance

        Returns:
            BacktestConfig with matching parameters
        """
        return cls(
            symbols=list(live_config.scan.symbols),
            timeframes=list(live_config.scan.timeframes),
            patterns=list(live_config.scan.patterns),
            lookback_bars=live_config.scan.lookback_bars,
            min_magnitude_pct=live_config.scan.min_magnitude_pct,
            min_risk_reward=live_config.scan.min_risk_reward,
            target_delta=live_config.execution.target_delta,
            delta_range_min=live_config.execution.delta_range_min,
            delta_range_max=live_config.execution.delta_range_max,
            min_dte=live_config.execution.min_dte,
            max_dte=live_config.execution.max_dte,
            target_dte=live_config.execution.target_dte,
            tfc_reeval_enabled=live_config.execution.tfc_reeval_enabled,
            tfc_reeval_min_strength=live_config.execution.tfc_reeval_min_strength,
            tfc_reeval_block_on_flip=live_config.execution.tfc_reeval_block_on_flip,
            max_hourly_entries_per_day=live_config.execution.max_hourly_entries_per_day,
            max_concurrent_positions=live_config.execution.max_concurrent_positions,
            capital_tracking_enabled=live_config.capital.enabled,
            virtual_capital=live_config.capital.virtual_capital,
            sizing_mode=live_config.capital.sizing_mode,
            fixed_dollar_amount=live_config.capital.fixed_dollar_amount,
            pct_of_capital=live_config.capital.pct_of_capital,
            max_portfolio_heat=live_config.capital.max_portfolio_heat,
            settlement_days=live_config.capital.settlement_days,
        )

    def get_max_loss_pct(self, timeframe: str) -> float:
        """Get timeframe-specific max loss percentage."""
        return self.max_loss_pct_by_timeframe.get(timeframe, 0.50)

    def get_target_rr(self, timeframe: str) -> float:
        """Get timeframe-specific target R:R."""
        return self.target_rr_by_timeframe.get(timeframe, 1.5)

    def get_max_holding(self, timeframe: str) -> int:
        """Get timeframe-specific max holding bars."""
        return self.max_holding_bars.get(timeframe, 18)

    def get_target_dte(self, timeframe: str) -> int:
        """Get timeframe-specific target DTE for strike selection."""
        return self.dte_by_timeframe.get(timeframe, self.target_dte)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        if not self.symbols:
            issues.append('No symbols configured')
        valid_timeframes = {'1H', '1D', '1W', '1M'}
        for tf in self.timeframes:
            if tf not in valid_timeframes:
                issues.append(f'Invalid timeframe: {tf}')
        if self.virtual_capital <= 0 and self.capital_tracking_enabled:
            issues.append('virtual_capital must be positive when capital tracking enabled')
        if self.fixed_dollar_amount <= 0:
            issues.append('fixed_dollar_amount must be positive')
        if self.start_date >= self.end_date:
            issues.append('start_date must be before end_date')
        return issues
