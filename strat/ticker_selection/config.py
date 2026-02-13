"""
Ticker selection pipeline configuration.

Thresholds, scoring weights, and paths. All values can be overridden
via environment variables prefixed with ``TICKER_SEL_``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os


@dataclass
class TickerSelectionConfig:
    """Configuration for the dynamic ticker selection pipeline."""

    # Master switch
    enabled: bool = False

    # Universe filters
    min_price: float = 5.0
    max_price: float = 500.0
    min_dollar_volume: float = 10_000_000.0  # $10M
    min_atr_percent: float = 1.5  # 1.5% minimum ATR as % of price

    # Pipeline caps
    max_screened: int = 500  # Max symbols after snapshot screen
    max_scan_symbols: int = 100  # Max symbols sent to STRAT scanner (~17s each)
    max_candidates: int = 12  # Max final candidates written to JSON
    max_dynamic_symbols: int = 12  # Cap on symbols daemon reads from candidates

    # Scoring weights (must sum to 1.0)
    weight_tfc: float = 0.40
    weight_pattern: float = 0.25
    weight_proximity: float = 0.20
    weight_atr: float = 0.15

    # Core ETFs always included in daemon scans
    core_symbols: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM', 'DIA',
    ])

    # Paths
    candidates_path: str = 'data/candidates/candidates.json'
    universe_cache_path: str = 'data/candidates/universe.json'

    # Staleness: daemon ignores candidates older than this (seconds)
    stale_threshold_seconds: int = 93600  # 26 hours

    # Alpaca account for data calls
    alpaca_account: str = 'SMALL'

    @classmethod
    def from_env(cls) -> 'TickerSelectionConfig':
        """Create config with environment variable overrides."""
        cfg = cls()
        cfg.enabled = os.environ.get(
            'TICKER_SELECTION_ENABLED', 'false'
        ).lower() == 'true'

        if v := os.environ.get('TICKER_SEL_MIN_PRICE'):
            cfg.min_price = float(v)
        if v := os.environ.get('TICKER_SEL_MAX_PRICE'):
            cfg.max_price = float(v)
        if v := os.environ.get('TICKER_SEL_MIN_DOLLAR_VOLUME'):
            cfg.min_dollar_volume = float(v)
        if v := os.environ.get('TICKER_SEL_MIN_ATR_PCT'):
            cfg.min_atr_percent = float(v)
        if v := os.environ.get('TICKER_SEL_MAX_SCAN'):
            cfg.max_scan_symbols = int(v)
        if v := os.environ.get('TICKER_SEL_MAX_CANDIDATES'):
            cfg.max_candidates = int(v)
        if v := os.environ.get('TICKER_SEL_ALPACA_ACCOUNT'):
            cfg.alpaca_account = v

        return cfg
