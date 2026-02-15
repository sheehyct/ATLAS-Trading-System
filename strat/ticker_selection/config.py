"""
Ticker selection pipeline configuration.

Thresholds, scoring weights, and paths. All values can be overridden
via environment variables prefixed with ``TICKER_SEL_``.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


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
    max_workers: int = 6  # Parallel workers for STRAT scanning (I/O bound)

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

    # Finviz enrichment (informational only, no scoring impact)
    finviz_enrichment_enabled: bool = True
    finviz_cache_ttl_hours: int = 6
    finviz_max_workers: int = 4

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
        if v := os.environ.get('TICKER_SEL_MAX_WORKERS'):
            cfg.max_workers = int(v)
        if v := os.environ.get('TICKER_SEL_ALPACA_ACCOUNT'):
            cfg.alpaca_account = v
        if v := os.environ.get('TICKER_SEL_FINVIZ_ENABLED'):
            cfg.finviz_enrichment_enabled = v.lower() in ('true', '1')
        if v := os.environ.get('TICKER_SEL_FINVIZ_CACHE_TTL'):
            cfg.finviz_cache_ttl_hours = int(v)

        return cfg
