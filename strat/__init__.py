"""
STRAT Layer 2 - Bar Classification and Pattern Detection

This package implements Rob Smith's STRAT methodology for:
- Bar classification (1, 2U, 2D, 3) using previous bar comparison
- Pattern detection (3-1-2, 2-1-2, 2-2) with measured move targets
- Integration with ATLAS Layer 1 regime detection
- Paper trading infrastructure (Session 83K-41)
"""

from strat.bar_classifier import (
    StratBarClassifier,
    classify_bars,
    format_bar_classifications,
    get_bar_sequence_string
)

from strat.paper_trading import (
    PaperTrade,
    PaperTradeLog,
    create_paper_trade,
)

from strat.paper_signal_scanner import (
    PaperSignalScanner,
    DetectedSignal,
    scan_for_signals,
    get_actionable_signals,
)

__all__ = [
    # Bar classifier
    'StratBarClassifier',
    'classify_bars',
    'format_bar_classifications',
    'get_bar_sequence_string',
    # Paper trading
    'PaperTrade',
    'PaperTradeLog',
    'create_paper_trade',
    'PaperSignalScanner',
    'DetectedSignal',
    'scan_for_signals',
    'get_actionable_signals',
]
