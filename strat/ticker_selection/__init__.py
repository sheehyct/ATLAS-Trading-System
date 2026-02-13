"""
ATLAS Dynamic Ticker Selection Pipeline.

Replaces the static 17-symbol list with dynamic universe discovery:
  Universe (~4,000) -> Screener (~300-500) -> STRAT Scan (~30-60)
  -> TFC Filter (~10-20) -> Scored Candidates (8-15)
  -> data/candidates/candidates.json -> Daemon reads automatically
"""

from strat.ticker_selection.config import TickerSelectionConfig
from strat.ticker_selection.pipeline import TickerSelectionPipeline, run_selection

__all__ = [
    'TickerSelectionConfig',
    'TickerSelectionPipeline',
    'run_selection',
]
