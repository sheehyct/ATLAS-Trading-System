"""
Multi-timeframe inside bar convergence detection.

Detects when multiple higher timeframes (Monthly, Weekly, Daily) have
inside bars (Type 1) simultaneously — a "coiled spring" setup where a
single intraday break can cascade through all timeframes at once,
flipping TFC from 1/4 to 4/4 in one bar.

This module is pure data extraction: it takes symbols, fetches bar data,
classifies bars, and returns convergence metadata.  It does not modify
the pipeline or scoring — those integrations happen in Phase 2.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strat.bar_classifier import classify_bars_nb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InsideBarInfo:
    """Per-timeframe inside bar state."""
    timeframe: str           # '1M', '1W', '1D'
    is_inside: bool          # True if last closed bar is Type 1
    prior_bar_high: float    # Break above = 2U resolution
    prior_bar_low: float     # Break below = 2D resolution
    prior_bar_direction: str  # 'U', 'D', or 'N' (bar before the inside bar)


@dataclass
class ConvergenceMetadata:
    """Convergence analysis result for a single symbol."""
    inside_bar_count: int                          # Number of TFs with Type 1
    inside_bar_timeframes: List[str]               # Which TFs (e.g. ['1M', '1W'])
    trigger_levels: Dict[str, Dict[str, float]]    # {tf: {'high': x, 'low': y}}
    convergence_score: float                       # 0-100 composite
    bullish_trigger: Optional[float]               # Highest high across inside TFs
    bearish_trigger: Optional[float]               # Lowest low across inside TFs
    trigger_spread_pct: float                      # |bull - bear| / price * 100
    prior_direction_alignment: str                 # 'bullish', 'bearish', 'mixed'
    is_convergence: bool                           # True if inside_bar_count >= 2
    tier: int = 2                                  # Set later by classifier
    bar_states: List[InsideBarInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for JSON output in candidates.json."""
        return {
            'inside_bar_count': self.inside_bar_count,
            'inside_bar_timeframes': self.inside_bar_timeframes,
            'trigger_levels': self.trigger_levels,
            'convergence_score': round(self.convergence_score, 1),
            'bullish_trigger': self.bullish_trigger,
            'bearish_trigger': self.bearish_trigger,
            'trigger_spread_pct': round(self.trigger_spread_pct, 2),
            'prior_direction_alignment': self.prior_direction_alignment,
            'is_convergence': self.is_convergence,
        }


# ---------------------------------------------------------------------------
# Timeframe-to-Alpaca mapping
# ---------------------------------------------------------------------------

_TF_ALPACA_MAP = {
    '1M': {'amount': 1, 'unit': 'Month'},
    '1W': {'amount': 1, 'unit': 'Week'},
    '1D': {'amount': 1, 'unit': 'Day'},
}


# ---------------------------------------------------------------------------
# Convergence Analyzer
# ---------------------------------------------------------------------------

class ConvergenceAnalyzer:
    """
    Detects multi-TF inside bar convergence and calculates trigger levels.

    Analyzes Monthly, Weekly, and Daily bars for each symbol to find
    "coiled spring" setups where multiple timeframes have inside bars.

    Parameters
    ----------
    alpaca_account : str
        Alpaca account key for credentials (default 'SMALL').
    """

    ANALYSIS_TIMEFRAMES = ['1M', '1W', '1D']

    def __init__(self, alpaca_account: str = 'SMALL'):
        self._alpaca_account = alpaca_account
        self._client = None

    def _get_client(self):
        """Lazy-init the Alpaca data client."""
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            from config.settings import get_alpaca_credentials

            creds = get_alpaca_credentials(self._alpaca_account)
            self._client = StockHistoricalDataClient(
                api_key=creds['api_key'],
                secret_key=creds['secret_key'],
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_symbol(self, symbol: str, current_price: float = 0.0) -> ConvergenceMetadata:
        """
        Analyze a single symbol for multi-TF inside bar convergence.

        Parameters
        ----------
        symbol : str
            Stock ticker.
        current_price : float
            Current price for trigger spread calculation.
            If 0, uses the daily close.

        Returns
        -------
        ConvergenceMetadata
        """
        bar_states: List[InsideBarInfo] = []

        for tf in self.ANALYSIS_TIMEFRAMES:
            try:
                state = self._get_bar_state(symbol, tf)
                bar_states.append(state)
            except Exception as e:
                logger.warning(f"Convergence: {symbol} {tf} bar fetch failed: {e}")
                bar_states.append(InsideBarInfo(
                    timeframe=tf, is_inside=False,
                    prior_bar_high=0.0, prior_bar_low=0.0,
                    prior_bar_direction='N',
                ))

        return self._build_metadata(bar_states, current_price)

    def analyze_batch(
        self, symbols: List[str], prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, ConvergenceMetadata]:
        """
        Analyze a batch of symbols for convergence.

        Parameters
        ----------
        symbols : list[str]
            Symbols to analyze.
        prices : dict[str, float] | None
            Optional current prices keyed by symbol.

        Returns
        -------
        dict[str, ConvergenceMetadata]
        """
        prices = prices or {}
        results: Dict[str, ConvergenceMetadata] = {}

        for sym in symbols:
            try:
                results[sym] = self.analyze_symbol(sym, prices.get(sym, 0.0))
            except Exception as e:
                logger.error(f"Convergence: {sym} analysis failed: {e}")

        return results

    # ------------------------------------------------------------------
    # Bar state extraction
    # ------------------------------------------------------------------

    def _get_bar_state(self, symbol: str, timeframe: str) -> InsideBarInfo:
        """
        Fetch recent bars for one TF and classify the last bar.

        We need at least 3 bars:
          - bar[-3]: to classify bar[-2] (need its direction)
          - bar[-2]: the "prior" bar whose H/L set trigger levels
          - bar[-1]: the last closed bar (is it inside?)
        """
        bars_df = self._fetch_bars(symbol, timeframe, limit=5)

        if bars_df is None or len(bars_df) < 3:
            return InsideBarInfo(
                timeframe=timeframe, is_inside=False,
                prior_bar_high=0.0, prior_bar_low=0.0,
                prior_bar_direction='N',
            )

        highs = bars_df['high'].values.astype(np.float64)
        lows = bars_df['low'].values.astype(np.float64)

        classifications = classify_bars_nb(highs, lows)

        last_class = int(classifications[-1])
        is_inside = (last_class == 1)

        # Prior bar's high/low define trigger levels
        prior_high = float(highs[-2])
        prior_low = float(lows[-2])

        # Direction of the bar before the inside bar (bar[-2]'s classification)
        prior_class = int(classifications[-2])
        if prior_class == 2:
            prior_dir = 'U'
        elif prior_class == -2:
            prior_dir = 'D'
        else:
            prior_dir = 'N'

        return InsideBarInfo(
            timeframe=timeframe,
            is_inside=is_inside,
            prior_bar_high=prior_high,
            prior_bar_low=prior_low,
            prior_bar_direction=prior_dir,
        )

    def _fetch_bars(self, symbol: str, timeframe: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC bars from Alpaca for a given timeframe.

        Returns a DataFrame with columns: open, high, low, close, volume.
        """
        from datetime import datetime, timedelta
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_config = _TF_ALPACA_MAP[timeframe]
        unit = getattr(TimeFrameUnit, tf_config['unit'])
        alpaca_tf = TimeFrame(tf_config['amount'], unit)

        # Lookback enough to get `limit` bars
        lookback_days = {
            '1M': limit * 35,
            '1W': limit * 10,
            '1D': limit * 3,
        }[timeframe]

        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        client = self._get_client()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_tf,
            start=start,
            end=end,
            limit=limit,
        )

        try:
            bars = client.get_stock_bars(request)
            df = bars.df
            if df.empty:
                return None
            # Multi-index: (symbol, timestamp) -> flatten
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel('symbol')
            return df
        except Exception as e:
            logger.warning(f"Alpaca bars fetch failed {symbol}/{timeframe}: {e}")
            return None

    # ------------------------------------------------------------------
    # Metadata construction & scoring
    # ------------------------------------------------------------------

    def _build_metadata(
        self, bar_states: List[InsideBarInfo], current_price: float,
    ) -> ConvergenceMetadata:
        """Build ConvergenceMetadata from per-TF bar states."""
        inside_states = [s for s in bar_states if s.is_inside]
        inside_count = len(inside_states)
        inside_tfs = [s.timeframe for s in inside_states]

        # Trigger levels per inside TF
        trigger_levels: Dict[str, Dict[str, float]] = {}
        for s in inside_states:
            trigger_levels[s.timeframe] = {
                'high': round(s.prior_bar_high, 2),
                'low': round(s.prior_bar_low, 2),
            }

        # Aggregate triggers: highest high, lowest low across inside TFs
        bullish_trigger: Optional[float] = None
        bearish_trigger: Optional[float] = None
        if inside_states:
            bullish_trigger = round(max(s.prior_bar_high for s in inside_states), 2)
            bearish_trigger = round(min(s.prior_bar_low for s in inside_states), 2)

        # Trigger spread as % of price
        trigger_spread_pct = 0.0
        if bullish_trigger is not None and bearish_trigger is not None and current_price > 0:
            trigger_spread_pct = abs(bullish_trigger - bearish_trigger) / current_price * 100

        # Direction alignment of prior bars
        prior_direction_alignment = self._assess_direction_alignment(bar_states)

        # Convergence score
        convergence_score = self._calculate_score(
            inside_count, trigger_spread_pct, prior_direction_alignment,
        )

        return ConvergenceMetadata(
            inside_bar_count=inside_count,
            inside_bar_timeframes=inside_tfs,
            trigger_levels=trigger_levels,
            convergence_score=round(convergence_score, 1),
            bullish_trigger=bullish_trigger,
            bearish_trigger=bearish_trigger,
            trigger_spread_pct=round(trigger_spread_pct, 2),
            prior_direction_alignment=prior_direction_alignment,
            is_convergence=(inside_count >= 2),
            bar_states=bar_states,
        )

    @staticmethod
    def _assess_direction_alignment(bar_states: List[InsideBarInfo]) -> str:
        """
        Assess whether prior bar directions are aligned.

        Only considers timeframes with inside bars, since those are
        the ones whose prior bar direction matters for cascade direction.
        """
        inside_states = [s for s in bar_states if s.is_inside]
        if not inside_states:
            return 'mixed'

        directions = [s.prior_bar_direction for s in inside_states]
        # Filter out 'N' (neutral/outside/reference) for alignment assessment
        directional = [d for d in directions if d in ('U', 'D')]

        if not directional:
            return 'mixed'
        if all(d == 'U' for d in directional):
            return 'bullish'
        if all(d == 'D' for d in directional):
            return 'bearish'
        return 'mixed'

    @staticmethod
    def _calculate_score(
        inside_count: int,
        trigger_spread_pct: float,
        direction_alignment: str,
    ) -> float:
        """
        Calculate convergence score (0-100).

        Formula:
            score = inside_count_score * 0.50
                  + trigger_proximity_score * 0.30
                  + direction_alignment_score * 0.20
        """
        # Short-circuit: no inside bars = no convergence signal
        if inside_count == 0:
            return 0.0

        # Component 1: Inside bar count (50% weight)
        count_scores = {1: 20, 2: 70, 3: 100}
        inside_count_score = count_scores.get(min(inside_count, 3), 0)

        # Component 2: Trigger proximity (30% weight)
        # Tighter spread = triggers are closer = more explosive
        if trigger_spread_pct <= 0:
            # No price data available — use neutral score, not penalty
            proximity_score = 50.0
        elif trigger_spread_pct < 1.0:
            proximity_score = 100.0
        elif trigger_spread_pct < 2.0:
            proximity_score = 80.0
        elif trigger_spread_pct < 4.0:
            proximity_score = 50.0
        else:
            proximity_score = 20.0

        # Component 3: Direction alignment (20% weight)
        alignment_scores = {
            'bullish': 100.0,
            'bearish': 100.0,
            'mixed': 30.0,
        }
        alignment_score = alignment_scores.get(direction_alignment, 30.0)

        # Weighted composite
        score = (
            inside_count_score * 0.50
            + proximity_score * 0.30
            + alignment_score * 0.20
        )
        return score
