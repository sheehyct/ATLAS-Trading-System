"""
STRAT Paper Trading Signal Scanner - Session 83K-41

Scans for STRAT patterns across all timeframes and symbols for paper trading.
Designed to detect ALL patterns per strategic pivot decision:
- Include ALL patterns: 2-2, 3-2, 3-2-2, 2-1-2, 3-1-2
- Include ALL timeframes: 1H, 1D, 1W, 1M
- Capture market context (VIX, ATR, regime) at detection time

Usage:
    from strat.paper_signal_scanner import PaperSignalScanner

    scanner = PaperSignalScanner()
    signals = scanner.scan_all_timeframes('SPY')
    scanner.print_signals(signals)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

from strat.bar_classifier import classify_bars_nb
from strat.pattern_detector import (
    # Completed pattern detectors (entry already happened)
    detect_312_patterns_nb,
    detect_212_patterns_nb,
    detect_22_patterns_nb,
    detect_32_patterns_nb,
    detect_322_patterns_nb,
    # Session 83K-71: Setup detectors (waiting for live break)
    detect_312_setups_nb,
    detect_212_setups_nb,
    detect_22_setups_nb,
    detect_322_setups_nb,
    detect_outside_bar_setups_nb,
)
from strat.paper_trading import (
    PaperTrade,
    create_paper_trade,
)
# Session 83K-52: Import from single source of truth
from strat.tier1_detector import PatternType, Timeframe
# Session EQUITY-41: Import pattern registry for bidirectional flag
from strat.pattern_registry import is_bidirectional_pattern
from integrations.tiingo_data_fetcher import TiingoDataFetcher
from strat.timeframe_continuity_adapter import TimeframeContinuityAdapter


@dataclass
class SignalContext:
    """Market context at signal detection time."""
    vix: float = 0.0
    atr_14: float = 0.0
    atr_percent: float = 0.0
    volume_20d_avg: float = 0.0
    current_volume: float = 0.0
    volume_ratio: float = 0.0
    market_regime: str = ''
    tfc_score: int = 0
    tfc_alignment: str = ''
    tfc_passes: bool = False
    risk_multiplier: float = 1.0
    priority_rank: int = 0


@dataclass
class DetectedSignal:
    """A detected STRAT pattern signal ready for paper trading."""
    pattern_type: str
    direction: str          # 'CALL' or 'PUT'
    symbol: str
    timeframe: str
    detected_time: datetime
    entry_trigger: float
    stop_price: float
    target_price: float
    magnitude_pct: float
    risk_reward: float
    context: SignalContext

    # Session 83K-68: Setup-based detection fields
    signal_type: str = 'COMPLETED'     # 'SETUP' or 'COMPLETED'
    setup_bar_high: float = 0.0        # Level to monitor for bullish break
    setup_bar_low: float = 0.0         # Level to monitor for bearish break
    setup_bar_timestamp: Optional[datetime] = None  # When setup bar closed

    # Session EQUITY-41: Pattern bidirectionality (from pattern_registry)
    # BIDIRECTIONAL (3-?, 3-1-?, X-1-?): Break determines direction
    # UNIDIRECTIONAL (3-2D-?, 3-2U-?, X-2D-?, X-2U-?): Only reversal direction triggers
    is_bidirectional: bool = True  # Default True for safety (check both directions)

    def to_paper_trade(self) -> PaperTrade:
        """Convert signal to PaperTrade for tracking."""
        return create_paper_trade(
            pattern_type=self.pattern_type,
            timeframe=self.timeframe,
            symbol=self.symbol,
            direction=self.direction,
            pattern_detected_time=self.detected_time,
            entry_trigger=self.entry_trigger,
            target_price=self.target_price,
            stop_price=self.stop_price,
            vix=self.context.vix,
            atr=self.context.atr_14,
            market_regime=self.context.market_regime,
        )


class PaperSignalScanner:
    """
    Scans for STRAT patterns across all timeframes for paper trading.

    Designed per Session 83K-40 strategic pivot:
    - Include ALL patterns to validate/invalidate backtest assumptions
    - Capture comprehensive market context
    - Output signals ready for paper trade entry
    """

    # Symbols to scan (NVDA excluded due to data issues)
    DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL']

    # All timeframes to scan
    # Session EQUITY-62: Added 4H for proper TFC evaluation (5 timeframes)
    DEFAULT_TIMEFRAMES = ['1H', '4H', '1D', '1W', '1M']

    # All pattern types to detect
    ALL_PATTERNS = ['2-2', '3-2', '3-2-2', '2-1-2', '3-1-2']

    def __init__(self, use_alpaca: bool = True, use_thetadata: bool = False):
        """
        Initialize scanner.

        Args:
            use_alpaca: Use Alpaca for underlying data (fallback to Tiingo if fails)
            use_thetadata: Use ThetaData for options context
        """
        self.use_alpaca = use_alpaca
        self.use_thetadata = use_thetadata
        self._vbt = None
        self._thetadata = None
        self._tiingo = None
        # Session EQUITY-62: TFC adapter needs timeframes in descending order for proper continuity checking
        self._tfc_adapter = TimeframeContinuityAdapter(timeframes=['1M', '1W', '1D', '4H', '1H'])

    def _get_tiingo(self):
        """Lazy load Tiingo fetcher."""
        if self._tiingo is None:
            try:
                self._tiingo = TiingoDataFetcher()
            except Exception as e:
                warnings.warn(f"Failed to initialize Tiingo: {e}")
                self._tiingo = False  # Mark as unavailable
        return self._tiingo if self._tiingo else None

    def _get_vbt(self):
        """Lazy load VectorBT Pro and configure Alpaca credentials."""
        if self._vbt is None:
            import os
            import vectorbtpro as vbt
            from dotenv import load_dotenv

            # Load .env file to get Alpaca credentials
            env_path = Path(__file__).parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)

            # Configure Alpaca credentials for VBT Pro
            api_key = os.environ.get('ALPACA_API_KEY', '')
            secret_key = os.environ.get('ALPACA_SECRET_KEY', '')

            if api_key and secret_key:
                vbt.AlpacaData.set_custom_settings(
                    client_config=dict(
                        api_key=api_key,
                        secret_key=secret_key,
                        paper=True  # Use paper trading endpoint
                    )
                )

            self._vbt = vbt
        return self._vbt

    def _fetch_data(self, symbol: str, timeframe: str,
                    lookback_bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for pattern detection.

        Primary: Alpaca (all timeframes)
        Fallback: Tiingo (daily/weekly/monthly only, no hourly/intraday)

        Args:
            symbol: Stock symbol
            timeframe: '15m', '30m', '1H', '1D', '1W', '1M'
            lookback_bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV columns
        """
        vbt = self._get_vbt()

        # Calculate start date based on timeframe
        # Session EQUITY-18: Add 15m and 30m support for faster timeframe scanning
        if timeframe == '15m':
            days = lookback_bars // 26 + 30  # ~26 15-min bars per day, add buffer
            tf_alpaca = '15Min'
            tf_tiingo = None  # Tiingo doesn't support intraday
        elif timeframe == '30m':
            days = lookback_bars // 13 + 30  # ~13 30-min bars per day, add buffer
            tf_alpaca = '30Min'
            tf_tiingo = None  # Tiingo doesn't support intraday
        elif timeframe == '1H':
            days = lookback_bars // 7 + 30  # ~7 bars per day, add buffer
            tf_alpaca = '1Hour'
            tf_tiingo = None  # Tiingo doesn't support hourly
        elif timeframe == '1D':
            days = lookback_bars + 30
            tf_alpaca = '1Day'
            tf_tiingo = '1D'
        elif timeframe == '1W':
            days = lookback_bars * 7 + 30
            tf_alpaca = '1Week'
            tf_tiingo = '1W'
        elif timeframe == '1M':
            days = lookback_bars * 30 + 30
            tf_alpaca = '1Month'
            tf_tiingo = '1M'
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        end = datetime.now()
        start = end - timedelta(days=days)

        # Try Alpaca first
        try:
            # Session 83K-72: For intraday timeframes, add 1 day to end date
            # to include today's bars (Alpaca end date is exclusive)
            # Session EQUITY-18: Include 15m and 30m as intraday timeframes
            is_intraday = timeframe in ('15m', '30m', '1H')
            end_date = end + timedelta(days=1) if is_intraday else end

            data = vbt.AlpacaData.pull(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                timeframe=tf_alpaca,
                tz='America/New_York'
            )
            df = data.get()

            # Session EQUITY-18: Filter to market hours for all intraday timeframes
            # This ensures consistent behavior for 15m, 30m, and 1H
            if is_intraday:
                df = self._align_hourly_bars(df)

            return df

        except Exception as e:
            # Alpaca failed, try Tiingo fallback (daily/weekly/monthly only)
            if tf_tiingo is None:
                warnings.warn(f"Failed to fetch {symbol} {timeframe} (no Tiingo fallback for intraday): {e}")
                return None

            tiingo = self._get_tiingo()
            if tiingo is None:
                warnings.warn(f"Failed to fetch {symbol} {timeframe} (Tiingo unavailable): {e}")
                return None

            try:
                tiingo_data = tiingo.fetch(
                    symbol,
                    start_date=start.strftime('%Y-%m-%d'),
                    end_date=end.strftime('%Y-%m-%d'),
                    timeframe=tf_tiingo,
                    use_cache=True
                )
                df = tiingo_data.get()
                return df

            except Exception as tiingo_err:
                warnings.warn(f"Failed to fetch {symbol} {timeframe} from both Alpaca and Tiingo: {tiingo_err}")
                return None

    def _align_hourly_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align hourly bars to market open (09:30, 10:30, 11:30, etc.).

        Session 83K-34: CRITICAL for correct pattern detection.
        Alpaca returns clock-aligned bars (10:00, 11:00) but STRAT
        requires market-open-aligned bars.
        """
        if df.empty:
            return df

        # Filter to market hours (09:30-16:00 ET)
        df = df.between_time('09:30', '16:00')

        return df

    # =========================================================================
    # Session 83K-80: 15-Minute Base Resampling Methods
    # =========================================================================

    def _fetch_15min_data(self, symbol: str, lookback_bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch 15-minute data from Alpaca.

        Session 83K-80: Base timeframe for multi-TF resampling.
        15-min bars are market-aligned (9:30, 9:45, 10:00, etc.)

        Args:
            symbol: Stock symbol
            lookback_bars: Number of 15-min bars to fetch

        Returns:
            DataFrame with OHLCV columns, filtered to market hours
        """
        vbt = self._get_vbt()

        # Calculate days needed (26 bars per day * lookback)
        days = lookback_bars // 26 + 30  # ~26 15-min bars per day, add buffer

        end = datetime.now()
        start = end - timedelta(days=days)

        try:
            data = vbt.AlpacaData.pull(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),  # Include today
                timeframe='15Min',
                tz='America/New_York'
            )
            df = data.get()

            # Filter to market hours (09:30-16:00 ET)
            df = df.between_time('09:30', '16:00')

            return df

        except Exception as e:
            warnings.warn(f"Failed to fetch 15-min data for {symbol}: {e}")
            return None

    def _resample_to_htf(self, base_df: pd.DataFrame, target_tf: str) -> Optional[pd.DataFrame]:
        """
        Resample 15-min OHLCV data to higher timeframe.

        Session 83K-80: Core resampling method for HTF scanning architecture fix.

        Args:
            base_df: DataFrame with 15-min OHLC data (market-aligned, 9:30 start)
            target_tf: Target timeframe ('1H', '1D', '1W', '1M')

        Returns:
            DataFrame with resampled OHLC data (last bar is "running"/incomplete)

        CRITICAL: Bars are market-aligned:
        - 15min: 9:30-9:45, 9:45-10:00, ...
        - 1H: 9:30-10:30, 10:30-11:30, ... (aligned to market open)
        - 1D: 9:30 to 16:00
        - 1W: Monday-Friday
        - 1M: Calendar month
        """
        if base_df is None or base_df.empty:
            return None

        # Map target TF to pandas frequency with proper offset for market alignment
        # '1h' with offset='30min' creates bars at 9:30, 10:30, 11:30, etc.
        freq_map = {
            '1H': ('1h', '30min'),   # Hourly, offset by 30 min for market alignment
            '1D': ('1D', None),       # Daily
            '1W': ('W-FRI', None),    # Weekly (Friday close)
            '1M': ('ME', None),       # Month end
        }

        if target_tf not in freq_map:
            raise ValueError(f"Unknown target timeframe: {target_tf}")

        freq, offset = freq_map[target_tf]

        try:
            # Use pandas resampling with offset for market alignment
            if offset:
                resampled = base_df.resample(freq, offset=offset).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            else:
                resampled = base_df.resample(freq).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

            return resampled

        except Exception as e:
            warnings.warn(f"Failed to resample to {target_tf}: {e}")
            return None

    def _fetch_vix(self) -> float:
        """Fetch current VIX level."""
        try:
            import yfinance as yf
            vix = yf.download('^VIX', period='1d', progress=False)
            if not vix.empty:
                # Handle both flat columns and MultiIndex columns from yfinance
                if 'Close' in vix.columns:
                    close_val = vix['Close'].iloc[-1]
                elif ('Close', '^VIX') in vix.columns:
                    close_val = vix[('Close', '^VIX')].iloc[-1]
                else:
                    # Fallback: get first column that contains 'Close'
                    close_cols = [c for c in vix.columns if 'Close' in str(c)]
                    if close_cols:
                        close_val = vix[close_cols[0]].iloc[-1]
                    else:
                        return 0.0
                # Ensure we return a scalar float
                if hasattr(close_val, 'item'):
                    return float(close_val.item())
                return float(close_val)
        except Exception as e:
            warnings.warn(f"Failed to fetch VIX: {e}")
        return 0.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from OHLC data."""
        if len(df) < period:
            return 0.0

        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        tr1 = high - low
        tr2 = abs(high - np.roll(close, 1))
        tr3 = abs(low - np.roll(close, 1))

        tr = np.maximum(np.maximum(tr1, tr2), tr3)[1:]  # Skip first (NaN)

        if len(tr) < period:
            return 0.0

        atr = np.mean(tr[-period:])
        return float(atr)

    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volume vs 20-day average."""
        if len(df) < 20 or 'Volume' not in df.columns:
            return 1.0

        avg_volume = df['Volume'].iloc[-21:-1].mean()
        current_volume = df['Volume'].iloc[-1]

        if avg_volume > 0:
            return current_volume / avg_volume
        return 1.0

    def _get_market_context(self, df: pd.DataFrame) -> SignalContext:
        """Get market context for signal."""
        vix = self._fetch_vix()
        atr = self._calculate_atr(df)
        current_price = df['Close'].iloc[-1] if not df.empty else 0
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0
        volume_ratio = self._calculate_volume_ratio(df)

        return SignalContext(
            vix=vix,
            atr_14=atr,
            atr_percent=atr_pct,
            volume_ratio=volume_ratio,
        )

    def evaluate_tfc(
        self, symbol: str, detection_timeframe: str, direction: int
    ):
        """Run timeframe continuity adapter and return assessment."""

        direction_label = "bullish" if direction == 1 else "bearish"

        def _fetch(tf: str):
            return self._fetch_data(symbol, tf, lookback_bars=50)

        return self._tfc_adapter.evaluate(
            fetcher=_fetch,
            detection_timeframe=detection_timeframe,
            direction=direction_label,
        )

    def _get_full_bar_sequence(self, pattern_type: str, classifications: np.ndarray,
                                idx: int, direction: int) -> str:
        """
        Get full bar sequence string for a detected pattern.

        Session 83K-44: Every directional bar must be classified as 2U or 2D.
        This method converts numeric classifications to proper STRAT notation.

        Args:
            pattern_type: Base pattern type ('2-2', '3-2', '3-2-2', '2-1-2', '3-1-2')
            classifications: Array of bar classifications (1, 2, -2, 3)
            idx: Index of the trigger bar
            direction: 1 for bullish, -1 for bearish

        Returns:
            Full bar sequence string (e.g., '2U-1-2U', '3-2D-2U', '2D-2U')

        Bar Classification Key:
            1  = Inside bar (1)
            2  = 2U (bullish directional - higher high)
            -2 = 2D (bearish directional - lower low)
            3  = Outside bar (3)
        """
        def bar_to_str(bar_class: int) -> str:
            """Convert numeric classification to string."""
            if bar_class == 1:
                return "1"
            elif bar_class == 2:
                return "2U"
            elif bar_class == -2:
                return "2D"
            elif abs(bar_class) == 3:
                return "3"
            else:
                return "?"

        # Get bar classifications for pattern
        if idx < 2:
            # Not enough bars for 3-bar pattern, return simple format
            return f"{pattern_type}{'U' if direction > 0 else 'D'}"

        if pattern_type == '2-2':
            # 2-2: bar at i-1 (first 2) + bar at i (trigger 2)
            bar1 = int(classifications[idx - 1])
            bar2 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}"

        elif pattern_type == '3-2':
            # 3-2: outside bar at i-1 + directional bar at i
            bar1 = int(classifications[idx - 1])
            bar2 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}"

        elif pattern_type == '3-2-2':
            # 3-2-2: outside bar at i-2 + directional at i-1 + directional at i
            bar1 = int(classifications[idx - 2])
            bar2 = int(classifications[idx - 1])
            bar3 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}-{bar_to_str(bar3)}"

        elif pattern_type == '2-1-2':
            # 2-1-2: directional at i-2 + inside at i-1 + directional at i
            bar1 = int(classifications[idx - 2])
            bar2 = int(classifications[idx - 1])
            bar3 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}-{bar_to_str(bar3)}"

        elif pattern_type == '3-1-2':
            # 3-1-2: outside at i-2 + inside at i-1 + directional at i
            bar1 = int(classifications[idx - 2])
            bar2 = int(classifications[idx - 1])
            bar3 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}-{bar_to_str(bar3)}"

        else:
            # Unknown pattern type, return simple format
            return f"{pattern_type}{'U' if direction > 0 else 'D'}"

    def _detect_patterns(self, df: pd.DataFrame,
                         pattern_type: str) -> List[Dict]:
        """
        Detect specific pattern type in data.

        Args:
            df: OHLCV DataFrame
            pattern_type: '2-2', '3-2', '3-2-2', '2-1-2', '3-1-2'

        Returns:
            List of detected patterns with entry/stop/target prices
        """
        if df is None or len(df) < 5:
            return []

        # Get classifications
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        classifications = classify_bars_nb(high, low)

        patterns = []

        # Pattern detectors return: (entries_mask, stops, targets, directions)
        # - entries_mask: boolean array (True at pattern trigger bar)
        # - stops: stop loss prices
        # - targets: target prices (structural levels)
        # - directions: 1 for bullish (CALL), -1 for bearish (PUT), 0 for no pattern
        if pattern_type == '2-2':
            result = detect_22_patterns_nb(classifications, high, low)
            entries_mask, stops, targets, directions = result[:4]
        elif pattern_type == '3-2':
            result = detect_32_patterns_nb(classifications, high, low)
            entries_mask, stops, targets, directions = result[:4]
        elif pattern_type == '3-2-2':
            result = detect_322_patterns_nb(classifications, high, low)
            entries_mask, stops, targets, directions = result[:4]
        elif pattern_type == '2-1-2':
            result = detect_212_patterns_nb(classifications, high, low)
            entries_mask, stops, targets, directions = result[:4]
        elif pattern_type == '3-1-2':
            result = detect_312_patterns_nb(classifications, high, low)
            entries_mask, stops, targets, directions = result[:4]
        else:
            return []

        # Extract pattern occurrences
        for i in range(len(entries_mask)):
            if entries_mask[i]:  # Pattern detected at this bar
                direction = 'CALL' if directions[i] > 0 else 'PUT'

                # Session 83K-68: CRITICAL FIX - Entry trigger uses SETUP bar (i-1), not trigger bar (i)
                # Per STRAT methodology: Entry happens LIVE when bar breaks setup bar's high/low
                # - 3-1-2, 2-1-2: i-1 is inside bar (entry = inside bar high/low)
                # - 2-2, 3-2-2: i-1 is first directional bar (entry = bar high/low)
                # The trigger bar (i) has ALREADY closed above/below this level by definition
                entry = high[i-1] if directions[i] > 0 else low[i-1]

                # Store setup bar levels for monitoring (Session 83K-68)
                setup_bar_high = high[i-1]
                setup_bar_low = low[i-1]

                stop = stops[i]
                target = targets[i]

                # Session 83K-44: Extract full bar sequence for pattern classification
                # Get bar classifications for the pattern bars
                bar_sequence = self._get_full_bar_sequence(
                    pattern_type, classifications, i, directions[i]
                )

                # BUGFIX Session 83K-44: Validate target geometry against DISPLAYED entry
                # Detector validates against different entry (e.g., high[i-1] for 2-2 patterns)
                # Must re-validate and recalculate if target geometry is invalid
                if not np.isnan(target) and not np.isnan(stop) and stop > 0:
                    if directions[i] > 0:  # Bullish - target must be ABOVE entry
                        if target <= entry:
                            # Target geometrically invalid - use measured move (1.5 R:R)
                            risk = entry - stop
                            if risk > 0:
                                target = entry + (risk * 1.5)
                    elif directions[i] < 0:  # Bearish - target must be BELOW entry
                        if target >= entry:
                            # Target geometrically invalid - use measured move (1.5 R:R)
                            risk = stop - entry
                            if risk > 0:
                                target = entry - (risk * 1.5)

                # Calculate magnitude and R:R
                # Skip if target is NaN (no valid target set)
                if entry > 0 and not np.isnan(target) and target > 0:
                    magnitude = abs(target - entry) / entry * 100
                    risk = abs(entry - stop) if stop > 0 and not np.isnan(stop) else 0
                    reward = abs(target - entry)
                    rr = reward / risk if risk > 0 else 0

                    patterns.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'signal': directions[i],
                        'direction': direction,
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'magnitude_pct': magnitude,
                        'risk_reward': rr,
                        'bar_sequence': bar_sequence,  # Session 83K-44: Full bar sequence
                        # Session 83K-68: Setup-based detection fields
                        'signal_type': 'COMPLETED',  # This is a completed pattern
                        'setup_bar_high': setup_bar_high,
                        'setup_bar_low': setup_bar_low,
                        'setup_bar_timestamp': df.index[i-1] if i > 0 else df.index[i],
                    })

        return patterns

    def _detect_setups(self, df: pd.DataFrame) -> List[Dict]:
        """
        Session 83K-71: Detect SETUP patterns (ending in inside bar or directional bar)
        that are waiting for a live break to become completed patterns.

        Per STRAT methodology:
        - X-1 setups (3-1, 2-1): Inside bar waiting for break to become X-1-2
        - Entry is LIVE when price breaks inside bar high (bull) or low (bear)

        Args:
            df: OHLCV DataFrame

        Returns:
            List of detected setups with entry triggers and signal_type='SETUP'
        """
        if df is None or len(df) < 3:
            return []

        # Get classifications
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        classifications = classify_bars_nb(high, low)

        setups = []

        # Helper to convert classification to bar string
        def bar_to_str(bar_class: int) -> str:
            if bar_class == 1:
                return '1'
            elif bar_class == 2:
                return '2U'
            elif bar_class == -2:
                return '2D'
            elif bar_class == 3:
                return '3'
            return str(bar_class)

        # =====================================================================
        # 3-1 Setups: Outside bar followed by Inside bar
        # Waiting for break to become 3-1-2U (bullish) or 3-1-2D (bearish)
        # =====================================================================
        result_312 = detect_312_setups_nb(classifications, high, low)
        setup_mask, bull_trigger, bear_trigger, stop_long, stop_short, target_long, target_short = result_312

        # CRITICAL: Exclude the last bar (live/incomplete bar) from 3-bar setup detection.
        # For 3-bar patterns (X-1-?), the inside bar must be CLOSED.
        # The NEXT bar (live) is what we watch for the break, not the setup bar itself.
        # Without this exclusion, we'd detect a LIVE bar as the inside bar and then
        # trigger entry when that SAME bar moves - which is incorrect.
        last_bar_idx = len(setup_mask) - 1
        for i in range(len(setup_mask)):
            # Skip the last bar - it's live/incomplete and cannot be a valid setup bar
            if i == last_bar_idx:
                continue
            if setup_mask[i]:
                # This is a 3-1 setup at bar i (inside bar)
                # Create TWO potential signals: bullish and bearish break
                setup_bar_high = high[i]
                setup_bar_low = low[i]

                # Bullish setup (break above inside bar high)
                if not np.isnan(bull_trigger[i]) and not np.isnan(stop_long[i]):
                    entry = bull_trigger[i]
                    stop = stop_long[i]
                    target = target_long[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = entry - stop if stop > 0 else 0
                        reward = target - entry
                        magnitude = (target - entry) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': 1,  # Bullish
                            'direction': 'CALL',
                            'entry': entry,
                            'stop': stop,
                            'target': target,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"3-1-?",  # Pending completion
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '3-1-2',  # Will become 3-1-2U on break
                        })

                # Bearish setup (break below inside bar low)
                if not np.isnan(bear_trigger[i]) and not np.isnan(stop_short[i]):
                    entry = bear_trigger[i]
                    stop = stop_short[i]
                    target = target_short[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = stop - entry if stop > 0 else 0
                        reward = entry - target
                        magnitude = (entry - target) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': -1,  # Bearish
                            'direction': 'PUT',
                            'entry': entry,
                            'stop': stop,
                            'target': target,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"3-1-?",  # Pending completion
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '3-1-2',  # Will become 3-1-2D on break
                        })

        # =====================================================================
        # 2-1 Setups: Directional bar followed by Inside bar
        # Waiting for break to become 2-1-2U (bullish) or 2-1-2D (bearish)
        # =====================================================================
        result_212 = detect_212_setups_nb(classifications, high, low)
        setup_mask, first_bar_dir, bull_trigger, bear_trigger, stop_long, stop_short, target_long, target_short = result_212

        # Same exclusion for 2-1 setups - inside bar must be CLOSED
        last_bar_idx_212 = len(setup_mask) - 1
        for i in range(len(setup_mask)):
            # Skip the last bar - it's live/incomplete and cannot be a valid setup bar
            if i == last_bar_idx_212:
                continue
            if setup_mask[i]:
                # This is a 2-1 setup at bar i (inside bar)
                first_dir = bar_to_str(int(first_bar_dir[i]))  # 2U or 2D
                setup_bar_high = high[i]
                setup_bar_low = low[i]

                # Bullish setup (break above inside bar high)
                if not np.isnan(bull_trigger[i]) and not np.isnan(stop_long[i]):
                    entry = bull_trigger[i]
                    stop = stop_long[i]
                    target = target_long[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = entry - stop if stop > 0 else 0
                        reward = target - entry
                        magnitude = (target - entry) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': 1,  # Bullish
                            'direction': 'CALL',
                            'entry': entry,
                            'stop': stop,
                            'target': target,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"{first_dir}-1-?",  # Pending completion
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '2-1-2',  # Will become 2-1-2U on break
                        })

                # Bearish setup (break below inside bar low)
                if not np.isnan(bear_trigger[i]) and not np.isnan(stop_short[i]):
                    entry = bear_trigger[i]
                    stop = stop_short[i]
                    target = target_short[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = stop - entry if stop > 0 else 0
                        reward = entry - target
                        magnitude = (entry - target) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': -1,  # Bearish
                            'direction': 'PUT',
                            'entry': entry,
                            'stop': stop,
                            'target': target,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"{first_dir}-1-?",  # Pending completion
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '2-1-2',  # Will become 2-1-2D on break
                        })

        # =====================================================================
        # 3-2 Setups: Outside bar followed by Directional bar
        # Session EQUITY-21: FIXED - 3-2 patterns trade the REVERSAL direction
        #   - 3-2D-? = CALL (entry at 2D bar HIGH, waiting for 3-2D-2U)
        #   - 3-2U-? = PUT (entry at 2U bar LOW, waiting for 3-2U-2D)
        # NOT bidirectional - we only trade the reversal, not continuation
        # =====================================================================
        result_322 = detect_322_setups_nb(classifications, high, low)
        (setup_mask_322, setup_dir_322, long_trigger_322, short_trigger_322,
         stop_long_322, stop_short_322, target_long_322, target_short_322) = result_322

        # Session EQUITY-27 FIX: Exclude the forming bar for daily/weekly/monthly timeframes.
        # For resampled HTF data, the last bar is INCOMPLETE (today's running bar).
        # Using it as a setup bar causes premature entries before the bar closes.
        # This matches the 3-1/2-1 exclusion logic above.
        last_bar_idx_322 = len(setup_mask_322) - 1
        for i in range(len(setup_mask_322)):
            if i == last_bar_idx_322:
                continue  # Skip forming bar - pattern not confirmed until bar closes
            if setup_mask_322[i]:
                # Get pattern prefix from bar before the directional bar
                prev_bar_class = classifications[i-1] if i > 0 else 0
                first_bar_str = bar_to_str(int(prev_bar_class))  # Should be '3'
                dir_bar_str = bar_to_str(int(setup_dir_322[i]))  # '2D' or '2U'

                setup_bar_high = high[i]
                setup_bar_low = low[i]

                # EQUITY-21: 3-2 patterns trade the REVERSAL
                # 3-2D-? = waiting for 3-2D-2U (reversal up) -> CALL
                # 3-2U-? = waiting for 3-2U-2D (reversal down) -> PUT
                if setup_dir_322[i] == -2:  # 3-2D -> CALL (reversal up)
                    entry_long = long_trigger_322[i]  # 2D bar's HIGH
                    stop_long = stop_long_322[i]
                    target_long = target_long_322[i]

                    if entry_long > 0 and not np.isnan(target_long) and target_long > 0:
                        risk = entry_long - stop_long if stop_long > 0 else 0
                        reward = target_long - entry_long
                        magnitude = (target_long - entry_long) / entry_long * 100 if entry_long > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': 1,
                            'direction': 'CALL',
                            'entry': entry_long,
                            'stop': stop_long,
                            'target': target_long,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"{first_bar_str}-{dir_bar_str}-?",
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '3-2-2',
                        })

                elif setup_dir_322[i] == 2:  # 3-2U -> PUT (reversal down)
                    entry_short = short_trigger_322[i]  # 2U bar's LOW
                    stop_short = stop_short_322[i]
                    target_short = target_short_322[i]

                    if entry_short > 0 and not np.isnan(target_short) and target_short > 0:
                        risk = stop_short - entry_short if stop_short > 0 else 0
                        reward = entry_short - target_short
                        magnitude = (entry_short - target_short) / entry_short * 100 if entry_short > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': -1,
                            'direction': 'PUT',
                            'entry': entry_short,
                            'stop': stop_short,
                            'target': target_short,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"{first_bar_str}-{dir_bar_str}-?",
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '3-2-2',
                        })

        # =====================================================================
        # 2-2 Setups: Directional bar REVERSAL setups
        # Session EQUITY-22: REVERSAL ONLY - match 3-2 logic from EQUITY-21
        #   - (X)-2D-? = CALL (entry at 2D bar HIGH, waiting for X-2D-2U reversal)
        #   - (X)-2U-? = PUT (entry at 2U bar LOW, waiting for X-2U-2D reversal)
        # Continuations (2U-2U, 2D-2D) do NOT create new entries
        # =====================================================================
        result_22 = detect_22_setups_nb(classifications, high, low)
        (setup_mask_22, setup_dir_22, long_trigger_22, short_trigger_22,
         stop_long_22, stop_short_22, target_long_22, target_short_22) = result_22

        # Session EQUITY-27 FIX: Exclude the forming bar for daily/weekly/monthly timeframes.
        # For resampled HTF data, the last bar is INCOMPLETE (today's running bar).
        # Using it as a setup bar causes premature entries before the bar closes.
        # This matches the 3-1/2-1 exclusion logic above.
        last_bar_idx_22 = len(setup_mask_22) - 1
        for i in range(len(setup_mask_22)):
            if i == last_bar_idx_22:
                continue  # Skip forming bar - pattern not confirmed until bar closes
            if setup_mask_22[i]:
                # Get pattern prefix from bar before this directional bar
                prev_bar_class = classifications[i-1] if i > 0 else 0
                first_bar_str = bar_to_str(int(prev_bar_class))
                dir_bar_str = bar_to_str(int(setup_dir_22[i]))

                # Skip if previous bar was outside (already handled by 3-2 setups)
                if prev_bar_class == 3:
                    continue

                # Session CRYPTO-MONITOR-3: Skip if previous bar is NOT directional
                # A valid 2-2 pattern requires reference bar to be 2U or 2D
                # Inside bar (1) as reference creates invalid patterns like "1-2D-?"
                if abs(prev_bar_class) != 2:
                    continue

                setup_bar_high = high[i]
                setup_bar_low = low[i]

                # EQUITY-22: 2-2 patterns trade the REVERSAL (like 3-2 patterns)
                # (X)-2D-? = waiting for X-2D-2U (reversal up) -> CALL
                # (X)-2U-? = waiting for X-2U-2D (reversal down) -> PUT
                if setup_dir_22[i] == -2:  # 2D bar -> CALL (reversal up)
                    entry_long = long_trigger_22[i]  # 2D bar's HIGH
                    stop_long = stop_long_22[i]
                    target_long = target_long_22[i]

                    if entry_long > 0 and not np.isnan(target_long) and target_long > 0:
                        risk = entry_long - stop_long if stop_long > 0 else 0
                        reward = target_long - entry_long
                        magnitude = (target_long - entry_long) / entry_long * 100 if entry_long > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': 1,
                            'direction': 'CALL',
                            'entry': entry_long,
                            'stop': stop_long,
                            'target': target_long,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"{first_bar_str}-{dir_bar_str}-?",
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '2-2',
                        })

                elif setup_dir_22[i] == 2:  # 2U bar -> PUT (reversal down)
                    entry_short = short_trigger_22[i]  # 2U bar's LOW
                    stop_short = stop_short_22[i]
                    target_short = target_short_22[i]

                    if entry_short > 0 and not np.isnan(target_short) and target_short > 0:
                        risk = stop_short - entry_short if stop_short > 0 else 0
                        reward = entry_short - target_short
                        magnitude = (entry_short - target_short) / entry_short * 100 if entry_short > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append({
                            'index': i,
                            'timestamp': df.index[i],
                            'signal': -1,
                            'direction': 'PUT',
                            'entry': entry_short,
                            'stop': stop_short,
                            'target': target_short,
                            'magnitude_pct': magnitude,
                            'risk_reward': rr,
                            'bar_sequence': f"{first_bar_str}-{dir_bar_str}-?",
                            'signal_type': 'SETUP',
                            'setup_bar_high': setup_bar_high,
                            'setup_bar_low': setup_bar_low,
                            'setup_bar_timestamp': df.index[i],
                            'setup_pattern': '2-2',
                        })

        # =====================================================================
        # 3-? Setups: Pure Outside bar BIDIRECTIONAL setup
        # Session EQUITY-20: INCLUDE last closed bar for outside bar setups
        # If yesterday was a 3 (outside bar), today can break either direction
        # =====================================================================
        result_3 = detect_outside_bar_setups_nb(classifications, high, low)
        (setup_mask_3, long_trigger_3, short_trigger_3,
         stop_long_3, stop_short_3, target_long_3, target_short_3) = result_3

        # Session EQUITY-27 FIX: Exclude the forming bar for daily/weekly/monthly timeframes.
        # For resampled HTF data, the last bar is INCOMPLETE (today's running bar).
        # Using it as a setup bar causes premature entries before the bar closes.
        # This matches the 3-1/2-1 exclusion logic above.
        last_bar_idx_3 = len(setup_mask_3) - 1
        for i in range(len(setup_mask_3)):
            if i == last_bar_idx_3:
                continue  # Skip forming bar - pattern not confirmed until bar closes
            if setup_mask_3[i]:
                setup_bar_high = high[i]
                setup_bar_low = low[i]

                # Create CALL setup (break above outside bar high)
                entry_long = long_trigger_3[i]
                stop_long = stop_long_3[i]
                target_long = target_long_3[i]

                if entry_long > 0 and not np.isnan(target_long) and target_long > 0:
                    risk = entry_long - stop_long if stop_long > 0 else 0
                    reward = target_long - entry_long
                    magnitude = (target_long - entry_long) / entry_long * 100 if entry_long > 0 else 0
                    rr = reward / risk if risk > 0 else 0

                    setups.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'signal': 1,
                        'direction': 'CALL',
                        'entry': entry_long,
                        'stop': stop_long,
                        'target': target_long,
                        'magnitude_pct': magnitude,
                        'risk_reward': rr,
                        'bar_sequence': '3-?',
                        'signal_type': 'SETUP',
                        'setup_bar_high': setup_bar_high,
                        'setup_bar_low': setup_bar_low,
                        'setup_bar_timestamp': df.index[i],
                        'setup_pattern': '3-2',
                    })

                # Create PUT setup (break below outside bar low)
                entry_short = short_trigger_3[i]
                stop_short = stop_short_3[i]
                target_short = target_short_3[i]

                if entry_short > 0 and not np.isnan(target_short) and target_short > 0:
                    risk = stop_short - entry_short if stop_short > 0 else 0
                    reward = entry_short - target_short
                    magnitude = (entry_short - target_short) / entry_short * 100 if entry_short > 0 else 0
                    rr = reward / risk if risk > 0 else 0

                    setups.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'signal': -1,
                        'direction': 'PUT',
                        'entry': entry_short,
                        'stop': stop_short,
                        'target': target_short,
                        'magnitude_pct': magnitude,
                        'risk_reward': rr,
                        'bar_sequence': '3-?',
                        'signal_type': 'SETUP',
                        'setup_bar_high': setup_bar_high,
                        'setup_bar_low': setup_bar_low,
                        'setup_bar_timestamp': df.index[i],
                        'setup_pattern': '3-2',
                    })

        return setups

    def scan_symbol_timeframe(self, symbol: str, timeframe: str,
                              lookback_bars: int = 50) -> List[DetectedSignal]:
        """
        Scan a single symbol/timeframe combination for all patterns.

        Args:
            symbol: Stock symbol
            timeframe: '1H', '1D', '1W', '1M'
            lookback_bars: How many recent bars to check

        Returns:
            List of detected signals
        """
        df = self._fetch_data(symbol, timeframe, lookback_bars)
        if df is None or df.empty:
            return []

        base_context = self._get_market_context(df)
        signals = []

        # Session EQUITY-47: Track TFC pass/fail for scan summary
        tfc_passed = 0
        tfc_failed = 0

        # =================================================================
        # COMPLETED patterns (entry already happened, historical)
        # Session CRYPTO-MONITOR-3: Only include patterns from the LAST bar
        # Patterns from older bars are stale - they should have been traded already.
        # =================================================================
        last_bar_idx = len(df) - 1
        for pattern_type in self.ALL_PATTERNS:
            patterns = self._detect_patterns(df, pattern_type)

            for p in patterns:
                # Only include patterns from the most recent CLOSED bar
                if p['index'] == last_bar_idx:
                    # Session 83K-44: Use full bar sequence from detection
                    pattern_name = p.get('bar_sequence', f"{pattern_type}{'U' if p['direction'] == 'CALL' else 'D'}")

                    # Session 83K-68: Include setup-based detection fields
                    setup_ts = p.get('setup_bar_timestamp')
                    if hasattr(setup_ts, 'to_pydatetime'):
                        setup_ts = setup_ts.to_pydatetime()

                    # Calculate TFC score based on signal direction (EQUITY-23 fix)
                    # CALL = 1 (want 2U bars), PUT = -1 (want 2D bars)
                    direction_int = 1 if p['direction'] == 'CALL' else -1
                    tfc_assessment = self.evaluate_tfc(symbol, timeframe, direction_int)
                    tfc_alignment = tfc_assessment.alignment_label()

                    # Session EQUITY-47: TFC logging for observability
                    # Use pattern_name for consistency with other logging
                    logger.info(
                        f"TFC Eval: {symbol} {timeframe} {pattern_name} - "
                        f"score={tfc_assessment.strength}/10, "
                        f"alignment={tfc_alignment}, "
                        f"passes_flexible={tfc_assessment.passes_flexible}, "
                        f"risk_multiplier={tfc_assessment.risk_multiplier:.2f}, "
                        f"priority_rank={tfc_assessment.priority_rank}"
                    )

                    # Session EQUITY-47: Track TFC pass/fail counts
                    if tfc_assessment.passes_flexible:
                        tfc_passed += 1
                    else:
                        tfc_failed += 1

                    # Create context with TFC for this signal
                    context = SignalContext(
                        vix=base_context.vix,
                        atr_14=base_context.atr_14,
                        atr_percent=base_context.atr_percent,
                        volume_ratio=base_context.volume_ratio,
                        tfc_score=tfc_assessment.strength,
                        tfc_alignment=tfc_alignment,
                        tfc_passes=tfc_assessment.passes_flexible,
                        risk_multiplier=tfc_assessment.risk_multiplier,
                        priority_rank=tfc_assessment.priority_rank,
                    )

                    signal = DetectedSignal(
                        pattern_type=pattern_name,
                        direction=p['direction'],
                        symbol=symbol,
                        timeframe=timeframe,
                        detected_time=p['timestamp'].to_pydatetime() if hasattr(p['timestamp'], 'to_pydatetime') else p['timestamp'],
                        entry_trigger=p['entry'],
                        stop_price=p['stop'],
                        target_price=p['target'],
                        magnitude_pct=p['magnitude_pct'],
                        risk_reward=p['risk_reward'],
                        context=context,
                        # Session 83K-68: Setup-based detection fields
                        signal_type=p.get('signal_type', 'COMPLETED'),
                        setup_bar_high=p.get('setup_bar_high', 0.0),
                        setup_bar_low=p.get('setup_bar_low', 0.0),
                        setup_bar_timestamp=setup_ts,
                        # Session EQUITY-41: Set bidirectionality from pattern registry
                        is_bidirectional=is_bidirectional_pattern(pattern_name),
                    )
                    signals.append(signal)

        # =================================================================
        # Session 83K-71: SETUP patterns (waiting for live break)
        # Only consider the LAST bar as a valid setup (current actionable)
        # Session EQUITY-18: Add setup validity checking (port from crypto scanner)
        # =================================================================
        setups = self._detect_setups(df)

        for p in setups:
            # Session CRYPTO-11: SETUP signals at second-to-last bar (most recent CLOSED inside bar)
            # We skip the last bar in _detect_setups (it's live/incomplete), so the most
            # recent actionable setup is at index len(df) - 2
            # Also include slightly older setups (up to 3 bars back) that haven't been broken yet
            if p['index'] >= len(df) - 4:
                # Session EQUITY-18: Validate setup is still active
                # For X-1 patterns (3-1-2, 2-1-2): Valid if bars stay inside setup bar range
                # For X-2 patterns (3-2-2, 2-2): Valid if entry level not yet triggered
                setup_idx = p['index']
                setup_high = p.get('setup_bar_high', 0.0)
                setup_low = p.get('setup_bar_low', 0.0)
                setup_pattern = p.get('setup_pattern', '')
                entry_price = p.get('entry', 0.0)
                direction = p.get('direction', '')

                # Check if setup is still valid
                # For X-1 patterns (3-1-2, 2-1-2): Valid if bars stay inside the setup bar range
                # For X-2 patterns (3-2-2, 2-2): Valid if entry level not yet triggered
                # For 3-? patterns (3-2 bidirectional): Valid until range is broken (pattern completes)
                setup_still_valid = True
                for j in range(setup_idx + 1, len(df)):
                    bar_high = df['High'].iloc[j]
                    bar_low = df['Low'].iloc[j]

                    if setup_pattern in ('3-1-2', '2-1-2'):
                        # Inside bar patterns: check if range was broken
                        if bar_high > setup_high or bar_low < setup_low:
                            setup_still_valid = False
                            break
                    elif setup_pattern in ('3-2-2', '2-2'):
                        # Directional bar patterns: check if entry was triggered
                        if direction == 'CALL' and bar_high >= entry_price:
                            # Long entry triggered (broke above)
                            setup_still_valid = False
                            break
                        elif direction == 'PUT' and bar_low <= entry_price:
                            # Short entry triggered (broke below)
                            setup_still_valid = False
                            break
                    elif setup_pattern == '3-2':
                        # Session CRYPTO-MONITOR-2 FIX: 3-? bidirectional (outside bar) setups
                        # MUST invalidate when range is broken - that means pattern COMPLETED
                        # The entry monitor handles the TRIGGER in real-time, but once the
                        # pattern completes (any bar breaks the outside bar range), the setup
                        # is no longer valid - it already triggered or should have.
                        if bar_high > setup_high or bar_low < setup_low:
                            setup_still_valid = False
                            break
                    else:
                        # Default: check if range was broken
                        if bar_high > setup_high or bar_low < setup_low:
                            setup_still_valid = False
                            break

                if not setup_still_valid:
                    continue  # Skip this invalidated setup

                setup_ts = p.get('setup_bar_timestamp')
                if hasattr(setup_ts, 'to_pydatetime'):
                    setup_ts = setup_ts.to_pydatetime()

                # Calculate TFC score based on signal direction (EQUITY-23 fix)
                # CALL = 1 (want 2U bars), PUT = -1 (want 2D bars)
                direction_int = 1 if p['direction'] == 'CALL' else -1
                tfc_assessment = self.evaluate_tfc(symbol, timeframe, direction_int)
                tfc_alignment = tfc_assessment.alignment_label()

                # Session EQUITY-47: TFC logging for observability (SETUP patterns)
                # Use bar_sequence for consistency with other logging
                logger.info(
                    f"TFC Eval: {symbol} {timeframe} {p['bar_sequence']} (SETUP) - "
                    f"score={tfc_assessment.strength}/10, "
                    f"alignment={tfc_alignment}, "
                    f"passes_flexible={tfc_assessment.passes_flexible}, "
                    f"risk_multiplier={tfc_assessment.risk_multiplier:.2f}, "
                    f"priority_rank={tfc_assessment.priority_rank}"
                )

                # Session EQUITY-47: Track TFC pass/fail counts
                if tfc_assessment.passes_flexible:
                    tfc_passed += 1
                else:
                    tfc_failed += 1

                # Create context with TFC for this signal
                context = SignalContext(
                    vix=base_context.vix,
                    atr_14=base_context.atr_14,
                    atr_percent=base_context.atr_percent,
                    volume_ratio=base_context.volume_ratio,
                    tfc_score=tfc_assessment.strength,
                    tfc_alignment=tfc_alignment,
                    tfc_passes=tfc_assessment.passes_flexible,
                    risk_multiplier=tfc_assessment.risk_multiplier,
                    priority_rank=tfc_assessment.priority_rank,
                )

                signal = DetectedSignal(
                    pattern_type=p['bar_sequence'],
                    direction=p['direction'],
                    symbol=symbol,
                    timeframe=timeframe,
                    detected_time=p['timestamp'].to_pydatetime() if hasattr(p['timestamp'], 'to_pydatetime') else p['timestamp'],
                    entry_trigger=p['entry'],
                    stop_price=p['stop'],
                    target_price=p['target'],
                    magnitude_pct=p['magnitude_pct'],
                    risk_reward=p['risk_reward'],
                    context=context,
                    # SETUP signals - waiting for live break
                    signal_type='SETUP',
                    setup_bar_high=p.get('setup_bar_high', 0.0),
                    setup_bar_low=p.get('setup_bar_low', 0.0),
                    setup_bar_timestamp=setup_ts,
                    # Session EQUITY-41: Set bidirectionality from pattern registry
                    is_bidirectional=is_bidirectional_pattern(p['bar_sequence']),
                )
                signals.append(signal)

        # Session EQUITY-47: Log TFC breakdown summary
        if tfc_passed > 0 or tfc_failed > 0:
            logger.info(
                f"TFC Summary: {symbol} {timeframe} - "
                f"signals={len(signals)}, TFC_passed={tfc_passed}, TFC_failed={tfc_failed}"
            )

        return signals

    # =========================================================================
    # Session 83K-80: Multi-Timeframe Scanning via 15-min Resampling
    # =========================================================================

    def scan_symbol_all_timeframes_resampled(self, symbol: str,
                                              lookback_bars: int = 50) -> List[DetectedSignal]:
        """
        Scan symbol across ALL timeframes using 15-min base resampling.

        Session 83K-80: HTF Scanning Architecture Fix.

        This method:
        1. Fetches 15-min data once
        2. Resamples to 1H, 1D, 1W, 1M
        3. Detects SETUP patterns on all timeframes
        4. Returns unified list of signals

        CRITICAL: This approach fixes the HTF scanning bug where:
        - Daily scans at 5PM missed entry opportunities
        - Weekly scans on Friday missed 5 days of entries
        - Monthly scans on 28th missed 4 weeks of entries

        With resampling, we detect setups within 15 minutes of inside bar close
        and entry_monitor polls every 60s for live triggers.

        Args:
            symbol: Stock symbol
            lookback_bars: Number of bars to look back per timeframe

        Returns:
            List of detected signals across all timeframes
        """
        all_signals = []

        # Calculate 15-min bars needed for monthly resampling
        # ~26 15-min bars/day * ~21 trading days/month * lookback_bars months + buffer
        base_lookback = lookback_bars * 30 * 26

        # Fetch 15-min data once
        base_df = self._fetch_15min_data(symbol, base_lookback)

        if base_df is None or base_df.empty:
            warnings.warn(f"Failed to fetch 15-min data for {symbol}")
            return all_signals

        # Scan each timeframe via resampling
        for tf in self.DEFAULT_TIMEFRAMES:
            resampled_df = self._resample_to_htf(base_df, tf)

            if resampled_df is None or resampled_df.empty:
                warnings.warn(f"Failed to resample {symbol} to {tf}")
                continue

            # Get base market context from resampled data
            base_context = self._get_market_context(resampled_df)

            # =================================================================
            # COMPLETED patterns (entry already happened, historical)
            # Session CRYPTO-MONITOR-3: Only include patterns from the LAST bar
            # Session EQUITY-54: Add TFC evaluation for resampled scans
            # =================================================================
            last_bar_idx_resamp = len(resampled_df) - 1
            for pattern_type in self.ALL_PATTERNS:
                patterns = self._detect_patterns(resampled_df, pattern_type)

                for p in patterns:
                    # Only include patterns from the most recent CLOSED bar
                    if p['index'] == last_bar_idx_resamp:
                        pattern_name = p.get('bar_sequence', f"{pattern_type}{'U' if p['direction'] == 'CALL' else 'D'}")

                        setup_ts = p.get('setup_bar_timestamp')
                        if hasattr(setup_ts, 'to_pydatetime'):
                            setup_ts = setup_ts.to_pydatetime()

                        # Session EQUITY-54: Calculate TFC for resampled patterns
                        direction_int = 1 if p['direction'] == 'CALL' else -1
                        tfc_assessment = self.evaluate_tfc(symbol, tf, direction_int)
                        tfc_alignment = tfc_assessment.alignment_label()

                        # Create context with TFC data
                        context = SignalContext(
                            vix=base_context.vix,
                            atr_14=base_context.atr_14,
                            atr_percent=base_context.atr_percent,
                            volume_ratio=base_context.volume_ratio,
                            tfc_score=tfc_assessment.strength,
                            tfc_alignment=tfc_alignment,
                            tfc_passes=tfc_assessment.passes_flexible,
                            risk_multiplier=tfc_assessment.risk_multiplier,
                            priority_rank=tfc_assessment.priority_rank,
                        )

                        signal = DetectedSignal(
                            pattern_type=pattern_name,
                            direction=p['direction'],
                            symbol=symbol,
                            timeframe=tf,
                            detected_time=p['timestamp'].to_pydatetime() if hasattr(p['timestamp'], 'to_pydatetime') else p['timestamp'],
                            entry_trigger=p['entry'],
                            stop_price=p['stop'],
                            target_price=p['target'],
                            magnitude_pct=p['magnitude_pct'],
                            risk_reward=p['risk_reward'],
                            context=context,
                            signal_type=p.get('signal_type', 'COMPLETED'),
                            setup_bar_high=p.get('setup_bar_high', 0.0),
                            setup_bar_low=p.get('setup_bar_low', 0.0),
                            setup_bar_timestamp=setup_ts,
                            # Session EQUITY-41: Set bidirectionality from pattern registry
                            is_bidirectional=is_bidirectional_pattern(pattern_name),
                        )
                        all_signals.append(signal)

            # =================================================================
            # SETUP patterns (waiting for live break)
            # Session EQUITY-57: Fixed bug - _detect_setups excludes last bar (incomplete),
            # so checking index == len-1 matched NOTHING. Now matches non-resampled logic.
            # Include recent setups (up to 3 bars back) that haven't been broken yet.
            # Session EQUITY-54: Add TFC evaluation for resampled scans
            # =================================================================
            setups = self._detect_setups(resampled_df)

            for p in setups:
                # SETUP signals: Include recent closed bars (len-4 to len-2)
                # _detect_setups excludes last bar (len-1), so most recent is len-2
                if p['index'] >= len(resampled_df) - 4:
                    setup_ts = p.get('setup_bar_timestamp')
                    if hasattr(setup_ts, 'to_pydatetime'):
                        setup_ts = setup_ts.to_pydatetime()

                    # Session EQUITY-54: Calculate TFC for SETUP patterns
                    direction_int = 1 if p['direction'] == 'CALL' else -1
                    tfc_assessment = self.evaluate_tfc(symbol, tf, direction_int)
                    tfc_alignment = tfc_assessment.alignment_label()

                    # Create context with TFC data
                    setup_context = SignalContext(
                        vix=base_context.vix,
                        atr_14=base_context.atr_14,
                        atr_percent=base_context.atr_percent,
                        volume_ratio=base_context.volume_ratio,
                        tfc_score=tfc_assessment.strength,
                        tfc_alignment=tfc_alignment,
                        tfc_passes=tfc_assessment.passes_flexible,
                        risk_multiplier=tfc_assessment.risk_multiplier,
                        priority_rank=tfc_assessment.priority_rank,
                    )

                    signal = DetectedSignal(
                        pattern_type=p['bar_sequence'],
                        direction=p['direction'],
                        symbol=symbol,
                        timeframe=tf,
                        detected_time=p['timestamp'].to_pydatetime() if hasattr(p['timestamp'], 'to_pydatetime') else p['timestamp'],
                        entry_trigger=p['entry'],
                        stop_price=p['stop'],
                        target_price=p['target'],
                        magnitude_pct=p['magnitude_pct'],
                        risk_reward=p['risk_reward'],
                        context=setup_context,
                        signal_type='SETUP',
                        setup_bar_high=p.get('setup_bar_high', 0.0),
                        setup_bar_low=p.get('setup_bar_low', 0.0),
                        setup_bar_timestamp=setup_ts,
                        # Session EQUITY-41: Set bidirectionality from pattern registry
                        is_bidirectional=is_bidirectional_pattern(p['bar_sequence']),
                    )
                    all_signals.append(signal)

        return all_signals

    def scan_all_timeframes(self, symbol: str) -> List[DetectedSignal]:
        """
        Scan all timeframes for a single symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of all detected signals across timeframes
        """
        all_signals = []

        for tf in self.DEFAULT_TIMEFRAMES:
            signals = self.scan_symbol_timeframe(symbol, tf)
            all_signals.extend(signals)

        return all_signals

    def scan_all_symbols(self) -> Dict[str, List[DetectedSignal]]:
        """
        Scan all default symbols across all timeframes.

        Returns:
            Dict mapping symbol to list of detected signals
        """
        results = {}

        for symbol in self.DEFAULT_SYMBOLS:
            print(f"Scanning {symbol}...")
            signals = self.scan_all_timeframes(symbol)
            results[symbol] = signals
            print(f"  Found {len(signals)} signals")

        return results

    def get_latest_signals(self, max_age_bars: int = 3) -> List[DetectedSignal]:
        """
        Get the most recent actionable signals across all symbols.

        Args:
            max_age_bars: Maximum age of signal in bars

        Returns:
            List of actionable signals sorted by magnitude
        """
        all_signals = []

        for symbol in self.DEFAULT_SYMBOLS:
            for tf in self.DEFAULT_TIMEFRAMES:
                df = self._fetch_data(symbol, tf, lookback_bars=20)
                if df is None or df.empty:
                    continue

                context = self._get_market_context(df)

                for pattern_type in self.ALL_PATTERNS:
                    patterns = self._detect_patterns(df, pattern_type)

                    for p in patterns:
                        # Only include signals from last max_age_bars
                        if p['index'] >= len(df) - max_age_bars:
                            # Use bar_sequence if available, otherwise construct simple name
                            pattern_name = p.get('bar_sequence', f"{pattern_type}{'U' if p['direction'] == 'CALL' else 'D'}")

                            signal = DetectedSignal(
                                pattern_type=pattern_name,
                                direction=p['direction'],
                                symbol=symbol,
                                timeframe=tf,
                                detected_time=p['timestamp'].to_pydatetime() if hasattr(p['timestamp'], 'to_pydatetime') else p['timestamp'],
                                entry_trigger=p['entry'],
                                stop_price=p['stop'],
                                target_price=p['target'],
                                magnitude_pct=p['magnitude_pct'],
                                risk_reward=p['risk_reward'],
                                context=context,
                                # Session EQUITY-41: Set bidirectionality from pattern registry
                                is_bidirectional=is_bidirectional_pattern(pattern_name),
                            )
                            all_signals.append(signal)

        # Sort by magnitude (higher magnitude = more attractive)
        all_signals.sort(key=lambda s: s.magnitude_pct, reverse=True)

        return all_signals

    def print_signals(self, signals: List[DetectedSignal]) -> None:
        """Print signals in a readable format."""
        if not signals:
            print("No signals detected.")
            return

        print("\n" + "=" * 80)
        print("STRAT PATTERN SIGNALS")
        print("=" * 80)

        for i, s in enumerate(signals, 1):
            print(f"\n[{i}] {s.pattern_type} {s.direction} on {s.symbol} ({s.timeframe})")
            print(f"    Detected: {s.detected_time}")
            print(f"    Entry Trigger: ${s.entry_trigger:.2f}")
            print(f"    Stop: ${s.stop_price:.2f}")
            print(f"    Target: ${s.target_price:.2f}")
            print(f"    Magnitude: {s.magnitude_pct:.2f}%")
            print(f"    Risk/Reward: {s.risk_reward:.2f}:1")
            print(f"    VIX: {s.context.vix:.1f}")

        print("\n" + "=" * 80)

    def to_dataframe(self, signals: List[DetectedSignal]) -> pd.DataFrame:
        """Convert signals to DataFrame for analysis."""
        if not signals:
            return pd.DataFrame()

        rows = []
        for s in signals:
            row = {
                'pattern_type': s.pattern_type,
                'direction': s.direction,
                'symbol': s.symbol,
                'timeframe': s.timeframe,
                'detected_time': s.detected_time,
                'entry_trigger': s.entry_trigger,
                'stop_price': s.stop_price,
                'target_price': s.target_price,
                'magnitude_pct': s.magnitude_pct,
                'risk_reward': s.risk_reward,
                'vix': s.context.vix,
                'atr_14': s.context.atr_14,
                'volume_ratio': s.context.volume_ratio,
            }
            rows.append(row)

        return pd.DataFrame(rows)


# =============================================================================
# Convenience Functions
# =============================================================================

def scan_for_signals(symbols: Optional[List[str]] = None,
                     timeframes: Optional[List[str]] = None) -> List[DetectedSignal]:
    """
    Quick scan for STRAT pattern signals.

    Args:
        symbols: List of symbols to scan (default: SPY, QQQ, IWM, DIA, AAPL)
        timeframes: List of timeframes (default: 1H, 1D, 1W, 1M)

    Returns:
        List of detected signals
    """
    scanner = PaperSignalScanner()

    symbols = symbols or scanner.DEFAULT_SYMBOLS
    timeframes = timeframes or scanner.DEFAULT_TIMEFRAMES

    all_signals = []

    for symbol in symbols:
        for tf in timeframes:
            signals = scanner.scan_symbol_timeframe(symbol, tf)
            all_signals.extend(signals)

    return all_signals


def get_actionable_signals(max_age_bars: int = 3) -> List[DetectedSignal]:
    """
    Get actionable signals from the most recent bars.

    Args:
        max_age_bars: Maximum age of signal in bars

    Returns:
        List of actionable signals sorted by magnitude
    """
    scanner = PaperSignalScanner()
    return scanner.get_latest_signals(max_age_bars)
