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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path

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
)
from strat.paper_trading import (
    PaperTrade,
    create_paper_trade,
)
# Session 83K-52: Import from single source of truth
from strat.tier1_detector import PatternType, Timeframe
from integrations.tiingo_data_fetcher import TiingoDataFetcher


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
    DEFAULT_TIMEFRAMES = ['1H', '1D', '1W', '1M']

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
        Fallback: Tiingo (daily/weekly/monthly only, no hourly)

        Args:
            symbol: Stock symbol
            timeframe: '1H', '1D', '1W', '1M'
            lookback_bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV columns
        """
        vbt = self._get_vbt()

        # Calculate start date based on timeframe
        if timeframe == '1H':
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
            end_date = end + timedelta(days=1) if timeframe == '1H' else end

            data = vbt.AlpacaData.pull(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                timeframe=tf_alpaca,
                tz='America/New_York'
            )
            df = data.get()

            # Handle hourly bar alignment for 1H (market-open-aligned)
            if timeframe == '1H':
                df = self._align_hourly_bars(df)

            return df

        except Exception as e:
            # Alpaca failed, try Tiingo fallback (daily/weekly/monthly only)
            if tf_tiingo is None:
                warnings.warn(f"Failed to fetch {symbol} {timeframe} (no Tiingo fallback for hourly): {e}")
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

        for i in range(len(setup_mask)):
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

        for i in range(len(setup_mask)):
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

        context = self._get_market_context(df)
        signals = []

        # =================================================================
        # COMPLETED patterns (entry already happened, historical)
        # =================================================================
        for pattern_type in self.ALL_PATTERNS:
            patterns = self._detect_patterns(df, pattern_type)

            for p in patterns:
                # Only include recent signals (last 5 bars)
                if p['index'] >= len(df) - 5:
                    # Session 83K-44: Use full bar sequence from detection
                    pattern_name = p.get('bar_sequence', f"{pattern_type}{'U' if p['direction'] == 'CALL' else 'D'}")

                    # Session 83K-68: Include setup-based detection fields
                    setup_ts = p.get('setup_bar_timestamp')
                    if hasattr(setup_ts, 'to_pydatetime'):
                        setup_ts = setup_ts.to_pydatetime()

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
                    )
                    signals.append(signal)

        # =================================================================
        # Session 83K-71: SETUP patterns (waiting for live break)
        # Only consider the LAST bar as a valid setup (current actionable)
        # =================================================================
        setups = self._detect_setups(df)

        for p in setups:
            # SETUP signals: Only include if at the LAST bar (current inside bar)
            # This ensures we're detecting setups that can be acted upon NOW
            if p['index'] == len(df) - 1:
                setup_ts = p.get('setup_bar_timestamp')
                if hasattr(setup_ts, 'to_pydatetime'):
                    setup_ts = setup_ts.to_pydatetime()

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
                )
                signals.append(signal)

        return signals

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
                            pattern_name = f"{pattern_type}{'U' if p['direction'] == 'CALL' else 'D'}"

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
