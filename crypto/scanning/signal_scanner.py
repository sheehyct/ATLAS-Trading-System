"""
STRAT Signal Scanner for Crypto Perpetual Futures.

Scans for STRAT patterns on BTC/ETH perpetuals using Coinbase INTX data.
Adapted from strat/paper_signal_scanner.py for 24/7 crypto markets.

Key Differences from Equities Scanner:
- No market hours filter (crypto is 24/7)
- Friday maintenance window handling (5-6 PM ET)
- Different timeframe hierarchy: 1w -> 1d -> 4h -> 1h -> 15m
- Uses Coinbase API instead of Alpaca

Usage:
    from crypto.scanning import CryptoSignalScanner

    scanner = CryptoSignalScanner()
    signals = scanner.scan_all_timeframes('BTC-PERP-INTX')
    scanner.print_signals(signals)
"""

import logging
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from crypto import config
from crypto.exchange.coinbase_client import CoinbaseClient
from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext
from strat.bar_classifier import classify_bars_nb
from strat.pattern_detector import (
    detect_212_patterns_nb,
    detect_22_patterns_nb,
    detect_312_patterns_nb,
    detect_322_patterns_nb,
    detect_32_patterns_nb,
    # Setup detectors for patterns waiting for live break
    detect_212_setups_nb,
    detect_312_setups_nb,
    detect_322_setups_nb,
    detect_22_setups_nb,
    detect_outside_bar_setups_nb,
)

logger = logging.getLogger(__name__)


class CryptoSignalScanner:
    """
    STRAT pattern scanner for crypto perpetual futures.

    Scans for all STRAT pattern types across multiple timeframes.
    Handles 24/7 crypto markets with Friday maintenance window.
    """

    # Default symbols to scan
    DEFAULT_SYMBOLS = config.CRYPTO_SYMBOLS

    # Crypto timeframe hierarchy (different from equities)
    DEFAULT_TIMEFRAMES = config.TIMEFRAMES

    # All pattern types to detect
    ALL_PATTERNS = config.SCAN_PATTERN_TYPES

    def __init__(self, client: Optional[CoinbaseClient] = None):
        """
        Initialize scanner with Coinbase client.

        Args:
            client: CoinbaseClient instance (creates new if None)
        """
        self.client = client or CoinbaseClient(simulation_mode=True)

    # =========================================================================
    # MAINTENANCE WINDOW HANDLING
    # =========================================================================

    def is_maintenance_window(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if given time is within Coinbase maintenance window.

        Maintenance window: Friday 5-6 PM ET (22:00-23:00 UTC standard time)

        Args:
            dt: Datetime to check (defaults to now UTC)

        Returns:
            True if within maintenance window
        """
        if not config.MAINTENANCE_WINDOW_ENABLED:
            return False

        if dt is None:
            dt = datetime.now(timezone.utc)

        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Check if Friday
        if dt.weekday() != config.MAINTENANCE_DAY:
            return False

        # Check if within maintenance hours
        hour = dt.hour
        return config.MAINTENANCE_START_HOUR_UTC <= hour < config.MAINTENANCE_END_HOUR_UTC

    def _overlaps_maintenance(self, bar_start: datetime, timeframe: str) -> bool:
        """
        Check if a bar overlaps the maintenance window.

        Args:
            bar_start: Bar start timestamp (UTC)
            timeframe: Timeframe string (e.g., '15m', '1h', '4h')

        Returns:
            True if any part of bar overlaps maintenance window
        """
        if not config.MAINTENANCE_WINDOW_ENABLED:
            return False

        # Ensure timezone-aware
        if bar_start.tzinfo is None:
            bar_start = bar_start.replace(tzinfo=timezone.utc)

        # Calculate bar duration in hours
        tf_hours = {
            "15m": 0.25,
            "1h": 1,
            "4h": 4,
            "1d": 24,
            "1w": 168,
        }
        duration_hours = tf_hours.get(timeframe, 1)

        # Check if bar start or end falls in maintenance
        bar_end = pd.Timestamp(bar_start) + pd.Timedelta(hours=duration_hours)

        # Check both start and end
        return self.is_maintenance_window(bar_start) or self.is_maintenance_window(
            bar_end.to_pydatetime().replace(tzinfo=timezone.utc)
        )

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_data(
        self, symbol: str, timeframe: str, lookback_bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Coinbase.

        No market hours filter needed for 24/7 crypto markets.
        Marks bars that overlap maintenance window.

        Args:
            symbol: Trading symbol (e.g., 'BTC-PERP-INTX')
            timeframe: Timeframe ('15m', '1h', '4h', '1d', '1w')
            lookback_bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV columns and is_maintenance_gap flag
        """
        try:
            # Map timeframe format for Coinbase client
            tf_map = {
                "15m": "15m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
                "1w": "1w",
            }
            interval = tf_map.get(timeframe, "1h")

            df = self.client.get_historical_ohlcv(
                symbol=symbol,
                interval=interval,
                limit=lookback_bars,
            )

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None

            # Standardize column names to match equities format
            df.columns = [c.capitalize() for c in df.columns]
            if "Open" not in df.columns:
                # Handle lowercase columns
                rename_map = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
                df = df.rename(columns=rename_map)

            # Mark bars overlapping maintenance window
            df["is_maintenance_gap"] = df.index.map(
                lambda x: self._overlaps_maintenance(x.to_pydatetime(), timeframe)
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return None

    # =========================================================================
    # MARKET CONTEXT
    # =========================================================================

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from OHLC data."""
        if len(df) < period:
            return 0.0

        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(np.maximum(tr1, tr2), tr3)[1:]

        if len(tr) < period:
            return 0.0

        return float(np.mean(tr[-period:]))

    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volume vs 20-bar average."""
        if len(df) < 21 or "Volume" not in df.columns:
            return 1.0

        avg_volume = df["Volume"].iloc[-21:-1].mean()
        current_volume = df["Volume"].iloc[-1]

        if avg_volume > 0:
            return current_volume / avg_volume
        return 1.0

    def _get_market_context(self, df: pd.DataFrame) -> CryptoSignalContext:
        """Get market context for signal."""
        atr = self._calculate_atr(df)
        current_price = df["Close"].iloc[-1] if not df.empty else 0
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0
        volume_ratio = self._calculate_volume_ratio(df)

        return CryptoSignalContext(
            atr_14=atr,
            atr_percent=atr_pct,
            volume_ratio=volume_ratio,
        )

    # =========================================================================
    # BAR SEQUENCE FORMATTING
    # =========================================================================

    def _get_bar_sequence(
        self, pattern_type: str, classifications: np.ndarray, idx: int, direction: int
    ) -> str:
        """
        Get full bar sequence string for detected pattern.

        Args:
            pattern_type: Base pattern type ('2-2', '3-2', etc.)
            classifications: Array of bar classifications
            idx: Index of trigger bar
            direction: 1 for bullish, -1 for bearish

        Returns:
            Full bar sequence (e.g., '2D-1-2U', '3-2D-2U')
        """

        def bar_to_str(bar_class: int) -> str:
            if bar_class == 1:
                return "1"
            elif bar_class == 2:
                return "2U"
            elif bar_class == -2:
                return "2D"
            elif abs(bar_class) == 3:
                return "3"
            return "?"

        if idx < 2:
            return f"{pattern_type}{'U' if direction > 0 else 'D'}"

        if pattern_type == "2-2":
            bar1 = int(classifications[idx - 1])
            bar2 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}"

        elif pattern_type == "3-2":
            bar1 = int(classifications[idx - 1])
            bar2 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}"

        elif pattern_type == "3-2-2":
            bar1 = int(classifications[idx - 2])
            bar2 = int(classifications[idx - 1])
            bar3 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}-{bar_to_str(bar3)}"

        elif pattern_type == "2-1-2":
            bar1 = int(classifications[idx - 2])
            bar2 = int(classifications[idx - 1])
            bar3 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}-{bar_to_str(bar3)}"

        elif pattern_type == "3-1-2":
            bar1 = int(classifications[idx - 2])
            bar2 = int(classifications[idx - 1])
            bar3 = int(classifications[idx])
            return f"{bar_to_str(bar1)}-{bar_to_str(bar2)}-{bar_to_str(bar3)}"

        return f"{pattern_type}{'U' if direction > 0 else 'D'}"

    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================

    def _detect_patterns(self, df: pd.DataFrame, pattern_type: str) -> List[Dict]:
        """
        Detect specific pattern type in OHLCV data.

        Args:
            df: OHLCV DataFrame
            pattern_type: '2-2', '3-2', '3-2-2', '2-1-2', '3-1-2'

        Returns:
            List of detected patterns with prices and metadata
        """
        if df is None or len(df) < 5:
            return []

        # Skip bars with maintenance gaps for pattern detection
        if df["is_maintenance_gap"].any():
            # Find continuous segment without gaps
            gap_indices = df[df["is_maintenance_gap"]].index
            if len(gap_indices) > 0:
                # Only use bars after the last gap
                last_gap_idx = df.index.get_loc(gap_indices[-1])
                if last_gap_idx >= len(df) - 3:
                    # Not enough bars after gap
                    logger.debug(f"Skipping {pattern_type} - maintenance gap too recent")
                    return []

        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)
        classifications = classify_bars_nb(high, low)

        patterns = []

        # Select detector based on pattern type
        if pattern_type == "2-2":
            result = detect_22_patterns_nb(classifications, high, low)
        elif pattern_type == "3-2":
            result = detect_32_patterns_nb(classifications, high, low)
        elif pattern_type == "3-2-2":
            result = detect_322_patterns_nb(classifications, high, low)
        elif pattern_type == "2-1-2":
            result = detect_212_patterns_nb(classifications, high, low)
        elif pattern_type == "3-1-2":
            result = detect_312_patterns_nb(classifications, high, low)
        else:
            return []

        entries_mask, stops, targets, directions = result[:4]

        # Extract pattern occurrences
        for i in range(len(entries_mask)):
            if entries_mask[i]:
                direction = "LONG" if directions[i] > 0 else "SHORT"

                # Entry uses setup bar (i-1), not trigger bar (i)
                entry = high[i - 1] if directions[i] > 0 else low[i - 1]
                setup_bar_high = high[i - 1]
                setup_bar_low = low[i - 1]

                stop = stops[i]
                target = targets[i]

                # Get full bar sequence
                bar_sequence = self._get_bar_sequence(
                    pattern_type, classifications, i, directions[i]
                )

                # Validate target geometry
                if not np.isnan(target) and not np.isnan(stop) and stop > 0:
                    if directions[i] > 0:  # Bullish
                        if target <= entry:
                            risk = entry - stop
                            if risk > 0:
                                target = entry + (risk * 1.5)
                    elif directions[i] < 0:  # Bearish
                        if target >= entry:
                            risk = stop - entry
                            if risk > 0:
                                target = entry - (risk * 1.5)

                # Calculate magnitude and R:R
                if entry > 0 and not np.isnan(target) and target > 0:
                    magnitude = abs(target - entry) / entry * 100
                    risk = abs(entry - stop) if stop > 0 and not np.isnan(stop) else 0
                    reward = abs(target - entry)
                    rr = reward / risk if risk > 0 else 0

                    # Check if bar overlaps maintenance
                    has_gap = bool(df["is_maintenance_gap"].iloc[i])

                    patterns.append(
                        {
                            "index": i,
                            "timestamp": df.index[i],
                            "signal": directions[i],
                            "direction": direction,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "magnitude_pct": magnitude,
                            "risk_reward": rr,
                            "bar_sequence": bar_sequence,
                            "signal_type": "COMPLETED",
                            "setup_bar_high": setup_bar_high,
                            "setup_bar_low": setup_bar_low,
                            "setup_bar_timestamp": df.index[i - 1] if i > 0 else df.index[i],
                            "has_maintenance_gap": has_gap,
                        }
                    )

        return patterns

    def _detect_setups(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect SETUP patterns (X-1 formations waiting for break).

        Args:
            df: OHLCV DataFrame

        Returns:
            List of setups with trigger levels
        """
        if df is None or len(df) < 3:
            return []

        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)
        classifications = classify_bars_nb(high, low)

        setups = []

        def bar_to_str(bar_class: int) -> str:
            if bar_class == 1:
                return "1"
            elif bar_class == 2:
                return "2U"
            elif bar_class == -2:
                return "2D"
            elif bar_class == 3:
                return "3"
            return str(bar_class)

        # 3-1 Setups (waiting to become 3-1-2)
        result_312 = detect_312_setups_nb(classifications, high, low)
        (
            setup_mask,
            bull_trigger,
            bear_trigger,
            stop_long,
            stop_short,
            target_long,
            target_short,
        ) = result_312

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
                setup_bar_high = high[i]
                setup_bar_low = low[i]
                has_gap = bool(df["is_maintenance_gap"].iloc[i])

                # Prior bar info for detecting 2-bar patterns (Session CRYPTO-8)
                # For 3-1, prior bar is the outside bar (type 3)
                prior_bar_type = int(classifications[i - 1]) if i > 0 else 0
                prior_bar_high = high[i - 1] if i > 0 else 0.0
                prior_bar_low = low[i - 1] if i > 0 else 0.0

                # Bullish setup
                if not np.isnan(bull_trigger[i]) and not np.isnan(stop_long[i]):
                    entry = bull_trigger[i]
                    stop = stop_long[i]
                    target = target_long[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = entry - stop if stop > 0 else 0
                        reward = target - entry
                        magnitude = (target - entry) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append(
                            {
                                "index": i,
                                "timestamp": df.index[i],
                                "signal": 1,
                                "direction": "LONG",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "magnitude_pct": magnitude,
                                "risk_reward": rr,
                                "bar_sequence": "3-1-?",
                                "signal_type": "SETUP",
                                "setup_bar_high": setup_bar_high,
                                "setup_bar_low": setup_bar_low,
                                "setup_bar_timestamp": df.index[i],
                                "setup_pattern": "3-1-2",
                                "has_maintenance_gap": has_gap,
                                "prior_bar_type": prior_bar_type,
                                "prior_bar_high": prior_bar_high,
                                "prior_bar_low": prior_bar_low,
                            }
                        )

                # Bearish setup
                if not np.isnan(bear_trigger[i]) and not np.isnan(stop_short[i]):
                    entry = bear_trigger[i]
                    stop = stop_short[i]
                    target = target_short[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = stop - entry if stop > 0 else 0
                        reward = entry - target
                        magnitude = (entry - target) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append(
                            {
                                "index": i,
                                "timestamp": df.index[i],
                                "signal": -1,
                                "direction": "SHORT",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "magnitude_pct": magnitude,
                                "risk_reward": rr,
                                "bar_sequence": "3-1-?",
                                "signal_type": "SETUP",
                                "setup_bar_high": setup_bar_high,
                                "setup_bar_low": setup_bar_low,
                                "setup_bar_timestamp": df.index[i],
                                "setup_pattern": "3-1-2",
                                "has_maintenance_gap": has_gap,
                                "prior_bar_type": prior_bar_type,
                                "prior_bar_high": prior_bar_high,
                                "prior_bar_low": prior_bar_low,
                            }
                        )

        # 2-1 Setups (waiting to become 2-1-2)
        result_212 = detect_212_setups_nb(classifications, high, low)
        (
            setup_mask,
            first_bar_dir,
            bull_trigger,
            bear_trigger,
            stop_long,
            stop_short,
            target_long,
            target_short,
        ) = result_212

        # Same exclusion for 2-1 setups - inside bar must be CLOSED
        last_bar_idx_212 = len(setup_mask) - 1
        for i in range(len(setup_mask)):
            # Skip the last bar - it's live/incomplete and cannot be a valid setup bar
            if i == last_bar_idx_212:
                continue
            if setup_mask[i]:
                first_dir = bar_to_str(int(first_bar_dir[i]))
                setup_bar_high = high[i]
                setup_bar_low = low[i]
                has_gap = bool(df["is_maintenance_gap"].iloc[i])

                # Prior bar info for detecting 2-bar patterns (Session CRYPTO-8)
                # When inside bar breaks opposite direction, pattern becomes X-2D or X-2U
                prior_bar_type = int(classifications[i - 1]) if i > 0 else 0
                prior_bar_high = high[i - 1] if i > 0 else 0.0
                prior_bar_low = low[i - 1] if i > 0 else 0.0

                # Bullish setup
                if not np.isnan(bull_trigger[i]) and not np.isnan(stop_long[i]):
                    entry = bull_trigger[i]
                    stop = stop_long[i]
                    target = target_long[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = entry - stop if stop > 0 else 0
                        reward = target - entry
                        magnitude = (target - entry) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append(
                            {
                                "index": i,
                                "timestamp": df.index[i],
                                "signal": 1,
                                "direction": "LONG",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "magnitude_pct": magnitude,
                                "risk_reward": rr,
                                "bar_sequence": f"{first_dir}-1-?",
                                "signal_type": "SETUP",
                                "setup_bar_high": setup_bar_high,
                                "setup_bar_low": setup_bar_low,
                                "setup_bar_timestamp": df.index[i],
                                "setup_pattern": "2-1-2",
                                "has_maintenance_gap": has_gap,
                                "prior_bar_type": prior_bar_type,
                                "prior_bar_high": prior_bar_high,
                                "prior_bar_low": prior_bar_low,
                            }
                        )

                # Bearish setup
                if not np.isnan(bear_trigger[i]) and not np.isnan(stop_short[i]):
                    entry = bear_trigger[i]
                    stop = stop_short[i]
                    target = target_short[i]
                    if entry > 0 and not np.isnan(target) and target > 0:
                        risk = stop - entry if stop > 0 else 0
                        reward = entry - target
                        magnitude = (entry - target) / entry * 100 if entry > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append(
                            {
                                "index": i,
                                "timestamp": df.index[i],
                                "signal": -1,
                                "direction": "SHORT",
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "magnitude_pct": magnitude,
                                "risk_reward": rr,
                                "bar_sequence": f"{first_dir}-1-?",
                                "signal_type": "SETUP",
                                "setup_bar_high": setup_bar_high,
                                "setup_bar_low": setup_bar_low,
                                "setup_bar_timestamp": df.index[i],
                                "setup_pattern": "2-1-2",
                                "has_maintenance_gap": has_gap,
                                "prior_bar_type": prior_bar_type,
                                "prior_bar_high": prior_bar_high,
                                "prior_bar_low": prior_bar_low,
                            }
                        )

        # =====================================================================
        # 3-2 Setups: Outside bar followed by Directional bar
        # Session EQUITY-19: BIDIRECTIONAL - Create BOTH LONG and SHORT setups
        # Per STRAT: "The forming bar is always '1' until it breaks"
        # =====================================================================
        result_322 = detect_322_setups_nb(classifications, high, low)
        (setup_mask_322, setup_dir_322, long_trigger_322, short_trigger_322,
         stop_long_322, stop_short_322, target_long_322, target_short_322) = result_322

        # Session EQUITY-20: INCLUDE last closed bar for bidirectional setups
        # The last bar in data is the LAST CLOSED bar, NOT the forming bar
        for i in range(len(setup_mask_322)):
            if setup_mask_322[i]:
                prev_bar_class = classifications[i - 1] if i > 0 else 0
                first_bar_str = bar_to_str(int(prev_bar_class))
                dir_bar_str = bar_to_str(int(setup_dir_322[i]))

                setup_bar_high = high[i]
                setup_bar_low = low[i]
                has_gap = bool(df["is_maintenance_gap"].iloc[i])

                # Create LONG setup (break above directional bar high)
                entry_long = long_trigger_322[i]
                stop_long = stop_long_322[i]
                target_long = target_long_322[i]

                if entry_long > 0 and not np.isnan(target_long) and target_long > 0:
                    risk = entry_long - stop_long if stop_long > 0 else 0
                    reward = target_long - entry_long
                    magnitude = (target_long - entry_long) / entry_long * 100 if entry_long > 0 else 0
                    rr = reward / risk if risk > 0 else 0

                    setups.append(
                        {
                            "index": i,
                            "timestamp": df.index[i],
                            "signal": 1,
                            "direction": "LONG",
                            "entry": entry_long,
                            "stop": stop_long,
                            "target": target_long,
                            "magnitude_pct": magnitude,
                            "risk_reward": rr,
                            "bar_sequence": f"{first_bar_str}-{dir_bar_str}-?",
                            "signal_type": "SETUP",
                            "setup_bar_high": setup_bar_high,
                            "setup_bar_low": setup_bar_low,
                            "setup_bar_timestamp": df.index[i],
                            "setup_pattern": "3-2-2",
                            "has_maintenance_gap": has_gap,
                            "prior_bar_type": int(prev_bar_class),
                            "prior_bar_high": high[i - 1] if i > 0 else 0.0,
                            "prior_bar_low": low[i - 1] if i > 0 else 0.0,
                        }
                    )

                # Create SHORT setup (break below directional bar low)
                entry_short = short_trigger_322[i]
                stop_short = stop_short_322[i]
                target_short = target_short_322[i]

                if entry_short > 0 and not np.isnan(target_short) and target_short > 0:
                    risk = stop_short - entry_short if stop_short > 0 else 0
                    reward = entry_short - target_short
                    magnitude = (entry_short - target_short) / entry_short * 100 if entry_short > 0 else 0
                    rr = reward / risk if risk > 0 else 0

                    setups.append(
                        {
                            "index": i,
                            "timestamp": df.index[i],
                            "signal": -1,
                            "direction": "SHORT",
                            "entry": entry_short,
                            "stop": stop_short,
                            "target": target_short,
                            "magnitude_pct": magnitude,
                            "risk_reward": rr,
                            "bar_sequence": f"{first_bar_str}-{dir_bar_str}-?",
                            "signal_type": "SETUP",
                            "setup_bar_high": setup_bar_high,
                            "setup_bar_low": setup_bar_low,
                            "setup_bar_timestamp": df.index[i],
                            "setup_pattern": "3-2-2",
                            "has_maintenance_gap": has_gap,
                            "prior_bar_type": int(prev_bar_class),
                            "prior_bar_high": high[i - 1] if i > 0 else 0.0,
                            "prior_bar_low": low[i - 1] if i > 0 else 0.0,
                        }
                    )

        # =====================================================================
        # 2-2 Setups: Directional bar REVERSAL setups
        # Session EQUITY-22: REVERSAL ONLY - match 3-2 logic from EQUITY-21
        #   - (X)-2D-? = LONG (entry at 2D bar HIGH, waiting for X-2D-2U reversal)
        #   - (X)-2U-? = SHORT (entry at 2U bar LOW, waiting for X-2U-2D reversal)
        # Continuations (2U-2U, 2D-2D) do NOT create new entries
        # =====================================================================
        result_22 = detect_22_setups_nb(classifications, high, low)
        (setup_mask_22, setup_dir_22, long_trigger_22, short_trigger_22,
         stop_long_22, stop_short_22, target_long_22, target_short_22) = result_22

        # Session EQUITY-20: INCLUDE last closed bar for setups
        for i in range(len(setup_mask_22)):
            if setup_mask_22[i]:
                prev_bar_class = classifications[i - 1] if i > 0 else 0
                first_bar_str = bar_to_str(int(prev_bar_class))
                dir_bar_str = bar_to_str(int(setup_dir_22[i]))

                # Skip if previous bar was outside (already handled by 3-2 setups)
                if prev_bar_class == 3:
                    continue

                setup_bar_high = high[i]
                setup_bar_low = low[i]
                has_gap = bool(df["is_maintenance_gap"].iloc[i])

                # EQUITY-22: 2-2 patterns trade the REVERSAL (like 3-2 patterns)
                # (X)-2D-? = waiting for X-2D-2U (reversal up) -> LONG
                # (X)-2U-? = waiting for X-2U-2D (reversal down) -> SHORT
                if setup_dir_22[i] == -2:  # 2D bar -> LONG (reversal up)
                    entry_long = long_trigger_22[i]  # 2D bar's HIGH
                    stop_long = stop_long_22[i]
                    target_long = target_long_22[i]

                    if entry_long > 0 and not np.isnan(target_long) and target_long > 0:
                        risk = entry_long - stop_long if stop_long > 0 else 0
                        reward = target_long - entry_long
                        magnitude = (target_long - entry_long) / entry_long * 100 if entry_long > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append(
                            {
                                "index": i,
                                "timestamp": df.index[i],
                                "signal": 1,
                                "direction": "LONG",
                                "entry": entry_long,
                                "stop": stop_long,
                                "target": target_long,
                                "magnitude_pct": magnitude,
                                "risk_reward": rr,
                                "bar_sequence": f"{first_bar_str}-{dir_bar_str}-?",
                                "signal_type": "SETUP",
                                "setup_bar_high": setup_bar_high,
                                "setup_bar_low": setup_bar_low,
                                "setup_bar_timestamp": df.index[i],
                                "setup_pattern": "2-2",
                                "has_maintenance_gap": has_gap,
                                "prior_bar_type": int(prev_bar_class),
                                "prior_bar_high": high[i - 1] if i > 0 else 0.0,
                                "prior_bar_low": low[i - 1] if i > 0 else 0.0,
                            }
                        )

                elif setup_dir_22[i] == 2:  # 2U bar -> SHORT (reversal down)
                    entry_short = short_trigger_22[i]  # 2U bar's LOW
                    stop_short = stop_short_22[i]
                    target_short = target_short_22[i]

                    if entry_short > 0 and not np.isnan(target_short) and target_short > 0:
                        risk = stop_short - entry_short if stop_short > 0 else 0
                        reward = entry_short - target_short
                        magnitude = (entry_short - target_short) / entry_short * 100 if entry_short > 0 else 0
                        rr = reward / risk if risk > 0 else 0

                        setups.append(
                            {
                                "index": i,
                                "timestamp": df.index[i],
                                "signal": -1,
                                "direction": "SHORT",
                                "entry": entry_short,
                                "stop": stop_short,
                                "target": target_short,
                                "magnitude_pct": magnitude,
                                "risk_reward": rr,
                                "bar_sequence": f"{first_bar_str}-{dir_bar_str}-?",
                                "signal_type": "SETUP",
                                "setup_bar_high": setup_bar_high,
                                "setup_bar_low": setup_bar_low,
                                "setup_bar_timestamp": df.index[i],
                                "setup_pattern": "2-2",
                                "has_maintenance_gap": has_gap,
                                "prior_bar_type": int(prev_bar_class),
                                "prior_bar_high": high[i - 1] if i > 0 else 0.0,
                                "prior_bar_low": low[i - 1] if i > 0 else 0.0,
                            }
                        )

        # =====================================================================
        # 3-? Setups: Pure Outside bar BIDIRECTIONAL setup
        # Session EQUITY-20: If last bar was a 3 (outside), create bidirectional setup
        # =====================================================================
        result_3 = detect_outside_bar_setups_nb(classifications, high, low)
        (setup_mask_3, long_trigger_3, short_trigger_3,
         stop_long_3, stop_short_3, target_long_3, target_short_3) = result_3

        for i in range(len(setup_mask_3)):
            if setup_mask_3[i]:
                setup_bar_high = high[i]
                setup_bar_low = low[i]
                has_gap = bool(df["is_maintenance_gap"].iloc[i])

                # Create LONG setup (break above outside bar high)
                entry_long = long_trigger_3[i]
                stop_long = stop_long_3[i]
                target_long = target_long_3[i]

                if entry_long > 0 and not np.isnan(target_long) and target_long > 0:
                    risk = entry_long - stop_long if stop_long > 0 else 0
                    reward = target_long - entry_long
                    magnitude = (target_long - entry_long) / entry_long * 100 if entry_long > 0 else 0
                    rr = reward / risk if risk > 0 else 0

                    setups.append(
                        {
                            "index": i,
                            "timestamp": df.index[i],
                            "signal": 1,
                            "direction": "LONG",
                            "entry": entry_long,
                            "stop": stop_long,
                            "target": target_long,
                            "magnitude_pct": magnitude,
                            "risk_reward": rr,
                            "bar_sequence": "3-?",
                            "signal_type": "SETUP",
                            "setup_bar_high": setup_bar_high,
                            "setup_bar_low": setup_bar_low,
                            "setup_bar_timestamp": df.index[i],
                            "setup_pattern": "3-2",
                            "has_maintenance_gap": has_gap,
                            "prior_bar_type": 3,
                            "prior_bar_high": setup_bar_high,
                            "prior_bar_low": setup_bar_low,
                        }
                    )

                # Create SHORT setup (break below outside bar low)
                entry_short = short_trigger_3[i]
                stop_short = stop_short_3[i]
                target_short = target_short_3[i]

                if entry_short > 0 and not np.isnan(target_short) and target_short > 0:
                    risk = stop_short - entry_short if stop_short > 0 else 0
                    reward = entry_short - target_short
                    magnitude = (entry_short - target_short) / entry_short * 100 if entry_short > 0 else 0
                    rr = reward / risk if risk > 0 else 0

                    setups.append(
                        {
                            "index": i,
                            "timestamp": df.index[i],
                            "signal": -1,
                            "direction": "SHORT",
                            "entry": entry_short,
                            "stop": stop_short,
                            "target": target_short,
                            "magnitude_pct": magnitude,
                            "risk_reward": rr,
                            "bar_sequence": "3-?",
                            "signal_type": "SETUP",
                            "setup_bar_high": setup_bar_high,
                            "setup_bar_low": setup_bar_low,
                            "setup_bar_timestamp": df.index[i],
                            "setup_pattern": "3-2",
                            "has_maintenance_gap": has_gap,
                            "prior_bar_type": 3,
                            "prior_bar_high": setup_bar_high,
                            "prior_bar_low": setup_bar_low,
                        }
                    )

        return setups

    # =========================================================================
    # PUBLIC SCANNING METHODS
    # =========================================================================

    def scan_symbol_timeframe(
        self, symbol: str, timeframe: str, lookback_bars: int = 50
    ) -> List[CryptoDetectedSignal]:
        """
        Scan a single symbol/timeframe for all STRAT patterns.

        Args:
            symbol: Trading symbol (e.g., 'BTC-PERP-INTX')
            timeframe: Timeframe ('15m', '1h', '4h', '1d', '1w')
            lookback_bars: How many bars to analyze

        Returns:
            List of detected signals
        """
        df = self._fetch_data(symbol, timeframe, lookback_bars)
        if df is None or df.empty:
            return []

        base_context = self._get_market_context(df)
        signals = []

        # Detect COMPLETED patterns
        for pattern_type in self.ALL_PATTERNS:
            patterns = self._detect_patterns(df, pattern_type)

            for p in patterns:
                # Only include recent signals (last 5 bars)
                if p["index"] >= len(df) - 5:
                    setup_ts = p.get("setup_bar_timestamp")
                    if hasattr(setup_ts, "to_pydatetime"):
                        setup_ts = setup_ts.to_pydatetime()

                    # Calculate TFC score based on signal direction (EQUITY-23 fix)
                    # LONG = 1 (want 2U bars), SHORT = -1 (want 2D bars)
                    direction_int = 1 if p["direction"] == "LONG" else -1
                    tfc_score = self.get_tfc_score(symbol, direction_int)
                    tfc_alignment = f"{tfc_score}/4 {'BULLISH' if direction_int == 1 else 'BEARISH'}"

                    # Create context with TFC for this signal
                    context = CryptoSignalContext(
                        atr_14=base_context.atr_14,
                        atr_percent=base_context.atr_percent,
                        volume_ratio=base_context.volume_ratio,
                        tfc_score=tfc_score,
                        tfc_alignment=tfc_alignment,
                    )

                    signal = CryptoDetectedSignal(
                        pattern_type=p["bar_sequence"],
                        direction=p["direction"],
                        symbol=symbol,
                        timeframe=timeframe,
                        # CRYPTO-MONITOR-1 FIX: Use scan time, not bar timestamp
                        # Bar timestamp is preserved in setup_bar_timestamp for reference
                        detected_time=datetime.now(timezone.utc),
                        entry_trigger=p["entry"],
                        stop_price=p["stop"],
                        target_price=p["target"],
                        magnitude_pct=p["magnitude_pct"],
                        risk_reward=p["risk_reward"],
                        context=context,
                        signal_type="COMPLETED",
                        setup_bar_high=p.get("setup_bar_high", 0.0),
                        setup_bar_low=p.get("setup_bar_low", 0.0),
                        setup_bar_timestamp=setup_ts,
                        has_maintenance_gap=p.get("has_maintenance_gap", False),
                    )
                    signals.append(signal)

        # Detect SETUP patterns (still valid - entry not yet triggered)
        setups = self._detect_setups(df)
        for p in setups:
            setup_idx = p["index"]
            setup_high = p.get("setup_bar_high", 0.0)
            setup_low = p.get("setup_bar_low", 0.0)
            setup_pattern = p.get("setup_pattern", "")
            entry_price = p.get("entry", 0.0)
            direction = p.get("direction", "")

            # Check if setup is still valid
            # For X-1 patterns (3-1-2, 2-1-2): Valid if bars stay inside the setup bar range
            # For X-2 patterns (3-2-2, 2-2): Valid if entry level not yet triggered
            # For 3-? patterns (3-2 bidirectional): ALWAYS valid - entry monitor handles trigger
            setup_still_valid = True
            for j in range(setup_idx + 1, len(df)):
                bar_high = df["High"].iloc[j]
                bar_low = df["Low"].iloc[j]

                if setup_pattern in ("3-1-2", "2-1-2"):
                    # Inside bar patterns: check if range was broken
                    if bar_high > setup_high or bar_low < setup_low:
                        setup_still_valid = False
                        break
                elif setup_pattern in ("3-2-2", "2-2"):
                    # Directional bar patterns (reversal setups):
                    # Session EQUITY-18 FIX: Check BOTH conditions:
                    # 1. Entry triggered (pattern completed)
                    # 2. Opposite side broken (pattern failed - continuation not reversal)
                    if direction == "LONG":
                        if bar_high >= entry_price:
                            # Long entry triggered (broke above) - pattern completed
                            setup_still_valid = False
                            break
                        elif bar_low < setup_low:
                            # Broke below setup bar low - pattern failed (price continuing down)
                            setup_still_valid = False
                            break
                    elif direction == "SHORT":
                        if bar_low <= entry_price:
                            # Short entry triggered (broke below) - pattern completed
                            setup_still_valid = False
                            break
                        elif bar_high > setup_high:
                            # Broke above setup bar high - pattern failed (price continuing up)
                            setup_still_valid = False
                            break
                elif setup_pattern == "3-2":
                    # Session EQUITY-24 FIX: 3-? bidirectional (outside bar) setups
                    # Do NOT invalidate based on range break - breaking the range IS the trigger!
                    # The entry monitor will detect which direction broke first (LONG or SHORT)
                    # and handle the entry accordingly.
                    pass  # Always valid - let entry monitor handle
                else:
                    # Default: check if range was broken
                    if bar_high > setup_high or bar_low < setup_low:
                        setup_still_valid = False
                        break

            if setup_still_valid:
                setup_ts = p.get("setup_bar_timestamp")
                if hasattr(setup_ts, "to_pydatetime"):
                    setup_ts = setup_ts.to_pydatetime()

                # Calculate TFC score based on signal direction (EQUITY-23 fix)
                # LONG = 1 (want 2U bars), SHORT = -1 (want 2D bars)
                direction_int = 1 if p["direction"] == "LONG" else -1
                tfc_score = self.get_tfc_score(symbol, direction_int)
                tfc_alignment = f"{tfc_score}/4 {'BULLISH' if direction_int == 1 else 'BEARISH'}"

                # Create context with TFC for this signal
                context = CryptoSignalContext(
                    atr_14=base_context.atr_14,
                    atr_percent=base_context.atr_percent,
                    volume_ratio=base_context.volume_ratio,
                    tfc_score=tfc_score,
                    tfc_alignment=tfc_alignment,
                )

                signal = CryptoDetectedSignal(
                    pattern_type=p["bar_sequence"],
                    direction=p["direction"],
                    symbol=symbol,
                    timeframe=timeframe,
                    # CRYPTO-MONITOR-1 FIX: Use scan time, not bar timestamp
                    # Bar timestamp is preserved in setup_bar_timestamp for reference
                    detected_time=datetime.now(timezone.utc),
                    entry_trigger=p["entry"],
                    stop_price=p["stop"],
                    target_price=p["target"],
                    magnitude_pct=p["magnitude_pct"],
                    risk_reward=p["risk_reward"],
                    context=context,
                    signal_type="SETUP",
                    setup_bar_high=p.get("setup_bar_high", 0.0),
                    setup_bar_low=p.get("setup_bar_low", 0.0),
                    setup_bar_timestamp=setup_ts,
                    prior_bar_type=p.get("prior_bar_type", 0),
                    prior_bar_high=p.get("prior_bar_high", 0.0),
                    prior_bar_low=p.get("prior_bar_low", 0.0),
                    has_maintenance_gap=p.get("has_maintenance_gap", False),
                )
                signals.append(signal)

        return signals

    def scan_all_timeframes(self, symbol: str) -> List[CryptoDetectedSignal]:
        """
        Scan all timeframes for a single symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of all detected signals across timeframes
        """
        all_signals = []

        for tf in self.DEFAULT_TIMEFRAMES:
            signals = self.scan_symbol_timeframe(symbol, tf)
            all_signals.extend(signals)

        return all_signals

    def scan_all_symbols(self) -> Dict[str, List[CryptoDetectedSignal]]:
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

    def get_tfc_score(self, symbol: str, direction: int) -> int:
        """
        Calculate Full Timeframe Continuity (FTFC) score.

        Checks alignment of bar classifications across timeframes.
        Score of 4 = all timeframes aligned (strongest setup).

        Args:
            symbol: Trading symbol
            direction: 1 for bullish (want 2U bars), -1 for bearish (want 2D bars)

        Returns:
            Score from 0-4
        """
        score = 0
        target_class = 2 * direction  # 2 for bull, -2 for bear

        # Check each timeframe (1w, 1d, 4h, 1h)
        for tf in ["1w", "1d", "4h", "1h"]:
            df = self._fetch_data(symbol, tf, lookback_bars=5)
            if df is not None and not df.empty:
                high = df["High"].values.astype(np.float64)
                low = df["Low"].values.astype(np.float64)
                classifications = classify_bars_nb(high, low)

                # Check last bar classification
                if len(classifications) > 0:
                    last_class = int(classifications[-1])
                    if last_class == target_class:
                        score += 1

        return score

    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================

    def print_signals(self, signals: List[CryptoDetectedSignal]) -> None:
        """Print signals in readable format."""
        if not signals:
            print("No signals detected.")
            return

        print("\n" + "=" * 80)
        print("CRYPTO STRAT SIGNALS")
        if self.is_maintenance_window():
            print("** MAINTENANCE WINDOW ACTIVE - Trading paused **")
        print("=" * 80)

        for i, s in enumerate(signals, 1):
            gap_warning = " [MAINT GAP]" if s.has_maintenance_gap else ""
            print(
                f"\n[{i}] {s.pattern_type} {s.direction} on {s.symbol} ({s.timeframe}){gap_warning}"
            )
            print(f"    Type: {s.signal_type}")
            print(f"    Detected: {s.detected_time}")
            print(f"    Entry: ${s.entry_trigger:,.2f}")
            print(f"    Stop: ${s.stop_price:,.2f}")
            print(f"    Target: ${s.target_price:,.2f}")
            print(f"    Magnitude: {s.magnitude_pct:.2f}%")
            print(f"    R:R: {s.risk_reward:.2f}:1")

        print("\n" + "=" * 80)

    def to_dataframe(self, signals: List[CryptoDetectedSignal]) -> pd.DataFrame:
        """Convert signals to DataFrame for analysis."""
        if not signals:
            return pd.DataFrame()

        rows = []
        for s in signals:
            rows.append(
                {
                    "pattern_type": s.pattern_type,
                    "direction": s.direction,
                    "symbol": s.symbol,
                    "timeframe": s.timeframe,
                    "signal_type": s.signal_type,
                    "detected_time": s.detected_time,
                    "entry_trigger": s.entry_trigger,
                    "stop_price": s.stop_price,
                    "target_price": s.target_price,
                    "magnitude_pct": s.magnitude_pct,
                    "risk_reward": s.risk_reward,
                    "atr_14": s.context.atr_14,
                    "volume_ratio": s.context.volume_ratio,
                    "has_maintenance_gap": s.has_maintenance_gap,
                }
            )

        return pd.DataFrame(rows)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def scan_crypto(
    symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None
) -> List[CryptoDetectedSignal]:
    """
    Quick scan for crypto STRAT signals.

    Args:
        symbols: List of symbols to scan (default: BTC-PERP-INTX, ETH-PERP-INTX)
        timeframes: List of timeframes (default: 1w, 1d, 4h, 1h, 15m)

    Returns:
        List of detected signals
    """
    scanner = CryptoSignalScanner()

    symbols = symbols or scanner.DEFAULT_SYMBOLS
    timeframes = timeframes or scanner.DEFAULT_TIMEFRAMES

    all_signals = []

    for symbol in symbols:
        for tf in timeframes:
            signals = scanner.scan_symbol_timeframe(symbol, tf)
            all_signals.extend(signals)

    return all_signals
