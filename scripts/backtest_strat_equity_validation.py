"""
STRAT Equity Validation Backtest

CRITICAL VALIDATION BEFORE OPTIONS MODULE IMPLEMENTATION

Purpose:
    Validate that STRAT patterns (3-1-2, 2-1-2) achieve >60% magnitude hit rates
    on equity data BEFORE investing 15+ hours building options module.

    If patterns don't work on equities, options module is pointless.
    This equity validation is faster (no options data required) and proves the edge exists.

Success Criteria:
    - Magnitude hit rate >= 60% with full timeframe continuity
    - Risk-reward ratio >= 2:1
    - Sufficient pattern occurrences (100+ total across all stocks)
    - Days to magnitude distribution for DTE calibration

Output:
    - CSV file with pattern occurrences and outcomes
    - Statistics summary printed to console
    - GO/NO-GO decision for options module implementation

Usage:
    python scripts/backtest_strat_equity_validation.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import vectorbtpro as vbt
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the main .env file (not .env.example or .env.development)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add project root to path (MUST be before local imports)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Session 83K-53: VIX correlation analysis (import after path setup)
from analysis.vix_data import fetch_vix_data, get_vix_at_date, categorize_vix, get_vix_bucket_name

from strat.bar_classifier import StratBarClassifier, classify_bars
from strat.pattern_detector import StratPatternDetector
from strat.timeframe_continuity import TimeframeContinuityChecker, get_continuity_strength
from strat.tier1_detector import Tier1Detector, Timeframe as T1Timeframe, PatternType
from integrations.tiingo_data_fetcher import TiingoDataFetcher
# Session 83K-53: Import expanded symbols for cross-instrument comparison
from validation.strat_validator import EXPANDED_SYMBOLS, TICKER_CATEGORIES, get_ticker_category
# ATLAS integration not used in initial validation (testing patterns alone first)
# from strat.atlas_integration import filter_strat_signals


# Magnitude hit detection tolerance (prevent floating point rounding errors)
# Example: target=196.37 but bar_high=196.369999 should still count as hit
MAGNITUDE_EPSILON = 0.01  # 1 cent tolerance


# Configuration
VALIDATION_CONFIG = {
    'backtest_period': {
        'start': '2017-01-01',
        'end': '2025-01-01',
        'rationale': '8 years covering multiple regimes: bull (2017-19), COVID (2020), volatility (2021), bear (2022), recovery (2023-24)'
    },
    'stock_universe': {
        'symbols': [
            # Technology (10 stocks, $40-200 range)
            'INTC', 'AMD', 'QCOM', 'TXN', 'ADI', 'MRVL', 'ON', 'MCHP', 'SWKS', 'NXPI',
            # Financials (10 stocks, $40-150 range)
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'SCHW', 'AXP', 'USB', 'PNC',
            # Healthcare (10 stocks, $50-200 range)
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'LLY', 'BMY',
            # Consumer (10 stocks, $40-200 range)
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
            # Industrials (10 stocks, $40-200 range)
            'HON', 'UNP', 'CAT', 'BA', 'GE', 'MMM', 'DE', 'LMT', 'RTX', 'UPS'
        ],
        'count': 50,
        'price_range': '$40-200 (suitable for options execution)',
        'rationale': 'Diversified sectors, institutional-quality names'
    },
    'timeframes': {
        'base': '1H',  # Base data download (always hourly for maximum granularity)
        'detection': ['1H', '1D', '1W', '1M'],  # Timeframes to detect patterns on
        'continuity_check': ['1M', '1W', '1D', '4H', '1H'],  # Session 66: Reverted 2D (degraded performance)
        'rationale': 'Multi-timeframe continuity checking without 2D (Session 64 baseline)'
    },
    'pattern_types': ['3-1-2 Up', '3-1-2 Down', '2-1-2 Up', '2-1-2 Down', '2-2 Up', '2-2 Down', '3-2 Up', '3-2 Down'],
    'filters': {
        'require_full_continuity': False,  # Changed to flexible (Session 56 Bug #3 fix)
        'use_flexible_continuity': True,   # Use timeframe-appropriate continuity
        'min_continuity_strength': 3,      # Require 3/5 timeframes aligned minimum
        'use_atlas_regime': False,  # First test patterns alone, then with ATLAS
        'min_pattern_quality': 'HIGH',
        'require_continuation_bars': True,  # Session 63: Mandate continuation bar filter
        'min_continuation_bars': 2,  # Session 63: Require 2+ continuation bars (35→73% hit rate improvement)
        'use_2d_hybrid_timeframe': False,  # Session 66: Disabled 2D (degraded R:R by 19%)
        'use_tier1_detector': True,  # Session 71: Use Tier1Detector for pattern detection (single source of truth)
        'include_22_down': False  # Session 69: 2-2 Down has negative expectancy without heavy filtering
    },
    'metrics': {
        'magnitude_window': 5,  # Bars to check for magnitude hit (patterns can take 1-5 bars)
        # Session 83K-64: Max holding reduced to match DTE (with 3-day exit buffer)
        # Rationale: 90% of patterns hit magnitude in 1-5 bars anyway.
        # DTE must >= max holding equivalent days to avoid option expiring before exit.
        # Formula: max_holding = (DTE - 3 day buffer) converted to bars
        'max_holding_bars': {
            '1H': 28,   # 7 day DTE - 3 day buffer = 4 days = ~28 hourly bars
            '1D': 18,   # 21 day DTE - 3 day buffer = 18 trading days
            '1W': 4,    # 35 day DTE - 3 day buffer = ~4.5 weeks = 4 bars
            '1M': 2,    # 75 day DTE - 3 day buffer = ~2.4 months = 2 bars
        },
        'magnitude_hit_threshold': 0.50,  # 50% minimum hit rate (realistic per research)
        'risk_reward_threshold': 2.0,  # 2:1 minimum R:R for GO decision
        'min_pattern_count': 100  # Need 100+ patterns for statistical significance
    }
}


class EquityValidationBacktest:
    """
    Validate STRAT patterns on equity data before options module implementation.

    This backtest tests whether patterns achieve sufficient magnitude hit rates
    to justify building the options execution module.
    """

    def __init__(self, config: dict = None):
        """
        Initialize equity validation backtest.

        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary (uses VALIDATION_CONFIG if not provided)
        """
        self.config = config or VALIDATION_CONFIG
        self.results = []
        self.summary_stats = {}

    def pull_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = '1H'
    ) -> pd.DataFrame:
        """
        Pull historical OHLC data for a symbol using Tiingo (30+ years) or Alpaca (7 years).

        Data Source Selection:
        - Tiingo: For historical data >6 years old (30+ years available, free tier)
        - Alpaca: For recent data <=6 years (higher quality, paid source)

        Parameters:
        -----------
        symbol : str
            Stock symbol
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
        timeframe : str
            Data timeframe (default: '1H')

        Returns:
        --------
        pd.DataFrame
            OHLC data with datetime index
        """
        try:
            # Determine optimal data source based on start date
            start_date = pd.to_datetime(start)
            years_ago = (pd.Timestamp.now() - start_date).days / 365.25

            alpaca_api_key = os.getenv('ALPACA_API_KEY')
            alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
            tiingo_api_key = os.getenv('TIINGO_API_KEY')

            ohlc = None

            # Try Alpaca for recent data (<= 6 years)
            if years_ago <= 6 and alpaca_api_key and alpaca_secret_key:
                try:
                    print(f"  Using Alpaca (high-quality recent data)")
                    data = vbt.AlpacaData.pull(
                        symbol,
                        start=start,
                        end=end,
                        timeframe=timeframe,
                        client_config={
                            'api_key': alpaca_api_key,
                            'secret_key': alpaca_secret_key,
                            'paper': True
                        },
                        tz='America/New_York'  # CRITICAL: Prevent UTC date shifts
                    )
                    ohlc = data.get()
                except Exception as e:
                    print(f"  Alpaca failed: {str(e)[:80]}...")
                    print(f"  Falling back to Tiingo...")

            # Use Tiingo if Alpaca failed or for historical data > 6 years
            if ohlc is None:
                if not tiingo_api_key:
                    raise ValueError("Tiingo API key required. Set TIINGO_API_KEY in .env file.")

                print(f"  Using Tiingo (30+ years historical data)")
                fetcher = TiingoDataFetcher(api_key=tiingo_api_key)
                vbt_data = fetcher.fetch(symbol, start_date=start, end_date=end, timeframe=timeframe)
                ohlc = vbt_data.get()

            # Verify no weekend bars
            if len(ohlc) > 0:
                weekend_bars = ohlc.index[ohlc.index.dayofweek >= 5]
                if len(weekend_bars) > 0:
                    print(f"WARNING: {symbol} has {len(weekend_bars)} weekend bars!")

            return ohlc

        except Exception as e:
            print(f"ERROR pulling data for {symbol}: {e}")
            return None

    def resample_multi_timeframe(
        self,
        hourly_data: pd.DataFrame,
        timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample hourly data to multiple timeframes.

        Parameters:
        -----------
        hourly_data : pd.DataFrame
            Hourly OHLC data
        timeframes : list
            List of timeframes to resample to (e.g., ['1M', '1W', '1D', '4H', '1H'])

        Returns:
        --------
        dict
            Dictionary mapping timeframe -> resampled DataFrame
        """
        resampled = {}

        for tf in timeframes:
            if tf == '1H':
                # Base timeframe (no resampling needed)
                resampled[tf] = hourly_data
            else:
                # Resample to target timeframe
                tf_data = hourly_data.resample(tf).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()

                resampled[tf] = tf_data

        return resampled

    def detect_patterns_with_continuity(
        self,
        symbol: str,
        hourly_data: pd.DataFrame,
        detection_timeframe: str,
        continuity_timeframes: List[str]
    ) -> List[Dict]:
        """
        Detect STRAT patterns on specified timeframe with continuity filtering.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        hourly_data : pd.DataFrame
            Hourly OHLC data (base data for resampling)
        detection_timeframe : str
            Timeframe to detect patterns on ('1H', '1D', '1W', '1M')
        continuity_timeframes : list
            Timeframes for continuity check

        Returns:
        --------
        list of dict
            List of pattern occurrences with metadata
        """
        patterns_found = []

        # Resample hourly data to all needed timeframes
        all_timeframes = list(set(continuity_timeframes + [detection_timeframe]))
        mtf_data = self.resample_multi_timeframe(hourly_data, all_timeframes)

        # Get data for detection timeframe
        detection_data = mtf_data[detection_timeframe]

        # Run bar classification on detection timeframe
        high = detection_data['High']
        low = detection_data['Low']
        close = detection_data['Close']

        # Step 1: Classify bars
        from strat.bar_classifier import classify_bars_nb
        classifications = classify_bars_nb(high.values, low.values)

        # Step 2: Run pattern detector with classifications
        pattern_result = StratPatternDetector.run(classifications, high, low)

        # Get pattern signals (correct attribute names)
        entries_312 = pattern_result.entries_312
        directions_312 = pattern_result.directions_312
        stops_312 = pattern_result.stops_312
        targets_312 = pattern_result.targets_312

        entries_212 = pattern_result.entries_212
        directions_212 = pattern_result.directions_212
        stops_212 = pattern_result.stops_212
        targets_212 = pattern_result.targets_212

        entries_22 = pattern_result.entries_22
        directions_22 = pattern_result.directions_22
        stops_22 = pattern_result.stops_22
        targets_22 = pattern_result.targets_22

        entries_32 = pattern_result.entries_32
        directions_32 = pattern_result.directions_32
        stops_32 = pattern_result.stops_32
        targets_32 = pattern_result.targets_32

        # Process 3-1-2 Up patterns (bullish: directions_312 == 1)
        for i in range(len(entries_312)):
            if entries_312.iloc[i] and directions_312.iloc[i] == 1:
                pattern_date = entries_312.index[i]

                # Market open filter: For hourly patterns, need 3 bars minimum
                # 9:30 (bar 1), 10:30 (bar 2), 11:30 (bar 3 triggers pattern)
                if detection_timeframe == '1H':
                    if pattern_date.hour < 11 or (pattern_date.hour == 11 and pattern_date.minute < 30):
                        continue  # Skip patterns before 11:30 AM

                # Entry price = inside bar high (one bar before trigger)
                pattern_loc = detection_data.index.get_loc(pattern_date)
                if pattern_loc < 1:
                    continue  # Need at least 1 bar before trigger
                inside_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[inside_bar_date, 'High']

                # Check timeframe continuity at this date
                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bullish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                    if not continuity['passes_flexible']:
                        continue
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bullish'
                    )
                    if self.config['filters']['require_full_continuity']:
                        if not continuity['full_continuity']:
                            continue

                # Record pattern
                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '3-1-2 Up',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_312.iloc[i],
                    'target_price': targets_312.iloc[i],
                    'direction': 'bullish',
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        # Process 3-1-2 Down patterns (bearish: directions_312 == -1)
        for i in range(len(entries_312)):
            if entries_312.iloc[i] and directions_312.iloc[i] == -1:
                pattern_date = entries_312.index[i]

                # Market open filter: For hourly patterns, need 3 bars minimum
                if detection_timeframe == '1H':
                    if pattern_date.hour < 11 or (pattern_date.hour == 11 and pattern_date.minute < 30):
                        continue  # Skip patterns before 11:30 AM

                # Entry price = inside bar low (one bar before trigger)
                pattern_loc = detection_data.index.get_loc(pattern_date)
                if pattern_loc < 1:
                    continue  # Need at least 1 bar before trigger
                inside_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[inside_bar_date, 'Low']

                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bearish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                    if not continuity['passes_flexible']:
                        continue
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bearish'
                    )
                    if self.config['filters']['require_full_continuity']:
                        if not continuity['full_continuity']:
                            continue

                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '3-1-2 Down',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_312.iloc[i],
                    'target_price': targets_312.iloc[i],
                    'direction': 'bearish',
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        # Process 2-1-2 Up patterns (bullish: directions_212 == 1)
        for i in range(len(entries_212)):
            if entries_212.iloc[i] and directions_212.iloc[i] == 1:
                pattern_date = entries_212.index[i]

                # Market open filter: For hourly patterns, need 3 bars minimum
                if detection_timeframe == '1H':
                    if pattern_date.hour < 11 or (pattern_date.hour == 11 and pattern_date.minute < 30):
                        continue  # Skip patterns before 11:30 AM

                # Entry price = inside bar high (one bar before trigger)
                pattern_loc = detection_data.index.get_loc(pattern_date)
                if pattern_loc < 1:
                    continue  # Need at least 1 bar before trigger
                inside_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[inside_bar_date, 'High']

                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bullish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                    if not continuity['passes_flexible']:
                        continue
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bullish'
                    )
                    if self.config['filters']['require_full_continuity']:
                        if not continuity['full_continuity']:
                            continue

                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '2-1-2 Up',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_212.iloc[i],
                    'target_price': targets_212.iloc[i],
                    'direction': 'bullish',
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        # Process 2-1-2 Down patterns (bearish: directions_212 == -1)
        for i in range(len(entries_212)):
            if entries_212.iloc[i] and directions_212.iloc[i] == -1:
                pattern_date = entries_212.index[i]

                # Market open filter: For hourly patterns, need 3 bars minimum
                if detection_timeframe == '1H':
                    if pattern_date.hour < 11 or (pattern_date.hour == 11 and pattern_date.minute < 30):
                        continue  # Skip patterns before 11:30 AM

                # Entry price = inside bar low (one bar before trigger)
                pattern_loc = detection_data.index.get_loc(pattern_date)
                if pattern_loc < 1:
                    continue  # Need at least 1 bar before trigger
                inside_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[inside_bar_date, 'Low']

                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bearish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                    if not continuity['passes_flexible']:
                        continue
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bearish'
                    )
                    if self.config['filters']['require_full_continuity']:
                        if not continuity['full_continuity']:
                            continue

                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '2-1-2 Down',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_212.iloc[i],
                    'target_price': targets_212.iloc[i],
                    'direction': 'bearish',
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        # Process 2-2 Up patterns (bullish: directions_22 == 1)
        # 2D-2U: Bearish → Bullish reversal (failed breakdown)
        for i in range(len(entries_22)):
            if entries_22.iloc[i] and directions_22.iloc[i] == 1:
                pattern_date = entries_22.index[i]

                # Market open filter: For hourly patterns, need 2 bars minimum
                # 10:30 (bar 1), 11:30 (bar 2 triggers pattern)
                if detection_timeframe == '1H':
                    if pattern_date.hour < 11 or (pattern_date.hour == 11 and pattern_date.minute < 30):
                        continue  # Skip patterns before 11:30 AM

                # CORRECTED (Session 59): Entry is LIVE when bar breaks previous bar's extreme
                # For 2D-2U: Entry when price breaks ABOVE previous bar (i-1) HIGH
                # All bars open as "1" (open = previous close), entry = previous bar HIGH
                pattern_loc = detection_data.index.get_loc(pattern_date)
                if pattern_loc < 1:
                    continue  # Need previous bar for entry
                prev_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[prev_bar_date, 'High']  # Previous 2D bar HIGH

                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bullish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                    if not continuity['passes_flexible']:
                        continue
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bullish'
                    )
                    if self.config['filters']['require_full_continuity']:
                        if not continuity['full_continuity']:
                            continue

                # Record pattern
                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '2-2 Up',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_22.iloc[i],
                    'target_price': targets_22.iloc[i],
                    'direction': 'bullish',
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        # Process 2-2 Down patterns (bearish: directions_22 == -1)
        # 2U-2D: Bullish → Bearish reversal (failed breakout)
        for i in range(len(entries_22)):
            if entries_22.iloc[i] and directions_22.iloc[i] == -1:
                pattern_date = entries_22.index[i]

                # Market open filter: For hourly patterns, need 2 bars minimum
                if detection_timeframe == '1H':
                    if pattern_date.hour < 11 or (pattern_date.hour == 11 and pattern_date.minute < 30):
                        continue  # Skip patterns before 11:30 AM

                # CORRECTED (Session 59): Entry is LIVE when bar breaks previous bar's extreme
                # For 2U-2D: Entry when price breaks BELOW previous bar (i-1) LOW
                # All bars open as "1" (open = previous close), entry = previous bar LOW
                pattern_loc = detection_data.index.get_loc(pattern_date)
                if pattern_loc < 1:
                    continue  # Need previous bar for entry
                prev_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[prev_bar_date, 'Low']  # Previous 2U bar LOW

                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bearish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                    if not continuity['passes_flexible']:
                        continue
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bearish'
                    )
                    if self.config['filters']['require_full_continuity']:
                        if not continuity['full_continuity']:
                            continue

                # Record pattern
                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '2-2 Down',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_22.iloc[i],
                    'target_price': targets_22.iloc[i],
                    'direction': 'bearish',
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        # Process 3-2 Up patterns (bullish: directions_32 == 1)
        # 3D-2U or 3-2U: Outside bar down → Bullish reversal
        for i in range(len(entries_32)):
            if entries_32.iloc[i] and directions_32.iloc[i] == 1:
                pattern_date = entries_32.index[i]

                # CRITICAL FIX (Session 61): Entry uses PREVIOUS BAR (live entry concept)
                # All bars open as "1" (inside bar, open = previous close)
                # Entry happens when bar i breaks above bar i-1 high
                pattern_loc = detection_data.index.get_loc(pattern_date)

                # Skip if pattern is at first bar (no previous bar for entry)
                if pattern_loc < 1:
                    continue

                prev_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[prev_bar_date, 'High']  # Outside bar high

                # Verify timeframe continuity
                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bullish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bullish'
                    )

                if not continuity.get('passes_flexible', continuity.get('full_continuity', False)):
                    continue

                # Record pattern
                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '3-2 Up',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_32.iloc[i],
                    'target_price': targets_32.iloc[i],
                    'direction': 1,
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        # Process 3-2 Down patterns (bearish: directions_32 == -1)
        # 3U-2D or 3-2D: Outside bar up → Bearish reversal
        for i in range(len(entries_32)):
            if entries_32.iloc[i] and directions_32.iloc[i] == -1:
                pattern_date = entries_32.index[i]

                # CRITICAL FIX (Session 61): Entry uses PREVIOUS BAR (live entry concept)
                pattern_loc = detection_data.index.get_loc(pattern_date)

                # Skip if pattern is at first bar
                if pattern_loc < 1:
                    continue

                prev_bar_date = detection_data.index[pattern_loc - 1]
                entry_price = detection_data.loc[prev_bar_date, 'Low']  # Outside bar low

                # Verify timeframe continuity
                high_dict = {tf: mtf_data[tf]['High'] for tf in continuity_timeframes if tf in mtf_data}
                low_dict = {tf: mtf_data[tf]['Low'] for tf in continuity_timeframes if tf in mtf_data}

                continuity_checker = TimeframeContinuityChecker(timeframes=continuity_timeframes)

                # Use flexible or full continuity based on config
                if self.config['filters'].get('use_flexible_continuity', False):
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date,
                        direction='bearish',
                        min_strength=self.config['filters']['min_continuity_strength'],
                        detection_timeframe=detection_timeframe
                    )
                else:
                    # Legacy full continuity mode
                    continuity = continuity_checker.check_flexible_continuity_at_datetime(
                        high_dict, low_dict, pattern_date, direction='bearish'
                    )

                if not continuity.get('passes_flexible', continuity.get('full_continuity', False)):
                    continue

                # Record pattern
                patterns_found.append({
                    'symbol': symbol,
                    'pattern_type': '3-2 Down',
                    'entry_date': pattern_date,
                    'entry_price': entry_price,
                    'stop_price': stops_32.iloc[i],
                    'target_price': targets_32.iloc[i],
                    'direction': -1,
                    'continuity_strength': continuity['strength'],
                    'full_continuity': continuity.get('full_continuity', continuity.get('passes_flexible', False)),
                    'detection_timeframe': detection_timeframe
                })

        return patterns_found

    def detect_patterns_with_tier1(
        self,
        symbol: str,
        hourly_data: pd.DataFrame,
        detection_timeframe: str,
        continuity_timeframes: List[str]
    ) -> List[Dict]:
        """
        Detect STRAT patterns using Tier1Detector (Session 71 integration).

        Uses Tier1Detector for consistent pattern detection across all modules.
        Continuation bar filter is applied by Tier1Detector internally.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        hourly_data : pd.DataFrame
            Hourly OHLC data (base data for resampling)
        detection_timeframe : str
            Timeframe to detect patterns on ('1H', '1D', '1W', '1M')
        continuity_timeframes : list
            Timeframes for continuity check (NOTE: Tier1Detector doesn't use this,
            but kept for API compatibility)

        Returns:
        --------
        list of dict
            List of pattern occurrences with metadata
        """
        patterns_found = []

        # Resample hourly data to detection timeframe
        if detection_timeframe == '1H':
            detection_data = hourly_data
        else:
            detection_data = hourly_data.resample(detection_timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()

        if len(detection_data) < 10:
            return patterns_found

        # Map detection timeframe to Tier1Detector timeframe
        tf_map = {
            '1H': T1Timeframe.DAILY,  # Use daily for hourly (closest match)
            '1D': T1Timeframe.DAILY,
            '1W': T1Timeframe.WEEKLY,
            '1M': T1Timeframe.MONTHLY
        }
        tier1_tf = tf_map.get(detection_timeframe, T1Timeframe.WEEKLY)

        # Initialize Tier1Detector
        min_cont_bars = self.config['filters'].get('min_continuation_bars', 2)
        include_22_down = self.config['filters'].get('include_22_down', False)

        detector = Tier1Detector(
            min_continuation_bars=min_cont_bars,
            include_22_down=include_22_down
        )

        # Detect patterns using Tier1Detector
        signals = detector.detect_patterns(detection_data, timeframe=tier1_tf)

        # Convert PatternSignal objects to dict format (for compatibility with rest of script)
        for signal in signals:
            # Map PatternType to pattern_type string (comprehensive mapping)
            pattern_type_map = {
                # 3-1-2 patterns
                PatternType.PATTERN_312_UP: '3-1-2 Up',
                PatternType.PATTERN_312_DOWN: '3-1-2 Down',
                # 2-1-2 patterns (legacy simple)
                PatternType.PATTERN_212_UP: '2-1-2 Up',
                PatternType.PATTERN_212_DOWN: '2-1-2 Down',
                # 2-1-2 patterns (detailed variants)
                PatternType.PATTERN_212_2U12U: '2U-1-2U',  # Bullish continuation
                PatternType.PATTERN_212_2D12D: '2D-1-2D',  # Bearish continuation
                PatternType.PATTERN_212_2D12U: '2D-1-2U',  # Bullish reversal
                PatternType.PATTERN_212_2U12D: '2U-1-2D',  # Bearish reversal
                # 2-2 patterns
                PatternType.PATTERN_22_UP: '2-2 Up',
                PatternType.PATTERN_22_DOWN: '2-2 Down',
                # 3-2 patterns
                PatternType.PATTERN_32_UP: '3-2 Up',
                PatternType.PATTERN_32_DOWN: '3-2 Down',
                # 3-2-2 patterns (detailed variants)
                PatternType.PATTERN_322_32U2U: '3-2U-2U',
                PatternType.PATTERN_322_32D2D: '3-2D-2D',
                PatternType.PATTERN_322_32D2U: '3-2D-2U',
                PatternType.PATTERN_322_32U2D: '3-2U-2D',
                # 3-2-2 patterns (legacy)
                PatternType.PATTERN_322_UP: '3-2-2 Up',
                PatternType.PATTERN_322_DOWN: '3-2-2 Down',
            }
            pattern_type_str = pattern_type_map.get(signal.pattern_type, 'Unknown')

            # Determine direction string
            direction_str = 'bullish' if signal.direction == 1 else 'bearish'

            # Market open filter for hourly patterns
            if detection_timeframe == '1H':
                if signal.timestamp.hour < 11 or (signal.timestamp.hour == 11 and signal.timestamp.minute < 30):
                    continue  # Skip patterns before 11:30 AM

            patterns_found.append({
                'symbol': symbol,
                'pattern_type': pattern_type_str,
                'entry_date': signal.timestamp,
                'entry_price': signal.entry_price,
                'stop_price': signal.stop_price,
                'target_price': signal.target_price,
                'direction': direction_str,
                'continuity_strength': 3,  # Tier1Detector doesn't track this, use default
                'full_continuity': True,  # Tier1Detector patterns passed all filters
                'detection_timeframe': detection_timeframe,
                'continuation_bars': signal.continuation_bars,  # Already filtered by Tier1Detector
                'tier1_filtered': True  # Mark as Tier1-filtered pattern
            })

        return patterns_found

    def measure_pattern_outcome(
        self,
        pattern: dict,
        future_data: pd.DataFrame,
        max_holding_bars: int = 30
    ) -> dict:
        """
        Measure whether pattern reached magnitude target and calculate metrics.

        Parameters:
        -----------
        pattern : dict
            Pattern data (entry_price, stop_price, target_price, direction)
        future_data : pd.DataFrame
            OHLC data after pattern trigger (for measuring outcome)
        max_holding_bars : int
            Maximum bars to hold before considering pattern failed

        Returns:
        --------
        dict
            Pattern outcome metrics:
            - magnitude_hit: bool (did price reach target?)
            - bars_to_magnitude: int (bars to reach target, or None if not reached)
            - stop_hit: bool (did price hit stop?)
            - bars_to_stop: int (bars to hit stop, or None)
            - actual_pnl_pct: float (actual P&L if exited at target or stop)
            - continuation_bars: int (count of consecutive 2D/2U bars after entry)
        """
        entry_price = pattern['entry_price']
        stop_price = pattern['stop_price']
        target_price = pattern['target_price']
        direction = pattern['direction']

        magnitude_hit = False
        bars_to_magnitude = None
        stop_hit = False
        bars_to_stop = None

        # Count continuation bars (directional bars after pattern entry in 5-bar window)
        # Session 55 insight: Patterns with 2+ continuation bars show higher hit rates
        # Session 57 fix: Allow inside bars (1.0) without breaking, but break on opposite directional
        # Rationale: Opposite bar = pattern failed, inside bar = consolidation (pattern still valid)
        continuation_bars = 0
        if len(future_data) > 0:
            future_classifications = classify_bars(future_data['High'], future_data['Low'])
            # Scan 5-bar window after entry (or fewer if less data available)
            scan_window = min(5, len(future_classifications))
            for i in range(scan_window):
                bar_classification = future_classifications[i]
                if direction == 'bullish':
                    # Bullish pattern: Count 2U bars (directional up)
                    if bar_classification == 2.0:
                        continuation_bars += 1
                    # Break on reversal bar (2D = -2.0)
                    elif bar_classification == -2.0:
                        break  # Reversal = pattern failed
                    # Break on outside bar (exhaustion signal)
                    elif bar_classification == 3.0:
                        break  # Outside bar = exhaustion
                    # Inside bars (1.0) - continue without counting or breaking
                elif direction == 'bearish':
                    # Bearish pattern: Count 2D bars (directional down)
                    if bar_classification == -2.0:
                        continuation_bars += 1
                    # Break on reversal bar (2U = 2.0)
                    elif bar_classification == 2.0:
                        break  # Reversal = pattern failed
                    # Break on outside bar (exhaustion signal)
                    elif bar_classification == 3.0:
                        break  # Outside bar = exhaustion
                    # Inside bars (1.0) - continue without counting or breaking

        # Track outcomes bar by bar
        for bar_idx in range(min(len(future_data), max_holding_bars)):
            bar_high = future_data['High'].iloc[bar_idx]
            bar_low = future_data['Low'].iloc[bar_idx]

            if direction == 'bullish':
                # Check if target hit (with epsilon tolerance for rounding)
                if bar_high >= (target_price - MAGNITUDE_EPSILON) and not magnitude_hit:
                    magnitude_hit = True
                    bars_to_magnitude = bar_idx

                # Check if stop hit
                if bar_low <= stop_price and not stop_hit:
                    stop_hit = True
                    bars_to_stop = bar_idx

            elif direction == 'bearish':
                # Check if target hit (price falls to target, with epsilon tolerance)
                if bar_low <= (target_price + MAGNITUDE_EPSILON) and not magnitude_hit:
                    magnitude_hit = True
                    bars_to_magnitude = bar_idx

                # Check if stop hit (price rises to stop)
                if bar_high >= stop_price and not stop_hit:
                    stop_hit = True
                    bars_to_stop = bar_idx

            # If both hit, break early
            if magnitude_hit or stop_hit:
                break

        # Calculate actual P&L
        if magnitude_hit:
            # Exited at target
            if direction == 'bullish':
                actual_pnl_pct = (target_price - entry_price) / entry_price * 100
            else:
                actual_pnl_pct = (entry_price - target_price) / entry_price * 100
        elif stop_hit:
            # Exited at stop (loss)
            if direction == 'bullish':
                actual_pnl_pct = (stop_price - entry_price) / entry_price * 100
            else:
                actual_pnl_pct = (entry_price - stop_price) / entry_price * 100
        else:
            # Held until max bars, exit at close
            final_close = future_data['Close'].iloc[min(len(future_data)-1, max_holding_bars-1)]
            if direction == 'bullish':
                actual_pnl_pct = (final_close - entry_price) / entry_price * 100
            else:
                actual_pnl_pct = (entry_price - final_close) / entry_price * 100

        return {
            'magnitude_hit': magnitude_hit,
            'bars_to_magnitude': bars_to_magnitude,
            'stop_hit': stop_hit,
            'bars_to_stop': bars_to_stop,
            'actual_pnl_pct': actual_pnl_pct,
            'continuation_bars': continuation_bars
        }

    def run_validation(self) -> Dict[str, pd.DataFrame]:
        """
        Run full validation backtest across all symbols and detection timeframes.

        Returns:
        --------
        dict
            Dictionary mapping detection_timeframe -> results DataFrame
        """
        print("=== STRAT EQUITY VALIDATION BACKTEST ===")
        print(f"Period: {self.config['backtest_period']['start']} to {self.config['backtest_period']['end']}")
        print(f"Universe: {self.config['stock_universe']['count']} stocks")
        print(f"Patterns: {', '.join(self.config['pattern_types'])}")
        print(f"Filter: Full timeframe continuity {'REQUIRED' if self.config['filters']['require_full_continuity'] else 'OPTIONAL'}")

        # Session 71: Tier1Detector integration status
        use_tier1 = self.config['filters'].get('use_tier1_detector', False)
        if use_tier1:
            print(f"Detection: Tier1Detector (min_continuation_bars={self.config['filters'].get('min_continuation_bars', 2)})")
        else:
            print(f"Detection: Legacy (StratPatternDetector + manual filters)")
        print()

        symbols = self.config['stock_universe']['symbols']
        start_date = self.config['backtest_period']['start']
        end_date = self.config['backtest_period']['end']
        detection_timeframes = self.config['timeframes']['detection']
        continuity_timeframes = self.config['timeframes']['continuity_check']
        # Session 83K-53: Timeframe-specific holding windows
        max_holding_bars_config = self.config['metrics']['max_holding_bars']

        all_results = {}

        # Session 83K-53: Fetch VIX data for entire date range (once for all timeframes)
        try:
            # Add buffer before start for lookback
            vix_start = (pd.Timestamp(start_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            vix_end = (pd.Timestamp(end_date) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            vix_data = fetch_vix_data(vix_start, vix_end)
            print(f"VIX data loaded: {len(vix_data)} days ({vix_data.index.min().date()} to {vix_data.index.max().date()})")
        except Exception as e:
            print(f"WARNING: Could not fetch VIX data: {e}")
            print("Continuing without VIX analysis...")
            vix_data = None

        # Run validation for each detection timeframe
        for detection_tf in detection_timeframes:
            # Session 83K-53: Get timeframe-specific holding window
            if isinstance(max_holding_bars_config, dict):
                max_holding_bars = max_holding_bars_config.get(detection_tf, 30)
            else:
                max_holding_bars = max_holding_bars_config  # Backward compatibility

            print(f"\n{'='*80}")
            print(f"DETECTION TIMEFRAME: {detection_tf} (max_holding_bars: {max_holding_bars})")
            print(f"{'='*80}")

            all_patterns = []

            for symbol_idx, symbol in enumerate(symbols, 1):
                print(f"[{symbol_idx}/{len(symbols)}] Processing {symbol}...", end='')

                # Pull hourly data (base data for resampling)
                hourly_data = self.pull_historical_data(symbol, start_date, end_date, '1H')

                if hourly_data is None or len(hourly_data) < 100:
                    print(f" SKIP (insufficient data)")
                    continue

                # Detect patterns on detection timeframe
                # Session 71: Use Tier1Detector or legacy method based on config
                if use_tier1:
                    patterns = self.detect_patterns_with_tier1(
                        symbol, hourly_data, detection_tf, continuity_timeframes
                    )
                else:
                    patterns = self.detect_patterns_with_continuity(
                        symbol, hourly_data, detection_tf, continuity_timeframes
                    )

                if len(patterns) == 0:
                    print(f" 0 patterns")
                    continue

                # Resample hourly data to detection timeframe for outcome measurement
                if detection_tf == '1H':
                    detection_data = hourly_data
                else:
                    detection_data = hourly_data.resample(detection_tf).agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                    }).dropna()

                # Measure outcomes for each pattern
                for pattern in patterns:
                    entry_date = pattern['entry_date']

                    # Get future data for outcome measurement (same timeframe as detection)
                    future_data = detection_data[detection_data.index > entry_date]

                    if len(future_data) < 5:
                        # Not enough future data to measure outcome
                        continue

                    # Measure outcome
                    outcome = self.measure_pattern_outcome(pattern, future_data, max_holding_bars)

                    # Session 63/71: Apply continuation bar filter if enabled
                    # SKIP if using Tier1Detector (already filtered by detector)
                    if not pattern.get('tier1_filtered', False):
                        if self.config['filters'].get('require_continuation_bars', False):
                            min_cont_bars = self.config['filters'].get('min_continuation_bars', 2)
                            if outcome['continuation_bars'] < min_cont_bars:
                                # Skip patterns with insufficient continuation bars
                                # Session 58 proved: 0-1 bars = 35% hit rate, 2+ bars = 73% hit rate
                                continue

                    # Combine pattern + outcome
                    pattern_with_outcome = {**pattern, **outcome}

                    # Session 83K-53: Add VIX data at entry date
                    if vix_data is not None:
                        entry_date = pattern.get('entry_date')
                        if entry_date is not None:
                            vix_value = get_vix_at_date(vix_data, pd.Timestamp(entry_date))
                            pattern_with_outcome['vix_at_entry'] = vix_value
                            pattern_with_outcome['vix_bucket'] = categorize_vix(vix_value)
                            pattern_with_outcome['vix_bucket_name'] = get_vix_bucket_name(pattern_with_outcome['vix_bucket'])
                        else:
                            pattern_with_outcome['vix_at_entry'] = None
                            pattern_with_outcome['vix_bucket'] = 0
                            pattern_with_outcome['vix_bucket_name'] = 'UNKNOWN'
                    else:
                        pattern_with_outcome['vix_at_entry'] = None
                        pattern_with_outcome['vix_bucket'] = 0
                        pattern_with_outcome['vix_bucket_name'] = 'UNKNOWN'

                    all_patterns.append(pattern_with_outcome)

                print(f" {len(patterns)} patterns")

            # Convert to DataFrame
            results_df = pd.DataFrame(all_patterns)

            if len(results_df) == 0:
                print(f"\nWARNING: No patterns found on {detection_tf} timeframe!")
                all_results[detection_tf] = results_df
                continue

            # Save results to CSV
            output_file = f'scripts/strat_validation_{detection_tf}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")

            # Calculate summary statistics for this timeframe
            self.calculate_summary_stats(results_df, detection_tf)

            all_results[detection_tf] = results_df

        return all_results

    def calculate_summary_stats(self, results_df: pd.DataFrame, detection_timeframe: str = '1H'):
        """
        Calculate and print summary statistics for a specific detection timeframe.

        Parameters:
        -----------
        results_df : pd.DataFrame
            Full results DataFrame
        detection_timeframe : str
            Timeframe patterns were detected on
        """
        print(f"\n=== VALIDATION RESULTS ({detection_timeframe}) ===")

        # Overall statistics
        total_patterns = len(results_df)
        magnitude_hits = results_df['magnitude_hit'].sum()
        magnitude_hit_rate = magnitude_hits / total_patterns if total_patterns > 0 else 0

        print(f"\nTotal Patterns: {total_patterns}")
        print(f"Magnitude Hits: {magnitude_hits} ({magnitude_hit_rate:.1%})")

        # By pattern type
        print("\nBy Pattern Type:")
        for pattern_type in results_df['pattern_type'].unique():
            subset = results_df[results_df['pattern_type'] == pattern_type]
            hits = subset['magnitude_hit'].sum()
            hit_rate = hits / len(subset) if len(subset) > 0 else 0
            print(f"  {pattern_type}: {hits}/{len(subset)} ({hit_rate:.1%})")

        # Bars to magnitude (for hits only)
        hits_only = results_df[results_df['magnitude_hit'] == True]
        if len(hits_only) > 0:
            print("\nBars to Magnitude (for hits):")
            print(f"  Median: {hits_only['bars_to_magnitude'].median():.1f} bars")
            print(f"  75th percentile: {hits_only['bars_to_magnitude'].quantile(0.75):.1f} bars")
            print(f"  90th percentile: {hits_only['bars_to_magnitude'].quantile(0.90):.1f} bars")

        # Risk-reward analysis
        wins = results_df[results_df['magnitude_hit'] == True]
        losses = results_df[results_df['stop_hit'] == True]

        if len(wins) > 0 and len(losses) > 0:
            avg_win_pct = wins['actual_pnl_pct'].mean()
            avg_loss_pct = losses['actual_pnl_pct'].mean()
            risk_reward = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0

            print(f"\nRisk-Reward:")
            print(f"  Average Win: {avg_win_pct:.2f}%")
            print(f"  Average Loss: {avg_loss_pct:.2f}%")
            print(f"  Risk-Reward Ratio: {risk_reward:.2f}:1")

        # GO/NO-GO Decision
        print("\n=== GO/NO-GO DECISION ===")

        go_criteria = []
        no_go_criteria = []

        # Criterion 1: Magnitude hit rate >= 60%
        if magnitude_hit_rate >= self.config['metrics']['magnitude_hit_threshold']:
            go_criteria.append(f"Magnitude hit rate {magnitude_hit_rate:.1%} >= 60%")
        else:
            no_go_criteria.append(f"Magnitude hit rate {magnitude_hit_rate:.1%} < 60% (FAIL)")

        # Criterion 2: Sufficient pattern count >= 100
        if total_patterns >= self.config['metrics']['min_pattern_count']:
            go_criteria.append(f"Pattern count {total_patterns} >= 100")
        else:
            no_go_criteria.append(f"Pattern count {total_patterns} < 100 (INSUFFICIENT DATA)")

        # Criterion 3: Risk-reward >= 2:1
        if len(wins) > 0 and len(losses) > 0:
            if risk_reward >= self.config['metrics']['risk_reward_threshold']:
                go_criteria.append(f"Risk-reward {risk_reward:.2f}:1 >= 2:1")
            else:
                no_go_criteria.append(f"Risk-reward {risk_reward:.2f}:1 < 2:1 (POOR EXPECTANCY)")

        # Print decision
        if len(no_go_criteria) == 0:
            print("DECISION: GO - Proceed to Options Module Implementation")
            print("\nCriteria Met:")
            for criterion in go_criteria:
                print(f"  - {criterion}")
        else:
            print("DECISION: NO-GO - Debug patterns before Options Module")
            print("\nFailed Criteria:")
            for criterion in no_go_criteria:
                print(f"  - {criterion}")
            if len(go_criteria) > 0:
                print("\nPassed Criteria:")
                for criterion in go_criteria:
                    print(f"  - {criterion}")

        print("\n" + "="*60)


def main():
    """
    Run STRAT equity validation backtest.

    Session 83K-53: Added CLI support for expanded universe.
    Session 83K-83: Added CLI support for timeframe filtering.

    Usage:
        python scripts/backtest_strat_equity_validation.py                    # Default 50 stocks
        python scripts/backtest_strat_equity_validation.py --universe expanded  # 16 ETFs/mega caps
        python scripts/backtest_strat_equity_validation.py --universe index     # 4 index ETFs only
        python scripts/backtest_strat_equity_validation.py --timeframes 1D,1W,1M  # Non-hourly only
    """
    import argparse

    parser = argparse.ArgumentParser(description='STRAT Equity Validation Backtest')
    parser.add_argument(
        '--universe',
        type=str,
        choices=['default', 'expanded', 'index', 'sector'],
        default='default',
        help='Symbol universe: default (50 stocks), expanded (16 ETFs/mega caps), index (4 ETFs), sector (4 sector ETFs)'
    )
    parser.add_argument(
        '--timeframes',
        type=str,
        default=None,
        help='Comma-separated timeframes to run (e.g., "1D,1W,1M"). Default: all (1H,1D,1W,1M)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Comma-separated symbols to run (e.g., "SPY,QQQ"). Overrides --universe'
    )
    args = parser.parse_args()

    # Select symbols based on --symbols or --universe choice
    if args.symbols:
        # Custom symbols override everything
        symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"[INFO] Using CUSTOM symbols: {symbols}")
    elif args.universe == 'expanded':
        symbols = EXPANDED_SYMBOLS
        print(f"[INFO] Using EXPANDED universe: {len(symbols)} symbols")
    elif args.universe == 'index':
        symbols = TICKER_CATEGORIES['index_etf']
        print(f"[INFO] Using INDEX universe: {symbols}")
    elif args.universe == 'sector':
        symbols = TICKER_CATEGORIES['sector_etf']
        print(f"[INFO] Using SECTOR universe: {symbols}")
    else:
        symbols = None  # Use default from VALIDATION_CONFIG
        print("[INFO] Using DEFAULT universe (50 stocks)")

    # Create config with selected symbols and timeframes
    config = VALIDATION_CONFIG.copy()

    if symbols:
        config['stock_universe'] = {
            'symbols': symbols,
            'count': len(symbols),
        }

    # Handle timeframes filter
    if args.timeframes:
        timeframes_list = [tf.strip() for tf in args.timeframes.split(',')]
        valid_tfs = ['1H', '1D', '1W', '1M']
        for tf in timeframes_list:
            if tf not in valid_tfs:
                print(f"WARNING: Invalid timeframe '{tf}'. Valid options: {valid_tfs}")
                return None
        config['timeframes'] = config['timeframes'].copy()
        config['timeframes']['detection'] = timeframes_list
        print(f"[INFO] Using TIMEFRAMES: {timeframes_list}")

    backtest = EquityValidationBacktest(config)
    results = backtest.run_validation()

    return results


if __name__ == '__main__':
    main()
