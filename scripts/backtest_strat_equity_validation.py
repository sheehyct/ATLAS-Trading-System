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
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strat.bar_classifier import StratBarClassifier, classify_bars
from strat.pattern_detector import StratPatternDetector
from strat.timeframe_continuity import TimeframeContinuityChecker, get_continuity_strength
# ATLAS integration not used in initial validation (testing patterns alone first)
# from strat.atlas_integration import filter_strat_signals


# Magnitude hit detection tolerance (prevent floating point rounding errors)
# Example: target=196.37 but bar_high=196.369999 should still count as hit
MAGNITUDE_EPSILON = 0.01  # 1 cent tolerance


# Configuration
VALIDATION_CONFIG = {
    'backtest_period': {
        'start': '2020-01-01',
        'end': '2025-01-01',
        'rationale': '5 years including March 2020 crash for stress testing'
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
        'continuity_check': ['1M', '1W', '1D', '4H', '1H'],
        'rationale': 'Multi-timeframe pattern detection + alignment for high-conviction signals'
    },
    'pattern_types': ['3-1-2 Up', '3-1-2 Down', '2-1-2 Up', '2-1-2 Down'],
    'filters': {
        'require_full_continuity': False,  # Changed to flexible (Session 56 Bug #3 fix)
        'use_flexible_continuity': True,   # Use timeframe-appropriate continuity
        'min_continuity_strength': 3,      # Require 3/5 timeframes aligned minimum
        'use_atlas_regime': False,  # First test patterns alone, then with ATLAS
        'min_pattern_quality': 'HIGH'
    },
    'metrics': {
        'magnitude_window': 5,  # Bars to check for magnitude hit (patterns can take 1-5 bars)
        'max_holding_bars': 30,  # Maximum bars to hold before considering failed
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
        Pull historical OHLC data for a symbol.

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
            data = vbt.YFData.pull(
                symbol,
                start=start,
                end=end,
                timeframe=timeframe,
                tz='America/New_York'
            )

            # Get OHLC DataFrame
            ohlc = data.get()

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

        # Count continuation bars (consecutive directional bars after pattern entry)
        # Session 55 insight: Patterns with 2+ continuation bars show higher hit rates
        continuation_bars = 0
        if len(future_data) > 0:
            future_classifications = classify_bars(future_data['High'], future_data['Low'])
            for bar_classification in future_classifications:
                if direction == 'bullish':
                    # Bullish pattern: Count consecutive 2U bars (classification == 2.0)
                    if bar_classification == 2.0:
                        continuation_bars += 1
                    else:
                        break  # Stop on inside bar (1) or opposite bar (2D)
                elif direction == 'bearish':
                    # Bearish pattern: Count consecutive 2D bars (classification == -2.0)
                    if bar_classification == -2.0:
                        continuation_bars += 1
                    else:
                        break  # Stop on inside bar (1) or opposite bar (2U)

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
        print()

        symbols = self.config['stock_universe']['symbols']
        start_date = self.config['backtest_period']['start']
        end_date = self.config['backtest_period']['end']
        detection_timeframes = self.config['timeframes']['detection']
        continuity_timeframes = self.config['timeframes']['continuity_check']
        max_holding_bars = self.config['metrics']['max_holding_bars']

        all_results = {}

        # Run validation for each detection timeframe
        for detection_tf in detection_timeframes:
            print(f"\n{'='*80}")
            print(f"DETECTION TIMEFRAME: {detection_tf}")
            print(f"{'='*80}")

            all_patterns = []

            for symbol_idx, symbol in enumerate(symbols, 1):
                print(f"[{symbol_idx}/{len(symbols)}] Processing {symbol}...", end='')

                # Pull hourly data (base data for resampling)
                hourly_data = self.pull_historical_data(symbol, start_date, end_date, '1H')

                if hourly_data is None or len(hourly_data) < 100:
                    print(f" SKIP (insufficient data)")
                    continue

                # Detect patterns on detection timeframe with continuity filtering
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

                    # Combine pattern + outcome
                    pattern_with_outcome = {**pattern, **outcome}
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
    """Run STRAT equity validation backtest."""
    backtest = EquityValidationBacktest()
    results = backtest.run_validation()

    return results


if __name__ == '__main__':
    main()
