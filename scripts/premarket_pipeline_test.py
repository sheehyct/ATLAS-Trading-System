#!/usr/bin/env python3
"""
Premarket Pipeline Test Script

Tests the full signal pipeline:
1. Fetch data from Alpaca (including premarket)
2. Run bar classification
3. Detect patterns (COMPLETED and SETUP)
4. Send Discord alerts for detected patterns
5. Show what entries would trigger

Usage:
    uv run python scripts/premarket_pipeline_test.py [--dry-run]

    --dry-run: Analyze only, don't send Discord alerts
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strat.paper_signal_scanner import PaperSignalScanner, DetectedSignal
from strat.bar_classifier import classify_bars_nb
from strat.signal_automation.signal_store import SignalStore, StoredSignal
from strat.signal_automation.alerters.discord_alerter import DiscordAlerter

# Load environment variables from .env file if present
from dotenv import load_dotenv
load_dotenv()


# Test configuration
TEST_SYMBOLS = ['SPY', 'QQQ', 'TSLA']
TEST_TIMEFRAMES = ['15m', '30m', '1H']  # Faster timeframes for more signals

BAR_TYPE_MAP = {
    1: '1 (Inside)',
    2: '2U (Up)',
    -2: '2D (Down)',
    3: '3 (Outside)'
}


def fetch_and_classify(scanner: PaperSignalScanner, symbol: str, timeframe: str, bars: int = 20) -> Optional[pd.DataFrame]:
    """Fetch data and add bar classification."""
    try:
        # Map timeframe for Alpaca
        tf_map = {'15m': '15Min', '30m': '30Min', '1H': '1H', '1D': '1D'}
        alpaca_tf = tf_map.get(timeframe, timeframe)

        df = scanner._fetch_data(symbol, alpaca_tf, bars)
        if df is None or len(df) < 4:
            return None

        # Classify bars
        h = df['High'].values.astype(np.float64)
        l = df['Low'].values.astype(np.float64)
        classes = classify_bars_nb(h, l)
        df['bar_type'] = classes
        df['bar_type_str'] = [BAR_TYPE_MAP.get(c, str(c)) for c in classes]

        return df
    except Exception as e:
        print(f"  Error fetching {symbol} {timeframe}: {e}")
        return None


def analyze_patterns(df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
    """Analyze patterns in the data."""
    if df is None or len(df) < 4:
        return {'error': 'Insufficient data'}

    classes = df['bar_type'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    # Get last 3 bars for pattern analysis
    last_3 = classes[-3:]
    pattern_seq = '-'.join([BAR_TYPE_MAP.get(c, '?').split()[0] for c in last_3])

    # Current bar info
    current_bar = {
        'type': BAR_TYPE_MAP.get(classes[-1], str(classes[-1])),
        'high': highs[-1],
        'low': lows[-1],
        'close': closes[-1],
        'time': df.index[-1]
    }

    # Check for SETUP patterns (ending in inside bar or directional bar)
    setups = []

    # X-1-? SETUP (inside bar waiting for break)
    if classes[-1] == 1:  # Current bar is inside
        prev_type = BAR_TYPE_MAP.get(classes[-2], '?').split()[0]
        setups.append({
            'pattern': f'{prev_type}-1-?',
            'type': 'SETUP',
            'call_trigger': highs[-2],  # Break above inside bar's parent high
            'put_trigger': lows[-2],    # Break below inside bar's parent low
            'description': 'Inside bar - waiting for directional break'
        })

    # X-2D-? SETUP (down bar waiting for opposite break)
    if classes[-1] == -2:  # Current bar is 2D
        prev_type = BAR_TYPE_MAP.get(classes[-2], '?').split()[0]
        setups.append({
            'pattern': f'{prev_type}-2D-?',
            'type': 'SETUP',
            'call_trigger': highs[-1],  # Break above 2D bar high = 2U
            'put_trigger': None,        # Already went down
            'description': '2D bar - watching for 2U reversal'
        })

    # X-2U-? SETUP (up bar waiting for opposite break)
    if classes[-1] == 2:  # Current bar is 2U
        prev_type = BAR_TYPE_MAP.get(classes[-2], '?').split()[0]
        setups.append({
            'pattern': f'{prev_type}-2U-?',
            'type': 'SETUP',
            'call_trigger': None,       # Already went up
            'put_trigger': lows[-1],    # Break below 2U bar low = 2D
            'description': '2U bar - watching for 2D reversal'
        })

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'pattern_sequence': pattern_seq,
        'current_bar': current_bar,
        'setups': setups,
        'last_5_bars': [(df.index[i].strftime('%H:%M'), BAR_TYPE_MAP.get(classes[i], '?')) for i in range(-5, 0)]
    }


def run_scanner_test(scanner: PaperSignalScanner, symbol: str, timeframe: str) -> List[DetectedSignal]:
    """Run the actual scanner to detect signals."""
    try:
        # Map timeframe
        tf_map = {'15m': '15Min', '30m': '30Min', '1H': '1H'}
        scanner_tf = tf_map.get(timeframe, timeframe)

        signals = scanner.scan_symbol_timeframe(symbol, scanner_tf)
        return signals
    except Exception as e:
        print(f"  Scanner error for {symbol} {timeframe}: {e}")
        return []


def send_test_alert(alerter: DiscordAlerter, signal: DetectedSignal) -> bool:
    """Send a test Discord alert for a detected signal."""
    try:
        # Convert DetectedSignal to StoredSignal for alerter
        stored = StoredSignal(
            signal_key=f"TEST_{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_{datetime.now().strftime('%Y%m%d%H%M')}",
            pattern_type=signal.pattern_type,
            direction=signal.direction,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            detected_time=datetime.now(),
            entry_trigger=getattr(signal, 'trigger_high', 0) or getattr(signal, 'trigger_low', 0),
            stop_price=getattr(signal, 'stop_price', 0),
            target_price=getattr(signal, 'target_price', 0),
            magnitude_pct=getattr(signal, 'magnitude_pct', 0),
            risk_reward=getattr(signal, 'risk_reward', 0),
            signal_type=signal.signal_type,
            status='DETECTED'
        )

        alerter.send_signal_alert(stored)
        return True
    except Exception as e:
        print(f"  Alert error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Premarket Pipeline Test')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, no Discord alerts')
    args = parser.parse_args()

    print("=" * 80)
    print("PREMARKET PIPELINE TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN (no alerts)' if args.dry_run else 'FULL TEST (will send Discord alerts)'}")
    print("=" * 80)
    print()

    # Initialize components
    scanner = PaperSignalScanner()
    alerter = None

    if not args.dry_run:
        try:
            webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
            if not webhook_url:
                raise ValueError("DISCORD_WEBHOOK_URL not set in environment or .env file")
            alerter = DiscordAlerter(webhook_url=webhook_url)
            print(f"Discord alerter initialized (webhook configured)")
        except Exception as e:
            print(f"Warning: Could not initialize Discord alerter: {e}")
            print("Continuing in dry-run mode")
            args.dry_run = True

    all_results = []
    all_signals = []

    # Analyze each symbol/timeframe combination
    for symbol in TEST_SYMBOLS:
        print(f"\n{'='*40}")
        print(f"ANALYZING: {symbol}")
        print(f"{'='*40}")

        for timeframe in TEST_TIMEFRAMES:
            print(f"\n  [{timeframe}]")

            # Fetch and classify
            df = fetch_and_classify(scanner, symbol, timeframe)
            if df is None:
                print(f"    No data available")
                continue

            # Analyze patterns
            analysis = analyze_patterns(df, symbol, timeframe)
            all_results.append(analysis)

            # Show last 5 bars
            print(f"    Last 5 bars:")
            for time_str, bar_type in analysis['last_5_bars']:
                print(f"      {time_str}: {bar_type}")

            print(f"    Pattern: {analysis['pattern_sequence']}")
            print(f"    Current: {analysis['current_bar']['type']} @ {analysis['current_bar']['close']:.2f}")

            # Show setups
            if analysis['setups']:
                print(f"    SETUPS DETECTED:")
                for setup in analysis['setups']:
                    print(f"      {setup['pattern']} ({setup['type']})")
                    if setup['call_trigger']:
                        print(f"        CALL trigger: > {setup['call_trigger']:.2f}")
                    if setup['put_trigger']:
                        print(f"        PUT trigger: < {setup['put_trigger']:.2f}")

            # Run actual scanner
            print(f"    Running scanner...")
            signals = run_scanner_test(scanner, symbol, timeframe)

            if signals:
                print(f"    SCANNER FOUND {len(signals)} SIGNALS:")
                for sig in signals:
                    print(f"      {sig.pattern_type} {sig.direction} ({sig.signal_type})")
                    all_signals.append(sig)

                    # Send alert if not dry run
                    if not args.dry_run and alerter:
                        print(f"      Sending Discord alert...")
                        if send_test_alert(alerter, sig):
                            print(f"      Alert sent!")
                        else:
                            print(f"      Alert failed!")
            else:
                print(f"    No signals from scanner")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Symbols analyzed: {', '.join(TEST_SYMBOLS)}")
    print(f"Timeframes: {', '.join(TEST_TIMEFRAMES)}")
    print(f"Total signals detected: {len(all_signals)}")

    if all_signals:
        print("\nAll detected signals:")
        for sig in all_signals:
            print(f"  {sig.symbol} {sig.timeframe} {sig.pattern_type} {sig.direction} ({sig.signal_type})")

    if not args.dry_run:
        print(f"\nDiscord alerts sent: {len(all_signals)}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return len(all_signals)


if __name__ == '__main__':
    sys.exit(0 if main() >= 0 else 1)
