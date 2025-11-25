#!/usr/bin/env python3
"""
Paper Trading Script for Tier 1 STRAT Options.

Session 70: Implements paper trading workflow for validated patterns.

Features:
- Scans multiple underlyings for weekly Tier 1 patterns
- Generates options trade signals with proper strike selection
- Tracks paper trades and performance
- Saves results for analysis

TIER 1 PATTERNS (Session 69 validated):
- 2-1-2 Up/Down @ 1W (80.7% win, 563.6% expectancy)
- 2-2 Up (2D-2U) @ 1W (86.2% win, 409.5% expectancy)
- 3-1-2 Up/Down @ 1W (72.7% win, 462.7% expectancy)

Usage:
    # Run weekly scan
    python scripts/paper_trade_options.py

    # Scan specific symbols
    python scripts/paper_trade_options.py --symbols SPY QQQ AAPL

    # Generate CSV report
    python scripts/paper_trade_options.py --output trades.csv
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import vectorbtpro as vbt

from config.settings import get_alpaca_credentials, get_tiingo_key
from strat.tier1_detector import Tier1Detector, Timeframe, PatternSignal
from strat.options_module import (
    OptionsExecutor,
    OptionContract,
    OptionType,
    OptionTrade,
)
from integrations.tiingo_data_fetcher import TiingoDataFetcher


# Default watchlist for paper trading
DEFAULT_SYMBOLS = [
    'SPY',   # S&P 500 ETF
    'QQQ',   # NASDAQ 100 ETF
    'IWM',   # Russell 2000 ETF
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'NVDA',  # NVIDIA
    'TSLA',  # Tesla
    'AMZN',  # Amazon
]

# Paper trading settings
PAPER_TRADE_CONFIG = {
    'capital_per_trade': 500,      # Max $ per trade
    'max_positions': 5,            # Max concurrent positions
    'min_continuation_bars': 2,    # Required continuation bars
    'default_dte_weekly': 35,      # Days to expiration
    'lookback_weeks': 52,          # Weeks of data for pattern detection
}


class PaperTradingSession:
    """
    Paper trading session for Tier 1 options.

    Tracks:
    - Active signals detected
    - Generated option trades
    - Paper positions (simulated)
    - Performance metrics
    """

    def __init__(
        self,
        symbols: list = None,
        output_dir: str = 'results/paper_trades'
    ):
        """
        Initialize paper trading session.

        Args:
            symbols: List of symbols to scan (default: DEFAULT_SYMBOLS)
            output_dir: Directory for saving results
        """
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.detector = Tier1Detector(
            min_continuation_bars=PAPER_TRADE_CONFIG['min_continuation_bars']
        )
        self.executor = OptionsExecutor(
            default_dte_weekly=PAPER_TRADE_CONFIG['default_dte_weekly']
        )
        self.fetcher = TiingoDataFetcher()

        # Session data
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.signals = []
        self.trades = []
        self.prices = {}

    def run_scan(self) -> pd.DataFrame:
        """
        Run weekly scan for Tier 1 patterns across all symbols.

        Returns:
            DataFrame with detected signals and generated trades
        """
        print('='*60)
        print(f'Paper Trading Scan - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print('='*60)
        print(f'Symbols: {", ".join(self.symbols)}')
        print(f'Lookback: {PAPER_TRADE_CONFIG["lookback_weeks"]} weeks')
        print()

        all_trades = []

        for symbol in self.symbols:
            print(f'\n[SCANNING] {symbol}...')

            try:
                # Fetch weekly data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(weeks=PAPER_TRADE_CONFIG['lookback_weeks'])).strftime('%Y-%m-%d')

                data = self.fetcher.fetch(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1W'
                )
                df = data.get()

                if df.empty:
                    print(f'  No data available')
                    continue

                # Store current price
                current_price = df['Close'].iloc[-1]
                self.prices[symbol] = current_price
                print(f'  Current price: ${current_price:.2f}')
                print(f'  Data bars: {len(df)}')

                # Detect patterns
                signals = self.detector.detect_patterns(df, timeframe=Timeframe.WEEKLY)

                # Filter to recent signals (last 2 weeks)
                recent_cutoff = df.index[-1] - timedelta(weeks=2)
                recent_signals = [s for s in signals if s.timestamp >= recent_cutoff]

                if recent_signals:
                    print(f'  PATTERNS FOUND: {len(recent_signals)}')

                    # Generate option trades
                    trades = self.executor.generate_option_trades(
                        recent_signals,
                        underlying=symbol,
                        underlying_price=current_price,
                        capital_per_trade=PAPER_TRADE_CONFIG['capital_per_trade']
                    )

                    for trade in trades:
                        print(f'    {trade.pattern_signal.pattern_type.value}:')
                        print(f'      Entry: ${trade.entry_trigger:.2f}, '
                              f'Target: ${trade.target_exit:.2f}, '
                              f'Stop: ${trade.stop_exit:.2f}')
                        print(f'      Option: {trade.contract.osi_symbol}')
                        print(f'      R:R: {trade.pattern_signal.risk_reward:.2f}')

                    self.signals.extend(recent_signals)
                    self.trades.extend(trades)
                    all_trades.extend(trades)
                else:
                    print(f'  No recent patterns')

            except Exception as e:
                print(f'  Error: {e}')
                continue

        # Generate summary
        print('\n' + '='*60)
        print('SCAN SUMMARY')
        print('='*60)
        print(f'Total signals: {len(self.signals)}')
        print(f'Total trades: {len(self.trades)}')

        if self.trades:
            df = self.executor.trades_to_dataframe(self.trades)
            print(f'\nBy pattern type:')
            print(df['pattern_type'].value_counts().to_string())

            print(f'\nBy underlying:')
            print(df['underlying'].value_counts().to_string())

            return df

        return pd.DataFrame()

    def save_results(self, df: pd.DataFrame):
        """
        Save scan results to files.

        Args:
            df: DataFrame with trades
        """
        if df.empty:
            print('\nNo trades to save')
            return

        # Save CSV
        csv_path = self.output_dir / f'trades_{self.session_id}.csv'
        df.to_csv(csv_path, index=False)
        print(f'\nSaved trades to: {csv_path}')

        # Save JSON summary
        summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'symbols_scanned': self.symbols,
            'total_signals': len(self.signals),
            'total_trades': len(self.trades),
            'by_pattern_type': df['pattern_type'].value_counts().to_dict(),
            'by_underlying': df['underlying'].value_counts().to_dict(),
            'avg_risk_reward': df['risk_reward'].mean(),
        }

        json_path = self.output_dir / f'summary_{self.session_id}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved summary to: {json_path}')

    def fetch_option_quotes(self, trades: list = None) -> pd.DataFrame:
        """
        Fetch live option quotes for generated trades.

        Args:
            trades: List of trades (default: session trades)

        Returns:
            DataFrame with option quotes
        """
        trades = trades or self.trades

        if not trades:
            print('No trades to fetch quotes for')
            return pd.DataFrame()

        print('\nFetching option quotes...')

        quotes = []
        for trade in trades:
            try:
                quote_data = self.executor.fetch_option_quote(trade.contract)
                if quote_data is not None and not quote_data.empty:
                    last_price = quote_data['Close'].iloc[-1]
                    quotes.append({
                        'osi_symbol': trade.contract.osi_symbol,
                        'underlying': trade.contract.underlying,
                        'strike': trade.contract.strike,
                        'last_price': last_price,
                        'volume': quote_data['Volume'].iloc[-1] if 'Volume' in quote_data else 0,
                    })
                    print(f'  {trade.contract.osi_symbol}: ${last_price:.2f}')
            except Exception as e:
                print(f'  {trade.contract.osi_symbol}: Error - {e}')

        return pd.DataFrame(quotes)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Paper Trade Tier 1 STRAT Options')
    parser.add_argument('--symbols', nargs='+', help='Symbols to scan')
    parser.add_argument('--output', help='Output CSV path')
    parser.add_argument('--quotes', action='store_true', help='Fetch live option quotes')
    args = parser.parse_args()

    # Initialize session
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    session = PaperTradingSession(symbols=symbols)

    # Run scan
    df = session.run_scan()

    # Save results
    session.save_results(df)

    # Fetch quotes if requested
    if args.quotes and not df.empty:
        quotes_df = session.fetch_option_quotes()
        if not quotes_df.empty:
            quotes_path = session.output_dir / f'quotes_{session.session_id}.csv'
            quotes_df.to_csv(quotes_path, index=False)
            print(f'Saved quotes to: {quotes_path}')

    # Custom output path
    if args.output and not df.empty:
        df.to_csv(args.output, index=False)
        print(f'Saved to custom path: {args.output}')

    print('\nDone!')


if __name__ == '__main__':
    main()
