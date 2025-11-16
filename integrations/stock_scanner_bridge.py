"""
Stock Scanner Bridge for ATLAS System

Connects the momentum stock scanner (C:/Dev/strat-stock-scanner) with the
ATLAS vectorbt backtesting system for multi-asset portfolio strategies.

This bridge allows:
1. Using scanner's stock universe and momentum detection
2. Historical backtesting with VectorBT Pro
3. Semi-annual rebalance simulation
4. Integration with ATLAS regime system

Usage:
    from integrations.stock_scanner_bridge import MomentumPortfolioBacktest

    backtest = MomentumPortfolioBacktest(
        universe='technology',
        top_n=10,
        volume_threshold=1.25
    )

    results = backtest.run(
        start_date='2015-01-01',
        end_date='2025-01-01',
        initial_capital=100000
    )
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import vectorbtpro as vbt

# Add scanner directory to path
SCANNER_PATH = Path("C:/Dev/strat-stock-scanner")
if SCANNER_PATH.exists():
    sys.path.insert(0, str(SCANNER_PATH))

try:
    from momentum_52w_detector import Momentum52WDetector, MomentumSignal
except ImportError:
    print("WARNING: Could not import momentum_52w_detector from scanner.")
    print("Make sure C:/Dev/strat-stock-scanner exists with momentum_52w_detector.py")


class MomentumPortfolioBacktest:
    """
    Multi-asset backtest using 52-week high momentum selection.

    Implements George & Hwang (2004) portfolio methodology:
    - Semi-annual rebalance (February, August)
    - Equal-weight top N stocks near 52-week highs
    - Optional volume confirmation
    - Compatible with ATLAS regime system
    """

    # Stock universes (from scanner)
    UNIVERSES = {
        'technology': [
            "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "AMD", "CRM",
            "ADBE", "NFLX", "INTC", "CSCO", "ACN", "IBM", "NOW", "QCOM", "TXN", "INTU",
            "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "PANW", "CRWD"
        ],
        'sp500_proxy': [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "LLY", "TSLA",
            "JPM", "WMT", "V", "XOM", "UNH", "MA", "PG", "COST", "JNJ", "HD",
            "NFLX", "BAC", "ABBV", "CRM", "MRK", "KO", "CVX", "AMD", "PEP", "ORCL",
            "TMO", "ACN", "MCD", "CSCO", "ABT", "DHR", "DIS", "ADBE", "WFC", "TXN"
        ],
        'healthcare': [
            "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
            "AMGN", "CVS", "MDT", "GILD", "CI", "REGN", "SYK", "VRTX", "ZTS", "HUM"
        ],
        'financials': [
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "CB", "AXP",
            "PNC", "USB", "TFC", "COF", "BK", "AIG", "MET", "AFL", "PRU", "ALL"
        ]
    }

    def __init__(
        self,
        universe: str = 'sp500_proxy',
        top_n: int = 20,
        volume_threshold: Optional[float] = None,
        min_distance: float = 0.90,
        rebalance_frequency: str = 'semi_annual'
    ):
        """
        Initialize momentum portfolio backtest.

        Args:
            universe: Stock universe to scan ('technology', 'sp500_proxy', etc.)
            top_n: Number of stocks to hold in portfolio
            volume_threshold: Volume filter (None = disabled, 1.25 = standard)
            min_distance: Minimum distance from 52w high (default 0.90)
            rebalance_frequency: 'semi_annual' (Feb/Aug) or 'quarterly'
        """
        self.universe_name = universe
        self.universe = self.UNIVERSES.get(universe, self.UNIVERSES['sp500_proxy'])
        self.top_n = top_n
        self.volume_threshold = volume_threshold
        self.min_distance = min_distance
        self.rebalance_frequency = rebalance_frequency

    def get_rebalance_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        Generate rebalance dates.

        Args:
            start_date: Backtest start (YYYY-MM-DD)
            end_date: Backtest end (YYYY-MM-DD)

        Returns:
            List of rebalance dates
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if self.rebalance_frequency == 'semi_annual':
            # Generate February 1 and August 1 for each year
            dates = []
            for year in range(start.year, end.year + 1):
                for month in [2, 8]:
                    date = pd.Timestamp(year=year, month=month, day=1)
                    if start <= date <= end:
                        dates.append(date)
        elif self.rebalance_frequency == 'quarterly':
            dates = pd.date_range(start, end, freq='QS')
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_frequency}")

        return [d.strftime('%Y-%m-%d') for d in dates]

    def select_portfolio_at_date(
        self,
        data,  # VBT YFData object
        rebalance_date: str
    ) -> List[str]:
        """
        Select top N stocks at rebalance date using momentum criteria.

        Args:
            data: VBT YFData object with OHLCV data
            rebalance_date: Date to select portfolio

        Returns:
            List of selected tickers
        """
        # Get OHLCV data from VBT object
        open_df = data.get('Open')
        high_df = data.get('High')
        low_df = data.get('Low')
        close_df = data.get('Close')
        volume_df = data.get('Volume')

        # Convert rebalance date to timezone-aware (match data index timezone)
        rebalance_ts = pd.Timestamp(rebalance_date)
        if close_df.index.tz is not None:
            rebalance_ts = rebalance_ts.tz_localize(close_df.index.tz)

        # Get historical data up to rebalance date
        historical_close = close_df.loc[:rebalance_ts]

        if len(historical_close) < 252:
            # Not enough history, return empty
            return []

        # Analyze each stock
        signals = []

        for symbol in self.universe:
            if symbol not in close_df.columns:
                continue

            # Extract bars for this symbol (last 252 days)
            symbol_close = historical_close[symbol].tail(252)

            if len(symbol_close) < 252:
                continue

            # Get corresponding OHLCV data
            symbol_open = open_df.loc[symbol_close.index, symbol]
            symbol_high = high_df.loc[symbol_close.index, symbol]
            symbol_low = low_df.loc[symbol_close.index, symbol]
            symbol_volume = volume_df.loc[symbol_close.index, symbol]

            # Convert to bar format expected by detector
            bars = []
            for idx in symbol_close.index:
                bars.append({
                    't': idx.isoformat(),
                    'o': float(symbol_open.loc[idx]),
                    'h': float(symbol_high.loc[idx]),
                    'l': float(symbol_low.loc[idx]),
                    'c': float(symbol_close.loc[idx]),
                    'v': float(symbol_volume.loc[idx])
                })

            # Analyze stock
            signal = Momentum52WDetector.analyze_stock(
                ticker=symbol,
                bars=bars,
                volume_threshold=self.volume_threshold,
                min_distance=self.min_distance
            )

            if signal:
                signals.append(signal)

        # Rank by momentum score
        signals.sort(key=lambda s: s.momentum_score, reverse=True)

        # Select top N
        selected = [s.ticker for s in signals[:self.top_n]]

        return selected

    def _build_allocation_matrix(
        self,
        close: pd.DataFrame,
        rebalance_dates: List[str],
        portfolios: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Build allocation matrix from portfolio selections.

        Args:
            close: Close price DataFrame (from VBT data.get('Close'))
            rebalance_dates: List of rebalance dates
            portfolios: Dict mapping rebalance_date -> list of selected tickers

        Returns:
            DataFrame with allocation percentages (rows=dates, cols=symbols)
        """

        # Initialize allocation matrix with zeros
        allocations = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        # Fill allocations at each rebalance date
        for i, rebalance_date in enumerate(rebalance_dates):
            rebalance_ts = pd.Timestamp(rebalance_date)

            # Make timezone-aware if data index is timezone-aware
            if close.index.tz is not None:
                rebalance_ts = rebalance_ts.tz_localize(close.index.tz)

            # Find nearest trading day (handle weekends/holidays)
            if rebalance_ts not in close.index:
                # Find first trading day on or after rebalance date
                future_dates = close.index[close.index >= rebalance_ts]
                if len(future_dates) == 0:
                    continue  # Skip if beyond data range
                rebalance_ts = future_dates[0]

            # Get selected stocks for this rebalance
            selected_stocks = portfolios[rebalance_date]
            if not selected_stocks:
                continue  # Skip if no stocks selected

            # Calculate equal weight
            weight = 1.0 / len(selected_stocks)

            # Determine end date for this allocation period
            if i < len(rebalance_dates) - 1:
                # Next rebalance date
                next_rebalance_ts = pd.Timestamp(rebalance_dates[i + 1])
                # Make timezone-aware if needed
                if close.index.tz is not None:
                    next_rebalance_ts = next_rebalance_ts.tz_localize(close.index.tz)
                # Find nearest trading day
                if next_rebalance_ts not in close.index:
                    future_dates = close.index[close.index >= next_rebalance_ts]
                    next_rebalance_ts = future_dates[0] if len(future_dates) > 0 else close.index[-1]
                end_ts = next_rebalance_ts
            else:
                # Last rebalance - hold until end of data
                end_ts = close.index[-1]

            # Set allocations (forward-fill until next rebalance)
            for stock in selected_stocks:
                if stock in allocations.columns:
                    allocations.loc[rebalance_ts:end_ts, stock] = weight

        return allocations

    def _extract_metrics(self, pf) -> Dict:
        """
        Extract key metrics from VectorBT portfolio.

        Args:
            pf: VectorBT Portfolio object

        Returns:
            Dictionary with metrics
        """
        # Calculate CAGR manually from total return and duration
        years = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days / 365.25
        total_return = pf.total_return
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        metrics = {
            'init_cash': float(pf.init_cash),
            'final_value': float(pf.final_value),
            'total_return': float(total_return),
            'cagr': float(cagr),
            'sharpe': float(pf.sharpe_ratio),
            'sortino': float(pf.sortino_ratio),
            'max_dd': float(pf.max_drawdown),
            'calmar': float(pf.calmar_ratio),
            'total_trades': int(len(pf.orders.readable)),
            'duration_days': int((pf.wrapper.index[-1] - pf.wrapper.index[0]).days),
            'duration_years': round(years, 2)
        }

        return metrics

    def run(
        self,
        start_date: str = '2015-01-01',
        end_date: str = '2025-01-01',
        initial_capital: float = 100000,
        fees: float = 0.001,
        slippage: float = 0.001
    ) -> Dict:
        """
        Run historical backtest of momentum portfolio strategy.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            fees: Trading fees (0.1% default)
            slippage: Slippage (0.1% default)

        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*80}")
        print(f"52-WEEK HIGH MOMENTUM PORTFOLIO BACKTEST")
        print(f"{'='*80}")
        print(f"\nUniverse: {self.universe_name} ({len(self.universe)} stocks)")
        print(f"Portfolio Size: Top {self.top_n} stocks")
        print(f"Volume Filter: {self.volume_threshold}x" if self.volume_threshold else "Volume Filter: Disabled")
        print(f"Rebalance: {self.rebalance_frequency}")
        print(f"Period: {start_date} to {end_date}")

        # Download data
        print(f"\nDownloading data for {len(self.universe)} stocks...")
        try:
            data = vbt.YFData.pull(
                self.universe,
                start=start_date,
                end=end_date
            )
            close = data.get('Close')
        except Exception as e:
            print(f"Error downloading data: {e}")
            return {'error': str(e)}
        print(f"Downloaded {len(close)} days of data")
        print(f"Date range: {close.index[0]} to {close.index[-1]}")

        # Generate rebalance dates
        rebalance_dates = self.get_rebalance_dates(start_date, end_date)
        print(f"\nRebalance schedule: {len(rebalance_dates)} rebalances")
        if rebalance_dates:
            print(f"First rebalance: {rebalance_dates[0]}")
            print(f"Last rebalance: {rebalance_dates[-1]}")
        else:
            print("ERROR: No rebalance dates generated")
            return {'error': 'No rebalance dates generated - check date range'}

        # Select portfolios at each rebalance date
        print(f"\nSelecting portfolios at each rebalance date...")

        rebalance_portfolios = {}
        for rebalance_date in rebalance_dates:
            selected = self.select_portfolio_at_date(data, rebalance_date)
            rebalance_portfolios[rebalance_date] = selected
            print(f"  {rebalance_date}: {len(selected)} stocks selected")

        # Build allocation matrix from portfolio selections
        print(f"\nBuilding allocation matrix...")
        allocations = self._build_allocation_matrix(
            close=close,
            rebalance_dates=rebalance_dates,
            portfolios=rebalance_portfolios
        )
        print(f"Allocation matrix shape: {allocations.shape}")

        # Run VBT portfolio backtest
        print(f"\nRunning VectorBT portfolio backtest...")
        pf = vbt.Portfolio.from_orders(
            close=close,
            size=allocations,
            size_type='targetpercent',
            group_by=True,  # Treat as single portfolio
            cash_sharing=True,  # Share cash across assets
            init_cash=initial_capital,
            fees=fees,
            slippage=slippage,
            freq='D'
        )

        # Extract metrics
        metrics = self._extract_metrics(pf)

        # Print results
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"Initial Capital: ${metrics['init_cash']:,.2f}")
        print(f"Final Value: ${metrics['final_value']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"CAGR: {metrics['cagr']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino']:.2f}")
        print(f"Max Drawdown: {metrics['max_dd']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"\nGate 1 Validation:")
        print(f"  Sharpe >= 0.8: {'PASS' if metrics['sharpe'] >= 0.8 else 'FAIL'}")
        print(f"  CAGR >= 10%: {'PASS' if metrics['cagr'] >= 0.10 else 'FAIL'}")

        results = {
            'portfolio': pf,
            'metrics': metrics,
            'allocations': allocations,
            'universe': self.universe_name,
            'universe_size': len(self.universe),
            'portfolio_size': self.top_n,
            'rebalance_dates': rebalance_dates,
            'portfolios': rebalance_portfolios,
            'period': f"{start_date} to {end_date}"
        }

        return results


def test_scanner_integration():
    """Test that scanner integration is working."""

    print("\n" + "="*80)
    print("TESTING STOCK SCANNER INTEGRATION")
    print("="*80)

    # Test 1: Import check
    print("\nTest 1: Import momentum detector...")
    try:
        from momentum_52w_detector import Momentum52WDetector
        print("  [OK] Successfully imported Momentum52WDetector")
    except ImportError as e:
        print(f"  [FAIL] Failed to import: {e}")
        return False

    # Test 2: Detector functionality
    print("\nTest 2: Test momentum detection...")

    # Create sample bars (simulating stock near 52w high)
    sample_bars = []
    for i in range(252):
        price = 100 + (i * 0.1)  # Uptrend
        sample_bars.append({
            't': f"2024-01-01T00:00:00",
            'o': price,
            'h': price + 1,
            'l': price - 1,
            'c': price,
            'v': 1000000
        })

    signal = Momentum52WDetector.analyze_stock(
        ticker="TEST",
        bars=sample_bars,
        volume_threshold=None,
        min_distance=0.90
    )

    if signal:
        print(f"  [OK] Detected momentum signal:")
        print(f"    Distance from high: {signal.distance_from_high:.4f}")
        print(f"    Momentum score: {signal.momentum_score}")
    else:
        print(f"  [FAIL] No signal detected (expected signal)")

    # Test 3: Portfolio backtest initialization
    print("\nTest 3: Initialize portfolio backtest...")
    try:
        backtest = MomentumPortfolioBacktest(
            universe='technology',
            top_n=10,
            volume_threshold=1.25
        )
        print(f"  [OK] Portfolio backtest initialized")
        print(f"    Universe: {len(backtest.universe)} stocks")
        print(f"    Portfolio size: {backtest.top_n}")
    except Exception as e:
        print(f"  [FAIL] Failed to initialize: {e}")
        return False

    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE")
    print("="*80)

    return True


if __name__ == "__main__":
    # Run integration tests
    test_scanner_integration()

    print("\n\nTo use scanner for backtesting:")
    print("="*80)
    print("""
    from integrations.stock_scanner_bridge import MomentumPortfolioBacktest

    # Initialize backtest
    backtest = MomentumPortfolioBacktest(
        universe='technology',  # or 'sp500_proxy', 'healthcare', etc.
        top_n=20,               # Top 20 stocks
        volume_threshold=1.25,  # Volume filter (None = disabled)
        rebalance_frequency='semi_annual'
    )

    # Run backtest
    results = backtest.run(
        start_date='2015-01-01',
        end_date='2025-01-01',
        initial_capital=100000
    )

    # View selected portfolios
    for date, stocks in results['portfolios'].items():
        print(f"{date}: {stocks}")
    """)
