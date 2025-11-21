"""
Phase 1: Comprehensive Multi-Stock Backtest Comparison

Tests 3 systems on 2020-2024 data:
1. System A1: S&P 500 with ATR/Volume filter + Top-5 momentum
2. System A3: Fixed Tech universe + Top-5 momentum
3. Baseline: SPY Buy & Hold

Purpose:
    Validate that dynamic ATR/volume filtering improves stock selection
    vs fixed sector universe, and both vs SPY baseline.

Metrics:
    - Total return
    - Sharpe ratio
    - Maximum drawdown
    - Win rate per stock
    - Rebalance count

Reference:
    Scanner implementation: C:/Dev/strat-stock-scanner
    ATR/Volume filters from strat_detector.py
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime
from typing import List, Dict, Tuple

# Import ATLAS components
from regime.academic_jump_model import AcademicJumpModel


# Stock universes
SP500_TOP_100 = [
    # Top 100 S&P 500 by market cap (approximate)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "LLY", "TSLA",
    "JPM", "WMT", "V", "XOM", "UNH", "MA", "PG", "COST", "JNJ", "HD",
    "NFLX", "BAC", "ABBV", "CRM", "MRK", "KO", "CVX", "AMD", "PEP", "ORCL",
    "TMO", "ACN", "MCD", "CSCO", "ABT", "DHR", "DIS", "ADBE", "WFC", "TXN",
    "INTC", "QCOM", "IBM", "INTU", "AMGN", "NOW", "PM", "GE", "RTX", "CAT",
    "SPGI", "AMAT", "UNP", "GS", "BKNG", "LOW", "SCHW", "NEE", "HON", "PFE",
    "BLK", "SYK", "AXP", "LMT", "C", "T", "ELV", "ADI", "SBUX", "GILD",
    "REGN", "MDLZ", "TJX", "MMC", "PLD", "VRTX", "BA", "CVS", "ISRG", "CB",
    "MO", "CI", "BMY", "ZTS", "SO", "DUK", "TMUS", "DE", "EOG", "FI",
    "BDX", "PNC", "USB", "SLB", "CL", "WM", "NOC", "NSC", "ITW", "MMM"
]

TECH_30 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "AMD", "CRM",
    "ADBE", "NFLX", "INTC", "CSCO", "ACN", "IBM", "NOW", "QCOM", "TXN", "INTU",
    "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "PANW", "CRWD"
]

# ATR/Volume filter thresholds (from scanner)
MIN_ATR = 1.50              # Minimum $1.50 absolute range
MIN_ATR_PERCENT = 1.5       # Minimum 1.5% volatility
MIN_DOLLAR_VOLUME = 10_000_000  # Minimum $10M liquidity

# Regime allocation rules
REGIME_ALLOCATION = {
    'TREND_BULL': 1.00,      # 100% deployed
    'TREND_NEUTRAL': 0.70,   # 70% deployed
    'TREND_BEAR': 0.30,      # 30% deployed
    'CRASH': 0.00            # 0% deployed (cash only)
}


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (from scanner implementation).

    ATR = average of True Range over period
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()

    return atr


def calculate_momentum_score(close: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Calculate 52-week high momentum score.

    Score = current_price / 52-week_high
    """
    high_52w = close.rolling(window=lookback, min_periods=lookback).max()
    momentum = close / high_52w
    return momentum


def apply_atr_volume_filter(
    data: Dict[str, pd.DataFrame],
    date: pd.Timestamp
) -> List[str]:
    """
    Apply ATR/volume filters to universe at specific date.

    Returns list of tickers that pass all filters.
    """
    passing_tickers = []

    for ticker, df in data.items():
        # Get data up to rebalance date
        hist = df.loc[:date]

        if len(hist) < 20:  # Need 20 days for ATR and volume
            continue

        # Get most recent values
        close = hist['Close'].iloc[-1]

        # Calculate ATR (14-day)
        atr = calculate_atr(hist['High'], hist['Low'], hist['Close'], period=14).iloc[-1]

        # Calculate average volume (20-day)
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]

        # Calculate metrics
        atr_percent = (atr / close * 100) if close > 0 else 0
        dollar_volume = avg_volume * close

        # Apply filters
        if (atr >= MIN_ATR and
            atr_percent >= MIN_ATR_PERCENT and
            dollar_volume >= MIN_DOLLAR_VOLUME):
            passing_tickers.append(ticker)

    return passing_tickers


def select_top_momentum_stocks(
    data: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    universe: List[str],
    top_n: int = 5
) -> List[str]:
    """
    Select top N stocks by 52-week momentum from universe.
    """
    momentum_scores = {}

    for ticker in universe:
        if ticker not in data:
            continue

        df = data[ticker]
        hist = df.loc[:date]

        if len(hist) < 252:  # Need 1 year for momentum
            continue

        # Calculate momentum
        momentum = calculate_momentum_score(hist['Close'], lookback=252).iloc[-1]

        if not pd.isna(momentum) and momentum >= 0.90:  # Within 10% of 52w high
            momentum_scores[ticker] = momentum

    # Sort by momentum and take top N
    sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [t[0] for t in sorted_tickers[:top_n]]

    return top_tickers


def load_universe_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for all tickers in universe.
    """
    print(f"Loading data for {len(tickers)} tickers...")

    data = {}
    failed = []

    for ticker in tickers:
        try:
            ticker_data = vbt.YFData.pull(
                ticker,
                start=start_date,
                end=end_date,
                timeframe='1d',
                tz='America/New_York'
            )
            df = ticker_data.get()

            if len(df) > 0:
                data[ticker] = df
            else:
                failed.append(ticker)
        except Exception as e:
            print(f"  Failed to load {ticker}: {e}")
            failed.append(ticker)

    print(f"  Loaded {len(data)} tickers successfully")
    if failed:
        print(f"  Failed: {len(failed)} tickers - {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

    return data


def run_atlas_detection(spy_data: pd.DataFrame) -> pd.Series:
    """Run ATLAS regime detection on SPY."""
    print("\nRunning ATLAS regime detection...")

    atlas = AcademicJumpModel()

    try:
        atlas_regimes, lambda_series, theta_df = atlas.online_inference(
            spy_data,
            lookback=1000,
            default_lambda=1.5
        )
    except Exception as e:
        print(f"  Warning: ATLAS inference failed: {e}")
        atlas_regimes = pd.Series('TREND_NEUTRAL', index=spy_data.index)

    # Align and forward-fill
    atlas_regimes_aligned = atlas_regimes.reindex(spy_data.index).fillna('TREND_NEUTRAL')

    # Log distribution
    regime_counts = atlas_regimes_aligned.value_counts()
    print(f"  Regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(atlas_regimes_aligned) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

    return atlas_regimes_aligned


def backtest_system_a1_atr_filter(
    spy_data: pd.DataFrame,
    universe_data: Dict[str, pd.DataFrame],
    atlas_regimes: pd.Series,
    universe_name: str
) -> Dict:
    """
    Backtest System A1: ATR/Volume filter + Top-5 momentum.
    """
    print(f"\n[System A1] {universe_name} with ATR/Volume filter...")

    # Rebalance dates
    rebalance_dates = []
    for year in range(spy_data.index[0].year, spy_data.index[-1].year + 1):
        for month in [2, 8]:
            try:
                date = pd.Timestamp(year=year, month=month, day=1, tz=spy_data.index.tz)
                if date >= spy_data.index[0] and date <= spy_data.index[-1]:
                    rebalance_dates.append(date)
            except:
                continue

    print(f"  Rebalance dates: {len(rebalance_dates)}")

    # Track portfolio allocation over time
    portfolio_weights = pd.DataFrame(0.0, index=spy_data.index, columns=['weight'])
    portfolio_stocks = {}  # date -> list of selected stocks

    for i, rebalance_date in enumerate(rebalance_dates):
        # Find closest trading day
        if rebalance_date not in spy_data.index:
            future_dates = spy_data.index[spy_data.index >= rebalance_date]
            if len(future_dates) == 0:
                continue
            rebalance_date = future_dates[0]

        # Step 1: Apply ATR/volume filter
        filtered_tickers = apply_atr_volume_filter(universe_data, rebalance_date)

        # Step 2: Select top-5 by momentum from filtered universe
        selected_tickers = select_top_momentum_stocks(
            universe_data,
            rebalance_date,
            filtered_tickers,
            top_n=5
        )

        # Step 3: Get regime and allocation
        regime = atlas_regimes.loc[rebalance_date]
        allocation = REGIME_ALLOCATION.get(regime, 0.70)

        print(f"\n  Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
        print(f"    Filtered: {len(filtered_tickers)} stocks pass ATR/volume")
        print(f"    Selected: {', '.join(selected_tickers) if selected_tickers else 'None'}")
        print(f"    Regime: {regime} ({allocation:.0%} allocation)")

        # Store selection
        portfolio_stocks[rebalance_date] = selected_tickers

        # Set allocation until next rebalance
        next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else spy_data.index[-1]
        if len(selected_tickers) > 0:
            portfolio_weights.loc[rebalance_date:next_rebalance, 'weight'] = allocation
        else:
            portfolio_weights.loc[rebalance_date:next_rebalance, 'weight'] = 0.0

    # Calculate portfolio returns
    # Equal-weight selected stocks, scaled by regime allocation
    portfolio_returns = pd.Series(0.0, index=spy_data.index)

    for rebalance_date, selected_tickers in portfolio_stocks.items():
        if not selected_tickers:
            continue

        # Get next rebalance date
        future_rebalances = [d for d in portfolio_stocks.keys() if d > rebalance_date]
        next_rebalance = future_rebalances[0] if future_rebalances else spy_data.index[-1]

        # Calculate equal-weight portfolio returns for this period
        period_returns = pd.Series(0.0, index=spy_data.index)
        weight_per_stock = 1.0 / len(selected_tickers)

        for ticker in selected_tickers:
            if ticker in universe_data:
                stock_returns = universe_data[ticker]['Close'].pct_change()
                period_returns += weight_per_stock * stock_returns

        # Apply to portfolio (only for rebalance period)
        mask = (spy_data.index >= rebalance_date) & (spy_data.index < next_rebalance)
        portfolio_returns[mask] = period_returns[mask]

    # Scale by regime allocation
    regime_weights = portfolio_weights['weight'].shift(1).fillna(0)
    final_returns = portfolio_returns * regime_weights

    # Calculate metrics
    init_cash = 10000
    portfolio_value = init_cash * (1 + final_returns).cumprod()

    total_return = (portfolio_value.iloc[-1] / init_cash) - 1
    mean_return = final_returns.mean() * 252
    std_return = final_returns.std() * np.sqrt(252)
    sharpe = mean_return / std_return if std_return > 0 else 0

    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_drawdown = drawdown.min()

    num_rebalances = len([s for s in portfolio_stocks.values() if s])

    print(f"\n  Results:")
    print(f"    Total Return: {total_return:.2%}")
    print(f"    Sharpe Ratio: {sharpe:.2f}")
    print(f"    Max Drawdown: {max_drawdown:.2%}")
    print(f"    Rebalances: {num_rebalances}")

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': num_rebalances,
        'portfolio_value': portfolio_value,
        'portfolio_stocks': portfolio_stocks
    }


def backtest_system_a3_fixed_universe(
    spy_data: pd.DataFrame,
    universe_data: Dict[str, pd.DataFrame],
    atlas_regimes: pd.Series,
    universe_name: str
) -> Dict:
    """
    Backtest System A3: Fixed universe (no filter) + Top-5 momentum.
    """
    print(f"\n[System A3] {universe_name} fixed universe (no ATR filter)...")

    # Same logic as A1, but without ATR filter step
    rebalance_dates = []
    for year in range(spy_data.index[0].year, spy_data.index[-1].year + 1):
        for month in [2, 8]:
            try:
                date = pd.Timestamp(year=year, month=month, day=1, tz=spy_data.index.tz)
                if date >= spy_data.index[0] and date <= spy_data.index[-1]:
                    rebalance_dates.append(date)
            except:
                continue

    print(f"  Rebalance dates: {len(rebalance_dates)}")

    portfolio_weights = pd.DataFrame(0.0, index=spy_data.index, columns=['weight'])
    portfolio_stocks = {}

    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date not in spy_data.index:
            future_dates = spy_data.index[spy_data.index >= rebalance_date]
            if len(future_dates) == 0:
                continue
            rebalance_date = future_dates[0]

        # Select top-5 by momentum from FULL universe (no filter)
        selected_tickers = select_top_momentum_stocks(
            universe_data,
            rebalance_date,
            list(universe_data.keys()),
            top_n=5
        )

        regime = atlas_regimes.loc[rebalance_date]
        allocation = REGIME_ALLOCATION.get(regime, 0.70)

        print(f"\n  Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
        print(f"    Selected: {', '.join(selected_tickers) if selected_tickers else 'None'}")
        print(f"    Regime: {regime} ({allocation:.0%} allocation)")

        portfolio_stocks[rebalance_date] = selected_tickers

        next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else spy_data.index[-1]
        if len(selected_tickers) > 0:
            portfolio_weights.loc[rebalance_date:next_rebalance, 'weight'] = allocation
        else:
            portfolio_weights.loc[rebalance_date:next_rebalance, 'weight'] = 0.0

    # Calculate returns (same as A1)
    portfolio_returns = pd.Series(0.0, index=spy_data.index)

    for rebalance_date, selected_tickers in portfolio_stocks.items():
        if not selected_tickers:
            continue

        future_rebalances = [d for d in portfolio_stocks.keys() if d > rebalance_date]
        next_rebalance = future_rebalances[0] if future_rebalances else spy_data.index[-1]

        period_returns = pd.Series(0.0, index=spy_data.index)
        weight_per_stock = 1.0 / len(selected_tickers)

        for ticker in selected_tickers:
            if ticker in universe_data:
                stock_returns = universe_data[ticker]['Close'].pct_change()
                period_returns += weight_per_stock * stock_returns

        mask = (spy_data.index >= rebalance_date) & (spy_data.index < next_rebalance)
        portfolio_returns[mask] = period_returns[mask]

    regime_weights = portfolio_weights['weight'].shift(1).fillna(0)
    final_returns = portfolio_returns * regime_weights

    init_cash = 10000
    portfolio_value = init_cash * (1 + final_returns).cumprod()

    total_return = (portfolio_value.iloc[-1] / init_cash) - 1
    mean_return = final_returns.mean() * 252
    std_return = final_returns.std() * np.sqrt(252)
    sharpe = mean_return / std_return if std_return > 0 else 0

    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_drawdown = drawdown.min()

    num_rebalances = len([s for s in portfolio_stocks.values() if s])

    print(f"\n  Results:")
    print(f"    Total Return: {total_return:.2%}")
    print(f"    Sharpe Ratio: {sharpe:.2f}")
    print(f"    Max Drawdown: {max_drawdown:.2%}")
    print(f"    Rebalances: {num_rebalances}")

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': num_rebalances,
        'portfolio_value': portfolio_value,
        'portfolio_stocks': portfolio_stocks
    }


def backtest_spy_baseline(spy_data: pd.DataFrame) -> Dict:
    """Backtest SPY buy-and-hold baseline."""
    print("\n[Baseline] SPY Buy-and-Hold...")

    spy_returns = spy_data['Close'].pct_change()

    init_cash = 10000
    portfolio_value = init_cash * (1 + spy_returns).cumprod()

    total_return = (portfolio_value.iloc[-1] / init_cash) - 1
    mean_return = spy_returns.mean() * 252
    std_return = spy_returns.std() * np.sqrt(252)
    sharpe = mean_return / std_return if std_return > 0 else 0

    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_drawdown = drawdown.min()

    print(f"\n  Results:")
    print(f"    Total Return: {total_return:.2%}")
    print(f"    Sharpe Ratio: {sharpe:.2f}")
    print(f"    Max Drawdown: {max_drawdown:.2%}")

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': 1,
        'portfolio_value': portfolio_value
    }


def main():
    """Run Phase 1 comprehensive backtest."""
    print("=" * 80)
    print("PHASE 1: COMPREHENSIVE BACKTEST COMPARISON")
    print("=" * 80)

    # Load SPY for regime detection
    print("\nLoading SPY data...")
    spy_data = vbt.YFData.pull('SPY', start='2020-01-01', end='2024-12-31', timeframe='1d', tz='America/New_York').get()
    print(f"  SPY: {len(spy_data)} days")

    # Run ATLAS regime detection
    atlas_regimes = run_atlas_detection(spy_data)

    # Load universes
    sp500_data = load_universe_data(SP500_TOP_100, '2020-01-01', '2024-12-31')
    tech_data = load_universe_data(TECH_30, '2020-01-01', '2024-12-31')

    # Run backtests
    results_a1 = backtest_system_a1_atr_filter(spy_data, sp500_data, atlas_regimes, "S&P 500 Top 100")
    results_a3 = backtest_system_a3_fixed_universe(spy_data, tech_data, atlas_regimes, "Technology 30")
    results_baseline = backtest_spy_baseline(spy_data)

    # Comparison
    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS COMPARISON")
    print("=" * 80)

    comparison = pd.DataFrame({
        'System A1 (ATR Filter)': [
            results_a1['total_return'],
            results_a1['sharpe_ratio'],
            results_a1['max_drawdown'],
            results_a1['num_trades']
        ],
        'System A3 (Fixed Tech)': [
            results_a3['total_return'],
            results_a3['sharpe_ratio'],
            results_a3['max_drawdown'],
            results_a3['num_trades']
        ],
        'SPY Baseline': [
            results_baseline['total_return'],
            results_baseline['sharpe_ratio'],
            results_baseline['max_drawdown'],
            results_baseline['num_trades']
        ]
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Trades'])

    print(comparison.to_string())

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find winner
    returns = {
        'A1': results_a1['total_return'],
        'A3': results_a3['total_return'],
        'SPY': results_baseline['total_return']
    }

    winner = max(returns, key=returns.get)
    winner_name = {
        'A1': 'System A1 (ATR Filter)',
        'A3': 'System A3 (Fixed Tech)',
        'SPY': 'SPY Baseline'
    }[winner]

    print(f"\nWINNER: {winner_name}")
    print(f"  Total Return: {returns[winner]:.2%}")

    if winner == 'A1':
        print("\n  ATR/Volume filtering successfully selected high-momentum stocks across sectors")
    elif winner == 'A3':
        print("\n  Technology sector concentration outperformed diversified ATR filtering")
    else:
        print("\n  Neither system beat buy-and-hold - consider Phase 2 tests or strategy revision")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)

    return {
        'a1': results_a1,
        'a3': results_a3,
        'baseline': results_baseline
    }


if __name__ == '__main__':
    main()
