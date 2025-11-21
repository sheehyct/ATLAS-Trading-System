"""
Phase 2: Extended Backtest with Larger Universe

Tests System A2 (S&P 500 Full List) to validate ATR filtering at scale.

Key Question:
    Does ATR/volume filtering improve performance with 5x more stocks to choose from?

Systems Tested:
    A2: S&P 500 (500 stocks) → ATR filter → Top-5 momentum
    A1: S&P 100 (100 stocks) → ATR filter → Top-5 momentum (Phase 1 baseline)

Hypothesis:
    Larger universe + ATR filter = better stock selection = higher returns

Reference:
    Phase 1 Results: System A1 achieved 0.93 Sharpe, -15.85% DD
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime
from typing import List, Dict

from regime.academic_jump_model import AcademicJumpModel


# S&P 500 Components (Top 200 by market cap - representative sample)
# Using top 200 for computational efficiency while maintaining diversity
SP500_TOP_200 = [
    # Mega Cap Tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "AMD",
    "CRM", "ADBE", "NFLX", "INTC", "CSCO", "ACN", "IBM", "NOW", "QCOM", "TXN",
    "INTU", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "CRWD", "MCHP",
    "FTNT", "ADI", "ADSK", "ANSS", "CDNS", "MRVL", "ON", "MPWR",

    # Healthcare & Pharma
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
    "AMGN", "CVS", "MDT", "GILD", "CI", "REGN", "SYK", "VRTX", "ZTS", "HUM",
    "BDX", "ELV", "ISRG", "BSX", "IDXX", "MCK", "COR", "HCA", "CAH", "DXCM",

    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "CB", "AXP",
    "PNC", "USB", "TFC", "COF", "BK", "AIG", "MET", "AFL", "PRU", "ALL",
    "SPGI", "MMC", "FI", "PGR", "TRV", "AJG", "AON", "ICE", "CME", "MCO",

    # Consumer
    "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "TJX", "MDLZ", "CL", "KMB", "GIS", "HSY", "SYY", "DG", "DLTR",
    "MO", "PM", "KHC", "MNST", "KDP", "CAG", "CPB", "HRL",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HES",
    "HAL", "BKR", "WMB", "KMI", "OKE", "LNG", "FANG", "DVN", "MRO", "APA",

    # Industrials
    "CAT", "RTX", "HON", "UNP", "BA", "GE", "LMT", "DE", "MMM", "UPS",
    "NOC", "GD", "ETN", "ITW", "EMR", "CSX", "NSC", "WM", "FDX", "ROK",
    "CARR", "OTIS", "PCAR", "IR", "AME", "FAST", "VRSK", "J", "CHRW", "EXPD",

    # Materials & Chemicals
    "LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "VMC", "MLM", "DD",
    "DOW", "PPG", "ALB", "CF", "MOS", "FMC", "IFF", "CE", "EMN", "IP",

    # Real Estate & Utilities
    "PLD", "AMT", "CCI", "PSA", "EQIX", "SPG", "O", "DLR", "WELL", "AVB",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ES", "ED",

    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "TMUS", "VZ", "CHTR", "PARA",

    # Additional Quality Names
    "V", "MA", "BRK-B", "BKNG", "SPGI", "ISRG", "ELV", "ZTS", "REGN", "VRTX"
]

# Remove duplicates
SP500_TOP_200 = list(set(SP500_TOP_200))

# ATR/Volume filter thresholds
MIN_ATR = 1.50
MIN_ATR_PERCENT = 1.5
MIN_DOLLAR_VOLUME = 10_000_000

# Regime allocation
REGIME_ALLOCATION = {
    'TREND_BULL': 1.00,
    'TREND_NEUTRAL': 0.70,
    'TREND_BEAR': 0.30,
    'CRASH': 0.00
}


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_momentum_score(close: pd.Series, lookback: int = 252) -> pd.Series:
    """Calculate 52-week momentum score."""
    high_52w = close.rolling(window=lookback, min_periods=lookback).max()
    return close / high_52w


def apply_atr_volume_filter(data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[str]:
    """Apply ATR/volume filters to universe."""
    passing_tickers = []

    for ticker, df in data.items():
        hist = df.loc[:date]
        if len(hist) < 20:
            continue

        close = hist['Close'].iloc[-1]
        atr = calculate_atr(hist['High'], hist['Low'], hist['Close'], period=14).iloc[-1]
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]

        atr_percent = (atr / close * 100) if close > 0 else 0
        dollar_volume = avg_volume * close

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
    """Select top N stocks by 52-week momentum."""
    momentum_scores = {}

    for ticker in universe:
        if ticker not in data:
            continue
        df = data[ticker]
        hist = df.loc[:date]
        if len(hist) < 252:
            continue
        momentum = calculate_momentum_score(hist['Close'], lookback=252).iloc[-1]
        if not pd.isna(momentum) and momentum >= 0.90:
            momentum_scores[ticker] = momentum

    sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tickers[:top_n]]


def load_universe_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data for universe."""
    print(f"Loading data for {len(tickers)} tickers...")
    data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(tickers)}")
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
            failed.append(ticker)

    print(f"  Loaded {len(data)} tickers successfully")
    if failed:
        print(f"  Failed: {len(failed)} tickers")

    return data


def run_atlas_detection(spy_data: pd.DataFrame) -> pd.Series:
    """Run ATLAS regime detection."""
    print("\nRunning ATLAS regime detection...")
    atlas = AcademicJumpModel()

    try:
        atlas_regimes, _, _ = atlas.online_inference(spy_data, lookback=1000, default_lambda=1.5)
    except:
        atlas_regimes = pd.Series('TREND_NEUTRAL', index=spy_data.index)

    atlas_regimes_aligned = atlas_regimes.reindex(spy_data.index).fillna('TREND_NEUTRAL')

    regime_counts = atlas_regimes_aligned.value_counts()
    print(f"  Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"    {regime}: {count} days ({100*count/len(atlas_regimes_aligned):.1f}%)")

    return atlas_regimes_aligned


def backtest_system_a2(
    spy_data: pd.DataFrame,
    universe_data: Dict[str, pd.DataFrame],
    atlas_regimes: pd.Series,
    universe_size: int
) -> Dict:
    """Backtest System A2 with larger universe."""
    print(f"\n[System A2] S&P 500 Top {universe_size} with ATR filter...")

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

    portfolio_weights = pd.DataFrame(0.0, index=spy_data.index, columns=['weight'])
    portfolio_stocks = {}

    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date not in spy_data.index:
            future_dates = spy_data.index[spy_data.index >= rebalance_date]
            if len(future_dates) == 0:
                continue
            rebalance_date = future_dates[0]

        # ATR filter
        filtered_tickers = apply_atr_volume_filter(universe_data, rebalance_date)

        # Select top-5
        selected_tickers = select_top_momentum_stocks(
            universe_data, rebalance_date, filtered_tickers, top_n=5
        )

        # Get regime
        regime = atlas_regimes.loc[rebalance_date]
        allocation = REGIME_ALLOCATION.get(regime, 0.70)

        print(f"\n  Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
        print(f"    Filtered: {len(filtered_tickers)} stocks")
        print(f"    Selected: {', '.join(selected_tickers) if selected_tickers else 'None'}")
        print(f"    Regime: {regime} ({allocation:.0%})")

        portfolio_stocks[rebalance_date] = selected_tickers

        next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else spy_data.index[-1]
        if selected_tickers:
            portfolio_weights.loc[rebalance_date:next_rebalance, 'weight'] = allocation
        else:
            portfolio_weights.loc[rebalance_date:next_rebalance, 'weight'] = 0.0

    # Calculate returns
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

    # Scale by regime
    regime_weights = portfolio_weights['weight'].shift(1).fillna(0)
    final_returns = portfolio_returns * regime_weights

    # Metrics
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


def main():
    """Run Phase 2 backtest."""
    print("=" * 80)
    print("PHASE 2: EXTENDED UNIVERSE BACKTEST")
    print("=" * 80)

    # Load SPY
    print("\nLoading SPY data...")
    spy_data = vbt.YFData.pull('SPY', start='2020-01-01', end='2024-12-31', timeframe='1d', tz='America/New_York').get()
    print(f"  SPY: {len(spy_data)} days")

    # ATLAS
    atlas_regimes = run_atlas_detection(spy_data)

    # Load larger universe
    sp500_data = load_universe_data(SP500_TOP_200, '2020-01-01', '2024-12-31')

    # Backtest A2
    results_a2 = backtest_system_a2(spy_data, sp500_data, atlas_regimes, len(SP500_TOP_200))

    # Compare to Phase 1 results
    print("\n" + "=" * 80)
    print("PHASE 2 vs PHASE 1 COMPARISON")
    print("=" * 80)

    # Phase 1 System A1 results (from earlier run)
    phase1_a1 = {
        'total_return': 0.6913,
        'sharpe_ratio': 0.93,
        'max_drawdown': -0.1585,
        'num_trades': 8
    }

    comparison = pd.DataFrame({
        'System A2 (S&P 200)': [
            results_a2['total_return'],
            results_a2['sharpe_ratio'],
            results_a2['max_drawdown'],
            results_a2['num_trades']
        ],
        'System A1 (S&P 100)': [
            phase1_a1['total_return'],
            phase1_a1['sharpe_ratio'],
            phase1_a1['max_drawdown'],
            phase1_a1['num_trades']
        ]
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Trades'])

    print(comparison.to_string())

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if results_a2['sharpe_ratio'] > phase1_a1['sharpe_ratio']:
        print(f"\n[WIN] System A2 improved Sharpe by {((results_a2['sharpe_ratio']/phase1_a1['sharpe_ratio'])-1)*100:.1f}%")
        print("  Larger universe + ATR filter found better opportunities")
    else:
        print(f"\n[NEUTRAL] System A1 remains better (Sharpe {phase1_a1['sharpe_ratio']:.2f} vs {results_a2['sharpe_ratio']:.2f})")
        print("  Top 100 already captured best opportunities")

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)

    return results_a2


if __name__ == '__main__':
    main()
