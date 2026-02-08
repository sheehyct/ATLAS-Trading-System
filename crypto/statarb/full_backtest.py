"""
Full Statistical Arbitrage Backtest - All Crypto Pairs

Tests all combinations of: BTC, ETH, SOL, ADA, XRP
"""

import logging
import warnings
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# Configuration
SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
BACKTEST_DAYS = 365
INITIAL_CAPITAL = 1000.0
POSITION_SIZE_PCT = 10.0
TAKER_FEE_RATE = 0.0002

def calculate_half_life(spread):
    spread = np.asarray(spread)
    spread = spread[~np.isnan(spread)]
    if len(spread) < 20:
        return np.nan
    spread_lag = spread[:-1]
    spread_delta = spread[1:] - spread_lag
    X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
    try:
        beta = np.linalg.lstsq(X, spread_delta, rcond=None)[0]
        theta = -np.log(1 + beta[1])
        if theta <= 0:
            return np.inf
        return np.log(2) / theta
    except (ValueError, TypeError, ZeroDivisionError, np.linalg.LinAlgError):
        return np.nan

def test_pair(prices, sym1, sym2, zscore_window=20, entry_th=2.0, exit_th=0.0):
    """Run single pair backtest."""
    close1 = prices[(sym1, 'close')]
    close2 = prices[(sym2, 'close')]
    
    # Cointegration
    log_p1, log_p2 = np.log(close1), np.log(close2)
    test_stat, p_value, crit = coint(log_p1, log_p2)
    hedge_ratio = np.polyfit(log_p2, log_p1, 1)[0]
    spread = log_p1 - hedge_ratio * log_p2
    half_life = calculate_half_life(spread.values)
    
    # Z-score
    zscore = (spread - spread.rolling(zscore_window).mean()) / spread.rolling(zscore_window).std()
    
    # Signals
    long_spread = (zscore.shift(1) > -entry_th) & (zscore <= -entry_th)
    short_spread = (zscore.shift(1) < entry_th) & (zscore >= entry_th)
    long_exit = ((zscore.shift(1) < exit_th) & (zscore >= exit_th)) | (zscore >= 3.0)
    short_exit = ((zscore.shift(1) > exit_th) & (zscore <= exit_th)) | (zscore <= -3.0)
    
    # Build arrays
    data = pd.concat([close1, close2], axis=1, keys=[sym1, sym2])
    long_entries = pd.DataFrame(False, index=data.index, columns=[sym1, sym2])
    short_entries = pd.DataFrame(False, index=data.index, columns=[sym1, sym2])
    exits = pd.DataFrame(False, index=data.index, columns=[sym1, sym2])
    
    long_entries.loc[long_spread, sym1] = True
    short_entries.loc[long_spread, sym2] = True
    short_entries.loc[short_spread, sym1] = True
    long_entries.loc[short_spread, sym2] = True
    exits.loc[long_exit | short_exit, :] = True
    
    # Simulate
    try:
        pf = vbt.Portfolio.from_signals(
            data, entries=long_entries, short_entries=short_entries, exits=exits,
            size=POSITION_SIZE_PCT, size_type="valuepercent100",
            group_by=True, cash_sharing=True, call_seq="auto",
            init_cash=INITIAL_CAPITAL, fees=TAKER_FEE_RATE,
        )
        stats = pf.stats()
        trades = pf.trades
        num_trades = len(trades.records_arr) if hasattr(trades, 'records_arr') else 0
        
        return {
            'pair': f"{sym1.replace('-USD','')}/{sym2.replace('-USD','')}",
            'p_value': p_value,
            'half_life': half_life,
            'cointegrated': p_value < 0.05,
            'hedge_ratio': hedge_ratio,
            'total_return': float(stats.get("Total Return [%]", 0)) / 100,
            'sharpe': float(stats.get("Sharpe Ratio", 0)) if not pd.isna(stats.get("Sharpe Ratio")) else 0,
            'max_dd': float(stats.get("Max Drawdown [%]", 0)) / 100,
            'num_trades': num_trades,
            'win_rate': float(trades.win_rate) if num_trades > 0 and not pd.isna(trades.win_rate) else 0,
            'fees': float(stats.get("Total Fees Paid", 0)),
        }
    except Exception as e:
        return None

def main():
    print("=" * 80)
    print("  STATISTICAL ARBITRAGE - ALL CRYPTO PAIRS BACKTEST")
    print("=" * 80)
    print(f"  Symbols: {', '.join([s.replace('-USD','') for s in SYMBOLS])}")
    print(f"  Period: {BACKTEST_DAYS} days | Capital: ${INITIAL_CAPITAL}")
    print("=" * 80)
    print()
    
    # Fetch data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
    
    print("Fetching data...")
    dfs = {}
    for sym in SYMBOLS:
        data = vbt.YFData.pull(sym, start=start_date, end=end_date, silence_warnings=True)
        df = data.get()
        df.columns = [c.lower() for c in df.columns]
        dfs[sym] = df
        print(f"  {sym}: {len(df)} bars")
    
    prices = pd.concat(dfs, axis=1, keys=dfs.keys()).ffill().dropna()
    print(f"  Aligned: {len(prices)} bars")
    print()
    
    # Test all pairs
    print("COINTEGRATION ANALYSIS")
    print("-" * 80)
    print(f"{'Pair':<12} {'P-Value':>10} {'Half-Life':>12} {'Coint?':>8} {'Hedge Ratio':>12}")
    print("-" * 80)
    
    all_pairs = list(combinations(SYMBOLS, 2))
    results = []
    
    for sym1, sym2 in all_pairs:
        result = test_pair(prices, sym1, sym2)
        if result:
            results.append(result)
            hl = f"{result['half_life']:.1f}d" if not np.isinf(result['half_life']) else "INF"
            coint_str = "YES" if result['cointegrated'] else "no"
            print(f"{result['pair']:<12} {result['p_value']:>10.4f} {hl:>12} {coint_str:>8} {result['hedge_ratio']:>12.4f}")
    
    print()
    print("BACKTEST RESULTS (Sorted by Sharpe Ratio)")
    print("-" * 80)
    print(f"{'Pair':<12} {'Return':>10} {'Sharpe':>8} {'Max DD':>8} {'Trades':>8} {'Win Rate':>10} {'Fees':>8}")
    print("-" * 80)
    
    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    for r in results:
        print(f"{r['pair']:<12} {r['total_return']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}% {r['num_trades']:>8} {r['win_rate']*100:>9.1f}% ${r['fees']:>6.2f}")
    
    print()
    print("=" * 80)
    print("  TOP 3 PAIRS BY SHARPE RATIO")
    print("=" * 80)
    for i, r in enumerate(results[:3], 1):
        coint_note = " [COINTEGRATED]" if r['cointegrated'] else ""
        print(f"  {i}. {r['pair']}: Sharpe {r['sharpe']:.2f}, Return {r['total_return']*100:+.2f}%{coint_note}")
    
    print()
    
    # Check for cointegrated pairs
    coint_pairs = [r for r in results if r['cointegrated']]
    if coint_pairs:
        print("  COINTEGRATED PAIRS FOUND:")
        for r in coint_pairs:
            print(f"    - {r['pair']}: p={r['p_value']:.4f}, half-life={r['half_life']:.1f} days")
    else:
        print("  WARNING: No statistically significant cointegration found (p < 0.05)")
        print("  Strategy is trading correlation, not mean reversion - higher regime risk")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
