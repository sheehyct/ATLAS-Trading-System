"""
ADA/XRP Parameter Optimization with Walk-Forward Validation

Anti-overfitting measures:
1. Train/Test split (70/30) - optimize on training, verify on test
2. Limited parameter grid - only sensible ranges
3. Stability check - parameters must work across both periods
4. No curve-fitting exotic combinations

Author: ATLAS Development Team
Date: January 2026
"""

import warnings
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")

# Configuration
SYMBOL1 = 'ADA-USD'
SYMBOL2 = 'XRP-USD'
INITIAL_CAPITAL = 1000.0
LEVERAGE = 3.0
POSITION_PCT = 20.0
FEE_RATE = 0.0002 * LEVERAGE  # Scaled for leverage

# Parameter grid (intentionally limited to avoid overfitting)
ZSCORE_WINDOWS = [10, 15, 20, 30, 40]  # 5 options
ENTRY_THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5]  # 5 options  
EXIT_THRESHOLDS = [0.0, 0.25, 0.5]  # 3 options
# Total: 5 * 5 * 3 = 75 combinations (reasonable)


def run_backtest(prices, window, entry_th, exit_th):
    """Run single backtest, return metrics dict."""
    close1 = prices[(SYMBOL1, 'close')]
    close2 = prices[(SYMBOL2, 'close')]
    
    # Spread and Z-score
    log_p1, log_p2 = np.log(close1), np.log(close2)
    hedge_ratio = np.polyfit(log_p2, log_p1, 1)[0]
    spread = log_p1 - hedge_ratio * log_p2
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    # Signals
    long_spread = (zscore.shift(1) > -entry_th) & (zscore <= -entry_th)
    short_spread = (zscore.shift(1) < entry_th) & (zscore >= entry_th)
    long_exit = ((zscore.shift(1) < exit_th) & (zscore >= exit_th)) | (zscore >= 3.0)
    short_exit = ((zscore.shift(1) > exit_th) & (zscore <= exit_th)) | (zscore <= -3.0)
    
    # Build arrays
    data = pd.concat([close1, close2], axis=1, keys=[SYMBOL1, SYMBOL2])
    long_entries = pd.DataFrame(False, index=data.index, columns=[SYMBOL1, SYMBOL2])
    short_entries = pd.DataFrame(False, index=data.index, columns=[SYMBOL1, SYMBOL2])
    exits = pd.DataFrame(False, index=data.index, columns=[SYMBOL1, SYMBOL2])
    
    long_entries.loc[long_spread, SYMBOL1] = True
    short_entries.loc[long_spread, SYMBOL2] = True
    short_entries.loc[short_spread, SYMBOL1] = True
    long_entries.loc[short_spread, SYMBOL2] = True
    exits.loc[long_exit | short_exit, :] = True
    
    try:
        pf = vbt.Portfolio.from_signals(
            data, entries=long_entries, short_entries=short_entries, exits=exits,
            size=POSITION_PCT * LEVERAGE, size_type="valuepercent100",
            group_by=True, cash_sharing=True, call_seq="auto",
            init_cash=INITIAL_CAPITAL, fees=FEE_RATE,
        )
        
        stats = pf.stats()
        trades = pf.trades
        num_trades = len(trades.records_arr) if hasattr(trades, 'records_arr') else 0
        
        return {
            'total_return': float(stats.get("Total Return [%]", 0)) / 100,
            'sharpe': float(stats.get("Sharpe Ratio", 0)) if not pd.isna(stats.get("Sharpe Ratio")) else 0,
            'sortino': float(stats.get("Sortino Ratio", 0)) if not pd.isna(stats.get("Sortino Ratio")) else 0,
            'max_dd': float(stats.get("Max Drawdown [%]", 0)) / 100,
            'num_trades': num_trades,
            'win_rate': float(trades.win_rate) if num_trades > 0 and not pd.isna(trades.win_rate) else 0,
            'profit_factor': float(trades.profit_factor) if num_trades > 0 and hasattr(trades, 'profit_factor') and not pd.isna(trades.profit_factor) else 0,
        }
    except (ValueError, TypeError, KeyError, ZeroDivisionError, AttributeError):
        return None


def main():
    print()
    print("=" * 90)
    print("  ADA/XRP PARAMETER OPTIMIZATION WITH WALK-FORWARD VALIDATION")
    print("=" * 90)
    print()
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("  Fetching 1 year of daily data...")
    dfs = {}
    for sym in [SYMBOL1, SYMBOL2]:
        data = vbt.YFData.pull(sym, start=start_date.strftime("%Y-%m-%d"), 
                               end=end_date.strftime("%Y-%m-%d"), silence_warnings=True)
        df = data.get()
        df.columns = [c.lower() for c in df.columns]
        dfs[sym] = df
    
    prices = pd.concat(dfs, axis=1, keys=dfs.keys()).ffill().dropna()
    total_bars = len(prices)
    print(f"  Total bars: {total_bars}")
    print()
    
    # Split data 70/30
    split_idx = int(total_bars * 0.70)
    train_prices = prices.iloc[:split_idx].copy()
    test_prices = prices.iloc[split_idx:].copy()
    
    train_start = train_prices.index[0].strftime("%Y-%m-%d")
    train_end = train_prices.index[-1].strftime("%Y-%m-%d")
    test_start = test_prices.index[0].strftime("%Y-%m-%d")
    test_end = test_prices.index[-1].strftime("%Y-%m-%d")
    
    print("  WALK-FORWARD SPLIT")
    print("  " + "-" * 86)
    print(f"  Training Period:   {train_start} to {train_end} ({len(train_prices)} bars, 70%)")
    print(f"  Testing Period:    {test_start} to {test_end} ({len(test_prices)} bars, 30%)")
    print()
    
    # Parameter grid
    param_combinations = list(product(ZSCORE_WINDOWS, ENTRY_THRESHOLDS, EXIT_THRESHOLDS))
    print(f"  Testing {len(param_combinations)} parameter combinations...")
    print()
    
    # Phase 1: Optimize on training data
    print("=" * 90)
    print("  PHASE 1: IN-SAMPLE OPTIMIZATION (Training Data)")
    print("=" * 90)
    print()
    
    train_results = []
    for window, entry, exit_th in param_combinations:
        result = run_backtest(train_prices, window, entry, exit_th)
        if result and result['num_trades'] >= 3:  # Minimum trades filter
            train_results.append({
                'window': window,
                'entry': entry,
                'exit': exit_th,
                **result
            })
    
    # Sort by Sharpe
    train_results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    print("  Top 10 In-Sample Results:")
    print("  " + "-" * 86)
    print(f"  {'Window':>6} {'Entry':>6} {'Exit':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}")
    print("  " + "-" * 86)
    
    for r in train_results[:10]:
        print(f"  {r['window']:>6} {r['entry']:>6.2f} {r['exit']:>6.2f} {r['total_return']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}% {r['num_trades']:>7} {r['win_rate']*100:>7.1f}%")
    
    print()
    
    # Phase 2: Validate top 5 on test data
    print("=" * 90)
    print("  PHASE 2: OUT-OF-SAMPLE VALIDATION (Test Data)")
    print("=" * 90)
    print()
    
    top5_params = train_results[:5]
    validation_results = []
    
    print("  Testing top 5 parameters on unseen data...")
    print()
    print(f"  {'Window':>6} {'Entry':>6} {'Exit':>6} | {'Train Ret':>10} {'Train Sharpe':>12} | {'Test Ret':>10} {'Test Sharpe':>11} | {'Stable?':>8}")
    print("  " + "-" * 100)
    
    for params in top5_params:
        window, entry, exit_th = params['window'], params['entry'], params['exit']
        
        # Run on test data
        test_result = run_backtest(test_prices, window, entry, exit_th)
        
        if test_result:
            train_sharpe = params['sharpe']
            test_sharpe = test_result['sharpe']
            
            # Stability check: test Sharpe should be at least 50% of train Sharpe
            # and both should be positive
            is_stable = (test_sharpe > 0) and (train_sharpe > 0) and (test_sharpe >= train_sharpe * 0.5)
            stable_str = "YES" if is_stable else "NO"
            
            validation_results.append({
                'window': window,
                'entry': entry,
                'exit': exit_th,
                'train_return': params['total_return'],
                'train_sharpe': train_sharpe,
                'train_trades': params['num_trades'],
                'test_return': test_result['total_return'],
                'test_sharpe': test_sharpe,
                'test_trades': test_result['num_trades'],
                'test_max_dd': test_result['max_dd'],
                'test_win_rate': test_result['win_rate'],
                'is_stable': is_stable,
            })
            
            print(f"  {window:>6} {entry:>6.2f} {exit_th:>6.2f} | {params['total_return']*100:>+9.2f}% {train_sharpe:>12.2f} | {test_result['total_return']*100:>+9.2f}% {test_sharpe:>11.2f} | {stable_str:>8}")
    
    print()
    
    # Phase 3: Final recommendation
    print("=" * 90)
    print("  PHASE 3: FINAL RECOMMENDATION")
    print("=" * 90)
    print()
    
    # Filter stable results and sort by test Sharpe
    stable_results = [r for r in validation_results if r['is_stable']]
    
    if stable_results:
        stable_results.sort(key=lambda x: x['test_sharpe'], reverse=True)
        best = stable_results[0]
        
        print("  RECOMMENDED PARAMETERS (Passed Stability Check)")
        print("  " + "-" * 86)
        print()
        print(f"    Z-Score Window:     {best['window']} days")
        print(f"    Entry Threshold:    {best['entry']:.2f} (enter when |Z| > {best['entry']:.2f})")
        print(f"    Exit Threshold:     {best['exit']:.2f} (exit when |Z| < {best['exit']:.2f})")
        print()
        print("    IN-SAMPLE (Training) Performance:")
        print(f"      Return:           {best['train_return']*100:+.2f}%")
        print(f"      Sharpe:           {best['train_sharpe']:.2f}")
        print(f"      Trades:           {best['train_trades']}")
        print()
        print("    OUT-OF-SAMPLE (Test) Performance:")
        print(f"      Return:           {best['test_return']*100:+.2f}%")
        print(f"      Sharpe:           {best['test_sharpe']:.2f}")
        print(f"      Max Drawdown:     {best['test_max_dd']*100:.2f}%")
        print(f"      Trades:           {best['test_trades']}")
        print(f"      Win Rate:         {best['test_win_rate']*100:.1f}%")
        print()
        
        # Run full backtest with best params
        print("  FULL PERIOD BACKTEST (with recommended parameters)")
        print("  " + "-" * 86)
        full_result = run_backtest(prices, best['window'], best['entry'], best['exit'])
        
        if full_result:
            net_pnl = full_result['total_return'] * INITIAL_CAPITAL
            final_equity = INITIAL_CAPITAL + net_pnl
            
            print()
            print(f"    Period:             {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"    Initial Capital:    ${INITIAL_CAPITAL:,.2f}")
            print(f"    Final Equity:       ${final_equity:,.2f}")
            print(f"    Net Return:         {full_result['total_return']*100:+.2f}%")
            print(f"    Sharpe Ratio:       {full_result['sharpe']:.2f}")
            print(f"    Sortino Ratio:      {full_result['sortino']:.2f}")
            print(f"    Max Drawdown:       {full_result['max_dd']*100:.2f}%")
            print(f"    Total Trades:       {full_result['num_trades']}")
            print(f"    Win Rate:           {full_result['win_rate']*100:.1f}%")
            print(f"    Profit Factor:      {full_result['profit_factor']:.2f}")
        
    else:
        print("  WARNING: No parameters passed stability check!")
        print()
        print("  This suggests the in-sample results may be overfit.")
        print("  The strategy may not be robust for live trading.")
        print()
        
        # Show best test result anyway
        if validation_results:
            validation_results.sort(key=lambda x: x['test_sharpe'], reverse=True)
            best_test = validation_results[0]
            print("  Best test performance (but failed stability):")
            print(f"    Window: {best_test['window']}, Entry: {best_test['entry']}, Exit: {best_test['exit']}")
            print(f"    Test Sharpe: {best_test['test_sharpe']:.2f}, Test Return: {best_test['test_return']*100:+.2f}%")
    
    print()
    print("=" * 90)
    print("  OVERFITTING ANALYSIS")
    print("=" * 90)
    print()
    
    if validation_results:
        avg_train_sharpe = np.mean([r['train_sharpe'] for r in validation_results])
        avg_test_sharpe = np.mean([r['test_sharpe'] for r in validation_results])
        degradation = (avg_train_sharpe - avg_test_sharpe) / avg_train_sharpe * 100 if avg_train_sharpe > 0 else 0
        
        print(f"  Average In-Sample Sharpe:    {avg_train_sharpe:.2f}")
        print(f"  Average Out-of-Sample Sharpe: {avg_test_sharpe:.2f}")
        print(f"  Performance Degradation:      {degradation:.1f}%")
        print()
        
        if degradation < 30:
            print("  VERDICT: Low degradation (<30%) - Parameters appear robust")
        elif degradation < 50:
            print("  VERDICT: Moderate degradation (30-50%) - Some overfitting likely")
        else:
            print("  VERDICT: High degradation (>50%) - Significant overfitting detected")
    
    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
