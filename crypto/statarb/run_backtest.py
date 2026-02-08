"""
Comprehensive Statistical Arbitrage Pairs Trading Backtest Runner.

Tests cointegration pairs trading strategies across crypto pairs:
- BTC-ETH (most liquid, strongest expected cointegration)
- BTC-SOL
- ETH-SOL

Uses VectorBT Pro for simulation with Coinbase CFM fee modeling.

Usage:
    python -m crypto.statarb.run_backtest

Author: ATLAS Development Team
Date: January 2026
"""

import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# VectorBT Pro
try:
    import vectorbtpro as vbt
    logger.info(f"VectorBT Pro version: {vbt.__version__}")
except ImportError:
    logger.error("VectorBT Pro not installed. Run: pip install vectorbtpro")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Pairs to test - Yahoo Finance format
PAIRS_TO_TEST = [
    ("BTC-USD", "ETH-USD"),  # Strongest expected cointegration
    ("BTC-USD", "SOL-USD"),
    ("ETH-USD", "SOL-USD"),
]

# Backtest periods
BACKTEST_DAYS = 365  # 1 year (YF has good daily data)

# Z-score parameter grids for optimization
ZSCORE_WINDOWS = [10, 20, 30, 60]  # days for daily timeframe
ENTRY_THRESHOLDS = [1.5, 2.0, 2.5]
EXIT_THRESHOLDS = [0.0, 0.25, 0.5]

# Capital and position sizing
INITIAL_CAPITAL = 1000.0  # $1000 starting capital
POSITION_SIZE_PCT = 10.0  # 10% per leg (20% total exposure)

# Coinbase CFM fee structure
TAKER_FEE_RATE = 0.0002  # 0.02%
MIN_FEE_PER_CONTRACT = 0.15  # $0.15 minimum


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_yahoo_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance.
    
    Yahoo Finance works well for crypto daily data.
    """
    logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
    
    dfs = {}
    
    for symbol in symbols:
        try:
            data = vbt.YFData.pull(
                symbol,
                start=start_date,
                end=end_date,
                silence_warnings=True,
            )
            df = data.get()
            
            if df is not None and not df.empty:
                # Rename columns to standard format
                df.columns = [c.lower() for c in df.columns]
                dfs[symbol] = df
                logger.info(f"  {symbol}: {len(df)} bars fetched")
            else:
                logger.warning(f"  {symbol}: No data returned")
                
        except Exception as e:
            logger.error(f"  {symbol}: Failed to fetch - {e}")
            continue
    
    if not dfs:
        raise ValueError("No data fetched for any symbols")
    
    # Combine into multi-column DataFrame
    combined = pd.concat(dfs, axis=1, keys=dfs.keys())
    
    # Align timestamps (some symbols may have gaps)
    combined = combined.ffill()
    combined = combined.dropna()
    
    logger.info(f"Combined dataset: {len(combined)} aligned bars")
    
    return combined


# =============================================================================
# COINTEGRATION ANALYSIS
# =============================================================================

def test_cointegration(
    price1: pd.Series,
    price2: pd.Series,
    significance_level: float = 0.05,
) -> Dict:
    """
    Run Engle-Granger cointegration test.
    
    Returns dict with p-value, hedge ratio, half-life.
    """
    from statsmodels.tsa.stattools import coint
    
    # Use log prices
    log_p1 = np.log(price1)
    log_p2 = np.log(price2)
    
    # Run cointegration test
    test_stat, p_value, crit_values = coint(log_p1, log_p2)
    
    # Calculate hedge ratio via OLS
    hedge_ratio = np.polyfit(log_p2, log_p1, 1)[0]
    
    # Calculate spread
    spread = log_p1 - hedge_ratio * log_p2
    
    # Calculate half-life
    half_life = calculate_half_life(spread.values)
    
    return {
        "p_value": p_value,
        "test_statistic": test_stat,
        "critical_values": {
            "1%": crit_values[0],
            "5%": crit_values[1],
            "10%": crit_values[2],
        },
        "hedge_ratio": hedge_ratio,
        "half_life": half_life,
        "is_cointegrated": p_value < significance_level,
    }


def calculate_half_life(spread: np.ndarray) -> float:
    """
    Calculate Ornstein-Uhlenbeck half-life.
    """
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
        
    except Exception:
        return np.nan


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def run_single_backtest(
    prices: pd.DataFrame,
    symbol1: str,
    symbol2: str,
    zscore_window: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    hedge_ratio: Optional[float] = None,
    include_fees: bool = True,
) -> Dict:
    """
    Run a single pairs trading backtest.
    
    Returns dict with performance metrics.
    """
    # Extract close prices
    close1 = prices[(symbol1, "close")]
    close2 = prices[(symbol2, "close")]
    
    # Test cointegration
    coint_result = test_cointegration(close1, close2)
    
    # Use calculated hedge ratio if not provided
    if hedge_ratio is None:
        hedge_ratio = coint_result["hedge_ratio"]
    
    # Calculate spread (log prices)
    log_p1 = np.log(close1)
    log_p2 = np.log(close2)
    spread = log_p1 - hedge_ratio * log_p2
    
    # Calculate rolling Z-score
    zscore_mean = spread.rolling(window=zscore_window).mean()
    zscore_std = spread.rolling(window=zscore_window).std()
    zscore = (spread - zscore_mean) / zscore_std
    
    # Generate signals
    # Long spread: Z crosses below -entry_threshold
    long_spread_entry = (zscore.shift(1) > -entry_threshold) & (zscore <= -entry_threshold)
    
    # Short spread: Z crosses above +entry_threshold
    short_spread_entry = (zscore.shift(1) < entry_threshold) & (zscore >= entry_threshold)
    
    # Exits
    long_exit = (zscore.shift(1) < exit_threshold) & (zscore >= exit_threshold)
    long_exit |= zscore >= stop_threshold
    
    short_exit = (zscore.shift(1) > exit_threshold) & (zscore <= exit_threshold)
    short_exit |= zscore <= -stop_threshold
    
    # Build signal arrays for VBT
    data = pd.concat([close1, close2], axis=1, keys=[symbol1, symbol2])
    
    long_entries = data.copy()
    long_entries[:] = False
    short_entries = data.copy()
    short_entries[:] = False
    exits = data.copy()
    exits[:] = False
    
    # Long spread: long asset1, short asset2
    long_entries.loc[long_spread_entry, symbol1] = True
    short_entries.loc[long_spread_entry, symbol2] = True
    
    # Short spread: short asset1, long asset2
    short_entries.loc[short_spread_entry, symbol1] = True
    long_entries.loc[short_spread_entry, symbol2] = True
    
    # Exits
    exits.loc[long_exit | short_exit, :] = True
    
    # Fee calculation
    if include_fees:
        # Use simple percentage fee for compatibility
        fee_pct = TAKER_FEE_RATE
    else:
        fee_pct = 0.0
    
    # Run VBT simulation
    try:
        pf = vbt.Portfolio.from_signals(
            data,
            entries=long_entries,
            short_entries=short_entries,
            exits=exits,
            size=POSITION_SIZE_PCT,
            size_type="valuepercent100",
            group_by=True,
            cash_sharing=True,
            call_seq="auto",
            init_cash=INITIAL_CAPITAL,
            fees=fee_pct,
        )
    except Exception as e:
        logger.error(f"Portfolio simulation failed: {e}")
        return None
    
    # Extract metrics
    try:
        stats = pf.stats()
        
        # Trade metrics
        trades = pf.trades
        num_trades = len(trades.records_arr) if hasattr(trades, 'records_arr') else 0
        
        if num_trades > 0:
            try:
                win_rate = float(trades.win_rate)
            except (ValueError, TypeError, AttributeError):
                win_rate = 0.0
            try:
                expectancy = float(trades.expectancy)
            except (ValueError, TypeError, AttributeError):
                expectancy = 0.0
        else:
            win_rate = 0.0
            expectancy = 0.0
        
        result = {
            # Parameters
            "symbol1": symbol1,
            "symbol2": symbol2,
            "zscore_window": zscore_window,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "hedge_ratio": hedge_ratio,
            
            # Cointegration
            "coint_p_value": coint_result["p_value"],
            "half_life": coint_result["half_life"],
            "is_cointegrated": coint_result["is_cointegrated"],
            
            # Performance
            "total_return": float(stats.get("Total Return [%]", 0)) / 100,
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0)) if not pd.isna(stats.get("Sharpe Ratio", 0)) else 0.0,
            "sortino_ratio": float(stats.get("Sortino Ratio", 0)) if not pd.isna(stats.get("Sortino Ratio", 0)) else 0.0,
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0)) / 100,
            "calmar_ratio": float(stats.get("Calmar Ratio", 0)) if not pd.isna(stats.get("Calmar Ratio", 0)) else 0.0,
            
            # Trades
            "num_trades": num_trades,
            "win_rate": float(win_rate) if not np.isnan(win_rate) else 0.0,
            "expectancy": float(expectancy) if not np.isnan(expectancy) else 0.0,
            "total_fees": float(stats.get("Total Fees Paid", 0)),
            
            # Objects for further analysis
            "portfolio": pf,
            "zscore": zscore,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_parameter_scan(
    prices: pd.DataFrame,
    symbol1: str,
    symbol2: str,
    zscore_windows: List[int] = ZSCORE_WINDOWS,
    entry_thresholds: List[float] = ENTRY_THRESHOLDS,
    exit_thresholds: List[float] = EXIT_THRESHOLDS,
) -> pd.DataFrame:
    """
    Run parameter optimization scan for a single pair.
    """
    results = []
    
    total_combos = len(zscore_windows) * len(entry_thresholds) * len(exit_thresholds)
    count = 0
    
    for window in zscore_windows:
        for entry in entry_thresholds:
            for exit_th in exit_thresholds:
                count += 1
                
                result = run_single_backtest(
                    prices=prices,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    zscore_window=window,
                    entry_threshold=entry,
                    exit_threshold=exit_th,
                )
                
                if result is not None:
                    # Remove non-serializable objects
                    result_clean = {k: v for k, v in result.items() 
                                   if k not in ["portfolio", "zscore"]}
                    results.append(result_clean)
    
    df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    if not df.empty:
        df = df.sort_values("sharpe_ratio", ascending=False)
    
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_separator(char: str = "=", length: int = 70):
    print(char * length)


def print_header(text: str):
    print_separator()
    print(f"  {text}")
    print_separator()


def main():
    """Main execution function."""
    
    print_header("STATISTICAL ARBITRAGE PAIRS TRADING BACKTEST")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pairs to Test: {len(PAIRS_TO_TEST)}")
    print(f"Backtest Period: {BACKTEST_DAYS} days")
    print()
    
    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=BACKTEST_DAYS)).strftime("%Y-%m-%d")
    
    print(f"Date Range: {start_date} to {end_date}")
    print()
    
    # Collect all unique symbols
    all_symbols = list(set(
        sym for pair in PAIRS_TO_TEST for sym in pair
    ))
    
    # Fetch data once for all symbols
    print_header("FETCHING DATA (Yahoo Finance - Daily)")
    
    try:
        prices = fetch_yahoo_data(
            symbols=all_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Phase 1: Test cointegration for all pairs
    print_header("COINTEGRATION ANALYSIS")
    
    coint_results = []
    
    for symbol1, symbol2 in PAIRS_TO_TEST:
        try:
            close1 = prices[(symbol1, "close")]
            close2 = prices[(symbol2, "close")]
            
            result = test_cointegration(close1, close2)
            result["pair"] = f"{symbol1}/{symbol2}"
            coint_results.append(result)
            
            status = "YES" if result["is_cointegrated"] else "NO"
            hl_str = f"{result['half_life']:6.1f}" if not np.isinf(result['half_life']) else "  INF "
            print(f"  {result['pair']:20} | p-value: {result['p_value']:.4f} | "
                  f"Half-life: {hl_str} days | Cointegrated: {status}")
            
        except Exception as e:
            logger.warning(f"  {symbol1}/{symbol2}: Cointegration test failed - {e}")
    
    print()
    
    # Phase 2: Run backtests for all pairs
    print_header("BACKTEST RESULTS (Default Parameters)")
    print("  Z-score Window: 20 days, Entry: 2.0, Exit: 0.0")
    print()
    
    backtest_results = []
    
    for symbol1, symbol2 in PAIRS_TO_TEST:
        print(f"  Testing {symbol1}/{symbol2}...")
        
        result = run_single_backtest(
            prices=prices,
            symbol1=symbol1,
            symbol2=symbol2,
            zscore_window=20,
            entry_threshold=2.0,
            exit_threshold=0.0,
        )
        
        if result is not None:
            backtest_results.append(result)
            
            print(f"    Total Return: {result['total_return']*100:+.2f}%")
            print(f"    Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"    Max Drawdown: {result['max_drawdown']*100:.2f}%")
            print(f"    Num Trades:   {result['num_trades']}")
            print(f"    Win Rate:     {result['win_rate']*100:.1f}%")
            print(f"    Total Fees:   ${result['total_fees']:.2f}")
            print()
        else:
            print(f"    FAILED")
            print()
    
    # Phase 3: Parameter optimization for best pair
    print_header("PARAMETER OPTIMIZATION")
    
    if backtest_results:
        # Find best pair by Sharpe
        best_result = max(backtest_results, key=lambda x: x.get("sharpe_ratio", -999))
        best_pair = (best_result["symbol1"], best_result["symbol2"])
        
        print(f"Optimizing: {best_pair[0]}/{best_pair[1]}")
        print(f"Testing {len(ZSCORE_WINDOWS) * len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)} combinations...")
        print()
        
        opt_df = run_parameter_scan(
            prices=prices,
            symbol1=best_pair[0],
            symbol2=best_pair[1],
        )
        
        if not opt_df.empty:
            print("Top 5 Parameter Combinations:")
            print_separator("-")
            
            top5 = opt_df.head(5)
            for idx, row in top5.iterrows():
                print(f"  Window: {int(row['zscore_window']):3d} | "
                      f"Entry: {row['entry_threshold']:.1f} | "
                      f"Exit: {row['exit_threshold']:.2f} | "
                      f"Sharpe: {row['sharpe_ratio']:.2f} | "
                      f"Return: {row['total_return']*100:+.2f}% | "
                      f"Trades: {int(row['num_trades'])}")
    
    print()
    
    # Summary
    print_header("SUMMARY")
    
    if backtest_results:
        print("\nAll Pairs Results:")
        print_separator("-")
        
        for r in backtest_results:
            coint_status = "Yes" if r.get("is_cointegrated", False) else "No"
            print(f"  {r['symbol1']}/{r['symbol2']:12} | "
                  f"Coint: {coint_status:3} | "
                  f"Return: {r['total_return']*100:+6.2f}% | "
                  f"Sharpe: {r['sharpe_ratio']:5.2f} | "
                  f"MaxDD: {r['max_drawdown']*100:5.1f}% | "
                  f"Trades: {r['num_trades']:3d} | "
                  f"WinRate: {r.get('win_rate', 0)*100:4.1f}%")
    else:
        print("No successful backtests.")
    
    print()
    print_header("BACKTEST COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
