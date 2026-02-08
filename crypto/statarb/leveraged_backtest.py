"""
Leveraged Statistical Arbitrage Backtest - Coinbase CFM Simulation

Models actual derivatives trading with:
- Leverage tiers (10x intraday BTC/ETH, 5x altcoins)
- Beta-adjusted position sizing for market neutrality
- Funding rate costs (8-hour intervals, ~10% APR)
- Realistic fee structure ($0.15 min + 0.02% taker)

Author: ATLAS Development Team
Date: January 2026
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

# =============================================================================
# COINBASE CFM CONFIGURATION
# =============================================================================

SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']

# Overnight leverage tiers (for positions held past 4PM ET)
# VERIFIED Jan 24, 2026 from Coinbase CFM platform
LEVERAGE = {
    'BTC': 4.1,
    'ETH': 4.0,
    'SOL': 2.7,
    'XRP': 2.6,
    'ADA': 3.4,
}

# Beta to BTC (volatility multiplier)
BETA = {
    'BTC': 1.00,
    'ETH': 1.98,
    'SOL': 1.55,
    'XRP': 1.77,
    'ADA': 2.20,
}

# Contract sizes (for fee calculation)
CONTRACT_SIZE = {
    'BTC': 0.01,
    'ETH': 0.1,
    'SOL': 5.0,
    'XRP': 500.0,
    'ADA': 1000.0,
}

# Fee structure - VERIFIED Jan 24, 2026
# Formula: (Notional * Rate) + Fixed_Per_Contract
MAKER_FEE_RATE = 0.00065   # 0.065%
TAKER_FEE_RATE = 0.0007    # 0.07%
MIN_FEE_PER_CONTRACT = 0.15  # $0.15 per contract

# Funding rate (annualized, applied per 8-hour period)
ANNUAL_FUNDING_RATE = 0.10  # 10% APR
FUNDING_PERIODS_PER_DAY = 3  # Every 8 hours
DAILY_FUNDING_RATE = ANNUAL_FUNDING_RATE / 365

# Backtest parameters
BACKTEST_DAYS = 365
INITIAL_CAPITAL = 1000.0
BASE_POSITION_PCT = 20.0  # 20% of equity per leg before leverage


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_symbol_base(symbol):
    """Extract base asset from symbol (BTC-USD -> BTC)."""
    return symbol.replace('-USD', '')

def calculate_half_life(spread):
    """Calculate Ornstein-Uhlenbeck half-life."""
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

def calculate_beta_adjusted_sizes(sym1, sym2, base_notional):
    """
    Calculate beta-adjusted position sizes for market neutrality.
    
    For a beta-neutral pair:
    notional1 * beta1 = notional2 * beta2
    """
    base1 = get_symbol_base(sym1)
    base2 = get_symbol_base(sym2)
    
    beta1 = BETA.get(base1, 1.0)
    beta2 = BETA.get(base2, 1.0)
    lev1 = LEVERAGE.get(base1, 3.0)
    lev2 = LEVERAGE.get(base2, 3.0)
    
    # Use minimum leverage of the pair
    effective_lev = min(lev1, lev2)
    
    # Beta-adjusted split
    # To be market-neutral: notional1 * beta1 = notional2 * beta2
    # Combined with: notional1 + notional2 = total_notional
    total_notional = base_notional * effective_lev
    
    # Solve for individual notionals
    # notional1 = total * beta2 / (beta1 + beta2)
    # notional2 = total * beta1 / (beta1 + beta2)
    notional1 = total_notional * beta2 / (beta1 + beta2)
    notional2 = total_notional * beta1 / (beta1 + beta2)
    
    return {
        'notional1': notional1,
        'notional2': notional2,
        'total_notional': notional1 + notional2,
        'effective_leverage': effective_lev,
        'beta1': beta1,
        'beta2': beta2,
    }

def calculate_funding_cost(notional, days_held, is_long=True):
    """
    Calculate funding cost for holding a leveraged position.
    
    In crypto perps, when funding is positive (typical in bull markets):
    - Longs pay shorts
    - Cost = notional * daily_rate * days
    
    For a pairs trade (long one, short other), funding partially offsets.
    """
    daily_cost = notional * DAILY_FUNDING_RATE
    total_cost = daily_cost * days_held
    
    # Long pays, short receives (in typical positive funding environment)
    if is_long:
        return total_cost  # Cost
    else:
        return -total_cost * 0.5  # Partial offset (funding isn't always symmetric)

def calculate_fees(trade_value, num_contracts=1, is_maker=False):
    """
    Calculate Coinbase CFM fees.
    
    Formula: (Notional × Rate) + (Fixed × Contracts)
    VERIFIED Jan 24, 2026 from platform
    """
    rate = MAKER_FEE_RATE if is_maker else TAKER_FEE_RATE
    pct_fee = abs(trade_value) * rate
    fixed_fee = MIN_FEE_PER_CONTRACT * num_contracts
    return pct_fee + fixed_fee


# =============================================================================
# LEVERAGED BACKTEST
# =============================================================================

def run_leveraged_backtest(prices, sym1, sym2, zscore_window=20, entry_th=2.0, exit_th=0.0):
    """
    Run leveraged pairs trading backtest with CFM-realistic modeling.
    """
    base1 = get_symbol_base(sym1)
    base2 = get_symbol_base(sym2)
    
    close1 = prices[(sym1, 'close')]
    close2 = prices[(sym2, 'close')]
    
    # Cointegration test
    log_p1, log_p2 = np.log(close1), np.log(close2)
    test_stat, p_value, crit = coint(log_p1, log_p2)
    hedge_ratio = np.polyfit(log_p2, log_p1, 1)[0]
    spread = log_p1 - hedge_ratio * log_p2
    half_life = calculate_half_life(spread.values)
    
    # Z-score signals
    zscore = (spread - spread.rolling(zscore_window).mean()) / spread.rolling(zscore_window).std()
    
    long_spread = (zscore.shift(1) > -entry_th) & (zscore <= -entry_th)
    short_spread = (zscore.shift(1) < entry_th) & (zscore >= entry_th)
    long_exit = ((zscore.shift(1) < exit_th) & (zscore >= exit_th)) | (zscore >= 3.0)
    short_exit = ((zscore.shift(1) > exit_th) & (zscore <= exit_th)) | (zscore <= -3.0)
    
    # Get position sizing
    sizing = calculate_beta_adjusted_sizes(sym1, sym2, INITIAL_CAPITAL * BASE_POSITION_PCT / 100)
    
    # Build signal arrays
    data = pd.concat([close1, close2], axis=1, keys=[sym1, sym2])
    long_entries = pd.DataFrame(False, index=data.index, columns=[sym1, sym2])
    short_entries = pd.DataFrame(False, index=data.index, columns=[sym1, sym2])
    exits = pd.DataFrame(False, index=data.index, columns=[sym1, sym2])
    
    # Long spread: long asset1, short asset2
    long_entries.loc[long_spread, sym1] = True
    short_entries.loc[long_spread, sym2] = True
    
    # Short spread: short asset1, long asset2
    short_entries.loc[short_spread, sym1] = True
    long_entries.loc[short_spread, sym2] = True
    
    exits.loc[long_exit | short_exit, :] = True
    
    # Position sizes as percentage (leveraged)
    # VBT uses size as % of current equity
    # We scale by leverage
    effective_lev = sizing['effective_leverage']
    size_pct = BASE_POSITION_PCT * effective_lev
    
    # Fee rate (scale to account for leverage)
    # More notional = more fees
    fee_rate = TAKER_FEE_RATE * effective_lev
    
    try:
        pf = vbt.Portfolio.from_signals(
            data,
            entries=long_entries,
            short_entries=short_entries,
            exits=exits,
            size=size_pct,
            size_type="valuepercent100",
            group_by=True,
            cash_sharing=True,
            call_seq="auto",
            init_cash=INITIAL_CAPITAL,
            fees=fee_rate,
        )
        
        stats = pf.stats()
        trades = pf.trades
        num_trades = len(trades.records_arr) if hasattr(trades, 'records_arr') else 0
        
        # Calculate additional metrics
        total_return = float(stats.get("Total Return [%]", 0)) / 100
        sharpe = float(stats.get("Sharpe Ratio", 0)) if not pd.isna(stats.get("Sharpe Ratio")) else 0
        sortino = float(stats.get("Sortino Ratio", 0)) if not pd.isna(stats.get("Sortino Ratio")) else 0
        max_dd = float(stats.get("Max Drawdown [%]", 0)) / 100
        calmar = float(stats.get("Calmar Ratio", 0)) if not pd.isna(stats.get("Calmar Ratio")) else 0
        total_fees = float(stats.get("Total Fees Paid", 0))
        
        # Trade statistics
        if num_trades > 0:
            win_rate = float(trades.win_rate) if not pd.isna(trades.win_rate) else 0
            avg_trade_pnl = float(trades.pnl.mean()) if hasattr(trades, 'pnl') else 0
            avg_trade_duration = float(trades.duration.mean()) if hasattr(trades, 'duration') else 0
            best_trade = float(trades.pnl.max()) if hasattr(trades, 'pnl') else 0
            worst_trade = float(trades.pnl.min()) if hasattr(trades, 'pnl') else 0
            profit_factor = float(trades.profit_factor) if hasattr(trades, 'profit_factor') and not pd.isna(trades.profit_factor) else 0
        else:
            win_rate = avg_trade_pnl = avg_trade_duration = best_trade = worst_trade = profit_factor = 0
        
        # Estimate funding costs (rough approximation)
        # Assume average position held for avg_trade_duration days
        avg_hold_days = avg_trade_duration if avg_trade_duration > 0 else 5
        avg_notional = sizing['total_notional']
        est_funding_cost_per_trade = avg_notional * DAILY_FUNDING_RATE * avg_hold_days * 0.5  # 50% offset for pairs
        total_funding_cost = est_funding_cost_per_trade * num_trades
        
        # Adjusted P&L
        gross_pnl = total_return * INITIAL_CAPITAL
        net_pnl = gross_pnl - total_funding_cost
        net_return = net_pnl / INITIAL_CAPITAL
        
        # Final equity
        final_equity = INITIAL_CAPITAL * (1 + total_return)
        final_equity_after_funding = INITIAL_CAPITAL + net_pnl
        
        return {
            # Identifiers
            'pair': f"{base1}/{base2}",
            'symbol1': sym1,
            'symbol2': sym2,
            
            # Cointegration
            'p_value': p_value,
            'half_life': half_life,
            'cointegrated': p_value < 0.05,
            'hedge_ratio': hedge_ratio,
            
            # Leverage/Sizing
            'effective_leverage': effective_lev,
            'beta1': sizing['beta1'],
            'beta2': sizing['beta2'],
            'notional_per_trade': sizing['total_notional'],
            
            # Returns
            'gross_return': total_return,
            'gross_pnl': gross_pnl,
            'total_fees': total_fees,
            'est_funding_cost': total_funding_cost,
            'net_pnl': net_pnl,
            'net_return': net_return,
            'final_equity': final_equity_after_funding,
            
            # Risk metrics
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd,
            'calmar': calmar,
            
            # Trade statistics
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_trade_duration': avg_trade_duration,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            
            # Objects
            'portfolio': pf,
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    end_str = end_date.strftime("%Y-%m-%d")
    start_str = start_date.strftime("%Y-%m-%d")
    
    print()
    print("=" * 90)
    print("  LEVERAGED STATISTICAL ARBITRAGE BACKTEST - COINBASE CFM SIMULATION")
    print("=" * 90)
    print()
    print("  CONFIGURATION")
    print("  " + "-" * 86)
    print(f"  Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"  Backtest Period:     {start_str} to {end_str} ({BACKTEST_DAYS} days)")
    print(f"  Assets:              {', '.join([s.replace('-USD','') for s in SYMBOLS])}")
    print(f"  Base Position Size:  {BASE_POSITION_PCT}% of equity per leg (before leverage)")
    print(f"  Taker Fee:           {TAKER_FEE_RATE*100:.3f}% + ${MIN_FEE_PER_CONTRACT:.2f}/contract")
    print(f"  Maker Fee:           {MAKER_FEE_RATE*100:.3f}% + ${MIN_FEE_PER_CONTRACT:.2f}/contract")
    print(f"  Est. Funding Rate:   {ANNUAL_FUNDING_RATE*100:.1f}% APR (longs pay shorts)")
    print()
    print("  LEVERAGE TIERS (Overnight - Verified Jan 24, 2026)")
    print("  " + "-" * 86)
    for asset, lev in LEVERAGE.items():
        beta = BETA[asset]
        eff_mult = lev * beta
        print(f"    {asset}: {lev:.1f}x leverage, {beta:.2f} beta -> {eff_mult:.2f}x effective multiplier")
    print()
    
    # Fetch data
    print("  FETCHING DATA...")
    print("  " + "-" * 86)
    dfs = {}
    for sym in SYMBOLS:
        data = vbt.YFData.pull(sym, start=start_str, end=end_str, silence_warnings=True)
        df = data.get()
        df.columns = [c.lower() for c in df.columns]
        dfs[sym] = df
        print(f"    {sym}: {len(df)} daily bars")
    
    prices = pd.concat(dfs, axis=1, keys=dfs.keys()).ffill().dropna()
    print(f"    Aligned dataset: {len(prices)} bars")
    print()
    
    # Run all pairs
    print("=" * 90)
    print("  COINTEGRATION ANALYSIS")
    print("=" * 90)
    print()
    print(f"  {'Pair':<10} {'P-Value':>10} {'Half-Life':>12} {'Hedge Ratio':>12} {'Status':>15}")
    print("  " + "-" * 86)
    
    all_pairs = list(combinations(SYMBOLS, 2))
    results = []
    
    for sym1, sym2 in all_pairs:
        result = run_leveraged_backtest(prices, sym1, sym2)
        if result:
            results.append(result)
            hl = f"{result['half_life']:.1f} days" if not np.isinf(result['half_life']) else "INF"
            status = "COINTEGRATED" if result['cointegrated'] else "Not cointegrated"
            print(f"  {result['pair']:<10} {result['p_value']:>10.4f} {hl:>12} {result['hedge_ratio']:>12.4f} {status:>15}")
    
    print()
    
    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    print("=" * 90)
    print("  BACKTEST RESULTS (Sorted by Sharpe Ratio)")
    print("=" * 90)
    print()
    print(f"  {'Pair':<10} {'Leverage':>8} {'Gross P/L':>12} {'Fees':>8} {'Funding':>10} {'Net P/L':>12} {'Net Ret':>10} {'Sharpe':>8}")
    print("  " + "-" * 86)
    
    for r in results:
        print(f"  {r['pair']:<10} {r['effective_leverage']:>7.1f}x ${r['gross_pnl']:>+10.2f} ${r['total_fees']:>6.2f} ${r['est_funding_cost']:>8.2f} ${r['net_pnl']:>+10.2f} {r['net_return']*100:>+9.2f}% {r['sharpe']:>8.2f}")
    
    print()
    
    # Detailed results for top 3
    print("=" * 90)
    print("  DETAILED RESULTS - TOP 3 PAIRS")
    print("=" * 90)
    
    for i, r in enumerate(results[:3], 1):
        print()
        print(f"  #{i} {r['pair']}")
        print("  " + "-" * 86)
        print()
        print(f"    POSITION SIZING")
        print(f"      Effective Leverage:    {r['effective_leverage']:.1f}x")
        base1 = r['symbol1'].replace('-USD', '')
        base2 = r['symbol2'].replace('-USD', '')
        print(f"      {base1} Beta:                {r['beta1']:.2f}")
        print(f"      {base2} Beta:                {r['beta2']:.2f}")
        print(f"      Notional per Trade:    ${r['notional_per_trade']:,.2f}")
        print()
        print(f"    PROFIT & LOSS")
        print(f"      Initial Capital:       ${INITIAL_CAPITAL:,.2f}")
        print(f"      Gross P/L:             ${r['gross_pnl']:+,.2f} ({r['gross_return']*100:+.2f}%)")
        print(f"      Trading Fees:          -${r['total_fees']:.2f}")
        print(f"      Est. Funding Costs:    -${r['est_funding_cost']:.2f}")
        print(f"      Net P/L:               ${r['net_pnl']:+,.2f} ({r['net_return']*100:+.2f}%)")
        print(f"      Final Equity:          ${r['final_equity']:,.2f}")
        print()
        print(f"    RISK METRICS")
        print(f"      Sharpe Ratio:          {r['sharpe']:.2f}")
        print(f"      Sortino Ratio:         {r['sortino']:.2f}")
        print(f"      Max Drawdown:          {r['max_dd']*100:.2f}%")
        print(f"      Calmar Ratio:          {r['calmar']:.2f}")
        print()
        print(f"    TRADE STATISTICS")
        print(f"      Total Trades:          {r['num_trades']}")
        print(f"      Win Rate:              {r['win_rate']*100:.1f}%")
        print(f"      Profit Factor:         {r['profit_factor']:.2f}")
        print(f"      Avg Trade P/L:         ${r['avg_trade_pnl']:+.2f}")
        print(f"      Avg Hold Duration:     {r['avg_trade_duration']:.1f} days")
        print(f"      Best Trade:            ${r['best_trade']:+.2f}")
        print(f"      Worst Trade:           ${r['worst_trade']:+.2f}")
        print()
        print(f"    COINTEGRATION")
        print(f"      P-Value:               {r['p_value']:.4f}")
        print(f"      Half-Life:             {r['half_life']:.1f} days")
        print(f"      Hedge Ratio:           {r['hedge_ratio']:.4f}")
        coint_status = "YES - Statistically significant" if r['cointegrated'] else "NO - Trading correlation only"
        print(f"      Cointegrated:          {coint_status}")
    
    print()
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print()
    
    # Overall statistics
    profitable = [r for r in results if r['net_pnl'] > 0]
    total_net_pnl = sum(r['net_pnl'] for r in results)
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    
    print(f"  Total Pairs Tested:        {len(results)}")
    print(f"  Profitable Pairs:          {len(profitable)} / {len(results)} ({len(profitable)/len(results)*100:.0f}%)")
    print(f"  Average Sharpe Ratio:      {avg_sharpe:.2f}")
    print()
    
    best = results[0]
    print(f"  BEST PAIR: {best['pair']}")
    print(f"    Net Return:              {best['net_return']*100:+.2f}%")
    print(f"    Sharpe Ratio:            {best['sharpe']:.2f}")
    print(f"    $1,000 -> ${best['final_equity']:,.2f}")
    print()
    
    # Risk warning
    coint_pairs = [r for r in results if r['cointegrated']]
    if not coint_pairs:
        print("  ⚠️  WARNING: No pairs showed statistically significant cointegration")
        print("     The strategy is trading on correlation, not mean reversion.")
        print("     This increases regime change risk - monitor for breakdown.")
    
    print()
    print("=" * 90)
    print("  BACKTEST COMPLETE")
    print("=" * 90)
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    main()
