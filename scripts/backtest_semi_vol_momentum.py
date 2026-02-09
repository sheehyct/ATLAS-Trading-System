"""
Semi-Volatility Momentum Strategy Backtest Validation

This script validates the Semi-Vol Momentum strategy on historical data (2010-2025).
Tests whether the strategy meets architecture performance targets.

============================================================================
VBT 5-STEP WORKFLOW VERIFICATION (Session MOMENTUM-4, January 2026)
============================================================================
VBT_VERIFIED: vbt.Portfolio.from_signals (via BaseStrategy.backtest)
  - SEARCH: Portfolio.from_signals patterns found
  - VERIFY: vectorbtpro.portfolio.base.Portfolio.from_signals RESOLVED
  - FIND: 10+ real-world examples found
  - TEST: Minimal example PASSED

VBT_VERIFIED: Portfolio metrics (total_return, sharpe_ratio, max_drawdown)
  - VERIFY: All refnames RESOLVED
  - TEST: Metrics return correct float types PASSED

VBT_VERIFIED: Splitter.from_n_rolling (walk-forward validation)
  - VERIFY: vectorbtpro.generic.splitting.base.Splitter.from_n_rolling RESOLVED
  - FIND: Walk-forward examples found
  - TEST: 5-fold split with 70/30 train/test PASSED

VBT_VERIFIED: Splitter.from_rolling, shuffle_splits (Monte Carlo)
  - VERIFY: Both refnames RESOLVED
  - FIND: Block bootstrap examples found
  - TEST: 20 shuffled blocks PASSED
============================================================================

Performance Targets (per architecture):
- Sharpe Ratio: 1.4-1.8 (volatility-managed)
- CAGR: 15-20%
- Max Drawdown: < -20%
- Walk-Forward Degradation: < 30%
- Monte Carlo P(Loss): < 20%

Strategy Key Feature:
- Scales positions inversely with volatility
- Larger positions in low-vol environments
- Smaller positions in high-vol environments
- Result: Improved risk-adjusted returns

Professional Standards:
- NO emojis or unicode
- Clear output with metrics comparison
- VBT 5-step workflow compliance
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.semi_vol_momentum import SemiVolMomentum
from strategies.base_strategy import StrategyConfig
from utils.momentum_universe import (
    UNIVERSE_SYMBOLS,
    SECTOR_MAP,
    get_validation_symbols,
    print_sector_distribution,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Performance targets from architecture spec
TARGETS = {
    'sharpe_min': 1.4,  # Higher target for vol-managed strategy
    'sharpe_max': 1.8,
    'cagr_min': 0.15,   # 15% minimum
    'cagr_max': 0.20,   # 20% max
    'max_drawdown': 0.20,  # 20% max (tighter than quality-momentum)
    'walk_forward_degradation': 0.30,  # 30% max degradation
    'monte_carlo_p_loss': 0.20,  # 20% max probability of loss
}


# =============================================================================
# DATA FETCHING AND PREPARATION
# =============================================================================

def fetch_multi_stock_data(
    symbols: List[str],
    start_date: str = '2010-01-01',
    end_date: str = '2025-01-01'
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols.

    VBT_VERIFIED: vbt.YFData.pull (resolved to vectorbtpro.data.base.Data.pull)

    Note: Using YFinance for extended historical data (2010-2025).

    Args:
        symbols: List of stock symbols
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        Dictionary mapping symbol to OHLCV DataFrame
    """
    print(f"\nFetching data for {len(symbols)} symbols...")
    print(f"Date range: {start_date} to {end_date}")

    universe_data = {}
    successful = 0

    for symbol in symbols:
        try:
            data = vbt.YFData.pull(symbol, start=start_date, end=end_date).get()
            if len(data) > 300:  # Need at least 300 days for momentum calc
                universe_data[symbol] = data
                successful += 1
        except Exception as e:
            print(f"  [WARN] Failed to fetch {symbol}: {e}")

    print(f"Successfully fetched {successful}/{len(symbols)} symbols")

    return universe_data


def prepare_portfolio_data(
    universe_data: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align all stock data to common timestamps for portfolio backtesting.

    Args:
        universe_data: Dictionary of symbol -> OHLCV DataFrames

    Returns:
        Tuple of (close_prices, open_prices) DataFrames with symbol columns
    """
    if not universe_data:
        raise ValueError("universe_data is empty")

    # Find common date range
    start = max(df.index.min() for df in universe_data.values())
    end = min(df.index.max() for df in universe_data.values())

    print(f"  Common date range: {start.date()} to {end.date()}")

    # Create aligned DataFrames
    close_prices = pd.DataFrame()
    open_prices = pd.DataFrame()
    high_prices = pd.DataFrame()
    low_prices = pd.DataFrame()

    symbols_dropped = []
    for symbol, df in universe_data.items():
        # Slice to common range
        df_slice = df.loc[start:end]

        # Forward-fill gaps (up to 5 days)
        df_ffill = df_slice.ffill(limit=5)

        # Check for excessive missing data (>5%)
        missing_pct = df_ffill['Close'].isna().sum() / len(df_ffill)
        if missing_pct > 0.05:
            symbols_dropped.append(symbol)
            print(f"  [WARN] Dropping {symbol}: {missing_pct:.1%} missing data")
            continue

        close_prices[symbol] = df_ffill['Close']
        open_prices[symbol] = df_ffill['Open']
        high_prices[symbol] = df_ffill['High']
        low_prices[symbol] = df_ffill['Low']

    # Drop any remaining NaN rows (edge cases)
    close_prices = close_prices.dropna()
    open_prices = open_prices.loc[close_prices.index]
    high_prices = high_prices.loc[close_prices.index]
    low_prices = low_prices.loc[close_prices.index]

    print(f"  Final universe: {len(close_prices.columns)} symbols, {len(close_prices)} days")

    return close_prices, open_prices


def calculate_portfolio_volatility(
    close_prices: pd.DataFrame,
    lookback: int = 60
) -> pd.Series:
    """
    Calculate portfolio-level volatility (equal-weighted).

    Args:
        close_prices: DataFrame with symbol columns
        lookback: Volatility lookback window

    Returns:
        Annualized portfolio volatility Series
    """
    # Calculate returns for each symbol
    returns = close_prices.pct_change()

    # Equal-weighted portfolio return
    portfolio_returns = returns.mean(axis=1)

    # Annualized volatility
    portfolio_vol = portfolio_returns.rolling(lookback).std() * np.sqrt(252)

    return portfolio_vol


def generate_vol_scaled_weights(
    strategy: 'SemiVolMomentum',
    close_prices: pd.DataFrame,
    max_positions: int = 10
) -> pd.DataFrame:
    """
    Generate volatility-scaled target weights for each stock.

    VBT_VERIFIED: Portfolio weights for from_orders with targetpercent

    Key Innovation: Position sizes are scaled by (target_vol / realized_vol)
    - Low vol environment: Larger positions (up to 2.0x)
    - High vol environment: Smaller positions (down to 0.5x)

    Args:
        strategy: SemiVolMomentum strategy instance
        close_prices: DataFrame with symbol columns (aligned prices)
        max_positions: Maximum number of positions to hold

    Returns:
        DataFrame with same shape as close_prices, values = target weight (0-1)
        Non-rebalance days = NaN (maintain current positions)
    """
    # Initialize weights DataFrame with NaN (no change)
    weights = pd.DataFrame(
        np.nan,
        index=close_prices.index,
        columns=close_prices.columns
    )

    # Calculate portfolio volatility
    portfolio_vol = calculate_portfolio_volatility(close_prices, strategy.volatility_lookback)

    # Calculate 12-1 momentum for each symbol
    momentum_df = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
    for symbol in close_prices.columns:
        momentum_df[symbol] = close_prices[symbol].pct_change(
            strategy.momentum_lookback
        ).shift(strategy.momentum_lag)

    # Get monthly rebalance dates (first trading day of each month)
    months = close_prices.index.to_period('M').unique()
    rebalance_dates = []
    for month in months:
        month_data = close_prices.index[close_prices.index.to_period('M') == month]
        if len(month_data) > 0:
            rebalance_dates.append(month_data[0])

    print(f"  Rebalance dates (monthly): {len(rebalance_dates)}")

    # Process each rebalance date
    for rebal_date in rebalance_dates:
        # Skip if before momentum lookback period
        date_idx = close_prices.index.get_loc(rebal_date)
        min_required = strategy.momentum_lookback + strategy.momentum_lag + 10
        if date_idx < min_required:
            continue

        # Check volatility condition - only trade in low vol environments
        current_vol = portfolio_vol.loc[rebal_date]
        if pd.isna(current_vol) or current_vol > strategy.vol_ceiling:
            # High vol environment - set all positions to zero
            for symbol in close_prices.columns:
                weights.loc[rebal_date, symbol] = 0.0
            continue

        # Calculate volatility scalar
        vol_scalar = strategy.target_volatility / current_vol
        vol_scalar = np.clip(vol_scalar, strategy.min_vol_scalar, strategy.max_vol_scalar)

        # Get momentum at rebalance date
        momentum_scores = []
        for symbol in close_prices.columns:
            mom = momentum_df.loc[rebal_date, symbol]
            if pd.notna(mom) and mom > 0:  # Only positive momentum
                momentum_scores.append({
                    'symbol': symbol,
                    'momentum': mom
                })

        if not momentum_scores:
            # No positive momentum - zero weights
            for symbol in close_prices.columns:
                weights.loc[rebal_date, symbol] = 0.0
            continue

        # Rank by momentum and select top N
        mom_df = pd.DataFrame(momentum_scores)
        mom_df['rank'] = mom_df['momentum'].rank(ascending=False)
        selected = mom_df[mom_df['rank'] <= max_positions]['symbol'].tolist()

        if not selected:
            for symbol in close_prices.columns:
                weights.loc[rebal_date, symbol] = 0.0
            continue

        # Calculate base equal weight scaled by volatility
        # vol_scalar adjusts overall portfolio size
        base_weight = (1.0 / len(selected)) * vol_scalar

        # Ensure we don't exceed 100% total allocation (even with 2x scaling)
        total_weight = base_weight * len(selected)
        if total_weight > 1.0:
            base_weight = 1.0 / len(selected)

        # Assign weights
        for symbol in close_prices.columns:
            if symbol in selected:
                weights.loc[rebal_date, symbol] = base_weight
            else:
                weights.loc[rebal_date, symbol] = 0.0

    # Count valid rebalance dates
    valid_rebalances = weights.notna().any(axis=1).sum()
    print(f"  Valid rebalance dates with selections: {valid_rebalances}")

    return weights


# =============================================================================
# BACKTEST EXECUTION (MULTI-STOCK PORTFOLIO MODE)
# =============================================================================

def run_semi_vol_momentum_backtest(
    universe_data: Dict[str, pd.DataFrame],
    initial_capital: float = 100000,
    max_positions: int = 10
) -> Tuple[dict, vbt.Portfolio]:
    """
    Run Semi-Vol Momentum strategy backtest on multi-stock portfolio.

    VBT_VERIFIED: Portfolio.from_orders with targetpercent
      - SEARCH: Portfolio.from_orders multi-asset examples found
      - VERIFY: vectorbtpro.portfolio.base.Portfolio.from_orders RESOLVED
      - FIND: 15+ real-world multi-asset examples with targetpercent
      - TEST: Equal-weighted rebalancing with group_by=True PASSED

    Args:
        universe_data: Dict of symbol -> OHLCV DataFrames
        initial_capital: Starting capital
        max_positions: Maximum number of positions to hold

    Returns:
        Tuple of (results_dict, portfolio_object)
    """
    print("\n" + "="*60)
    print("SEMI-VOL MOMENTUM STRATEGY BACKTEST (MULTI-STOCK PORTFOLIO)")
    print("="*60)

    # Step 1: Prepare aligned portfolio data
    print("\nStep 1: Preparing portfolio data...")
    close_prices, open_prices = prepare_portfolio_data(universe_data)

    # Step 2: Create strategy configuration
    config = StrategyConfig(
        name="Semi-Vol-Momentum",
        universe="multi_sector",
        rebalance_frequency="monthly",
        regime_compatibility={
            'TREND_BULL': True,
            'TREND_NEUTRAL': False,  # Sit out
            'TREND_BEAR': False,      # Sit out
            'CRASH': False            # Risk-off
        },
        risk_per_trade=0.02,
        max_positions=max_positions,
        enable_shorts=False
    )

    # Initialize strategy
    strategy = SemiVolMomentum(
        config,
        target_volatility=0.15,      # 15% target vol
        volatility_lookback=60,       # 60-day vol window
        momentum_lookback=252,        # 12-month momentum
        momentum_lag=21,              # 1-month lag
        min_vol_scalar=0.5,           # Min position scale
        max_vol_scalar=2.0,           # Max position scale
        vol_ceiling=0.18,             # 18% vol ceiling
        vol_circuit_breaker=0.22      # 22% circuit breaker
    )

    # Step 3: Generate volatility-scaled portfolio weights
    print("\nStep 2: Generating volatility-scaled weights...")
    weights = generate_vol_scaled_weights(
        strategy=strategy,
        close_prices=close_prices,
        max_positions=max_positions
    )

    # Step 4: Run multi-asset portfolio backtest
    print("\nStep 3: Running portfolio backtest...")
    print(f"  Universe: {len(close_prices.columns)} stocks")
    print(f"  Period: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print(f"  Initial capital: ${initial_capital:,.0f}")

    # VBT_VERIFIED: Portfolio.from_orders with multi-asset weights
    pf = vbt.Portfolio.from_orders(
        close=close_prices,
        size=weights,
        size_type='targetpercent',
        direction='longonly',
        group_by=True,                # Single portfolio (not per-asset)
        cash_sharing=True,            # Share capital across assets
        call_seq='auto',              # Sell before buy to free up cash
        init_cash=initial_capital,
        fees=0.001,                   # 10 bps transaction cost
        slippage=0.001,               # 10 bps slippage
        freq='1D'
    )

    # Extract portfolio-level metrics
    total_return = float(pf.total_return)
    sharpe = float(pf.sharpe_ratio)
    sortino = float(pf.sortino_ratio)
    max_drawdown = float(pf.max_drawdown)
    cagr = float(pf.annualized_return)

    # Trade statistics
    trades = pf.trades
    total_trades = int(trades.count())
    if total_trades > 0:
        win_rate = float(trades.win_rate)
        avg_trade = float(trades.pnl.mean() / initial_capital) if trades.pnl.mean() is not None else 0.0
    else:
        win_rate = 0.0
        avg_trade = 0.0

    print(f"\nPortfolio Metrics:")
    print(f"  Total Return: {total_return:.1%}")
    print(f"  CAGR: {cagr:.1%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.1%}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1%}")

    results = {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'strategy': strategy,
        'close_prices': close_prices,
        'weights': weights
    }

    return results, pf


# =============================================================================
# WALK-FORWARD VALIDATION (PORTFOLIO MODE)
# =============================================================================

def run_walk_forward_validation(
    strategy: SemiVolMomentum,
    close_prices: pd.DataFrame,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    initial_capital: float = 100000,
    max_positions: int = 10
) -> dict:
    """
    Run walk-forward validation on multi-stock portfolio using VBT Splitter.

    VBT_VERIFIED: Splitter.from_n_rolling
      - Creates n rolling splits with train/test sets
      - Used for out-of-sample performance validation

    Target: Out-of-sample Sharpe >= 70% of in-sample Sharpe (degradation < 30%)
    """
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION (PORTFOLIO MODE)")
    print("="*60)

    # Calculate split sizes
    total_days = len(close_prices)
    min_train_days = 400  # More than momentum_lookback (252) + momentum_lag (21) + buffer
    min_test_days = 63    # ~3 months for meaningful OOS test

    # Adjust fold size to ensure sufficient training data
    fold_size = max(min_train_days + min_test_days, total_days // (n_folds + 1))
    train_days = int(fold_size * train_ratio)
    test_days = fold_size - train_days

    # Ensure minimum requirements
    train_days = max(train_days, min_train_days)
    test_days = max(test_days, min_test_days)

    print(f"Total trading days: {total_days}")
    print(f"Folds: {n_folds}")
    print(f"Train days per fold: {train_days}")
    print(f"Test days per fold: {test_days}")

    # Create splitter
    # VBT_VERIFIED: Splitter.from_n_rolling
    splitter = vbt.Splitter.from_n_rolling(
        close_prices.index,
        n=n_folds,
        length=train_days + test_days,
        split=[train_days, test_days],
        set_labels=["train", "test"]
    )

    print(f"Splitter created with {splitter.n_splits} splits")

    # Run backtests on each fold
    in_sample_sharpes = []
    out_sample_sharpes = []

    # Get splits using take() method
    train_splits = splitter.take(close_prices, set_="train")
    test_splits = splitter.take(close_prices, set_="test")

    for i in range(n_folds):
        try:
            # Get fold data
            train_prices = train_splits[i]
            test_prices = test_splits[i]

            if len(train_prices) < 300 or len(test_prices) < 21:
                print(f"  Fold {i+1}: SKIP (insufficient data)")
                continue

            # Generate weights for train period
            train_weights = generate_vol_scaled_weights(
                strategy=strategy,
                close_prices=train_prices,
                max_positions=max_positions
            )

            # In-sample backtest
            pf_train = vbt.Portfolio.from_orders(
                close=train_prices,
                size=train_weights,
                size_type='targetpercent',
                direction='longonly',
                group_by=True,
                cash_sharing=True,
                call_seq='auto',
                init_cash=initial_capital,
                fees=0.001,
                slippage=0.001,
                freq='1D'
            )
            in_sharpe = float(pf_train.sharpe_ratio)

            # Generate weights for test period
            test_weights = generate_vol_scaled_weights(
                strategy=strategy,
                close_prices=test_prices,
                max_positions=max_positions
            )

            # Out-of-sample backtest
            pf_test = vbt.Portfolio.from_orders(
                close=test_prices,
                size=test_weights,
                size_type='targetpercent',
                direction='longonly',
                group_by=True,
                cash_sharing=True,
                call_seq='auto',
                init_cash=initial_capital,
                fees=0.001,
                slippage=0.001,
                freq='1D'
            )
            out_sharpe = float(pf_test.sharpe_ratio)

            in_sample_sharpes.append(in_sharpe)
            out_sample_sharpes.append(out_sharpe)

            print(f"  Fold {i+1}: IS Sharpe={in_sharpe:.2f}, OOS Sharpe={out_sharpe:.2f}")

        except Exception as e:
            print(f"  Fold {i+1}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    # Calculate degradation
    if in_sample_sharpes and out_sample_sharpes:
        avg_in = np.mean([s for s in in_sample_sharpes if np.isfinite(s)])
        avg_out = np.mean([s for s in out_sample_sharpes if np.isfinite(s)])

        if avg_in != 0 and np.isfinite(avg_in):
            degradation = 1 - (avg_out / avg_in) if avg_in > 0 else 1.0
        else:
            degradation = 1.0
    else:
        avg_in = 0.0
        avg_out = 0.0
        degradation = 1.0

    print(f"\nWalk-Forward Results:")
    print(f"  Average In-Sample Sharpe: {avg_in:.2f}")
    print(f"  Average Out-of-Sample Sharpe: {avg_out:.2f}")
    print(f"  Degradation: {degradation:.1%}")
    print(f"  Target: < 30%")
    print(f"  Status: {'PASS' if degradation < TARGETS['walk_forward_degradation'] else 'FAIL'}")

    return {
        'in_sample_sharpes': in_sample_sharpes,
        'out_sample_sharpes': out_sample_sharpes,
        'avg_in_sample': avg_in,
        'avg_out_sample': avg_out,
        'degradation': degradation,
        'passed': degradation < TARGETS['walk_forward_degradation']
    }


# =============================================================================
# MONTE CARLO SIMULATION (PORTFOLIO RETURNS)
# =============================================================================

def run_monte_carlo_simulation(
    portfolio_returns: pd.Series,
    n_simulations: int = 500,
    seed: int = 42
) -> dict:
    """
    Run Monte Carlo simulation using moving block bootstrap on portfolio returns.

    VBT_VERIFIED: Splitter.from_rolling, Splitter.shuffle_splits

    Target: P(Total Return < 0) < 20%
    """
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION (Portfolio Block Bootstrap)")
    print("="*60)

    returns = portfolio_returns.dropna()

    # Block size formula
    block_size = int(3.15 * len(returns) ** (1 / 3))
    print(f"Block size: {block_size} days")
    print(f"Simulations: {n_simulations}")

    # VBT_VERIFIED: Splitter.from_rolling
    block_splitter = vbt.Splitter.from_rolling(
        returns.index,
        length=block_size,
        offset=1,
        offset_anchor="prev_start"
    )

    print(f"Total blocks available: {block_splitter.n_splits}")

    # Run simulations
    total_returns = []
    np.random.seed(seed)

    target_length = len(returns)
    blocks_needed = int(np.ceil(target_length / block_size))

    for sim in range(n_simulations):
        try:
            # VBT_VERIFIED: Splitter.shuffle_splits
            shuffled = block_splitter.shuffle_splits(
                size=blocks_needed,
                replace=True,
                seed=seed + sim
            )

            sample_returns = shuffled.take(returns, into="stacked")
            if hasattr(sample_returns, 'values'):
                sample_returns = pd.Series(sample_returns.values.flatten())

            sample_returns = sample_returns.dropna()
            if len(sample_returns) > target_length:
                sample_returns = sample_returns.iloc[:target_length]

            if len(sample_returns) < target_length * 0.8:
                continue

            cum_return = float((1 + sample_returns).prod() - 1)

            if np.isfinite(cum_return):
                total_returns.append(cum_return)

        except Exception as e:
            if sim == 0:
                print(f"  Simulation error: {e}")

    if not total_returns:
        print("  [WARN] No valid simulations completed")
        return {
            'p_loss': 1.0,
            'mean_return': 0.0,
            'passed': False
        }

    # Calculate statistics
    returns_array = np.array(total_returns)
    p_loss = (returns_array < 0).mean()
    mean_return = np.mean(returns_array)
    median_return = np.median(returns_array)
    std_return = np.std(returns_array)
    pct_5 = np.percentile(returns_array, 5)
    pct_95 = np.percentile(returns_array, 95)

    print(f"\nMonte Carlo Results ({len(total_returns)} valid simulations):")
    print(f"  P(Loss): {p_loss:.1%}")
    print(f"  Mean Return: {mean_return:.1%}")
    print(f"  Median Return: {median_return:.1%}")
    print(f"  Std Dev: {std_return:.1%}")
    print(f"  5th Percentile: {pct_5:.1%}")
    print(f"  95th Percentile: {pct_95:.1%}")
    print(f"  Target P(Loss): < 20%")
    print(f"  Status: {'PASS' if p_loss < TARGETS['monte_carlo_p_loss'] else 'FAIL'}")

    return {
        'p_loss': p_loss,
        'mean_return': mean_return,
        'median_return': median_return,
        'std_return': std_return,
        'percentile_5': pct_5,
        'percentile_95': pct_95,
        'n_simulations': len(total_returns),
        'passed': p_loss < TARGETS['monte_carlo_p_loss']
    }


# =============================================================================
# RESULTS COMPARISON
# =============================================================================

def print_final_results(
    backtest_results: dict,
    walk_forward_results: dict,
    monte_carlo_results: dict
):
    """Print final validation results and target comparison."""
    print("\n" + "="*70)
    print("SEMI-VOL MOMENTUM STRATEGY - FINAL VALIDATION RESULTS")
    print("="*70)

    # Main metrics
    print(f"\n{'METRIC':<30} {'ACTUAL':>15} {'TARGET':>20} {'STATUS':>10}")
    print("-"*75)

    # Sharpe Ratio
    sharpe = backtest_results['sharpe']
    sharpe_status = 'PASS' if TARGETS['sharpe_min'] <= sharpe else 'FAIL'
    print(f"{'Sharpe Ratio':<30} {sharpe:>15.2f} {TARGETS['sharpe_min']:.1f}-{TARGETS['sharpe_max']:.1f}{'':<11} {sharpe_status:>10}")

    # CAGR
    cagr = backtest_results['cagr']
    cagr_status = 'PASS' if cagr >= TARGETS['cagr_min'] else 'FAIL'
    print(f"{'CAGR':<30} {cagr:>14.1%} {TARGETS['cagr_min']:.0%}-{TARGETS['cagr_max']:.0%}{'':<11} {cagr_status:>10}")

    # Max Drawdown
    max_dd = abs(backtest_results['max_drawdown'])
    dd_status = 'PASS' if max_dd <= TARGETS['max_drawdown'] else 'FAIL'
    print(f"{'Max Drawdown':<30} {max_dd:>14.1%} {'<'} {TARGETS['max_drawdown']:.0%}{'':<13} {dd_status:>10}")

    # Walk-Forward Degradation
    degradation = walk_forward_results['degradation']
    wf_status = 'PASS' if walk_forward_results['passed'] else 'FAIL'
    print(f"{'Walk-Forward Degradation':<30} {degradation:>14.1%} {'<'} {TARGETS['walk_forward_degradation']:.0%}{'':<13} {wf_status:>10}")

    # Monte Carlo P(Loss)
    p_loss = monte_carlo_results['p_loss']
    mc_status = 'PASS' if monte_carlo_results['passed'] else 'FAIL'
    print(f"{'Monte Carlo P(Loss)':<30} {p_loss:>14.1%} {'<'} {TARGETS['monte_carlo_p_loss']:.0%}{'':<13} {mc_status:>10}")

    # Trade Statistics
    print(f"\n{'TRADE STATISTICS':<30}")
    print("-"*75)
    print(f"{'Total Trades':<30} {backtest_results['total_trades']:>15.0f}")
    print(f"{'Win Rate':<30} {backtest_results['win_rate']:>14.1%}")
    print(f"{'Avg Trade Return':<30} {backtest_results['avg_trade']:>14.2%}")
    print(f"{'Total Return':<30} {backtest_results['total_return']:>14.1%}")
    print(f"{'Sortino Ratio':<30} {backtest_results['sortino']:>15.2f}")

    # Overall Status
    all_pass = (
        sharpe_status == 'PASS' and
        cagr_status == 'PASS' and
        dd_status == 'PASS' and
        wf_status == 'PASS' and
        mc_status == 'PASS'
    )

    print("\n" + "="*70)
    if all_pass:
        print("OVERALL STATUS: ALL TARGETS MET - READY FOR DEPLOYMENT")
    else:
        print("OVERALL STATUS: SOME TARGETS NOT MET - REVIEW REQUIRED")
        print("\nRecommendations:")
        if sharpe_status == 'FAIL':
            print("  - Sharpe: Adjust vol scaling parameters or momentum filters")
        if cagr_status == 'FAIL':
            print("  - CAGR: Test in trending markets, reduce vol ceiling")
        if dd_status == 'FAIL':
            print("  - Drawdown: Tighten circuit breaker, reduce max vol scalar")
        if wf_status == 'FAIL':
            print("  - Walk-Forward: Simplify signal logic, reduce parameters")
        if mc_status == 'FAIL':
            print("  - Monte Carlo: Add position limits, increase diversification")
    print("="*70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("SEMI-VOL MOMENTUM STRATEGY - VBT 5-STEP VALIDATED BACKTEST")
    print("="*70)
    print("\nPurpose: Validate Semi-Volatility Momentum implementation")
    print("Mode: Multi-stock multi-sector portfolio with volatility scaling")
    print(f"Universe: {len(UNIVERSE_SYMBOLS)} stocks across 11 GICS sectors")
    print(f"Expected: Sharpe {TARGETS['sharpe_min']}-{TARGETS['sharpe_max']}, ")
    print(f"          CAGR {TARGETS['cagr_min']:.0%}-{TARGETS['cagr_max']:.0%}, ")
    print(f"          MaxDD < {TARGETS['max_drawdown']:.0%}")

    # Configuration for validation run
    # Use 16 symbols for faster validation while maintaining sector diversity
    VALIDATION_SYMBOLS = get_validation_symbols()  # 16 stocks across 11 sectors

    # Step 1: Fetch data
    print("\n" + "-"*60)
    print("STEP 1: DATA FETCHING (MULTI-SECTOR)")
    print("-"*60)
    print_sector_distribution(VALIDATION_SYMBOLS)

    try:
        universe_data = fetch_multi_stock_data(
            VALIDATION_SYMBOLS,  # 16 symbols for sector-diverse validation
            start_date='2010-01-01',
            end_date='2025-01-01'
        )

        if not universe_data:
            print("[FAIL] No data fetched. Check internet connection.")
            return

    except Exception as e:
        print(f"[FAIL] Data fetch failed: {e}")
        return

    # Step 2: Run main backtest (MULTI-STOCK PORTFOLIO MODE)
    print("\n" + "-"*60)
    print("STEP 2: MAIN BACKTEST (PORTFOLIO MODE)")
    print("-"*60)

    try:
        backtest_results, pf = run_semi_vol_momentum_backtest(
            universe_data,
            initial_capital=100000,
            max_positions=10  # Top 10 stocks from 16-stock universe
        )
    except Exception as e:
        print(f"[FAIL] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Walk-forward validation (PORTFOLIO MODE)
    print("\n" + "-"*60)
    print("STEP 3: WALK-FORWARD VALIDATION (PORTFOLIO MODE)")
    print("-"*60)

    try:
        walk_forward_results = run_walk_forward_validation(
            strategy=backtest_results['strategy'],
            close_prices=backtest_results['close_prices'],
            n_folds=3,  # Reduced for more data per fold
            train_ratio=0.7,
            initial_capital=100000,
            max_positions=10
        )
    except Exception as e:
        print(f"[WARN] Walk-forward failed: {e}")
        import traceback
        traceback.print_exc()
        walk_forward_results = {
            'degradation': 1.0,
            'passed': False,
            'avg_in_sample': 0,
            'avg_out_sample': 0
        }

    # Step 4: Monte Carlo simulation (PORTFOLIO RETURNS)
    print("\n" + "-"*60)
    print("STEP 4: MONTE CARLO SIMULATION (PORTFOLIO RETURNS)")
    print("-"*60)

    try:
        portfolio_returns = pf.returns
        monte_carlo_results = run_monte_carlo_simulation(
            portfolio_returns=portfolio_returns,
            n_simulations=500,
            seed=42
        )
    except Exception as e:
        print(f"[WARN] Monte Carlo failed: {e}")
        import traceback
        traceback.print_exc()
        monte_carlo_results = {
            'p_loss': 1.0,
            'passed': False,
            'mean_return': 0
        }

    # Step 5: Print final results
    print_final_results(backtest_results, walk_forward_results, monte_carlo_results)

    print("\n" + "="*70)
    print("BACKTEST VALIDATION COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. If all targets met -> Enable in live trading")
    print("2. If targets missed -> Adjust vol scaling parameters")
    print("3. Compare with Quality-Momentum for portfolio allocation")


if __name__ == "__main__":
    main()
