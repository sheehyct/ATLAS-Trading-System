"""
Session 83B: Compare Synthetic (Black-Scholes) vs Real (ThetaData) Options P/L

Enhanced version with:
- 6 symbols: SPY, QQQ, AAPL, IWM, DIA, NVDA
- Dynamic ATM strike lookup via Tiingo
- Data availability validation before backtest
- Per-symbol metrics breakdown

Key Metrics:
- Price difference (synthetic vs real) per option
- P/L discrepancy analysis
- Greeks accuracy comparison
- Data source distribution (thetadata/black_scholes/mixed)

Usage:
    uv run python exploratory/compare_synthetic_vs_real_pnl.py

Requires:
    - ThetaData terminal running (localhost:25503)
    - Tiingo API access for price data
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strat.options_module import (
    OptionsBacktester, OptionTrade, OptionContract, OptionType, OptionStrategy
)
from strat.tier1_detector import PatternSignal, PatternType, Timeframe
from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher
from integrations.tiingo_data_fetcher import TiingoDataFetcher
from exploratory.comparison_config import (
    ComparisonConfig, SymbolMetrics, DataAvailability,
    STRIKE_INTERVALS, APPROXIMATE_PRICES
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_atm_strike(
    symbol: str,
    trade_date: datetime,
    config: ComparisonConfig,
    price_cache: Dict[str, pd.DataFrame] = None
) -> float:
    """
    Get ATM strike price for a symbol on a given date.

    Uses Tiingo to fetch the underlying price, then rounds to the nearest
    valid strike interval for the symbol.

    Args:
        symbol: Underlying symbol (e.g., 'SPY')
        trade_date: Date to get ATM strike for
        config: ComparisonConfig with strike intervals
        price_cache: Optional cache of price DataFrames to avoid repeated fetches

    Returns:
        ATM strike price rounded to valid interval
    """
    if price_cache is None:
        price_cache = {}

    # Check cache first
    if symbol in price_cache:
        df = price_cache[symbol]
        # Find the closest date in the DataFrame
        trade_ts = pd.Timestamp(trade_date)
        if df.index.tz is not None:
            trade_ts = trade_ts.tz_localize(df.index.tz)

        # Get the price on or before trade_date
        mask = df.index <= trade_ts
        if mask.any():
            close_col = 'close' if 'close' in df.columns else 'Close'
            price = df.loc[mask, close_col].iloc[-1]
            return config.round_to_strike(price, symbol)

    # Fallback to approximate price
    approx_price = config.get_approximate_price(symbol)
    return config.round_to_strike(approx_price, symbol)


def validate_data_availability(
    config: ComparisonConfig,
    thetadata_fetcher: ThetaDataOptionsFetcher = None
) -> Dict[str, DataAvailability]:
    """
    Validate data availability for all symbols before running backtest.

    Checks:
    1. Tiingo has underlying price data
    2. ThetaData terminal is connected
    3. ThetaData has expirations for symbol
    4. Sample quote retrieval works

    Args:
        config: ComparisonConfig with symbols and date range
        thetadata_fetcher: Optional pre-initialized fetcher

    Returns:
        Dict mapping symbol to DataAvailability result
    """
    results = {}
    tiingo = TiingoDataFetcher()

    # Initialize ThetaData fetcher if not provided
    if thetadata_fetcher is None:
        thetadata_fetcher = ThetaDataOptionsFetcher(auto_connect=True)

    thetadata_connected = thetadata_fetcher.is_available

    for symbol in config.symbols:
        availability = DataAvailability(symbol=symbol)
        availability.thetadata_connected = thetadata_connected

        # Check Tiingo data
        try:
            start_str = config.start_date.strftime('%Y-%m-%d')
            end_str = config.end_date.strftime('%Y-%m-%d')
            data = tiingo.fetch(symbol, start_date=start_str, end_date=end_str)
            if data is not None:
                df = data.get() if hasattr(data, 'get') else data
                if df is not None and len(df) > 0:
                    availability.tiingo_available = True
        except Exception as e:
            availability.error_message = f"Tiingo error: {e}"

        # Check ThetaData expirations (use _provider which is the internal attribute)
        if thetadata_connected and thetadata_fetcher._provider is not None:
            try:
                expirations = thetadata_fetcher._provider.get_expirations(symbol)
                if expirations and len(expirations) > 0:
                    availability.thetadata_has_expirations = True

                    # Try a sample quote
                    try:
                        sample_exp = expirations[0]
                        strikes = thetadata_fetcher._provider.get_strikes(symbol, sample_exp)
                        if strikes and len(strikes) > 0:
                            availability.sample_quote_ok = True
                    except Exception:
                        pass
            except Exception as e:
                if availability.error_message:
                    availability.error_message += f"; ThetaData error: {e}"
                else:
                    availability.error_message = f"ThetaData error: {e}"

        results[symbol] = availability

    return results


def create_sample_trades(
    config: ComparisonConfig,
    price_cache: Dict[str, pd.DataFrame] = None
) -> List[OptionTrade]:
    """
    Create sample option trades for comparison testing.

    Uses configuration for symbols, dates, and strike calculation.

    Args:
        config: ComparisonConfig with test parameters
        price_cache: Cache of price DataFrames for ATM strike lookup

    Returns:
        List of OptionTrade objects for backtesting
    """
    if price_cache is None:
        price_cache = {}

    trades = []

    # Pattern configurations - 3 patterns per symbol
    pattern_configs = [
        (PatternType.PATTERN_212_UP, 'call'),
        (PatternType.PATTERN_312_UP, 'call'),
        (PatternType.PATTERN_22_UP, 'call'),
    ]

    # Spread trade dates across the date range
    date_range_days = (config.end_date - config.start_date).days
    days_per_symbol = date_range_days // len(config.symbols)

    for i, symbol in enumerate(config.symbols):
        for j, (pattern_type, opt_type) in enumerate(pattern_configs[:config.patterns_per_symbol]):
            # Spread trades across the date range
            trade_offset = i * days_per_symbol + j * (days_per_symbol // config.patterns_per_symbol)
            trade_date = config.start_date + timedelta(days=trade_offset)

            # Skip weekends
            while trade_date.weekday() >= 5:
                trade_date += timedelta(days=1)

            expiration = trade_date + timedelta(days=config.expiration_offset_days)
            # Skip weekends for expiration too
            while expiration.weekday() >= 5:
                expiration += timedelta(days=1)

            # Get ATM strike
            if config.use_dynamic_strikes:
                strike = get_atm_strike(symbol, trade_date, config, price_cache)
            else:
                strike = config.round_to_strike(
                    config.get_approximate_price(symbol), symbol
                )

            # Create pattern signal
            signal = PatternSignal(
                pattern_type=pattern_type,
                direction=1 if opt_type == 'call' else -1,
                entry_price=strike * 1.005 if opt_type == 'call' else strike * 0.995,
                stop_price=strike * 0.98 if opt_type == 'call' else strike * 1.02,
                target_price=strike * 1.02 if opt_type == 'call' else strike * 0.98,
                timestamp=pd.Timestamp(trade_date),
                timeframe=Timeframe.WEEKLY,
                continuation_bars=2,
                is_filtered=True,
                risk_reward=2.0,
            )

            # Create option contract
            contract = OptionContract(
                underlying=symbol,
                expiration=expiration,
                option_type=OptionType.CALL if opt_type == 'call' else OptionType.PUT,
                strike=strike,
            )

            # Create trade
            trade = OptionTrade(
                pattern_signal=signal,
                contract=contract,
                strategy=OptionStrategy.LONG_CALL if opt_type == 'call' else OptionStrategy.LONG_PUT,
                entry_trigger=signal.entry_price,
                target_exit=signal.target_price,
                stop_exit=signal.stop_price,
                quantity=1,
                option_premium=0.0,
            )

            trades.append(trade)

    return trades


def run_comparison_backtest(
    trades: List[OptionTrade],
    price_data_dict: Dict[str, pd.DataFrame],
    use_thetadata: bool = True
) -> pd.DataFrame:
    """
    Run backtest with or without ThetaData.

    Args:
        trades: List of OptionTrade objects
        price_data_dict: Pre-fetched price data by symbol
        use_thetadata: If False, forces Black-Scholes fallback

    Returns:
        DataFrame with backtest results
    """
    # Create backtester
    if use_thetadata:
        fetcher = ThetaDataOptionsFetcher(auto_connect=True)
        backtester = OptionsBacktester(
            thetadata_provider=fetcher if fetcher.is_available else None
        )
    else:
        backtester = OptionsBacktester(thetadata_provider=None)

    # Run backtest for each symbol's trades
    all_results = []
    symbols = list(set(t.contract.underlying for t in trades))

    for symbol in symbols:
        if symbol not in price_data_dict:
            continue

        symbol_trades = [t for t in trades if t.contract.underlying == symbol]
        if not symbol_trades:
            continue

        results = backtester.backtest_trades(symbol_trades, price_data_dict[symbol])
        if results is not None and len(results) > 0:
            results['symbol'] = symbol
            all_results.append(results)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def calculate_symbol_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    symbol: str
) -> SymbolMetrics:
    """
    Calculate metrics for a single symbol.

    Args:
        real_df: Results DataFrame for real (ThetaData) backtest
        synthetic_df: Results DataFrame for synthetic (B-S) backtest
        symbol: Symbol to calculate metrics for

    Returns:
        SymbolMetrics object with computed values
    """
    metrics = SymbolMetrics(symbol=symbol)

    # Filter to this symbol
    real_sym = real_df[real_df['symbol'] == symbol] if 'symbol' in real_df.columns else real_df
    synth_sym = synthetic_df[synthetic_df['symbol'] == symbol] if 'symbol' in synthetic_df.columns else synthetic_df

    if real_sym.empty:
        metrics.data_available = False
        return metrics

    metrics.trade_count = len(real_sym)

    # Data source distribution
    if 'data_source' in real_sym.columns:
        source_dist = real_sym['data_source'].value_counts(normalize=True)
        metrics.thetadata_pct = source_dist.get('thetadata', 0) * 100
        metrics.black_scholes_pct = source_dist.get('black_scholes', 0) * 100
        metrics.mixed_pct = source_dist.get('mixed', 0) * 100

    # Price metrics
    if 'option_cost' in real_sym.columns and 'option_cost' in synth_sym.columns:
        real_costs = real_sym['option_cost'].dropna()
        synth_costs = synth_sym['option_cost'].dropna()

        if len(real_costs) > 0 and len(synth_costs) > 0:
            min_len = min(len(real_costs), len(synth_costs))
            diff = real_costs.values[:min_len] - synth_costs.values[:min_len]
            metrics.price_mae = np.abs(diff).mean()

            with np.errstate(divide='ignore', invalid='ignore'):
                pct_diff = np.abs(diff) / np.abs(synth_costs.values[:min_len])
                pct_diff = pct_diff[~np.isinf(pct_diff) & ~np.isnan(pct_diff)]
                if len(pct_diff) > 0:
                    metrics.price_mape = pct_diff.mean() * 100

    # P/L metrics
    if 'pnl' in real_sym.columns and 'pnl' in synth_sym.columns:
        real_pnl = real_sym['pnl'].dropna()
        synth_pnl = synth_sym['pnl'].dropna()

        if len(real_pnl) > 0 and len(synth_pnl) > 0:
            min_len = min(len(real_pnl), len(synth_pnl))
            diff = real_pnl.values[:min_len] - synth_pnl.values[:min_len]
            metrics.pnl_mae = np.abs(diff).mean()
            metrics.pnl_rmse = np.sqrt((diff ** 2).mean())

    return metrics


def calculate_discrepancy_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    config: ComparisonConfig
) -> Tuple[dict, List[SymbolMetrics]]:
    """
    Calculate discrepancy metrics between real and synthetic pricing.

    Returns:
        Tuple of (aggregate_metrics dict, list of SymbolMetrics)
    """
    # Check if both DataFrames have data
    if real_df.empty or synthetic_df.empty:
        return {
            'price_mae': None,
            'price_mape': None,
            'pnl_mae': None,
            'pnl_rmse': None,
            'real_trades': len(real_df),
            'synthetic_trades': len(synthetic_df),
        }, []

    # Calculate per-symbol metrics
    symbol_metrics = []
    for symbol in config.symbols:
        metrics = calculate_symbol_metrics(real_df, synthetic_df, symbol)
        symbol_metrics.append(metrics)

    # Aggregate metrics
    aggregate = {}

    # Data source distribution (overall)
    if 'data_source' in real_df.columns:
        source_dist = real_df['data_source'].value_counts(normalize=True)
        aggregate['thetadata_pct'] = source_dist.get('thetadata', 0) * 100
        aggregate['black_scholes_pct'] = source_dist.get('black_scholes', 0) * 100
        aggregate['mixed_pct'] = source_dist.get('mixed', 0) * 100

    # Overall price metrics
    if 'option_cost' in real_df.columns and 'option_cost' in synthetic_df.columns:
        real_costs = real_df['option_cost'].dropna()
        synth_costs = synthetic_df['option_cost'].dropna()

        if len(real_costs) > 0 and len(synth_costs) > 0:
            min_len = min(len(real_costs), len(synth_costs))
            diff = real_costs.values[:min_len] - synth_costs.values[:min_len]
            aggregate['price_mae'] = np.abs(diff).mean()

            with np.errstate(divide='ignore', invalid='ignore'):
                pct_diff = np.abs(diff) / np.abs(synth_costs.values[:min_len])
                pct_diff = pct_diff[~np.isinf(pct_diff) & ~np.isnan(pct_diff)]
                aggregate['price_mape'] = pct_diff.mean() * 100 if len(pct_diff) > 0 else None

    # Overall P/L metrics
    if 'pnl' in real_df.columns and 'pnl' in synthetic_df.columns:
        real_pnl = real_df['pnl'].dropna()
        synth_pnl = synthetic_df['pnl'].dropna()

        if len(real_pnl) > 0 and len(synth_pnl) > 0:
            min_len = min(len(real_pnl), len(synth_pnl))
            diff = real_pnl.values[:min_len] - synth_pnl.values[:min_len]
            aggregate['pnl_mae'] = np.abs(diff).mean()
            aggregate['pnl_rmse'] = np.sqrt((diff ** 2).mean())

    # Win rate comparison
    if 'win' in real_df.columns and 'win' in synthetic_df.columns:
        aggregate['real_win_rate'] = real_df['win'].mean() * 100
        aggregate['synthetic_win_rate'] = synthetic_df['win'].mean() * 100
        aggregate['win_rate_diff'] = aggregate['real_win_rate'] - aggregate['synthetic_win_rate']

    aggregate['real_trades'] = len(real_df)
    aggregate['synthetic_trades'] = len(synthetic_df)

    return aggregate, symbol_metrics


def print_per_symbol_table(symbol_metrics: List[SymbolMetrics]):
    """Print per-symbol metrics in a formatted table."""
    print("\n" + "=" * 80)
    print("[4] Per-Symbol Results")
    print("=" * 80)
    print(f"| {'Symbol':<8} | {'Trades':>6} | {'ThetaData%':>10} | {'B-S%':>6} | {'Price MAE':>10} | {'MAPE':>8} |")
    print("-" * 80)

    for m in symbol_metrics:
        if not m.data_available:
            print(f"| {m.symbol:<8} | {'N/A':>6} | {'N/A':>10} | {'N/A':>6} | {'N/A':>10} | {'N/A':>8} |")
        else:
            price_mae_str = f"${m.price_mae:.2f}" if m.price_mae is not None else "N/A"
            mape_str = f"{m.price_mape:.1f}%" if m.price_mape is not None else "N/A"
            print(f"| {m.symbol:<8} | {m.trade_count:>6} | {m.thetadata_pct:>9.1f}% | {m.black_scholes_pct:>5.1f}% | {price_mae_str:>10} | {mape_str:>8} |")

    print("=" * 80)


def main():
    print("=" * 80)
    print("Session 83B: Expanded Synthetic vs Real Options P/L Comparison")
    print("=" * 80)

    # Create configuration
    config = ComparisonConfig(
        symbols=['SPY', 'QQQ', 'AAPL', 'IWM', 'DIA', 'NVDA'],
        start_date=datetime(2024, 10, 1),
        end_date=datetime(2024, 11, 15),
        patterns_per_symbol=3,
        use_dynamic_strikes=True,
        validate_data_first=True,
    )

    print(f"\nConfiguration:")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Date range: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
    print(f"  Patterns per symbol: {config.patterns_per_symbol}")
    print(f"  Total expected trades: {len(config.symbols) * config.patterns_per_symbol}")

    # Validate data availability
    if config.validate_data_first:
        print("\n[1] Validating data availability...")
        availability = validate_data_availability(config)

        print("\n  Data Availability:")
        print(f"  | {'Symbol':<8} | {'Tiingo':>8} | {'ThetaData':>10} | {'Options':>8} | {'Status':>10} |")
        print("  " + "-" * 60)
        for symbol, avail in availability.items():
            tiingo_str = "OK" if avail.tiingo_available else "FAIL"
            theta_str = "OK" if avail.thetadata_connected else "FAIL"
            opts_str = "OK" if avail.thetadata_has_expirations else "FAIL"
            print(f"  | {symbol:<8} | {tiingo_str:>8} | {theta_str:>10} | {opts_str:>8} | {avail.status_str():>10} |")

        # Filter to available symbols
        available_symbols = [s for s, a in availability.items() if a.tiingo_available]
        if not available_symbols:
            print("\n  ERROR: No symbols have available data. Exiting.")
            return None, None, None

        if len(available_symbols) < len(config.symbols):
            print(f"\n  WARNING: Only {len(available_symbols)}/{len(config.symbols)} symbols available")
            config.symbols = available_symbols

    # Pre-fetch price data for all symbols
    print("\n[2] Fetching price data...")
    tiingo = TiingoDataFetcher()
    price_data_dict = {}

    # Determine date range with buffer
    min_date = config.start_date - timedelta(days=10)
    max_date = config.end_date + timedelta(days=45)

    for symbol in config.symbols:
        try:
            start_str = min_date.strftime('%Y-%m-%d')
            end_str = max_date.strftime('%Y-%m-%d')
            data = tiingo.fetch(symbol, start_date=start_str, end_date=end_str)
            if data is not None:
                df = data.get() if hasattr(data, 'get') else data
                if df is not None and len(df) > 0:
                    # Remove timezone for compatibility
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    # Normalize column names
                    df.columns = [c.lower() for c in df.columns]
                    price_data_dict[symbol] = df
                    print(f"  {symbol}: {len(df)} bars loaded")
        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")

    # Create sample trades with dynamic strikes
    print("\n[3] Creating sample trades...")
    trades = create_sample_trades(config, price_data_dict)
    print(f"  Created {len(trades)} sample trades")

    # Show trade distribution
    trade_counts = {}
    for t in trades:
        sym = t.contract.underlying
        trade_counts[sym] = trade_counts.get(sym, 0) + 1
    print(f"  Distribution: {trade_counts}")

    # Run with ThetaData (real pricing)
    print("\n[4] Running backtest with ThetaData (real pricing)...")
    real_results = run_comparison_backtest(trades, price_data_dict, use_thetadata=True)
    print(f"  Completed: {len(real_results)} trades processed")

    # Run without ThetaData (synthetic/Black-Scholes only)
    print("\n[5] Running backtest with Black-Scholes only (synthetic)...")
    synthetic_results = run_comparison_backtest(trades, price_data_dict, use_thetadata=False)
    print(f"  Completed: {len(synthetic_results)} trades processed")

    # Calculate discrepancy metrics
    print("\n[6] Calculating discrepancy metrics...")
    aggregate_metrics, symbol_metrics = calculate_discrepancy_metrics(
        real_results, synthetic_results, config
    )

    # Display aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE DISCREPANCY ANALYSIS RESULTS")
    print("=" * 80)

    print("\n--- Data Source Distribution (Real Backtest) ---")
    if 'thetadata_pct' in aggregate_metrics:
        print(f"  ThetaData:     {aggregate_metrics.get('thetadata_pct', 0):.1f}%")
        print(f"  Black-Scholes: {aggregate_metrics.get('black_scholes_pct', 0):.1f}%")
        print(f"  Mixed:         {aggregate_metrics.get('mixed_pct', 0):.1f}%")

    print("\n--- Pricing Discrepancy ---")
    if aggregate_metrics.get('price_mae') is not None:
        print(f"  Price MAE:  ${aggregate_metrics['price_mae']:.4f} per share")
    if aggregate_metrics.get('price_mape') is not None:
        print(f"  Price MAPE: {aggregate_metrics['price_mape']:.2f}%")

    print("\n--- P/L Discrepancy ---")
    if aggregate_metrics.get('pnl_mae') is not None:
        print(f"  P/L MAE:  ${aggregate_metrics['pnl_mae']:.2f}")
    if aggregate_metrics.get('pnl_rmse') is not None:
        print(f"  P/L RMSE: ${aggregate_metrics['pnl_rmse']:.2f}")

    print("\n--- Win Rate Comparison ---")
    if 'real_win_rate' in aggregate_metrics:
        print(f"  Real (ThetaData):     {aggregate_metrics['real_win_rate']:.1f}%")
        print(f"  Synthetic (B-S):      {aggregate_metrics['synthetic_win_rate']:.1f}%")
        print(f"  Difference:           {aggregate_metrics['win_rate_diff']:+.1f}%")

    print("\n--- Trade Counts ---")
    print(f"  Real backtest:      {aggregate_metrics['real_trades']} trades")
    print(f"  Synthetic backtest: {aggregate_metrics['synthetic_trades']} trades")

    # Display per-symbol results
    if symbol_metrics:
        print_per_symbol_table(symbol_metrics)

    # Save detailed results
    output_dir = Path(__file__).parent.parent / 'docs' / 'archives'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not real_results.empty:
        real_results.to_csv(output_dir / 'session83b_real_results.csv', index=False)
        print(f"\n  Saved real results to {output_dir / 'session83b_real_results.csv'}")

    if not synthetic_results.empty:
        synthetic_results.to_csv(output_dir / 'session83b_synthetic_results.csv', index=False)
        print(f"  Saved synthetic results to {output_dir / 'session83b_synthetic_results.csv'}")

    # Save per-symbol metrics
    if symbol_metrics:
        metrics_df = pd.DataFrame([m.to_dict() for m in symbol_metrics])
        metrics_df.to_csv(output_dir / 'session83b_per_symbol_metrics.csv', index=False)
        print(f"  Saved per-symbol metrics to {output_dir / 'session83b_per_symbol_metrics.csv'}")

    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)

    return aggregate_metrics, symbol_metrics, real_results


if __name__ == "__main__":
    main()
