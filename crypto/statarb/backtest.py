"""
Pairs trading backtest using VectorBT Pro.

Implements complete statistical arbitrage backtest with:
- Cointegration-based pair selection
- Z-score signal generation
- Coinbase CFM fee modeling
- Beta-adjusted position sizing
- Comprehensive performance metrics

Usage:
    from crypto.statarb.backtest import run_pairs_backtest
    
    result = run_pairs_backtest(
        symbol1="BTC-USD",
        symbol2="ETH-USD",
        start_date="2023-01-01",
        end_date="2024-01-01",
        interval="1h",
    )
    print(result.summary())
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# VectorBT Pro import
try:
    import vectorbtpro as vbt
except ImportError:
    raise ImportError("VectorBT Pro required. Install with: pip install vectorbtpro")

# Local imports
from .cointegration import engle_granger_test, calculate_half_life, CointegrationResult
from .spread import (
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    generate_signals,
    calculate_position_sizes,
)

# Fee imports
try:
    from crypto.trading.fees import (
        TAKER_FEE_RATE,
        MIN_FEE_PER_CONTRACT,
        calculate_round_trip_fee,
        calculate_breakeven_move,
    )
except ImportError:
    # Fallback fee constants
    TAKER_FEE_RATE = 0.0002  # 0.02%
    MIN_FEE_PER_CONTRACT = 0.15


@dataclass
class PairsBacktestResult:
    """Complete results from pairs trading backtest."""
    
    # Pair information
    symbol1: str
    symbol2: str
    start_date: datetime
    end_date: datetime
    interval: str
    
    # Cointegration metrics
    cointegration: Optional[CointegrationResult] = None
    half_life: float = np.nan
    hedge_ratio: float = 1.0
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    avg_trade_duration: float = 0.0  # in bars
    
    # Fee impact
    total_fees_paid: float = 0.0
    gross_return: float = 0.0  # Before fees
    fee_drag: float = 0.0  # Fees as % of gross return
    
    # Z-score statistics
    avg_entry_zscore: float = 0.0
    avg_exit_zscore: float = 0.0
    
    # Raw objects for further analysis
    portfolio: Any = None  # VBT Portfolio object
    prices_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def summary(self) -> str:
        """Generate text summary of backtest results."""
        lines = [
            "=" * 60,
            f"PAIRS TRADING BACKTEST: {self.symbol1} / {self.symbol2}",
            "=" * 60,
            f"Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Interval: {self.interval}",
            "",
            "COINTEGRATION:",
            f"  P-value: {self.cointegration.p_value:.4f}" if self.cointegration else "  N/A",
            f"  Hedge Ratio: {self.hedge_ratio:.4f}",
            f"  Half-life: {self.half_life:.1f} bars",
            "",
            "PERFORMANCE:",
            f"  Total Return: {self.total_return * 100:.2f}%",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {self.sortino_ratio:.2f}",
            f"  Max Drawdown: {self.max_drawdown * 100:.2f}%",
            f"  Calmar Ratio: {self.calmar_ratio:.2f}",
            "",
            "TRADES:",
            f"  Total Trades: {self.num_trades}",
            f"  Win Rate: {self.win_rate * 100:.1f}%",
            f"  Profit Factor: {self.profit_factor:.2f}",
            f"  Avg Trade Return: {self.avg_trade_return * 100:.2f}%",
            f"  Avg Duration: {self.avg_trade_duration:.1f} bars",
            "",
            "FEES:",
            f"  Total Fees: ${self.total_fees_paid:.2f}",
            f"  Gross Return: {self.gross_return * 100:.2f}%",
            f"  Fee Drag: {self.fee_drag * 100:.2f}%",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol1": self.symbol1,
            "symbol2": self.symbol2,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "interval": self.interval,
            "half_life": self.half_life,
            "hedge_ratio": self.hedge_ratio,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "total_fees_paid": self.total_fees_paid,
            "cointegration_pvalue": self.cointegration.p_value if self.cointegration else None,
        }


def fetch_crypto_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Fetch crypto price data via Coinbase public API.
    
    Args:
        symbols: List of trading pairs (e.g., ["BTC-USD", "ETH-USD"])
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Candle interval
        
    Returns:
        DataFrame with MultiIndex columns (symbol, OHLCV)
    """
    try:
        from crypto.exchange.coinbase_client import CoinbaseClient
        client = CoinbaseClient(simulation_mode=True)
    except ImportError:
        client = None
    
    dfs = {}
    
    for symbol in symbols:
        try:
            if client:
                df = client.get_historical_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    limit=5000,  # Get max available
                )
            else:
                # Fallback to VBT data if client not available
                df = _fetch_via_vbt(symbol, start_date, end_date, interval)
            
            if not df.empty:
                # Filter date range
                df = df.loc[start_date:end_date]
                dfs[symbol] = df
                logger.info("Fetched %d bars for %s", len(df), symbol)
                
        except Exception as e:
            logger.error("Failed to fetch %s: %s", symbol, e)
            continue
    
    if not dfs:
        raise ValueError("No data fetched for any symbols")
    
    # Combine into multi-column DataFrame
    combined = pd.concat(dfs, axis=1, keys=dfs.keys())
    
    # Forward fill missing data (different symbols may have gaps)
    combined = combined.ffill()
    
    # Drop rows where any symbol is NaN
    combined = combined.dropna()
    
    return combined


def _fetch_via_vbt(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    """
    Fallback data fetch using VBT Pro's built-in sources.
    """
    # Try Binance for crypto
    binance_symbol = symbol.replace("-USD", "USDT")
    
    try:
        data = vbt.BinanceData.pull(
            binance_symbol,
            start=start_date,
            end=end_date,
            timeframe=interval,
        )
        return data.get()
    except Exception:
        pass
    
    # Try Yahoo Finance
    yf_symbol = symbol.replace("-USD", "-USD")
    try:
        data = vbt.YFData.pull(
            yf_symbol,
            start=start_date,
            end=end_date,
        )
        return data.get()
    except Exception:
        pass
    
    return pd.DataFrame()


def create_fee_func(
    notional_per_trade: float,
    contracts_per_trade: int = 1,
) -> callable:
    """
    Create VBT-compatible fee function for Coinbase CFM.
    
    Args:
        notional_per_trade: Expected notional value per trade
        contracts_per_trade: Expected number of contracts
        
    Returns:
        Fee function: f(col, i, val) -> fee
    """
    def coinbase_fee(col, i, val):
        """Calculate fee per trade value."""
        trade_notional = abs(val)
        
        # Percentage-based fee
        pct_fee = trade_notional * TAKER_FEE_RATE
        
        # Minimum fee floor (estimate contracts from notional)
        est_contracts = max(1, int(trade_notional / notional_per_trade * contracts_per_trade))
        min_fee = MIN_FEE_PER_CONTRACT * est_contracts
        
        return max(pct_fee, min_fee)
    
    return coinbase_fee


def run_pairs_backtest(
    symbol1: str,
    symbol2: str,
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    interval: str = "1h",
    # Z-score parameters
    zscore_window: int = 20,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.0,
    stop_zscore: float = 3.0,
    # Hedge ratio
    hedge_ratio: Optional[float] = None,  # None = calculate from data
    rolling_hedge_window: Optional[int] = None,  # None = fixed hedge ratio
    # Position sizing
    initial_capital: float = 10000.0,
    position_size_pct: float = 10.0,  # % of equity per leg
    # Fees
    include_fees: bool = True,
    # Filters
    min_half_life: float = 1.0,
    max_half_life: float = 168.0,  # 1 week for hourly
    # Data (optional - provide if already fetched)
    prices_df: Optional[pd.DataFrame] = None,
) -> PairsBacktestResult:
    """
    Run complete pairs trading backtest.
    
    Args:
        symbol1: First symbol (long side when spread undervalued)
        symbol2: Second symbol (short side when spread undervalued)
        start_date: Backtest start date
        end_date: Backtest end date  
        interval: Candle interval ("1h", "4h", "1d")
        zscore_window: Rolling window for Z-score calculation
        entry_zscore: Z-score threshold for entry (absolute value)
        exit_zscore: Z-score threshold for exit
        stop_zscore: Z-score stop-loss threshold
        hedge_ratio: Fixed hedge ratio (None to calculate)
        rolling_hedge_window: Window for rolling hedge ratio (None = fixed)
        initial_capital: Starting capital in USD
        position_size_pct: Position size as % of equity per leg
        include_fees: Include Coinbase CFM fees
        min_half_life: Minimum half-life filter (bars)
        max_half_life: Maximum half-life filter (bars)
        prices_df: Pre-fetched price data (optional)
        
    Returns:
        PairsBacktestResult with all metrics and analysis
    """
    logger.info("Starting pairs backtest: %s / %s", symbol1, symbol2)
    
    # Fetch data if not provided
    if prices_df is None:
        prices_df = fetch_crypto_data(
            symbols=[symbol1, symbol2],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
    
    # Extract close prices
    close1 = prices_df[(symbol1, "close")] if (symbol1, "close") in prices_df.columns else prices_df[symbol1]["close"]
    close2 = prices_df[(symbol2, "close")] if (symbol2, "close") in prices_df.columns else prices_df[symbol2]["close"]
    
    # Test cointegration
    coint_result = engle_granger_test(close1, close2, significance_level=0.05)
    
    logger.info(
        "Cointegration test: p-value=%.4f, hedge_ratio=%.4f, half_life=%.1f",
        coint_result.p_value,
        coint_result.hedge_ratio,
        coint_result.half_life,
    )
    
    # Use calculated hedge ratio if not provided
    if hedge_ratio is None:
        hedge_ratio = coint_result.hedge_ratio
    
    # Calculate spread and Z-score
    if rolling_hedge_window:
        hr_series = calculate_hedge_ratio(close1, close2, window=rolling_hedge_window)
        spread = np.log(close1) - hr_series * np.log(close2)
    else:
        spread = calculate_spread(close1, close2, hedge_ratio=hedge_ratio)
    
    zscore = calculate_zscore(spread, window=zscore_window)
    
    # Generate entry signals
    # Long spread (long asset1, short asset2): Z < -entry_zscore
    # Short spread (short asset1, long asset2): Z > entry_zscore
    long_spread_entry = zscore.shift(1) > -entry_zscore
    long_spread_entry &= zscore <= -entry_zscore
    
    short_spread_entry = zscore.shift(1) < entry_zscore
    short_spread_entry &= zscore >= entry_zscore
    
    # Exit signals (Z-score crosses toward zero)
    long_spread_exit = (zscore.shift(1) < exit_zscore) & (zscore >= exit_zscore)
    long_spread_exit |= zscore >= stop_zscore  # Stop out
    
    short_spread_exit = (zscore.shift(1) > exit_zscore) & (zscore <= exit_zscore)
    short_spread_exit |= zscore <= -stop_zscore  # Stop out
    
    # Build signal arrays for VBT
    # Asset 1: Long on long_spread, Short on short_spread
    # Asset 2: Short on long_spread, Long on short_spread
    
    data = pd.concat([close1, close2], axis=1, keys=[symbol1, symbol2])
    
    long_entries = data.copy()
    long_entries[:] = False
    short_entries = data.copy()
    short_entries[:] = False
    
    # When long spread: long asset1, short asset2
    long_entries.loc[long_spread_entry, symbol1] = True
    short_entries.loc[long_spread_entry, symbol2] = True
    
    # When short spread: short asset1, long asset2
    short_entries.loc[short_spread_entry, symbol1] = True
    long_entries.loc[short_spread_entry, symbol2] = True
    
    # Exits
    exits = data.copy()
    exits[:] = False
    exits.loc[long_spread_exit | short_spread_exit, :] = True
    
    # Fee function
    if include_fees:
        fee_func = create_fee_func(
            notional_per_trade=initial_capital * position_size_pct / 100,
            contracts_per_trade=10,  # Estimate
        )
    else:
        fee_func = 0.0
    
    # Run VBT simulation
    pf = vbt.Portfolio.from_signals(
        data,
        entries=long_entries,
        short_entries=short_entries,
        exits=exits,
        size=position_size_pct,
        size_type="valuepercent100",
        group_by=True,  # Group as single portfolio
        cash_sharing=True,
        call_seq="auto",  # Sell before buy
        init_cash=initial_capital,
        fees=fee_func,
    )
    
    # Extract performance metrics
    stats = pf.stats()
    
    # Trade-level metrics
    trades = pf.trades
    if len(trades.records_arr) > 0:
        num_trades = len(trades.records_arr)
        winning_trades = trades.winning
        losing_trades = trades.losing
        
        win_rate = len(winning_trades.records_arr) / num_trades if num_trades > 0 else 0
        
        avg_win = winning_trades.pnl.mean() if len(winning_trades.records_arr) > 0 else 0
        avg_loss = abs(losing_trades.pnl.mean()) if len(losing_trades.records_arr) > 0 else 0
        
        profit_factor = avg_win * len(winning_trades.records_arr) / (avg_loss * len(losing_trades.records_arr)) if len(losing_trades.records_arr) > 0 and avg_loss > 0 else np.inf
        
        avg_trade_return = trades.pnl.mean() / initial_capital if num_trades > 0 else 0
        avg_duration = trades.duration.mean() if num_trades > 0 else 0
    else:
        num_trades = 0
        win_rate = 0
        profit_factor = 0
        avg_trade_return = 0
        avg_win = 0
        avg_loss = 0
        avg_duration = 0
    
    # Fee calculations
    total_fees = float(stats.get("Total Fees Paid", 0))
    total_return = float(stats.get("Total Return [%]", 0)) / 100
    
    # Estimate gross return (return + fees as % of capital)
    gross_return = total_return + (total_fees / initial_capital) if initial_capital > 0 else total_return
    fee_drag = total_fees / (initial_capital * gross_return) if gross_return > 0 else 0
    
    # Build result
    result = PairsBacktestResult(
        symbol1=symbol1,
        symbol2=symbol2,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        interval=interval,
        cointegration=coint_result,
        half_life=coint_result.half_life,
        hedge_ratio=hedge_ratio,
        total_return=total_return,
        sharpe_ratio=float(stats.get("Sharpe Ratio", 0)),
        sortino_ratio=float(stats.get("Sortino Ratio", 0)),
        max_drawdown=float(stats.get("Max Drawdown [%]", 0)) / 100,
        calmar_ratio=float(stats.get("Calmar Ratio", 0)),
        num_trades=num_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_return=avg_trade_return,
        avg_winning_trade=avg_win / initial_capital if avg_win else 0,
        avg_losing_trade=avg_loss / initial_capital if avg_loss else 0,
        avg_trade_duration=avg_duration,
        total_fees_paid=total_fees,
        gross_return=gross_return,
        fee_drag=fee_drag,
        portfolio=pf,
        prices_df=prices_df,
        signals_df=pd.DataFrame({
            "spread": spread,
            "zscore": zscore,
            "long_spread_entry": long_spread_entry,
            "short_spread_entry": short_spread_entry,
        }),
    )
    
    return result


def run_parameter_scan(
    symbol1: str,
    symbol2: str,
    start_date: str,
    end_date: str,
    interval: str = "1h",
    zscore_windows: List[int] = [10, 20, 30, 60],
    entry_thresholds: List[float] = [1.5, 2.0, 2.5, 3.0],
    exit_thresholds: List[float] = [-0.5, 0.0, 0.5],
    initial_capital: float = 10000.0,
    position_size_pct: float = 10.0,
) -> pd.DataFrame:
    """
    Run parameter optimization scan.
    
    Args:
        symbol1: First symbol
        symbol2: Second symbol  
        start_date: Start date
        end_date: End date
        interval: Timeframe
        zscore_windows: List of Z-score windows to test
        entry_thresholds: List of entry thresholds to test
        exit_thresholds: List of exit thresholds to test
        initial_capital: Starting capital
        position_size_pct: Position size %
        
    Returns:
        DataFrame with results for each parameter combination
    """
    # Fetch data once
    prices_df = fetch_crypto_data(
        symbols=[symbol1, symbol2],
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )
    
    results = []
    
    total = len(zscore_windows) * len(entry_thresholds) * len(exit_thresholds)
    count = 0
    
    for window in zscore_windows:
        for entry in entry_thresholds:
            for exit_th in exit_thresholds:
                count += 1
                logger.info("Running %d/%d: window=%d, entry=%.1f, exit=%.1f",
                           count, total, window, entry, exit_th)
                
                try:
                    result = run_pairs_backtest(
                        symbol1=symbol1,
                        symbol2=symbol2,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        zscore_window=window,
                        entry_zscore=entry,
                        exit_zscore=exit_th,
                        initial_capital=initial_capital,
                        position_size_pct=position_size_pct,
                        prices_df=prices_df,
                    )
                    
                    results.append({
                        "zscore_window": window,
                        "entry_threshold": entry,
                        "exit_threshold": exit_th,
                        **result.to_dict(),
                    })
                    
                except Exception as e:
                    logger.warning("Failed: %s", e)
                    continue
    
    return pd.DataFrame(results)


# Convenience function for quick testing
def quick_test(
    symbol1: str = "BTC-USD",
    symbol2: str = "ETH-USD",
    days: int = 90,
) -> PairsBacktestResult:
    """
    Quick test with default parameters.
    
    Args:
        symbol1: First symbol
        symbol2: Second symbol
        days: Number of days to backtest
        
    Returns:
        PairsBacktestResult
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    
    return run_pairs_backtest(
        symbol1=symbol1,
        symbol2=symbol2,
        start_date=start_date,
        end_date=end_date,
        interval="1h",
        zscore_window=24,  # 1 day for hourly
        entry_zscore=2.0,
        exit_zscore=0.0,
        initial_capital=1000.0,
        position_size_pct=20.0,
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    result = quick_test("BTC-USD", "ETH-USD", days=90)
    print(result.summary())
