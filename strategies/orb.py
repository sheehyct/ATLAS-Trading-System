"""
Opening Range Breakout (ORB) Strategy - Refactored to BaseStrategy

Strategy Description:
- Entry: Breakout of first 30-minute range with directional bias and volume confirmation
- Exit: End-of-day (3:55 PM ET) OR ATR-based stop loss
- Position Sizing: ATR-based with capital constraint

Research-Backed Requirements (MANDATORY):
1. Volume confirmation: 2.0x average volume (HARDCODED)
2. Directional bias: Opening bar close > open for longs
3. NO signal exits (RSI/MACD would cut winners)
4. ATR stops: 2.5x multiplier baseline
5. Target metrics: Sharpe > 2.0, R:R > 3:1, Net expectancy > 0.5%

Reference: STRATEGY_2_IMPLEMENTATION_ADDENDUM.md
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import time
from typing import Dict, Tuple, Optional
import pandas_market_calendars as mcal
import os
from dotenv import load_dotenv
from pydantic import Field

from strategies.base_strategy import BaseStrategy, StrategyConfig
from utils.position_sizing import calculate_position_size_atr

# Load environment variables from root .env
load_dotenv()


class ORBConfig(StrategyConfig):
    """
    Configuration for Opening Range Breakout strategy.

    Extends StrategyConfig with ORB-specific parameters.
    """
    # ORB-specific parameters
    symbol: str = 'SPY'
    opening_minutes: int = Field(default=30, ge=5, le=60)
    atr_period: int = Field(default=14, ge=5, le=30)
    atr_stop_multiplier: float = Field(default=2.5, ge=1.0, le=5.0)
    volume_multiplier: float = Field(default=2.0, ge=1.5, le=3.0)

    # Data fetching
    start_date: str = '2016-01-01'
    end_date: str = '2025-10-14'


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy inheriting from BaseStrategy.

    Implements mandatory volume confirmation, directional bias,
    and ATR-based position sizing.

    Usage:
        >>> config = ORBConfig(name="ORB", symbol="SPY", risk_per_trade=0.02)
        >>> strategy = ORBStrategy(config)
        >>> data_5min, data_daily = strategy.fetch_data()
        >>> pf = strategy.backtest(data_5min, initial_capital=10000)
        >>> metrics = strategy.get_performance_metrics(pf)
    """

    def __init__(self, config: ORBConfig):
        """Initialize ORB strategy with configuration"""
        super().__init__(config)

        # Type hint for IDE support
        self.config: ORBConfig = config

        # Data storage (populated by fetch_data())
        self.data_5min = None
        self.data_daily = None
        self._opening_range_cache = None
        self._atr_cache = None

    def fetch_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch intraday and daily data from Alpaca.

        Args:
            start_date: Start date (YYYY-MM-DD), defaults to config
            end_date: End date (YYYY-MM-DD), defaults to config

        Returns:
            Tuple of (5min_data, daily_data)

        Note:
            This method is NOT part of BaseStrategy abstract interface.
            It's ORB-specific for data fetching. The fetched data is then
            passed to backtest() method.
        """
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date

        # Get Alpaca credentials from environment (MID account has Algo Trader Plus)
        api_key = os.getenv('ALPACA_MID_KEY')
        api_secret = os.getenv('ALPACA_MID_SECRET')

        if not api_key or not api_secret:
            raise ValueError(
                "Alpaca MID account credentials not found. "
                "Ensure ALPACA_MID_KEY and ALPACA_MID_SECRET are set in config/.env"
            )

        # Configure Alpaca credentials
        vbt.AlpacaData.set_custom_settings(
            client_config=dict(
                api_key=api_key,
                secret_key=api_secret
            )
        )

        # Fetch 5-minute data for intraday signals
        data_5min = vbt.AlpacaData.pull(
            self.config.symbol,
            start=start,
            end=end,
            timeframe='5Min',
            tz='America/New_York'
        ).get()

        # Fetch daily data for ATR calculation
        data_daily = vbt.AlpacaData.pull(
            self.config.symbol,
            start=start,
            end=end,
            timeframe='1D',
            tz='America/New_York'
        ).get()

        # CRITICAL: Filter to RTH only (9:30 AM - 4:00 PM ET)
        # This is the bug we're fixing - between_time filters intraday data
        print(f"DEBUG: Before RTH filter: {len(data_5min)} bars")
        data_5min = data_5min.between_time('09:30', '16:00')
        print(f"DEBUG: After RTH filter: {len(data_5min)} bars")

        # Filter NYSE trading days only (no weekends/holidays)
        nyse = mcal.get_calendar('NYSE')
        trading_days = nyse.valid_days(start_date=start, end_date=end)

        print(f"DEBUG: Trading days from calendar: {len(trading_days)}")

        # FIX: Normalize both sides to dates (remove time and timezone) for proper comparison
        data_5min_dates = pd.Series(data_5min.index.date, index=data_5min.index)
        trading_days_dates = pd.DatetimeIndex(trading_days).date

        # Filter using date-only comparison
        data_5min = data_5min[data_5min_dates.isin(trading_days_dates)]
        data_daily_dates = pd.Series(data_daily.index.date, index=data_daily.index)
        data_daily = data_daily[data_daily_dates.isin(trading_days_dates)]

        print(f"DEBUG: After trading days filter: {len(data_5min)} 5min bars, {len(data_daily)} daily bars")

        # Store for use by generate_signals()
        self.data_5min = data_5min
        self.data_daily = data_daily

        print(f"\nFetched 5-minute data: {len(data_5min)} bars")
        print(f"Fetched daily data: {len(data_daily)} bars")

        if len(data_5min) > 0:
            print(f"Date range: {data_5min.index[0]} to {data_5min.index[-1]}")
        else:
            print("WARNING: No data after filtering! Check date range and filters.")

        return data_5min, data_daily

    def _calculate_opening_range(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate opening range for each trading day.

        Args:
            data: 5-minute OHLCV DataFrame

        Returns:
            Dict containing opening_high, opening_low, opening_close, opening_open
        """
        # Group by trading day
        daily_groups = data.groupby(data.index.date)

        opening_high_list = []
        opening_low_list = []
        opening_close_list = []
        opening_open_list = []
        dates = []

        # Calculate opening range for each day
        n_bars = self.config.opening_minutes // 5  # Convert minutes to 5-min bars

        for date, day_data in daily_groups:
            # Get first N bars (opening range)
            opening_bars = day_data.iloc[:n_bars]

            if len(opening_bars) < n_bars:
                # Skip days with insufficient data
                continue

            opening_high_list.append(opening_bars['High'].max())
            opening_low_list.append(opening_bars['Low'].min())
            opening_close_list.append(opening_bars['Close'].iloc[-1])
            opening_open_list.append(opening_bars['Open'].iloc[0])
            # Make timestamp timezone-aware to match intraday data
            dates.append(pd.Timestamp(date, tz=data.index.tz))

        # Create Series with daily values
        opening_high = pd.Series(opening_high_list, index=dates)
        opening_low = pd.Series(opening_low_list, index=dates)
        opening_close = pd.Series(opening_close_list, index=dates)
        opening_open = pd.Series(opening_open_list, index=dates)

        # Broadcast daily values to intraday bars using map by date pattern
        # VBT-verified pattern from mcp__vectorbt-pro__search (Discord examples)
        # Reference: "entries.reindex(price_data.index, method='ffill')"
        # Pattern: Create dict mapping date -> value, then map intraday dates to values
        # This correctly handles timezone-aware indices and avoids NaN propagation
        opening_high_dict = dict(zip(opening_high.index.date, opening_high.values))
        opening_low_dict = dict(zip(opening_low.index.date, opening_low.values))
        opening_close_dict = dict(zip(opening_close.index.date, opening_close.values))
        opening_open_dict = dict(zip(opening_open.index.date, opening_open.values))

        intraday_dates = pd.Series(data.index.date, index=data.index)
        opening_high_ff = intraday_dates.map(opening_high_dict)
        opening_low_ff = intraday_dates.map(opening_low_dict)
        opening_close_ff = intraday_dates.map(opening_close_dict)
        opening_open_ff = intraday_dates.map(opening_open_dict)

        return {
            'opening_high': opening_high_ff,
            'opening_low': opening_low_ff,
            'opening_close': opening_close_ff,
            'opening_open': opening_open_ff
        }

    def generate_signals(self, data: pd.DataFrame, regime: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Generate entry/exit signals with MANDATORY volume confirmation.

        Implements BaseStrategy abstract method.

        Args:
            data: 5-minute OHLCV DataFrame with DatetimeIndex
            regime: Optional market regime (currently not used by ORB)

        Returns:
            Dict containing:
            - long_entries: Boolean Series for long entries
            - long_exits: Boolean Series for long exits (EOD)
            - short_entries: Boolean Series for short entries (if enabled)
            - short_exits: Boolean Series for short exits (EOD)
            - stop_distance: ATR-based stop distance
            - volume_confirmed: Boolean Series tracking volume filter
        """
        # Calculate opening range
        opening_range = self._calculate_opening_range(data)
        opening_high = opening_range['opening_high']
        opening_low = opening_range['opening_low']
        opening_close = opening_range['opening_close']
        opening_open = opening_range['opening_open']

        # Directional bias from opening bar
        bullish_opening = opening_close > opening_open
        bearish_opening = opening_close < opening_open

        # Price breakout signals
        price_breakout_long = data['Close'] > opening_high
        price_breakout_short = data['Close'] < opening_low

        # CRITICAL: Volume confirmation (MANDATORY, HARDCODED at 2.0x)
        volume_ma = data['Volume'].rolling(window=20).mean()
        volume_surge = data['Volume'] > (volume_ma * self.config.volume_multiplier)

        # Calculate ATR for stops (needs daily data)
        # If data_daily not available, estimate from 5min data
        if self.data_daily is not None and len(self.data_daily) > 0:
            atr_indicator = vbt.talib("ATR").run(
                self.data_daily['High'],
                self.data_daily['Low'],
                self.data_daily['Close'],
                timeperiod=self.config.atr_period
            )
            atr_daily = atr_indicator.real

            # Forward-fill ATR to intraday bars using map by date pattern
            atr_dict = dict(zip(atr_daily.index.date, atr_daily.values))
            intraday_dates_atr = pd.Series(data.index.date, index=data.index)
            atr_intraday = intraday_dates_atr.map(atr_dict)
        else:
            # Fallback: Calculate ATR from 5min data (less ideal)
            atr_indicator = vbt.talib("ATR").run(
                data['High'],
                data['Low'],
                data['Close'],
                timeperiod=self.config.atr_period * 78  # Approximate daily equivalent
            )
            atr_intraday = atr_indicator.real

        stop_distance = atr_intraday * self.config.atr_stop_multiplier

        # Time filter: Only allow entries after opening range ends
        # Calculate entry start time (9:30 AM + opening_minutes)
        from datetime import datetime, timedelta
        market_open = datetime.strptime("09:30", "%H:%M")
        entry_start = market_open + timedelta(minutes=self.config.opening_minutes)
        entry_start_time = entry_start.time()  # e.g., 10:00 AM for 30-min range
        can_enter = data.index.time >= entry_start_time

        # Generate entry signals (ALL conditions required)
        long_entries = (
            price_breakout_long &
            bullish_opening &
            volume_surge &
            can_enter
        )

        short_entries = (
            price_breakout_short &
            bearish_opening &
            volume_surge &
            can_enter &
            self.config.enable_shorts
        )

        # EOD exit signals (3:55 PM ET - 5 minutes before close)
        eod_time = time(15, 55)
        eod_exit = pd.Series(data.index.time == eod_time, index=data.index)

        # Print signal summary
        total_long = long_entries.sum()
        total_short = short_entries.sum()
        volume_confirmed_pct = volume_surge.sum() / len(volume_surge) * 100

        print(f"\n=== Signal Generation Summary ===")
        print(f"Long entry signals: {total_long}")
        print(f"Short entry signals: {total_short}")
        print(f"Volume confirmation rate: {volume_confirmed_pct:.1f}%")
        print(f"Avg ATR: ${atr_intraday.mean():.2f}")
        print(f"Avg stop distance: ${stop_distance.mean():.2f}")

        return {
            'long_entries': long_entries,
            'long_exits': eod_exit,
            'short_entries': short_entries,
            'short_exits': eod_exit,
            'stop_distance': stop_distance,
            # Extra info for analysis
            'atr': atr_intraday,
            'volume_confirmed': volume_surge,
            'volume_ma': volume_ma,
            'opening_high': opening_high,
            'opening_low': opening_low
        }

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes using ATR-based method with capital constraint.

        Implements BaseStrategy abstract method.

        Args:
            data: OHLCV DataFrame
            capital: Current account capital
            stop_distance: ATR-based stop distances

        Returns:
            Position sizes (number of shares) as pandas Series
        """
        # Use utils/position_sizing.py (Gate 1 validated)
        position_sizes, actual_risks, constrained = calculate_position_size_atr(
            init_cash=capital,
            close=data['Close'],
            atr=stop_distance / self.config.atr_stop_multiplier,  # Back out ATR from stop distance
            atr_multiplier=self.config.atr_stop_multiplier,
            risk_pct=self.config.risk_per_trade
        )

        return position_sizes

    def get_strategy_name(self) -> str:
        """Return strategy name for logging/reporting"""
        return f"Opening Range Breakout (ORB) - {self.config.symbol}"

    def validate_parameters(self) -> bool:
        """
        Validate ORB-specific parameters.

        Checks:
        - opening_minutes is reasonable (5-60 minutes)
        - atr_stop_multiplier is within acceptable range (1.5-5.0)

        Returns:
            True if all parameters valid

        Raises:
            AssertionError: If validation fails
        """
        assert 5 <= self.config.opening_minutes <= 60, \
            f"opening_minutes {self.config.opening_minutes} outside range [5, 60]"

        assert 1.5 <= self.config.atr_stop_multiplier <= 5.0, \
            f"atr_stop_multiplier {self.config.atr_stop_multiplier} outside range [1.5, 5.0]"

        return True

    # Additional helper methods (not part of BaseStrategy interface)

    def analyze_expectancy(
        self,
        pf: vbt.Portfolio,
        transaction_costs: float = 0.0035
    ) -> Dict:
        """
        Comprehensive expectancy analysis with efficiency factors.

        MANDATORY Gate 2 requirement from STRATEGY_2_IMPLEMENTATION_ADDENDUM.md

        Args:
            pf: VectorBT Portfolio from backtest()
            transaction_costs: Total costs per trade (default: 0.35%)

        Returns:
            Dict with expectancy metrics and viability assessment
        """
        trades = pf.trades

        if trades.count() == 0:
            print("[WARNING] No trades executed - cannot calculate expectancy")
            return None

        win_rate = trades.win_rate

        # Use VBT Pro built-in properties
        if trades.winning.count() > 0:
            avg_win = trades.winning.returns.mean()
        else:
            avg_win = 0.0

        if trades.losing.count() > 0:
            avg_loss = abs(trades.losing.returns.mean())
        else:
            avg_loss = 0.0

        if avg_loss == 0:
            print("[WARNING] No losing trades - cannot calculate R:R")
            rr_ratio = np.inf if avg_win > 0 else 0
        else:
            rr_ratio = avg_win / avg_loss

        # 1. Theoretical expectancy
        theoretical_exp = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # 2. Realized expectancy (80% efficiency from fixed fractional sizing)
        efficiency_factor = 0.80
        realized_exp = theoretical_exp * efficiency_factor

        # 3. Net expectancy (after transaction costs)
        net_exp = realized_exp - transaction_costs

        # Print detailed report
        print("\n" + "=" * 70)
        print("EXPECTANCY ANALYSIS (Gate 2 Requirement)")
        print("=" * 70)

        print(f"\nInput Statistics:")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Winner: {avg_win:.2%}")
        print(f"  Avg Loser: {avg_loss:.2%}")
        print(f"  R:R Ratio: {rr_ratio:.2f}:1")

        print(f"\nExpectancy Breakdown:")
        print(f"  1. Theoretical: {theoretical_exp:.4f} ({theoretical_exp*100:.2f}% per trade)")
        print(f"     Formula: ({win_rate:.2%} x {avg_win:.2%}) - ({1-win_rate:.2%} x {avg_loss:.2%})")

        print(f"\n  2. Realized ({efficiency_factor:.0%} efficiency):")
        print(f"     {realized_exp:.4f} ({realized_exp*100:.2f}% per trade)")
        print(f"     Reason: Fixed fractional sizing drag")

        print(f"\n  3. Net (after costs):")
        print(f"     {net_exp:.4f} ({net_exp*100:.2f}% per trade)")
        print(f"     Costs: {transaction_costs:.2%} per trade")

        # Viability assessment
        print(f"\n{'=' * 70}")
        print("VIABILITY ASSESSMENT")
        print(f"{'=' * 70}")

        viable = False
        assessment = ""

        if net_exp >= 0.008:
            assessment = "[PASS] EXCELLENT"
            detail = "Net expectancy > 0.8% per trade - comfortable margin"
            viable = True
        elif net_exp >= 0.005:
            assessment = "[PASS] GOOD"
            detail = "Net expectancy > 0.5% per trade - viable strategy"
            viable = True
        elif net_exp >= 0.003:
            assessment = "[FAIL] MARGINAL"
            detail = "Net expectancy 0.3-0.5% - barely viable, sensitive to costs"
            viable = False
        elif net_exp >= 0.000:
            assessment = "[FAIL] BREAKEVEN"
            detail = "Net expectancy near zero - not profitable"
            viable = False
        else:
            assessment = "[FAIL] NEGATIVE"
            detail = "Negative net expectancy - losing money"
            viable = False

        print(f"\n{assessment}")
        print(f"  {detail}")

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'rr_ratio': rr_ratio,
            'theoretical': theoretical_exp,
            'realized': realized_exp,
            'net': net_exp,
            'viable': viable,
            'assessment': assessment
        }
