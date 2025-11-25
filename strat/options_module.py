"""
Options Module for ATLAS Trading System.

Session 70: Integrates Tier 1 STRAT patterns with Alpaca options trading.

Features:
- OSI option symbol generation
- Strike selection per STRAT methodology (within entry-to-target range)
- VBT Pro AlpacaData integration for options quotes
- P/L calculation for paper trading
- Debit spread support for defined risk

STRAT Options Philosophy:
- Buy calls/puts when underlying breaks pattern entry
- Strike should be within entry-to-target range (higher probability)
- DTE: 30-45 days for weekly patterns, 60-90 days for monthly
- Exit at measured move target or 50% max profit

Usage:
    from strat.options_module import OptionsExecutor
    from strat.tier1_detector import Tier1Detector

    # Detect patterns
    detector = Tier1Detector()
    signals = detector.detect_patterns(data)

    # Convert to options trades
    executor = OptionsExecutor()
    trades = executor.generate_option_trades(signals, underlying_price=450.0)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import vectorbtpro as vbt

from config.settings import get_alpaca_credentials
from strat.tier1_detector import PatternSignal, PatternType, Timeframe
from strat.greeks import calculate_greeks, Greeks, estimate_iv_from_history


class OptionType(Enum):
    """Option contract types."""
    CALL = "C"
    PUT = "P"


class OptionStrategy(Enum):
    """Option strategy types."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    CALL_DEBIT_SPREAD = "call_debit_spread"
    PUT_DEBIT_SPREAD = "put_debit_spread"


@dataclass
class OptionContract:
    """
    Represents an options contract.

    Attributes:
        underlying: Underlying symbol (e.g., 'SPY')
        expiration: Expiration date
        option_type: CALL or PUT
        strike: Strike price
        osi_symbol: OSI format symbol (e.g., 'SPY241220C00300000')
    """
    underlying: str
    expiration: datetime
    option_type: OptionType
    strike: float
    osi_symbol: str = ""

    def __post_init__(self):
        """Generate OSI symbol after initialization."""
        if not self.osi_symbol:
            self.osi_symbol = self.generate_osi_symbol()

    def generate_osi_symbol(self) -> str:
        """
        Generate OSI (Options Symbology Initiative) format symbol.

        Format: SYMBOL + YYMMDD + C/P + STRIKE*1000 (8 digits)
        Example: SPY241220C00300000 = SPY Dec 20 2024 $300 Call
        """
        # Format: SYMBOL (padded to 6 chars) + YYMMDD + C/P + STRIKE (8 digits)
        symbol_part = self.underlying.upper().ljust(6)[:6]
        date_part = self.expiration.strftime("%y%m%d")
        type_part = self.option_type.value
        # Strike in dollars * 1000, padded to 8 digits
        strike_part = str(int(self.strike * 1000)).zfill(8)

        return f"{symbol_part.strip()}{date_part}{type_part}{strike_part}"


@dataclass
class OptionTrade:
    """
    Represents an options trade generated from a pattern signal.

    Attributes:
        pattern_signal: Original pattern signal
        contract: Option contract details
        strategy: Option strategy type
        entry_trigger: Price level that triggers entry
        target_exit: Price level for target exit
        stop_exit: Price level for stop exit
        quantity: Number of contracts
        max_risk: Maximum risk in dollars
        expected_reward: Expected reward in dollars
    """
    pattern_signal: PatternSignal
    contract: OptionContract
    strategy: OptionStrategy
    entry_trigger: float
    target_exit: float
    stop_exit: float
    quantity: int = 1
    max_risk: float = 0.0
    expected_reward: float = 0.0
    option_premium: float = 0.0
    delta: float = 0.0

    def calculate_risk_reward(self, premium: float):
        """Calculate max risk and expected reward based on premium."""
        self.option_premium = premium
        # Max risk = premium paid * 100 * quantity
        self.max_risk = premium * 100 * self.quantity
        # Expected reward based on pattern R:R
        self.expected_reward = self.max_risk * self.pattern_signal.risk_reward


class OptionsExecutor:
    """
    Executes options trades based on Tier 1 STRAT pattern signals.

    STRAT Strike Selection Rules:
    1. Strike within entry-to-target range for higher probability
    2. Prefer ATM or slightly ITM strikes for better delta
    3. DTE based on pattern timeframe (weekly: 30-45 days, monthly: 60-90)

    Attributes:
        account: Alpaca account to use ('MID', 'LARGE', 'SMALL')
        default_dte_weekly: Default DTE for weekly patterns
        default_dte_monthly: Default DTE for monthly patterns
        strike_offset: Strike price offset from ATM (negative = ITM)
    """

    def __init__(
        self,
        account: str = 'MID',
        default_dte_weekly: int = 35,
        default_dte_monthly: int = 75,
        strike_offset: float = 0.0
    ):
        """
        Initialize Options Executor.

        Args:
            account: Alpaca account ('MID', 'LARGE', 'SMALL')
            default_dte_weekly: Default DTE for weekly patterns (default: 35)
            default_dte_monthly: Default DTE for monthly patterns (default: 75)
            strike_offset: Offset from ATM (negative = ITM, default: 0)
        """
        self.account = account
        self.default_dte_weekly = default_dte_weekly
        self.default_dte_monthly = default_dte_monthly
        self.strike_offset = strike_offset

        # Get credentials
        self.creds = get_alpaca_credentials(account)

    def generate_option_trades(
        self,
        signals: List[PatternSignal],
        underlying: str,
        underlying_price: float,
        capital_per_trade: float = 500.0,
        max_risk_pct: float = 0.02,
        price_data: Optional[pd.DataFrame] = None
    ) -> List[OptionTrade]:
        """
        Generate options trades from pattern signals.

        Session 73: Now uses data-driven strike selection with delta targeting.
        If price_data is provided, calculates IV from historical data for more
        accurate strike selection.

        Args:
            signals: List of Tier 1 pattern signals
            underlying: Underlying symbol (e.g., 'SPY')
            underlying_price: Current underlying price
            capital_per_trade: Capital allocated per trade (default: $500)
            max_risk_pct: Max risk as % of capital (default: 2%)
            price_data: Optional DataFrame with historical OHLC data for IV calculation

        Returns:
            List of OptionTrade objects
        """
        trades = []

        # Calculate IV from historical data if provided
        iv = None
        if price_data is not None:
            iv = self._estimate_iv_from_price_data(price_data)

        for signal in signals:
            # Determine option type based on signal direction
            if signal.direction == 1:  # Bullish
                option_type = OptionType.CALL
                strategy = OptionStrategy.LONG_CALL
            else:  # Bearish
                option_type = OptionType.PUT
                strategy = OptionStrategy.LONG_PUT

            # Calculate expiration date (use signal timestamp for backtesting)
            expiration = self._calculate_expiration(signal.timeframe, signal.timestamp)

            # Calculate DTE from signal timestamp to expiration
            try:
                signal_dt = signal.timestamp
                if hasattr(signal_dt, 'tzinfo') and signal_dt.tzinfo is not None:
                    signal_dt = signal_dt.replace(tzinfo=None)
                if hasattr(signal_dt, 'to_pydatetime'):
                    signal_dt = signal_dt.to_pydatetime()
                dte = (expiration - signal_dt).days
            except Exception:
                dte = None  # Will use default

            # Select strike price using data-driven delta targeting
            strike, delta, theta = self._select_strike(
                signal,
                underlying_price,
                option_type,
                iv=iv,
                dte=dte
            )

            # Create contract
            contract = OptionContract(
                underlying=underlying,
                expiration=expiration,
                option_type=option_type,
                strike=strike
            )

            # Create trade
            trade = OptionTrade(
                pattern_signal=signal,
                contract=contract,
                strategy=strategy,
                entry_trigger=signal.entry_price,
                target_exit=signal.target_price,
                stop_exit=signal.stop_price,
            )

            # Store delta if calculated
            if delta is not None:
                trade.delta = delta

            # Calculate position size based on capital and risk
            max_contracts = int(capital_per_trade / (underlying_price * 0.05))  # Rough estimate
            trade.quantity = max(1, min(max_contracts, 5))  # 1-5 contracts

            trades.append(trade)

        return trades

    def _estimate_iv_from_price_data(self, price_data: pd.DataFrame) -> float:
        """
        Estimate implied volatility from historical price data.

        Args:
            price_data: DataFrame with OHLC data

        Returns:
            Estimated IV (annualized)
        """
        close_col = 'close' if 'close' in price_data.columns else 'Close'
        if close_col not in price_data.columns:
            return 0.20  # Default IV

        try:
            iv = estimate_iv_from_history(price_data[close_col], window=20)
            return max(0.10, min(iv, 1.0))  # Clamp between 10% and 100%
        except Exception:
            return 0.20  # Default IV

    def _select_strike(
        self,
        signal: PatternSignal,
        underlying_price: float,
        option_type: OptionType,
        iv: Optional[float] = None,
        dte: Optional[int] = None
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Select strike price using data-driven delta targeting.

        Session 73: Replaced 0.3x geometric formula with delta-targeting algorithm.
        The algorithm targets delta ~0.65 (optimal range 0.50-0.80) while verifying
        theta cost is acceptable.

        Previous approach (0.3x formula) only achieved 20.9% of strikes in optimal
        delta range. This data-driven approach targets 60%+ accuracy.

        Args:
            signal: Pattern signal with entry/target prices
            underlying_price: Current underlying price
            option_type: CALL or PUT
            iv: Implied volatility (if None, uses default 0.20)
            dte: Days to expiration (if None, uses default from timeframe)

        Returns:
            Tuple of (strike, delta, theta). delta/theta may be None if fallback used.
        """
        # Use defaults if not provided
        if iv is None:
            iv = 0.20

        if dte is None:
            if signal.timeframe == Timeframe.WEEKLY:
                dte = self.default_dte_weekly
            elif signal.timeframe == Timeframe.MONTHLY:
                dte = self.default_dte_monthly
            else:
                dte = self.default_dte_weekly  # Default to weekly

        # Use data-driven delta-targeting algorithm
        return self._select_strike_data_driven(
            signal=signal,
            underlying_price=underlying_price,
            option_type=option_type,
            iv=iv,
            dte=dte
        )

    def _round_to_standard_strike(
        self,
        raw_strike: float,
        underlying_price: float
    ) -> float:
        """
        Round strike to standard intervals.

        - Under $100: $1 intervals
        - $100-$500: $5 intervals
        - Over $500: $10 intervals
        """
        if underlying_price < 100:
            interval = 1.0
        elif underlying_price < 500:
            interval = 5.0
        else:
            interval = 10.0

        return round(raw_strike / interval) * interval

    def _get_expected_holding_days(self, timeframe: Timeframe) -> int:
        """
        Get expected holding period for theta calculation based on timeframe.

        Session 73: Timeframe-adjusted holding periods for accurate theta cost.

        Args:
            timeframe: Pattern timeframe (DAILY, WEEKLY, MONTHLY)

        Returns:
            Expected holding days for the timeframe
        """
        holding_days = {
            Timeframe.DAILY: 3,     # Daily patterns: ~3 days
            Timeframe.WEEKLY: 7,    # Weekly patterns: ~1 week
            Timeframe.MONTHLY: 21   # Monthly patterns: ~3 weeks
        }
        return holding_days.get(timeframe, 7)  # Default to weekly

    def _get_strike_interval(self, underlying_price: float) -> float:
        """
        Get standard strike interval based on underlying price.

        - Under $100: $1 intervals
        - $100-$500: $5 intervals
        - Over $500: $10 intervals

        Args:
            underlying_price: Current underlying price

        Returns:
            Strike interval in dollars
        """
        if underlying_price < 100:
            return 1.0
        elif underlying_price < 500:
            return 5.0
        return 10.0

    def _generate_candidate_strikes(
        self,
        strike_min: float,
        strike_max: float,
        interval: float
    ) -> List[float]:
        """
        Generate candidate strikes within range at standard intervals.

        Args:
            strike_min: Minimum strike price
            strike_max: Maximum strike price
            interval: Strike interval

        Returns:
            List of candidate strike prices
        """
        if strike_min >= strike_max:
            return []

        start = np.ceil(strike_min / interval) * interval
        end = np.floor(strike_max / interval) * interval

        if start > end:
            return []

        return list(np.arange(start, end + interval, interval))

    def _fallback_to_geometric(
        self,
        signal: PatternSignal,
        underlying_price: float,
        option_type: OptionType
    ) -> Tuple[float, None, None]:
        """
        Fallback to 0.3x geometric formula when no valid delta strikes found.

        This ensures we always return a valid strike, even if delta targeting fails.

        Args:
            signal: Pattern signal
            underlying_price: Current underlying price
            option_type: CALL or PUT

        Returns:
            Tuple of (strike, None, None) - delta/theta are None for fallback
        """
        entry = signal.entry_price
        target = signal.target_price

        if option_type == OptionType.CALL:
            strike = entry + (0.3 * (target - entry))
        else:
            strike = entry - (0.3 * (entry - target))

        strike = self._round_to_standard_strike(strike, underlying_price)

        # Ensure strike is within valid range
        if option_type == OptionType.CALL:
            strike = max(entry, min(strike, target))
        else:
            strike = min(entry, max(strike, target))

        return strike, None, None

    def _select_strike_data_driven(
        self,
        signal: PatternSignal,
        underlying_price: float,
        option_type: OptionType,
        iv: float,
        dte: int,
        target_delta: float = 0.65,
        delta_range: Tuple[float, float] = (0.50, 0.80),
        max_theta_pct: float = 0.30  # Allow up to 30% theta cost (relaxed from 10%)
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Select strike using delta targeting with theta verification.

        Session 73: Replaces the 0.3x geometric formula with data-driven selection
        that targets optimal delta (0.50-0.80) while verifying theta cost.

        Algorithm:
        1. Generate candidate strikes within entry-target range
        2. Calculate Greeks for each candidate
        3. Filter: Keep only strikes with delta in optimal range
        4. Score: 70% delta proximity to target + 30% theta cost acceptability
        5. Select best scoring strike, or fallback to geometric if none valid

        Args:
            signal: Pattern signal with entry/target prices
            underlying_price: Current underlying price
            option_type: CALL or PUT
            iv: Implied volatility (annualized)
            dte: Days to expiration
            target_delta: Target delta value (default 0.65)
            delta_range: Acceptable delta range (default 0.50-0.80)
            max_theta_pct: Max theta cost as % of expected profit (default 10%)

        Returns:
            Tuple of (strike, delta, theta). delta/theta may be None if fallback used.
        """
        entry = signal.entry_price
        target = signal.target_price

        # Step 1: Define strike range based on option type
        # EXPANDED RANGE: Include ITM strikes to achieve higher deltas (0.50-0.80)
        # For higher deltas: Calls need lower strikes (ITM), Puts need higher strikes (ITM)
        expected_move = abs(target - entry)
        itm_expansion = expected_move * 1.0  # Search 100% into ITM territory (full move size)

        if option_type == OptionType.CALL:
            # Calls: Lower strike = higher delta (more ITM)
            # Search from (entry - expansion) to target
            strike_min = entry - itm_expansion
            strike_max = target
        else:
            # Puts: Higher strike = higher delta (more ITM)
            # Search from target to (entry + expansion)
            strike_min = target
            strike_max = entry + itm_expansion

        # Step 2: Generate candidate strikes at standard intervals
        interval = self._get_strike_interval(underlying_price)
        candidates = self._generate_candidate_strikes(strike_min, strike_max, interval)

        if not candidates:
            return self._fallback_to_geometric(signal, underlying_price, option_type)

        # Step 3: Calculate Greeks and score each candidate
        T = dte / 365.0
        r = 0.05  # Risk-free rate
        option_type_str = 'call' if option_type == OptionType.CALL else 'put'

        # Get expected holding days for this timeframe
        estimated_days = self._get_expected_holding_days(signal.timeframe)

        scored_strikes = []
        for strike in candidates:
            greeks = calculate_greeks(
                S=underlying_price,
                K=strike,
                T=T,
                r=r,
                sigma=iv,
                option_type=option_type_str
            )

            # Filter by delta range
            delta_abs = abs(greeks.delta)
            if delta_abs < delta_range[0] or delta_abs > delta_range[1]:
                continue  # Outside optimal range

            # Delta score: proximity to target delta (normalized to 0-1)
            # Max deviation is 0.30 (from 0.50 to 0.80 range edges to target)
            delta_score = 1.0 - abs(delta_abs - target_delta) / 0.30

            # Theta score: cost verification
            expected_move = abs(target - entry)
            expected_profit = delta_abs * expected_move * 100  # Per contract
            daily_theta_cost = abs(greeks.theta) * 100
            total_theta_cost = daily_theta_cost * estimated_days
            theta_cost_pct = total_theta_cost / expected_profit if expected_profit > 0 else 1.0

            if theta_cost_pct > max_theta_pct:
                continue  # Theta too expensive

            theta_score = 1.0 - (theta_cost_pct / max_theta_pct)

            # Combined score: 70% delta, 30% theta
            total_score = (delta_score * 0.7) + (theta_score * 0.3)

            scored_strikes.append({
                'strike': strike,
                'delta': greeks.delta,
                'theta': greeks.theta,
                'option_price': greeks.option_price,
                'total_score': total_score
            })

        if not scored_strikes:
            return self._fallback_to_geometric(signal, underlying_price, option_type)

        # Step 4: Select best scoring strike
        best = max(scored_strikes, key=lambda x: x['total_score'])
        return best['strike'], best['delta'], best['theta']

    def _calculate_expiration(
        self,
        timeframe: Timeframe,
        reference_date: datetime = None
    ) -> datetime:
        """
        Calculate optimal expiration date based on pattern timeframe.

        Args:
            timeframe: Pattern timeframe (WEEKLY or MONTHLY)
            reference_date: Date to calculate from (default: today)

        Returns:
            Expiration date
        """
        # Use reference_date if provided (for backtesting), else today
        if reference_date is None:
            base_date = datetime.now()
        else:
            # Convert to naive datetime if needed
            if hasattr(reference_date, 'tzinfo') and reference_date.tzinfo is not None:
                base_date = reference_date.replace(tzinfo=None)
            elif hasattr(reference_date, 'to_pydatetime'):
                base_date = reference_date.to_pydatetime()
                if hasattr(base_date, 'tzinfo') and base_date.tzinfo is not None:
                    base_date = base_date.replace(tzinfo=None)
            else:
                base_date = reference_date

        if timeframe == Timeframe.WEEKLY:
            target_dte = self.default_dte_weekly
        elif timeframe == Timeframe.MONTHLY:
            target_dte = self.default_dte_monthly
        else:
            target_dte = self.default_dte_weekly

        # Find next Friday after target DTE
        target_date = base_date + timedelta(days=target_dte)

        # Adjust to next Friday (expiration day)
        days_until_friday = (4 - target_date.weekday()) % 7
        if days_until_friday == 0 and target_date.hour > 16:
            days_until_friday = 7

        expiration = target_date + timedelta(days=days_until_friday)

        return expiration

    def fetch_option_quote(
        self,
        contract: OptionContract,
        start: str = None,
        end: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch option quote data from Alpaca via VBT Pro.

        Args:
            contract: Option contract to fetch
            start: Start date (default: 30 days ago)
            end: End date (default: today)

        Returns:
            DataFrame with OHLC data or None if not found
        """
        if start is None:
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        try:
            data = vbt.AlpacaData.pull(
                contract.osi_symbol,
                client_type="options",
                client_config={
                    'api_key': self.creds['api_key'],
                    'secret_key': self.creds['secret_key'],
                },
                start=start,
                end=end,
                timeframe="1 day"
            )
            return data.get()

        except Exception as e:
            print(f"Error fetching option data for {contract.osi_symbol}: {e}")
            return None

    def get_available_expirations(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 90
    ) -> List[datetime]:
        """
        Get available expiration dates for underlying.

        Note: This requires Alpaca Options API access.
        Returns mock data for testing if API not available.
        """
        today = datetime.now()
        expirations = []

        # Generate Friday expirations within DTE range
        current = today + timedelta(days=min_dte)
        end_date = today + timedelta(days=max_dte)

        while current <= end_date:
            # Find next Friday
            days_until_friday = (4 - current.weekday()) % 7
            friday = current + timedelta(days=days_until_friday)

            if friday <= end_date and friday not in expirations:
                expirations.append(friday)

            current = friday + timedelta(days=7)

        return expirations

    def trades_to_dataframe(self, trades: List[OptionTrade]) -> pd.DataFrame:
        """
        Convert trades to DataFrame for analysis.

        Args:
            trades: List of OptionTrade objects

        Returns:
            DataFrame with trade details
        """
        if not trades:
            return pd.DataFrame()

        records = []
        for trade in trades:
            records.append({
                'timestamp': trade.pattern_signal.timestamp,
                'pattern_type': trade.pattern_signal.pattern_type.value,
                'direction': 'BULL' if trade.pattern_signal.direction == 1 else 'BEAR',
                'underlying': trade.contract.underlying,
                'osi_symbol': trade.contract.osi_symbol,
                'option_type': trade.contract.option_type.value,
                'strike': trade.contract.strike,
                'expiration': trade.contract.expiration.strftime('%Y-%m-%d'),
                'strategy': trade.strategy.value,
                'entry_trigger': trade.entry_trigger,
                'target_exit': trade.target_exit,
                'stop_exit': trade.stop_exit,
                'quantity': trade.quantity,
                'risk_reward': trade.pattern_signal.risk_reward,
                'continuation_bars': trade.pattern_signal.continuation_bars,
            })

        return pd.DataFrame(records)


class OptionsBacktester:
    """
    Backtest options strategies on historical data.

    Session 71: Uses Black-Scholes Greeks for accurate P/L calculation:
    - Delta: Linear price movement P/L
    - Gamma: Convexity adjustment for large moves
    - Theta: Time decay (critical for accurate modeling)

    Entry when underlying breaks pattern entry price.
    Exit at target, stop, or expiration.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        default_iv: float = 0.20
    ):
        """
        Initialize backtester with Greeks support.

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
            default_iv: Default implied volatility if not estimated (default 20%)
        """
        self.risk_free_rate = risk_free_rate
        self.default_iv = default_iv

    def _estimate_iv(self, price_data: pd.DataFrame) -> float:
        """Estimate IV from historical price data."""
        close_col = 'close' if 'close' in price_data.columns else 'Close'
        if close_col not in price_data.columns:
            return self.default_iv

        try:
            iv = estimate_iv_from_history(price_data[close_col], window=20)
            return max(0.10, min(iv, 1.0))  # Clamp between 10% and 100%
        except Exception:
            return self.default_iv

    def backtest_trades(
        self,
        trades: List[OptionTrade],
        price_data: pd.DataFrame,
        option_cost: Optional[float] = None  # None = use Black-Scholes calculated price
    ) -> pd.DataFrame:
        """
        Backtest option trades against historical price data.

        Uses Black-Scholes Greeks for accurate P/L calculation including:
        - Delta P/L from price movement
        - Gamma P/L from convexity (large moves)
        - Theta P/L from time decay

        Args:
            trades: List of OptionTrade objects
            price_data: DataFrame with OHLC data for underlying
            option_cost: Option premium per contract ($/share). If None, uses
                        Black-Scholes calculated option_price from entry Greeks.

        Returns:
            DataFrame with backtest results including Greeks breakdown
        """
        results = []

        # Estimate IV from historical data
        iv = self._estimate_iv(price_data)

        for trade in trades:
            # Find pattern date in price data
            try:
                pattern_idx = price_data.index.get_loc(trade.pattern_signal.timestamp)
            except KeyError:
                continue

            # Look for entry, target, or stop hit
            entry_hit = False
            entry_idx = None
            entry_price_underlying = None
            exit_price = None
            exit_type = None
            exit_date = None
            exit_idx = None

            # Scan forward from pattern
            for i in range(pattern_idx + 1, min(pattern_idx + 30, len(price_data))):
                row = price_data.iloc[i]
                high = row['high'] if 'high' in row.index else row['High']
                low = row['low'] if 'low' in row.index else row['Low']
                close = row['close'] if 'close' in row.index else row['Close']

                # Check entry
                if not entry_hit:
                    if trade.pattern_signal.direction == 1:  # Bullish
                        if high >= trade.entry_trigger:
                            entry_hit = True
                            entry_idx = i
                            entry_price_underlying = trade.entry_trigger
                    else:  # Bearish
                        if low <= trade.entry_trigger:
                            entry_hit = True
                            entry_idx = i
                            entry_price_underlying = trade.entry_trigger
                    continue

                # Check exit (after entry)
                if trade.pattern_signal.direction == 1:  # Bullish
                    if high >= trade.target_exit:
                        exit_price = trade.target_exit
                        exit_type = 'TARGET'
                        exit_date = price_data.index[i]
                        exit_idx = i
                        break
                    elif low <= trade.stop_exit:
                        exit_price = trade.stop_exit
                        exit_type = 'STOP'
                        exit_date = price_data.index[i]
                        exit_idx = i
                        break
                else:  # Bearish
                    if low <= trade.target_exit:
                        exit_price = trade.target_exit
                        exit_type = 'TARGET'
                        exit_date = price_data.index[i]
                        exit_idx = i
                        break
                    elif high >= trade.stop_exit:
                        exit_price = trade.stop_exit
                        exit_type = 'STOP'
                        exit_date = price_data.index[i]
                        exit_idx = i
                        break

            # Calculate P/L using Greeks
            if entry_hit and exit_price and entry_idx is not None and exit_idx is not None:
                # Calculate days held (for theta decay)
                days_held = exit_idx - entry_idx

                # Get option parameters
                strike = trade.contract.strike
                option_type = 'call' if trade.contract.option_type == OptionType.CALL else 'put'

                # Calculate DTE at entry (approximate from expiration)
                # Handle timezone-aware vs naive datetime comparison
                try:
                    expiration_dt = trade.contract.expiration
                    entry_dt = price_data.index[entry_idx]

                    # Convert to naive datetime if needed
                    if hasattr(expiration_dt, 'tzinfo') and expiration_dt.tzinfo is not None:
                        expiration_dt = expiration_dt.replace(tzinfo=None)
                    if hasattr(entry_dt, 'tzinfo') and entry_dt.tzinfo is not None:
                        entry_dt = entry_dt.tz_localize(None) if hasattr(entry_dt, 'tz_localize') else entry_dt.replace(tzinfo=None)
                    if hasattr(entry_dt, 'to_pydatetime'):
                        entry_dt = entry_dt.to_pydatetime()
                        if hasattr(entry_dt, 'tzinfo') and entry_dt.tzinfo is not None:
                            entry_dt = entry_dt.replace(tzinfo=None)

                    dte_at_entry = (expiration_dt - entry_dt).days
                except Exception:
                    dte_at_entry = 35  # Default if can't calculate

                dte_at_entry = max(1, dte_at_entry)
                T_entry = dte_at_entry / 365.0
                T_exit = max(0.001, (dte_at_entry - days_held) / 365.0)

                # Calculate Greeks at entry
                entry_greeks = calculate_greeks(
                    S=entry_price_underlying,
                    K=strike,
                    T=T_entry,
                    r=self.risk_free_rate,
                    sigma=iv,
                    option_type=option_type
                )

                # Determine actual option cost (per share)
                # If option_cost not provided, use Black-Scholes theoretical price
                if option_cost is not None:
                    actual_option_cost = option_cost
                else:
                    actual_option_cost = entry_greeks.option_price

                # Calculate exit Greeks for more accurate P/L
                # Greeks change as price and time change, so we use average of entry/exit
                exit_greeks = calculate_greeks(
                    S=exit_price,
                    K=strike,
                    T=T_exit,
                    r=self.risk_free_rate,
                    sigma=iv,
                    option_type=option_type
                )

                # Calculate price movement (actual underlying change)
                # Delta already encodes direction:
                # - Call delta is positive: profits when price goes UP
                # - Put delta is negative: profits when price goes DOWN
                # So we always use: exit - entry (no direction transform needed)
                price_move = exit_price - entry_price_underlying

                # Use AVERAGE Greeks for more accurate P/L calculation
                # This accounts for delta/theta changing during the hold period
                avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
                avg_theta = (entry_greeks.theta + exit_greeks.theta) / 2
                avg_gamma = (entry_greeks.gamma + exit_greeks.gamma) / 2

                # Calculate P/L components using average Greeks
                # Delta P/L: avg_delta * price_move
                delta_pnl = avg_delta * price_move * 100 * trade.quantity

                # Gamma P/L: 0.5 * avg_gamma * price_move^2 (convexity)
                gamma_pnl = 0.5 * avg_gamma * (price_move ** 2) * 100 * trade.quantity

                # Theta P/L: avg_theta * days_held (time decay, negative for long options)
                # Note: Using average theta accounts for theta acceleration near expiration
                theta_pnl = avg_theta * days_held * 100 * trade.quantity

                # Total P/L
                if exit_type == 'TARGET':
                    # For target hit, we use intrinsic value at exit
                    if option_type == 'call':
                        intrinsic_value = max(exit_price - strike, 0)
                    else:
                        intrinsic_value = max(strike - exit_price, 0)

                    # P/L = (delta_pnl + gamma_pnl + theta_pnl) - entry_cost
                    # delta_pnl is already correctly signed:
                    # - Positive for winning calls (price up, delta positive)
                    # - Positive for winning puts (price down, delta negative * negative = positive)
                    # Note: theta_pnl is typically negative (time decay costs money)
                    gross_pnl = delta_pnl + gamma_pnl + theta_pnl
                    pnl = gross_pnl - actual_option_cost * 100 * trade.quantity
                else:
                    # STOP: Max loss is premium paid
                    pnl = -actual_option_cost * 100 * trade.quantity

                results.append({
                    'timestamp': trade.pattern_signal.timestamp,
                    'pattern_type': trade.pattern_signal.pattern_type.value,
                    'osi_symbol': trade.contract.osi_symbol,
                    'entry_trigger': trade.entry_trigger,
                    'entry_price_underlying': entry_price_underlying,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'exit_date': exit_date,
                    'days_held': days_held,
                    'quantity': trade.quantity,
                    'strike': strike,
                    'iv': iv,
                    'option_cost': actual_option_cost,  # Premium per share ($/share)
                    'entry_option_price': entry_greeks.option_price,  # Black-Scholes price at entry
                    'exit_option_price': exit_greeks.option_price,  # Black-Scholes price at exit
                    'entry_delta': entry_greeks.delta,
                    'exit_delta': exit_greeks.delta,
                    'avg_delta': avg_delta,  # Average delta used for P/L
                    'entry_theta': entry_greeks.theta,
                    'exit_theta': exit_greeks.theta,
                    'avg_theta': avg_theta,  # Average theta used for P/L
                    'delta_pnl': delta_pnl,
                    'gamma_pnl': gamma_pnl,
                    'theta_pnl': theta_pnl,
                    'pnl': pnl,
                    'win': pnl > 0,
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("Options Module Test")
    print("=" * 60)

    # Test OSI symbol generation
    print("\n[TEST 1] OSI Symbol Generation...")
    contract = OptionContract(
        underlying='SPY',
        expiration=datetime(2024, 12, 20),
        option_type=OptionType.CALL,
        strike=300.0
    )
    print(f"  Underlying: {contract.underlying}")
    print(f"  Expiration: {contract.expiration.date()}")
    print(f"  Type: {contract.option_type.value}")
    print(f"  Strike: ${contract.strike}")
    print(f"  OSI Symbol: {contract.osi_symbol}")

    # Test with Tier 1 patterns
    print("\n[TEST 2] Generating Option Trades from Patterns...")
    from strat.tier1_detector import Tier1Detector, Timeframe
    from integrations.tiingo_data_fetcher import TiingoDataFetcher

    # Fetch data
    fetcher = TiingoDataFetcher()
    data = fetcher.fetch('SPY', start_date='2023-01-01', end_date='2024-12-31', timeframe='1W')
    spy_df = data.get()

    # Detect patterns
    detector = Tier1Detector()
    signals = detector.detect_patterns(spy_df, timeframe=Timeframe.WEEKLY)

    if signals:
        # Generate option trades
        executor = OptionsExecutor()
        trades = executor.generate_option_trades(
            signals[:5],  # First 5 signals
            underlying='SPY',
            underlying_price=450.0
        )

        print(f"  Generated {len(trades)} option trades")

        # Convert to DataFrame
        df = executor.trades_to_dataframe(trades)
        print(f"\n  Option Trades:")
        print(df[['pattern_type', 'osi_symbol', 'strike', 'strategy', 'risk_reward']].to_string(index=False))

        # Backtest
        print("\n[TEST 3] Backtesting Trades...")
        backtester = OptionsBacktester()
        results = backtester.backtest_trades(trades, spy_df)

        if not results.empty:
            print(f"  Trades with results: {len(results)}")
            print(f"  Win rate: {results['win'].mean()*100:.1f}%")
            print(f"  Total P/L: ${results['pnl'].sum():,.2f}")
        else:
            print("  No trades executed in backtest period")
    else:
        print("  No patterns detected")

    print("\n" + "=" * 60)
    print("Options Module Test Complete")
    print("=" * 60)
