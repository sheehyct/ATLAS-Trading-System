"""
Black-Scholes Options Price Provider

Offline fallback for options pricing when ThetaData is unavailable.
Uses strat/greeks.py calculate_greeks() for theoretical pricing.

Estimates implied volatility from historical underlying data when
no market IV is available.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np

from strat.backtesting.data_providers.base import OptionsQuoteResult

logger = logging.getLogger(__name__)


class BlackScholesProvider:
    """
    Options pricing via Black-Scholes model.

    Uses strat/greeks.py for theoretical option prices with
    historical volatility as a proxy for implied volatility.

    For backtesting without a ThetaData terminal. Provides
    approximate prices that won't match market quotes exactly
    but preserve relative P&L behavior.

    Usage:
        provider = BlackScholesProvider()
        quote = provider.get_quote('SPY', '2024-03-15', 450.0, 'C',
                                    as_of=datetime(2024, 2, 1))
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        default_iv: float = 0.20,
        bid_ask_spread_pct: float = 0.05,
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default: 5%)
            default_iv: Default implied volatility if can't estimate (default: 20%)
            bid_ask_spread_pct: Simulated bid-ask spread as % of mid (default: 5%)
        """
        self._risk_free_rate = risk_free_rate
        self._default_iv = default_iv
        self._spread_pct = bid_ask_spread_pct
        self._iv_cache: dict = {}  # (symbol, date_str) -> estimated IV

    def set_historical_data(self, symbol: str, close_prices) -> None:
        """
        Pre-load historical close prices for IV estimation.

        Args:
            symbol: Underlying symbol
            close_prices: pandas Series or array of close prices
        """
        try:
            import pandas as pd
            if isinstance(close_prices, pd.Series):
                prices = close_prices.values
            else:
                prices = np.array(close_prices)

            if len(prices) < 20:
                return

            # Calculate rolling 20-day historical volatility
            returns = np.diff(np.log(prices))
            # Use 20-day rolling windows
            for i in range(20, len(returns)):
                window = returns[i-20:i]
                hv = np.std(window) * np.sqrt(252)
                if isinstance(close_prices, pd.Series):
                    date_str = str(close_prices.index[i+1].date())
                else:
                    date_str = str(i)
                self._iv_cache[(symbol, date_str)] = hv

        except Exception as e:
            logger.debug("Failed to estimate IV for %s: %s", symbol, e)

    def get_quote(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        option_type: str,
        as_of: datetime,
    ) -> Optional[OptionsQuoteResult]:
        """
        Calculate theoretical option price using Black-Scholes.

        Uses historical volatility or default IV.
        Simulates bid/ask spread around the theoretical price.
        """
        try:
            from strat.greeks import calculate_greeks

            # Calculate time to expiration
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            dte = (exp_date - as_of).days
            if dte <= 0:
                return None  # Expired
            T = dte / 365.0

            # Get IV estimate
            date_str = as_of.strftime('%Y-%m-%d')
            sigma = self._iv_cache.get((symbol, date_str), self._default_iv)

            # We need the underlying price - approximate from strike for now
            # In practice the engine should provide this, but for standalone use:
            # Assume ATM (S ≈ K) if no better info
            S = strike  # Will be overridden by caller if underlying price available

            bs_type = 'call' if option_type.upper() in ('C', 'CALL') else 'put'
            greeks = calculate_greeks(S, strike, T, self._risk_free_rate, sigma, bs_type)

            mid = greeks.option_price
            if mid <= 0:
                mid = 0.01  # Minimum price

            # Simulate bid/ask spread
            half_spread = mid * self._spread_pct / 2
            bid = max(0.01, mid - half_spread)
            ask = mid + half_spread

            return OptionsQuoteResult(
                symbol=symbol,
                strike=strike,
                expiration=expiration,
                option_type=option_type.upper() if len(option_type) == 1 else option_type[0].upper(),
                bid=bid,
                ask=ask,
                mid=mid,
                iv=sigma,
                delta=greeks.delta,
                gamma=greeks.gamma,
                theta=greeks.theta,
                vega=greeks.vega,
                underlying_price=S,
                timestamp=as_of,
                data_source='blackscholes',
            )

        except ImportError:
            logger.error("strat.greeks module not available")
            return None
        except Exception as e:
            logger.debug("B-S pricing failed: %s", e)
            return None

    def get_quote_with_underlying(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        option_type: str,
        as_of: datetime,
        underlying_price: float,
    ) -> Optional[OptionsQuoteResult]:
        """
        Calculate option price with known underlying price.

        More accurate than get_quote() since we don't have to guess
        the underlying price.
        """
        try:
            from strat.greeks import calculate_greeks

            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            dte = (exp_date - as_of).days
            if dte <= 0:
                return None
            T = dte / 365.0

            date_str = as_of.strftime('%Y-%m-%d')
            sigma = self._iv_cache.get((symbol, date_str), self._default_iv)

            bs_type = 'call' if option_type.upper() in ('C', 'CALL') else 'put'
            greeks = calculate_greeks(
                underlying_price, strike, T,
                self._risk_free_rate, sigma, bs_type,
            )

            mid = max(0.01, greeks.option_price)
            half_spread = mid * self._spread_pct / 2
            bid = max(0.01, mid - half_spread)
            ask = mid + half_spread

            return OptionsQuoteResult(
                symbol=symbol,
                strike=strike,
                expiration=expiration,
                option_type=option_type.upper() if len(option_type) == 1 else option_type[0].upper(),
                bid=bid,
                ask=ask,
                mid=mid,
                iv=sigma,
                delta=greeks.delta,
                gamma=greeks.gamma,
                theta=greeks.theta,
                vega=greeks.vega,
                underlying_price=underlying_price,
                timestamp=as_of,
                data_source='blackscholes',
            )
        except Exception as e:
            logger.debug("B-S pricing with underlying failed: %s", e)
            return None

    def find_expiration(
        self,
        symbol: str,
        target_dte: int,
        as_of: datetime,
        min_dte: int = 7,
        max_dte: int = 21,
    ) -> Optional[str]:
        """
        Find third-Friday expiration closest to target DTE.

        Pure calculation - no API call needed.
        """
        target_date = as_of + timedelta(days=target_dte)
        year = target_date.year
        month = target_date.month

        # Third Friday of target month
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)

        dte = (third_friday - as_of).days
        if dte < min_dte:
            # Try next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            third_friday = first_friday + timedelta(weeks=2)

        return third_friday.strftime('%Y-%m-%d')

    def get_strikes(
        self,
        symbol: str,
        expiration: str,
    ) -> List[float]:
        """
        Generate synthetic strike range.

        Since B-S doesn't have a real strike chain, generate strikes
        at $1 intervals around the assumed underlying price.
        Returns a reasonable range for backtesting.
        """
        # Without knowing the underlying price, generate a generic range
        # The engine will filter to relevant strikes
        # Default range: $100-$800 at $5 intervals (covers most liquid equities)
        return [float(s) for s in range(100, 805, 5)]
