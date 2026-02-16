"""
ThetaData Options Price Provider

Wraps the existing ThetaDataRESTClient to implement the
OptionsPriceProvider protocol for backtesting.

Provides real historical NBBO quotes with bid/ask spread
for realistic fill modeling.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

from strat.backtesting.data_providers.base import OptionsQuoteResult

logger = logging.getLogger(__name__)


class ThetaDataProvider:
    """
    Options pricing via ThetaData REST API.

    Wraps ThetaDataRESTClient for historical NBBO quotes.
    Entry fills at ask, exit fills at bid for conservative modeling.

    Usage:
        provider = ThetaDataProvider()
        if provider.connect():
            quote = provider.get_quote('SPY', '2024-03-15', 450.0, 'C',
                                       as_of=datetime(2024, 2, 1))
    """

    def __init__(self):
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        """Initialize and connect to ThetaData terminal."""
        try:
            from integrations.thetadata_client import ThetaDataRESTClient
            self._client = ThetaDataRESTClient()
            self._connected = self._client.connect()
            if self._connected:
                logger.info("ThetaData provider connected")
            else:
                logger.warning("ThetaData provider failed to connect")
            return self._connected
        except ImportError:
            logger.error("ThetaData client not available")
            return False
        except Exception as e:
            logger.error("ThetaData connection error: %s", e)
            return False

    def get_quote(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        option_type: str,
        as_of: datetime,
    ) -> Optional[OptionsQuoteResult]:
        """
        Get historical options quote from ThetaData.

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            expiration: Expiration date YYYY-MM-DD
            strike: Strike price in dollars
            option_type: 'C' or 'P'
            as_of: Historical date to price the option

        Returns:
            OptionsQuoteResult with bid/ask/mid, or None
        """
        if not self._connected or not self._client:
            return None

        try:
            exp_dt = datetime.strptime(expiration, '%Y-%m-%d')
            td_type = option_type.upper()
            if td_type not in ('C', 'P'):
                td_type = 'C' if option_type.upper() in ('CALL', 'C') else 'P'

            quote = self._client.get_quote(
                underlying=symbol,
                expiration=exp_dt,
                strike=strike,
                option_type=td_type,
                as_of=as_of,
            )

            if quote is None:
                return None

            return OptionsQuoteResult(
                symbol=symbol,
                strike=strike,
                expiration=expiration,
                option_type=td_type,
                bid=quote.bid,
                ask=quote.ask,
                mid=quote.mid,
                iv=quote.iv,
                delta=quote.delta,
                gamma=quote.gamma,
                theta=quote.theta,
                vega=quote.vega,
                underlying_price=quote.underlying_price,
                timestamp=quote.timestamp,
                data_source='thetadata',
            )
        except Exception as e:
            logger.debug("ThetaData quote failed for %s %s %s %s @ %s: %s",
                         symbol, expiration, strike, option_type, as_of, e)
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
        Find expiration date closest to target DTE.

        Uses ThetaDataRESTClient.find_valid_expiration() which
        calculates third-Friday expirations for historical dates.
        """
        if not self._connected or not self._client:
            return None

        try:
            exp_dt = self._client.find_valid_expiration(
                underlying=symbol,
                target_date=as_of,
                target_dte=target_dte,
                min_dte=min_dte,
                max_dte=max_dte,
            )
            if exp_dt:
                return exp_dt.strftime('%Y-%m-%d')
            return None
        except Exception as e:
            logger.debug("ThetaData expiration search failed: %s", e)
            return None

    def get_strikes(
        self,
        symbol: str,
        expiration: str,
    ) -> List[float]:
        """Get available strikes for a symbol/expiration."""
        if not self._connected or not self._client:
            return []

        try:
            exp_dt = datetime.strptime(expiration, '%Y-%m-%d')
            return self._client.get_strikes_cached(symbol, exp_dt)
        except Exception as e:
            logger.debug("ThetaData strikes fetch failed: %s", e)
            return []
