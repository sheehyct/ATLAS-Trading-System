"""
Exchange integration module for crypto derivatives.

Provides Coinbase Advanced Trade API client with:
- Historical OHLCV data fetching
- Order execution (market, limit, stop)
- Position management
- Simulation mode for paper trading
"""

from crypto.exchange.coinbase_client import CoinbaseClient

__all__ = ["CoinbaseClient"]
