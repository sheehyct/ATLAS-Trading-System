"""
ATLAS System Integrations

External integrations for the ATLAS trading system:
- Stock Scanner Bridge: Connect momentum scanner to VectorBT backtesting
"""

from .stock_scanner_bridge import MomentumPortfolioBacktest, test_scanner_integration

__all__ = ['MomentumPortfolioBacktest', 'test_scanner_integration']
