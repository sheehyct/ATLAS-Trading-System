"""
ATLAS System Integrations

External integrations for the ATLAS trading system:
- Alpaca Trading Client: Order execution (no VBT dependency)
- Stock Scanner Bridge: Connect momentum scanner to VectorBT backtesting (requires VBT)
- ThetaData: Historical options data for backtesting (Session 79)

Note: VBT-dependent modules are NOT auto-imported to allow dashboard deployment
without VectorBT Pro license. Import them explicitly when needed:
    from integrations.stock_scanner_bridge import MomentumPortfolioBacktest
"""

# Only export names - don't auto-import VBT-dependent modules
# This allows dashboard to import alpaca_trading_client without triggering VBT import
__all__ = [
    # Alpaca (no VBT dependency - safe to import)
    'AlpacaTradingClient',
    # VBT-dependent (import explicitly when needed)
    'MomentumPortfolioBacktest',
    'test_scanner_integration',
    # ThetaData (import explicitly when needed)
    'ThetaDataRESTClient',
    'ThetaDataProviderBase',
    'OptionsQuote',
    'create_thetadata_client',
    'ThetaDataOptionsFetcher',
]


def __getattr__(name):
    """Lazy import for VBT-dependent modules."""
    if name == 'AlpacaTradingClient':
        from .alpaca_trading_client import AlpacaTradingClient
        return AlpacaTradingClient
    elif name in ('MomentumPortfolioBacktest', 'test_scanner_integration'):
        from .stock_scanner_bridge import MomentumPortfolioBacktest, test_scanner_integration
        if name == 'MomentumPortfolioBacktest':
            return MomentumPortfolioBacktest
        return test_scanner_integration
    elif name in ('ThetaDataRESTClient', 'ThetaDataProviderBase', 'OptionsQuote', 'create_thetadata_client'):
        from .thetadata_client import (
            ThetaDataRESTClient, ThetaDataProviderBase, OptionsQuote, create_thetadata_client
        )
        return locals()[name]
    elif name == 'ThetaDataOptionsFetcher':
        from .thetadata_options_fetcher import ThetaDataOptionsFetcher
        return ThetaDataOptionsFetcher
    raise AttributeError(f"module 'integrations' has no attribute '{name}'")
