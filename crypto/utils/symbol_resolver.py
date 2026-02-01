"""
Symbol resolution utilities for spot/derivative mapping.

Session EQUITY-99: Implements two-layer data architecture where:
- Signal Detection: Use SPOT data (BTC-USD, ETH-USD) for cleaner price action
- Execution: Use DERIVATIVE data (BIP-20DEC30-CDE) for actual trading

This addresses the problem of artificial long wicks in CFM derivatives
during low liquidity periods that create false STRAT signals.
"""

from typing import Optional

from crypto import config


class SymbolResolver:
    """
    Resolves between spot and derivative symbols.

    Provides utilities to:
    - Map derivatives to their spot counterparts for data fetching
    - Map spot symbols back to derivatives for execution
    - Check if spot data is available for a given symbol
    - Extract base asset (BTC, ETH) from any symbol format

    Usage:
        >>> SymbolResolver.get_spot_symbol("BIP-20DEC30-CDE")
        'BTC-USD'
        >>> SymbolResolver.has_spot_data("BIP-20DEC30-CDE")
        True
        >>> SymbolResolver.get_base_asset("BIP-20DEC30-CDE")
        'BTC'
    """

    @staticmethod
    def get_spot_symbol(derivative_symbol: str) -> Optional[str]:
        """
        Get spot symbol for a derivative.

        Args:
            derivative_symbol: Derivative symbol (e.g., "BIP-20DEC30-CDE")

        Returns:
            Spot symbol (e.g., "BTC-USD") or None if no mapping exists
        """
        return config.DERIVATIVE_TO_SPOT.get(derivative_symbol)

    @staticmethod
    def get_derivative_symbol(spot_symbol: str) -> Optional[str]:
        """
        Get derivative symbol for a spot symbol.

        Args:
            spot_symbol: Spot symbol (e.g., "BTC-USD")

        Returns:
            Derivative symbol (e.g., "BIP-20DEC30-CDE") or None if no mapping exists
        """
        return config.SPOT_TO_DERIVATIVE.get(spot_symbol)

    @staticmethod
    def has_spot_data(symbol: str) -> bool:
        """
        Check if spot data is available for this symbol.

        Works with both derivative symbols (BIP-20DEC30-CDE) and
        spot symbols (BTC-USD) by extracting the base asset.

        Args:
            symbol: Any trading symbol

        Returns:
            True if spot data is available for the underlying asset
        """
        base = SymbolResolver.get_base_asset(symbol)
        return base in config.SPOT_DATA_AVAILABLE

    @staticmethod
    def get_base_asset(symbol: str) -> str:
        """
        Extract base asset from any symbol format.

        Handles:
        - CFM derivatives: BIP-20DEC30-CDE -> BTC
        - INTX perpetuals: BTC-PERP-INTX -> BTC
        - Spot symbols: BTC-USD -> BTC
        - StatArb symbols: ADA-USD -> ADA

        Args:
            symbol: Trading symbol in any format

        Returns:
            Base asset code (BTC, ETH, SOL, ADA, XRP, etc.)
        """
        # Extract prefix before first dash
        prefix = symbol.split("-")[0]

        # Map CFM product codes to base assets
        # Uses SYMBOL_TO_BASE_ASSET from config for CFM mapping
        return config.SYMBOL_TO_BASE_ASSET.get(prefix, prefix)

    @staticmethod
    def resolve_data_symbol(
        trading_symbol: str,
        use_spot: Optional[bool] = None,
    ) -> str:
        """
        Resolve which symbol to use for data fetching.

        High-level convenience method that combines spot availability check
        and symbol resolution in one call.

        Args:
            trading_symbol: The symbol to trade (derivative)
            use_spot: Override for USE_SPOT_FOR_SIGNALS config.
                     None uses config default.

        Returns:
            Symbol to use for data fetching (spot if available, else original)
        """
        if use_spot is None:
            use_spot = config.USE_SPOT_FOR_SIGNALS

        if not use_spot:
            return trading_symbol

        if not SymbolResolver.has_spot_data(trading_symbol):
            return trading_symbol

        spot = SymbolResolver.get_spot_symbol(trading_symbol)
        return spot if spot else trading_symbol

    @staticmethod
    def resolve_price_symbol(
        trading_symbol: str,
        use_spot: Optional[bool] = None,
    ) -> str:
        """
        Resolve which symbol to use for price fetching (trigger detection).

        Similar to resolve_data_symbol but uses USE_SPOT_FOR_TRIGGERS config.

        Args:
            trading_symbol: The symbol to trade (derivative)
            use_spot: Override for USE_SPOT_FOR_TRIGGERS config.
                     None uses config default.

        Returns:
            Symbol to use for price fetching (spot if available, else original)
        """
        if use_spot is None:
            use_spot = config.USE_SPOT_FOR_TRIGGERS

        if not use_spot:
            return trading_symbol

        if not SymbolResolver.has_spot_data(trading_symbol):
            return trading_symbol

        spot = SymbolResolver.get_spot_symbol(trading_symbol)
        return spot if spot else trading_symbol
