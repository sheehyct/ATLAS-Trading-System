"""
Unit tests for SymbolResolver.

Session EQUITY-99: Tests for spot/derivative symbol mapping utilities.
"""

import pytest
from unittest.mock import patch

from crypto.utils.symbol_resolver import SymbolResolver
from crypto import config


class TestSymbolResolverGetSpotSymbol:
    """Tests for SymbolResolver.get_spot_symbol()."""

    def test_btc_derivative_returns_spot(self):
        """BIP-20DEC30-CDE should map to BTC-USD."""
        result = SymbolResolver.get_spot_symbol("BIP-20DEC30-CDE")
        assert result == "BTC-USD"

    def test_eth_derivative_returns_spot(self):
        """ETP-20DEC30-CDE should map to ETH-USD."""
        result = SymbolResolver.get_spot_symbol("ETP-20DEC30-CDE")
        assert result == "ETH-USD"

    def test_unknown_derivative_returns_none(self):
        """Unknown derivatives should return None."""
        result = SymbolResolver.get_spot_symbol("UNKNOWN-SYMBOL")
        assert result is None

    def test_spot_symbol_returns_none(self):
        """Spot symbols should return None (no mapping needed)."""
        result = SymbolResolver.get_spot_symbol("BTC-USD")
        assert result is None


class TestSymbolResolverGetDerivativeSymbol:
    """Tests for SymbolResolver.get_derivative_symbol()."""

    def test_btc_spot_returns_derivative(self):
        """BTC-USD should map to BIP-20DEC30-CDE."""
        result = SymbolResolver.get_derivative_symbol("BTC-USD")
        assert result == "BIP-20DEC30-CDE"

    def test_eth_spot_returns_derivative(self):
        """ETH-USD should map to ETP-20DEC30-CDE."""
        result = SymbolResolver.get_derivative_symbol("ETH-USD")
        assert result == "ETP-20DEC30-CDE"

    def test_unknown_spot_returns_none(self):
        """Unknown spot symbols should return None."""
        result = SymbolResolver.get_derivative_symbol("SOL-USD")
        assert result is None

    def test_derivative_symbol_returns_none(self):
        """Derivative symbols should return None (no mapping needed)."""
        result = SymbolResolver.get_derivative_symbol("BIP-20DEC30-CDE")
        assert result is None


class TestSymbolResolverHasSpotData:
    """Tests for SymbolResolver.has_spot_data()."""

    def test_btc_derivative_has_spot_data(self):
        """BTC derivative should have spot data available."""
        assert SymbolResolver.has_spot_data("BIP-20DEC30-CDE") is True

    def test_eth_derivative_has_spot_data(self):
        """ETH derivative should have spot data available."""
        assert SymbolResolver.has_spot_data("ETP-20DEC30-CDE") is True

    def test_btc_spot_has_spot_data(self):
        """BTC spot symbol should have spot data available."""
        assert SymbolResolver.has_spot_data("BTC-USD") is True

    def test_eth_spot_has_spot_data(self):
        """ETH spot symbol should have spot data available."""
        assert SymbolResolver.has_spot_data("ETH-USD") is True

    def test_sol_has_spot_data(self):
        """SOL should have spot data available (in SPOT_DATA_AVAILABLE)."""
        assert SymbolResolver.has_spot_data("SOL-USD") is True

    def test_ada_no_spot_data(self):
        """ADA should not have spot data available (StatArb only)."""
        assert SymbolResolver.has_spot_data("ADA-USD") is False

    def test_xrp_no_spot_data(self):
        """XRP should not have spot data available (StatArb only)."""
        assert SymbolResolver.has_spot_data("XRP-USD") is False


class TestSymbolResolverGetBaseAsset:
    """Tests for SymbolResolver.get_base_asset()."""

    def test_btc_derivative_base_asset(self):
        """BIP-20DEC30-CDE should return BTC."""
        result = SymbolResolver.get_base_asset("BIP-20DEC30-CDE")
        assert result == "BTC"

    def test_eth_derivative_base_asset(self):
        """ETP-20DEC30-CDE should return ETH."""
        result = SymbolResolver.get_base_asset("ETP-20DEC30-CDE")
        assert result == "ETH"

    def test_btc_spot_base_asset(self):
        """BTC-USD should return BTC."""
        result = SymbolResolver.get_base_asset("BTC-USD")
        assert result == "BTC"

    def test_eth_spot_base_asset(self):
        """ETH-USD should return ETH."""
        result = SymbolResolver.get_base_asset("ETH-USD")
        assert result == "ETH"

    def test_sol_spot_base_asset(self):
        """SOL-USD should return SOL."""
        result = SymbolResolver.get_base_asset("SOL-USD")
        assert result == "SOL"

    def test_ada_spot_base_asset(self):
        """ADA-USD should return ADA."""
        result = SymbolResolver.get_base_asset("ADA-USD")
        assert result == "ADA"

    def test_intx_perpetual_base_asset(self):
        """BTC-PERP-INTX should return BTC."""
        result = SymbolResolver.get_base_asset("BTC-PERP-INTX")
        assert result == "BTC"


class TestSymbolResolverResolveDataSymbol:
    """Tests for SymbolResolver.resolve_data_symbol()."""

    def test_btc_derivative_resolves_to_spot_when_enabled(self):
        """BTC derivative should resolve to spot when USE_SPOT_FOR_SIGNALS=True."""
        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            result = SymbolResolver.resolve_data_symbol("BIP-20DEC30-CDE")
            assert result == "BTC-USD"

    def test_btc_derivative_stays_derivative_when_disabled(self):
        """BTC derivative should stay derivative when USE_SPOT_FOR_SIGNALS=False."""
        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', False):
            result = SymbolResolver.resolve_data_symbol("BIP-20DEC30-CDE")
            assert result == "BIP-20DEC30-CDE"

    def test_explicit_override_true(self):
        """Explicit use_spot=True should override config."""
        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', False):
            result = SymbolResolver.resolve_data_symbol("BIP-20DEC30-CDE", use_spot=True)
            assert result == "BTC-USD"

    def test_explicit_override_false(self):
        """Explicit use_spot=False should override config."""
        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            result = SymbolResolver.resolve_data_symbol("BIP-20DEC30-CDE", use_spot=False)
            assert result == "BIP-20DEC30-CDE"

    def test_ada_stays_ada_no_spot_data(self):
        """ADA should stay ADA even when spot enabled (no spot data)."""
        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            result = SymbolResolver.resolve_data_symbol("ADA-USD")
            assert result == "ADA-USD"


class TestSymbolResolverResolvePriceSymbol:
    """Tests for SymbolResolver.resolve_price_symbol()."""

    def test_btc_derivative_resolves_to_spot_when_enabled(self):
        """BTC derivative should resolve to spot when USE_SPOT_FOR_TRIGGERS=True."""
        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', True):
            result = SymbolResolver.resolve_price_symbol("BIP-20DEC30-CDE")
            assert result == "BTC-USD"

    def test_btc_derivative_stays_derivative_when_disabled(self):
        """BTC derivative should stay derivative when USE_SPOT_FOR_TRIGGERS=False."""
        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', False):
            result = SymbolResolver.resolve_price_symbol("BIP-20DEC30-CDE")
            assert result == "BIP-20DEC30-CDE"

    def test_explicit_override_true(self):
        """Explicit use_spot=True should override config."""
        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', False):
            result = SymbolResolver.resolve_price_symbol("BIP-20DEC30-CDE", use_spot=True)
            assert result == "BTC-USD"

    def test_explicit_override_false(self):
        """Explicit use_spot=False should override config."""
        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', True):
            result = SymbolResolver.resolve_price_symbol("BIP-20DEC30-CDE", use_spot=False)
            assert result == "BIP-20DEC30-CDE"


class TestSymbolResolverStatArbCompatibility:
    """Tests to ensure StatArb symbols (ADA, XRP) work correctly."""

    def test_ada_has_no_spot_mapping(self):
        """ADA-USD should have no spot mapping (StatArb uses derivative data)."""
        assert SymbolResolver.get_spot_symbol("ADA-USD") is None

    def test_xrp_has_no_spot_mapping(self):
        """XRP-USD should have no spot mapping (StatArb uses derivative data)."""
        assert SymbolResolver.get_spot_symbol("XRP-USD") is None

    def test_ada_resolve_data_stays_ada(self):
        """ADA-USD should not change in resolve_data_symbol."""
        result = SymbolResolver.resolve_data_symbol("ADA-USD", use_spot=True)
        assert result == "ADA-USD"

    def test_xrp_resolve_price_stays_xrp(self):
        """XRP-USD should not change in resolve_price_symbol."""
        result = SymbolResolver.resolve_price_symbol("XRP-USD", use_spot=True)
        assert result == "XRP-USD"


class TestSymbolResolverConfigMapping:
    """Tests to verify config mappings are correct."""

    def test_derivative_to_spot_mapping_exists(self):
        """DERIVATIVE_TO_SPOT config should have BTC and ETH mappings."""
        assert "BIP-20DEC30-CDE" in config.DERIVATIVE_TO_SPOT
        assert "ETP-20DEC30-CDE" in config.DERIVATIVE_TO_SPOT

    def test_spot_to_derivative_mapping_exists(self):
        """SPOT_TO_DERIVATIVE config should have BTC and ETH mappings."""
        assert "BTC-USD" in config.SPOT_TO_DERIVATIVE
        assert "ETH-USD" in config.SPOT_TO_DERIVATIVE

    def test_spot_data_available_contains_expected(self):
        """SPOT_DATA_AVAILABLE should contain BTC, ETH, SOL."""
        assert "BTC" in config.SPOT_DATA_AVAILABLE
        assert "ETH" in config.SPOT_DATA_AVAILABLE
        assert "SOL" in config.SPOT_DATA_AVAILABLE

    def test_spot_data_available_excludes_altcoins(self):
        """SPOT_DATA_AVAILABLE should NOT contain ADA, XRP."""
        assert "ADA" not in config.SPOT_DATA_AVAILABLE
        assert "XRP" not in config.SPOT_DATA_AVAILABLE

    def test_bidirectional_mapping_consistency(self):
        """DERIVATIVE_TO_SPOT and SPOT_TO_DERIVATIVE should be inverses."""
        for deriv, spot in config.DERIVATIVE_TO_SPOT.items():
            assert config.SPOT_TO_DERIVATIVE.get(spot) == deriv
