"""
Tests for crypto/trading/beta.py - Beta and capital efficiency calculations.

Tests cover:
- CRYPTO_BETA_TO_BTC and leverage tier constants
- calculate_effective_multiplier() - leverage x beta
- get_effective_multipliers() - all symbols
- rank_by_capital_efficiency() - sorted ranking
- project_pnl_on_btc_move() - P&L projection
- compare_instruments_on_btc_move() - DataFrame comparison
- calculate_rolling_beta() - pandas rolling calculation
- calculate_beta_from_ranges() - Day Up/Down method
- update_beta_from_current_levels() - batch update
- calculate_beta_adjusted_size() - normalized sizing
- select_best_instrument() - signal selection

Session: EQUITY-82
"""

import pytest
import pandas as pd
import numpy as np

from crypto.trading.beta import (
    CRYPTO_BETA_TO_BTC,
    INTRADAY_LEVERAGE,
    SWING_LEVERAGE,
    calculate_effective_multiplier,
    get_effective_multipliers,
    rank_by_capital_efficiency,
    project_pnl_on_btc_move,
    compare_instruments_on_btc_move,
    calculate_rolling_beta,
    calculate_beta_from_ranges,
    update_beta_from_current_levels,
    calculate_beta_adjusted_size,
    select_best_instrument,
)


# =============================================================================
# CONSTANT VALIDATION TESTS
# =============================================================================


class TestCryptoBetaConstants:
    """Tests for beta constant definitions."""

    def test_btc_beta_is_one(self):
        """BTC beta to itself should be 1.0."""
        assert CRYPTO_BETA_TO_BTC["BTC"] == 1.0

    def test_all_betas_positive(self):
        """All beta values should be positive."""
        for symbol, beta in CRYPTO_BETA_TO_BTC.items():
            assert beta > 0, f"{symbol} has non-positive beta: {beta}"

    def test_expected_symbols_present(self):
        """Expected symbols should be in beta dictionary."""
        expected = {"BTC", "ETH", "SOL", "XRP", "ADA"}
        assert expected == set(CRYPTO_BETA_TO_BTC.keys())

    def test_eth_beta_greater_than_one(self):
        """ETH typically has beta > 1 relative to BTC."""
        assert CRYPTO_BETA_TO_BTC["ETH"] > 1.0

    def test_ada_has_highest_beta(self):
        """ADA has highest beta in the set."""
        ada_beta = CRYPTO_BETA_TO_BTC["ADA"]
        for symbol, beta in CRYPTO_BETA_TO_BTC.items():
            if symbol != "ADA":
                assert ada_beta >= beta, f"ADA beta {ada_beta} not >= {symbol} beta {beta}"


class TestLeverageTierConstants:
    """Tests for leverage tier definitions."""

    def test_intraday_btc_eth_have_10x(self):
        """BTC and ETH should have 10x intraday leverage."""
        assert INTRADAY_LEVERAGE["BTC"] == 10.0
        assert INTRADAY_LEVERAGE["ETH"] == 10.0

    def test_intraday_alts_have_5x(self):
        """SOL, XRP, ADA should have 5x intraday leverage."""
        assert INTRADAY_LEVERAGE["SOL"] == 5.0
        assert INTRADAY_LEVERAGE["XRP"] == 5.0
        assert INTRADAY_LEVERAGE["ADA"] == 5.0

    def test_swing_leverage_lower_than_intraday(self):
        """Swing leverage should be lower than intraday."""
        for symbol in INTRADAY_LEVERAGE:
            assert SWING_LEVERAGE[symbol] < INTRADAY_LEVERAGE[symbol]

    def test_swing_btc_eth_leverage(self):
        """BTC and ETH swing leverage - verified Jan 24, 2026."""
        assert SWING_LEVERAGE["BTC"] == 4.1
        assert SWING_LEVERAGE["ETH"] == 4.0

    def test_swing_alts_leverage(self):
        """SOL, XRP, ADA swing leverage - verified Jan 24, 2026."""
        assert SWING_LEVERAGE["SOL"] == 2.7
        assert SWING_LEVERAGE["XRP"] == 2.6
        assert SWING_LEVERAGE["ADA"] == 3.4


# =============================================================================
# EFFECTIVE MULTIPLIER TESTS
# =============================================================================


class TestCalculateEffectiveMultiplier:
    """Tests for calculate_effective_multiplier function."""

    def test_btc_intraday_multiplier(self):
        """BTC intraday: 10x leverage x 1.0 beta = 10.0."""
        result = calculate_effective_multiplier("BTC", "intraday")
        assert result == 10.0

    def test_eth_intraday_multiplier(self):
        """ETH intraday: 10x leverage x 1.98 beta = 19.8."""
        result = calculate_effective_multiplier("ETH", "intraday")
        assert result == pytest.approx(19.8, rel=0.01)

    def test_ada_intraday_multiplier(self):
        """ADA intraday: 5x leverage x 2.2 beta = 11.0."""
        result = calculate_effective_multiplier("ADA", "intraday")
        assert result == 11.0

    def test_sol_intraday_multiplier(self):
        """SOL intraday: 5x leverage x 1.55 beta = 7.75."""
        result = calculate_effective_multiplier("SOL", "intraday")
        assert result == pytest.approx(7.75, rel=0.01)

    def test_swing_tier_lower_multipliers(self):
        """Swing tier should produce lower multipliers."""
        for symbol in CRYPTO_BETA_TO_BTC:
            intraday = calculate_effective_multiplier(symbol, "intraday")
            swing = calculate_effective_multiplier(symbol, "swing")
            assert swing < intraday, f"{symbol} swing {swing} not < intraday {intraday}"

    def test_symbol_with_suffix_stripped(self):
        """Symbol with -USD suffix should work."""
        result = calculate_effective_multiplier("BTC-USD", "intraday")
        assert result == 10.0

    def test_lowercase_symbol_works(self):
        """Lowercase symbol should work."""
        result = calculate_effective_multiplier("btc", "intraday")
        assert result == 10.0

    def test_unknown_symbol_defaults(self):
        """Unknown symbol should use default values."""
        result = calculate_effective_multiplier("UNKNOWN", "intraday")
        # Default beta = 1.0, default intraday leverage = 5.0
        assert result == 5.0


class TestGetEffectiveMultipliers:
    """Tests for get_effective_multipliers function."""

    def test_returns_all_symbols(self):
        """Should return multipliers for all tracked symbols."""
        result = get_effective_multipliers("intraday")
        assert set(result.keys()) == set(CRYPTO_BETA_TO_BTC.keys())

    def test_intraday_values_correct(self):
        """Intraday multipliers should match manual calculation."""
        result = get_effective_multipliers("intraday")
        assert result["BTC"] == 10.0
        assert result["ETH"] == pytest.approx(19.8, rel=0.01)

    def test_swing_values_correct(self):
        """Swing multipliers should be lower."""
        intraday = get_effective_multipliers("intraday")
        swing = get_effective_multipliers("swing")
        for symbol in intraday:
            assert swing[symbol] < intraday[symbol]


class TestRankByCapitalEfficiency:
    """Tests for rank_by_capital_efficiency function."""

    def test_returns_sorted_list(self):
        """Should return list sorted by multiplier descending."""
        result = rank_by_capital_efficiency("intraday")
        multipliers = [m for _, m in result]
        assert multipliers == sorted(multipliers, reverse=True)

    def test_eth_first_intraday(self):
        """ETH should be first for intraday (highest multiplier)."""
        result = rank_by_capital_efficiency("intraday")
        assert result[0][0] == "ETH"

    def test_all_symbols_included(self):
        """All symbols should be in ranking."""
        result = rank_by_capital_efficiency("intraday")
        symbols = [s for s, _ in result]
        assert set(symbols) == set(CRYPTO_BETA_TO_BTC.keys())

    def test_returns_tuples(self):
        """Should return list of (symbol, multiplier) tuples."""
        result = rank_by_capital_efficiency("intraday")
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)


# =============================================================================
# P&L PROJECTION TESTS
# =============================================================================


class TestProjectPnlOnBtcMove:
    """Tests for project_pnl_on_btc_move function."""

    def test_btc_move_calculation(self):
        """BTC P&L projection should be straightforward."""
        result = project_pnl_on_btc_move(
            btc_move_percent=0.03,  # 3% BTC move
            account_value=1000,
            symbol="BTC",
            leverage_tier="intraday",
        )
        # BTC: 3% move, 10x leverage, $1000 account = $10,000 notional
        # P&L = $10,000 * 0.03 = $300
        assert result["expected_pnl"] == 300.0
        assert result["pnl_percent"] == 0.30

    def test_high_beta_amplifies_move(self):
        """High beta asset should have amplified move."""
        ada_result = project_pnl_on_btc_move(
            btc_move_percent=0.03,
            account_value=1000,
            symbol="ADA",
            leverage_tier="intraday",
        )
        # ADA: beta 2.2, so expected move = 3% * 2.2 = 6.6%
        assert ada_result["expected_move"] == pytest.approx(0.066, rel=0.01)
        assert ada_result["beta"] == 2.2

    def test_result_contains_required_fields(self):
        """Result should contain all required fields."""
        result = project_pnl_on_btc_move(0.03, 1000, "ETH", "intraday")
        required_fields = [
            "symbol", "btc_move", "beta", "expected_move",
            "leverage", "notional", "expected_pnl", "pnl_percent"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_symbol_normalized(self):
        """Symbol should be normalized to uppercase without suffix."""
        result = project_pnl_on_btc_move(0.03, 1000, "btc-usd", "intraday")
        assert result["symbol"] == "BTC"

    def test_swing_tier_uses_lower_leverage(self):
        """Swing tier should use lower leverage."""
        intraday = project_pnl_on_btc_move(0.03, 1000, "BTC", "intraday")
        swing = project_pnl_on_btc_move(0.03, 1000, "BTC", "swing")
        assert swing["leverage"] < intraday["leverage"]
        assert swing["expected_pnl"] < intraday["expected_pnl"]


class TestCompareInstrumentsOnBtcMove:
    """Tests for compare_instruments_on_btc_move function."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        result = compare_instruments_on_btc_move(0.03, 1000, "intraday")
        assert isinstance(result, pd.DataFrame)

    def test_includes_all_symbols(self):
        """DataFrame should include all tracked symbols."""
        result = compare_instruments_on_btc_move(0.03, 1000, "intraday")
        assert len(result) == len(CRYPTO_BETA_TO_BTC)

    def test_sorted_by_pnl_descending(self):
        """DataFrame should be sorted by expected_pnl descending."""
        result = compare_instruments_on_btc_move(0.03, 1000, "intraday")
        pnl_values = result["expected_pnl"].tolist()
        assert pnl_values == sorted(pnl_values, reverse=True)

    def test_eth_first_for_positive_move(self):
        """ETH should have highest P&L for positive BTC move."""
        result = compare_instruments_on_btc_move(0.03, 1000, "intraday")
        assert result.iloc[0]["symbol"] == "ETH"


# =============================================================================
# ROLLING BETA TESTS
# =============================================================================


class TestCalculateRollingBeta:
    """Tests for calculate_rolling_beta function."""

    def test_returns_series(self):
        """Should return a pandas Series."""
        asset_prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        btc_prices = pd.Series([50000, 51000, 50500, 51500, 52000, 51800, 52500])
        result = calculate_rolling_beta(asset_prices, btc_prices, window=3)
        assert isinstance(result, pd.Series)

    def test_first_values_are_nan(self):
        """First window-1 values should be NaN."""
        asset_prices = pd.Series([100, 102, 101, 103, 105])
        btc_prices = pd.Series([50000, 51000, 50500, 51500, 52000])
        result = calculate_rolling_beta(asset_prices, btc_prices, window=3)
        # First 3 values should be NaN (window=3, plus one for returns)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_positive_correlation_gives_positive_beta(self):
        """Positively correlated assets should have positive beta."""
        # Create perfectly correlated price series
        btc_prices = pd.Series([100, 110, 105, 115, 120, 118, 125, 130])
        # Asset moves 2x BTC
        asset_prices = pd.Series([10, 12, 11, 13, 14, 13.6, 15, 16])
        result = calculate_rolling_beta(asset_prices, btc_prices, window=5)
        # Last value should be approximately 2.0 (asset moves 2x)
        assert result.dropna().iloc[-1] > 0

    def test_same_length_output(self):
        """Output should have same length as input."""
        asset_prices = pd.Series([100, 102, 101, 103, 105])
        btc_prices = pd.Series([50000, 51000, 50500, 51500, 52000])
        result = calculate_rolling_beta(asset_prices, btc_prices, window=3)
        assert len(result) == len(asset_prices)


# =============================================================================
# BETA FROM RANGES TESTS
# =============================================================================


class TestCalculateBetaFromRanges:
    """Tests for calculate_beta_from_ranges function."""

    def test_equal_ranges_gives_beta_one(self):
        """Equal percentage ranges should give beta = 1.0."""
        result = calculate_beta_from_ranges(
            asset_high=110, asset_low=100,
            btc_high=55000, btc_low=50000
        )
        # Both have 10% range
        assert result == pytest.approx(1.0, rel=0.01)

    def test_double_range_gives_beta_two(self):
        """Asset with 2x range should have beta = 2.0."""
        result = calculate_beta_from_ranges(
            asset_high=120, asset_low=100,  # 20% range
            btc_high=55000, btc_low=50000   # 10% range
        )
        assert result == pytest.approx(2.0, rel=0.01)

    def test_half_range_gives_beta_half(self):
        """Asset with 0.5x range should have beta = 0.5."""
        result = calculate_beta_from_ranges(
            asset_high=105, asset_low=100,  # 5% range
            btc_high=55000, btc_low=50000   # 10% range
        )
        assert result == pytest.approx(0.5, rel=0.01)

    def test_zero_btc_range_returns_one(self):
        """Zero BTC range should return default beta of 1.0."""
        result = calculate_beta_from_ranges(
            asset_high=110, asset_low=100,
            btc_high=50000, btc_low=50000  # No range
        )
        assert result == 1.0

    def test_ada_empirical_values(self):
        """Test with actual ADA Day Up/Down values from session."""
        # From CRYPTO-BETA session notes
        result = calculate_beta_from_ranges(
            asset_high=0.3737, asset_low=0.3464,  # ADA
            btc_high=90273, btc_low=87156,        # BTC
        )
        # Should be approximately 2.2
        assert result == pytest.approx(2.2, rel=0.1)


class TestUpdateBetaFromCurrentLevels:
    """Tests for update_beta_from_current_levels function."""

    def test_returns_dict_with_all_symbols(self):
        """Should return betas for all provided symbols."""
        levels = {
            "BTC": {"day_up": 90273, "day_down": 87156},
            "ETH": {"day_up": 3100, "day_down": 2900},
            "ADA": {"day_up": 0.37, "day_down": 0.34},
        }
        result = update_beta_from_current_levels(levels)
        assert "BTC" in result
        assert "ETH" in result
        assert "ADA" in result

    def test_btc_always_beta_one(self):
        """BTC beta should always be 1.0."""
        levels = {
            "BTC": {"day_up": 90000, "day_down": 85000},
            "ETH": {"day_up": 3100, "day_down": 2900},
        }
        result = update_beta_from_current_levels(levels)
        assert result["BTC"] == 1.0

    def test_raises_without_btc(self):
        """Should raise error if BTC not in levels."""
        levels = {
            "ETH": {"day_up": 3100, "day_down": 2900},
        }
        with pytest.raises(ValueError, match="BTC levels required"):
            update_beta_from_current_levels(levels)


# =============================================================================
# BETA-ADJUSTED SIZING TESTS
# =============================================================================


class TestCalculateBetaAdjustedSize:
    """Tests for calculate_beta_adjusted_size function."""

    def test_btc_no_adjustment(self):
        """BTC (beta=1.0) should have no beta adjustment."""
        result = calculate_beta_adjusted_size(
            account_value=1000,
            target_risk_usd=50,
            entry_price=90000,
            stop_price=89000,
            symbol="BTC",
            normalize_to_btc=True,
        )
        # With beta=1.0, adjusted = raw
        assert result["raw_notional"] == result["beta_adjusted_notional"]

    def test_high_beta_reduces_size(self):
        """High beta asset should have reduced position size."""
        btc_result = calculate_beta_adjusted_size(
            account_value=1000, target_risk_usd=50,
            entry_price=90000, stop_price=89000,
            symbol="BTC", normalize_to_btc=True,
        )
        ada_result = calculate_beta_adjusted_size(
            account_value=1000, target_risk_usd=50,
            entry_price=0.35, stop_price=0.34,
            symbol="ADA", normalize_to_btc=True,
        )
        # ADA with higher beta should have smaller adjusted notional
        # (relative to raw) compared to BTC
        btc_ratio = btc_result["beta_adjusted_notional"] / btc_result["raw_notional"]
        ada_ratio = ada_result["beta_adjusted_notional"] / ada_result["raw_notional"]
        assert ada_ratio < btc_ratio

    def test_leverage_capping(self):
        """Position should be capped at max leverage."""
        result = calculate_beta_adjusted_size(
            account_value=100,  # Small account
            target_risk_usd=500,  # Large risk target
            entry_price=90000,
            stop_price=89000,
            symbol="BTC",
            leverage_tier="intraday",
            normalize_to_btc=True,
        )
        # Max notional = $100 * 10x = $1000
        assert result["final_notional"] <= 100 * 10
        assert result["leverage_capped"] == True

    def test_result_contains_required_fields(self):
        """Result should contain all required fields."""
        result = calculate_beta_adjusted_size(
            account_value=1000, target_risk_usd=50,
            entry_price=90000, stop_price=89000,
            symbol="BTC",
        )
        required_fields = [
            "symbol", "beta", "entry_price", "stop_price", "stop_percent",
            "target_risk_usd", "raw_notional", "beta_adjusted_notional",
            "final_notional", "leverage_used", "max_leverage",
            "actual_risk_usd", "beta_adjusted_risk_usd", "leverage_capped"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_normalize_false_uses_raw(self):
        """With normalize_to_btc=False, should use raw notional."""
        result = calculate_beta_adjusted_size(
            account_value=1000, target_risk_usd=50,
            entry_price=0.35, stop_price=0.34,
            symbol="ADA",
            normalize_to_btc=False,
        )
        # Without normalization, adjusted = raw (before leverage cap)
        assert result["beta_adjusted_notional"] == result["raw_notional"]


# =============================================================================
# INSTRUMENT SELECTION TESTS
# =============================================================================


class TestSelectBestInstrument:
    """Tests for select_best_instrument function."""

    def test_returns_highest_efficiency(self):
        """Should return signal with highest effective multiplier."""
        signals = [
            {"symbol": "BTC", "entry": 90000, "stop": 89000, "target": 92000},
            {"symbol": "ETH", "entry": 3000, "stop": 2900, "target": 3200},
            {"symbol": "ADA", "entry": 0.35, "stop": 0.34, "target": 0.40},
        ]
        result = select_best_instrument(signals, 1000, "intraday")
        # ETH has highest effective multiplier (19.8)
        assert result["symbol"] == "ETH"

    def test_filters_by_rr_ratio(self):
        """Should filter out signals below min R:R ratio."""
        signals = [
            {"symbol": "BTC", "entry": 90000, "stop": 89000, "target": 90500},  # R:R = 0.5
            {"symbol": "ETH", "entry": 3000, "stop": 2900, "target": 3200},     # R:R = 2.0
        ]
        result = select_best_instrument(signals, 1000, "intraday", min_rr_ratio=1.5)
        # BTC filtered out due to low R:R
        assert result["symbol"] == "ETH"

    def test_returns_none_if_no_valid_signals(self):
        """Should return None if no signals pass R:R filter."""
        signals = [
            {"symbol": "BTC", "entry": 90000, "stop": 89000, "target": 90500},  # R:R = 0.5
        ]
        result = select_best_instrument(signals, 1000, "intraday", min_rr_ratio=1.5)
        assert result is None

    def test_empty_signals_returns_none(self):
        """Empty signal list should return None."""
        result = select_best_instrument([], 1000, "intraday")
        assert result is None

    def test_adds_efficiency_metrics(self):
        """Selected signal should have efficiency metrics added."""
        signals = [
            {"symbol": "ETH", "entry": 3000, "stop": 2900, "target": 3200},
        ]
        result = select_best_instrument(signals, 1000, "intraday")
        assert "effective_multiplier" in result
        assert "rr_ratio" in result
        assert "beta" in result

    def test_does_not_modify_original(self):
        """Should not modify original signal dictionary."""
        original = {"symbol": "ETH", "entry": 3000, "stop": 2900, "target": 3200}
        signals = [original]
        result = select_best_instrument(signals, 1000, "intraday")
        assert "effective_multiplier" not in original
        assert "effective_multiplier" in result
