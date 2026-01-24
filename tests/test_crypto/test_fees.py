"""
Tests for crypto/trading/fees.py - Coinbase CFM fee calculations.

Tests cover:
- Fee constants (TAKER_FEE_RATE, MIN_FEE_PER_CONTRACT, CONTRACT_MULTIPLIERS)
- calculate_fee() - percentage vs minimum floor
- calculate_round_trip_fee() - entry + exit
- calculate_breakeven_move() - fee impact on returns
- calculate_num_contracts() - notional to contracts
- calculate_notional_from_contracts() - reverse calculation
- create_coinbase_fee_func() - VBT integration
- create_fixed_pct_fee_func() - simple percentage
- analyze_fee_impact() - comprehensive analysis

Session: EQUITY-82
"""

import pytest

from crypto.trading.fees import (
    TAKER_FEE_RATE,
    MAKER_FEE_RATE,
    MIN_FEE_PER_CONTRACT,
    CONTRACT_MULTIPLIERS,
    calculate_fee,
    calculate_round_trip_fee,
    calculate_breakeven_move,
    calculate_num_contracts,
    calculate_notional_from_contracts,
    create_coinbase_fee_func,
    create_fixed_pct_fee_func,
    analyze_fee_impact,
)


# =============================================================================
# CONSTANT VALIDATION TESTS
# =============================================================================


class TestFeeConstants:
    """Tests for fee constant definitions."""

    def test_taker_fee_rate(self):
        """Taker fee should be 0.02% (0.0002)."""
        assert TAKER_FEE_RATE == 0.0002

    def test_maker_fee_rate_zero(self):
        """Maker fee should be 0.00%."""
        assert MAKER_FEE_RATE == 0.0

    def test_min_fee_per_contract(self):
        """Minimum fee should be $0.15 per contract."""
        assert MIN_FEE_PER_CONTRACT == 0.15

    def test_btc_contract_multiplier(self):
        """BTC nano contract should be 0.01 BTC."""
        assert CONTRACT_MULTIPLIERS["BTC"] == 0.01
        assert CONTRACT_MULTIPLIERS["BIP"] == 0.01

    def test_eth_contract_multiplier(self):
        """ETH nano contract should be 0.10 ETH."""
        assert CONTRACT_MULTIPLIERS["ETH"] == 0.10
        assert CONTRACT_MULTIPLIERS["ETP"] == 0.10

    def test_sol_contract_multiplier(self):
        """SOL nano contract should be 5 SOL."""
        assert CONTRACT_MULTIPLIERS["SOL"] == 5.0

    def test_xrp_contract_multiplier(self):
        """XRP nano contract should be 500 XRP."""
        assert CONTRACT_MULTIPLIERS["XRP"] == 500.0

    def test_ada_contract_multiplier(self):
        """ADA nano contract should be 1000 ADA."""
        assert CONTRACT_MULTIPLIERS["ADA"] == 1000.0


# =============================================================================
# CALCULATE FEE TESTS
# =============================================================================


class TestCalculateFee:
    """Tests for calculate_fee function."""

    def test_percentage_fee_dominates_large_trades(self):
        """For large trades, percentage fee should dominate."""
        # $50,000 notional * 0.02% = $10
        # 5 contracts * $0.15 = $0.75
        # Max = $10
        fee = calculate_fee(notional_value=50000, num_contracts=5, is_maker=False)
        assert fee == 10.0

    def test_minimum_fee_dominates_small_trades(self):
        """For small trades, minimum fee should dominate."""
        # $1,000 notional * 0.02% = $0.20
        # 10 contracts * $0.15 = $1.50
        # Max = $1.50
        fee = calculate_fee(notional_value=1000, num_contracts=10, is_maker=False)
        assert fee == 1.50

    def test_maker_fee_is_zero(self):
        """Maker orders should have zero percentage fee."""
        fee = calculate_fee(notional_value=50000, num_contracts=1, is_maker=True)
        # Only minimum fee applies: 1 * $0.15 = $0.15
        assert fee == 0.15

    def test_zero_notional_returns_zero(self):
        """Zero notional should return zero fee."""
        fee = calculate_fee(notional_value=0, num_contracts=5, is_maker=False)
        assert fee == 0.0

    def test_negative_notional_returns_zero(self):
        """Negative notional should return zero fee."""
        fee = calculate_fee(notional_value=-1000, num_contracts=5, is_maker=False)
        assert fee == 0.0

    def test_zero_contracts_returns_zero(self):
        """Zero contracts should return zero fee."""
        fee = calculate_fee(notional_value=1000, num_contracts=0, is_maker=False)
        assert fee == 0.0

    def test_single_contract_minimum(self):
        """Single contract should have at least $0.15 fee."""
        fee = calculate_fee(notional_value=100, num_contracts=1, is_maker=False)
        # $100 * 0.02% = $0.02
        # 1 * $0.15 = $0.15
        # Max = $0.15
        assert fee == 0.15

    def test_breakeven_point(self):
        """Find the breakeven point where fees are equal."""
        # For 1 contract: $0.15 minimum
        # Percentage fee = notional * 0.0002
        # $0.15 = notional * 0.0002
        # notional = $750
        fee_at_750 = calculate_fee(notional_value=750, num_contracts=1, is_maker=False)
        assert fee_at_750 == pytest.approx(0.15, rel=0.01)


# =============================================================================
# ROUND TRIP FEE TESTS
# =============================================================================


class TestCalculateRoundTripFee:
    """Tests for calculate_round_trip_fee function."""

    def test_double_entry_fee(self):
        """Round trip should be approximately 2x single fee for same order type."""
        single = calculate_fee(notional_value=50000, num_contracts=5, is_maker=False)
        round_trip = calculate_round_trip_fee(50000, 5, entry_is_maker=False, exit_is_maker=False)
        assert round_trip == 2 * single

    def test_mixed_maker_taker(self):
        """Mixed maker/taker should have different entry/exit fees."""
        round_trip = calculate_round_trip_fee(50000, 5, entry_is_maker=True, exit_is_maker=False)
        entry_fee = calculate_fee(50000, 5, is_maker=True)
        exit_fee = calculate_fee(50000, 5, is_maker=False)
        assert round_trip == entry_fee + exit_fee

    def test_both_maker_minimal_fees(self):
        """Both maker orders should only pay minimum fees."""
        round_trip = calculate_round_trip_fee(50000, 5, entry_is_maker=True, exit_is_maker=True)
        # 5 contracts * $0.15 * 2 = $1.50
        assert round_trip == 1.50

    def test_small_trade_round_trip(self):
        """Small trade round trip example from docstring."""
        # $5000 notional, 12 contracts
        round_trip = calculate_round_trip_fee(5000, 12, entry_is_maker=False, exit_is_maker=False)
        # Each side: max(5000 * 0.0002, 12 * 0.15) = max(1.00, 1.80) = 1.80
        # Round trip = 1.80 * 2 = 3.60
        assert round_trip == pytest.approx(3.60, rel=0.01)


# =============================================================================
# BREAKEVEN MOVE TESTS
# =============================================================================


class TestCalculateBreakevenMove:
    """Tests for calculate_breakeven_move function."""

    def test_breakeven_calculation(self):
        """Breakeven move should equal fees / notional."""
        breakeven = calculate_breakeven_move(5000, 12, entry_is_maker=False, exit_is_maker=False)
        # From previous test: round trip = $3.60
        # Breakeven = $3.60 / $5000 = 0.00072
        assert breakeven == pytest.approx(0.00072, rel=0.01)

    def test_zero_notional_returns_zero(self):
        """Zero notional should return zero breakeven."""
        breakeven = calculate_breakeven_move(0, 10)
        assert breakeven == 0.0

    def test_negative_notional_returns_zero(self):
        """Negative notional should return zero breakeven."""
        breakeven = calculate_breakeven_move(-1000, 10)
        assert breakeven == 0.0

    def test_maker_orders_lower_breakeven(self):
        """Maker orders should have lower breakeven."""
        taker_be = calculate_breakeven_move(50000, 5, False, False)
        maker_be = calculate_breakeven_move(50000, 5, True, True)
        assert maker_be < taker_be


# =============================================================================
# CONTRACT CALCULATION TESTS
# =============================================================================


class TestCalculateNumContracts:
    """Tests for calculate_num_contracts function."""

    def test_btc_contracts(self):
        """BTC contract calculation."""
        # $5000 notional, $90,000 BTC price
        # Contract = 0.01 BTC = $900
        # Contracts = $5000 / $900 = 5.55 -> 5
        num = calculate_num_contracts(notional_value=5000, price=90000, symbol="BTC")
        assert num == 5

    def test_ada_contracts(self):
        """ADA contract calculation from docstring."""
        # $5000 notional, $0.35 ADA price
        # Contract = 1000 ADA = $350
        # Contracts = $5000 / $350 = 14.28 -> 14
        num = calculate_num_contracts(notional_value=5000, price=0.35, symbol="ADA")
        assert num == 14

    def test_eth_contracts(self):
        """ETH contract calculation."""
        # $5000 notional, $3000 ETH price
        # Contract = 0.10 ETH = $300
        # Contracts = $5000 / $300 = 16.67 -> 16
        num = calculate_num_contracts(notional_value=5000, price=3000, symbol="ETH")
        assert num == 16

    def test_symbol_with_suffix(self):
        """Symbol with -USD suffix should work."""
        num = calculate_num_contracts(notional_value=5000, price=90000, symbol="BTC-USD")
        assert num == 5

    def test_lowercase_symbol(self):
        """Lowercase symbol should work."""
        num = calculate_num_contracts(notional_value=5000, price=90000, symbol="btc")
        assert num == 5

    def test_unknown_symbol_raises(self):
        """Unknown symbol should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown symbol"):
            calculate_num_contracts(notional_value=5000, price=100, symbol="UNKNOWN")

    def test_zero_price_returns_zero(self):
        """Zero price should return zero contracts."""
        num = calculate_num_contracts(notional_value=5000, price=0, symbol="BTC")
        assert num == 0

    def test_rounds_down(self):
        """Should round down to whole contracts."""
        # Create a scenario that gives a fractional result
        num = calculate_num_contracts(notional_value=999, price=90000, symbol="BTC")
        # 999 / 900 = 1.11 -> 1
        assert num == 1


class TestCalculateNotionalFromContracts:
    """Tests for calculate_notional_from_contracts function."""

    def test_btc_notional(self):
        """BTC notional calculation."""
        # 5 contracts * 0.01 BTC * $90,000 = $4,500
        notional = calculate_notional_from_contracts(5, price=90000, symbol="BTC")
        assert notional == 4500.0

    def test_ada_notional_from_docstring(self):
        """ADA notional calculation from docstring."""
        # 12 contracts * 1000 ADA * $0.3415 = $4,098
        notional = calculate_notional_from_contracts(12, price=0.3415, symbol="ADA")
        assert notional == pytest.approx(4098.0, rel=0.01)

    def test_round_trip_consistency(self):
        """Contracts -> notional -> contracts should be consistent."""
        original_notional = 5000
        price = 0.35
        symbol = "ADA"

        contracts = calculate_num_contracts(original_notional, price, symbol)
        recovered_notional = calculate_notional_from_contracts(contracts, price, symbol)

        # Recovered should be <= original (due to rounding down)
        assert recovered_notional <= original_notional
        # And close to original
        assert recovered_notional >= original_notional * 0.9

    def test_unknown_symbol_raises(self):
        """Unknown symbol should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown symbol"):
            calculate_notional_from_contracts(5, price=100, symbol="UNKNOWN")


# =============================================================================
# VBT INTEGRATION TESTS
# =============================================================================


class TestCreateCoinbaseFeeFunc:
    """Tests for create_coinbase_fee_func function."""

    def test_returns_callable(self):
        """Should return a callable function."""
        fee_func = create_coinbase_fee_func("BTC", is_maker=False)
        assert callable(fee_func)

    def test_callable_signature(self):
        """Fee function should accept col, i, val parameters."""
        fee_func = create_coinbase_fee_func("BTC", is_maker=False)
        # Should not raise
        result = fee_func(0, 0, 1000)
        assert isinstance(result, float)

    def test_positive_fee_for_positive_value(self):
        """Should return positive fee for positive trade value."""
        fee_func = create_coinbase_fee_func("BTC", is_maker=False)
        fee = fee_func(0, 0, 5000)
        assert fee > 0

    def test_positive_fee_for_negative_value(self):
        """Should return positive fee for negative trade value (sells)."""
        fee_func = create_coinbase_fee_func("BTC", is_maker=False)
        fee = fee_func(0, 0, -5000)
        assert fee > 0

    def test_zero_value_returns_zero(self):
        """Zero trade value should return zero fee."""
        fee_func = create_coinbase_fee_func("BTC", is_maker=False)
        fee = fee_func(0, 0, 0)
        assert fee == 0.0

    def test_maker_lower_fees(self):
        """Maker fee function should return lower fees."""
        taker_func = create_coinbase_fee_func("BTC", is_maker=False)
        maker_func = create_coinbase_fee_func("BTC", is_maker=True)

        taker_fee = taker_func(0, 0, 50000)
        maker_fee = maker_func(0, 0, 50000)

        assert maker_fee <= taker_fee


class TestCreateFixedPctFeeFunc:
    """Tests for create_fixed_pct_fee_func function."""

    def test_returns_callable(self):
        """Should return a callable function."""
        fee_func = create_fixed_pct_fee_func(0.0002)
        assert callable(fee_func)

    def test_percentage_calculation(self):
        """Should calculate simple percentage fee."""
        fee_func = create_fixed_pct_fee_func(0.001)  # 0.1%
        fee = fee_func(0, 0, 10000)
        assert fee == 10.0  # 0.1% of 10000

    def test_absolute_value_used(self):
        """Should use absolute value of trade."""
        fee_func = create_fixed_pct_fee_func(0.001)
        pos_fee = fee_func(0, 0, 10000)
        neg_fee = fee_func(0, 0, -10000)
        assert pos_fee == neg_fee

    def test_default_rate(self):
        """Default rate should be 0.0002 (0.02%)."""
        fee_func = create_fixed_pct_fee_func()
        fee = fee_func(0, 0, 10000)
        assert fee == 2.0  # 0.02% of 10000


# =============================================================================
# FEE IMPACT ANALYSIS TESTS
# =============================================================================


class TestAnalyzeFeeImpact:
    """Tests for analyze_fee_impact function."""

    def test_returns_dict_with_required_fields(self):
        """Should return dict with all required fields."""
        result = analyze_fee_impact(
            account_value=1000,
            leverage=5,
            price=0.35,
            symbol="ADA",
            stop_percent=0.02,
            target_percent=0.04,
        )
        required_fields = [
            "notional", "num_contracts", "round_trip_fee", "breakeven_move",
            "fee_as_pct_of_target", "fee_as_pct_of_stop",
            "net_target_pct", "net_rr_ratio"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_notional_calculation(self):
        """Notional should be account_value * leverage (adjusted for contracts)."""
        result = analyze_fee_impact(
            account_value=1000, leverage=5, price=0.35,
            symbol="ADA", stop_percent=0.02, target_percent=0.04,
        )
        # Notional = 1000 * 5 = 5000, but adjusted for whole contracts
        # 14 contracts * 1000 ADA * $0.35 = $4,900
        assert result["notional"] == pytest.approx(4900, rel=0.1)

    def test_num_contracts_whole_number(self):
        """num_contracts should be a whole number."""
        result = analyze_fee_impact(
            account_value=1000, leverage=5, price=0.35,
            symbol="ADA", stop_percent=0.02, target_percent=0.04,
        )
        assert isinstance(result["num_contracts"], int)

    def test_net_rr_lower_than_gross(self):
        """Net R:R ratio should be lower than gross due to fees."""
        result = analyze_fee_impact(
            account_value=1000, leverage=5, price=0.35,
            symbol="ADA", stop_percent=0.02, target_percent=0.04,
        )
        # Gross R:R = 0.04 / 0.02 = 2.0
        gross_rr = 2.0
        assert result["net_rr_ratio"] < gross_rr

    def test_breakeven_smaller_than_target(self):
        """Breakeven move should be smaller than target."""
        result = analyze_fee_impact(
            account_value=1000, leverage=5, price=0.35,
            symbol="ADA", stop_percent=0.02, target_percent=0.04,
        )
        assert result["breakeven_move"] < 0.04

    def test_net_target_accounts_for_fees(self):
        """Net target should be target minus breakeven."""
        result = analyze_fee_impact(
            account_value=1000, leverage=5, price=0.35,
            symbol="ADA", stop_percent=0.02, target_percent=0.04,
        )
        expected_net = 0.04 - result["breakeven_move"]
        assert result["net_target_pct"] == pytest.approx(expected_net, rel=0.01)

    def test_fee_pct_of_target_reasonable(self):
        """Fee as percent of target should be reasonable (<10% typically)."""
        result = analyze_fee_impact(
            account_value=1000, leverage=5, price=0.35,
            symbol="ADA", stop_percent=0.02, target_percent=0.04,
        )
        # Fees should be a small fraction of expected profit
        assert result["fee_as_pct_of_target"] < 0.10
