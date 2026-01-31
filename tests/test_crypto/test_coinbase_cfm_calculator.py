"""
Unit tests for Coinbase CFM P/L Calculator.

Tests FIFO matching, fee calculation, and P/L computation for:
- Crypto perpetuals (BIP, ETP, SOP, ADP, XRP)
- Commodity futures (SLRH, GOLJ)
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from crypto.analytics.coinbase_cfm_calculator import (
    CoinbaseCFMCalculator,
    CFMTransaction,
    CFMLot,
    CFMRealizedPL,
    CFMOpenPosition,
    extract_base_symbol,
    classify_product,
    calculate_fee,
    TAKER_FEE_RATE,
    MAKER_FEE_RATE,
    FIXED_FEE_PER_CONTRACT,
    CRYPTO_PERPS,
    COMMODITY_FUTURES,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator():
    """Create fresh calculator instance."""
    return CoinbaseCFMCalculator()


@pytest.fixture
def sample_fills():
    """Sample fills data mimicking Coinbase API response."""
    now = datetime.now(timezone.utc)
    return [
        # Open BIP long position
        {
            "entry_id": "fill-001",
            "order_id": "order-001",
            "product_id": "BIP-20DEC30-CDE",
            "side": "BUY",
            "size": "0.5",
            "price": "50000.00",
            "commission": "17.65",  # 0.07% * 25000 + 0.15
            "trade_time": (now - timedelta(hours=5)).isoformat(),
            "liquidity_indicator": "TAKER",
        },
        # Close BIP long position (partial)
        {
            "entry_id": "fill-002",
            "order_id": "order-002",
            "product_id": "BIP-20DEC30-CDE",
            "side": "SELL",
            "size": "0.25",
            "price": "51000.00",
            "commission": "9.08",  # 0.07% * 12750 + 0.15
            "trade_time": (now - timedelta(hours=2)).isoformat(),
            "liquidity_indicator": "TAKER",
        },
        # Open SLRH position
        {
            "entry_id": "fill-003",
            "order_id": "order-003",
            "product_id": "SLRH-20MAR26-CDE",
            "side": "BUY",
            "size": "10",
            "price": "25.50",
            "commission": "0.33",
            "trade_time": (now - timedelta(hours=1)).isoformat(),
            "liquidity_indicator": "MAKER",
        },
    ]


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestExtractBaseSymbol:
    """Test extract_base_symbol function."""

    def test_bitcoin_perp(self):
        assert extract_base_symbol("BIP-20DEC30-CDE") == "BIP"

    def test_ethereum_perp(self):
        assert extract_base_symbol("ETP-20DEC30-CDE") == "ETP"

    def test_silver_futures(self):
        assert extract_base_symbol("SLRH-20MAR26-CDE") == "SLRH"

    def test_gold_futures(self):
        assert extract_base_symbol("GOLJ-20MAR26-CDE") == "GOLJ"

    def test_empty_string(self):
        assert extract_base_symbol("") == ""

    def test_no_dash(self):
        assert extract_base_symbol("BIP") == "BIP"


class TestClassifyProduct:
    """Test classify_product function."""

    def test_crypto_perps(self):
        for symbol in CRYPTO_PERPS:
            product_id = f"{symbol}-20DEC30-CDE"
            assert classify_product(product_id) == "crypto_perp"

    def test_commodity_futures(self):
        for symbol in COMMODITY_FUTURES:
            product_id = f"{symbol}-20MAR26-CDE"
            assert classify_product(product_id) == "commodity_future"

    def test_unknown_product(self):
        assert classify_product("UNKNOWN-20DEC30-CDE") == "unknown"

    def test_empty_string(self):
        assert classify_product("") == "unknown"


class TestCalculateFee:
    """Test calculate_fee function."""

    def test_taker_fee(self):
        notional = 10000.0
        expected = (notional * TAKER_FEE_RATE) + FIXED_FEE_PER_CONTRACT
        assert calculate_fee(notional, is_maker=False) == pytest.approx(expected)

    def test_maker_fee(self):
        notional = 10000.0
        expected = (notional * MAKER_FEE_RATE) + FIXED_FEE_PER_CONTRACT
        assert calculate_fee(notional, is_maker=True) == pytest.approx(expected)

    def test_multiple_contracts(self):
        notional = 10000.0
        num_contracts = 5
        expected = (notional * TAKER_FEE_RATE) + (FIXED_FEE_PER_CONTRACT * num_contracts)
        assert calculate_fee(notional, is_maker=False, num_contracts=num_contracts) == pytest.approx(expected)

    def test_zero_notional(self):
        assert calculate_fee(0) == FIXED_FEE_PER_CONTRACT


# =============================================================================
# CFMTransaction TESTS
# =============================================================================

class TestCFMTransaction:
    """Test CFMTransaction dataclass."""

    def test_from_coinbase_fill(self, sample_fills):
        fill = sample_fills[0]
        txn = CFMTransaction.from_coinbase_fill(fill)

        assert txn.fill_id == "fill-001"
        assert txn.order_id == "order-001"
        assert txn.product_id == "BIP-20DEC30-CDE"
        assert txn.side == "BUY"
        assert txn.size == 0.5
        assert txn.price == 50000.0
        assert txn.fee == 17.65
        assert txn.is_maker is False

    def test_base_symbol_property(self, sample_fills):
        txn = CFMTransaction.from_coinbase_fill(sample_fills[0])
        assert txn.base_symbol == "BIP"

    def test_product_type_property(self, sample_fills):
        txn = CFMTransaction.from_coinbase_fill(sample_fills[0])
        assert txn.product_type == "crypto_perp"

        txn = CFMTransaction.from_coinbase_fill(sample_fills[2])
        assert txn.product_type == "commodity_future"

    def test_notional_property(self, sample_fills):
        txn = CFMTransaction.from_coinbase_fill(sample_fills[0])
        assert txn.notional == pytest.approx(25000.0)

    def test_to_dict(self, sample_fills):
        txn = CFMTransaction.from_coinbase_fill(sample_fills[0])
        d = txn.to_dict()

        assert d["fill_id"] == "fill-001"
        assert d["base_symbol"] == "BIP"
        assert d["product_type"] == "crypto_perp"
        assert d["notional"] == pytest.approx(25000.0)


# =============================================================================
# CALCULATOR TESTS - BASIC
# =============================================================================

class TestCalculatorBasic:
    """Test basic calculator functionality."""

    def test_empty_fills(self, calculator):
        calculator.process_fills([])
        assert calculator.get_realized_pnl() == []
        assert calculator.get_open_positions() == []

    def test_single_open_position(self, calculator):
        fills = [{
            "entry_id": "fill-001",
            "order_id": "order-001",
            "product_id": "BIP-20DEC30-CDE",
            "side": "BUY",
            "size": "1.0",
            "price": "50000.00",
            "commission": "35.15",
            "trade_time": datetime.now(timezone.utc).isoformat(),
        }]

        calculator.process_fills(fills)

        assert len(calculator.get_realized_pnl()) == 0
        positions = calculator.get_open_positions()
        assert len(positions) == 1
        assert positions[0].base_symbol == "BIP"
        assert positions[0].quantity == pytest.approx(1.0)


# =============================================================================
# CALCULATOR TESTS - FIFO MATCHING
# =============================================================================

class TestCalculatorFIFO:
    """Test FIFO P/L matching."""

    def test_simple_round_trip(self, calculator):
        """Test opening and fully closing a position."""
        now = datetime.now(timezone.utc)
        fills = [
            {
                "entry_id": "fill-001",
                "order_id": "order-001",
                "product_id": "BIP-20DEC30-CDE",
                "side": "BUY",
                "size": "1.0",
                "price": "50000.00",
                "commission": "35.15",
                "trade_time": (now - timedelta(hours=2)).isoformat(),
            },
            {
                "entry_id": "fill-002",
                "order_id": "order-002",
                "product_id": "BIP-20DEC30-CDE",
                "side": "SELL",
                "size": "1.0",
                "price": "51000.00",
                "commission": "35.85",
                "trade_time": now.isoformat(),
            },
        ]

        calculator.process_fills(fills)

        # Should have one realized P/L and no open positions
        realized = calculator.get_realized_pnl()
        assert len(realized) == 1
        assert realized[0].quantity == pytest.approx(1.0)
        assert realized[0].entry_price == pytest.approx(50000.0)
        assert realized[0].exit_price == pytest.approx(51000.0)

        # Gross P/L = (51000 - 50000) * 1 = 1000
        assert realized[0].gross_pnl == pytest.approx(1000.0, rel=0.01)

        # No open positions
        assert len(calculator.get_open_positions()) == 0

    def test_partial_close(self, calculator):
        """Test partially closing a position (FIFO)."""
        now = datetime.now(timezone.utc)
        fills = [
            {
                "entry_id": "fill-001",
                "order_id": "order-001",
                "product_id": "ETP-20DEC30-CDE",
                "side": "BUY",
                "size": "10.0",
                "price": "3000.00",
                "commission": "21.15",
                "trade_time": (now - timedelta(hours=2)).isoformat(),
            },
            {
                "entry_id": "fill-002",
                "order_id": "order-002",
                "product_id": "ETP-20DEC30-CDE",
                "side": "SELL",
                "size": "4.0",
                "price": "3100.00",
                "commission": "8.83",
                "trade_time": now.isoformat(),
            },
        ]

        calculator.process_fills(fills)

        # Should have one realized P/L for 4 units
        realized = calculator.get_realized_pnl()
        assert len(realized) == 1
        assert realized[0].quantity == pytest.approx(4.0)

        # Should have open position for remaining 6 units
        positions = calculator.get_open_positions()
        assert len(positions) == 1
        assert positions[0].quantity == pytest.approx(6.0)

    def test_fifo_order_multiple_lots(self, calculator):
        """Test that FIFO matches oldest lots first."""
        now = datetime.now(timezone.utc)
        fills = [
            # First lot at $50,000
            {
                "entry_id": "fill-001",
                "order_id": "order-001",
                "product_id": "BIP-20DEC30-CDE",
                "side": "BUY",
                "size": "0.5",
                "price": "50000.00",
                "commission": "17.65",
                "trade_time": (now - timedelta(hours=3)).isoformat(),
            },
            # Second lot at $52,000
            {
                "entry_id": "fill-002",
                "order_id": "order-002",
                "product_id": "BIP-20DEC30-CDE",
                "side": "BUY",
                "size": "0.5",
                "price": "52000.00",
                "commission": "18.35",
                "trade_time": (now - timedelta(hours=2)).isoformat(),
            },
            # Close 0.5 at $55,000 - should match against FIRST lot
            {
                "entry_id": "fill-003",
                "order_id": "order-003",
                "product_id": "BIP-20DEC30-CDE",
                "side": "SELL",
                "size": "0.5",
                "price": "55000.00",
                "commission": "19.40",
                "trade_time": now.isoformat(),
            },
        ]

        calculator.process_fills(fills)

        realized = calculator.get_realized_pnl()
        assert len(realized) == 1
        # Should have matched first lot at $50,000
        assert realized[0].entry_price == pytest.approx(50000.0)
        assert realized[0].exit_price == pytest.approx(55000.0)

        # Remaining lot should be the $52,000 one
        positions = calculator.get_open_positions()
        assert len(positions) == 1
        assert positions[0].avg_entry_price == pytest.approx(52000.0, rel=0.01)


class TestCalculatorShortPositions:
    """Test short position handling."""

    def test_short_round_trip(self, calculator):
        """Test opening short and covering."""
        now = datetime.now(timezone.utc)
        fills = [
            # Open short
            {
                "entry_id": "fill-001",
                "order_id": "order-001",
                "product_id": "GOLJ-20MAR26-CDE",
                "side": "SELL",
                "size": "5.0",
                "price": "2000.00",
                "commission": "7.15",
                "trade_time": (now - timedelta(hours=2)).isoformat(),
            },
            # Cover short (buy to close)
            {
                "entry_id": "fill-002",
                "order_id": "order-002",
                "product_id": "GOLJ-20MAR26-CDE",
                "side": "BUY",
                "size": "5.0",
                "price": "1950.00",
                "commission": "6.98",
                "trade_time": now.isoformat(),
            },
        ]

        calculator.process_fills(fills)

        realized = calculator.get_realized_pnl()
        assert len(realized) == 1
        assert realized[0].side == "SELL"
        # Short profit when price goes down
        # Gross P/L = (2000 - 1950) * 5 = 250
        assert realized[0].gross_pnl == pytest.approx(250.0, rel=0.01)


# =============================================================================
# CALCULATOR TESTS - SUMMARIES
# =============================================================================

class TestCalculatorSummaries:
    """Test summary methods."""

    def test_realized_pnl_total(self, calculator, sample_fills):
        calculator.process_fills(sample_fills)
        totals = calculator.get_realized_pnl_total()

        assert "gross_pnl" in totals
        assert "total_fees" in totals
        assert "net_pnl" in totals
        assert "trade_count" in totals

    def test_pnl_by_product(self, calculator, sample_fills):
        calculator.process_fills(sample_fills)
        by_product = calculator.get_pnl_by_product()

        # Should have entry for BIP (partial close)
        if "BIP" in by_product:
            assert "net_pnl" in by_product["BIP"]
            assert "trade_count" in by_product["BIP"]
            assert "win_rate" in by_product["BIP"]

    def test_pnl_by_product_type(self, calculator, sample_fills):
        calculator.process_fills(sample_fills)
        by_type = calculator.get_pnl_by_product_type()

        assert "crypto_perp" in by_type
        assert "commodity_future" in by_type

    def test_performance_metrics_empty(self, calculator):
        """Test performance metrics with no trades."""
        metrics = calculator.get_performance_metrics()
        assert metrics["trade_count"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0

    def test_performance_metrics_with_trades(self, calculator):
        """Test performance metrics with some trades."""
        now = datetime.now(timezone.utc)
        fills = [
            # Winning trade
            {"entry_id": "1", "order_id": "1", "product_id": "BIP-20DEC30-CDE",
             "side": "BUY", "size": "1.0", "price": "50000.00", "commission": "35.15",
             "trade_time": (now - timedelta(hours=3)).isoformat()},
            {"entry_id": "2", "order_id": "2", "product_id": "BIP-20DEC30-CDE",
             "side": "SELL", "size": "1.0", "price": "52000.00", "commission": "36.55",
             "trade_time": (now - timedelta(hours=2)).isoformat()},
            # Losing trade
            {"entry_id": "3", "order_id": "3", "product_id": "ETP-20DEC30-CDE",
             "side": "BUY", "size": "1.0", "price": "3000.00", "commission": "2.25",
             "trade_time": (now - timedelta(hours=1)).isoformat()},
            {"entry_id": "4", "order_id": "4", "product_id": "ETP-20DEC30-CDE",
             "side": "SELL", "size": "1.0", "price": "2900.00", "commission": "2.18",
             "trade_time": now.isoformat()},
        ]

        calculator.process_fills(fills)
        metrics = calculator.get_performance_metrics()

        assert metrics["trade_count"] == 2
        assert metrics["win_count"] == 1
        assert metrics["loss_count"] == 1
        assert metrics["win_rate"] == pytest.approx(50.0)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_fill_data(self, calculator):
        """Test handling of invalid fill data."""
        fills = [
            {"entry_id": "", "order_id": "", "product_id": "", "side": "",
             "size": "", "price": "", "commission": ""},
        ]
        # Should not raise
        calculator.process_fills(fills)
        assert len(calculator.get_realized_pnl()) == 0

    def test_duplicate_fill_ids(self, calculator):
        """Test handling of duplicate fill IDs."""
        now = datetime.now(timezone.utc)
        fills = [
            {"entry_id": "fill-001", "order_id": "order-001", "product_id": "BIP-20DEC30-CDE",
             "side": "BUY", "size": "1.0", "price": "50000.00", "commission": "35.15",
             "trade_time": (now - timedelta(hours=1)).isoformat()},
            {"entry_id": "fill-001", "order_id": "order-002", "product_id": "BIP-20DEC30-CDE",
             "side": "BUY", "size": "1.0", "price": "51000.00", "commission": "35.85",
             "trade_time": now.isoformat()},
        ]
        # Should handle gracefully
        calculator.process_fills(fills)
        positions = calculator.get_open_positions()
        assert len(positions) == 1
        assert positions[0].quantity == pytest.approx(2.0)

    def test_flip_position(self, calculator):
        """Test flipping from long to short in one trade."""
        now = datetime.now(timezone.utc)
        fills = [
            # Long 1 BTC
            {"entry_id": "1", "order_id": "1", "product_id": "BIP-20DEC30-CDE",
             "side": "BUY", "size": "1.0", "price": "50000.00", "commission": "35.15",
             "trade_time": (now - timedelta(hours=1)).isoformat()},
            # Sell 2 BTC (close long + open short)
            {"entry_id": "2", "order_id": "2", "product_id": "BIP-20DEC30-CDE",
             "side": "SELL", "size": "2.0", "price": "51000.00", "commission": "71.55",
             "trade_time": now.isoformat()},
        ]

        calculator.process_fills(fills)

        # Should have 1 realized P/L (closed long)
        realized = calculator.get_realized_pnl()
        assert len(realized) == 1
        assert realized[0].quantity == pytest.approx(1.0)

        # Should have 1 open short position
        positions = calculator.get_open_positions()
        assert len(positions) == 1
        assert positions[0].side == "SELL"
        assert positions[0].quantity == pytest.approx(1.0)


# =============================================================================
# OPEN POSITION TESTS
# =============================================================================

class TestOpenPosition:
    """Test CFMOpenPosition functionality."""

    def test_update_unrealized_pnl_long(self, calculator):
        """Test unrealized P/L calculation for long position."""
        now = datetime.now(timezone.utc)
        fills = [
            {"entry_id": "1", "order_id": "1", "product_id": "BIP-20DEC30-CDE",
             "side": "BUY", "size": "1.0", "price": "50000.00", "commission": "35.15",
             "trade_time": now.isoformat()},
        ]

        calculator.process_fills(fills)
        positions = calculator.get_open_positions()

        assert len(positions) == 1
        pos = positions[0]

        # Update with current price
        pos.update_unrealized_pnl(52000.0)

        # Unrealized P/L = (52000 - 50000) * 1 - fees
        assert pos.unrealized_pnl > 0
        assert pos.current_price == 52000.0

    def test_to_dict(self, calculator):
        """Test CFMOpenPosition.to_dict()."""
        now = datetime.now(timezone.utc)
        fills = [
            {"entry_id": "1", "order_id": "1", "product_id": "BIP-20DEC30-CDE",
             "side": "BUY", "size": "1.0", "price": "50000.00", "commission": "35.15",
             "trade_time": now.isoformat()},
        ]

        calculator.process_fills(fills)
        positions = calculator.get_open_positions()
        pos_dict = positions[0].to_dict()

        assert pos_dict["product_id"] == "BIP-20DEC30-CDE"
        assert pos_dict["base_symbol"] == "BIP"
        assert pos_dict["asset_name"] == "Bitcoin"
        assert pos_dict["side"] == "BUY"
        assert pos_dict["product_type"] == "crypto_perp"


# =============================================================================
# FUNDING TESTS
# =============================================================================

class TestFundingPayments:
    """Test funding payment tracking."""

    def test_add_funding_payments(self, calculator):
        payments = [
            {"product_id": "BIP-20DEC30-CDE", "amount": 5.50, "timestamp": "2026-01-15T08:00:00Z"},
            {"product_id": "BIP-20DEC30-CDE", "amount": -2.25, "timestamp": "2026-01-15T16:00:00Z"},
        ]

        calculator.add_funding_payments(payments)
        summary = calculator.get_funding_summary()

        assert summary["total_paid"] == pytest.approx(5.50)
        assert summary["total_received"] == pytest.approx(2.25)
        assert summary["net_funding"] == pytest.approx(-3.25)  # received - paid
        assert summary["payment_count"] == 2

    def test_empty_funding_payments(self, calculator):
        summary = calculator.get_funding_summary()
        assert summary["total_paid"] == 0
        assert summary["total_received"] == 0
        assert summary["net_funding"] == 0
        assert summary["payment_count"] == 0
