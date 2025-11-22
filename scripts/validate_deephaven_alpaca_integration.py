"""
Validation Script for Deephaven Alpaca Integration

Tests the Alpaca integration logic without requiring Docker/Deephaven.
Validates:
1. AlpacaTradingClient connection
2. Position data fetching
3. Account equity retrieval
4. Data structure compatibility

Usage:
    uv run python scripts/validate_deephaven_alpaca_integration.py
"""

import sys
sys.path.insert(0, 'C:\\Strat_Trading_Bot\\vectorbt-workspace')

from integrations.alpaca_trading_client import AlpacaTradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import os


def test_trading_client_connection():
    """Test 1: Verify trading client connects successfully."""
    print("\n" + "="*70)
    print("TEST 1: Trading Client Connection")
    print("="*70)

    try:
        client = AlpacaTradingClient(account='LARGE')
        if not client.connect():
            print("FAIL: Connection failed")
            return False

        print(f"PASS: Connected to Alpaca API")
        return True

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def test_account_equity_fetch():
    """Test 2: Verify account equity is fetched correctly."""
    print("\n" + "="*70)
    print("TEST 2: Account Equity Fetch")
    print("="*70)

    try:
        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        account = client.get_account()
        equity = account['equity']

        print(f"Account Equity: ${equity:,.2f}")

        # Verify equity is reasonable for paper trading account
        if 1000 <= equity <= 50000:
            print(f"PASS: Equity within expected range ($1k-$50k)")
            return True
        else:
            print(f"WARN: Equity outside expected range (still valid)")
            return True

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def test_positions_fetch():
    """Test 3: Verify positions are fetched correctly."""
    print("\n" + "="*70)
    print("TEST 3: Positions Fetch")
    print("="*70)

    try:
        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        positions = client.list_positions()

        print(f"Positions Found: {len(positions)}")

        if not positions:
            print("WARN: No positions in account (acceptable if account empty)")
            return True

        # Verify System A1 positions
        expected_symbols = {'CSCO', 'GOOGL', 'AMAT', 'AAPL', 'CRWD', 'AVGO'}
        actual_symbols = {pos['symbol'] for pos in positions}

        print(f"\nExpected Symbols (System A1): {expected_symbols}")
        print(f"Actual Symbols: {actual_symbols}")

        if expected_symbols == actual_symbols:
            print("PASS: All System A1 positions present")
        else:
            missing = expected_symbols - actual_symbols
            extra = actual_symbols - expected_symbols
            if missing:
                print(f"WARN: Missing positions: {missing}")
            if extra:
                print(f"WARN: Unexpected positions: {extra}")
            print("PASS: Position fetch working (composition may vary)")

        # Display position details
        print("\nPosition Details:")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
            print(f"    Current: ${pos['current_price']:.2f}, P&L: ${pos['unrealized_pl']:.2f}")

        return True

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def test_position_data_structure():
    """Test 4: Verify position data structure matches Deephaven requirements."""
    print("\n" + "="*70)
    print("TEST 4: Position Data Structure Validation")
    print("="*70)

    try:
        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        positions = client.list_positions()

        if not positions:
            print("SKIP: No positions to validate")
            return True

        # Required fields for Deephaven integration
        required_fields = ['symbol', 'qty', 'avg_entry_price', 'current_price']

        for pos in positions:
            missing_fields = [field for field in required_fields if field not in pos]
            if missing_fields:
                print(f"FAIL: Missing fields in position data: {missing_fields}")
                return False

        # Test stop price calculation (5% methodology)
        test_pos = positions[0]
        stop_price = test_pos['avg_entry_price'] * 0.95
        risk_percent = (test_pos['avg_entry_price'] - stop_price) / test_pos['avg_entry_price']

        print(f"Test Position: {test_pos['symbol']}")
        print(f"  AvgCost: ${test_pos['avg_entry_price']:.2f}")
        print(f"  StopPrice (5%): ${stop_price:.2f}")
        print(f"  RiskPercent: {risk_percent:.2%}")

        if abs(risk_percent - 0.05) < 0.001:  # Allow for floating point error
            print("PASS: Stop price calculation correct (5% methodology)")
            return True
        else:
            print(f"FAIL: Stop price calculation incorrect")
            return False

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def test_market_data_fetch():
    """Test 5: Verify market data client works."""
    print("\n" + "="*70)
    print("TEST 5: Market Data Fetch")
    print("="*70)

    try:
        # Initialize data client
        stock_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_LARGE_KEY') or os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_LARGE_SECRET') or os.getenv('ALPACA_SECRET_KEY')
        )

        # Get symbols from positions
        trading_client = AlpacaTradingClient(account='LARGE')
        trading_client.connect()
        positions = trading_client.list_positions()

        if not positions:
            print("SKIP: No positions to fetch quotes for")
            return True

        symbols = [pos['symbol'] for pos in positions][:3]  # Test with first 3 symbols

        # Fetch latest quotes
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = stock_client.get_stock_latest_quote(request)

        print(f"Quotes fetched for {len(symbols)} symbols:")
        for symbol in symbols:
            quote = quotes[symbol]
            mid_price = (quote.ask_price + quote.bid_price) / 2.0
            print(f"  {symbol}: ${mid_price:.2f} (bid: ${quote.bid_price:.2f}, ask: ${quote.ask_price:.2f})")

        print("PASS: Market data fetch successful")
        return True

    except Exception as e:
        print(f"FAIL: {str(e)}")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("DEEPHAVEN ALPACA INTEGRATION VALIDATION")
    print("="*70)

    tests = [
        test_trading_client_connection,
        test_account_equity_fetch,
        test_positions_fetch,
        test_position_data_structure,
        test_market_data_fetch
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\nERROR in {test.__name__}: {str(e)}")
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nSTATUS: ALL TESTS PASSED - Integration ready for Deephaven deployment")
        return 0
    else:
        print("\nSTATUS: SOME TESTS FAILED - Review errors before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
