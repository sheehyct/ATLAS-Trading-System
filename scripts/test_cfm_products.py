#!/usr/bin/env python3
"""
Test script to discover available perpetual futures products on Coinbase.

Purpose: Find correct product IDs for US CFM perpetuals (BIP/EIP) vs INTX products.

Usage:
    uv run python scripts/test_cfm_products.py
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def test_list_products():
    """List all perpetual futures products available via API."""
    from coinbase.rest import RESTClient

    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: Missing COINBASE_API_KEY or COINBASE_API_SECRET in .env")
        return

    # Format private key if needed
    if "BEGIN EC PRIVATE KEY" not in api_secret:
        api_secret = f"-----BEGIN EC PRIVATE KEY-----\n{api_secret}\n-----END EC PRIVATE KEY-----"

    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret)
        print("Authenticated successfully!\n")
    except Exception as e:
        print(f"ERROR: Failed to authenticate: {e}")
        return

    # List all products with futures filter
    print("=" * 60)
    print("PERPETUAL FUTURES PRODUCTS")
    print("=" * 60)

    try:
        # Try listing products with futures filter
        products = client.get_products(
            product_type="FUTURE",
            contract_expiry_type="PERPETUAL",
        )

        perpetuals = []
        if hasattr(products, "products"):
            perpetuals = products.products
        elif isinstance(products, dict) and "products" in products:
            perpetuals = products["products"]

        if not perpetuals:
            print("No perpetual products found with filter.")
            print("\nTrying to list ALL products...")
            all_products = client.get_products()
            if hasattr(all_products, "products"):
                all_prods = all_products.products
            elif isinstance(all_products, dict) and "products" in all_products:
                all_prods = all_products["products"]
            else:
                all_prods = []

            # Filter for perpetuals manually
            for p in all_prods:
                product_id = p.get("product_id", "") if isinstance(p, dict) else getattr(p, "product_id", "")
                product_type = p.get("product_type", "") if isinstance(p, dict) else getattr(p, "product_type", "")
                if "PERP" in product_id.upper() or product_type == "FUTURE":
                    perpetuals.append(p)

        print(f"\nFound {len(perpetuals)} perpetual/futures products:\n")

        for p in perpetuals:
            if isinstance(p, dict):
                product_id = p.get("product_id", "N/A")
                base = p.get("base_currency_id", p.get("base_name", "N/A"))
                quote = p.get("quote_currency_id", p.get("quote_name", "N/A"))
                status = p.get("status", "N/A")
                product_type = p.get("product_type", "N/A")
            else:
                product_id = getattr(p, "product_id", "N/A")
                base = getattr(p, "base_currency_id", getattr(p, "base_name", "N/A"))
                quote = getattr(p, "quote_currency_id", getattr(p, "quote_name", "N/A"))
                status = getattr(p, "status", "N/A")
                product_type = getattr(p, "product_type", "N/A")

            print(f"  {product_id}")
            print(f"    Base: {base}, Quote: {quote}")
            print(f"    Type: {product_type}, Status: {status}")
            print()

    except Exception as e:
        print(f"ERROR listing products: {e}")
        import traceback
        traceback.print_exc()


def test_specific_products():
    """Test fetching data for specific product IDs."""
    from crypto.exchange.coinbase_client import CoinbaseClient

    # Products to test - exhaustive search for US derivatives
    test_products = [
        # Nano Bitcoin Perp Style (BIP) - the actual product user trades
        "BIP",            # Direct BIP symbol
        "BIP-USD",        # BIP with quote currency
        "BIP-PERP",       # BIP perp format
        "BIPZ0",          # Contract month format (Dec)
        "BIP-PERP-CDX",   # BIP with CDX venue
        # Nano Ether equivalent
        "EIP",            # Direct EIP symbol
        "EIP-USD",        # EIP with quote currency
        # Generic perp formats we already tested
        "BTC-PERP",       # Hyphen format
        "ETH-PERP",       # Hyphen format
        "BTC_PERP",       # Underscore format
        "BTC-PERP-CDX",   # CDX venue suffix
        # Working INTX for comparison
        "BTC-PERP-INTX",  # Current INTX product (should work)
    ]

    print("\n" + "=" * 60)
    print("TESTING SPECIFIC PRODUCT IDS")
    print("=" * 60)

    client = CoinbaseClient()

    for product_id in test_products:
        print(f"\n--- {product_id} ---")
        try:
            # Try to get current price
            price = client.get_current_price(product_id)
            if price:
                print(f"  Current Price: ${price:,.2f}")
            else:
                print(f"  Price: Not available")

            # Try to get recent candles
            df = client.get_historical_ohlcv(
                symbol=product_id,
                interval="1h",
                limit=3,
            )
            if df is not None and not df.empty:
                print(f"  Recent 1h candles:")
                for idx, row in df.tail(3).iterrows():
                    print(f"    {idx}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")
            else:
                print(f"  Candles: Not available")

        except Exception as e:
            print(f"  ERROR: {e}")


def compare_prices():
    """Compare prices between INTX and potential CFM products."""
    from crypto.exchange.coinbase_client import CoinbaseClient

    print("\n" + "=" * 60)
    print("PRICE COMPARISON: INTX vs CFM")
    print("=" * 60)

    client = CoinbaseClient()

    pairs = [
        ("BTC-PERP-INTX", "BTC-PERP"),
        ("ETH-PERP-INTX", "ETH-PERP"),
    ]

    for intx, cfm in pairs:
        print(f"\n{intx.split('-')[0]}:")

        intx_price = None
        cfm_price = None

        try:
            intx_price = client.get_current_price(intx)
            print(f"  INTX ({intx}): ${intx_price:,.2f}" if intx_price else f"  INTX ({intx}): Not available")
        except Exception as e:
            print(f"  INTX ({intx}): ERROR - {e}")

        try:
            cfm_price = client.get_current_price(cfm)
            print(f"  CFM  ({cfm}): ${cfm_price:,.2f}" if cfm_price else f"  CFM  ({cfm}): Not available")
        except Exception as e:
            print(f"  CFM  ({cfm}): ERROR - {e}")

        if intx_price and cfm_price:
            diff = cfm_price - intx_price
            print(f"  Difference: ${diff:,.2f} ({diff/intx_price*100:.2f}%)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("COINBASE CFM PRODUCT DISCOVERY")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    # Test 1: List all perpetual products
    test_list_products()

    # Test 2: Try specific product IDs
    test_specific_products()

    # Test 3: Compare prices
    compare_prices()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If BTC-PERP/ETH-PERP work, update crypto/config.py")
    print("2. If not, check API response for actual product ID format")
    print("3. Update CRYPTO_SYMBOLS in config with correct IDs")


if __name__ == "__main__":
    main()
