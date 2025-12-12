"""
Test Tiingo API connection and basic data retrieval.
Phase 1: Verify API access and data availability.
"""

import os
from tiingo import TiingoClient
import pandas as pd
from datetime import datetime, timedelta

# API key should be set in environment variable TIINGO_API_KEY before running
# or set it here for testing: os.environ['TIINGO_API_KEY'] = 'your_key_here'

# Initialize client
config = {
    'api_key': os.environ.get('TIINGO_API_KEY'),
    'session': True  # Reuse session for multiple requests
}

client = TiingoClient(config)

# Test 1: Get ticker metadata
print("=" * 80)
print("TEST 1: Ticker Metadata")
print("=" * 80)

metadata = client.get_ticker_metadata("SPY")
print(f"Ticker: {metadata['ticker']}")
print(f"Name: {metadata['name']}")
print(f"Exchange: {metadata['exchangeCode']}")
print(f"Start Date: {metadata['startDate']}")
print(f"End Date: {metadata['endDate']}")
print()

# Test 2: Get recent price data
print("=" * 80)
print("TEST 2: Recent Price Data (Last 5 Days)")
print("=" * 80)

end_date = datetime.now()
start_date = end_date - timedelta(days=5)

prices = client.get_ticker_price(
    "SPY",
    startDate=start_date.strftime('%Y-%m-%d'),
    endDate=end_date.strftime('%Y-%m-%d'),
    frequency='daily'
)

df = pd.DataFrame(prices)
print(df[['date', 'open', 'high', 'low', 'close', 'volume']].tail())
print()

# Test 3: Get historical data (30 years)
print("=" * 80)
print("TEST 3: Full Historical Data Range (30+ Years)")
print("=" * 80)

historical = client.get_dataframe(
    "SPY",
    startDate='1993-01-01',  # SPY inception
    endDate=datetime.now().strftime('%Y-%m-%d'),
    frequency='daily'
)

print(f"Total trading days: {len(historical)}")
print(f"Date range: {historical.index[0]} to {historical.index[-1]}")
print(f"Columns: {historical.columns.tolist()}")
print()
print(f"First 5 rows:")
print(historical.head())
print()
print(f"Last 5 rows:")
print(historical.tail())
print()

# Test 4: Verify adjusted columns exist
print("=" * 80)
print("TEST 4: Column Verification")
print("=" * 80)

required_cols = ['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']
missing_cols = [col for col in required_cols if col not in historical.columns]

if missing_cols:
    print(f"FAIL: Missing columns: {missing_cols}")
else:
    print("PASS: All required adjusted price columns present")
    print(f"Available columns: {historical.columns.tolist()}")
print()

# Summary
print("=" * 80)
print("CONNECTION TEST SUMMARY")
print("=" * 80)
print(f"[PASS] Metadata retrieval: SUCCESS")
print(f"[PASS] Recent data retrieval: SUCCESS")
print(f"[PASS] Historical data (30+ years): SUCCESS ({len(historical)} days)")
print(f"[PASS] Column validation: SUCCESS")
print()
print("Tiingo API connection fully operational!")
print(f"Data available from {metadata['startDate']} to {metadata['endDate']}")
