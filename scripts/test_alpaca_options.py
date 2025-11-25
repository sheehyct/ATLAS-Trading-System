#!/usr/bin/env python3
"""
Test Alpaca Options Data Access via VBT Pro.

Session 70: Verify Alpaca Algotrader Plus tier supports options data.
"""

import sys
sys.path.insert(0, '.')

import vectorbtpro as vbt
from datetime import datetime, timedelta
from config.settings import get_alpaca_credentials

print('='*60)
print('Testing Alpaca Options Data Access')
print('='*60)

# Get credentials
print('\n[1] Loading Alpaca credentials...')
creds = get_alpaca_credentials('MID')
print(f'  API Key: {creds["api_key"][:8]}...')
print(f'  Base URL: {creds["base_url"]}')

# Test options symbol
# Format: SYMBOL + YYMMDD + C/P + STRIKE*1000
# SPY option expiring Dec 2024, $580 call
# Note: Use a recent expiration for testing
today = datetime.now()
# Find next Friday (options expire on Fridays)
days_until_friday = (4 - today.weekday()) % 7
if days_until_friday == 0:
    days_until_friday = 7
next_friday = today + timedelta(days=days_until_friday + 14)  # 2 weeks out

# Format OSI symbol
exp_str = next_friday.strftime('%y%m%d')
test_strike = 600  # Approximate SPY price
osi_symbol = f'SPY{exp_str}C00{test_strike}000'

print(f'\n[2] Test OSI Symbol: {osi_symbol}')
print(f'  Expiration: {next_friday.strftime("%Y-%m-%d")}')
print(f'  Strike: ${test_strike}')

# Try to fetch options data
print('\n[3] Attempting to fetch options data via VBT Pro...')

try:
    # Method 1: Using client_type="options"
    print('  Method 1: vbt.AlpacaData.pull with client_type="options"')

    data = vbt.AlpacaData.pull(
        osi_symbol,
        client_type="options",
        client_config={
            'api_key': creds['api_key'],
            'secret_key': creds['secret_key'],
        },
        start=(today - timedelta(days=7)).strftime('%Y-%m-%d'),
        end=today.strftime('%Y-%m-%d'),
        timeframe="1 day"
    )

    df = data.get()
    print(f'  SUCCESS: Retrieved {len(df)} bars')
    print(f'  Columns: {list(df.columns)}')
    if len(df) > 0:
        print(f'  Last row:')
        print(df.tail(1))

except Exception as e:
    error_msg = str(e)
    print(f'  Error: {error_msg[:200]}...' if len(error_msg) > 200 else f'  Error: {error_msg}')

    # Check if it's an auth/subscription error
    if '401' in error_msg or 'unauthorized' in error_msg.lower():
        print('\n  [!] 401 Unauthorized - Check API credentials')
    elif 'subscription' in error_msg.lower() or 'access' in error_msg.lower():
        print('\n  [!] Subscription issue - May need Alpaca Algotrader Plus tier')
    elif 'not found' in error_msg.lower():
        print('\n  [!] Symbol not found - May be invalid OSI symbol or no data')

# Try listing available options
print('\n[4] Attempting to list SPY options...')
try:
    from alpaca.data.historical import OptionHistoricalDataClient

    options_client = OptionHistoricalDataClient(
        api_key=creds['api_key'],
        secret_key=creds['secret_key']
    )

    print('  OptionHistoricalDataClient created successfully')
    print('  This indicates Alpaca SDK supports options')

except ImportError:
    print('  alpaca.data.historical.OptionHistoricalDataClient not available')
    print('  May need to update alpaca-py SDK')
except Exception as e:
    print(f'  Error: {e}')

# Summary
print('\n' + '='*60)
print('Test Summary')
print('='*60)
print('''
Alpaca Options Data Access requires:
1. Alpaca Algotrader Plus subscription (options enabled)
2. Valid OSI symbol format for existing options
3. VBT Pro with client_type="options"

If data access failed:
- Verify Alpaca subscription includes options data
- Check that option symbol exists and has recent data
- Consider using a more liquid option (ATM, near expiration)
''')
