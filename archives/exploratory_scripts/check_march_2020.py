from data.alpaca import fetch_alpaca_data
import pandas as pd

data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3300)
print(f"Data length: {len(data)}")
print(f"First date: {data.index[0]}")
print(f"Last date: {data.index[-1]}")

march_2020_start = pd.to_datetime('2020-03-01').tz_localize('UTC')
if march_2020_start >= data.index[0]:
    # Find index of March 2020
    march_idx = None
    for i, date in enumerate(data.index):
        if date >= march_2020_start:
            march_idx = i
            break

    print(f"\nMarch 2020 at index: {march_idx}")
    print(f"With 1500 lookback, inference starts at index 1500")
    print(f"March 2020 in results: {march_idx > 1500}")
    print(f"\nTo test March 2020, need lookback < {march_idx}")
    print(f"Recommended lookback: {march_idx - 50} (with buffer)")
