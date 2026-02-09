"""Quick test of Yahoo Finance coverage for all crypto symbols."""
import vectorbtpro as vbt

symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
print('Testing Yahoo Finance coverage (1 year)...')
print('-' * 40)

for s in symbols:
    try:
        data = vbt.YFData.pull(s, start='2024-01-01', end='2026-01-23', silence_warnings=True)
        df = data.get()
        if df is not None and not df.empty:
            print(f'{s}: {len(df)} bars OK')
        else:
            print(f'{s}: NO DATA')
    except Exception as e:
        print(f'{s}: FAILED - {e}')
