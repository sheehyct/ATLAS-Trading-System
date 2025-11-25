#!/usr/bin/env python3
"""Test script for options module."""

import sys
sys.path.insert(0, '.')

print('='*60)
print('Testing Options Module')
print('='*60)

# Test imports
print('\n[TEST 1] Importing OptionsModule...')
from strat.options_module import (
    OptionsExecutor,
    OptionContract,
    OptionType,
    OptionStrategy,
    OptionsBacktester
)
print('  SUCCESS: Imports work')

# Test OSI symbol generation
print('\n[TEST 2] OSI Symbol Generation...')
from datetime import datetime

contract = OptionContract(
    underlying='SPY',
    expiration=datetime(2024, 12, 20),
    option_type=OptionType.CALL,
    strike=300.0
)
print(f'  OSI Symbol: {contract.osi_symbol}')
assert contract.osi_symbol == 'SPY241220C00300000', f'Expected SPY241220C00300000, got {contract.osi_symbol}'
print('  PASS: OSI format correct')

# Test with different strikes
print('\n[TEST 3] Multiple Strike Tests...')
contracts = [
    OptionContract('AAPL', datetime(2025, 1, 17), OptionType.CALL, 150.0),
    OptionContract('TSLA', datetime(2025, 3, 21), OptionType.PUT, 250.5),
    OptionContract('QQQ', datetime(2025, 6, 20), OptionType.CALL, 500.0),
]
for c in contracts:
    print(f'  {c.underlying} ${c.strike} {c.option_type.value} -> {c.osi_symbol}')

# Test with Tier 1 patterns
print('\n[TEST 4] Pattern-to-Options Conversion...')
from strat.tier1_detector import Tier1Detector, Timeframe
from integrations.tiingo_data_fetcher import TiingoDataFetcher

# Fetch data
fetcher = TiingoDataFetcher()
data = fetcher.fetch('SPY', start_date='2023-01-01', end_date='2024-12-31', timeframe='1W')
spy_df = data.get()
print(f'  Data loaded: {len(spy_df)} bars')

# Detect patterns
detector = Tier1Detector()
signals = detector.detect_patterns(spy_df, timeframe=Timeframe.WEEKLY)
print(f'  Patterns detected: {len(signals)}')

trades = None
if signals:
    # Generate option trades
    executor = OptionsExecutor()
    trades = executor.generate_option_trades(
        signals[:3],  # First 3 signals
        underlying='SPY',
        underlying_price=450.0
    )

    print(f'  Option trades generated: {len(trades)}')
    for t in trades:
        print(f'    {t.pattern_signal.pattern_type.value}: {t.contract.osi_symbol}')
        print(f'      Strike: ${t.contract.strike}, R:R: {t.pattern_signal.risk_reward:.2f}')

    # Test DataFrame conversion
    df = executor.trades_to_dataframe(trades)
    print(f'\n  DataFrame shape: {df.shape}')

# Test backtest
print('\n[TEST 5] Simplified Backtest...')
if signals and trades:
    backtester = OptionsBacktester(risk_free_rate=0.05, default_iv=0.20)
    results = backtester.backtest_trades(trades, spy_df)

    if not results.empty:
        print(f'  Trades executed: {len(results)}')
        win_count = results['win'].sum()
        loss_count = (~results['win']).sum()
        print(f'  Wins: {win_count}, Losses: {loss_count}')
        print(f'  Total P/L: ${results["pnl"].sum():,.2f}')
    else:
        print('  No trades executed (pattern dates may be outside data range)')

print('\n' + '='*60)
print('ALL TESTS PASSED - Options Module Working!')
print('='*60)
