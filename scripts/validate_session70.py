#!/usr/bin/env python3
"""
Session 70 End-to-End Validation Script.

Validates all components implemented in Session 70:
1. Centralized config.settings (.env fix)
2. Tier 1 Pattern Detector
3. Options Module
4. Alpaca Options Data Access
5. Paper Trading Pipeline

Run: uv run python scripts/validate_session70.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(title):
    print('\n' + '='*60)
    print(title)
    print('='*60)

def print_pass(msg):
    print(f'  [PASS] {msg}')

def print_fail(msg):
    print(f'  [FAIL] {msg}')

def main():
    print_header('SESSION 70 END-TO-END VALIDATION')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    results = {
        'config_fix': False,
        'tier1_detector': False,
        'options_module': False,
        'alpaca_options': False,
        'paper_trading': False,
    }

    # ========================================
    # TEST 1: Centralized Config
    # ========================================
    print_header('TEST 1: Centralized Config System')

    try:
        from config.settings import (
            load_config,
            get_alpaca_credentials,
            get_tiingo_key,
            is_config_loaded
        )

        load_config()

        # Test Tiingo key (the key that was missing before)
        tiingo_key = get_tiingo_key()
        assert tiingo_key is not None, 'Tiingo key is None'
        assert len(tiingo_key) > 10, 'Tiingo key too short'
        print_pass(f'Tiingo API key loaded: {tiingo_key[:8]}...')

        # Test Alpaca credentials for all accounts
        for account in ['SMALL', 'MID', 'LARGE']:
            creds = get_alpaca_credentials(account)
            assert creds['api_key'] is not None, f'{account} API key is None'
            print_pass(f'Alpaca {account} credentials loaded')

        results['config_fix'] = True

    except Exception as e:
        print_fail(f'Config error: {e}')

    # ========================================
    # TEST 2: Tier 1 Pattern Detector
    # ========================================
    print_header('TEST 2: Tier 1 Pattern Detector')

    try:
        from strat.tier1_detector import (
            Tier1Detector,
            Timeframe,
            PatternType,
            detect_tier1_patterns
        )
        from integrations.tiingo_data_fetcher import TiingoDataFetcher

        # Test initialization validation
        try:
            bad_detector = Tier1Detector(min_continuation_bars=1)
            print_fail('Should reject min_bars < 2')
        except ValueError:
            print_pass('Correctly rejects min_continuation_bars < 2')

        # Test pattern detection
        fetcher = TiingoDataFetcher()
        data = fetcher.fetch('SPY', start_date='2020-01-01', end_date='2024-12-31', timeframe='1W')
        spy_df = data.get()

        detector = Tier1Detector(min_continuation_bars=2)
        signals = detector.detect_patterns(spy_df, timeframe=Timeframe.WEEKLY)

        assert len(signals) > 0, 'No patterns detected'
        print_pass(f'Detected {len(signals)} Tier 1 patterns')

        # Verify 2-2 Down exclusion
        has_22_down = any(s.pattern_type == PatternType.PATTERN_22_DOWN for s in signals)
        assert not has_22_down, '2-2 Down should be excluded by default'
        print_pass('2-2 Down (2U-2D) correctly excluded')

        # Verify continuation bar filter
        for sig in signals[:5]:
            assert sig.continuation_bars >= 2, 'Pattern should have 2+ continuation bars'
        print_pass('Continuation bar filter working (min 2 bars)')

        # Generate summary
        summary = detector.generate_summary(signals)
        print_pass(f'Summary: {summary["total_signals"]} signals, patterns: {list(summary["by_pattern_type"].keys())}')

        results['tier1_detector'] = True

    except Exception as e:
        print_fail(f'Tier 1 detector error: {e}')
        import traceback
        traceback.print_exc()

    # ========================================
    # TEST 3: Options Module
    # ========================================
    print_header('TEST 3: Options Module')

    try:
        from strat.options_module import (
            OptionsExecutor,
            OptionContract,
            OptionType,
            OptionsBacktester
        )

        # Test OSI symbol generation
        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2024, 12, 20),
            option_type=OptionType.CALL,
            strike=300.0
        )
        assert contract.osi_symbol == 'SPY241220C00300000', f'Wrong OSI: {contract.osi_symbol}'
        print_pass('OSI symbol generation correct')

        # Test options executor with patterns
        if results['tier1_detector']:
            executor = OptionsExecutor()
            trades = executor.generate_option_trades(
                signals[:5],
                underlying='SPY',
                underlying_price=450.0
            )
            assert len(trades) > 0, 'No trades generated'
            print_pass(f'Generated {len(trades)} option trades from patterns')

            # Test DataFrame conversion
            df = executor.trades_to_dataframe(trades)
            assert 'osi_symbol' in df.columns, 'Missing osi_symbol column'
            print_pass('Trades DataFrame conversion working')

            # Test backtester
            backtester = OptionsBacktester()
            results_df = backtester.backtest_trades(trades, spy_df)
            print_pass(f'Backtester ran: {len(results_df)} results')

        results['options_module'] = True

    except Exception as e:
        print_fail(f'Options module error: {e}')
        import traceback
        traceback.print_exc()

    # ========================================
    # TEST 4: Alpaca Options Data Access
    # ========================================
    print_header('TEST 4: Alpaca Options Data Access')

    try:
        import vectorbtpro as vbt

        creds = get_alpaca_credentials('MID')

        # Generate test OSI symbol
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7 or 7
        exp_date = today + timedelta(days=days_until_friday + 14)
        exp_str = exp_date.strftime('%y%m%d')
        test_osi = f'SPY{exp_str}C00600000'

        print(f'  Testing symbol: {test_osi}')

        data = vbt.AlpacaData.pull(
            test_osi,
            client_type='options',
            client_config={
                'api_key': creds['api_key'],
                'secret_key': creds['secret_key'],
            },
            start=(today - timedelta(days=7)).strftime('%Y-%m-%d'),
            end=today.strftime('%Y-%m-%d'),
            timeframe='1 day'
        )

        df = data.get()
        assert len(df) > 0, 'No options data retrieved'
        print_pass(f'Retrieved {len(df)} bars of options data')
        print_pass(f'Columns: {list(df.columns)[:5]}...')

        results['alpaca_options'] = True

    except Exception as e:
        print_fail(f'Alpaca options error: {e}')

    # ========================================
    # TEST 5: Paper Trading Pipeline
    # ========================================
    print_header('TEST 5: Paper Trading Pipeline')

    try:
        # Import paper trading components
        from scripts.paper_trade_options import PaperTradingSession

        # Quick test with 1 symbol
        session = PaperTradingSession(symbols=['SPY'])
        df = session.run_scan()

        print_pass('Paper trading session runs without error')
        print_pass(f'Scan returned {len(df)} trades')

        results['paper_trading'] = True

    except Exception as e:
        print_fail(f'Paper trading error: {e}')
        import traceback
        traceback.print_exc()

    # ========================================
    # SUMMARY
    # ========================================
    print_header('VALIDATION SUMMARY')

    total = len(results)
    passed = sum(results.values())

    print(f'\nResults:')
    for test, passed_test in results.items():
        status = '[PASS]' if passed_test else '[FAIL]'
        print(f'  {status} {test}')

    print(f'\nTotal: {passed}/{total} tests passed')

    if passed == total:
        print('\n' + '*'*60)
        print('SESSION 70 VALIDATION COMPLETE - ALL TESTS PASSED!')
        print('*'*60)
        print('''
Implemented Features:
1. Centralized config.settings - Permanent .env fix
2. Tier 1 Pattern Detector - Validated patterns with filters
3. Options Module - OSI symbols, strike selection, backtesting
4. Alpaca Options Data - VBT Pro integration working
5. Paper Trading Pipeline - Ready for weekly scans

Next Steps:
- Run weekly paper trading scans
- Track option performance over 30 days
- Fine-tune strike selection based on results
''')
    else:
        print('\n[WARNING] Some tests failed - review errors above')

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
