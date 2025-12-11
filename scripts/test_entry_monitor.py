"""Test entry monitor manually - Session 83K-67"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strat.signal_automation.signal_store import SignalStore, TIMEFRAME_PRIORITY
from strat.signal_automation.entry_monitor import EntryMonitor, EntryMonitorConfig
from integrations.alpaca_trading_client import AlpacaTradingClient
from typing import Dict, List

def main():
    # Setup - SignalStore auto-loads on init
    store = SignalStore()

    client = AlpacaTradingClient('SMALL')
    client.connect()

    def price_fetcher(symbols: List[str]) -> Dict[str, float]:
        quotes = client.get_stock_quotes(symbols)
        prices = {}
        for symbol, quote in quotes.items():
            if isinstance(quote, dict) and 'mid' in quote:
                prices[symbol] = quote['mid']
            elif isinstance(quote, (int, float)):
                prices[symbol] = float(quote)
        return prices

    # Create monitor (without market hours restriction for testing)
    config = EntryMonitorConfig(
        poll_interval=60,
        market_hours_only=False,  # Allow testing after hours
    )

    monitor = EntryMonitor(
        signal_store=store,
        price_fetcher=price_fetcher,
        config=config,
    )

    print('ENTRY MONITOR TEST (Bypassing Market Hours)')
    print('=' * 60)
    print(f'Pending signals: {len(monitor.get_pending_signals())}')
    print()

    # Check triggers
    triggered = monitor.check_triggers()

    print(f'TRIGGERED SIGNALS: {len(triggered)}')
    print('-' * 60)
    for event in triggered:
        sig = event.signal
        print(f'  [{sig.timeframe}] {sig.symbol} {sig.pattern_type} {sig.direction}')
        print(f'       Trigger: ${event.trigger_price:.2f} | Current: ${event.current_price:.2f}')
        print(f'       Priority: {event.priority}')
        print()

    print()
    print('Monitor Stats:')
    stats = monitor.get_stats()
    for k, v in stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
