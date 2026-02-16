"""
CLI Entry Point for STRAT Backtesting Pipeline

Replaces the monolithic backtest_strat_options_unified.py with a
modular command-line interface.

Usage:
    python -m strat.backtesting.runners.cli --symbol SPY --timeframes 1D 1W
    python -m strat.backtesting.runners.cli --symbol SPY QQQ --no-capital-tracking
    python -m strat.backtesting.runners.cli --config path/to/config.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from strat.backtesting.config import BacktestConfig
from strat.backtesting.engine import BacktestEngine


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='STRAT Options Backtesting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data selection
    parser.add_argument('--symbol', '-s', nargs='+', default=['SPY'],
                        help='Symbols to backtest (default: SPY)')
    parser.add_argument('--timeframes', '-t', nargs='+', default=['1H', '1D', '1W', '1M'],
                        help='Timeframes to test (default: 1H 1D 1W 1M)')
    parser.add_argument('--start', default='2019-01-01',
                        help='Start date YYYY-MM-DD (default: 2019-01-01)')
    parser.add_argument('--end', default='2025-01-01',
                        help='End date YYYY-MM-DD (default: 2025-01-01)')

    # Risk settings
    parser.add_argument('--risk', type=float, default=300.0,
                        help='Fixed dollar amount per trade (default: 300)')
    parser.add_argument('--capital', type=float, default=3000.0,
                        help='Virtual capital (default: 3000)')
    parser.add_argument('--max-positions', type=int, default=5,
                        help='Max concurrent positions (default: 5)')

    # Feature toggles
    parser.add_argument('--no-capital-tracking', action='store_true',
                        help='Disable capital constraints (unlimited capital)')
    parser.add_argument('--no-trailing-stops', action='store_true',
                        help='Disable trailing stops')
    parser.add_argument('--no-partial-exits', action='store_true',
                        help='Disable partial exits')
    parser.add_argument('--no-pattern-invalidation', action='store_true',
                        help='Disable Type 3 pattern invalidation')

    # Data source
    parser.add_argument('--price-source', choices=['thetadata', 'blackscholes'],
                        default='thetadata',
                        help='Options price source (default: thetadata)')

    # Output
    parser.add_argument('--output', '-o', default='data/backtests',
                        help='Output directory (default: data/backtests)')
    parser.add_argument('--csv', action='store_true',
                        help='Export trades to CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')

    # Config file
    parser.add_argument('--config', type=str,
                        help='Path to JSON config file (overrides other args)')

    return parser.parse_args(argv)


def build_config(args) -> BacktestConfig:
    """Build BacktestConfig from CLI arguments."""
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return BacktestConfig(**data)

    return BacktestConfig(
        symbols=args.symbol,
        timeframes=args.timeframes,
        start_date=args.start,
        end_date=args.end,
        fixed_dollar_amount=args.risk,
        virtual_capital=args.capital,
        max_concurrent_positions=args.max_positions,
        capital_tracking_enabled=not args.no_capital_tracking,
        use_trailing_stop=not args.no_trailing_stops,
        partial_exit_enabled=not args.no_partial_exits,
        pattern_invalidation_enabled=not args.no_pattern_invalidation,
        options_price_source=args.price_source,
        output_dir=args.output,
    )


def setup_price_provider(config: BacktestConfig):
    """Initialize the appropriate options price provider."""
    if config.options_price_source == 'thetadata':
        from strat.backtesting.data_providers.thetadata_provider import ThetaDataProvider
        provider = ThetaDataProvider()
        if provider.connect():
            return provider
        logging.warning("ThetaData connection failed, falling back to Black-Scholes")

    if config.options_price_source == 'blackscholes':
        from strat.backtesting.data_providers.blackscholes_provider import BlackScholesProvider
        return BlackScholesProvider()

    # No provider available - engine will use fallback estimates
    return None


def main(argv=None):
    """Main entry point."""
    args = parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    config = build_config(args)

    # Validate
    issues = config.validate()
    if issues:
        for issue in issues:
            logging.error("Config error: %s", issue)
        sys.exit(1)

    # Initialize price provider
    price_provider = setup_price_provider(config)

    # Run backtest
    engine = BacktestEngine(config, price_provider=price_provider)
    results = engine.run()

    # Print summary
    print(results.summary())

    # Export CSV if requested
    if args.csv and not results.trades_df.empty:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / f"backtest_trades_{config.start_date}_{config.end_date}.csv"
        results.trades_df.to_csv(csv_path, index=False)
        print(f"\nTrades exported to: {csv_path}")

    return results


if __name__ == '__main__':
    main()
