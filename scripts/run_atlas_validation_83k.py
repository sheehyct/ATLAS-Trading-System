#!/usr/bin/env python3
"""
ATLAS STRAT Validation Execution Script - Session 83K

Executes comprehensive ATLAS Production Readiness validation for STRAT strategies
using REAL ThetaData options data.

Validation Matrix (120 total runs - Session 83K-35: Added 1H):
| Batch | Pattern | Timeframes     | Symbols | Runs |
|-------|---------|----------------|---------|------|
| 1     | 3-1-2   | 1H,1D,1W,1M    | 6       | 24   |
| 2     | 2-1-2   | 1H,1D,1W,1M    | 6       | 24   |
| 3     | 2-2 Up  | 1H,1D,1W,1M    | 6       | 24   |
| 4     | 3-2     | 1H,1D,1W,1M    | 6       | 24   |
| 5     | 3-2-2   | 1H,1D,1W,1M    | 6       | 24   |

NOTE: Hourly (1H) uses market-open-aligned bars (09:30, 10:30, 11:30, etc.)
      NOT clock-aligned bars. This is CRITICAL for STRAT time rules.

ThetaData Integration:
- Historical quotes (bid/ask) - tick level
- Greeks 1st Order (delta, theta, vega, rho)
- Implied volatility
- 8 years of historical data (2016-2024)

Usage:
    # Run full validation
    python scripts/run_atlas_validation_83k.py

    # Run single batch
    python scripts/run_atlas_validation_83k.py --batch 3-1-2

    # Resume from checkpoint
    python scripts/run_atlas_validation_83k.py --resume

    # Dry run (verify setup without running)
    python scripts/run_atlas_validation_83k.py --dry-run

    # Skip specific patterns/symbols
    python scripts/run_atlas_validation_83k.py --skip-patterns 3-2-2 --skip-symbols NVDA

Output:
    validation_results/session_83k/
    ├── checkpoint.json           # Resume state
    ├── summary/
    │   ├── master_report.json    # Full summary
    │   └── master_report.csv     # CSV export
    ├── by_pattern/
    │   ├── 312/batch_*.json      # Per-batch results
    │   ├── 212/
    │   ├── 22/
    │   ├── 32/
    │   └── 322/
    └── bugs/
        └── bugs_discovered.json  # Any bugs found
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.strat_validator import (
    ATLASSTRATValidator,
    ThetaDataStatus,
    DEFAULT_PATTERNS,
    DEFAULT_TIMEFRAMES,
    DEFAULT_SYMBOLS,
    EXPANDED_SYMBOLS,  # Session 83K-53
    TICKER_CATEGORIES,  # Session 83K-53
)


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging with file and console handlers."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Log file with timestamp
    log_file = logs_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)


def print_banner():
    """Print session banner."""
    print()
    print("=" * 70)
    print("ATLAS STRAT VALIDATION - Session 83K")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Validation Matrix:")
    print(f"  Patterns:   {', '.join(DEFAULT_PATTERNS)}")
    print(f"  Timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
    print(f"  Symbols:    {', '.join(DEFAULT_SYMBOLS)}")
    print(f"  Total Runs: {len(DEFAULT_PATTERNS) * len(DEFAULT_TIMEFRAMES) * len(DEFAULT_SYMBOLS)}")
    print()
    print("ThetaData Standard Tier:")
    print("  - Historical quotes (bid/ask)")
    print("  - Greeks 1st Order (delta, theta, vega, rho)")
    print("  - Implied volatility")
    print("  - 8 years of data")
    print("=" * 70)
    print()


def verify_thetadata(validator: ATLASSTRATValidator) -> bool:
    """Verify ThetaData connection and print status."""
    print("Verifying ThetaData connection...")

    status = validator.verify_thetadata_connection()

    print()
    print("ThetaData Status:")
    print(f"  Connected:        {status.connected}")
    print(f"  Quotes Available: {status.quotes_available}")
    print(f"  Greeks Available: {status.greeks_available}")

    if status.symbols_checked:
        available = [s for s, ok in status.symbols_checked.items() if ok]
        unavailable = [s for s, ok in status.symbols_checked.items() if not ok]
        print(f"  Symbols OK:       {', '.join(available) if available else 'None'}")
        if unavailable:
            print(f"  Symbols MISSING:  {', '.join(unavailable)}")

    if status.error_message:
        print(f"  Error:            {status.error_message}")

    print()

    return status.connected


def run_dry_run(validator: ATLASSTRATValidator, args: argparse.Namespace) -> None:
    """Execute dry run - verify setup without running validation."""
    print("=" * 70)
    print("DRY RUN - Verifying Setup")
    print("=" * 70)
    print()

    # Check ThetaData
    if verify_thetadata(validator):
        print("[OK] ThetaData connection verified")
    else:
        print("[WARNING] ThetaData not available - will use Black-Scholes fallback")

    # Check data fetcher
    print()
    print("Testing data fetch...")
    try:
        from validation.strat_validator import DataFetcher
        fetcher = DataFetcher()
        data = fetcher.get_data('SPY', '1D')
        print(f"[OK] SPY daily data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"[ERROR] Data fetch failed: {e}")

    # Check strategy
    print()
    print("Testing strategy initialization...")
    try:
        from strategies.strat_options_strategy import STRATOptionsStrategy
        strategy = STRATOptionsStrategy()
        print(f"[OK] STRATOptionsStrategy initialized")
        print(f"     Patterns: {strategy.config.pattern_types}")
    except Exception as e:
        print(f"[ERROR] Strategy init failed: {e}")

    # Check validation runner
    print()
    print("Testing validation runner...")
    try:
        from validation import ValidationRunner
        runner = ValidationRunner()
        print("[OK] ValidationRunner initialized")
    except Exception as e:
        print(f"[ERROR] ValidationRunner failed: {e}")

    print()
    print("=" * 70)
    print("DRY RUN COMPLETE")
    print("=" * 70)

    # Show what would be run
    # Session 83K-53: Use selected_symbols from args
    patterns = [p for p in DEFAULT_PATTERNS if p not in (args.skip_patterns or [])]
    timeframes = [t for t in DEFAULT_TIMEFRAMES if t not in (args.skip_timeframes or [])]
    symbols = [s for s in args.selected_symbols if s not in (args.skip_symbols or [])]

    if args.batch:
        patterns = [args.batch]

    total = len(patterns) * len(timeframes) * len(symbols)

    print()
    print(f"Validation would run {total} combinations:")
    print(f"  Patterns:   {patterns}")
    print(f"  Timeframes: {timeframes}")
    print(f"  Symbols:    {symbols}")


def run_single_batch(validator: ATLASSTRATValidator, pattern: str, args: argparse.Namespace) -> None:
    """Run validation for a single pattern."""
    print(f"\nRunning single batch: {pattern}")
    print()

    result = validator.run_batch(
        pattern=pattern,
        skip_timeframes=args.skip_timeframes,
        skip_symbols=args.skip_symbols,
        resume=args.resume,
    )

    # Print batch summary
    print()
    print(f"Batch {pattern} Results:")
    print(f"  Total:    {result.total_runs}")
    print(f"  Passed:   {result.passed_runs}")
    print(f"  Failed:   {result.failed_runs}")
    print(f"  Errors:   {result.error_runs}")
    print(f"  Pass Rate: {result.pass_rate*100:.1f}%")
    print(f"  ThetaData Coverage: {result.avg_thetadata_coverage:.1f}%")
    print(f"  Time:     {result.total_execution_time:.1f}s")


def run_full_validation(validator: ATLASSTRATValidator, args: argparse.Namespace) -> None:
    """Run full validation across all patterns."""
    print("\nStarting full validation...")
    print()

    results = validator.run_full_validation(
        resume=args.resume,
        skip_patterns=args.skip_patterns,
        skip_timeframes=args.skip_timeframes,
        skip_symbols=args.skip_symbols,
    )

    # Print summary
    print()
    print(validator.print_summary())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ATLAS STRAT Validation Execution Script - Session 83K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_atlas_validation_83k.py              # Full validation
  python scripts/run_atlas_validation_83k.py --batch 3-1-2  # Single batch
  python scripts/run_atlas_validation_83k.py --resume     # Resume from checkpoint
  python scripts/run_atlas_validation_83k.py --dry-run    # Verify setup only
        """
    )

    parser.add_argument(
        '--batch',
        type=str,
        choices=DEFAULT_PATTERNS,
        help='Run single batch for specific pattern'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Verify setup without running validation'
    )

    parser.add_argument(
        '--skip-patterns',
        nargs='+',
        choices=DEFAULT_PATTERNS,
        help='Patterns to skip'
    )

    parser.add_argument(
        '--skip-timeframes',
        nargs='+',
        choices=DEFAULT_TIMEFRAMES,
        help='Timeframes to skip'
    )

    parser.add_argument(
        '--skip-symbols',
        nargs='+',
        choices=DEFAULT_SYMBOLS,
        help='Symbols to skip'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_results/session_83k',
        help='Output directory for results'
    )

    parser.add_argument(
        '--no-thetadata',
        action='store_true',
        help='Skip ThetaData requirement (use Black-Scholes only)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Session 83K-14: Holdout mode for sparse pattern strategies
    parser.add_argument(
        '--holdout',
        action='store_true',
        help='Use holdout validation (70/30 split) instead of walk-forward. '
             'Recommended for sparse pattern strategies like STRAT.'
    )

    parser.add_argument(
        '--include-nvda',
        action='store_true',
        help='Include NVDA in validation (excluded by default due to pre-split strike data issues)'
    )

    # Session 83K-53: Expanded ticker universe support
    parser.add_argument(
        '--universe',
        type=str,
        choices=['default', 'expanded', 'index_only', 'sector_only'],
        default='default',
        help='Symbol universe to validate: default (6 symbols), expanded (16 symbols), '
             'index_only (4 ETFs), sector_only (4 sector ETFs)'
    )

    args = parser.parse_args()

    # Session 83K-53: Select symbol universe
    if args.universe == 'expanded':
        selected_symbols = EXPANDED_SYMBOLS
        print(f"[INFO] Using EXPANDED universe: {len(selected_symbols)} symbols")
    elif args.universe == 'index_only':
        selected_symbols = TICKER_CATEGORIES['index_etf']
        print(f"[INFO] Using INDEX_ONLY universe: {selected_symbols}")
    elif args.universe == 'sector_only':
        selected_symbols = TICKER_CATEGORIES['sector_etf']
        print(f"[INFO] Using SECTOR_ONLY universe: {selected_symbols}")
    else:
        selected_symbols = DEFAULT_SYMBOLS
        print(f"[INFO] Using DEFAULT universe: {len(selected_symbols)} symbols")

    # Store selected symbols in args for use by run functions
    args.selected_symbols = selected_symbols

    # Session 83K-14: Auto-exclude NVDA unless explicitly included
    # NVDA pre-split strikes (pre-July 2021) cause ThetaData 472 errors
    if not args.include_nvda:
        if args.skip_symbols is None:
            args.skip_symbols = ['NVDA']
        elif 'NVDA' not in args.skip_symbols:
            args.skip_symbols.append('NVDA')
        print("[INFO] NVDA auto-excluded (pre-split strike data issues). Use --include-nvda to override.")

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    print_banner()

    # Initialize validator
    # Session 83K-14: Pass holdout_mode for sparse pattern strategies
    # Session 83K-53: Pass selected symbols for expanded universe support
    validator = ATLASSTRATValidator(
        output_dir=output_dir,
        require_thetadata=not args.no_thetadata,
        holdout_mode=args.holdout,
        symbols=args.selected_symbols,
    )

    if args.holdout:
        print("[INFO] Holdout mode enabled (70/30 train/test split)")
        print("       Recommended for sparse pattern strategies like STRAT.")

    try:
        if args.dry_run:
            run_dry_run(validator, args)
        elif args.batch:
            # Verify ThetaData first
            if not args.no_thetadata:
                if not verify_thetadata(validator):
                    print("\n[ERROR] ThetaData not available. Use --no-thetadata to skip.")
                    sys.exit(1)
            run_single_batch(validator, args.batch, args)
        else:
            # Full validation
            if not args.no_thetadata:
                if not verify_thetadata(validator):
                    print("\n[ERROR] ThetaData not available. Use --no-thetadata to skip.")
                    sys.exit(1)
            run_full_validation(validator, args)

        print()
        print(f"Results saved to: {output_dir}")
        print()
        print("=" * 70)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        print("Progress saved to checkpoint. Use --resume to continue.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        print(f"\n[ERROR] Validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
