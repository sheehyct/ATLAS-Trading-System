#!/usr/bin/env python3
"""
CLI entry point for the ATLAS Ticker Selection Pipeline.

Usage:
    python scripts/run_ticker_selection.py              # Full pipeline
    python scripts/run_ticker_selection.py --dry-run    # Screen only, no write
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strat.ticker_selection.config import TickerSelectionConfig
from strat.ticker_selection.pipeline import TickerSelectionPipeline


def main():
    parser = argparse.ArgumentParser(description='ATLAS Ticker Selection Pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Run pipeline without writing candidates.json')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ticker_selection.log', mode='a'),
        ],
    )

    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    config = TickerSelectionConfig.from_env()
    pipeline = TickerSelectionPipeline(config)
    result = pipeline.run(dry_run=args.dry_run)

    # Summary
    stats = result.get('pipeline_stats', {})
    n = stats.get('final_candidates', 0)
    dur = stats.get('scan_duration_seconds', 0)
    print(f"\nDone: {n} candidates in {dur}s")

    if args.dry_run:
        print("(dry run - candidates.json NOT written)")


if __name__ == '__main__':
    main()
