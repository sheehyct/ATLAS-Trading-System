#!/usr/bin/env python3
"""
Generate synthetic candidates.json for testing the daemon and Discord summary.

Output goes to data/candidates/synthetic_candidates.json by default.
The daemon rejects files with "synthetic": true in pipeline_stats, so
this can NEVER accidentally drive live trading.

Usage:
    python scripts/generate_synthetic_candidates.py                    # 12 defaults
    python scripts/generate_synthetic_candidates.py --tickers AAPL NVDA TSLA
    python scripts/generate_synthetic_candidates.py --count 6
    python scripts/generate_synthetic_candidates.py --live-enrich      # Real Finviz data
    python scripts/generate_synthetic_candidates.py --preview          # Discord preview only
    python scripts/generate_synthetic_candidates.py --write-live       # Overwrite real candidates.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Synthetic data pool: ~20 candidates with realistic values
# ---------------------------------------------------------------------------
CANDIDATE_POOL = [
    # (symbol, pattern, direction, timeframe, entry, stop, target, current,
    #  tfc_score, tfc_align, sector, industry, earnings, tgt_price, recom, headlines)
    ('NVDA', '3-1-2U', 'CALL', '1D', 135.50, 131.20, 144.80, 134.10, 4, '4/4 BULLISH',
     'Technology', 'Semiconductors', 'Feb 26 AMC', 170.0, 'Strong Buy',
     ['Nvidia CEO talks AI infrastructure demand', 'NVDA hits new data center revenue record', 'Analyst raises NVDA target to $180']),
    ('AAPL', '2-1-2U', 'CALL', '1H', 242.30, 239.50, 248.00, 241.80, 3, '3/4 BULLISH',
     'Technology', 'Consumer Electronics', 'Jan 29 AMC', 260.0, 'Buy',
     ['Apple Vision Pro 2 rumored for Q3', 'AAPL services revenue hits new high', 'iPhone 17 production ramp begins']),
    ('MSFT', '2-1-2U', 'CALL', '1D', 425.00, 419.50, 438.00, 423.60, 3, '3/4 BULLISH',
     'Technology', 'Software - Infrastructure', 'Apr 22 AMC', 480.0, 'Buy',
     ['Microsoft Azure AI revenue surges 40%', 'MSFT Copilot adoption accelerates', 'Cloud market share gains continue']),
    ('AVGO', '3-1-2U', 'CALL', '1H', 195.00, 190.80, 205.50, 193.70, 4, '4/4 BULLISH',
     'Technology', 'Semiconductors', 'Mar 06 AMC', 225.0, 'Strong Buy',
     ['Broadcom AI networking revenue doubles', 'VMware integration ahead of schedule', 'Custom ASIC demand surges']),
    ('CRM', '2-1-2U', 'CALL', '1D', 325.40, 319.00, 338.00, 323.90, 3, '3/4 BULLISH',
     'Technology', 'Software - Application', 'Feb 26 AMC', 365.0, 'Buy',
     ['Salesforce Agentforce platform gains traction', 'CRM margins expand on AI efficiency', 'New enterprise deals accelerate']),
    ('META', '2-1-2U', 'CALL', '1H', 615.00, 605.00, 640.00, 612.30, 3, '3/4 BULLISH',
     'Communication Services', 'Internet Content', 'Apr 23 AMC', 680.0, 'Buy',
     ['Meta AI assistant reaches 500M users', 'Reels ad revenue growth accelerates', 'WhatsApp payments launch in 5 markets']),
    ('GOOG', '2-2-?', 'CALL', '1D', 185.00, 181.50, 194.00, 183.80, 3, '3/4 BULLISH',
     'Communication Services', 'Internet Content', 'Apr 22 AMC', 205.0, 'Buy',
     ['Google Cloud AI revenue exceeds expectations', 'Gemini 2 benchmark results impress', 'YouTube Premium subs hit 100M']),
    ('AMZN', '3-1-2U', 'CALL', '1D', 215.00, 210.50, 226.00, 213.40, 4, '4/4 BULLISH',
     'Consumer Cyclical', 'Internet Retail', 'Apr 24 AMC', 240.0, 'Strong Buy',
     ['AWS revenue growth reaccelerates', 'Amazon logistics cost per unit drops 8%', 'Prime membership hits 250M globally']),
    ('JPM', '2-1-2U', 'CALL', '1H', 258.00, 254.50, 266.00, 256.90, 3, '3/4 BULLISH',
     'Financial Services', 'Banks - Diversified', 'Apr 11 BMO', 275.0, 'Buy',
     ['JPMorgan trading revenue beats estimates', 'Net interest income guidance raised', 'Consumer credit quality remains strong']),
    ('GS', '2-1-2D', 'PUT', '1D', 595.00, 605.00, 575.00, 598.20, 3, '3/4 BEARISH',
     'Financial Services', 'Capital Markets', 'Apr 14 BMO', 580.0, 'Hold',
     ['Goldman Sachs M&A pipeline softens', 'Asset management flows slow', 'Trading revenue below peer average']),
    ('XOM', '2-2-?', 'PUT', '1H', 108.50, 111.00, 104.00, 109.80, 2, '2/4 BEARISH',
     'Energy', 'Oil & Gas Integrated', 'Apr 25 BMO', 115.0, 'Hold',
     ['Oil prices drop on demand concerns', 'Exxon Guyana output expansion delayed', 'Refining margins compress in Q1']),
    ('CVX', '2-1-2D', 'PUT', '1D', 155.00, 158.50, 148.00, 156.20, 3, '3/4 BEARISH',
     'Energy', 'Oil & Gas Integrated', 'Apr 25 BMO', 165.0, 'Hold',
     ['Chevron Hess acquisition faces FTC review', 'Permian basin output plateaus', 'Natural gas prices remain depressed']),
    ('UNH', '3-1-2U', 'CALL', '1D', 535.00, 525.00, 560.00, 531.80, 4, '4/4 BULLISH',
     'Healthcare', 'Healthcare Plans', 'Apr 15 BMO', 590.0, 'Strong Buy',
     ['UnitedHealth Medicare enrollment beats estimates', 'Optum Health margins expand', 'Drug cost management initiatives deliver']),
    ('LLY', '2-1-2U', 'CALL', '1H', 820.00, 805.00, 855.00, 815.50, 3, '3/4 BULLISH',
     'Healthcare', 'Drug Manufacturers', 'Apr 30 BMO', 900.0, 'Buy',
     ['Lilly GLP-1 supply constraints easing', 'Mounjaro market share gains continue', 'Alzheimer drug trial data expected Q2']),
    ('COST', '2-1-2U', 'CALL', '1D', 960.00, 948.00, 985.00, 957.30, 3, '3/4 BULLISH',
     'Consumer Defensive', 'Discount Stores', 'Mar 06 AMC', 1000.0, 'Buy',
     ['Costco same-store sales beat by 200bps', 'E-commerce growth accelerates to 25%', 'Membership fee increase drives margin']),
    ('WMT', '2-2-?', 'CALL', '1H', 92.00, 90.20, 95.50, 91.40, 2, '2/4 BULLISH',
     'Consumer Defensive', 'Discount Stores', 'Feb 20 BMO', 98.0, 'Buy',
     ['Walmart grocery market share hits record', 'Walmart+ membership growth strong', 'Ad revenue grows 30% YoY']),
    ('CAT', '2-1-2D', 'PUT', '1D', 365.00, 372.00, 350.00, 368.50, 3, '3/4 BEARISH',
     'Industrials', 'Farm & Heavy Construction', 'Apr 24 BMO', 375.0, 'Hold',
     ['Caterpillar backlog declines sequentially', 'Construction spending cools in US', 'Mining equipment orders slow globally']),
    ('GE', '3-1-2U', 'CALL', '1H', 195.00, 190.50, 206.00, 193.80, 4, '4/4 BULLISH',
     'Industrials', 'Specialty Industrial Machinery', 'Apr 22 BMO', 220.0, 'Strong Buy',
     ['GE Aerospace engine deliveries accelerate', 'Defense orders surge on NATO spending', 'Service revenue margins expand to 28%']),
    ('XLF', '2U-2U-?', 'PUT', '1H', 51.92, 52.15, 51.59, 51.69, 4, '4/4 BEARISH',
     'Financial Services', 'ETF', '', None, '',
     ['Financial sector rotation continues', 'Yield curve impact on bank earnings', 'Regional bank concerns resurface']),
    ('ABBV', '2-1-2U', 'CALL', '1D', 195.00, 190.00, 206.00, 193.20, 3, '3/4 BULLISH',
     'Healthcare', 'Drug Manufacturers', 'Apr 25 BMO', 210.0, 'Buy',
     ['AbbVie immunology franchise grows 15%', 'Skyrizi/Rinvoq offset Humira erosion', 'Oncology pipeline shows promise']),
]


def _build_candidate_entry(row, rank):
    """Build a single candidate dict matching pipeline._build_output schema."""
    (symbol, pattern, direction, timeframe, entry, stop, target, current,
     tfc_score, tfc_alignment, sector, industry,
     earnings, target_price, recommendation, headlines) = row

    base_pattern = pattern.rstrip('UD').rstrip('-') if pattern[-1] in 'UD' else pattern
    tfc_direction = 'bullish' if direction == 'CALL' else 'bearish'
    distance_pct = round(abs(entry - current) / current * 100, 2) if current else 0
    risk_reward = round(abs(target - entry) / abs(entry - stop), 2) if abs(entry - stop) > 0 else 0
    atr_pct = round(abs(entry - stop) / current * 100, 2) if current else 2.5

    return {
        'symbol': symbol,
        'composite_score': round(75 - (rank - 1) * 2.5, 1),
        'rank': rank,
        'pattern': {
            'type': pattern,
            'base_type': base_pattern,
            'signal_type': 'SETUP',
            'direction': direction,
            'timeframe': timeframe,
            'is_bidirectional': '?' in pattern,
        },
        'levels': {
            'entry_trigger': entry,
            'stop_price': stop,
            'target_price': target,
            'current_price': current,
            'distance_to_trigger_pct': distance_pct,
        },
        'tfc': {
            'score': tfc_score,
            'alignment': tfc_alignment,
            'direction': tfc_direction,
            'passes_flexible': True,
            'risk_multiplier': 1.0 if tfc_score >= 4 else 0.5,
            'priority_rank': 1 if tfc_score >= 4 else 2,
        },
        'metrics': {
            'atr_percent': atr_pct,
            'dollar_volume': 50_000_000.0,
            'risk_reward': risk_reward,
        },
        'scoring_breakdown': {
            'tfc_component': min(40.0, tfc_score * 10.0),
            'pattern_component': 15.0 if '3-1-2' in pattern else 7.5,
            'proximity_component': max(0, 20.0 - distance_pct * 4),
            'atr_component': 15.0 if 2.0 <= atr_pct <= 5.0 else 7.5,
        },
        'finviz': {
            'sector': sector,
            'industry': industry,
            'earnings_date': earnings,
            'analyst_recommendation': recommendation,
            'target_price': target_price,
            'news_headlines': headlines[:3],
        },
    }


def _live_enrich(candidates):
    """Replace synthetic Finviz data with real scrapes."""
    try:
        from strat.ticker_selection.enrichment import FinvizEnricher
        enricher = FinvizEnricher()
        symbols = [c['symbol'] for c in candidates]
        enrichment_map = enricher.enrich_candidates(symbols)
        for c in candidates:
            sym = c['symbol']
            if sym in enrichment_map and not enrichment_map[sym].fetch_error:
                c['finviz'] = enrichment_map[sym].to_dict()
        print(f"[live-enrich] Enriched {len(enrichment_map)} symbols from Finviz")
    except Exception as e:
        print(f"[live-enrich] Failed, keeping synthetic data: {e}")


def _format_discord_preview(candidates):
    """Render Discord summary to stdout."""
    lines = [
        "**Ticker Selection Pipeline Complete** _(SYNTHETIC)_",
        f"Universe: 11,478 | Screened: 500 | Patterns: 83 | "
        f"Final: {len(candidates)}",
        "Duration: 0.1s (synthetic)",
        "",
        "**Top Candidates:**",
    ]
    for candidate in candidates[:8]:
        pat = candidate['pattern']
        direction_label = '[CALL]' if pat['direction'] == 'CALL' else '[PUT]'
        lines.append(
            f"{direction_label} **{candidate['symbol']}** {pat['type']} "
            f"{pat['timeframe']} "
            f"| Score: {candidate['composite_score']} "
            f"| TFC: {candidate['tfc']['alignment']} "
            f"| ATR: {candidate['metrics']['atr_percent']}%"
        )
        finviz = candidate.get('finviz', {})
        parts = []
        if finviz.get('sector'):
            parts.append(finviz['sector'])
        if finviz.get('earnings_date'):
            parts.append(f"Earn: {finviz['earnings_date']}")
        if finviz.get('target_price') is not None:
            parts.append(f"Tgt: ${finviz['target_price']:.0f}")
        if finviz.get('analyst_recommendation'):
            parts.append(f"Rec: {finviz['analyst_recommendation']}")
        if parts:
            lines.append(f"  _({' | '.join(parts)})_")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic candidates.json for daemon testing'
    )
    parser.add_argument(
        '--tickers', nargs='+', default=None,
        help='Specific tickers to include (from built-in pool)',
    )
    parser.add_argument(
        '--count', type=int, default=12,
        help='Number of candidates to generate (default: 12)',
    )
    parser.add_argument(
        '--live-enrich', action='store_true',
        help='Fetch real Finviz data instead of synthetic',
    )
    parser.add_argument(
        '--preview', action='store_true',
        help='Print Discord summary to stdout, do not write file',
    )
    parser.add_argument(
        '--write-live', action='store_true',
        help='Write to the REAL candidates.json path (data/candidates/candidates.json)',
    )
    args = parser.parse_args()

    # Select candidates from pool
    if args.tickers:
        pool_map = {row[0]: row for row in CANDIDATE_POOL}
        selected = []
        for ticker in args.tickers:
            ticker = ticker.upper()
            if ticker in pool_map:
                selected.append(pool_map[ticker])
            else:
                print(f"Warning: {ticker} not in pool, skipping "
                      f"(available: {', '.join(pool_map.keys())})")
        if not selected:
            print("No valid tickers. Exiting.")
            sys.exit(1)
    else:
        selected = CANDIDATE_POOL[:args.count]

    # Build candidate entries
    candidates = []
    for i, row in enumerate(selected):
        candidates.append(_build_candidate_entry(row, rank=i + 1))

    # Live enrichment (replaces synthetic Finviz data)
    if args.live_enrich:
        _live_enrich(candidates)

    # Build full output
    now = datetime.now(timezone.utc).isoformat()
    output = {
        'version': '1.0',
        'generated_at': now,
        'pipeline_stats': {
            'universe_size': 11478,
            'screened_size': 500,
            'patterns_found': 83,
            'tfc_qualified': 83,
            'final_candidates': len(candidates),
            'scan_duration_seconds': 0.1,
            'synthetic': True,   # <-- daemon rejects this
        },
        'core_symbols': ['SPY', 'QQQ', 'IWM', 'DIA'],
        'candidates': candidates,
    }

    if args.preview:
        print(_format_discord_preview(candidates))
        print(f"\n--- {len(candidates)} synthetic candidates generated at {now} ---")
        return

    # Write to file
    if args.write_live:
        path = Path('data/candidates/candidates.json')
        print("WARNING: Writing to LIVE candidates path.")
        print("  The daemon will REJECT this file (synthetic: true).")
    else:
        path = Path('data/candidates/synthetic_candidates.json')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(candidates)} synthetic candidates to {path}")
    print(f"  generated_at: {now}")
    print(f"  synthetic: true (daemon will reject this file)")


if __name__ == '__main__':
    main()
