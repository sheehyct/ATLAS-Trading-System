#!/usr/bin/env python3
"""
Signal API Server - Session 83K-76

Minimal Flask API to serve STRAT signals from VPS to Railway dashboard.
Runs as a systemd service on port 5000.

Usage:
    uv run python scripts/signal_api.py
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, jsonify
from strat.signal_automation.signal_store import SignalStore

logger = logging.getLogger(__name__)

TRADE_METADATA_FILE = project_root / 'data' / 'executions' / 'trade_metadata.json'

app = Flask(__name__)
signal_store = SignalStore()


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'signal_count': len(signal_store)})


@app.route('/signals')
def get_all_signals():
    """Get all signals as JSON."""
    signals = signal_store.load_signals()
    return jsonify({
        key: signal.to_dict()
        for key, signal in signals.items()
    })


@app.route('/signals/pending')
def get_pending_signals():
    """Get SETUP signals awaiting entry trigger."""
    signals = signal_store.get_setup_signals_for_monitoring()
    return jsonify([s.to_dict() for s in signals])


@app.route('/signals/active')
def get_active_signals():
    """Get all non-expired signals."""
    all_signals = signal_store.load_signals()
    active = [
        s for s in all_signals.values()
        if s.status not in ('EXPIRED', 'CONVERTED')
    ]
    return jsonify([s.to_dict() for s in active])


@app.route('/signals/triggered')
def get_triggered_signals():
    """Get signals that have been triggered."""
    all_signals = signal_store.load_signals()
    triggered = [
        s for s in all_signals.values()
        if s.status in ('TRIGGERED', 'HISTORICAL_TRIGGERED')
    ]
    return jsonify([s.to_dict() for s in triggered])


@app.route('/signals/stats')
def get_stats():
    """Get signal store statistics."""
    return jsonify(signal_store.get_stats())


def _load_trade_metadata_file() -> dict:
    """Load trade_metadata.json from disk, returning empty dict on failure."""
    if not TRADE_METADATA_FILE.exists():
        return {}
    try:
        with open(TRADE_METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading trade_metadata.json: {e}")
        return {}


def _merge_signal_store_tfc(metadata: dict) -> None:
    """Overlay signal store TFC data onto metadata dict (mutates in place)."""
    try:
        all_signals = signal_store.load_signals()
    except Exception as e:
        logger.error(f"Error loading signal store: {e}")
        return

    for signal in all_signals.values():
        if not signal.executed_osi_symbol:
            continue

        osi = signal.executed_osi_symbol
        signal_data = {
            'pattern_type': signal.pattern_type,
            'timeframe': signal.timeframe,
            'tfc_score': signal.tfc_score,
            'tfc_alignment': signal.tfc_alignment,
            'direction': signal.direction,
            'symbol': signal.symbol,
            'entry_trigger': signal.entry_trigger,
            'stop_price': signal.stop_price,
            'target_price': signal.target_price,
        }

        if osi not in metadata:
            metadata[osi] = signal_data
        elif signal.tfc_score and not metadata[osi].get('tfc_score'):
            metadata[osi]['tfc_score'] = signal.tfc_score
            metadata[osi]['tfc_alignment'] = signal.tfc_alignment


@app.route('/trade_metadata')
def get_trade_metadata():
    """
    EQUITY-103: Trade metadata for closed trade pattern/TFC correlation.

    Merges two data sources:
    1. trade_metadata.json (written by executor at order time + backfill script)
    2. signal_store signals with executed_osi_symbol (for TFC writeback data)

    Returns:
        Dict mapping OSI symbol -> {pattern_type, timeframe, tfc_score, tfc_alignment, ...}
    """
    metadata = _load_trade_metadata_file()
    _merge_signal_store_tfc(metadata)
    return jsonify(metadata)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
