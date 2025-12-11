#!/usr/bin/env python3
"""
Signal API Server - Session 83K-76

Minimal Flask API to serve STRAT signals from VPS to Railway dashboard.
Runs as a systemd service on port 5000.

Usage:
    uv run python scripts/signal_api.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, jsonify
from strat.signal_automation.signal_store import SignalStore

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
