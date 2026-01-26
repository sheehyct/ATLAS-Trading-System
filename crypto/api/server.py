"""
Crypto Daemon REST API Server - Session CRYPTO-6

Minimal Flask API exposing crypto daemon data for dashboard integration.
Runs as a thread within the daemon process on port 8080.

Endpoints:
    /health      - Health check with daemon status
    /status      - Full daemon status
    /positions   - Open positions with P&L
    /signals     - Pending SETUP signals
    /performance - Account summary and performance metrics
    /trades      - Trade history (supports ?status=open|closed&limit=N)

Usage:
    from crypto.api.server import init_api, run_api

    init_api(daemon_instance)
    run_api(host='0.0.0.0', port=8080)
"""

import logging
from typing import TYPE_CHECKING, Optional

from flask import Flask, jsonify, request

if TYPE_CHECKING:
    from crypto.scanning.daemon import CryptoSignalDaemon

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global daemon reference (set via init_api)
_daemon: Optional["CryptoSignalDaemon"] = None


def init_api(daemon_instance: "CryptoSignalDaemon") -> None:
    """
    Initialize API with daemon reference.

    Args:
        daemon_instance: Running CryptoSignalDaemon instance
    """
    global _daemon
    _daemon = daemon_instance
    logger.info("API initialized with daemon reference")


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.route('/health')
def health():
    """
    Health check endpoint.

    Returns:
        JSON with status, daemon_running, and uptime
    """
    if _daemon is None:
        return jsonify({
            'status': 'error',
            'message': 'Daemon not initialized'
        }), 503

    status = _daemon.get_status()
    return jsonify({
        'status': 'ok',
        'daemon_running': status.get('running', False),
        'uptime_seconds': status.get('uptime_seconds', 0)
    })


@app.route('/status')
def get_status():
    """
    Full daemon status endpoint.

    Returns:
        Complete daemon status dictionary
    """
    if _daemon is None:
        return jsonify({'error': 'Daemon not initialized'}), 503

    return jsonify(_daemon.get_status())


@app.route('/positions')
def get_positions():
    """
    Open positions with unrealized P&L.

    Returns:
        List of position dictionaries
    """
    if _daemon is None:
        return jsonify([])

    try:
        positions = _daemon.get_open_positions()
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return jsonify([])


@app.route('/signals')
def get_signals():
    """
    Pending SETUP signals awaiting trigger.

    Returns:
        List of signal dictionaries
    """
    if _daemon is None:
        return jsonify([])

    try:
        signals = _daemon.get_pending_setups()
        # Convert to dicts if needed
        result = []
        for signal in signals:
            if hasattr(signal, 'to_dict'):
                result.append(signal.to_dict())
            elif isinstance(signal, dict):
                result.append(signal)
            else:
                # Fallback: serialize key attributes
                result.append({
                    'symbol': getattr(signal, 'symbol', ''),
                    'timeframe': getattr(signal, 'timeframe', ''),
                    'pattern_type': getattr(signal, 'pattern_type', ''),
                    'direction': getattr(signal, 'direction', ''),
                    'entry_trigger': getattr(signal, 'entry_trigger', 0),
                    'stop_price': getattr(signal, 'stop_price', 0),
                    'target_price': getattr(signal, 'target_price', 0),
                    'magnitude_pct': getattr(signal, 'magnitude_pct', 0),
                    'risk_reward': getattr(signal, 'risk_reward', 0),
                    'signal_type': getattr(signal, 'signal_type', ''),
                    'detected_time': str(getattr(signal, 'detected_time', '')),
                })
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return jsonify([])


@app.route('/performance')
def get_performance():
    """
    Account summary and performance metrics.

    Returns:
        Dictionary with account_summary and performance_metrics
    """
    if _daemon is None or _daemon.paper_trader is None:
        return jsonify({
            'account_summary': {},
            'performance_metrics': {}
        })

    try:
        return jsonify({
            'account_summary': _daemon.paper_trader.get_account_summary(),
            'performance_metrics': _daemon.paper_trader.get_performance_metrics()
        })
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        return jsonify({
            'account_summary': {},
            'performance_metrics': {}
        })


@app.route('/trades')
def get_trades():
    """
    Trade history with optional filtering.

    Query params:
        status: Filter by 'open' or 'closed'
        limit: Maximum trades to return (default 50)

    Returns:
        List of trade dictionaries
    """
    if _daemon is None or _daemon.paper_trader is None:
        return jsonify([])

    try:
        status_filter = request.args.get('status', None)
        limit = int(request.args.get('limit', 50))

        trades = _daemon.paper_trader.get_trade_history(limit=limit)

        # Filter by status if specified
        if status_filter == 'open':
            trades = [t for t in trades if t.get('status') == 'OPEN']
        elif status_filter == 'closed':
            trades = [t for t in trades if t.get('status') == 'CLOSED']

        return jsonify(trades)
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        return jsonify([])


def _transform_pnl_response(raw: dict) -> dict:
    """
    Transform PaperTrader format to documented API format.

    Session DB-2: Normalizes response format for API contract compliance.

    PaperTrader returns: {'gross': float, 'fees': float, 'funding': float, 'net': float, 'trades': int}
    API documents:       {'total_pnl': float, 'trade_count': int, 'win_rate': float}

    Args:
        raw: Raw response from PaperTrader.get_pnl_by_strategy()

    Returns:
        Normalized response matching documented API format
    """
    def transform_strategy(data: dict) -> dict:
        if not data:
            return {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0}
        return {
            'total_pnl': data.get('net', 0),
            'trade_count': data.get('trades', 0),
            'win_rate': data.get('win_rate', 0),  # Pass through if available
        }

    return {
        'strat': transform_strategy(raw.get('strat', {})),
        'statarb': transform_strategy(raw.get('statarb', {})),
        'combined': transform_strategy(raw.get('combined', {})),
    }


@app.route('/pnl_by_strategy')
def get_pnl_by_strategy():
    """
    P&L breakdown by trading strategy (EQUITY-93B).

    Returns:
        Dictionary with P&L breakdown by strategy:
        {
            'strat': {'total_pnl': float, 'trade_count': int, 'win_rate': float},
            'statarb': {'total_pnl': float, 'trade_count': int, 'win_rate': float},
            'combined': {'total_pnl': float, 'trade_count': int, 'win_rate': float}
        }
    """
    if _daemon is None or _daemon.paper_trader is None:
        return jsonify({
            'strat': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0},
            'statarb': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0},
            'combined': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0}
        })

    try:
        # Check if paper_trader has get_pnl_by_strategy method
        if hasattr(_daemon.paper_trader, 'get_pnl_by_strategy'):
            # Session DB-2: Transform to documented API format
            raw = _daemon.paper_trader.get_pnl_by_strategy()
            return jsonify(_transform_pnl_response(raw))

        # Fallback: Calculate from trade history
        trades = _daemon.paper_trader.get_trade_history(limit=500)
        closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

        def calc_strategy_stats(strategy_trades):
            if not strategy_trades:
                return {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0}
            total_pnl = sum(t.get('pnl', 0) or 0 for t in strategy_trades)
            wins = sum(1 for t in strategy_trades if (t.get('pnl') or 0) > 0)
            return {
                'total_pnl': total_pnl,
                'trade_count': len(strategy_trades),
                'win_rate': (wins / len(strategy_trades) * 100) if strategy_trades else 0
            }

        strat_trades = [t for t in closed_trades if t.get('strategy') == 'strat']
        statarb_trades = [t for t in closed_trades if t.get('strategy') == 'statarb']

        return jsonify({
            'strat': calc_strategy_stats(strat_trades),
            'statarb': calc_strategy_stats(statarb_trades),
            'combined': calc_strategy_stats(closed_trades)
        })
    except Exception as e:
        logger.error(f"Error fetching P&L by strategy: {e}")
        return jsonify({
            'strat': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0},
            'statarb': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0},
            'combined': {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0}
        })


# =============================================================================
# SERVER RUNNER
# =============================================================================


def run_api(host: str = '0.0.0.0', port: int = 8080) -> None:
    """
    Run the Flask API server.

    Args:
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to listen on (default: 8080)

    Note:
        This function blocks. Run in a thread for non-blocking operation.
    """
    # Disable Flask's default logging noise
    import logging as flask_logging
    flask_logging.getLogger('werkzeug').setLevel(flask_logging.WARNING)

    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)
