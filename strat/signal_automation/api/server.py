"""
Equity Daemon REST API Server - Session EQUITY-33

Minimal Flask API exposing equity daemon data for dashboard integration.
Runs as a thread within the daemon process on port 8081.

Endpoints:
    /health              - Health check with daemon status
    /status              - Full daemon status
    /positions           - Open option positions with P&L
    /positions_with_signals - Positions linked to their STRAT signals (for trade progress chart)
    /signals             - Pending SETUP signals
    /signals/triggered   - Triggered signals awaiting execution
    /signals/by_category - Signals grouped by category (setups, triggered, low_magnitude)

Usage:
    from strat.signal_automation.api.server import init_api, run_api

    init_api(daemon_instance)
    run_api(host='0.0.0.0', port=8081)
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, List, Any

from flask import Flask, jsonify, request

if TYPE_CHECKING:
    from strat.signal_automation.daemon import SignalDaemon

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global daemon reference (set via init_api)
_daemon: Optional["SignalDaemon"] = None


def init_api(daemon_instance: "SignalDaemon") -> None:
    """
    Initialize API with daemon reference.

    Args:
        daemon_instance: Running SignalDaemon instance
    """
    global _daemon
    _daemon = daemon_instance
    logger.info("Equity API initialized with daemon reference")


def run_api(host: str = '0.0.0.0', port: int = 8081, debug: bool = False) -> None:
    """
    Run the Flask API server.

    Args:
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to bind to (default: 8081)
        debug: Enable Flask debug mode
    """
    logger.info(f"Starting Equity API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _signal_to_dict(signal) -> Dict[str, Any]:
    """Convert a StoredSignal to a JSON-serializable dictionary."""
    return {
        'signal_key': getattr(signal, 'signal_key', ''),
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
        'status': getattr(signal, 'status', ''),
        'detected_time': str(getattr(signal, 'detected_time', '')),
        'executed_osi_symbol': getattr(signal, 'executed_osi_symbol', None),
        'tfc_score': getattr(signal, 'tfc_score', 0),
        'tfc_alignment': getattr(signal, 'tfc_alignment', ''),
    }


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

    return jsonify({
        'status': 'ok',
        'daemon_running': _daemon._is_running,
        'uptime_seconds': (
            (_daemon._start_time and
             ((__import__('datetime').datetime.now() - _daemon._start_time).total_seconds()))
            or 0
        )
    })


@app.route('/status')
def get_status():
    """
    Full daemon status endpoint.

    Returns:
        Dictionary with daemon status information
    """
    if _daemon is None:
        return jsonify({'error': 'Daemon not initialized'}), 503

    return jsonify({
        'running': _daemon._is_running,
        'scan_count': _daemon._scan_count,
        'signal_count': _daemon._signal_count,
        'execution_count': _daemon._execution_count,
        'error_count': _daemon._error_count,
        'start_time': str(_daemon._start_time) if _daemon._start_time else None,
    })


@app.route('/positions')
def get_positions():
    """
    Open option positions with unrealized P&L and signal linkage.

    Session EQUITY-34: Now includes pattern and timeframe from originating signal.

    Returns:
        List of position dictionaries from Alpaca with pattern/timeframe/entry_time
    """
    if _daemon is None or _daemon.executor is None:
        return jsonify([])

    try:
        positions = _daemon.executor.get_positions()

        # Session EQUITY-34: Enhance with signal data
        for pos in positions:
            pos['pattern'] = '-'
            pos['timeframe'] = '-'
            pos['entry_time_et'] = None

            osi_symbol = pos.get('symbol', '')
            if osi_symbol and _daemon.signal_store:
                try:
                    signal = _daemon.signal_store.get_signal_by_osi_symbol(osi_symbol)
                    if signal:
                        pos['pattern'] = signal.pattern_type
                        pos['timeframe'] = signal.timeframe
                        # Format entry time in ET
                        if signal.detected_time:
                            try:
                                import pytz
                                et = pytz.timezone('America/New_York')
                                dt = signal.detected_time
                                if dt.tzinfo is None:
                                    dt = pytz.utc.localize(dt)
                                dt_et = dt.astimezone(et)
                                pos['entry_time_et'] = dt_et.strftime('%m/%d %H:%M')
                            except Exception:
                                pos['entry_time_et'] = str(signal.detected_time)[:16]
                except Exception as e:
                    logger.debug(f"Could not look up signal for {osi_symbol}: {e}")

        return jsonify(positions)
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return jsonify([])


@app.route('/positions_with_signals')
def get_positions_with_signals():
    """
    Positions linked to their originating STRAT signals.

    Session EQUITY-33: Enables the Trade Progress to Target chart on Railway.

    Returns:
        List of trade dictionaries with:
        - name: Display name (e.g., "SPY 3-1-2U 1H")
        - entry: Entry trigger price (underlying stock)
        - current: Current underlying stock price
        - target: Target price (underlying stock)
        - stop: Stop price (underlying stock)
        - direction: CALL or PUT
        - pnl_pct: Current P&L percentage
        - osi_symbol: OCC option symbol
    """
    if _daemon is None or _daemon.executor is None:
        return jsonify([])

    try:
        # Get open positions from Alpaca
        positions = _daemon.executor.get_positions()
        if not positions:
            return jsonify([])

        # Get underlying prices for all positions
        underlying_symbols = set()
        position_signals = []

        for pos in positions:
            osi_symbol = pos.get('symbol', '')

            # Look up the signal for this position
            signal = _daemon.signal_store.get_signal_by_osi_symbol(osi_symbol)

            if signal:
                underlying_symbols.add(signal.symbol)
                position_signals.append((pos, signal))

        # Fetch current underlying prices
        underlying_prices = {}
        if underlying_symbols:
            try:
                quotes = _daemon.executor._trading_client.get_stock_quotes(
                    list(underlying_symbols)
                )
                for symbol, quote in quotes.items():
                    if isinstance(quote, dict):
                        underlying_prices[symbol] = quote.get('mid', quote.get('last', 0))
                    elif isinstance(quote, (int, float)):
                        underlying_prices[symbol] = float(quote)
            except Exception as e:
                logger.debug(f"Could not fetch underlying prices: {e}")

        # Build trade data
        trades = []
        for pos, signal in position_signals:
            current_underlying = underlying_prices.get(
                signal.symbol, signal.entry_trigger
            )

            trade = {
                'name': f"{signal.symbol} {signal.pattern_type} {signal.timeframe}",
                'entry': signal.entry_trigger,
                'current': current_underlying,
                'target': signal.target_price,
                'stop': signal.stop_price,
                'direction': signal.direction,
                'pnl_pct': float(pos.get('unrealized_plpc', 0)),
                'osi_symbol': pos.get('symbol', ''),
            }
            trades.append(trade)

        return jsonify(trades)

    except Exception as e:
        logger.error(f"Error fetching positions with signals: {e}")
        return jsonify([])


@app.route('/signals')
def get_signals():
    """
    All signals in the signal store.

    Query params:
        status: Filter by status (DETECTED, ALERTED, TRIGGERED, etc.)
        symbol: Filter by symbol
        timeframe: Filter by timeframe

    Returns:
        List of signal dictionaries
    """
    if _daemon is None:
        return jsonify([])

    try:
        # Get filter params
        status_filter = request.args.get('status', None)
        symbol_filter = request.args.get('symbol', None)
        timeframe_filter = request.args.get('timeframe', None)

        # Get signals based on filters
        if symbol_filter:
            signals = _daemon.signal_store.get_signals_by_symbol(symbol_filter)
        elif timeframe_filter:
            signals = _daemon.signal_store.get_signals_by_timeframe(timeframe_filter)
        else:
            # Get recent signals (last 7 days by default)
            signals = _daemon.signal_store.get_recent_signals(days=7)

        # Apply status filter if specified
        if status_filter:
            signals = [s for s in signals if s.status == status_filter]

        return jsonify([_signal_to_dict(s) for s in signals])

    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return jsonify([])


@app.route('/signals/triggered')
def get_triggered_signals():
    """
    Signals that have been triggered (entry price hit).

    Returns:
        List of triggered signal dictionaries
    """
    if _daemon is None:
        return jsonify([])

    try:
        signals = _daemon.signal_store.get_historical_triggered_signals()
        return jsonify([_signal_to_dict(s) for s in signals])
    except Exception as e:
        logger.error(f"Error fetching triggered signals: {e}")
        return jsonify([])


@app.route('/signals/setups')
def get_setup_signals():
    """
    SETUP signals awaiting entry trigger break.

    Returns:
        List of setup signal dictionaries
    """
    if _daemon is None:
        return jsonify([])

    try:
        signals = _daemon.signal_store.get_setup_signals_for_monitoring()
        return jsonify([_signal_to_dict(s) for s in signals])
    except Exception as e:
        logger.error(f"Error fetching setup signals: {e}")
        return jsonify([])


@app.route('/signals/pending')
def get_pending_signals():
    """
    Alias for /signals/setups - pending SETUP signals awaiting entry trigger.

    Dashboard Overhaul: Added to match options_loader._fetch_from_api() calls.

    Returns:
        List of pending signal dictionaries with TFC data
    """
    return get_setup_signals()


@app.route('/signals/closed')
def get_closed_signals():
    """
    Signals that have been converted to trades (closed trades).

    Dashboard Overhaul: Added for closed trades with TFC data on Railway.

    Query params:
        days: Number of days to look back (default: 30)

    Returns:
        List of closed signal dictionaries with TFC fields
    """
    if _daemon is None:
        return jsonify([])

    try:
        days = int(request.args.get('days', 30))

        # Get signals that have been executed (have OSI symbol)
        all_signals = _daemon.signal_store.get_recent_signals(days=days)
        closed = [
            s for s in all_signals
            if s.executed_osi_symbol and s.status in ('CONVERTED', 'EXECUTED')
        ]

        return jsonify([_signal_to_dict(s) for s in closed])

    except Exception as e:
        logger.error(f"Error fetching closed signals: {e}")
        return jsonify([])


@app.route('/signals/by_category')
def get_signals_by_category():
    """
    Signals grouped by category for dashboard tabs.

    Returns:
        Dictionary with keys: setups, triggered, low_magnitude
    """
    if _daemon is None:
        return jsonify({
            'setups': [],
            'triggered': [],
            'low_magnitude': []
        })

    try:
        # Get all recent signals
        all_signals = _daemon.signal_store.get_recent_signals(days=7)

        # Categorize
        setups = []
        triggered = []
        low_magnitude = []

        min_magnitude = 0.5  # Default threshold

        for signal in all_signals:
            if signal.signal_type == 'SETUP':
                if signal.magnitude_pct < min_magnitude:
                    low_magnitude.append(_signal_to_dict(signal))
                else:
                    setups.append(_signal_to_dict(signal))
            elif signal.status in ('TRIGGERED', 'HISTORICAL_TRIGGERED'):
                triggered.append(_signal_to_dict(signal))

        return jsonify({
            'setups': setups,
            'triggered': triggered,
            'low_magnitude': low_magnitude
        })

    except Exception as e:
        logger.error(f"Error categorizing signals: {e}")
        return jsonify({
            'setups': [],
            'triggered': [],
            'low_magnitude': []
        })


@app.route('/trade_metadata')
def get_trade_metadata():
    """
    EQUITY-102: Trade metadata for closed trade pattern/TFC correlation.

    Merges two data sources:
    1. trade_metadata.json (written by executor at order time)
    2. signal_store signals with executed_osi_symbol (for TFC writeback data)

    Returns:
        Dict mapping OSI symbol -> {pattern_type, timeframe, tfc_score, tfc_alignment, ...}
    """
    metadata = {}

    # Source 1: Load trade_metadata.json from disk
    try:
        metadata_file = Path('data/executions/trade_metadata.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.debug(f"Loaded {len(metadata)} entries from trade_metadata.json")
    except Exception as e:
        logger.error(f"Error loading trade_metadata.json: {e}")

    # Source 2: Merge signal_store data (has updated TFC from EQUITY-102 writeback)
    if _daemon is not None and _daemon.signal_store is not None:
        try:
            all_signals = _daemon.signal_store.load_signals()
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
                    'detected_time': str(signal.detected_time) if signal.detected_time else None,
                }

                if osi not in metadata:
                    metadata[osi] = signal_data
                elif signal.tfc_score and not metadata[osi].get('tfc_score'):
                    # Signal store TFC takes priority over missing/zero file data
                    metadata[osi]['tfc_score'] = signal.tfc_score
                    metadata[osi]['tfc_alignment'] = signal.tfc_alignment

            logger.debug(f"Merged signal_store data, total entries: {len(metadata)}")
        except Exception as e:
            logger.error(f"Error merging signal_store data: {e}")

    return jsonify(metadata)
