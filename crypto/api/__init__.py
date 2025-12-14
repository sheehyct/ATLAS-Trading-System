"""
Crypto Daemon REST API Module - Session CRYPTO-6

Provides REST API endpoints for exposing crypto daemon data to the dashboard.
Runs as a thread within the main daemon process.

Usage:
    from crypto.api.server import init_api, run_api

    # In daemon start():
    init_api(daemon_instance)
    api_thread = threading.Thread(target=run_api, kwargs={'port': 8080})
    api_thread.start()
"""

from crypto.api.server import app, init_api, run_api

__all__ = ['app', 'init_api', 'run_api']
