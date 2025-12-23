"""
Equity Daemon API - Session EQUITY-33

REST API for exposing equity daemon data to remote dashboards.
"""

from strat.signal_automation.api.server import app, init_api, run_api

__all__ = ['app', 'init_api', 'run_api']
