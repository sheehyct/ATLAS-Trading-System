"""
Config package for ATLAS Trading System.

Provides centralized configuration loading from root .env file.
"""

from config.settings import (
    load_config,
    get_alpaca_credentials,
    get_tiingo_key,
    get_alphavantage_key,
    get_openai_key,
    get_github_token,
    get_default_account,
    get_vbt_alpaca_config,
    is_config_loaded,
    get_thetadata_config,
    is_thetadata_available,
)

__all__ = [
    'load_config',
    'get_alpaca_credentials',
    'get_tiingo_key',
    'get_alphavantage_key',
    'get_openai_key',
    'get_github_token',
    'get_default_account',
    'get_vbt_alpaca_config',
    'is_config_loaded',
    'get_thetadata_config',
    'is_thetadata_available',
]
