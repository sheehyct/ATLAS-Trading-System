"""
Centralized Configuration Loading for ATLAS Trading System.

Session 70: Created to fix recurring .env loading issues.
- Single source of truth for all environment variables
- Validates required credentials at startup
- Prevents Alpaca 401 errors from incorrect .env paths

Usage:
    from config.settings import load_config, get_alpaca_credentials, get_tiingo_key

    # At app startup (call once)
    load_config()

    # Get credentials as needed
    creds = get_alpaca_credentials('MID')
    tiingo_key = get_tiingo_key()
"""

import os
from pathlib import Path
from typing import Dict, Optional

# Flag to track if config has been loaded
_CONFIG_LOADED = False


def load_config(force_reload: bool = False) -> None:
    """
    Load environment variables from project root .env file.

    This function should be called once at application startup.
    It loads from the ROOT .env file (not config/.env) to ensure
    all credentials including TIINGO_API_KEY are available.

    Args:
        force_reload: If True, reload even if already loaded

    Raises:
        FileNotFoundError: If .env file doesn't exist
        ValueError: If required environment variables are missing
    """
    global _CONFIG_LOADED

    if _CONFIG_LOADED and not force_reload:
        return

    # Import here to avoid circular imports
    from dotenv import load_dotenv

    # Always load from project root .env (the complete file)
    # This file contains ALL credentials including TIINGO_API_KEY
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'

    # Railway and other cloud platforms set env vars directly
    # Only load .env file if it exists (local development)
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        # Check if we're in a cloud environment with env vars already set
        if not os.getenv('RAILWAY_ENVIRONMENT') and not os.getenv('TIINGO_API_KEY'):
            raise FileNotFoundError(
                f"Required .env file not found at {env_path}. "
                "Copy .env.example to .env and fill in credentials."
            )
        # Cloud environment - env vars already available, no .env needed

    _CONFIG_LOADED = True

    # Validate critical environment variables (graceful for dashboards)
    _validate_required_vars()


def _validate_required_vars(strict: bool = False) -> None:
    """
    Validate that all required environment variables are set.

    Args:
        strict: If True, raise ValueError on missing vars. If False, warn only.

    Raises:
        ValueError: If strict=True and any required variable is missing
    """
    import warnings

    required_vars = [
        'TIINGO_API_KEY',      # Required for historical data
        'ALPACA_MID_KEY',      # Primary trading account
        'ALPACA_MID_SECRET',
    ]

    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.strip() == '':
            missing.append(var)

    if missing:
        msg = f"Missing environment variables: {missing}. Some features may be unavailable."
        if strict:
            raise ValueError(msg + " Check your .env file at project root.")
        else:
            warnings.warn(msg, UserWarning)


def get_alpaca_credentials(account: str = 'MID') -> Dict[str, str]:
    """
    Get Alpaca credentials for specified account.

    Args:
        account: Account type - 'SMALL', 'MID', or 'LARGE'

    Returns:
        Dict with api_key, secret_key, and base_url

    Raises:
        ValueError: If account type is invalid

    Note:
        For Railway/cloud deployments, you can use simplified env vars:
        - ALPACA_API_KEY and ALPACA_SECRET_KEY (used as fallback for any account)
        For local development with multiple accounts, use:
        - ALPACA_MID_KEY, ALPACA_MID_SECRET
        - ALPACA_LARGE_KEY, ALPACA_LARGE_SECRET
    """
    # Ensure config is loaded
    load_config()

    account = account.upper()

    # Fallback keys for simplified Railway deployments
    fallback_key = os.getenv('ALPACA_API_KEY')
    fallback_secret = os.getenv('ALPACA_SECRET_KEY')

    if account == 'SMALL':
        return {
            'api_key': os.getenv('ALPACA_API_KEY') or fallback_key,
            'secret_key': os.getenv('ALPACA_SECRET_KEY') or fallback_secret,
            'base_url': os.getenv('ALPACA_ENDPOINT', 'https://paper-api.alpaca.markets')
        }
    elif account == 'MID':
        return {
            'api_key': os.getenv('ALPACA_MID_KEY') or fallback_key,
            'secret_key': os.getenv('ALPACA_MID_SECRET') or fallback_secret,
            'base_url': os.getenv('ALPACA_MID_ENDPOINT', 'https://paper-api.alpaca.markets')
        }
    elif account == 'LARGE':
        return {
            'api_key': os.getenv('ALPACA_LARGE_KEY') or fallback_key,
            'secret_key': os.getenv('ALPACA_LARGE_SECRET') or fallback_secret,
            'base_url': os.getenv('ALPACA_LARGE_ENDPOINT', 'https://paper-api.alpaca.markets')
        }
    else:
        raise ValueError(f"Invalid account type: {account}. Use 'SMALL', 'MID', or 'LARGE'.")


def get_tiingo_key() -> str:
    """
    Get Tiingo API key.

    Returns:
        Tiingo API key string

    Raises:
        ValueError: If TIINGO_API_KEY is not set
    """
    load_config()
    key = os.getenv('TIINGO_API_KEY')
    if not key:
        raise ValueError("TIINGO_API_KEY not set in environment")
    return key


def get_alphavantage_key() -> Optional[str]:
    """Get AlphaVantage API key (optional)."""
    load_config()
    return os.getenv('ALPHAVANTAGE_API_KEY')


def get_openai_key() -> Optional[str]:
    """Get OpenAI API key (optional, for Whisper transcription)."""
    load_config()
    return os.getenv('OPENAI_API_KEY')


def get_github_token() -> Optional[str]:
    """Get GitHub token for VectorBT Pro documentation access."""
    load_config()
    return os.getenv('GITHUB_TOKEN') or os.getenv('VECTORBT_TOKEN')


def get_default_account() -> str:
    """Get the default account to use."""
    load_config()
    return os.getenv('DEFAULT_ACCOUNT', 'MID')


def get_thetadata_config() -> Dict[str, any]:
    """
    Get ThetaData terminal configuration.

    ThetaData uses local terminal architecture - REST calls go to localhost.
    The terminal handles authentication via its own creds.txt file.
    No API key needed in REST calls.

    Returns:
        Dict with host, port, timeout, and enabled flag

    Example:
        config = get_thetadata_config()
        base_url = f"http://{config['host']}:{config['port']}"
    """
    load_config()
    return {
        'host': os.getenv('THETADATA_HOST', '127.0.0.1'),
        'port': int(os.getenv('THETADATA_PORT', '25510')),
        'timeout': int(os.getenv('THETADATA_TIMEOUT', '30')),
        'enabled': os.getenv('THETADATA_ENABLED', 'false').lower() == 'true'
    }


def is_thetadata_available() -> bool:
    """
    Check if ThetaData integration is enabled.

    Returns:
        True if THETADATA_ENABLED=true in environment
    """
    load_config()
    return os.getenv('THETADATA_ENABLED', 'false').lower() == 'true'


def is_config_loaded() -> bool:
    """Check if configuration has been loaded."""
    return _CONFIG_LOADED


# Convenience function for VBT Pro AlpacaData
def get_vbt_alpaca_config(account: str = 'MID') -> Dict[str, str]:
    """
    Get Alpaca config in VBT Pro format.

    Returns:
        Dict ready for vbt.AlpacaData.fetch(client_config=...)
    """
    creds = get_alpaca_credentials(account)
    return {
        'api_key': creds['api_key'],
        'secret_key': creds['secret_key'],
    }
