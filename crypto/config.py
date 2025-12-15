"""
Configuration for crypto derivatives trading module.

Defines symbols, risk parameters, timeframes, and trading rules.
Supports perpetual futures with funding rates and leverage tiers.
"""

from typing import Dict, List

# =============================================================================
# EXCHANGE CONFIGURATION
# =============================================================================

# Primary exchange for derivatives
# Options: "coinbase_spot" (spot with simulated leverage),
#          "coinbase_intx" (international derivatives),
#          "hyperliquid" (decentralized perps)
PRIMARY_EXCHANGE: str = "coinbase_spot"  # Start with spot, upgrade later

# =============================================================================
# SYMBOLS
# =============================================================================

# Active trading symbols
# Format depends on exchange:
#   - coinbase_spot: "BTC-USD", "ETH-USD", "SOL-USD"
#   - coinbase_intx: "BTC-PERP-INTX", "ETH-PERP-INTX"
#   - hyperliquid: "BTC", "ETH", "SOL"
CRYPTO_SYMBOLS: List[str] = [
    "BTC-PERP-INTX",  # Bitcoin Perpetual (primary)
    "ETH-PERP-INTX",  # Ethereum Perpetual
    # "SOL-PERP-INTX",  # Add if available
]

# Spot symbols (for data fallback)
CRYPTO_SPOT_SYMBOLS: List[str] = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
]

# =============================================================================
# LEVERAGE CONFIGURATION
# =============================================================================

# Leverage tiers based on holding period
LEVERAGE_TIERS: Dict[str, Dict[str, float]] = {
    "intraday": {
        "BTC": 10.0,   # Must close before 4PM ET
        "ETH": 10.0,
        "SOL": 10.0,
    },
    "swing": {
        "BTC": 4.0,    # Holding through funding periods, 24/7
        "ETH": 4.0,
        "SOL": 3.0,    # More volatile
    },
}

# Default leverage tier for position sizing
DEFAULT_LEVERAGE_TIER: str = "swing"  # Conservative default

# =============================================================================
# INTRADAY LEVERAGE WINDOW (Coinbase INTX)
# =============================================================================
# Coinbase offers up to 10x leverage for intraday positions.
# Intraday window: 6PM ET to 4PM ET next day (22 hours)
# Unavailable window: 4PM ET to 6PM ET (2 hours) - swing leverage only
#
# Positions must be closed before 4PM ET to use intraday leverage.
# If holding past 4PM ET, margin requirements increase to swing tier.

# Intraday window start time (6PM ET = 18:00)
INTRADAY_WINDOW_START_HOUR_ET: int = 18  # 6PM ET

# Intraday window end time (4PM ET = 16:00 next day)
INTRADAY_WINDOW_END_HOUR_ET: int = 16  # 4PM ET

# Enable automatic tier switching based on time of day
ENABLE_TIME_BASED_LEVERAGE: bool = True

# Symbol-specific max leverage (legacy, use LEVERAGE_TIERS)
SYMBOL_MAX_LEVERAGE: Dict[str, float] = {
    "BTC-USD": 4.0,   # Default to swing tier
    "ETH-USD": 4.0,
    "SOL-USD": 3.0,
}

# =============================================================================
# FUNDING RATE CONFIGURATION (Perpetual Futures)
# =============================================================================

# Funding interval (hours) - most exchanges use 8h
FUNDING_INTERVAL_HOURS: int = 8

# Funding times (UTC) - typically 00:00, 08:00, 16:00
FUNDING_TIMES_UTC: List[str] = ["00:00", "08:00", "16:00"]

# Estimated annual funding rate for backtesting (when positive, longs pay shorts)
# Historical BTC funding averages ~10-15% APR in bull markets
DEFAULT_ANNUAL_FUNDING_RATE: float = 0.10  # 10% APR

# Convert to per-funding-period rate
# 0.10 / (365 * 3) = ~0.0000913 per 8h period = ~0.009% per period
FUNDING_RATE_PER_PERIOD: float = DEFAULT_ANNUAL_FUNDING_RATE / (365 * 3)

# Include funding in P&L calculations
INCLUDE_FUNDING_IN_PNL: bool = True

# =============================================================================
# CONTRACT SPECIFICATIONS
# =============================================================================

# Contract types
CONTRACT_TYPES: Dict[str, str] = {
    "BTC-USD": "spot",            # Spot (simulated leverage)
    "ETH-USD": "spot",
    "SOL-USD": "spot",
    "BTC-PERP-INTX": "perpetual", # INTX Perpetual futures (no expiry)
    "ETH-PERP-INTX": "perpetual",
}

# Dated futures expiration (for future support)
# Format: "SYMBOL": "YYYY-MM-DD"
FUTURES_EXPIRY: Dict[str, str] = {
    # "BTC-26DEC25-CDE": "2025-12-26",
}

# =============================================================================
# MARGIN REQUIREMENTS
# =============================================================================

# Initial margin requirement (1/leverage)
# 10x leverage = 10% initial margin
INITIAL_MARGIN_PERCENT: Dict[str, float] = {
    "BTC": 0.10,   # 10% = 10x max
    "ETH": 0.10,
    "SOL": 0.10,
}

# Maintenance margin (liquidation threshold)
MAINTENANCE_MARGIN_PERCENT: Dict[str, float] = {
    "BTC": 0.05,   # 5% = liquidation at ~20x effective leverage
    "ETH": 0.05,
    "SOL": 0.067,  # ~15x max before liquidation
}

# =============================================================================
# RISK PARAMETERS
# =============================================================================

# Default risk per trade
DEFAULT_RISK_PERCENT: float = 0.02  # 2%

# Maximum leverage (can be overridden per symbol and tier)
DEFAULT_MAX_LEVERAGE: float = 4.0  # Conservative swing default

# Minimum position size in USD
MIN_POSITION_USD: float = 10.0

# Maximum position size as percent of account
MAX_POSITION_PERCENT: float = 0.50  # 50% of account max per position

# =============================================================================
# TIMEFRAMES
# =============================================================================

# Multi-timeframe hierarchy for STRAT analysis
TIMEFRAMES: List[str] = ["1w", "1d", "4h", "1h", "15m"]

# Base timeframe for resampling (used for HTF pattern detection)
BASE_TIMEFRAME: str = "15m"

# Timeframes that can have absolute vetoes
VETO_TIMEFRAMES: List[str] = ["1w", "1d"]

# =============================================================================
# TRADING RULES
# =============================================================================

# Minimum continuity score to take a trade
MIN_CONTINUITY_SCORE: int = 2  # Out of 4

# Minimum reward:risk ratio
MIN_REWARD_RISK: float = 1.5

# Trade frequency (expected)
EXPECTED_TRADES_PER_MONTH: int = 3  # 1-5 per month

# =============================================================================
# SIMULATION / PAPER TRADING
# =============================================================================

# Default paper trading balance
DEFAULT_PAPER_BALANCE: float = 1000.0

# Enable simulation mode by default (safe default)
SIMULATION_MODE: bool = True

# Paper trading data directory
PAPER_TRADING_DATA_DIR: str = "crypto/simulation/data"

# =============================================================================
# SCANNING / DAEMON
# =============================================================================

# Scan interval in seconds
SCAN_INTERVAL_SECONDS: int = 900  # 15 minutes (matches BASE_TIMEFRAME)

# How many bars to fetch for analysis
LOOKBACK_BARS: int = 100

# Enable HTF resampling for pattern detection
ENABLE_HTF_RESAMPLING: bool = True

# Entry monitor polling interval (for live trigger detection)
ENTRY_MONITOR_POLL_SECONDS: int = 60  # 1 minute

# =============================================================================
# MAINTENANCE WINDOW (Coinbase INTX)
# =============================================================================
# Coinbase crypto futures have a 1-hour maintenance window every Friday.
# During this window, no trading occurs and bar data may be incomplete.
# Bars overlapping this window should be excluded from pattern detection.

MAINTENANCE_WINDOW_ENABLED: bool = True

# Maintenance window timing (UTC)
# Friday 5-6 PM ET = Friday 22:00-23:00 UTC (standard time)
# Note: During daylight saving time, this shifts by 1 hour
MAINTENANCE_DAY: int = 4  # 0=Monday, 4=Friday
MAINTENANCE_START_HOUR_UTC: int = 22  # 22:00 UTC = 5 PM ET
MAINTENANCE_END_HOUR_UTC: int = 23  # 23:00 UTC = 6 PM ET

# =============================================================================
# SIGNAL FILTERS
# =============================================================================

# Minimum target magnitude (as percentage of entry price)
MIN_MAGNITUDE_PCT: float = 0.5

# Minimum reward:risk ratio for signals
# Lowered from 1.5 to 1.0 to allow daily timeframe patterns (Session CRYPTO-8)
MIN_SIGNAL_RISK_REWARD: float = 1.0

# Signal expiry (how long SETUP signals remain valid)
SIGNAL_EXPIRY_HOURS: int = 24

# Pattern types to scan (all STRAT patterns)
SCAN_PATTERN_TYPES: List[str] = ["2-2", "3-2", "3-2-2", "2-1-2", "3-1-2"]

# =============================================================================
# API LIMITS
# =============================================================================

# Coinbase API rate limits
COINBASE_REQUESTS_PER_SECOND: int = 10
COINBASE_MAX_CANDLES_PER_REQUEST: int = 300

# =============================================================================
# LOGGING
# =============================================================================

# Log all trades to file
LOG_TRADES: bool = True
TRADE_LOG_PATH: str = "crypto/logs/trades.log"


# =============================================================================
# LEVERAGE TIER HELPER FUNCTIONS
# =============================================================================


def is_intraday_window(now_et: "datetime.datetime") -> bool:
    """
    Check if current time is within intraday leverage window.

    Intraday window: 6PM ET to 4PM ET next day (22 hours).
    Unavailable: 4PM ET to 6PM ET (2 hours).

    Args:
        now_et: Current datetime in ET timezone

    Returns:
        True if intraday leverage is available
    """
    hour = now_et.hour
    # Intraday window: 18:00 (6PM) to 16:00 (4PM) next day
    # This means: hour >= 18 OR hour < 16
    # Unavailable: 16:00 to 18:00 (4PM to 6PM)
    return hour >= INTRADAY_WINDOW_START_HOUR_ET or hour < INTRADAY_WINDOW_END_HOUR_ET


def get_current_leverage_tier(now_et: "datetime.datetime") -> str:
    """
    Get the current leverage tier based on time of day.

    Args:
        now_et: Current datetime in ET timezone

    Returns:
        "intraday" or "swing"
    """
    if not ENABLE_TIME_BASED_LEVERAGE:
        return DEFAULT_LEVERAGE_TIER

    if is_intraday_window(now_et):
        return "intraday"
    return "swing"


def get_max_leverage_for_symbol(
    symbol: str,
    now_et: "datetime.datetime",
) -> float:
    """
    Get the maximum leverage for a symbol based on current time.

    Args:
        symbol: Trading symbol (e.g., "BTC-PERP-INTX")
        now_et: Current datetime in ET timezone

    Returns:
        Maximum leverage (10.0 for intraday, 4.0 for swing)
    """
    tier = get_current_leverage_tier(now_et)

    # Extract base asset from symbol (BTC-PERP-INTX -> BTC)
    base_asset = symbol.split("-")[0]

    return LEVERAGE_TIERS.get(tier, {}).get(base_asset, DEFAULT_MAX_LEVERAGE)


def time_until_intraday_close_et(now_et: "datetime.datetime") -> "datetime.timedelta":
    """
    Calculate time remaining until intraday window closes (4PM ET).

    If currently outside intraday window, returns timedelta(0).

    Args:
        now_et: Current datetime in ET timezone

    Returns:
        Timedelta until 4PM ET, or 0 if outside window
    """
    from datetime import timedelta

    if not is_intraday_window(now_et):
        return timedelta(0)

    hour = now_et.hour
    # Calculate hours until 16:00 (4PM)
    if hour >= INTRADAY_WINDOW_START_HOUR_ET:
        # After 6PM, need to go through midnight to 4PM
        # Hours remaining = (24 - hour) + 16
        hours_remaining = (24 - hour) + INTRADAY_WINDOW_END_HOUR_ET
    else:
        # Before 4PM, just subtract
        hours_remaining = INTRADAY_WINDOW_END_HOUR_ET - hour

    # Subtract current minutes for more precision
    minutes_remaining = 60 - now_et.minute
    hours_remaining -= 1  # Adjust for partial hour

    return timedelta(hours=hours_remaining, minutes=minutes_remaining)
