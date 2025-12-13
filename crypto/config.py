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
        "BTC": 10.0,   # Close before funding (8h)
        "ETH": 10.0,
        "SOL": 10.0,
    },
    "swing": {
        "BTC": 4.0,    # Holding through funding periods
        "ETH": 4.0,
        "SOL": 3.0,    # More volatile
    },
}

# Default leverage tier for position sizing
DEFAULT_LEVERAGE_TIER: str = "swing"  # Conservative default

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
