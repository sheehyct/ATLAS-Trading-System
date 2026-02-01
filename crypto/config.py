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
#   - coinbase_cfm: "BIP-20DEC30-CDE", "ETP-20DEC30-CDE" (US CFM venue)
#   - hyperliquid: "BTC", "ETH", "SOL"
#
# NOTE: CFM products have expiration dates in format: SYMBOL-DDMMMYY-CDE
# BIP = Nano Bitcoin Perp Style Futures (CFM - what user trades)
# ETP = Nano Ether Perp Style Futures (CFM)
# Contract rollover: Update symbols when approaching expiration
CRYPTO_SYMBOLS: List[str] = [
    "BIP-20DEC30-CDE",  # Nano Bitcoin Perp Style (CFM - primary)
    "ETP-20DEC30-CDE",  # Nano Ether Perp Style (CFM)
]

# Legacy INTX symbols (for reference/fallback)
INTX_SYMBOLS: List[str] = [
    "BTC-PERP-INTX",  # Bitcoin Perpetual (INTX)
    "ETH-PERP-INTX",  # Ethereum Perpetual (INTX)
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
# CORRECTED Session Jan 23, 2026: SOL/XRP/ADA are 5x intraday, not 10x
# CORRECTED Session Jan 24, 2026: Overnight/swing values verified from platform
LEVERAGE_TIERS: Dict[str, Dict[str, float]] = {
    "intraday": {
        "BTC": 10.0,   # Must close before 4PM ET
        "ETH": 10.0,
        "SOL": 5.0,    # Altcoins get 5x max intraday
        "XRP": 5.0,
        "ADA": 5.0,
    },
    "overnight": {
        # Verified from Coinbase CFM platform Jan 24, 2026
        "BTC": 4.1,
        "ETH": 4.0,
        "SOL": 2.7,
        "XRP": 2.6,
        "ADA": 3.4,
    },
    # Legacy alias for backward compatibility
    "swing": {
        "BTC": 4.1,
        "ETH": 4.0,
        "SOL": 2.7,
        "XRP": 2.6,
        "ADA": 3.4,
    },
}

# =============================================================================
# BETA TO BTC (Volatility Multiplier)
# =============================================================================
# Empirical beta values - how much each asset moves relative to BTC
# Calculated from Day Down/Day Up ranges (Jan 23, 2026 snapshot)
# Should be recalculated periodically as market dynamics change
#
# Key Insight: Lower leverage altcoins can outperform higher leverage BTC/ETH
# when their beta exceeds the leverage differential.
# Example: ADA (5x, 2.2 beta) beats BTC (10x, 1.0 beta) on same market move

CRYPTO_BETA_TO_BTC: Dict[str, float] = {
    "BTC": 1.00,
    "ETH": 1.98,
    "SOL": 1.55,
    "XRP": 1.77,
    "ADA": 2.20,
}

# Effective multiplier = Leverage × Beta
# This determines true capital efficiency
# Ranking (intraday): ETH (19.8) > ADA (11.0) > BTC (10.0) > XRP (8.85) > SOL (7.75)
EFFECTIVE_MULTIPLIER_INTRADAY: Dict[str, float] = {
    symbol: LEVERAGE_TIERS["intraday"].get(symbol, 5.0) * CRYPTO_BETA_TO_BTC.get(symbol, 1.0)
    for symbol in CRYPTO_BETA_TO_BTC
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
    # CFM Nano Perp Style Futures (dated contracts that roll)
    "BIP-20DEC30-CDE": "dated_future",  # Nano Bitcoin - expires Dec 30, 2025
    "ETP-20DEC30-CDE": "dated_future",  # Nano Ether - expires Dec 30, 2025
}

# Contract sizes for CFM Nano Perp Style Futures
# VERIFIED Jan 24, 2026 from Coinbase CFM platform
CONTRACT_SIZES: Dict[str, float] = {
    "BTC": 0.01,     # 0.01 BTC per contract
    "ETH": 0.1,      # 0.1 ETH per contract
    "SOL": 5.0,      # 5 SOL per contract
    "XRP": 500.0,    # 500 XRP per contract
    "ADA": 1000.0,   # 1,000 ADA per contract
}

# Dated futures expiration
# Format: "SYMBOL": "YYYY-MM-DD"
# NOTE: These are "Perp Style" futures - they roll quarterly but have expiration
FUTURES_EXPIRY: Dict[str, str] = {
    "BIP-20DEC30-CDE": "2025-12-30",  # Nano Bitcoin Dec contract
    "ETP-20DEC30-CDE": "2025-12-30",  # Nano Ether Dec contract
}

# Symbol to base asset mapping (for leverage lookups)
# Maps CFM product codes to their underlying asset
SYMBOL_TO_BASE_ASSET: Dict[str, str] = {
    "BIP": "BTC",  # Nano Bitcoin Perp Style -> BTC
    "ETP": "ETH",  # Nano Ether Perp Style -> ETH
    "BTC": "BTC",  # Direct mapping
    "ETH": "ETH",
    "SOL": "SOL",
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
# FEE STRUCTURE (Coinbase CFM)
# =============================================================================
# VERIFIED Jan 24, 2026 from Coinbase CFM platform
# Fee = (Notional * Rate) + Fixed_Per_Contract
# Fee tiers are based on trailing 30-day derivatives volume

# Default tier fee rates (adjusts based on volume tier)
MAKER_FEE_RATE: float = 0.00065    # 0.065% for maker (limit orders)
TAKER_FEE_RATE: float = 0.0007     # 0.07% for taker (market orders)

# Fixed per-contract fee (covers NFA, exchange, and vendor costs)
MIN_FEE_PER_CONTRACT: float = 0.15  # $0.15 per contract ALL trades

# Liquidation fee
LIQUIDATION_FEE_RATE: float = 0.001  # 0.10%


def calculate_trade_fee(notional: float, is_maker: bool = False, num_contracts: int = 1) -> float:
    """
    Calculate Coinbase CFM trade fee.
    
    Fee = (Notional × Rate) + (Fixed × Contracts)
    
    Args:
        notional: Total notional value of trade in USD
        is_maker: True for limit orders, False for market orders
        num_contracts: Number of contracts (for fixed fee component)
    
    Returns:
        Total fee in USD
    
    Example:
        >>> calculate_trade_fee(1000, is_maker=False, num_contracts=1)
        0.85  # $1000 × 0.07% + $0.15 = $0.70 + $0.15 = $0.85
    """
    rate = MAKER_FEE_RATE if is_maker else TAKER_FEE_RATE
    percentage_fee = notional * rate
    fixed_fee = MIN_FEE_PER_CONTRACT * num_contracts
    return percentage_fee + fixed_fee


def calculate_round_trip_fee(notional: float, is_maker: bool = False, num_contracts: int = 1) -> float:
    """Calculate total fee for entry + exit."""
    return calculate_trade_fee(notional, is_maker, num_contracts) * 2


def calculate_breakeven_move(notional: float, is_maker: bool = False, num_contracts: int = 1) -> float:
    """Calculate minimum price move percentage needed to break even after fees."""
    round_trip = calculate_round_trip_fee(notional, is_maker, num_contracts)
    return round_trip / notional  # As decimal (multiply by 100 for percentage)

# =============================================================================
# RISK PARAMETERS
# =============================================================================

# Default risk per trade (used only when LEVERAGE_FIRST_SIZING is False)
DEFAULT_RISK_PERCENT: float = 0.02  # 2%

# Maximum leverage (can be overridden per symbol and tier)
DEFAULT_MAX_LEVERAGE: float = 4.0  # Conservative swing default

# Minimum position size in USD
MIN_POSITION_USD: float = 10.0

# Maximum position size as percent of account
MAX_POSITION_PERCENT: float = 0.50  # 50% of account max per position

# Session EQUITY-59: Leverage-first sizing mode
# When True: Always use full available leverage (10x intraday, 4x swing)
# Actual risk floats based on stop distance. Used for paper trading and data collection.
# When False: Risk-based sizing (2% risk per trade, leverage as needed)
LEVERAGE_FIRST_SIZING: bool = True

# =============================================================================
# TIMEFRAMES
# =============================================================================

# Multi-timeframe hierarchy for STRAT analysis
# Session EQUITY-34: Removed 15m per user request - only 1H and above
TIMEFRAMES: List[str] = ["1w", "1d", "4h", "1h"]

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
# FEE PROFITABILITY FILTER (Session EQUITY-99)
# =============================================================================
# Reject trades where fees consume too much of expected profit

# Enable fee profitability filter at entry
FEE_PROFITABILITY_FILTER_ENABLED: bool = True

# Maximum acceptable fee as percentage of target profit
# 0.20 = reject if fees > 20% of expected profit
MAX_FEE_PCT_OF_TARGET: float = 0.20


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
# TEMPORARY: Disabled (0.0) for testing to capture more trade data (Session EQUITY-23)
MIN_SIGNAL_RISK_REWARD: float = 0.0

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
# SPOT/DERIVATIVE SYMBOL MAPPING (Session EQUITY-99)
# =============================================================================
# Use spot data for cleaner price action in signal detection
# Execute trades on derivatives for actual trading
#
# Problem: CFM derivatives can have artificial long wicks during low liquidity
# periods, creating false STRAT signals. Using spot data for signal detection
# provides cleaner price action while executing trades on derivatives.

DERIVATIVE_TO_SPOT: Dict[str, str] = {
    "BIP-20DEC30-CDE": "BTC-USD",
    "ETP-20DEC30-CDE": "ETH-USD",
}
SPOT_TO_DERIVATIVE: Dict[str, str] = {v: k for k, v in DERIVATIVE_TO_SPOT.items()}

# Base assets that have reliable spot data available
SPOT_DATA_AVAILABLE: set = {"BTC", "ETH", "SOL"}

# Feature toggles - enable/disable spot data usage
USE_SPOT_FOR_SIGNALS: bool = True  # Use spot data for pattern detection
USE_SPOT_FOR_TRIGGERS: bool = True  # Use spot prices for entry triggers

# Maximum divergence threshold between spot and derivative prices
# Skip trade if spot/derivative prices diverge more than this percentage
MAX_SPOT_DERIVATIVE_DIVERGENCE: float = 0.02  # 2%


# =============================================================================
# LEVERAGE TIER HELPER FUNCTIONS
# =============================================================================


def is_weekend_leverage_window(now_et: "datetime.datetime") -> bool:
    """
    Check if current time is within weekend leverage window (4x only).

    Weekend window (swing leverage only):
    - Friday 4PM ET to Sunday 6PM ET

    Similar to equity futures weekend handling.

    Args:
        now_et: Current datetime in ET timezone

    Returns:
        True if in weekend window (10x leverage unavailable)
    """
    weekday = now_et.weekday()  # Monday=0, Friday=4, Saturday=5, Sunday=6
    hour = now_et.hour

    # Friday after 4PM ET
    if weekday == 4 and hour >= INTRADAY_WINDOW_END_HOUR_ET:
        return True

    # All of Saturday
    if weekday == 5:
        return True

    # Sunday before 6PM ET
    if weekday == 6 and hour < INTRADAY_WINDOW_START_HOUR_ET:
        return True

    return False


def is_intraday_window(now_et: "datetime.datetime") -> bool:
    """
    Check if current time is within intraday leverage window.

    Intraday window: 6PM ET to 4PM ET next day (22 hours) - weekdays only.
    Unavailable: 4PM ET to 6PM ET (2 hours).
    Weekend: No intraday leverage (Friday 4PM ET to Sunday 6PM ET).

    Args:
        now_et: Current datetime in ET timezone

    Returns:
        True if intraday leverage (10x) is available
    """
    # Session EQUITY-59: No intraday leverage during weekend
    if is_weekend_leverage_window(now_et):
        return False

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
        symbol: Trading symbol (e.g., "BTC-PERP-INTX" or "BIP-20DEC30-CDE")
        now_et: Current datetime in ET timezone

    Returns:
        Maximum leverage (10.0 for intraday, 4.0 for swing)
    """
    tier = get_current_leverage_tier(now_et)

    # Extract base asset from symbol (BTC-PERP-INTX -> BTC, BIP-20DEC30-CDE -> BIP)
    symbol_prefix = symbol.split("-")[0]

    # Map CFM symbols to base assets (BIP -> BTC, ETP -> ETH)
    base_asset = SYMBOL_TO_BASE_ASSET.get(symbol_prefix, symbol_prefix)

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
