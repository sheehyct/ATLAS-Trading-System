# EQUITY-84: Shared utilities for signal automation
# Phase 4 refactoring - Common utilities used across coordinators

"""
Utilities package for signal automation.

Extracted components:
- MarketHoursValidator: NYSE market hours validation (EQUITY-86)
- MarketSchedule: Market schedule dataclass
- ThreadSafeCache: Thread-safe caching with TTL (planned)
- DaemonStats: Thread-safe metric counters (planned)
"""

from strat.signal_automation.utils.market_hours import (
    MarketHoursValidator,
    MarketSchedule,
    is_market_hours,
    get_market_hours_validator,
)

__all__ = [
    'MarketHoursValidator',
    'MarketSchedule',
    'is_market_hours',
    'get_market_hours_validator',
]
