"""
Risk-Free Rate Module for ATLAS Options Trading System.

Session 78: Provides date-based risk-free rate lookup for accurate
historical backtesting and Greeks calculations.

The risk-free rate significantly impacts option pricing:
- 2020 (COVID era): ~0.25% (near-zero rates)
- 2022-2023: 4.5-5.25% (rate hikes)
- 2024-2025: ~5.0% (elevated rates)

Using a hardcoded 5% rate for 2020 backtests would significantly
overvalue options and distort Greeks calculations.

Usage:
    from strat.risk_free_rate import get_risk_free_rate

    # Get rate for a specific date
    rate = get_risk_free_rate(datetime(2020, 3, 15))  # Returns 0.0025
    rate = get_risk_free_rate(datetime(2024, 1, 15))  # Returns 0.0525
"""

from datetime import datetime
from typing import Union
import pandas as pd


# Historical risk-free rates (1-month Treasury or Fed Funds Rate approximations)
# Source: Federal Reserve Economic Data (FRED)
# Format: (start_date, rate) - rate applies from start_date until next entry
RATE_HISTORY = [
    (datetime(2008, 1, 1), 0.0400),   # Pre-financial crisis
    (datetime(2008, 12, 1), 0.0025),  # Financial crisis, ZIRP begins
    (datetime(2015, 12, 1), 0.0050),  # First rate hike post-crisis
    (datetime(2016, 12, 1), 0.0075),
    (datetime(2017, 6, 1), 0.0125),
    (datetime(2017, 12, 1), 0.0150),
    (datetime(2018, 3, 1), 0.0175),
    (datetime(2018, 6, 1), 0.0200),
    (datetime(2018, 9, 1), 0.0225),
    (datetime(2018, 12, 1), 0.0250),
    (datetime(2019, 8, 1), 0.0225),   # Rate cuts begin
    (datetime(2019, 9, 1), 0.0200),
    (datetime(2019, 10, 1), 0.0175),
    (datetime(2020, 3, 1), 0.0025),   # COVID emergency cuts (near-zero)
    (datetime(2022, 3, 1), 0.0050),   # Rate hikes begin
    (datetime(2022, 5, 1), 0.0100),
    (datetime(2022, 6, 1), 0.0175),
    (datetime(2022, 7, 1), 0.0250),
    (datetime(2022, 9, 1), 0.0325),
    (datetime(2022, 11, 1), 0.0400),
    (datetime(2022, 12, 1), 0.0450),
    (datetime(2023, 2, 1), 0.0475),
    (datetime(2023, 3, 1), 0.0500),
    (datetime(2023, 5, 1), 0.0525),   # Peak rate
    (datetime(2023, 7, 1), 0.0550),   # Terminal rate
    (datetime(2024, 9, 1), 0.0500),   # First cut
    (datetime(2024, 11, 1), 0.0475),  # Second cut
]

# Default rate for dates outside history
DEFAULT_RATE = 0.05


def get_risk_free_rate(date: Union[datetime, pd.Timestamp]) -> float:
    """
    Get the risk-free rate for a given date.

    Uses historical Federal Funds Rate / 1-month Treasury approximations
    for accurate option pricing in backtests.

    Args:
        date: The date to get the rate for. Can be datetime or pd.Timestamp.

    Returns:
        Annual risk-free rate as a decimal (e.g., 0.05 for 5%)

    Example:
        >>> get_risk_free_rate(datetime(2020, 3, 15))
        0.0025  # Near-zero rates during COVID

        >>> get_risk_free_rate(datetime(2023, 7, 15))
        0.055  # Peak rates during hiking cycle
    """
    # Convert pandas Timestamp to datetime if needed
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()

    # Handle timezone-aware datetimes
    if hasattr(date, 'tzinfo') and date.tzinfo is not None:
        date = date.replace(tzinfo=None)

    # Search through rate history (newest to oldest)
    for threshold_date, rate in reversed(RATE_HISTORY):
        if date >= threshold_date:
            return rate

    # Date is before our history - use first rate or default
    if RATE_HISTORY:
        return RATE_HISTORY[0][1]
    return DEFAULT_RATE


def get_current_risk_free_rate() -> float:
    """
    Get the current risk-free rate (for live trading).

    Returns:
        Current annual risk-free rate as a decimal
    """
    return get_risk_free_rate(datetime.now())


if __name__ == "__main__":
    # Test the module
    print("Risk-Free Rate Module Test")
    print("=" * 50)

    test_dates = [
        datetime(2019, 1, 15),   # Pre-COVID
        datetime(2020, 3, 15),   # COVID crash
        datetime(2020, 6, 15),   # COVID recovery, ZIRP
        datetime(2022, 6, 15),   # Rate hikes beginning
        datetime(2023, 7, 15),   # Peak rates
        datetime(2024, 1, 15),   # Elevated rates
        datetime(2024, 10, 15),  # After first cut
    ]

    for date in test_dates:
        rate = get_risk_free_rate(date)
        print(f"{date.strftime('%Y-%m-%d')}: {rate:.2%}")

    print("\n" + "=" * 50)
    print(f"Current rate: {get_current_risk_free_rate():.2%}")
