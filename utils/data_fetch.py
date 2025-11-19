"""
Data fetching utilities with mandatory timezone handling.

This module enforces correct date/timezone patterns for market data fetches.
See CLAUDE.md "CRITICAL: Date and Timezone Handling" section.
"""

import vectorbtpro as vbt
import pandas as pd
from typing import Optional, Dict, Any


def fetch_us_stocks(
    symbols: str or list,
    start: str,
    end: str,
    timeframe: str = '1d',
    source: str = 'alpaca',
    client_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> vbt.Data:
    """
    Fetch US stock data with mandatory timezone enforcement.

    CRITICAL: This function ALWAYS sets tz='America/New_York' to prevent
    UTC date shifts that cause weekend dates and misalignment with TradingView.

    Args:
        symbols: Stock symbol(s) to fetch (e.g., 'AAPL' or ['AAPL', 'MSFT'])
        start: Start date in format 'YYYY-MM-DD' (e.g., '2025-11-01')
        end: End date in format 'YYYY-MM-DD' (e.g., '2025-11-20')
        timeframe: Timeframe specification (default: '1d' for daily)
        source: Data source ('alpaca' or 'tiingo', default: 'alpaca')
        client_config: API credentials dict (api_key, secret_key, etc.)
        **kwargs: Additional parameters passed to data source

    Returns:
        vbt.Data: Market data with proper timezone (America/New_York)

    Raises:
        ValueError: If source is not 'alpaca' or 'tiingo'
        AssertionError: If fetched data contains weekend dates

    Examples:
        >>> # Fetch AAPL data with Alpaca
        >>> data = fetch_us_stocks(
        ...     'AAPL',
        ...     start='2025-11-01',
        ...     end='2025-11-20',
        ...     client_config=dict(
        ...         api_key='YOUR_KEY',
        ...         secret_key='YOUR_SECRET',
        ...         paper=True
        ...     )
        ... )

        >>> # Verify no weekend dates
        >>> df = data.get()
        >>> for idx in df.index:
        ...     assert idx.strftime('%A') not in ['Saturday', 'Sunday']

    See Also:
        - CLAUDE.md: "CRITICAL: Date and Timezone Handling"
        - OpenMemory: Tag "date-handling"
    """
    # Validate source
    if source not in ['alpaca', 'tiingo']:
        raise ValueError(f"source must be 'alpaca' or 'tiingo', got: {source}")

    # CRITICAL: Always set timezone to America/New_York for US markets
    kwargs['tz'] = 'America/New_York'

    # Fetch data from specified source
    if source == 'alpaca':
        if client_config is None:
            raise ValueError("client_config required for Alpaca (api_key, secret_key, paper)")

        data = vbt.AlpacaData.pull(
            symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            client_config=client_config,
            **kwargs
        )

    elif source == 'tiingo':
        if client_config is None:
            raise ValueError("client_config required for Tiingo (api_key)")

        data = vbt.TiingoData.pull(
            symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            client_config=client_config,
            **kwargs
        )

    # VERIFICATION: Check for weekend dates
    df = data.get()
    weekend_dates = []

    for idx in df.index:
        weekday = idx.strftime('%A')
        if weekday in ['Saturday', 'Sunday']:
            weekend_dates.append(f"{idx.date()} ({weekday})")

    if weekend_dates:
        raise AssertionError(
            f"Weekend dates found in fetched data: {weekend_dates}\n"
            f"This indicates a timezone or data source issue."
        )

    # Verify timezone
    if df.index.tz.zone != 'America/New_York':
        raise AssertionError(
            f"Timezone mismatch: expected 'America/New_York', got '{df.index.tz.zone}'"
        )

    return data


def verify_bar_classifications(
    data: pd.DataFrame,
    expected: Dict[str, str],
    verbose: bool = True
) -> tuple:
    """
    Verify bar classifications match expected values (e.g., from TradingView).

    Args:
        data: DataFrame with 'High' and 'Low' columns and datetime index
        expected: Dict mapping date strings to expected bar types (e.g., {'2025-11-19': '2U'})
        verbose: If True, print detailed comparison

    Returns:
        tuple: (matches, total, accuracy_pct)

    Example:
        >>> expected = {'2025-11-19': '2U', '2025-11-18': '3'}
        >>> matches, total, acc = verify_bar_classifications(df, expected)
        >>> print(f"Accuracy: {acc:.1f}%")
    """
    from strat import classify_bars, format_bar_classifications

    # Classify bars
    classifications = classify_bars(data['High'], data['Low'])
    labels = format_bar_classifications(classifications, skip_reference=True)

    # Get dates (skip reference bar)
    dates = list(data.index[1:])

    matches = 0
    total = 0

    if verbose:
        print("=" * 70)
        print("BAR CLASSIFICATION VERIFICATION")
        print("=" * 70)
        print(f"{'DATE':<12} {'DAY':<4} | {'OUR':<4} | {'EXPECTED':<8} | {'MATCH':<5}")
        print("-" * 70)

    # Compare newest to oldest
    for i in range(len(dates) - 1, -1, -1):
        date = dates[i]
        our_label = labels[i]
        date_str = date.strftime('%Y-%m-%d')
        day_name = date.strftime('%a')

        expected_label = expected.get(date_str, '???')

        if expected_label != '???':
            total += 1
            if our_label == expected_label:
                matches += 1
                match_str = "✓"
            else:
                match_str = "✗"
        else:
            match_str = "-"

        if verbose:
            print(f"{date_str} {day_name} | {our_label:<4} | {expected_label:<8} | {match_str:<5}")

    accuracy = (100.0 * matches / total) if total > 0 else 0.0

    if verbose:
        print("-" * 70)
        print(f"ACCURACY: {matches}/{total} = {accuracy:.1f}%")
        print("=" * 70)

    return matches, total, accuracy
