"""
Real-Time VIX Spike Detection (Paper Trading Version)

Detects flash crash conditions using intraday VIX data from Yahoo Finance.

IMPORTANT: This implementation uses yfinance for paper trading purposes.
Before deploying with live funds, upgrade to massive.com for real-time
unlimited index data including VIX.

Key Features:
- Intraday spike detection (open to current within trading day)
- Multi-day spike detection (1-day and 3-day changes)
- Absolute VIX level thresholds
- After-hours monitoring capability
- Fallback to daily data when intraday unavailable
- Robust error handling for data source issues

Thresholds (calibrated for CRASH regime):
- Intraday: +20% from market open (09:30 EST)
- 1-Day: +20% from previous close
- 3-Day: +50% from 3 days ago
- Absolute: VIX >= 35 (extreme fear regardless of change)

Usage:
    detector = VIXSpikeDetector()
    is_crash = detector.detect_crash()

    if is_crash:
        print(f"CRASH detected: {detector.get_details()}")
        print(detector.format_report())
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple
import pytz


class VIXSpikeDetector:
    """
    Real-time VIX spike detection for flash crash identification.

    Uses intraday 1-minute VIX data during market hours, falls back to
    daily data after hours or on weekends.

    Data Source: Yahoo Finance (yfinance)
    Note: For live trading, upgrade to massive.com for real-time data.
    """

    # Crash detection thresholds
    INTRADAY_THRESHOLD = 0.20  # 20% from open to current
    ONE_DAY_THRESHOLD = 0.20   # 20% from yesterday's close
    THREE_DAY_THRESHOLD = 0.50 # 50% from 3 days ago
    ABSOLUTE_VIX_THRESHOLD = 35.0  # VIX >= 35 = extreme fear

    # Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(self):
        """Initialize VIX spike detector."""
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.vix_ticker = yf.Ticker("^VIX")

        # Cache for results (prevent excessive API calls)
        self._last_check: Optional[datetime] = None
        self._cached_result: Optional[Dict] = None
        self._cache_duration_seconds = 60  # Cache for 1 minute

    def detect_crash(self) -> bool:
        """
        Check if VIX indicates crash conditions.

        Returns:
            True if any crash threshold exceeded, False otherwise
        """
        details = self.get_details()
        return details['is_crash']

    def get_details(self) -> Dict:
        """
        Get detailed VIX spike analysis.

        Returns:
            Dict with:
            - is_crash (bool): True if crash detected
            - vix_current (float): Current VIX level
            - intraday_change_pct (float): Intraday % change
            - one_day_change_pct (float): 1-day % change
            - three_day_change_pct (float): 3-day % change
            - triggers (list): Which thresholds were exceeded
            - timestamp (datetime): When check was performed
            - market_open (bool): Whether market is currently open
            - data_source_error (str): Error message if data fetch failed
        """
        # Check cache
        now = datetime.now(self.eastern_tz)
        if self._cached_result and self._last_check:
            elapsed = (now - self._last_check).total_seconds()
            if elapsed < self._cache_duration_seconds:
                return self._cached_result

        # Perform fresh check
        result = self._check_vix_conditions(now)

        # Update cache
        self._last_check = now
        self._cached_result = result

        return result

    def _check_vix_conditions(self, now: datetime) -> Dict:
        """
        Perform VIX spike detection across all thresholds.

        Args:
            now: Current time (Eastern timezone)

        Returns:
            Dict with detection results
        """
        market_open = self._is_market_open(now)

        # Initialize result values
        vix_current = None
        vix_open_today = None
        vix_prev_close = None
        vix_3d_ago = None
        intraday_change_pct = None
        one_day_change_pct = None
        three_day_change_pct = None
        data_source_error = None

        try:
            if market_open:
                # During market hours: use intraday data
                intraday_data = self._get_intraday_vix()
                if not intraday_data.empty:
                    vix_current = float(intraday_data['Close'].iloc[-1])
                    vix_open_today = float(intraday_data['Close'].iloc[0])
                    intraday_change_pct = (vix_current / vix_open_today - 1) * 100

                    # Also get daily data for multi-day checks
                    daily_data = self._get_daily_vix(days_back=5)
                    if len(daily_data) >= 2:
                        vix_prev_close = float(daily_data['Close'].iloc[-2])
                        one_day_change_pct = (vix_current / vix_prev_close - 1) * 100
                    if len(daily_data) >= 4:
                        vix_3d_ago = float(daily_data['Close'].iloc[-4])
                        three_day_change_pct = (vix_current / vix_3d_ago - 1) * 100
                else:
                    data_source_error = "No intraday VIX data available"
            else:
                # After hours/weekend: use daily data only
                daily_data = self._get_daily_vix(days_back=5)
                if not daily_data.empty:
                    vix_current = float(daily_data['Close'].iloc[-1])
                    if len(daily_data) >= 2:
                        vix_prev_close = float(daily_data['Close'].iloc[-2])
                        one_day_change_pct = (vix_current / vix_prev_close - 1) * 100
                    if len(daily_data) >= 4:
                        vix_3d_ago = float(daily_data['Close'].iloc[-4])
                        three_day_change_pct = (vix_current / vix_3d_ago - 1) * 100
                else:
                    data_source_error = "No daily VIX data available"

        except Exception as e:
            # Fallback to None values on error
            data_source_error = f"VIX data fetch failed: {str(e)}"
            print(f"Warning: {data_source_error}")

        # Check thresholds
        triggers = []

        if vix_current is not None and vix_current >= self.ABSOLUTE_VIX_THRESHOLD:
            triggers.append(f"Absolute VIX >= {self.ABSOLUTE_VIX_THRESHOLD} ({vix_current:.2f})")

        if intraday_change_pct is not None and intraday_change_pct >= self.INTRADAY_THRESHOLD * 100:
            triggers.append(f"Intraday spike >= {self.INTRADAY_THRESHOLD*100:.0f}% ({intraday_change_pct:+.1f}%)")

        if one_day_change_pct is not None and one_day_change_pct >= self.ONE_DAY_THRESHOLD * 100:
            triggers.append(f"1-day spike >= {self.ONE_DAY_THRESHOLD*100:.0f}% ({one_day_change_pct:+.1f}%)")

        if three_day_change_pct is not None and three_day_change_pct >= self.THREE_DAY_THRESHOLD * 100:
            triggers.append(f"3-day spike >= {self.THREE_DAY_THRESHOLD*100:.0f}% ({three_day_change_pct:+.1f}%)")

        return {
            'is_crash': len(triggers) > 0,
            'vix_current': vix_current,
            'vix_open_today': vix_open_today,
            'vix_prev_close': vix_prev_close,
            'vix_3d_ago': vix_3d_ago,
            'intraday_change_pct': intraday_change_pct,
            'one_day_change_pct': one_day_change_pct,
            'three_day_change_pct': three_day_change_pct,
            'triggers': triggers,
            'timestamp': now,
            'market_open': market_open,
            'data_source_error': data_source_error
        }

    def _get_intraday_vix(self) -> pd.DataFrame:
        """
        Get today's 1-minute VIX data.

        Returns:
            DataFrame with intraday VIX bars (Close, Open, High, Low, Volume)
        """
        try:
            # Get 1-minute data for today
            hist = self.vix_ticker.history(period='1d', interval='1m')
            return hist
        except Exception as e:
            print(f"Warning: Failed to fetch intraday VIX: {e}")
            return pd.DataFrame()

    def _get_daily_vix(self, days_back: int = 5) -> pd.DataFrame:
        """
        Get daily VIX data for last N days.

        Args:
            days_back: Number of days to fetch

        Returns:
            DataFrame with daily VIX bars
        """
        try:
            hist = self.vix_ticker.history(period=f'{days_back}d', interval='1d')
            return hist
        except Exception as e:
            print(f"Warning: Failed to fetch daily VIX: {e}")
            return pd.DataFrame()

    def _is_market_open(self, now: datetime) -> bool:
        """
        Check if US stock market is currently open.

        Args:
            now: Current time (Eastern timezone)

        Returns:
            True if market is open, False otherwise
        """
        # Check if weekday (0 = Monday, 6 = Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check if within market hours
        current_time = now.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def format_report(self) -> str:
        """
        Generate human-readable crash detection report.

        Returns:
            Formatted string with detection results
        """
        details = self.get_details()

        lines = []
        lines.append("=" * 60)
        lines.append("VIX CRASH DETECTION REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {details['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        lines.append(f"Market Status: {'OPEN' if details['market_open'] else 'CLOSED'}")
        lines.append(f"Data Source: Yahoo Finance (yfinance)")
        lines.append("")

        if details['data_source_error']:
            lines.append(f"WARNING: DATA ERROR: {details['data_source_error']}")
            lines.append("Crash detection may be unreliable - proceeding with caution")
            lines.append("")

        if details['vix_current']:
            lines.append(f"Current VIX: {details['vix_current']:.2f}")
        else:
            lines.append("Current VIX: N/A")
        lines.append("")

        lines.append("THRESHOLDS:")
        lines.append(f"  Intraday spike: >= {self.INTRADAY_THRESHOLD*100:.0f}%")
        lines.append(f"  1-day spike: >= {self.ONE_DAY_THRESHOLD*100:.0f}%")
        lines.append(f"  3-day spike: >= {self.THREE_DAY_THRESHOLD*100:.0f}%")
        lines.append(f"  Absolute VIX: >= {self.ABSOLUTE_VIX_THRESHOLD:.0f}")
        lines.append("")

        lines.append("CURRENT CONDITIONS:")
        if details['intraday_change_pct'] is not None:
            lines.append(f"  Intraday: {details['vix_open_today']:.2f} -> {details['vix_current']:.2f} ({details['intraday_change_pct']:+.1f}%)")
        if details['one_day_change_pct'] is not None:
            lines.append(f"  1-Day: {details['vix_prev_close']:.2f} -> {details['vix_current']:.2f} ({details['one_day_change_pct']:+.1f}%)")
        if details['three_day_change_pct'] is not None:
            lines.append(f"  3-Day: {details['vix_3d_ago']:.2f} -> {details['vix_current']:.2f} ({details['three_day_change_pct']:+.1f}%)")
        lines.append("")

        if details['is_crash']:
            lines.append("*** CRASH DETECTED ***")
            lines.append("")
            lines.append("TRIGGERS:")
            for trigger in details['triggers']:
                lines.append(f"  * {trigger}")
            lines.append("")
            lines.append("RECOMMENDED ACTION: Switch to CRASH regime (0% allocation)")
        else:
            lines.append("* NO CRASH DETECTED")
            lines.append("Market volatility within acceptable bounds")

        lines.append("=" * 60)

        return "\n".join(lines)


if __name__ == '__main__':
    """Test VIX crash detection."""
    detector = VIXSpikeDetector()
    print(detector.format_report())
