"""
VIX Acceleration Detector for Flash Crash Detection

This module implements rapid VIX spike detection to identify flash crashes
that the academic jump model misses due to 20-60 day smoothing.

Key Use Case: August 5, 2024 flash crash (VIX +64.90% in 1 day)
- Academic model: Lagged by days (uses 20-60 day smoothing)
- VIX acceleration: Detected same day (percentage change threshold)

Architecture:
- Layer 1B-1: VIX Acceleration (this module) - Flash crashes (minutes-hours)
- Layer 1A: Academic Clustering (existing) - Slow trends (weeks-months)

References:
- Session 34 exploration: docs/exploration/SESSION_34_VIX_STRAT_CROSS_ASSET_ENHANCEMENT.md
- Validation: August 5 2024, March 2020, 2020-2024 backtest
"""

import pandas as pd
import vectorbtpro as vbt
from typing import Optional, Tuple


def fetch_vix_data(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch VIX data from Yahoo Finance using VectorBT Pro.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format or relative (e.g., '1 year ago')
    end_date : str
        End date in YYYY-MM-DD format or relative (e.g., 'today')

    Returns
    -------
    pd.Series
        VIX close prices indexed by date

    Examples
    --------
    >>> vix_data = fetch_vix_data('2024-08-01', '2024-08-10')
    >>> print(vix_data.loc['2024-08-05'])  # Should show VIX spike
    38.57

    Notes
    -----
    - Uses Yahoo Finance symbol: ^VIX
    - Data includes regular US market hours only
    - VIX is calculated by CBOE based on SPX options implied volatility
    """
    vix_data = vbt.YFData.pull('^VIX', start=start_date, end=end_date)
    vix_close = vix_data.get()['Close']
    return vix_close


def detect_vix_spike(
    vix_close: pd.Series,
    threshold_1d: float = 0.20,
    threshold_3d: float = 0.50
) -> pd.Series:
    """
    Detect rapid VIX acceleration using percentage change thresholds.

    Flash crash indicator: VIX spikes >20% in 1 day OR >50% in 3 days.

    Parameters
    ----------
    vix_close : pd.Series
        VIX close prices indexed by date
    threshold_1d : float, optional
        1-day percentage change threshold (default: 0.20 = 20%)
    threshold_3d : float, optional
        3-day percentage change threshold (default: 0.50 = 50%)

    Returns
    -------
    pd.Series
        Boolean series indicating spike days (True = flash crash detected)

    Examples
    --------
    >>> vix_data = fetch_vix_data('2024-08-01', '2024-08-10')
    >>> spikes = detect_vix_spike(vix_data)
    >>> print(spikes.loc['2024-08-05'])  # August 5 flash crash
    True

    >>> # March 2020 crash (gradual escalation)
    >>> vix_data_2020 = fetch_vix_data('2020-03-01', '2020-03-31')
    >>> spikes_2020 = detect_vix_spike(vix_data_2020)
    >>> print(spikes_2020.sum())  # Multiple spike days

    Notes
    -----
    Threshold Calibration (from August 5, 2024):
    - 1-day change: 64.90% (VIX 23.39 → 38.57)
    - Threshold 20%: Conservative (catches major spikes)
    - Threshold 30%: More conservative (fewer false positives)

    Design Rationale:
    - OR logic: Trigger on EITHER 1-day OR 3-day spike
    - 1-day threshold catches flash crashes (Aug 5, 2024)
    - 3-day threshold catches rapid escalation (March 2020)
    - NaN values automatically set to False (no spike)
    """
    # Calculate percentage changes
    vix_change_1d = vix_close.pct_change()
    vix_change_3d = vix_close.pct_change(periods=3)

    # Detect spikes (OR logic: either threshold triggers)
    spike_1d = vix_change_1d > threshold_1d
    spike_3d = vix_change_3d > threshold_3d

    # Combine: spike if EITHER threshold exceeded
    spike_detected = spike_1d | spike_3d

    # Handle NaN values (first few days have no comparison)
    spike_detected = spike_detected.fillna(False)

    return spike_detected


def classify_vix_severity(
    vix_close: pd.Series,
    threshold_flash_crash: float = 0.30,
    threshold_elevated: float = 0.15
) -> pd.Series:
    """
    Classify VIX acceleration severity into discrete levels.

    Severity Levels:
    - FLASH_CRASH: VIX spike >30% (extreme, immediate CRASH regime)
    - ELEVATED: VIX spike >15% but <30% (heightened volatility)
    - NORMAL: VIX spike <15% (routine fluctuations)

    Parameters
    ----------
    vix_close : pd.Series
        VIX close prices indexed by date
    threshold_flash_crash : float, optional
        1-day change threshold for FLASH_CRASH (default: 0.30 = 30%)
    threshold_elevated : float, optional
        1-day change threshold for ELEVATED (default: 0.15 = 15%)

    Returns
    -------
    pd.Series
        String series with severity levels ('FLASH_CRASH', 'ELEVATED', 'NORMAL')

    Examples
    --------
    >>> vix_data = fetch_vix_data('2024-08-01', '2024-08-10')
    >>> severity = classify_vix_severity(vix_data)
    >>> print(severity.loc['2024-08-05'])
    'FLASH_CRASH'

    Notes
    -----
    Use Cases:
    - FLASH_CRASH: Override ATLAS to CRASH regime immediately
    - ELEVATED: Warning signal, reduce position sizes
    - NORMAL: No override, use academic clustering regime

    Threshold Rationale (from August 5, 2024):
    - Aug 5 change: 64.90% → FLASH_CRASH (well above 30%)
    - Aug 2 change: 25.82% → ELEVATED (between 15-30%)
    """
    vix_change_1d = vix_close.pct_change()

    # Initialize as NORMAL
    severity = pd.Series('NORMAL', index=vix_close.index)

    # Classify based on thresholds
    severity[vix_change_1d > threshold_elevated] = 'ELEVATED'
    severity[vix_change_1d > threshold_flash_crash] = 'FLASH_CRASH'

    return severity


def get_vix_regime_override(
    vix_close: pd.Series,
    threshold_1d: float = 0.20,
    threshold_3d: float = 0.50
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate ATLAS regime override signals based on VIX acceleration.

    Convenience function combining spike detection and severity classification.

    Parameters
    ----------
    vix_close : pd.Series
        VIX close prices indexed by date
    threshold_1d : float, optional
        1-day percentage change threshold (default: 0.20 = 20%)
    threshold_3d : float, optional
        3-day percentage change threshold (default: 0.50 = 50%)

    Returns
    -------
    spike_detected : pd.Series
        Boolean series indicating spike days
    severity : pd.Series
        String series with severity levels

    Examples
    --------
    >>> vix_data = fetch_vix_data('2024-08-01', '2024-08-10')
    >>> spikes, severity = get_vix_regime_override(vix_data)
    >>>
    >>> # Override ATLAS regime with VIX
    >>> atlas_regime = academic_model.online_inference(spy_data)
    >>> atlas_regime[spikes] = 'CRASH'  # VIX override

    Notes
    -----
    Integration Pattern:
    1. Academic model generates baseline regime (bull/bear/neutral/crash)
    2. VIX acceleration detects flash crashes
    3. VIX override to CRASH takes precedence
    4. CRASH veto in strat/atlas_integration.py rejects bullish patterns
    """
    spike_detected = detect_vix_spike(vix_close, threshold_1d, threshold_3d)
    severity = classify_vix_severity(vix_close)

    return spike_detected, severity


# Module-level validation
if __name__ == '__main__':
    print("VIX Acceleration Module - Validation Tests")
    print("=" * 60)

    # Test 1: August 5, 2024 flash crash
    print("\nTest 1: August 5, 2024 Flash Crash Detection")
    vix_aug_2024 = fetch_vix_data('2024-08-01', '2024-08-10')
    spikes_aug, severity_aug = get_vix_regime_override(vix_aug_2024)

    print(f"VIX on Aug 5: {vix_aug_2024.loc['2024-08-05']:.2f}")
    print(f"Spike detected: {spikes_aug.loc['2024-08-05']}")
    print(f"Severity: {severity_aug.loc['2024-08-05']}")

    assert spikes_aug.loc['2024-08-05'] == True, "Failed to detect Aug 5 flash crash"
    assert severity_aug.loc['2024-08-05'] == 'FLASH_CRASH', "Severity misclassified"
    print("[PASS] August 5, 2024 flash crash detected correctly")

    # Test 2: Normal volatility period (should have minimal spikes)
    print("\nTest 2: Normal Volatility Period (2021-2022)")
    vix_2021 = fetch_vix_data('2021-01-01', '2021-12-31')
    spikes_2021, _ = get_vix_regime_override(vix_2021)

    false_positive_rate = spikes_2021.sum() / len(spikes_2021)
    print(f"Spike days in 2021: {spikes_2021.sum()} / {len(spikes_2021)}")
    print(f"False positive rate: {false_positive_rate:.2%}")

    assert false_positive_rate < 0.05, "Too many false positives (>5%)"
    print("[PASS] False positive rate acceptable")

    print("\n" + "=" * 60)
    print("All validation tests passed!")
    print("VIX Acceleration Module ready for integration")
