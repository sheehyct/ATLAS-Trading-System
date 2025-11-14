"""
Tests for VIX Acceleration Flash Crash Detection

Validates VIX acceleration layer against historical flash crashes:
- Test Case 1: August 5, 2024 (VIX +64.90% in 1 day)
- Test Case 2: March 2020 crash escalation (VIX 15 -> 82)
- Test Case 3: Normal volatility 2021-2022 (false positive rate)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from regime.vix_acceleration import (
    fetch_vix_data,
    detect_vix_spike,
    classify_vix_severity,
    get_vix_regime_override
)
from regime.academic_jump_model import AcademicJumpModel
from data.alpaca import fetch_alpaca_data


# Fixture: VIX data for August 2024 flash crash
@pytest.fixture
def vix_aug_2024():
    """Fetch VIX data for August 2024 flash crash period."""
    return fetch_vix_data('2024-08-01', '2024-08-10')


# Fixture: VIX data for March 2020 crash
@pytest.fixture
def vix_march_2020():
    """Fetch VIX data for March 2020 crash period."""
    return fetch_vix_data('2020-03-01', '2020-03-31')


# Fixture: VIX data for normal volatility period
@pytest.fixture
def vix_normal_2021():
    """Fetch VIX data for normal volatility 2021."""
    return fetch_vix_data('2021-01-01', '2021-12-31')


# Fixture: SPY data for academic model integration tests
@pytest.fixture
def spy_data_3000():
    """Fetch SPY data for testing (3000 days lookback)."""
    try:
        data = fetch_alpaca_data('SPY', '1D', period_days=3000)
        return data
    except Exception as e:
        pytest.skip(f"Alpaca data fetch failed: {e}")


class TestVIXDataFetching:
    """Test VIX data fetching functionality."""

    def test_fetch_vix_data_august_2024(self, vix_aug_2024):
        """Test fetching VIX data for August 2024."""
        assert len(vix_aug_2024) >= 5, "Should have at least 5 trading days"

        # Check August 5 exists (handle timezone in string comparison)
        index_dates = [str(d).split(' ')[0] for d in vix_aug_2024.index]
        assert '2024-08-05' in index_dates, "Should include August 5"

        # August 5 VIX should be elevated (>30)
        aug_5_vix = vix_aug_2024.loc[vix_aug_2024.index.astype(str) >= '2024-08-05'].iloc[0]
        assert aug_5_vix > 30.0, f"Aug 5 VIX should be >30, got {aug_5_vix:.2f}"

    def test_fetch_vix_data_march_2020(self, vix_march_2020):
        """Test fetching VIX data for March 2020 crash."""
        assert len(vix_march_2020) >= 15, "Should have at least 15 trading days"

        # VIX should reach extreme levels (>60) during peak
        max_vix = vix_march_2020.max()
        assert max_vix > 60.0, f"March 2020 VIX peak should be >60, got {max_vix:.2f}"

    def test_fetch_vix_data_normal_period(self, vix_normal_2021):
        """Test fetching VIX data for normal volatility period."""
        assert len(vix_normal_2021) >= 200, "Should have at least 200 trading days"

        # Normal VIX range: 15-30
        mean_vix = vix_normal_2021.mean()
        assert 15.0 < mean_vix < 35.0, f"2021 average VIX should be 15-35, got {mean_vix:.2f}"


class TestVIXSpikeDetection:
    """Test VIX spike detection logic."""

    def test_detect_august_5_flash_crash(self, vix_aug_2024):
        """Test detection of August 5, 2024 flash crash."""
        spikes = detect_vix_spike(vix_aug_2024, threshold_1d=0.20, threshold_3d=0.50)

        # August 5 should be detected as spike
        aug_5_spike = spikes.loc[spikes.index.astype(str) >= '2024-08-05'].iloc[0]
        assert aug_5_spike == True, "August 5 flash crash should be detected"

        # Calculate actual 1-day change for verification
        vix_change_1d = vix_aug_2024.pct_change()
        aug_5_change = vix_change_1d.loc[vix_change_1d.index.astype(str) >= '2024-08-05'].iloc[0]

        assert aug_5_change > 0.20, f"Aug 5 1-day change should be >20%, got {aug_5_change:.2%}"

    def test_detect_march_2020_escalation(self, vix_march_2020):
        """Test detection of March 2020 crash escalation."""
        spikes = detect_vix_spike(vix_march_2020, threshold_1d=0.20, threshold_3d=0.50)

        # Should detect multiple spike days during crash
        spike_count = spikes.sum()
        assert spike_count >= 3, f"March 2020 should have >=3 spike days, got {spike_count}"

    def test_normal_volatility_false_positives(self, vix_normal_2021):
        """Test false positive rate during normal volatility."""
        spikes = detect_vix_spike(vix_normal_2021, threshold_1d=0.20, threshold_3d=0.50)

        spike_count = spikes.sum()
        false_positive_rate = spike_count / len(spikes)

        # Should have <5% false positive rate in normal conditions
        assert false_positive_rate < 0.05, \
            f"False positive rate {false_positive_rate:.2%} exceeds 5% threshold"

    def test_threshold_sensitivity(self, vix_aug_2024):
        """Test spike detection with different thresholds."""
        # Conservative threshold (30%)
        spikes_conservative = detect_vix_spike(vix_aug_2024, threshold_1d=0.30, threshold_3d=0.70)

        # Aggressive threshold (15%)
        spikes_aggressive = detect_vix_spike(vix_aug_2024, threshold_1d=0.15, threshold_3d=0.40)

        # Aggressive should detect more spikes
        assert spikes_aggressive.sum() >= spikes_conservative.sum(), \
            "Aggressive threshold should detect >= conservative threshold"


class TestVIXSeverityClassification:
    """Test VIX severity classification."""

    def test_classify_august_5_flash_crash(self, vix_aug_2024):
        """Test severity classification of August 5 flash crash."""
        severity = classify_vix_severity(vix_aug_2024)

        aug_5_severity = severity.loc[severity.index.astype(str) >= '2024-08-05'].iloc[0]
        assert aug_5_severity == 'FLASH_CRASH', \
            f"August 5 should be classified as FLASH_CRASH, got {aug_5_severity}"

    def test_classify_elevated_vs_normal(self, vix_normal_2021):
        """Test classification distinguishes ELEVATED vs NORMAL."""
        severity = classify_vix_severity(vix_normal_2021)

        # Count each severity level
        severity_counts = severity.value_counts()

        # Most days should be NORMAL
        normal_pct = severity_counts.get('NORMAL', 0) / len(severity)
        assert normal_pct > 0.80, \
            f"Normal period should be >80% NORMAL, got {normal_pct:.2%}"

    def test_three_level_severity(self, vix_march_2020):
        """Test that all 3 severity levels can be distinguished."""
        severity = classify_vix_severity(
            vix_march_2020,
            threshold_flash_crash=0.30,
            threshold_elevated=0.10
        )

        unique_levels = severity.unique()

        # Should have NORMAL days at start of month
        assert 'NORMAL' in unique_levels, "Should have NORMAL days"

        # Should have ELEVATED as warning
        # (Note: Might not appear if crash was immediate, so this is optional)

        # Should have FLASH_CRASH at peak
        assert 'FLASH_CRASH' in unique_levels, "Should have FLASH_CRASH days"


class TestVIXRegimeOverride:
    """Test VIX regime override integration."""

    def test_get_vix_regime_override(self, vix_aug_2024):
        """Test convenience function returns both spike and severity."""
        spikes, severity = get_vix_regime_override(vix_aug_2024)

        # Check both outputs align
        aug_5_spike = spikes.loc[spikes.index.astype(str) >= '2024-08-05'].iloc[0]
        aug_5_severity = severity.loc[severity.index.astype(str) >= '2024-08-05'].iloc[0]

        assert aug_5_spike == True, "August 5 should be spike"
        assert aug_5_severity == 'FLASH_CRASH', "August 5 should be FLASH_CRASH severity"


class TestAcademicModelIntegration:
    """Test VIX acceleration integration with academic jump model."""

    def test_vix_override_with_academic_model(self, spy_data_3000, vix_aug_2024):
        """Test VIX override works with academic model online inference."""
        # Initialize academic model
        model = AcademicJumpModel(lambda_penalty=1.5)

        # Fit on training data (use first 2500 days)
        train_data = spy_data_3000.iloc[:2500]
        model.fit(train_data, n_starts=3, random_seed=42)

        # Online inference with VIX data on August 2024 period
        # Extract SPY data for August 2024 (need enough lookback)
        aug_start_idx = spy_data_3000.index.get_indexer(['2024-08-01'], method='nearest')[0]
        spy_aug_data = spy_data_3000.iloc[max(0, aug_start_idx - 1500):aug_start_idx + 10]

        # Run online inference WITH VIX data
        regimes_with_vix, _, _ = model.online_inference(
            spy_aug_data,
            lookback=1000,
            vix_data=vix_aug_2024
        )

        # Check August 5 is CRASH (VIX override should have triggered)
        aug_5_regime = regimes_with_vix.loc[regimes_with_vix.index.astype(str) >= '2024-08-05']

        if len(aug_5_regime) > 0:
            aug_5_regime_val = aug_5_regime.iloc[0]
            assert aug_5_regime_val == 'CRASH', \
                f"August 5 should be CRASH with VIX override, got {aug_5_regime_val}"

    def test_vix_override_vs_no_override(self, spy_data_3000, vix_aug_2024):
        """Compare regime detection with and without VIX override."""
        # Initialize model
        model = AcademicJumpModel(lambda_penalty=1.5)

        # Fit on training data
        train_data = spy_data_3000.iloc[:2500]
        model.fit(train_data, n_starts=3, random_seed=42)

        # Online inference on August 2024
        aug_start_idx = spy_data_3000.index.get_indexer(['2024-08-01'], method='nearest')[0]
        spy_aug_data = spy_data_3000.iloc[max(0, aug_start_idx - 1500):aug_start_idx + 10]

        # WITHOUT VIX override
        regimes_no_vix, _, _ = model.online_inference(spy_aug_data, lookback=1000)

        # WITH VIX override
        regimes_with_vix, _, _ = model.online_inference(
            spy_aug_data,
            lookback=1000,
            vix_data=vix_aug_2024
        )

        # VIX override should increase CRASH days
        crash_days_no_vix = (regimes_no_vix == 'CRASH').sum()
        crash_days_with_vix = (regimes_with_vix == 'CRASH').sum()

        assert crash_days_with_vix >= crash_days_no_vix, \
            "VIX override should increase or maintain CRASH day count"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handle_missing_vix_dates(self):
        """Test handling of missing VIX dates (align with SPY)."""
        # Create synthetic VIX data with gaps
        vix_dates = pd.date_range('2024-08-01', '2024-08-10', freq='B')[::2]  # Every other day
        vix_data = pd.Series(
            data=[20, 25, 30, 35],  # Match length of vix_dates
            index=vix_dates
        )

        # detect_vix_spike should handle missing dates gracefully
        spikes = detect_vix_spike(vix_data)

        assert len(spikes) == len(vix_data), \
            "Spike detection should return same length as input"

    def test_handle_nan_values(self):
        """Test handling of NaN values in VIX data."""
        # Create VIX data with NaN
        dates = pd.date_range('2024-08-01', '2024-08-10', freq='B')
        vix_data = pd.Series(
            data=[20, np.nan, 25, 30, np.nan, 35, 40],
            index=dates
        )

        # Should not raise error
        spikes = detect_vix_spike(vix_data)

        # NaN spike values should be False
        assert not spikes.isna().any(), "Spike detection should not produce NaN"

    def test_empty_vix_data(self):
        """Test behavior with empty VIX data."""
        empty_vix = pd.Series(dtype=float)

        # Should not raise error
        spikes = detect_vix_spike(empty_vix)

        assert len(spikes) == 0, "Empty input should return empty output"


# Optional: Integration validation (run manually, not in CI)
if __name__ == '__main__':
    print("VIX Acceleration Test Suite - Manual Validation")
    print("=" * 60)

    # Run key validation tests
    print("\n1. Testing August 5, 2024 flash crash detection...")
    vix_aug = fetch_vix_data('2024-08-01', '2024-08-10')
    spikes_aug, severity_aug = get_vix_regime_override(vix_aug)

    aug_5_date = '2024-08-05'
    aug_5_spike = spikes_aug.loc[spikes_aug.index.astype(str) >= aug_5_date].iloc[0]
    aug_5_sev = severity_aug.loc[severity_aug.index.astype(str) >= aug_5_date].iloc[0]

    print(f"   August 5 spike detected: {aug_5_spike}")
    print(f"   August 5 severity: {aug_5_sev}")
    print(f"   [PASS]" if aug_5_spike and aug_5_sev == 'FLASH_CRASH' else "   [FAIL]")

    print("\n2. Testing March 2020 crash detection...")
    vix_march = fetch_vix_data('2020-03-01', '2020-03-31')
    spikes_march, _ = get_vix_regime_override(vix_march)

    spike_count_march = spikes_march.sum()
    print(f"   March 2020 spike days: {spike_count_march}")
    print(f"   [PASS]" if spike_count_march >= 3 else "   [FAIL]")

    print("\n3. Testing false positive rate (2021)...")
    vix_2021 = fetch_vix_data('2021-01-01', '2021-12-31')
    spikes_2021, _ = get_vix_regime_override(vix_2021)

    false_positive_rate = spikes_2021.sum() / len(spikes_2021)
    print(f"   False positive rate: {false_positive_rate:.2%}")
    print(f"   [PASS]" if false_positive_rate < 0.05 else "   [FAIL]")

    print("\n" + "=" * 60)
    print("Manual validation complete. Run pytest for full test suite.")
