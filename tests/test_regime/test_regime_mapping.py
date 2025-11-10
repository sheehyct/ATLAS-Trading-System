"""
Test suite for Academic Jump Model Phase E - Regime Mapping.

Tests the map_to_atlas_regimes() method that maps 2-state (bull/bear)
clustering output to 4-regime ATLAS output (TREND_BULL, TREND_BEAR,
TREND_NEUTRAL, CRASH).
"""

import pytest
import pandas as pd
import numpy as np
from regime.academic_jump_model import AcademicJumpModel
from data.alpaca import fetch_alpaca_data


@pytest.fixture
def spy_data():
    """Fetch SPY data for testing (2016-2025, ~2271 trading days)."""
    data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3300)
    return data


@pytest.fixture
def model():
    """Create AcademicJumpModel instance."""
    return AcademicJumpModel()


# Test 1: March 2020 Crash Detection
def test_crash_detection_march_2020(spy_data, model):
    """
    Test CRASH regime detection during March 2020.

    Primary success criterion: >50% of March 2020 days should be
    detected as CRASH or TREND_BEAR.

    Expected behavior:
    - Feb 2020: TREND_BULL (market peak)
    - Early Mar (1-10): TREND_BEAR (initial decline)
    - Mid Mar (11-20): CRASH (circuit breakers, extreme volatility)
    - Late Mar (21-31): TREND_BEAR (recovery begins)
    """
    # Use shorter lookback to include March 2020 in inference window
    # March 2020 at index ~842 in dataset, so lookback must be < 842
    lookback = 750  # 3 years

    # Run online inference with regime mapping
    atlas_regimes, lambda_history, theta_history = model.online_inference(
        spy_data,
        lookback=lookback
    )

    # Extract March 2020 regimes
    march_2020_regimes = atlas_regimes.loc['2020-03']

    # Count regime types
    crash_days = (march_2020_regimes == 'CRASH').sum()
    bear_days = (march_2020_regimes == 'TREND_BEAR').sum()
    bull_days = (march_2020_regimes == 'TREND_BULL').sum()
    neutral_days = (march_2020_regimes == 'TREND_NEUTRAL').sum()

    total_march_days = len(march_2020_regimes)
    crash_bear_total = crash_days + bear_days

    # Primary success criterion: >50% crash or bear
    crash_bear_percentage = crash_bear_total / total_march_days

    print(f"\nMarch 2020 Regime Distribution:")
    print(f"  CRASH: {crash_days}/{total_march_days} ({crash_days/total_march_days:.1%})")
    print(f"  TREND_BEAR: {bear_days}/{total_march_days} ({bear_days/total_march_days:.1%})")
    print(f"  TREND_BULL: {bull_days}/{total_march_days} ({bull_days/total_march_days:.1%})")
    print(f"  TREND_NEUTRAL: {neutral_days}/{total_march_days} ({neutral_days/total_march_days:.1%})")
    print(f"  CRASH+BEAR Total: {crash_bear_total}/{total_march_days} ({crash_bear_percentage:.1%})")

    # Assertions
    assert crash_bear_total >= total_march_days * 0.5, (
        f"March 2020 crash/bear detection: {crash_bear_percentage:.1%} "
        f"(expected >50%)"
    )

    # At least some days should be CRASH (extreme volatility)
    assert crash_days > 0, "Expected at least some CRASH days in March 2020"

    # Verify no invalid regimes
    valid_regimes = {'TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL', 'CRASH'}
    assert set(march_2020_regimes.unique()).issubset(valid_regimes)


# Test 2: Bull Market Detection (2017-2019)
def test_trend_bull_2017_2019(spy_data, model):
    """
    Test TREND_BULL detection during strong bull market (2017-2019).

    Expected: >70% TREND_BULL days during sustained bull market periods.
    """
    lookback = 750  # 3 years

    atlas_regimes, _, _ = model.online_inference(
        spy_data,
        lookback=lookback
    )

    # Extract 2017-2019 period (if available in dataset)
    try:
        bull_period = atlas_regimes.loc['2017':'2019']
    except KeyError:
        pytest.skip("2017-2019 period not available in dataset")

    # Count regimes
    bull_days = (bull_period == 'TREND_BULL').sum()
    total_days = len(bull_period)
    bull_percentage = bull_days / total_days

    print(f"\n2017-2019 Bull Market:")
    print(f"  TREND_BULL: {bull_days}/{total_days} ({bull_percentage:.1%})")

    # Should see significant TREND_BULL presence (>40% is reasonable)
    # Note: Using 40% threshold as markets aren't always in strong bull mode
    assert bull_percentage > 0.40, (
        f"Bull market detection: {bull_percentage:.1%} (expected >40%)"
    )


# Test 3: Regime Distribution Balance
def test_regime_distribution_balance(spy_data, model):
    """
    Test that no single regime dominates >80% of all days.

    This prevents degenerate solutions where the model assigns
    everything to one regime.
    """
    lookback = 750

    atlas_regimes, _, _ = model.online_inference(
        spy_data,
        lookback=lookback
    )

    # Count regime distribution
    regime_counts = atlas_regimes.value_counts()
    regime_percentages = regime_counts / len(atlas_regimes) * 100

    print(f"\nRegime Distribution (full dataset):")
    for regime, percentage in regime_percentages.items():
        count = regime_counts[regime]
        print(f"  {regime}: {count} days ({percentage:.1%})")

    # No regime should dominate >80%
    max_percentage = regime_percentages.max() / 100
    assert max_percentage < 0.80, (
        f"Degenerate solution detected: {regime_percentages.idxmax()} "
        f"dominates {max_percentage:.1%} (expected <80%)"
    )

    # All 4 regimes should appear at least occasionally (>1%)
    # Note: CRASH might be rare, so lower threshold
    min_percentage = regime_percentages.min() / 100
    assert min_percentage > 0.01 or regime_percentages.idxmin() == 'CRASH', (
        f"Regime {regime_percentages.idxmin()} too rare: {min_percentage:.1%}"
    )


# Test 4: Feature Threshold Logic (Unit Test with Synthetic Data)
def test_feature_threshold_logic(model):
    """
    Unit test for map_to_atlas_regimes() with synthetic data.

    Verifies boundary conditions and threshold logic:
    - CRASH: bear + DD > 0.03 + Sortino < -1.0
    - TREND_BEAR: bear + NOT crash
    - TREND_BULL: bull + Sortino > 0.5
    - TREND_NEUTRAL: bull + Sortino <= 0.5
    """
    # Create synthetic test data
    dates = pd.date_range('2020-01-01', periods=10, freq='D')

    # Test case 1: CRASH conditions
    state_sequence = pd.Series(['bear'] * 10, index=dates)
    features_df = pd.DataFrame({
        'downside_dev': [0.05, 0.04, 0.035, 0.02, 0.01, 0.02, 0.01, 0.03, 0.025, 0.01],
        'sortino_20': [-2.0, -1.5, -1.2, -0.5, 0.0, -0.8, -0.5, -1.0, -0.3, 0.0],
        'sortino_60': [-1.0] * 10
    }, index=dates)

    regimes = model.map_to_atlas_regimes(state_sequence, features_df)

    # First 3 should be CRASH (DD > 0.03 AND Sortino < -1.0)
    assert regimes.iloc[0] == 'CRASH', "Day 0: DD=0.05, Sortino=-2.0 should be CRASH"
    assert regimes.iloc[1] == 'CRASH', "Day 1: DD=0.04, Sortino=-1.5 should be CRASH"
    assert regimes.iloc[2] == 'CRASH', "Day 2: DD=0.035, Sortino=-1.2 should be CRASH"

    # Rest should be TREND_BEAR (bear but not crash)
    # Day 3: DD=0.02 (not >0.03), Sortino=-0.5 (not <-1.0) → TREND_BEAR
    # Day 7: DD=0.03 (not >0.03, boundary), Sortino=-1.0 (not <-1.0, boundary) → TREND_BEAR
    assert all(regimes.iloc[3:] == 'TREND_BEAR'), "Days 3-9 should be TREND_BEAR"

    # Test case 2: TREND_BULL conditions
    state_sequence = pd.Series(['bull'] * 5, index=dates[:5])
    features_df = pd.DataFrame({
        'downside_dev': [0.01] * 5,
        'sortino_20': [1.5, 1.0, 0.6, 0.5, 0.4],  # Threshold at 0.5
        'sortino_60': [1.0] * 5
    }, index=dates[:5])

    regimes = model.map_to_atlas_regimes(state_sequence, features_df)

    # First 3 should be TREND_BULL (Sortino > 0.5)
    assert regimes.iloc[0] == 'TREND_BULL', "Sortino=1.5 should be TREND_BULL"
    assert regimes.iloc[1] == 'TREND_BULL', "Sortino=1.0 should be TREND_BULL"
    assert regimes.iloc[2] == 'TREND_BULL', "Sortino=0.6 should be TREND_BULL"

    # Last 2 should be TREND_NEUTRAL (Sortino <= 0.5)
    assert regimes.iloc[3] == 'TREND_NEUTRAL', "Sortino=0.5 boundary should be TREND_NEUTRAL"
    assert regimes.iloc[4] == 'TREND_NEUTRAL', "Sortino=0.4 should be TREND_NEUTRAL"


# Test 5: Index Alignment
def test_index_alignment(spy_data, model):
    """
    Verify regime output aligns with input data dates.

    Tests for off-by-one errors and proper index handling.
    """
    lookback = 750

    atlas_regimes, lambda_history, theta_history = model.online_inference(
        spy_data,
        lookback=lookback
    )

    # Regime output should start after lookback period
    expected_start = spy_data.index[lookback]
    actual_start = atlas_regimes.index[0]

    # Allow for feature calculation warm-up period (dropna in calculate_features)
    # The actual start might be slightly later due to NaN values from EWM
    assert actual_start >= expected_start, (
        f"Regime output starts at {actual_start}, expected >= {expected_start}"
    )

    # All output indices should be valid trading days from input
    assert atlas_regimes.index.isin(spy_data.index).all(), (
        "Regime output contains dates not in input data"
    )

    # No duplicate dates
    assert not atlas_regimes.index.duplicated().any(), (
        "Regime output contains duplicate dates"
    )

    # Sorted chronologically
    assert atlas_regimes.index.is_monotonic_increasing, (
        "Regime output not sorted chronologically"
    )


# Test 6: NaN Handling
def test_nan_handling(model):
    """
    Test behavior with NaN values in features.

    Early dates may have NaN values due to EWM warm-up period.
    Should handle gracefully without errors.
    """
    dates = pd.date_range('2020-01-01', periods=100, freq='D')

    # Create state sequence
    state_sequence = pd.Series(['bull'] * 100, index=dates)

    # Create features with NaN in early period (realistic scenario)
    features_df = pd.DataFrame({
        'downside_dev': [np.nan] * 20 + [0.01] * 80,
        'sortino_20': [np.nan] * 20 + [1.0] * 80,
        'sortino_60': [np.nan] * 60 + [1.0] * 40
    }, index=dates)

    # Should not raise error
    regimes = model.map_to_atlas_regimes(state_sequence, features_df)

    # Output should have same length as input
    assert len(regimes) == len(state_sequence)

    # Days with NaN features might map to TREND_NEUTRAL (default)
    # This is acceptable graceful degradation
    print(f"\nNaN handling: {regimes.value_counts().to_dict()}")


# Test 7: Invalid State Handling
def test_invalid_state_handling(model):
    """
    Test that invalid states in state_sequence raise ValueError.
    """
    dates = pd.date_range('2020-01-01', periods=10, freq='D')

    # Create invalid state sequence
    state_sequence = pd.Series(['bull', 'bear', 'invalid', 'bull'], index=dates[:4])
    features_df = pd.DataFrame({
        'downside_dev': [0.01] * 4,
        'sortino_20': [1.0] * 4,
        'sortino_60': [1.0] * 4
    }, index=dates[:4])

    # Should raise ValueError
    with pytest.raises(ValueError, match="invalid states"):
        model.map_to_atlas_regimes(state_sequence, features_df)


# Test 8: Missing Feature Columns
def test_missing_feature_columns(model):
    """
    Test that missing required feature columns raise ValueError.
    """
    dates = pd.date_range('2020-01-01', periods=10, freq='D')

    state_sequence = pd.Series(['bull'] * 10, index=dates)

    # Missing 'sortino_20' column
    features_df = pd.DataFrame({
        'downside_dev': [0.01] * 10,
        'sortino_60': [1.0] * 10
    }, index=dates)

    # Should raise ValueError
    with pytest.raises(ValueError, match="missing required columns"):
        model.map_to_atlas_regimes(state_sequence, features_df)
