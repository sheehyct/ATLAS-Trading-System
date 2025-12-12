"""Test March 2020 with lambda=1 to see if lower penalty allows switching."""

import pandas as pd
from regime.academic_jump_model import AcademicJumpModel
from data.alpaca import fetch_alpaca_data

# Fetch data
data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3300)

# Run with lambda=1
model = AcademicJumpModel()
regime_states, _, _ = model.online_inference(
    data,
    lookback=750,
    default_lambda=1.0  # Very low penalty
)

# Analyze March 2020
regimes_march = regime_states.loc['2020-03']
bear_days = (regimes_march == 'bear').sum()
bull_days = (regimes_march == 'bull').sum()
bear_pct = bear_days / len(regimes_march) * 100

print("=" * 80)
print("TEST: Lambda=1.0 for March 2020")
print("=" * 80)
print(f"Bull days: {bull_days} ({bull_days/len(regimes_march)*100:.1f}%)")
print(f"Bear days: {bear_days} ({bear_pct:.1f}%)")
print(f"Total: {len(regimes_march)}")
print()
print(f"Target: >=50% bear (>= {len(regimes_march)//2} days)")
print(f"Result: {'PASS' if bear_pct >= 50 else 'FAIL'}")
print("=" * 80)
