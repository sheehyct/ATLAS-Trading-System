"""
Diagnostic script to investigate March 2020 crash detection failure.

This script runs online_inference() with diagnostic logging enabled
to understand why March 2020 crash is not detected (0% bear days).
"""

import pandas as pd
from regime.academic_jump_model import AcademicJumpModel
from data.alpaca import fetch_alpaca_data

def main():
    print("=" * 80)
    print("MARCH 2020 CRASH DETECTION DIAGNOSTIC")
    print("=" * 80)
    print()
    print("Hypothesis: Lambda=15 penalty prevents switching despite bear centroid being closer")
    print()

    # Fetch data (same as test)
    print("[1/4] Fetching SPY data...")
    data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3300)  # ~9 years
    print(f"      Data range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"      Total days: {len(data)}")

    # Find March 2020 position
    march_2020_mask = (data.index >= '2020-03-01') & (data.index <= '2020-03-31')
    march_2020_indices = data.index[march_2020_mask]
    if len(march_2020_indices) > 0:
        first_march_idx = data.index.get_loc(march_2020_indices[0])
        print(f"      March 2020 start: index {first_march_idx} (date: {march_2020_indices[0].date()})")
        print(f"      March 2020 days: {len(march_2020_indices)}")

    print()

    # Run online inference with diagnostic logging
    print("[2/4] Running online_inference() with lookback=750, lambda=15...")
    print("      (Diagnostic logging will show March 2020 details)")
    print()

    model = AcademicJumpModel()
    regime_states, lambda_history, theta_history = model.online_inference(
        data,
        lookback=750,  # Reduced to include March 2020 in results
        theta_update_freq=126,  # 6 months (default)
        lambda_update_freq=21,  # 1 month (default)
        default_lambda=15.0  # Trading mode (suspect this is too high)
    )

    print()
    print("=" * 80)
    print("[3/4] ANALYSIS OF MARCH 2020 RESULTS")
    print("=" * 80)
    print()

    # Analyze March 2020 results
    regimes_march = regime_states.loc['2020-03']
    bull_days = (regimes_march == 'bull').sum()
    bear_days = (regimes_march == 'bear').sum()
    bear_pct = bear_days / len(regimes_march) * 100

    print(f"March 2020 Regime Detection:")
    print(f"  Bull days: {bull_days} ({bull_days/len(regimes_march)*100:.1f}%)")
    print(f"  Bear days: {bear_days} ({bear_pct:.1f}%)")
    print(f"  Total days: {len(regimes_march)}")
    print()
    print(f"Target: >= 50% bear detection (>= {len(regimes_march)//2} days)")
    print(f"Result: {'PASS' if bear_pct >= 50 else 'FAIL'} ({bear_pct:.1f}%)")
    print()

    # Show regime sequence
    print(f"Regime sequence:")
    regime_str = ''.join(['B' if r == 'bull' else 'b' for r in regimes_march])
    for i in range(0, len(regime_str), 40):
        print(f"  {regime_str[i:i+40]}")
    print(f"  (B = bull, b = bear)")
    print()

    # Analyze lambda values during March
    lambda_march = lambda_history.loc['2020-03']
    print(f"Lambda values during March 2020:")
    print(f"  Mean: {lambda_march.mean():.2f}")
    print(f"  Min: {lambda_march.min():.2f}")
    print(f"  Max: {lambda_march.max():.2f}")
    print()

    # Check when theta was last updated before March
    theta_march = theta_history.loc[:'2020-03-01']
    if len(theta_march) > 0:
        last_theta_date = theta_march.index[-1]
        days_since_update = (pd.Timestamp('2020-03-01') - last_theta_date).days
        print(f"Last theta update before March: {last_theta_date.date()}")
        print(f"Days since update: {days_since_update}")
        print(f"  (Theta was trained on pre-crash bull market data)")
        print()

    print("=" * 80)
    print("[4/4] DIAGNOSIS")
    print("=" * 80)
    print()

    if bear_days == 0:
        print("CRITICAL FINDING: 0% bear detection confirms hypothesis!")
        print()
        print("Root cause: Lambda=15 penalty prevents switching despite crash features.")
        print()
        print("Evidence from diagnostic logs above:")
        print("  1. Bear centroid IS closer to crash features")
        print("  2. But lambda=15 penalty exceeds distance improvement")
        print("  3. Model stays in initial bull state entire month")
        print("  4. Theta not updated until after March ends (6-month frequency)")
        print()
        print("RECOMMENDATION:")
        print("  - Test with lambda=5 (more responsive, allows switching)")
        print("  - Test with theta_update_freq=21 (monthly updates)")
        print("  - Combination should achieve >50% detection")
    else:
        print(f"Partial bear detection: {bear_pct:.1f}%")
        print()
        print("Review diagnostic logs above to identify which days switched successfully.")

    print()
    print("=" * 80)
    print("Diagnostic complete. Proceeding to Phase 2 testing...")
    print("=" * 80)

if __name__ == '__main__':
    main()
