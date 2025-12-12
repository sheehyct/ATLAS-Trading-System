"""
Quick test to verify label mapping bug hypothesis.

Tests if state_labels_ changes during online_inference and causes
retroactive remapping of historical states.
"""

import pandas as pd
import numpy as np
from regime.academic_jump_model import AcademicJumpModel
from data.alpaca import fetch_alpaca_data

# Fetch data
data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3300)

# Create model and run inference
model = AcademicJumpModel()

# Track state_labels_ during inference by monkey-patching
label_history = []
original_update_theta = model._update_theta_online

def tracked_update_theta(features, lambda_value, n_starts=10):
    result = original_update_theta(features, lambda_value, n_starts)
    label_history.append({
        'labels': model.state_labels_.copy(),
        'theta': result.copy()
    })
    return result

model._update_theta_online = tracked_update_theta

# Run online inference
regime_states, lambda_series, theta_df = model.online_inference(
    data,
    lookback=750,
    theta_update_freq=126,
    lambda_update_freq=21,
    default_lambda=15.0
)

print("=" * 80)
print("LABEL MAPPING BUG INVESTIGATION")
print("=" * 80)
print()

print(f"Total theta updates: {len(label_history)}")
print()

# Check if labels flipped
if len(label_history) >= 2:
    print("Label history:")
    for i, entry in enumerate(label_history):
        print(f"  Update {i}: {entry['labels']}")
        print(f"    Theta[0]: DD={entry['theta'][0,0]:.4f}, S20={entry['theta'][0,1]:.3f}")
        print(f"    Theta[1]: DD={entry['theta'][1,0]:.4f}, S20={entry['theta'][1,1]:.3f}")

    print()

    # Check if labels changed
    labels_changed = False
    for i in range(1, len(label_history)):
        if label_history[i]['labels'] != label_history[i-1]['labels']:
            labels_changed = True
            print(f"[CRITICAL] Labels CHANGED between update {i-1} and {i}!")
            print(f"  Before: {label_history[i-1]['labels']}")
            print(f"  After:  {label_history[i]['labels']}")

    if not labels_changed:
        print("[INFO] Labels remained consistent throughout inference")

    print()

print(f"Final state_labels_: {model.state_labels_}")
print()

# Analyze March 2020
regimes_march = regime_states.loc['2020-03']
bear_days = (regimes_march == 'bear').sum()
bull_days = (regimes_march == 'bull').sum()

print(f"March 2020 results:")
print(f"  Bull: {bull_days} days ({bull_days/len(regimes_march)*100:.1f}%)")
print(f"  Bear: {bear_days} days ({bear_days/len(regimes_march)*100:.1f}%)")
print()

print("=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print()

if labels_changed:
    print("BUG CONFIRMED: Labels changed during inference!")
    print()
    print("Impact: Historical states get retroactively remapped using final labels.")
    print("Example: If March used labels {0:'bear'} but final is {0:'bull'},")
    print("         then March states (which were 0) get mapped to 'bull'!")
    print()
    print("FIX REQUIRED: Store labels with each state instead of applying final labels to all history.")
else:
    print("Labels were consistent. Bug hypothesis INCORRECT.")
    print("Need to investigate alternative explanation for 0% bear detection.")
