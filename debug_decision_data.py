#!/usr/bin/env python3
"""Debug script to check what's in the merged dataframe"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/sirius/Desktop/Workspace/Crafter - Editable')

from vis.data_manager import DataManager

csv_path = '/Users/sirius/Desktop/Workspace/Crafter - Editable/archive/run_20260218_222805/logs/event_log_30.03_22.57.54.csv'

print("Loading data...")
dm = DataManager()
dm.load_data(csv_path)

print(f"\n📊 Dataframe info:")
print(f"  Shape: {dm.event_df.shape}")
print(f"  Columns: {list(dm.event_df.columns)}")

print(f"\n🔍 Decision attribution columns:")
for col in ['action_probability', 'value_estimate', 'exploration_bonus', 'world_model_score']:
    if col in dm.event_df.columns:
        series = dm.event_df[col]
        print(f"  {col}:")
        print(f"    Type: {series.dtype}")
        print(f"    Non-null: {series.notna().sum()} / {len(series)}")
        print(f"    Min: {series.min()}, Max: {series.max()}")
        print(f"    Sample values: {series.dropna().head(3).tolist()}")
    else:
        print(f"  {col}: NOT FOUND")

print(f"\n📈 Extracted arrays:")
print(f"  action_prob: len={len(dm.action_prob)}, has_values={len(dm.action_prob) > 0 and np.any(np.isfinite(dm.action_prob))}")
print(f"  value_est: len={len(dm.value_est)}, has_values={len(dm.value_est) > 0 and np.any(np.isfinite(dm.value_est))}")
print(f"  dreamer_explore: len={len(dm.dreamer_explore)}, has_values={len(dm.dreamer_explore) > 0 and np.any(np.isfinite(dm.dreamer_explore))}")
print(f"  dreamer_wm_score: len={len(dm.dreamer_wm_score)}, has_values={len(dm.dreamer_wm_score) > 0 and np.any(np.isfinite(dm.dreamer_wm_score))}")

print(f"\n⚠️  Normalized arrays:")
print(f"  action_prob_norm: len={len(dm.action_prob_norm)}, has_values={np.any(dm.action_prob_norm) if len(dm.action_prob_norm) > 0 else False}")
print(f"  value_norm: len={len(dm.value_norm)}, has_values={np.any(dm.value_norm) if len(dm.value_norm) > 0 else False}")
print(f"  dreamer_explore_norm: len={len(dm.dreamer_explore_norm)}, has_values={np.any(dm.dreamer_explore_norm) if len(dm.dreamer_explore_norm) > 0 else False}")
print(f"  dreamer_wm_score_norm: len={len(dm.dreamer_wm_score_norm)}, has_values={np.any(dm.dreamer_wm_score_norm) if len(dm.dreamer_wm_score_norm) > 0 else False}")

print("\n✓ Debug complete")
