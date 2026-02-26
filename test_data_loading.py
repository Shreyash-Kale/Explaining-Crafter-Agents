#!/usr/bin/env python3
"""Test script to verify decision attribution data loading"""

import sys
import os
import numpy as np
sys.path.insert(0, '/Users/sirius/Desktop/Workspace/Crafter - Editable')

from vis.data_manager import DataManager

# Test loading data
csv_path = '/Users/sirius/Desktop/Workspace/Crafter - Editable/archive/run_20260218_222805/logs/event_log_17.03_12.14.10.csv'

print("Creating DataManager...")
dm = DataManager()

print(f"Loading CSV: {csv_path}")
success = dm.load_data(csv_path)

if success:
    print(f"✓ Data loaded successfully")
    print(f"  Time steps: {len(dm.time_steps)}")
    print(f"  Action prob samples: {len(dm.action_prob)}")
    print(f"  Value estimate samples: {len(dm.value_est)}")
    print(f"  PPO entropy samples: {len(dm.ppo_entropy)}")
    print(f"  Dreamer explore samples: {len(dm.dreamer_explore)}")
    
    # Check if decision attribution data was loaded
    if len(dm.action_prob) > 0 and np.any(dm.action_prob):
        print(f"✓ Action probability data is available")
    else:
        print(f"✗ No action probability data found")
        
    if len(dm.value_est) > 0 and np.any(dm.value_est):
        print(f"✓ Value estimate data is available")
    else:
        print(f"✗ No value estimate data found")
else:
    print(f"✗ Failed to load data")

print("\nDone!")
