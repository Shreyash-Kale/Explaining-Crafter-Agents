import glob, pandas as pd, os

ckpt = 290000
base_dir = f'/Users/sirius/Desktop/Crafter/results/dreamer_v2/checkpoint_{ckpt}'
episode_dirs = sorted(glob.glob(os.path.join(base_dir, 'episode_*')))

rewards = []
for ep_dir in episode_dirs:
    # each ep_dir holds exactly one CSV file
    csv = glob.glob(os.path.join(ep_dir, '*.csv'))[0]
    df = pd.read_csv(csv)
    # adjust column name if needed
    if 'total_reward' in df.columns:
        total = df['total_reward'].iloc[-1]
    else:
        total = df['reward'].sum()  # sum per-step rewards
    
    rewards.append(total)

# now compute mean and std
import numpy as np
rewards = np.array(rewards)
print(f"Dreamer @{ckpt}: mean = {rewards.mean():.2f}, std = {rewards.std():.2f}")
