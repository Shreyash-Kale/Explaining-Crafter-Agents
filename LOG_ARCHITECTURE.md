# Crafter RL Logging & Visualization Architecture

## 1. Why 3 Menu Options to Open Files?

The three menu options provide different workflows for loading episodes:

```
File Menu:
├─ Open Random Log and Video
│  └─ Randomly picks one episode from the logs/ directory
│     (Quick exploratory workflow - good for discovering interesting episodes)
│
├─ Open from Results ► (submenu)
│  ├─ Browse Results Directory...
│  │  └─ File dialog to navigate results/dreamer_v2/checkpoint_*/episode_*/
│  └─ Recent Episodes (auto-populated if available)
│     └─ Quick access to recently generated episodes
│
└─ Open from Logs Directory...
   └─ File dialog to browse custom log directories
      (For when logs are in non-standard locations)
```

**Why 3 options?**
- **Random**: Fast exploration without browsing
- **Results**: Organized by training checkpoint/episode structure
- **Logs**: Flexible for custom log locations or batch-loaded episodes

---

## 2. PPO vs Dreamer: What's in Your Archive?

### No PPO Logs Present ❌
Your archive contains **only Dreamer training**, no PPO logs.
- PPO is referenced in `default_folders/examples/run_ppo.py` (reference implementation only)
- Your actual training used DreamerV2 (see `run_20260218_222805`)

### Decision Attribution: Dreamer vs PPO

Decision attribution data **differs by algorithm**:

| Component | Dreamer | PPO |
|-----------|---------|-----|
| Action Probability | ✅ `action_probability` | ✅ `action_probability` |
| Value Estimate | ✅ `value_estimate` | ✅ `value_estimate` |
| Exploration Signal | ✅ `exploration_bonus` | ❌ (not recorded) |
| World-Model Signal | ✅ `world_model_score` | ❌ (not recorded) |
| Policy Entropy | ❌ (not recorded) | ✅ `entropy` |
| Advantage | ❌ (not recorded) | ✅ `advantage` |

The viz automatically detects which columns are present and displays the appropriate metrics.

---

## 3. Complete Log File Architecture

### Directory Structure

```
archive/run_20260218_222805/
├── logs/                          # Episode recordings (event_logs + videos)
│   ├── event_log_30.03_22.57.54.csv    # One episode = one CSV
│   ├── event_log_30.03_23.02.27.csv    # Format: time_step | action | reward | ...
│   ├── 20250314T103326-ach4-len194.mp4 # Corresponding
│   └── ... (many episodes)
│
├── training_logs/                 # Organized by checkpoint step
│   ├── log_90000/                 # Training checkpoint at step 90,000
│   │   ├── decision_attribution.csv    # 90K samples of policy decisions
│   │   ├── dreamer_training_log_90k.csv # Training metrics snapshot
│   │   ├── dreamer_training.txt         # Console output
│   │   └── dreamer_training_metrics.png # Loss curves
│   ├── log_96000/
│   ├── log_150000/
│   ├── log_200000/
│   ├── log_230000/
│   ├── log_290000/
│   └── all_training_csv_logs/     # Consolidated training data
│
└── results/                       # Organized by checkpoint & episode
    └── dreamer_v2/
        ├── checkpoint_90000/
        │   ├── episode_0/
        │   │   ├── data.csv
        │   │   └── video.mp4
        │   └── episode_1/
        └── checkpoint_200000/
            ├── episode_0/
            └── ...
```

---

## 4. What's in Each Log Type?

### A. Event Log CSV (One per Episode)

**File**: `logs/event_log_30.03_22.57.54.csv`  
**Source**: Recorded during a single episode playthrough  
**Rows**: One per timestep (typically 100-300 rows per episode)  

**Columns (42 total)**:

```
Metadata:
- time_step               # 0, 1, 2, ... N (episode progress)
- action                  # 0-16 (action ID taken at this step)
- reward                  # -1.0 to 3.0 (immediate reward this step)
- cumulative_reward       # Running sum of rewards

State Info:
- health, food, drink, energy  # Life support resources (0-9 range)

Inventory (Resources):
- sapling, wood, stone, coal, iron, diamond  # Materials collected
- wood_pickaxe, stone_pickaxe, iron_pickaxe  # Tools crafted
- wood_sword, stone_sword, iron_sword         # Weapons crafted

Achievements (Binary: 0 or 1):
- collect_coal, collect_diamond, ..., place_table, wake_up  # 22 achievements
```

**Example Row**:
```csv
time_step=15, action=9, reward=0.0, cumulative_reward=1.5, 
health=9, food=9, ..., wood=3, stone=1, ..., 
collect_wood=1, place_table=0, ...
```

---

### B. Decision Attribution CSV (Per Training Checkpoint)

**File**: `training_logs/log_290000/decision_attribution.csv`  
**Source**: Computed during training every ~1000 training steps  
**Rows**: ~200 samples (one per ~1000 training timesteps)  

**Columns (6)**:

```
step                  # Training step (90000, 91000, 92000, ...)
action_taken          # The action the policy chose
action_probability    # P(action | state) from policy
world_model_score     # How confident the world-model was (0-1)
exploration_bonus     # RND/novelty bonus (typically const ~2.8)
value_estimate        # V(state) from critic head (-0.3 to 0.2)
```

**Example**:
```csv
step,action_taken,action_probability,world_model_score,exploration_bonus,value_estimate
230000,5,0.058290,0.162305,2.832884,0.078069
231000,7,0.059890,0.211292,2.832955,-0.023993
...
```

**Why Separate?**
- Episode logs capture actual gameplay (what happened)
- Decision attribution captures training policy state (what the model believed)
- They're at different temporal scales (episodes vs training steps)

---

### C. Training Metrics CSV

**File**: `training_logs/log_290000/dreamer_training_log_290k.csv`  
**Source**: Aggregate statistics at each checkpoint interval  
**Rows**: One row per checkpoint (5K-10K step intervals)  

**Columns (~50)**:

```
step                           # Training step
avg_reward                     # Average episode reward (Crafter score)
avg_length                     # Average episode length (steps)
component_discount             # Temporal discount factor used
component_reward               # Component-wise reward breakdown
component_TimeLimittruncated   # Episodes that hit time limit

achievement_collect_coal       # How many times achieved (episode count)
achievement_collect_iron       # ...
... (all 22 achievements)
```

**Example**:
```csv
step=290000, avg_reward=1.5, avg_length=169.4,
achievement_collect_coal=0, achievement_collect_wood=57, ...
```

---

### D. Training Text Log

**File**: `training_logs/log_290000/dreamer_training.txt`  
**Source**: Console output saved to disk  

**Content**:
```
Starting DreamerV2 training with 4 parallel environments...
Resuming from previous checkpoint at step 230000

Step 235000, Avg Reward: 1.28, Avg Length: 167.8
  inventory: (non-numeric data)
  achievements: (non-numeric data)
  reward: 0.024
  
Step 240000, Avg Reward: 1.28, Avg Length: 165.2
  ...
```

---

## 5. How the Visualization Integrates Separate Data

### Merge Workflow

```
Episode Load (vis/data_manager.py):
│
├─ Load: logs/event_log_30.03_22.57.54.csv (161 rows)
│   └─ Columns: time_step, action, reward, health, achievements...
│
├─ Search: training_logs/log_*/decision_attribution.csv
│   ├─ Find ALL decision_attribution files (~200 training samples each)
│   └─ Combine them (209 rows total)
│
├─ Resample: Training samples (209) → Episode steps (161)
│   └─ Uniformly interpolate training signals across episode
│   └─ Matches training policy behavior to episode timeline
│
└─ Result: Enriched DataFrame with all columns
    ├─ Episode data: action, reward, health, achievements
    └─ Decision signals: value_estimate, action_probability, 
                         exploration_bonus, world_model_score
```

### Why Resample?

- **Training steps** (90000, 91000, ...) don't match **episode steps** (0, 1, 2, ...)
- Solution: Spread the ~200 training samples uniformly across the ~160 episode steps
- This shows how the policy's beliefs evolved during the episode

### Result: What You See in Visualization

```
Decision Attribution Plot displays:
- Value estimate (purple line)       ← From value critic head
- Action probability (brown dashed)  ← From policy head
- Exploration bonus (orange)         ← From RND/curiosity
- World-model score (green dashed)   ← From prediction confidence
```

---

## 6. Data Flow Diagram

```
Training Phase:
  Environment Rollout → Policy Decisions
       ↓                    ↓
   [action taken]    [log decision attribution]
       ↓                    ↓
  training_logs/log_*/decision_attribution.csv
  training_logs/log_*/dreamer_training_log.csv
  training_logs/log_*/dreamer_training.txt

  (Every ~1000 steps records policy state)

Episode Playback Phase:
  Environment Episode → (concurrent recording)
       ↓
  logs/event_log_*.csv  +  logs/*.mp4

  (Saves actual gameplay without decision signals)

Visualization:
  load event_log.csv ──┐
                       ├─→ [Resample & Merge] ──→ Plots
  load training decision_attribution.csv ──┘
  
  - Episode trajectory (x-axis: time_step 0-160)
  - Resampled policy beliefs (training data interpolated)
```

---

## 7. Summary Table

| Log Type | Location | Frequency | Rows | Purpose |
|----------|----------|-----------|------|---------|
| **Event Log** | `logs/event_log_*.csv` | Per episode | ~160 | Actual gameplay recording |
| **Decision Attribution** | `training_logs/log_*/decision_attribution.csv` | Every ~1K training steps | ~200 | Policy internals (for viz overlay) |
| **Training Metrics** | `training_logs/log_*/dreamer_training_log.csv` | Per checkpoint | ~50 | Aggregate performance stats |
| **Training Text** | `training_logs/log_*/dreamer_training.txt` | Per checkpoint | varies | Human-readable console output |

---

## 8. Key Insight: Why Separate?

The system **separates episode recording from training analysis**:

- **During Training**: Save policy decisions (what the model believed)
- **During Playback**: Record outcomes (what actually happened)
- **In Visualization**: Combine them to show how the policy's beliefs led to outcomes

This separation allows:
✅ Efficient training (minimal episode overhead)  
✅ Clean data collection (no training artifacts in episode logs)  
✅ Rich analysis (beliefs + outcomes side-by-side)  

---
