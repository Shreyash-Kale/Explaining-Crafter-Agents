# Crafter RL Logging & Visualization Architecture

**Last updated:** 2026-04-26

This document reflects the current directory layout, CSV schemas, and data-flow as implemented in the live code. The old archive-centric layout (where logs lived under `archive/run_*/`) is still valid for legacy data but the active pipeline uses `data/` as its root.

---

## 1. Why 3 Menu Options to Open Files?

The three File menu entries provide different workflows for loading episodes:

```text
File Menu
├─ Open Random Log and Video
│  └─ Picks a validated timestamp-matched CSV/MP4 pair
│     from the configured logs directory.
│     (Quick exploration — no browsing required)
│
├─ Open from Results ▶ (submenu)
│  ├─ Browse Results Directory...
│  │  └─ File dialog starting at data/eval/
│  │     Navigate to: checkpoint_<N>/episode_<NNN>/
│  └─ Recent Episodes (auto-populated from latest checkpoint_*)
│
└─ Open from Logs Directory...
   └─ File dialog for any directory containing CSV + MP4 pairs
```

**Why 3 options?**

- **Random** — fast exploration without browsing; picks randomly from validated timestamp-matched pairs
- **Results** — structured by eval checkpoint and episode number; the standard path after `run_eval.py`
- **Logs** — flexible fallback for custom log locations or archived run data

The app also has a **Study** menu for marking task boundaries during user-study sessions. See §5 for study-log details.

---

## 2. Algorithm Compatibility: Dreamer vs PPO

The visualizer detects which attribution columns are present and renders the appropriate set. PPO logs are display-compatible; the project trains DreamerV2 only.

| Column | Dreamer | PPO |
| --- | --- | --- |
| `action_probability` | yes | yes |
| `value_estimate` | yes | yes |
| `exploration_bonus` | yes | — |
| `world_model_score` | yes | — |
| `logit` | yes | — |
| `entropy` | — | yes |
| `advantage` | — | yes |

Algorithm inference (`vis/explainer.py → infer_algorithm`):

- `exploration_bonus` + `world_model_score` present → Dreamer
- `entropy` + `advantage` present → PPO
- Neither → unknown (generic explanation text)

---

## 3. Directory Structure

### 3.1 Active pipeline (`data/`)

```text
data/
├── checkpoints/
│   └── ckpt_<target_steps>/
│       ├── ckpt-<step>.index
│       ├── ckpt-<step>.data-00000-of-00001
│       ├── checkpoint
│       └── tensorboard/
│           └── <YYYYMMDD-HHMMSS>/
│               └── events.out.tfevents.*
│
├── training_logs/
│   └── log_<target_steps>/
│       ├── dreamer_training_log.csv      ← step-gated aggregate metrics
│       ├── decision_attribution.csv      ← sparse training-time attribution
│       ├── dreamer_training.txt          ← console output
│       ├── dreamer_training_metrics.png  ← 4-panel reward/length/component chart
│       └── recon_<step>.png             ← orig vs reconstruction (every 10k steps)
│
├── eval/
│   └── checkpoint_<step>/
│       ├── episode_001/
│       │   ├── ckpt<checkpoint>_episode001.csv   ← per-step episode log + attribution
│       │   ├── ckpt<checkpoint>_episode001.mp4   ← 15 fps video
│       │   └── stats.jsonl                       ← crafter.Recorder summary
│       ├── episode_002/
│       │   └── ...
│       └── ...
│
└── study_logs/
    ├── raw/
    │   └── <participantID>_<YYYYMMDD_HHMMSS>.csv  ← full structured event log
    ├── reports/
    │   └── <session>_report.txt                    ← human-readable timeline
    └── analysis/
        └── <session>_analysis.csv                  ← flat no-JSON analysis CSV
```

### 3.2 Legacy archive (`archive/`)

Old runs (pre-2026-04) are stored under `archive/run_<timestamp>/` with `logs/`, `training_logs/`, and `results/dreamer_v2/checkpoint_<N>/episode_<NNN>/` sub-trees. The visualizer config (`vis/config.py`) automatically prefers the latest archive run over `data/eval/` when an archive exists, so legacy data loads transparently.

---

## 4. What's in Each Log Type

### A. Eval Episode CSV

**File:** `data/eval/checkpoint_<step>/episode_<NNN>/ckpt<ckpt>_episode<NNN>.csv`
**Source:** Written by `dreamer/env.py → run_episode` during evaluation
**Rows:** One per timestep (typically 100–1,000 rows per episode)

**Columns:**

```text
Transition basics:
  time_step           — 0, 1, 2, ... N
  action              — 0–16 (action ID)
  reward              — immediate reward this step
  cumulative_reward   — running total
  done                — True at episode end

Player stats (0–9 range):
  health, food, drink, energy

Inventory (counts):
  sapling, wood, stone, coal, iron, diamond
  wood_pickaxe, stone_pickaxe, iron_pickaxe
  wood_sword, stone_sword, iron_sword

Achievement flags (binary, 22 total):
  collect_coal, collect_diamond, collect_drink, collect_iron,
  collect_sapling, collect_stone, collect_wood,
  defeat_skeleton, defeat_zombie, eat_cow, eat_plant,
  make_iron_pickaxe, make_iron_sword, make_stone_pickaxe,
  make_stone_sword, make_wood_pickaxe, make_wood_sword,
  place_furnace, place_plant, place_stone, place_table, wake_up

Decision attribution (per-step, from policy.decision_attribution):
  logit               — raw actor logit for the taken action
  action_probability  — softmax probability of the taken action
  value_estimate      — critic output V(s) at this step
  exploration_bonus   — policy entropy (Categorical) = actor uncertainty
  world_model_score   — mean absolute encoder embedding ‖embed‖
```

**Key difference from legacy:** Attribution columns are written per step directly — no resampling. `None` fills attribution columns if the policy object lacks a `decision_attribution` method (e.g., random-action baselines).

**Example row (partial):**

```csv
time_step,action,reward,cumulative_reward,...,action_probability,value_estimate,exploration_bonus,world_model_score
15,5,0.1,1.6,...,0.2341,0.087,1.932,0.4182
```

---

### B. Training Decision Attribution CSV

**File:** `data/training_logs/log_<step>/decision_attribution.csv`
**Source:** Written during training every ~1,000 env steps (first env only)
**Rows:** ~200–300 samples per training run

**Columns:**

```text
step                — training env-step count
action_taken        — action chosen by actor
action_probability  — softmax probability of that action
world_model_score   — ‖embed‖ of the observation
exploration_bonus   — Categorical entropy of action distribution
value_estimate      — critic output at the current latent state
```

**Limitation (unchanged from v2):** Only ~1,000-step intervals, first env only. Good for confirming actor-collapse (all action_probability → 1.0) but not for per-step episode analysis. Use the eval episode CSV for per-step attribution instead.

---

### C. Training Metrics CSV

**File:** `data/training_logs/log_<step>/dreamer_training_log.csv`
**Source:** Written every `log_interval` env steps (default 5,000), step-gated — writes even when no episode has completed
**Rows:** One per log interval

**Columns:**

```text
step                            — env-step count
avg_reward                      — mean episode reward (nan if no episode yet)
avg_length                      — mean episode length (nan if no episode yet)
component_<key>                 — mean of info-dict reward components
achievement_<name>              — sum of achievement flags over recent episodes
```

---

### D. Decoder Reconstruction PNGs

**File:** `data/training_logs/log_<step>/recon_<step>.png`
**Source:** Written by `train.py` every 10,000 env steps
**Content:** Side-by-side [original observation | decoder reconstruction], both normalized to [0,255].

**How to read:** At ≤10k steps the reconstruction will be blurry but should show rough color regions. By 50k steps it should be recognizably frame-like if the world model is learning. Uniform gray = silent world model (same failure as v2 runs).

**Standalone checker:**

```bash
python generate_recon.py \
    --checkpoint-dir data/checkpoints/ckpt_750000 \
    --out recon_check.png
```

---

### E. Study Session Log

**File:** `data/study_logs/raw/<participantID>_<datetime>.csv`
**Source:** Written by `vis/study_logger.py → StudyLogger` when study logging is enabled
**Enabled via:** `--study-logging` CLI flag, `VIS_STUDY_LOGGING=1` env var, or interactive prompt at startup

**Columns:**

```text
event_seq, timestamp_iso, monotonic_ms, elapsed_s
session_id, participant_id
event_type, event_category
target_id, interaction_type
episode_id, frame, time_step
source           — user_input | system_init | system_sync | unknown
ui_state_json    — delta-compressed UI state (only when changed)
event_payload_json
```

**Event categories:** session, study, video, timeline, ui, plot, visibility, window.

**Generated reports (on session close):**

- `data/study_logs/reports/<session>_report.txt` — human-readable timeline, user events only, with elapsed-time deltas
- `data/study_logs/analysis/<session>_analysis.csv` — flat no-JSON CSV with key payload fields as explicit columns (plot, plot_step, duration_ms, tab, achievement, playback_speed, view, episode_total_steps, episode_reward_total)

**Manual report generation:**

```bash
python vis/log_report.py --all
python vis/log_report.py data/study_logs/raw/P1_20260426_143022.csv
```

---

## 5. How the Visualizer Loads Data

### 5.1 CSV/MP4 pairing

`vis/main.py → _build_valid_log_video_pairs` tries three strategies in order:

1. **Same base name** — `ckpt660000_episode001.csv` → `ckpt660000_episode001.mp4`
2. **Timestamp match** — parses `event_log_DD.MM_HH.MM.SS.csv` and `YYYYMMDDTHHMMSS-achN-lenN.mp4` filenames; pairs within a 15-minute window
3. **User dialog** — prompts for manual MP4 selection when strategies 1 and 2 both fail

### 5.2 Data loading (vis/data_manager.py)

For eval episodes produced by `run_eval.py`, attribution columns are already per-step:

```text
Load episode CSV (N rows, all columns present)
     │
     ▼
Normalize signal ranges
     │
     ▼
Separate into:
  ├─ time_steps, reward_log, action_log
  ├─ reward_components (achievement + inventory + stats columns)
  └─ attribution signals (action_probability, value_estimate,
                          exploration_bonus, world_model_score)
     │
     ▼
Ready for visualization — no resampling needed
```

For **legacy** episode CSVs (from `logs/event_log_*.csv` produced during live gameplay, which lack attribution columns), `data_manager.py` can still fall back to loading a separate `decision_attribution.csv` and resampling it across the episode length. This path remains for backward compatibility with archived data.

### 5.3 What gets plotted

**Charts panel (right column, top):**

- Cumulative reward curve
- Reward components (per-step)
- Resource/inventory traces

**Decision attribution plot (right column, bottom):**

- `value_estimate` — purple
- `action_probability` — dusty rose
- `exploration_bonus` — amber
- `world_model_score` — slate blue
- Current-step marker

**Explanation panel (left column, bottom):**

- Per-step deterministic natural-language text from `vis/explainer.py`
- Describes action confidence, value trend, world-model quality, vital stat warnings, and achievement unlocks

**Achievements panel (right column, switchable):**

- All 22 Crafter achievement flags, grouped by completed / not completed
- Switches in/out via the "Show Achievements" toggle button

---

## 6. Data Flow Diagram

```text
Training Phase
─────────────
  4 parallel Crafter envs ──► DreamerPolicy.__call__
           │                        │
           │                        ▼ (every ~1000 steps)
           │              decision_attribution.csv
           ▼                        │
    env.step(action) ──► replay buffer ──► train_batch
                                              │
                                     ▼ (every 10k steps)
                                  recon_<step>.png
                                     │
                                  dreamer_training_log.csv
                                  dreamer_training.txt
                                  dreamer_training_metrics.png

Evaluation Phase
────────────────
  run_eval.py
    └─► for each episode:
          create_environment(episode_dir)   ← crafter.Recorder attached
          DreamerPolicy(load_checkpoint=True)
          run_episode(env, policy_fn)
            ├─► per-step: action + policy.decision_attribution(obs)
            ├─► write ckpt<N>_episode<NNN>.csv  (all columns inc. attribution)
            ├─► write ckpt<N>_episode<NNN>.mp4  (15 fps)
            └─► crafter.Recorder writes stats.jsonl

  find_best_episode.py
    └─► scans stats.jsonl files → ranks by (unique achievements, reward)

Visualization Phase
───────────────────
  python -m vis.main
    ├─► File menu → pick CSV + MP4
    ├─► data_manager.py loads CSV → all columns available per step
    ├─► video_player.py loads MP4
    ├─► timeline.py maps frame ↔ step
    ├─► widgets.py renders charts + decision plot
    ├─► explainer.py generates per-step text
    └─► (if study logging) study_logger.py records all user interactions
          └─► on close: log_report.py writes _report.txt + _analysis.csv
```

---

## 7. Summary Table

| Log type | Location | Written by | Frequency | Primary use |
| --- | --- | --- | --- | --- |
| Episode CSV + attribution | `data/eval/checkpoint_N/episode_NNN/` | `run_eval.py` → `run_episode` | Per episode | Visualization + explanation |
| Episode video | same directory | `run_episode` | Per episode | Video playback |
| Crafter stats | same directory | `crafter.Recorder` | Per episode | `find_best_episode.py` ranking |
| Training decision attribution | `data/training_logs/log_N/` | `train.py` (every ~1k steps) | Sparse (~1k-step gaps) | Training health check |
| Training metrics CSV | same directory | `train.py` (every 5k steps) | Step-gated | Reward / achievement trends |
| Reconstruction PNG | same directory | `train.py` (every 10k steps) | Periodic | World-model sanity check |
| Study session log | `data/study_logs/raw/` | `StudyLogger` | Per interaction event | User-study data collection |
| Session report | `data/study_logs/reports/` | `log_report.py` (on close) | Per session | Qualitative review |
| Analysis CSV | `data/study_logs/analysis/` | `log_report.py` (on close) | Per session | Quantitative analysis |

---

## 8. Key Architectural Principle

Episode CSVs and training logs are **separate streams by design**:

- **Training logs** capture what the model believed at training time (sparse, policy-state snapshots).
- **Eval episode CSVs** capture what actually happened in a rollout with the saved checkpoint, with attribution computed per step from the loaded model.
- **Visualization** reads eval episode CSVs directly — no post-hoc merging is needed for data produced after 2026-04-26.

This separation allows efficient training (no per-step overhead), clean eval data (no training artifacts), and correct attribution (computed from the actual policy used in the episode, not resampled from a different training phase).
