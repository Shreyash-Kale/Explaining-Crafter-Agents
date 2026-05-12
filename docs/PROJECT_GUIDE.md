# Crafter RL Platform - Technical Project Guide

This guide documents the current implementation of the repository and aligns with the live code across the Dreamer training pipeline, evaluation scripts, and visualization application.

Scope:

- README.md is the quickstart.
- This document is the deeper architecture and workflow reference.

---

## 1. Current System Overview

The project centers on DreamerV2 training, standalone evaluation, and an interactive PyQt visualization tool designed for user studies on agent explainability.

What is implemented today:

- DreamerV2 training loop with parallel environment stepping and epsilon-greedy exploration.
- Full world-model components: RSSM with discrete latent variables, CNN encoder/decoder, reward predictor, and a binary discount (episode-continuation) predictor.
- Target critic with soft EMA updates; REINFORCE actor gradient; KL balancing with separate free-nats floors.
- In-training decoder reconstruction snapshots saved periodically to the log directory.
- Decision-attribution logging during training and evaluation.
- Standalone evaluation script with per-step checkpoint selection.
- Episode ranking tool that scores eval runs by achievement diversity and reward.
- Standalone world-model reconstruction checker.
- Interactive visualization with timeline-synced plots, achievements panel, and explanation text.
- Structured user-study session logging with dwell-time tracking and on-close report generation.

Important clarification:

- PPO is not a training pipeline in this repository.
- The visualizer can still display PPO-style columns (`entropy`, `advantage`) when those fields exist in a loaded CSV.

---

## 2. Repository Map

### Training and Evaluation

- Training orchestration: [dreamer/train.py](dreamer/train.py)
- Policy wrapper and replay integration: [dreamer/policy.py](dreamer/policy.py)
- Dreamer model components: [dreamer/core.py](dreamer/core.py)
- Environment and episode export utilities: [dreamer/env.py](dreamer/env.py)
- Standalone evaluation runner: [scripts/run_eval.py](scripts/run_eval.py)
- Episode ranker: [scripts/find_best_episode.py](scripts/find_best_episode.py)
- World-model reconstruction checker: [generate_recon.py](generate_recon.py)
- Output archiving helper: [scripts/archive_outputs.sh](scripts/archive_outputs.sh)

### Visualization Application

- Main window and app entry point: [vis/main.py](vis/main.py)
- Charts, info panel, and explanation panel widgets: [vis/widgets.py](vis/widgets.py)
- CSV loading and signal normalization: [vis/data_manager.py](vis/data_manager.py)
- Timeline mapping: [vis/timeline.py](vis/timeline.py)
- Video playback widget: [vis/video_player.py](vis/video_player.py)
- Deterministic template explainer: [vis/explainer.py](vis/explainer.py)
- Semantic event detection from trajectories: [vis/SemanticEventDetector.py](vis/SemanticEventDetector.py)
- Structured user-study logger: [vis/study_logger.py](vis/study_logger.py)
- Study log report generator: [vis/log_report.py](vis/log_report.py)
- Visualization path defaults and study-log config: [vis/config.py](vis/config.py)

### Dependencies

- macOS / Apple Silicon: [requirements.txt](requirements.txt)
- Linux / Windows eval-only (no PyQt): [scripts/requirements_eval.txt](scripts/requirements_eval.txt)

---

## 3. Environment and Dependencies

### macOS (Apple Silicon) — Full Install

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
```

Pinned runtime stack:

- tensorflow-macos 2.16.1 + tensorflow 2.16.1 + tf-keras 2.16.0
- tensorflow-probability 0.24.0
- gym 0.25.2
- crafter 1.8.3
- numpy 1.23.5
- pandas 2.2.3
- matplotlib 3.10.1
- opencv-python 4.11.0.86
- imageio 2.37.0
- PyQt5 5.15.11
- pyqtgraph 0.13.7

### Linux / Windows — Evaluation Only (no visualization)

```bash
pip install -r scripts/requirements_eval.txt
```

This omits PyQt5 and pyqtgraph and is intended for lab machines that run evaluation episodes only.

Implementation note:

- All Dreamer modules set `TF_USE_LEGACY_KERAS=1` at import time for compatibility with the pinned TF/Keras stack.
- The training script also sets GPU memory growth (`tf.config.experimental.set_memory_growth`) to avoid reserving all VRAM on NVIDIA hardware.

---

## 4. Architecture

### 4.1 Training Loop (dreamer/train.py)

- `train_dreamer(...)` creates `num_envs` Crafter environments.
- Actions are produced from `DreamerPolicy` with an epsilon-greedy exploration floor (`expl_amount=0.1`) applied during training only.
- Checkpoint and log directories are rewritten to step-suffixed targets:
  - checkpoint directory becomes `.../ckpt_<target_steps>`
  - log directory becomes `.../log_<target_steps>`
- Decision attribution is periodically logged to `decision_attribution.csv` with columns:
  - `step`, `action_taken`, `action_probability`, `world_model_score`, `exploration_bonus`, `value_estimate`
- Aggregate metrics are appended to `dreamer_training_log.csv`.
- A 4-panel PNG training chart (`dreamer_training_metrics.png`) is saved at each log interval.
- Every 10,000 steps a side-by-side original/reconstruction PNG (`recon_<step>.png`) is saved by running one observation through the encoder → RSSM → decoder path. This lets you visually verify that the world model is learning without interrupting training.

### 4.2 Policy Layer (dreamer/policy.py)

`DreamerPolicy` bridges environment interaction and DreamerV2 updates.

Key behaviours:

- **Epsilon-greedy exploration**: `__call__` accepts `expl_amount` (default 0.1); during training, that fraction of steps take a uniformly random action to keep the replay buffer diverse.
- **Enhanced replay buffer** (`EnhancedReplayBuffer`): prioritized experience replay with episode-boundary tracking so sequences do not cross episode ends. Priorities are updated after each training batch based on model loss.
- **Training trigger**: calls `train_batch` every `training_interval` env steps once the buffer has enough data.
- **Checkpoint save**: saves the full checkpoint at every `save_interval` steps.
- **Attribution delegation**: `log_decision_attribution` and `decision_attribution` delegate to `DreamerV2.log_decision_attribution`, which computes action probability, world-model score, exploration bonus (policy entropy), and value estimate from the current latent state.

### 4.3 Dreamer Core (dreamer/core.py)

#### Sub-networks

| Class | Role |
| --- | --- |
| `RSSM` | Recurrent State-Space Model. GRU cell + prior/posterior discrete latent heads (32 categories × 32 classes). |
| `Encoder` | 4-layer CNN → 1024-d embedding. Normalises inputs to [0,1] internally. |
| `Decoder` | Transposed-CNN decoder. Mean constrained to (0,1) via sigmoid; fixed σ=0.1 so reconstruction gradients are tight even for small errors. |
| `DenseDecoder` | Shared head for reward and discount prediction. Reward head uses `dist='mse'` (fixed σ=1.0) to avoid std collapse under sparse rewards. Discount head uses `dist='binary'`. |
| `Actor` | 4-layer MLP → `OneHotCategorical` for discrete actions. |
| `Critic` / `target_critic` | 4-layer MLP → scalar value. Both warmed up to 2048-d input at build time. |

#### Training Hyperparameters (Crafter overrides from defaults)

| Parameter | Value | Reason |
| --- | --- | --- |
| `actor_entropy` | 3e-3 | Default 1e-3 is too weak; REINFORCE sharpens logits faster than entropy can push back, causing policy collapse. |
| `actor_grad` | `'reinforce'` | Discrete `OneHotCategorical` gives zero gradient through `dynamics` backprop without a Gumbel estimator. |
| `actor_lr` | 1e-4 | Crafter default; entropy bonus is strong enough to hold policy at this rate. |
| `critic_lr` | 1e-4 | Crafter default. |
| `model_lr` | 1e-4 | Prevents world model over-fitting on early noisy sparse-reward rollouts. |
| `gamma` | 0.999 | Crafter config; sparse reward chains need a long effective horizon. |
| `imagination_horizon` | 15 | DreamerV2 paper default. |
| `free_nats` | 1.0 | Forces the encoder to encode signal even when the prior is a good match; without this the posterior trivially collapses. |
| `kl_balance` | 0.8 | 80% weight on training the prior; 20% on regularising the posterior. |

#### KL Balancing

`train_model` implements the DreamerV2 stop-gradient KL balance:

- `lhs = KL(sg(posterior) ∥ prior)` — gradients flow to the prior only.
- `rhs = KL(posterior ∥ sg(prior))` — gradients flow to the posterior only.
- `kl = 0.8 · lhs + 0.2 · rhs`, free-nats applied to each term separately.

#### Target Critic

`target_critic` is a slow copy of the online critic updated by soft EMA (τ=0.01) after every training batch. Lambda-returns for the actor are bootstrapped from the target critic; the online critic is trained to regress those target values. This prevents the critic from chasing its own tail.

#### Discount Predictor

`discount_predictor` is a binary `DenseDecoder` trained with BCE to predict `P(episode continues)` at each imagined step. Lambda returns in imagination are dampened by `gamma × discount_pred` at each step, so bootstrapping dies at predicted episode boundaries rather than using a hard horizon cutoff.

#### Lambda Returns

`compute_return` implements GAE-style lambda returns with per-step effective discount:

```python
disc = gamma * discount_predictions
next_values = concat(values[:, 1:], values[:, -1:])  # bootstrap last value
delta = rewards + disc * next_values - values
```

Returns are accumulated in reverse order using `disc * lambda_` as the per-step decay.

### 4.4 Episode Export Flow (dreamer/env.py)

- `create_environment(output_dir)` builds a Crafter env, patches `metadata.render_modes` for Gym 0.25 compatibility, and wraps it with `crafter.Recorder`.
- `run_episode(...)` writes:
  - Per-step CSV with columns: `time_step`, `action`, `reward`, `cumulative_reward`, `done`, all achievement flags, all inventory items (wood, stone, coal, iron, diamond, pickaxes, swords, sapling), player stats (health, food, drink, energy), and attribution fields (logit, action_probability, value_estimate, exploration_bonus, world_model_score).
  - Optional MP4 at 15 fps.
- The CSV and video base filename is normalized from the directory path: `ckpt{checkpoint_num}_episode{episode_num}` when checkpoint info is present, or `episode_{episode_num}` otherwise.
- `run_with_dreamer(...)` batches multiple evaluation episodes into `results/dreamer_v2/checkpoint_<N>/episode_<NNN>/` subdirectories.

### 4.5 Visualization App (vis/main.py + vis/widgets.py)

The main window layout:

```text
┌─────────────────────┬──────────────────────────────┐
│  Left column        │  Right column                │
│  ─────────────────  │  ────────────────────────    │
│  Video player       │  Charts  OR  Achievements    │
│                     │  panel (stacked widget)       │
│  [Toggle button]    │                              │
│  ─────────────────  │  ────────────────────────    │
│  Explanation panel  │  Decision attribution plot   │
│  (always visible)   │  OR empty placeholder        │
└─────────────────────┴──────────────────────────────┘
          Timeline slider (below video, full width)
```

The toggle button switches the right column between Charts and Achievements view. In Achievements mode the bottom-right area is replaced with an empty placeholder.

#### Menu Structure

- **File**
  - Open Random Log and Video (picks a validated timestamp-matched CSV/MP4 pair from the logs directory)
  - Open from Results → Browse Results Directory / Recent checkpoint episodes
  - Open from Logs Directory
  - Exit
- **View** — toggles for Cumulative Rewards, Reward Components, Decision Attribution, and Auto Reload (dev)
- **Study** — manual study markers: Task Start/End, Question Start/End, Answer Submit, Think Aloud Start/End

#### Auto-Reload Dev Mode

`setup_auto_reload` installs a `QFileSystemWatcher` on `vis/*.py`. When any source file changes, a 250ms debounce timer fires `restart_process`, which calls `os.execv` to hot-restart the process. The current study-logging setting is propagated to the restarted process via `VIS_STUDY_LOGGING` environment variable.

#### CSV/MP4 Pairing

`_build_valid_log_video_pairs` attempts three strategies in order:

1. Same base name (`event_log_17.03_12.13.09.csv` → `event_log_17.03_12.13.09.mp4`).
2. Timestamp-matched: parses `event_log_DD.MM_HH.MM.SS.csv` and `YYYYMMDDTHHMMSS-achN-lenN.mp4` filenames, pairs within a 15-minute window.
3. Falls back to asking the user to select an MP4.

### 4.6 Explanation Generation (vis/explainer.py)

- `generate_explanation(step_row, prev_row, algorithm)` creates deterministic multi-sentence step-level text.
- Algorithm is auto-inferred from available columns if not specified:
  - `exploration_bonus` + `world_model_score` → Dreamer
  - `entropy` + `advantage` → PPO
  - Otherwise → unknown
- Dreamer output describes action confidence, value trend, reward context, world-model prediction quality, exploration bonus level, and vital stat warnings (health/food/drink/energy drops).
- PPO output describes entropy (exploitative vs. exploratory) and advantage relative to baseline.
- Can be previewed standalone with a CSV: `python vis/explainer.py data.csv --rows 20`.

### 4.7 Study Logger (vis/study_logger.py)

`StudyLogger` records structured interaction events to a per-session CSV in `data/study_logs/raw/`. It is enabled via:

- CLI flag `--study-logging`
- Environment variable `VIS_STUDY_LOGGING=1`
- Interactive prompt at startup (only when stdin is a TTY)

When disabled, `NoOpStudyLogger` is a drop-in no-op with the same interface.

Logged event categories:

| Category | Events |
| --- | --- |
| session | session_start, session_end, episode_loaded, episode_context |
| study | task_start/end, question_start/end, answer_submit, think_aloud_start/end |
| video | play_started, play_paused, step_forward/backward, restart, speed_change |
| timeline | slider_scrub_start/end, slider_jump, navigation_event |
| ui | toggle_info_plots, tab_switch, achievement_clicked, view_toggle |
| plot | plot_hover_start/end, plot_click, plot_viewport_changed, legend_toggle, decision_point_click/hover |
| visibility | impression_start/end (with dwell time in ms) |
| window | window_focus_gained/lost, layout_change |

Throttling: high-frequency events (`frame_changed`, `navigation_event`, `layout_resize`) are throttled per-key to avoid flooding the log.

UI state is written as a delta-compressed JSON column: only emitted when state actually changes.

On session close the logger calls `_generate_session_reports` (if study logging was active), which produces:

- `data/study_logs/reports/<session>_report.txt` — human-readable timeline
- `data/study_logs/analysis/<session>_analysis.csv` — flat CSV with key payload fields exploded into columns

### 4.8 Log Report Generation (vis/log_report.py)

Standalone or called on close by the main window.

```bash
python vis/log_report.py                              # auto-finds latest log in data/study_logs/raw/
python vis/log_report.py data/study_logs/raw/P4_*.csv # specific file
python vis/log_report.py --all                        # all logs
```

Outputs:

- `<session>_report.txt`: human-readable timeline showing only user-initiated events, with elapsed-time deltas. System-init events are counted but not listed. Dwell impressions shorter than 1 second are filtered out.
- `<session>_analysis.csv`: flat no-JSON CSV with fields: event_seq, timestamp_iso, elapsed_s, delta_ms, participant_id, session_id, episode_id, episode_number, time_step, event_type, event_category, source, is_user_action, target_id, plus extracted payload fields (plot, plot_step, duration_ms, tab, achievement, playback_speed, view) and episode context (total_steps, reward_total).

### 4.9 Semantic Event Detector (vis/SemanticEventDetector.py)

`SemanticEventDetector.detect_events(time_steps, actions, reward_components)` walks a trajectory and emits typed events:

- `resource_collected` — component starting with `collect_`
- `item_crafted` — component starting with `make_`
- `structure_built` — component starting with `place_`
- `enemy_defeated` — component starting with `defeat_`
- `survival` — `wake_up` component
- `exploration` — first movement action in a consecutive movement block

Each event carries type, human-readable description, importance level, component name, and value.

---

## 5. Data Model and File Layout

### 5.1 CSV Fields Used by the Visualizer

Required:

- `time_step`
- `action`
- `reward`

Optional (visualized when present):

- `executed_action`
- `action_probability`
- `value` or `value_estimate`
- Dreamer signals: `exploration_bonus`, `world_model_score`, `logit`
- PPO signals: `entropy`, `advantage`
- Inventory: `wood`, `stone`, `coal`, `iron`, `diamond`, `sapling`, `wood_pickaxe`, `stone_pickaxe`, `iron_pickaxe`, `wood_sword`, `stone_sword`, `iron_sword`
- Stats: `health`, `food`, `drink`, `energy`
- Achievement flags: `collect_wood`, `place_table`, etc.

### 5.2 Training Artifacts

Typical outputs from a Dreamer training run:

```text
./data/
├── checkpoints/
│   └── ckpt_<target_steps>/
│       ├── ckpt-<step>.index
│       ├── ckpt-<step>.data-*
│       ├── checkpoint
│       └── tensorboard/<datetime>/
└── training_logs/
    └── log_<target_steps>/
        ├── dreamer_training_log.csv
        ├── decision_attribution.csv
        ├── dreamer_training.txt
        ├── dreamer_training_metrics.png
        └── recon_<step>.png   (every 10k steps)
```

### 5.3 Evaluation Artifacts

Produced by `scripts/run_eval.py`:

```text
data/eval/
└── checkpoint_<step>/
    ├── episode_001/
    │   ├── ckpt<ckpt>_episode001.csv
    │   ├── ckpt<ckpt>_episode001.mp4
    │   └── stats.jsonl   (from crafter.Recorder)
    ├── episode_002/
    ...
```

### 5.4 Study Log Artifacts

```text
data/study_logs/
├── raw/
│   └── <participant>_<datetime>.csv   ← full event log with JSON payloads
├── reports/
│   └── <session>_report.txt           ← human-readable timeline
└── analysis/
    └── <session>_analysis.csv         ← flat CSV for pandas/Excel
```

### 5.5 Visualization Source Directories

`vis/config.py` resolves input directories:

1. If `archive/run_*` exists, pick the latest run and use its `logs/` and `results/`.
2. Otherwise, fall back to `data/eval/`.

---

## 6. Running Workflows

### 6.1 Train Dreamer (Python API)

```python
from dreamer.train import train_dreamer

train_dreamer(
    env_name='CrafterReward-v1',
    total_steps=250000,
    num_envs=4,
    save_interval=10000,
    checkpoint_dir='./data/checkpoints',
    log_dir='./data/training_logs',
)
```

### 6.2 Train Dreamer (CLI)

```bash
python -m dreamer.train \
    --mode train \
    --steps 750000 \
    --checkpoint-dir ./data/checkpoints \
    --log-dir ./data/training_logs \
    --save-interval 10000
```

To continue from an existing checkpoint:

```bash
python -m dreamer.train \
    --mode train \
    --steps 250000 \
    --checkpoint-dir ./data/checkpoints/ckpt_500000 \
    --log-dir ./data/training_logs \
    --load-checkpoint
```

### 6.3 Run Evaluation

Use the latest checkpoint step found in a folder:

```bash
python scripts/run_eval.py \
    --checkpoint-dir data/checkpoints/ckpt_750000 \
    --num-episodes 50 \
    --out-dir data/eval
```

Target a specific step within that folder (useful when a folder contains multiple `ckpt-N` files):

```bash
python scripts/run_eval.py \
    --checkpoint-dir data/checkpoints/ckpt_750000 \
    --checkpoint-step 660000 \
    --num-episodes 50 \
    --out-dir data/eval \
    --no-video
```

### 6.4 Find the Best Evaluation Episode

```bash
python scripts/find_best_episode.py --eval-dir data/eval --top 10
```

Ranks episodes by unique achievement types (primary) then total reward (secondary). Reads `stats.jsonl` files written by `crafter.Recorder`.

### 6.5 Check World-Model Reconstruction

```bash
python generate_recon.py \
    --checkpoint-dir data/checkpoints/ckpt_750000 \
    --out recon_check.png
```

Saves a side-by-side original/reconstruction PNG using one observation from a fresh Crafter env. Safe to run while training is active.

### 6.6 Run Visualization

```bash
python -m vis.main
```

For a user study session:

```bash
python -m vis.main --study-logging
# OR
VIS_STUDY_LOGGING=1 python -m vis.main
```

Use the File menu to load random logs, browse results episodes, or open log files directly.

### 6.7 Generate Study Log Reports

```bash
# From Python (also called automatically on session close)
python vis/log_report.py --all

# Specific file
python vis/log_report.py data/study_logs/raw/P1_20260426_143022.csv
```

---

## 7. Analysis Workflow in the UI

1. Load a CSV + MP4 pair via File menu (auto-paired by timestamp or manual selection).
2. Use timeline/video controls to align visual events with numeric signals.
3. Inspect cumulative reward and component traces for regime shifts.
4. Hover the decision-attribution plot (bottom right) to compare confidence, value, world-model score, and exploration bonus at any step.
5. Switch to Achievements mode to inspect completion status for all 22 Crafter achievements.
6. Read the Explanation panel (bottom left) for per-step natural-language attribution text.
7. For study sessions: use the Study menu to mark task boundaries; reports are generated automatically on close.

---

## 8. Troubleshooting

- **UI imports fail**: launch from repo root with `python -m vis.main`.
- **No files shown in quick-open**: verify `data/eval/` exists and contains episode subdirectories, or that `archive/run_*` contains `logs/` and `results/`.
- **Missing attribution curves**: confirm the CSV contains the relevant columns or that training produced `decision_attribution.csv`.
- **TensorFlow compatibility issues**: use pinned versions from requirements and keep `TF_USE_LEGACY_KERAS=1` intact.
- **Missing video pair**: the app prompts for MP4 selection when auto-matching fails.
- **Checkpoint not found in run_eval.py**: verify `--checkpoint-dir` points to the folder containing `ckpt-N.index` files; use `--checkpoint-step` to pick a specific step when multiple exist.
- **Study log not being written**: check that `data/study_logs/raw/` is writable; confirm study logging is enabled (`VIS_STUDY_LOGGING=1` or `--study-logging`).
- **Policy collapse during training**: if the agent stops exploring, check that `actor_entropy=3e-3` is not reduced and that epsilon-greedy `expl_amount=0.1` is active.

---

## 9. FAQ

- **Can I analyze non-Dreamer CSV logs?**
  Yes. If the required columns (`time_step`, `action`, `reward`) exist, the visualizer will plot what is available.

- **Does the project train PPO agents?**
  Not in the current code. PPO fields are visualization-compatible only.

- **Where should old training outputs go?**
  Use [scripts/archive_outputs.sh](scripts/archive_outputs.sh) to move generated folders into timestamped `archive/run_*` directories.

- **How do I pick the best episode to show in the UI?**
  Run `scripts/find_best_episode.py` after evaluation to rank by achievement diversity and reward.

- **Why does the decoder use a fixed σ=0.1?**
  A learned σ grew unbounded on sparse Crafter rewards, which killed the mean's gradient. The fixed value keeps reconstruction loss meaningful throughout training.

- **Why REINFORCE instead of dynamics backprop for the actor?**
  Discrete `OneHotCategorical` samples produce zero gradients through dynamics backpropagation without a straight-through Gumbel estimator. REINFORCE is the correct gradient estimator for this action type.

- **What does the discount predictor do?**
  It learns `P(episode continues)` for each imagined step. Lambda-returns are multiplied by `gamma × discount_pred` at each imagination step, so value bootstrapping gracefully dies at predicted episode ends rather than at an arbitrary horizon cutoff.

---

## 10. Quick Command Reference

### Setup

```bash
python -m venv crafter_env && source crafter_env/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python -m dreamer.train --mode train --steps 750000 \
    --checkpoint-dir ./data/checkpoints --log-dir ./data/training_logs
```

### Evaluate

```bash
python scripts/run_eval.py --checkpoint-dir data/checkpoints/ckpt_750000 \
    --num-episodes 50 --out-dir data/eval
```

### Find Best Episode

```bash
python scripts/find_best_episode.py --eval-dir data/eval
```

### Check Reconstruction

```bash
python generate_recon.py --checkpoint-dir data/checkpoints/ckpt_750000
```

### Visualize

```bash
python -m vis.main
```

### Study Session

```bash
python -m vis.main --study-logging
```

### Generate Log Reports

```bash
python vis/log_report.py --all
```
