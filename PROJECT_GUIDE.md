# Crafter RL Platform - Technical Project Guide

This guide documents the current implementation of the repository and aligns with the live code in the Dreamer and visualization modules.

Scope:

- README.md is the quickstart.
- This document is the deeper architecture and workflow reference.

---

## 1. Current System Overview

The project currently centers on Dreamer-based training and PyQt-based analysis for Crafter episodes.

What is implemented today:

- DreamerV2 training loop with parallel environment stepping.
- Decision-attribution logging during training.
- Episode rollout/export utilities that write CSV and MP4 artifacts.
- Interactive visualization with timeline-synced plots, achievements view, and explanation text.

Important clarification:

- PPO is not a training pipeline in this repository.
- The visualizer can still display PPO-style columns (`entropy`, `advantage`) when those fields exist in a loaded CSV.

---

## 2. Repository Map

- Training orchestration: [dreamer/train.py](dreamer/train.py)
- Policy wrapper and replay integration: [dreamer/policy.py](dreamer/policy.py)
- Dreamer model components: [dreamer/core.py](dreamer/core.py)
- Environment + episode export utilities: [dreamer/env.py](dreamer/env.py)
- Main visualization window: [vis/main.py](vis/main.py)
- Charts, info panel, explanation panel: [vis/widgets.py](vis/widgets.py)
- CSV loading and signal normalization: [vis/data_manager.py](vis/data_manager.py)
- Timeline mapping: [vis/timeline.py](vis/timeline.py)
- Video playback widget: [vis/video_player.py](vis/video_player.py)
- Deterministic template explainer: [vis/explainer.py](vis/explainer.py)
- Visualization path defaults: [vis/config.py](vis/config.py)
- Output archiving helper: [scripts/archive_outputs.sh](scripts/archive_outputs.sh)
- Dependency pins: [requirements.txt](requirements.txt)

---

## 3. Environment and Dependencies

Install:

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
```

Pinned runtime stack includes:

- tensorflow-macos 2.16.1
- tensorflow-probability 0.20.0
- gym 0.25.2
- crafter 1.8.3
- numpy 1.23.5
- pandas 2.2.3
- matplotlib 3.10.1
- opencv-python 4.11.0.86
- imageio 2.37.0
- PyQt5 5.15.11
- pyqtgraph 0.13.7

Implementation note:

- Dreamer modules set `TF_USE_LEGACY_KERAS=1` for compatibility with the pinned TF/Keras stack.

---

## 4. Architecture

### 4.1 Training Loop (dreamer/train.py)

- `train_dreamer(...)` creates `num_envs` Crafter environments.
- Actions are produced from `DreamerPolicy`, environments are stepped, and transitions are fed back through policy updates.
- Checkpoint and log directories are rewritten to step-suffixed targets:
  - checkpoint directory becomes `.../ckpt_<target_steps>`
  - log directory becomes `.../log_<target_steps>`
- Decision attribution is periodically logged to `decision_attribution.csv` with columns:
  - `step`, `action_taken`, `action_probability`, `world_model_score`, `exploration_bonus`, `value_estimate`
- Aggregate metrics are appended to `dreamer_training_log.csv`.

### 4.2 Policy Layer (dreamer/policy.py)

- `DreamerPolicy` acts as the bridge between environment interaction and Dreamer updates.
- It exposes methods used by training/analysis code, including decision-attribution calls used for explainability data.

### 4.3 Dreamer Core (dreamer/core.py)

- Contains Dreamer model internals (RSSM-style latent dynamics, encoder/decoder, actor/critic components).
- Supports action/value attribution values consumed by logging and visualization.

### 4.4 Episode Export Flow (dreamer/env.py)

- `create_environment(...)` builds a Crafter env and wraps it with `crafter.Recorder`.
- `run_episode(...)` writes per-step CSV rows and optional MP4 output.
- Export filenames are normalized using checkpoint/episode info when present in folder paths.

### 4.5 Visualization App (vis/main.py + vis/widgets.py)

The main window is split into:

- Left: video player.
- Right: stacked charts or achievements panel.
- Bottom: mode-dependent panel.

Bottom panel behavior:

- Charts mode: decision-attribution plot.
- Achievements mode: explanation toolbox text panel.

Menu flows:

- Open Random Log and Video
- Open from Results
- Open from Logs Directory

View toggles:

- Cumulative rewards
- Reward components
- Decision attribution

### 4.6 Explanation Generation (vis/explainer.py)

- `generate_explanation(step_row, prev_row, algorithm)` creates deterministic step-level text.
- Supports Dreamer, PPO-style, and unknown signal sets.
- Algorithm can be inferred from available columns.

---

## 5. Data Model and File Layout

### 5.1 CSV Fields Used by the Visualizer

Required fields:

- `time_step`
- `action`
- `reward`

Optional fields:

- `executed_action`
- `action_probability`
- `value` or `value_estimate`
- Dreamer-style: `exploration_bonus`, `world_model_score`
- PPO-style: `entropy`, `advantage`
- inventory/resource columns (either explicit columns or inventory-derived content)

### 5.2 Training Artifacts

Typical outputs from Dreamer training:

- `.../ckpt_<target_steps>/` (checkpoint files)
- `.../log_<target_steps>/dreamer_training_log.csv`
- `.../log_<target_steps>/decision_attribution.csv`
- `.../log_<target_steps>/dreamer_training.txt`

### 5.3 Visualization Source Directories

`vis/config.py` resolves input directories this way:

1. If `archive/run_*` exists, pick the latest run and use its `logs/` and `results/`.
2. Otherwise, use root-level `logs/` and `results/`.

This makes archived runs first-class inputs for the UI.

---

## 6. Running Workflows

### 6.1 Train Dreamer (Python API)

```bash
python - <<'PY'
from dreamer.train import train_dreamer

train_dreamer(
    env_name='CrafterReward-v1',
    total_steps=250000,
    num_envs=4,
    save_interval=10000,
    checkpoint_dir='./dreamer_checkpoints',
    log_dir='./training_logs',
)
PY
```

### 6.2 Train Dreamer (CLI)

```bash
python -m dreamer.train --mode train --steps 250000 --checkpoint-dir ./dreamer_checkpoints --log-dir ./training_logs
```

### 6.3 Run Visualization

```bash
python -m vis.main
```

Use the File menu to load random logs, browse results episodes, or open log files directly.

---

## 7. Analysis Workflow in the UI

1. Load a CSV + MP4 pair.
2. Use timeline/video controls to align visual events with numeric signals.
3. Inspect cumulative reward and component traces for regime shifts.
4. Use decision attribution to compare confidence/value/exploration signals.
5. Switch to Achievements mode to inspect completion status and explanation text at the current step.

---

## 8. Troubleshooting

- UI imports fail: launch from repo root with `python -m vis.main`.
- No files shown in quick-open: verify `logs/` and `results/` exist under the latest archive run or project root.
- Missing attribution curves: confirm CSV contains corresponding columns or that training produced `decision_attribution.csv`.
- TensorFlow compatibility issues: use pinned versions from requirements and keep `TF_USE_LEGACY_KERAS=1` behavior intact.
- Missing video pair: the app will prompt for MP4 selection when auto-matching fails.

---

## 9. FAQ

- Can I analyze non-Dreamer CSV logs?
  Yes. If required columns exist, the visualizer will plot what is available.

- Does the project train PPO agents?
  Not in the current code. PPO fields are visualization-compatible only.

- Where should old outputs go?
  Use [scripts/archive_outputs.sh](scripts/archive_outputs.sh) to move generated folders into timestamped archive runs.

---

## 10. Quick Commands

Setup:

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
```

Train:

```bash
python -m dreamer.train --mode train --steps 250000 --checkpoint-dir ./dreamer_checkpoints --log-dir ./training_logs
```

Visualize:

```bash
python -m vis.main
```
