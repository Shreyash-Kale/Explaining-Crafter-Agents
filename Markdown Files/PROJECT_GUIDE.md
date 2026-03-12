# Crafter RL Platform – Comprehensive Technical Guide

This guide thoroughly explains the project: what it does, how it’s built, how to run it end-to-end, and how to analyze sequential decision-making using the visualization interface. It is intended for practitioners who want a deep understanding of the system and how to extend it.

Scope note:

- `README.md` is the quickstart and day-to-day entry point.
- This document is the deep technical reference.

---

## 1. Overview

The project is a full-stack platform for training and analyzing reinforcement learning agents in the Crafter environment using PPO & DreamerV2. It comprises:

- A custom DreamerV2 implementation tailored to Crafter observations/actions.
- Parallel environment rollout and prioritized replay.
- Structured logs, per-step decision attribution, and episode exports.
- A PyQt5 visual analysis interface that synchronizes gameplay video with metrics, rewards, and semantic signals.

The platform helps answer questions like: Which actions led to rewards? What did the agent believe (value/probability/exploration) before taking an action? Which resource components contributed to progress? How do achievements unfold across an episode?

---

## 2. Repository Map

- Training pipeline: [dreamer/train.py](dreamer/train.py)
- Policy glue + replay buffer: [dreamer/policy.py](dreamer/policy.py)
- DreamerV2 core models: [dreamer/core.py](dreamer/core.py)
- Environment setup + recorder: [dreamer/env.py](dreamer/env.py)
- Visualization main app: [vis/main.py](vis/main.py)
  - Plots and UI components: [vis/widgets.py](vis/widgets.py)
  - Data loader/normalizer: [vis/data_manager.py](vis/data_manager.py)
  - Timeline synchronization: [vis/timeline.py](vis/timeline.py)
  - Video player controls: [vis/video_player.py](vis/video_player.py)
  - Template explainer module: [vis/explainer.py](vis/explainer.py)
- Semantic event templates (optional, currently not wired): [SemanticEventDetector.py](SemanticEventDetector.py)
- Visualization config/paths: [vis/config.py](vis/config.py)
- Dependencies: [requirements.txt](requirements.txt)

---

## 3. Environment and Dependencies

Confirmed runtime libraries:

- TensorFlow macOS 2.16.1 and TensorFlow Probability 0.20.0
- Gym 0.25.2 (pre v1 API) and Crafter 1.8.3
- NumPy, Pandas, Matplotlib (headless);
- OpenCV for video; PyQt5 and pyqtgraph for the viz

Install via:

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
```

Notes:
- Code sets `TF_USE_LEGACY_KERAS=1` where needed to align TF/Keras APIs.
- On Apple Silicon, TensorFlow-MacOS is used (already pinned in requirements).

---

## 4. Technical Architecture

### 4.1 Training Loop (dreamer/train.py)

- Spawns `num_envs` parallel `CrafterReward-v1` environments.
- Alternates rollout and learning: for each step, actions are produced by the policy, environments step, transitions are added to replay, and periodic updates/logging occur.
- Directory management automatically suffixes checkpoint/log dirs by target step (e.g., `ckpt_250000`, `log_250000`).
- Decision attribution logging occurs every ~1000 steps for one env: records action probability, exploration bonus, world-model score, and value estimate.

### 4.2 Policy and Replay (dreamer/policy.py)

- `DreamerPolicy` wraps the agent and provides:
  - Global step tracking and checkpoint resume.
  - `EnhancedReplayBuffer` with episode boundary awareness:
    - Avoids sampling sequences that cross terminal boundaries.
    - Prioritized sampling with importance weights (alpha/beta schedule).
  - Batched sequence sampling (`batch_size` × `sequence_length`) feeding DreamerV2.
  - `log_decision_attribution(obs, action)` computing signals for analysis (action prob, exploration/world-model, value).

### 4.3 DreamerV2 Core (dreamer/core.py)

- Implements a DreamerV2-style model stack:
  - `RSSM`: recurrent state-space model with discrete latents (categorical, OneHotCategorical) and GRU core.
  - `Encoder`: CNN turns pixel observations (uint8) into embeddings.
  - `Decoder`: deconvolutional decoder reconstructs pixels for representation learning.
  - Actor/Critic heads (present in file) produce action distributions and value estimates from latent states.
  - “Imagination” unrolls latent dynamics (`imagination_horizon`) to train actor/critic via predicted returns.
- Representation loss: reconstruction and KL between posterior/prior latents.
- Critic loss: value fit on imagined targets (e.g., lambda-returns).

### 4.4 Environment Integration (dreamer/env.py)
### 4.5 Visualization Stack
- Main app (`vis/main.py`) composes:
  - Left: `VideoPlayerWidget` (OpenCV-backed frame display + play/step controls).
  - Right: `VisualizationWidget` (pyqtgraph plots) or `InfoPanel` (achievements view) via a stacked widget.
  - Bottom: mode-aware panel:
    - charts mode: decision attribution plot
    - achievements mode: Explanation Toolbox (`ExplanationPanel`) with step-level NLP templates
  - Decision signals: `action_probability`, `value`/`value_estimate`, `entropy`, `advantage`, `exploration_bonus`, `world_model_score`
  - Normalized traces for overlays (0–1 scaling, guarded against degenerate ranges)
- Plots in `vis/widgets.py`:
  - Cumulative reward line + per-step bar graph; `DecisionPoint` markers with action-aware tooltips; vertical cursor.
  - Reward component stacked areas/lines for non-zero series; dynamic legend.
  - Decision attribution overlay: always `value` + `action_probability`; plus PPO (`entropy`, `advantage`) or DreamerV2 (`exploration_bonus`, `world_model_score`) when available.
- Synchronization:
  - Slider percent → video frame → episode step using `frame_step_ratio` mapping.
  - Video `frame_changed` events update slider and plots; the app updates both `InfoPanel` and `ExplanationPanel` at the current step.

### 4.6 Template Explanation Toolbox (vis/explainer.py + vis/widgets.py)

- `vis/explainer.py` provides deterministic template generation via `generate_explanation(...)`.
- Input: current step row, previous step row, and algorithm mode (`dreamer` / `ppo` / `unknown`).
- Output: concise natural language text describing confidence, value trend, and algorithm-specific signals.
- `ExplanationPanel` is rendered in the bottom pane during Achievements mode and stays synchronized with timeline/video scrubbing.

---

## 5. Data and Directory Layouts

### 5.1 Expected CSV columns

- Required: `time_step`, `action`, `reward`
- Optional (used when present):
  - `executed_action`, `action_probability`, `value` or `value_estimate`
  - PPO: `entropy`, `advantage`
  - DreamerV2: `exploration_bonus`, `world_model_score`
  - `inventory` as a Python-style dict string, e.g., `{"wood": 3, "stone": 1}`

`VisDataManager` expands inventory keys into component time series. Zero-only components are hidden from the plot.

### 5.2 Training outputs

- Checkpoints: `dreamer_checkpoints/ckpt_<STEPS>/ckpt-<N>`
- Logs: `logs_dreamer/log_<STEPS>/dreamer_training_log.csv`, `decision_attribution.csv`, plus text logs
- Videos: `videos/` (episode recordings if enabled)

### 5.3 Results for Viz

- Logs directory: `logs/` (used by viz quick-open)
- Results directory: `results/dreamer_v2/checkpoint_*/episode_*/`
  - Typical episode contents: `data.csv`, one or more `.mp4` files

---

## 6. Running the System

For minimal setup/run commands, prefer `README.md` first; this section keeps the full workflow examples.

### 6.1 Setup environment

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
```

### 6.2 Train an agent

```bash
python - <<'PY'
from dreamer.train import train_dreamer
train_dreamer(
    env_name='CrafterReward-v1',
    total_steps=250000,
    num_envs=4,
    save_interval=10000,
)
PY
```

To resume from a checkpoint:

```bash
python - <<'PY'
from dreamer.train import train_dreamer
train_dreamer(
    total_steps=100000,
    load_checkpoint=True,
    checkpoint_dir='./dreamer_checkpoints/ckpt_250000',
)
PY
```

### 6.3 Visualize an episode

```bash
python -m vis.main
```

In the app:

- File → Open Random Log and Video (looks in `logs/`)
- File → Open from Results (navigates `results/dreamer_v2/...`)
- Use the bottom slider or video controls; charts, achievements, and explanation text update in sync.
- Toggle button behavior:
  - `Show Achievements`: right pane shows achievements; bottom pane shows Explanation Toolbox.
  - `Show Charts`: right pane shows charts; bottom pane shows decision attribution plot.
- View menu toggles cumulative, components, and decision attribution visibility.

---

## 7. Sequential Decision Analysis Workflow

1. Identify a reward jump in the cumulative timeline; hover the `DecisionPoint` to see action, reward size, and the step index.
2. Glance at the decision attribution overlay: were value and action probability high? Was entropy/exploration elevated beforehand? This explains confidence and exploration behavior.
3. Inspect reward components: which resource signals changed? Inventory-derived lines show resource collection/crafting effects.
4. Switch to the Info panel: see achievements unlocked near the current step and dependencies required for missing ones.
5. Use the Explanation Toolbox while in Achievements mode to read a textual summary for the current step.
6. Move frame-by-frame to study short sequences around pivotal decisions; correlate video context with attribution signals and template narration.

This workflow reveals credit assignment patterns, exploration efficacy, and prerequisite milestones for complex behaviors.

---

## 8. Configuration and Hyperparameters

- `train_dreamer(...)` parameters:
  - `env_name`, `total_steps`, `num_envs`, `log_interval`, `save_interval`
  - `checkpoint_dir`, `log_dir`, `video_dir`, `load_checkpoint`
- `DreamerPolicy(...)`:
  - `replay_capacity`, `batch_size`, `sequence_length`, `training_interval`, `save_interval`
  - `parallel_envs`, `checkpoint_path`
- `DreamerV2(...)`:
  - `actor_entropy` (exploration), `imagination_horizon`, encoder/decoder sizes, RSSM latent sizes/classes

Tips:

- Increase `num_envs` for better sample throughput; watch CPU/GPU limits.
- Tune `actor_entropy` to balance exploration vs exploitation.
- Lengthen `sequence_length` if long-term dependencies matter; ensure replay can hold enough sequences.

---

## 9. Performance and Scaling

- Parallel envs are CPU-bound; use moderate values (e.g., 4–8) on laptops.
- TensorFlow ops benefit from Apple Neural Engine/GPU where available; keep observations uint8 and batch operations in TF.
- Reduce UI overhead during training by using headless plotting (`matplotlib.use("Agg")`)—already set.

---

## 10. Troubleshooting

- TensorFlow/Keras API mismatch: ensure `TF_USE_LEGACY_KERAS=1` and versions pinned in [requirements.txt](requirements.txt).
- No video in viz: confirm `.mp4` exists next to the CSV (same folder). `vis/main.py` will prompt if missing.
- Empty components: likely zero-only series; check `inventory` column content in CSV.
- Slow playback: lower video resolution or limit frame rate; the player scales frames to the widget size.
- Checkpoint not found: verify path naming (`ckpt_<STEPS>/ckpt-<N>`). `DreamerPolicy` prints loaded step.
- Import error on launch: run `python -m vis.main` from project root (not `python vis/main.py`).

---

## 11. FAQ

- Can I visualize PPO logs? Yes—if your CSV includes `entropy`/`advantage`, the decision attribution plot will show PPO overlays.
- Can I generate natural language explanations? Yes—`vis/explainer.py` powers the Explanation Toolbox shown in Achievements mode.
- Can I extend event detection? Use [SemanticEventDetector.py](SemanticEventDetector.py) and wire events into `vis/widgets.py` (icons or bands) keyed to `time_step`.
- Do I need videos? The viz works best with video; without it, you can still scrub plots via the slider.

---

## Appendix: Recent Cleanup Notes

Removed legacy standalone scripts that were not in the core runtime/import path:

- `MIXTAPE_Save.py`
- `test_data_loading.py`
- `debug_decision_data.py`
- `analyze_rewards.py`
- duplicate root `VisConfig.py` (active config is `vis/config.py`)

---

## 12. Roadmap and Extensions

- Add TD-error, advantage, and logits to tooltips for deeper diagnostics.
- Record imagined trajectories (DreamerV2) and overlay as predicted outcome bands.
- Bookmark/tag key steps in the UI and export a review set.
- Batch import of episodes with a comparison dashboard (per-checkpoint).

---

## 13. Glossary

- RSSM: Recurrent State-Space Model—latent dynamics with a recurrent core and discrete variables.
- Decision attribution: signals explaining a decision (probability, value, exploration/world-model scores).
- Inventory-derived components: numerical traces extracted from item counts/resources in the episode log.

---

## 14. Quick Commands

These commands mirror `README.md` and are repeated here for convenience.

Setup:

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
```

Train:

```bash
python - <<'PY'
from dreamer.train import train_dreamer
train_dreamer(total_steps=250000, num_envs=4, save_interval=10000)
PY
```

Visualize:

```bash
python -m vis.main
```

This document is meant to be exhaustive yet navigable. Use it as the central reference while developing, training, and analyzing agents with this platform.
