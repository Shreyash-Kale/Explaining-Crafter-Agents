# Explaining Crafter Agents

A research toolkit for **training, evaluating, and explaining** a reinforcement-learning
agent that plays [Crafter](https://github.com/danijar/crafter) (a 2D Minecraft-like
survival game). It combines a **DreamerV2** training pipeline with an interactive
**PyQt visualization app** that plays back an episode video alongside synchronized
reward plots, decision-attribution traces, and plain-English explanations of *why*
the agent took each action.

The visualizer is built for **user studies on agent explainability**, so it also
includes structured interaction logging (what the participant clicked, hovered, and
looked at, and for how long).

> **New to this project? Start here, then read [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**
> for the deep architecture and data-flow reference. This README gets you running;
> the guide explains how everything works internally.

---

## The big picture

There are three things this repo does. You can use them independently.

| Stage | What it does | Entry point |
| --- | --- | --- |
| **1. Train** | Trains a DreamerV2 world-model agent in Crafter. Produces checkpoints + training logs. | `python -m dreamer.train` |
| **2. Evaluate** | Runs a trained checkpoint for N episodes. Produces per-episode CSV logs + MP4 videos + stats. | `python scripts/run_eval.py` |
| **3. Visualize / Explain** | Loads an episode (CSV + MP4) into a desktop app that syncs video, plots, and explanations. | `python -m vis.main` |

A typical end-to-end flow:

```text
train a model ──▶ evaluate it on fresh episodes ──▶ pick the best episode ──▶ explore it in the visualizer
 (dreamer/)        (scripts/run_eval.py)             (scripts/find_best_episode.py)   (vis/)
```

**Important:** this repo trains **DreamerV2 only**. It does *not* train PPO. The
visualizer can still display PPO-style columns (`entropy`, `advantage`) if they
happen to exist in a loaded CSV, but no PPO training code lives here.

---

## Requirements

- **Python 3.11** (developed on 3.11.8).
- A working C/C++ toolchain for the native deps (TensorFlow, OpenCV, PyQt5).
- For the visualizer: a desktop environment (PyQt5 needs a display).

There are **three** requirements files for three situations:

| File | Use it when | Includes GUI? |
| --- | --- | --- |
| [requirements.txt](requirements.txt) | macOS / Apple Silicon — full install (train + eval + visualize) | ✅ |
| [requirements_windows.txt](requirements_windows.txt) | Windows via WSL2 (Ubuntu) + CUDA, or Linux — full install | ✅ |
| [scripts/requirements_eval.txt](scripts/requirements_eval.txt) | A headless lab machine that only needs to train/evaluate (no PyQt) | ❌ |

Core pinned stack (identical across the full installs):

- tensorflow / tensorflow-macos **2.16.1** + tf-keras **2.16.0**
- tensorflow-probability **0.24.0**
- gym **0.25.2**, crafter **1.8.3**
- numpy **1.23.5**, pandas **2.2.3**, matplotlib **3.10.1**
- opencv-python **4.11.0.86**, imageio **2.37.0**
- PyQt5 **5.15.11**, pyqtgraph **0.13.7** (visualizer only)

> **Why the pins matter:** every Dreamer module sets `TF_USE_LEGACY_KERAS=1` at import
> time so TensorFlow 2.16 uses the Keras 2 API. Changing the TF/Keras versions will
> break the world-model code. Keep the pins unless you know what you're doing.

---

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.11)
python -m venv crafter_env
source crafter_env/bin/activate          # Windows/WSL: source crafter_env/bin/activate

# 2. Install dependencies (macOS shown; pick the file that matches your platform)
pip install -r requirements.txt

# 3. Launch the visualization app (loads a sample episode automatically)
python -m vis.main
```

> Always run the app as a **module from the project root** (`python -m vis.main`),
> not `python vis/main.py`. Module mode is what makes the package imports resolve.

The app opens with a sample episode already loaded (from `data/eval/` or the newest
`archive/run_*/`). Use the **File** menu to load other episodes.

---

## Workflows

### 1. Train a DreamerV2 agent

**CLI (simplest):**

```bash
python -m dreamer.train \
    --mode train \
    --steps 750000 \
    --checkpoint-dir ./data/checkpoints \
    --log-dir ./data/training_logs \
    --save-interval 10000
```

**Resume from an existing checkpoint** (point `--checkpoint-dir` at the run folder and add `--load-checkpoint`):

```bash
python -m dreamer.train \
    --mode train \
    --steps 250000 \
    --checkpoint-dir ./data/checkpoints/ckpt_500000 \
    --log-dir ./data/training_logs \
    --load-checkpoint
```

**Python API equivalent:**

```python
from dreamer.train import train_dreamer

train_dreamer(
    env_name='CrafterReward-v1',
    total_steps=750000,
    num_envs=4,            # parallel Crafter environments for data collection
    save_interval=10000,
    checkpoint_dir='./data/checkpoints',
    log_dir='./data/training_logs',
)
```

Training writes **step-suffixed** folders so runs never overwrite each other:

- Checkpoints → `./data/checkpoints/ckpt_<target_steps>/`
- Logs → `./data/training_logs/log_<target_steps>/`, containing:
  - `dreamer_training_log.csv` — aggregate metrics over time
  - `decision_attribution.csv` — per-step explainability signals
  - `dreamer_training.txt` — human-readable progress log
  - `dreamer_training_metrics.png` — 4-panel chart, refreshed each log interval
  - `recon_<step>.png` — world-model reconstruction snapshot, every 10k steps
    (left = real frame, right = what the model imagined — a quick "is it learning?" check)

### 2. Evaluate a trained checkpoint

Runs N full episodes with a checkpoint and saves a CSV + MP4 + `stats.jsonl` per episode.

```bash
python scripts/run_eval.py \
    --checkpoint-dir data/checkpoints/ckpt_750000 \
    --num-episodes 50 \
    --out-dir data/eval
```

By default it loads the **highest-step** checkpoint in the folder. To pin a specific
step (useful when a folder holds several `ckpt-N` files), and skip video for speed:

```bash
python scripts/run_eval.py \
    --checkpoint-dir data/checkpoints/ckpt_750000 \
    --checkpoint-step 660000 \
    --num-episodes 50 \
    --no-video
```

Output layout:

```text
data/eval/
└── checkpoint_<step>/
    ├── episode_001/
    │   ├── ckpt<step>_episode001.csv   ← per-step trajectory (the visualizer reads this)
    │   ├── ckpt<step>_episode001.mp4   ← episode video
    │   └── stats.jsonl                 ← achievement/reward summary (from crafter.Recorder)
    ├── episode_002/
    └── ...
```

### 3. Find the best episode to showcase

Ranks all evaluated episodes by achievement diversity (primary) then total reward
(secondary), reading the `stats.jsonl` files:

```bash
python scripts/find_best_episode.py --eval-dir data/eval --top 10
```

### 4. Visualize and explain an episode

```bash
python -m vis.main
```

**File menu:**
- **Open Random Log and Video** — picks a validated, timestamp-matched CSV/MP4 pair.
- **Open from Results** — browse `data/eval/` or jump to recent checkpoint episodes.
- **Open from Logs Directory** — choose a CSV directly (it auto-finds the matching video).

**Right-pane toggle (button under the video):**
- **Show Charts** — reward plots + the decision-attribution comparison plot (bottom right).
- **Show Achievements** — completion status for all 22 Crafter achievements.

The **Explanation panel** (bottom left) is always visible and shows deterministic,
per-step natural-language text describing the agent's action confidence, value trend,
reward context, world-model prediction quality, exploration level, and vital-stat warnings.

### 5. Run a user-study session (interaction logging)

Enable structured logging of every participant interaction (clicks, hovers, dwell time,
timeline scrubs, study markers):

```bash
python -m vis.main --study-logging
# or:
VIS_STUDY_LOGGING=1 python -m vis.main
```

The app prompts for a participant ID, then writes to `data/study_logs/`:

```text
data/study_logs/
├── raw/        ← full event log with JSON payloads (one CSV per session)
├── reports/    ← human-readable timeline (_report.txt), generated on session close
└── analysis/   ← flat CSV for pandas/Excel (_analysis.csv)
```

Use the **Study** menu to mark Task / Question / Think-Aloud boundaries during a session.
Reports regenerate automatically when you close the app, or manually:

```bash
python vis/log_report.py --all                              # all raw logs
python vis/log_report.py data/study_logs/raw/P1_<...>.csv   # one session
```

### 6. (Optional) Sanity-check world-model reconstruction

Saves a side-by-side original/reconstruction PNG from a checkpoint. Safe to run while
training is live:

```bash
python generate_recon.py --checkpoint-dir data/checkpoints/ckpt_750000 --out recon_check.png
```

### 7. (Optional) Archive old outputs

Non-destructive cleanup that *moves* generated folders (`logs/`, `results/`,
`training_logs/`, `videos/`, etc.) into a timestamped `archive/run_<datetime>/`:

```bash
bash scripts/archive_outputs.sh
```

The visualizer automatically prefers the **newest** `archive/run_*/` if one exists
(see "Where the visualizer reads data from" below).

---

## Where the visualizer reads data from

`vis/config.py` resolves input directories at startup:

1. **If `archive/run_*/` exists**, it uses the newest run's `logs/` and `results/`.
2. **Otherwise**, it falls back to `data/eval/` for both.

This is why archiving a run changes what the app loads by default. To analyze a fresh
evaluation, either keep its output in `data/eval/` (no archive present) or archive it
into `archive/run_*/logs` and `.../results`.

Study logs always go to `data/study_logs/` regardless of the above.

---

## CSV columns the visualizer understands

A loaded episode CSV needs at minimum: `time_step`, `action`, `reward`.
Everything else is optional and plotted only when present:

- **Attribution / explainability:** `action_probability`, `value` or `value_estimate`,
  `logit`, `exploration_bonus`, `world_model_score` (Dreamer), `entropy`, `advantage` (PPO).
- **Inventory:** `wood`, `stone`, `coal`, `iron`, `diamond`, `sapling`,
  `wood_pickaxe`, `stone_pickaxe`, `iron_pickaxe`, `wood_sword`, `stone_sword`, `iron_sword`.
- **Vitals:** `health`, `food`, `drink`, `energy`.
- **Achievement flags:** `collect_wood`, `place_table`, `defeat_zombie`, … (22 total).

The explainer auto-detects the algorithm from which columns exist
(`exploration_bonus` + `world_model_score` → Dreamer; `entropy` + `advantage` → PPO).

---

## Repository map

### Training & evaluation (`dreamer/`, `scripts/`, root)

- [dreamer/train.py](dreamer/train.py) — training loop, CLI, logging, recon snapshots
- [dreamer/policy.py](dreamer/policy.py) — `DreamerPolicy`: env interaction, replay buffer, attribution
- [dreamer/core.py](dreamer/core.py) — DreamerV2 model (RSSM, encoder/decoder, actor, critics)
- [dreamer/env.py](dreamer/env.py) — Crafter env setup + `run_episode` CSV/MP4 export
- [scripts/run_eval.py](scripts/run_eval.py) — standalone evaluation runner
- [scripts/find_best_episode.py](scripts/find_best_episode.py) — rank episodes by achievements/reward
- [scripts/archive_outputs.sh](scripts/archive_outputs.sh) — move outputs into `archive/run_*/`
- [generate_recon.py](generate_recon.py) — standalone world-model reconstruction check

### Visualization app (`vis/`)

- [vis/main.py](vis/main.py) — PyQt main window, menus, video↔plot sync, app entry point
- [vis/widgets.py](vis/widgets.py) — reward charts, achievements panel, explanation panel
- [vis/data_manager.py](vis/data_manager.py) — CSV loading, signal normalization, attribution shaping
- [vis/timeline.py](vis/timeline.py) — frame↔step mapping and timeline control
- [vis/video_player.py](vis/video_player.py) — MP4 playback widget
- [vis/explainer.py](vis/explainer.py) — deterministic natural-language step explanations
- [vis/SemanticEventDetector.py](vis/SemanticEventDetector.py) — typed events from a trajectory
- [vis/study_logger.py](vis/study_logger.py) — structured user-study interaction logger
- [vis/log_report.py](vis/log_report.py) — turn raw study logs into reports + analysis CSVs
- [vis/config.py](vis/config.py) — default log/result paths, colors, study-log folders

### Other directories

- `data/` — checkpoints, eval outputs, study logs, sample media (created/used at runtime)
- `archive/` — timestamped archived runs (preferred by the visualizer when present)
- `docs/` — documentation (see below)
- `default_folders/` — a **vendored copy of the upstream Crafter benchmark repo**
  (danijar/crafter): reference example runners, score files, and plotting scripts.
  Not part of this project's pipeline; kept for reference.

---

## Documentation

- **[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)** — the main technical reference:
  architecture, training hyperparameters, KL balancing, data model, full command
  reference, troubleshooting, and FAQ. **Read this next.**
- `docs/Markdown Files/` — supporting notes: `ARCHITECTURE_ANALYSIS.md`,
  `LOG_ARCHITECTURE.md`, `TRAINING_AUDIT.md`, `NLP_Explainations.md`,
  `TUTORIAL_SCRIPT.md`, `ADVISOR_PITCH.md`.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError` / import errors launching the UI | Run from the repo root as a module: `python -m vis.main`. |
| Nothing loads in the app's quick-open | Ensure `data/eval/` has episode subfolders, **or** `archive/run_*/` has `logs/` and `results/`. |
| Attribution curves missing | The CSV must contain the relevant columns; training must have produced `decision_attribution.csv`. |
| TensorFlow / Keras errors | Use the pinned versions and keep `TF_USE_LEGACY_KERAS=1` (set automatically at import). |
| No video pairs with a CSV | The app falls back to asking you to select an MP4 manually. |
| `run_eval.py` can't find a checkpoint | Point `--checkpoint-dir` at the folder containing `ckpt-N.index`; use `--checkpoint-step` when several exist. |
| Study log not written | Confirm `data/study_logs/raw/` is writable and logging is enabled (`--study-logging` or `VIS_STUDY_LOGGING=1`). |
| Agent stops exploring during training | Don't lower `actor_entropy` (3e-3) or the epsilon-greedy `expl_amount` (0.1) — see PROJECT_GUIDE §8. |
| UI too small/large on a HiDPI display | Override font scale, e.g. `VIS_UI_SCALE=1.2 python -m vis.main`. |
