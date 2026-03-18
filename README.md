# Explaining Crafter Agents

Dreamer-based reinforcement learning and analysis toolkit for Crafter, with a PyQt visualization app for timeline, reward, attribution, and explanation playback.

## Quick Start

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
python -m vis.main
```

Run the UI from the project root with module mode (`python -m vis.main`).

## Current Capabilities

- DreamerV2 training with parallel environment rollout.
- Step-level decision attribution logging during training.
- Episode generation with CSV + MP4 output.
- Visualization that synchronizes video, timeline, reward plots, and attribution traces.
- Deterministic explanation panel (`vis/explainer.py`) shown in Achievements mode.

## Main Workflows

### 1) Train Dreamer

Example using the Python API:

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
    video_dir='./videos',
)
PY
```

Example resume:

```bash
python - <<'PY'
from dreamer.train import train_dreamer

train_dreamer(
    total_steps=100000,
    load_checkpoint=True,
    checkpoint_dir='./dreamer_checkpoints/ckpt_250000',
    log_dir='./training_logs',
)
PY
```

You can also use the CLI:

```bash
python -m dreamer.train --mode train --steps 250000 --checkpoint-dir ./dreamer_checkpoints --log-dir ./training_logs
```

### 2) Launch Visualization

```bash
python -m vis.main
```

File menu options in the app:

- Open Random Log and Video
- Open from Results
- Open from Logs Directory

Mode toggle behavior:

- Show Achievements: right pane switches to achievements, bottom pane shows the Explanation Toolbox.
- Show Charts: right pane switches to charts, bottom pane shows the decision-attribution plot.

## Output Conventions

Training (`dreamer/train.py`) creates step-suffixed directories:

- Checkpoints: `.../ckpt_<target_steps>/`
- Logs: `.../log_<target_steps>/`
- Decision attribution file: `decision_attribution.csv` inside the log folder
- Summary metrics file: `dreamer_training_log.csv` inside the log folder

Episode exports (`dreamer/env.py`) are written under results-style folders and include:

- A CSV trajectory file
- A matching MP4 video (when recording is enabled)

## Visualization Data Paths

`vis/config.py` prefers archived runs when available:

- If `archive/run_*/` exists, the UI defaults to the newest run's `logs/` and `results/`.
- Otherwise it falls back to project-root `logs/` and `results/`.

This matches the archive script behavior in `scripts/archive_outputs.sh`.

## Repository Map

- `dreamer/train.py`: Dreamer training loop and CLI
- `dreamer/policy.py`: policy wrapper + replay interactions
- `dreamer/core.py`: DreamerV2 model components
- `dreamer/env.py`: episode rollout and export helpers
- `vis/main.py`: main PyQt application and view switching
- `vis/widgets.py`: charts, achievement panel, explanation panel
- `vis/data_manager.py`: CSV loading, normalization, attribution shaping
- `vis/timeline.py`: frame-step mapping and timeline control
- `vis/video_player.py`: MP4 playback widget
- `vis/explainer.py`: deterministic natural-language explanations
- `vis/config.py`: default log/result path resolution

## Dependencies

Pinned dependencies live in `requirements.txt`.
Core packages:

- tensorflow-macos 2.16.1
- tensorflow-probability 0.20.0
- gym 0.25.2
- crafter 1.8.3
- PyQt5 5.15.11
- pyqtgraph 0.13.7

## Documentation

For deeper architecture and data-flow details, see:

- `Markdown Files/PROJECT_GUIDE.md`
