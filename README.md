# Explaining Crafter Agents

Reinforcement learning training and analysis platform for the Crafter environment, centered on DreamerV2 and a PyQt5 visualization tool for sequential decision analysis.

## What This Project Includes

- DreamerV2 training pipeline with parallel environment rollout.
- Checkpoint/log management with step-based output folders.
- Decision-attribution logging (`action_probability`, `value_estimate`, `exploration_bonus`, `world_model_score`).
- Visualization app that syncs video playback with reward curves, components, and attribution traces.

## Repository Map

- Training entry point: `dreamer/train.py`
- Policy and replay: `dreamer/policy.py`
- Dreamer core models: `dreamer/core.py`
- Environment integration: `dreamer/env.py`
- Visualization app: `vis/main.py`
- Visualization internals:
	- `vis/widgets.py`
	- `vis/data_manager.py`
	- `vis/timeline.py`
	- `vis/video_player.py`
	- `vis/config.py`
- Full technical guide: `Markdown Files/PROJECT_GUIDE.md`

## Requirements

Main pinned dependencies are in `requirements.txt`:

- `tensorflow-macos==2.16.1`
- `tensorflow-probability==0.20.0`
- `gym==0.25.2`
- `crafter==1.8.3`
- `PyQt5==5.15.11`
- `pyqtgraph==0.13.7`

## Setup

```bash
python -m venv crafter_env
source crafter_env/bin/activate
pip install -r requirements.txt
```

## Train DreamerV2

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

Resume from a checkpoint:

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

## Launch Visualization

Run as a module from the project root:

```bash
python -m vis.main
```

Do not run `vis/main.py` directly with `python vis/main.py`; it uses package-relative imports.

## Typical Output Layout

- Checkpoints: `dreamer_checkpoints/ckpt_<STEPS>/ckpt-<N>`
- Training logs: `logs_dreamer/log_<STEPS>/`
	- `dreamer_training_log.csv`
	- `decision_attribution.csv`
- Videos: `videos/`
- Viz-ready logs/results:
	- `logs/`
	- `results/dreamer_v2/checkpoint_*/episode_*/`

## Notes on Recent Cleanup

The following legacy standalone scripts were removed because they were not part of the core runtime/import path:

- `MIXTAPE_Save.py`
- `test_data_loading.py`
- `debug_decision_data.py`
- `analyze_rewards.py`
- duplicate root `VisConfig.py` (use `vis/config.py`)

`SemanticEventDetector.py` is currently optional and not wired into the active visualization pipeline.

## Documentation

For architecture, data flow, and extension guidance, use:

- `Markdown Files/PROJECT_GUIDE.md`
