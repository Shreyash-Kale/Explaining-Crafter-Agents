"""
Standalone evaluation script — runs N episodes with a DreamerV2 checkpoint
and saves per-episode CSVs, videos, and stats.jsonl files.

Designed to run on a lab PC (Linux/Windows) after copying over this repo + a checkpoint.

Usage:
    python scripts/run_eval.py \
        --checkpoint-dir data/checkpoints/ckpt_470000 \
        --num-episodes 50 \
        --out-dir data/eval

After running, use scripts/find_best_episode.py to rank results.
"""

import os
import sys
import argparse
import re
from pathlib import Path

# Must be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # suppress TF info/warnings

# Ensure repo root is on path regardless of where the script is called from
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def find_latest_checkpoint(checkpoint_dir: str) -> int | None:
    """Return the highest step number found in checkpoint_dir."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    files = list(ckpt_dir.glob("ckpt-*.index"))
    if not files:
        return None
    steps = [int(re.search(r"ckpt-(\d+)", f.name).group(1)) for f in files]
    return max(steps)


def run(checkpoint_dir: str, num_episodes: int, out_dir: str, record_video: bool):
    from dreamer.env import create_environment, run_episode
    from dreamer.policy import DreamerPolicy

    ckpt_dir = Path(checkpoint_dir)
    checkpoint_step = find_latest_checkpoint(str(ckpt_dir))
    if checkpoint_step is None:
        print(f"[error] No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    print(f"Checkpoint dir : {ckpt_dir}")
    print(f"Checkpoint step: {checkpoint_step}")
    print(f"Episodes       : {num_episodes}")
    print(f"Output dir     : {out_dir}")
    print()

    out_root = Path(out_dir) / f"checkpoint_{checkpoint_step}"

    for ep_idx in range(1, num_episodes + 1):
        ep_dir = out_root / f"episode_{ep_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        env = create_environment(str(ep_dir))
        agent = DreamerPolicy(
            env,
            training=False,
            checkpoint_dir=str(ckpt_dir),
            load_checkpoint=True,
            checkpoint_number=checkpoint_step,
        )

        def policy_fn(obs):
            return agent(obs)
        policy_fn.__self__ = agent

        total_reward, csv_path = run_episode(
            env, policy_fn,
            output_dir=str(ep_dir),
            episode_id=ep_idx,
            record_video=record_video,
        )
        print(f"  Episode {ep_idx:3d}/{num_episodes}  reward={total_reward:.3f}  → {ep_dir.name}")

    print(f"\nDone. Results saved to {out_root}")
    print("Run  python scripts/find_best_episode.py  to rank episodes.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DreamerV2 checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        default="data/checkpoints/ckpt_470000",
        help="Path to checkpoint directory (e.g. data/checkpoints/ckpt_470000)",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--out-dir", default="data/eval", help="Root output directory")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 recording")
    args = parser.parse_args()

    run(
        checkpoint_dir=args.checkpoint_dir,
        num_episodes=args.num_episodes,
        out_dir=args.out_dir,
        record_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
