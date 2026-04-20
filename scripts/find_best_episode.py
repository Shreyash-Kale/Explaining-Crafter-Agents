"""
Scans all eval episode directories and ranks episodes by:
  1. Most unique achievement types in a single episode
  2. Highest total reward

Usage:
    python scripts/find_best_episode.py [--eval-dir data/eval]
"""

import json
import argparse
from pathlib import Path


ACHIEVEMENT_KEYS = [
    "achievement_collect_coal",
    "achievement_collect_diamond",
    "achievement_collect_drink",
    "achievement_collect_iron",
    "achievement_collect_sapling",
    "achievement_collect_stone",
    "achievement_collect_wood",
    "achievement_defeat_skeleton",
    "achievement_defeat_zombie",
    "achievement_eat_cow",
    "achievement_eat_plant",
    "achievement_make_iron_pickaxe",
    "achievement_make_iron_sword",
    "achievement_make_stone_pickaxe",
    "achievement_make_stone_sword",
    "achievement_make_wood_pickaxe",
    "achievement_make_wood_sword",
    "achievement_place_furnace",
    "achievement_place_plant",
    "achievement_place_stone",
    "achievement_place_table",
    "achievement_wake_up",
]


def parse_stats(stats_path: Path) -> list[dict]:
    records = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_episode(records: list[dict]) -> dict:
    """Aggregate stats across all lines in a stats.jsonl (usually 1-2 lines)."""
    best_reward = max(r.get("reward", 0) for r in records)
    best_length = max(r.get("length", 0) for r in records)

    # Sum achievement counts across records, then check which types are non-zero
    totals = {k: 0 for k in ACHIEVEMENT_KEYS}
    for r in records:
        for k in ACHIEVEMENT_KEYS:
            totals[k] += r.get(k, 0)

    unique_types = [k.replace("achievement_", "") for k, v in totals.items() if v > 0]
    return {
        "reward": best_reward,
        "length": best_length,
        "unique_count": len(unique_types),
        "unique_types": unique_types,
        "achievement_totals": {k.replace("achievement_", ""): v for k, v in totals.items() if v > 0},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/eval", help="Root eval directory")
    parser.add_argument("--top", type=int, default=10, help="How many top episodes to show")
    args = parser.parse_args()

    eval_root = Path(args.eval_dir)
    episodes = []

    for stats_file in sorted(eval_root.rglob("stats.jsonl")):
        try:
            records = parse_stats(stats_file)
            if not records:
                continue
            score = score_episode(records)
            # Infer checkpoint and episode from path
            parts = stats_file.parts
            checkpoint = next((p for p in parts if "checkpoint_" in p), "?")
            episode = next((p for p in parts if "episode_" in p), "?")
            episodes.append({
                "path": str(stats_file.parent),
                "checkpoint": checkpoint,
                "episode": episode,
                **score,
            })
        except Exception as e:
            print(f"  [warn] skipping {stats_file}: {e}")

    if not episodes:
        print(f"No stats.jsonl files found under {eval_root}")
        return

    print(f"\nFound {len(episodes)} episodes across {eval_root}\n")

    # Rank by unique achievement types (primary) then reward (secondary)
    by_diversity = sorted(episodes, key=lambda x: (x["unique_count"], x["reward"]), reverse=True)
    print(f"{'='*70}")
    print(f"TOP {args.top} BY UNIQUE ACHIEVEMENT TYPES")
    print(f"{'='*70}")
    for i, ep in enumerate(by_diversity[: args.top], 1):
        print(f"\n#{i}  {ep['checkpoint']} / {ep['episode']}")
        print(f"    reward={ep['reward']:.2f}  length={ep['length']}  unique={ep['unique_count']}")
        print(f"    types: {', '.join(ep['unique_types'])}")
        print(f"    path: {ep['path']}")

    print(f"\n{'='*70}")
    by_reward = sorted(episodes, key=lambda x: (x["reward"], x["unique_count"]), reverse=True)
    print(f"TOP {args.top} BY REWARD")
    print(f"{'='*70}")
    for i, ep in enumerate(by_reward[: args.top], 1):
        print(f"\n#{i}  {ep['checkpoint']} / {ep['episode']}")
        print(f"    reward={ep['reward']:.2f}  length={ep['length']}  unique={ep['unique_count']}")
        print(f"    types: {', '.join(ep['unique_types'])}")
        print(f"    path: {ep['path']}")


if __name__ == "__main__":
    main()
