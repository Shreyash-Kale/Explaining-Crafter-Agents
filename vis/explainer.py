"""Template-based natural language explanations for RL decisions.

This module is deterministic by design so explanations remain stable across runs.
It consumes one step row (plus optional previous row) and produces concise text.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


ACTION_MAPPING = {
    0: "noop",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "do",
    6: "sleep",
    7: "place_stone",
    8: "place_table",
    9: "place_furnace",
    10: "place_plant",
    11: "make_wood_pickaxe",
    12: "make_stone_pickaxe",
    13: "make_iron_pickaxe",
    14: "make_wood_sword",
    15: "make_stone_sword",
    16: "make_iron_sword",
}

# Tunable thresholds for deterministic language buckets.
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.50
VALUE_DELTA_THRESHOLD = 0.05
WORLD_MODEL_HIGH = 0.75
WORLD_MODEL_MEDIUM = 0.40
ENTROPY_HIGH = 0.60
ADVANTAGE_POSITIVE = 0.10
ADVANTAGE_NEGATIVE = -0.10


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float when possible, otherwise return default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_action_name(action: Any, executed_action: Any = None) -> str:
    """Return a readable action name from executed_action/action fields."""
    chosen = executed_action if executed_action is not None else action
    if isinstance(chosen, str):
        stripped = chosen.strip()
        if stripped:
            # Support numeric strings such as "6".
            try:
                action_id = int(stripped)
                return ACTION_MAPPING.get(action_id, f"unknown_{action_id}")
            except ValueError:
                return stripped.lower()

    try:
        action_id = int(chosen)
        return ACTION_MAPPING.get(action_id, f"unknown_{action_id}")
    except (TypeError, ValueError):
        return "unknown"


def _confidence_label(prob: float) -> str:
    if prob >= CONFIDENCE_HIGH:
        return "high confidence"
    if prob >= CONFIDENCE_MEDIUM:
        return "moderate confidence"
    return "low confidence"


def _value_direction(prev_value: float, curr_value: float) -> tuple[str, str]:
    delta = curr_value - prev_value
    if delta > VALUE_DELTA_THRESHOLD:
        return "rising", "anticipating future reward"
    if delta < -VALUE_DELTA_THRESHOLD:
        return "falling", "not expecting an immediate payoff"
    return "stable", "neutral about the immediate outcome"


def _dreamer_world_model_text(world_model_score: float) -> str:
    if world_model_score >= WORLD_MODEL_HIGH:
        return "World-model confidence was high, so the observed outcome matched prediction well."
    if world_model_score >= WORLD_MODEL_MEDIUM:
        return "World-model confidence was moderate, so the outcome was somewhat surprising."
    return "World-model confidence was low, so the outcome was likely unexpected."


def _ppo_entropy_text(entropy: float) -> str:
    if entropy >= ENTROPY_HIGH:
        return "Policy entropy was high, indicating exploratory behavior."
    return "Policy entropy was low, indicating exploitative behavior."


def _ppo_advantage_text(advantage: float) -> str:
    if advantage >= ADVANTAGE_POSITIVE:
        return "Advantage was positive, so this action outperformed baseline expectation."
    if advantage <= ADVANTAGE_NEGATIVE:
        return "Advantage was negative, so this action underperformed baseline expectation."
    return "Advantage was near neutral at this step."


def infer_algorithm(step_row: Dict[str, Any]) -> str:
    """Infer algorithm from available columns in the row."""
    keys = set(step_row.keys())
    if {"exploration_bonus", "world_model_score"}.issubset(keys):
        return "dreamer"
    if {"entropy", "advantage"}.issubset(keys):
        return "ppo"
    return "unknown"


def generate_explanation(
    step_row: Dict[str, Any],
    prev_row: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Generate a deterministic explanation string for one timestep.

    Parameters
    ----------
    step_row:
        Dict-like row with keys such as time_step, action, action_probability,
        value_estimate, exploration_bonus/world_model_score or entropy/advantage.
    prev_row:
        Previous row for value trend estimation. If omitted, current value is reused.
    algorithm:
        Optional explicit value ("dreamer", "ppo", or "unknown").
    """
    algo = (algorithm or infer_algorithm(step_row)).lower()

    step = step_row.get("time_step", step_row.get("step", "?"))
    action_name = _normalize_action_name(
        action=step_row.get("action"),
        executed_action=step_row.get("executed_action"),
    )

    action_prob = _safe_float(step_row.get("action_probability"), default=0.5)
    value_curr = _safe_float(
        step_row.get("value_estimate", step_row.get("value")), default=0.0
    )
    if prev_row is None:
        value_prev = value_curr
    else:
        value_prev = _safe_float(
            prev_row.get("value_estimate", prev_row.get("value")), default=value_curr
        )

    confidence = _confidence_label(action_prob)
    direction, meaning = _value_direction(value_prev, value_curr)

    lines = [
        (
            f"At step {step}, the agent chose {action_name} with {confidence} "
            f"(action probability: {action_prob:.2f})."
        ),
        f"Its value estimate was {direction}, meaning it was {meaning}.",
    ]

    if algo == "dreamer":
        world_model_score = _safe_float(step_row.get("world_model_score"), default=0.5)
        exploration_bonus = _safe_float(step_row.get("exploration_bonus"), default=0.0)
        lines.append(_dreamer_world_model_text(world_model_score))
        if exploration_bonus >= 0.70:
            lines.append("Exploration bonus was elevated, suggesting unfamiliar territory.")
    elif algo == "ppo":
        entropy = _safe_float(step_row.get("entropy"), default=0.5)
        advantage = _safe_float(step_row.get("advantage"), default=0.0)
        lines.append(_ppo_entropy_text(entropy))
        lines.append(_ppo_advantage_text(advantage))
    else:
        lines.append("Algorithm-specific attribution signals were not available for this step.")

    achievement = step_row.get("achievement_unlocked") or step_row.get("achievement")
    if achievement:
        lines.append(f"This step appears to unlock: {achievement}.")

    return " ".join(lines)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Preview template explanations from a CSV.")
    parser.add_argument("csv_path", help="Path to input CSV")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to preview")
    parser.add_argument(
        "--algorithm",
        choices=["dreamer", "ppo", "unknown"],
        default=None,
        help="Force algorithm mode instead of auto-detection",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    limit = max(1, min(args.rows, len(df)))
    for idx in range(limit):
        curr = df.iloc[idx].to_dict()
        prev = df.iloc[max(0, idx - 1)].to_dict()
        print(f"[{idx}] {generate_explanation(curr, prev, algorithm=args.algorithm)}")
