#!/usr/bin/env python3
"""
log_report.py — Generate human-readable and analysis-ready reports from study log CSVs.

Three outputs per session log:
  <session>_report.txt       — human-readable timeline (for qualitative review)
  <session>_analysis.csv     — flat, no-JSON CSV (for pandas / Excel analysis)

Handles both old schema (no source column) and new schema (source as top-level column).

Usage:
    python vis/log_report.py                        # auto-finds latest log in study_logs/
    python vis/log_report.py study_logs/P4_*.csv    # specific file(s)
    python vis/log_report.py --all                  # process all logs in study_logs/
    python vis/log_report.py --all --output-dir reports/
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# ── Events skipped in the report (too noisy / covered by bookend events) ──────
SKIP_EVENTS = {
    "frame_changed",     # continuous during playback; bookended by play_started/paused
    "slider_scrub",      # mid-scrub; bookended by slider_scrub_start/end
    "plot_hover",        # mid-hover; bookended by plot_hover_start/end
    "navigation_event",  # internal sync event, not a user action
    "layout_change",     # window resize noise
    "impression_start",  # we only care about impression_end (dwell time)
    "ui_state",          # internal state snapshot, not a user action
}

# Minimum dwell time for impression_end to appear in the report
MIN_DWELL_MS = 1000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_payload(row: dict) -> dict:
    try:
        return json.loads(row.get("event_payload_json", "{}") or "{}") or {}
    except Exception:
        return {}


def _get_source(row: dict, payload: dict) -> str:
    """Works with both old schema (no source column) and new schema."""
    source = row.get("source", "").strip()
    if not source:
        source = payload.get("source", "unknown")
    return source or "unknown"


def _format_delta(prev_ms: float, curr_ms: float) -> str:
    delta = curr_ms - prev_ms
    if delta < 0:
        return ""
    if delta < 1000:
        return f"+{int(delta)}ms"
    elif delta < 60_000:
        return f"+{delta / 1000:.1f}s"
    else:
        m = int(delta // 60_000)
        s = int((delta % 60_000) / 1000)
        return f"+{m}m{s:02d}s"


def _describe(row: dict, payload: dict) -> str | None:
    """Return a human-readable description for a log event, or None to skip it."""
    et = row["event_type"]
    step = row.get("time_step", "?")
    target = row.get("target_id", "")

    if et == "session_start":
        return "Session started"

    elif et == "session_end":
        total = payload.get("total_elapsed_s", "?")
        return f"Session ended  (total: {total}s)"

    elif et == "episode_loaded":
        ep = payload.get("episode_id", target)
        return f"Episode loaded: {ep}"

    elif et == "episode_context":
        steps = payload.get("total_steps", "?")
        reward = payload.get("reward_total", "?")
        return f"Episode context snapshot: {steps} steps, cumulative reward {reward}"

    elif et == "play_started":
        return f"Playback started @ step {step}"

    elif et == "play_paused":
        return f"Playback paused @ step {step}"

    elif et == "step_forward":
        return f"Stepped forward → step {step}"

    elif et == "step_backward":
        return f"Stepped back → step {step}"

    elif et == "restart":
        return "Video restarted"

    elif et == "speed_change":
        speed = payload.get("playback_speed", payload.get("speed_text", "?"))
        return f"Speed changed to {speed}x"

    elif et == "slider_scrub_start":
        return f"Scrub started @ step {step}"

    elif et == "slider_scrub_end":
        return f"Scrub ended → step {step}"

    elif et == "slider_jump":
        return f"Jumped to step {step}"

    elif et == "tab_switch":
        tab = payload.get("tab", target)
        return f"Switched tab → {tab}"

    elif et == "toggle_info_plots":
        return "Toggled info panel"

    elif et == "view_toggle":
        view = payload.get("view", target)
        return f"Toggled view → {view}"

    elif et == "achievement_clicked":
        ach = payload.get("achievement", target)
        return f"Clicked achievement: {ach}"

    elif et == "plot_hover_start":
        plot = payload.get("plot", target)
        hstep = payload.get("step", step)
        return f"Hover start: {plot} @ step {hstep}"

    elif et == "plot_hover_end":
        plot = payload.get("plot", target)
        dur = payload.get("duration_ms", "?")
        hstep = payload.get("step", step)
        return f"Hover end: {plot}  (dwell {dur}ms) @ step {hstep}"

    elif et == "plot_click":
        plot = payload.get("plot", target)
        return f"Clicked: {plot} @ step {payload.get('step', step)}"

    elif et == "plot_viewport_changed":
        plot = payload.get("plot", target)
        return f"Zoomed/panned: {plot}"

    elif et == "legend_toggle":
        plot = payload.get("plot", target)
        return f"Toggled legend on: {plot}"

    elif et == "decision_point_click":
        return f"Clicked decision point @ step {payload.get('step', step)}"

    elif et == "decision_point_hover":
        return f"Hovered decision point @ step {payload.get('step', step)}"

    elif et == "impression_end":
        dur = int(payload.get("duration_ms", 0))
        if dur < MIN_DWELL_MS:
            return None  # skip brief flickers
        return f"Viewed '{target}' for {dur / 1000:.1f}s"

    elif et == "window_focus_gained":
        return "Window gained focus"

    elif et == "window_focus_lost":
        return "Window lost focus"

    elif et in ("task_start", "task_end", "question_start", "question_end"):
        return et.replace("_", " ").title()

    elif et == "answer_submit":
        return "Answer submitted"

    elif et == "think_aloud_start":
        return "Think-aloud started"

    elif et == "think_aloud_end":
        return "Think-aloud ended"

    elif et in SKIP_EVENTS:
        return None

    else:
        return et  # fallback: show raw event type


# ── Core report builder ───────────────────────────────────────────────────────

def generate_report(csv_path: Path) -> str:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return f"Empty log file: {csv_path}"

    # ── Enrich each row with parsed payload and source ────────────────────────
    for row in rows:
        payload = _parse_payload(row)
        row["_payload"] = payload
        row["_source"] = _get_source(row, payload)

    # ── Session metadata ──────────────────────────────────────────────────────
    participant_id = rows[0].get("participant_id", "?")
    session_id = rows[0].get("session_id", "?")

    system_count = sum(1 for r in rows if r["_source"] == "system_init")
    user_count = len(rows) - system_count

    try:
        t_start = float(rows[0].get("elapsed_s", 0))
        t_end = float(rows[-1].get("elapsed_s", 0))
        total_s = int(t_end - t_start)
        duration_str = f"{total_s // 60}m {total_s % 60:02d}s"
    except Exception:
        duration_str = "unknown"

    # ── Collect episode metadata (from episode_context events) ────────────────
    ep_context: dict[str, dict] = {}
    for row in rows:
        if row["event_type"] == "episode_context":
            ep_id = row.get("episode_id", "")
            if ep_id:
                ep_context[ep_id] = row["_payload"]

    # Count distinct episodes
    seen_episodes: list[str] = []
    for row in rows:
        ep = row.get("episode_id", "")
        if ep and (not seen_episodes or seen_episodes[-1] != ep):
            seen_episodes.append(ep)

    # ── Build report lines ────────────────────────────────────────────────────
    W = 80
    SEP = "=" * W
    THIN = "─" * W

    lines = [
        SEP,
        f"  Study Session Report  —  Participant: {participant_id}",
        f"  Session ID   : {session_id}",
        f"  Duration     : {duration_str}",
        f"  Episodes     : {len(seen_episodes)}",
        f"  Total events : {len(rows)}  |  User-initiated: {user_count}  |  System-init: {system_count}",
        f"  Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        SEP,
    ]

    if system_count:
        lines.append(
            f"\n  [System-init events: {system_count} rows skipped from timeline below]\n"
        )

    current_ep: str = ""
    ep_event_num = 0
    prev_mono_ms: float | None = None

    for row in rows:
        source = row["_source"]
        payload = row["_payload"]
        et = row["event_type"]
        ep_id = row.get("episode_id", "")

        # ── Episode section header ────────────────────────────────────────────
        if ep_id and ep_id != current_ep:
            current_ep = ep_id
            ep_event_num = 0
            ctx = ep_context.get(ep_id, {})
            steps = ctx.get("total_steps", "?")
            reward = ctx.get("reward_total", "?")
            lines.append("")
            lines.append(THIN)
            lines.append(f"  Episode: {ep_id}")
            lines.append(f"  {steps} steps  |  reward: {reward}")
            lines.append(THIN)

        # ── Skip system-init and noisy events ─────────────────────────────────
        if source == "system_init":
            continue
        if et in SKIP_EVENTS:
            continue

        desc = _describe(row, payload)
        if desc is None:
            continue

        ep_event_num += 1

        try:
            mono_ms = float(row.get("monotonic_ms", 0))
        except Exception:
            mono_ms = 0.0

        ts = row.get("timestamp_iso", "")
        ts_short = ts[11:23] if len(ts) >= 23 else ts  # HH:MM:SS.mmm

        delta_str = _format_delta(prev_mono_ms, mono_ms) if prev_mono_ms is not None else ""
        prev_mono_ms = mono_ms

        lines.append(f"  [{ep_event_num:>4}]  {ts_short}  {delta_str:>10}  —  {desc}")

    lines += ["", SEP, "  End of report", SEP, ""]
    return "\n".join(lines)


# ── Analysis CSV ─────────────────────────────────────────────────────────────
#
# A flat, no-JSON CSV suitable for Excel or pandas.
# Skips only the truly repetitive mid-stream events; everything else stays.
# Key payload fields are extracted into their own columns.

ANALYSIS_SKIP = {
    "frame_changed",  # fires every frame during playback — extremely noisy
    "slider_scrub",   # intermediate mid-scrub events (start/end are kept)
    "plot_hover",     # intermediate mid-hover events (start/end are kept)
}

ANALYSIS_COLUMNS = [
    "event_seq",
    "timestamp_iso",
    "elapsed_s",
    "delta_ms",         # milliseconds since the previous row
    "participant_id",
    "session_id",
    "episode_id",
    "episode_number",   # ordinal 1-N — which episode within the session
    "time_step",        # playback position at time of event
    "event_type",
    "event_category",
    "source",           # user_input | system_init | system_sync | unknown
    "is_user_action",   # TRUE when source == user_input
    "target_id",
    # ── Payload fields extracted into their own columns ──────────────────────
    "plot",             # which plot was interacted with
    "plot_step",        # step position on the plot (for hover / click)
    "duration_ms",      # dwell time (for hover_end / impression_end)
    "tab",              # tab name (for tab_switch)
    "achievement",      # achievement name (for achievement_clicked)
    "playback_speed",   # speed value (for speed_change)
    "view",             # view name (for view_toggle)
    # ── Episode context (filled from episode_context snapshot) ───────────────
    "episode_total_steps",
    "episode_reward_total",
]


def _sibling_dir(csv_path: Path, subfolder: str) -> Path:
    """
    Return the output directory for a generated file.
    If the raw CSV lives inside a 'raw/' folder, place output in the sibling
    subfolder (e.g. raw/ → reports/ or analysis/).
    Otherwise fall back to the CSV's own directory.
    """
    if csv_path.parent.name == "raw":
        out = csv_path.parent.parent / subfolder
    else:
        out = csv_path.parent
    out.mkdir(parents=True, exist_ok=True)
    return out


def generate_analysis_csv(csv_path: Path, output_dir: Path | None = None) -> Path:
    """
    Read a raw study log CSV and write a flat analysis CSV with no JSON blobs.
    Returns the path of the written file.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Pre-scan: collect episode metadata from episode_context events
    ep_context: dict[str, dict] = {}
    for row in rows:
        if row["event_type"] == "episode_context":
            payload = _parse_payload(row)
            ep_id = row.get("episode_id", "")
            if ep_id:
                ep_context[ep_id] = payload

    out_rows = []
    episode_number = 0
    current_ep = ""
    prev_mono_ms: float | None = None

    for row in rows:
        et = row["event_type"]
        if et in ANALYSIS_SKIP:
            continue

        payload = _parse_payload(row)
        source = _get_source(row, payload)

        # Track episode number (increments each time episode_id changes)
        ep_id = row.get("episode_id", "")
        if ep_id and ep_id != current_ep:
            current_ep = ep_id
            episode_number += 1

        # Compute delta from previous row
        try:
            mono_ms = float(row.get("monotonic_ms", 0))
        except Exception:
            mono_ms = 0.0
        delta_ms = round(mono_ms - prev_mono_ms, 1) if prev_mono_ms is not None else ""
        prev_mono_ms = mono_ms

        ctx = ep_context.get(ep_id, {})

        out_rows.append({
            "event_seq":            row.get("event_seq", ""),
            "timestamp_iso":        row.get("timestamp_iso", ""),
            "elapsed_s":            row.get("elapsed_s", ""),
            "delta_ms":             delta_ms,
            "participant_id":       row.get("participant_id", ""),
            "session_id":           row.get("session_id", ""),
            "episode_id":           ep_id,
            "episode_number":       episode_number if ep_id else "",
            "time_step":            row.get("time_step", ""),
            "event_type":           et,
            "event_category":       row.get("event_category", ""),
            "source":               source,
            "is_user_action":       "TRUE" if source == "user_input" else "FALSE",
            "target_id":            row.get("target_id", ""),
            "plot":                 payload.get("plot", ""),
            "plot_step":            payload.get("step", ""),
            "duration_ms":          payload.get("duration_ms", ""),
            "tab":                  payload.get("tab", ""),
            "achievement":          payload.get("achievement", ""),
            "playback_speed":       payload.get("playback_speed", ""),
            "view":                 payload.get("view", ""),
            "episode_total_steps":  ctx.get("total_steps", ""),
            "episode_reward_total": ctx.get("reward_total", ""),
        })

    out_dir = output_dir or _sibling_dir(csv_path, "analysis")
    out_path = out_dir / (csv_path.stem + "_analysis.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    return out_path


def generate_txt_report(csv_path: Path, output_dir: Path | None = None) -> Path:
    """Write the human-readable timeline report and return its path."""
    out_dir = output_dir or _sibling_dir(csv_path, "reports")
    out_path = out_dir / (csv_path.stem + "_report.txt")
    out_path.write_text(generate_report(csv_path), encoding="utf-8")
    return out_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a human-readable timeline report from a study log CSV."
    )
    parser.add_argument(
        "csv_files", nargs="*", help="Path(s) to study log CSV file(s)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all CSVs in study_logs/"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports (default: same folder as each CSV)",
    )
    args = parser.parse_args()

    if args.all:
        paths = sorted(Path("study_logs").glob("*.csv"))
        if not paths:
            print("No CSV files found in study_logs/")
            sys.exit(1)
    elif args.csv_files:
        paths = [Path(p) for p in args.csv_files]
    else:
        # Auto-find the most recently modified CSV
        candidates = sorted(
            Path("study_logs").glob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No CSV files found in study_logs/. Pass a path explicitly.")
            sys.exit(1)
        paths = [candidates[0]]
        print(f"Auto-selected: {paths[0]}")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in paths:
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            continue

        txt_path      = generate_txt_report(csv_path,      output_dir=out_dir)
        analysis_path = generate_analysis_csv(csv_path,    output_dir=out_dir)
        print(f"  txt report  : {txt_path}")
        print(f"  analysis CSV: {analysis_path}")


if __name__ == "__main__":
    main()
