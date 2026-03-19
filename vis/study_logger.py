import atexit
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QEvent, QObject  # type: ignore[import-not-found]


class _TaggedVisibilityFilter(QObject):
    """Event filter that measures dwell time and emits impression events."""

    def __init__(self, logger: "StudyLogger", tag: str, widget: Any):
        super().__init__(widget)
        self.logger = logger
        self.tag = tag
        self.widget = widget
        self.visible_since: Optional[float] = None
        self._last_known_visible: Optional[bool] = None

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        event_type = event.type()

        if event_type == QEvent.Show:
            self._on_visibility_state(True)
        elif event_type == QEvent.Hide:
            self._on_visibility_state(False)

        return False

    def _on_visibility_state(self, is_visible: bool):
        if self._last_known_visible is is_visible:
            return
        self._last_known_visible = is_visible

        if is_visible:
            self._start_impression()
        else:
            self._end_impression(reason="leave")

    def close(self):
        self._end_impression(reason="close")

    def _start_impression(self):
        if self.visible_since is not None:
            return

        self.visible_since = time.monotonic()
        self.logger.log(
            "impression_start",
            {
                "target_id": self.tag,
                "visible_ratio": self._visible_ratio(),
            },
        )

    def _end_impression(self, reason: str):
        if self.visible_since is None:
            return

        now = time.monotonic()
        duration_ms = int((now - self.visible_since) * 1000)
        self.visible_since = None

        # Ignore flicker-level impressions that are likely layout churn.
        if reason != "close" and duration_ms < 100:
            return

        self.logger.log(
            "impression_end",
            {
                "target_id": self.tag,
                "reason": reason,
                "duration_ms": duration_ms,
                "visible_ratio": self._visible_ratio(),
            },
        )

    def _visible_ratio(self) -> float:
        try:
            return 1.0 if self.widget.isVisible() else 0.0
        except Exception:
            return 1.0


class StudyLogger:
    """Structured session logger for user-study interactions in the Vis UI."""

    EVENT_TYPES = {
        # Session lifecycle
        "session_start",
        "session_end",
        "episode_loaded",
        "episode_context",
        "task_start",
        "task_end",
        "question_start",
        "question_end",
        "answer_submit",
        "think_aloud_start",
        "think_aloud_end",
        # Video controls
        "play_started",
        "play_paused",
        "step_forward",
        "step_backward",
        "restart",
        "speed_change",
        # Timeline and navigation
        "slider_scrub_start",
        "slider_scrub",
        "slider_scrub_end",
        "slider_jump",
        "frame_changed",
        "navigation_event",
        # UI state and panels
        "toggle_info_plots",
        "tab_switch",
        "achievement_clicked",
        "view_toggle",
        "ui_state",
        # Plot interactions
        "plot_hover",
        "plot_click",
        "plot_viewport_changed",
        "legend_toggle",
        "decision_point_click",
        "decision_point_hover",
        "plot_hover_start",
        "plot_hover_end",
        # Visibility / impressions
        "impression_start",
        "impression_end",
        # Window / layout
        "window_focus_gained",
        "window_focus_lost",
        "layout_change",
    }

    COLUMNS = [
        "event_seq",
        "timestamp_iso",
        "monotonic_ms",
        "elapsed_s",
        "session_id",
        "participant_id",
        "event_type",
        "event_category",
        "target_id",
        "interaction_type",
        "episode_id",
        "frame",
        "time_step",
        "ui_state_json",
        "event_payload_json",
    ]

    def __init__(self, output_dir: str = "study_logs", participant_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.participant_id = participant_id or "P_unknown"
        self.wall_start = time.time()
        self.monotonic_start = time.monotonic()
        self.event_seq = 0

        self.session_id = f"{self.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.episode_id = ""
        self.current_frame = 0
        self.current_step = 0
        self.ui_state: Dict[str, Any] = {
            "panel": "plots",
            "theme": "light",
            "playback_speed": 1.0,
            "active_tab": "completed",
            "filters": {},
            "navigation": {},
            "viewport": {},
        }

        self._last_emit_by_key: Dict[str, float] = {}
        self._visibility_filters = []

        self._event_category = {
            "session": {
                "session_start",
                "session_end",
                "episode_loaded",
                "episode_context",
            },
            "study": {
                "task_start",
                "task_end",
                "question_start",
                "question_end",
                "answer_submit",
                "think_aloud_start",
                "think_aloud_end",
            },
            "video": {
                "play_started",
                "play_paused",
                "step_forward",
                "step_backward",
                "restart",
                "speed_change",
                "frame_changed",
            },
            "timeline": {
                "slider_scrub_start",
                "slider_scrub",
                "slider_scrub_end",
                "slider_jump",
                "navigation_event",
            },
            "ui": {
                "toggle_info_plots",
                "tab_switch",
                "achievement_clicked",
                "view_toggle",
                "ui_state",
            },
            "plot": {
                "plot_hover",
                "plot_click",
                "plot_viewport_changed",
                "legend_toggle",
                "decision_point_click",
                "decision_point_hover",
                "plot_hover_start",
                "plot_hover_end",
            },
            "visibility": {
                "impression_start",
                "impression_end",
            },
            "window": {
                "window_focus_gained",
                "window_focus_lost",
                "layout_change",
            },
        }

        filename = f"{self.session_id}.csv"
        self._path = self.output_dir / filename
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._file.flush()

        atexit.register(self.close)
        self.log("session_start", {"study_logger": "v2"})

    @property
    def log_path(self) -> Path:
        return self._path

    def set_episode(self, episode_id: str, source: str = "system_init"):
        self.episode_id = episode_id or ""
        self.log("episode_loaded", {"episode_id": self.episode_id, "source": source})

    def set_frame(self, frame: int, step: int):
        self.current_frame = int(max(0, frame))
        self.current_step = int(max(0, step))

    def update_ui_state(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            self.ui_state[key] = value

    def set_viewport_state(self, plot_name: str, x_range: Any, y_range: Any):
        viewport = dict(self.ui_state.get("viewport", {}))
        viewport[plot_name] = {
            "x_range": list(x_range) if x_range is not None else None,
            "y_range": list(y_range) if y_range is not None else None,
        }
        self.ui_state["viewport"] = viewport

    def should_emit(self, throttle_key: str, min_interval_s: float) -> bool:
        now = time.monotonic()
        last = self._last_emit_by_key.get(throttle_key)
        if last is not None and (now - last) < min_interval_s:
            return False
        self._last_emit_by_key[throttle_key] = now
        return True

    def attach_visibility_tag(self, tag: str, widget: Any):
        watcher = _TaggedVisibilityFilter(self, tag, widget)
        widget.installEventFilter(watcher)
        self._visibility_filters.append(watcher)

    def log(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        if event_type not in self.EVENT_TYPES:
            return

        payload = payload or {}
        payload.setdefault("source", "unknown")
        self.event_seq += 1

        category = "other"
        for name, events in self._event_category.items():
            if event_type in events:
                category = name
                break

        target_id = payload.get("target_id", "")
        interaction_type = payload.get("interaction_type", event_type)

        row = {
            "event_seq": self.event_seq,
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "monotonic_ms": int((time.monotonic() - self.monotonic_start) * 1000),
            "elapsed_s": round(time.time() - self.wall_start, 3),
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "event_type": event_type,
            "event_category": category,
            "target_id": target_id,
            "interaction_type": interaction_type,
            "episode_id": self.episode_id,
            "frame": self.current_frame,
            "time_step": self.current_step,
            "ui_state_json": json.dumps(self.ui_state, ensure_ascii=True, separators=(",", ":")),
            "event_payload_json": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        }

        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if getattr(self, "_file", None) is None or self._file.closed:
            return

        for watcher in self._visibility_filters:
            watcher.close()

        self.log("session_end", {"total_elapsed_s": round(time.time() - self.wall_start, 3)})
        self._file.flush()
        self._file.close()


class NoOpStudyLogger:
    """Drop-in logger that performs no persistence when study logging is disabled."""

    def __init__(self):
        self.participant_id = "disabled"
        self.session_id = "disabled"
        self.episode_id = ""
        self.current_frame = 0
        self.current_step = 0
        self.ui_state: Dict[str, Any] = {
            "panel": "plots",
            "theme": "light",
            "playback_speed": 1.0,
            "active_tab": "completed",
            "filters": {},
            "navigation": {},
            "viewport": {},
            "layout": {},
        }

    def set_episode(self, episode_id: str, source: str = "system_init"):
        self.episode_id = episode_id or ""

    def set_frame(self, frame: int, step: int):
        self.current_frame = int(max(0, frame))
        self.current_step = int(max(0, step))

    def update_ui_state(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            self.ui_state[key] = value

    def set_viewport_state(self, plot_name: str, x_range: Any, y_range: Any):
        viewport = dict(self.ui_state.get("viewport", {}))
        viewport[plot_name] = {
            "x_range": list(x_range) if x_range is not None else None,
            "y_range": list(y_range) if y_range is not None else None,
        }
        self.ui_state["viewport"] = viewport

    def should_emit(self, throttle_key: str, min_interval_s: float) -> bool:
        return False

    def attach_visibility_tag(self, tag: str, widget: Any):
        return

    def log(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        return

    def close(self):
        return
