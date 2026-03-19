# main.py - Main application entry point for VisGUI system

import sys
import os
import re
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSplitter, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtCore import Qt, QTimer, QFileSystemWatcher
from PyQt5.QtWidgets import QStackedWidget, QPushButton
from PyQt5.QtWidgets import QCheckBox

from .video_player import VideoPlayerWidget
from .widgets import VisualizationWidget, InfoPanel, ExplanationPanel
from .data_manager import DataManager
from .timeline import TimelineController
from .config import DEFAULT_LOG_DIR, RESULTS_LOG_DIR, VIZ_COLORS, DEFAULT_FPS
from .study_logger import StudyLogger, NoOpStudyLogger
import random


def _study_logging_enabled_from_runtime() -> bool:
    env_value = os.getenv("VIS_STUDY_LOGGING", "").strip().lower()
    if env_value:
        return env_value in {"1", "true", "yes", "on", "y"}

    if "--study-logging" in sys.argv:
        return True
    if "--no-study-logging" in sys.argv:
        return False

    if not sys.stdin or not sys.stdin.isatty():
        return False

    try:
        answer = input("Enable study logging? (y/n) [n]: ").strip().lower()
    except EOFError:
        return False

    return answer in {"y", "yes"}

class MainWindow(QMainWindow):
    """Main application window containing video player and visualization panels"""
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Crafter Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.study_logging_enabled = _study_logging_enabled_from_runtime()
        if self.study_logging_enabled:
            participant_id, ok = QInputDialog.getText(
                self,
                "Study Session",
                "Enter participant ID:",
            )
            self.logger = StudyLogger(participant_id=participant_id.strip() if ok and participant_id else None)
            print("Study logging enabled")
            print(f"Session log: {self.logger.log_path}")
        else:
            self.logger = NoOpStudyLogger()
            print("Study logging disabled")
            print("Enable with --study-logging, VIS_STUDY_LOGGING=1, or answer y at startup")
        self.logger.update_ui_state(theme="light", panel="plots")
        self._loading_phase = True

        self.frame_offset = 1
        self.showing_info_panel = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(True)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.setStyleSheet(
            """
            QSplitter::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #b7b7b7, stop:0.5 #8f8f8f, stop:1 #b7b7b7);
                border-left: 1px solid #7a7a7a;
                border-right: 1px solid #7a7a7a;
            }
            """
        )
        main_layout.addWidget(self.main_splitter)

        self.left_column = QWidget()
        self.left_column_layout = QVBoxLayout(self.left_column)
        self.left_column_layout.setContentsMargins(0, 0, 0, 0)
        self.left_column_layout.setSpacing(6)

        self.right_column = QWidget()
        self.right_column_layout = QVBoxLayout(self.right_column)
        self.right_column_layout.setContentsMargins(0, 0, 0, 0)
        self.right_column_layout.setSpacing(6)

        self.main_splitter.addWidget(self.left_column)
        self.main_splitter.addWidget(self.right_column)
        self.main_splitter.splitterMoved.connect(
            lambda pos, index: self._on_splitter_moved("main_splitter", pos, index)
        )

        self.data_manager = DataManager()

        self.toggle_button = QPushButton()
        self.toggle_button.setFixedHeight(32)
        self.toggle_button.setMinimumWidth(150)
        self.toggle_button.setStyleSheet(
            "QPushButton { background-color: #f7f7f7; border: 1px solid #9a9a9a; border-radius: 6px; padding: 0 12px; }"
            "QPushButton:hover { background-color: #ffffff; border-color: #6f6f6f; }"
            "QPushButton:pressed { background-color: #e8e8e8; }"
        )
        self.toggle_button.clicked.connect(self.toggle_view)

        self.video_player = VideoPlayerWidget()
        self.video_player.add_info_widget(self.toggle_button)
        self.left_column_layout.addWidget(self.video_player, 3)

        self.explanation_panel = ExplanationPanel(self.data_manager)
        self.left_column_layout.addWidget(self.explanation_panel, 1)

        self.right_widget = QStackedWidget()
        self.visualization = VisualizationWidget(self.data_manager)
        self.info_panel = InfoPanel()
        self.info_panel.data_manager = self.data_manager
        self.right_widget.addWidget(self.visualization)
        self.right_widget.addWidget(self.info_panel)
        self.right_widget.setCurrentIndex(0)
        self.right_column_layout.addWidget(self.right_widget, 3)

        self.bottom_right_container = QWidget()
        self.bottom_right_layout = QVBoxLayout(self.bottom_right_container)
        self.bottom_right_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_right_layout.setSpacing(0)
        self.right_column_layout.addWidget(self.bottom_right_container, 1)

        self.empty_bottom_panel = QWidget()
        self.empty_bottom_panel.setStyleSheet(
            "QWidget { background-color: #f5f5f5; border: 1px dashed #d0d0d0; border-radius: 4px; }"
        )

        self.timeline = TimelineController()
        self.video_player.add_bottom_widget(self.timeline)

        self.timeline.position_changed.connect(self.on_timeline_position_changed)
        self.timeline.position_changed.connect(self.visualization.update_decision_marker)
        self.video_player.frame_changed.connect(self.on_video_frame_changed)

        self._connect_logging_signals()
        self._attach_visibility_tracking()

        self.setup_auto_reload()
        self.setup_menu()

        self.main_splitter.setSizes([500, 840])
        self._mount_right_bottom_panel()
        self.update_toggle_button_label()

        self.open_random_files()
        self._loading_phase = False

    def setup_auto_reload(self):
        """Watch vis/*.py and restart process automatically after file saves."""
        self.auto_reload_enabled = True
        self._restarting = False

        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(250)
        self._reload_timer.timeout.connect(self.restart_process)
        self._mount_right_bottom_panel()
        self._watcher = QFileSystemWatcher(self)
        self.vis_dir = os.path.dirname(__file__)
        self.refresh_watched_files()
        self._watcher.addPath(self.vis_dir)
        self._watcher.fileChanged.connect(self.on_source_changed)
        self._watcher.directoryChanged.connect(self.on_source_dir_changed)

    def refresh_watched_files(self):
        """Keep watcher list synced with current vis python files."""
        desired = {
            os.path.join(self.vis_dir, name)
            for name in os.listdir(self.vis_dir)
            if name.endswith('.py') and os.path.isfile(os.path.join(self.vis_dir, name))
        }
        current = set(self._watcher.files())
        to_add = sorted(desired - current)
        to_remove = sorted(current - desired)
        if to_add:
            self._watcher.addPaths(to_add)
        if to_remove:
            self._watcher.removePaths(to_remove)

    def on_source_changed(self, path):
        if not self.auto_reload_enabled:
            return
        # QFileSystemWatcher can drop changed files; add it back.
        if os.path.exists(path) and path not in self._watcher.files():
            self._watcher.addPath(path)
        self._reload_timer.start()

    def on_source_dir_changed(self, _path):
        if not self.auto_reload_enabled:
            return
        self.refresh_watched_files()
        self._reload_timer.start()

    def restart_process(self):
        """Hot-restart the GUI process to reflect code changes without manual relaunch."""
        if self._restarting:
            return
        self._restarting = True
        python_exec = sys.executable
        os.execv(python_exec, [python_exec] + sys.argv)

    def toggle_view(self):
        if self.showing_info_panel:
            self.right_widget.setCurrentIndex(0)  # Show plots
            self.showing_info_panel = False
            panel = "plots"
        else:
            self.right_widget.setCurrentIndex(1)  # Show info panel
            self.showing_info_panel = True
            panel = "achievements"
        self.logger.update_ui_state(panel=panel)
        self.logger.log(
            "toggle_info_plots",
            {
                "target_id": "toggle_info_plots",
                "interaction_type": "button_click",
                "switched_to": panel,
                "source": "user_input",
            },
        )
        self._mount_right_bottom_panel()
        self.update_toggle_button_label()

    def update_toggle_button_label(self):
        """Keep the footer toggle label descriptive of the next action."""
        if self.showing_info_panel:
            self.toggle_button.setText("Show Charts")
        else:
            self.toggle_button.setText("Show Achievements")
        self._mount_right_bottom_panel()
    def _connect_logging_signals(self):
        self.video_player.interaction_event.connect(self._on_widget_interaction)
        self.timeline.interaction_event.connect(self._on_widget_interaction)
        self.visualization.interaction_event.connect(self._on_widget_interaction)
        self.info_panel.interaction_event.connect(self._on_widget_interaction)

    def _attach_visibility_tracking(self):
        self.logger.attach_visibility_tag("video_panel", self.video_player)
        self.logger.attach_visibility_tag("timeline_slider", self.timeline)
        self.logger.attach_visibility_tag("cumulative_plot", self.visualization.cumulative_plot)
        self.logger.attach_visibility_tag("components_plot", self.visualization.components_plot)
        self.logger.attach_visibility_tag("info_panel", self.info_panel)
        self.logger.attach_visibility_tag("explanation_panel", self.explanation_panel)
        self.logger.attach_visibility_tag("decision_container", self.bottom_right_container)

    def _on_widget_interaction(self, event_type, payload):
        payload = payload or {}
        payload.setdefault("source", "system_init" if self._loading_phase else "user_input")

        if event_type == "speed_change":
            self.logger.update_ui_state(playback_speed=payload.get("playback_speed", 1.0))

        if event_type == "tab_switch":
            self.logger.update_ui_state(active_tab=payload.get("tab", "completed"))

        if event_type == "plot_viewport_changed":
            plot_name = payload.get("plot", "unknown")
            self.logger.set_viewport_state(
                plot_name,
                payload.get("x_range"),
                payload.get("y_range"),
            )

        self.logger.log(event_type, payload)

    def _on_view_toggle(self, view_name, visible):
        self.visualization.toggle_view(view_name, visible)
        filters_state = dict(self.logger.ui_state.get("filters", {}))
        filters_state[view_name] = bool(visible)
        self.logger.update_ui_state(filters=filters_state)
        self.logger.log(
            "view_toggle",
            {
                "target_id": "view_menu",
                "interaction_type": "menu_toggle",
                "view": view_name,
                "visible": bool(visible),
                "source": "user_input",
            },
        )

    def _on_auto_reload_toggled(self, checked):
        self.auto_reload_enabled = checked
        self.logger.log(
            "view_toggle",
            {
                "target_id": "auto_reload",
                "interaction_type": "menu_toggle",
                "view": "auto_reload",
                "visible": bool(checked),
                "source": "user_input",
            },
        )

    def _log_study_marker(self, marker_event_type):
        self.logger.log(
            marker_event_type,
            {
                "target_id": "study_menu",
                "interaction_type": "study_marker",
                "marker": marker_event_type,
                "source": "user_input",
            },
        )

    def _on_splitter_moved(self, splitter_name, pos, index):
        if not self.logger.should_emit(f"layout_splitter_{splitter_name}", 0.10):
            return

        sizes = self.main_splitter.sizes()
        layout_state = dict(self.logger.ui_state.get("layout", {}))
        layout_state[splitter_name] = list(sizes)
        self.logger.update_ui_state(layout=layout_state)

        self.logger.log(
            "layout_change",
            {
                "target_id": splitter_name,
                "interaction_type": "splitter_move",
                "splitter": splitter_name,
                "position": int(pos),
                "handle_index": int(index),
                "sizes": list(sizes),
                "source": "user_input",
            },
        )



    
    def find_all_csv_files(self):
        """Find all CSV files in both logs and results directories"""
        csv_files = []
        
        # Check logs directory
        if os.path.exists(DEFAULT_LOG_DIR):
            log_csv_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.csv')]
            csv_files.extend([(os.join(DEFAULT_LOG_DIR, f), 'log') for f in log_csv_files])
        
        # Check results directory
        if os.path.exists(RESULTS_LOG_DIR):
            # Recursively walk through results directory structure
            for root, dirs, files in os.walk(RESULTS_LOG_DIR):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append((os.path.join(root, file), 'result'))
        
        return csv_files

    def open_random_files(self):
        """Randomly select only a validated CSV/MP4 pair from logs."""

        if not os.path.isdir(DEFAULT_LOG_DIR):
            print(f"Logs directory not found: {DEFAULT_LOG_DIR}")
            return

        pairs = self._build_valid_log_video_pairs(DEFAULT_LOG_DIR)
        if not pairs:
            print("No validated CSV/MP4 pairs found in logs directory.")
            return

        csv_path, video_path = random.choice(pairs)
        self.load_data(csv_path, video_path)

    def _extract_csv_timestamp(self, csv_path):
        """Parse event log timestamp from filenames like event_log_17.03_12.13.09.csv."""

        name = os.path.basename(csv_path)
        match = re.match(r"event_log_(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})\.csv$", name)
        if not match:
            return None

        day, month, hour, minute, second = map(int, match.groups())
        try:
            year = datetime.fromtimestamp(os.path.getmtime(csv_path)).year
            return datetime(year, month, day, hour, minute, second)
        except (ValueError, OSError):
            return None

    def _extract_video_timestamp(self, video_name):
        """Parse MP4 timestamp from filenames like 20250317T121309-ach1-len155.mp4."""

        match = re.match(r"(\d{8}T\d{6})-ach\d+-len\d+\.mp4$", video_name)
        if not match:
            return None

        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            return None

    def _find_matching_video_for_csv(self, csv_path, video_dir, max_seconds=900):
        """Find the nearest timestamp-matched video for a CSV; return None when ambiguous."""

        csv_time = self._extract_csv_timestamp(csv_path)
        if csv_time is None:
            return None

        candidates = []
        for filename in os.listdir(video_dir):
            if not filename.endswith('.mp4'):
                continue
            video_time = self._extract_video_timestamp(filename)
            if video_time is None:
                continue
            delta_seconds = abs((video_time - csv_time).total_seconds())
            if delta_seconds <= max_seconds:
                candidates.append((delta_seconds, filename))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return os.path.join(video_dir, candidates[0][1])

    def _build_valid_log_video_pairs(self, log_dir):
        """Build CSV/MP4 pairs using timestamp matching and strict filtering."""

        pairs = []
        for filename in os.listdir(log_dir):
            if not filename.endswith('.csv'):
                continue
            csv_path = os.path.join(log_dir, filename)

            base_name = os.path.splitext(filename)[0]
            base_video = os.path.join(log_dir, f"{base_name}.mp4")
            if os.path.exists(base_video):
                pairs.append((csv_path, base_video))
                continue

            matched_video = self._find_matching_video_for_csv(csv_path, log_dir)
            if matched_video:
                pairs.append((csv_path, matched_video))

        return pairs

    def setup_menu(self):
        """Create the application menu bar with actions"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Open random video and log
        random_action = file_menu.addAction('Open Random Log and Video')
        random_action.triggered.connect(self.open_random_files)
        
        # Open submenu for results
        results_menu = file_menu.addMenu('Open from Results')
        
        # Action to browse results directory
        browse_results = results_menu.addAction('Browse Results Directory...')
        browse_results.triggered.connect(self.browse_results_directory)
        
        # Add quick access to recent episodes if they exist
        if os.path.exists(RESULTS_LOG_DIR):
            self.populate_recent_episodes_menu(results_menu)
        
        # Open logs action
        open_logs_action = file_menu.addAction('Open from Logs Directory...')
        open_logs_action.triggered.connect(self.open_log_files)
        
        # Exit action
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Toggle visualization types
        show_cumulative = view_menu.addAction('Show Cumulative Rewards')
        show_cumulative.setCheckable(True)
        show_cumulative.setChecked(True)
        show_cumulative.triggered.connect(lambda checked: self._on_view_toggle('cumulative', checked))
        
        show_components = view_menu.addAction('Show Reward Components')
        show_components.setCheckable(True)
        show_components.setChecked(True)
        show_components.triggered.connect(lambda checked: self._on_view_toggle('components', checked))

        show_decision = view_menu.addAction('Show Decision Attribution')
        show_decision.setCheckable(True)
        show_decision.setChecked(True)
        show_decision.triggered.connect(lambda checked: self._on_view_toggle('decision', checked))

        auto_reload = view_menu.addAction('Auto Reload (Dev)')
        auto_reload.setCheckable(True)
        auto_reload.setChecked(True)
        auto_reload.triggered.connect(self._on_auto_reload_toggled)

        study_menu = menubar.addMenu('Study')
        study_actions = [
            ('Task Start', 'task_start'),
            ('Task End', 'task_end'),
            ('Question Start', 'question_start'),
            ('Question End', 'question_end'),
            ('Answer Submit', 'answer_submit'),
            ('Think Aloud Start', 'think_aloud_start'),
            ('Think Aloud End', 'think_aloud_end'),
        ]
        for label, marker_event in study_actions:
            action = study_menu.addAction(label)
            action.triggered.connect(lambda _checked=False, event_name=marker_event: self._log_study_marker(event_name))



    def populate_recent_episodes_menu(self, menu):
        """Add menu items for recent episodes in the results directory"""
        # Look for dreamer_v2 directory
        dreamer_dir = os.path.join(RESULTS_LOG_DIR, 'dreamer_v2')
        if not os.path.exists(dreamer_dir):
            return
            
        # Look for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(dreamer_dir) 
                        if os.path.isdir(os.path.join(dreamer_dir, d)) 
                        and d.startswith('checkpoint_')]
        
        if not checkpoint_dirs:
            return
        
        # Sort by checkpoint number (descending)
        checkpoint_dirs.sort(reverse=True)
        
        # For the most recent checkpoint, add episode entries
        recent_checkpoint = checkpoint_dirs[0]
        checkpoint_path = os.path.join(dreamer_dir, recent_checkpoint)
        
        # Add a submenu for this checkpoint
        checkpoint_menu = menu.addMenu(f"Recent: {recent_checkpoint}")
        
        # Find episode directories
        episode_dirs = [d for d in os.listdir(checkpoint_path)
                    if os.path.isdir(os.path.join(checkpoint_path, d))
                    and d.startswith('episode_')]
        
        # Sort by episode number
        episode_dirs.sort()
        
        # Add the most recent 5 episodes
        for episode_dir in episode_dirs[-5:]:
            episode_path = os.path.join(checkpoint_path, episode_dir)
            
            # Look for data.csv in this episode
            data_path = os.path.join(episode_path, 'data.csv')
            if os.path.exists(data_path):
                # Add menu item for this episode
                episode_action = checkpoint_menu.addAction(f"{episode_dir}")
                # Use lambda with default arg to capture current value
                episode_action.triggered.connect(
                    lambda checked, path=data_path: self.load_episode(path)
                )

    # VisMain.py  –  inside load_episode()
    def load_episode(self, csv_path):
        dir_path = os.path.dirname(csv_path)

        # NEW: take *any* mp4 in this folder
        video_files = [f for f in os.listdir(dir_path) if f.endswith(".mp4")]
        video_path  = os.path.join(dir_path, video_files[0]) if video_files else None

        if not video_path:               # still nothing? ask user
            video_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", dir_path, "Video Files (*.mp4)"
            )
            if not video_path:
                return

        self.load_data(csv_path, video_path)


    def browse_results_directory(self):
        """Browse the results directory structure to find CSV files"""
        if not os.path.exists(RESULTS_LOG_DIR):
            print("Results directory does not exist.")
            return
            
        # Use directory dialog to browse the complex structure
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Episode Directory", RESULTS_LOG_DIR
        )
        
        if not dir_path:
            return  # User cancelled
            
        # Look for data.csv in this directory
        csv_path = os.path.join(dir_path, 'data.csv')
        if not os.path.exists(csv_path):
            # Try to find any CSV
            csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(dir_path, csv_files[0])
            else:
                print(f"No CSV files found in {dir_path}")
                return
        
        # Find video in the same directory
        video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
        
        video_path = None
        if video_files:
            # Take the first video file
            video_path = os.path.join(dir_path, video_files[0])
        
        # If no video found, look for video elsewhere
        if not video_path:
            video_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", dir_path, "Video Files (*.mp4)"
            )
            if not video_path:
                return  # User cancelled
        
        # Load the data
        self.load_data(csv_path, video_path)

    def open_log_files(self):
        """Open dialog to select files from logs directory"""
        log_file, _ = QFileDialog.getOpenFileName(
            self, "Select Event Log File", DEFAULT_LOG_DIR, "CSV Files (*.csv)"
        )
        
        if log_file:
            self._handle_file_selection(log_file)

    def _handle_file_selection(self, log_file):
        """Handle file selection from either directory"""
        # Try to find corresponding video with same base name
        base_name = os.path.splitext(os.path.basename(log_file))[0]
        video_dir = os.path.dirname(log_file)
        possible_video = os.path.join(video_dir, f"{base_name}.mp4")
        video_file = possible_video if os.path.exists(possible_video) else None

        if not video_file:
            video_file = self._find_matching_video_for_csv(log_file, video_dir)
        
        # If no matching video found, ask user to select
        if not video_file:
            video_file, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", video_dir, "Video Files (*.mp4)"
            )
            
        if video_file:
            self.load_data(log_file, video_file)
    
    def open_files(self):
        """Open dialog to select log and video files"""
        
        log_file, _ = QFileDialog.getOpenFileName(
            self, "Select Event Log File", DEFAULT_LOG_DIR, "CSV Files (*.csv)"
        )
        
        if log_file:
            # Try to find corresponding video with same base name
            base_name = os.path.splitext(os.path.basename(log_file))[0]
            video_dir = os.path.dirname(log_file)
            possible_video = os.path.join(video_dir, f"{base_name}.mp4")
            
            video_file = possible_video if os.path.exists(possible_video) else None
            
            # If no matching video found, ask user to select
            if not video_file:
                video_file, _ = QFileDialog.getOpenFileName(
                    self, "Select Video File", video_dir, "Video Files (*.mp4)"
                )
            
            if video_file:
                self.load_data(log_file, video_file)
    
    def load_latest_data(self):
        """Find and load the most recent log and video files"""
        
        try:
            # Find CSV files in log directory
            csv_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.csv')]
            
            if not csv_files:
                return
            
            # Sort by modification time (newest first)
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(DEFAULT_LOG_DIR, x)), reverse=True)
            latest_csv = os.path.join(DEFAULT_LOG_DIR, csv_files[0])
            
            # Find video files in log directory
            video_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.mp4')]
            
            if not video_files:
                return
                
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(os.path.join(DEFAULT_LOG_DIR, x)), reverse=True)
            latest_video = os.path.join(DEFAULT_LOG_DIR, video_files[0])
            
            self.load_data(latest_csv, latest_video)
            
        except Exception as e:
            print(f"Error loading latest data: {e}")
    
    def load_data(self, log_file, video_file):
        """Load data from log and video files"""

        print(f"\n📊 Loading CSV: {log_file}")
        print(f"🎬 Loading Video: {video_file}\n")
        source = "system_init" if self._loading_phase else "user_input"
        self.logger.set_episode(os.path.basename(log_file), source=source)

        if not self.data_manager.load_data(log_file):
            print(f"Failed to load data from {log_file}")
            return False
    
        # Load event data
        self.data_manager.load_data(log_file)

        total_steps = len(self.data_manager.time_steps)
        reward_values = [float(v) for v in self.data_manager.reward_log]
        positive_count = sum(1 for v in reward_values if v > 0)
        negative_count = sum(1 for v in reward_values if v < 0)
        reward_total = float(sum(reward_values)) if reward_values else 0.0

        ppo_values = self.data_manager.get_ppo_entropy_norm()
        dreamer_values = self.data_manager.get_dreamer_explore_norm()
        if len(ppo_values) == total_steps and np.any(ppo_values):
            algorithm_hint = "ppo"
        elif len(dreamer_values) == total_steps and np.any(dreamer_values):
            algorithm_hint = "dreamer"
        else:
            algorithm_hint = "unknown"

        component_summary = {}
        for name, values in self.data_manager.reward_components.items():
            numeric = [float(v) for v in values]
            if not numeric:
                continue
            component_summary[name] = {
                "mean": round(float(sum(numeric) / len(numeric)), 4),
                "max": round(float(max(numeric)), 4),
                "min": round(float(min(numeric)), 4),
                "non_zero_count": int(sum(1 for v in numeric if abs(v) > 1e-9)),
            }

        self.logger.log(
            "episode_context",
            {
                "target_id": "episode_context",
                "interaction_type": "episode_snapshot",
                "episode_id": os.path.basename(log_file),
                "total_steps": int(total_steps),
                "reward_total": round(reward_total, 4),
                "positive_reward_count": int(positive_count),
                "negative_reward_count": int(negative_count),
                "algorithm_hint": algorithm_hint,
                "component_summary": component_summary,
                "source": source,
            },
        )
        
                
        # Pass data to visualization
        self.visualization.set_data(
            self.data_manager.time_steps,
            self.data_manager.reward_log,
            self.data_manager.action_log,  # Use action IDs directly
            self.data_manager.reward_components
        )

        # Build / refresh the decision-attribution comparison plot
        self.visualization.rebuild_decision_plot()
        if hasattr(self.visualization, "decision_plot") and self.visualization.decision_plot is not None:
            self.logger.attach_visibility_tag("decision_plot", self.visualization.decision_plot)
        self._mount_right_bottom_panel()

        
        # Pass video to player
        self.video_player.load_video(video_file)

        
        # Setup timeline controller
        total_steps = len(self.data_manager.time_steps)
        total_frames = self.video_player.total_frames
        self.timeline.setup(total_steps, total_frames)
        self.timeline.update_mapping_status(0, self.timeline.frame_to_step(0, offset=self.frame_offset), offset=self.frame_offset)

        # Prime right-side achievements and bottom explanation at episode start.
        self.info_panel.update_state(0)
        self.explanation_panel.update_step(0)
        
        # Update window title
        self.setWindowTitle(f"Crafter Analysis - {os.path.basename(log_file)}")

    def focusInEvent(self, event):
        self.logger.log(
            "window_focus_gained",
            {
                "target_id": "main_window",
                "interaction_type": "focus_in",
                "source": "user_input",
            },
        )
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.logger.log(
            "window_focus_lost",
            {
                "target_id": "main_window",
                "interaction_type": "focus_out",
                "source": "user_input",
            },
        )
        super().focusOutEvent(event)

    def resizeEvent(self, event):
        if self.logger.should_emit("layout_resize", 0.25):
            size = event.size()
            layout_state = dict(self.logger.ui_state.get("layout", {}))
            layout_state["window_size"] = {
                "width": int(size.width()),
                "height": int(size.height()),
            }
            self.logger.update_ui_state(layout=layout_state)
            self.logger.log(
                "layout_change",
                {
                    "target_id": "main_window",
                    "interaction_type": "window_resize",
                    "width": int(size.width()),
                    "height": int(size.height()),
                    "source": "user_input" if not self._loading_phase else "system_init",
                },
            )
        super().resizeEvent(event)

    def _mount_bottom_panel(self):
        """Backward-compatible wrapper for old call sites."""
        self._mount_right_bottom_panel()

    def _mount_right_bottom_panel(self):
        """Charts mode shows decision plot; achievements mode keeps this area empty."""
        while self.bottom_right_layout.count():
            item = self.bottom_right_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if self.showing_info_panel:
            self.bottom_right_layout.addWidget(self.empty_bottom_panel)
            self.empty_bottom_panel.setVisible(True)
            return

        if hasattr(self.visualization, 'decision_plot'):
            self.bottom_right_layout.addWidget(self.visualization.decision_plot)
            self.visualization.decision_plot.setVisible(True)
        else:
            self.bottom_right_layout.addWidget(self.empty_bottom_panel)
    
    def on_timeline_position_changed(self, position):
        """Handle timeline position changes (0-100%)"""
        
        # Update video position
        self.video_player.seek_percent(position)
        
        # Calculate corresponding step for visualization
        frame = self.video_player.current_frame
        step = self.timeline.frame_to_step(frame, offset=self.frame_offset)
        self.timeline.update_mapping_status(frame, step, offset=self.frame_offset)
        
        # Update visualization position without triggering back-propagation
        self.visualization.update_position(step, from_timeline=True)

        self.info_panel.update_state(step)
        self.explanation_panel.update_step(step)

        self.logger.set_frame(frame, step)
        if self.logger.should_emit("navigation_event", 0.20):
            self.logger.log(
                "navigation_event",
                {
                    "target_id": "timeline_sync",
                    "interaction_type": "timeline_move",
                    "position_percent": round(position, 2),
                    "frame": int(frame),
                    "step": int(step),
                    "source": "system_sync",
                },
            )
    
        # Define an offset for synchronization 
        self.frame_offset = 1

    # Update the on_video_frame_changed method
    def on_video_frame_changed(self, frame):
        """Handle video frame changes"""

         # Avoid processing the same frame repeatedly at video end
        if hasattr(self, 'last_processed_frame') and self.last_processed_frame == frame:
            return
        self.last_processed_frame = frame

        # Update timeline position
        position = (frame / self.video_player.total_frames) * 100
        self.timeline.set_position(position, from_video=True)
        
        # Calculate corresponding step for visualization with offset
        step = self.timeline.frame_to_step(frame, offset=self.frame_offset)
        self.timeline.update_mapping_status(frame, step, offset=self.frame_offset)
        
        # Update visualization position
        self.visualization.update_position(step, from_video=True)

        self.info_panel.update_state(step)
        self.explanation_panel.update_step(step)
        self.logger.set_frame(frame, step)
        if self.logger.should_emit("frame_changed", 0.15):
            self.logger.log(
                "frame_changed",
                {
                    "target_id": "video_frame",
                    "interaction_type": "playback_tick",
                    "frame": int(frame),
                    "step": int(step),
                    "source": "system_sync",
                },
            )

    def closeEvent(self, event):
        self.logger.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


