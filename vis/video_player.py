# video_player.py - Qt-based video player component

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider,
                           QHBoxLayout, QPushButton, QStyle, QSizePolicy,
                           QToolButton, QMenu, QAction, QActionGroup)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize

class VideoPlayerWidget(QWidget):
    """Widget for playing and controlling video playback"""
    
    # Signal emitted when the current frame changes
    frame_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        
        # Create the video display and controls layout
        self.init_ui()
        
        # Initialize video playback variables
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playback_speed = 1.0
        self.playing = False
        
        # Setup timer for playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # Initially, no video is loaded
        self.reset()
    
    def init_ui(self):
        """Initialize user interface components"""

        control_height = 36
        icon_button_width = 48
        control_button_style = (
            "QPushButton { background-color: #f7f7f7; border: 1px solid #9a9a9a; border-radius: 6px; }"
            "QPushButton:hover { background-color: #ffffff; border-color: #6f6f6f; }"
            "QPushButton:pressed { background-color: #e8e8e8; }"
        )
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(4)
        
        # Create video display label
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.main_layout.addWidget(self.video_label)
        
        # Single aligned footer row with separate left/right groups so macOS
        # native control metrics do not drift visually.
        self.footer_row = QHBoxLayout()
        self.footer_row.setContentsMargins(0, 0, 0, 0)
        self.footer_row.setSpacing(8)

        self.controls_widget = QWidget()
        self.controls_widget.setFixedHeight(control_height)
        self.controls_row = QHBoxLayout(self.controls_widget)
        self.controls_row.setContentsMargins(0, 0, 0, 0)
        self.controls_row.setSpacing(6)

        self.status_widget = QWidget()
        self.status_widget.setFixedHeight(control_height)
        self.status_row = QHBoxLayout(self.status_widget)
        self.status_row.setContentsMargins(0, 0, 0, 0)
        self.status_row.setSpacing(8)
        
        # Step backward button
        self.back_button = QPushButton()
        self.back_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.back_button.setFixedSize(icon_button_width, control_height)
        self.back_button.setStyleSheet(control_button_style)
        self.back_button.clicked.connect(lambda: self.seek_relative(-1))
        self.controls_row.addWidget(self.back_button, 0, Qt.AlignVCenter)

        # Play/pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setFixedSize(icon_button_width, control_height)
        self.play_button.setStyleSheet(control_button_style)
        self.play_button.clicked.connect(self.toggle_play)
        self.controls_row.addWidget(self.play_button, 0, Qt.AlignVCenter)
        
        # Step forward button
        self.forward_button = QPushButton()
        self.forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.forward_button.setFixedSize(icon_button_width, control_height)
        self.forward_button.setStyleSheet(control_button_style)
        self.forward_button.clicked.connect(lambda: self.seek_relative(1))
        self.controls_row.addWidget(self.forward_button, 0, Qt.AlignVCenter)

        # Playback speed menu button (cleaner macOS-style than a plain dropdown).
        self.speed_button = QToolButton()
        self.speed_button.setFixedSize(82, control_height)
        self.speed_button.setText("1x")
        self.speed_button.setPopupMode(QToolButton.InstantPopup)
        self.speed_button.setStyleSheet(
            "QToolButton { background-color: #f7f7f7; border: 1px solid #9a9a9a; border-radius: 6px; padding: 0 8px; font-size: 12px; }"
            "QToolButton:hover { background-color: #ffffff; border-color: #6f6f6f; }"
            "QToolButton::menu-indicator { image: none; width: 0; }"
        )

        self.speed_menu = QMenu(self.speed_button)
        self.speed_group = QActionGroup(self)
        self.speed_group.setExclusive(True)
        self.speed_actions = {}
        for speed_text in ["1x", "2x", "3x", "5x"]:
            action = QAction(speed_text, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, s=speed_text: self.set_playback_speed(s))
            self.speed_menu.addAction(action)
            self.speed_group.addAction(action)
            self.speed_actions[speed_text] = action
        self.speed_button.setMenu(self.speed_menu)
        self.controls_row.addWidget(self.speed_button, 0, Qt.AlignVCenter)

        # Restart button: jump back to frame 0 without auto-playing.
        self.restart_button = QPushButton()
        self.restart_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.restart_button.setFixedSize(icon_button_width, control_height)
        self.restart_button.setStyleSheet(control_button_style)
        self.restart_button.clicked.connect(self.restart_to_start)
        self.controls_row.addWidget(self.restart_button, 0, Qt.AlignVCenter)

        self.info_label = QLabel("Frame: 0 / 0")
        self.info_label.setFixedHeight(control_height)
        self.info_label.setMinimumWidth(110)
        self.info_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.info_label.setStyleSheet("font-size: 12px; padding: 0; margin: 0;")
        self.status_row.addWidget(self.info_label, 0, Qt.AlignVCenter)

        self.footer_row.addWidget(self.controls_widget, 0, Qt.AlignVCenter)
        self.footer_row.addStretch(1)
        self.footer_row.addWidget(self.status_widget, 0, Qt.AlignVCenter)

        self.main_layout.addLayout(self.footer_row)

    def add_info_widget(self, widget):
        """Attach an external control widget to the right-side status group."""
        if hasattr(widget, 'setFixedHeight'):
            widget.setFixedHeight(self.info_label.height())
        self.status_row.addWidget(widget, 0, Qt.AlignVCenter)

    def add_bottom_widget(self, widget):
        """Attach an external widget below the footer row inside the video panel."""
        self.main_layout.addWidget(widget)
    
    def load_video(self, video_path):
        """Load a video file for playback"""
        
        # Close any previously opened video
        if self.cap is not None:
            self.cap.release()
        
        # Open the new video file
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            self.video_label.setText(f"Error opening video: {video_path}")
            return False
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default if unable to determine
        
        # Reset to first frame
        self.current_frame = 0
        self.set_playback_speed("1x")
        self.show_frame(0)
        
        # Update UI to show video is loaded
        self.update_info_label()
        
        return True
    
    def reset(self):
        """Reset player state when no video is loaded"""
        
        # Clear the video display
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_frame_display(blank_image)
        
        # Reset playback state
        self.current_frame = 0
        self.total_frames = 0
        self.playing = False
        self.playback_speed = 1.0
        
        # Stop the playback timer
        self.timer.stop()
        
        # Update UI
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.set_playback_speed("1x")
        self.info_label.setText("No video loaded")
    
    def update_frame_display(self, frame):
        """Update the video display with the given frame"""
        
        # Convert OpenCV's BGR format to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create a scaled pixmap to fit the label
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the pixmap to the label
        self.video_label.setPixmap(pixmap)
    
    def show_frame(self, frame_idx):
        """Show a specific frame from the video"""
        
        if self.cap is None or not self.cap.isOpened():
            return False
        
        # Ensure frame index is within bounds
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = self.cap.read()
        
        if ret:
            # Update the display
            self.update_frame_display(frame)
            
            # Update current frame
            self.current_frame = frame_idx
            self.update_info_label()
            
            # Emit signal that frame has changed
            self.frame_changed.emit(frame_idx)
            
            return True
        else:
            print(f"Error reading frame {frame_idx}")
            return False
    
    def next_frame(self):
        """Display the next frame during playback"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Check if we've reached the end
        if self.current_frame >= self.total_frames - 1:
            self.toggle_play()  # Stop playback
            return
        
        # Show the next frame
        self.show_frame(self.current_frame + 1)

    def get_timer_interval_ms(self):
        """Return timer interval for the currently selected playback speed."""

        effective_fps = max(self.fps * self.playback_speed, 1.0)
        return max(1, int(1000 / effective_fps))

    def set_playback_speed(self, speed_text):
        """Update playback speed from dropdown selection."""

        if speed_text in self.speed_actions:
            self.speed_actions[speed_text].setChecked(True)
            self.speed_button.setText(speed_text)

        try:
            self.playback_speed = float(speed_text.rstrip('x'))
        except ValueError:
            self.playback_speed = 1.0

        if self.playing:
            self.timer.start(self.get_timer_interval_ms())
    
    def toggle_play(self):
        """Toggle between play and pause"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        self.playing = not self.playing
        
        if self.playing:
            # Start playback
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start(self.get_timer_interval_ms())
        else:
            # Pause playback
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()

    def restart_to_start(self):
        """Return to frame 0 and stay paused until user presses play."""

        if self.cap is None or not self.cap.isOpened():
            return

        # Ensure restart does not auto-play.
        self.playing = False
        self.timer.stop()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        # Seek to the first frame; this emits frame_changed and syncs plots.
        self.show_frame(0)
    
    def seek_percent(self, percent):
        """Seek to position specified as percentage (0-100)"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Calculate frame from percentage
        frame = int((percent / 100) * self.total_frames)
        
        # Show the frame
        self.show_frame(frame)
    
    def seek_relative(self, frames):
        """Seek a relative number of frames from current position"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Calculate new frame position
        new_frame = self.current_frame + frames
        
        # Show the frame
        self.show_frame(new_frame)
    
    def update_info_label(self):
        """Update the information label with current frame details"""
        
        self.info_label.setText(f"Frame: {self.current_frame} / {self.total_frames}")
    
    def resizeEvent(self, event):
        """Handle resize events to scale the video display"""
        
        super().resizeEvent(event)
        
        # If video is loaded, update the displayed frame to match new size
        if self.cap is not None and self.cap.isOpened():
            # Save current position
            current = self.current_frame
            
            # Redisplay current frame at new size
            self.show_frame(current)
