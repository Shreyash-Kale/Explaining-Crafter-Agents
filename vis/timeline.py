# timeline_controller.py - Unified timeline control for synchronization

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt, pyqtSignal



class TimelineController(QWidget):
    """Controls for synchronized timeline between video and visualization"""
    
    # Signal emitted when position changes
    position_changed = pyqtSignal(float)  # 0-100 percentage
    interaction_event = pyqtSignal(str, object)
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.init_ui()
        
        # Initialize variables
        self.total_steps = 0
        self.total_frames = 0
        self.frame_step_ratio = 1.0
        self._is_dragging = False
        self._drag_start_value = 0
    
    def init_ui(self):
        """Initialize the UI components"""
        
        # Create horizontal layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        
        # Create position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)  # Use 0-1000 for more precision
        self.position_slider.setValue(0)
        self.position_slider.valueChanged.connect(self.on_slider_value_changed)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        layout.addWidget(self.position_slider)

        # Step counter – primary timeline unit.
        self.step_frame_label = QLabel("Step: 0 / 0")
        layout.addWidget(self.step_frame_label)

        self.position_label = QLabel("0.0%")
        self.position_label.setMinimumWidth(52)
        self.position_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.position_label)
        
        # Set fixed height
        self.setFixedHeight(40)
    
    def setup(self, total_steps, total_frames):
        """Initialize with total steps and frames"""
        
        self.total_steps = max(1, total_steps)
        self.total_frames = max(1, total_frames)
        
        # Calculate ratio for mapping between frames and steps
        self.frame_step_ratio = self.total_frames / self.total_steps
        
        # Reset position
        self.set_position(0)
        self.update_mapping_status(0, 0)
    
    def set_position(self, position, from_video=False, from_viz=False):
        """Set the position (0-100%)"""
        
        # Update slider (scaled to 0-1000 internally)
        slider_value = int(position * 10)
        
        # Only update if value has changed to avoid feedback loops
        if self.position_slider.value() != slider_value:
            # Temporarily block signals to avoid feedback
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(slider_value)
            self.position_slider.blockSignals(False)
        
        # Update label
        self.position_label.setText(f"{position:.1f}%")
        
        # Emit signal unless this change came from a component that already knows
        if not from_video and not from_viz:
            self.position_changed.emit(position)
    
    def on_slider_value_changed(self, value):
        """Handle slider value changes"""
        
        # Convert 0-1000 to 0-100%
        position = value / 10.0
        
        # Update label
        self.position_label.setText(f"{position:.1f}%")

        if self._is_dragging:
            self.interaction_event.emit(
                "slider_scrub",
                {
                    "target_id": "timeline_slider",
                    "interaction_type": "drag",
                    "slider_value": int(value),
                    "position_percent": round(position, 2),
                },
            )
        
        # Emit signal
        self.position_changed.emit(position)

    def on_slider_pressed(self):
        self._is_dragging = True
        self._drag_start_value = int(self.position_slider.value())
        self.interaction_event.emit(
            "slider_scrub_start",
            {
                "target_id": "timeline_slider",
                "interaction_type": "press",
                "slider_value": self._drag_start_value,
            },
        )

    def on_slider_released(self):
        value = int(self.position_slider.value())
        position = value / 10.0
        delta = value - int(self._drag_start_value)
        abs_delta = abs(delta)

        if abs_delta <= 5:
            scrub_intent = "micro_adjust"
        elif value >= 995 and delta > 0:
            scrub_intent = "jump_to_end"
        elif value <= 5 and delta < 0:
            scrub_intent = "jump_to_start"
        elif delta > 0:
            scrub_intent = "seek_forward"
        else:
            scrub_intent = "seek_backward"

        self._is_dragging = False
        self.interaction_event.emit(
            "slider_scrub_end",
            {
                "target_id": "timeline_slider",
                "interaction_type": "release",
                "slider_value": value,
                "position_percent": round(position, 2),
                "delta_slider_value": int(delta),
                "delta_percent": round(delta / 10.0, 2),
                "scrub_intent": scrub_intent,
            },
        )
        self.interaction_event.emit(
            "slider_jump",
            {
                "target_id": "timeline_slider",
                "interaction_type": "jump",
                "slider_value": value,
                "position_percent": round(position, 2),
                "delta_slider_value": int(delta),
                "delta_percent": round(delta / 10.0, 2),
                "scrub_intent": scrub_intent,
            },
        )
    
    def frame_to_step(self, frame, offset=0):
        """Convert a frame number to a step index with offset"""
        
        if self.total_frames <= 0 or self.total_steps <= 0:
            return 0
        
        # Calculate step based on frame with offset
        step = int((frame + offset) / self.frame_step_ratio)
        
        # Add debugging to verify the mapping
        # print(f"Mapping frame {frame} to step {step} (with offset {offset})")
        
        # Ensure step is within valid range
        return max(0, min(step, self.total_steps - 1))

    def update_mapping_status(self, frame, step, offset=0):
        """Update step counter label for the current cursor position."""

        if self.total_steps <= 0:
            self.step_frame_label.setText("Step: 0 / 0")
            return

        safe_step = max(0, min(int(step), self.total_steps - 1))
        self.step_frame_label.setText(f"Step: {safe_step} / {self.total_steps - 1}")




    
    def step_to_frame(self, step):
        """Convert a step index to a frame number"""
        
        if self.total_frames <= 0 or self.total_steps <= 0:
            return 0
        
        # Calculate frame based on step
        frame = int(step * self.frame_step_ratio)
        
        # Ensure within bounds
        return max(0, min(frame, self.total_frames - 1))

