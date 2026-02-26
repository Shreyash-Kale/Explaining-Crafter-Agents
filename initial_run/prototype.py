import cv2
import tkinter as tk
from tkinter import ttk
import webbrowser
import tempfile
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import time

from initial_run.visualization import (
    plot_cumulative_reward_interactive_enhanced,
    plot_reward_decomposition_interactive_enhanced,
    create_decision_analysis_dashboard
)
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json

# Create a simple HTTP server for real-time updates
class VisualizationServer(BaseHTTPRequestHandler):
    # Shared variable to store current visualization state
    current_state = {"step": 0, "action": "", "html": ""}
    
    def do_GET(self):
        if self.path == "/":
            # Serve the main visualization page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            
            # Create HTML with auto-refresh script
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Agent Analysis Visualization</title>
                <script>
                    // Check for updates every 100ms
                    setInterval(function() {{
                        fetch('/state')
                            .then(response => response.json())
                            .then(data => {{
                                if (data.step != currentStep) {{
                                    document.getElementById('visualization').innerHTML = data.html;
                                    currentStep = data.step;
                                }}
                            }});
                    }}, 100);
                    let currentStep = {VisualizationServer.current_state["step"]};
                </script>
            </head>
            <body>
                <div id="visualization">
                    {VisualizationServer.current_state["html"]}
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == "/state":
            # Return current state as JSON
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(VisualizationServer.current_state).encode())

# Function to update visualization state
def update_visualization_state(step, action, html_content):
    VisualizationServer.current_state = {
        "step": step,
        "action": action,
        "html": html_content
    }

class VideoController:
    def __init__(self, video_path, label, on_frame_change=None, time_steps=None):
        self.video_path = video_path
        self.label = label
        self.on_frame_change = on_frame_change
        self.time_steps = time_steps  # Now properly defined as a parameter  
        self.cap = cv2.VideoCapture(video_path)
        self.speed_multiplier = 1.0  # Add this line to initialize speed

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:  # Safeguard against invalid FPS
            self.fps = 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize state
        self.current_frame = 0
        self.playing = False
        self.update_id = None  # Track the update callback
        self.speed_multiplier = 1.0  # Default playback speed

        # Show first frame immediately
        self.show_current_frame()

        # Add keyboard bindings
        label.focus_set()  # Ensure label can receive keyboard events
        label.bind("<space>", lambda e: self.play_pause())
        label.bind("<Right>", lambda e: self.seek_by_steps(5))  # 5 steps forward
        label.bind("<Left>", lambda e: self.seek_by_steps(-5))  # 5 steps backward

    def update(self):
        """Regular update for playback with speed control"""
        if self.playing and self.current_frame < self.total_frames - 1:
            # Measure start time for this frame
            start_time = time.time()
            
            # Advance the frame and show it
            self.current_frame += 1
            self.show_current_frame()
            
            # Update slider position as video plays
            if hasattr(self, 'position_callback'):
                position = (self.current_frame / self.total_frames) * 100
                self.position_callback(position)
            
            # Calculate how long processing took
            elapsed = time.time() - start_time
            
            # Adjust wait time based on speed multiplier (key change here!)
            target_frame_time = (1.0 / self.fps) / self.speed_multiplier
            wait_time = max(1, int((target_frame_time - elapsed) * 1000))
            
            # Cancel any existing update and schedule a new one
            if self.update_id:
                self.label.after_cancel(self.update_id)
            self.update_id = self.label.after(wait_time, self.update)
        elif self.playing:
            # End of video reached, stop playing
            self.playing = False

    def set_speed(self, speed_multiplier):
        """Set playback speed"""
        self.speed_multiplier = float(speed_multiplier)
        return self.speed_multiplier

    def play_pause(self):
        """Toggle playback state with debouncing"""
        # Debounce to prevent multiple rapid toggles
        current_time = time.time()
        if hasattr(self, 'last_toggle_time') and current_time - self.last_toggle_time < 0.3:
            return self.playing
        self.last_toggle_time = current_time
        
        # Toggle playing state
        self.playing = not self.playing
        
        # If we're stopping, cancel any pending updates
        if not self.playing and self.update_id:
            self.label.after_cancel(self.update_id)
            self.update_id = None
        
        # If we're starting, begin the update cycle
        if self.playing:
            self.update()
        
        return self.playing

    def show_current_frame(self):
        """Display the current frame without advancing"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((700, 550), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.config(image=imgtk)
            self.label.image = imgtk
            
            # Notify about frame change
            if self.on_frame_change:
                self.on_frame_change(self.current_frame)
        
    def seek(self, position):
        """Seek to position (0-100)"""
        # If playing, pause first to avoid conflicts
        was_playing = self.playing
        self.playing = False
        
        # Calculate and set new position
        frame_idx = int((float(position) / 100) * self.total_frames)
        self.current_frame = min(max(0, frame_idx), self.total_frames - 1)
        
        # Show the new frame
        self.show_current_frame()
        
        # Resume if we were playing
        self.playing = was_playing
        if self.playing:
            self.update()
        
        return self.current_frame
    
    def seek_relative(self, seconds):
        """Jump forward or backward by specified seconds"""
        fps = self.fps if self.fps > 0 else 30
        frames_to_jump = int(seconds * fps)
        new_frame = min(max(0, self.current_frame + frames_to_jump), self.total_frames - 1)
        return self.seek(new_frame / self.total_frames * 100)
    
    def seek_by_steps(self, step_count):
        """Jump forward or backward by specified number of steps instead of seconds"""
        # Skip if time_steps is not provided
        if self.time_steps is None:
            return self.seek_relative(step_count)  # Fall back to seconds
            
        # Get total frames and steps from the video and data
        total_frames = self.total_frames
        total_steps = len(self.time_steps)
        
        # Calculate the current step based on current frame
        current_step_idx = min(int((self.current_frame / total_frames) * total_steps), total_steps - 1)
        
        # Calculate new step index with bounds checking
        new_step_idx = min(max(0, current_step_idx + step_count), total_steps - 1)
        
        # Convert step index back to frame number
        frame_ratio = new_step_idx / total_steps
        new_frame = int(frame_ratio * total_frames)
        
        # Use existing seek method to set to the new frame
        return self.seek(new_frame / total_frames * 100)
    
def create_synchronized_dashboard(event_df, time_steps, action_log, reward_log, reward_components, current_step):
    """
    Creates a synchronized dashboard using the visualization functions from visualization.py
    with added synchronization markers at the current position.
    """
    # Use the create_decision_analysis_dashboard function directly from visualization.py
    # This will internally call the other two visualization functions
    dashboard = create_decision_analysis_dashboard(
        reward_log, 
        time_steps, 
        action_log, 
        reward_components, 
        event_df
    )
    
    # Add vertical line at current position for sync indication
    current_idx = min(current_step, len(time_steps) - 1)
    if current_idx >= 0:
        dashboard.add_vline(
            x=time_steps[current_idx],
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=1, col=1
        )
        dashboard.add_vline(
            x=time_steps[current_idx],
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )
    
    # Add annotation about current time with better styling
    if current_idx >= 0:
        current_action = action_log[current_idx] if current_idx < len(action_log) else "unknown"
        
        dashboard.add_annotation(
            text=f"<b>Current Step:</b> {time_steps[current_idx]}<br><b>Action:</b> {current_action}",
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(245,245,245,0.9)",
            bordercolor="#888888",
            borderwidth=1,
            borderpad=4
        )
    
    return dashboard

def main():
    # Find the latest recording in logs folder
    logs_dir = './logs'
    video_files = [f for f in os.listdir(logs_dir) if f.endswith('.mp4')]
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.csv')]
    
    if not video_files or not log_files:
        print("No video or log files found in ./logs directory")
        return
    
    # Sort by creation time to get the most recent
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
    
    latest_video = os.path.join(logs_dir, video_files[0])
    latest_log = os.path.join(logs_dir, log_files[0])
    
    print(f"Using video: {latest_video}")
    print(f"Using log: {latest_log}")
    
    # Load event data from CSV
    event_df = pd.read_csv(latest_log)
    reward_log = event_df['reward'].tolist()
    time_steps = event_df['time_step'].tolist()
    action_log = event_df['action'].tolist()
    
    # Extract reward components
    reward_components = {}
    for col in event_df.columns:
        if col not in ['time_step', 'action', 'reward', 'cumulative_reward']:
            reward_components[col] = event_df[col].tolist()
    
    # Create the main window
    root = tk.Tk()
    root.title("Crafter Analysis Tool")
    root.geometry("1200x700")
    
    video_frame = ttk.Frame(root)
    video_frame.pack(fill=tk.BOTH, expand=True)
    
    # Video display using OpenCV
    video_label = tk.Label(video_frame)
    video_label.pack(fill=tk.BOTH, expand=True)
    
    # Create temporary HTML file for visualization
    temp_html = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    temp_html.close()
    
    # Status variables
    status_var = tk.StringVar(value="Ready")
    last_update_time = [time.time()]  # Use list for mutable reference
    
    # Function to update visualization with throttling
    def update_visualization(frame):
        current_time = time.time()
        # Throttle updates to max 5 per second to avoid browser overload
        if current_time - last_update_time[0] < 0.2:
            return
            
        last_update_time[0] = current_time
        
        # Convert frame to step index
        total_frames = controller.total_frames
        total_steps = len(time_steps)
        
        # Map frame number to step index
        step_index = min(int((frame / total_frames) * total_steps), total_steps - 1)
        
        status_var.set(f"Updating visualization to step {step_index}...")
        
        # Create dashboard with highlighted current position
        fig = create_synchronized_dashboard(
            event_df, 
            time_steps, 
            action_log, 
            reward_log, 
            reward_components,
            step_index
        )
        
        # Save to HTML and trigger browser refresh
        html_content = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
        with open(temp_html.name, 'w') as f:
            f.write(html_content)
            
        # Use JavaScript to force browser refresh
        js_refresh = f"""
        <script>
        // Add timestamp to force refresh
        window.location.href = window.location.href.split("?")[0] + "?t={int(time.time())}";
        </script>
        """
        with open(temp_html.name + ".refresh", 'w') as f:
            f.write(js_refresh)
            
        status_var.set(f"At step {step_index} / {total_steps-1}")
    
    # Initialize video controller with update callback
    controller = VideoController(latest_video, video_label, update_visualization, time_steps)
    
    # Add controls frame
    controls_frame = ttk.Frame(root)
    controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Add play/pause button
    play_pause_var = tk.StringVar(value="Play")
    
    def toggle_play():
        is_playing = controller.play_pause()
        play_pause_var.set("Pause" if is_playing else "Play")
    
    play_button = ttk.Button(controls_frame, textvariable=play_pause_var, command=toggle_play)
    play_button.pack(side=tk.LEFT, padx=5)
    
    # Add seek slider
    position_var = tk.DoubleVar(value=0)
    
    def on_slider_change(value):
        controller.seek(float(value))
    
    seek_slider = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                          variable=position_var, command=on_slider_change)
    seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Add speed control dropdown
    speed_frame = ttk.Frame(controls_frame)
    speed_frame.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
    
    speed_var = tk.StringVar(value="1.0x")
    speed_options = ["0.5x", "1.0x", "1.5x", "2.0x", "3.0x", "5.0x"]
    
    def change_speed(event=None):
        # Extract numeric value from string (remove 'x' suffix)
        speed = float(speed_var.get().replace('x', ''))
        controller.set_speed(speed)
        status_var.set(f"Playback speed set to {speed}x")
    
    speed_dropdown = ttk.Combobox(speed_frame, textvariable=speed_var, 
                                values=speed_options, width=5, state="readonly")
    speed_dropdown.pack(side=tk.LEFT)
    speed_dropdown.bind("<<ComboboxSelected>>", change_speed)
    
    # Add status label
    status_label = ttk.Label(controls_frame, textvariable=status_var)
    status_label.pack(side=tk.RIGHT, padx=10)
    
    # Initialize visualization and open browser
    update_visualization(0)
    webbrowser.open('file://' + temp_html.name)
    
    # Start the application
    root.mainloop()
    
    # Clean up
    try:
        os.unlink(temp_html.name)
        os.unlink(temp_html.name + ".refresh")
    except:
        pass

if __name__ == "__main__":
    main()
