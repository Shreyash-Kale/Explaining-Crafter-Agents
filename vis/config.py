# ---- VisConfig.py ----
import os

# Get the absolute path to the project root
CRAFTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prefer the most recent archived run if present
ARCHIVE_DIR = os.path.join(CRAFTER_DIR, 'archive')

def _latest_archive_run_dir():
    if not os.path.isdir(ARCHIVE_DIR):
        return None
    run_dirs = [
        d for d in os.listdir(ARCHIVE_DIR)
        if os.path.isdir(os.path.join(ARCHIVE_DIR, d)) and d.startswith('run_')
    ]
    if not run_dirs:
        return None
    run_dirs.sort()
    return os.path.join(ARCHIVE_DIR, run_dirs[-1])


_latest_run = _latest_archive_run_dir()

# Use archive paths when available; fallback to live folders
if _latest_run:
    DEFAULT_LOG_DIR = os.path.join(_latest_run, 'logs')
    RESULTS_LOG_DIR = os.path.join(_latest_run, 'results')
else:
    DEFAULT_LOG_DIR = os.path.join(CRAFTER_DIR, 'logs')
    RESULTS_LOG_DIR = os.path.join(CRAFTER_DIR, 'results')

# Debug prints to help troubleshoot
print(f"Looking for logs in: {DEFAULT_LOG_DIR}")
print(f"Looking for results in: {RESULTS_LOG_DIR}")

# Visualization settings
VIZ_COLORS = {
    'cumulative': '#1f77b4',  # Blue for overall progress
    'reward': '#ff7f0e',      # Orange for immediate rewards
    'position': '#d62728',    # Red for position marker
    'significant': '#2ca02c', # Green for significant events
    # Health and resource colors matching game
    'health': '#d62728',      # Red
    'food': '#2ca02c',        # Green
    'drink': '#1f77b4',       # Blue
    'energy': '#ffbf00',      # Golden
    'wood': '#8c564b',        # Brown
    'stone': '#7f7f7f',       # Grey
    'diamond': '#17becf',     # Cyan
    'iron': '#e377c2'         # Pink
}

# Video player settings
DEFAULT_FPS = 30

# Create required directories if they don't exist (only for live runs)
if not _latest_run:
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_LOG_DIR, exist_ok=True)
