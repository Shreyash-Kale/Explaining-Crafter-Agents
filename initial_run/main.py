from crafter.Code.dreamer_files.environment import create_environment, run_episode, random_policy
from initial_run.visualization import (plot_cumulative_reward_interactive_enhanced, 
                plot_reward_decomposition_interactive_enhanced,
                create_decision_analysis_dashboard)
import cv2
import numpy as np
import threading
import time
import os

def main():
    env = create_environment()
    
    # Run an episode and get the results
    print("Running episode with random policy...")
    cumulative_reward, reward_log, combined_reward_log, time_steps, final_obs, action_log, event_df = run_episode(env, random_policy)
    
    print("Total Reward:", cumulative_reward)
    print(f"Episode length: {len(time_steps)} steps")
    
    # Create the visualization dashboard
    print("Creating interactive visualization dashboard...")
    dashboard = create_decision_analysis_dashboard(
        reward_log, 
        time_steps, 
        action_log, 
        {k: [step.get(k, 0) for step in combined_reward_log] for k in combined_reward_log[0].keys()},
        event_df
    )
    
    # Start video playback if available
    video_files = [f for f in os.listdir("logs") if f.endswith(".mp4")]
    if video_files:
        try:
            latest_video = os.path.join("logs", video_files[-1])
            print(f"Found gameplay video: {latest_video}")
            video_thread = threading.Thread(
                target=play_synchronized_video,
                args=(latest_video, len(time_steps))
            )
            video_thread.daemon = True
            video_thread.start()
        except Exception as e:
            print(f"Could not start video playback: {e}")
    
    # Display the dashboard
    dashboard.show()
    
    # Save the dashboard to HTML for sharing
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dashboard.write_html(f"logs/decision_analysis_{timestamp}.html")
    print(f"Analysis dashboard saved to logs/decision_analysis_{timestamp}.html")


def play_synchronized_video(video_path, total_steps):
    """
    Plays the recorded gameplay video with controls synchronized to visualization.
    """
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return
        
        # Give the OS a moment to initialize resources
        time.sleep(0.5)
        
        try:
            cv2.namedWindow('Gameplay', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gameplay', 640, 480)
        except Exception as e:
            print(f"Error creating window: {e}")
            print("This may be due to a missing display or OpenCV installation issue.")
            print("Continuing without video playback...")
            cap.release()
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS if unable to determine
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            print("Warning: Could not determine frame count")
            frame_count = total_steps * 10  # Rough estimate
        
        # Calculate frame display timing based on step count
        frames_per_step = frame_count / total_steps if total_steps > 0 else 1
        
        playing = True
        current_frame = 0
        
        print("Video playback started. Press SPACE to pause/play, ESC to exit.")
        
        while True:
            if not cap.isOpened():
                break
                
            if playing:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Display frame number and corresponding step
                current_step = min(int(current_frame / frames_per_step), total_steps - 1)
                cv2.putText(
                    frame, f"Step: {current_step}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )
                
                cv2.imshow('Gameplay', frame)
                current_frame += 1
            
            # Handle keyboard input with reduced framerate for stability
            key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == 32:  # SPACE key
                playing = not playing
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Video playback error: {e}")
        print("Continuing without video playback...")


if __name__ == "__main__":
    main()
