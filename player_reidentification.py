#!/usr/bin/env python3
"""
Player Re-Identification System for Sports Analytics
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

def main():
    """
    Main function - UPDATE THESE PATHS FOR YOUR SYSTEM
    """
    # =================================================================
    # IMPORTANT: UPDATE THESE PATHS WITH YOUR ACTUAL FILE LOCATIONS
    # =================================================================
    
    # Path to your YOLO model (replace with actual filename)
    MODEL_PATH = str(Path.home() / "player_tracking" / "models" / "best.pt")
    
    # Path to your input video
    VIDEO_PATH = str(Path.home() / "player_tracking" / "videos" / "15sec_input_720p.mp4")
    
    # Output paths
    OUTPUT_VIDEO = str(Path.home() / "player_tracking" / "output" / "tracked_output.mp4")
    OUTPUT_LOG = str(Path.home() / "player_tracking" / "output" / "tracking_log.txt")
    
    # =================================================================
    # CHECK IF FILES EXIST
    # =================================================================
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        print("Please check the filename and path.")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video file not found at: {VIDEO_PATH}")
        print("Please check the filename and path.")
        return
    
    print(f"✅ Model found: {MODEL_PATH}")
    print(f"✅ Video found: {VIDEO_PATH}")
    
    # =================================================================
    # INITIALIZE YOLO MODEL
    # =================================================================
    
    print("🔄 Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return
    
    # =================================================================
    # OPEN VIDEO
    # =================================================================
    
    print("🔄 Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video opened: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # =================================================================
    # SETUP VIDEO WRITER
    # =================================================================
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # =================================================================
    # SIMPLE TRACKING VARIABLES
    # =================================================================
    
    player_tracks = {}  # {track_id: {'positions': [], 'last_seen': frame_num}}
    next_track_id = 1
    frame_count = 0
    
    # Colors for visualization
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    print("🔄 Processing video...")
    
    # =================================================================
    # MAIN PROCESSING LOOP
    # =================================================================
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:  # Every second at 30fps
            print(f"📊 Processing frame {frame_count}/{total_frames}")
        
        # Run YOLO detection
        results = model(frame, conf=0.5, verbose=False)
        
        current_detections = []
        
        # Extract player detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Assuming class 0 is 'player' - adjust based on your model
                    if int(box.cls) == 0:
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        current_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': conf
                        })
        
        # =================================================================
        # SIMPLE TRACKING LOGIC
        # =================================================================
        
        # For first frame or when no tracks exist, create new tracks
        if not player_tracks or frame_count == 1:
            for i, detection in enumerate(current_detections):
                player_tracks[next_track_id] = {
                    'positions': [detection['center']],
                    'last_seen': frame_count,
                    'bbox': detection['bbox']
                }
                next_track_id += 1
        else:
            # Simple nearest neighbor tracking
            unmatched_detections = current_detections.copy()
            
            for track_id in list(player_tracks.keys()):
                track = player_tracks[track_id]
                
                if not unmatched_detections:
                    break
                
                # Find closest detection
                min_distance = float('inf')
                best_match = None
                best_idx = -1
                
                last_pos = track['positions'][-1]
                
                for idx, detection in enumerate(unmatched_detections):
                    distance = np.sqrt(
                        (detection['center'][0] - last_pos[0])**2 + 
                        (detection['center'][1] - last_pos[1])**2
                    )
                    
                    if distance < min_distance and distance < 100:  # Max 100 pixels
                        min_distance = distance
                        best_match = detection
                        best_idx = idx
                
                # Update track if match found
                if best_match:
                    track['positions'].append(best_match['center'])
                    track['last_seen'] = frame_count
                    track['bbox'] = best_match['bbox']
                    unmatched_detections.pop(best_idx)
            
            # Create new tracks for unmatched detections
            for detection in unmatched_detections:
                player_tracks[next_track_id] = {
                    'positions': [detection['center']],
                    'last_seen': frame_count,
                    'bbox': detection['bbox']
                }
                next_track_id += 1
        
        # Remove old tracks (not seen for 30 frames)
        tracks_to_remove = []
        for track_id, track in player_tracks.items():
            if frame_count - track['last_seen'] > 30:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del player_tracks[track_id]
        
        # =================================================================
        # VISUALIZATION
        # =================================================================
        
        vis_frame = frame.copy()
        
        # Draw tracks
        for track_id, track in player_tracks.items():
            if frame_count - track['last_seen'] <= 1:  # Only active tracks
                color = colors[(track_id - 1) % len(colors)]
                bbox = track['bbox']
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw player ID
                label = f"Player {track_id}"
                cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw trajectory
                positions = track['positions']
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(vis_frame, positions[i-1], positions[i], color, 2)
        
        # Add frame info
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Players: {len([t for t in player_tracks.values() if frame_count - t['last_seen'] <= 1])}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(vis_frame)
    
    # =================================================================
    # CLEANUP AND RESULTS
    # =================================================================
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Write summary log
    with open(OUTPUT_LOG, 'w') as f:
        f.write(f"Player Re-Identification Results\n")
        f.write(f"================================\n")
        f.write(f"Total frames processed: {frame_count}\n")
        f.write(f"Total unique players detected: {next_track_id - 1}\n")
        f.write(f"Final active tracks: {len(player_tracks)}\n\n")
        
        for track_id, track in player_tracks.items():
            f.write(f"Player {track_id}:\n")
            f.write(f"  - First seen: Frame {track['last_seen'] - len(track['positions']) + 1}\n")
            f.write(f"  - Last seen: Frame {track['last_seen']}\n")
            f.write(f"  - Total positions tracked: {len(track['positions'])}\n\n")
    
    print("🎉 Processing complete!")
    print(f"📹 Output video: {OUTPUT_VIDEO}")
    print(f"📊 Log file: {OUTPUT_LOG}")
    print(f"👥 Total players tracked: {next_track_id - 1}")

if __name__ == "__main__":
    print("🚀 Starting Player Re-Identification System")
    print("=" * 50)
    main()