# Player Re-Identification System for Sports Analytics

A Python-based system for tracking and re-identifying players in sports videos using YOLO object detection and simple tracking algorithms.

## Features

- **Player Detection**: Uses YOLO models to detect players in video frames
- **Player Tracking**: Implements simple nearest-neighbor tracking to maintain player identities across frames
- **Trajectory Visualization**: Draws player movement paths and bounding boxes
- **Multi-Player Support**: Tracks multiple players simultaneously with unique IDs
- **Output Generation**: Creates annotated video output and detailed logging

## Requirements

### System Requirements
- macOS (optimized for macOS setup)
- Python 3.7+
- Sufficient storage space for video processing

### Python Dependencies
```bash
pip install opencv-python
pip install ultralytics
pip install numpy
```

## Installation

1. **Clone or download the script** to your local machine

2. **Install required packages**:
   ```bash
   pip install opencv-python ultralytics numpy
   ```

3. **Create the directory structure**:
   ```bash
   mkdir -p ~/player_tracking/models
   mkdir -p ~/player_tracking/videos
   mkdir -p ~/player_tracking/output
   ```

## Setup

### 1. Prepare Your YOLO Model
- Place your trained YOLO model file (`.pt` format) in `~/player_tracking/models/`
- The model should be trained to detect players (class 0 should be 'player')

### 2. Prepare Your Video
- Place your input video file in `~/player_tracking/videos/`
- Supported formats: MP4, AVI, MOV, etc.

### 3. Update File Paths
Edit the script and update these variables with your actual filenames:

```python
# Update these paths in the main() function
MODEL_PATH = str(Path.home() / "player_tracking" / "models" / "your_model_file.pt")
VIDEO_PATH = str(Path.home() / "player_tracking" / "videos" / "your_video_file.mp4")
```

## Usage

### Basic Usage
```bash
python3 player_tracking.py
```

### Expected Output
The script will create:
- **Annotated video**: `~/player_tracking/output/tracked_output.mp4`
- **Tracking log**: `~/player_tracking/output/tracking_log.txt`

## How It Works

### 1. Player Detection
- Uses YOLO model to detect players in each frame
- Filters detections by confidence threshold (default: 0.5)
- Extracts bounding boxes and center points

### 2. Player Tracking
- **First Frame**: Creates new tracks for all detected players
- **Subsequent Frames**: Uses nearest-neighbor algorithm to match detections with existing tracks
- **Distance Threshold**: Maximum 100 pixels for track association
- **Track Management**: Removes tracks not seen for 30+ frames

### 3. Visualization
- **Bounding Boxes**: Different colors for each player
- **Player IDs**: Numbered labels above each player
- **Trajectories**: Lines showing player movement paths
- **Frame Info**: Current frame number and active player count

## Configuration Options

### Detection Parameters
```python
# In the YOLO detection call
results = model(frame, conf=0.5, verbose=False)
```
- `conf`: Confidence threshold (0.0-1.0)
- `verbose`: Enable/disable detection logging

### Tracking Parameters
```python
# Maximum distance for track association
if distance < min_distance and distance < 100:  # Adjust this value
```

```python
# Frames before removing inactive tracks
if frame_count - track['last_seen'] > 30:  # Adjust this value
```

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Check that your `.pt` file exists in the models directory
   - Verify the filename matches exactly in the script

2. **"Video file not found"**
   - Ensure your video file is in the videos directory
   - Check file permissions and format compatibility

3. **Poor tracking performance**
   - Adjust confidence threshold for YOLO detection
   - Modify distance threshold for track association
   - Consider using more sophisticated tracking algorithms

4. **Memory issues with large videos**
   - Process videos in smaller chunks
   - Reduce video resolution if possible
   - Monitor system memory usage

### Performance Tips

- **Faster Processing**: Lower the confidence threshold cautiously
- **Better Accuracy**: Use a well-trained YOLO model specific to your sport
- **Reduced Memory Usage**: Process shorter video segments
- **Quality vs Speed**: Adjust frame processing intervals for real-time applications

## Output Files

### Video Output
- **Format**: MP4 with H.264 encoding
- **Features**: Bounding boxes, player IDs, trajectories, frame info
- **Location**: `~/player_tracking/output/tracked_output.mp4`

### Log File
Contains detailed tracking statistics:
- Total frames processed
- Number of unique players detected
- Per-player tracking information (first seen, last seen, total positions)

## Customization

### Adding New Features
- **Player Statistics**: Extend the tracking data structure
- **Advanced Tracking**: Implement Kalman filters or Deep SORT
- **Real-time Processing**: Add camera input support
- **Export Options**: Add CSV export for tracking data

### Modifying Visualization
- **Colors**: Edit the `colors` list for different player colors
- **Display Elements**: Modify the visualization section to add/remove elements
- **Font Styles**: Adjust OpenCV text parameters

