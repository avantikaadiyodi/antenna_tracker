# Antenna Tracker Visualization

A Python program that visualizes a dish antenna tracking a moving target in 3D space.

## Features

- **3D Visualization**: Real-time animation of a parabolic dish antenna tracking a moving target
- **Telemetry Plots**: Live plots showing azimuth, elevation, and range over time
- **Video Export**: Save animations as MP4 videos for sharing and documentation

## Requirements

- Python 3.x
- numpy
- matplotlib
- ffmpeg (for video export)

## Installation

Install the required Python packages:
```bash
pip install numpy matplotlib
```

For video export functionality, you also need ffmpeg installed on your system:
- **Windows**: Download from https://ffmpeg.org/ or use `choco install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg` or `sudo yum install ffmpeg`
- **macOS**: `brew install ffmpeg`

## Usage

### Basic Usage (Display Only)

Run the visualization without saving:
```bash
python antenna_vis.py
```

This will open a window showing the 3D animation and telemetry plots.

### Save as Video

To save the animation as a video file:
```bash
python antenna_vis.py --save
```

The video will be saved to `./track_plots/YYYYMMDD/track_plot_HHMMSS.mp4`

### Custom Output Directory

Specify a custom output directory:
```bash
python antenna_vis.py --save --output-dir ./my_videos
```

### Command-Line Options

- `--save`: Save the animation as a video file (MP4 format)
- `--output-dir DIR`: Specify output directory for saved videos (default: `./track_plots`)
- `--help`: Show help message

## Project Structure

```
antenna_tracker/
├── antenna_vis.py          # Main visualization script
├── README.md               # This file
├── prompts.txt            # Useful prompts for the project
└── track_plots/           # Generated videos (created automatically)
    └── YYYYMMDD/
        └── track_plot_HHMMSS.mp4
```


## Output Structure

When saving videos, the program creates the following directory structure:
```
track_plots/
└── 20260117/              # Date (YYYYMMDD)
    ├── track_plot_123045.mp4
    ├── track_plot_145230.mp4
    └── ...
```

### Video Specifications

- **Format**: MP4 (H.264)
- **Frame Rate**: 20 fps
- **Bitrate**: 1800 kbps
- **Resolution**: 1400x800 pixels (14x8 inches at 100 DPI)
- **Duration**: ~10 seconds (200 frames)


## How It Works

1. **Trajectory Generation**: Creates a dummy 3D trajectory (figure-8 pattern)
2. **Antenna Pointing**: Calculates rotation matrices to point the dish at the target
3. **Visualization**: Animates the dish orientation and plots telemetry data
4. **Video Export**: Uses FFmpeg to encode the animation as MP4

## Customization

You can modify the following parameters in the code:

- **Trajectory**: Edit `generate_trajectory()` function to change the target path
- **Antenna Size**: Modify `diameter` and `depth` in `create_parabolic_dish()`
- **Animation Speed**: Change `interval` parameter in `FuncAnimation()`
- **Video Quality**: Adjust `fps` and `bitrate` in `save_animation()`
