# 3D Tracklet Visualization Tool

![Visualization Example](Screenshot%202025-08-01%20at%2012.55.58.png)

This project provides a Python script to generate and visualize synthetic 3D bounding box tracklets for objects (vehicles) moving along a multi-lane road. The visualization is interactive and rendered using Plotly, allowing you to explore detection and tracking results in a realistic traffic scenario.

## Features

- **Synthetic Scene Generation**: Simulates vehicles of various types (cars, trucks, buses, motorcycles) moving along lanes, with realistic sizes, speeds, and non-overlapping trajectories.
- **Detection and Tracking Simulation**: For each object, both "detector" and "tracker" bounding boxes are generated, with configurable noise, lag, and jitter.
- **Interactive 3D Visualization**: Visualizes all bounding boxes as wireframes, with options to show/hide detectors, trackers, or individual objects. Tracklet centerlines and lane markings are also displayed.
- **Self-contained HTML Output**: The visualization is exported as a standalone HTML file for easy sharing and inspection.

## Requirements

- Python 3.7+
- [NumPy](https://numpy.org/)
- [Plotly](https://plotly.com/python/)

Install dependencies with:
```bash
pip install numpy plotly
```

## Usage

Run the script directly:
```bash
python visualize_3d_tracklets.py
```

This will generate a file named `visualization.html` in the current directory. Open this file in your web browser to explore the 3D scene.

## Customization

You can modify the parameters in the `main()` function of `visualize_3d_tracklets.py` to control the scenario:

- `num_frames`: Number of time steps (length of each tracklet).
- `lane_count`: Number of lanes (increasing this spaces out vehicles more).
- `vehicle_counts`: Dictionary specifying the number of each vehicle type.
- `detector_noise_std`, `tracker_spatial_jitter_std`: Control the amount of noise/jitter in detections and tracker outputs.
- `tracker_lag_frames`: How many frames the tracker lags behind the true position.
- `random_seed`: For reproducibility.

Example for 5 spaced-out, short tracklets:
```python
boxes = generate_road_scene(
    num_frames=40,
    dt=0.2,
    lane_count=5,
    lane_width=3.6,
    vehicle_counts={"car": 3, "truck": 1, "bus": 1},
    detector_noise_std=0.05,
    tracker_spatial_jitter_std=0.2,
    tracker_lag_frames=2,
    random_seed=2025,
)
```

## Visualization Controls

- **Dropdown Menu**: Filter to show all objects, only detectors, only trackers, or a specific object.
- **Legend**: Click to toggle visibility of tracklet centerlines.
- **3D Navigation**: Rotate, zoom, and pan using your mouse.

## Code Structure

- `visualize_3d_tracklets.py`: Main script containing scene generation and visualization logic.
- `generate_road_scene`: Function to create a realistic multi-lane road scenario.
- `visualize_scene`: Function to build and export the interactive Plotly visualization.

## Extending

- Add new vehicle types by editing the `VEHICLE_SPECS` dictionary.
- Adjust lane geometry, noise models, or car-following logic for more complex scenarios.
- Integrate with real detection/tracking data by adapting the input format.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Developed by [Your Name or Organization].
