import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Dict, Tuple, Optional


Box = Dict[str, object]  # keys: id(int), frame(int), type(str), center(tuple), size(tuple)


# Canonical vehicle specifications (L, W, H) in meters and speed ranges in m/s
VEHICLE_SPECS = {
    "car": {"size": (4.5, 1.9, 1.6), "v_range": (10.0, 28.0)},     # 36–100 km/h
    "truck": {"size": (10.0, 2.5, 3.5), "v_range": (6.0, 20.0)},  # 22–72 km/h
    "bus": {"size": (12.0, 2.6, 3.2), "v_range": (8.0, 22.0)},    # 29–79 km/h
    "motorcycle": {"size": (2.2, 0.8, 1.2), "v_range": (12.0, 33.0)}, # 43–120 km/h
}


def generate_synthetic_scene(
    num_objects: int = 5,
    num_frames: int = 50,
    dt: float = 1.0,
    pos_range: Tuple[float, float] = (-20.0, 20.0),
    vel_range: Tuple[float, float] = (-1.5, 1.5),
    size_range: Tuple[float, float] = (1.5, 4.0),
    detector_noise_std: float = 0.05,
    tracker_spatial_jitter_std: float = 0.15,
    tracker_lag_frames: int = 1,
    random_seed: Optional[int] = 42,
) -> List[Box]:
    """
    Generate synthetic 3D scene data for detector and tracker boxes.
    - Objects move linearly with gaussian noise added to detector.
    - Tracker lags by 'tracker_lag_frames' and has extra jitter.
    - size is constant per object.
    """
    rng = np.random.default_rng(random_seed)

    # Initialize per-object properties
    starts = rng.uniform(pos_range[0], pos_range[1], size=(num_objects, 3))
    vels = rng.uniform(vel_range[0], vel_range[1], size=(num_objects, 3))
    sizes = rng.uniform(size_range[0], size_range[1], size=(num_objects, 3))

    boxes: List[Box] = []

    # Precompute "true" centers without noise
    true_centers = np.zeros((num_objects, num_frames, 3))
    for oid in range(num_objects):
        for t in range(num_frames):
            true_centers[oid, t] = starts[oid] + vels[oid] * (t * dt)

    # Detector: noisy around true positions
    for oid in range(num_objects):
        for t in range(num_frames):
            det_center = true_centers[oid, t] + rng.normal(0, detector_noise_std, size=3)
            boxes.append({
                "id": int(oid),
                "frame": int(t),
                "type": "detector",
                "center": tuple(det_center.tolist()),
                "size": tuple(sizes[oid].tolist()),
            })

    # Tracker: lag + jitter
    for oid in range(num_objects):
        for t in range(num_frames):
            ref_t = max(0, t - tracker_lag_frames)
            base = true_centers[oid, ref_t]
            trk_center = base + rng.normal(0, tracker_spatial_jitter_std, size=3)
            boxes.append({
                "id": int(oid),
                "frame": int(t),
                "type": "tracker",
                "center": tuple(trk_center.tolist()),
                "size": tuple(sizes[oid].tolist()),
            })

    return boxes


def _cuboid_wireframe(center: Tuple[float, float, float],
                      size: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return x,y,z arrays describing 12 edges of a 3D cuboid as a continuous
    sequence with None breaks for Plotly scatter3d line plotting.
    """
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0

    # 8 corners
    corners = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz + dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz - dz],
        [cx + dx, cy + dy, cz + dz],
    ])

    # define edges as pairs of corner indices
    edges = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7),
    ]

    xs, ys, zs = [], [], []
    for (i, j) in edges:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])

    return np.array(xs), np.array(ys), np.array(zs)


def _color_for_id(oid: int) -> str:
    # Distinct but deterministic color set; extendable
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    return palette[oid % len(palette)]


def generate_road_scene(
    num_frames: int = 80,
    dt: float = 0.2,
    lane_count: int = 4,
    lane_width: float = 3.6,
    vehicle_counts: Dict[str, int] = None,
    detector_noise_std: float = 0.05,
    tracker_spatial_jitter_std: float = 0.15,
    tracker_lag_frames: int = 1,
    random_seed: Optional[int] = 123,
) -> List[Box]:
    """Generate a realistic multi-lane road scenario with multiple vehicle types, avoiding intersections.

    X: along road (forward), Y: lateral (left), Z: up.
    Lanes centered around Y=0, spaced by lane_width.

    Non-intersection strategy:
      - Assign vehicles to lanes and place them with min headway gaps based on length + buffer.
      - Use a simple car-following model to maintain time headway and enforce no-overlap constraints at each step.
      - No lane changes to keep lateral separation; small lateral jitter within lane only.
    """
    rng = np.random.default_rng(random_seed)
    if vehicle_counts is None:
        vehicle_counts = {"car": 8, "truck": 3, "bus": 2, "motorcycle": 3}

    # Build list of lanes centered around 0
    lane_indices = np.arange(lane_count)
    offset = (lane_count - 1) / 2.0
    lane_centers = ((lane_indices - offset) * lane_width).tolist()

    # Create vehicles and assign to lanes in round-robin to distribute traffic
    vehicles = []
    oid = 0
    all_types = []
    for vtype, count in vehicle_counts.items():
        all_types.extend([vtype] * count)
    rng.shuffle(all_types)

    for i, vtype in enumerate(all_types):
        size = VEHICLE_SPECS[vtype]["size"]
        vmin, vmax = VEHICLE_SPECS[vtype]["v_range"]
        lane_y = lane_centers[i % lane_count]
        desired_speed = rng.uniform(vmin, vmax)
        # start x spaced to avoid overlap within lane
        # initialize near a base offset with extra random spacing
        vehicles.append({
            "id": oid,
            "type": vtype,
            "size": size,
            "lane_y": float(lane_y),
            "length": float(size[0]),
            "width": float(size[1]),
            "height": float(size[2]),
            "speed_des": float(desired_speed),
        })
        oid += 1

    # Group vehicles by lane
    lane_to_vehicles: Dict[float, List[Dict[str, object]]] = {y: [] for y in lane_centers}
    for v in vehicles:
        lane_to_vehicles[v["lane_y"]].append(v)

    # Sort vehicles within each lane by initial longitudinal position (back to front)
    # Place with headway: min_gap = 0.5*len_back + 0.5*len_front + buffer
    MIN_BUFFER = 3.0  # meters extra gap between bumpers
    START_X_BACK = -120.0
    START_X_SPREAD = 40.0

    for lane_y, lane_list in lane_to_vehicles.items():
        rng.shuffle(lane_list)
        # compute initial x positions with safe gaps
        x_positions = []
        for v in lane_list:
            length = v["length"]
            if not x_positions:
                x = START_X_BACK + rng.uniform(0, START_X_SPREAD)
            else:
                prev_idx = len(x_positions) - 1
                prev_len = lane_list[prev_idx]["length"]
                min_gap = 0.5 * prev_len + 0.5 * length + MIN_BUFFER
                x = x_positions[-1] + min_gap + rng.uniform(2.0, 6.0)
            x_positions.append(x)
        # attach state
        for v, x in zip(lane_list, x_positions):
            v["x"] = float(x)
            v["y"] = float(lane_y + rng.normal(0.0, 0.1))
            v["z"] = float(v["height"] / 2.0)
            v["vx"] = float(rng.uniform(max(0.0, v["speed_des"] - 2.0), v["speed_des"]))

    # Simulation parameters for following model
    TIME_HEADWAY = 1.2   # seconds desired
    MAX_ACCEL = 2.0      # m/s^2
    MAX_DECEL = -4.0     # m/s^2 (braking)
    MIN_V = 0.0

    boxes: List[Box] = []

    # Pre-build a flattened list of vehicles for iteration order, but maintain lane groups
    lanes_ordered = [lane_to_vehicles[y] for y in sorted(lane_to_vehicles.keys())]

    for t in range(num_frames):
        # Update dynamics per lane (front to back to propagate constraints)
        for lane in lanes_ordered:
            if not lane:
                continue
            # Sort by current x (front first)
            lane.sort(key=lambda v: v["x"])  # back to front
            lane_front_to_back = list(reversed(lane))

            leader = None
            for v in lane_front_to_back:
                length = v["length"]
                # Desired speed with small noise
                v_des = max(MIN_V, v["speed_des"] + rng.normal(0.0, 0.3))

                if leader is None:
                    # Free-road vehicle accelerates towards desired speed
                    dv = v_des - v["vx"]
                    ax = np.clip(dv / max(dt, 1e-3), MAX_DECEL, MAX_ACCEL)
                else:
                    # Car-following: keep time headway and space gap
                    gap = (leader["x"] - 0.5 * leader["length"]) - (v["x"] + 0.5 * length)
                    desired_gap = max(2.0, 0.5 * leader["length"] + 0.5 * length + MIN_BUFFER + v["vx"] * TIME_HEADWAY)
                    # control towards maintaining desired gap
                    gap_error = gap - desired_gap
                    # proportional control on gap error + speed match
                    dv_lead = leader["vx"] - v["vx"]
                    ax = 0.6 * gap_error / max(dt * max(v["vx"], 1.0), 1.0) + 0.4 * dv_lead / max(dt, 1e-3)
                    ax = float(np.clip(ax, MAX_DECEL, MAX_ACCEL))

                    # Emergency braking if overlap would occur next step
                    next_tail = v["x"] + v["vx"] * dt + 0.5 * length
                    next_head_leader = leader["x"] + leader["vx"] * dt - 0.5 * leader["length"]
                    if next_tail >= next_head_leader:
                        ax = min(ax, -6.0)

                # integrate
                v["vx"] = float(np.clip(v["vx"] + ax * dt, MIN_V, v_des + 5.0))
                v["x"] = float(v["x"] + v["vx"] * dt)
                # small within-lane jitter (does not cause intersection due to lane separation)
                v["y"] = float(v["lane_y"] + rng.normal(0.0, 0.08))
                leader = v

            # After integration pass, enforce no-overlap by pushing back tails if needed
            lane.sort(key=lambda v: v["x"])  # back to front
            for i in range(len(lane) - 1, 0, -1):  # front to back indices
                front = lane[i]
                back = lane[i - 1]
                min_gap = 0.5 * front["length"] + 0.5 * back["length"] + MIN_BUFFER
                tail_back = back["x"] + 0.5 * back["length"]
                head_front = front["x"] - 0.5 * front["length"]
                if tail_back > head_front - 1e-3:
                    # push back vehicle to maintain min gap and reduce its speed
                    back["x"] = head_front - min_gap + 0.5 * back["length"]
                    back["vx"] = min(back["vx"], front["vx"])  # avoid re-colliding next step

        # Emit detector & tracker boxes for this frame (no intersections guaranteed per lane)
        for lane in lanes_ordered:
            for v in lane:
                center_true = np.array([v["x"], v["y"], v["z"]], dtype=float)

                det_center = center_true + rng.normal(0, detector_noise_std, size=3)
                boxes.append({
                    "id": int(v["id"]),
                    "frame": int(t),
                    "type": "detector",
                    "center": (float(det_center[0]), float(det_center[1]), float(det_center[2])),
                    "size": (float(v["length"]), float(v["width"]), float(v["height"]))
                })

                ref_t = max(0, t - tracker_lag_frames)
                # approximate past state with constant-velocity rollback
                x_ref = v["x"] - v["vx"] * (t - ref_t) * dt
                y_ref = v["lane_y"]
                z_ref = v["z"]
                base = np.array([x_ref, y_ref, z_ref])
                trk_center = base + rng.normal(0, tracker_spatial_jitter_std, size=3)
                boxes.append({
                    "id": int(v["id"]),
                    "frame": int(t),
                    "type": "tracker",
                    "center": (float(trk_center[0]), float(trk_center[1]), float(trk_center[2])),
                    "size": (float(v["length"]), float(v["width"]), float(v["height"]))
                })

    return boxes


def visualize_scene(
    boxes: List[Box],
    html_path: str = "visualization.html",
    show_tracklet_lines: bool = True,
    draw_ground_lanes: bool = True,
):
    """
    Build an interactive Plotly 3D visualization with dropdowns:
    - Show All
    - Detectors Only
    - Trackers Only
    - Specific Object ID (both types for that ID)
    Exports to a self-contained HTML.
    """
    # Group by (id, type) and also collect centers per id for lines
    by_group: Dict[Tuple[int, str], List[Box]] = {}
    centers_by_id: Dict[int, List[Tuple[int, Tuple[float, float, float]]]] = {}

    for b in boxes:
        key = (int(b["id"]), str(b["type"]))
        by_group.setdefault(key, []).append(b)
        centers_by_id.setdefault(int(b["id"]), []).append((int(b["frame"]), b["center"]))

    # Sort centers by frame for lines
    for oid in centers_by_id:
        centers_by_id[oid] = sorted(centers_by_id[oid], key=lambda x: x[0])

    traces = []
    trace_meta = []  # for filtering logic: {kind: 'detector'|'tracker'|'line'|'lane', id: int}

    # Optional: draw ground plane lane lines for context
    if draw_ground_lanes:
        # Estimate bounds from data
        centers = np.array([b["center"] for b in boxes])
        x_min, x_max = float(np.min(centers[:, 0])) - 10.0, float(np.max(centers[:, 0])) + 10.0
        # Try to infer lane Ys by simple clustering of Y values (round to nearest 0.5m)
        ys = centers[:, 1]
        lane_candidates = np.unique(np.round(ys / 0.5) * 0.5)
        # Only keep lines that have enough points near them
        lane_lines = []
        for y0 in lane_candidates:
            if np.sum(np.abs(ys - y0) < 0.4) > max(5, len(ys) * 0.02):
                lane_lines.append(float(y0))
        # Limit number of lines for clarity
        lane_lines = sorted(lane_lines)[:12]
        for y0 in lane_lines:
            traces.append(go.Scatter3d(
                x=[x_min, x_max, None], y=[y0, y0, None], z=[0, 0, None],
                mode="lines",
                line=dict(color="#CCCCCC", width=2, dash="dash"),
                name=f"lane y={y0:.1f}",
                showlegend=False,
            ))
            trace_meta.append({"kind": "lane", "id": -1})

    # Add box wireframes as individual traces per box for granular legend filtering
    for (oid, typ), group in by_group.items():
        color = _color_for_id(oid)
        width = 3 if typ == "detector" else 2
        dash = "solid" if typ == "detector" else "dot"
        opacity = 1.0 if typ == "detector" else 0.9

        for b in group:
            x, y, z = _cuboid_wireframe(b["center"], b["size"])
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
                name=f"{typ} | id={oid} | f={b['frame']}",
                legendgroup=f"{typ}_id_{oid}",
                hovertemplate=(
                    f"type: {typ}<br>"
                    f"id: {oid}<br>"
                    "frame: %{customdata[0]}<br>"
                    "center: (%{customdata[1]:.2f}, %{customdata[2]:.2f}, %{customdata[3]:.2f})<br>"
                    "size: (%{customdata[4]:.2f}, %{customdata[5]:.2f}, %{customdata[6]:.2f})<extra></extra>"
                ),
                customdata=[[
                    b["frame"],
                    b["center"][0], b["center"][1], b["center"][2],
                    b["size"][0], b["size"][1], b["size"][2],
                ]] * len(x),
                showlegend=False,
            ))
            trace_meta.append({"kind": typ, "id": oid})

    # Add center lines per object id (tracklet)
    if show_tracklet_lines:
        for oid, seq in centers_by_id.items():
            color = _color_for_id(oid)
            pts = np.array([c for _, c in seq])
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color),
                name=f"tracklet id={oid}",
                legendgroup=f"tracklet_id_{oid}",
                hovertemplate="id: " + str(oid) + "<br>x:%{x:.2f} y:%{y:.2f} z:%{z:.2f}<extra></extra>",
                showlegend=True,
            ))
            trace_meta.append({"kind": "line", "id": oid})

    # Build visibility masks for dropdown filters
    n = len(traces)

    def mask_all():
        return [True] * n

    def mask_detectors_only():
        return [m["kind"] == "detector" for m in trace_meta]

    def mask_trackers_only():
        return [m["kind"] == "tracker" for m in trace_meta]

    def mask_object(oid: int):
        return [m["id"] == oid for m in trace_meta]

    # Dropdown buttons
    unique_ids = sorted({m["id"] for m in trace_meta if m["id"] >= 0})
    buttons = [
        dict(label="Show All", method="update", args=[{"visible": mask_all()}]),
        dict(label="Detectors Only", method="update", args=[{"visible": mask_detectors_only()}]),
        dict(label="Trackers Only", method="update", args=[{"visible": mask_trackers_only()}]),
    ]
    for oid in unique_ids:
        buttons.append(dict(label=f"Object {oid}", method="update", args=[{"visible": mask_object(oid)}]))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Tracklet and Detection Visualization (Road Scenario)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            zaxis=dict(range=[0, max(6.0, float(np.max([b['size'][2] for b in boxes])) + 1.0)])
        ),
        width=1200,
        height=820,
        updatemenus=[dict(
            type="dropdown",
            buttons=buttons,
            x=1.15,
            y=1.0,
            xanchor="left",
            yanchor="top",
            showactive=True,
        )],
        legend=dict(
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=0, r=320, t=50, b=0),
    )

    # Export self-contained HTML
    pio.write_html(fig, file=html_path, auto_open=False, include_plotlyjs="cdn")
    print(f"Wrote visualization to {html_path}")


def main():
    # Default: run the realistic road scenario
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
    visualize_scene(boxes, html_path="visualization.html", show_tracklet_lines=True, draw_ground_lanes=True)


if __name__ == "__main__":
    main()
