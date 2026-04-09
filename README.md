<<<<<<< Updated upstream
# DON&T — Deconflicted Optimal Navigation & Trajectory-learning

A reinforcement learning framework for training autonomous fixed-wing UAVs to navigate waypoint missions while avoiding collisions in shared airspace. Built with Stable Baselines3 and Gymnasium, featuring curriculum learning for progressive difficulty scaling.
=======
# DSA Sweeper

DSA Sweeper is a multi-drone search simulation that flies a deterministic motion model over a grayscale heatmap, continuously decays searched areas with a camera footprint model, and exports one per-drone JSON timeseries of flight and camera state.

The current codebase is set up for three main workflows:
>>>>>>> Stashed changes

1. Run the visualizer and watch the drones search the map.
2. Call the export pipeline programmatically or from `make` with an image plus georeferenced corners, then save one JSON timeseries per drone.
3. Run the planner as an incremental session where an external system submits drone observations and receives the next recommended targets back.

<<<<<<< Updated upstream
## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Monitoring](#monitoring)
=======
## What The System Does

- Uses a deterministic simulated flight controller for each drone.
- Plans search targets from high-value heatmap clusters.
- Coordinates drones with lightweight reserved-path and reserved-coverage logic so they avoid duplicating work.
- Moves the camera continuously using local, reachable pitch and yaw changes instead of snapping to a global best point.
- Lets you control how quickly searched heat decays, from instant clearing to gradual fade-out.
- Supports an external observe-and-plan loop where another system can submit drone positions plus projection points on every cycle.
- Exports per-drone JSON files with one record per timestep.
- Each timestep record includes:
  - drone latitude and longitude
  - drone AGL and bearing
  - camera ground projection latitude and longitude
  - camera orientation vector components `cam_x`, `cam_y`, `cam_z`

## Current Behavior

The runtime is currently configured around:
>>>>>>> Stashed changes

- A deterministic `SimulatedFlightController`
- A grayscale heatmap image
- A camera model with dynamic pitch and yaw
- A CPU-friendly central planner
- Corner-based geographic export in WGS84 latitude and longitude fields

<<<<<<< Updated upstream
## Updates/Notes

- Noticed when looking at logs that the explained varience was never improving once the drones seemingly learned general navigation. 
- I believe the old heading reward was overpowering the drones need for deconfliction. This action was being emphasized by the training episodes not properly terminating on a crash. 
- Added a new reward for relative distance to target in place of the heading reward. The new reward give points for drones shortening their min distance to the target.
- After some testing, I found a minimal heading reward to actually be necessary, without it my drones are simply learning to fly in circles. 
- Ive spent pretty much all of my time trying to tune the rewards for the drones to avoid eachother. However, there seems to be a gap in my knowledge and I cannot get it to work. For now, I have added a hard safety turn that will force the drones to turn away from eachother. To go with this, there is also a penalty for it having to be activated, hopefully this will still allow for learning to take place instead of relying on the fallback.
- Added vector normalization to account for the fact that flight zones and waypoints hit can vary wildly.

After much testing and fidling with the parameters, I have yet to be able to get the drones to consistently fly waypoints while also deconflicting. Beyond just visually being abale to see this struggle, when checking the tensorboard logs, information such as the explained varience and loss numbers tell me the policies are not converging as expected. This is leading towards the path of more multi-agent RL (MARL) systems. Currently the system that is using basic PPO has produced the best results, but it is still unnacceptable for what I need from this project. Based on the current system, I would expect MARL to be able to produce better results. MARL systems that might be good to test will be MAPPO and QMix.This information is somewhat backed by the <b>eval_visualizer.ipynb</b> notebook which I have included in the repository. It shows a sample of the current system's performance and its inconsistencies that would not be acceptable on a deployed platform.

## Overview

The system trains UAV autopilots using reinforcement learning to accomplish complex missions:

- Navigate through sequential waypoint missions
- Avoid collisions with other aircraft in shared airspace (1–5 simultaneous UAVs)
- Respect geofence boundaries
- Generalize across dynamic environments via curriculum learning

Training progresses through 7 phases, expanding the operational area (200m → 1500m) and incrementally adding aircraft (1 → 5 UAVs). Rewards are normalized across phases using `VecNormalize` so the policy gradient signal stays stable as episode complexity grows.
=======
The DON’T / RL motion path is not part of the active runtime anymore.

## Repository Layout

```text
.
├── api_showcase.ipynb
├── config.yaml
├── Makefile
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py
    ├── export_paths_geojson.py
    ├── geo_reference.py
    ├── main.py
    ├── runtime_builder.py
    ├── session_api.py
    ├── visualizer.py
    ├── drone_controller/
    │   ├── flight_controller.py
    │   └── motion.py
    ├── heat_map_updates/
    │   ├── camera_projection.py
    │   ├── heatmap_loader.py
    │   └── heatmap_updates.py
    └── planner_folder/
        ├── central_planner.py
        ├── geometry.py
        └── planner.py
```
>>>>>>> Stashed changes

## Installation

<<<<<<< Updated upstream
## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
├── deconfliction_factory.py   # Main entry point — reads config and dispatches mode
├── train.py                   # Training loop with curriculum callback
├── test.py                    # Inference, visualization, and video generation
├── eval.py                    # Multi-mission benchmarking across algorithms
├── gym_env.py                 # Gymnasium environment (rewards, obs, collision logic)
│
├── config/
│   ├── train_config.yaml
│   ├── test_config.yaml
│   └── eval_config.yaml
│
└── flight_engine/
    ├── simulator.py           # FixedWingAircraft with Dubins path following
    ├── flight_calcs.py        # Turn and motion calculations
    ├── wp_manager.py          # Waypoint queue and arrival logic
    ├── trans_coorders.py      # Geographic ↔ local coordinate transforms
    ├── helpers.py             # Position, FlightMode, utility functions
    └── visualizer.py          # Matplotlib multi-UAV plotting
```

---

## Quick Start

All modes are launched through `deconfliction_factory.py`:

```bash
# Train a new model
python deconfliction_factory.py --mode train

# Run inference on a trained model
python deconfliction_factory.py --mode test

# Benchmark multiple saved models
python deconfliction_factory.py --mode eval
```

Or with make:

```bash
make train
make test
make eval
```

Edit the corresponding yaml in `config/` before running.

---

## Training

### Training Config: [`config/train_config.yaml`](config/train_config.yaml)

Training checkpoints are saved at each curriculum phase transition and on completion:

```
models/
├── sac_uav_curriculum_phase_1_step_100000.zip
├── sac_uav_curriculum_phase_1_step_100000_vecnorm.pkl
├── ...
└── sac_uav_curriculum.zip
└── sac_uav_curriculum_vecnorm.pkl
```

Press `Ctrl+C` at any time to save an `_interrupted` checkpoint and exit cleanly.

### Reward design

Three-tier collision penalty ensures avoidance is never a tradeoff against waypoints:

| Tier | Range | Effect |
|---|---|---|
| Anticipatory shaping | `caution_dist × 3` → 0 | Quadratic per-step penalty; gives dense gradient |
| One-shot critical | First entry into `critical_dist` | Large single penalty per pair per episode |
| Terminal crash | Any pair inside `critical_dist` | Large per-UAV penalty + episode ends immediately |

The terminal crash penalty is calibrated so no waypoint sequence can rationalize accepting a collision.

---

## Testing & Evaluation

### Testing Config: [`config/test_config.yaml`](config/test_config.yaml)

### Test mode

Runs the trained policy on manually defined missions. Set `save_visuals: false` for a fast stats-only run, or `true` to render an MP4 video of the full episode.

**Important:** always load the `_vecnorm.pkl` that was saved with your model. The normalizer statistics are part of the trained system — loading a model without its normalizer will produce incorrect policy behavior.

### Eval mode

### Eval Config: [`config/eval_config.yaml`](config/eval_config.yaml)

Generates random missions and benchmarks each configured algorithm over `num_missions` episodes, then prints a ranked summary table:

```
Algorithm | Mean Reward | Std | Avg Steps | WP Completion | Crash Rate
```

---

## Monitoring
=======
Create an environment and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Current required packages in [requirements.txt](requirements.txt):

- `numpy`
- `matplotlib`
- `PyYAML`
- `opencv-python`
- `imageio`

## Quick Start

Run the visualizer:

```bash
make run
```

Compile-check the code:
>>>>>>> Stashed changes

```bash
make compile
```

<<<<<<< Updated upstream
Open http://localhost:6006 to track episode rewards, value function estimates, policy entropy, and curriculum phase transitions in real time.
=======
Open the interactive API walkthrough notebook:

```bash
jupyter notebook api_showcase.ipynb
```

That notebook mirrors the README with runnable examples for both public APIs.
It now also shows the config-backed API call style with both `config=` and
`config_path=`.

Export per-drone JSON files using the current config:

```bash
make export-paths
```

By default, `make export-paths` follows `export.save_video` from
`config.yaml`. With the current config, that creates:

- `exports/drone_0.json`
- `exports/drone_1.json`
- `exports/drone_2.json`
- `exports/drone_3.json`
- `exports/replay.mp4`

The replay video is written at `1 / step_seconds` frames per second, so its
playback time matches the exported timestamps.

Export stop behavior:

- default `make export-paths` behavior is `MAX_STEPS=2000`
- set `MAX_STEPS=none` to remove the step cap and run until the planner finishes
- set `STOP_WHEN_COVERED_PERCENT=<0..100>` to stop early once that percentage of the initial heatmap value has been cleared
- set `SAVE_VIDEO=false` to disable replay video output for that export
- if both are set, export stops at whichever limit is reached first
- `STOP_WHEN_COVERED_PERCENT` is based on cleared initial heatmap value, not raw geometric area

Examples:

```bash
make export-paths OUTPUT=exports MAX_STEPS=none
make export-paths OUTPUT=exports MAX_STEPS=none STOP_WHEN_COVERED_PERCENT=95
make export-paths OUTPUT=exports SAVE_VIDEO=false
```

If `OUTPUT` has a file extension, it is treated as a base filename:

```bash
make export-paths OUTPUT=paths.json
```

That creates:

- `paths_drone_0.json`
- `paths_drone_1.json`
- `paths_drone_2.json`
- `paths_drone_3.json`
- `paths.mp4`

If you pass a `.geojson` filename as the base output name, the exporter still writes `.json` files because the output is no longer GeoJSON.

CLI options:

- `--max-steps none` runs until the planner finishes or another stop condition is reached
- `--stop-when-covered-percent <0..100>` stops early once that percentage of the initial heatmap value has been cleared
- `--save-video` overrides config and enables replay video generation
- `--no-video` overrides config and disables replay video generation
- `--video-output <path>` writes the replay video to an explicit location

Equivalent CLI examples:

```bash
python3 -m src.export_paths_geojson --config config.yaml --max-steps none --output exports
python3 -m src.export_paths_geojson --config config.yaml --max-steps none --stop-when-covered-percent 95 --output exports
```

Override search decay from `make`:

```bash
make export-paths OUTPUT=exports SEARCH_DECAY_PERCENT_PER_100MS=10
```

## Decay And Step Rate Quick Reference

You can set both the search decay rate and the planner timestep from the config
file or from both public APIs.

- Config: `heatmap.search_decay_percent_per_100ms: 10.0`
- API v1: `export_drone_geojsons_from_image(..., search_decay_percent_per_100ms=10.0)`
- API v2: `create_incremental_session_from_image(..., search_decay_percent_per_100ms=10.0)`
- Config: `planner.step_seconds: 0.1`
- API v1: `export_drone_geojsons_from_image(..., step_seconds=0.1)`
- API v2: `create_incremental_session_from_image(..., step_seconds=0.1)`

For API v2, the decay setting and the elapsed time are separate:

- `search_decay_percent_per_100ms` sets the policy
- `step_seconds` sets the default timestep when no per-call override is provided
- `observe_and_plan(..., dt_seconds=...)` tells the session how much real time that update represents

Examples:

- `100` means covered heat clears immediately
- `10` means a max-value cell takes about 1 second of continuous coverage to clear
- `1` means a max-value cell takes about 10 seconds of continuous coverage to clear

## Core Entry Points

### Visualizer

- Module: [src.main](src/main.py)
- Command: `python3 -m src.main`
- Make target: `make run`

This loads [config.yaml](config.yaml), opens a matplotlib UI, and animates:

- drone positions
- assigned flight targets
- assigned camera targets
- current camera footprint

### Export API

- Module: [src.export_paths_geojson](src/export_paths_geojson.py)
- Main high-level callable:
  - `from src import export_drone_geojsons_from_image`

This is the intended API surface if another system wants to provide:

- an image
- image scale
- `tl`, `tr`, `br`, `bl` corner coordinates

and receive back per-drone JSON timestep arrays.

### Incremental Session API

- Module: [src.session_api](src/session_api.py)
- High-level callable:
  - `from src import create_incremental_session_from_image`

This is the intended API surface when another system owns motion and sensing. The outside system:

1. Initializes a session once.
2. Submits the latest drone positions and projection points.
3. Receives the next recommended targets.
4. Moves the drones externally.
5. Repeats.

## Python APIs

There are two Python API versions exposed from `src`. They solve different problems.

Both public APIs can be used in two styles:

- explicit inputs: pass `image_source`, `corners`, and `source_image_ppm` yourself
- config-backed inputs: pass `config=` or `config_path=` and let the API read those values from the YAML config

If neither `config` nor `config_path` is supplied, the repo's default
`config.yaml` is loaded automatically. Any explicit call arguments still
override the config values.

### Choose The Right API

#### API v1: One-Shot Export

Use `export_drone_geojsons_from_image(...)` when you want to:

- give the system an image and its georeferenced corners directly, or let it read them from config
- let the internal simulator run the search
- get back per-drone JSON timestep outputs

This is the right API for offline simulation, batch export, or testing.

#### API v2: Incremental Session

Use `create_incremental_session_from_image(...)` when you want to:

- keep the drone motion outside this repo
- send the latest drone positions and projection points each cycle
- get back the next recommended targets each cycle
- preserve planner and map state across repeated calls

This is the right API for integration with another flight stack, autonomy loop, or live system.

### API v1: One-Shot JSON Export

The export API can use explicit image inputs or bootstrap them from
`config=` / `config_path=`. The decay rate can come from config or be
overridden directly on this API call with
`search_decay_percent_per_100ms=...`. Drone IDs and starting locations can
also come from config or be overridden per call. You can still mix config and
API usage overall, but `drone_ids` and `initial_drone_positions` are full
replacement lists, not partial patches.

Import:

```python
from src import export_drone_geojsons_from_image
from src.config import load_config
```

Config-backed call:

```python
config = load_config("config.yaml")
exports = export_drone_geojsons_from_image(
    config=config,
    max_steps=None,
    stop_when_covered_percent=95.0,
    save_video=False,
)

print(exports.keys())
first_drone_id = next(iter(exports))
print(exports[first_drone_id][0])
```

Config-path shorthand:

```python
exports = export_drone_geojsons_from_image(
    config_path="config.yaml",
    max_steps=200,
)
```

Explicit image/corners call:

```python
exports = export_drone_geojsons_from_image(
    "resized.png",
    {
        "tl": {"lat": 47.92830697389202, "lon": -97.1052720857056},
        "tr": {"lat": 47.92830697389202, "lon": -97.07851323120872},
        "br": {"lat": 47.91031953734, "lon": -97.07851323120872},
        "bl": {"lat": 47.91031953734, "lon": -97.1052720857056},
    },
    source_image_ppm=0.5,
    max_steps=None,
    stop_when_covered_percent=95.0,
    greedy_paths_enabled=True,
    initial_altitudes_agl=[126.0, 128.0, 124.0, 126.0],
    drone_ids=["alpha", "beta", "charlie", "delta"],
    initial_drone_positions=[
        {"e": 920.0, "n": 1000.0, "heading": 0.0},
        {"e": 1000.0, "n": 1080.0, "heading": 90.0},
        {"lat": 47.9189, "lon": -97.0924, "heading": 180.0},
        {"lat": 47.9197, "lon": -97.0914, "heading": 270.0},
    ],
    search_decay_percent_per_100ms=100.0,
    step_seconds=0.1,
    save_video=False,
)
```

Stop behavior in Python:

- default `export_drone_geojsons_from_image(..., max_steps=2000)`
- pass `max_steps=None` to run until planner completion
- pass `stop_when_covered_percent=95.0` to stop once 95% of the initial heatmap value has been cleared
- if both are set, whichever limit is reached first ends the export

In-memory image call:

```python
import numpy as np
from src import export_drone_geojsons_from_image

image = np.zeros((1000, 1000), dtype=np.uint8)

exports = export_drone_geojsons_from_image(
    image,
    {
        "tl": (47.92830697389202, -97.1052720857056),
        "tr": (47.92830697389202, -97.07851323120872),
        "br": (47.91031953734, -97.07851323120872),
        "bl": (47.91031953734, -97.1052720857056),
    },
    source_image_ppm=0.5,
    max_steps=500,
    greedy_paths_enabled=False,
    initial_altitudes_agl=[126.0],
    search_decay_percent_per_100ms=25.0,
    step_seconds=0.1,
)
```

API v1 inputs:

- `config`
  - optional loaded `SweeperConfig`
  - provides default image path, source image ppm, geo corners, and runtime settings
- `config_path`
  - optional YAML path loaded when `config` is not supplied
  - defaults to the repo's `config.yaml` when neither `config` nor `config_path` is provided
- `image_source`
  - optional image path string, `Path`, or NumPy image array
  - when omitted, the API uses `config.heatmap.image_path`
- `corners`
  - optional `tl`, `tr`, `br`, `bl`
  - when omitted, the API uses `config.geo_reference`
- `source_image_ppm`
  - optional source image pixels per meter
  - when omitted, the API uses `config.heatmap.source_image_ppm`
- `drone_ids`
  - optional drone ID list
  - when omitted, the API uses `config.planner.drone_ids` or falls back to `drone_0`, `drone_1`, ...
  - when provided, it must be a full list with one unique ID per active drone
- `initial_drone_positions`
  - optional per-drone starting positions
  - each entry must provide either `e` and `n` or `lat` and `lon`
  - each entry may also include an optional starting `heading`
  - when omitted, the API uses `config.planner.initial_drone_positions` or the default centered-circle spawn pattern
  - when provided, it must be a full list with one entry per active drone
- `max_steps`
  - optional planner horizon
  - defaults to `2000`
  - pass `None` in Python or `--max-steps none` in the CLI to run until the planner finishes or another stop condition is reached
- `stop_when_covered_percent`
  - optional early-stop threshold from `0` to `100`
  - pass `95.0` in Python or `--stop-when-covered-percent 95` in the CLI to stop once 95% of the initial heatmap value has been cleared by observation
  - this is based on cleared initial heatmap value, not literal geometric map area
  - if both `max_steps` and `stop_when_covered_percent` are set, the export stops at whichever limit is reached first
- `greedy_paths_enabled`
  - optional override for greedy intermediate waypoints on the route to the main hotspot
- `initial_altitudes_agl`
  - optional one-value or per-drone initial AGL override in meters
  - a single value is broadcast to every drone
  - this is the creation-time AGL for API v1; use API v2 observations when you need to update AGL over time
- `search_decay_percent_per_100ms`
  - optional search persistence override
  - `100` clears covered heat immediately
  - `1` removes 1% of full-scale heat per 100 ms of observation, so a max-value cell takes about 10 seconds of continuous coverage to fully clear
- `step_seconds`
  - optional planner timestep override in seconds
  - controls simulated motion distance per step, heatmap decay per internal step, and exported timestamps
- `video_output`
  - optional replay video destination
  - no video is written unless you provide a path
- `save_video`
  - optional boolean override for whether replay video writing is enabled
  - defaults to `config.export.save_video`
  - this only has an effect when `video_output` is also provided

If you pass `image_source`, `corners`, or `source_image_ppm` explicitly, those
values override what the selected config would have provided.

API v1 return value:

```python
{
    "drone_0": [{...timestep record...}, {...timestep record...}, ...],
    "drone_1": [{...timestep record...}, {...timestep record...}, ...],
    ...
}
```

Those dictionary keys follow the active drone IDs, so if you configure
`["alpha", "beta"]`, the export keys become `alpha` and `beta`.

Each per-drone export is a JSON array with one record per simulated timestep.
The first record is the initial planner state at `timestamp = 0.0`, and each
additional record advances by the configured `step_seconds` value.

Each timestep record contains:

- `timestamp`: simulated time in seconds
- `drone_lat`, `drone_lon`: drone geographic position
- `agl`: drone altitude above ground in meters
- `bearing`: drone heading in degrees, normalized to `0..360`
- `cam_proj_lat`, `cam_proj_lon`: camera center-ray ground intersection in geographic coordinates
- `cam_x`, `cam_y`, `cam_z`: camera center-ray orientation vector in the planner's local world frame
  - `cam_x`: east component
  - `cam_y`: north component
  - `cam_z`: vertical component
  - negative `cam_z` means the camera is pointing downward

Example record:

```python
{
    "timestamp": 0.1,
    "drone_lat": 47.91966514297415,
    "drone_lon": -97.08964068052376,
    "agl": 126.0,
    "bearing": 12.0,
    "cam_proj_lat": 47.92071159293584,
    "cam_proj_lon": -97.08942189466998,
    "cam_x": 0.09491582661957682,
    "cam_y": 0.6753611989040085,
    "cam_z": -0.7313537016191705,
}
```

### API v2: Incremental Observe-And-Plan Session

The session API can also bootstrap image path, source image ppm, and geo
corners from `config=` / `config_path=`. The decay rate can come from config
or be overridden when the session is created with
`search_decay_percent_per_100ms=...`. The real elapsed time for each update is
then supplied separately with `observe_and_plan(..., dt_seconds=...)`. Drone IDs
and starting locations can also come from config or be overridden per call, but
API overrides for those fields must be full-list replacements.

Import:

```python
from src import create_incremental_session_from_image
from src.config import load_config
```

Basic lifecycle:

1. Create the session once from an image plus corners, or from config.
2. Optionally call `get_plan()` to inspect the initial recommendation.
3. On each cycle, call `observe_and_plan(...)`.
4. Move the drones externally.
5. Submit fresh observations again.

Example:

```python
config = load_config("config.yaml")
session = create_incremental_session_from_image(
    config=config,
    drone_ids=["alpha", "beta", "charlie", "delta"],
    initial_drone_positions=[
        {"e": 920.0, "n": 1000.0, "heading": 0.0},
        {"e": 1000.0, "n": 1080.0, "heading": 90.0},
        {"lat": 47.9189, "lon": -97.0924, "heading": 180.0},
        {"lat": 47.9197, "lon": -97.0914, "heading": 270.0},
    ],
)

initial_plan = session.get_plan()

step_result = session.observe_and_plan(
    {
        "alpha": {
            "position": {"lat": 47.9193, "lon": -97.0897, "agl": 130.0},
            "projection_point": {"lat": 47.9188, "lon": -97.0892},
        },
        "beta": {
            "position": {"lat": 47.9193, "lon": -97.0940},
            "projection_point": {"lat": 47.9187, "lon": -97.0943},
        },
        "charlie": {
            "position": {"lat": 47.9160, "lon": -97.0920},
            "projection_point": {"lat": 47.9153, "lon": -97.0921},
        },
        "delta": {
            "position": {"lat": 47.9220, "lon": -97.0920},
            "projection_point": {"lat": 47.9212, "lon": -97.0918},
        },
    },
    dt_seconds=0.1,
)

print(step_result["drones"]["alpha"]["next_target"])
print(step_result["drones"]["alpha"]["camera_target"])
```

Config-path shorthand:

```python
session = create_incremental_session_from_image(config_path="config.yaml")
```

Explicit image/corners creation is still supported by passing `image_source`,
`corners`, and `source_image_ppm` directly.

API v2 session creation inputs:

- `config`
  - optional loaded `SweeperConfig`
  - provides default image path, source image ppm, geo corners, and runtime settings
- `config_path`
  - optional YAML path loaded when `config` is not supplied
  - defaults to the repo's `config.yaml` when neither `config` nor `config_path` is provided
- `image_source`
  - optional image path string, `Path`, or NumPy image array
  - when omitted, the API uses `config.heatmap.image_path`
- `corners`
  - optional `tl`, `tr`, `br`, `bl`
  - when omitted, the API uses `config.geo_reference`
- `source_image_ppm`
  - optional source image pixels per meter
  - when omitted, the API uses `config.heatmap.source_image_ppm`
- `drone_ids`
  - optional drone ID list
  - when omitted, the API uses `config.planner.drone_ids` or falls back to `drone_0`, `drone_1`, ...
  - when provided, it must be a full list with one unique ID per active drone
- `initial_drone_positions`
  - optional per-drone starting positions
  - each entry must provide either `e` and `n` or `lat` and `lon`
  - each entry may also include an optional starting `heading`
  - when omitted, the API uses `config.planner.initial_drone_positions` or the default centered-circle spawn pattern
  - when provided, it must be a full list with one entry per active drone
- `greedy_paths_enabled`, `initial_altitudes_agl`, `search_decay_percent_per_100ms`, `step_seconds`
  - optional runtime overrides applied on top of the selected config

AGL and decay control in API v2:

- `initial_altitudes_agl=[...]` sets the starting AGL for each drone when the session is created
- `search_decay_percent_per_100ms=...` sets how quickly covered heat fades
- `step_seconds=...` sets the session's default timestep in seconds
- `observe_and_plan(...)` can update a drone's live altitude by including `position.agl`
- `observe_and_plan(..., dt_seconds=...)` lets an external loop scale decay by the real elapsed time since the last update
- if `dt_seconds` is omitted later, the session uses the configured `step_seconds`
- a single `initial_altitudes_agl` value is broadcast to every drone
- if you omit AGL on a later observation, the session keeps that drone's previous AGL
- observation keys must match the configured drone IDs

API v2 observation format:

`observe_and_plan(...)` accepts either:

- a dictionary keyed by your configured drone IDs
- a list of observations in drone index order

For each drone, the observation may include:

- `position` with either:
  - `e`, `n`, optional `heading`, optional `agl`
  - or `lat`, `lon`, optional `heading`, optional `agl`
  - `agl` may also be provided as `altitude_agl`, `altitude_m`, or `altitude`
- `projection_point` with either:
  - `e`, `n`
  - or `lat`, `lon`
- optional `camera_pitch_deg`
- optional `camera_yaw_deg`

If `heading` is omitted, the session infers it from motion since the previous observation when possible.
If `agl` is omitted, the session keeps the drone’s previous AGL.
If `dt_seconds` is omitted, the session uses the configured `step_seconds` value for decay scaling.

API v2 response shape:

```python
{
    "finished": False,
    "planner": {
        "drone_ids": ["alpha", "beta", "charlie", "delta"],
        "greedy_paths_enabled": True,
        "initial_altitudes_agl": [126.0, 128.0, 124.0, 126.0],
        "search_decay_percent_per_100ms": 10.0,
        "step_seconds": 0.1,
    },
    "drones": {
        "alpha": {
            "state": {...},
            "next_target": {...},
            "camera_target": {...},
            "camera_projection_point": {...},
        },
        ...
    },
}
```

For `state`, `next_target`, and `camera_target`, the API includes both:

- local planner coordinates: `e`, `n`
- geographic coordinates: `lat`, `lon`
- state altitude above ground: `agl`

`next_target` also includes the planner’s scoring fields, such as:

- `cluster_mean`
- `unique_route_gain`
- `overlap_gain`
- `route_gain`
- `effective_distance`
- `target_standoff_radius`
- `distance_efficiency`

When greedy paths are enabled, `next_target` can also include:

- `greedy_subtarget`
- `greedy_prefix_steps`
- `greedy_progress_ratio`
- `main_target`

When the underlying hotspot is too close to the image edge, `next_target` may
also include:

- `edge_approach_target`
- `edge_margin_m`
- `main_target_edge_distance`

`camera_target` also includes camera scoring fields, such as:

- `footprint_value`
- `footprint_mean`
- `local_value`

## Command Line Export

The same export logic is available from the module CLI:

```bash
python3 -m src.export_paths_geojson --config config.yaml --output exports
```

CLI stop behavior:

- default `--max-steps 2000`
- use `--max-steps none` to run until planner completion
- use `--stop-when-covered-percent 95` to stop once 95% of the initial heatmap value has been cleared
- replay video saving follows `export.save_video` by default
- use `--save-video` or `--no-video` to override that config value for one run
- if both are supplied, whichever limit is reached first ends the export

Override greedy intermediate waypoint behavior from the CLI:

```bash
python3 -m src.export_paths_geojson --config config.yaml --greedy-paths-enabled --output exports
python3 -m src.export_paths_geojson --config config.yaml --no-greedy-paths --output exports
```

Override initial drone altitudes from the CLI:

```bash
python3 -m src.export_paths_geojson \
  --config config.yaml \
  --initial-altitudes-agl 126 128 124 126 \
  --search-decay-percent-per-100ms 10 \
  --output exports
```

Override the planner timestep:

```bash
python3 -m src.export_paths_geojson \
  --config config.yaml \
  --step-seconds 0.2 \
  --output exports
```

Override the image path:

```bash
python3 -m src.export_paths_geojson \
  --config config.yaml \
  --image resized.png \
  --source-image-ppm 0.5 \
  --output exports
```

Override all corners directly:

```bash
python3 -m src.export_paths_geojson \
  --config config.yaml \
  --image resized.png \
  --source-image-ppm 0.5 \
  --tl-lat 47.92830697389202 --tl-lon -97.1052720857056 \
  --tr-lat 47.92830697389202 --tr-lon -97.07851323120872 \
  --br-lat 47.91031953734    --br-lon -97.07851323120872 \
  --bl-lat 47.91031953734    --bl-lon -97.1052720857056 \
  --output exports
```

Equivalent `make` command:

```bash
make export-paths \
  IMAGE=resized.png \
  SOURCE_IMAGE_PPM=0.5 \
  STEP_SECONDS=0.2 \
  MAX_STEPS=none \
  STOP_WHEN_COVERED_PERCENT=95 \
  SAVE_VIDEO=false \
  TL_LAT=47.92830697389202 TL_LON=-97.1052720857056 \
  TR_LAT=47.92830697389202 TR_LON=-97.07851323120872 \
  BR_LAT=47.91031953734    BR_LON=-97.07851323120872 \
  BL_LAT=47.91031953734    BL_LON=-97.1052720857056 \
  OUTPUT=exports
```

## Geo Referencing

The export path is now corner-based.

Required corner order:

- `tl`: top left
- `tr`: top right
- `br`: bottom right
- `bl`: bottom left

This is important. If the corners are out of order, the exported lat/lon records will be wrong.

### How Coordinates Are Computed

Internally, the planner still works in a local meter-space frame:

- `e` increases to the right
- `n` increases upward

During export, each point is converted to geographic coordinates by bilinear interpolation over the four image corners. That means:

- the planner stays simple and fast
- exported geographic fields are written as separate WGS84 values: `*_lat` and `*_lon`
- callers can provide true image corners instead of relying on a single centroid anchor

### Current Default Corners

The current [config.yaml](config.yaml) includes both:

- explicit corners
- the original centroid

The explicit corners were derived from your provided centroid:

- `47.91931325561601, -97.09189265845716`

and the current image footprint after resize:

- `1000 x 1000` pixels
- `0.5` source pixels per meter
- `1.0` meter output resolution
- effective footprint: `2000 m x 2000 m`

Those derived corners are a good working default, but exact surveyed corner coordinates are better if you have them.

## Configuration

The runtime config lives in [config.yaml](config.yaml) and is loaded by [src/config.py](src/config.py).

### `heatmap`

```yaml
heatmap:
  image_path: resized.png
  source_image_ppm: 0.5
  resolution: 1.0
  search_decay_percent_per_100ms: 100.0
```

- `image_path`: input heatmap image
  - grayscale values are used directly by the planner
  - brighter source pixels mean higher search value
- `source_image_ppm`: source image pixels per meter
- `resolution`: planner resolution in meters per pixel after resize
- `search_decay_percent_per_100ms`: percentage points of normalized heat removed per 100 ms of continuous coverage
  - `100` clears immediately
  - `1` takes about 10 seconds of continuous coverage to clear a max-value cell

### `geo_reference`

```yaml
geo_reference:
  tl: {lat: ..., lon: ...}
  tr: {lat: ..., lon: ...}
  br: {lat: ..., lon: ...}
  bl: {lat: ..., lon: ...}
  centroid_lat: ...
  centroid_lon: ...
```

- The export prefers the four explicit corners.
- If corners are missing but centroid is present, the exporter can derive an axis-aligned approximation.
- For accurate export, provide all four corners.

### `camera`

```yaml
camera:
  agl: 126.0
  pitch: -50.0
  yaw: 0.0
  min_pitch: -80.0
  max_pitch: -25.0
  min_yaw: -20.0
  max_yaw: 20.0
  pitch_turn_rate_deg: 3.0
  yaw_turn_rate_deg: 4.0
  matrix:
    - [12285.0, 0.0, 4624.0]
    - [0.0, 12285.0, 3472.0]
    - [0.0, 0.0, 1.0]
```

- `agl`: camera altitude above ground in meters
  - used as the default projection altitude when a drone-specific AGL is not supplied
- `pitch`, `yaw`: starting camera pose
- `min_*`, `max_*`: camera motion limits
- `*_turn_rate_deg`: camera slew-rate limits in degrees per second
  - each planner step can change camera pose by `*_turn_rate_deg * step_seconds`
- `matrix`: camera intrinsics

### `planner`

```yaml
planner:
  num_searchers: 4
  cluster_size: 40
  top_k_clusters: 3
  camera_reach_steps: 3
  greedy_paths_enabled: true
  drone_ids: [drone_0, drone_1, drone_2, drone_3]
  initial_altitudes_agl: [126.0, 126.0, 126.0, 126.0]
  target_radius_percent: 0.25
  drone_speed: 40.0
  step_seconds: 0.1
  max_turn_rate_deg: 12.0
```

- `num_searchers`: number of drones
- `cluster_size`: coarse block size used for search scoring
- `top_k_clusters`: number of top hotspots to consider
- `camera_reach_steps`: how many pitch/yaw turn-rate steps the camera planner looks ahead
- `greedy_paths_enabled`: when enabled, `next_target` may be a short-horizon greedy intermediate waypoint while `main_target` still points at the underlying hotspot
- `drone_ids`: optional explicit IDs for each drone
  - when set, must contain exactly one unique ID per drone
- `initial_drone_positions`: optional explicit start locations for each drone
  - each entry may use either `e`/`n` local coordinates or `lat`/`lon` geo coordinates
  - each entry may also include an optional starting heading in degrees
  - if omitted, the planner uses the default centered-circle spawn pattern
- `initial_altitudes_agl`: one shared value or one per drone; used when the drones are created and as the initial AGL in the projection model, in meters
  - both public APIs can override this at call time, and API v2 can also update AGL later through observations
- `target_radius_percent`: fraction of the camera footprint span treated as usable stand-off search reach when comparing target travel cost
- `drone_speed`: deterministic drone speed in meters per second
- `step_seconds`: duration of one planner/simulator step in seconds
  - both public APIs can override this at call time
- `drone_speed`: deterministic drone speed in meters per second
  - each internal step moves `drone_speed * step_seconds` meters
- `max_turn_rate_deg`: heading turn-rate limit in degrees per second
  - each internal step can change heading by `max_turn_rate_deg * step_seconds`

### `display`

These values only affect the visualizer:

- figure size
- animation interval
  - when `display.sync_to_runtime` is `true`, the visualizer ignores the manual
    interval for simulation timing and instead keeps simulation time aligned to
    `planner.step_seconds`; `display.interval_ms` becomes the redraw cadence, so
    lower values make motion look smoother without changing speed
- colors
- colormap

### `export`

```yaml
export:
  save_video: true
```

- `save_video`: default replay-video behavior for export commands that write
  JSON outputs
  - `true` writes a replay video alongside the JSON outputs unless a CLI or
    API override disables it
  - `false` skips replay video creation unless a CLI or API override enables it

## How Planning Works

### Flight Motion

[SimulatedFlightController](src/drone_controller/flight_controller.py) advances each drone toward its assigned target using deterministic motion with a max turn rate.

When map bounds are active, the controller chooses an in-bounds heading and, if
needed, a shorter forward step instead of letting the drone step outside the map
and then snapping it back. That keeps edge behavior smoother and more physical.

### Heatmap Search

[HeatMapUpdates](src/heat_map_updates/heatmap_updates.py) removes searched areas by projecting the current camera footprint into the map and decaying the covered region.

In the incremental session API, the planner can also update the map from an externally observed projection point. It keeps the configured footprint shape and shifts that footprint so it is centered on the submitted observed ground point before decaying the searched area.

### Multi-Drone Assignment

[choose_best_cluster_target(...)](src/planner_folder/planner.py) picks targets from high-value clusters while considering:

- cluster value
- distance
- reserved target spacing
- short-horizon path conflicts
- predicted unique coverage vs overlap with other drones
- value-per-travel efficiency, so slightly better far-away clusters do not beat good nearby work too easily

When `planner.greedy_paths_enabled` is on, the planner still chooses a main
hotspot target, but it may publish a nearer greedy subtarget from the short
predicted route prefix if that prefix already carries the drone through useful
value. This keeps the path opportunistic without losing the underlying main
search intent.

If a hotspot is too close to the image boundary, the planner also publishes an
interior approach waypoint instead of sending the drone all the way to the edge.
The original edge hotspot is preserved as `main_target` in the API payload.

This keeps drones from repeatedly converging on the same region.

The runtime still validates drone states against the map bounds, but normal
edge behavior now comes from in-bounds steering in the controller rather than a
hard position snap after motion.

### Camera Motion

[choose_camera_projection_target(...)](src/planner_folder/planner.py) does not chase a global best point on the whole map.

Instead, it:

- enumerates pitch and yaw poses the camera can actually reach soon
- projects the full candidate camera footprint onto the heatmap
- scores the actual heatmap value that footprint would hit
- uses footprint mean and nearby local value as tie-breakers
- chooses the best reachable pose

At runtime, the next camera pitch/yaw step is also chosen from an in-bounds
local neighborhood, so intermediate turn-limited camera motion cannot walk the
projection center off the map. Zero-hit camera targets are rejected.

That gives you a camera that keeps moving toward nearby useful coverage while staying cheap enough for edge CPU use.

## Export Output

Each drone file is a JSON array of timestep records.

Filenames:

- directory output writes `<drone_id>.json` for each active drone ID
- file-style output writes `<stem>_<drone_id>.json` for each active drone ID

Record fields:

- `timestamp`: simulated time in seconds
- `drone_lat`, `drone_lon`: drone position in WGS84
- `agl`: altitude above ground in meters
- `bearing`: drone heading in degrees
- `cam_proj_lat`, `cam_proj_lon`: camera center-ray intersection with the ground in WGS84
- `cam_x`, `cam_y`, `cam_z`: camera center-ray direction vector from the drone in local world coordinates

Coordinate conventions:

- local `x` points east
- local `y` points north
- local `z` points up
- the exported camera ray usually has a negative `cam_z` because the camera points toward the ground

## Edge CPU Notes

The current implementation is biased toward lightweight planning:

- deterministic motion instead of RL inference
- coarse cluster scoring instead of full-resolution global optimization
- short-horizon reservation logic
- local camera reachability instead of whole-map camera search
- observation-driven session API that reuses the same lightweight planner state

## Development Notes

- The public import surface is intentionally small:
  - `from src import export_drone_geojsons_from_image`
  - `from src import create_incremental_session_from_image`
- The lower-level export helpers live in [src/export_paths_geojson.py](src/export_paths_geojson.py).
- The incremental control loop lives in [src/session_api.py](src/session_api.py).

## Troubleshooting

### Exported paths look geographically wrong

Check:

- corner order is `tl`, `tr`, `br`, `bl`
- corner values match the input image
- `source_image_ppm` is correct
- `resolution` is correct

### Visualizer or export cannot read the image

Check:

- `heatmap.image_path` exists
- `opencv-python` is installed
- if OpenCV is unavailable, the loader can fall back to matplotlib for file reads, but OpenCV is still the preferred path

### The camera is not moving enough

Check:

- `camera.min_pitch`, `camera.max_pitch`
- `camera.min_yaw`, `camera.max_yaw`
- `camera.pitch_turn_rate_deg`
- `camera.yaw_turn_rate_deg`
- `planner.camera_reach_steps`

## Verified Commands

These paths are currently wired and compile:

```bash
make compile
make run
make export-paths OUTPUT=exports
python3 -m src.export_paths_geojson --config config.yaml --output exports
```

The current image-plus-corners export path was also exercised end-to-end with the repo’s current image and produced per-drone JSON timestep files successfully.

The incremental session API was also exercised against the current image with repeated `observe_and_plan(...)` calls and returned stable next-target and camera-target payloads.
>>>>>>> Stashed changes
