# DON&T — Deconflicted Optimal Navigation & Trajectory-learning

This repository trains and evaluates a shared MAPPO policy for fixed-wing UAV swarm deconfliction, waypoint following, and boundary-aware navigation.

The current setup is intentionally stationary. Training does not use curriculum; it samples randomized multi-UAV missions from the ranges in `config/tuning_config.yaml`.

## What It Does

- Trains one shared MAPPO actor with a centralized critic
- Samples randomized 2-5 UAV missions with 6-20 waypoints per UAV during training
- Uses reference-guided warm-start supervision before PPO updates take over
- Applies runtime waypoint re-approach assist when a UAV blows a tight pass and starts orbiting
- Scales episode timeout by both mission size and planned route distance
- Tests a trained checkpoint on manual or generated missions
- Supports live waypoint appends and queue replacement through a trained-policy runtime wrapper
- Benchmarks a trained checkpoint across randomized evaluation missions
- Produces optional MP4 rollouts for visual inspection

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Project Structure

```text
src/
├── deconfliction_factory.py   # Mode dispatcher
├── train.py                   # MAPPO training loop
├── test.py                    # MAPPO inference + optional video export
├── eval.py                    # MAPPO random-mission benchmark
├── inference_setup.py         # Shared test/runtime environment setup helpers
├── mappo.py                   # Policy, critic, normalization, GAE helpers
├── mappo_runtime.py           # Shared rollout/inference helpers
├── trained_policy_runtime.py  # Deployment-style trained-policy runtime wrapper
├── pettingzoo_env.py          # Parallel multi-UAV environment
└── flight_engine/             # Aircraft dynamics and geometry helpers

config/
├── train_config.yaml
├── test_config.yaml
├── eval_config.yaml
└── tuning_config.yaml
```

## Quick Start

All modes run through `deconfliction_factory.py`:

```bash
python src/deconfliction_factory.py --mode train
python src/deconfliction_factory.py --mode test
python src/deconfliction_factory.py --mode eval
```

Or with `make`:

```bash
make train
make test
make eval
```

## Configuration

`config/train_config.yaml`

- PPO optimizer, rollout, warm-start, checkpoint, and TensorBoard settings
- `latest_model_name` controls the stable checkpoint path used by test and eval

`config/tuning_config.yaml`

- Shared environment limits such as agent count, safety thresholds, and map ranges
- Training mission sampling ranges
- Reward shaping for progress, completion, safety, boundary pressure, circling, and stagnation
- Guidance settings for deconfliction and waypoint re-approach behavior

`config/test_config.yaml`

- MAPPO checkpoint path
- Manual mission definitions or generated mission settings
- Video export controls
- Visual overlay toggles such as `show_planned_dubins_paths`

`config/eval_config.yaml`

- MAPPO checkpoint path
- Number of randomized benchmark missions
- Evaluation mission size controls

## Training

Training currently uses:

- `32` parallel environments
- `10,000,000` total timesteps
- reference warm-start before PPO updates
- randomized rectangular flight boxes from `500 m` to `1800 m`
- randomized training missions with `2-5` UAVs and `6-20` waypoints per UAV

Episode timeout is not fixed per mission anymore. The environment starts from `tuning.env.shared.max_steps` and can scale upward using:

- assigned waypoint count
- longest planned route distance
- `timeout_max_steps` as a hard cap

This helps large missions avoid timing out simply because they sampled more work.

### Checkpoints

Training saves checkpoints into `train.save_dir`:

```text
models_mappo/
├── mappo_uav_policy_latest.pt
├── mappo_uav_policy_step_100000.pt
├── ...
└── mappo_uav_policy.pt
```

`mappo_uav_policy_latest.pt` is refreshed during training using `train.latest_checkpoint_interval`.

Important: periodic checkpoints are now written after the PPO optimizer step, so `*_latest.pt` reflects the newest trained weights rather than a pre-update snapshot from the same timestep.

`Ctrl+C` writes an `_interrupted.pt` checkpoint before exiting.

### TensorBoard

TensorBoard logs are written under `train.tensorboard.log_dir`. By default, each training launch gets its own timestamped subdirectory, and `latest_run.txt` points to the newest run.

Launch TensorBoard with:

```bash
tensorboard --logdir models_mappo/tensorboard
```

Useful training curves include:

- `train/mean_return_50`
- `train/completion_rate_50`
- `train/crash_rate_50`
- `train/timeout_rate_50`
- `train/waypoint_throughput_per_min_50`
- `train/geofence_outside_steps_50`
- `train/circling_steps_50`
- `policy/reference_mae`

Useful episode-level diagnostics include:

- `episode/circling_steps`
- `episode/circling_breakouts`
- `episode/waypoint_reapproach_steps`
- `episode/waypoint_reapproach_events`
- `episode/deconfliction_time_s`
- `episode/geofence_outside_steps`

## Waypoint Re-approach

The environment includes a runtime waypoint re-approach assist for fixed-wing geometry failures.

If a UAV gets too close to a waypoint, cannot capture it cleanly within its turn limits, and starts stalling or circling, the environment:

- resets the local reference route from the aircraft's current state
- temporarily overrides the executed turn command with the reference breakout/rejoin command
- keeps that assist active for a short minimum window
- releases control once the aircraft has opened the geometry and can make a sane inbound pass again

Those assisted steps are masked out of the PPO actor loss so the policy is not trained as if it chose the override itself.

## Testing

Test mode runs the checkpoint in `config/test_config.yaml` against either:

- `mission_mode: "missions"` for hand-authored scenarios
- `mission_mode: "gen_mission"` for generated missions

Useful switches:

- `save_visuals: false` for a fast stats-only run
- `save_visuals: true` to export plots and, if enabled, an MP4 under `reports/`
- `show_planned_dubins_paths: false` to hide the pregenerated/planned Dubins overlay in visual outputs
- `model_path: "models_mappo/mappo_uav_policy_latest.pt"` to keep testing the rolling latest checkpoint

Test mode keeps the mission definition fixed for the episode. Live waypoint mutation is not enabled there.

## Trained Runtime

Live waypoint edits are available only through `src/trained_policy_runtime.py`, which is intended for deployment or external integration around an already-trained checkpoint.

It is not enabled in:

- training
- normal `test` mode
- `eval` mode

The runtime wrapper builds the same test-style environment, but with live waypoint updates enabled and episode termination on waypoint completion disabled. That means a UAV can finish its current queue, loiter, and later receive more waypoints without resetting the episode.

When you launch your own integration from the repo root, run it with `PYTHONPATH=src` so the runtime modules resolve.

Example:

```python
from trained_policy_runtime import TrainedPolicyRuntime

runtime = TrainedPolicyRuntime.from_config()

runtime.append_waypoints(
    "UAV-1",
    [
        {"id": "dispatch-101", "lat": 37.7751, "lon": -122.4179},
        {"id": "dispatch-102", "lat": 37.7758, "lon": -122.4168},
    ],
)

runtime.replace_waypoint_queue(
    "UAV-2",
    [
        {"id": "dispatch-201", "lat": 37.7747, "lon": -122.4149},
        {"id": "dispatch-202", "lat": 37.7752, "lon": -122.4138},
    ],
)

runtime.replace_waypoint_queue(
    "UAV-3",
    [{"id": "dispatch-301", "lat": 37.7739, "lon": -122.4185}],
    replace_current=True,
)

while True:
    result = runtime.step()
    if result.terminated or result.truncated:
        break
```

Live state access:

```python
uav_1 = runtime.agent_state("UAV-1")
print(uav_1["position"]["lat"], uav_1["position"]["lon"])
print(uav_1["bearing_deg"], uav_1["mode"])
print(uav_1["current_target"])
print(uav_1["targets_by_id"][uav_1["current_target_id"]])

target = runtime.target_state("UAV-1", uav_1["current_target_id"])
print(target["id"], target["status"])

all_uavs = runtime.agent_states()
all_targets_for_uav_1 = runtime.target_states("UAV-1")

result = runtime.step()
live_states_after_step = result.agent_states
```

Queue semantics:

- `append_waypoints(agent, [...])` keeps the current active waypoint and adds the new waypoints to the tail of the queue.
- If that UAV has already completed its queue and is loitering, the first appended waypoint becomes the new current target and navigation resumes immediately.
- `replace_waypoint_queue(agent, [...])` replaces only the queued backlog. The current active waypoint stays in place.
- `replace_waypoint_queue(agent, [...], replace_current=True)` replaces the full remaining mission immediately, including the current active waypoint.
- Passing an empty list with `replace_current=True` clears the remaining mission and the UAV returns to loiter.
- Runtime waypoint payloads are dicts shaped like `{"id": "...", "lat": ..., "lon": ...}`. If you pass bare `(lat, lon)` pairs, the runtime auto-generates a unique waypoint ID for that UAV. Duplicate waypoint IDs for the same UAV are rejected.

When a live update is applied, the runtime refreshes waypoint caches, reference guidance, observation history, and timeout accounting so the trained policy sees the new mission state on the next control step.

State semantics:

- `runtime.agent_state(agent)` returns the latest snapshot for one UAV.
- `runtime.agent_states()` returns the latest snapshot for every UAV in the runtime session.
- `runtime.target_state(agent, target_id)` returns one target/waypoint dict by its unique ID for that UAV.
- `runtime.target_states(agent)` returns all of that UAV's targets keyed by waypoint ID.
- `result.agent_states` mirrors those snapshots immediately after each `runtime.step()`.
- `position.lat` and `position.lon` are the live geodetic coordinates.
- `bearing_deg` is the live compass bearing in `[0, 360)`.
- `heading_rad` and `heading_deg` expose the same orientation in the simulator's signed heading convention.
- `mode` and `flight_mode` expose the current flight mode such as `NAVIGATING` or `LOITERING`.
- `current_waypoint`, `current_target`, `queued_waypoints`, and `hit_waypoints` are waypoint dicts with unique IDs.
- `current_target_id` gives the active target ID directly.
- `targets.current`, `targets.queued`, `targets.hit`, and `targets.remaining` group the same waypoint dicts by status.
- `targets_by_id` and `waypoints_by_id` expose the UAV's target state keyed by waypoint ID, with each value including `id`, `lat`, `lon`, and `status`.
- `completed_waypoints` remains the count of hit waypoints, while `hit_waypoints` keeps the full hit-waypoint history for that UAV.
- Snapshots remain available after a step that terminates or truncates the episode, so you can still inspect final aircraft state before calling `reset()`.

## Evaluation

Eval mode benchmarks one checkpoint over randomized missions and writes results to `results/eval_results.json`.

The evaluation summary now includes more than reward alone. It records mission-success and behavior metrics such as:

- waypoint completion rate
- crash and timeout outcomes
- waypoint throughput
- geofence exits and outside steps
- minimum pairwise separation
- deconfliction time
- circling steps and circling breakouts
- waypoint re-approach steps and events
- flown distance versus planned route distance

This makes it easier to spot policies that achieve reward by circling, stalling, or drifting instead of completing missions cleanly.
