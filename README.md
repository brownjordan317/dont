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
├── mappo.py                   # Policy, critic, normalization, GAE helpers
├── mappo_runtime.py           # Shared rollout/inference helpers
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
- `model_path: "models_mappo/mappo_uav_policy_latest.pt"` to keep testing the rolling latest checkpoint

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
