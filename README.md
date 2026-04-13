# DON&T — Deconflicted Optimal Navigation & Trajectory-learning

This repository is now focused on a single MARL path: MAPPO for fixed-wing UAV deconfliction and waypoint following.

The training setup is intentionally stationary. MAPPO does not use curriculum here; it trains on randomized multi-agent missions sampled from the ranges in `config/tuning_config.yaml`.

## What It Does

- Trains a shared MAPPO policy with a centralized critic
- Samples 1–5 UAV missions inside randomized square flight boxes
- Tests a trained checkpoint on hand-authored missions
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

All entrypoints still run through `deconfliction_factory.py`:

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

- Optimizer, rollout, checkpoint, latest-checkpoint, and TensorBoard settings for MAPPO
- No curriculum settings

`config/tuning_config.yaml`

- Shared environment limits like `min_agents`, `max_agents`, and collision thresholds
- Training distribution ranges such as `train.box_min_m` and `train.box_max_m`
- Reward shaping, hard-safety, anti-circling, and guidance knobs

`config/test_config.yaml`

- MAPPO checkpoint path
- Manual or generated missions for visual/manual inspection

`config/eval_config.yaml`

- MAPPO checkpoint path
- Number of random benchmark missions
- Random benchmark mission size and UAV count controls

## Training

Training saves checkpoints into `train.save_dir` as:

```text
models_mappo/
├── mappo_uav_policy_latest.pt
├── mappo_uav_policy_step_100000.pt
├── ...
└── mappo_uav_policy.pt
```

`mappo_uav_policy_latest.pt` is refreshed during training using `train.latest_checkpoint_interval`, so test mode can keep pointing at one stable checkpoint path.

`Ctrl+C` writes an `_interrupted.pt` checkpoint before exiting.

TensorBoard logs are written under `train.tensorboard.log_dir`. By default, each training launch gets its own timestamped subdirectory, and `latest_run.txt` points to the newest one. After installing the updated requirements, you can inspect runs with:

```bash
tensorboard --logdir models_mappo/tensorboard
```

## Testing

Test mode runs the MAPPO checkpoint in `config/test_config.yaml` against either manual missions or generated missions, depending on `test.mission_mode`.

- Set `save_visuals: false` for a fast stats-only run
- Set `save_visuals: true` to also export an MP4 into `reports/`

## Evaluation

Eval mode benchmarks one MAPPO checkpoint over randomized missions and writes a summary JSON to `results/eval_results.json`.

The benchmark currently reports reward, completion, crash rate, safety interventions, and minimum separation statistics.
