# DON&T — Deconflicted Optimal Navigation & Trajectory-learning

A reinforcement learning framework for training autonomous fixed-wing UAVs to navigate waypoint missions while avoiding collisions in shared airspace. Built with Stable Baselines3 and Gymnasium, featuring curriculum learning for progressive difficulty scaling.

---

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Monitoring](#monitoring)

---

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

---

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

```bash
tensorboard --logdir=./training_logs
```

Open http://localhost:6006 to track episode rewards, value function estimates, policy entropy, and curriculum phase transitions in real time.