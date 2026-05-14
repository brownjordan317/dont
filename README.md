# DON&T HRL- Deconflicted Optimal Navigation & Trajectory Hierachical Reinforcement Learning

DON&T HRL trains fixed-wing UAV policies for waypoint navigation, collision avoidance, and geofence-aware multi-agent flight. The current system is hierarchical: low-level policies learn specialized continuous turn-rate behavior, while a high-level manager selects which low-level behavior should control each aircraft during a mission.

The environment models coordinated fixed-wing aircraft in randomized rectangular flight boxes. Aircraft must complete waypoint missions, maintain separation, avoid geofence breaches, and handle conflict cases where several vehicles are simultaneously competing for safe maneuvering space. Training is intentionally stationary; scenario variation comes from the sampling ranges in `config/tuning.yaml` rather than a curriculum schedule.

## Project Status

This project is still a work in progress. The current results show useful pieces of behavior, especially in route following and short-horizon avoidance, but the full hierarchical system is not yet at the level I want. The individual skills are still inconsistent, the avoid skill has a high geofence failure rate, and the manager is limited by the quality of the skills it composes. I am not satisfied with the current performance and plan to keep improving the skill training, boundary behavior, anti-circling recovery, and manager handoff logic.

## Changes Since v2

DON&T HRL now uses a hierarchical policy layout instead of one shared MAPPO policy for every behavior. The route and avoid skills are trained as separate continuous-control policies, and the manager learns when to select each pretrained skill during mission execution.

Route training now emphasizes curved waypoint tracking through Dubins-style reference guidance and warm-start imitation before PPO. Avoid training is waypoint-free and focused on local survival, collision avoidance, geofence compliance, and avoiding degenerate circling behavior. Manager training composes those two skills through categorical skill selection rather than direct turn-rate control.

The environment and configs have also been simplified around the active training modes. Old inactive curriculum/recoverability paths were removed, per-skill evaluation is supported, and safety failures such as geofence breaches are terminal by default.

## Architecture

The project is organized around three trainable roles:

- `route_skill`: continuous waypoint-following policy
- `avoid_skill`: continuous local safety policy
- `manager`: categorical policy that selects between the pretrained route and avoid skills

All roles use the same underlying PettingZoo-style parallel environment and MAPPO implementation. The route and avoid skills are trained independently first, then loaded by the manager so it can learn skill selection instead of directly learning low-level turn-rate control.

## Skill Roles

The `route_skill` is a continuous-control waypoint follower. For each aircraft, it receives the standard flight observation, current waypoint geometry, Dubins-style reference-route features, nearby traffic features, and geofence context. It outputs one normalized turn command that is converted into a fixed-wing turn rate by the base environment. The route skill is trained on waypoint missions and optimized for waypoint completion, route progress, bounded turning, and geofence compliance. It is warm-started from a reference Dubins controller before PPO so the actor begins with a usable curved-path tracking prior.

The `avoid_skill` is a continuous-control safety policy. It uses the same action interface as the route skill, but its training environment removes the waypoint objective and focuses on close-quarters survival. Aircraft fly in waypoint-free conflict scenarios where collision, geofence breach, and degenerate survival motion can terminate the episode. The reward emphasizes remaining alive, avoiding critical separation failures, staying inside the geofence, and avoiding synchronized circling patterns that satisfy survival without producing useful evasive behavior. When used by the manager, waypoint-related inputs are neutralized so the avoid policy behaves as a local safety maneuver rather than a second route follower.

The `manager` is a categorical high-level policy over the pretrained low-level skills. It does not output turn rates; it selects `route_skill` or `avoid_skill` for each aircraft at each manager decision step. Its observation is the base flight observation augmented with manager-specific features, including the route skill action, avoid skill action, neighbor avoidance pressure, boundary pressure, current avoidance pressure, avoid-option state, avoid handoff safety, and avoid hold progress. Sticky avoid logic enforces minimum avoid durations and safe handoff conditions, while hard-coded breakout guards can temporarily force route-follow execution when avoid remains active in a low-threat circling loop. Forced override steps are masked out of PPO training.

## Behavioral Demo Vids and Descriptions
In the exampls folder I have some videos demoing different behaviors. 
- `examples/skill_eval/avoid/avoid_mission_001.mp4`: This video shows a breif demo of the avoid skill. This skill has been the biggest struggle for me. I have been relatively unsuccesful in training the system to properly handle deconfliction. I can get it to survive and show signs of "dodging". However, results are consistently inconsistent. The system surives much longer than random movements would, but the odds of a long term survival are very low. On the bright side, it works well enough to make safe movements in large more realistic environments. Either way, i wouldnt trust it to fly something in the real world.
- `examples/skill_eval/avoid/route_mission_001.mp4`: This video demos the route skill. If I am being completely honest with myself, this skill would probably be better as a hardcoded ability. Within the video, the dotted predrawn lines is a hard calculated optimal path for comparison. As can be seen, just manually taking control of the drone is practically gauranteed to be better. However, I would still call the training of this skill a relative success. The primary thing I wanted for the route skill was to learn dubins like paths rather than just point to point flight. 
- `examples/No_circle_fix_no_decon.mp4`: This video demonstrates a reason why I believe the route skill is better of without RL. The drones begin the test with good flight patterns. However, the drones often get stuck in tight circling patterns around the waypoints. While this is something that is technically fixable with RL, it is far easier to hardcode breakouts from that behavior. Furthermore, just hardcoding would be able to completely prevent the circling in the first place.
- `examples/no_decon.mp4`: This video demonstrates the rl route skill being aided by a hard coded loiter breakout. The hard coded loiter breakout allows the missions to actually continue to completion rather than timing out.

## Environment Behavior

Episodes are bounded by mission completion, safety failures, and timeout. Collision-critical separation failures and geofence breaches are terminal by default. The environment also tracks behavior diagnostics such as waypoint throughput, geofence outside steps, minimum pairwise separation, deconfliction time, flown distance, circling breakouts, and waypoint reapproach events.

Waypoint following uses fixed-wing dynamics, bounded turn rates, configurable turn response lag, and Dubins-style reference guidance. Anti-circling assistance is intentionally implemented as hard-coded guard behavior: if an aircraft stalls around a waypoint or the manager keeps an aircraft in avoid during a low-threat loop, the environment can force a route-follow/reapproach interval and exclude that forced action from actor training.

## Skill Evaluation Results

The latest skill-only evaluation used 100 randomized route-skill missions and 100 randomized avoid-skill survival episodes. Visual export was disabled for this run. Results are reported as mean +/- standard deviation across episodes.

### Route Skill

| Metric | Result |
| --- | ---: |
| Reward | 1778.11 +/- 2484.62 |
| Episode steps | 737.96 +/- 456.75 |
| Episode duration | 221.39 +/- 137.02 s |
| Reward per step | 1.84 +/- 6.72 |
| Distance per UAV | 5901.28 +/- 4883.46 m |
| Waypoint completion rate | 92.00% +/- 18.92% |
| Waypoint throughput | 3.03 +/- 1.13/min |
| Mission completion rate | 79.00% +/- 40.73% |
| Failure rate | 21.00% +/- 40.73% |
| Crash rate | 0.00% +/- 0.00% |
| Timeout rate | 2.00% +/- 14.00% |
| Boundary compliance | 81.00% +/- 39.23% |
| Geofence exits | 0.19 +/- 0.39 |
| Geofence outside steps | 0.19 +/- 0.39 |
| Distance / planned distance | 1.19 +/- 0.99 |

The route skill completed 79 of 100 missions, with 19 geofence violations and 2 max-step timeouts. Its waypoint completion rate is higher than its full mission completion rate, which means most failed episodes still made substantial route progress before termination. The distance-to-planned ratio of 1.19 suggests that the policy usually follows a reasonably efficient curved route, but the large standard deviation in reward, steps, distance, and distance ratio shows that behavior is still inconsistent across sampled missions. The main weakness is boundary handling: 19% of episodes left the geofence despite no collision failures.

### Avoid Skill

| Metric | Result |
| --- | ---: |
| Reward | -442.44 +/- 798.27 |
| Episode steps | 325.56 +/- 169.64 |
| Episode duration | 8.14 +/- 4.24 s |
| Reward per step | -2.71 +/- 4.96 |
| Distance per UAV | 227.72 +/- 117.76 m |
| Survival completion rate | 17.00% +/- 37.56% |
| Failure rate | 100.00% +/- 0.00% |
| Crash rate | 11.00% +/- 31.29% |
| Timeout rate | 0.00% +/- 0.00% |
| Boundary compliance | 28.00% +/- 44.90% |
| Geofence exits | 0.72 +/- 0.45 |
| Geofence outside steps | 0.72 +/- 0.45 |
| Minimum pairwise separation | 22.46 +/- 8.70 m |

The avoid skill produced 17 survival completions, 72 geofence violations, and 11 critical separation violations. These results indicate that the policy often avoids immediate collisions better than random motion would, but it does not reliably keep the aircraft inside the operating box. The minimum pairwise separation average of 22.46 m is above the critical threshold in many episodes, so the dominant failure mode is boundary escape rather than direct collision. The high geofence violation rate also explains the negative mean reward: the skill has learned some reactive avoidance behavior, but it is not yet a robust standalone deconfliction policy.

### Manager

The manager is currently weak because it is composing two low-level skills that are not yet reliable enough on their own. In principle, the manager should choose `route_skill` when an aircraft can safely make mission progress and choose `avoid_skill` when local traffic or boundary pressure makes route following unsafe. That only works cleanly if each skill has a dependable meaning. Right now, `route_skill` usually makes waypoint progress but still fails a meaningful number of episodes through geofence violations, while `avoid_skill` shows short-horizon evasive behavior but frequently survives by drifting out of bounds or entering low-progress behavior.

This makes the manager's training signal messy. If the manager selects route, the aircraft may advance toward the mission but inherit route's boundary and circling problems. If it selects avoid, the aircraft may reduce immediate collision pressure but create a new boundary failure or stay in avoid longer than useful. The result is that the manager is not simply learning "when to switch"; it is also being forced to compensate for two imperfect controllers with different failure modes.

The current manager performance should therefore be interpreted as a system-level limitation, not just a manager-policy failure. Better route geofence compliance, more reliable avoid behavior near boundaries, and stronger anti-circling recovery would make the manager's action space much cleaner. Until then, the hard-coded sticky-avoid and breakout guards are necessary because the learned hierarchy does not consistently recover from those cases by itself.

## Configuration

The main configuration files are:

- `config/tuning.yaml`: shared environment, flight, reward, and guidance settings
- `config/train_route_skill.yaml`: route-skill PPO and warm-start settings
- `config/train_avoid_skill.yaml`: avoid-skill survival training settings
- `config/train_manager.yaml`: manager PPO settings and low-level skill checkpoint paths
- `config/train.yaml`: default manager-training entrypoint
- `config/test.yaml`: test rollout, mission, checkpoint, and video settings
- `config/eval.yaml`: randomized evaluation settings

Training and evaluation modes are dispatched through `src/deconfliction_factory.py`.

## Repository Layout

```text
src/
├── deconfliction_factory.py      # CLI mode dispatcher
├── config_utils.py               # YAML loading and mode-specific config merge helpers
├── inference_setup.py            # Shared test/eval/runtime environment setup
├── trained_policy_runtime.py     # Runtime wrapper for live waypoint updates
├── env/                          # Base environment, HRL wrappers, rewards, scenarios
├── flight_engine/                # Fixed-wing dynamics, waypoints, geometry, Dubins paths
├── mappo/                        # MAPPO policy, runtime helpers, rollout utilities
├── train/                        # Training loop and PPO update utilities
├── test/                         # Test rollout and visualization code
└── eval/                         # Evaluation runners and result aggregation

config/
├── tuning.yaml
├── train.yaml
├── train_route_skill.yaml
├── train_avoid_skill.yaml
├── train_manager.yaml
├── test.yaml
└── eval.yaml
```

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Common entrypoints:

```bash
make train_route_skill
make train_avoid_skill
make train_manager
make train
make test
make eval
make eval_skills
```

Equivalent direct form:

```bash
python src/deconfliction_factory.py --mode train
python src/deconfliction_factory.py --mode test
python src/deconfliction_factory.py --mode eval
```

## Outputs

Training checkpoints are written to the save directory configured by the active training file. TensorBoard logs are written under the configured tensorboard log directory, usually inside the corresponding model directory. Test videos and rollout plots are written under `reports/` when visual output is enabled. Evaluation summaries are written under `results/`.

## Acknowledgment

OpenAI Codex was used as a development assistant for this project, including help with implementation, debugging, refactoring, experiment analysis, and documentation.
