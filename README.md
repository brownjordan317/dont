# **D**econflicted **O**ptimal **N**avigation & **T**rajectory-learning (DON<sup><span style="font-size:0.5em">&amp;</span></sup>T)


A reinforcement learning framework for training autonomous fixed-wing UAVs to navigate waypoint missions while avoiding collisions in shared airspace. Built with Stable Baselines3 (A2C algorithm) and Gymnasium, featuring curriculum learning for progressive difficulty scaling.

---

## 🎯 Overview

This system trains UAV autopilots using **reinforcement learning** to accomplish complex missions:

- **Navigate** through sequential waypoint missions
- **Avoid collisions** with other aircraft in shared airspace
- **Respect geofence boundaries** (stay within operational areas)
- **Handle dynamic environments** with 1-5 simultaneous aircraft

The training uses **curriculum learning** that progressively increases difficulty by:
1. Expanding operational area (200m → 1500m boxes)
2. Adding more aircraft (1 → 5 UAVs)
3. Creating more complex scenarios over time

---

## ✨ Key Features

### Core Capabilities
- ✈️ **Realistic Fixed-Wing Physics**: Dubins-style path following with turning radius constraints
- 🎓 **Curriculum Learning**: 7-phase progression from simple to complex scenarios
- 🤝 **Multi-Agent Deconfliction**: Dynamic collision avoidance between multiple UAVs
- 🗺️ **Geographic Positioning**: Real-world lat/lon coordinates with meter-scale local transforms
- 🎯 **Flexible Missions**: Support for auto-generated or manually-defined waypoint sequences

### Technical Highlights
- Sequential waypoint arrival detection using geometric line segment checks
- Reward shaping for smooth, efficient flight paths
- Loiter mode when missions complete
- TensorBoard integration for training monitoring
- Matplotlib visualizations for testing

---

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
# Core RL and Environment
pip install -r requirements.txt
```
---

## 📁 Project Structure

### Main Entry Point
- **`deconfliction_factory.py`**: The main factory that reads configuration and launches training or testing

### Core Files
- **`test_config.yaml & train_config.yaml`**: All configuration settings (THIS IS WHAT YOU EDIT)
- **`train.py`**: Training loop with curriculum callback
- **`test.py`**: Model evaluation and visualization generation
- **`gym_env.py`**: Gymnasium environment implementing the RL interface
- **`deconfliction_factory.py`**: Handles the running of the train and test scripts

### Flight Engine (flight_engine/)
- **`simulator.py`**: FixedWingAircraft class with Dubins path following
- **`flight_calcs.py`**: FlightDynamics for turn calculations and motion
- **`helpers.py`**: Position dataclass, FlightMode enum, utility functions
- **`trans_coorders.py`**: Geographic ↔ Local coordinate transformations
- **`wp_manager.py`**: Waypoint queue and arrival logic
- **`visualizer.py`**: Matplotlib plotting for multi-UAV scenarios

### Utilities
- **`just_fly_a_path.py`**: Standalone physics test (no RL, just simulation)

---

## 🚀 Quick Start Guide

### Running the System

The system is controlled entirely through **`deconfliction_factory.py`** and the yaml files in the config directory.

#### Step 1: Choose Your Parameters

Edit `train_config.yaml` or `test_config.yaml` and set the parameters you want to use.

#### Step 2: Run the Factory

```bash
make train
# or
make test
```

That's it! The factory will:
1. Read the correct yaml file
2. Import the appropriate module (train or test)
3. Execute with your configuration

---

## ⚙️ Configuration Reference

All settings are in the yaml files. Here's what each section does:

### Training Configuration

#### Hyperparameters

```yaml
train:
  # Model Settings
  model_name: "a2c_uav_curriculum"  # Name for saved model
  learning_rate: 1e-4               # Adam optimizer learning rate
  max_grad_norm: 0.5                # Gradient clipping threshold
  ent_coeff: 0.01                   # Entropy bonus (exploration)
  vf_coeff: 0.25                    # Value function loss weight
  
  # Network Architecture
  policy_kwargs:
    net_arch: [512, 512]            # Two hidden layers, 512 neurons each
  
  # Logging
  tensorboard_log: "./training_logs/"  # TensorBoard output directory
  verbose: 0                           # 0=silent, 1=info, 2=debug
  device: "cpu"                        # "cpu" or "cuda"
  save_dir: "./models/"                # Where to save model checkpoints
```

#### Flight Simulator Settings

```yaml
  # Physics Parameters
  drone_speed: 25 # The speed of the drone
  speed_var: 3 # The random variations that the speed could change
  drone_turn_rate: 30 # The turn rate of the drone
  turn_var: 5 # The random variation that the turn rate could be off
  dt: 0.6                  # Simulation timestep in seconds (Essentially tunes the max travel distance per timestep)
  max_steps: 10_000        # Max steps per episode before reset
  
  # Mission Generation
  boundary_margin: 0.15    # Keep waypoints 15% away from geofence edges
  mission_waypoint_count: 3  # Number of waypoints maintained in queue
  caution_dist: 100 # This is the distance between each other that drones start becoming nervous
  critical_dist: 35 # This is the critical distance that will lead the system to a crash

```

**Important Notes**:
- **`dt`**: Larger values (0.6) = faster training but less realistic physics. Smaller values (0.1) = slower but more accurate.
- **`mission_waypoint_count`**: The system always maintains this many waypoints. When one is reached, a new one spawns.

#### Curriculum Learning

This is where the progressive difficulty is defined:

```yaml
  # Curriculum Control
  total_timesteps: 1_000_000    # Total training steps
  change_frequency: 5_000       # Steps between environment changes
  
  curriculum:
    0.00:  # At 0% progress (start of training)
      min_box_size: 200
      max_box_size: 250
      num_drones: 1
    
    0.10:  # At 10% progress (100k steps)
      min_box_size: 250
      max_box_size: 300
      num_drones: 1
    
    0.20:  # At 20% progress (200k steps)
      min_box_size: 300
      max_box_size: 400
      num_drones: 1
    
    0.35:  # At 35% progress (350k steps)
      min_box_size: 400
      max_box_size: 600
      num_drones: 2  # Second drone added!
    
    0.50:
      min_box_size: 600
      max_box_size: 800
      num_drones: 3
    
    0.65:
      min_box_size: 800
      max_box_size: 1000
      num_drones: 4
    
    0.80:  # At 80% progress (800k steps)
      min_box_size: 1000
      max_box_size: 1500
      num_drones: 5  # Maximum complexity!
```

**How It Works**:
- Every `change_frequency` steps (5000), a new random box is generated
- Box size sampled uniformly between `min_box_size` and `max_box_size`
- Drones are added/removed to match `num_drones` for current phase
- Phases are triggered based on training progress percentage

**Customization**:
- Start with smaller `total_timesteps` (e.g., 100,000) for testing
- Increase `change_frequency` (e.g., 10,000) for more stable learning per environment
- Add intermediate phases for smoother progression

---

### Testing Configuration

```yaml
test:
  # Model Loading
  model_path: "a2c_uav_curriculum_good.zip"  # Path to trained model
  test_name: "test_1"                        # Name for output files
  save_dir: "tests"                          # Output directory
  
  # Test Environment
  env:
    origin: [37.7749, -122.4194]  # San Francisco coordinates
    box_size: 1500                # Operational area size (meters)
    max_steps: 4000               # Maximum simulation steps
    dt: 0.05                      # Timestep (smaller = more accurate)
```

**Testing Notes**:
- **`dt`**: Use smaller values (0.05) for smoother, more realistic test visualizations
- **`origin`**: Can be any lat/lon on Earth
- **`box_size`**: Should match or exceed the largest training box

#### Manual Mission Definition

Define specific waypoint missions for each UAV:

```yaml
  missions:
    "UAV-1":
      initial_position: [37.7749, -122.4194]  # Starting lat/lon
      initial_heading: 45                     # Degrees (0=North, 90=East)
      cruise_speed: 25                        # m/s
      turning_radius: 60                      # meters
      waypoints:
        # List of [latitude, longitude] waypoints
        - [37.7755, -122.4188]  # First waypoint
        - [37.7762, -122.4182]  # Second waypoint
        - [37.7768, -122.4176]  # etc...
        # ... can have as many as you want
    
    "UAV-2":
      initial_position: [37.7742, -122.4190]
      initial_heading: 270  # Heading west
      cruise_speed: 20
      turning_radius: 60
      waypoints:
        - [37.7742, -122.4182]
        - [37.7742, -122.4172]
        # ... more waypoints ...
```

**Mission Design Tips**:
- Each UAV gets its own complete mission
- Waypoints are followed in sequence
- Make sure waypoints are within the `box_size` boundary
- Test for potential collision points between UAVs

---

## 🎓 Training Mode

### What Happens During Training

The system will:
1. **Initialize** a single UAV at a random global location
2. **Create** a small operational area (200-250m)
3. **Generate** random waypoint missions continuously
4. **Train** the policy using the Gymnasium environment
5. **Progress** through curriculum phases automatically
6. **Save** model checkpoints at each phase transition
7. **Display** rich console output showing progress

### Console Output

You'll see colored panels showing:

```
╭─────────────────────────────────╮
│ Starting Training Mode          │
╰─────────────────────────────────╯

✓ Config loaded successfully from factory.yaml

╭─────────────────────────────────╮
│ Curriculum Update               │
│ Step: 5000                      │
│ Area: 237m × 244m               │
│ UAVs: 1                         │
╰─────────────────────────────────╯

╭─────────────────────────────────╮
│ UAV ADDED                       │
│ Total UAVs: 2                   │
╰─────────────────────────────────╯

╭─────────────────────────────────╮
│ MODEL SAVED                     │
│ Phase: 3                        │
│ Progress ≥ 35%                  │
│ Saved to:                       │
│ ./models/a2c_phase_3_step_...   │
╰─────────────────────────────────╯
```

### Saved Files

Training creates:

```
models/
├── a2c_phase_0_step_0.zip         # Phase checkpoints
├── a2c_phase_1_step_100000.zip
├── a2c_phase_2_step_200000.zip
├── ...
└── a2c_uav_curriculum.zip         # Final trained model

training_logs/
└── a2c_uav_curriculum_1/
    └── events.out.tfevents...     # TensorBoard logs
```

### Monitoring with TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir=./training_logs
```

Then open http://localhost:6006 in your browser to see:
- Episode rewards over time
- Value function estimates
- Policy entropy
- And more metrics

### Interrupting Training

Press `Ctrl+C` to stop training. The model will be saved as:
```
a2c_uav_curriculum_interrupted.zip
```

You can resume training by loading this model and continuing.

---

## 🧪 Testing Mode

### Running Tests

1. **Ensure you have a trained model**:
   - Train one yourself, OR
   - Download a pre-trained model

2. **Edit `test_config.yaml`**:
   ```yaml
   mode: "test"
   test:
     model_path: "a2c_uav_curriculum_good.zip"  # Your model file
   ```

3. **Define test missions** in the `missions` section (see Configuration Reference)

4. **Run**:
   ```bash
   make test
   ```

### What Happens During Testing

1. **Loads** the trained model from disk
2. **Creates** the test environment with your specified origin and box size
3. **Initializes** UAVs at specified positions with defined waypoint missions
4. **Runs** the simulation using the trained policy (deterministic mode)
5. **Records** all positions, waypoints, and performance metrics
6. **Generates** high-quality visualization plots
7. **Saves** results to the `save_dir`

### Test Output

```
reports/
└── test_1.png  # Visualization with flight paths, waypoints, and metrics
```

### Understanding Test Visualizations

The generated plot shows:

- **Flight Paths**: Colored lines showing each UAV's trajectory
- **Start Positions**: Circles marking where each UAV began
- **End Positions**: Squares showing final positions
- **Waypoints**: X markers showing all waypoints that existed
- **Geofence**: Red dashed box showing operational boundary
- **Metrics**: Title includes steps taken, total reward, and waypoint arrivals

**What to Look For**:
- ✅ Smooth, curved flight paths (realistic fixed-wing behavior)
- ✅ Waypoints being reached in sequence
- ✅ Aircraft maintaining separation (collision avoidance working)
- ✅ All paths staying within the geofence
- ❌ Jerky paths indicate poor policy or too large `dt`
- ❌ Collisions or near-misses indicate insufficient training
- ❌ Geofence violations indicate boundary penalty needs tuning

---
## 🧪 Physics Testing

Use `just_fly_a_path.py` to test physics without RL:

### What It Does

Creates 3 UAVs with different specs and random waypoint missions, then:
1. Simulates pure physics (no RL policy)
2. Generates frame-by-frame visualizations
3. Saves final trajectory plot

### Running Physics Test

```bash
python just_fly_a_path.py
```

### Output

```
physics_frames/
├── step_0000.png
├── step_0005.png
├── step_0010.png
├── ...
└── step_1500.png

final_physics_test.png  # Complete trajectory visualization
```

### Use Cases

- **Verify** turning radius and cruise speed are realistic
- **Debug** waypoint arrival detection
- **Validate** coordinate transformations
- **Test** physics before integrating with RL
- **Create animations** by combining frames into GIF/video