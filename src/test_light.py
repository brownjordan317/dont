import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from stable_baselines3 import A2C, PPO, SAC

# Import your custom environment modules
from gym_env import MultiUAVEnv
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position

console = Console()

def create_light_environment(config):
    env_cfg = config["test"]["env"]
    origin = [float(x) for x in env_cfg["origin"]]
    box_size = env_cfg["box_size"]
    
    uavs = []
    for id_tag, params in config["test"]["missions"].items():
        uav = FixedWingAircraft(
            id_tag=id_tag,
            initial_position=Position(*params["initial_position"]),
            initial_heading=params["initial_heading"],
            cruise_speed=params["cruise_speed"],
            turning_radius=params["turning_radius"],
            mission=list(params["waypoints"]),
        )
        uavs.append(uav)

    lat_off = (box_size / 2.0) / 111320.0
    lon_off = (box_size / 2.0) / (111320.0 * np.cos(np.radians(origin[0])))
    tl = (origin[0] + lat_off, origin[1] - lon_off)
    br = (origin[0] - lat_off, origin[1] + lon_off)

    return MultiUAVEnv(
        uavs, 
        tl=tl, 
        br=br, 
        dt=0.05, 
        mode=config["test"]["mode"],
        inference_mode=config["test"].get("inference_mode", False)
    )

def test(config):
    console.print(Panel.fit("[bold cyan]UAV Inference Engine (Light Mode)[/bold cyan]"))
    
    # Load Model
    algo_name = config["test"].get("algorithm", "SAC").upper()
    algo_map = {"SAC": SAC, "A2C": A2C, "PPO": PPO}
    model = algo_map[algo_name].load(
        config["test"]["model_path"],
        device=config["test"].get("device", "cpu")
    )
    
    # Setup Env
    env = create_light_environment(config)
    obs, _ = env.reset()
    
    max_steps = config["test"]["env"]["max_steps"]
    total_reward = 0
    step_count = 0
    done = False
    info = {"waypoints_hit": 0}

    # Simulation with Progress Bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[yellow]Simulating...", total=max_steps)

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Update progress bar
            progress.update(task, advance=1)

    # Final Summary Table or List
    console.print("\n" + "="*40)
    console.print(f"[bold green]Final Episode Results:[/bold green]")
    console.print(f" • Score: [bold white]{total_reward:.2f}[/bold white]")
    console.print(f" • Waypoints Hit: [bold cyan]{info.get('waypoints_hit', 0)}[/bold cyan]")
    console.print(f" • Completion: {'Terminated' if done else 'Timed Out'} at {step_count} steps")
    console.print("="*40 + "\n")

    env.close()

if __name__ == "__main__":
    # Load your config and call run_light_session(config) here
    pass