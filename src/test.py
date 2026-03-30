import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import A2C, PPO, SAC
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
import os

from gym_env import MultiUAVEnv
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position
from config_utils import get_tuning_section
from vec_env_utils import make_eval_env, reset_rollout_env, step_rollout_env, unwrap_env

console = Console()


def _heading_to_radians(value):
    heading = float(value)
    if abs(heading) > (2 * np.pi):
        return np.deg2rad(heading)
    return heading

# ================================================================
# ENV CREATION
# ================================================================

def create_test_environment(scenario, origin, config, inference_mode=True):
    """
    Create test environment
    
    Args:
        inference_mode: If True, skips reward calculations for faster inference
    """
    tuning = get_tuning_section(config, "test")
    env_cfg = tuning["env"]
    reward_cfg = tuning["rewards"]
    hard_safety_cfg = tuning["hard_safety"]
    anti_circling_cfg = tuning["anti_circling"]
    box_size = scenario["box_size"]
    uavs = []
    for id, params in config["test"]["missions"].items():
        uav = FixedWingAircraft(
            id_tag=id,
            initial_position=Position(
                params["initial_position"][0], 
                params["initial_position"][1]
            ),
            initial_heading=_heading_to_radians(params["initial_heading"]),
            cruise_speed=params["cruise_speed"],
            turning_radius=params["turning_radius"],
            mission=list(params["waypoints"]),
        )
        uavs.append(uav)

    lat_off = (box_size / 2.0) / 111_320.0
    lon_off = (box_size / 2.0) / (111_320.0 * np.cos(np.radians(origin[0])))
    tl = (origin[0] + lat_off, origin[1] - lon_off)
    br = (origin[0] - lat_off, origin[1] + lon_off)

    return MultiUAVEnv(
        uavs, 
        tl=tl, br=br, 
        dt=env_cfg["dt"],
        max_steps=env_cfg["max_steps"],
        mode=env_cfg["mode"],
        caution_dist=env_cfg["caution_dist"],
        critical_dist=env_cfg["critical_dist"],
        inference_mode=inference_mode,
        reward_config=reward_cfg,
        hard_safety_config=hard_safety_cfg,
        anti_circling_config=anti_circling_cfg,
        ), tl, br

# ================================================================
# EPISODE RECORDING (for video generation)
# ================================================================

def run_and_record_episode(model, env, transformer, max_steps):
    base_env = unwrap_env(env)
    obs = reset_rollout_env(env)

    uav_data = [{'id': ac.id_tag, 'positions': [], 'headings': [], 
                 'waypoints_visited': [], 'all_waypoints': [], 
                 'current_targets': []} for ac in base_env.aircraft_list]

    done = False
    step = 0
    total_reward = 0
    episode_metrics = None

    while not done and step < max_steps:
        for i, ac in enumerate(base_env.aircraft_list):
            pos = ac.position.to_tuple()
            x, y = transformer.geo_to_local(pos[0], pos[1])
            uav_data[i]['positions'].append((x, y))
            uav_data[i]['headings'].append(ac.heading)

            wp_obj = ac.waypoint_manager.current_waypoint
            if wp_obj:
                wp = wp_obj.to_tuple()
                wp_x, wp_y = transformer.geo_to_local(wp[0], wp[1])
                uav_data[i]['current_targets'].append((wp_x, wp_y))
                if (wp_x, wp_y) not in uav_data[i]['all_waypoints']:
                    uav_data[i]['all_waypoints'].append((wp_x, wp_y))
            else:
                uav_data[i]['current_targets'].append(None)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = step_rollout_env(env, action)
        if done:
            episode_metrics = info.get("episode_metrics")
        # Faster check for waypoint hits
        for i, ac in enumerate(base_env.aircraft_list):
            if getattr(ac, 'last_waypoint_hit_pos', None):
                h_x, h_y = transformer.geo_to_local(*ac.last_waypoint_hit_pos)
                uav_data[i]['waypoints_visited'].append((h_x, h_y))
                ac.last_waypoint_hit_pos = None

        total_reward += reward
        step += 1

    if episode_metrics is None:
        episode_metrics = base_env.get_uav_metrics()

    return uav_data, step, total_reward, info["waypoints_hit"], episode_metrics

# ================================================================
# LIGHT MODE (no video, just statistics)
# ================================================================

def run_light_episode(model, env, max_steps, model_cls):
    """Run episode without recording positions - just get metrics"""
    base_env = unwrap_env(env)
    obs = reset_rollout_env(env)
    
    total_reward = 0
    step_count = 0
    done = False
    info = {"waypoints_hit": 0}
    episode_metrics = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[yellow]Simulating {model_cls}...", total=max_steps)

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = step_rollout_env(env, action)
            if done:
                episode_metrics = info.get("episode_metrics")
            
            total_reward += reward
            step_count += 1
            
            progress.update(task, advance=1)
        
        if not done:
            truncated = True
        else:
            truncated = False

    if episode_metrics is None:
        episode_metrics = base_env.get_uav_metrics()
    
    return (
        step_count,
        total_reward,
        info.get('waypoints_hit', 0),
        done,
        truncated,
        episode_metrics,
    )

# ================================================================
# VIDEO GENERATION (Optimized with Blitting & Pre-calc)
# ================================================================

def create_video(
        uav_data, 
        tl, br, 
        transformer, 
        scenario_name, _, 
        arrivals, save_path, 
        config, fps=30, speed_multiplier=1
    ):
    console.print("[cyan]Creating video...[/cyan]")
    env_cfg = get_tuning_section(config, "test")["env"]
    tl_x, tl_y = transformer.geo_to_local(tl[0], tl[1])
    br_x, br_y = transformer.geo_to_local(br[0], br[1])

    max_steps = max(len(data['positions']) for data in uav_data)
    frame_indices = list(range(0, max_steps, max(1, int(speed_multiplier))))
    total_frames = len(frame_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, len(uav_data)))
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.add_patch(
        Rectangle(
            (tl_x, br_y), 
            br_x - tl_x, tl_y - br_y, 
            linewidth=2, edgecolor='red', 
            facecolor='none', linestyle='--'
            )
        )
    
    uav_artists = []
    trail_len = 120
    # Pre-calculate alpha arrays for performance
    alphas = np.linspace(0.02, 0.7, trail_len)

    for i, data in enumerate(uav_data):
        color = colors[i]
        line, = ax.plot(
            [], [], color=color, alpha=0.10, linewidth=1, animated=True
        )
        marker, = ax.plot(
            [], [], marker='o', markersize=6, color=color, 
            markeredgecolor='black', zorder=15, animated=True
        )
        trail = ax.scatter(
            [], [], s=20, zorder=10, animated=True
        )
        
        c30 = Circle(
            (0, 0), env_cfg["caution_dist"], color=color, 
            fill=False, linestyle='--', alpha=0.35, animated=True
        )
        c5 = Circle(
            (0, 0), env_cfg["critical_dist"], color=color, 
            fill=True, alpha=0.2, animated=True
        )
        glow = Circle(
            (0, 0), 0, color=color, alpha=0.0, zorder=5, 
            linewidth=0, animated=True
        )

        # Using Quiver for arrows is much faster than recreating ax.arrow
        quiver = ax.quiver(
            [0], [0], [0], [0], color=color, scale=1, scale_units='xy', 
            width=0.005, headwidth=4, zorder=20, animated=True
        )

        ax.add_patch(c30); ax.add_patch(c5); ax.add_patch(glow)

        uav_artists.append({
            'positions': np.array(data['positions']),
            'headings': np.array(data['headings']),
            'current_targets': data['current_targets'],
            'line': line, 'marker': marker, 'trail': trail,
            'c30': c30, 'c5': c5, 'glow': glow, 'quiver': quiver,
            'color_rgb': mcolors.to_rgb(color)
        })

    ax.set_xlim(tl_x - 100, br_x + 100)
    ax.set_ylim(br_y - 100, tl_y + 100)
    ax.set_aspect('equal')
    title_text = ax.text(
        0.5, 1.05, '', transform=ax.transAxes, ha='center', 
        fontweight='bold', animated=True
    )

    def update(frame_num):
        step = frame_indices[frame_num]
        title_text.set_text(
            f"{scenario_name}\nStep: {step}/{max_steps} | Arrivals: {arrivals}"
        )
        changed_artists = [title_text]

        for art in uav_artists:
            if step >= len(art['positions']): continue
            pos = art['positions'][step]
            
            # Update Artists
            art['marker'].set_data([pos[0]], [pos[1]])
            art['line'].set_data(
                art['positions'][:step+1, 0], 
                art['positions'][:step+1, 1]
            )
            art['c30'].set_center(pos); art['c5'].set_center(pos)

            # Trail Logic (Optimized RGBA slicing)
            start_idx = max(0, step - trail_len)
            trail_pts = art['positions'][start_idx:step]
            if len(trail_pts) > 0:
                art['trail'].set_offsets(trail_pts)
                rgba = np.zeros((len(trail_pts), 4))
                rgba[:, :3] = art['color_rgb']
                rgba[:, 3] = alphas[-len(trail_pts):] # Match alpha length to points
                art['trail'].set_facecolors(rgba)

            # Heading Arrow (Quiver update)
            heading = art['headings'][step]
            art['quiver'].set_offsets(pos)
            art['quiver'].set_UVC(40 * np.sin(heading), 40 * np.cos(heading))

            # Pulse Logic
            target = art['current_targets'][step]
            if target:
                rad_base = env_cfg["wp_hit_radius"]
                pulse = (rad_base * 0.7) + (rad_base * 0.3) * (0.5 * (1 + np.sin(frame_num * 0.05)))
                art['glow'].set_center(target)
                art['glow'].set_radius(pulse)
                art['glow'].set_alpha(0.25 * (pulse / rad_base))
            else:
                art['glow'].set_alpha(0.0)

            changed_artists.extend(
                [
                    art['marker'], art['line'], 
                    art['trail'], art['c30'], 
                    art['c5'], art['glow'], 
                    art['quiver']
                    ]
            )
        
        return changed_artists

    with Progress(
        SpinnerColumn(), 
        TextColumn("{task.description}"), 
        BarColumn(), 
        TimeRemainingColumn(), 
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Rendering...", total=total_frames)
        
        # blit=True is the key to speed
        anim = FuncAnimation(
            fig, update, frames=total_frames, 
            interval=1000/fps, blit=True
        )
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        
        # Hook into writer to update progress bar without slowing down render
        original_grab = writer.grab_frame
        def grab_with_progress(*args, **kwargs):
            original_grab(*args, **kwargs)
            progress.update(task, advance=1)
        writer.grab_frame = grab_with_progress
        
        anim.save(save_path, writer=writer)
    plt.close()

# ================================================================
# MAIN TEST FUNCTION
# ================================================================

def test(config):
    tuning = get_tuning_section(config, "test")
    env_cfg = tuning["env"]
    # Determine mode (light or full video)
    light_mode = not config["test"].get("save_visuals", False)
    light_mode = False
    if light_mode:
        console.print(Panel.fit("[bold cyan]UAV Inference Engine (Light Mode)[/bold cyan]"))
    else:
        console.print(Panel.fit("[bold white]Flight Path Visualizer[/bold white]"))
    
    os.makedirs(config["test"]["save_dir"], exist_ok=True)
    
    origin = [float(x) for x in env_cfg["origin"]]
    scenario_info = {
        "name": config["test"]["test_name"], 
        "box_size": env_cfg["box_size"]
    }
    
    # Create environment with inference mode enabled
    raw_env, tl, br = create_test_environment(
        scenario_info, 
        origin, 
        config, 
        inference_mode=config["test"]["inference_mode"]
    )
    env, stats_path = make_eval_env(
        lambda raw_env=raw_env: raw_env,
        model_path=config["test"]["model_path"],
        vecnormalize_path=config["test"].get("vecnormalize_path"),
        norm_reward=False,
    )
    base_env = unwrap_env(env)

    # Load model
    algo_map = {"SAC": SAC, "A2C": A2C, "PPO": PPO}
    model_cls = algo_map.get(config["test"].get("algorithm", "SAC").upper(), SAC)
    model = model_cls.load(
        config["test"]["model_path"],
        env=env,
        device=config["test"].get("device", "cpu")
    )
    max_steps = env_cfg["max_steps"]
    
    base_env.clear_missions = False

    if stats_path:
        console.print(f"[green]Loaded VecNormalize stats:[/green] {stats_path}")
    else:
        console.print("[yellow]No VecNormalize stats found; using raw observations.[/yellow]")

    if light_mode:
        # Light mode - just run and get stats
        steps, total_reward, arrivals, done, truncated, metrics = run_light_episode(model, env, max_steps, model_cls)
        
        # Display results
        console.print("\n" + "="*50)
        console.print(f"[bold green]Episode Results:[/bold green]")
        console.print(f" • Total Reward: [bold white]{total_reward:.2f}[/bold white]")
        console.print(f" • Waypoints Hit: [bold cyan]{arrivals}[/bold cyan]")
        console.print(f" • Steps: [bold yellow]{steps}[/bold yellow]")
        console.print(f" • Status: [bold]{'Completed' if done else 'Timed Out'}[/bold]")
        console.print(f" • Caution Violations: [yellow]{metrics['safety_violations']['caution']['total_count']}[/yellow]")
        console.print(f" • Critical Violations: [red]{metrics['safety_violations']['critical']['total_count']}[/red]")
        console.print(f" • Geofence Violations: [red]{metrics['safety_violations']['geofence']['total_count']}[/red]")
        console.print(f" • Hard Safety Uses: [magenta]{metrics['safety_violations']['hard_safety']['total_count']}[/magenta]")
        console.print(f" • Anti-Circling Uses: [cyan]{metrics['safety_violations']['anti_circling']['total_count']}[/cyan]")
        
        # Per-UAV distance stats
        console.print(f"\n[bold cyan]Distance Traveled per UAV:[/bold cyan]")
        for stat in metrics['mission_stats']:
            dist_m = stat['dist_navigating']
            console.print(f" • {stat['id']}: [green]{dist_m:.2f} m[/green] ({stat['waypoints_reached']} waypoints)")
        
        console.print("="*50 + "\n")
        
    else:
        # Full mode - record positions and create video
        uav_data, steps, total_reward, arrivals, metrics = run_and_record_episode(
            model, env, base_env.transformer, max_steps
        )
        
        console.print(f"\n[bold green]Episode Finished[/bold green]")
        console.print(f"Total Reward: {total_reward:.2f}")
        console.print(f"Waypoint Arrivals: {arrivals}")
        console.print(f"Steps: {steps}")
        console.print(f"Caution Violations: {metrics['safety_violations']['caution']['total_count']}")
        console.print(f"Critical Violations: {metrics['safety_violations']['critical']['total_count']}")
        console.print(f"Hard Safety Uses: {metrics['safety_violations']['hard_safety']['total_count']}")
        console.print(f"Anti-Circling Uses: {metrics['safety_violations']['anti_circling']['total_count']}")
        
        # Per-UAV distance stats
        console.print(f"\n[bold cyan]Distance Traveled:[/bold cyan]")
        for stat in metrics['mission_stats']:
            dist_km = stat['dist_navigating'] / 1000.0
            console.print(f" • {stat['id']}: {dist_km:.2f} km ({stat['waypoints_reached']} waypoints)")
        console.print()
        
        if config["test"].get("create_video", True):
            vid_name = f"{config['test']['test_name'].replace(' ', '_')}.mp4"
            create_video(
                uav_data, tl, br, base_env.transformer, config["test"]["test_name"], 0, arrivals, 
                os.path.join(config["test"]["save_dir"], vid_name), config, 
                fps=config["test"].get("video_fps", 30), 
                speed_multiplier=config["test"].get("video_speed", 1)
            )
    
    env.close()
