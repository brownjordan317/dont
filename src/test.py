import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import A2C
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import os
import yaml
import sys

from gym_env import MultiUAVEnv
from flight_engine.simulator import FixedWingAircraft
from flight_engine.helpers import Position

console = Console()

# ================================================================
# ENV CREATION
# ================================================================

def create_test_environment(scenario, origin, config):
    box_size = scenario["box_size"]

    uavs = []
    for id, params in config["test"]["missions"].items():
        uav = FixedWingAircraft(
            id_tag=id,
            initial_position=Position(
                params["initial_position"][0],
                params["initial_position"][1]
            ),
            initial_heading=params["initial_heading"],
            cruise_speed=params["cruise_speed"],
            turning_radius=params["turning_radius"],
            mission=list(params["waypoints"]),
        )
        uavs.append(uav)

    lat_off = (box_size / 2.0) / 111_320.0
    lon_off = (box_size / 2.0) / (111_320.0 * np.cos(np.radians(origin[0])))

    tl = (origin[0] + lat_off, origin[1] - lon_off)
    br = (origin[0] - lat_off, origin[1] + lon_off)

    env = MultiUAVEnv(
        uavs,
        tl=tl,
        br=br,
        dt=0.05,
        mode=config["test"]["mode"]
    )

    return env, tl, br

# ================================================================
# EPISODE RECORDING
# ================================================================

def run_and_record_episode(model, env, transformer, max_steps):
    obs, _ = env.reset()

    uav_data = []
    for ac in env.aircraft_list:
        uav_data.append({
            'id': ac.id_tag,
            'positions': [],
            'headings': [],
            'waypoints_visited': [],
            'all_waypoints': [],
            'current_targets': []
        })

    done = False
    truncated = False
    step = 0
    total_reward = 0

    while not (done or truncated) and step < max_steps:
        for i, ac in enumerate(env.aircraft_list):
            pos = ac.position.to_tuple()
            x, y = transformer.geo_to_local(pos[0], pos[1])

            uav_data[i]['positions'].append((x, y))
            uav_data[i]['headings'].append(ac.heading)

            if ac.waypoint_manager.current_waypoint:
                wp = ac.waypoint_manager.current_waypoint.to_tuple()
                wp_x, wp_y = transformer.geo_to_local(wp[0], wp[1])

                uav_data[i]['current_targets'].append((wp_x, wp_y))

                if (wp_x, wp_y) not in uav_data[i]['all_waypoints']:
                    uav_data[i]['all_waypoints'].append((wp_x, wp_y))
            else:
                uav_data[i]['current_targets'].append(None)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        for i, ac in enumerate(env.aircraft_list):
            if hasattr(ac, 'last_waypoint_hit_pos') and ac.last_waypoint_hit_pos:
                h_x, h_y = transformer.geo_to_local(
                    ac.last_waypoint_hit_pos[0],
                    ac.last_waypoint_hit_pos[1]
                )
                uav_data[i]['waypoints_visited'].append((h_x, h_y))
                ac.last_waypoint_hit_pos = None

        total_reward += reward
        step += 1

    return uav_data, step, total_reward, info["waypoints_hit"]

# ================================================================
# VIDEO WITH GLOW + FADE + SHAPE CHANGE
# ================================================================

def create_video(uav_data, tl, br, transformer,
                 scenario_name, total_reward,
                 arrivals, save_path,
                 fps=30, speed_multiplier=1):

    console.print("[cyan]Creating video...[/cyan]")

    tl_x, tl_y = transformer.geo_to_local(tl[0], tl[1])
    br_x, br_y = transformer.geo_to_local(br[0], br[1])

    max_steps = max(len(data['positions']) for data in uav_data)
    frame_indices = list(range(0, max_steps, max(1, int(speed_multiplier))))
    total_frames = len(frame_indices)

    colors = plt.cm.tab10(np.linspace(0, 1, len(uav_data)))
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    # Geofence
    ax.add_patch(Rectangle(
        (tl_x, br_y),
        br_x - tl_x,
        tl_y - br_y,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        linestyle='--'
    ))

    uav_artists = []

    for i, data in enumerate(uav_data):
        color = colors[i]

        line, = ax.plot([], [], color=color, alpha=0.10, linewidth=1)
        marker, = ax.plot([], [], marker='o', markersize=6,
                          color=color, markeredgecolor='black', zorder=15)

        trail = ax.scatter([], [], s=20, zorder=10)

        # Safety circles (RESTORED)
        c30 = Circle((0, 0), 30, color=color, fill=False,
                     linestyle='--', alpha=0.35)
        c5 = Circle((0, 0), 5, color=color, fill=True,
                    alpha=0.2)

        ax.add_patch(c30)
        ax.add_patch(c5)

        # Heading arrow placeholder
        arrow = None

        glow = Circle((0, 0),
              0,
              color=color,
              alpha=0.0,
              zorder=5,
              linewidth=0)

        ax.add_patch(glow)

        uav_artists.append({
            'positions': np.array(data['positions']),
            'headings': np.array(data['headings']),
            'current_targets': data['current_targets'],
            'line': line,
            'marker': marker,
            'trail': trail,
            'c30': c30,
            'c5': c5,
            'arrow': arrow,
            'glow': glow,
            'color': color,
            'last_target': None,
            'trail_reset_counter': 0
        })

    ax.set_xlim(tl_x - 100, br_x + 100)
    ax.set_ylim(br_y - 100, tl_y + 100)
    ax.set_aspect('equal')

    title_text = ax.text(0.5, 1.05, '',
                         transform=ax.transAxes,
                         ha='center',
                         fontweight='bold')

    def update(frame_num):
        step = frame_indices[frame_num]

        title_text.set_text(
            f"{scenario_name}\nStep: {step}/{max_steps} | Arrivals: {arrivals}"
        )

        updated = [title_text]

        for art in uav_artists:

            if step >= len(art['positions']):
                continue

            pos = art['positions'][step]

            # Aircraft marker
            art['marker'].set_data([pos[0]], [pos[1]])
            art['line'].set_data(
                art['positions'][:step+1, 0],
                art['positions'][:step+1, 1]
            )

            # Safety circles
            art['c30'].set_center(pos)
            art['c5'].set_center(pos)

            # ======================================================
            # LONGER FADING TRAIL
            # ======================================================
            trail_len = 120   # MUCH longer trail

            if art['trail_reset_counter'] > 0:
                trail_len = 25
                art['trail_reset_counter'] -= 1

            start_idx = max(0, step - trail_len)
            trail_pts = art['positions'][start_idx:step]

            if len(trail_pts) > 0:
                art['trail'].set_offsets(trail_pts)

                alphas = np.linspace(0.02, 0.7, len(trail_pts))
                rgb = mcolors.to_rgb(art['color'])
                rgba = np.zeros((len(trail_pts), 4))
                rgba[:, :3] = rgb
                rgba[:, 3] = alphas
                art['trail'].set_facecolors(rgba)

            # ======================================================
            # HEADING VISUALIZATION
            # Heading is radians, NORTH = 0
            #
            # Matplotlib uses:
            # 0 rad = +X (east)
            #
            # Conversion:
            # x = sin(theta)
            # y = cos(theta)
            # ======================================================

            if art['arrow'] is not None:
                art['arrow'].remove()

            if step < len(art['headings']):
                heading = art['headings'][step]

                dx = 40 * np.sin(heading)
                dy = 40 * np.cos(heading)

                art['arrow'] = ax.arrow(
                    pos[0], pos[1],
                    dx, dy,
                    head_width=12,
                    head_length=15,
                    linewidth=2,
                    color=art['color'],
                    zorder=20
                )

                updated.append(art['arrow'])

            # ======================================================
            # ACTIVE WAYPOINT PULSE ONLY (NO DIAMOND)
            # ======================================================

            MAX_GLOW_RADIUS = 30   # meters
            MIN_GLOW_RADIUS = 20
            PULSE_SPEED = 0.05

            if step < len(art['current_targets']):
                target = art['current_targets'][step]

                if target is not None:

                    if art['last_target'] != target:
                        art['trail_reset_counter'] = 25
                        art['last_target'] = target

                    # Smooth bounded pulse in meters
                    pulse = MIN_GLOW_RADIUS + \
                            (MAX_GLOW_RADIUS - MIN_GLOW_RADIUS) * \
                            (0.5 * (1 + np.sin(frame_num * PULSE_SPEED)))

                    art['glow'].set_center(target)
                    art['glow'].set_radius(pulse)

                    # Optional: fade alpha with size for nicer look
                    alpha_scale = pulse / MAX_GLOW_RADIUS
                    art['glow'].set_alpha(0.25 * alpha_scale)

                else:
                    art['glow'].set_alpha(0.0)

            updated.extend([
                art['marker'],
                art['line'],
                art['trail'],
                art['c30'],
                art['c5'],
                art['glow']
            ])

        return updated

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task(
            "[cyan]Rendering video...", total=total_frames
        )

        anim = FuncAnimation(
            fig, update,
            frames=total_frames,
            interval=1000/fps,
            blit=False
        )

        writer = FFMpegWriter(fps=fps, bitrate=2000)

        original_grab = writer.grab_frame

        def grab_with_progress(*args, **kwargs):
            original_grab(*args, **kwargs)
            progress.update(task, advance=1)

        writer.grab_frame = grab_with_progress
        anim.save(save_path, writer=writer)

    plt.close()


# ================================================================
# TEST ENTRY
# ================================================================

def test(config):
    console.print(Panel.fit("[bold white]Flight Path Visualizer[/bold white]"))
    os.makedirs(config["test"]["save_dir"], exist_ok=True)

    model = A2C.load(config["test"]["model_path"], device='cpu')
    origin = [
        float(config["test"]["env"]["origin"][0]),
        float(config["test"]["env"]["origin"][1])
    ]

    scenario_info = {
        "name": config["test"]["test_name"],
        "box_size": config["test"]["env"]["box_size"]
    }

    env, tl, br = create_test_environment(scenario_info, origin, config)

    uav_data, steps, total_reward, arrivals = run_and_record_episode(
        model, env, env.transformer,
        config["test"]["env"]["max_steps"]
    )

    console.print("\n[bold green]Episode Finished[/bold green]")
    console.print(f"Total Reward: {total_reward:.2f}")
    console.print(f"Waypoint Arrivals: {arrivals}\n")

    if config["test"].get("create_video", True):
        vid_name = f"{config['test']['test_name'].replace(' ', '_')}.mp4"
        vid_path = os.path.join(config["test"]["save_dir"], vid_name)

        create_video(
            uav_data,
            tl,
            br,
            env.transformer,
            config["test"]["test_name"],
            total_reward,
            arrivals,
            vid_path,
            fps=config["test"].get("video_fps", 30),
            speed_multiplier=config["test"].get("video_speed", 1)
        )

    env.close()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_test_2.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    test(config)
