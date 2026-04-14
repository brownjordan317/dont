from __future__ import annotations
from contextlib import nullcontext
import os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config_utils import get_tuning_section
from flight_engine.dubins import build_centered_dubins_route
from flight_engine.navigation_utils import (
    build_box_bounds,
    local_box_geometry,
    ordered_waypoint_tuples,
)
from flight_engine.trans_coorders import CoordinateTransformer
from mappo import MAPPOPolicy
from mappo_runtime import (
    action_dict_from_step,
    sample_info,
    select_actions,
    validate_policy_env,
)
from pettingzoo_env import MultiUAVParallelEnv

console = Console()


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                console.print(
                    "[yellow]CUDA was requested but is not available; falling back to CPU.[/yellow]"
                )
                return "cpu"
        except ModuleNotFoundError:
            return "cpu"
    return requested_device


def resolve_test_mission_mode(config) -> str:
    test_cfg = config["test"]
    raw_mode = str(test_cfg.get("mission_mode", "auto")).strip().lower()
    mode_aliases = {
        "manual": "manual",
        "manual_mission": "manual",
        "manual mission": "manual",
        "gen_mission": "gen_mission",
        "generated": "gen_mission",
        "generated_mission": "gen_mission",
        "generated mission": "gen_mission",
        "random": "gen_mission",
        "auto": "auto",
    }
    if raw_mode not in mode_aliases:
        raise ValueError(
            "Unsupported test mission_mode. "
            "Expected one of: manual, gen_mission, auto."
        )

    resolved_mode = mode_aliases[raw_mode]
    if resolved_mode == "auto":
        return "manual" if test_cfg.get("missions") else "gen_mission"
    return resolved_mode


def summarize_waypoint_counts(metrics: dict) -> tuple[int, int]:
    mission_stats = metrics.get("mission_stats", [])
    team_reached = sum(int(stat.get("waypoints_reached", 0)) for stat in mission_stats)
    team_assigned = sum(int(stat.get("assigned_waypoints", 0)) for stat in mission_stats)
    return team_reached, team_assigned


def build_planned_dubins_path(
    env: MultiUAVParallelEnv,
    agent: str,
    transformer: CoordinateTransformer,
    *,
    sample_step_m: float | None = None,
) -> list[tuple[float, float]]:
    aircraft = env.aircraft_by_agent[agent]
    mission_waypoints = ordered_waypoint_tuples(aircraft)
    start_position = aircraft.initial_pos.to_tuple()
    start_x, start_y = transformer.geo_to_local(start_position[0], start_position[1])
    if not mission_waypoints:
        return [(float(start_x), float(start_y))]

    waypoint_positions_local = [
        transformer.geo_to_local(latitude, longitude)
        for latitude, longitude in mission_waypoints
    ]
    effective_step = (
        sample_step_m
        if sample_step_m is not None
        else max(1.5, min(5.0, float(aircraft.base_turning_radius) / 10.0))
    )
    return build_centered_dubins_route(
        start_position_local=(float(start_x), float(start_y)),
        start_heading_rad=float(aircraft.initial_heading),
        waypoint_positions_local=[
            (float(local_x), float(local_y))
            for local_x, local_y in waypoint_positions_local
        ],
        turn_radius_m=float(aircraft.base_turning_radius),
        sample_step_m=float(effective_step),
        bounds_local=(
            float(env.local_min_x),
            float(env.local_max_x),
            float(env.local_min_y),
            float(env.local_max_y),
        ),
    )


def _manual_agent_sort_key(agent_name: str) -> tuple[int, str]:
    prefix, sep, suffix = agent_name.partition("-")
    if prefix == "UAV" and sep and suffix.isdigit():
        return int(suffix), agent_name
    return 10_000, agent_name


def _find_supplemental_start_position(
    *,
    used_positions: list[np.ndarray],
    center_local: np.ndarray,
    local_max_x: float,
    local_max_y: float,
    edge_margin_m: float,
    min_separation_m: float,
) -> tuple[float, float]:
    x_min = edge_margin_m
    x_max = max(local_max_x - edge_margin_m, x_min + 1.0)
    y_min = edge_margin_m
    y_max = max(local_max_y - edge_margin_m, y_min + 1.0)

    def candidate_is_valid(candidate: np.ndarray) -> bool:
        if not (x_min <= candidate[0] <= x_max and y_min <= candidate[1] <= y_max):
            return False
        return all(
            float(np.linalg.norm(candidate - existing)) >= min_separation_m
            for existing in used_positions
        )

    angle_count = 24
    for ring_idx in range(6):
        radius = 0.0 if ring_idx == 0 else min_separation_m * float(ring_idx)
        phase = 0.0 if ring_idx % 2 == 0 else (np.pi / angle_count)
        for angle_idx in range(angle_count):
            angle = phase + ((2.0 * np.pi * angle_idx) / angle_count)
            offset = np.asarray(
                [np.cos(angle) * radius, np.sin(angle) * radius],
                dtype=np.float64,
            )
            candidate = center_local + offset
            if candidate_is_valid(candidate):
                return float(candidate[0]), float(candidate[1])

    best_candidate = None
    best_clearance = -np.inf
    grid_x = np.linspace(x_min, x_max, num=9)
    grid_y = np.linspace(y_min, y_max, num=9)
    for x_value in grid_x:
        for y_value in grid_y:
            candidate = np.asarray([x_value, y_value], dtype=np.float64)
            clearance = min(
                (float(np.linalg.norm(candidate - existing)) for existing in used_positions),
                default=np.inf,
            )
            if clearance > best_clearance:
                best_clearance = clearance
                best_candidate = candidate

    if best_candidate is None:
        best_candidate = np.asarray(
            [
                min(max(center_local[0], x_min), x_max),
                min(max(center_local[1], y_min), y_max),
            ],
            dtype=np.float64,
        )
    return float(best_candidate[0]), float(best_candidate[1])


def expand_generated_manual_missions(
    *,
    configured_manual_missions: dict,
    num_drones: int,
    max_agents: int,
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    min_start_separation_m: float,
    caution_dist: float,
    waypoint_arrival_radius: float,
) -> dict:
    if num_drones > max_agents:
        raise ValueError(
            "test.gen_mission.num_drones cannot exceed the configured "
            f"env max_agents ({max_agents})."
        )

    sorted_templates = [
        configured_manual_missions[agent_name]
        for agent_name in sorted(
            configured_manual_missions,
            key=_manual_agent_sort_key,
        )
    ]
    if not sorted_templates:
        raise ValueError(
            "test.mission_mode is set to gen_mission but no test.missions were "
            "provided to define the aircraft initialization state."
        )

    transformer, local_max_x, local_max_y = local_box_geometry(top_left, bottom_right)
    edge_margin_m = max(
        float(waypoint_arrival_radius),
        float(min_start_separation_m) * 0.5,
        float(caution_dist) * 0.75,
        20.0,
    )
    agent_names = [f"UAV-{idx + 1}" for idx in range(num_drones)]
    manual_missions = {}
    used_positions_local: list[np.ndarray] = []
    center_local = np.asarray(
        [local_max_x * 0.5, local_max_y * 0.5],
        dtype=np.float64,
    )

    for agent_idx, agent_name in enumerate(agent_names):
        if agent_name in configured_manual_missions:
            template = configured_manual_missions[agent_name]
        else:
            template = sorted_templates[agent_idx % len(sorted_templates)]

        if agent_name in configured_manual_missions:
            lat, lon = (
                float(configured_manual_missions[agent_name]["initial_position"][0]),
                float(configured_manual_missions[agent_name]["initial_position"][1]),
            )
            local_x, local_y = transformer.geo_to_local(lat, lon)
        else:
            if used_positions_local:
                seed_positions = np.vstack(used_positions_local)
                seed_center = np.mean(seed_positions, axis=0)
            else:
                seed_center = center_local
            local_x, local_y = _find_supplemental_start_position(
                used_positions=used_positions_local,
                center_local=seed_center,
                local_max_x=local_max_x,
                local_max_y=local_max_y,
                edge_margin_m=edge_margin_m,
                min_separation_m=float(min_start_separation_m),
            )
            lat, lon = transformer.local_to_geo(local_x, local_y)

        used_positions_local.append(
            np.asarray([float(local_x), float(local_y)], dtype=np.float64)
        )
        manual_missions[agent_name] = {
            "initial_position": [float(lat), float(lon)],
            "initial_heading": float(template["initial_heading"]),
            "cruise_speed": float(template["cruise_speed"]),
            "turning_radius": float(template["turning_radius"]),
            "waypoints": list(template.get("waypoints", [])),
        }

    return manual_missions


def create_test_environment(config):
    test_cfg = config["test"]
    tuning = get_tuning_section(config, "test")
    env_cfg = tuning["env"]
    mission_mode = resolve_test_mission_mode(config)
    generated_cfg = test_cfg.get("gen_mission", {})
    configured_manual_missions = test_cfg.get("missions", {})
    flight_cfg = tuning["flight"]

    origin = tuple(float(value) for value in env_cfg["origin"])
    top_left, bottom_right = build_box_bounds(origin, float(env_cfg["box_size"]))

    if mission_mode == "manual":
        manual_missions = configured_manual_missions
        if not manual_missions:
            raise ValueError(
                "test.mission_mode is set to manual but no test.missions were provided."
            )
        mission_waypoint_count = max(
            (
                len(params.get("waypoints", []))
                for params in manual_missions.values()
            ),
            default=1,
        )
        reset_options = None
    else:
        if not configured_manual_missions:
            raise ValueError(
                "test.mission_mode is set to gen_mission but no test.missions were "
                "provided to define the aircraft initialization state."
            )

        num_drones = int(
            generated_cfg.get("num_drones", len(configured_manual_missions))
        )
        if num_drones <= 0:
            raise ValueError("test.gen_mission.num_drones must be at least 1.")
        manual_missions = expand_generated_manual_missions(
            configured_manual_missions=configured_manual_missions,
            num_drones=num_drones,
            max_agents=int(env_cfg["max_agents"]),
            top_left=top_left,
            bottom_right=bottom_right,
            min_start_separation_m=float(
                flight_cfg.get(
                    "min_start_separation_m",
                    env_cfg["caution_dist"] * 1.5,
                )
            ),
            caution_dist=float(env_cfg["caution_dist"]),
            waypoint_arrival_radius=float(
                env_cfg.get(
                    "waypoint_arrival_radius",
                    env_cfg.get("wp_hit_radius", 30.0),
                )
            ),
        )
        mission_waypoint_count = int(
            generated_cfg.get(
                "num_waypoints",
                env_cfg.get("mission_waypoint_count", 3),
            )
        )
        if mission_waypoint_count <= 0:
            raise ValueError("test.gen_mission.num_waypoints must be at least 1.")
        if not (env_cfg["min_agents"] <= len(manual_missions) <= env_cfg["max_agents"]):
            raise ValueError(
                "The selected generated-mission aircraft count must be within the "
                f"configured env min/max agent bounds [{env_cfg['min_agents']}, {env_cfg['max_agents']}]."
            )
        reset_options = {"generate_random_waypoints": True}

    env = MultiUAVParallelEnv(
        dt=env_cfg["dt"],
        max_steps=env_cfg["max_steps"],
        timeout_scale_with_mission_size=env_cfg.get(
            "timeout_scale_with_mission_size",
            False,
        ),
        timeout_steps_per_additional_waypoint=env_cfg.get(
            "timeout_steps_per_additional_waypoint",
            0,
        ),
        timeout_scale_with_route_distance=env_cfg.get(
            "timeout_scale_with_route_distance",
            False,
        ),
        timeout_steps_per_additional_route_km=env_cfg.get(
            "timeout_steps_per_additional_route_km",
            0.0,
        ),
        timeout_max_steps=env_cfg.get("timeout_max_steps"),
        timeout_reference_waypoints=env_cfg.get("timeout_reference_waypoints"),
        timeout_reference_route_distance_m=env_cfg.get(
            "timeout_reference_route_distance_m"
        ),
        boundary_margin=env_cfg["boundary_margin"],
        mission_waypoint_count=mission_waypoint_count,
        waypoint_arrival_radius=env_cfg.get(
            "waypoint_arrival_radius",
            env_cfg.get("wp_hit_radius", 30.0),
        ),
        obs_stack_size=env_cfg["obs_stack_size"],
        caution_dist=env_cfg["caution_dist"],
        critical_dist=env_cfg["critical_dist"],
        min_agents=env_cfg["min_agents"],
        max_agents=env_cfg["max_agents"],
        origin=origin,
        top_left=top_left,
        bottom_right=bottom_right,
        flight_config=tuning["flight"],
        reward_config=tuning["rewards"],
        guidance_config=tuning["guidance"],
        manual_missions=manual_missions,
    )
    return env, top_left, bottom_right, reset_options


def run_light_episode(
    policy: MAPPOPolicy,
    env: MultiUAVParallelEnv,
    max_steps: int,
    label: str = "MAPPO",
    *,
    deterministic: bool = True,
    show_progress: bool = True,
):
    total_reward = 0.0
    step_count = 0
    last_info = {"waypoints_hit": 0}

    progress_ctx = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress_ctx if show_progress else nullcontext() as progress:
        task = (
            progress.add_task(f"[yellow]Simulating {label}...", total=max_steps)
            if show_progress
            else None
        )
        while env.agents and step_count < max_steps:
            step = select_actions(
                policy,
                env,
                deterministic=deterministic,
                update_stats=False,
            )
            _, rewards, terminations, truncations, infos = env.step(
                action_dict_from_step(step)
            )

            total_reward += float(next(iter(rewards.values()), 0.0))
            step_count += 1
            last_info = sample_info(infos) if infos else last_info
            if show_progress and task is not None:
                progress.update(task, advance=1)

            if any(terminations.values()) or any(truncations.values()):
                break

    metrics = last_info.get("episode_metrics", env.get_episode_metrics())
    terminated = bool(last_info.get("terminated", False))
    truncated = bool(last_info.get("truncated", step_count >= max_steps))
    team_reached, team_assigned = summarize_waypoint_counts(metrics)

    return (
        step_count,
        total_reward,
        team_reached,
        team_assigned,
        terminated,
        truncated,
        metrics,
    )


def run_and_record_episode(
    policy: MAPPOPolicy,
    env: MultiUAVParallelEnv,
    transformer,
    max_steps: int,
    *,
    deterministic: bool = True,
):
    tracked_agents = [agent for agent in env.possible_agents if agent in env.aircraft_by_agent]
    uav_data = [
        {
            "id": agent,
            "positions": [],
            "headings": [],
            "waypoints_visited": [],
            "all_waypoints": [],
            "current_targets": [],
            "planned_path": [],
        }
        for agent in tracked_agents
    ]

    for idx, agent in enumerate(tracked_agents):
        uav_data[idx]["planned_path"] = build_planned_dubins_path(
            env,
            agent,
            transformer,
        )

    done = False
    step_count = 0
    total_reward = 0.0
    last_info = {"waypoints_hit": 0}

    while env.agents and not done and step_count < max_steps:
        for idx, agent in enumerate(tracked_agents):
            aircraft = env.aircraft_by_agent[agent]
            position = aircraft.position.to_tuple()
            x_pos, y_pos = transformer.geo_to_local(position[0], position[1])
            uav_data[idx]["positions"].append((x_pos, y_pos))
            uav_data[idx]["headings"].append(float(aircraft.heading))

            waypoint = aircraft.waypoint_manager.current_waypoint
            if waypoint is not None:
                wp_x, wp_y = transformer.geo_to_local(waypoint.latitude, waypoint.longitude)
                uav_data[idx]["current_targets"].append((wp_x, wp_y))
                if (wp_x, wp_y) not in uav_data[idx]["all_waypoints"]:
                    uav_data[idx]["all_waypoints"].append((wp_x, wp_y))
            else:
                uav_data[idx]["current_targets"].append(None)

        step = select_actions(
            policy,
            env,
            deterministic=deterministic,
            update_stats=False,
        )
        _, rewards, terminations, truncations, infos = env.step(
            action_dict_from_step(step)
        )

        total_reward += float(next(iter(rewards.values()), 0.0))
        step_count += 1
        last_info = sample_info(infos) if infos else last_info
        done = bool(any(terminations.values()) or any(truncations.values()))

        for idx, agent in enumerate(tracked_agents):
            aircraft = env.aircraft_by_agent[agent]
            if getattr(aircraft, "last_waypoint_hit_pos", None):
                hit_x, hit_y = transformer.geo_to_local(*aircraft.last_waypoint_hit_pos)
                uav_data[idx]["waypoints_visited"].append((hit_x, hit_y))
                aircraft.last_waypoint_hit_pos = None

    metrics = last_info.get("episode_metrics", env.get_episode_metrics())
    team_reached, team_assigned = summarize_waypoint_counts(metrics)
    return (
        uav_data,
        step_count,
        total_reward,
        team_reached,
        team_assigned,
        metrics,
    )


def create_video(
    uav_data,
    top_left,
    bottom_right,
    transformer,
    scenario_name,
    arrivals,
    total_waypoints,
    save_path,
    config,
    *,
    fps: int = 30,
    speed_multiplier: int = 1,
):
    console.print("[cyan]Creating video...[/cyan]")
    env_cfg = get_tuning_section(config, "test")["env"]
    top_left_x, top_left_y = transformer.geo_to_local(top_left[0], top_left[1])
    bottom_right_x, bottom_right_y = transformer.geo_to_local(
        bottom_right[0],
        bottom_right[1],
    )

    max_steps = max(len(data["positions"]) for data in uav_data)
    frame_indices = list(range(0, max_steps, max(1, int(speed_multiplier))))
    total_frames = len(frame_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, len(uav_data)))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.add_patch(
        Rectangle(
            (top_left_x, bottom_right_y),
            bottom_right_x - top_left_x,
            top_left_y - bottom_right_y,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
    )

    uav_artists = []
    trail_len = 120
    alphas = np.linspace(0.02, 0.7, trail_len)

    for idx, data in enumerate(uav_data):
        color = colors[idx]
        planned_path = np.asarray(data.get("planned_path", []), dtype=np.float32)
        if len(planned_path) >= 2:
            ax.plot(
                planned_path[:, 0],
                planned_path[:, 1],
                color=color,
                linewidth=2.2,
                alpha=0.45,
                linestyle=":",
                zorder=4,
            )
        line, = ax.plot([], [], color=color, alpha=0.10, linewidth=1, animated=True)
        marker, = ax.plot(
            [],
            [],
            marker="o",
            markersize=6,
            color=color,
            markeredgecolor="black",
            zorder=15,
            animated=True,
        )
        trail = ax.scatter([], [], s=20, zorder=10, animated=True)
        caution_circle = Circle(
            (0, 0),
            env_cfg["caution_dist"],
            color=color,
            fill=False,
            linestyle="--",
            alpha=0.35,
            animated=True,
        )
        critical_circle = Circle(
            (0, 0),
            env_cfg["critical_dist"],
            color=color,
            fill=True,
            alpha=0.2,
            animated=True,
        )
        glow = Circle((0, 0), 0, color=color, alpha=0.0, zorder=12, animated=True)
        quiver = ax.quiver(
            [],
            [],
            [],
            [],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.004,
            zorder=14,
            animated=True,
        )

        ax.add_patch(caution_circle)
        ax.add_patch(critical_circle)
        ax.add_patch(glow)

        uav_artists.append(
            {
                "positions": np.asarray(data["positions"], dtype=np.float32),
                "headings": np.asarray(data["headings"], dtype=np.float32),
                "current_targets": data["current_targets"],
                "marker": marker,
                "line": line,
                "trail": trail,
                "c30": caution_circle,
                "c5": critical_circle,
                "glow": glow,
                "quiver": quiver,
                "color_rgb": mcolors.to_rgb(color),
            }
        )

    ax.set_xlim(top_left_x - 100, bottom_right_x + 100)
    ax.set_ylim(bottom_right_y - 100, top_left_y + 100)
    ax.set_aspect("equal")
    title_text = ax.text(
        0.5,
        1.05,
        "",
        transform=ax.transAxes,
        ha="center",
        fontweight="bold",
        animated=True,
    )

    def update(frame_num):
        step = frame_indices[frame_num]
        title_text.set_text(
            f"{scenario_name}\nStep: {step}/{max_steps} | Team Waypoints: {arrivals}/{total_waypoints}"
        )
        changed_artists = [title_text]

        for artist in uav_artists:
            if step >= len(artist["positions"]):
                continue
            position = artist["positions"][step]
            artist["marker"].set_data([position[0]], [position[1]])
            artist["line"].set_data(
                artist["positions"][: step + 1, 0],
                artist["positions"][: step + 1, 1],
            )
            artist["c30"].set_center(position)
            artist["c5"].set_center(position)

            start_idx = max(0, step - trail_len)
            trail_points = artist["positions"][start_idx:step]
            if len(trail_points) > 0:
                artist["trail"].set_offsets(trail_points)
                rgba = np.zeros((len(trail_points), 4))
                rgba[:, :3] = artist["color_rgb"]
                rgba[:, 3] = alphas[-len(trail_points) :]
                artist["trail"].set_facecolors(rgba)

            heading = artist["headings"][step]
            artist["quiver"].set_offsets(position)
            artist["quiver"].set_UVC(40 * np.sin(heading), 40 * np.cos(heading))

            target = artist["current_targets"][step]
            if target:
                radius = max(
                    float(
                        env_cfg.get(
                            "waypoint_arrival_radius",
                            env_cfg.get("wp_hit_radius", 30.0),
                        )
                    ),
                    1.0,
                )
                pulse = (radius * 0.7) + (radius * 0.3) * (
                    0.5 * (1 + np.sin(frame_num * 0.05))
                )
                artist["glow"].set_center(target)
                artist["glow"].set_radius(pulse)
                artist["glow"].set_alpha(0.25 * (pulse / radius))
            else:
                artist["glow"].set_alpha(0.0)

            changed_artists.extend(
                [
                    artist["marker"],
                    artist["line"],
                    artist["trail"],
                    artist["c30"],
                    artist["c5"],
                    artist["glow"],
                    artist["quiver"],
                ]
            )

        return changed_artists

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Rendering...", total=total_frames)
        animation = FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=1000 / fps,
            blit=True,
        )
        writer = FFMpegWriter(fps=fps, bitrate=2000)

        original_grab_frame = writer.grab_frame

        def grab_with_progress(*args, **kwargs):
            original_grab_frame(*args, **kwargs)
            progress.update(task, advance=1)

        writer.grab_frame = grab_with_progress
        animation.save(save_path, writer=writer)

    plt.close()


def create_dubins_sample_visualization(config):
    test_cfg = config["test"]
    tuning = get_tuning_section(config, "test")
    env_cfg = tuning["env"]
    save_dir = test_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    env, top_left, bottom_right, reset_options = create_test_environment(config)
    if reset_options is None:
        env.reset()
    else:
        env.reset(options=reset_options)

    transformer = env.transformer
    top_left_x, top_left_y = transformer.geo_to_local(top_left[0], top_left[1])
    bottom_right_x, bottom_right_y = transformer.geo_to_local(
        bottom_right[0],
        bottom_right[1],
    )

    tracked_agents = [
        agent for agent in env.possible_agents if agent in env.aircraft_by_agent
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tracked_agents), 1)))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=140)
    ax.add_patch(
        Rectangle(
            (top_left_x, bottom_right_y),
            bottom_right_x - top_left_x,
            top_left_y - bottom_right_y,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
    )

    for idx, agent in enumerate(tracked_agents):
        aircraft = env.aircraft_by_agent[agent]
        color = colors[idx]
        start_lat, start_lon = aircraft.initial_pos.to_tuple()
        start_x, start_y = transformer.geo_to_local(start_lat, start_lon)
        mission_waypoints = ordered_waypoint_tuples(aircraft)
        waypoint_points = np.asarray(
            [
                transformer.geo_to_local(latitude, longitude)
                for latitude, longitude in mission_waypoints
            ],
            dtype=np.float32,
        )
        planned_path = np.asarray(
            build_planned_dubins_path(env, agent, transformer),
            dtype=np.float32,
        )

        if len(planned_path) >= 2:
            ax.plot(
                planned_path[:, 0],
                planned_path[:, 1],
                color=color,
                linewidth=2.4,
                alpha=0.95,
                label=f"{agent} (R={aircraft.base_turning_radius:.0f}m)",
                zorder=5,
            )
        ax.scatter(
            [start_x],
            [start_y],
            color=color,
            s=70,
            marker="o",
            edgecolors="black",
            zorder=8,
        )
        ax.quiver(
            [start_x],
            [start_y],
            [35.0 * np.sin(float(aircraft.initial_heading))],
            [35.0 * np.cos(float(aircraft.initial_heading))],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.004,
            zorder=8,
        )

        if len(waypoint_points) > 0:
            ax.plot(
                waypoint_points[:, 0],
                waypoint_points[:, 1],
                linestyle=":",
                linewidth=1.2,
                color=color,
                alpha=0.35,
                zorder=4,
            )
            ax.scatter(
                waypoint_points[:, 0],
                waypoint_points[:, 1],
                color=color,
                s=32,
                marker="x",
                zorder=7,
            )
            arrival_radius = max(
                float(
                    env_cfg.get(
                        "waypoint_arrival_radius",
                        env_cfg.get("wp_hit_radius", 30.0),
                    )
                ),
                1.0,
            )
            for wp_x, wp_y in waypoint_points:
                ax.add_patch(
                    Circle(
                        (float(wp_x), float(wp_y)),
                        arrival_radius,
                        edgecolor=color,
                        facecolor="none",
                        linewidth=0.8,
                        alpha=0.18,
                        zorder=3,
                    )
                )

    ax.set_xlim(top_left_x - 100, bottom_right_x + 100)
    ax.set_ylim(bottom_right_y - 100, top_left_y + 100)
    ax.set_aspect("equal")
    ax.set_title(
        f"{test_cfg['test_name']} Dubins Sample\n"
        f"Planned Dubins paths from test config ({resolve_test_mission_mode(config)})"
    )
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    if tracked_agents:
        ax.legend(loc="upper right")

    save_path = os.path.join(
        save_dir,
        f"{test_cfg['test_name'].replace(' ', '_')}_dubins_sample.png",
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    env.close()

    console.print(f"[green]Dubins sample saved:[/green] {save_path}")
    return save_path


def print_episode_report(total_reward, arrivals, total_waypoints, steps, truncated, metrics):
    termination_reason = metrics.get("episode_summary", {}).get("termination_reason")
    if termination_reason == "completed":
        outcome = "Completed"
    elif termination_reason == "critical_violation":
        outcome = "Critical Violation"
    elif termination_reason == "mission_complete_loitering":
        outcome = "Mission Complete Loiter"
    elif truncated:
        outcome = "Timed Out"
    else:
        outcome = "Ended"
    console.print("\n" + "=" * 50)
    console.print("[bold green]Episode Results:[/bold green]")
    console.print(f" • Total Reward: [bold white]{total_reward:.2f}[/bold white]")
    console.print(
        f" • Team Waypoints Hit: [bold cyan]{arrivals}/{total_waypoints}[/bold cyan]"
    )
    console.print(f" • Steps: [bold yellow]{steps}[/bold yellow]")
    console.print(f" • Status: [bold]{outcome}[/bold]")
    console.print(
        f" • Caution Violations: [yellow]{metrics['safety_violations']['caution']['total_count']}[/yellow]"
    )
    console.print(
        f" • Critical Violations: [red]{metrics['safety_violations']['critical']['total_count']}[/red]"
    )
    console.print(
        f" • Geofence Violations: [red]{metrics['safety_violations']['geofence']['total_count']}[/red]"
    )
    console.print("\n[bold cyan]Distance Traveled per UAV:[/bold cyan]")
    for stat in metrics["mission_stats"]:
        distance_m = stat["dist_navigating"]
        console.print(
            f" • {stat['id']}: [green]{distance_m:.2f} m[/green] ({stat['waypoints_reached']} waypoints)"
        )
    console.print("=" * 50 + "\n")


def test(config):
    test_cfg = config["test"]

    light_mode = not bool(test_cfg.get("save_visuals", False))
    if light_mode:
        console.print(Panel.fit("[bold cyan]MAPPO Inference Engine[/bold cyan]"))
    else:
        console.print(Panel.fit("[bold white]MAPPO Flight Path Visualizer[/bold white]"))

    os.makedirs(test_cfg["save_dir"], exist_ok=True)

    env, top_left, bottom_right, reset_options = create_test_environment(config)
    policy = MAPPOPolicy.load(
        test_cfg["model_path"],
        device=resolve_device(str(test_cfg.get("device", "cpu"))),
    )
    validate_policy_env(policy, env)

    if reset_options is None:
        env.reset()
    else:
        env.reset(options=reset_options)
    max_steps = int(env.max_steps)
    deterministic = bool(test_cfg.get("deterministic", True))

    if light_mode:
        steps, total_reward, arrivals, total_waypoints, _, truncated, metrics = run_light_episode(
            policy,
            env,
            max_steps,
            label="MAPPO",
            deterministic=deterministic,
        )
        print_episode_report(
            total_reward,
            arrivals,
            total_waypoints,
            steps,
            truncated,
            metrics,
        )
    else:
        uav_data, steps, total_reward, arrivals, total_waypoints, metrics = run_and_record_episode(
            policy,
            env,
            env.transformer,
            max_steps,
            deterministic=deterministic,
        )
        print_episode_report(
            total_reward,
            arrivals,
            total_waypoints,
            steps,
            False,
            metrics,
        )

        if bool(test_cfg.get("create_video", True)):
            video_name = f"{test_cfg['test_name'].replace(' ', '_')}.mp4"
            create_video(
                uav_data,
                top_left,
                bottom_right,
                env.transformer,
                test_cfg["test_name"],
                arrivals,
                total_waypoints,
                os.path.join(test_cfg["save_dir"], video_name),
                config,
                fps=int(test_cfg.get("video_fps", 30)),
                speed_multiplier=int(test_cfg.get("video_speed", 1)),
            )

    env.close()
