from __future__ import annotations

import numpy as np
from rich.console import Console

from config_utils import get_tuning_section
from flight_engine.navigation_utils import build_box_bounds, local_box_geometry
from mappo import MAPPOPolicy
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


def create_test_environment(
    config,
    *,
    terminate_on_all_waypoints_complete: bool = True,
    allow_live_waypoint_updates: bool = False,
):
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
        terminate_on_all_waypoints_complete=terminate_on_all_waypoints_complete,
        allow_live_waypoint_updates=allow_live_waypoint_updates,
    )
    return env, top_left, bottom_right, reset_options


def load_policy_for_config(config) -> MAPPOPolicy:
    test_cfg = config["test"]
    return MAPPOPolicy.load(
        test_cfg["model_path"],
        device=resolve_device(str(test_cfg.get("device", "cpu"))),
    )
