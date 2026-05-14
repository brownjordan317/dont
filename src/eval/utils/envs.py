from __future__ import annotations

from config_utils import get_tuning_section
from inference_setup import maybe_wrap_manager_env, resolve_device
from mappo import MAPPOPolicy
from env.pettingzoo_env import MultiUAVParallelEnv


def build_eval_guidance_config(config: dict, *, section: str = "eval") -> dict:
    tuning = get_tuning_section(config, section)
    return dict(tuning["guidance"])


def build_base_eval_env(config: dict, *, section: str = "eval") -> MultiUAVParallelEnv:
    tuning = get_tuning_section(config, section)
    env_cfg = tuning["env"]
    guidance_config = build_eval_guidance_config(config, section=section)
    return MultiUAVParallelEnv(
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
        mission_waypoint_count=config["drone_settings"]["num_wps_per_drone"],
        waypoint_arrival_radius=env_cfg.get("waypoint_arrival_radius", 30.0),
        obs_stack_size=env_cfg["obs_stack_size"],
        separation_dist=env_cfg["separation_dist"],
        min_agents=env_cfg["min_agents"],
        max_agents=env_cfg["max_agents"],
        reset_generation_attempts=env_cfg.get("reset_generation_attempts", 128),
        reset_min_feasible_cpa_m=env_cfg.get("reset_min_feasible_cpa_m"),
        reset_min_boundary_time_ratio=env_cfg.get("reset_min_boundary_time_ratio", 0.35),
        reset_heading_jitter_rad=env_cfg.get("reset_heading_jitter_rad", 0.2),
        map_size_range_m=(env_cfg["box_min_m"], env_cfg["box_max_m"]),
        terminate_on_geofence_violation=env_cfg.get(
            "terminate_on_geofence_violation",
            True,
        ),
        geofence_breach_grace_steps=env_cfg.get("geofence_breach_grace_steps", 1),
        flight_config=tuning["flight"],
        reward_config=tuning["rewards"],
        guidance_config=guidance_config,
    )


def load_eval_policy(config: dict, *, section: str = "eval") -> MAPPOPolicy:
    cfg = config[section]
    return MAPPOPolicy.load(
        cfg["model_path"],
        device=resolve_device(str(cfg.get("device", "cpu"))),
    )


def build_manager_eval_env(config: dict, policy: MAPPOPolicy):
    env = build_base_eval_env(config, section="eval")
    return maybe_wrap_manager_env(
        env,
        config=config,
        policy=policy,
        config_section="eval",
    )
