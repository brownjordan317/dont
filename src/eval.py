from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os

from geopy.distance import distance as geopy_distance
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config_utils import get_tuning_section
from flight_engine.navigation_utils import planned_route_distance_m
from inference_setup import resolve_device
from mappo import MAPPOPolicy
from mappo_runtime import validate_policy_env
from pettingzoo_env import MultiUAVParallelEnv
from test import run_light_episode

console = Console()


def box_dimensions_m(top_left, bottom_right):
    if not top_left or not bottom_right:
        return 0.0, 0.0

    mid_lat = (top_left[0] + bottom_right[0]) / 2.0
    mid_lon = (top_left[1] + bottom_right[1]) / 2.0

    width_m = geopy_distance(
        (mid_lat, top_left[1]),
        (mid_lat, bottom_right[1]),
    ).meters
    height_m = geopy_distance(
        (top_left[0], mid_lon),
        (bottom_right[0], mid_lon),
    ).meters
    return float(width_m), float(height_m)


class ResultsWriter:
    def __init__(self, path: str, config: dict):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.data = {
            "meta": {
                "schema_version": 3,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
            },
            "missions": [],
            "benchmark": None,
        }
        self._flush()

    def append_mission(self, record: dict):
        self.data["missions"].append(record)
        self._flush()

    def finalize(self, benchmark: dict):
        self.data["benchmark"] = benchmark
        self._flush()
        console.print(f"[green]Results saved -> {self.path}[/green]")

    def _flush(self):
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as handle:
            json.dump(self.data, handle, indent=2)
        os.replace(tmp_path, self.path)


def ordered_waypoints(aircraft) -> list:
    waypoints = []
    current_waypoint = aircraft.waypoint_manager.current_waypoint
    if current_waypoint is not None:
        waypoints.append(current_waypoint.to_tuple())
    waypoints.extend(
        waypoint.to_tuple()
        for waypoint in aircraft.waypoint_manager.waypoint_queue
    )
    return waypoints


def build_mission_definition(env: MultiUAVParallelEnv, config: dict) -> dict:
    top_left = (env.max_lat, env.min_lon)
    bottom_right = (env.min_lat, env.max_lon)
    width_m, height_m = box_dimensions_m(top_left, bottom_right)
    uav_records = []
    total_planned_distance = 0.0

    for agent in env.agents:
        aircraft = env.aircraft_by_agent[agent]
        mission_waypoints = ordered_waypoints(aircraft)
        planned_distance = planned_route_distance_m(
            aircraft,
            env.transformer,
            start_position=aircraft.initial_pos.to_tuple(),
        )
        total_planned_distance += planned_distance
        uav_records.append(
            {
                "id": aircraft.id_tag,
                "initial_position": aircraft.initial_pos.to_tuple(),
                "initial_heading_rad": float(aircraft.initial_heading),
                "initial_heading_deg": float(np.degrees(aircraft.initial_heading)),
                "cruise_speed_mps": float(aircraft.base_cruise_speed),
                "turning_radius_m": float(aircraft.base_turning_radius),
                "waypoints": mission_waypoints,
                "planned_route_distance_m": float(planned_distance),
            }
        )

    return {
        "origin": (
            float((top_left[0] + bottom_right[0]) / 2.0),
            float((top_left[1] + bottom_right[1]) / 2.0),
        ),
        "top_left": top_left,
        "bottom_right": bottom_right,
        "box_width_m": float(width_m),
        "box_height_m": float(height_m),
        "box_area_km2": float((width_m * height_m) / 1_000_000.0),
        "num_drones": len(uav_records),
        "episode_max_steps": int(env.max_steps),
        "base_max_steps": int(env.base_max_steps),
        "timeout_scaled": bool(env.max_steps != env.base_max_steps),
        "timeout_max_route_distance_m": float(env._episode_timeout_max_route_distance_m),
        "timeout_reference_route_distance_m": float(
            env._episode_timeout_reference_route_distance_m
        ),
        "waypoints_per_drone": (
            float(sum(len(record["waypoints"]) for record in uav_records) / len(uav_records))
            if uav_records
            else 0.0
        ),
        "total_waypoints": int(sum(len(record["waypoints"]) for record in uav_records)),
        "planned_route_distance_m_total": float(total_planned_distance),
        "planned_route_distance_m_max": (
            float(max(record["planned_route_distance_m"] for record in uav_records))
            if uav_records
            else 0.0
        ),
        "planned_route_distance_m_avg": (
            float(total_planned_distance / len(uav_records))
            if uav_records
            else 0.0
        ),
        "uavs": uav_records,
    }


def summarize_mission(
    *,
    policy_path: str,
    reward: float,
    steps: int,
    terminated: bool,
    truncated: bool,
    metrics: dict,
    mission_definition: dict,
    dt: float,
) -> dict:
    mission_stats = metrics.get("mission_stats", [])
    safety = metrics.get("safety_violations", {})
    episode_summary = metrics.get("episode_summary", {})
    reward_breakdown = metrics.get("reward_breakdown", {})
    telemetry = metrics.get("telemetry", [])

    reached_waypoints = sum(drone.get("waypoints_reached", 0) for drone in mission_stats)
    total_waypoints = sum(
        drone.get("assigned_waypoints", mission_definition["waypoints_per_drone"])
        for drone in mission_stats
    )
    total_distance = sum(drone.get("dist_navigating", 0.0) for drone in mission_stats)
    avg_distance = float(total_distance / len(mission_stats)) if mission_stats else 0.0
    completion_rate = float(reached_waypoints / total_waypoints) if total_waypoints else 0.0
    termination_reason = episode_summary.get("termination_reason") or "ended"
    min_pairwise_distance = episode_summary.get("min_pairwise_distance_m")
    crashed = termination_reason == "critical_violation"
    completed_mission = termination_reason == "completed"
    is_truncated = bool(truncated or termination_reason == "max_steps")

    summary = {
        "waypoints_reached_total": int(reached_waypoints),
        "waypoints_total": int(total_waypoints),
        "waypoints_remaining_total": int(max(total_waypoints - reached_waypoints, 0)),
        "waypoint_completion_rate": completion_rate,
        "waypoint_throughput_per_min": (
            float(reached_waypoints / max((steps * dt) / 60.0, 1e-6))
            if steps > 0
            else 0.0
        ),
        "avg_distance_per_uav_m": avg_distance,
        "total_distance_m": float(total_distance),
        "planned_route_distance_m_total": float(
            mission_definition["planned_route_distance_m_total"]
        ),
        "distance_vs_planned_ratio": (
            float(total_distance / mission_definition["planned_route_distance_m_total"])
            if mission_definition["planned_route_distance_m_total"]
            else None
        ),
        "caution_events": int(safety.get("caution", {}).get("total_count", 0)),
        "critical_events": int(safety.get("critical", {}).get("total_count", 0)),
        "geofence_exits": int(safety.get("geofence", {}).get("total_count", 0)),
        "geofence_outside_steps": int(
            safety.get("geofence", {}).get("outside_step_total", 0)
        ),
        "min_pairwise_distance_m": (
            float(min_pairwise_distance) if min_pairwise_distance is not None else None
        ),
        "min_pairwise_pair": episode_summary.get("min_pairwise_pair"),
        "min_pairwise_time_s": episode_summary.get("min_pairwise_time_s"),
        "uavs_completed": int(episode_summary.get("uavs_completed", 0)),
        "deconfliction_time_s": float(episode_summary.get("deconfliction_time_s", 0.0)),
        "circling_steps_total": int(episode_summary.get("circling_steps_total", 0)),
        "circling_breakouts_total": int(
            episode_summary.get("circling_breakouts_total", 0)
        ),
        "waypoint_reapproach_steps_total": int(
            episode_summary.get("waypoint_reapproach_steps_total", 0)
        ),
        "waypoint_reapproach_events_total": int(
            episode_summary.get("waypoint_reapproach_events_total", 0)
        ),
        "duration_s": float(steps * dt),
        "reward_per_step": float(reward / steps) if steps else 0.0,
        "completed_mission": completed_mission,
        "crashed": crashed,
        "failed_mission": bool(not completed_mission),
        "truncated": is_truncated,
        "timed_out": bool(termination_reason == "max_steps"),
        "boundary_compliant": bool(
            int(safety.get("geofence", {}).get("total_count", 0)) == 0
            and int(safety.get("geofence", {}).get("outside_step_total", 0)) == 0
        ),
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy": {"model_path": policy_path},
        "reward": float(reward),
        "steps": int(steps),
        "duration_s": summary["duration_s"],
        "done": bool(terminated),
        "truncated": summary["truncated"],
        "termination_reason": termination_reason,
        "completed_mission": summary["completed_mission"],
        "summary": summary,
        "mission_stats": mission_stats,
        "episode_summary": episode_summary,
        "reward_breakdown": reward_breakdown,
        "safety_violations": safety,
        "telemetry": telemetry,
    }


def update_history(history: dict, mission_record: dict):
    summary = mission_record["summary"]
    history["rewards"].append(mission_record["reward"])
    history["steps"].append(mission_record["steps"])
    history["durations_s"].append(mission_record["duration_s"])
    history["reward_per_step"].append(summary["reward_per_step"])
    history["wps"].append(summary["waypoint_completion_rate"])
    history["throughput"].append(summary["waypoint_throughput_per_min"])
    history["distances"].append(summary["avg_distance_per_uav_m"])
    history["distance_ratios"].append(
        float(summary["distance_vs_planned_ratio"])
        if summary["distance_vs_planned_ratio"] is not None
        else np.nan
    )
    history["caution_counts"].append(summary["caution_events"])
    history["critical_counts"].append(summary["critical_events"])
    history["geofence_counts"].append(summary["geofence_exits"])
    history["outside_steps"].append(summary["geofence_outside_steps"])
    history["circling_steps"].append(summary["circling_steps_total"])
    history["circling_breakouts"].append(summary["circling_breakouts_total"])
    history["deconfliction_times"].append(summary["deconfliction_time_s"])
    history["min_pairwise_distances"].append(
        float(summary["min_pairwise_distance_m"])
        if summary["min_pairwise_distance_m"] is not None
        else np.nan
    )
    history["timed_out"] += int(summary["timed_out"])
    history["boundary_compliant"] += int(summary["boundary_compliant"])
    history["status_counts"][mission_record["termination_reason"]] += 1
    if summary["crashed"]:
        history["crashes"] += 1
    if summary["failed_mission"]:
        history["failed"] += 1
    if summary["completed_mission"]:
        history["completed"] += 1
    if mission_record["termination_reason"] == "max_steps":
        history["max_steps"] += 1


def build_benchmark(history: dict) -> dict:
    rewards = np.asarray(history["rewards"], dtype=float)
    steps = np.asarray(history["steps"], dtype=float)
    durations = np.asarray(history["durations_s"], dtype=float)
    reward_per_step = np.asarray(history["reward_per_step"], dtype=float)
    distances = np.asarray(history["distances"], dtype=float)
    wps = np.asarray(history["wps"], dtype=float)
    throughput = np.asarray(history["throughput"], dtype=float)
    distance_ratios = np.asarray(history["distance_ratios"], dtype=float)
    caution = np.asarray(history["caution_counts"], dtype=float)
    critical = np.asarray(history["critical_counts"], dtype=float)
    geofence = np.asarray(history["geofence_counts"], dtype=float)
    outside_steps = np.asarray(history["outside_steps"], dtype=float)
    circling_steps = np.asarray(history["circling_steps"], dtype=float)
    circling_breakouts = np.asarray(history["circling_breakouts"], dtype=float)
    deconfliction_times = np.asarray(history["deconfliction_times"], dtype=float)
    min_pairwise = np.asarray(history["min_pairwise_distances"], dtype=float)
    finite_min_pairwise = min_pairwise[np.isfinite(min_pairwise)]
    finite_distance_ratios = distance_ratios[np.isfinite(distance_ratios)]
    mission_count = max(int(len(rewards)), 1)

    return {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "std_reward": float(rewards.std()) if rewards.size else 0.0,
        "median_reward": float(np.median(rewards)) if rewards.size else 0.0,
        "p10_reward": float(np.percentile(rewards, 10)) if rewards.size else 0.0,
        "p90_reward": float(np.percentile(rewards, 90)) if rewards.size else 0.0,
        "avg_steps": float(steps.mean()) if steps.size else 0.0,
        "avg_duration_s": float(durations.mean()) if durations.size else 0.0,
        "avg_reward_per_step": float(reward_per_step.mean()) if reward_per_step.size else 0.0,
        "avg_dist_m": float(distances.mean()) if distances.size else 0.0,
        "wp_completion_rate": float(wps.mean()) if wps.size else 0.0,
        "avg_waypoint_throughput_per_min": float(throughput.mean()) if throughput.size else 0.0,
        "mission_completion_rate": float(history["completed"] / mission_count),
        "crash_rate": float(history["crashes"] / mission_count),
        "failure_rate": float(history["failed"] / mission_count),
        "timeout_rate": float(history["timed_out"] / mission_count),
        "boundary_compliance_rate": float(history["boundary_compliant"] / mission_count),
        "avg_caution_events": float(caution.mean()) if caution.size else 0.0,
        "avg_critical_events": float(critical.mean()) if critical.size else 0.0,
        "avg_geofence_exits": float(geofence.mean()) if geofence.size else 0.0,
        "avg_geofence_outside_steps": float(outside_steps.mean()) if outside_steps.size else 0.0,
        "avg_circling_steps": float(circling_steps.mean()) if circling_steps.size else 0.0,
        "avg_circling_breakouts": (
            float(circling_breakouts.mean()) if circling_breakouts.size else 0.0
        ),
        "avg_deconfliction_time_s": (
            float(deconfliction_times.mean()) if deconfliction_times.size else 0.0
        ),
        "avg_distance_vs_planned_ratio": (
            float(finite_distance_ratios.mean()) if finite_distance_ratios.size else None
        ),
        "avg_min_pairwise_distance_m": (
            float(finite_min_pairwise.mean()) if finite_min_pairwise.size else None
        ),
        "status_counts": dict(history["status_counts"]),
        "num_missions": int(len(rewards)),
    }


def print_benchmark(summary: dict):
    table = Table(title="MAPPO Evaluation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Mean Reward", f"{summary['mean_reward']:.2f}")
    table.add_row("Reward Std", f"{summary['std_reward']:.2f}")
    table.add_row("Avg Steps", f"{summary['avg_steps']:.1f}")
    table.add_row("Waypoint Completion", f"{summary['wp_completion_rate']:.2%}")
    table.add_row("Waypoint Throughput", f"{summary['avg_waypoint_throughput_per_min']:.2f}/min")
    table.add_row("Mission Completion", f"{summary['mission_completion_rate']:.2%}")
    table.add_row("Crash Rate", f"{summary['crash_rate']:.2%}")
    table.add_row("Timeout Rate", f"{summary['timeout_rate']:.2%}")
    table.add_row("Boundary Compliance", f"{summary['boundary_compliance_rate']:.2%}")
    table.add_row("Failure Rate", f"{summary['failure_rate']:.2%}")
    table.add_row("Avg Circling Steps", f"{summary['avg_circling_steps']:.1f}")
    table.add_row("Avg Deconfliction Time", f"{summary['avg_deconfliction_time_s']:.1f} s")
    table.add_row(
        "Avg Dist/Planned",
        str(summary["avg_distance_vs_planned_ratio"]),
    )
    table.add_row("Avg Min Separation", str(summary["avg_min_pairwise_distance_m"]))
    console.print(table)


def eval(config):
    eval_cfg = config["eval"]
    tuning = get_tuning_section(config, "eval")
    env_cfg = tuning["env"]

    console.print(Panel.fit("[bold blue]MAPPO Evaluation[/bold blue]"))

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
        mission_waypoint_count=config["drone_settings"]["num_wps_per_drone"],
        waypoint_arrival_radius=env_cfg.get("waypoint_arrival_radius", 30.0),
        obs_stack_size=env_cfg["obs_stack_size"],
        caution_dist=env_cfg["caution_dist"],
        critical_dist=env_cfg["critical_dist"],
        min_agents=env_cfg["min_agents"],
        max_agents=env_cfg["max_agents"],
        map_size_range_m=(env_cfg["box_min_m"], env_cfg["box_max_m"]),
        flight_config=tuning["flight"],
        reward_config=tuning["rewards"],
        guidance_config=tuning["guidance"],
    )

    policy = MAPPOPolicy.load(
        eval_cfg["model_path"],
        device=resolve_device(str(eval_cfg.get("device", "cpu"))),
    )
    validate_policy_env(policy, env)

    writer = ResultsWriter(config["output"]["results_path"], config)
    history = {
        "rewards": [],
        "steps": [],
        "durations_s": [],
        "reward_per_step": [],
        "wps": [],
        "throughput": [],
        "distances": [],
        "distance_ratios": [],
        "caution_counts": [],
        "critical_counts": [],
        "geofence_counts": [],
        "outside_steps": [],
        "circling_steps": [],
        "circling_breakouts": [],
        "deconfliction_times": [],
        "min_pairwise_distances": [],
        "crashes": 0,
        "failed": 0,
        "completed": 0,
        "max_steps": 0,
        "timed_out": 0,
        "boundary_compliant": 0,
        "status_counts": Counter(),
    }

    num_missions = int(eval_cfg["num_missions"])
    deterministic = bool(eval_cfg.get("deterministic", True))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Evaluating MAPPO...", total=num_missions)

            for mission_index in range(num_missions):
                env.reset(options={"num_agents": int(config["drone_settings"]["num_drones"])})
                mission_definition = build_mission_definition(env, config)
                steps, reward, _, _, terminated, truncated, metrics = run_light_episode(
                    policy,
                    env,
                    int(env.max_steps),
                    label="MAPPO",
                    deterministic=deterministic,
                    show_progress=False,
                )
                mission_record = summarize_mission(
                    policy_path=eval_cfg["model_path"],
                    reward=reward,
                    steps=steps,
                    terminated=terminated,
                    truncated=truncated,
                    metrics=metrics,
                    mission_definition=mission_definition,
                    dt=float(env_cfg["dt"]),
                )
                mission_record["mission_index"] = mission_index
                mission_record["mission"] = mission_definition
                writer.append_mission(mission_record)
                update_history(history, mission_record)
                progress.update(task, advance=1)

        benchmark = build_benchmark(history)
        writer.finalize(benchmark)
        print_benchmark(benchmark)
    finally:
        env.close()
