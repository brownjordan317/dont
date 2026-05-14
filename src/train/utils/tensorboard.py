from __future__ import annotations

import os
import time

import numpy as np
from rich.console import Console
import yaml

from train.utils.checkpoints import checkpoint_stem

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None

console = Console()

SUCCESS_TERMINATION_REASONS = {
    "completed",
    "survival_completed",
}


def is_successful_termination(termination_reason: str) -> bool:
    return str(termination_reason) in SUCCESS_TERMINATION_REASONS


def create_tensorboard_writer(train_cfg: dict, config: dict):
    tensorboard_cfg = train_cfg.get("tensorboard", {})
    if not bool(tensorboard_cfg.get("enabled", True)):
        return None

    if SummaryWriter is None:
        console.print(
            "[yellow]TensorBoard logging is enabled, but the `tensorboard` package "
            "is not installed. Install the updated requirements to enable it.[/yellow]"
        )
        return None

    base_log_dir = str(
        tensorboard_cfg.get(
            "log_dir",
            os.path.join(
                train_cfg["save_dir"],
                "tensorboard",
                checkpoint_stem(train_cfg["model_name"]),
            ),
        )
    )
    separate_runs = bool(tensorboard_cfg.get("separate_runs", True))
    if separate_runs:
        run_name = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
        log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(base_log_dir, exist_ok=True)
        latest_run_path = os.path.join(base_log_dir, "latest_run.txt")
        with open(latest_run_path, "w") as handle:
            handle.write(log_dir + "\n")
    else:
        log_dir = base_log_dir

    flush_secs = max(int(tensorboard_cfg.get("flush_secs", 30)), 1)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
    writer.add_text(
        "config/yaml",
        f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}\n```",
        global_step=0,
    )
    if separate_runs:
        console.print(f"[cyan]TensorBoard base dir:[/cyan] {base_log_dir}")
        console.print(f"[cyan]TensorBoard run dir:[/cyan] {log_dir}")
    else:
        console.print(f"[cyan]TensorBoard log dir:[/cyan] {log_dir}")
    return writer


def tensorboard_interval(
    train_cfg: dict,
    key: str,
    default: int,
) -> int:
    tensorboard_cfg = train_cfg.get("tensorboard", {})
    return max(int(tensorboard_cfg.get(key, default)), 1)


def log_episode_metrics(
    writer,
    *,
    train_cfg: dict,
    episode_index: int,
    episode_return: float,
    episode_length: int,
    final_metrics: dict,
    termination_reason: str,
) -> None:
    if writer is None:
        return

    episode_metrics_interval = tensorboard_interval(
        train_cfg,
        "episode_metrics_interval",
        5,
    )
    if (episode_index % episode_metrics_interval) != 0:
        return

    episode_detail_interval = tensorboard_interval(
        train_cfg,
        "episode_detail_interval",
        max(episode_metrics_interval * 4, 1),
    )
    log_detailed_metrics = (episode_index % episode_detail_interval) == 0

    def log_numeric_scalars(prefix: str, values) -> None:
        if isinstance(values, dict):
            for key, value in values.items():
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                log_numeric_scalars(child_prefix, value)
            return

        if isinstance(values, (int, float, np.integer, np.floating, bool)):
            writer.add_scalar(prefix, float(values), episode_index)

    mission_stats = final_metrics.get("mission_stats", [])
    episode_summary = final_metrics.get("episode_summary", {})
    reward_breakdown = final_metrics.get("reward_breakdown", {})
    safety = final_metrics.get("safety_violations", {})

    reached_waypoints = sum(
        int(drone.get("waypoints_reached", 0))
        for drone in mission_stats
    )
    remaining_waypoints = sum(
        int(drone.get("waypoints_remaining", 0))
        for drone in mission_stats
    )
    assigned_waypoints = sum(
        int(drone.get("assigned_waypoints", 0))
        for drone in mission_stats
    )
    completion_rate = (
        float(reached_waypoints / assigned_waypoints)
        if assigned_waypoints
        else 0.0
    )
    min_pairwise_distance = episode_summary.get("min_pairwise_distance_m")

    writer.add_scalar("episode/return", float(episode_return), episode_index)
    writer.add_scalar("episode/length", int(episode_length), episode_index)
    writer.add_scalar(
        "episode/completed",
        1.0 if is_successful_termination(termination_reason) else 0.0,
        episode_index,
    )
    writer.add_scalar(
        "episode/survival_completed",
        1.0 if termination_reason == "survival_completed" else 0.0,
        episode_index,
    )
    writer.add_scalar(
        "episode/safety_failure",
        1.0
        if termination_reason
        in {"critical_violation", "geofence_violation", "degenerate_survival_motion"}
        else 0.0,
        episode_index,
    )
    writer.add_scalar(
        "episode/degenerate_survival_motion",
        1.0 if termination_reason == "degenerate_survival_motion" else 0.0,
        episode_index,
    )
    writer.add_scalar(
        "episode/crashed",
        1.0
        if termination_reason == "critical_violation"
        else 0.0,
        episode_index,
    )
    writer.add_scalar(
        "episode/geofence_violation",
        1.0 if termination_reason == "geofence_violation" else 0.0,
        episode_index,
    )
    writer.add_scalar(
        "episode/max_steps_timeout",
        1.0 if termination_reason == "max_steps" else 0.0,
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoint_completion_rate",
        completion_rate,
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoints_reached",
        reached_waypoints,
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoints_remaining",
        remaining_waypoints,
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoint_throughput_per_min",
        float(episode_summary.get("waypoint_throughput_per_min", 0.0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/circling_steps",
        int(episode_summary.get("circling_steps_total", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/circling_breakouts",
        int(episode_summary.get("circling_breakouts_total", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoint_reapproach_steps",
        int(episode_summary.get("waypoint_reapproach_steps_total", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoint_reapproach_events",
        int(episode_summary.get("waypoint_reapproach_events_total", 0)),
        episode_index,
    )
    route_skill_steps = int(episode_summary.get("route_skill_steps_total", 0))
    avoid_skill_steps = int(episode_summary.get("avoid_skill_steps_total", 0))
    requested_route_skill_steps = int(
        episode_summary.get("requested_route_skill_steps_total", 0)
    )
    requested_avoid_skill_steps = int(
        episode_summary.get("requested_avoid_skill_steps_total", 0)
    )
    forced_avoid_steps = int(episode_summary.get("forced_avoid_steps_total", 0))
    forced_route_breakout_steps = int(
        episode_summary.get("forced_route_breakout_steps_total", 0)
    )
    avoid_loop_breakout_events = int(
        episode_summary.get("avoid_option_loop_breakout_events_total", 0)
    )
    total_skill_steps = route_skill_steps + avoid_skill_steps
    total_requested_skill_steps = requested_route_skill_steps + requested_avoid_skill_steps
    if total_skill_steps > 0:
        writer.add_scalar(
            "episode/route_skill_steps",
            route_skill_steps,
            episode_index,
        )
        writer.add_scalar(
            "episode/avoid_skill_steps",
            avoid_skill_steps,
            episode_index,
        )
        writer.add_scalar(
            "episode/route_skill_fraction",
            float(route_skill_steps / total_skill_steps),
            episode_index,
        )
        writer.add_scalar(
            "episode/avoid_skill_fraction",
            float(avoid_skill_steps / total_skill_steps),
            episode_index,
        )
        writer.add_scalar(
            "episode/forced_avoid_steps",
            forced_avoid_steps,
            episode_index,
        )
        writer.add_scalar(
            "episode/forced_avoid_fraction",
            float(forced_avoid_steps / total_skill_steps),
            episode_index,
        )
        writer.add_scalar(
            "episode/forced_route_breakout_steps",
            forced_route_breakout_steps,
            episode_index,
        )
        writer.add_scalar(
            "episode/forced_route_breakout_fraction",
            float(forced_route_breakout_steps / total_skill_steps),
            episode_index,
        )
        writer.add_scalar(
            "episode/avoid_option_loop_breakout_events",
            avoid_loop_breakout_events,
            episode_index,
        )
    if total_requested_skill_steps > 0:
        writer.add_scalar(
            "episode/requested_route_skill_steps",
            requested_route_skill_steps,
            episode_index,
        )
        writer.add_scalar(
            "episode/requested_avoid_skill_steps",
            requested_avoid_skill_steps,
            episode_index,
        )
        writer.add_scalar(
            "episode/requested_route_skill_fraction",
            float(requested_route_skill_steps / total_requested_skill_steps),
            episode_index,
        )
        writer.add_scalar(
            "episode/requested_avoid_skill_fraction",
            float(requested_avoid_skill_steps / total_requested_skill_steps),
            episode_index,
        )
    if not log_detailed_metrics:
        return

    writer.add_scalar(
        "episode/caution_events",
        int(safety.get("caution", {}).get("total_count", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/critical_events",
        int(safety.get("critical", {}).get("total_count", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/geofence_exits",
        int(safety.get("geofence", {}).get("total_count", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/geofence_outside_steps",
        int(safety.get("geofence", {}).get("outside_step_total", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/uavs_completed",
        int(episode_summary.get("uavs_completed", 0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/deconfliction_steps",
        float(episode_summary.get("deconfliction_steps_total", 0.0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/deconfliction_time_s",
        float(episode_summary.get("deconfliction_time_s", 0.0)),
        episode_index,
    )
    if min_pairwise_distance is not None:
        writer.add_scalar(
            "episode/min_pairwise_distance_m",
            float(min_pairwise_distance),
            episode_index,
        )

    log_numeric_scalars("episode_reward", reward_breakdown)
