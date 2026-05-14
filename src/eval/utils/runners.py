from __future__ import annotations

import os
from pathlib import Path

from test import create_video, run_and_record_episode, run_light_episode

from .results import build_mission_definition, summarize_mission


def run_policy_episode(
    *,
    policy,
    env,
    config: dict,
    policy_path: str,
    label: str,
    deterministic: bool,
    record_visual: bool = False,
    visual_path: str | None = None,
    show_progress: bool = False,
) -> dict:
    mission_definition = build_mission_definition(env, config)
    base_env = getattr(env, "base_env", env)
    if record_visual:
        uav_data, steps, reward, _, _, metrics = run_and_record_episode(
            policy,
            env,
            base_env.transformer,
            int(base_env.max_steps),
            deterministic=deterministic,
            include_planned_paths=True,
        )
        if visual_path:
            Path(visual_path).parent.mkdir(parents=True, exist_ok=True)
            create_video(
                uav_data,
                (base_env.max_lat, base_env.min_lon),
                (base_env.min_lat, base_env.max_lon),
                base_env.transformer,
                label,
                sum(stat.get("waypoints_reached", 0) for stat in metrics.get("mission_stats", [])),
                sum(stat.get("assigned_waypoints", 0) for stat in metrics.get("mission_stats", [])),
                visual_path,
                config,
                fps=int(config.get("eval", {}).get("skill_eval", {}).get("video_fps", 30)),
                speed_multiplier=int(
                    config.get("eval", {}).get("skill_eval", {}).get("video_speed_multiplier", 1)
                ),
                show_planned_paths=True,
            )
    else:
        steps, reward, _, _, _, _, metrics = run_light_episode(
            policy,
            env,
            int(base_env.max_steps),
            label=label,
            deterministic=deterministic,
            show_progress=show_progress,
        )
    episode_summary = metrics.get("episode_summary", {})
    terminated = bool(episode_summary.get("termination_reason") not in {None, "max_steps"})
    truncated = bool(episode_summary.get("termination_reason") == "max_steps")
    record = summarize_mission(
        policy_path=policy_path,
        reward=reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        metrics=metrics,
        mission_definition=mission_definition,
        dt=float(base_env.dt),
        label=label,
    )
    if visual_path:
        record["visual_path"] = os.path.abspath(visual_path)
    return record
