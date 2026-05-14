from __future__ import annotations

from collections import deque
import os
import random
import time
from typing import List

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel

from config_utils import get_tuning_section, load_mode_config
from env.hrl_manager_env import HierarchicalManagerEnv
from env.hrl_skill_env import AvoidSkillTrainingEnv, RouteSkillTrainingEnv
from flight_engine.helpers import FlightMode
from mappo import MAPPOPolicy, RolloutBatch, build_rollout_batch
from mappo.runtime import (
    action_dict_from_step,
    sample_info,
    select_actions_for_envs,
)
from env.pettingzoo_env import MultiUAVParallelEnv
from train.utils.checkpoints import (
    best_checkpoint_filename,
    best_checkpoint_score,
    checkpoint_filename,
    checkpoint_stem,
    latest_checkpoint_filename,
    save_checkpoint,
    scheduled_learning_rate,
    should_stop_for_stale_best_checkpoint,
    should_update_best_checkpoint,
)
from train.utils.tensorboard import (
    create_tensorboard_writer,
    is_successful_termination,
    log_episode_metrics,
    tensorboard_interval,
)
from train.utils.ppo_update import update_policy

console = Console()


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_rollout() -> dict:
    return {
        "obs": [],
        "states": [],
        "raw_actions": [],
        "log_probs": [],
        "masks": [],
        "values": [],
        "rewards": [],
        "dones": [],
    }


def reward_vector_from_step(env, rewards: dict, active_agents: list[str]) -> np.ndarray:
    reward_vector = np.zeros(int(env.max_agents), dtype=np.float32)
    for agent in active_agents:
        idx = int(env.agent_name_to_index[agent])
        reward_vector[idx] = float(rewards.get(agent, 0.0))
    return reward_vector


def masked_mean_reward(reward_vector: np.ndarray, mask: np.ndarray) -> float:
    reward_vector = np.asarray(reward_vector, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    valid_count = float(mask.sum())
    if valid_count <= 0.0:
        return 0.0
    return float((reward_vector * mask).sum() / valid_count)


def policy_train_mask_after_step(env, fallback_mask: np.ndarray) -> np.ndarray:
    train_mask = getattr(env, "_last_policy_train_mask", None)
    if train_mask is None:
        return np.asarray(fallback_mask, dtype=np.float32).copy()
    train_mask = np.asarray(train_mask, dtype=np.float32)
    if train_mask.shape != np.asarray(fallback_mask).shape:
        return np.asarray(fallback_mask, dtype=np.float32).copy()
    return train_mask.copy()


def reset_training_env(env, *, seed=None, attempts: int = 3):
    attempts = max(int(attempts), 1)
    last_error: RuntimeError | None = None
    for attempt_idx in range(attempts):
        attempt_seed = None if seed is None else int(seed) + attempt_idx
        try:
            return env.reset(seed=attempt_seed)
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return env.reset(seed=seed)


def concat_rollout_batches(batches: List[RolloutBatch]) -> RolloutBatch:
    return RolloutBatch(
        obs=np.concatenate([batch.obs for batch in batches], axis=0),
        states=np.concatenate([batch.states for batch in batches], axis=0),
        raw_actions=np.concatenate([batch.raw_actions for batch in batches], axis=0),
        log_probs=np.concatenate([batch.log_probs for batch in batches], axis=0),
        masks=np.concatenate([batch.masks for batch in batches], axis=0),
        values=np.concatenate([batch.values for batch in batches], axis=0),
        rewards=np.concatenate([batch.rewards for batch in batches], axis=0),
        returns=np.concatenate([batch.returns for batch in batches], axis=0),
        advantages=np.concatenate([batch.advantages for batch in batches], axis=0),
    )


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        console.print(
            "[yellow]CUDA was requested but is not available; falling back to CPU.[/yellow]"
        )
        return "cpu"
    return requested_device


def configure_torch_runtime(train_cfg: dict, device: str) -> None:
    if device != "cpu":
        return

    requested_threads = train_cfg.get("torch_num_threads")
    if requested_threads is None:
        requested_threads = 4

    requested_threads = max(int(requested_threads), 1)
    current_threads = torch.get_num_threads()
    if current_threads != requested_threads:
        torch.set_num_threads(requested_threads)
        console.print(
            f"[cyan]PyTorch CPU threads:[/cyan] {current_threads} -> {requested_threads}"
        )


def build_training_env(
    tuning: dict,
    *,
    env_overrides: dict | None = None,
    reward_overrides: dict | None = None,
    guidance_overrides: dict | None = None,
) -> MultiUAVParallelEnv:
    env_cfg = dict(tuning["env"])
    if env_overrides:
        env_cfg.update(env_overrides)
    reward_cfg = dict(tuning["rewards"])
    if reward_overrides:
        reward_cfg.update(reward_overrides)
    guidance_cfg = dict(tuning["guidance"])
    if guidance_overrides:
        guidance_cfg.update(guidance_overrides)
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
        mission_waypoint_count=env_cfg["mission_waypoint_count"],
        mission_waypoint_count_min=env_cfg.get("mission_waypoint_count_min"),
        mission_waypoint_count_max=env_cfg.get("mission_waypoint_count_max"),
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
        terminate_on_all_waypoints_complete=env_cfg.get(
            "terminate_on_all_waypoints_complete",
            True,
        ),
        terminate_on_critical_violation=env_cfg.get(
            "terminate_on_critical_violation",
            True,
        ),
        terminate_on_geofence_violation=env_cfg.get(
            "terminate_on_geofence_violation",
            True,
        ),
        geofence_breach_grace_steps=env_cfg.get("geofence_breach_grace_steps", 1),
        refill_random_waypoints_on_completion=env_cfg.get(
            "refill_random_waypoints_on_completion",
            False,
        ),
        disable_waypoint_navigation=env_cfg.get(
            "disable_waypoint_navigation",
            False,
        ),
        allow_live_waypoint_updates=env_cfg.get("allow_live_waypoint_updates", False),
        flight_config=tuning["flight"],
        reward_config=reward_cfg,
        guidance_config=guidance_cfg,
    )


def training_role(train_cfg: dict) -> str:
    return str(train_cfg.get("role", "manager")).strip().lower()


def role_display_name(role: str) -> str:
    return {
        "route_skill": "Route Skill",
        "avoid_skill": "Avoid Skill",
        "manager": "HRL Manager",
    }.get(role, role.replace("_", " ").title())


def role_actor_type(role: str) -> str:
    return "categorical_skill" if role == "manager" else "gaussian"


def role_critic_value_dim(role: str, env) -> int:
    if role in {"avoid_skill", "manager"}:
        return int(env.max_agents)
    return 1


def direct_env_from_training_env(env):
    current = env
    while hasattr(current, "base_env"):
        current = getattr(current, "base_env")
    return current


def guidance_action_dict(env, vector_name: str) -> dict:
    base_env = direct_env_from_training_env(env)
    guidance_vector = getattr(base_env, vector_name)
    return {
        agent: np.asarray(
            [guidance_vector[base_env.agent_name_to_index[agent]]],
            dtype=np.float32,
        )
        for agent in env.agents
    }


def route_reference_action_dict(env) -> dict:
    return guidance_action_dict(env, "_last_reference_action_vector")


def avoid_reference_action_dict(env) -> dict:
    return guidance_action_dict(env, "_last_avoidance_action_vector")


def guidance_warmstart_targets(
    env,
    action_dim: int,
    *,
    vector_name: str,
    require_waypoint: bool,
) -> tuple[np.ndarray, np.ndarray]:
    base_env = direct_env_from_training_env(env)
    guidance_vector = getattr(base_env, vector_name)
    targets = np.zeros((env.max_agents, action_dim), dtype=np.float32)
    masks = np.zeros(env.max_agents, dtype=np.float32)
    for agent in env.agents:
        idx = env.agent_name_to_index[agent]
        aircraft = base_env.aircraft_by_agent[agent]
        targets[idx, 0] = float(guidance_vector[idx])
        has_target = (
            aircraft.waypoint_manager.current_waypoint is not None
            or not require_waypoint
        )
        if has_target and aircraft.flight_mode != FlightMode.LOITERING:
            masks[idx] = 1.0
    return targets, masks


def route_warmstart_targets(env, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return guidance_warmstart_targets(
        env,
        action_dim,
        vector_name="_last_reference_action_vector",
        require_waypoint=True,
    )


def avoid_warmstart_targets(env, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return guidance_warmstart_targets(
        env,
        action_dim,
        vector_name="_last_avoidance_action_vector",
        require_waypoint=False,
    )


def manager_warmstart_targets(env, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    env.get_obs_matrix()
    targets = np.zeros((env.max_agents, action_dim), dtype=np.float32)
    masks = np.zeros(env.max_agents, dtype=np.float32)
    feature_names = getattr(env, "manager_feature_names", ())
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    extra_cache = getattr(env, "_manager_extra_cache", None)

    for agent in env.agents:
        idx = env.agent_name_to_index[agent]
        aircraft = env.base_env.aircraft_by_agent[agent]
        action_controls_aircraft = bool(
            aircraft.flight_mode != FlightMode.LOITERING
            and aircraft.waypoint_manager.current_waypoint is not None
        )
        if not action_controls_aircraft:
            continue

        current_pressure = 0.0
        boundary_pressure = 0.0
        handoff_safe = 1.0
        avoid_active = 0.0
        if extra_cache is not None:
            if "current_avoidance_pressure" in feature_index:
                current_pressure = float(
                    extra_cache[idx, feature_index["current_avoidance_pressure"]]
                )
            if "boundary_avoidance_pressure" in feature_index:
                boundary_pressure = float(
                    extra_cache[idx, feature_index["boundary_avoidance_pressure"]]
                )
            if "avoid_option_handoff_safe" in feature_index:
                handoff_safe = float(
                    extra_cache[idx, feature_index["avoid_option_handoff_safe"]]
                )
            if "avoid_option_active" in feature_index:
                avoid_active = float(
                    extra_cache[idx, feature_index["avoid_option_active"]]
                )

        avoid_threshold = max(
            float(getattr(env, "avoid_option_handoff_pressure_threshold", 0.20)),
            float(getattr(env, "avoid_option_handoff_boundary_pressure_threshold", 0.20)),
        )
        should_avoid = bool(
            current_pressure > avoid_threshold
            or boundary_pressure > avoid_threshold
            or (avoid_active >= 0.5 and handoff_safe < 0.5)
        )
        skill_idx = env.avoid_skill_index if should_avoid else env.route_skill_index
        targets[idx, skill_idx] = 1.0
        masks[idx] = 1.0
    return targets, masks


def manager_reference_action_dict(env) -> dict:
    targets, masks = manager_warmstart_targets(env, env.action_dim)
    return {
        agent: targets[env.agent_name_to_index[agent]]
        for agent in env.agents
        if float(masks[env.agent_name_to_index[agent]]) > 0.0
    }


def run_manager_warmstart(
    *,
    policy: MAPPOPolicy,
    envs: List,
    train_cfg: dict,
    seed: int | None,
    writer,
    starting_timestep: int,
) -> dict:
    return run_guidance_warmstart(
        policy=policy,
        envs=envs,
        train_cfg=train_cfg,
        seed=seed,
        writer=writer,
        starting_timestep=starting_timestep,
        prefix="manager",
        display_name="Manager warm-start",
        target_fn=manager_warmstart_targets,
        action_dict_fn=manager_reference_action_dict,
        seed_offset=30_000,
    )


def run_guidance_warmstart(
    *,
    policy: MAPPOPolicy,
    envs: List,
    train_cfg: dict,
    seed: int | None,
    writer,
    starting_timestep: int,
    prefix: str,
    display_name: str,
    target_fn,
    action_dict_fn,
    seed_offset: int,
) -> dict:
    max_steps = max(int(train_cfg.get(f"{prefix}_warmstart_max_steps", 0)), 0)
    if max_steps <= 0 or not envs:
        return {"timesteps": 0, "updates": 0, "loss": 0.0, "mae": 0.0}

    num_envs = max(
        1,
        min(int(train_cfg.get(f"{prefix}_warmstart_num_envs", len(envs))), len(envs)),
    )
    warm_envs = envs[:num_envs]
    learning_rate = float(
        train_cfg.get(f"{prefix}_warmstart_learning_rate", train_cfg["learning_rate"])
    )
    batch_steps = max(int(train_cfg.get(f"{prefix}_warmstart_batch_steps", 32)), 1)
    log_interval = max(int(train_cfg.get(f"{prefix}_warmstart_log_interval", 10_000)), 1)
    max_grad_norm = float(train_cfg["max_grad_norm"])
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=learning_rate)

    for env_idx, env in enumerate(warm_envs):
        env_seed = None if seed is None else int(seed) + int(seed_offset) + env_idx
        reset_training_env(env, seed=env_seed)

    timesteps = 0
    updates = 0
    losses: list[float] = []
    maes: list[float] = []
    obs_batches: list[np.ndarray] = []
    state_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    last_log_step = 0

    while timesteps < max_steps:
        for env in warm_envs:
            active_indices = [env.agent_name_to_index[agent] for agent in env.agents]
            if not active_indices:
                reset_training_env(env)
                continue

            targets, masks = target_fn(env, policy.action_dim)
            valid_indices = [
                idx
                for idx in active_indices
                if float(masks[idx]) > 0.0
            ]
            if valid_indices:
                obs_batches.append(env.get_obs_matrix()[valid_indices])
                state_batches.append(np.asarray(env.state(), dtype=np.float32))
                target_batches.append(targets[valid_indices])

            _, _, terminations, truncations, _ = env.step(action_dict_fn(env))
            timesteps += 1
            done = bool(any(terminations.values()) or any(truncations.values()))
            if done:
                reset_training_env(env)

            if len(obs_batches) >= batch_steps or timesteps >= max_steps:
                merged_obs = np.concatenate(obs_batches, axis=0) if obs_batches else None
                if merged_obs is not None:
                    merged_states = np.asarray(state_batches, dtype=np.float32)
                    merged_targets = np.concatenate(target_batches, axis=0)
                    policy.update_normalizers(merged_obs, merged_states)
                    obs_tensor = torch.as_tensor(
                        policy.normalize_obs(merged_obs),
                        dtype=torch.float32,
                        device=policy.device,
                    )
                    target_tensor = torch.as_tensor(
                        merged_targets,
                        dtype=torch.float32,
                        device=policy.device,
                    )
                    optimizer.zero_grad()
                    predictions = policy.actor_mean_action(obs_tensor)
                    loss = torch.nn.functional.smooth_l1_loss(
                        predictions,
                        target_tensor,
                        beta=float(train_cfg.get(f"{prefix}_warmstart_delta", 0.1)),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        mae = torch.abs(predictions - target_tensor).mean()
                    losses.append(float(loss.item()))
                    maes.append(float(mae.item()))
                    updates += 1
                    global_step = starting_timestep + timesteps
                    if writer is not None:
                        writer.add_scalar(f"warmstart/{prefix}_loss", losses[-1], global_step)
                        writer.add_scalar(f"warmstart/{prefix}_mae", maes[-1], global_step)

                obs_batches.clear()
                state_batches.clear()
                target_batches.clear()

            if timesteps - last_log_step >= log_interval:
                console.print(
                    f"[cyan]{display_name}:[/cyan] "
                    f"{timesteps:,}/{max_steps:,} steps | "
                    f"loss={float(np.mean(losses[-100:])) if losses else 0.0:.4f} | "
                    f"mae={float(np.mean(maes[-100:])) if maes else 0.0:.4f}"
                )
                last_log_step = timesteps

            if timesteps >= max_steps:
                break

    return {
        "timesteps": timesteps,
        "updates": updates,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mae": float(np.mean(maes)) if maes else 0.0,
    }


def run_route_warmstart(
    *,
    policy: MAPPOPolicy,
    envs: List,
    train_cfg: dict,
    seed: int | None,
    writer,
    starting_timestep: int,
) -> dict:
    return run_guidance_warmstart(
        policy=policy,
        envs=envs,
        train_cfg=train_cfg,
        seed=seed,
        writer=writer,
        starting_timestep=starting_timestep,
        prefix="route_skill",
        display_name="Route warm-start",
        target_fn=route_warmstart_targets,
        action_dict_fn=route_reference_action_dict,
        seed_offset=10_000,
    )


def run_avoid_warmstart(
    *,
    policy: MAPPOPolicy,
    envs: List,
    train_cfg: dict,
    seed: int | None,
    writer,
    starting_timestep: int,
) -> dict:
    return run_guidance_warmstart(
        policy=policy,
        envs=envs,
        train_cfg=train_cfg,
        seed=seed,
        writer=writer,
        starting_timestep=starting_timestep,
        prefix="avoid_skill",
        display_name="Avoid warm-start",
        target_fn=avoid_warmstart_targets,
        action_dict_fn=avoid_reference_action_dict,
        seed_offset=20_000,
    )


def load_manager_skill_policies(train_cfg: dict, device: str) -> tuple[MAPPOPolicy, MAPPOPolicy]:
    manager_cfg = train_cfg.get("manager", {})
    route_skill_path = manager_cfg.get("route_skill_path")
    avoid_skill_path = manager_cfg.get("avoid_skill_path")
    if not route_skill_path or not avoid_skill_path:
        raise ValueError(
            "Manager training requires manager.route_skill_path, "
            "and manager.avoid_skill_path."
        )
    missing_paths = [
        path
        for path in (str(route_skill_path), str(avoid_skill_path))
        if not os.path.exists(path)
    ]
    if missing_paths:
        missing_list = ", ".join(missing_paths)
        raise FileNotFoundError(
            "Manager training requires pretrained skill checkpoints. Missing: "
            f"{missing_list}. Run `make train` to auto-bootstrap the full HRL stack, "
            "or run the route and avoid skill trainers first."
        )

    validate_route_checkpoint_contract(str(route_skill_path), device=device)
    validate_avoid_checkpoint_contract(str(avoid_skill_path), device=device)
    route_policy = MAPPOPolicy.load(str(route_skill_path), device=device)
    avoid_policy = MAPPOPolicy.load(str(avoid_skill_path), device=device)
    return route_policy, avoid_policy


def checkpoint_train_config(path: str, device: str = "cpu") -> dict:
    payload = MAPPOPolicy._load_checkpoint_payload(str(path), device)
    return (
        payload.get("extra", {})
        .get("config", {})
        .get("train", {})
    )


def route_checkpoint_contract_mismatches(
    path: str,
    expected_train_cfg: dict,
    *,
    expected_tuning_cfg: dict | None = None,
    device: str = "cpu",
) -> list[str]:
    saved_train_cfg = checkpoint_train_config(path, device=device)
    expected_role = str(expected_train_cfg.get("role", "route_skill")).strip().lower()
    prefix = "route_skill"
    mismatches = []
    if str(saved_train_cfg.get("role", "")).strip().lower() != expected_role:
        mismatches.append(f"role is not {expected_role}")
    for key in (
        f"{prefix}_caution_dist",
        "route_skill_completion_bonus",
        "route_skill_geofence_penalty",
        "route_skill_harsh_turn_penalty",
        "route_skill_harsh_turn_threshold",
        "route_skill_geofence_breach_grace_steps",
        "route_skill_geofence_breach_penalty",
        "route_skill_reference_route_boundary_margin_m",
        "route_skill_reference_route_generation_attempts",
        "route_skill_time_pressure_penalty",
        "route_skill_remaining_waypoint_penalty",
        "route_skill_warmstart_max_steps",
        "route_skill_warmstart_num_envs",
        "route_skill_warmstart_learning_rate",
        "route_skill_warmstart_batch_steps",
        "route_skill_warmstart_delta",
        f"{prefix}_mission_waypoint_count",
        f"{prefix}_box_size_m",
        "log_std_init",
    ):
        if key not in expected_train_cfg:
            continue
        expected_value = float(expected_train_cfg[key])
        actual_value = saved_train_cfg.get(key)
        if actual_value is None or not np.isclose(float(actual_value), expected_value):
            mismatches.append(f"{key} checkpoint={actual_value!r} expected={expected_value!r}")
    for key in (
        "route_skill_geofence_breach_terminates",
        "route_skill_require_reference_route_in_bounds",
    ):
        if key not in expected_train_cfg:
            continue
        expected_value = bool(expected_train_cfg[key])
        actual_value = saved_train_cfg.get(key)
        if actual_value is None or bool(actual_value) != expected_value:
            mismatches.append(f"{key} checkpoint={actual_value!r} expected={expected_value!r}")
    return mismatches


def avoid_checkpoint_contract_mismatches(
    path: str,
    expected_train_cfg: dict,
    *,
    expected_tuning_cfg: dict | None = None,
    device: str = "cpu",
) -> list[str]:
    saved_train_cfg = checkpoint_train_config(path, device=device)
    mismatches = []
    if str(saved_train_cfg.get("role", "")).strip().lower() != "avoid_skill":
        mismatches.append("role is not avoid_skill")

    payload = MAPPOPolicy._load_checkpoint_payload(path, device)
    actual_critic_value_dim = int(
        payload.get(
            "critic_value_dim",
            MAPPOPolicy._infer_critic_value_dim(payload.get("critic_state_dict", {})),
        )
    )
    expected_env_cfg = (expected_tuning_cfg or {}).get("env", {})
    expected_critic_value_dim = int(
        expected_env_cfg.get(
            "max_agents",
            expected_env_cfg.get(
                "shared",
                {},
            ).get(
                "max_agents",
                expected_train_cfg.get("avoid_skill_survival_agents", 1),
            ),
        )
    )
    if actual_critic_value_dim != expected_critic_value_dim:
        mismatches.append(
            "critic_value_dim "
            f"checkpoint={actual_critic_value_dim!r} "
            f"expected={expected_critic_value_dim!r}"
        )

    for key in (
        "avoid_skill_survival_degenerate_motion_terminates",
        "avoid_skill_survival_conflict_require_valid_candidate",
    ):
        expected_value = bool(expected_train_cfg.get(key, True))
        actual_value = saved_train_cfg.get(key)
        if actual_value is None or bool(actual_value) != expected_value:
            mismatches.append(f"{key} checkpoint={actual_value!r} expected={expected_value!r}")

    for key in ("avoid_skill_survival_scenario_mode",):
        expected_value = str(expected_train_cfg.get(key, "random")).strip().lower()
        actual_value = saved_train_cfg.get(key)
        if actual_value is None or str(actual_value).strip().lower() != expected_value:
            mismatches.append(f"{key} checkpoint={actual_value!r} expected={expected_value!r}")

    for key in (
        "avoid_skill_survival_agents_min",
        "avoid_skill_survival_agents",
        "avoid_skill_env_dt_s",
        "avoid_skill_survival_box_size_min_m",
        "avoid_skill_survival_box_size_max_m",
        "avoid_skill_survival_conflict_edge_margin_m",
        "avoid_skill_survival_conflict_min_pairs",
        "avoid_skill_survival_conflict_target_cpa_m",
        "avoid_skill_survival_conflict_max_trigger_cpa_m",
        "avoid_skill_survival_conflict_min_feasible_cpa_m",
        "avoid_skill_survival_start_margin_m",
        "avoid_skill_survival_max_steps",
        "avoid_skill_survival_step_reward",
        "avoid_skill_survival_crash_penalty",
        "avoid_skill_survival_team_failure_penalty_fraction",
        "avoid_skill_survival_generation_attempts",
        "avoid_skill_survival_min_boundary_time_ratio",
        "avoid_skill_survival_start_validation_steps",
        "avoid_skill_survival_boundary_escape_steps",
        "avoid_skill_survival_degenerate_min_steps",
        "avoid_skill_survival_degenerate_consecutive_steps",
        "avoid_skill_survival_degenerate_turn_fraction_threshold",
        "avoid_skill_survival_degenerate_turn_agreement_threshold",
        "avoid_skill_warmstart_max_steps",
        "avoid_skill_warmstart_num_envs",
        "avoid_skill_warmstart_learning_rate",
        "avoid_skill_warmstart_batch_steps",
        "avoid_skill_warmstart_delta",
    ):
        if key not in expected_train_cfg:
            continue
        expected_value = float(expected_train_cfg[key])
        actual_value = saved_train_cfg.get(key)
        if actual_value is None or not np.isclose(float(actual_value), expected_value):
            mismatches.append(f"{key} checkpoint={actual_value!r} expected={expected_value!r}")

    return mismatches


def validate_avoid_checkpoint_contract(
    path: str,
    *,
    expected_train_cfg: dict | None = None,
    device: str = "cpu",
) -> None:
    if expected_train_cfg is None:
        expected_train_cfg = load_mode_config("train_avoid_skill")["train"]
    mismatches = avoid_checkpoint_contract_mismatches(
        path,
        expected_train_cfg,
        expected_tuning_cfg=load_mode_config("train_avoid_skill").get("tuning", {}),
        device=device,
    )
    if mismatches:
        raise ValueError(
            "Avoid skill checkpoint was trained with an outdated/incompatible "
            "avoid-skill contract: "
            + "; ".join(mismatches)
            + ". Retrain it with `make train_avoid_skill` before manager training."
        )


def validate_route_checkpoint_contract(
    path: str,
    *,
    expected_train_cfg: dict | None = None,
    device: str = "cpu",
) -> None:
    if expected_train_cfg is None:
        expected_train_cfg = load_mode_config("train_route_skill")["train"]
    mismatches = route_checkpoint_contract_mismatches(
        path,
        expected_train_cfg,
        expected_tuning_cfg=load_mode_config("train_route_skill").get("tuning", {}),
        device=device,
    )
    if mismatches:
        raise ValueError(
            "Route skill checkpoint was trained with an outdated/incompatible "
            "route-skill contract: "
            + "; ".join(mismatches)
            + ". Retrain it with `make train_route_skill` before manager training."
        )


def infer_state_max_agents(*, state_dim: int, obs_dim: int) -> int:
    denom = int(obs_dim) + 1
    if denom <= 0:
        return 0
    numerator = int(state_dim) - 2
    if numerator <= 0 or numerator % denom != 0:
        return 0
    return int(numerator // denom)


def obs_self_feature_mapping(
    *,
    source_obs_dim: int,
    target_env,
) -> list[tuple[int, int]]:
    base_env = direct_env_from_training_env(target_env)
    if not all(
        hasattr(base_env, attr)
        for attr in (
            "obs_stack_size",
            "base_obs_dim",
            "self_feature_count",
        )
    ):
        return []

    obs_stack_size = int(base_env.obs_stack_size)
    target_base_obs_dim = int(base_env.base_obs_dim)
    self_feature_count = int(base_env.self_feature_count)
    if obs_stack_size <= 0 or target_base_obs_dim <= 0 or self_feature_count <= 0:
        return []
    if int(source_obs_dim) % obs_stack_size != 0:
        return []

    source_base_obs_dim = int(source_obs_dim) // obs_stack_size
    copied_self_features = min(self_feature_count, source_base_obs_dim, target_base_obs_dim)
    if copied_self_features <= 0:
        return []

    mapping: list[tuple[int, int]] = []
    for stack_idx in range(obs_stack_size):
        source_stack_offset = stack_idx * source_base_obs_dim
        target_stack_offset = stack_idx * target_base_obs_dim
        for feature_idx in range(copied_self_features):
            mapping.append(
                (
                    source_stack_offset + feature_idx,
                    target_stack_offset + feature_idx,
                )
            )
    return mapping


def validate_skill_policy(skill_name: str, policy: MAPPOPolicy, base_env: MultiUAVParallelEnv) -> None:
    mismatches = []
    allows_projected_skill_obs = bool(
        obs_self_feature_mapping(
            source_obs_dim=int(policy.obs_dim),
            target_env=base_env,
        )
    )
    allows_projected_skill_state = (
        allows_projected_skill_obs
        and infer_state_max_agents(
            state_dim=int(policy.state_dim),
            obs_dim=int(policy.obs_dim),
        )
        > 0
    )
    if int(policy.obs_dim) != int(base_env.obs_dim) and not allows_projected_skill_obs:
        mismatches.append(f"obs_dim checkpoint={policy.obs_dim} env={base_env.obs_dim}")
    if int(policy.state_dim) != int(base_env.state_dim) and not allows_projected_skill_state:
        mismatches.append(f"state_dim checkpoint={policy.state_dim} env={base_env.state_dim}")
    if int(policy.action_dim) != 1:
        mismatches.append(f"action_dim checkpoint={policy.action_dim} expected=1")
    if mismatches:
        raise ValueError(
            f"{skill_name} checkpoint is incompatible with the base training environment: "
            + "; ".join(mismatches)
        )


def build_role_training_env(
    *,
    tuning: dict,
    train_cfg: dict,
    role: str,
    device: str,
    route_skill_policy: MAPPOPolicy | None = None,
    avoid_skill_policy: MAPPOPolicy | None = None,
):
    env_overrides = None
    reward_overrides = None
    guidance_overrides = None
    if role == "route_skill":
        prefix = "route_skill"
        max_agents = int(tuning["env"]["max_agents"])
        route_num_agents = int(train_cfg.get(f"{prefix}_training_num_agents", train_cfg.get("route_skill_training_num_agents", 1)))
        route_num_agents = int(np.clip(route_num_agents, 1, max_agents))
        env_overrides = {
            "min_agents": route_num_agents,
            "max_agents": route_num_agents,
            "mission_waypoint_count": int(
                train_cfg.get(
                    f"{prefix}_mission_waypoint_count",
                    tuning["env"].get("mission_waypoint_count", 3),
                )
            ),
            "mission_waypoint_count_min": int(
                train_cfg.get(
                    f"{prefix}_mission_waypoint_count",
                    tuning["env"].get("mission_waypoint_count_min", 3),
                )
            ),
            "mission_waypoint_count_max": int(
                train_cfg.get(
                    f"{prefix}_mission_waypoint_count",
                    tuning["env"].get("mission_waypoint_count_max", 3),
                )
            ),
            "box_min_m": float(
                train_cfg.get(
                    f"{prefix}_box_size_m",
                    tuning["env"].get("box_max_m", 2000.0),
                )
            ),
            "box_max_m": float(
                train_cfg.get(
                    f"{prefix}_box_size_m",
                    tuning["env"].get("box_max_m", 2000.0),
                )
            ),
            "separation_dist": float(
                train_cfg.get(
                    f"{prefix}_separation_dist",
                    tuning["env"]["separation_dist"],
                )
            ),
        }
        reward_overrides = {
            "completion_bonus": float(
                train_cfg.get("route_skill_completion_bonus", 1800.0)
            ),
            "geofence_penalty": float(
                train_cfg.get(
                    "route_skill_geofence_penalty",
                    tuning["rewards"].get("geofence_penalty", 28.0),
                )
            ),
            "harsh_turn_penalty": float(
                train_cfg.get(
                    "route_skill_harsh_turn_penalty",
                    tuning["rewards"].get("harsh_turn_penalty", 0.35),
                )
            ),
            "harsh_turn_threshold": float(
                train_cfg.get(
                    "route_skill_harsh_turn_threshold",
                    tuning["rewards"].get("harsh_turn_threshold", 0.6),
                )
            ),
        }
    elif role == "avoid_skill":
        survival_agents = int(train_cfg["avoid_skill_survival_agents"])
        survival_box_size_min_m = float(train_cfg["avoid_skill_survival_box_size_min_m"])
        survival_box_size_max_m = float(train_cfg["avoid_skill_survival_box_size_max_m"])
        survival_max_steps = int(
            train_cfg.get(
                "avoid_skill_survival_max_steps",
                tuning["env"]["max_steps"],
            )
        )
        survival_dt = float(
            train_cfg.get(
                "avoid_skill_env_dt_s",
                tuning["env"]["dt"],
            )
        )
        env_overrides = {
            "dt": survival_dt,
            "min_agents": min(
                int(tuning["env"].get("min_agents", 2)),
                survival_agents,
            ),
            "max_agents": max(
                int(tuning["env"].get("max_agents", survival_agents)),
                survival_agents,
            ),
            "mission_waypoint_count": 0,
            "mission_waypoint_count_min": 0,
            "mission_waypoint_count_max": 0,
            "box_min_m": survival_box_size_min_m,
            "box_max_m": survival_box_size_max_m,
            "max_steps": survival_max_steps,
            "timeout_max_steps": survival_max_steps,
            "timeout_scale_with_mission_size": False,
            "timeout_scale_with_route_distance": False,
            "terminate_on_all_waypoints_complete": False,
            "refill_random_waypoints_on_completion": False,
            "disable_waypoint_navigation": True,
        }
    elif role == "manager":
        manager_cfg = train_cfg.get("manager", {})
        env_overrides = {
            "dt": float(manager_cfg["env_dt_s"]),
            "max_steps": int(manager_cfg["max_steps"]),
            "timeout_max_steps": int(manager_cfg["timeout_max_steps"]),
        }

    base_env = build_training_env(
        tuning,
        env_overrides=env_overrides,
        reward_overrides=reward_overrides,
        guidance_overrides=guidance_overrides,
    )
    if role == "route_skill":
        return RouteSkillTrainingEnv(base_env=base_env, train_cfg=train_cfg)
    if role == "avoid_skill":
        return AvoidSkillTrainingEnv(base_env=base_env, train_cfg=train_cfg)
    if role != "manager":
        return base_env

    if route_skill_policy is None or avoid_skill_policy is None:
        raise ValueError("Manager role requires route and avoid skill policies.")

    validate_skill_policy("Route skill", route_skill_policy, base_env)
    validate_skill_policy("Avoid skill", avoid_skill_policy, base_env)
    manager_cfg = train_cfg.get("manager", {})
    manager_reset_options = None
    if str(manager_cfg.get("scenario_mode", "")).strip():
        manager_reset_options = {
            "scenario_mode": str(manager_cfg["scenario_mode"]),
            "num_agents": int(manager_cfg["conflict_num_agents"]),
            "box_width_m": float(manager_cfg["conflict_box_size_m"]),
            "box_height_m": float(manager_cfg["conflict_box_size_m"]),
            "mission_waypoint_count": int(manager_cfg["conflict_mission_waypoint_count"]),
            "conflict_start_radius_fraction": float(manager_cfg["conflict_start_radius_fraction"]),
            "conflict_crossing_radius_fraction": float(manager_cfg["conflict_crossing_radius_fraction"]),
            "conflict_boundary_radius_fraction": float(manager_cfg["conflict_boundary_radius_fraction"]),
            "conflict_boundary_route_offset_m": float(manager_cfg["conflict_boundary_route_offset_m"]),
            "conflict_boundary_focus_probability": float(manager_cfg["conflict_boundary_focus_probability"]),
            "conflict_boundary_focus_center_fraction": float(manager_cfg["conflict_boundary_focus_center_fraction"]),
            "conflict_boundary_focus_target_time_ratio": float(manager_cfg["conflict_boundary_focus_target_time_ratio"]),
            "conflict_boundary_focus_max_time_ratio": float(manager_cfg["conflict_boundary_focus_max_time_ratio"]),
            "conflict_boundary_focus_scored_legs": int(manager_cfg["conflict_boundary_focus_scored_legs"]),
            "conflict_generation_attempts": int(manager_cfg["conflict_generation_attempts"]),
            "conflict_require_valid_candidate": bool(manager_cfg["conflict_require_valid_candidate"]),
            "conflict_require_boundary_viability": bool(manager_cfg["conflict_require_boundary_viability"]),
            "conflict_min_boundary_time_ratio": float(manager_cfg["conflict_min_boundary_time_ratio"]),
            "conflict_min_feasible_cpa_m": float(manager_cfg["conflict_min_feasible_cpa_m"]),
            "conflict_target_cpa_m": float(manager_cfg["conflict_target_cpa_m"]),
            "conflict_max_trigger_cpa_m": float(manager_cfg["conflict_max_trigger_cpa_m"]),
            "conflict_min_pairs": int(manager_cfg["conflict_min_pairs"]),
            "conflict_target_leg_time_s": float(manager_cfg["conflict_target_leg_time_s"]),
            "conflict_phase_jitter_rad": float(manager_cfg["conflict_phase_jitter_rad"]),
            "conflict_first_wp_lateral_jitter_m": float(manager_cfg["conflict_first_wp_lateral_jitter_m"]),
            "conflict_follow_lateral_offset_m": float(manager_cfg["conflict_follow_lateral_offset_m"]),
        }
    return HierarchicalManagerEnv(
        base_env=base_env,
        route_skill_policy=route_skill_policy,
        avoid_skill_policy=avoid_skill_policy,
        skill_deterministic=bool(manager_cfg["skill_deterministic"]),
        avoid_option_sticky_enabled=bool(manager_cfg["avoid_option_sticky_enabled"]),
        avoid_option_min_steps=int(manager_cfg["avoid_option_min_steps"]),
        avoid_option_handoff_pressure_threshold=float(
            manager_cfg["avoid_option_handoff_pressure_threshold"]
        ),
        avoid_option_handoff_boundary_pressure_threshold=float(
            manager_cfg["avoid_option_handoff_boundary_pressure_threshold"]
        ),
        avoid_option_handoff_min_separation_ratio=float(
            manager_cfg["avoid_option_handoff_min_separation_ratio"]
        ),
        avoid_option_loop_breakout_enabled=bool(
            manager_cfg.get("avoid_option_loop_breakout_enabled", True)
        ),
        avoid_option_loop_breakout_min_steps=int(
            manager_cfg.get("avoid_option_loop_breakout_min_steps", 24)
        ),
        avoid_option_loop_breakout_turns=float(
            manager_cfg.get("avoid_option_loop_breakout_turns", 0.35)
        ),
        avoid_option_loop_breakout_max_displacement_efficiency=float(
            manager_cfg.get(
                "avoid_option_loop_breakout_max_displacement_efficiency",
                0.75,
            )
        ),
        avoid_option_loop_breakout_hazard_threshold=float(
            manager_cfg.get("avoid_option_loop_breakout_hazard_threshold", 1.10)
        ),
        avoid_option_loop_breakout_boundary_threshold=float(
            manager_cfg.get("avoid_option_loop_breakout_boundary_threshold", 1.25)
        ),
        avoid_option_loop_breakout_min_separation_ratio=float(
            manager_cfg.get("avoid_option_loop_breakout_min_separation_ratio", 0.35)
        ),
        avoid_option_loop_breakout_route_steps=int(
            manager_cfg.get("avoid_option_loop_breakout_route_steps", 120)
        ),
        reset_options_template=manager_reset_options,
    )


def train(config):
    train_cfg = config["train"]
    tuning = get_tuning_section(config, "train")
    role = training_role(train_cfg)
    os.makedirs(train_cfg["save_dir"], exist_ok=True)

    seed = train_cfg.get("seed")
    if seed is not None:
        seed = int(seed)
    set_global_seed(seed)

    device = resolve_device(str(train_cfg.get("device", "cpu")))
    configure_torch_runtime(train_cfg, device)

    console.print(
        Panel.fit(
            f"[bold white]{role_display_name(role)} Trainer[/bold white]",
            subtitle="Hierarchical MAPPO" if role == "manager" else "Low-Level Skill Policy",
        )
    )

    writer = create_tensorboard_writer(train_cfg, config)
    envs = []
    num_envs = int(train_cfg["num_envs"])
    route_skill_policy = avoid_skill_policy = None
    if role == "manager":
        route_skill_policy, avoid_skill_policy = load_manager_skill_policies(
            train_cfg,
            device,
        )

    for env_idx in range(num_envs):
        env = build_role_training_env(
            tuning=tuning,
            train_cfg=train_cfg,
            role=role,
            device=device,
            route_skill_policy=route_skill_policy,
            avoid_skill_policy=avoid_skill_policy,
        )
        env_seed = None if seed is None else seed + env_idx
        reset_training_env(env, seed=env_seed)
        envs.append(env)

    reference_env = envs[0]
    policy = MAPPOPolicy(
        obs_dim=reference_env.obs_dim,
        state_dim=reference_env.state_dim,
        action_dim=reference_env.action_dim,
        actor_hidden_sizes=train_cfg["actor_hidden_sizes"],
        critic_hidden_sizes=train_cfg["critic_hidden_sizes"],
        device=device,
        log_std_init=float(train_cfg["log_std_init"]),
        obs_clip=float(train_cfg["obs_clip"]),
        state_clip=float(train_cfg["state_clip"]),
        actor_type=role_actor_type(role),
        critic_value_dim=role_critic_value_dim(role, reference_env),
    )
    optimizer = torch.optim.Adam(
        list(policy.actor.parameters()) + list(policy.critic.parameters()),
        lr=float(train_cfg["learning_rate"]),
    )

    total_timesteps = int(train_cfg["total_timesteps"])
    initial_learning_rate = float(train_cfg["learning_rate"])
    final_learning_rate = float(train_cfg.get("final_learning_rate", initial_learning_rate))
    rollout_steps = int(train_cfg["rollout_steps"])
    checkpoint_interval = int(train_cfg["checkpoint_interval"])
    latest_checkpoint_interval = int(train_cfg.get("latest_checkpoint_interval", checkpoint_interval))
    log_interval_updates = max(int(train_cfg.get("log_interval_updates", 1)), 1)

    timesteps = 0
    updates = 0
    next_checkpoint = checkpoint_interval
    next_latest_checkpoint = latest_checkpoint_interval

    current_episode_returns = [0.0 for _ in envs]
    current_episode_lengths = [0 for _ in envs]
    recent_returns = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)
    recent_completion = deque(maxlen=50)
    recent_crashes = deque(maxlen=50)
    recent_timeouts = deque(maxlen=50)
    recent_waypoint_throughput = deque(maxlen=50)
    recent_geofence_outside_steps = deque(maxlen=50)
    completed_episodes = 0
    latest_checkpoint_enabled = bool(train_cfg.get("save_latest_checkpoint", True))
    best_checkpoint_enabled = bool(train_cfg.get("save_best_checkpoint", True))
    best_checkpoint_primary_metric = str(
        train_cfg.get("best_checkpoint_primary_metric", "completion_rate")
    )
    best_checkpoint_min_episodes = int(train_cfg.get("best_checkpoint_min_episodes", 50))
    best_checkpoint_patience_timesteps = int(train_cfg.get("best_checkpoint_patience_timesteps", 0))
    best_checkpoint_stop_min_timesteps = int(train_cfg.get("best_checkpoint_stop_min_timesteps", 0))
    best_score: tuple[float, float, float, float, float] | None = None
    best_timestep: int | None = None
    train_start_time = time.perf_counter()

    try:
        if role == "route_skill":
            warmstart_metrics = run_route_warmstart(
                policy=policy,
                envs=envs,
                train_cfg=train_cfg,
                seed=seed,
                writer=writer,
                starting_timestep=timesteps,
            )
            timesteps += int(warmstart_metrics["timesteps"])
            if warmstart_metrics["timesteps"] > 0:
                console.print(
                    "[green]Route warm-start complete:[/green] "
                    f"{warmstart_metrics['timesteps']:,} steps | "
                    f"loss={warmstart_metrics['loss']:.4f} | "
                    f"mae={warmstart_metrics['mae']:.4f}"
                )
                for env in envs:
                    reset_training_env(env)
        elif role == "avoid_skill":
            warmstart_metrics = run_avoid_warmstart(
                policy=policy,
                envs=envs,
                train_cfg=train_cfg,
                seed=seed,
                writer=writer,
                starting_timestep=timesteps,
            )
            timesteps += int(warmstart_metrics["timesteps"])
            if warmstart_metrics["timesteps"] > 0:
                console.print(
                    "[green]Avoid warm-start complete:[/green] "
                    f"{warmstart_metrics['timesteps']:,} steps | "
                    f"loss={warmstart_metrics['loss']:.4f} | "
                    f"mae={warmstart_metrics['mae']:.4f}"
                )
                for env in envs:
                    reset_training_env(env)
        elif role == "manager":
            warmstart_metrics = run_manager_warmstart(
                policy=policy,
                envs=envs,
                train_cfg=train_cfg,
                seed=seed,
                writer=writer,
                starting_timestep=timesteps,
            )
            timesteps += int(warmstart_metrics["timesteps"])
            if warmstart_metrics["timesteps"] > 0:
                console.print(
                    "[green]Manager warm-start complete:[/green] "
                    f"{warmstart_metrics['timesteps']:,} steps | "
                    f"loss={warmstart_metrics['loss']:.4f} | "
                    f"mae={warmstart_metrics['mae']:.4f}"
                )
                for env in envs:
                    reset_training_env(env)

        while timesteps < total_timesteps:
            rollout_data = [empty_rollout() for _ in envs]
            steps_this_update = 0

            while steps_this_update < rollout_steps and timesteps < total_timesteps:
                policy_steps = select_actions_for_envs(
                    policy,
                    envs,
                    deterministic=False,
                    update_stats=True,
                )

                for env_idx, (env, step) in enumerate(zip(envs, policy_steps)):
                    active_agents = list(step.active_agents)
                    _, rewards, terminations, truncations, infos = env.step(
                        action_dict_from_step(step)
                    )

                    reward_vector = reward_vector_from_step(env, rewards, active_agents)
                    train_mask = policy_train_mask_after_step(env, step.mask)
                    reward = masked_mean_reward(reward_vector, step.mask)
                    done = bool(any(terminations.values()) or any(truncations.values()))
                    info = sample_info(infos)

                    rollout_data[env_idx]["obs"].append(step.normalized_obs_matrix)
                    rollout_data[env_idx]["states"].append(step.normalized_state)
                    rollout_data[env_idx]["raw_actions"].append(step.raw_action_matrix)
                    rollout_data[env_idx]["log_probs"].append(step.log_prob_vector)
                    rollout_data[env_idx]["masks"].append(train_mask)
                    rollout_data[env_idx]["values"].append(step.value)
                    rollout_data[env_idx]["rewards"].append(reward_vector)
                    rollout_data[env_idx]["dones"].append(float(done))

                    timesteps += 1
                    steps_this_update += 1
                    current_episode_returns[env_idx] += reward
                    current_episode_lengths[env_idx] += 1

                    if done:
                        # Always query the wrapper env for final metrics. HRL manager augments
                        # the base episode metrics with requested/executed skill counts; the
                        # sampled info may contain only the base env's episode_metrics.
                        final_metrics = env.get_episode_metrics()
                        if not final_metrics:
                            final_metrics = info.get("episode_metrics", {})
                        episode_summary = final_metrics.get("episode_summary", {})
                        termination_reason = episode_summary.get("termination_reason", info.get("termination_reason", "ended"))

                        recent_returns.append(float(current_episode_returns[env_idx]))
                        recent_lengths.append(int(current_episode_lengths[env_idx]))
                        recent_completion.append(1.0 if is_successful_termination(termination_reason) else 0.0)
                        recent_crashes.append(1.0 if termination_reason == "critical_violation" else 0.0)
                        recent_timeouts.append(1.0 if termination_reason == "max_steps" else 0.0)
                        recent_waypoint_throughput.append(float(episode_summary.get("waypoint_throughput_per_min", 0.0)))
                        recent_geofence_outside_steps.append(
                            float(final_metrics.get("safety_violations", {}).get("geofence", {}).get("outside_step_total", 0.0))
                        )
                        completed_episodes += 1
                        log_episode_metrics(
                            writer,
                            train_cfg=train_cfg,
                            episode_index=completed_episodes,
                            episode_return=float(current_episode_returns[env_idx]),
                            episode_length=int(current_episode_lengths[env_idx]),
                            final_metrics=final_metrics,
                            termination_reason=termination_reason,
                        )

                        current_episode_returns[env_idx] = 0.0
                        current_episode_lengths[env_idx] = 0
                        reset_training_env(env)

                    if steps_this_update >= rollout_steps or timesteps >= total_timesteps:
                        break

            batches = []
            for env_idx, env in enumerate(envs):
                data = rollout_data[env_idx]
                if not data["rewards"]:
                    continue
                if data["dones"][-1]:
                    bootstrap_value = (
                        np.zeros(reference_env.max_agents, dtype=np.float32)
                        if policy.critic_value_dim > 1
                        else 0.0
                    )
                else:
                    bootstrap_value = policy.value(env.state())
                batches.append(
                    build_rollout_batch(
                        obs=data["obs"],
                        states=data["states"],
                        raw_actions=data["raw_actions"],
                        log_probs=data["log_probs"],
                        masks=data["masks"],
                        values=data["values"],
                        rewards=data["rewards"],
                        dones=data["dones"],
                        bootstrap_value=bootstrap_value,
                        gamma=float(train_cfg["gamma"]),
                        gae_lambda=float(train_cfg["gae_lambda"]),
                    )
                )

            if not batches:
                break

            batch = concat_rollout_batches(batches)
            updates += 1
            current_learning_rate = scheduled_learning_rate(
                initial_learning_rate,
                final_learning_rate,
                timesteps / max(total_timesteps, 1),
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_learning_rate
            update_metrics = update_policy(policy, optimizer, batch, train_cfg)

            while checkpoint_interval > 0 and timesteps >= next_checkpoint:
                checkpoint_path = save_checkpoint(
                    policy,
                    save_dir=train_cfg["save_dir"],
                    filename=f"{checkpoint_stem(train_cfg['model_name'])}_step_{timesteps}.pt",
                    config=config,
                    timesteps=timesteps,
                    updates=updates,
                )
                console.print(f"[green]Checkpoint saved:[/green] {checkpoint_path}")
                next_checkpoint += checkpoint_interval

            while latest_checkpoint_enabled and latest_checkpoint_interval > 0 and timesteps >= next_latest_checkpoint:
                latest_path = save_checkpoint(
                    policy,
                    save_dir=train_cfg["save_dir"],
                    filename=latest_checkpoint_filename(train_cfg),
                    config=config,
                    timesteps=timesteps,
                    updates=updates,
                )
                console.print(f"[green]Latest checkpoint updated:[/green] {latest_path}")
                next_latest_checkpoint += latest_checkpoint_interval
                if writer is not None:
                    writer.flush()

            mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            mean_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            completion_rate = float(np.mean(recent_completion)) if recent_completion else 0.0
            crash_rate = float(np.mean(recent_crashes)) if recent_crashes else 0.0
            timeout_rate = float(np.mean(recent_timeouts)) if recent_timeouts else 0.0
            waypoint_throughput = float(np.mean(recent_waypoint_throughput)) if recent_waypoint_throughput else 0.0
            geofence_outside_steps = float(np.mean(recent_geofence_outside_steps)) if recent_geofence_outside_steps else 0.0
            train_metrics = {
                "mean_return": mean_return,
                "mean_length": mean_length,
                "completion_rate": completion_rate,
                "crash_rate": crash_rate,
                "timeout_rate": timeout_rate,
                "waypoint_throughput": waypoint_throughput,
                "geofence_outside_steps": geofence_outside_steps,
            }
            elapsed = max(time.perf_counter() - train_start_time, 1e-6)
            timesteps_per_sec = float(timesteps / elapsed)
            train_metrics_interval_updates = tensorboard_interval(train_cfg, "train_metrics_interval_updates", log_interval_updates)

            if best_checkpoint_enabled and should_update_best_checkpoint(
                train_metrics,
                best_score=best_score,
                recent_episode_count=len(recent_completion),
                min_recent_episodes=best_checkpoint_min_episodes,
                primary_metric=best_checkpoint_primary_metric,
            ):
                best_score = best_checkpoint_score(
                    train_metrics,
                    primary_metric=best_checkpoint_primary_metric,
                )
                best_timestep = int(timesteps)
                best_path = save_checkpoint(
                    policy,
                    save_dir=train_cfg["save_dir"],
                    filename=best_checkpoint_filename(train_cfg),
                    config=config,
                    timesteps=timesteps,
                    updates=updates,
                )
                console.print(
                    "[bold green]Best checkpoint updated:[/bold green] "
                    f"{best_path} | completion={completion_rate:.2%} | "
                    f"return={mean_return:.2f} | length={mean_length:.1f}"
                )
                if writer is not None:
                    writer.add_scalar("train/best_completion_rate_50", completion_rate, timesteps)
                    writer.add_scalar("train/best_mean_return_50", mean_return, timesteps)
                    writer.add_scalar("train/best_mean_length_50", mean_length, timesteps)
                    writer.flush()

            if should_stop_for_stale_best_checkpoint(
                timesteps=timesteps,
                best_timestep=best_timestep,
                patience_timesteps=best_checkpoint_patience_timesteps,
                min_timesteps=best_checkpoint_stop_min_timesteps,
            ):
                console.print(
                    "[yellow]Stopping training: no best-checkpoint improvement for "
                    f"{timesteps - int(best_timestep or timesteps):,} timesteps. "
                    f"Best checkpoint is {best_checkpoint_filename(train_cfg)}.[/yellow]"
                )
                break

            if writer is not None and (updates % train_metrics_interval_updates) == 0:
                writer.add_scalar("loss/total", update_metrics["loss"], timesteps)
                writer.add_scalar("loss/actor", update_metrics["actor_loss"], timesteps)
                writer.add_scalar("loss/critic", update_metrics["critic_loss"], timesteps)
                writer.add_scalar("policy/entropy", update_metrics["entropy"], timesteps)
                writer.add_scalar("policy/approx_kl", update_metrics["approx_kl"], timesteps)
                writer.add_scalar("train/mean_return_50", mean_return, timesteps)
                writer.add_scalar("train/mean_length_50", mean_length, timesteps)
                writer.add_scalar("train/completion_rate_50", completion_rate, timesteps)
                writer.add_scalar("train/crash_rate_50", crash_rate, timesteps)
                writer.add_scalar("train/timeout_rate_50", timeout_rate, timesteps)
                writer.add_scalar("train/waypoint_throughput_per_min_50", waypoint_throughput, timesteps)
                writer.add_scalar("train/geofence_outside_steps_50", geofence_outside_steps, timesteps)
                writer.add_scalar("train/timesteps_per_sec", timesteps_per_sec, timesteps)
                writer.add_scalar("train/updates", updates, timesteps)
                writer.add_scalar("train/episodes_completed", completed_episodes, timesteps)
                writer.add_scalar("train/learning_rate", float(optimizer.param_groups[0]["lr"]), timesteps)

            if updates % log_interval_updates == 0:
                console.print(
                    Panel.fit(
                        "\n".join(
                            [
                                f"Role: {role}",
                                f"Timesteps: {timesteps:,}/{total_timesteps:,}",
                                f"Updates: {updates}",
                                f"Mean Return (50 ep): {mean_return:.2f}",
                                f"Mean Length (50 ep): {mean_length:.1f}",
                                f"Completion Rate (50 ep): {completion_rate:.2%}",
                                f"Crash Rate (50 ep): {crash_rate:.2%}",
                                f"Timeout Rate (50 ep): {timeout_rate:.2%}",
                                f"Waypoint Throughput (50 ep): {waypoint_throughput:.2f}/min",
                                f"Geofence Outside Steps (50 ep): {geofence_outside_steps:.1f}",
                                f"Timesteps / Sec: {timesteps_per_sec:.1f}",
                                f"Actor Loss: {update_metrics['actor_loss']:.4f}",
                                f"Critic Loss: {update_metrics['critic_loss']:.4f}",
                                f"Entropy: {update_metrics['entropy']:.4f}",
                                f"Approx KL: {update_metrics['approx_kl']:.5f}",
                            ]
                        ),
                        title="[bold cyan]Training Update[/bold cyan]",
                    )
                )
                if writer is not None:
                    writer.flush()

        final_path = save_checkpoint(
            policy,
            save_dir=train_cfg["save_dir"],
            filename=checkpoint_filename(train_cfg["model_name"]),
            config=config,
            timesteps=timesteps,
            updates=updates,
        )
        console.print(f"[green]Final MAPPO checkpoint saved:[/green] {final_path}")
        if latest_checkpoint_enabled:
            latest_path = save_checkpoint(
                policy,
                save_dir=train_cfg["save_dir"],
                filename=latest_checkpoint_filename(train_cfg),
                config=config,
                timesteps=timesteps,
                updates=updates,
            )
            if latest_path != final_path:
                console.print(f"[green]Latest MAPPO checkpoint updated:[/green] {latest_path}")
        if writer is not None:
            writer.flush()

    except KeyboardInterrupt:
        interrupted_path = save_checkpoint(
            policy,
            save_dir=train_cfg["save_dir"],
            filename=f"{checkpoint_stem(train_cfg['model_name'])}_interrupted.pt",
            config=config,
            timesteps=timesteps,
            updates=updates,
            interrupted=True,
        )
        console.print(f"[yellow]Training interrupted. Saved:[/yellow] {interrupted_path}")
        if latest_checkpoint_enabled:
            latest_path = save_checkpoint(
                policy,
                save_dir=train_cfg["save_dir"],
                filename=latest_checkpoint_filename(train_cfg),
                config=config,
                timesteps=timesteps,
                updates=updates,
                interrupted=True,
            )
            if latest_path != interrupted_path:
                console.print(f"[yellow]Latest MAPPO checkpoint updated:[/yellow] {latest_path}")
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        for env in envs:
            env.close()


if __name__ == "__main__":
    print()
