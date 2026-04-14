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
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None

from config_utils import get_tuning_section
from flight_engine.helpers import FlightMode
from mappo import MAPPOPolicy, RolloutBatch, build_rollout_batch
from mappo_runtime import (
    action_dict_from_step,
    sample_info,
    select_actions_for_envs,
)
from pettingzoo_env import MultiUAVParallelEnv

console = Console()


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_filename(model_name: str) -> str:
    if model_name.endswith(".pt") or model_name.endswith(".pth"):
        return model_name
    return f"{model_name}.pt"


def checkpoint_stem(model_name: str) -> str:
    filename = checkpoint_filename(model_name)
    return os.path.splitext(filename)[0]


def save_checkpoint(
    policy: MAPPOPolicy,
    *,
    save_dir: str,
    filename: str,
    config: dict,
    timesteps: int,
    updates: int,
    interrupted: bool = False,
) -> str:
    path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(path) or save_dir, exist_ok=True)
    policy.save(
        path,
        extra={
            "config": config,
            "timesteps": int(timesteps),
            "updates": int(updates),
            "interrupted": bool(interrupted),
        },
    )
    return path


def latest_checkpoint_filename(train_cfg: dict) -> str:
    latest_name = train_cfg.get("latest_model_name")
    if latest_name:
        return checkpoint_filename(str(latest_name))
    return f"{checkpoint_stem(train_cfg['model_name'])}_latest.pt"


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
        1.0 if termination_reason == "completed" else 0.0,
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
    writer.add_scalar(
        "episode/circling_steps",
        float(episode_summary.get("circling_steps_total", 0.0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/circling_breakouts",
        float(episode_summary.get("circling_breakouts_total", 0.0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoint_reapproach_steps",
        float(episode_summary.get("waypoint_reapproach_steps_total", 0.0)),
        episode_index,
    )
    writer.add_scalar(
        "episode/waypoint_reapproach_events",
        float(episode_summary.get("waypoint_reapproach_events_total", 0.0)),
        episode_index,
    )
    if min_pairwise_distance is not None:
        writer.add_scalar(
            "episode/min_pairwise_distance_m",
            float(min_pairwise_distance),
            episode_index,
        )

    log_numeric_scalars("episode_reward", reward_breakdown)


def empty_rollout() -> dict:
    return {
        "obs": [],
        "states": [],
        "raw_actions": [],
        "log_probs": [],
        "masks": [],
        "reference_actions": [],
        "reference_masks": [],
        "values": [],
        "rewards": [],
        "dones": [],
    }


def concat_rollout_batches(batches: List[RolloutBatch]) -> RolloutBatch:
    return RolloutBatch(
        obs=np.concatenate([batch.obs for batch in batches], axis=0),
        states=np.concatenate([batch.states for batch in batches], axis=0),
        raw_actions=np.concatenate([batch.raw_actions for batch in batches], axis=0),
        log_probs=np.concatenate([batch.log_probs for batch in batches], axis=0),
        masks=np.concatenate([batch.masks for batch in batches], axis=0),
        reference_actions=np.concatenate(
            [batch.reference_actions for batch in batches], axis=0
        ),
        reference_masks=np.concatenate(
            [batch.reference_masks for batch in batches], axis=0
        ),
        values=np.concatenate([batch.values for batch in batches], axis=0),
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


def build_training_env(tuning: dict) -> MultiUAVParallelEnv:
    env_cfg = tuning["env"]
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
        caution_dist=env_cfg["caution_dist"],
        critical_dist=env_cfg["critical_dist"],
        min_agents=env_cfg["min_agents"],
        max_agents=env_cfg["max_agents"],
        map_size_range_m=(env_cfg["box_min_m"], env_cfg["box_max_m"]),
        flight_config=tuning["flight"],
        reward_config=tuning["rewards"],
        guidance_config=tuning["guidance"],
    )


def huber_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    abs_error = torch.abs(prediction - target)
    if delta <= 0.0:
        return abs_error
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return (0.5 * quadratic.pow(2) / delta) + linear


def reference_action_dict(env) -> dict:
    return {
        agent: np.asarray(
            [env._last_reference_action_vector[env.agent_name_to_index[agent]]],
            dtype=np.float32,
        )
        for agent in env.agents
    }


def reference_supervision_targets(env, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    reference_actions = np.zeros((env.max_agents, action_dim), dtype=np.float32)
    reference_masks = np.zeros(env.max_agents, dtype=np.float32)
    for agent in env.agents:
        idx = env.agent_name_to_index[agent]
        aircraft = env.aircraft_by_agent[agent]
        reference_actions[idx, 0] = float(env._last_reference_action_vector[idx])
        if (
            aircraft.waypoint_manager.current_waypoint is not None
            and aircraft.flight_mode != FlightMode.LOITERING
            and not env._deconfliction_active[idx]
        ):
            reference_masks[idx] = 1.0
    return reference_actions, reference_masks


def run_reference_warmstart(
    *,
    policy: MAPPOPolicy,
    envs: List[MultiUAVParallelEnv],
    train_cfg: dict,
    seed: int | None,
    writer,
    starting_timestep: int,
) -> dict:
    legacy_warmstart_steps = max(
        int(train_cfg.get("reference_warmstart_steps", 0)),
        0,
    )
    warmstart_max_steps = max(
        int(
            train_cfg.get(
                "reference_warmstart_max_steps",
                legacy_warmstart_steps,
            )
        ),
        0,
    )
    warmstart_min_steps = max(
        int(
            train_cfg.get(
                "reference_warmstart_min_steps",
                min(10_000, warmstart_max_steps) if warmstart_max_steps > 0 else 0,
            )
        ),
        0,
    )
    if warmstart_max_steps <= 0 or not envs:
        return {
            "timesteps": 0,
            "updates": 0,
            "reference_loss": 0.0,
            "reference_mae": 0.0,
            "converged": False,
            "stop_reason": "disabled",
        }
    warmstart_min_steps = min(warmstart_min_steps, warmstart_max_steps)

    warmstart_num_envs = max(
        1,
        min(
            int(train_cfg.get("reference_warmstart_num_envs", len(envs))),
            len(envs),
        ),
    )
    warmstart_lr = float(
        train_cfg.get(
            "reference_warmstart_learning_rate",
            train_cfg["learning_rate"],
        )
    )
    max_grad_norm = float(train_cfg["max_grad_norm"])
    reference_delta = float(train_cfg.get("reference_warmstart_delta", 0.1))
    log_interval = max(int(train_cfg.get("reference_warmstart_log_interval", 10_000)), 1)
    convergence_window = max(
        int(train_cfg.get("reference_warmstart_convergence_updates", 50)),
        1,
    )
    target_mae = float(train_cfg.get("reference_warmstart_target_mae", 0.12))
    target_loss = float(
        train_cfg.get("reference_warmstart_target_loss", target_mae)
    )
    post_warmstart_log_std = float(
        train_cfg.get("reference_warmstart_log_std", -2.5)
    )

    warmstart_envs = envs[:warmstart_num_envs]
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=warmstart_lr)

    for env_idx, env in enumerate(warmstart_envs):
        env_seed = None if seed is None else seed + 10_000 + env_idx
        env.reset(seed=env_seed)

    steps = 0
    updates = 0
    reference_losses: List[float] = []
    reference_maes: List[float] = []
    recent_loss_window = deque(maxlen=convergence_window)
    recent_mae_window = deque(maxlen=convergence_window)
    last_log_step = 0
    converged = False
    stop_reason = "max_steps"

    while steps < warmstart_max_steps:
        obs_batches = []
        state_batch = []
        reference_batch = []

        for env in warmstart_envs:
            active_indices = [env.agent_name_to_index[agent] for agent in env.agents]
            if not active_indices:
                continue

            obs_matrix = env.get_obs_matrix()
            state_batch.append(np.asarray(env.state(), dtype=np.float32))
            active_obs = obs_matrix[active_indices]
            active_reference = env._last_reference_action_vector[active_indices].astype(
                np.float32,
                copy=False,
            )[:, None]
            obs_batches.append(active_obs)
            reference_batch.append(active_reference)

        if obs_batches:
            merged_obs = np.concatenate(obs_batches, axis=0)
            merged_states = np.asarray(state_batch, dtype=np.float32)
            merged_reference = np.concatenate(reference_batch, axis=0)
            policy.update_normalizers(merged_obs, merged_states)
            normalized_obs = policy.normalize_obs(merged_obs)

            obs_tensor = torch.as_tensor(
                normalized_obs,
                dtype=torch.float32,
                device=policy.device,
            )
            reference_tensor = torch.as_tensor(
                merged_reference,
                dtype=torch.float32,
                device=policy.device,
            )

            actor_optimizer.zero_grad()
            mean_actions = policy.actor_mean_action(obs_tensor)
            reference_loss = huber_regression_loss(
                mean_actions,
                reference_tensor,
                reference_delta,
            ).mean()
            reference_mae = torch.abs(mean_actions - reference_tensor).mean()
            reference_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_grad_norm)
            actor_optimizer.step()

            updates += 1
            reference_losses.append(float(reference_loss.item()))
            reference_maes.append(float(reference_mae.item()))
            recent_loss_window.append(float(reference_loss.item()))
            recent_mae_window.append(float(reference_mae.item()))

            global_step = starting_timestep + steps
            if writer is not None:
                writer.add_scalar(
                    "warmstart/reference_loss",
                    float(reference_loss.item()),
                    global_step,
                )
                writer.add_scalar(
                    "warmstart/reference_mae",
                    float(reference_mae.item()),
                    global_step,
                )

            if (
                steps >= warmstart_min_steps
                and len(recent_mae_window) == convergence_window
                and float(np.mean(recent_mae_window)) <= target_mae
                and float(np.mean(recent_loss_window)) <= target_loss
            ):
                converged = True
                stop_reason = "converged"
                break

        for env in warmstart_envs:
            if steps >= warmstart_max_steps or converged:
                break
            _, _, terminations, truncations, _ = env.step(reference_action_dict(env))
            steps += 1
            done = bool(any(terminations.values()) or any(truncations.values()))
            if done:
                env.reset()

        if converged:
            break

        if (steps - last_log_step) >= log_interval:
            mean_loss = float(np.mean(reference_losses[-100:])) if reference_losses else 0.0
            mean_mae = float(np.mean(reference_maes[-100:])) if reference_maes else 0.0
            console.print(
                f"[cyan]Reference warm-start:[/cyan] {steps:,}/{warmstart_max_steps:,} "
                f"steps | loss={mean_loss:.4f} | mae={mean_mae:.4f} "
                f"| target_loss<={target_loss:.4f} target_mae<={target_mae:.4f}"
            )
            last_log_step = steps

    return {
        "timesteps": steps,
        "updates": updates,
        "reference_loss": float(np.mean(reference_losses)) if reference_losses else 0.0,
        "reference_mae": float(np.mean(reference_maes)) if reference_maes else 0.0,
        "reference_log_std": post_warmstart_log_std,
        "converged": converged,
        "stop_reason": stop_reason,
    }


def update_policy(
    policy: MAPPOPolicy,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    train_cfg: dict,
) -> dict:
    device = policy.device
    batch_size = int(batch.states.shape[0])
    minibatch_size = min(int(train_cfg["minibatch_size"]), batch_size)
    update_epochs = int(train_cfg["update_epochs"])
    clip_ratio = float(train_cfg["clip_ratio"])
    value_clip = float(train_cfg["value_clip"])
    entropy_coef = float(train_cfg["entropy_coef"])
    value_loss_coef = float(train_cfg["value_loss_coef"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    reference_tracking_coef = float(train_cfg.get("reference_tracking_coef", 0.0))
    reference_tracking_delta = float(train_cfg.get("reference_tracking_delta", 0.1))
    target_kl = train_cfg.get("target_kl")
    if target_kl is not None:
        target_kl = max(float(target_kl), 1e-6)

    advantages = batch.advantages.astype(np.float32, copy=True)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
    raw_actions = torch.as_tensor(batch.raw_actions, dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch.masks, dtype=torch.float32, device=device)
    reference_actions = torch.as_tensor(
        batch.reference_actions,
        dtype=torch.float32,
        device=device,
    )
    reference_masks = torch.as_tensor(
        batch.reference_masks,
        dtype=torch.float32,
        device=device,
    )
    old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)

    metrics = {
        "loss": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
        "reference_loss": [],
        "reference_mae": [],
        "reference_mask_fraction": [],
    }

    indices = np.arange(batch_size)
    stop_early = False

    for _ in range(update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            stop = start + minibatch_size
            batch_indices = indices[start:stop]

            obs_mb = obs[batch_indices]
            states_mb = states[batch_indices]
            raw_actions_mb = raw_actions[batch_indices]
            old_log_probs_mb = old_log_probs[batch_indices]
            masks_mb = masks[batch_indices]
            reference_actions_mb = reference_actions[batch_indices]
            reference_masks_mb = reference_masks[batch_indices]
            old_values_mb = old_values[batch_indices]
            returns_mb = returns[batch_indices]
            advantages_mb = advantages[batch_indices]

            max_agents = masks_mb.shape[1]
            obs_flat = obs_mb.reshape(-1, policy.obs_dim)
            raw_actions_flat = raw_actions_mb.reshape(-1, policy.action_dim)
            mask_flat = masks_mb.reshape(-1)
            old_log_probs_flat = old_log_probs_mb.reshape(-1)
            reference_actions_flat = reference_actions_mb.reshape(-1, policy.action_dim)
            reference_mask_flat = reference_masks_mb.reshape(-1)
            adv_flat = advantages_mb.unsqueeze(1).expand(-1, max_agents).reshape(-1)

            _, new_log_probs_flat, entropy_flat = policy.evaluate_actor(
                obs_flat,
                raw_actions_flat,
            )
            pre_step_kl = (
                ((old_log_probs_flat - new_log_probs_flat) * mask_flat).sum()
                / mask_flat.sum().clamp_min(1.0)
            )
            if target_kl is not None and abs(float(pre_step_kl.item())) > target_kl:
                stop_early = True
                break

            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            unclipped = ratio * adv_flat
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_flat

            valid_count = mask_flat.sum().clamp_min(1.0)
            actor_loss = -(
                torch.minimum(unclipped, clipped) * mask_flat
            ).sum() / valid_count
            entropy = (entropy_flat * mask_flat).sum() / valid_count

            if reference_tracking_coef > 0.0:
                mean_actions_flat = policy.actor_mean_action(obs_flat)
                reference_valid = reference_mask_flat.sum().clamp_min(1.0)
                reference_loss = (
                    huber_regression_loss(
                        mean_actions_flat,
                        reference_actions_flat,
                        reference_tracking_delta,
                    ).sum(dim=-1)
                    * reference_mask_flat
                ).sum() / reference_valid
                reference_mae = (
                    torch.abs(mean_actions_flat - reference_actions_flat).sum(dim=-1)
                    * reference_mask_flat
                ).sum() / reference_valid
                reference_mask_fraction = (
                    reference_mask_flat.mean()
                    if reference_mask_flat.numel() > 0
                    else torch.tensor(0.0, device=device)
                )
            else:
                reference_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                reference_mae = torch.tensor(0.0, dtype=torch.float32, device=device)
                reference_mask_fraction = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )

            values_pred = policy.evaluate_critic(states_mb)
            if value_clip > 0.0:
                value_delta = torch.clamp(
                    values_pred - old_values_mb,
                    -value_clip,
                    value_clip,
                )
                values_clipped = old_values_mb + value_delta
                value_loss_unclipped = (values_pred - returns_mb).pow(2)
                value_loss_clipped = (values_clipped - returns_mb).pow(2)
                critic_loss = 0.5 * torch.maximum(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
            else:
                critic_loss = 0.5 * (values_pred - returns_mb).pow(2).mean()

            loss = (
                actor_loss
                + (value_loss_coef * critic_loss)
                - (entropy_coef * entropy)
                + (reference_tracking_coef * reference_loss)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.actor.parameters()) + list(policy.critic.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                _, post_step_log_probs_flat, _ = policy.evaluate_actor(
                    obs_flat,
                    raw_actions_flat,
                )
                approx_kl = (
                    ((old_log_probs_flat - post_step_log_probs_flat) * mask_flat).sum()
                    / valid_count
                )

            metrics["loss"].append(float(loss.item()))
            metrics["actor_loss"].append(float(actor_loss.item()))
            metrics["critic_loss"].append(float(critic_loss.item()))
            metrics["entropy"].append(float(entropy.item()))
            metrics["approx_kl"].append(float(approx_kl.item()))
            metrics["reference_loss"].append(float(reference_loss.item()))
            metrics["reference_mae"].append(float(reference_mae.item()))
            metrics["reference_mask_fraction"].append(
                float(reference_mask_fraction.item())
            )

            if target_kl is not None and abs(float(approx_kl.item())) > target_kl:
                stop_early = True
                break

        if stop_early:
            break

    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in metrics.items()
    }


def train(config):
    train_cfg = config["train"]
    tuning = get_tuning_section(config, "train")
    os.makedirs(train_cfg["save_dir"], exist_ok=True)

    seed = train_cfg.get("seed")
    if seed is not None:
        seed = int(seed)
    set_global_seed(seed)

    device = resolve_device(str(train_cfg.get("device", "cpu")))
    configure_torch_runtime(train_cfg, device)

    console.print(
        Panel.fit(
            "[bold white]MAPPO Trainer[/bold white]",
            subtitle="Stationary Randomized MARL",
        )
    )

    writer = create_tensorboard_writer(train_cfg, config)
    envs = []
    num_envs = int(train_cfg["num_envs"])
    for env_idx in range(num_envs):
        env = build_training_env(tuning)
        env_seed = None if seed is None else seed + env_idx
        env.reset(seed=env_seed)
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
    )
    optimizer = torch.optim.Adam(
        list(policy.actor.parameters()) + list(policy.critic.parameters()),
        lr=float(train_cfg["learning_rate"]),
    )

    total_timesteps = int(train_cfg["total_timesteps"])
    rollout_steps = int(train_cfg["rollout_steps"])
    checkpoint_interval = int(train_cfg["checkpoint_interval"])
    latest_checkpoint_interval = int(
        train_cfg.get("latest_checkpoint_interval", checkpoint_interval)
    )
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
    recent_circling_steps = deque(maxlen=50)
    completed_episodes = 0
    latest_checkpoint_enabled = bool(train_cfg.get("save_latest_checkpoint", True))
    train_start_time = time.perf_counter()

    try:
        warmstart_metrics = run_reference_warmstart(
            policy=policy,
            envs=envs,
            train_cfg=train_cfg,
            seed=seed,
            writer=writer,
            starting_timestep=timesteps,
        )
        timesteps += int(warmstart_metrics["timesteps"])
        if warmstart_metrics["timesteps"] > 0:
            with torch.no_grad():
                policy.actor.log_std.fill_(float(warmstart_metrics["reference_log_std"]))
            console.print(
                "[green]Reference warm-start complete:[/green] "
                f"{warmstart_metrics['timesteps']:,} steps | "
                f"{warmstart_metrics['stop_reason']} | "
                f"loss={warmstart_metrics['reference_loss']:.4f} | "
                f"mae={warmstart_metrics['reference_mae']:.4f} | "
                f"log_std={warmstart_metrics['reference_log_std']:.2f}"
            )
            for env in envs:
                env.reset()

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
                    reference_actions, reference_masks = reference_supervision_targets(
                        env,
                        policy.action_dim,
                    )
                    _, rewards, terminations, truncations, infos = env.step(
                        action_dict_from_step(step)
                    )

                    reward = float(next(iter(rewards.values()), 0.0))
                    done = bool(any(terminations.values()) or any(truncations.values()))
                    info = sample_info(infos)

                    rollout_data[env_idx]["obs"].append(step.normalized_obs_matrix)
                    rollout_data[env_idx]["states"].append(step.normalized_state)
                    rollout_data[env_idx]["raw_actions"].append(step.raw_action_matrix)
                    rollout_data[env_idx]["log_probs"].append(step.log_prob_vector)
                    rollout_data[env_idx]["masks"].append(
                        env._last_policy_train_mask.copy()
                    )
                    rollout_data[env_idx]["reference_actions"].append(reference_actions)
                    rollout_data[env_idx]["reference_masks"].append(reference_masks)
                    rollout_data[env_idx]["values"].append(step.value)
                    rollout_data[env_idx]["rewards"].append(reward)
                    rollout_data[env_idx]["dones"].append(float(done))

                    timesteps += 1
                    steps_this_update += 1
                    current_episode_returns[env_idx] += reward
                    current_episode_lengths[env_idx] += 1

                    if done:
                        final_metrics = info.get("episode_metrics", env.get_episode_metrics())
                        episode_summary = final_metrics.get("episode_summary", {})
                        termination_reason = episode_summary.get("termination_reason", "ended")

                        recent_returns.append(float(current_episode_returns[env_idx]))
                        recent_lengths.append(int(current_episode_lengths[env_idx]))
                        recent_completion.append(
                            1.0 if termination_reason == "completed" else 0.0
                        )
                        recent_crashes.append(
                            1.0
                            if termination_reason == "critical_violation"
                            else 0.0
                        )
                        recent_timeouts.append(
                            1.0 if termination_reason == "max_steps" else 0.0
                        )
                        recent_waypoint_throughput.append(
                            float(episode_summary.get("waypoint_throughput_per_min", 0.0))
                        )
                        recent_geofence_outside_steps.append(
                            float(
                                final_metrics.get("safety_violations", {})
                                .get("geofence", {})
                                .get("outside_step_total", 0.0)
                            )
                        )
                        recent_circling_steps.append(
                            float(episode_summary.get("circling_steps_total", 0.0))
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
                        env.reset()

                    if steps_this_update >= rollout_steps or timesteps >= total_timesteps:
                        break

            batches = []
            for env_idx, env in enumerate(envs):
                data = rollout_data[env_idx]
                if not data["rewards"]:
                    continue
                bootstrap_value = (
                    0.0 if data["dones"][-1] else policy.value(env.state())
                )
                batches.append(
                    build_rollout_batch(
                        obs=data["obs"],
                        states=data["states"],
                        raw_actions=data["raw_actions"],
                        log_probs=data["log_probs"],
                        masks=data["masks"],
                        reference_actions=data["reference_actions"],
                        reference_masks=data["reference_masks"],
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

            while (
                latest_checkpoint_enabled
                and latest_checkpoint_interval > 0
                and timesteps >= next_latest_checkpoint
            ):
                latest_path = save_checkpoint(
                    policy,
                    save_dir=train_cfg["save_dir"],
                    filename=latest_checkpoint_filename(train_cfg),
                    config=config,
                    timesteps=timesteps,
                    updates=updates,
                )
                console.print(
                    f"[green]Latest checkpoint updated:[/green] {latest_path}"
                )
                next_latest_checkpoint += latest_checkpoint_interval
                if writer is not None:
                    writer.flush()

            mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            mean_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            completion_rate = float(np.mean(recent_completion)) if recent_completion else 0.0
            crash_rate = float(np.mean(recent_crashes)) if recent_crashes else 0.0
            timeout_rate = float(np.mean(recent_timeouts)) if recent_timeouts else 0.0
            waypoint_throughput = (
                float(np.mean(recent_waypoint_throughput))
                if recent_waypoint_throughput
                else 0.0
            )
            geofence_outside_steps = (
                float(np.mean(recent_geofence_outside_steps))
                if recent_geofence_outside_steps
                else 0.0
            )
            circling_steps = (
                float(np.mean(recent_circling_steps))
                if recent_circling_steps
                else 0.0
            )
            elapsed = max(time.perf_counter() - train_start_time, 1e-6)
            timesteps_per_sec = float(timesteps / elapsed)
            train_metrics_interval_updates = tensorboard_interval(
                train_cfg,
                "train_metrics_interval_updates",
                log_interval_updates,
            )

            if writer is not None and (updates % train_metrics_interval_updates) == 0:
                writer.add_scalar("loss/total", update_metrics["loss"], timesteps)
                writer.add_scalar("loss/actor", update_metrics["actor_loss"], timesteps)
                writer.add_scalar("loss/critic", update_metrics["critic_loss"], timesteps)
                writer.add_scalar(
                    "loss/reference",
                    update_metrics["reference_loss"],
                    timesteps,
                )
                writer.add_scalar("policy/entropy", update_metrics["entropy"], timesteps)
                writer.add_scalar("policy/approx_kl", update_metrics["approx_kl"], timesteps)
                writer.add_scalar(
                    "policy/reference_mae",
                    update_metrics["reference_mae"],
                    timesteps,
                )
                writer.add_scalar(
                    "policy/reference_mask_fraction",
                    update_metrics["reference_mask_fraction"],
                    timesteps,
                )
                writer.add_scalar("train/mean_return_50", mean_return, timesteps)
                writer.add_scalar("train/mean_length_50", mean_length, timesteps)
                writer.add_scalar("train/completion_rate_50", completion_rate, timesteps)
                writer.add_scalar("train/crash_rate_50", crash_rate, timesteps)
                writer.add_scalar("train/timeout_rate_50", timeout_rate, timesteps)
                writer.add_scalar(
                    "train/waypoint_throughput_per_min_50",
                    waypoint_throughput,
                    timesteps,
                )
                writer.add_scalar(
                    "train/geofence_outside_steps_50",
                    geofence_outside_steps,
                    timesteps,
                )
                writer.add_scalar(
                    "train/circling_steps_50",
                    circling_steps,
                    timesteps,
                )
                writer.add_scalar("train/timesteps_per_sec", timesteps_per_sec, timesteps)
                writer.add_scalar("train/updates", updates, timesteps)
                writer.add_scalar("train/episodes_completed", completed_episodes, timesteps)
                writer.add_scalar(
                    "train/learning_rate",
                    float(optimizer.param_groups[0]["lr"]),
                    timesteps,
                )

            if updates % log_interval_updates == 0:
                console.print(
                    Panel.fit(
                        "\n".join(
                            [
                                f"Timesteps: {timesteps:,}/{total_timesteps:,}",
                                f"Updates: {updates}",
                                f"Mean Return (50 ep): {mean_return:.2f}",
                                f"Mean Length (50 ep): {mean_length:.1f}",
                                f"Completion Rate (50 ep): {completion_rate:.2%}",
                                f"Crash Rate (50 ep): {crash_rate:.2%}",
                                f"Timeout Rate (50 ep): {timeout_rate:.2%}",
                                f"Waypoint Throughput (50 ep): {waypoint_throughput:.2f}/min",
                                f"Geofence Outside Steps (50 ep): {geofence_outside_steps:.1f}",
                                f"Circling Steps (50 ep): {circling_steps:.1f}",
                                f"Timesteps / Sec: {timesteps_per_sec:.1f}",
                                f"Actor Loss: {update_metrics['actor_loss']:.4f}",
                                f"Critic Loss: {update_metrics['critic_loss']:.4f}",
                                f"Reference Loss: {update_metrics['reference_loss']:.4f}",
                                f"Reference MAE: {update_metrics['reference_mae']:.4f}",
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
                console.print(
                    f"[green]Latest MAPPO checkpoint updated:[/green] {latest_path}"
                )
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
                console.print(
                    f"[yellow]Latest MAPPO checkpoint updated:[/yellow] {latest_path}"
                )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        for env in envs:
            env.close()


if __name__ == "__main__":
    print()
