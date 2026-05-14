from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from mappo import MAPPOPolicy


@dataclass
class PolicyStep:
    obs_matrix: np.ndarray
    normalized_obs_matrix: np.ndarray
    state: np.ndarray
    normalized_state: np.ndarray
    mask: np.ndarray
    action_matrix: np.ndarray
    raw_action_matrix: np.ndarray
    log_prob_vector: np.ndarray
    active_agents: List[str]
    active_indices: List[int]
    value: float | np.ndarray


def sample_info(info_dict: Dict[str, dict]) -> dict:
    return next(iter(info_dict.values()), {})


def active_agent_indices(env) -> List[int]:
    return [env.agent_name_to_index[agent] for agent in env.agents]


def validate_policy_env(policy: MAPPOPolicy, env) -> None:
    mismatches = []
    if int(policy.obs_dim) != int(env.obs_dim):
        mismatches.append(
            f"obs_dim checkpoint={policy.obs_dim} env={env.obs_dim}"
        )
    if int(policy.state_dim) != int(env.state_dim):
        mismatches.append(
            f"state_dim checkpoint={policy.state_dim} env={env.state_dim}"
        )
    if int(policy.action_dim) != int(env.action_dim):
        mismatches.append(
            f"action_dim checkpoint={policy.action_dim} env={env.action_dim}"
        )
    if mismatches:
        raise ValueError(
            "MAPPO checkpoint is incompatible with the configured environment: "
            + "; ".join(mismatches)
        )


def select_actions(
    policy: MAPPOPolicy,
    env,
    *,
    deterministic: bool = False,
    update_stats: bool = False,
) -> PolicyStep:
    obs_matrix = env.get_obs_matrix()
    state = env.state()
    active_agents = list(env.agents)
    active_indices = active_agent_indices(env)

    if not active_agents:
        raise ValueError("Cannot select MAPPO actions when the environment has no active agents.")

    active_obs = obs_matrix[active_indices]
    output = policy.act_parallel(
        [active_obs],
        np.asarray([state], dtype=np.float32),
        deterministic=deterministic,
        update_stats=update_stats,
    )

    action_matrix = np.zeros(
        (env.max_agents, policy.action_dim),
        dtype=np.float32,
    )
    raw_action_matrix = np.zeros_like(action_matrix)
    log_prob_vector = np.zeros(env.max_agents, dtype=np.float32)
    mask = np.zeros(env.max_agents, dtype=np.float32)

    active_actions = output["actions"][0]
    active_raw_actions = output["raw_actions"][0]
    active_log_probs = output["log_probs"][0]
    active_normalized_obs = output["normalized_obs"][0]
    normalized_state = np.asarray(output["normalized_states"][0], dtype=np.float32)

    action_matrix[active_indices] = active_actions
    raw_action_matrix[active_indices] = active_raw_actions
    log_prob_vector[active_indices] = active_log_probs
    mask[active_indices] = 1.0

    normalized_obs_matrix = np.zeros_like(obs_matrix, dtype=np.float32)
    normalized_obs_matrix[active_indices] = active_normalized_obs

    value = output["values"][0]
    if np.ndim(value) == 0:
        value = float(value)
    else:
        value = np.asarray(value, dtype=np.float32)

    return PolicyStep(
        obs_matrix=obs_matrix,
        normalized_obs_matrix=normalized_obs_matrix,
        state=state,
        normalized_state=normalized_state,
        mask=mask,
        action_matrix=action_matrix,
        raw_action_matrix=raw_action_matrix,
        log_prob_vector=log_prob_vector,
        active_agents=active_agents,
        active_indices=active_indices,
        value=value,
    )


def select_actions_for_envs(
    policy: MAPPOPolicy,
    envs: List,
    *,
    deterministic: bool = False,
    update_stats: bool = False,
) -> List[PolicyStep]:
    obs_matrices: List[np.ndarray] = []
    states: List[np.ndarray] = []
    active_agents_by_env: List[List[str]] = []
    active_indices_by_env: List[List[int]] = []
    obs_batches: List[np.ndarray] = []

    for env in envs:
        active_agents = list(env.agents)
        if not active_agents:
            raise ValueError("Cannot select MAPPO actions when an environment has no active agents.")

        obs_matrix = env.get_obs_matrix()
        state = env.state()
        active_indices = active_agent_indices(env)

        obs_matrices.append(obs_matrix)
        states.append(state)
        active_agents_by_env.append(active_agents)
        active_indices_by_env.append(active_indices)
        obs_batches.append(obs_matrix[active_indices])

    output = policy.act_parallel(
        obs_batches,
        np.asarray(states, dtype=np.float32),
        deterministic=deterministic,
        update_stats=update_stats,
    )

    steps: List[PolicyStep] = []
    for env_idx, env in enumerate(envs):
        action_matrix = np.zeros(
            (env.max_agents, policy.action_dim),
            dtype=np.float32,
        )
        raw_action_matrix = np.zeros_like(action_matrix)
        log_prob_vector = np.zeros(env.max_agents, dtype=np.float32)
        mask = np.zeros(env.max_agents, dtype=np.float32)

        active_indices = active_indices_by_env[env_idx]
        active_actions = output["actions"][env_idx]
        active_raw_actions = output["raw_actions"][env_idx]
        active_log_probs = output["log_probs"][env_idx]
        active_normalized_obs = output["normalized_obs"][env_idx]
        normalized_state = np.asarray(
            output["normalized_states"][env_idx],
            dtype=np.float32,
        )

        action_matrix[active_indices] = active_actions
        raw_action_matrix[active_indices] = active_raw_actions
        log_prob_vector[active_indices] = active_log_probs
        mask[active_indices] = 1.0

        normalized_obs_matrix = np.zeros_like(obs_matrices[env_idx], dtype=np.float32)
        normalized_obs_matrix[active_indices] = active_normalized_obs

        value = output["values"][env_idx]
        if np.ndim(value) == 0:
            value = float(value)
        else:
            value = np.asarray(value, dtype=np.float32)

        steps.append(
            PolicyStep(
                obs_matrix=obs_matrices[env_idx],
                normalized_obs_matrix=normalized_obs_matrix,
                state=states[env_idx],
                normalized_state=normalized_state,
                mask=mask,
                action_matrix=action_matrix,
                raw_action_matrix=raw_action_matrix,
                log_prob_vector=log_prob_vector,
                active_agents=active_agents_by_env[env_idx],
                active_indices=active_indices,
                value=value,
            )
        )

    return steps


def action_dict_from_step(step: PolicyStep) -> Dict[str, np.ndarray]:
    return {
        agent: step.action_matrix[global_idx]
        for agent, global_idx in zip(step.active_agents, step.active_indices)
    }
