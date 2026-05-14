from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

from flight_engine.helpers import FlightMode
from mappo import MAPPOPolicy


def _abs_angle_delta(after: float, before: float) -> float:
    return float(abs(np.arctan2(np.sin(after - before), np.cos(after - before))))


def _finite_row_max(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size <= 0:
        return 0.0
    return float(np.max(finite_values))


class HierarchicalManagerEnv:
    def __init__(
        self,
        *,
        base_env,
        route_skill_policy: MAPPOPolicy,
        avoid_skill_policy: MAPPOPolicy,
        skill_deterministic: bool = True,
        avoid_option_sticky_enabled: bool = True,
        avoid_option_min_steps: int = 8,
        avoid_option_handoff_pressure_threshold: float = 0.20,
        avoid_option_handoff_boundary_pressure_threshold: float = 0.20,
        avoid_option_handoff_min_separation_ratio: float = 1.0,
        avoid_option_loop_breakout_enabled: bool = True,
        avoid_option_loop_breakout_min_steps: int = 40,
        avoid_option_loop_breakout_turns: float = 0.60,
        avoid_option_loop_breakout_max_displacement_efficiency: float = 0.55,
        avoid_option_loop_breakout_hazard_threshold: float = 0.65,
        avoid_option_loop_breakout_boundary_threshold: float = 0.60,
        avoid_option_loop_breakout_min_separation_ratio: float = 0.60,
        avoid_option_loop_breakout_route_steps: int = 80,
        reset_options_template: dict | None = None,
    ):
        self.base_env = base_env
        self.route_skill_policy = route_skill_policy
        self.avoid_skill_policy = avoid_skill_policy
        self.skill_deterministic = bool(skill_deterministic)
        self.avoid_option_sticky_enabled = bool(avoid_option_sticky_enabled)
        self.avoid_option_min_steps = max(int(avoid_option_min_steps), 0)
        self.avoid_option_handoff_pressure_threshold = max(
            float(avoid_option_handoff_pressure_threshold),
            0.0,
        )
        self.avoid_option_handoff_boundary_pressure_threshold = max(
            float(avoid_option_handoff_boundary_pressure_threshold),
            0.0,
        )
        self.avoid_option_handoff_min_separation_ratio = max(
            float(avoid_option_handoff_min_separation_ratio),
            0.0,
        )
        self.avoid_option_loop_breakout_enabled = bool(
            avoid_option_loop_breakout_enabled
        )
        self.avoid_option_loop_breakout_min_steps = max(
            int(avoid_option_loop_breakout_min_steps),
            1,
        )
        self.avoid_option_loop_breakout_turns = max(
            float(avoid_option_loop_breakout_turns),
            0.25,
        )
        self.avoid_option_loop_breakout_max_displacement_efficiency = float(
            np.clip(avoid_option_loop_breakout_max_displacement_efficiency, 0.0, 1.0)
        )
        self.avoid_option_loop_breakout_hazard_threshold = max(
            float(avoid_option_loop_breakout_hazard_threshold),
            0.0,
        )
        self.avoid_option_loop_breakout_boundary_threshold = max(
            float(avoid_option_loop_breakout_boundary_threshold),
            0.0,
        )
        self.avoid_option_loop_breakout_min_separation_ratio = max(
            float(avoid_option_loop_breakout_min_separation_ratio),
            0.0,
        )
        self.avoid_option_loop_breakout_route_steps = max(
            int(avoid_option_loop_breakout_route_steps),
            1,
        )
        self.reset_options_template = dict(reset_options_template or {})

        self.skill_names = ("route_follow", "avoid")
        self.route_skill_index = 0
        self.avoid_skill_index = 1
        self.manager_feature_names = (
            "route_action",
            "avoid_action",
            "neighbor_avoidance_pressure",
            "boundary_avoidance_pressure",
            "current_avoidance_pressure",
            "avoid_option_active",
            "avoid_option_handoff_safe",
            "avoid_option_hold_fraction",
        )
        self.manager_feature_dim = len(self.manager_feature_names)

        self.dt = float(base_env.dt)
        self.max_agents = int(base_env.max_agents)
        self.possible_agents = list(base_env.possible_agents)
        self.agent_name_to_index = dict(base_env.agent_name_to_index)
        self.agents = list(base_env.agents)
        self.action_dim = len(self.skill_names)
        self.obs_dim = int(base_env.obs_dim) + self.manager_feature_dim
        self.state_dim = (self.max_agents * self.obs_dim) + self.max_agents + 2

        self.observation_spaces = {
            agent: spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self._manager_obs_cache = np.zeros((self.max_agents, self.obs_dim), dtype=np.float32)
        self._manager_extra_cache = np.zeros(
            (self.max_agents, self.manager_feature_dim),
            dtype=np.float32,
        )
        self._state_cache = np.zeros(self.state_dim, dtype=np.float32)
        self._route_action_cache = np.zeros(self.max_agents, dtype=np.float32)
        self._avoid_action_cache = np.zeros(self.max_agents, dtype=np.float32)
        self._last_policy_train_mask = np.zeros(self.max_agents, dtype=np.float32)
        self._last_selected_skill_indices = np.zeros(self.max_agents, dtype=np.int32)
        self._last_requested_skill_indices = np.zeros(self.max_agents, dtype=np.int32)
        self._last_forced_avoid_mask = np.zeros(self.max_agents, dtype=np.float32)
        self._last_forced_route_breakout_mask = np.zeros(
            self.max_agents,
            dtype=np.float32,
        )
        self._skill_step_totals = np.zeros((self.max_agents, self.action_dim), dtype=np.int32)
        self._requested_skill_step_totals = np.zeros(
            (self.max_agents, self.action_dim),
            dtype=np.int32,
        )
        self._forced_avoid_step_totals = np.zeros(self.max_agents, dtype=np.int32)
        self._forced_route_breakout_step_totals = np.zeros(
            self.max_agents,
            dtype=np.int32,
        )
        self._avoid_option_loop_breakout_events = np.zeros(
            self.max_agents,
            dtype=np.int32,
        )
        self._avoid_option_active = np.zeros(self.max_agents, dtype=bool)
        self._avoid_option_steps = np.zeros(self.max_agents, dtype=np.int32)
        self._avoid_option_loop_steps = np.zeros(self.max_agents, dtype=np.int32)
        self._avoid_option_loop_path_length = np.zeros(self.max_agents, dtype=np.float32)
        self._avoid_option_loop_heading_travel = np.zeros(
            self.max_agents,
            dtype=np.float32,
        )
        self._avoid_option_loop_start_local = np.zeros(
            (self.max_agents, 2),
            dtype=np.float32,
        )
        self._avoid_option_breakout_steps_remaining = np.zeros(
            self.max_agents,
            dtype=np.int32,
        )
        self._cache_valid = False

    def __getattr__(self, name: str):
        return getattr(self.base_env, name)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        return self.base_env.seed(seed)

    def _manager_feature_payload(self, idx: int) -> dict:
        return {
            name: float(self._manager_extra_cache[idx, feature_idx])
            for feature_idx, name in enumerate(self.manager_feature_names)
        }

    def _reset_options(self, options: dict | None) -> dict | None:
        if not self.reset_options_template and not options:
            return None
        reset_options = dict(self.reset_options_template)
        reset_options.update(dict(options or {}))
        return reset_options

    def _decode_skill_selection(self, raw_action: Any) -> int:
        action_arr = np.asarray(raw_action)
        if action_arr.ndim == 0:
            return int(np.clip(int(round(float(action_arr.item()))), 0, self.action_dim - 1))
        flat = action_arr.reshape(-1)
        if flat.size <= 0:
            return self.route_skill_index
        if flat.size == 1:
            return int(np.clip(int(round(float(flat[0]))), 0, self.action_dim - 1))
        return int(np.argmax(flat[: self.action_dim]))

    def _sync_agents(self) -> None:
        self.agents = list(self.base_env.agents)

    def _invalidate_cache(self) -> None:
        self._cache_valid = False

    def _obs_batch_for_policy(
        self,
        policy: MAPPOPolicy,
        *,
        base_obs: np.ndarray,
        active_indices: List[int],
        neutralize_waypoint_features: bool = False,
    ) -> np.ndarray:
        if int(policy.obs_dim) == int(self.base_env.obs_dim):
            policy_obs = np.asarray(base_obs[active_indices], dtype=np.float32)
            if neutralize_waypoint_features:
                policy_obs = policy_obs.copy()
                self._neutralize_waypoint_features_for_policy(
                    policy_obs,
                    policy_base_obs_dim=int(self.base_env.base_obs_dim),
                )
            return policy_obs

        obs_stack_size = int(self.base_env.obs_stack_size)
        source_base_obs_dim = int(policy.obs_dim) // max(obs_stack_size, 1)
        if (
            obs_stack_size <= 0
            or source_base_obs_dim <= 0
            or source_base_obs_dim * obs_stack_size != int(policy.obs_dim)
        ):
            raise ValueError(
                "Skill policy observation shape is incompatible with manager env: "
                f"policy_obs_dim={policy.obs_dim}, env_obs_dim={self.base_env.obs_dim}"
            )

        copied_self_features = min(
            int(self.base_env.self_feature_count),
            source_base_obs_dim,
            int(self.base_env.base_obs_dim),
        )
        projected_obs = np.zeros(
            (len(active_indices), int(policy.obs_dim)),
            dtype=np.float32,
        )
        self._fill_neutral_skill_non_self_features(
            projected_obs,
            source_base_obs_dim=source_base_obs_dim,
            copied_self_features=copied_self_features,
        )
        for row_idx, agent_idx in enumerate(active_indices):
            for stack_idx in range(obs_stack_size):
                source_offset = stack_idx * int(self.base_env.base_obs_dim)
                target_offset = stack_idx * source_base_obs_dim
                projected_obs[
                    row_idx,
                    target_offset: target_offset + copied_self_features,
                ] = base_obs[
                    agent_idx,
                    source_offset: source_offset + copied_self_features,
                ]
        if neutralize_waypoint_features:
            self._neutralize_waypoint_features_for_policy(
                projected_obs,
                policy_base_obs_dim=source_base_obs_dim,
            )
        return projected_obs

    def _neutralize_waypoint_features_for_policy(
        self,
        obs_batch: np.ndarray,
        *,
        policy_base_obs_dim: int,
    ) -> None:
        feature_indices = getattr(self.base_env, "self_feature_indices", {})
        neutral_values = {
            "waypoint_distance": 0.0,
            "waypoint_bearing_sin": 0.0,
            "waypoint_bearing_cos": 1.0,
            "next_leg_turn_sin": 0.0,
            "next_leg_turn_cos": 1.0,
            "has_next_waypoint": 0.0,
            "route_reference_action": 0.0,
            "has_active_waypoint": 0.0,
            "remaining_waypoints_ratio": 0.0,
        }
        obs_stack_size = int(self.base_env.obs_stack_size)
        for stack_idx in range(obs_stack_size):
            stack_offset = stack_idx * int(policy_base_obs_dim)
            for feature_name, neutral_value in neutral_values.items():
                feature_idx = feature_indices.get(feature_name)
                if feature_idx is None or int(feature_idx) >= int(policy_base_obs_dim):
                    continue
                obs_batch[:, stack_offset + int(feature_idx)] = float(neutral_value)

    def _fill_neutral_skill_non_self_features(
        self,
        projected_obs: np.ndarray,
        *,
        source_base_obs_dim: int,
        copied_self_features: int,
    ) -> None:
        tail_width = int(source_base_obs_dim) - int(copied_self_features)
        if tail_width <= 0:
            return
        neutral_neighbor = np.asarray(
            getattr(self.base_env, "NO_NEIGHBOR_FEATURE_VECTOR", ()),
            dtype=np.float32,
        )
        if neutral_neighbor.size <= 0:
            return

        neutral_tail = np.zeros(tail_width, dtype=np.float32)
        cursor = 0
        while cursor < tail_width:
            copy_width = min(int(neutral_neighbor.size), tail_width - cursor)
            neutral_tail[cursor: cursor + copy_width] = neutral_neighbor[:copy_width]
            cursor += copy_width

        obs_stack_size = int(self.base_env.obs_stack_size)
        for stack_idx in range(obs_stack_size):
            target_offset = stack_idx * int(source_base_obs_dim)
            tail_start = target_offset + int(copied_self_features)
            tail_stop = target_offset + int(source_base_obs_dim)
            projected_obs[:, tail_start:tail_stop] = neutral_tail

    def _state_for_policy(
        self,
        policy: MAPPOPolicy,
        *,
        base_state: np.ndarray,
        policy_obs_batch: np.ndarray,
    ) -> np.ndarray:
        if int(policy.state_dim) == int(self.base_env.state_dim):
            return np.asarray(base_state, dtype=np.float32)

        source_max_agents = 0
        denom = int(policy.obs_dim) + 1
        numerator = int(policy.state_dim) - 2
        if denom > 0 and numerator > 0 and numerator % denom == 0:
            source_max_agents = int(numerator // denom)
        if source_max_agents <= 0:
            raise ValueError(
                "Skill policy state shape is incompatible with manager env: "
                f"policy_state_dim={policy.state_dim}, env_state_dim={self.base_env.state_dim}"
            )

        projected_state = np.zeros(int(policy.state_dim), dtype=np.float32)
        copied_agents = min(source_max_agents, int(policy_obs_batch.shape[0]))
        for local_idx in range(copied_agents):
            start = local_idx * int(policy.obs_dim)
            stop = start + int(policy.obs_dim)
            projected_state[start:stop] = policy_obs_batch[local_idx]

        active_mask_start = source_max_agents * int(policy.obs_dim)
        projected_state[active_mask_start: active_mask_start + copied_agents] = 1.0
        projected_state[-2:] = np.asarray(
            [
                self.base_env.box_width_m / max(self.base_env.map_size_scale, 1.0),
                self.base_env.box_height_m / max(self.base_env.map_size_scale, 1.0),
            ],
            dtype=np.float32,
        )
        return projected_state

    def _skill_policy_actions(
        self,
        policy: MAPPOPolicy,
        *,
        base_obs: np.ndarray,
        base_state: np.ndarray,
        neutralize_waypoint_features: bool = False,
    ) -> np.ndarray:
        actions = np.zeros(self.max_agents, dtype=np.float32)
        active_indices = [
            self.agent_name_to_index[agent]
            for agent in self.base_env.agents
        ]
        if not active_indices:
            return actions

        active_obs = self._obs_batch_for_policy(
            policy,
            base_obs=base_obs,
            active_indices=active_indices,
            neutralize_waypoint_features=neutralize_waypoint_features,
        )
        policy_state = self._state_for_policy(
            policy,
            base_state=base_state,
            policy_obs_batch=active_obs,
        )
        output = policy.act_parallel(
            [active_obs],
            np.asarray([policy_state], dtype=np.float32),
            deterministic=self.skill_deterministic,
            update_stats=False,
        )
        active_actions = output["actions"][0].reshape(-1)
        actions[active_indices] = active_actions.astype(np.float32, copy=False)
        return actions

    def _predicted_critical_pressure(self, idx: int) -> float:
        predicted_critical_idx = getattr(
            self.base_env,
            "self_feature_indices",
            {},
        ).get("max_predicted_critical_pressure")
        if predicted_critical_idx is None:
            return 1.0 if self.base_env._deconfliction_active[idx] else 0.0
        return float(
            np.clip(
                self.base_env._base_obs_cache[idx, int(predicted_critical_idx)],
                0.0,
                1.0,
            )
        )

    def _current_min_separation_ratio(self, idx: int) -> float:
        other_indices = [
            self.agent_name_to_index[other_agent]
            for other_agent in self.base_env.agents
            if self.agent_name_to_index[other_agent] != idx
        ]
        if not other_indices:
            return 2.0
        min_separation = float(np.min(self.base_env._dist_matrix[idx, other_indices]))
        return float(
            np.clip(
                min_separation / max(float(self.base_env.caution_dist), 1.0),
                0.0,
                2.0,
            )
        )

    def _avoid_option_handoff_metrics(self, agent: str, idx: int) -> dict[str, float]:
        pair_conflict = _finite_row_max(
            self.base_env._last_pair_conflict_pressure_matrix[idx]
        )
        predicted_critical_pressure = self._predicted_critical_pressure(idx)
        boundary_pressure = max(
            float(self.base_env._last_boundary_avoidance_pressure[idx]),
            float(self.base_env._boundary_status(agent)[1]),
        )
        hazard_pressure = max(
            pair_conflict,
            float(self.base_env._last_neighbor_avoidance_pressure[idx]),
            1.0 if self.base_env._deconfliction_active[idx] else 0.0,
        )
        current_avoidance_pressure = max(
            hazard_pressure,
            predicted_critical_pressure,
            boundary_pressure,
        )
        min_separation_ratio = self._current_min_separation_ratio(idx)
        handoff_safe = bool(
            current_avoidance_pressure
            <= self.avoid_option_handoff_pressure_threshold
            and boundary_pressure
            <= self.avoid_option_handoff_boundary_pressure_threshold
            and min_separation_ratio
            >= self.avoid_option_handoff_min_separation_ratio
        )
        return {
            "current_avoidance_pressure": float(
                np.clip(current_avoidance_pressure, 0.0, 2.0)
            ),
            "boundary_pressure": float(np.clip(boundary_pressure, 0.0, 2.0)),
            "min_separation_ratio": float(min_separation_ratio),
            "handoff_safe": float(handoff_safe),
        }

    def _avoid_option_breakout_hazard_pressure(self, agent: str, idx: int) -> float:
        pair_conflict = _finite_row_max(
            self.base_env._last_pair_conflict_pressure_matrix[idx]
        )
        boundary_pressure = max(
            float(self.base_env._last_boundary_avoidance_pressure[idx]),
            float(self.base_env._boundary_status(agent)[1]),
        )
        return float(
            np.clip(
                max(
                    pair_conflict,
                    float(self.base_env._last_neighbor_avoidance_pressure[idx]),
                    self._predicted_critical_pressure(idx),
                    boundary_pressure,
                ),
                0.0,
                2.0,
            )
        )

    def _avoid_option_breakout_safe_from_cache(self, agent: str, idx: int) -> bool:
        boundary_pressure = self._manager_feature_value(
            idx,
            "boundary_avoidance_pressure",
        )
        min_separation_ratio = self._current_min_separation_ratio(idx)
        hazard_pressure = self._avoid_option_breakout_hazard_pressure(agent, idx)
        return bool(
            hazard_pressure <= self.avoid_option_loop_breakout_hazard_threshold
            and boundary_pressure
            <= self.avoid_option_loop_breakout_boundary_threshold
            and min_separation_ratio
            >= self.avoid_option_loop_breakout_min_separation_ratio
        )

    def _avoid_option_breakout_ready_from_cache(self, agent: str, idx: int) -> bool:
        if (
            not self.avoid_option_loop_breakout_enabled
            or not self._avoid_option_active[idx]
            or self._avoid_option_breakout_steps_remaining[idx] > 0
            or not self._avoid_option_breakout_safe_from_cache(agent, idx)
        ):
            return False

        if int(self._avoid_option_loop_steps[idx]) < int(
            self.avoid_option_loop_breakout_min_steps
        ):
            return False
        turn_count = float(self._avoid_option_loop_heading_travel[idx]) / (
            2.0 * np.pi
        )
        if turn_count < self.avoid_option_loop_breakout_turns:
            return False
        path_length = float(self._avoid_option_loop_path_length[idx])
        if path_length <= 1e-6:
            return False
        displacement = float(
            np.linalg.norm(
                self.base_env._local_pos_cache[idx]
                - self._avoid_option_loop_start_local[idx]
            )
        )
        displacement_efficiency = displacement / max(path_length, 1e-6)
        return bool(
            displacement_efficiency
            <= self.avoid_option_loop_breakout_max_displacement_efficiency
        )

    def _start_avoid_option_breakout(self, idx: int) -> None:
        if self._avoid_option_breakout_steps_remaining[idx] <= 0:
            self._avoid_option_loop_breakout_events[idx] += 1
        self._avoid_option_breakout_steps_remaining[idx] = int(
            self.avoid_option_loop_breakout_route_steps
        )

    def _clear_avoid_option_loop_state(self, idx: int) -> None:
        self._avoid_option_loop_steps[idx] = 0
        self._avoid_option_loop_path_length[idx] = 0.0
        self._avoid_option_loop_heading_travel[idx] = 0.0
        self._avoid_option_loop_start_local[idx] = self.base_env._local_pos_cache[idx]

    def _update_avoid_option_loop_state(
        self,
        *,
        active_agents: List[str],
        executed_skill_indices: dict[str, int],
        pre_local_positions: np.ndarray,
        pre_headings: np.ndarray,
    ) -> None:
        for agent in active_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.base_env.aircraft_by_agent[agent]
            if executed_skill_indices.get(agent) != self.avoid_skill_index:
                self._clear_avoid_option_loop_state(idx)
                continue

            if int(self._avoid_option_loop_steps[idx]) <= 0:
                self._avoid_option_loop_start_local[idx] = pre_local_positions[idx]
            self._avoid_option_loop_steps[idx] += 1
            self._avoid_option_loop_path_length[idx] += float(
                np.linalg.norm(
                    self.base_env._local_pos_cache[idx] - pre_local_positions[idx]
                )
            )
            self._avoid_option_loop_heading_travel[idx] += _abs_angle_delta(
                float(aircraft.heading),
                float(pre_headings[idx]),
            )

    def _refresh_cache(self) -> None:
        if self._cache_valid:
            return
        self._sync_agents()
        self._manager_extra_cache[:] = 0.0

        base_obs = self.base_env.get_obs_matrix()
        base_state = self.base_env.state()
        self._route_action_cache[:] = self._skill_policy_actions(
            self.route_skill_policy,
            base_obs=base_obs,
            base_state=base_state,
        )
        self._avoid_action_cache[:] = self._skill_policy_actions(
            self.avoid_skill_policy,
            base_obs=base_obs,
            base_state=base_state,
            neutralize_waypoint_features=True,
        )
        for agent in self.base_env.agents:
            idx = self.agent_name_to_index[agent]
            route_action = float(self._route_action_cache[idx])
            avoid_action = float(self._avoid_action_cache[idx])
            handoff_metrics = self._avoid_option_handoff_metrics(agent, idx)
            current_avoidance_pressure = float(
                handoff_metrics["current_avoidance_pressure"]
            )
            hold_denominator = max(int(self.avoid_option_min_steps), 1)
            avoid_option_hold_fraction = float(
                np.clip(
                    float(self._avoid_option_steps[idx]) / float(hold_denominator),
                    0.0,
                    2.0,
                )
            )

            self._manager_extra_cache[idx] = np.asarray(
                [
                    route_action,
                    avoid_action,
                    float(self.base_env._last_neighbor_avoidance_pressure[idx]),
                    float(self.base_env._last_boundary_avoidance_pressure[idx]),
                    current_avoidance_pressure,
                    1.0 if self._avoid_option_active[idx] else 0.0,
                    float(handoff_metrics["handoff_safe"]),
                    avoid_option_hold_fraction,
                ],
                dtype=np.float32,
            )

        self._manager_obs_cache[:] = np.concatenate(
            [base_obs, self._manager_extra_cache],
            axis=1,
        ).astype(np.float32, copy=False)
        active_mask = np.zeros(self.max_agents, dtype=np.float32)
        for agent in self.agents:
            active_mask[self.agent_name_to_index[agent]] = 1.0
        self._state_cache[:] = np.concatenate(
            [
                self._manager_obs_cache.reshape(-1),
                active_mask,
                np.asarray(
                    [
                        self.base_env.box_width_m / max(self.base_env.map_size_scale, 1.0),
                        self.base_env.box_height_m / max(self.base_env.map_size_scale, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32, copy=False)
        self._cache_valid = True

    def _manager_feature_value(self, idx: int, name: str) -> float:
        return float(
            self._manager_extra_cache[idx, self.manager_feature_names.index(name)]
        )

    def _avoid_option_can_handoff_from_cache(self, idx: int) -> bool:
        if not self.avoid_option_sticky_enabled:
            return True
        min_steps_met = int(self._avoid_option_steps[idx]) >= int(
            self.avoid_option_min_steps
        )
        handoff_safe = (
            self._manager_feature_value(idx, "avoid_option_handoff_safe") >= 0.5
        )
        return bool(min_steps_met and handoff_safe)

    def get_obs_matrix(self) -> np.ndarray:
        self._refresh_cache()
        return self._manager_obs_cache.copy()

    def state(self) -> np.ndarray:
        self._refresh_cache()
        return self._state_cache.copy()

    def reset(self, seed=None, options=None):
        reset_options = self._reset_options(options)
        self.base_env.reset(seed=seed, options=reset_options)
        self.base_env._refresh_skill_guidance_cache()
        self.base_env._update_obs_cache(fill_history=True)
        self._sync_agents()
        self._last_policy_train_mask[:] = 0.0
        self._last_selected_skill_indices[:] = self.route_skill_index
        self._last_requested_skill_indices[:] = self.route_skill_index
        self._last_forced_avoid_mask[:] = 0.0
        self._last_forced_route_breakout_mask[:] = 0.0
        self._skill_step_totals[:] = 0
        self._requested_skill_step_totals[:] = 0
        self._forced_avoid_step_totals[:] = 0
        self._forced_route_breakout_step_totals[:] = 0
        self._avoid_option_loop_breakout_events[:] = 0
        self._avoid_option_active[:] = False
        self._avoid_option_steps[:] = 0
        self._avoid_option_loop_steps[:] = 0
        self._avoid_option_loop_path_length[:] = 0.0
        self._avoid_option_loop_heading_travel[:] = 0.0
        self._avoid_option_loop_start_local[:] = self.base_env._local_pos_cache
        self._avoid_option_breakout_steps_remaining[:] = 0
        self._invalidate_cache()
        self._refresh_cache()

        obs = {
            agent: self._manager_obs_cache[self.agent_name_to_index[agent]].copy()
            for agent in self.agents
        }
        infos = {
            agent: {
                **self.base_env._shared_info(),
                "selected_skill": self.skill_names[int(self._last_selected_skill_indices[self.agent_name_to_index[agent]])],
                "requested_skill": self.skill_names[int(self._last_requested_skill_indices[self.agent_name_to_index[agent]])],
                "forced_avoid": bool(self._last_forced_avoid_mask[self.agent_name_to_index[agent]]),
                "forced_route_breakout": bool(self._last_forced_route_breakout_mask[self.agent_name_to_index[agent]]),
                "avoid_option_active": bool(self._avoid_option_active[self.agent_name_to_index[agent]]),
                "avoid_option_steps": int(self._avoid_option_steps[self.agent_name_to_index[agent]]),
                "manager_features": self._manager_feature_payload(
                    self.agent_name_to_index[agent]
                ),
            }
            for agent in self.agents
        }
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        self._sync_agents()
        if not self.agents:
            return {}, {}, {}, {}, {}

        self._refresh_cache()
        executed_skill_indices: dict[str, int] = {}
        forced_avoid_by_agent: dict[str, bool] = {}
        forced_route_breakout_by_agent: dict[str, bool] = {}
        requested_skill_indices: dict[str, int] = {}
        controllable_by_agent: dict[str, bool] = {}
        needs_route_reanchor = False
        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.base_env.aircraft_by_agent[agent]
            action_controls_aircraft = bool(
                aircraft.flight_mode != FlightMode.LOITERING
                and aircraft.waypoint_manager.current_waypoint is not None
            )
            skill_idx = (
                self._decode_skill_selection(
                    actions.get(agent, np.asarray([1.0, 0.0], dtype=np.float32))
                )
                if action_controls_aircraft
                else self.route_skill_index
            )
            self._last_requested_skill_indices[idx] = skill_idx
            requested_skill_indices[agent] = skill_idx
            controllable_by_agent[agent] = action_controls_aircraft

        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.base_env.aircraft_by_agent[agent]
            action_controls_aircraft = bool(controllable_by_agent.get(agent, False))
            skill_idx = requested_skill_indices.get(agent, self.route_skill_index)
            forced_avoid = False
            forced_route_breakout = False
            if (
                action_controls_aircraft
                and self._avoid_option_breakout_steps_remaining[idx] > 0
                and self._avoid_option_breakout_safe_from_cache(agent, idx)
            ):
                executed_skill_idx = self.route_skill_index
                forced_route_breakout = skill_idx != self.route_skill_index
            elif (
                action_controls_aircraft
                and self._avoid_option_breakout_ready_from_cache(agent, idx)
            ):
                self._start_avoid_option_breakout(idx)
                executed_skill_idx = self.route_skill_index
                forced_route_breakout = skill_idx != self.route_skill_index
            elif (
                action_controls_aircraft
                and self.avoid_option_sticky_enabled
                and self._avoid_option_active[idx]
            ):
                if self._avoid_option_can_handoff_from_cache(idx):
                    executed_skill_idx = skill_idx
                else:
                    executed_skill_idx = self.avoid_skill_index
                    forced_avoid = skill_idx != self.avoid_skill_index
            elif action_controls_aircraft:
                executed_skill_idx = skill_idx
            else:
                executed_skill_idx = self.route_skill_index

            if not action_controls_aircraft:
                self._avoid_option_active[idx] = False
                self._avoid_option_steps[idx] = 0
                self._avoid_option_breakout_steps_remaining[idx] = 0
                self._clear_avoid_option_loop_state(idx)
            elif executed_skill_idx == self.avoid_skill_index:
                self._avoid_option_active[idx] = True
            else:
                self._avoid_option_active[idx] = False
                self._avoid_option_steps[idx] = 0

            if (
                action_controls_aircraft
                and int(self._last_selected_skill_indices[idx]) == self.avoid_skill_index
                and executed_skill_idx == self.route_skill_index
            ):
                self.base_env._reset_reference_route_progress(
                    idx=idx,
                    aircraft=aircraft,
                )
                needs_route_reanchor = True

            executed_skill_indices[agent] = executed_skill_idx
            forced_avoid_by_agent[agent] = forced_avoid
            forced_route_breakout_by_agent[agent] = forced_route_breakout

        if needs_route_reanchor:
            self.base_env._refresh_skill_guidance_cache()
            self._invalidate_cache()

        self._refresh_cache()
        self._last_policy_train_mask[:] = 0.0
        pre_local_positions = np.asarray(
            self.base_env._local_pos_cache,
            dtype=np.float32,
        ).copy()
        pre_headings = np.zeros(self.max_agents, dtype=np.float32)
        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            pre_headings[idx] = float(self.base_env.aircraft_by_agent[agent].heading)

        low_level_actions = {}
        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.base_env.aircraft_by_agent[agent]
            action_controls_aircraft = bool(
                aircraft.flight_mode != FlightMode.LOITERING
                and aircraft.waypoint_manager.current_waypoint is not None
            )
            skill_idx = executed_skill_indices.get(agent, self.route_skill_index)
            forced_avoid = bool(forced_avoid_by_agent.get(agent, False))
            forced_route_breakout = bool(
                forced_route_breakout_by_agent.get(agent, False)
            )
            if action_controls_aircraft:
                self._last_selected_skill_indices[idx] = skill_idx
                self._skill_step_totals[idx, skill_idx] += 1
                requested_skill_idx = requested_skill_indices.get(
                    agent,
                    self.route_skill_index,
                )
                self._requested_skill_step_totals[idx, requested_skill_idx] += 1
                self._last_forced_avoid_mask[idx] = 1.0 if forced_avoid else 0.0
                self._last_forced_route_breakout_mask[idx] = (
                    1.0 if forced_route_breakout else 0.0
                )
                if forced_avoid:
                    self._forced_avoid_step_totals[idx] += 1
                if forced_route_breakout:
                    self._forced_route_breakout_step_totals[idx] += 1
                if skill_idx == self.avoid_skill_index:
                    self._avoid_option_steps[idx] += 1
                elif self._avoid_option_breakout_steps_remaining[idx] > 0:
                    self._avoid_option_breakout_steps_remaining[idx] -= 1
                self._last_policy_train_mask[idx] = (
                    0.0 if (forced_avoid or forced_route_breakout) else 1.0
                )
            else:
                skill_idx = self.route_skill_index
                self._last_selected_skill_indices[idx] = skill_idx
                self._last_requested_skill_indices[idx] = skill_idx
                self._last_forced_avoid_mask[idx] = 0.0
                self._last_forced_route_breakout_mask[idx] = 0.0

            selected_action = (
                self._avoid_action_cache[idx]
                if skill_idx == self.avoid_skill_index
                else self._route_action_cache[idx]
            )
            low_level_actions[agent] = np.asarray([selected_action], dtype=np.float32)

        _, rewards, terminations, truncations, infos = self.base_env.step(low_level_actions)
        self._update_avoid_option_loop_state(
            active_agents=list(low_level_actions.keys()),
            executed_skill_indices=executed_skill_indices,
            pre_local_positions=pre_local_positions,
            pre_headings=pre_headings,
        )
        self._sync_agents()
        self._invalidate_cache()

        if self.agents:
            self._refresh_cache()
            obs = {
                agent: self._manager_obs_cache[self.agent_name_to_index[agent]].copy()
                for agent in self.agents
            }
        else:
            obs = {}

        augmented_infos = {}
        for agent, info in infos.items():
            idx = self.agent_name_to_index[agent]
            augmented_infos[agent] = {
                **info,
                "selected_skill": self.skill_names[int(self._last_selected_skill_indices[idx])],
                "selected_skill_id": int(self._last_selected_skill_indices[idx]),
                "requested_skill": self.skill_names[int(self._last_requested_skill_indices[idx])],
                "requested_skill_id": int(self._last_requested_skill_indices[idx]),
                "forced_avoid": bool(self._last_forced_avoid_mask[idx]),
                "forced_route_breakout": bool(self._last_forced_route_breakout_mask[idx]),
                "avoid_option_active": bool(self._avoid_option_active[idx]),
                "avoid_option_steps": int(self._avoid_option_steps[idx]),
                "avoid_option_breakout_steps_remaining": int(
                    self._avoid_option_breakout_steps_remaining[idx]
                ),
                "manager_features": self._manager_feature_payload(idx),
            }

        return obs, rewards, terminations, truncations, augmented_infos

    def get_episode_metrics(self) -> dict:
        metrics = self.base_env.get_episode_metrics()
        episode_summary = metrics.setdefault("episode_summary", {})
        episode_summary["route_skill_steps_total"] = int(
            np.sum(self._skill_step_totals[:, self.route_skill_index])
        )
        episode_summary["avoid_skill_steps_total"] = int(
            np.sum(self._skill_step_totals[:, self.avoid_skill_index])
        )
        episode_summary["requested_route_skill_steps_total"] = int(
            np.sum(self._requested_skill_step_totals[:, self.route_skill_index])
        )
        episode_summary["requested_avoid_skill_steps_total"] = int(
            np.sum(self._requested_skill_step_totals[:, self.avoid_skill_index])
        )
        episode_summary["forced_avoid_steps_total"] = int(
            np.sum(self._forced_avoid_step_totals)
        )
        episode_summary["forced_route_breakout_steps_total"] = int(
            np.sum(self._forced_route_breakout_step_totals)
        )
        episode_summary["avoid_option_loop_breakout_events_total"] = int(
            np.sum(self._avoid_option_loop_breakout_events)
        )

        for mission_stat in metrics.get("mission_stats", []):
            agent = mission_stat["id"]
            idx = self.agent_name_to_index[agent]
            mission_stat["route_skill_steps"] = int(
                self._skill_step_totals[idx, self.route_skill_index]
            )
            mission_stat["avoid_skill_steps"] = int(
                self._skill_step_totals[idx, self.avoid_skill_index]
            )
            mission_stat["requested_route_skill_steps"] = int(
                self._requested_skill_step_totals[idx, self.route_skill_index]
            )
            mission_stat["requested_avoid_skill_steps"] = int(
                self._requested_skill_step_totals[idx, self.avoid_skill_index]
            )
            mission_stat["forced_avoid_steps"] = int(
                self._forced_avoid_step_totals[idx]
            )
            mission_stat["forced_route_breakout_steps"] = int(
                self._forced_route_breakout_step_totals[idx]
            )
            mission_stat["avoid_option_loop_breakout_events"] = int(
                self._avoid_option_loop_breakout_events[idx]
            )
        return metrics

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


HRLManagerEnv = HierarchicalManagerEnv
