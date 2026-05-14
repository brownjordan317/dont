from __future__ import annotations

from typing import Dict, List

import numpy as np

from flight_engine.helpers import FlightMode
from flight_engine.reference_guidance import time_to_boundary_ahead


class _SkillTrainingEnvBase:
    def __init__(self, *, base_env, train_cfg: dict):
        self.base_env = base_env
        self.train_cfg = train_cfg
        self._prev_metrics: Dict[str, dict] = {}

    def __getattr__(self, name: str):
        return getattr(self.base_env, name)

    def observation_space(self, agent: str):
        return self.base_env.observation_space(agent)

    def action_space(self, agent: str):
        return self.base_env.action_space(agent)

    def seed(self, seed=None):
        return self.base_env.seed(seed)

    def get_obs_matrix(self):
        return self.base_env.get_obs_matrix()

    def state(self):
        return self.base_env.state()

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()

    def _base_info_dict(self) -> dict:
        return {
            agent: self.base_env._shared_info()
            for agent in self.base_env.agents
        }

    def _post_reset_response(self):
        obs_matrix = self.base_env.get_obs_matrix()
        obs = {
            agent: obs_matrix[self.base_env.agent_name_to_index[agent]].copy()
            for agent in self.base_env.agents
        }
        return obs, self._base_info_dict()


class RouteSkillTrainingEnv(_SkillTrainingEnvBase):
    def __init__(self, *, base_env, train_cfg: dict):
        super().__init__(base_env=base_env, train_cfg=train_cfg)
        self.route_num_agents = int(
            np.clip(
                train_cfg.get(
                    "route_skill_training_num_agents",
                    base_env.min_agents,
                ),
                base_env.min_agents,
                base_env.max_agents,
            )
        )
        self.geofence_breach_terminates = bool(
            train_cfg.get("route_skill_geofence_breach_terminates", True)
        )
        self.geofence_breach_grace_steps = max(
            int(train_cfg.get("route_skill_geofence_breach_grace_steps", 1)),
            1,
        )
        self.geofence_breach_penalty = float(
            train_cfg.get("route_skill_geofence_breach_penalty", 3200.0)
        )
        self.require_reference_route_in_bounds = bool(
            train_cfg.get("route_skill_require_reference_route_in_bounds", True)
        )
        self.reference_route_boundary_margin_m = max(
            float(train_cfg.get("route_skill_reference_route_boundary_margin_m", 0.0)),
            0.0,
        )
        self.reference_route_generation_attempts = max(
            int(
                train_cfg.get(
                    "route_skill_reference_route_generation_attempts",
                    getattr(base_env, "reset_generation_attempts", 128),
                )
            ),
            1,
        )
        self.time_pressure_penalty = float(
            train_cfg.get("route_skill_time_pressure_penalty", 0.0)
        )
        self.remaining_waypoint_penalty = float(
            train_cfg.get("route_skill_remaining_waypoint_penalty", 0.0)
        )

    def _reset_options(self, options: dict | None) -> dict:
        reset_options = dict(options or {})
        if self.base_env.randomized:
            max_span = float(self.base_env.map_size_range_m[1])
            reset_options.setdefault("num_agents", self.route_num_agents)
            reset_options.setdefault("box_width_m", max_span)
            reset_options.setdefault("box_height_m", max_span)
        return reset_options

    def _refresh_route_reference_cache(self) -> None:
        self.base_env._reference_route_cache = [
            None
            for _ in range(self.base_env.max_agents)
        ]
        self.base_env._route_progress_anchor[:] = 0.0
        self.base_env._refresh_skill_guidance_cache()
        self.base_env._update_obs_cache(fill_history=True)

    def _reference_route_is_in_bounds(self, route) -> bool:
        if route is None or getattr(route, "points", None) is None:
            return False
        points = np.asarray(route.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            return False
        if not np.all(np.isfinite(points[:, :2])):
            return False

        margin = float(self.reference_route_boundary_margin_m)
        min_x = float(self.base_env.local_min_x) + margin
        max_x = float(self.base_env.local_max_x) - margin
        min_y = float(self.base_env.local_min_y) + margin
        max_y = float(self.base_env.local_max_y) - margin
        if min_x > max_x:
            min_x = float(self.base_env.local_min_x)
            max_x = float(self.base_env.local_max_x)
        if min_y > max_y:
            min_y = float(self.base_env.local_min_y)
            max_y = float(self.base_env.local_max_y)

        x = points[:, 0]
        y = points[:, 1]
        return bool(
            np.all((x >= min_x) & (x <= max_x))
            and np.all((y >= min_y) & (y <= max_y))
        )

    def _all_reference_routes_in_bounds(self, agents: List[str]) -> bool:
        for agent in agents:
            idx = self.base_env.agent_name_to_index[agent]
            aircraft = self.base_env.aircraft_by_agent[agent]
            if aircraft.waypoint_manager.current_waypoint is None:
                continue
            if not self._reference_route_is_in_bounds(
                self.base_env._reference_route_cache[idx]
            ):
                return False
        return True

    def _collect_metrics(self, agents: List[str]) -> dict[str, dict]:
        metrics: dict[str, dict] = {}
        for agent in agents:
            aircraft = self.base_env.aircraft_by_agent[agent]
            _, remaining_waypoints, assigned_waypoints = (
                self.base_env._mission_waypoint_counts(aircraft)
            )
            metrics[agent] = {
                "remaining_waypoint_fraction": float(
                    np.clip(
                        remaining_waypoints / max(assigned_waypoints, 1),
                        0.0,
                        1.0,
                    )
                ),
            }
        return metrics

    def reset(self, seed=None, options=None):
        reset_options = self._reset_options(options)
        attempts = (
            self.reference_route_generation_attempts
            if self.require_reference_route_in_bounds
            else 1
        )
        for attempt_idx in range(attempts):
            attempt_seed = (
                None
                if seed is None
                else int(seed) + attempt_idx
            )
            self.base_env.reset(seed=attempt_seed, options=reset_options)
            self._refresh_route_reference_cache()
            if (
                not self.require_reference_route_in_bounds
                or self._all_reference_routes_in_bounds(list(self.base_env.agents))
            ):
                self._prev_metrics = self._collect_metrics(list(self.base_env.agents))
                return self._post_reset_response()

        raise RuntimeError(
            "Route skill could not generate an in-bounds Dubins reference route "
            f"after {attempts} attempts. Increase route_skill_box_size_m, reduce "
            "turning radius pressure, or relax route_skill_reference_route_boundary_margin_m."
        )

    def _geofence_breach_agents(self, agents: List[str]) -> list[str]:
        if not self.geofence_breach_terminates:
            return []
        breached = []
        for agent in agents:
            idx = self.base_env.agent_name_to_index[agent]
            inside, _, _ = self.base_env._boundary_status(agent)
            outside_steps = int(self.base_env.geofence_outside_steps[idx])
            if not inside and outside_steps >= self.geofence_breach_grace_steps:
                breached.append(agent)
        return breached

    def step(self, actions):
        current_agents = list(self.base_env.agents)
        obs, rewards, terminations, truncations, infos = self.base_env.step(actions)
        curr_metrics = self._collect_metrics(current_agents)
        geofence_breach_agents = self._geofence_breach_agents(current_agents)
        geofence_breach_agent_set = set(geofence_breach_agents)
        geofence_breach = bool(geofence_breach_agents)
        episode_progress = float(
            np.clip(
                self.base_env.current_step / max(self.base_env.max_steps, 1),
                0.0,
                1.0,
            )
        )

        agent_shaping: dict[str, dict] = {}
        shaped_rewards: dict[str, float] = {}
        for agent in current_agents:
            curr = curr_metrics[agent]
            agent_remaining_waypoint_fraction = float(
                curr["remaining_waypoint_fraction"]
            )
            agent_time_pressure = float(
                self.time_pressure_penalty * agent_remaining_waypoint_fraction
            )
            agent_remaining_waypoint_pressure = float(
                self.remaining_waypoint_penalty
                * agent_remaining_waypoint_fraction
                * (0.25 + (0.75 * episode_progress))
            )
            agent_geofence_breach_penalty = float(
                self.geofence_breach_penalty
                if agent in geofence_breach_agent_set
                else 0.0
            )
            agent_reward_bonus = float(
                -agent_time_pressure
                - agent_remaining_waypoint_pressure
                - agent_geofence_breach_penalty
            )
            agent_shaping[agent] = {
                "mode": "route_skill",
                "remaining_waypoint_fraction": agent_remaining_waypoint_fraction,
                "episode_progress": episode_progress,
                "time_pressure": agent_time_pressure,
                "remaining_waypoint_pressure": agent_remaining_waypoint_pressure,
                "geofence_breach": float(agent in geofence_breach_agent_set),
                "geofence_breach_penalty": agent_geofence_breach_penalty,
                "reward_bonus": agent_reward_bonus,
            }
            shaped_rewards[agent] = float(rewards.get(agent, 0.0)) + agent_reward_bonus

        geofence_breach_penalty = float(
            np.mean(
                [agent_shaping[a]["geofence_breach_penalty"] for a in current_agents]
            )
            if current_agents
            else 0.0
        )

        if current_agents:
            remaining_waypoint_fraction = float(np.mean([agent_shaping[a]["remaining_waypoint_fraction"] for a in current_agents]))
            time_pressure = float(np.mean([agent_shaping[a]["time_pressure"] for a in current_agents]))
            remaining_waypoint_pressure = float(np.mean([agent_shaping[a]["remaining_waypoint_pressure"] for a in current_agents]))
            shaping_reward = float(np.mean([agent_shaping[a]["reward_bonus"] for a in current_agents]))
            geofence_breach_mean = float(np.mean([agent_shaping[a]["geofence_breach"] for a in current_agents]))
        else:
            remaining_waypoint_fraction = 0.0
            time_pressure = 0.0
            remaining_waypoint_pressure = 0.0
            shaping_reward = 0.0
            geofence_breach_mean = 0.0

        shaping_info = {
            "mode": "route_skill",
            "remaining_waypoint_fraction": float(remaining_waypoint_fraction),
            "episode_progress": float(episode_progress),
            "time_pressure": float(time_pressure),
            "remaining_waypoint_pressure": float(remaining_waypoint_pressure),
            "geofence_breach": float(geofence_breach_mean),
            "geofence_breach_count": float(len(geofence_breach_agents)),
            "geofence_breach_penalty": float(geofence_breach_penalty),
            "reward_bonus": float(shaping_reward),
        }
        if geofence_breach:
            self.base_env._termination_reason = "geofence_violation"
            terminations = {agent: True for agent in current_agents}
            truncations = {agent: False for agent in current_agents}
            final_metrics = self.base_env.get_episode_metrics()
            for agent in current_agents:
                infos.setdefault(agent, {})
                infos[agent]["terminated"] = True
                infos[agent]["truncated"] = False
                infos[agent]["termination_reason"] = "geofence_violation"
                infos[agent]["episode_metrics"] = final_metrics
            obs = {}
            self.base_env.agents = []

        for agent in current_agents:
            infos.setdefault(agent, {})
            infos[agent]["skill_shaping"] = shaping_info
            infos[agent]["agent_skill_shaping"] = agent_shaping[agent]

        self._prev_metrics = (
            {}
            if any(terminations.values()) or any(truncations.values())
            else self._collect_metrics(list(self.base_env.agents))
        )
        return obs, shaped_rewards, terminations, truncations, infos


class AvoidSkillTrainingEnv(_SkillTrainingEnvBase):
    def __init__(self, *, base_env, train_cfg: dict):
        super().__init__(base_env=base_env, train_cfg=train_cfg)

        self.base_env.disable_waypoint_navigation = True

        self.survival_agents = int(
            np.clip(
                train_cfg.get("avoid_skill_survival_agents"),
                base_env.min_agents,
                base_env.max_agents,
            )
        )
        self.survival_agents_min = int(
            np.clip(
                train_cfg.get(
                    "avoid_skill_survival_agents_min",
                    self.survival_agents,
                ),
                base_env.min_agents,
                base_env.max_agents,
            )
        )
        if self.survival_agents_min > self.survival_agents:
            raise ValueError(
                "avoid_skill_survival_agents_min must be <= "
                "avoid_skill_survival_agents."
            )
        self.survival_box_size_min_m = float(
            train_cfg.get("avoid_skill_survival_box_size_min_m")
        )
        self.survival_box_size_max_m = float(
            train_cfg.get("avoid_skill_survival_box_size_max_m")
        )
        if self.survival_box_size_min_m <= 0.0:
            raise ValueError("avoid_skill_survival_box_size_min_m must be positive.")
        if self.survival_box_size_max_m < self.survival_box_size_min_m:
            raise ValueError(
                "avoid_skill_survival_box_size_max_m must be >= "
                "avoid_skill_survival_box_size_min_m."
            )
        self.survival_start_margin_m = float(
            train_cfg.get("avoid_skill_survival_start_margin_m")
        )
        if self.survival_start_margin_m < 0.0:
            raise ValueError("avoid_skill_survival_start_margin_m must be non-negative.")
        self.survival_scenario_mode = str(
            train_cfg.get("avoid_skill_survival_scenario_mode", "random")
        ).strip().lower()
        if self.survival_scenario_mode not in {"random", "default", "conflict_course"}:
            raise ValueError(
                "avoid_skill_survival_scenario_mode must be one of: "
                "random, default, conflict_course."
            )
        self.survival_conflict_options = self._survival_conflict_options(train_cfg)
        self.survival_max_steps = max(
            int(train_cfg.get("avoid_skill_survival_max_steps", 600)),
            1,
        )
        self.survival_step_reward = float(
            train_cfg.get("avoid_skill_survival_step_reward", 1.0)
        )
        self.survival_crash_penalty = abs(
            float(train_cfg.get("avoid_skill_survival_crash_penalty", 1000.0))
        )
        self.survival_team_failure_penalty_fraction = float(
            np.clip(
                train_cfg.get("avoid_skill_survival_team_failure_penalty_fraction", 0.25),
                0.0,
                1.0,
            )
        )
        self.survival_generation_attempts = max(
            int(
                train_cfg.get(
                    "avoid_skill_survival_generation_attempts",
                    getattr(base_env, "reset_generation_attempts", 128),
                )
            ),
            1,
        )
        self.survival_min_boundary_time_ratio = float(
            np.clip(
                train_cfg.get(
                    "avoid_skill_survival_min_boundary_time_ratio",
                    base_env.reset_min_boundary_time_ratio,
                ),
                0.0,
                2.0,
            )
        )
        self.survival_start_validation_steps = max(
            int(train_cfg.get("avoid_skill_survival_start_validation_steps", 0)),
            0,
        )
        self.survival_boundary_escape_steps = max(
            int(train_cfg.get("avoid_skill_survival_boundary_escape_steps", 0)),
            0,
        )
        self.survival_degenerate_motion_terminates = bool(
            train_cfg.get("avoid_skill_survival_degenerate_motion_terminates", True)
        )
        self.survival_degenerate_min_steps = max(
            int(train_cfg.get("avoid_skill_survival_degenerate_min_steps", 30)),
            0,
        )
        self.survival_degenerate_consecutive_steps = max(
            int(train_cfg.get("avoid_skill_survival_degenerate_consecutive_steps", 24)),
            1,
        )
        self.survival_degenerate_turn_fraction_threshold = float(
            np.clip(
                train_cfg.get("avoid_skill_survival_degenerate_turn_fraction_threshold", 0.78),
                0.0,
                1.0,
            )
        )
        self.survival_degenerate_turn_agreement_threshold = float(
            np.clip(
                train_cfg.get("avoid_skill_survival_degenerate_turn_agreement_threshold", 0.90),
                0.0,
                1.0,
            )
        )
        self.geofence_breach_grace_steps = 1
        self._survival_degenerate_motion_steps = 0
        self._current_scenario_bucket = "survival"

    @staticmethod
    def _survival_conflict_options(train_cfg: dict) -> dict:
        options = {}
        for train_key, option_key in (
            (
                "avoid_skill_survival_conflict_start_radius_fraction",
                "conflict_start_radius_fraction",
            ),
            (
                "avoid_skill_survival_conflict_crossing_radius_fraction",
                "conflict_crossing_radius_fraction",
            ),
            (
                "avoid_skill_survival_conflict_boundary_radius_fraction",
                "conflict_boundary_radius_fraction",
            ),
            (
                "avoid_skill_survival_conflict_edge_margin_m",
                "conflict_edge_margin_m",
            ),
            (
                "avoid_skill_survival_conflict_target_leg_time_s",
                "conflict_target_leg_time_s",
            ),
            (
                "avoid_skill_survival_conflict_phase_jitter_rad",
                "conflict_phase_jitter_rad",
            ),
            (
                "avoid_skill_survival_conflict_first_wp_lateral_jitter_m",
                "conflict_first_wp_lateral_jitter_m",
            ),
            (
                "avoid_skill_survival_conflict_generation_attempts",
                "conflict_generation_attempts",
            ),
            (
                "avoid_skill_survival_conflict_min_feasible_cpa_m",
                "conflict_min_feasible_cpa_m",
            ),
            (
                "avoid_skill_survival_conflict_target_cpa_m",
                "conflict_target_cpa_m",
            ),
            (
                "avoid_skill_survival_conflict_max_trigger_cpa_m",
                "conflict_max_trigger_cpa_m",
            ),
            (
                "avoid_skill_survival_conflict_min_pairs",
                "conflict_min_pairs",
            ),
            (
                "avoid_skill_survival_conflict_require_valid_candidate",
                "conflict_require_valid_candidate",
            ),
            (
                "avoid_skill_survival_conflict_require_boundary_viability",
                "conflict_require_boundary_viability",
            ),
            (
                "avoid_skill_survival_conflict_min_boundary_time_ratio",
                "conflict_min_boundary_time_ratio",
            ),
        ):
            if train_key in train_cfg:
                options[option_key] = train_cfg[train_key]
        return options

    def _survival_episode_agent_count(self, seed: int | None) -> int:
        if self.survival_agents_min >= self.survival_agents:
            return int(self.survival_agents)
        rng = np.random.default_rng(seed)
        return int(rng.integers(self.survival_agents_min, self.survival_agents + 1))

    def _survival_episode_box_shape(self, seed: int | None) -> tuple[float, float]:
        if self.survival_box_size_min_m >= self.survival_box_size_max_m:
            box_size = float(self.survival_box_size_min_m)
            return box_size, box_size
        rng = np.random.default_rng(seed)
        return (
            float(
                rng.uniform(
                    float(self.survival_box_size_min_m),
                    float(self.survival_box_size_max_m),
                )
            ),
            float(
                rng.uniform(
                    float(self.survival_box_size_min_m),
                    float(self.survival_box_size_max_m),
                )
            ),
        )

    def _reset_options(
        self,
        options: dict | None,
        *,
        num_agents: int,
        box_width_m: float,
        box_height_m: float,
    ) -> dict:
        reset_options = dict(options or {})
        reset_options.update(
            {
                "scenario_mode": self.survival_scenario_mode,
                "skill_scenario_bucket": "survival",
                "num_agents": int(num_agents),
                "box_width_m": float(box_width_m),
                "box_height_m": float(box_height_m),
                "start_margin_m": self.survival_start_margin_m,
                "mission_waypoint_count": 0,
                "max_steps": self.survival_max_steps,
                "timeout_max_steps": self.survival_max_steps,
                "timeout_scale_with_mission_size": False,
                "timeout_scale_with_route_distance": False,
                "terminate_on_all_waypoints_complete": False,
                "refill_random_waypoints_on_completion": False,
                "disable_waypoint_navigation": True,
            }
        )
        reset_options.update(self.survival_conflict_options)
        return reset_options

    def _clear_survival_waypoints(self) -> None:
        self.base_env.disable_waypoint_navigation = True
        for agent in list(self.base_env.agents):
            aircraft = self.base_env.aircraft_by_agent[agent]
            aircraft.replace_waypoint_queue([], replace_current=True)
            aircraft.waypoint_manager.hit_waypoints = []
            aircraft.flight_mode = FlightMode.NAVIGATING
            aircraft.loiter_center = None

    def reset(self, seed=None, options=None):
        for attempt_idx in range(self.survival_generation_attempts):
            attempt_seed = None if seed is None else int(seed) + attempt_idx
            episode_num_agents = self._survival_episode_agent_count(attempt_seed)
            episode_box_width_m, episode_box_height_m = (
                self._survival_episode_box_shape(attempt_seed)
            )
            reset_options = self._reset_options(
                options,
                num_agents=episode_num_agents,
                box_width_m=episode_box_width_m,
                box_height_m=episode_box_height_m,
            )
            self.base_env.reset(seed=attempt_seed, options=reset_options)
            self._clear_survival_waypoints()
            self.base_env._refresh_skill_guidance_cache()
            self.base_env._update_obs_cache(fill_history=True)
            agents = list(self.base_env.agents)
            if (
                self._survival_initial_state_is_possible(agents)
                and self._survival_start_has_no_immediate_crash(agents)
            ):
                self._prev_metrics = {}
                self._survival_degenerate_motion_steps = 0
                obs, infos = self._post_reset_response()
                for info in infos.values():
                    info["skill_scenario_bucket"] = self._current_scenario_bucket
                return obs, infos

        raise RuntimeError(
            "Avoid survival could not generate a viable waypoint-free start after "
            f"{self.survival_generation_attempts} attempts. Increase "
            "avoid_skill_survival_box_size_max_m or reduce aircraft spacing/turning constraints."
        )

    def _geofence_breach_agents(self, agents: List[str]) -> list[str]:
        breached = []
        for agent in agents:
            idx = self.base_env.agent_name_to_index[agent]
            inside, _, _ = self.base_env._boundary_status(agent)
            outside_steps = int(self.base_env.geofence_outside_steps[idx])
            if not inside and outside_steps >= self.geofence_breach_grace_steps:
                breached.append(agent)
        return breached

    def _survival_initial_state_is_possible(self, agents: List[str]) -> bool:
        if not agents:
            return False
        active_indices = [self.base_env.agent_name_to_index[agent] for agent in agents]
        min_edge_clearance_m = max(
            float(self.base_env.critical_dist) * 1.5,
            float(self.base_env.caution_dist) * 2.0,
        )
        for agent in agents:
            aircraft = self.base_env.aircraft_by_agent[agent]
            if aircraft.waypoint_manager.current_waypoint is not None:
                return False
            inside, _, _ = self.base_env._boundary_status(agent)
            if not inside:
                return False
            idx = self.base_env.agent_name_to_index[agent]
            x_pos = float(self.base_env._local_pos_cache[idx, 0])
            y_pos = float(self.base_env._local_pos_cache[idx, 1])
            edge_clearance_m = min(
                y_pos - float(self.base_env.local_min_y),
                float(self.base_env.local_max_y) - y_pos,
                x_pos - float(self.base_env.local_min_x),
                float(self.base_env.local_max_x) - x_pos,
            )
            if edge_clearance_m < min_edge_clearance_m:
                return False
            boundary_time_ratio = time_to_boundary_ahead(
                pos_local=self.base_env._local_pos_cache[idx],
                heading=float(aircraft.heading),
                cruise_speed=float(aircraft.dynamics.cruise_speed),
                local_min_x=float(self.base_env.local_min_x),
                local_max_x=float(self.base_env.local_max_x),
                local_min_y=float(self.base_env.local_min_y),
                local_max_y=float(self.base_env.local_max_y),
                lookahead_time_s=float(self.base_env.caution_lookahead_time_s),
            )
            if boundary_time_ratio < self.survival_min_boundary_time_ratio:
                return False
            if not self._survival_agent_has_boundary_escape(agent):
                return False

        for pos, idx_a in enumerate(active_indices):
            for idx_b in active_indices[pos + 1 :]:
                if (
                    float(self.base_env._dist_matrix[idx_a, idx_b])
                    < float(self.base_env.min_start_separation_m)
                ):
                    return False
        return True

    def _survival_agent_has_boundary_escape(self, agent: str) -> bool:
        if self.survival_boundary_escape_steps <= 0:
            return True

        idx = self.base_env.agent_name_to_index[agent]
        aircraft = self.base_env.aircraft_by_agent[agent]
        start_pos = np.asarray(self.base_env._local_pos_cache[idx], dtype=np.float32)
        start_heading = float(aircraft.heading)
        cruise_speed = float(aircraft.dynamics.cruise_speed)
        max_turn_rate = max(float(aircraft.dynamics.max_turn_rate), 1e-6)
        dt = float(self.base_env.dt)

        for turn_sign in (-1.0, 1.0):
            pos = start_pos.astype(np.float32, copy=True)
            heading = start_heading
            survived = True
            for _ in range(self.survival_boundary_escape_steps):
                heading += float(turn_sign) * max_turn_rate * dt
                pos[0] += cruise_speed * np.sin(heading) * dt
                pos[1] += cruise_speed * np.cos(heading) * dt
                if (
                    pos[0] < float(self.base_env.local_min_x)
                    or pos[0] > float(self.base_env.local_max_x)
                    or pos[1] < float(self.base_env.local_min_y)
                    or pos[1] > float(self.base_env.local_max_y)
                ):
                    survived = False
                    break
            if survived:
                return True
        return False

    def _survival_start_has_no_immediate_crash(self, agents: List[str]) -> bool:
        if self.survival_start_validation_steps <= 0 or not agents:
            return True

        active_indices = [self.base_env.agent_name_to_index[agent] for agent in agents]
        positions = np.asarray(
            self.base_env._local_pos_cache[active_indices],
            dtype=np.float32,
        ).copy()
        headings = np.asarray(
            [
                float(self.base_env.aircraft_by_agent[agent].heading)
                for agent in agents
            ],
            dtype=np.float32,
        )
        turn_rates = np.asarray(
            [
                float(self.base_env.aircraft_by_agent[agent].actual_turn_rate)
                for agent in agents
            ],
            dtype=np.float32,
        )
        cruise_speeds = np.asarray(
            [
                float(self.base_env.aircraft_by_agent[agent].dynamics.cruise_speed)
                for agent in agents
            ],
            dtype=np.float32,
        )
        response_times = np.asarray(
            [
                max(
                    float(
                        self.base_env.aircraft_by_agent[agent].dynamics.turn_response_time_s
                    ),
                    0.0,
                )
                for agent in agents
            ],
            dtype=np.float32,
        )
        dt = float(self.base_env.dt)
        for _ in range(self.survival_start_validation_steps):
            smooth_mask = response_times > 1e-6
            if np.any(smooth_mask):
                alpha = 1.0 - np.exp(-dt / np.maximum(response_times[smooth_mask], 1e-6))
                turn_rates[smooth_mask] = turn_rates[smooth_mask] * (1.0 - alpha)
            turn_rates[~smooth_mask] = 0.0
            headings = headings + turn_rates * dt
            positions[:, 0] += cruise_speeds * np.sin(headings) * dt
            positions[:, 1] += cruise_speeds * np.cos(headings) * dt

            inside = (
                (positions[:, 0] >= float(self.base_env.local_min_x))
                & (positions[:, 0] <= float(self.base_env.local_max_x))
                & (positions[:, 1] >= float(self.base_env.local_min_y))
                & (positions[:, 1] <= float(self.base_env.local_max_y))
            )
            if not bool(np.all(inside)):
                return False
            for idx_a in range(len(positions)):
                for idx_b in range(idx_a + 1, len(positions)):
                    if (
                        float(np.linalg.norm(positions[idx_a] - positions[idx_b]))
                        < float(self.base_env.critical_dist)
                    ):
                        return False
        return True

    def _survival_degenerate_motion_status(
        self,
        agents: List[str],
    ) -> tuple[bool, dict[str, float]]:
        if not self.survival_degenerate_motion_terminates or not agents:
            self._survival_degenerate_motion_steps = 0
            return False, {
                "survival_degenerate_motion": 0.0,
                "survival_degenerate_motion_steps": 0.0,
                "survival_degenerate_turn_fraction": 0.0,
                "survival_degenerate_turn_agreement": 0.0,
                "survival_heading_coherence": 0.0,
            }

        turn_fractions: list[float] = []
        turn_signs: list[float] = []
        heading_vectors: list[np.ndarray] = []
        for agent in agents:
            idx = self.base_env.agent_name_to_index[agent]
            aircraft = self.base_env.aircraft_by_agent[agent]
            max_turn_rate = max(float(aircraft.dynamics.max_turn_rate), 1e-6)
            turn_fraction = float(
                np.clip(abs(float(aircraft.actual_turn_rate)) / max_turn_rate, 0.0, 1.0)
            )
            turn_fractions.append(turn_fraction)
            if turn_fraction >= self.survival_degenerate_turn_fraction_threshold:
                turn_signs.append(float(np.sign(float(aircraft.actual_turn_rate))))
            else:
                turn_signs.append(0.0)
            heading = float(aircraft.heading)
            heading_vectors.append(
                np.asarray([np.sin(heading), np.cos(heading)], dtype=np.float32)
            )

        mean_turn_fraction = float(np.mean(turn_fractions)) if turn_fractions else 0.0
        turn_agreement = abs(float(np.mean(turn_signs))) if turn_signs else 0.0
        heading_coherence = (
            float(np.linalg.norm(np.mean(heading_vectors, axis=0)))
            if heading_vectors
            else 0.0
        )
        same_turn_circling = bool(
            int(self.base_env.current_step) >= int(self.survival_degenerate_min_steps)
            and mean_turn_fraction >= self.survival_degenerate_turn_fraction_threshold
            and turn_agreement >= self.survival_degenerate_turn_agreement_threshold
        )
        if same_turn_circling:
            self._survival_degenerate_motion_steps += 1
        else:
            self._survival_degenerate_motion_steps = 0

        degenerate_motion = bool(
            self._survival_degenerate_motion_steps
            >= self.survival_degenerate_consecutive_steps
        )
        return degenerate_motion, {
            "survival_degenerate_motion": float(degenerate_motion),
            "survival_degenerate_motion_steps": float(
                self._survival_degenerate_motion_steps
            ),
            "survival_degenerate_turn_fraction": float(mean_turn_fraction),
            "survival_degenerate_turn_agreement": float(turn_agreement),
            "survival_heading_coherence": float(heading_coherence),
        }

    def _survival_step(
        self,
        *,
        current_agents: List[str],
        obs: dict,
        terminations: dict,
        truncations: dict,
        infos: dict,
    ):
        geofence_breach_agents = self._geofence_breach_agents(current_agents)
        geofence_breach_agent_set = set(geofence_breach_agents)
        geofence_breach = bool(geofence_breach_agents)

        critical_collision_agent_set: set[str] = set()
        if self.base_env._termination_reason == "critical_violation":
            for agent_a, agent_b in self.base_env.crit_dist_breakers:
                if agent_a in current_agents:
                    critical_collision_agent_set.add(agent_a)
                if agent_b in current_agents:
                    critical_collision_agent_set.add(agent_b)
        critical_collision = bool(
            self.base_env._termination_reason == "critical_violation"
        )

        degenerate_motion, degenerate_motion_metrics = (
            self._survival_degenerate_motion_status(current_agents)
        )
        survival_complete = bool(
            self.base_env._termination_reason == "max_steps"
            or any(truncations.values())
            or int(self.base_env.current_step) >= int(self.survival_max_steps)
        )

        terminal_reason = None
        if critical_collision:
            terminal_reason = "critical_violation"
        elif geofence_breach:
            terminal_reason = "geofence_violation"
        elif degenerate_motion:
            terminal_reason = "degenerate_survival_motion"
        elif survival_complete:
            terminal_reason = "survival_completed"

        failed = terminal_reason in {
            "critical_violation",
            "geofence_violation",
            "degenerate_survival_motion",
        }

        shaped_rewards = {}
        agent_shaping = {}
        team_failure_penalty = (
            self.survival_crash_penalty * self.survival_team_failure_penalty_fraction
            if failed
            else 0.0
        )
        culprit_failure_penalty = self.survival_crash_penalty - team_failure_penalty
        for agent in current_agents:
            geofence_penalized = bool(
                terminal_reason == "geofence_violation"
                and agent in geofence_breach_agent_set
            )
            collision_penalized = bool(
                terminal_reason == "critical_violation"
                and (
                    not critical_collision_agent_set
                    or agent in critical_collision_agent_set
                )
            )
            degenerate_penalized = bool(terminal_reason == "degenerate_survival_motion")
            agent_failed = bool(
                geofence_penalized or collision_penalized or degenerate_penalized
            )
            agent_culprit_penalty = float(culprit_failure_penalty) if agent_failed else 0.0
            agent_crash_penalty = float(team_failure_penalty + agent_culprit_penalty)
            agent_reward_bonus = float(self.survival_step_reward) - agent_crash_penalty
            shaped_rewards[agent] = float(agent_reward_bonus)
            agent_shaping[agent] = {
                "mode": "avoid_survival",
                "survival_step_reward": float(self.survival_step_reward),
                "survival_crash_penalty": float(agent_crash_penalty),
                "survival_team_failure_penalty": float(team_failure_penalty),
                "survival_culprit_failure_penalty": float(agent_culprit_penalty),
                "survival_terminal": float(terminal_reason is not None),
                "survival_completed": float(terminal_reason == "survival_completed"),
                "survival_failed": float(agent_failed),
                "geofence_breach": float(agent in geofence_breach_agent_set),
                "geofence_breach_penalty": (
                    float(agent_culprit_penalty)
                    if geofence_penalized
                    else 0.0
                ),
                "critical_collision": float(collision_penalized),
                "critical_collision_penalty": (
                    float(agent_culprit_penalty)
                    if collision_penalized
                    else 0.0
                ),
                "degenerate_survival_motion": float(degenerate_motion),
                "degenerate_survival_motion_penalty": (
                    float(agent_culprit_penalty)
                    if terminal_reason == "degenerate_survival_motion"
                    else 0.0
                ),
                **degenerate_motion_metrics,
                "episode_progress": float(
                    np.clip(
                        self.base_env.current_step / max(self.survival_max_steps, 1),
                        0.0,
                        1.0,
                    )
                ),
                "survival_time_s": float(self.base_env.current_step * self.base_env.dt),
                "reward_bonus": float(agent_reward_bonus),
            }
        reward_bonus = (
            float(np.mean(list(shaped_rewards.values()))) if shaped_rewards else 0.0
        )

        shaping_info = {
            "mode": "avoid_survival",
            "survival_step_reward": float(self.survival_step_reward),
            "survival_crash_penalty": (
                float(self.survival_crash_penalty) if failed else 0.0
            ),
            "survival_team_failure_penalty": float(team_failure_penalty),
            "survival_culprit_failure_penalty": float(culprit_failure_penalty if failed else 0.0),
            "survival_terminal": float(terminal_reason is not None),
            "survival_completed": float(terminal_reason == "survival_completed"),
            "survival_failed": float(failed),
            "survival_time_s": float(self.base_env.current_step * self.base_env.dt),
            "survival_step_count": float(self.base_env.current_step),
            "survival_max_steps": float(self.survival_max_steps),
            "survival_box_width_m": float(self.base_env.box_width_m),
            "survival_box_height_m": float(self.base_env.box_height_m),
            "geofence_breach": float(geofence_breach),
            "geofence_breach_count": float(len(geofence_breach_agents)),
            "geofence_breach_penalty": (
                float(self.survival_crash_penalty)
                if terminal_reason == "geofence_violation"
                else 0.0
            ),
            "critical_collision": float(critical_collision),
            "critical_collision_penalty": (
                float(self.survival_crash_penalty)
                if terminal_reason == "critical_violation"
                else 0.0
            ),
            "degenerate_survival_motion": float(degenerate_motion),
            "degenerate_survival_motion_penalty": (
                float(self.survival_crash_penalty)
                if terminal_reason == "degenerate_survival_motion"
                else 0.0
            ),
            **degenerate_motion_metrics,
            "reward_bonus": float(reward_bonus),
        }

        if terminal_reason is not None:
            self.base_env._termination_reason = terminal_reason
            terminations = {agent: True for agent in current_agents}
            truncations = {agent: False for agent in current_agents}
            final_metrics = self.base_env.get_episode_metrics()
            episode_summary = final_metrics.setdefault("episode_summary", {})
            episode_summary["termination_reason"] = terminal_reason
            episode_summary["survival_completed"] = bool(
                terminal_reason == "survival_completed"
            )
            episode_summary["survival_failed"] = bool(failed)
            episode_summary["survival_geofence_failed"] = bool(geofence_breach)
            episode_summary["survival_collision_failed"] = bool(critical_collision)
            episode_summary["survival_degenerate_motion_failed"] = bool(
                terminal_reason == "degenerate_survival_motion"
            )
            episode_summary["survival_steps"] = int(self.base_env.current_step)
            episode_summary["survival_time_s"] = float(
                self.base_env.current_step * self.base_env.dt
            )
            episode_summary["survival_max_steps"] = int(self.survival_max_steps)
            episode_summary["survival_box_width_m"] = float(self.base_env.box_width_m)
            episode_summary["survival_box_height_m"] = float(self.base_env.box_height_m)
            episode_summary["survival_crash_penalty"] = float(
                self.survival_crash_penalty
            )
            episode_summary["survival_step_reward"] = float(self.survival_step_reward)
            for agent in current_agents:
                infos.setdefault(agent, {})
                infos[agent]["terminated"] = True
                infos[agent]["truncated"] = False
                infos[agent]["termination_reason"] = terminal_reason
                infos[agent]["episode_metrics"] = final_metrics
            obs = {}
            self.base_env.agents = []

        for agent in current_agents:
            infos.setdefault(agent, {})
            infos[agent]["skill_shaping"] = shaping_info
            infos[agent]["agent_skill_shaping"] = agent_shaping[agent]
            infos[agent]["skill_scenario_bucket"] = self._current_scenario_bucket

        self._prev_metrics = {}
        return obs, shaped_rewards, terminations, truncations, infos

    def step(self, actions):
        current_agents = list(self.base_env.agents)
        obs, _, terminations, truncations, infos = self.base_env.step(actions)
        return self._survival_step(
            current_agents=current_agents,
            obs=obs,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
        )
