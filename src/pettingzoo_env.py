from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

try:
    from pettingzoo.utils.env import ParallelEnv
except ModuleNotFoundError:
    class ParallelEnv:  # pragma: no cover - lightweight fallback for local runs
        metadata = {}

from flight_engine.helpers import FlightMode, Position, clip_scalar, wrap_angle
from flight_engine.navigation_utils import (
    build_box_bounds,
    build_rect_bounds,
    heading_to_radians,
    order_route_points,
    planned_route_distance_m,
)
from flight_engine.reference_guidance import (
    ReferenceRoute,
    build_reference_route_local,
    compute_reference_action,
    dangerous_neighbor_turn_preview,
    time_to_boundary_ahead,
    turn_circle_feasibility_features,
)
from flight_engine.simulator import FixedWingAircraft
from flight_engine.trans_coorders import CoordinateTransformer


class MultiUAVParallelEnv(ParallelEnv):
    metadata = {
        "name": "dont_multi_uav_parallel_v0",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        dt: float = 0.3,
        max_steps: int = 1_500,
        timeout_scale_with_mission_size: bool = False,
        timeout_steps_per_additional_waypoint: int = 0,
        timeout_scale_with_route_distance: bool = False,
        timeout_steps_per_additional_route_km: float = 0.0,
        timeout_max_steps: Optional[int] = None,
        timeout_reference_waypoints: Optional[int] = None,
        timeout_reference_route_distance_m: Optional[float] = None,
        boundary_margin: float = 0.05,
        mission_waypoint_count: int = 3,
        mission_waypoint_count_min: Optional[int] = None,
        mission_waypoint_count_max: Optional[int] = None,
        waypoint_arrival_radius: float = 30.0,
        obs_stack_size: int = 4,
        caution_dist: float = 50.0,
        critical_dist: float = 10.0,
        min_agents: int = 2,
        max_agents: int = 5,
        map_size_range_m: Tuple[float, float] = (300.0, 1_500.0),
        origin: Optional[Tuple[float, float]] = None,
        top_left: Optional[Tuple[float, float]] = None,
        bottom_right: Optional[Tuple[float, float]] = None,
        flight_config: Optional[dict] = None,
        reward_config: Optional[dict] = None,
        guidance_config: Optional[dict] = None,
        manual_missions: Optional[Dict[str, dict]] = None,
        terminate_on_all_waypoints_complete: bool = True,
        refill_random_waypoints_on_completion: bool = False,
        allow_live_waypoint_updates: bool = False,
    ):
        super().__init__()
        flight_config = flight_config or {}
        reward_config = reward_config or {}
        guidance_config = guidance_config or {}

        self.dt = float(dt)
        self.base_max_steps = max(int(max_steps), 1)
        self.max_steps = self.base_max_steps
        self.timeout_scale_with_mission_size = bool(timeout_scale_with_mission_size)
        self.timeout_steps_per_additional_waypoint = max(
            int(timeout_steps_per_additional_waypoint),
            0,
        )
        self.timeout_scale_with_route_distance = bool(timeout_scale_with_route_distance)
        self.timeout_steps_per_additional_route_km = max(
            float(timeout_steps_per_additional_route_km),
            0.0,
        )
        if timeout_max_steps is None:
            timeout_max_steps = self.base_max_steps
        self.timeout_max_steps = max(int(timeout_max_steps), self.base_max_steps)
        self.timeout_reference_waypoints = (
            max(int(timeout_reference_waypoints), 1)
            if timeout_reference_waypoints is not None
            else None
        )
        self.timeout_reference_route_distance_m = (
            max(float(timeout_reference_route_distance_m), 0.0)
            if timeout_reference_route_distance_m is not None
            else None
        )
        self._episode_timeout_assigned_waypoints = 0
        self._episode_timeout_reference_waypoints = 0
        self._episode_timeout_max_route_distance_m = 0.0
        self._episode_timeout_reference_route_distance_m = 0.0
        self.boundary_margin = float(boundary_margin)
        self.mission_waypoint_count = int(mission_waypoint_count)
        if mission_waypoint_count_min is None:
            mission_waypoint_count_min = self.mission_waypoint_count
        if mission_waypoint_count_max is None:
            mission_waypoint_count_max = self.mission_waypoint_count
        self.mission_waypoint_count_min = max(int(mission_waypoint_count_min), 1)
        self.mission_waypoint_count_max = max(int(mission_waypoint_count_max), 1)
        if self.mission_waypoint_count_max < self.mission_waypoint_count_min:
            raise ValueError(
                "mission_waypoint_count_max must be greater than or equal to "
                "mission_waypoint_count_min."
            )
        self._current_mission_waypoint_count = self.mission_waypoint_count_max
        self.waypoint_arrival_radius = max(float(waypoint_arrival_radius), 1.0)
        self.obs_stack_size = max(int(obs_stack_size), 1)
        self.caution_dist = float(caution_dist)
        self.critical_dist = float(critical_dist)
        self.min_agents = int(min_agents)
        self.max_agents = int(max_agents)
        self.map_size_range_m = tuple(float(v) for v in map_size_range_m)
        self.default_origin = tuple(origin) if origin else None
        self.fixed_top_left = tuple(top_left) if top_left else None
        self.fixed_bottom_right = tuple(bottom_right) if bottom_right else None
        self.manual_missions = manual_missions or {}
        self.randomized = not bool(self.manual_missions)
        self.terminate_on_all_waypoints_complete = bool(
            terminate_on_all_waypoints_complete
        )
        self.refill_random_waypoints_on_completion = bool(
            refill_random_waypoints_on_completion
        )
        self.allow_live_waypoint_updates = bool(allow_live_waypoint_updates)

        self.possible_agents = [
            f"UAV-{idx + 1}"
            for idx in range(self.max_agents)
        ]
        self.manual_agents = [
            agent
            for agent in self.possible_agents
            if agent in self.manual_missions
        ]
        unknown_manual_agents = sorted(
            set(self.manual_missions.keys()) - set(self.possible_agents)
        )
        if unknown_manual_agents:
            raise ValueError(
                "Manual mission agent ids exceed the configured max_agents. "
                f"Unknown agents: {unknown_manual_agents}"
            )
        if not self.randomized and not self.manual_agents:
            raise ValueError("manual_missions was provided but no matching UAV ids were found.")

        self.agent_name_to_index = {
            agent: idx
            for idx, agent in enumerate(self.possible_agents)
        }
        self.agents: List[str] = []

        cruise_center = float(flight_config.get("cruise_speed_mps", 25.0))
        cruise_var = float(flight_config.get("cruise_speed_variation_mps", 0.0))
        turn_center = float(flight_config.get("turning_radius_m", 30.0))
        turn_var = float(flight_config.get("turning_radius_variation_m", 0.0))

        self.cruise_speed_min = float(
            flight_config.get(
                "cruise_speed_min_mps",
                max(3.0, cruise_center - cruise_var),
            )
        )
        self.cruise_speed_max = float(
            flight_config.get(
                "cruise_speed_max_mps",
                cruise_center + cruise_var,
            )
        )
        self.turning_radius_min = float(
            flight_config.get(
                "turning_radius_min_m",
                max(5.0, turn_center - turn_var),
            )
        )
        self.turning_radius_max = float(
            flight_config.get(
                "turning_radius_max_m",
                turn_center + turn_var,
            )
        )
        self.min_start_separation_m = float(
            flight_config.get("min_start_separation_m", self.caution_dist * 1.5)
        )
        self.turn_response_time_s = float(
            flight_config.get("turn_response_time_s", 0.0)
        )
        self.max_turn_rate_min = float(
            self.cruise_speed_min / max(self.turning_radius_max, 1e-6)
        )
        self.max_turn_rate_max = float(
            self.cruise_speed_max / max(self.turning_radius_min, 1e-6)
        )
        self.reward_waypoint_hit = float(
            reward_config.get("waypoint_hit_reward", 35.0)
        )
        self.reward_completion_bonus = float(
            reward_config.get("completion_bonus", 100.0)
        )
        self.reward_waypoint_proximity_bonus = float(
            reward_config.get("waypoint_proximity_bonus", 1.0)
        )
        self.reward_progress = float(
            reward_config.get("progress_reward", 0.0)
        )
        self.penalty_geofence = float(
            reward_config.get("geofence_penalty", 20.0)
        )
        self.penalty_boundary_soft = float(
            reward_config.get("boundary_soft_penalty", 0.0)
        )
        self.geofence_growth_exponent = float(
            reward_config.get("geofence_growth_exponent", 2.0)
        )
        self.boundary_soft_growth_exponent = float(
            reward_config.get("boundary_soft_growth_exponent", 1.5)
        )
        self.geofence_depth_cap = float(
            reward_config.get("geofence_depth_cap", 1.0)
        )
        self.penalty_crash = float(
            reward_config.get("crash_penalty", 150.0)
        )
        self.penalty_harsh_turn = float(
            reward_config.get("harsh_turn_penalty", 0.4)
        )
        self.penalty_circling = float(
            reward_config.get("circling_penalty", 0.35)
        )
        self.circling_activation_steps = max(
            int(reward_config.get("circling_activation_steps", 45)),
            1,
        )
        self.circling_activation_turns = max(
            float(reward_config.get("circling_activation_turns", 1.25)),
            0.25,
        )
        self.circling_min_distance_ratio = max(
            float(reward_config.get("circling_min_distance_ratio", 1.05)),
            1.0,
        )
        self.circling_progress_reset_m = max(
            float(reward_config.get("circling_progress_reset_m", 1.0)),
            1e-3,
        )
        self.circling_relief_steps = max(
            int(reward_config.get("circling_relief_steps", 10)),
            1,
        )
        self.circling_relief_turn_fraction = float(
            np.clip(
                reward_config.get("circling_relief_turn_fraction", 0.3),
                0.0,
                1.0,
            )
        )
        self.penalty_separation_margin = float(
            reward_config.get("separation_margin_penalty", 4.0)
        )
        self.penalty_stagnation = float(
            reward_config.get("stagnation_penalty", 0.0)
        )
        self.stagnation_step_threshold = max(
            int(reward_config.get("stagnation_step_threshold", 18)),
            1,
        )
        self.stagnation_progress_epsilon_m = max(
            float(reward_config.get("stagnation_progress_epsilon_m", 1.0)),
            1e-6,
        )
        self.penalty_incomplete_mission = float(
            reward_config.get("incomplete_mission_penalty", 0.0)
        )
        self.separation_margin_ratio = float(
            reward_config.get("separation_margin_ratio", 1.6)
        )
        self.penalty_existence = float(
            reward_config.get("existence_penalty", 0.02)
        )
        self.harsh_turn_threshold = float(
            reward_config.get("harsh_turn_threshold", 0.55)
        )
        self.boundary_buffer_ratio = float(
            guidance_config.get("boundary_buffer_ratio", 0.12)
        )
        self.caution_lookahead_time_s = float(
            guidance_config.get("caution_lookahead_time_s", 8.0)
        )
        self.predicted_caution_weight = float(
            guidance_config.get("predicted_caution_weight", 0.75)
        )
        self.deconfliction_progress_scale = float(
            np.clip(
                guidance_config.get("deconfliction_progress_scale", 0.35),
                0.0,
                1.0,
            )
        )
        self.head_on_conflict_weight = float(
            guidance_config.get("head_on_conflict_weight", 0.5)
        )
        self.conflict_commit_activation = float(
            guidance_config.get("conflict_commit_activation", 0.2)
        )
        self.conflict_commit_release_scale = float(
            guidance_config.get("conflict_commit_release_scale", 1.35)
        )
        self.conflict_commit_release_pressure = float(
            guidance_config.get("conflict_commit_release_pressure", 0.08)
        )

        self.guidance_lookahead_time_s = float(
            guidance_config.get("lookahead_time_s", 1.8)
        )
        self.guidance_min_lookahead_m = float(
            guidance_config.get("min_lookahead_m", 20.0)
        )
        self.guidance_max_lookahead_m = float(
            guidance_config.get("max_lookahead_m", 90.0)
        )
        self.guidance_deconfliction_hold_steps = int(
            guidance_config.get("deconfliction_hold_steps", 6)
        )
        self.guidance_route_commit_scale = float(
            guidance_config.get("route_commit_scale", 1.5)
        )
        self.guidance_turn_gain = float(
            guidance_config.get("turn_gain", 1.0)
        )
        self.guidance_turn_lookahead_scale = float(
            guidance_config.get("turn_lookahead_scale", 1.25)
        )
        self.guidance_turn_radius_floor_scale = float(
            guidance_config.get("turn_radius_floor_scale", 1.35)
        )
        self.waypoint_reapproach_min_steps = max(
            int(guidance_config.get("waypoint_reapproach_min_steps", 14)),
            1,
        )
        self.waypoint_reapproach_release_distance_scale = max(
            float(
                guidance_config.get(
                    "waypoint_reapproach_release_distance_scale",
                    2.4,
                )
            ),
            1.0,
        )
        self.waypoint_reapproach_release_arrival_scale = max(
            float(
                guidance_config.get(
                    "waypoint_reapproach_release_arrival_scale",
                    3.5,
                )
            ),
            1.0,
        )

        self.self_feature_count = 34
        self.neighbor_feature_count = 30
        self.action_dim = 1
        self.max_neighbors = max(1, self.max_agents - 1)
        self.base_obs_dim = (
            self.self_feature_count
            + (self.max_neighbors * self.neighbor_feature_count)
        )
        self.obs_dim = self.base_obs_dim * self.obs_stack_size
        self.state_dim = (
            self.max_agents * self.obs_dim
            + self.max_agents
            + 2
        )
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
                low=-1.0,
                high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self._rng = np.random.default_rng()
        self.current_step = 0
        self.aircraft_by_agent: Dict[str, FixedWingAircraft] = {}
        self._reward_component_names = [
            "waypoint_reward",
            "completion_bonus",
            "waypoint_proximity_bonus",
            "progress_reward",
            "geofence_penalty",
            "boundary_soft_penalty",
            "crash_penalty",
            "harsh_turn_penalty",
            "circling_penalty",
            "stagnation_penalty",
            "separation_margin_penalty",
            "existence_penalty",
            "incomplete_mission_penalty",
        ]
        self._reward_totals = {name: 0.0 for name in self._reward_component_names}

        self._base_obs_cache = np.zeros(
            (self.max_agents, self.base_obs_dim),
            dtype=np.float32,
        )
        self._obs_history = np.zeros(
            (self.max_agents, self.obs_stack_size, self.base_obs_dim),
            dtype=np.float32,
        )
        self._obs_cache = np.zeros((self.max_agents, self.obs_dim), dtype=np.float32)
        self._local_pos_cache = np.zeros((self.max_agents, 2), dtype=np.float32)
        self._prev_local_cache = np.zeros((self.max_agents, 2), dtype=np.float32)
        self._wp_local_cache = np.zeros((self.max_agents, 2), dtype=np.float32)
        self._next_wp_local_cache = np.zeros((self.max_agents, 2), dtype=np.float32)
        self._has_next_wp = np.zeros(self.max_agents, dtype=bool)
        self._dx_matrix = np.zeros((self.max_agents, self.max_agents), dtype=np.float32)
        self._dy_matrix = np.zeros((self.max_agents, self.max_agents), dtype=np.float32)
        self._dist_matrix = np.zeros((self.max_agents, self.max_agents), dtype=np.float32)
        self._last_rel_bearing_matrix = np.full(
            (self.max_agents, self.max_agents),
            np.nan,
            dtype=np.float32,
        )
        self._last_d_cpa_matrix = np.full(
            (self.max_agents, self.max_agents),
            np.nan,
            dtype=np.float32,
        )
        self._last_pair_conflict_pressure_matrix = np.full(
            (self.max_agents, self.max_agents),
            np.nan,
            dtype=np.float32,
        )
        self._segment_start_local = np.zeros((self.max_agents, 2), dtype=np.float32)
        self._segment_start_heading = np.zeros(self.max_agents, dtype=np.float32)
        self._segment_path_length = np.zeros(self.max_agents, dtype=np.float32)
        self._last_headings = np.zeros(self.max_agents, dtype=np.float32)
        self._last_heading_rates = np.zeros(self.max_agents, dtype=np.float32)
        self._last_actions = np.zeros(self.max_agents, dtype=np.float32)
        self._last_action_delta = np.zeros(self.max_agents, dtype=np.float32)
        self._last_reference_action_vector = np.zeros(self.max_agents, dtype=np.float32)
        self._route_progress_anchor = np.zeros(self.max_agents, dtype=np.float32)
        self._closest_wp_distance = np.full(self.max_agents, np.inf, dtype=np.float32)
        self._closest_wp_signature: List[Optional[Tuple[float, float]]] = [
            None for _ in range(self.max_agents)
        ]
        self._circling_wp_signature: List[Optional[Tuple[float, float]]] = [
            None for _ in range(self.max_agents)
        ]
        self._circling_active = np.zeros(self.max_agents, dtype=bool)
        self._circling_stagnation_steps = np.zeros(self.max_agents, dtype=np.int32)
        self._circling_angular_travel = np.zeros(self.max_agents, dtype=np.float32)
        self._circling_relief_progress = np.zeros(self.max_agents, dtype=np.int32)
        self._circling_steps_total = np.zeros(self.max_agents, dtype=np.int32)
        self._circling_breakouts_total = np.zeros(self.max_agents, dtype=np.int32)
        self._waypoint_reapproach_active = np.zeros(self.max_agents, dtype=bool)
        self._waypoint_reapproach_hold_remaining = np.zeros(
            self.max_agents,
            dtype=np.int32,
        )
        self._waypoint_reapproach_release_distance = np.zeros(
            self.max_agents,
            dtype=np.float32,
        )
        self._waypoint_reapproach_steps_total = np.zeros(
            self.max_agents,
            dtype=np.int32,
        )
        self._waypoint_reapproach_events_total = np.zeros(
            self.max_agents,
            dtype=np.int32,
        )
        self._last_commanded_action_vector = np.zeros(self.max_agents, dtype=np.float32)
        self._last_policy_train_mask = np.zeros(self.max_agents, dtype=np.float32)
        self._deconfliction_active = np.zeros(self.max_agents, dtype=bool)
        self._deconfliction_hold_remaining = np.zeros(self.max_agents, dtype=np.int32)
        self._deconfliction_steps_total = np.zeros(self.max_agents, dtype=np.int32)
        self._side_commitment = np.zeros((self.max_agents, self.max_agents), dtype=np.float32)
        self._reference_route_cache: List[Optional[ReferenceRoute]] = [
            None for _ in range(self.max_agents)
        ]
        self._completion_steps: List[Optional[int]] = [None for _ in range(self.max_agents)]
        self._min_pairwise_distance = np.inf
        self._min_pairwise_pair = None
        self._min_pairwise_step = None
        self._episode_reward_total = 0.0
        self._termination_reason = "not_started"
        self._crashed = False

        self.caution_dist_breakers: List[Tuple[str, str]] = []
        self.crit_dist_breakers: List[Tuple[str, str]] = []
        self.geofence_exit_counts = [0 for _ in range(self.max_agents)]
        self.geofence_outside_steps = [0 for _ in range(self.max_agents)]
        self._was_outside = [False for _ in range(self.max_agents)]

        self.min_lat = -1.0
        self.max_lat = 1.0
        self.min_lon = -1.0
        self.max_lon = 1.0
        self.transformer = CoordinateTransformer(0.0, 0.0)
        self.local_min_x = 0.0
        self.local_max_x = 1.0
        self.local_min_y = 0.0
        self.local_max_y = 1.0
        self.box_width_m = 1.0
        self.box_height_m = 1.0
        self.map_diag_m = 1.0
        self.map_size_scale = max(self.map_size_range_m[1], 1.0)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return seed

    def state(self) -> np.ndarray:
        active_mask = np.zeros(self.max_agents, dtype=np.float32)
        for agent in self.agents:
            active_mask[self.agent_name_to_index[agent]] = 1.0
        return np.concatenate(
            [
                self._obs_cache.reshape(-1),
                active_mask,
                np.asarray(
                    [
                        self.box_width_m / max(self.map_size_scale, 1.0),
                        self.box_height_m / max(self.map_size_scale, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def get_obs_matrix(self) -> np.ndarray:
        return self._obs_cache.copy()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        options = options or {}

        self.current_step = 0
        self._crashed = False
        self._termination_reason = "in_progress"
        self._episode_reward_total = 0.0
        self._reward_totals = {name: 0.0 for name in self._reward_component_names}
        self._obs_cache.fill(0.0)
        self._base_obs_cache.fill(0.0)
        self._obs_history.fill(0.0)
        self._local_pos_cache.fill(0.0)
        self._prev_local_cache.fill(0.0)
        self._wp_local_cache.fill(0.0)
        self._next_wp_local_cache.fill(0.0)
        self._has_next_wp[:] = False
        self._dx_matrix.fill(0.0)
        self._dy_matrix.fill(0.0)
        self._dist_matrix.fill(0.0)
        self._last_rel_bearing_matrix.fill(np.nan)
        self._last_d_cpa_matrix.fill(np.nan)
        self._last_pair_conflict_pressure_matrix.fill(np.nan)
        self._segment_start_local.fill(0.0)
        self._segment_start_heading[:] = 0.0
        self._segment_path_length[:] = 0.0
        self._last_headings[:] = 0.0
        self._last_heading_rates[:] = 0.0
        self._last_actions[:] = 0.0
        self._last_action_delta[:] = 0.0
        self._last_reference_action_vector[:] = 0.0
        self._route_progress_anchor[:] = 0.0
        self._closest_wp_distance[:] = np.inf
        self._closest_wp_signature = [None for _ in range(self.max_agents)]
        self._circling_wp_signature = [None for _ in range(self.max_agents)]
        self._circling_active[:] = False
        self._circling_stagnation_steps[:] = 0
        self._circling_angular_travel[:] = 0.0
        self._circling_relief_progress[:] = 0
        self._circling_steps_total[:] = 0
        self._circling_breakouts_total[:] = 0
        self._waypoint_reapproach_active[:] = False
        self._waypoint_reapproach_hold_remaining[:] = 0
        self._waypoint_reapproach_release_distance[:] = 0.0
        self._waypoint_reapproach_steps_total[:] = 0
        self._waypoint_reapproach_events_total[:] = 0
        self._last_commanded_action_vector[:] = 0.0
        self._last_policy_train_mask[:] = 0.0
        self._deconfliction_active[:] = False
        self._deconfliction_hold_remaining[:] = 0
        self._deconfliction_steps_total[:] = 0
        self._side_commitment.fill(0.0)
        self._reference_route_cache = [None for _ in range(self.max_agents)]
        self._completion_steps = [None for _ in range(self.max_agents)]
        self._min_pairwise_distance = np.inf
        self._min_pairwise_pair = None
        self._min_pairwise_step = None

        self.caution_dist_breakers = []
        self.crit_dist_breakers = []
        self.geofence_exit_counts = [0 for _ in range(self.max_agents)]
        self.geofence_outside_steps = [0 for _ in range(self.max_agents)]
        self._was_outside = [False for _ in range(self.max_agents)]

        if self.randomized:
            self._reset_random_episode(options)
        else:
            self._reset_manual_episode(options)
        self.max_steps = self._resolve_episode_max_steps(options)

        self._sync_local_caches()
        self._refresh_route_guidance_cache()
        self._update_obs_cache(fill_history=True)
        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            self._last_headings[idx] = float(self.aircraft_by_agent[agent].heading)
            self._segment_start_local[idx] = self._local_pos_cache[idx]
            self._segment_start_heading[idx] = float(
                self.aircraft_by_agent[agent].heading
            )
            self._sync_waypoint_progress_tracking(agent)

        obs = {
            agent: self._obs_cache[self.agent_name_to_index[agent]].copy()
            for agent in self.agents
        }
        infos = {
            agent: self._shared_info()
            for agent in self.agents
        }
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = list(self.agents)
        self.current_step += 1

        policy_action_vector = np.zeros(self.max_agents, dtype=np.float32)
        for agent in current_agents:
            idx = self.agent_name_to_index[agent]
            raw_action = actions.get(agent, np.zeros((1,), dtype=np.float32))
            policy_action_vector[idx] = float(
                np.clip(np.asarray(raw_action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0)
            )

        self._prev_local_cache[:] = self._local_pos_cache
        commanded_action_vector = policy_action_vector.copy()
        self._last_policy_train_mask[:] = 0.0
        was_deconfliction_active = self._deconfliction_active.copy()
        for agent in current_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            action_controls_aircraft = bool(
                aircraft.flight_mode != FlightMode.LOITERING
                and aircraft.waypoint_manager.current_waypoint is not None
            )
            assist_controls_aircraft = bool(
                action_controls_aircraft
                and self._should_apply_waypoint_reapproach_assist(
                    idx=idx,
                    aircraft=aircraft,
                )
            )
            if assist_controls_aircraft:
                commanded_action_vector[idx] = float(self._last_reference_action_vector[idx])
                self._waypoint_reapproach_steps_total[idx] += 1
            self._last_policy_train_mask[idx] = (
                1.0
                if action_controls_aircraft and not assist_controls_aircraft
                else 0.0
            )
        self._last_commanded_action_vector[:] = commanded_action_vector

        executed_action_vector = commanded_action_vector.copy()
        for agent in current_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            commanded_turn_rate = float(
                commanded_action_vector[idx] * aircraft.dynamics.max_turn_rate
            )
            aircraft.desired_turn_rate = commanded_turn_rate
            if aircraft.flight_mode == FlightMode.LOITERING:
                aircraft.desired_turn_rate = float(aircraft.dynamics.max_turn_rate)
                aircraft.actual_turn_rate = float(aircraft.dynamics.max_turn_rate)
                executed_action_vector[idx] = 1.0
                aircraft._update_loiter(
                    float(self._prev_local_cache[idx, 0]),
                    float(self._prev_local_cache[idx, 1]),
                    self.dt,
                    self.transformer,
                )
            else:
                aircraft.update_simple(
                    commanded_turn_rate,
                    self.dt,
                    self.transformer,
                )
                aircraft.path_history.append(
                    Position(
                        aircraft.position.latitude,
                        aircraft.position.longitude,
                    )
                )

        self._sync_local_caches()
        self._update_pairwise_distance_cache()
        pair_metrics = self._evaluate_pairwise_safety(current_agents)
        self._update_deconfliction_state(
            current_agents,
            pair_metrics,
            was_active=was_deconfliction_active,
        )

        waypoint_proximities = []
        progress_terms = []
        circling_scores = []
        stagnation_scores = []
        boundary_soft_risks = []
        outside_depths = []
        waypoint_hits = 0
        caution_pressure = np.asarray(
            pair_metrics["agent_caution_pressure"],
            dtype=np.float32,
        )

        for agent in current_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            step_distance = float(
                np.linalg.norm(self._local_pos_cache[idx] - self._prev_local_cache[idx])
            )
            self._segment_path_length[idx] += step_distance
            best_improvement = 0.0
            if aircraft.waypoint_manager.current_waypoint is not None:
                dist_before = float(
                    np.linalg.norm(self._wp_local_cache[idx] - self._prev_local_cache[idx])
                )
                dist_after = float(
                    np.linalg.norm(self._wp_local_cache[idx] - self._local_pos_cache[idx])
                )
            else:
                dist_before = 0.0
                dist_after = 0.0
            if aircraft.waypoint_manager.current_waypoint is not None:
                self._sync_waypoint_progress_tracking(
                    agent,
                    distance_to_wp=dist_before,
                )
                self._sync_waypoint_circling_tracking(agent)
                started_circling_this_step = False
                closest_before = float(
                    min(self._closest_wp_distance[idx], dist_before)
                )
                closest_after = float(min(closest_before, dist_after))
                angle_before = float(
                    np.arctan2(
                        float(self._prev_local_cache[idx, 1] - self._wp_local_cache[idx, 1]),
                        float(self._prev_local_cache[idx, 0] - self._wp_local_cache[idx, 0]),
                    )
                )
                angle_after = float(
                    np.arctan2(
                        float(self._local_pos_cache[idx, 1] - self._wp_local_cache[idx, 1]),
                        float(self._local_pos_cache[idx, 0] - self._wp_local_cache[idx, 0]),
                    )
                )
                angular_step = float(abs(wrap_angle(angle_after - angle_before)))
                proximity_scale = max(
                    float(aircraft.waypoint_manager.arrival_threshold),
                    1e-6,
                )
                proximity_before = float(
                    (
                        proximity_scale
                        / (closest_before + proximity_scale)
                    )
                    ** 2
                )
                proximity_after = float(
                    (
                        proximity_scale
                        / (closest_after + proximity_scale)
                    )
                    ** 2
                )
                self._closest_wp_distance[idx] = closest_after
                best_improvement = max(closest_before - dist_after, 0.0)
                progress_denominator = max(
                    closest_before,
                    float(aircraft.waypoint_manager.arrival_threshold),
                    1e-6,
                )
                progress_term = float(
                    np.clip(best_improvement / progress_denominator, 0.0, 1.0)
                )
                if self._deconfliction_active[idx] or float(caution_pressure[idx]) > 0.0:
                    progress_term *= self.deconfliction_progress_scale
                progress_terms.append(progress_term)
                max_turn_rate = max(float(aircraft.dynamics.max_turn_rate), 1e-6)
                turn_fraction = abs(float(aircraft.actual_turn_rate)) / max_turn_rate
                reapproach_assisted = bool(
                    self._waypoint_reapproach_active[idx]
                    and not self._deconfliction_active[idx]
                )
                if reapproach_assisted:
                    self._circling_active[idx] = False
                    self._circling_stagnation_steps[idx] = 0
                    self._circling_angular_travel[idx] = 0.0
                    self._circling_relief_progress[idx] = 0
                    waypoint_proximities.append(0.0)
                    circling_scores.append(0.0)
                    stagnation_scores.append(0.0)
                elif self._deconfliction_active[idx]:
                    self._circling_active[idx] = False
                    self._circling_stagnation_steps[idx] = 0
                    self._circling_angular_travel[idx] = 0.0
                    self._circling_relief_progress[idx] = 0
                    waypoint_proximities.append(0.0)
                    circling_scores.append(0.0)
                    stagnation_scores.append(0.0)
                else:
                    if self._circling_active[idx] and best_improvement >= self.circling_progress_reset_m:
                        self._circling_active[idx] = False
                        self._circling_breakouts_total[idx] += 1
                        self._circling_stagnation_steps[idx] = 0
                        self._circling_angular_travel[idx] = 0.0
                        self._circling_relief_progress[idx] = 0
                    elif best_improvement >= self.circling_progress_reset_m:
                        self._circling_stagnation_steps[idx] = 0
                        self._circling_angular_travel[idx] = 0.0
                        self._circling_relief_progress[idx] = 0
                    else:
                        self._circling_stagnation_steps[idx] += 1
                        self._circling_angular_travel[idx] += angular_step

                    if self._circling_active[idx]:
                        if turn_fraction <= self.circling_relief_turn_fraction:
                            self._circling_relief_progress[idx] += 1
                            if self._circling_relief_progress[idx] >= self.circling_relief_steps:
                                self._circling_active[idx] = False
                                self._circling_breakouts_total[idx] += 1
                                self._circling_stagnation_steps[idx] = 0
                                self._circling_angular_travel[idx] = 0.0
                                self._circling_relief_progress[idx] = 0
                        else:
                            self._circling_relief_progress[idx] = 0

                    arrival_threshold = max(
                        float(aircraft.waypoint_manager.arrival_threshold),
                        1e-6,
                    )
                    outside_capture_ratio = float(
                        max((dist_after / arrival_threshold) - self.circling_min_distance_ratio, 0.0)
                    )
                    turn_excess = float(
                        max(
                            (self._circling_angular_travel[idx] / (2.0 * np.pi))
                            - self.circling_activation_turns,
                            0.0,
                        )
                    )
                    stagnation_excess = float(
                        max(
                            self._circling_stagnation_steps[idx] - self.circling_activation_steps,
                            0.0,
                        )
                    )
                    if (
                        outside_capture_ratio > 0.0
                        and turn_excess > 0.0
                        and stagnation_excess > 0.0
                    ):
                        started_circling_this_step = not self._circling_active[idx]
                        self._circling_active[idx] = True
                        self._circling_relief_progress[idx] = 0

                    if self._circling_active[idx]:
                        waypoint_proximities.append(0.0)
                        circling_score = 1.0
                        self._circling_steps_total[idx] += 1
                    else:
                        waypoint_proximities.append(
                            float(
                                max(proximity_after - proximity_before, 0.0)
                            )
                        )
                        circling_score = 0.0
                    circling_scores.append(circling_score)
                    if (
                        not self._circling_active[idx]
                        and dist_after > (2.0 * arrival_threshold)
                        and best_improvement <= self.stagnation_progress_epsilon_m
                    ):
                        stagnation_excess = max(
                            int(self._circling_stagnation_steps[idx])
                            - self.stagnation_step_threshold,
                            0,
                        )
                        if stagnation_excess > 0:
                            stagnation_scores.append(
                                float(
                                    np.clip(
                                        stagnation_excess
                                        / max(self.stagnation_step_threshold, 1),
                                        0.0,
                                        1.0,
                                    )
                                )
                            )
                        else:
                            stagnation_scores.append(0.0)
                    else:
                        stagnation_scores.append(0.0)
            else:
                self._sync_waypoint_progress_tracking(agent)
                self._sync_waypoint_circling_tracking(agent)
                waypoint_proximities.append(0.0)
                progress_terms.append(0.0)
                circling_scores.append(0.0)
                stagnation_scores.append(0.0)

            current_wp = aircraft.waypoint_manager.current_waypoint
            waypoint_reached = False
            if current_wp is not None:
                waypoint_reached = self._check_line_segment_arrival_local(
                    self._prev_local_cache[idx],
                    self._local_pos_cache[idx],
                    self._wp_local_cache[idx],
                    aircraft.waypoint_manager.arrival_threshold,
                    start_heading=float(self._last_headings[idx]),
                    turn_rate=float(aircraft.actual_turn_rate),
                    cruise_speed=float(aircraft.dynamics.cruise_speed),
                    dt=self.dt,
                )
            if current_wp is not None and waypoint_reached:
                waypoint_hits += 1
                aircraft.last_waypoint_hit_pos = current_wp.to_tuple()
                aircraft.waypoint_manager.hit_waypoints.append(current_wp)
                aircraft.waypoint_manager.advance()
                self._clear_waypoint_reapproach(idx)
                self._closest_wp_signature[idx] = None
                self._closest_wp_distance[idx] = np.inf
                self._circling_wp_signature[idx] = None
                self._circling_active[idx] = False
                self._circling_stagnation_steps[idx] = 0
                self._circling_angular_travel[idx] = 0.0
                self._circling_relief_progress[idx] = 0
                self._reset_reference_route_progress(idx=idx, aircraft=aircraft)
            else:
                self._update_waypoint_reapproach_state(
                    idx=idx,
                    aircraft=aircraft,
                    distance_to_wp=dist_after,
                    best_improvement=best_improvement,
                )

            if (
                current_wp is not None
                and not self._waypoint_reapproach_active[idx]
                and (
                started_circling_this_step
                or self._should_reset_waypoint_capture(
                    idx=idx,
                    aircraft=aircraft,
                    distance_to_wp=dist_after,
                    best_improvement=best_improvement,
                )
                )
            ):
                self._start_waypoint_reapproach(
                    idx=idx,
                    aircraft=aircraft,
                    distance_to_wp=dist_after,
                )

            if not aircraft.waypoint_manager.has_waypoints():
                if self.randomized and self.refill_random_waypoints_on_completion:
                    self._assign_random_waypoints(aircraft)
                    self._clear_waypoint_reapproach(idx)
                    self._closest_wp_signature[idx] = None
                    self._closest_wp_distance[idx] = np.inf
                    self._circling_wp_signature[idx] = None
                    self._circling_active[idx] = False
                    self._circling_stagnation_steps[idx] = 0
                    self._circling_angular_travel[idx] = 0.0
                    self._circling_relief_progress[idx] = 0
                    self._reset_reference_route_progress(idx=idx, aircraft=aircraft)
                else:
                    if self._completion_steps[idx] is None:
                        self._completion_steps[idx] = self.current_step
                    self._clear_waypoint_reapproach(idx)
                    aircraft._enter_loiter()

            inside, soft_risk, outside_depth = self._boundary_status(agent)
            boundary_soft_risks.append(soft_risk)
            outside_depths.append(outside_depth)
            if not inside:
                self.geofence_outside_steps[idx] += 1
                if not self._was_outside[idx]:
                    self.geofence_exit_counts[idx] += 1
                    self._was_outside[idx] = True
            else:
                self._was_outside[idx] = False

            if aircraft.waypoint_manager.current_waypoint is None:
                self._clear_waypoint_reapproach(idx)
                self._closest_wp_signature[idx] = None
                self._closest_wp_distance[idx] = np.inf
                self._circling_wp_signature[idx] = None
                self._circling_active[idx] = False
                self._circling_stagnation_steps[idx] = 0
                self._circling_angular_travel[idx] = 0.0
                self._circling_relief_progress[idx] = 0
                self._reset_reference_route_progress(idx=idx, aircraft=aircraft)

        all_done = all(
            not self.aircraft_by_agent[agent].waypoint_manager.has_waypoints()
            for agent in current_agents
        )
        terminated = self._crashed or (
            all_done and self.terminate_on_all_waypoints_complete
        )
        truncated = self.current_step >= self.max_steps

        self._update_waypoint_local_cache()
        self._refresh_route_guidance_cache()

        team_reward = self._compute_team_reward(
            current_agents=current_agents,
            waypoint_hits=waypoint_hits,
            waypoint_proximities=waypoint_proximities,
            progress_terms=progress_terms,
            circling_scores=circling_scores,
            stagnation_scores=stagnation_scores,
            boundary_soft_risks=boundary_soft_risks,
            outside_depths=outside_depths,
            pair_metrics=pair_metrics,
            completed=all_done,
            truncated=truncated,
        )

        for agent in current_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            heading_rate = float(
                wrap_angle(aircraft.heading - self._last_headings[idx])
                / max(self.dt, 1e-6)
            )
            self._last_heading_rates[idx] = heading_rate
            self._last_headings[idx] = float(aircraft.heading)
        self._last_action_delta[:] = executed_action_vector - self._last_actions
        self._last_actions[:] = executed_action_vector
        self._episode_reward_total += team_reward

        if self._crashed:
            self._termination_reason = "critical_violation"
        elif all_done and self.terminate_on_all_waypoints_complete:
            self._termination_reason = "completed"
        elif truncated:
            self._termination_reason = "max_steps"
        elif all_done:
            self._termination_reason = "mission_complete_loitering"
        else:
            self._termination_reason = "in_progress"

        self._update_obs_cache()

        rewards = {agent: float(team_reward) for agent in current_agents}
        terminations = {agent: bool(terminated) for agent in current_agents}
        truncations = {agent: bool(truncated) for agent in current_agents}
        infos = {
            agent: self._shared_info(
                waypoints_hit=waypoint_hits,
            )
            for agent in current_agents
        }

        if terminated or truncated:
            final_metrics = self.get_episode_metrics()
            for agent in current_agents:
                infos[agent]["episode_metrics"] = final_metrics
            self.agents = []
            return {}, rewards, terminations, truncations, infos

        obs = {
            agent: self._obs_cache[self.agent_name_to_index[agent]].copy()
            for agent in current_agents
        }
        self.agents = current_agents
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        return None

    def runtime_agent_snapshot(self, agent: str) -> dict:
        aircraft = self._runtime_snapshot_aircraft(agent)
        latitude = float(aircraft.position.latitude)
        longitude = float(aircraft.position.longitude)
        local_x, local_y = self.transformer.geo_to_local(latitude, longitude)
        current_waypoint = aircraft.waypoint_manager.current_waypoint
        queued_waypoints = [
            waypoint.to_tuple()
            for waypoint in aircraft.waypoint_manager.waypoint_queue
        ]
        heading_rad = float(aircraft.heading)
        bearing_deg = float(np.degrees(heading_rad) % 360.0)
        return {
            "agent": agent,
            "active": bool(agent in self.agents),
            "mode": aircraft.flight_mode.value,
            "flight_mode": aircraft.flight_mode.value,
            "position": {
                "lat": latitude,
                "lon": longitude,
            },
            "position_latlon": (latitude, longitude),
            "local_position_m": {
                "x": float(local_x),
                "y": float(local_y),
            },
            "heading_rad": heading_rad,
            "heading_deg": float(np.degrees(heading_rad)),
            "bearing_deg": bearing_deg,
            "current_waypoint": (
                current_waypoint.to_tuple()
                if current_waypoint is not None
                else None
            ),
            "queued_waypoints": queued_waypoints,
            "completed_waypoints": int(len(aircraft.waypoint_manager.hit_waypoints)),
            "remaining_waypoints": int(
                len(queued_waypoints)
                + (1 if current_waypoint is not None else 0)
            ),
            "distance_traveled_m": float(aircraft.distance_traveled),
            "actual_turn_rate_rad_s": float(aircraft.actual_turn_rate),
            "desired_turn_rate_rad_s": float(aircraft.desired_turn_rate),
            "current_step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "sim_time_s": float(self.current_step * self.dt),
        }

    def runtime_agent_snapshots(self) -> dict[str, dict]:
        return {
            agent: self.runtime_agent_snapshot(agent)
            for agent in sorted(self.aircraft_by_agent)
        }

    def runtime_waypoint_snapshot(self, agent: str) -> dict:
        return self.runtime_agent_snapshot(agent)

    def append_runtime_waypoints(
        self,
        agent: str,
        waypoints: Any,
    ) -> dict:
        aircraft = self._runtime_waypoint_aircraft(agent)
        positions = self._coerce_runtime_waypoint_list(waypoints)
        if positions:
            aircraft.append_waypoints(positions)
            self._refresh_after_runtime_waypoint_update(agent)
        return self.runtime_waypoint_snapshot(agent)

    def replace_runtime_waypoint_queue(
        self,
        agent: str,
        waypoints: Any,
        *,
        replace_current: bool = False,
    ) -> dict:
        aircraft = self._runtime_waypoint_aircraft(agent)
        positions = self._coerce_runtime_waypoint_list(waypoints)
        aircraft.replace_waypoint_queue(
            positions,
            replace_current=replace_current,
        )
        self._refresh_after_runtime_waypoint_update(agent)
        return self.runtime_waypoint_snapshot(agent)

    def _runtime_waypoint_aircraft(self, agent: str) -> FixedWingAircraft:
        if not self.allow_live_waypoint_updates:
            raise RuntimeError(
                "Live waypoint updates are disabled for this environment. "
                "Use the trained-policy runtime path to enable them."
            )
        if agent not in self.aircraft_by_agent or agent not in self.agents:
            raise KeyError(
                f"Agent {agent!r} is not active in the current episode."
            )
        return self.aircraft_by_agent[agent]

    def _runtime_snapshot_aircraft(self, agent: str) -> FixedWingAircraft:
        if not self.allow_live_waypoint_updates:
            raise RuntimeError(
                "Live state snapshots are disabled for this environment. "
                "Use the trained-policy runtime path to enable them."
            )
        if agent not in self.aircraft_by_agent:
            raise KeyError(
                f"Agent {agent!r} is not available in the current runtime session."
            )
        return self.aircraft_by_agent[agent]

    def _coerce_runtime_waypoint_list(self, waypoints: Any) -> List[Position]:
        if waypoints is None:
            raise TypeError("waypoints cannot be None")
        if (
            isinstance(waypoints, Position)
            or isinstance(waypoints, dict)
            or self._is_scalar_waypoint_pair(waypoints)
        ):
            raw_waypoints = [waypoints]
        else:
            raw_waypoints = list(waypoints)
        return [
            self._coerce_runtime_waypoint(waypoint)
            for waypoint in raw_waypoints
        ]

    def _coerce_runtime_waypoint(self, waypoint: Any) -> Position:
        if isinstance(waypoint, Position):
            return Position(
                float(waypoint.latitude),
                float(waypoint.longitude),
            )
        if isinstance(waypoint, dict):
            if "latitude" in waypoint and "longitude" in waypoint:
                return Position(
                    float(waypoint["latitude"]),
                    float(waypoint["longitude"]),
                )
            if "lat" in waypoint and "lon" in waypoint:
                return Position(
                    float(waypoint["lat"]),
                    float(waypoint["lon"]),
                )
            raise TypeError(
                "Waypoint dicts must provide latitude/longitude or lat/lon keys."
            )
        if self._is_scalar_waypoint_pair(waypoint):
            latitude, longitude = waypoint
            return Position(float(latitude), float(longitude))
        raise TypeError(
            "Waypoints must be Position instances, (lat, lon) pairs, or "
            "dicts with latitude/longitude keys."
        )

    def _is_scalar_waypoint_pair(self, waypoint: Any) -> bool:
        if not isinstance(waypoint, (list, tuple, np.ndarray)):
            return False
        if len(waypoint) != 2:
            return False
        return bool(np.isscalar(waypoint[0]) and np.isscalar(waypoint[1]))

    def _reset_runtime_waypoint_tracking(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
    ) -> None:
        self._clear_waypoint_reapproach(idx)
        self._closest_wp_signature[idx] = None
        self._closest_wp_distance[idx] = np.inf
        self._circling_wp_signature[idx] = None
        self._circling_active[idx] = False
        self._circling_stagnation_steps[idx] = 0
        self._circling_angular_travel[idx] = 0.0
        self._circling_relief_progress[idx] = 0
        self._reset_reference_route_progress(idx=idx, aircraft=aircraft)

    def _refresh_after_runtime_waypoint_update(self, agent: str) -> None:
        aircraft = self.aircraft_by_agent[agent]
        idx = self.agent_name_to_index[agent]
        self._reset_runtime_waypoint_tracking(idx=idx, aircraft=aircraft)

        if aircraft.waypoint_manager.has_waypoints():
            self._completion_steps[idx] = None
            aircraft.flight_mode = FlightMode.NAVIGATING
            aircraft.loiter_center = None
        else:
            if self._completion_steps[idx] is None:
                self._completion_steps[idx] = self.current_step
            aircraft._enter_loiter()

        self._sync_local_caches()
        self._sync_waypoint_progress_tracking(agent)
        self._sync_waypoint_circling_tracking(agent)
        self._refresh_route_guidance_cache()
        refreshed_max_steps = self._resolve_episode_max_steps({})
        self.max_steps = max(
            int(self.max_steps),
            int(refreshed_max_steps),
            int(self.current_step) + 1,
        )
        self._update_obs_cache(fill_history=True)

    def _resolve_mission_waypoint_count(self, options: dict) -> int:
        requested = options.get("mission_waypoint_count")
        if requested is not None:
            return max(int(requested), 1)
        if self.mission_waypoint_count_min == self.mission_waypoint_count_max:
            return self.mission_waypoint_count_min
        return int(
            self._rng.integers(
                self.mission_waypoint_count_min,
                self.mission_waypoint_count_max + 1,
            )
        )

    def _episode_assigned_waypoint_count(self) -> int:
        return int(
            sum(
                aircraft.waypoint_manager.queue_size()
                + (1 if aircraft.waypoint_manager.current_waypoint is not None else 0)
                for aircraft in self.aircraft_by_agent.values()
            )
        )

    def _episode_max_planned_route_distance_m(self) -> float:
        if not self.aircraft_by_agent:
            return 0.0
        return float(
            max(
                planned_route_distance_m(
                    aircraft,
                    self.transformer,
                )
                for aircraft in self.aircraft_by_agent.values()
            )
        )

    def _resolve_episode_max_steps(self, options: dict) -> int:
        requested = options.get("max_steps")
        if requested is not None:
            assigned_waypoints = self._episode_assigned_waypoint_count()
            max_route_distance_m = self._episode_max_planned_route_distance_m()
            self._episode_timeout_assigned_waypoints = assigned_waypoints
            self._episode_timeout_reference_waypoints = assigned_waypoints
            self._episode_timeout_max_route_distance_m = max_route_distance_m
            self._episode_timeout_reference_route_distance_m = max_route_distance_m
            return max(int(requested), 1)

        assigned_waypoints = self._episode_assigned_waypoint_count()
        max_route_distance_m = self._episode_max_planned_route_distance_m()
        active_agents = max(len(self.agents), 1)
        reference_waypoints = self.timeout_reference_waypoints
        if reference_waypoints is None:
            reference_waypoints = max(self.mission_waypoint_count, 1) * active_agents
        reference_route_distance_m = self.timeout_reference_route_distance_m
        if reference_route_distance_m is None:
            reference_route_distance_m = max_route_distance_m
        self._episode_timeout_assigned_waypoints = assigned_waypoints
        self._episode_timeout_reference_waypoints = int(reference_waypoints)
        self._episode_timeout_max_route_distance_m = float(max_route_distance_m)
        self._episode_timeout_reference_route_distance_m = float(
            reference_route_distance_m
        )

        episode_max_steps = self.base_max_steps
        if (
            self.timeout_scale_with_mission_size
            and self.timeout_steps_per_additional_waypoint > 0
        ):
            extra_waypoints = max(assigned_waypoints - int(reference_waypoints), 0)
            episode_max_steps += (
                extra_waypoints * self.timeout_steps_per_additional_waypoint
            )
        if (
            self.timeout_scale_with_route_distance
            and self.timeout_steps_per_additional_route_km > 0.0
        ):
            extra_route_distance_m = max(
                max_route_distance_m - float(reference_route_distance_m),
                0.0,
            )
            episode_max_steps += int(
                np.ceil(
                    (extra_route_distance_m / 1000.0)
                    * self.timeout_steps_per_additional_route_km
                )
            )
        return int(
            min(
                max(episode_max_steps, self.base_max_steps),
                self.timeout_max_steps,
            )
        )

    def _reset_random_episode(self, options: dict):
        self._current_mission_waypoint_count = self._resolve_mission_waypoint_count(options)
        num_agents = int(
            options.get(
                "num_agents",
                self._rng.integers(self.min_agents, self.max_agents + 1),
            )
        )
        self.agents = list(self.possible_agents[:num_agents])

        if options.get("top_left") and options.get("bottom_right"):
            top_left = tuple(options["top_left"])
            bottom_right = tuple(options["bottom_right"])
        elif self.fixed_top_left and self.fixed_bottom_right:
            top_left = tuple(self.fixed_top_left)
            bottom_right = tuple(self.fixed_bottom_right)
        else:
            if self.default_origin:
                origin = self.default_origin
            else:
                origin = (
                    float(self._rng.uniform(-70.0, 70.0)),
                    float(self._rng.uniform(-170.0, 170.0)),
                )
            width_m = float(
                options.get(
                    "box_width_m",
                    self._rng.uniform(*self.map_size_range_m),
                )
            )
            height_m = float(
                options.get(
                    "box_height_m",
                    self._rng.uniform(*self.map_size_range_m),
                )
            )
            top_left, bottom_right = build_rect_bounds(origin, width_m, height_m)
        self._set_bounds(top_left, bottom_right)

        self.aircraft_by_agent = {}
        start_margin_request = max(
            float(self.turning_radius_max),
            self.min_start_separation_m * 0.5,
            self.caution_dist * 0.75,
        )
        start_margin_x = self._effective_edge_margin(
            span_m=self.box_width_m,
            requested_margin_m=start_margin_request,
            max_ratio=0.28,
        )
        start_margin_y = self._effective_edge_margin(
            span_m=self.box_height_m,
            requested_margin_m=start_margin_request,
            max_ratio=0.28,
        )
        start_positions = self._sample_local_points(
            count=len(self.agents),
            min_separation_m=self.min_start_separation_m,
            x_min=self.local_min_x + start_margin_x,
            x_max=self.local_max_x - start_margin_x,
            y_min=self.local_min_y + start_margin_y,
            y_max=self.local_max_y - start_margin_y,
        )
        for idx, agent in enumerate(self.agents):
            lat, lon = self.transformer.local_to_geo(
                float(start_positions[idx, 0]),
                float(start_positions[idx, 1]),
            )
            cruise_speed = float(
                self._rng.uniform(self.cruise_speed_min, self.cruise_speed_max)
            )
            turning_radius = float(
                self._rng.uniform(self.turning_radius_min, self.turning_radius_max)
            )
            aircraft = FixedWingAircraft(
                id_tag=agent,
                initial_position=Position(lat, lon),
                initial_heading=float(self._rng.uniform(-np.pi, np.pi)),
                cruise_speed=cruise_speed,
                turning_radius=turning_radius,
                turn_response_time_s=self.turn_response_time_s,
            )
            aircraft.waypoint_manager.arrival_threshold = self.waypoint_arrival_radius
            aircraft.path_history = [Position(lat, lon)]
            aircraft.last_waypoint_hit_pos = None
            aircraft.actual_turn_rate = 0.0
            aircraft.desired_turn_rate = 0.0
            self._assign_random_waypoints(aircraft)
            self.aircraft_by_agent[agent] = aircraft

    def _reset_manual_episode(self, options: dict):
        self._current_mission_waypoint_count = self._resolve_mission_waypoint_count(options)
        generate_random_waypoints = bool(
            options.get("generate_random_waypoints", False)
        )
        self.agents = list(self.manual_agents)
        top_left = tuple(options.get("top_left", self.fixed_top_left))
        bottom_right = tuple(options.get("bottom_right", self.fixed_bottom_right))
        if top_left is None or bottom_right is None:
            origin = tuple(options.get("origin", self.default_origin))
            box_size = float(options["box_size_m"])
            top_left, bottom_right = build_box_bounds(origin, box_size)
        self._set_bounds(top_left, bottom_right)

        self.aircraft_by_agent = {}
        for agent in self.agents:
            params = self.manual_missions[agent]
            lat, lon = params["initial_position"]
            aircraft = FixedWingAircraft(
                id_tag=agent,
                initial_position=Position(lat, lon),
                initial_heading=heading_to_radians(params["initial_heading"]),
                cruise_speed=float(params["cruise_speed"]),
                turning_radius=float(params["turning_radius"]),
                mission=(
                    []
                    if generate_random_waypoints
                    else list(params["waypoints"])
                ),
                turn_response_time_s=self.turn_response_time_s,
            )
            aircraft.waypoint_manager.arrival_threshold = self.waypoint_arrival_radius
            aircraft.path_history = [Position(lat, lon)]
            aircraft.last_waypoint_hit_pos = None
            aircraft.actual_turn_rate = 0.0
            aircraft.desired_turn_rate = 0.0
            if generate_random_waypoints:
                self._assign_random_waypoints(aircraft)
            self.aircraft_by_agent[agent] = aircraft

    def _set_bounds(
        self,
        top_left: Tuple[float, float],
        bottom_right: Tuple[float, float],
    ):
        self.min_lat, self.max_lat = sorted([top_left[0], bottom_right[0]])
        self.min_lon, self.max_lon = sorted([top_left[1], bottom_right[1]])
        self.transformer = CoordinateTransformer(self.min_lat, self.min_lon)

        east_x, _ = self.transformer.geo_to_local(self.min_lat, self.max_lon)
        _, north_y = self.transformer.geo_to_local(self.max_lat, self.min_lon)
        self.local_min_x = 0.0
        self.local_min_y = 0.0
        self.local_max_x = float(east_x)
        self.local_max_y = float(north_y)
        self.box_width_m = max(self.local_max_x - self.local_min_x, 1.0)
        self.box_height_m = max(self.local_max_y - self.local_min_y, 1.0)
        self.map_diag_m = max(
            float(np.hypot(self.box_width_m, self.box_height_m)),
            1.0,
        )
        self.map_size_scale = max(self.map_size_scale, self.box_width_m, self.box_height_m)

    def _sample_local_points(
        self,
        *,
        count: int,
        min_separation_m: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> np.ndarray:
        samples: List[np.ndarray] = []
        attempts = 0
        while len(samples) < count and attempts < 5_000:
            attempts += 1
            candidate = np.asarray(
                [
                    self._rng.uniform(x_min, x_max),
                    self._rng.uniform(y_min, y_max),
                ],
                dtype=np.float32,
            )
            if all(np.linalg.norm(candidate - existing) >= min_separation_m for existing in samples):
                samples.append(candidate)
        if len(samples) < count:
            while len(samples) < count:
                samples.append(
                    np.asarray(
                        [
                            self._rng.uniform(x_min, x_max),
                            self._rng.uniform(y_min, y_max),
                        ],
                        dtype=np.float32,
                    )
                )
        return np.stack(samples, axis=0)

    def _effective_edge_margin(
        self,
        *,
        span_m: float,
        requested_margin_m: float,
        max_ratio: float,
    ) -> float:
        return float(
            min(
                max(requested_margin_m, self.boundary_margin * span_m),
                max(span_m * max_ratio, 1.0),
            )
        )

    def _assign_random_waypoints(self, aircraft: FixedWingAircraft):
        waypoint_margin_request = max(
            float(self.turning_radius_max),
            float(aircraft.waypoint_manager.arrival_threshold) * 1.5,
            self.caution_dist * 0.5,
        )
        margin_x = self._effective_edge_margin(
            span_m=self.box_width_m,
            requested_margin_m=waypoint_margin_request,
            max_ratio=0.25,
        )
        margin_y = self._effective_edge_margin(
            span_m=self.box_height_m,
            requested_margin_m=waypoint_margin_request,
            max_ratio=0.25,
        )
        x_min = self.local_min_x + margin_x
        x_max = self.local_max_x - margin_x
        y_min = self.local_min_y + margin_y
        y_max = self.local_max_y - margin_y
        min_sep = min(
            max(
                float(aircraft.dynamics.turning_radius) * 1.5,
                float(aircraft.waypoint_manager.arrival_threshold) * 2.5,
                self.caution_dist * 0.9,
            ),
            0.35 * min(self.box_width_m, self.box_height_m),
        )

        points = self._sample_local_points(
            count=self._current_mission_waypoint_count,
            min_separation_m=min_sep,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        start_local = np.asarray(
            self.transformer.geo_to_local(
                aircraft.position.latitude,
                aircraft.position.longitude,
            ),
            dtype=np.float32,
        )
        ordered_points = order_route_points(
            start_local=start_local,
            start_heading=float(aircraft.heading),
            points=points,
            turn_radius=float(aircraft.dynamics.turning_radius),
            arrival_threshold=float(aircraft.waypoint_manager.arrival_threshold),
            caution_dist=self.caution_dist,
            box_width_m=self.box_width_m,
            box_height_m=self.box_height_m,
        )
        for x, y in ordered_points:
            lat, lon = self.transformer.local_to_geo(float(x), float(y))
            aircraft.add_wp(Position(lat, lon))
        if aircraft.waypoint_manager.has_waypoints():
            aircraft.flight_mode = FlightMode.NAVIGATING
            aircraft.loiter_center = None

    def _sync_local_caches(self):
        self._local_pos_cache.fill(0.0)
        if not self.agents:
            return
        lats = [self.aircraft_by_agent[agent].position.latitude for agent in self.agents]
        lons = [self.aircraft_by_agent[agent].position.longitude for agent in self.agents]
        x_vals, y_vals = self.transformer.geo_to_local(lats, lons)
        for agent, x_val, y_val in zip(self.agents, x_vals, y_vals):
            idx = self.agent_name_to_index[agent]
            self._local_pos_cache[idx, 0] = float(x_val)
            self._local_pos_cache[idx, 1] = float(y_val)
        self._update_waypoint_local_cache()
        self._update_pairwise_distance_cache()

    def _update_waypoint_local_cache(self):
        self._wp_local_cache[:] = self._local_pos_cache
        self._next_wp_local_cache[:] = self._local_pos_cache
        self._has_next_wp[:] = False
        wp_agents = []
        wp_lats = []
        wp_lons = []
        next_wp_agents = []
        next_wp_lats = []
        next_wp_lons = []
        for agent in self.agents:
            aircraft = self.aircraft_by_agent[agent]
            wp = aircraft.waypoint_manager.current_waypoint
            if wp is not None:
                wp_agents.append(agent)
                wp_lats.append(wp.latitude)
                wp_lons.append(wp.longitude)
            if aircraft.waypoint_manager.waypoint_queue:
                next_wp = aircraft.waypoint_manager.waypoint_queue[0]
                next_wp_agents.append(agent)
                next_wp_lats.append(next_wp.latitude)
                next_wp_lons.append(next_wp.longitude)
                self._has_next_wp[self.agent_name_to_index[agent]] = True
        if wp_agents:
            wp_x, wp_y = self.transformer.geo_to_local(wp_lats, wp_lons)
            for agent, x_val, y_val in zip(wp_agents, wp_x, wp_y):
                idx = self.agent_name_to_index[agent]
                self._wp_local_cache[idx, 0] = float(x_val)
                self._wp_local_cache[idx, 1] = float(y_val)
        if next_wp_agents:
            next_wp_x, next_wp_y = self.transformer.geo_to_local(next_wp_lats, next_wp_lons)
            for agent, x_val, y_val in zip(next_wp_agents, next_wp_x, next_wp_y):
                idx = self.agent_name_to_index[agent]
                self._next_wp_local_cache[idx, 0] = float(x_val)
                self._next_wp_local_cache[idx, 1] = float(y_val)

    def _sync_waypoint_progress_tracking(
        self,
        agent: str,
        *,
        distance_to_wp: Optional[float] = None,
    ) -> None:
        idx = self.agent_name_to_index[agent]
        aircraft = self.aircraft_by_agent[agent]
        waypoint = aircraft.waypoint_manager.current_waypoint
        if waypoint is None:
            self._closest_wp_signature[idx] = None
            self._closest_wp_distance[idx] = np.inf
            return

        signature = (float(waypoint.latitude), float(waypoint.longitude))
        if self._closest_wp_signature[idx] == signature:
            return

        if distance_to_wp is None:
            distance_to_wp = float(
                np.linalg.norm(self._wp_local_cache[idx] - self._local_pos_cache[idx])
            )
        self._closest_wp_signature[idx] = signature
        self._closest_wp_distance[idx] = float(max(distance_to_wp, 0.0))

    def _sync_waypoint_circling_tracking(self, agent: str) -> None:
        idx = self.agent_name_to_index[agent]
        aircraft = self.aircraft_by_agent[agent]
        waypoint = aircraft.waypoint_manager.current_waypoint
        if waypoint is None:
            self._circling_wp_signature[idx] = None
            self._circling_active[idx] = False
            self._circling_stagnation_steps[idx] = 0
            self._circling_angular_travel[idx] = 0.0
            self._circling_relief_progress[idx] = 0
            return

        signature = (float(waypoint.latitude), float(waypoint.longitude))
        if self._circling_wp_signature[idx] == signature:
            return

        self._circling_wp_signature[idx] = signature
        self._circling_active[idx] = False
        self._circling_stagnation_steps[idx] = 0
        self._circling_angular_travel[idx] = 0.0
        self._circling_relief_progress[idx] = 0

    def _reset_reference_route_progress(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
    ) -> None:
        self._segment_start_local[idx] = self._local_pos_cache[idx]
        self._segment_start_heading[idx] = float(aircraft.heading)
        self._segment_path_length[idx] = 0.0
        self._reference_route_cache[idx] = None
        self._route_progress_anchor[idx] = 0.0

    def _clear_waypoint_reapproach(self, idx: int) -> None:
        self._waypoint_reapproach_active[idx] = False
        self._waypoint_reapproach_hold_remaining[idx] = 0
        self._waypoint_reapproach_release_distance[idx] = 0.0

    def _should_apply_waypoint_reapproach_assist(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
    ) -> bool:
        if not self._waypoint_reapproach_active[idx]:
            return False
        if (
            aircraft.waypoint_manager.current_waypoint is None
            or aircraft.flight_mode == FlightMode.LOITERING
        ):
            self._clear_waypoint_reapproach(idx)
            return False
        return not self._deconfliction_active[idx]

    def _should_reset_waypoint_capture(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
        distance_to_wp: float,
        best_improvement: float,
    ) -> bool:
        waypoint = aircraft.waypoint_manager.current_waypoint
        if waypoint is None or self._deconfliction_active[idx]:
            return False

        arrival_threshold = max(
            float(aircraft.waypoint_manager.arrival_threshold),
            1e-6,
        )
        if distance_to_wp <= arrival_threshold:
            return False

        turning_radius = max(float(aircraft.dynamics.turning_radius), 1.0)
        close_capture_dist = max(
            turning_radius * 1.5,
            arrival_threshold * 2.5,
            float(aircraft.dynamics.cruise_speed) * self.dt * 2.0,
        )
        if distance_to_wp > close_capture_dist:
            return False
        if best_improvement > self.stagnation_progress_epsilon_m:
            return False
        if int(self._circling_stagnation_steps[idx]) < max(
            4,
            self.stagnation_step_threshold // 2,
        ):
            return False

        rel_wp = np.asarray(
            self._wp_local_cache[idx] - self._local_pos_cache[idx],
            dtype=np.float32,
        )
        rel_bearing_wp = wrap_angle(
            float(np.arctan2(rel_wp[0], rel_wp[1])) - float(aircraft.heading)
        )
        left_margin, right_margin = turn_circle_feasibility_features(
            pos_local=self._local_pos_cache[idx],
            wp_local=self._wp_local_cache[idx],
            heading=float(aircraft.heading),
            turning_radius=turning_radius,
        )
        constrained_capture = min(left_margin, right_margin) < -0.05
        badly_misaligned = abs(rel_bearing_wp) >= (np.pi * 0.35)
        overshot_waypoint = float(np.cos(rel_bearing_wp)) < 0.35
        return bool(
            (constrained_capture and badly_misaligned)
            or overshot_waypoint
        )

    def _should_release_waypoint_reapproach(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
        distance_to_wp: float,
        best_improvement: float,
    ) -> bool:
        waypoint = aircraft.waypoint_manager.current_waypoint
        if waypoint is None:
            return True

        arrival_threshold = max(
            float(aircraft.waypoint_manager.arrival_threshold),
            1e-6,
        )
        turning_radius = max(float(aircraft.dynamics.turning_radius), 1.0)
        release_distance = max(
            float(self._waypoint_reapproach_release_distance[idx]),
            turning_radius * self.waypoint_reapproach_release_distance_scale,
            arrival_threshold * self.waypoint_reapproach_release_arrival_scale,
        )
        if distance_to_wp >= release_distance:
            return True

        rel_wp = np.asarray(
            self._wp_local_cache[idx] - self._local_pos_cache[idx],
            dtype=np.float32,
        )
        rel_bearing_wp = wrap_angle(
            float(np.arctan2(rel_wp[0], rel_wp[1])) - float(aircraft.heading)
        )
        left_margin, right_margin = turn_circle_feasibility_features(
            pos_local=self._local_pos_cache[idx],
            wp_local=self._wp_local_cache[idx],
            heading=float(aircraft.heading),
            turning_radius=turning_radius,
        )
        constrained_capture = min(left_margin, right_margin) < -0.02
        aligned_for_retry = abs(rel_bearing_wp) <= (np.pi * 0.28)
        facing_waypoint = float(np.cos(rel_bearing_wp)) > 0.45
        return bool(
            best_improvement >= self.circling_progress_reset_m
            and distance_to_wp > (arrival_threshold * 1.5)
            and aligned_for_retry
            and facing_waypoint
            and not constrained_capture
        )

    def _update_waypoint_reapproach_state(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
        distance_to_wp: float,
        best_improvement: float,
    ) -> None:
        if not self._waypoint_reapproach_active[idx]:
            return
        if (
            aircraft.waypoint_manager.current_waypoint is None
            or aircraft.flight_mode == FlightMode.LOITERING
        ):
            self._clear_waypoint_reapproach(idx)
            return
        if self._deconfliction_active[idx]:
            return

        if self._waypoint_reapproach_hold_remaining[idx] > 0:
            self._waypoint_reapproach_hold_remaining[idx] -= 1
            if self._waypoint_reapproach_hold_remaining[idx] > 0:
                return

        if self._should_release_waypoint_reapproach(
            idx=idx,
            aircraft=aircraft,
            distance_to_wp=distance_to_wp,
            best_improvement=best_improvement,
        ):
            self._clear_waypoint_reapproach(idx)

    def _start_waypoint_reapproach(
        self,
        *,
        idx: int,
        aircraft: FixedWingAircraft,
        distance_to_wp: float,
    ) -> None:
        waypoint = aircraft.waypoint_manager.current_waypoint
        if waypoint is None:
            return

        if not self._waypoint_reapproach_active[idx]:
            self._waypoint_reapproach_events_total[idx] += 1
        if self._circling_active[idx]:
            self._circling_breakouts_total[idx] += 1
        arrival_threshold = max(
            float(aircraft.waypoint_manager.arrival_threshold),
            1e-6,
        )
        turning_radius = max(float(aircraft.dynamics.turning_radius), 1.0)
        self._waypoint_reapproach_active[idx] = True
        self._waypoint_reapproach_hold_remaining[idx] = (
            self.waypoint_reapproach_min_steps
        )
        self._waypoint_reapproach_release_distance[idx] = float(
            max(
                distance_to_wp + (0.75 * turning_radius),
                turning_radius * self.waypoint_reapproach_release_distance_scale,
                arrival_threshold * self.waypoint_reapproach_release_arrival_scale,
            )
        )
        self._closest_wp_signature[idx] = (
            float(waypoint.latitude),
            float(waypoint.longitude),
        )
        self._closest_wp_distance[idx] = float(max(distance_to_wp, 0.0))
        self._circling_active[idx] = False
        self._circling_stagnation_steps[idx] = 0
        self._circling_angular_travel[idx] = 0.0
        self._circling_relief_progress[idx] = 0
        self._reset_reference_route_progress(idx=idx, aircraft=aircraft)

    def _update_pairwise_distance_cache(self):
        self._dx_matrix.fill(0.0)
        self._dy_matrix.fill(0.0)
        self._dist_matrix.fill(0.0)
        if not self.agents:
            return
        indices = np.asarray(
            [self.agent_name_to_index[agent] for agent in self.agents],
            dtype=np.intp,
        )
        pos = self._local_pos_cache[indices]
        dx = pos[:, None, 0] - pos[None, :, 0]
        dy = pos[:, None, 1] - pos[None, :, 1]
        dist = np.hypot(dx, dy)
        active_grid = np.ix_(indices, indices)
        self._dx_matrix[active_grid] = dx
        self._dy_matrix[active_grid] = dy
        self._dist_matrix[active_grid] = dist

    def _refresh_route_guidance_cache(self):
        self._last_reference_action_vector[:] = 0.0

        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            if (
                aircraft.waypoint_manager.current_waypoint is None
                or aircraft.flight_mode == FlightMode.LOITERING
            ):
                continue

            pos_local = np.asarray(self._local_pos_cache[idx], dtype=np.float32)
            wp_local = np.asarray(self._wp_local_cache[idx], dtype=np.float32)
            route = build_reference_route_local(
                cached_route=self._reference_route_cache[idx],
                aircraft=aircraft,
                transformer=self.transformer,
                segment_start_local=self._segment_start_local[idx],
                segment_start_heading=float(self._segment_start_heading[idx]),
                bounds_local=(
                    float(self.local_min_x),
                    float(self.local_max_x),
                    float(self.local_min_y),
                    float(self.local_max_y),
                ),
            )
            self._reference_route_cache[idx] = route
            action, updated_anchor = compute_reference_action(
                pos_local=pos_local,
                wp_local=wp_local,
                route=route,
                route_progress_anchor=float(self._route_progress_anchor[idx]),
                arrival_threshold=float(aircraft.waypoint_manager.arrival_threshold),
                current_heading=float(aircraft.heading),
                cruise_speed=float(aircraft.dynamics.cruise_speed),
                turning_radius=float(aircraft.dynamics.turning_radius),
                max_turn_rate=float(aircraft.dynamics.max_turn_rate),
                dt=self.dt,
                lookahead_time_s=self.guidance_lookahead_time_s,
                min_lookahead_m=self.guidance_min_lookahead_m,
                max_lookahead_m=self.guidance_max_lookahead_m,
                route_commit_scale=self.guidance_route_commit_scale,
                turn_gain=self.guidance_turn_gain,
                turn_lookahead_scale=self.guidance_turn_lookahead_scale,
                turn_radius_floor_scale=self.guidance_turn_radius_floor_scale,
            )
            self._route_progress_anchor[idx] = updated_anchor
            self._last_reference_action_vector[idx] = action

    def _append_obs_history(self, *, fill_history: bool = False):
        if fill_history:
            self._obs_history[:] = self._base_obs_cache[:, None, :]
        else:
            if self.obs_stack_size > 1:
                self._obs_history[:, :-1] = self._obs_history[:, 1:]
            self._obs_history[:, -1] = self._base_obs_cache
        self._obs_cache[:] = self._obs_history.reshape(self.max_agents, self.obs_dim)

    def _update_obs_cache(self, *, fill_history: bool = False):
        self._base_obs_cache.fill(0.0)
        if not self.agents:
            self._append_obs_history(fill_history=fill_history)
            return

        active_indices = [
            self.agent_name_to_index[agent]
            for agent in self.agents
        ]
        local_pos_cache = self._local_pos_cache
        wp_local_cache = self._wp_local_cache
        next_wp_local_cache = self._next_wp_local_cache
        dist_matrix = self._dist_matrix
        map_diag_scale = max(self.map_diag_m, 1.0)
        cruise_speed_scale = max(self.cruise_speed_max, 1.0)
        max_turn_rate_scale = max(self.max_turn_rate_max, 1e-6)
        caution_dist_scale = max(self.caution_dist, 1.0)
        critical_dist_scale = max(self.critical_dist, 1e-6)
        lookahead_time_scale = max(self.caution_lookahead_time_s, 1e-6)
        box_width_scale = max(self.box_width_m, 1.0)
        box_height_scale = max(self.box_height_m, 1.0)

        current_rel_bearing_matrix = np.full(
            (self.max_agents, self.max_agents),
            np.nan,
            dtype=np.float32,
        )
        current_d_cpa_matrix = np.full(
            (self.max_agents, self.max_agents),
            np.nan,
            dtype=np.float32,
        )
        current_pair_conflict_pressure_matrix = np.full(
            (self.max_agents, self.max_agents),
            np.nan,
            dtype=np.float32,
        )

        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            wp_vec = wp_local_cache[idx] - local_pos_cache[idx]
            dist_wp = float(np.linalg.norm(wp_vec))
            bearing_wp = float(np.arctan2(wp_vec[0], wp_vec[1]))
            rel_bearing_wp = wrap_angle(bearing_wp - aircraft.heading)
            next_wp_exists = bool(self._has_next_wp[idx])
            if next_wp_exists:
                next_leg_vec = next_wp_local_cache[idx] - wp_local_cache[idx]
                next_leg_bearing = float(np.arctan2(next_leg_vec[0], next_leg_vec[1]))
                lookahead_turn = wrap_angle(next_leg_bearing - bearing_wp)
                next_leg_sin = float(np.sin(lookahead_turn))
                next_leg_cos = float(np.cos(lookahead_turn))
            else:
                next_leg_sin = 0.0
                next_leg_cos = 1.0

            x_pos = float(local_pos_cache[idx, 0])
            y_pos = float(local_pos_cache[idx, 1])
            d_south = clip_scalar(y_pos / box_height_scale, -1.0, 1.0)
            d_north = clip_scalar((self.box_height_m - y_pos) / box_height_scale, -1.0, 1.0)
            d_west = clip_scalar(x_pos / box_width_scale, -1.0, 1.0)
            d_east = clip_scalar((self.box_width_m - x_pos) / box_width_scale, -1.0, 1.0)
            max_turn_rate = max(float(aircraft.dynamics.max_turn_rate), 1e-6)
            _, boundary_soft_risk, _ = self._boundary_status(agent)
            boundary_time_ratio = time_to_boundary_ahead(
                pos_local=local_pos_cache[idx],
                heading=float(aircraft.heading),
                cruise_speed=float(aircraft.dynamics.cruise_speed),
                local_min_x=float(self.local_min_x),
                local_max_x=float(self.local_max_x),
                local_min_y=float(self.local_min_y),
                local_max_y=float(self.local_max_y),
                lookahead_time_s=self.caution_lookahead_time_s,
            )
            left_turn_feasibility, right_turn_feasibility = (
                turn_circle_feasibility_features(
                    pos_local=local_pos_cache[idx],
                    wp_local=wp_local_cache[idx],
                    heading=float(aircraft.heading),
                    turning_radius=float(aircraft.dynamics.turning_radius),
                )
            )
            episode_progress = float(
                clip_scalar(self.current_step / max(self.max_steps, 1), 0.0, 1.0)
            )
            has_active_waypoint = float(
                aircraft.waypoint_manager.current_waypoint is not None
            )
            loiter_mode = float(aircraft.flight_mode == FlightMode.LOITERING)
            total_assigned_waypoints = (
                len(aircraft.waypoint_manager.hit_waypoints)
                + aircraft.waypoint_manager.queue_size()
                + (1 if aircraft.waypoint_manager.current_waypoint is not None else 0)
            )
            remaining_waypoints_ratio = float(
                np.clip(
                    (
                        aircraft.waypoint_manager.queue_size()
                        + (1 if aircraft.waypoint_manager.current_waypoint is not None else 0)
                    )
                    / max(total_assigned_waypoints, 1),
                    0.0,
                    1.0,
                )
            )

            self_features = [
                clip_scalar(dist_wp / map_diag_scale, 0.0, 2.0),
                np.sin(rel_bearing_wp),
                np.cos(rel_bearing_wp),
                next_leg_sin,
                next_leg_cos,
                1.0 if next_wp_exists else 0.0,
                np.sin(aircraft.heading),
                np.cos(aircraft.heading),
                clip_scalar(
                    aircraft.dynamics.cruise_speed / cruise_speed_scale,
                    0.0,
                    2.0,
                ),
                clip_scalar(
                    aircraft.dynamics.max_turn_rate / max_turn_rate_scale,
                    0.0,
                    2.0,
                ),
                clip_scalar(aircraft.actual_turn_rate / max_turn_rate, -1.0, 1.0),
                clip_scalar(aircraft.desired_turn_rate / max_turn_rate, -1.0, 1.0),
                clip_scalar(
                    (aircraft.desired_turn_rate - aircraft.actual_turn_rate)
                    / max_turn_rate,
                    -2.0,
                    2.0,
                ),
                float(self._last_actions[idx]),
                float(self._last_action_delta[idx]),
                d_south,
                d_north,
                d_west,
                d_east,
                boundary_time_ratio,
                left_turn_feasibility,
                right_turn_feasibility,
                episode_progress,
                has_active_waypoint,
                loiter_mode,
                remaining_waypoints_ratio,
                clip_scalar(boundary_soft_risk, 0.0, 2.0),
                1.0 if self._deconfliction_active[idx] else 0.0,
            ]

            neighbor_features: List[float] = []
            dangerous_other_idx: Optional[int] = None
            dangerous_score = -np.inf
            max_pair_conflict_pressure = 0.0
            max_predicted_critical_pressure = 0.0
            weighted_turn_preference = 0.0
            weighted_turn_total = 0.0
            min_caution_time_ratio = 1.0
            neighbors = [
                other_idx
                for other_idx in active_indices
                if other_idx != idx
            ]
            neighbors.sort(key=lambda other_idx: dist_matrix[idx, other_idx])
            for slot in range(self.max_neighbors):
                if slot >= len(neighbors):
                    neighbor_features.extend(
                        [
                            1.0,
                            2.0,
                            1.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                    continue

                other_idx = neighbors[slot]
                other_agent = self.possible_agents[other_idx]
                other = self.aircraft_by_agent[other_agent]
                other_max_turn_rate = max(float(other.dynamics.max_turn_rate), 1e-6)
                rel_x = float(local_pos_cache[other_idx, 0] - local_pos_cache[idx, 0])
                rel_y = float(local_pos_cache[other_idx, 1] - local_pos_cache[idx, 1])
                dist = float(dist_matrix[idx, other_idx])
                rel_bearing = wrap_angle(
                    np.arctan2(rel_x, rel_y) - aircraft.heading
                )
                rel_heading = wrap_angle(other.heading - aircraft.heading)

                own_vx = aircraft.dynamics.cruise_speed * np.sin(aircraft.heading)
                own_vy = aircraft.dynamics.cruise_speed * np.cos(aircraft.heading)
                other_vx = other.dynamics.cruise_speed * np.sin(other.heading)
                other_vy = other.dynamics.cruise_speed * np.cos(other.heading)
                rel_vx = other_vx - own_vx
                rel_vy = other_vy - own_vy
                heading_sin = float(np.sin(aircraft.heading))
                heading_cos = float(np.cos(aircraft.heading))
                forward_offset = float((rel_x * heading_sin) + (rel_y * heading_cos))
                right_offset = float((rel_x * heading_cos) - (rel_y * heading_sin))
                forward_rel_speed = float((rel_vx * heading_sin) + (rel_vy * heading_cos))
                right_rel_speed = float((rel_vx * heading_cos) - (rel_vy * heading_sin))
                closing_speed = float(
                    -((rel_x * rel_vx) + (rel_y * rel_vy)) / max(dist, 1e-6)
                )
                speed_sq = float((rel_vx * rel_vx) + (rel_vy * rel_vy))
                if speed_sq > 1e-6:
                    t_cpa_raw = float(-((rel_x * rel_vx) + (rel_y * rel_vy)) / speed_sq)
                    if t_cpa_raw > 0.0:
                        t_cpa = clip_scalar(t_cpa_raw, 0.0, self.caution_lookahead_time_s)
                        projected_rel_x = rel_x + (rel_vx * t_cpa)
                        projected_rel_y = rel_y + (rel_vy * t_cpa)
                        d_cpa = float(np.hypot(projected_rel_x, projected_rel_y))
                    else:
                        t_cpa = self.caution_lookahead_time_s
                        projected_rel_x = rel_x
                        projected_rel_y = rel_y
                        d_cpa = dist
                else:
                    t_cpa = self.caution_lookahead_time_s
                    projected_rel_x = rel_x
                    projected_rel_y = rel_y
                    d_cpa = dist
                cpa_forward_offset = float(
                    (projected_rel_x * heading_sin) + (projected_rel_y * heading_cos)
                )
                cpa_right_offset = float(
                    (projected_rel_x * heading_cos) - (projected_rel_y * heading_sin)
                )

                current_severity = clip_scalar(
                    (self.caution_dist - dist) / max(self.caution_dist, 1e-6),
                    0.0,
                    1.0,
                )
                if d_cpa < self.caution_dist:
                    urgency = 1.0 - (t_cpa / lookahead_time_scale)
                    predicted_severity = clip_scalar(
                        urgency
                        * (
                            (self.caution_dist - d_cpa)
                            / max(self.caution_dist, 1e-6)
                        ),
                        0.0,
                        1.0,
                    )
                else:
                    predicted_severity = 0.0
                critical_distance_ratio = clip_scalar(
                    d_cpa / critical_dist_scale,
                    0.0,
                    2.0,
                )
                if d_cpa < self.critical_dist:
                    critical_urgency = 1.0 - (t_cpa / lookahead_time_scale)
                    predicted_critical_severity = clip_scalar(
                        critical_urgency
                        * (
                            (self.critical_dist - d_cpa)
                            / max(self.critical_dist, 1e-6)
                        ),
                        0.0,
                        1.0,
                    )
                else:
                    predicted_critical_severity = 0.0
                pair_conflict_pressure = clip_scalar(
                    max(current_severity, self.predicted_caution_weight * predicted_severity),
                    0.0,
                    1.0,
                )
                traffic_vs_waypoint_bearing = wrap_angle(rel_bearing - rel_bearing_wp)
                prev_rel_bearing = float(self._last_rel_bearing_matrix[idx, other_idx])
                if np.isfinite(prev_rel_bearing):
                    bearing_rate = float(
                        wrap_angle(rel_bearing - prev_rel_bearing) / max(self.dt, 1e-6)
                    )
                else:
                    bearing_rate = 0.0
                prev_d_cpa = float(self._last_d_cpa_matrix[idx, other_idx])
                if not np.isfinite(prev_d_cpa):
                    prev_d_cpa = d_cpa
                prev_pair_conflict = float(
                    self._last_pair_conflict_pressure_matrix[idx, other_idx]
                )
                if not np.isfinite(prev_pair_conflict):
                    prev_pair_conflict = pair_conflict_pressure

                current_rel_bearing_matrix[idx, other_idx] = rel_bearing
                current_d_cpa_matrix[idx, other_idx] = d_cpa
                current_pair_conflict_pressure_matrix[idx, other_idx] = pair_conflict_pressure

                danger_score_candidate = max(
                    pair_conflict_pressure,
                    predicted_critical_severity * 1.5,
                    current_severity,
                )
                if danger_score_candidate > dangerous_score:
                    dangerous_score = danger_score_candidate
                    dangerous_other_idx = other_idx

                side_pass_cue = float(self._side_commitment[idx, other_idx])
                if side_pass_cue == 0.0:
                    if abs(rel_bearing) < (np.pi / 6.0):
                        side_pass_cue = 1.0
                    else:
                        side_pass_cue = -float(np.sign(rel_bearing))
                        if side_pass_cue == 0.0:
                            side_pass_cue = 1.0

                max_pair_conflict_pressure = max(
                    max_pair_conflict_pressure,
                    pair_conflict_pressure,
                )
                max_predicted_critical_pressure = max(
                    max_predicted_critical_pressure,
                    predicted_critical_severity,
                )
                min_caution_time_ratio = min(
                    min_caution_time_ratio,
                    clip_scalar(t_cpa / lookahead_time_scale, 0.0, 1.0),
                )
                weighted_turn_preference += pair_conflict_pressure * side_pass_cue
                weighted_turn_total += pair_conflict_pressure

                neighbor_features.extend(
                    [
                        clip_scalar(dist / map_diag_scale, 0.0, 2.0),
                        clip_scalar(d_cpa / caution_dist_scale, 0.0, 2.0),
                        clip_scalar(t_cpa / lookahead_time_scale, 0.0, 1.0),
                        clip_scalar(
                            closing_speed / cruise_speed_scale,
                            -1.0,
                            1.0,
                        ),
                        pair_conflict_pressure,
                        critical_distance_ratio,
                        predicted_critical_severity,
                        side_pass_cue,
                        np.sin(rel_bearing),
                        np.cos(rel_bearing),
                        np.sin(rel_heading),
                        np.cos(rel_heading),
                        clip_scalar(
                            other.dynamics.cruise_speed / cruise_speed_scale,
                            0.0,
                            2.0,
                        ),
                        clip_scalar(
                            other.dynamics.max_turn_rate / max_turn_rate_scale,
                            0.0,
                            2.0,
                        ),
                        clip_scalar(
                            other.actual_turn_rate / other_max_turn_rate,
                            -1.0,
                            1.0,
                        ),
                        clip_scalar(forward_offset / map_diag_scale, -2.0, 2.0),
                        clip_scalar(right_offset / map_diag_scale, -2.0, 2.0),
                        clip_scalar(
                            forward_rel_speed / cruise_speed_scale,
                            -2.0,
                            2.0,
                        ),
                        clip_scalar(
                            right_rel_speed / cruise_speed_scale,
                            -2.0,
                            2.0,
                        ),
                        clip_scalar(cpa_forward_offset / map_diag_scale, -2.0, 2.0),
                        clip_scalar(cpa_right_offset / map_diag_scale, -2.0, 2.0),
                        np.sin(traffic_vs_waypoint_bearing),
                        np.cos(traffic_vs_waypoint_bearing),
                        clip_scalar(
                            bearing_rate / max(max_turn_rate, 1e-6),
                            -2.0,
                            2.0,
                        ),
                        clip_scalar(prev_d_cpa / caution_dist_scale, 0.0, 2.0),
                        clip_scalar(prev_pair_conflict, 0.0, 1.0),
                        float(self._last_commanded_action_vector[other_idx]),
                        float(self._last_reference_action_vector[other_idx]),
                        1.0 if self._deconfliction_active[other_idx] else 0.0,
                        1.0 if abs(self._side_commitment[idx, other_idx]) > 0.0 else 0.0,
                    ]
                )

            if weighted_turn_total > 1e-6:
                dominant_turn_preference = float(
                    np.clip(weighted_turn_preference / weighted_turn_total, -1.0, 1.0)
                )
            else:
                dominant_turn_preference = 0.0

            self_features.extend(
                [
                    min_caution_time_ratio,
                    clip_scalar(max_pair_conflict_pressure, 0.0, 1.0),
                    clip_scalar(max_predicted_critical_pressure, 0.0, 1.0),
                    dominant_turn_preference,
                ]
            )
            if dangerous_other_idx is None:
                left_sep_improvement, right_sep_improvement = 0.0, 0.0
            else:
                other = self.aircraft_by_agent[self.possible_agents[dangerous_other_idx]]
                left_sep_improvement, right_sep_improvement = (
                    dangerous_neighbor_turn_preview(
                        own_pos_local=local_pos_cache[idx],
                        other_pos_local=local_pos_cache[dangerous_other_idx],
                        own_heading=float(aircraft.heading),
                        other_heading=float(other.heading),
                        own_cruise_speed=float(aircraft.dynamics.cruise_speed),
                        other_cruise_speed=float(other.dynamics.cruise_speed),
                        own_max_turn_rate=float(aircraft.dynamics.max_turn_rate),
                        caution_dist=float(self.caution_dist),
                        dt=self.dt,
                    )
                )
            feature_vector = (
                self_features
                + [
                    left_sep_improvement,
                    right_sep_improvement,
                ]
                + neighbor_features
            )
            if len(feature_vector) != self.base_obs_dim:
                raise ValueError(
                    "Observation feature length mismatch: "
                    f"expected {self.base_obs_dim}, got {len(feature_vector)}"
                )
            self._base_obs_cache[idx] = np.asarray(feature_vector, dtype=np.float32)

        self._last_rel_bearing_matrix[:] = current_rel_bearing_matrix
        self._last_d_cpa_matrix[:] = current_d_cpa_matrix
        self._last_pair_conflict_pressure_matrix[:] = current_pair_conflict_pressure_matrix
        self._append_obs_history(fill_history=fill_history)

    def _update_deconfliction_state(
        self,
        active_agents: List[str],
        pair_metrics: dict,
        *,
        was_active: np.ndarray,
    ) -> None:
        caution_pressure = np.asarray(
            pair_metrics["agent_caution_pressure"],
            dtype=np.float32,
        )

        for agent in active_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            if (
                aircraft.waypoint_manager.current_waypoint is None
                or aircraft.flight_mode == FlightMode.LOITERING
            ):
                self._deconfliction_active[idx] = False
                self._deconfliction_hold_remaining[idx] = 0
                continue

            within_caution = bool(float(caution_pressure[idx]) > 0.0)

            if within_caution:
                self._deconfliction_active[idx] = True
                self._deconfliction_hold_remaining[idx] = (
                    self.guidance_deconfliction_hold_steps
                )
            elif self._deconfliction_active[idx]:
                if self._deconfliction_hold_remaining[idx] > 0:
                    self._deconfliction_hold_remaining[idx] -= 1
                    self._deconfliction_active[idx] = True
                else:
                    self._deconfliction_active[idx] = False
            else:
                self._deconfliction_hold_remaining[idx] = 0

            if self._deconfliction_active[idx]:
                self._deconfliction_steps_total[idx] += 1

            if was_active[idx] and not self._deconfliction_active[idx]:
                self._segment_start_local[idx] = self._local_pos_cache[idx]
                self._segment_start_heading[idx] = float(aircraft.heading)
                self._segment_path_length[idx] = 0.0
                self._route_progress_anchor[idx] = 0.0
                self._reference_route_cache[idx] = None

    def _evaluate_pairwise_safety(self, active_agents: Iterable[str]) -> dict:
        agents = list(active_agents)
        critical_pairs = []
        caution_pairs = []
        separation_margin_penalties = []
        agent_caution_pressure = np.zeros(self.max_agents, dtype=np.float32)
        agent_conflict_pressure = np.zeros(self.max_agents, dtype=np.float32)
        agent_critical_pressure = np.zeros(self.max_agents, dtype=np.float32)
        parallel_conflict_pressure = np.zeros(self.max_agents, dtype=np.float32)
        preferred_turn_score = np.zeros(self.max_agents, dtype=np.float32)
        preferred_turn_weight = np.zeros(self.max_agents, dtype=np.float32)
        separation_margin_dist = max(
            self.caution_dist * self.separation_margin_ratio,
            self.caution_dist + 1.0,
        )
        separation_margin_span = max(
            separation_margin_dist - self.caution_dist,
            1e-6,
        )
        critical_buffer_dist = min(
            self.caution_dist,
            max(self.critical_dist * 2.5, self.critical_dist + 20.0),
        )
        critical_buffer_span = max(
            critical_buffer_dist - self.critical_dist,
            1e-6,
        )
        if len(agents) >= 2:
            for agent_a, agent_b in combinations(agents, 2):
                idx_a = self.agent_name_to_index[agent_a]
                idx_b = self.agent_name_to_index[agent_b]
                dist = float(self._dist_matrix[idx_a, idx_b])
                if dist < self._min_pairwise_distance:
                    self._min_pairwise_distance = dist
                    self._min_pairwise_pair = (agent_a, agent_b)
                    self._min_pairwise_step = self.current_step

                current_severity = 0.0
                if dist < self.caution_dist:
                    current_severity = float(
                        np.clip(
                            (self.caution_dist - dist) / max(self.caution_dist, 1e-6),
                            0.0,
                            1.0,
                        )
                    )
                    caution_pairs.append((agent_a, agent_b))
                if self.caution_dist <= dist < separation_margin_dist:
                    separation_margin_penalties.append(
                        float(
                            np.clip(
                                (separation_margin_dist - dist) / separation_margin_span,
                                0.0,
                                1.0,
                            )
                            ** 2
                        )
                    )

                aircraft_a = self.aircraft_by_agent[agent_a]
                aircraft_b = self.aircraft_by_agent[agent_b]
                own_vx = aircraft_a.dynamics.cruise_speed * np.sin(aircraft_a.heading)
                own_vy = aircraft_a.dynamics.cruise_speed * np.cos(aircraft_a.heading)
                other_vx = aircraft_b.dynamics.cruise_speed * np.sin(aircraft_b.heading)
                other_vy = aircraft_b.dynamics.cruise_speed * np.cos(aircraft_b.heading)
                rel_position = np.asarray(
                    [
                        float(self._local_pos_cache[idx_b, 0] - self._local_pos_cache[idx_a, 0]),
                        float(self._local_pos_cache[idx_b, 1] - self._local_pos_cache[idx_a, 1]),
                    ],
                    dtype=np.float32,
                )
                rel_velocity = np.asarray(
                    [other_vx - own_vx, other_vy - own_vy],
                    dtype=np.float32,
                )
                rel_heading = float(
                    abs(
                        np.arctan2(
                            np.sin(aircraft_b.heading - aircraft_a.heading),
                            np.cos(aircraft_b.heading - aircraft_a.heading),
                        )
                    )
                )
                head_on_factor = float(
                    np.clip(rel_heading / np.pi, 0.0, 1.0)
                )
                parallel_factor = float(
                    np.clip(np.cos(rel_heading), 0.0, 1.0)
                )
                speed_sq = float(np.dot(rel_velocity, rel_velocity))
                predicted_severity = 0.0
                predicted_critical_pressure = 0.0
                current_critical_pressure = 0.0
                if dist < critical_buffer_dist:
                    current_critical_pressure = float(
                        np.clip(
                            (critical_buffer_dist - dist) / critical_buffer_span,
                            0.0,
                            1.0,
                        )
                    )
                if speed_sq > 1e-6:
                    t_cpa = float(
                        np.clip(
                            -np.dot(rel_position, rel_velocity) / speed_sq,
                            0.0,
                            self.caution_lookahead_time_s,
                        )
                    )
                    if t_cpa > 0.0:
                        cpa_distance = float(
                            np.linalg.norm(rel_position + rel_velocity * t_cpa)
                        )
                        if cpa_distance < self.caution_dist:
                            urgency = 1.0 - (t_cpa / max(self.caution_lookahead_time_s, 1e-6))
                            predicted_severity = float(
                                np.clip(
                                    urgency
                                    * (
                                        (self.caution_dist - cpa_distance)
                                        / max(self.caution_dist, 1e-6)
                                    ),
                                    0.0,
                                    1.0,
                                )
                            )
                        if cpa_distance < critical_buffer_dist:
                            critical_urgency = 1.0 - (
                                t_cpa / max(self.caution_lookahead_time_s, 1e-6)
                            )
                            predicted_critical_pressure = float(
                                np.clip(
                                    critical_urgency
                                    * (
                                        (critical_buffer_dist - cpa_distance)
                                        / critical_buffer_span
                                    ),
                                    0.0,
                                    1.0,
                                )
                            )
                conflict_pressure = float(
                    np.clip(
                        max(current_severity, self.predicted_caution_weight * predicted_severity)
                        * (
                            1.0
                            + self.head_on_conflict_weight * head_on_factor
                        ),
                        0.0,
                        1.0,
                    )
                )
                critical_pressure = float(
                    np.clip(
                        max(current_critical_pressure, predicted_critical_pressure)
                        * (1.0 + (0.5 * head_on_factor)),
                        0.0,
                        1.0,
                    )
                )
                rel_bearing_a = wrap_angle(
                    np.arctan2(rel_position[0], rel_position[1]) - aircraft_a.heading
                )
                if abs(rel_bearing_a) < (np.pi / 6.0):
                    raw_turn_pref_a = 1.0
                else:
                    raw_turn_pref_a = -np.sign(rel_bearing_a)
                    if raw_turn_pref_a == 0.0:
                        raw_turn_pref_a = 1.0

                rel_bearing_b = wrap_angle(
                    np.arctan2(-rel_position[0], -rel_position[1]) - aircraft_b.heading
                )
                if abs(rel_bearing_b) < (np.pi / 6.0):
                    raw_turn_pref_b = 1.0
                else:
                    raw_turn_pref_b = -np.sign(rel_bearing_b)
                    if raw_turn_pref_b == 0.0:
                        raw_turn_pref_b = 1.0

                committed_turn_a = float(self._side_commitment[idx_a, idx_b])
                committed_turn_b = float(self._side_commitment[idx_b, idx_a])
                commit_release_dist = self.caution_dist * self.conflict_commit_release_scale
                release_commitment = (
                    dist >= commit_release_dist
                    and conflict_pressure <= self.conflict_commit_release_pressure
                )
                if release_commitment:
                    self._side_commitment[idx_a, idx_b] = 0.0
                    self._side_commitment[idx_b, idx_a] = 0.0
                    committed_turn_a = 0.0
                    committed_turn_b = 0.0
                if (
                    abs(committed_turn_a) > 0.0
                    and abs(committed_turn_b) > 0.0
                    and dist < commit_release_dist
                ):
                    turn_pref_a = committed_turn_a
                    turn_pref_b = committed_turn_b
                else:
                    turn_pref_a = raw_turn_pref_a
                    turn_pref_b = raw_turn_pref_b
                    if conflict_pressure >= self.conflict_commit_activation:
                        self._side_commitment[idx_a, idx_b] = float(turn_pref_a)
                        self._side_commitment[idx_b, idx_a] = float(turn_pref_b)

                if max(conflict_pressure, critical_pressure) > 0.0:
                    agent_caution_pressure[idx_a] = max(
                        agent_caution_pressure[idx_a],
                        float(current_severity),
                    )
                    agent_caution_pressure[idx_b] = max(
                        agent_caution_pressure[idx_b],
                        float(current_severity),
                    )
                    agent_conflict_pressure[idx_a] = max(
                        agent_conflict_pressure[idx_a],
                        float(conflict_pressure),
                    )
                    agent_conflict_pressure[idx_b] = max(
                        agent_conflict_pressure[idx_b],
                        float(conflict_pressure),
                    )
                    agent_critical_pressure[idx_a] = max(
                        agent_critical_pressure[idx_a],
                        float(critical_pressure),
                    )
                    agent_critical_pressure[idx_b] = max(
                        agent_critical_pressure[idx_b],
                        float(critical_pressure),
                    )
                    parallel_pressure = float(conflict_pressure * parallel_factor)
                    parallel_conflict_pressure[idx_a] = max(
                        parallel_conflict_pressure[idx_a],
                        parallel_pressure,
                    )
                    parallel_conflict_pressure[idx_b] = max(
                        parallel_conflict_pressure[idx_b],
                        parallel_pressure,
                    )
                    preferred_turn_score[idx_a] += float(conflict_pressure * turn_pref_a)
                    preferred_turn_weight[idx_a] += float(conflict_pressure)
                    preferred_turn_score[idx_b] += float(conflict_pressure * turn_pref_b)
                    preferred_turn_weight[idx_b] += float(conflict_pressure)

                if dist < self.critical_dist:
                    critical_pairs.append((agent_a, agent_b))

        self.caution_dist_breakers.extend(caution_pairs)
        self.crit_dist_breakers.extend(critical_pairs)
        if critical_pairs:
            self._crashed = True

        preferred_turn = np.zeros(self.max_agents, dtype=np.float32)
        active_pref = preferred_turn_weight > 0.0
        preferred_turn[active_pref] = np.sign(preferred_turn_score[active_pref])
        neutral_pref = active_pref & (preferred_turn == 0.0)
        preferred_turn[neutral_pref] = 1.0

        return {
            "critical_pairs": critical_pairs,
            "caution_pairs": caution_pairs,
            "separation_margin_penalty": (
                float(np.mean(separation_margin_penalties))
                if separation_margin_penalties
                else 0.0
            ),
            "agent_caution_pressure": agent_caution_pressure,
            "agent_conflict_pressure": agent_conflict_pressure,
            "agent_critical_pressure": agent_critical_pressure,
            "parallel_conflict_pressure": parallel_conflict_pressure,
            "preferred_turn": preferred_turn,
        }

    def _boundary_status(self, agent: str) -> Tuple[bool, float, float]:
        idx = self.agent_name_to_index[agent]
        x_pos = float(self._local_pos_cache[idx, 0])
        y_pos = float(self._local_pos_cache[idx, 1])

        inside = (
            self.local_min_x <= x_pos <= self.local_max_x
            and self.local_min_y <= y_pos <= self.local_max_y
        )
        distances = np.asarray(
            [
                y_pos - self.local_min_y,
                self.local_max_y - y_pos,
                x_pos - self.local_min_x,
                self.local_max_x - x_pos,
            ],
            dtype=np.float32,
        )
        min_inside_distance = float(np.min(distances))
        buffer_dist = self.boundary_buffer_ratio * min(self.box_width_m, self.box_height_m)
        if min_inside_distance >= buffer_dist:
            soft_risk = 0.0
        else:
            soft_risk = float(np.clip((buffer_dist - min_inside_distance) / max(buffer_dist, 1.0), 0.0, 2.0))

        outside_depth = 0.0
        if not inside:
            outside_x = max(self.local_min_x - x_pos, 0.0, x_pos - self.local_max_x)
            outside_y = max(self.local_min_y - y_pos, 0.0, y_pos - self.local_max_y)
            outside_depth = float(np.hypot(outside_x, outside_y) / max(min(self.box_width_m, self.box_height_m), 1.0))

        return inside, soft_risk, outside_depth

    def _compute_team_reward(
        self,
        *,
        current_agents: List[str],
        waypoint_hits: int,
        waypoint_proximities: List[float],
        progress_terms: List[float],
        circling_scores: List[float],
        stagnation_scores: List[float],
        boundary_soft_risks: List[float],
        outside_depths: List[float],
        pair_metrics: dict,
        completed: bool,
        truncated: bool,
    ) -> float:
        waypoint_term = self.reward_waypoint_hit * float(waypoint_hits / max(len(current_agents), 1))
        completion_term = self.reward_completion_bonus if completed else 0.0
        proximity_term = self.reward_waypoint_proximity_bonus * float(
            np.mean(np.asarray(waypoint_proximities, dtype=np.float32))
            if waypoint_proximities
            else 0.0
        )
        progress_term = self.reward_progress * float(
            np.mean(np.asarray(progress_terms, dtype=np.float32))
            if progress_terms
            else 0.0
        )
        geofence_term = -self.penalty_geofence * float(
            np.mean(
                [
                    min(max(depth, 0.0), self.geofence_depth_cap)
                    ** self.geofence_growth_exponent
                    for depth in outside_depths
                ]
                if outside_depths
                else 0.0
            )
        )
        boundary_soft_term = -self.penalty_boundary_soft * float(
            np.mean(
                [
                    max(risk, 0.0) ** self.boundary_soft_growth_exponent
                    for risk in boundary_soft_risks
                ]
                if boundary_soft_risks
                else 0.0
            )
        )
        harsh_turn_terms = []
        caution_pressure = np.asarray(
            pair_metrics["agent_caution_pressure"],
            dtype=np.float32,
        )
        for agent in current_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            if (
                aircraft.waypoint_manager.current_waypoint is None
                or aircraft.flight_mode == FlightMode.LOITERING
            ):
                harsh_turn_terms.append(0.0)
                continue
            max_turn_rate = max(float(aircraft.dynamics.max_turn_rate), 1e-6)
            turn_fraction = abs(float(aircraft.actual_turn_rate)) / max_turn_rate
            harsh_excess = max(turn_fraction - self.harsh_turn_threshold, 0.0)
            dist_to_wp = float(
                np.linalg.norm(self._wp_local_cache[idx] - self._local_pos_cache[idx])
            )
            near_waypoint_capture = bool(
                dist_to_wp <= (2.0 * float(aircraft.waypoint_manager.arrival_threshold))
            )
            if self._deconfliction_active[idx] or float(caution_pressure[idx]) > 0.0:
                safe_weight = 0.0
            elif near_waypoint_capture:
                safe_weight = 0.0
            else:
                safe_weight = 1.0
            harsh_turn_terms.append(safe_weight * (harsh_excess ** 2))
        harsh_turn_term = -self.penalty_harsh_turn * float(
            np.mean(np.asarray(harsh_turn_terms, dtype=np.float32))
            if harsh_turn_terms
            else 0.0
        )
        circling_term = -self.penalty_circling * float(
            np.mean(np.asarray(circling_scores, dtype=np.float32))
            if circling_scores
            else 0.0
        )
        stagnation_term = -self.penalty_stagnation * float(
            np.mean(np.asarray(stagnation_scores, dtype=np.float32))
            if stagnation_scores
            else 0.0
        )
        separation_margin_term = -self.penalty_separation_margin * float(
            pair_metrics["separation_margin_penalty"]
        )
        crash_term = -self.penalty_crash if pair_metrics["critical_pairs"] else 0.0
        active_navigation_fraction = float(
            np.mean(
                [
                    1.0
                    if (
                        self.aircraft_by_agent[agent].waypoint_manager.current_waypoint is not None
                        and self.aircraft_by_agent[agent].flight_mode != FlightMode.LOITERING
                        and not self._deconfliction_active[self.agent_name_to_index[agent]]
                    )
                    else 0.0
                    for agent in current_agents
                ]
            )
            if current_agents
            else 0.0
        )
        existence_term = -self.penalty_existence * active_navigation_fraction
        if truncated and not completed:
            incomplete_fraction = float(
                np.mean(
                    [
                        (
                            self.aircraft_by_agent[agent].waypoint_manager.queue_size()
                            + (
                                1
                                if self.aircraft_by_agent[agent].waypoint_manager.current_waypoint
                                is not None
                                else 0
                            )
                        )
                        / max(
                            len(self.aircraft_by_agent[agent].waypoint_manager.hit_waypoints)
                            + self.aircraft_by_agent[agent].waypoint_manager.queue_size()
                            + (
                                1
                                if self.aircraft_by_agent[agent].waypoint_manager.current_waypoint
                                is not None
                                else 0
                            ),
                            1,
                        )
                        for agent in current_agents
                    ]
                )
                if current_agents
                else 0.0
            )
        else:
            incomplete_fraction = 0.0
        incomplete_term = -self.penalty_incomplete_mission * incomplete_fraction

        components = {
            "waypoint_reward": waypoint_term,
            "completion_bonus": completion_term,
            "waypoint_proximity_bonus": proximity_term,
            "progress_reward": progress_term,
            "geofence_penalty": geofence_term,
            "boundary_soft_penalty": boundary_soft_term,
            "crash_penalty": crash_term,
            "harsh_turn_penalty": harsh_turn_term,
            "circling_penalty": circling_term,
            "stagnation_penalty": stagnation_term,
            "separation_margin_penalty": separation_margin_term,
            "existence_penalty": existence_term,
            "incomplete_mission_penalty": incomplete_term,
        }
        for name, value in components.items():
            self._reward_totals[name] += float(value)
        return float(sum(components.values()))

    def _check_line_segment_arrival_local(
        self,
        start: np.ndarray,
        end: np.ndarray,
        point: np.ndarray,
        radius: float,
        *,
        start_heading: Optional[float] = None,
        turn_rate: Optional[float] = None,
        cruise_speed: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> bool:
        segment = end - start
        segment_norm_sq = float(np.dot(segment, segment))
        radius = float(radius)

        if (
            float(np.linalg.norm(point - start)) <= radius
            or float(np.linalg.norm(point - end)) <= radius
        ):
            return True

        if (
            start_heading is not None
            and turn_rate is not None
            and cruise_speed is not None
            and dt is not None
        ):
            turn_rate = float(turn_rate)
            cruise_speed = float(cruise_speed)
            dt = float(dt)
            turn_amount = turn_rate * dt
            if abs(turn_amount) > 1e-5 and cruise_speed > 1e-6:
                turn_radius = cruise_speed / max(abs(turn_rate), 1e-6)
                turn_sign = 1.0 if turn_rate > 0.0 else -1.0
                center = np.asarray(
                    [
                        float(start[0]) - (turn_sign * turn_radius * np.sin(start_heading)),
                        float(start[1]) + (turn_sign * turn_radius * np.cos(start_heading)),
                    ],
                    dtype=np.float64,
                )
                start_angle = float(np.arctan2(start[1] - center[1], start[0] - center[0]))
                point_angle = float(np.arctan2(point[1] - center[1], point[0] - center[0]))
                point_radius = float(np.linalg.norm(point - center))

                if turn_rate > 0.0:
                    swept_angle = float((point_angle - start_angle) % (2.0 * np.pi))
                    sweep_limit = min(float(turn_amount), 2.0 * np.pi)
                else:
                    swept_angle = float((start_angle - point_angle) % (2.0 * np.pi))
                    sweep_limit = min(float(-turn_amount), 2.0 * np.pi)

                if swept_angle <= sweep_limit and abs(point_radius - turn_radius) <= radius:
                    return True

        if segment_norm_sq <= 1e-9:
            return False

        start_to_point = point - start
        t = np.clip(
            np.dot(start_to_point, segment) / (segment_norm_sq + 1e-9),
            0.0,
            1.0,
        )
        closest = start + t * segment
        return bool(float(np.linalg.norm(point - closest)) <= radius)

    def _shared_info(
        self,
        *,
        waypoints_hit: int = 0,
    ) -> dict:
        cumulative_waypoints_hit = int(
            sum(
                len(aircraft.waypoint_manager.hit_waypoints)
                for aircraft in self.aircraft_by_agent.values()
            )
        )
        active_indices = [
            self.agent_name_to_index[agent]
            for agent in self.agents
        ]
        policy_trainable_fraction = (
            float(np.mean(self._last_policy_train_mask[active_indices]))
            if active_indices
            else 0.0
        )
        info = {
            "waypoints_hit": cumulative_waypoints_hit,
            "waypoints_hit_step": int(waypoints_hit),
            "crashed": bool(self._crashed),
            "terminated": bool(
                self._crashed
                or self._termination_reason == "completed"
            ),
            "truncated": bool(self.current_step >= self.max_steps),
            "termination_reason": self._termination_reason,
            "all_waypoints_completed": bool(
                all(
                    not aircraft.waypoint_manager.has_waypoints()
                    for aircraft in self.aircraft_by_agent.values()
                )
            ),
            "policy_trainable_fraction": policy_trainable_fraction,
            "sim_time_s": float(self.current_step * self.dt),
            "max_steps": int(self.max_steps),
            "base_max_steps": int(self.base_max_steps),
            "timeout_max_route_distance_m": float(
                self._episode_timeout_max_route_distance_m
            ),
            "team_reward_total": float(self._episode_reward_total),
            "state": self.state(),
        }
        if self.allow_live_waypoint_updates:
            info["agent_states"] = self.runtime_agent_snapshots()
        return info

    def get_episode_metrics(self) -> dict:
        def summarize_pairs(pairs: List[Tuple[str, str]]):
            counts = Counter(tuple(sorted(pair)) for pair in pairs)
            return [
                {"pair": list(pair), "count": count}
                for pair, count in sorted(counts.items())
            ]

        def motion_metrics_for(aircraft: FixedWingAircraft) -> dict:
            path = aircraft.path_history
            if len(path) < 2:
                return {
                    "net_displacement_efficiency": 1.0,
                    "mean_abs_turn_rate_deg_s": 0.0,
                    "max_abs_turn_rate_deg_s": 0.0,
                    "mean_heading_jerk_deg_s2": 0.0,
                    "turn_reversal_rate": 0.0,
                }

            lats = [point.latitude for point in path]
            lons = [point.longitude for point in path]
            x_vals, y_vals = self.transformer.geo_to_local(lats, lons)
            coords = np.column_stack([x_vals, y_vals]).astype(np.float32)
            deltas = np.diff(coords, axis=0)
            segment_lengths = np.linalg.norm(deltas, axis=1)
            valid_mask = segment_lengths > 1e-6
            if not np.any(valid_mask):
                return {
                    "net_displacement_efficiency": 0.0,
                    "mean_abs_turn_rate_deg_s": 0.0,
                    "max_abs_turn_rate_deg_s": 0.0,
                    "mean_heading_jerk_deg_s2": 0.0,
                    "turn_reversal_rate": 0.0,
                }

            headings = np.arctan2(deltas[valid_mask, 0], deltas[valid_mask, 1])
            displacement = float(np.linalg.norm(coords[-1] - coords[0]))
            net_displacement_efficiency = float(
                np.clip(
                    displacement / max(float(aircraft.distance_traveled), 1e-6),
                    0.0,
                    1.0,
                )
            )
            if headings.size < 2:
                return {
                    "net_displacement_efficiency": net_displacement_efficiency,
                    "mean_abs_turn_rate_deg_s": 0.0,
                    "max_abs_turn_rate_deg_s": 0.0,
                    "mean_heading_jerk_deg_s2": 0.0,
                    "turn_reversal_rate": 0.0,
                }

            heading_delta = np.asarray(
                [wrap_angle(curr - prev) for prev, curr in zip(headings[:-1], headings[1:])],
                dtype=np.float32,
            )
            turn_rates = heading_delta / max(self.dt, 1e-6)
            abs_turn_rates_deg = np.degrees(np.abs(turn_rates))
            if turn_rates.size >= 2:
                heading_jerk_deg = np.degrees(np.abs(np.diff(turn_rates))) / max(self.dt, 1e-6)
            else:
                heading_jerk_deg = np.zeros((0,), dtype=np.float32)

            active_turn_mask = np.abs(turn_rates) > np.deg2rad(1.0)
            active_turn_rates = turn_rates[active_turn_mask]
            if active_turn_rates.size >= 2:
                turn_reversals = (
                    np.sign(active_turn_rates[:-1]) * np.sign(active_turn_rates[1:]) < 0
                )
                turn_reversal_rate = float(np.mean(turn_reversals))
            else:
                turn_reversal_rate = 0.0

            return {
                "net_displacement_efficiency": net_displacement_efficiency,
                "mean_abs_turn_rate_deg_s": float(abs_turn_rates_deg.mean()),
                "max_abs_turn_rate_deg_s": float(abs_turn_rates_deg.max()),
                "mean_heading_jerk_deg_s2": (
                    float(heading_jerk_deg.mean()) if heading_jerk_deg.size else 0.0
                ),
                "turn_reversal_rate": turn_reversal_rate,
            }

        telemetry = []
        mission_stats = []
        reward_per_uav = []

        for agent in self.possible_agents:
            if agent not in self.aircraft_by_agent:
                continue
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            reached = len(aircraft.waypoint_manager.hit_waypoints)
            remaining = (
                aircraft.waypoint_manager.queue_size()
                + (1 if aircraft.waypoint_manager.current_waypoint else 0)
            )
            assigned = reached + remaining
            motion_metrics = motion_metrics_for(aircraft)
            telemetry.append(
                {
                    "id": agent,
                    "pos": aircraft.position.to_tuple(),
                    "speed": float(aircraft.dynamics.cruise_speed),
                    "heading": float(aircraft.heading),
                    "mode": aircraft.flight_mode.value,
                }
            )
            mission_stats.append(
                {
                    "id": agent,
                    "waypoints_reached": int(reached),
                    "assigned_waypoints": int(assigned),
                    "waypoints_remaining": int(remaining),
                    "completion_rate": float(reached / assigned) if assigned else 1.0,
                    "completed_mission": remaining == 0,
                    "completion_step": self._completion_steps[idx],
                    "completion_time_s": (
                        float(self._completion_steps[idx] * self.dt)
                        if self._completion_steps[idx] is not None
                        else None
                    ),
                    "dist_navigating": float(aircraft.distance_traveled),
                    "initial_position": aircraft.initial_pos.to_tuple(),
                    "final_position": aircraft.position.to_tuple(),
                    "initial_heading": float(aircraft.initial_heading),
                    "final_heading": float(aircraft.heading),
                    "cruise_speed_mps": float(aircraft.dynamics.cruise_speed),
                    "turning_radius_m": float(aircraft.dynamics.turning_radius),
                    "geofence_exits": int(self.geofence_exit_counts[idx]),
                    "deconfliction_steps": int(self._deconfliction_steps_total[idx]),
                    "circling_steps": int(self._circling_steps_total[idx]),
                    "circling_breakouts": int(self._circling_breakouts_total[idx]),
                    "waypoint_reapproach_steps": int(
                        self._waypoint_reapproach_steps_total[idx]
                    ),
                    "waypoint_reapproach_events": int(
                        self._waypoint_reapproach_events_total[idx]
                    ),
                    **motion_metrics,
                }
            )
            reward_per_uav.append(
                {
                    "id": agent,
                    "waypoints_reached": int(reached),
                }
            )

        waypoint_completion_rate = (
            float(
                np.mean([entry["completion_rate"] for entry in mission_stats])
            )
            if mission_stats
            else 0.0
        )
        team_waypoints_reached = int(
            sum(entry["waypoints_reached"] for entry in mission_stats)
        )
        team_waypoints_remaining = int(
            sum(entry["waypoints_remaining"] for entry in mission_stats)
        )
        sim_time_s = float(self.current_step * self.dt)
        waypoint_throughput_per_min = float(
            team_waypoints_reached / max(sim_time_s / 60.0, 1e-6)
        )
        avg_mean_abs_turn_rate = (
            float(np.mean([entry["mean_abs_turn_rate_deg_s"] for entry in mission_stats]))
            if mission_stats
            else 0.0
        )
        avg_heading_jerk = (
            float(np.mean([entry["mean_heading_jerk_deg_s2"] for entry in mission_stats]))
            if mission_stats
            else 0.0
        )
        avg_turn_reversal_rate = (
            float(np.mean([entry["turn_reversal_rate"] for entry in mission_stats]))
            if mission_stats
            else 0.0
        )
        deconfliction_steps_total = int(np.sum(self._deconfliction_steps_total))
        circling_steps_total = int(np.sum(self._circling_steps_total))
        circling_breakouts_total = int(np.sum(self._circling_breakouts_total))
        waypoint_reapproach_steps_total = int(
            np.sum(self._waypoint_reapproach_steps_total)
        )
        waypoint_reapproach_events_total = int(
            np.sum(self._waypoint_reapproach_events_total)
        )

        return {
            "telemetry": telemetry,
            "mission_stats": mission_stats,
            "episode_summary": {
                "steps": int(self.current_step),
                "max_steps": int(self.max_steps),
                "base_max_steps": int(self.base_max_steps),
                "timeout_scaled": bool(self.max_steps != self.base_max_steps),
                "timeout_assigned_waypoints": int(self._episode_timeout_assigned_waypoints),
                "timeout_reference_waypoints": int(
                    self._episode_timeout_reference_waypoints
                ),
                "timeout_max_route_distance_m": float(
                    self._episode_timeout_max_route_distance_m
                ),
                "timeout_reference_route_distance_m": float(
                    self._episode_timeout_reference_route_distance_m
                ),
                "sim_time_s": sim_time_s,
                "termination_reason": self._termination_reason,
                "uavs_completed": int(
                    sum(1 for step in self._completion_steps if step is not None)
                ),
                "all_waypoints_completed": all(
                    not aircraft.waypoint_manager.has_waypoints()
                    for aircraft in self.aircraft_by_agent.values()
                ),
                "min_pairwise_distance_m": (
                    float(self._min_pairwise_distance)
                    if self._min_pairwise_pair is not None
                    else None
                ),
                "min_pairwise_pair": (
                    list(self._min_pairwise_pair)
                    if self._min_pairwise_pair is not None
                    else None
                ),
                "min_pairwise_step": self._min_pairwise_step,
                "waypoint_completion_rate": waypoint_completion_rate,
                "team_waypoints_reached": team_waypoints_reached,
                "team_waypoints_remaining": team_waypoints_remaining,
                "waypoint_throughput_per_min": waypoint_throughput_per_min,
                "avg_mean_abs_turn_rate_deg_s": avg_mean_abs_turn_rate,
                "avg_heading_jerk_deg_s2": avg_heading_jerk,
                "avg_turn_reversal_rate": avg_turn_reversal_rate,
                "deconfliction_steps_total": deconfliction_steps_total,
                "deconfliction_time_s": float(deconfliction_steps_total * self.dt),
                "circling_steps_total": circling_steps_total,
                "circling_breakouts_total": circling_breakouts_total,
                "waypoint_reapproach_steps_total": waypoint_reapproach_steps_total,
                "waypoint_reapproach_events_total": waypoint_reapproach_events_total,
                "team_reward_total": float(self._episode_reward_total),
            },
            "reward_breakdown": {
                "team": {name: float(value) for name, value in self._reward_totals.items()},
                "net_total": float(sum(self._reward_totals.values())),
                "per_uav": reward_per_uav,
            },
            "safety_violations": {
                "caution": {
                    "total_count": len(self.caution_dist_breakers),
                    "pairs": self.caution_dist_breakers,
                    "pair_counts": summarize_pairs(self.caution_dist_breakers),
                },
                "critical": {
                    "total_count": len(self.crit_dist_breakers),
                    "pairs": self.crit_dist_breakers,
                    "pair_counts": summarize_pairs(self.crit_dist_breakers),
                },
                "geofence": {
                    "total_count": int(sum(self.geofence_exit_counts)),
                    "counts": list(self.geofence_exit_counts),
                    "outside_step_total": int(sum(self.geofence_outside_steps)),
                    "outside_step_counts": list(self.geofence_outside_steps),
                },
            },
        }
