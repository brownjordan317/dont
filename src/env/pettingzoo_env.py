from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

try:
    from pettingzoo.utils.env import ParallelEnv
except ModuleNotFoundError:
    class ParallelEnv:  # pragma: no cover - lightweight fallback for local runs
        metadata = {}

from flight_engine.helpers import FlightMode, Position, wrap_angle
from flight_engine.reference_guidance import ReferenceRoute
from flight_engine.simulator import FixedWingAircraft
from flight_engine.trans_coorders import CoordinateTransformer

from env.episode_info import EpisodeInfoMixin
from env.observation_guidance import ObservationGuidanceMixin
from env.runtime_waypoints import RuntimeWaypointMixin
from env.safety_rewards import SafetyRewardMixin
from env.scenario_generation import ScenarioGenerationMixin


class MultiUAVParallelEnv(
    RuntimeWaypointMixin,
    ScenarioGenerationMixin,
    ObservationGuidanceMixin,
    SafetyRewardMixin,
    EpisodeInfoMixin,
    ParallelEnv,
):
    metadata = {
        "name": "dont_multi_uav_parallel_v0",
        "render_modes": [],
        "is_parallelizable": True,
    }
    SELF_FEATURE_NAMES = (
        "waypoint_distance",
        "waypoint_bearing_sin",
        "waypoint_bearing_cos",
        "next_leg_turn_sin",
        "next_leg_turn_cos",
        "has_next_waypoint",
        "heading_sin",
        "heading_cos",
        "cruise_speed",
        "max_turn_rate",
        "actual_turn_rate",
        "desired_turn_rate",
        "turn_rate_error",
        "route_reference_action",
        "last_action",
        "last_action_delta",
        "boundary_distance_south",
        "boundary_distance_north",
        "boundary_distance_west",
        "boundary_distance_east",
        "boundary_time_ahead",
        "boundary_left_turn_feasibility",
        "boundary_right_turn_feasibility",
        "episode_progress",
        "has_active_waypoint",
        "loiter_mode",
        "remaining_waypoints_ratio",
        "boundary_soft_risk",
        "deconfliction_active",
        "nearest_caution_time",
        "max_pair_conflict_pressure",
        "max_predicted_critical_pressure",
        "dominant_turn_preference",
        "left_sep_improvement",
        "right_sep_improvement",
    )
    NO_NEIGHBOR_FEATURE_VECTOR = (
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
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    )

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
        separation_dist: float = 50.0,
        min_agents: int = 2,
        max_agents: int = 5,
        reset_generation_attempts: int = 128,
        reset_min_feasible_cpa_m: Optional[float] = None,
        reset_min_boundary_time_ratio: float = 0.35,
        reset_heading_jitter_rad: float = 0.2,
        map_size_range_m: Tuple[float, float] = (300.0, 1_500.0),
        origin: Optional[Tuple[float, float]] = None,
        top_left: Optional[Tuple[float, float]] = None,
        bottom_right: Optional[Tuple[float, float]] = None,
        flight_config: Optional[dict] = None,
        reward_config: Optional[dict] = None,
        guidance_config: Optional[dict] = None,
        manual_missions: Optional[Dict[str, dict]] = None,
        terminate_on_all_waypoints_complete: bool = True,
        terminate_on_critical_violation: bool = True,
        terminate_on_geofence_violation: bool = True,
        geofence_breach_grace_steps: int = 1,
        enable_inter_drone_awareness: bool = True,
        refill_random_waypoints_on_completion: bool = False,
        allow_live_waypoint_updates: bool = False,
        disable_waypoint_navigation: bool = False,
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
        self.disable_waypoint_navigation = bool(disable_waypoint_navigation)
        min_mission_waypoints = 0 if self.disable_waypoint_navigation else 1
        self.mission_waypoint_count = max(
            int(mission_waypoint_count),
            min_mission_waypoints,
        )
        if mission_waypoint_count_min is None:
            mission_waypoint_count_min = self.mission_waypoint_count
        if mission_waypoint_count_max is None:
            mission_waypoint_count_max = self.mission_waypoint_count
        self.mission_waypoint_count_min = max(
            int(mission_waypoint_count_min),
            min_mission_waypoints,
        )
        self.mission_waypoint_count_max = max(
            int(mission_waypoint_count_max),
            min_mission_waypoints,
        )
        if self.mission_waypoint_count_max < self.mission_waypoint_count_min:
            raise ValueError(
                "mission_waypoint_count_max must be greater than or equal to "
                "mission_waypoint_count_min."
            )
        self._current_mission_waypoint_count = self.mission_waypoint_count_max
        self.waypoint_arrival_radius = max(float(waypoint_arrival_radius), 1.0)
        self.obs_stack_size = max(int(obs_stack_size), 1)
        self.caution_dist = float(separation_dist)
        self.critical_dist = float(separation_dist)
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
        self.terminate_on_critical_violation = bool(terminate_on_critical_violation)
        self.terminate_on_geofence_violation = bool(terminate_on_geofence_violation)
        self.geofence_breach_grace_steps = max(int(geofence_breach_grace_steps), 1)
        self.enable_inter_drone_awareness = bool(enable_inter_drone_awareness)
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
            reward_config.get("completion_bonus", 0.0)
        )
        self.penalty_geofence = float(
            reward_config.get("geofence_penalty", 20.0)
        )
        self.penalty_geofence_breach = float(
            reward_config.get("geofence_breach_penalty", self.penalty_geofence)
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
        self.stagnation_step_threshold = max(
            int(reward_config.get("stagnation_step_threshold", 12)),
            1,
        )
        self.stagnation_progress_epsilon_m = max(
            float(reward_config.get("stagnation_progress_epsilon_m", 1.0)),
            1e-6,
        )
        self.penalty_incomplete_mission = float(
            reward_config.get("incomplete_mission_penalty", 0.0)
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
        self.avoidance_neighbor_influence_scale = max(
            float(guidance_config.get("avoidance_neighbor_influence_scale", 2.5)),
            1.0,
        )
        self.avoidance_boundary_influence_scale = max(
            float(guidance_config.get("avoidance_boundary_influence_scale", 1.75)),
            0.5,
        )
        self.avoidance_lateral_bias = max(
            float(guidance_config.get("avoidance_lateral_bias", 0.6)),
            0.0,
        )
        self.avoidance_center_pull = max(
            float(guidance_config.get("avoidance_center_pull", 0.3)),
            0.0,
        )
        self.manager_avoid_neighbor_threshold = float(
            guidance_config.get("manager_avoid_neighbor_threshold", 0.18)
        )
        self.manager_avoid_boundary_threshold = float(
            guidance_config.get("manager_avoid_boundary_threshold", 0.22)
        )
        if reset_min_feasible_cpa_m is None:
            reset_min_feasible_cpa_m = max(
                self.critical_dist * 2.0,
                self.caution_dist * 0.4,
            )
        self.reset_generation_attempts = max(int(reset_generation_attempts), 1)
        self.reset_min_feasible_cpa_m = float(reset_min_feasible_cpa_m)
        self.reset_min_boundary_time_ratio = float(
            np.clip(reset_min_boundary_time_ratio, 0.0, 2.0)
        )
        self.reset_heading_jitter_rad = max(
            float(reset_heading_jitter_rad),
            0.0,
        )

        self.self_feature_names = self.SELF_FEATURE_NAMES
        self.self_feature_indices = {
            name: idx for idx, name in enumerate(self.self_feature_names)
        }
        self.self_feature_count = len(self.self_feature_names)
        self.neighbor_feature_count = len(self.NO_NEIGHBOR_FEATURE_VECTOR)
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
            "geofence_penalty",
            "geofence_breach_penalty",
            "boundary_soft_penalty",
            "crash_penalty",
            "harsh_turn_penalty",
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
        self._active_mask_cache = np.zeros(self.max_agents, dtype=np.float32)
        self._state_cache = np.zeros(self.state_dim, dtype=np.float32)
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
        self._last_avoidance_action_vector = np.zeros(self.max_agents, dtype=np.float32)
        self._last_neighbor_avoidance_pressure = np.zeros(
            self.max_agents,
            dtype=np.float32,
        )
        self._last_boundary_avoidance_pressure = np.zeros(
            self.max_agents,
            dtype=np.float32,
        )
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
        return self._state_cache.copy()

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
        self._last_avoidance_action_vector[:] = 0.0
        self._last_neighbor_avoidance_pressure[:] = 0.0
        self._last_boundary_avoidance_pressure[:] = 0.0
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
        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            self._last_headings[idx] = float(self.aircraft_by_agent[agent].heading)
            self._segment_start_local[idx] = self._local_pos_cache[idx]
            self._segment_start_heading[idx] = float(
                self.aircraft_by_agent[agent].heading
            )
            self._sync_waypoint_progress_tracking(agent)
            self._sync_waypoint_circling_tracking(agent)
        self._refresh_skill_guidance_cache()
        self._update_obs_cache(fill_history=True)

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
                and (
                    self.disable_waypoint_navigation
                    or aircraft.waypoint_manager.current_waypoint is not None
                )
            )
            assist_controls_aircraft = bool(
                action_controls_aircraft
                and aircraft.waypoint_manager.current_waypoint is not None
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
                loiter_turn_rate = aircraft.loiter_turn_rate_command(self.dt)
                max_turn_rate = max(float(aircraft.dynamics.max_turn_rate), 1e-6)
                executed_action_vector[idx] = float(
                    np.clip(loiter_turn_rate / max_turn_rate, -1.0, 1.0)
                )
                aircraft._update_loiter(
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

        boundary_soft_risks = []
        outside_depths = []
        geofence_breach = False
        waypoint_hits = 0

        for agent in current_agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            step_distance = float(
                np.linalg.norm(self._local_pos_cache[idx] - self._prev_local_cache[idx])
            )
            self._segment_path_length[idx] += step_distance
            current_wp = aircraft.waypoint_manager.current_waypoint
            best_improvement = 0.0
            started_circling_this_step = False
            if current_wp is not None:
                dist_before = float(
                    np.linalg.norm(self._wp_local_cache[idx] - self._prev_local_cache[idx])
                )
                dist_after = float(
                    np.linalg.norm(self._wp_local_cache[idx] - self._local_pos_cache[idx])
                )
                self._sync_waypoint_progress_tracking(
                    agent,
                    distance_to_wp=dist_before,
                )
                self._sync_waypoint_circling_tracking(agent)
                closest_before = float(min(self._closest_wp_distance[idx], dist_before))
                closest_after = float(min(closest_before, dist_after))
                self._closest_wp_distance[idx] = closest_after
                best_improvement = max(closest_before - dist_after, 0.0)

                reapproach_assisted = bool(
                    self._waypoint_reapproach_active[idx]
                    and not self._deconfliction_active[idx]
                )
                if reapproach_assisted or self._deconfliction_active[idx]:
                    self._clear_circling_state(idx)
                elif best_improvement >= self.circling_progress_reset_m:
                    if self._circling_active[idx]:
                        self._circling_breakouts_total[idx] += 1
                    self._clear_circling_state(idx)
                else:
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
                    self._circling_stagnation_steps[idx] += 1
                    self._circling_angular_travel[idx] += angular_step

                    max_turn_rate = max(float(aircraft.dynamics.max_turn_rate), 1e-6)
                    turn_fraction = abs(float(aircraft.actual_turn_rate)) / max_turn_rate
                    if self._circling_active[idx]:
                        if turn_fraction <= self.circling_relief_turn_fraction:
                            self._circling_relief_progress[idx] += 1
                            if self._circling_relief_progress[idx] >= self.circling_relief_steps:
                                self._circling_breakouts_total[idx] += 1
                                self._clear_circling_state(idx)
                        else:
                            self._circling_relief_progress[idx] = 0

                    arrival_threshold = max(
                        float(aircraft.waypoint_manager.arrival_threshold),
                        1e-6,
                    )
                    outside_capture_ratio = float(
                        max(
                            (dist_after / arrival_threshold)
                            - self.circling_min_distance_ratio,
                            0.0,
                        )
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
                            self._circling_stagnation_steps[idx]
                            - self.circling_activation_steps,
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
                        self._circling_steps_total[idx] += 1
            else:
                dist_after = 0.0
                self._sync_waypoint_progress_tracking(agent)
                self._sync_waypoint_circling_tracking(agent)

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
                self._clear_waypoint_tracking(idx)
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
                and not waypoint_reached
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
                if self.disable_waypoint_navigation:
                    self._clear_waypoint_tracking(idx)
                    self._reset_reference_route_progress(idx=idx, aircraft=aircraft)
                    aircraft.flight_mode = FlightMode.NAVIGATING
                    aircraft.loiter_center = None
                elif self.randomized and self.refill_random_waypoints_on_completion:
                    self._assign_random_waypoints(aircraft)
                    self._clear_waypoint_tracking(idx)
                    self._reset_reference_route_progress(idx=idx, aircraft=aircraft)
                else:
                    if self._completion_steps[idx] is None:
                        self._completion_steps[idx] = self.current_step
                    self._clear_waypoint_tracking(idx)
                    aircraft._enter_loiter()

            inside, soft_risk, outside_depth = self._boundary_status(agent)
            boundary_soft_risks.append(soft_risk)
            outside_depths.append(outside_depth)
            if not inside:
                self.geofence_outside_steps[idx] += 1
                if not self._was_outside[idx]:
                    self.geofence_exit_counts[idx] += 1
                    self._was_outside[idx] = True
                if (
                    self.terminate_on_geofence_violation
                    and self.geofence_outside_steps[idx]
                    >= self.geofence_breach_grace_steps
                ):
                    geofence_breach = True
            else:
                self._was_outside[idx] = False

            if aircraft.waypoint_manager.current_waypoint is None:
                self._clear_waypoint_tracking(idx)
                self._reset_reference_route_progress(idx=idx, aircraft=aircraft)

        all_done = False if self.disable_waypoint_navigation else all(
            not self.aircraft_by_agent[agent].waypoint_manager.has_waypoints()
            for agent in current_agents
        )
        terminated = self._crashed or geofence_breach or (
            all_done and self.terminate_on_all_waypoints_complete
        )
        truncated = self.current_step >= self.max_steps

        self._update_waypoint_local_cache()
        self._refresh_skill_guidance_cache()

        team_reward = self._compute_team_reward(
            current_agents=current_agents,
            waypoint_hits=waypoint_hits,
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
        elif geofence_breach:
            self._termination_reason = "geofence_violation"
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
