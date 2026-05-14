from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from flight_engine.helpers import FlightMode, Position, clip_scalar, wrap_angle
from flight_engine.navigation_utils import (
    build_box_bounds,
    build_rect_bounds,
    heading_to_radians,
    order_route_points,
    planned_route_distance_m,
)
from flight_engine.reference_guidance import time_to_boundary_ahead
from flight_engine.simulator import FixedWingAircraft
from flight_engine.trans_coorders import CoordinateTransformer


class ScenarioGenerationMixin:
    def _resolve_mission_waypoint_count(self, options: dict) -> int:
        min_mission_waypoints = 0 if self.disable_waypoint_navigation else 1
        requested = options.get("mission_waypoint_count")
        if requested is not None:
            return max(int(requested), min_mission_waypoints)
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
                self._mission_waypoint_counts(aircraft)[2]
                for aircraft in self.aircraft_by_agent.values()
            )
        )

    def _mission_waypoint_counts(
        self,
        aircraft: FixedWingAircraft,
    ) -> tuple[int, int, int]:
        reached = len(aircraft.waypoint_manager.hit_waypoints)
        remaining = int(len(aircraft.waypoint_manager.remaining_waypoints()))
        return int(reached), int(remaining), int(reached + remaining)

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
        scenario_mode = str(options.get("scenario_mode", "random")).strip().lower()
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

        if scenario_mode == "conflict_course":
            self._reset_conflict_episode(options)
            return
        if scenario_mode not in {"random", "default"}:
            raise ValueError(
                "Unsupported random reset scenario_mode. "
                "Expected one of: random, default, conflict_course."
            )
        for _ in range(self.reset_generation_attempts):
            self.aircraft_by_agent = {}
            start_margin_request = float(
                options.get(
                    "start_margin_m",
                    max(
                        float(self.turning_radius_max),
                        self.min_start_separation_m * 0.5,
                        self.caution_dist * 0.75,
                    ),
                )
            )
            start_margin_max_ratio = float(
                options.get(
                    "start_margin_max_ratio",
                    0.28,
                )
            )
            start_margin_x = self._effective_edge_margin(
                span_m=self.box_width_m,
                requested_margin_m=start_margin_request,
                max_ratio=start_margin_max_ratio,
            )
            start_margin_y = self._effective_edge_margin(
                span_m=self.box_height_m,
                requested_margin_m=start_margin_request,
                max_ratio=start_margin_max_ratio,
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
                if self.disable_waypoint_navigation:
                    aircraft.replace_waypoint_queue([], replace_current=True)
                    aircraft.flight_mode = FlightMode.NAVIGATING
                else:
                    self._assign_random_waypoints(aircraft)
                self.aircraft_by_agent[agent] = aircraft
            if self.disable_waypoint_navigation:
                return
            _, is_valid = self._generated_scenario_score()
            if is_valid:
                return

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
                    if generate_random_waypoints or self.disable_waypoint_navigation
                    else list(params["waypoints"])
                ),
                turn_response_time_s=self.turn_response_time_s,
            )
            aircraft.waypoint_manager.arrival_threshold = self.waypoint_arrival_radius
            aircraft.path_history = [Position(lat, lon)]
            aircraft.last_waypoint_hit_pos = None
            aircraft.actual_turn_rate = 0.0
            aircraft.desired_turn_rate = 0.0
            if self.disable_waypoint_navigation:
                aircraft.replace_waypoint_queue([], replace_current=True)
                aircraft.flight_mode = FlightMode.NAVIGATING
            self.aircraft_by_agent[agent] = aircraft
        if generate_random_waypoints and not self.disable_waypoint_navigation:
            self._resample_generated_waypoints_until_feasible()

    def _clip_local_point(
        self,
        point_local: np.ndarray,
        *,
        margin_m: float,
    ) -> np.ndarray:
        x_min = self.local_min_x + margin_m
        x_max = self.local_max_x - margin_m
        y_min = self.local_min_y + margin_m
        y_max = self.local_max_y - margin_m
        if x_max < x_min:
            x_min = self.local_min_x
            x_max = self.local_max_x
        if y_max < y_min:
            y_min = self.local_min_y
            y_max = self.local_max_y
        return np.asarray(
            [
                float(np.clip(point_local[0], x_min, x_max)),
                float(np.clip(point_local[1], y_min, y_max)),
            ],
            dtype=np.float32,
        )

    def _align_generated_mission_heading(
        self,
        aircraft: FixedWingAircraft,
    ) -> None:
        waypoint = aircraft.waypoint_manager.current_waypoint
        if waypoint is None:
            return
        start_local = np.asarray(
            self.transformer.geo_to_local(
                aircraft.position.latitude,
                aircraft.position.longitude,
            ),
            dtype=np.float32,
        )
        wp_local = np.asarray(
            self.transformer.geo_to_local(
                waypoint.latitude,
                waypoint.longitude,
            ),
            dtype=np.float32,
        )
        target_vec = wp_local - start_local
        if float(np.linalg.norm(target_vec)) <= 1e-6:
            return
        heading = float(np.arctan2(target_vec[0], target_vec[1]))
        if self.reset_heading_jitter_rad > 0.0:
            heading += float(
                self._rng.uniform(
                    -self.reset_heading_jitter_rad,
                    self.reset_heading_jitter_rad,
                )
            )
        aligned_heading = float(wrap_angle(heading))
        aircraft.heading = aligned_heading
        aircraft.initial_heading = aligned_heading
        aircraft.actual_turn_rate = 0.0
        aircraft.desired_turn_rate = 0.0

    def _generated_scenario_candidate_specs(self) -> tuple[List[dict], List[float]]:
        candidate_specs: List[dict] = []
        boundary_time_ratios: List[float] = []
        for agent in self.agents:
            aircraft = self.aircraft_by_agent[agent]
            pos_local = np.asarray(
                self.transformer.geo_to_local(
                    aircraft.position.latitude,
                    aircraft.position.longitude,
                ),
                dtype=np.float32,
            )
            boundary_time_ratios.append(
                float(
                    time_to_boundary_ahead(
                        pos_local=pos_local,
                        heading=float(aircraft.heading),
                        cruise_speed=float(aircraft.dynamics.cruise_speed),
                        local_min_x=float(self.local_min_x),
                        local_max_x=float(self.local_max_x),
                        local_min_y=float(self.local_min_y),
                        local_max_y=float(self.local_max_y),
                        lookahead_time_s=float(self.caution_lookahead_time_s),
                    )
                )
            )
            waypoint = aircraft.waypoint_manager.current_waypoint
            if waypoint is None:
                continue
            wp_local = np.asarray(
                self.transformer.geo_to_local(
                    waypoint.latitude,
                    waypoint.longitude,
                ),
                dtype=np.float32,
            )
            candidate_specs.append(
                {
                    "agent": agent,
                    "start_local": pos_local,
                    "waypoint_locals": [wp_local],
                    "cruise_speed": float(aircraft.dynamics.cruise_speed),
                }
            )
        return candidate_specs, boundary_time_ratios

    def _generated_scenario_score(self) -> tuple[float, bool]:
        candidate_specs, boundary_time_ratios = self._generated_scenario_candidate_specs()
        pair_metrics = self._conflict_candidate_pair_metrics(candidate_specs)
        min_pair_cpa = (
            min(float(metrics["d_cpa"]) for metrics in pair_metrics)
            if pair_metrics
            else None
        )
        min_boundary_ratio = (
            min(float(ratio) for ratio in boundary_time_ratios)
            if boundary_time_ratios
            else 2.0
        )
        cpa_deficit = (
            max(self.reset_min_feasible_cpa_m - float(min_pair_cpa), 0.0)
            if min_pair_cpa is not None
            else 0.0
        )
        boundary_deficit = max(
            self.reset_min_boundary_time_ratio - float(min_boundary_ratio),
            0.0,
        )
        pair_score = (
            float(min_pair_cpa)
            if min_pair_cpa is not None
            else float(self.caution_dist) * 2.0
        )
        score = (
            pair_score
            + (float(self.caution_dist) * float(min_boundary_ratio))
            - (100.0 * cpa_deficit)
            - (50.0 * boundary_deficit)
        )
        is_valid = (cpa_deficit <= 0.0) and (boundary_deficit <= 0.0)
        return float(score), bool(is_valid)

    def _capture_generated_waypoint_state(self) -> dict[str, dict]:
        snapshot: dict[str, dict] = {}
        for agent in self.agents:
            aircraft = self.aircraft_by_agent[agent]
            snapshot[agent] = {
                "heading": float(aircraft.heading),
                "waypoints": [
                    (
                        float(waypoint.latitude),
                        float(waypoint.longitude),
                    )
                    for waypoint in aircraft.waypoint_manager.remaining_waypoints()
                ],
            }
        return snapshot

    def _restore_generated_waypoint_state(self, snapshot: dict[str, dict]) -> None:
        for agent, state in snapshot.items():
            aircraft = self.aircraft_by_agent[agent]
            aircraft.waypoint_manager.hit_waypoints = []
            aircraft.replace_waypoint_queue(
                [
                    Position(latitude, longitude)
                    for latitude, longitude in state["waypoints"]
                ],
                replace_current=True,
            )
            heading = float(state["heading"])
            aircraft.heading = heading
            aircraft.initial_heading = heading
            aircraft.actual_turn_rate = 0.0
            aircraft.desired_turn_rate = 0.0

    def _resample_generated_waypoints_until_feasible(self) -> None:
        best_snapshot: Optional[dict[str, dict]] = None
        best_score = float("-inf")
        for _ in range(self.reset_generation_attempts):
            for agent in self.agents:
                self._assign_random_waypoints(self.aircraft_by_agent[agent])
            score, is_valid = self._generated_scenario_score()
            if score > best_score:
                best_score = score
                best_snapshot = self._capture_generated_waypoint_state()
            if is_valid:
                return
        if best_snapshot is not None:
            self._restore_generated_waypoint_state(best_snapshot)

    def _conflict_candidate_pair_metrics(
        self,
        candidate_specs: List[dict],
    ) -> List[dict]:
        pair_metrics: List[dict] = []
        for idx_a, spec_a in enumerate(candidate_specs):
            waypoint_locals_a = spec_a.get("waypoint_locals", [])
            if not waypoint_locals_a:
                continue
            start_a = np.asarray(spec_a["start_local"], dtype=np.float32)
            first_wp_a = np.asarray(waypoint_locals_a[0], dtype=np.float32)
            leg_a = first_wp_a - start_a
            leg_a_dist = float(np.linalg.norm(leg_a))
            if leg_a_dist <= 1e-6:
                continue
            speed_a = max(float(spec_a["cruise_speed"]), 1e-6)
            vel_a = (leg_a / leg_a_dist) * speed_a
            eta_a = leg_a_dist / speed_a

            for idx_b in range(idx_a + 1, len(candidate_specs)):
                spec_b = candidate_specs[idx_b]
                waypoint_locals_b = spec_b.get("waypoint_locals", [])
                if not waypoint_locals_b:
                    continue
                start_b = np.asarray(spec_b["start_local"], dtype=np.float32)
                first_wp_b = np.asarray(waypoint_locals_b[0], dtype=np.float32)
                leg_b = first_wp_b - start_b
                leg_b_dist = float(np.linalg.norm(leg_b))
                if leg_b_dist <= 1e-6:
                    continue
                speed_b = max(float(spec_b["cruise_speed"]), 1e-6)
                vel_b = (leg_b / leg_b_dist) * speed_b
                eta_b = leg_b_dist / speed_b
                rel_position = start_b - start_a
                rel_velocity = vel_b - vel_a
                max_eval_time = max(
                    min(
                        eta_a,
                        eta_b,
                        self.caution_lookahead_time_s,
                    ),
                    0.0,
                )
                speed_sq = float(np.dot(rel_velocity, rel_velocity))
                if speed_sq > 1e-6:
                    t_cpa = clip_scalar(
                        -float(np.dot(rel_position, rel_velocity)) / speed_sq,
                        0.0,
                        max_eval_time,
                    )
                    d_cpa = float(
                        np.linalg.norm(rel_position + (rel_velocity * t_cpa))
                    )
                else:
                    t_cpa = 0.0
                    d_cpa = float(np.linalg.norm(rel_position))
                pair_metrics.append(
                    {
                        "pair": (idx_a, idx_b),
                        "d_cpa": float(d_cpa),
                        "t_cpa": float(t_cpa),
                        "max_eval_time": float(max_eval_time),
                    }
                )
        return pair_metrics

    def _conflict_candidate_boundary_ratios(
        self,
        candidate_specs: List[dict],
        *,
        max_legs: int,
    ) -> List[float]:
        boundary_ratios: List[float] = []
        leg_limit = max(int(max_legs), 1)
        for spec in candidate_specs:
            waypoint_locals = spec.get("waypoint_locals", [])
            if not waypoint_locals:
                continue
            current_local = np.asarray(spec["start_local"], dtype=np.float32)
            cruise_speed = max(float(spec.get("cruise_speed", 0.0)), 1e-6)
            for waypoint_local in waypoint_locals[:leg_limit]:
                next_local = np.asarray(waypoint_local, dtype=np.float32)
                leg_vec = next_local - current_local
                leg_dist = float(np.linalg.norm(leg_vec))
                if leg_dist <= 1e-6:
                    current_local = next_local
                    continue
                heading = float(np.arctan2(leg_vec[0], leg_vec[1]))
                boundary_ratios.append(
                    float(
                        time_to_boundary_ahead(
                            pos_local=current_local,
                            heading=heading,
                            cruise_speed=cruise_speed,
                            local_min_x=float(self.local_min_x),
                            local_max_x=float(self.local_max_x),
                            local_min_y=float(self.local_min_y),
                            local_max_y=float(self.local_max_y),
                            lookahead_time_s=float(self.caution_lookahead_time_s),
                        )
                    )
                )
                current_local = next_local
        return boundary_ratios

    @staticmethod
    def _conflict_candidate_score(
        pair_metrics: List[dict],
        *,
        min_feasible_cpa_m: float,
        target_cpa_m: float,
        max_trigger_cpa_m: float,
        min_conflict_pairs: int,
        boundary_time_ratios: Optional[List[float]] = None,
        min_boundary_time_ratio: Optional[float] = None,
        target_boundary_time_ratio: Optional[float] = None,
        max_boundary_time_ratio: Optional[float] = None,
    ) -> tuple[float, bool]:
        if not pair_metrics:
            return float("-inf"), False
        if (
            boundary_time_ratios is not None
            and not boundary_time_ratios
            and (
                min_boundary_time_ratio is not None
                or target_boundary_time_ratio is not None
                or max_boundary_time_ratio is not None
            )
        ):
            return float("-inf"), False

        trigger_pairs = [
            metrics
            for metrics in pair_metrics
            if metrics["d_cpa"] <= max_trigger_cpa_m
        ]
        if not trigger_pairs:
            closest_pair_cpa = min(
                float(metrics["d_cpa"])
                for metrics in pair_metrics
            )
            return (
                -1_000.0 - abs(closest_pair_cpa - target_cpa_m),
                False,
            )

        cpa_deficit_total = float(
            sum(
                max(min_feasible_cpa_m - float(metrics["d_cpa"]), 0.0)
                for metrics in trigger_pairs
            )
        )
        target_error = float(
            np.mean(
                [
                    abs(float(metrics["d_cpa"]) - target_cpa_m)
                    for metrics in trigger_pairs
                ]
            )
        )
        conflict_pair_bonus = float(
            min(len(trigger_pairs), max(min_conflict_pairs, 1))
        )
        min_boundary_ratio = (
            min(float(ratio) for ratio in boundary_time_ratios)
            if boundary_time_ratios
            else None
        )
        boundary_deficit = (
            max(float(min_boundary_time_ratio) - float(min_boundary_ratio), 0.0)
            if min_boundary_ratio is not None and min_boundary_time_ratio is not None
            else 0.0
        )
        boundary_excess = (
            max(float(min_boundary_ratio) - float(max_boundary_time_ratio), 0.0)
            if min_boundary_ratio is not None and max_boundary_time_ratio is not None
            else 0.0
        )
        boundary_target_error = (
            abs(float(min_boundary_ratio) - float(target_boundary_time_ratio))
            if min_boundary_ratio is not None
            and target_boundary_time_ratio is not None
            else 0.0
        )
        score = (
            (25.0 * conflict_pair_bonus)
            - (50.0 * cpa_deficit_total)
            - target_error
            - (40.0 * boundary_deficit)
            - (20.0 * boundary_excess)
            - (12.0 * boundary_target_error)
        )
        is_valid = (
            len(trigger_pairs) >= max(min_conflict_pairs, 1)
            and cpa_deficit_total <= 0.0
            and boundary_deficit <= 0.0
            and boundary_excess <= 0.0
        )
        return float(score), bool(is_valid)

    def _reset_conflict_episode(self, options: dict):
        self.aircraft_by_agent = {}
        map_center_local = np.asarray(
            [
                (self.local_min_x + self.local_max_x) * 0.5,
                (self.local_min_y + self.local_max_y) * 0.5,
            ],
            dtype=np.float32,
        )
        span_min = min(self.box_width_m, self.box_height_m)
        edge_margin_request = options.get("conflict_edge_margin_m")
        edge_margin = self._effective_edge_margin(
            span_m=span_min,
            requested_margin_m=(
                float(edge_margin_request)
                if edge_margin_request is not None
                else max(
                    float(self.turning_radius_max) * 1.5,
                    self.waypoint_arrival_radius * 2.0,
                    self.caution_dist * 1.25,
                )
            ),
            max_ratio=float(options.get("conflict_edge_margin_max_ratio", 0.28)),
        )
        start_radius_fraction = float(
            np.clip(
                options.get("conflict_start_radius_fraction", 0.24),
                0.02,
                0.98,
            )
        )
        crossing_radius_fraction = float(
            np.clip(
                options.get("conflict_crossing_radius_fraction", start_radius_fraction * 0.75),
                0.02,
                start_radius_fraction,
            )
        )
        boundary_radius_fraction = float(
            np.clip(
                options.get(
                    "conflict_boundary_radius_fraction",
                    max(start_radius_fraction, 0.9),
                ),
                crossing_radius_fraction,
                0.99,
            )
        )
        requested_start_radius_m = options.get("conflict_start_radius_m")
        requested_exit_radius_m = options.get("conflict_exit_radius_m")
        requested_follow_radius_m = options.get("conflict_follow_radius_m")
        target_leg_time_s = max(
            float(
                options.get(
                    "conflict_target_leg_time_s",
                    max(self.caution_lookahead_time_s * 0.75, 4.0),
                )
            ),
            1.0,
        )
        avg_conflict_speed = max(
            0.5 * (self.cruise_speed_min + self.cruise_speed_max),
            1.0,
        )
        requested_target_leg_distance_m = options.get("conflict_target_leg_distance_m")
        phase_jitter = float(options.get("conflict_phase_jitter_rad", np.pi / 14.0))
        lateral_jitter = float(
            options.get(
                "conflict_first_wp_lateral_jitter_m",
                max(self.caution_dist * 0.2, 8.0),
            )
        )
        boundary_focus_probability = float(
            np.clip(
                options.get("conflict_boundary_focus_probability", 0.0),
                0.0,
                1.0,
            )
        )
        boundary_focus_center_fraction = float(
            np.clip(
                options.get("conflict_boundary_focus_center_fraction", 0.72),
                0.0,
                0.95,
            )
        )
        boundary_focus_target_time_ratio = float(
            max(
                options.get(
                    "conflict_boundary_focus_target_time_ratio",
                    self.reset_min_boundary_time_ratio,
                ),
                self.reset_min_boundary_time_ratio,
            )
        )
        boundary_focus_max_time_ratio = float(
            max(
                options.get(
                    "conflict_boundary_focus_max_time_ratio",
                    boundary_focus_target_time_ratio,
                ),
                boundary_focus_target_time_ratio,
            )
        )
        boundary_focus_scored_legs = max(
            int(options.get("conflict_boundary_focus_scored_legs", 2)),
            1,
        )
        follow_lateral_offset = float(
            options.get(
                "conflict_follow_lateral_offset_m",
                max(self.caution_dist * 0.9, float(self.turning_radius_max) * 1.25),
            )
        )
        boundary_route_offset = float(
            options.get(
                "conflict_boundary_route_offset_m",
                max(
                    follow_lateral_offset * 1.25,
                    self.caution_dist * 1.5,
                    float(self.turning_radius_max) * 1.5,
                ),
            )
        )
        generation_attempts = max(
            int(options.get("conflict_generation_attempts", 256)),
            1,
        )
        min_feasible_cpa_m = float(
            options.get(
                "conflict_min_feasible_cpa_m",
                max(self.critical_dist * 2.0, self.caution_dist * 0.4),
            )
        )
        target_cpa_m = float(
            options.get(
                "conflict_target_cpa_m",
                self.caution_dist * 0.7,
            )
        )
        max_trigger_cpa_m = float(
            options.get(
                "conflict_max_trigger_cpa_m",
                self.caution_dist * 0.95,
            )
        )
        min_conflict_pairs = max(
            int(options.get("conflict_min_pairs", 1)),
            1,
        )
        require_valid_candidate = bool(
            options.get("conflict_require_valid_candidate", False)
        )
        require_boundary_viability = bool(
            options.get("conflict_require_boundary_viability", False)
        )
        min_boundary_time_ratio = float(
            np.clip(
                options.get(
                    "conflict_min_boundary_time_ratio",
                    self.reset_min_boundary_time_ratio,
                ),
                0.0,
                2.0,
            )
        )
        chosen_specs = None
        safe_fallback_specs = None
        safe_fallback_score = float("-inf")
        fallback_specs = None
        fallback_score = float("-inf")

        for _ in range(generation_attempts):
            center_local = np.asarray(map_center_local, dtype=np.float32)
            boundary_focus_active = bool(
                boundary_focus_probability > 0.0
                and float(self._rng.random()) < boundary_focus_probability
            )
            if boundary_focus_active:
                center_guard = edge_margin + max(
                    self.caution_dist * 1.75,
                    self.waypoint_arrival_radius * 3.0,
                )
                focus_axis = int(self._rng.integers(0, 4))
                if focus_axis == 0:
                    max_offset = max(
                        (self.local_max_x - map_center_local[0]) - center_guard,
                        0.0,
                    )
                    center_local[0] += float(
                        boundary_focus_center_fraction * max_offset
                    )
                elif focus_axis == 1:
                    max_offset = max(
                        (map_center_local[0] - self.local_min_x) - center_guard,
                        0.0,
                    )
                    center_local[0] -= float(
                        boundary_focus_center_fraction * max_offset
                    )
                elif focus_axis == 2:
                    max_offset = max(
                        (self.local_max_y - map_center_local[1]) - center_guard,
                        0.0,
                    )
                    center_local[1] += float(
                        boundary_focus_center_fraction * max_offset
                    )
                else:
                    max_offset = max(
                        (map_center_local[1] - self.local_min_y) - center_guard,
                        0.0,
                    )
                    center_local[1] -= float(
                        boundary_focus_center_fraction * max_offset
                    )

            available_radius = max(
                min(
                    center_local[0] - self.local_min_x,
                    self.local_max_x - center_local[0],
                    center_local[1] - self.local_min_y,
                    self.local_max_y - center_local[1],
                ) - edge_margin,
                1.0,
            )
            radius_floor = min(
                available_radius,
                max(
                    self.caution_dist * 1.75,
                    self.waypoint_arrival_radius * 3.0,
                ),
            )
            start_radius = float(
                np.clip(
                    float(requested_start_radius_m)
                    if requested_start_radius_m is not None
                    else available_radius * start_radius_fraction,
                    radius_floor,
                    available_radius,
                )
            )
            exit_radius = float(
                np.clip(
                    float(requested_exit_radius_m)
                    if requested_exit_radius_m is not None
                    else available_radius * crossing_radius_fraction,
                    min(radius_floor, available_radius),
                    available_radius,
                )
            )
            target_leg_distance = max(
                float(requested_target_leg_distance_m)
                if requested_target_leg_distance_m is not None
                else avg_conflict_speed * target_leg_time_s,
                float(exit_radius),
            )
            target_start_radius = float(
                np.clip(
                    target_leg_distance - exit_radius,
                    radius_floor,
                    available_radius,
                )
            )
            start_radius = min(start_radius, target_start_radius)
            follow_radius = float(
                np.clip(
                    float(requested_follow_radius_m)
                    if requested_follow_radius_m is not None
                    else available_radius * boundary_radius_fraction,
                    max(exit_radius, min(radius_floor, available_radius)),
                    available_radius,
                )
            )
            phase = float(self._rng.uniform(-np.pi, np.pi))
            candidate_specs = []
            for idx, agent in enumerate(self.agents):
                theta = phase + ((2.0 * np.pi * idx) / max(len(self.agents), 1))
                theta += float(self._rng.uniform(-phase_jitter, phase_jitter))
                radial = np.asarray(
                    [np.sin(theta), np.cos(theta)],
                    dtype=np.float32,
                )
                lateral = np.asarray(
                    [np.cos(theta), -np.sin(theta)],
                    dtype=np.float32,
                )

                start_local = self._clip_local_point(
                    center_local + (radial * start_radius),
                    margin_m=edge_margin,
                )
                first_wp_local = self._clip_local_point(
                    center_local
                    - (radial * exit_radius)
                    + (
                        lateral
                        * float(self._rng.uniform(-lateral_jitter, lateral_jitter))
                    ),
                    margin_m=edge_margin,
                )
                lateral_sign = 1.0 if (idx % 2) == 0 else -1.0
                second_wp_local = self._clip_local_point(
                    center_local
                    - (radial * follow_radius)
                    + (lateral * lateral_sign * follow_lateral_offset),
                    margin_m=edge_margin,
                )

                cruise_speed = float(
                    self._rng.uniform(self.cruise_speed_min, self.cruise_speed_max)
                )
                turning_radius = float(
                    self._rng.uniform(self.turning_radius_min, self.turning_radius_max)
                )

                waypoint_locals = [first_wp_local]
                if self._current_mission_waypoint_count >= 2:
                    waypoint_locals.append(second_wp_local)
                extra_waypoint_count = max(self._current_mission_waypoint_count - 2, 0)
                if extra_waypoint_count > 0:
                    boundary_base = center_local - (radial * follow_radius)
                    for extra_idx in range(extra_waypoint_count):
                        sweep_scale = float(extra_idx + 1) / float(extra_waypoint_count + 1)
                        sweep_sign = lateral_sign if (extra_idx % 2) == 0 else -lateral_sign
                        sweep_offset = follow_lateral_offset + (
                            boundary_route_offset * sweep_scale
                        )
                        boundary_wp_local = self._clip_local_point(
                            boundary_base + (lateral * sweep_sign * sweep_offset),
                            margin_m=edge_margin,
                        )
                        waypoint_locals.append(boundary_wp_local)

                candidate_specs.append(
                    {
                        "agent": agent,
                        "start_local": np.asarray(start_local, dtype=np.float32),
                        "waypoint_locals": [
                            np.asarray(point, dtype=np.float32)
                            for point in waypoint_locals
                        ],
                        "cruise_speed": float(cruise_speed),
                        "turning_radius": float(turning_radius),
                    }
                )

            pair_metrics = self._conflict_candidate_pair_metrics(candidate_specs)
            boundary_time_ratios = (
                self._conflict_candidate_boundary_ratios(
                    candidate_specs,
                    max_legs=boundary_focus_scored_legs,
                )
                if boundary_focus_active or require_boundary_viability
                else None
            )
            candidate_score, is_valid_candidate = self._conflict_candidate_score(
                pair_metrics,
                min_feasible_cpa_m=min_feasible_cpa_m,
                target_cpa_m=target_cpa_m,
                max_trigger_cpa_m=max_trigger_cpa_m,
                min_conflict_pairs=min_conflict_pairs,
                boundary_time_ratios=boundary_time_ratios,
                min_boundary_time_ratio=(
                    min_boundary_time_ratio
                    if boundary_focus_active or require_boundary_viability
                    else None
                ),
                target_boundary_time_ratio=(
                    boundary_focus_target_time_ratio
                    if boundary_focus_active
                    else None
                ),
                max_boundary_time_ratio=(
                    boundary_focus_max_time_ratio
                    if boundary_focus_active
                    else None
                ),
            )
            if is_valid_candidate:
                chosen_specs = candidate_specs
                break
            is_feasible_candidate = bool(
                pair_metrics
                and all(
                    float(metrics["d_cpa"]) >= float(min_feasible_cpa_m)
                    for metrics in pair_metrics
                )
            )
            if is_feasible_candidate and candidate_score > safe_fallback_score:
                safe_fallback_specs = candidate_specs
                safe_fallback_score = candidate_score
            if candidate_score > fallback_score:
                fallback_specs = candidate_specs
                fallback_score = candidate_score

        if chosen_specs is None and require_valid_candidate:
            raise RuntimeError(
                "Failed to generate a valid survivable conflict_course scenario "
                f"after {generation_attempts} attempts. "
                f"num_agents={len(self.agents)}, box=({self.box_width_m:.1f}m x "
                f"{self.box_height_m:.1f}m), min_feasible_cpa_m={min_feasible_cpa_m:.1f}, "
                f"target_cpa_m={target_cpa_m:.1f}, max_trigger_cpa_m={max_trigger_cpa_m:.1f}, "
                f"min_conflict_pairs={min_conflict_pairs}, "
                f"require_boundary_viability={require_boundary_viability}, "
                f"min_boundary_time_ratio={min_boundary_time_ratio:.2f}. "
                "Increase conflict_generation_attempts or relax the scenario constraints; "
                "strict mode will not fall back to an invalid or easier reset."
            )

        selected_specs = (
            chosen_specs
            if chosen_specs is not None
            else safe_fallback_specs
            if safe_fallback_specs is not None
            else fallback_specs
            if fallback_specs is not None
            else []
        )

        for spec in selected_specs:
            agent = str(spec["agent"])
            start_local = np.asarray(spec["start_local"], dtype=np.float32)
            waypoint_locals = [
                np.asarray(point, dtype=np.float32)
                for point in spec["waypoint_locals"]
            ]
            cruise_speed = float(spec["cruise_speed"])
            turning_radius = float(spec["turning_radius"])

            start_lat, start_lon = self.transformer.local_to_geo(
                float(start_local[0]),
                float(start_local[1]),
            )
            first_waypoint_local = waypoint_locals[0]
            heading = float(
                np.arctan2(
                    first_waypoint_local[0] - start_local[0],
                    first_waypoint_local[1] - start_local[1],
                )
            )
            mission = []
            for waypoint_idx, waypoint_local in enumerate(waypoint_locals):
                wp_lat, wp_lon = self.transformer.local_to_geo(
                    float(waypoint_local[0]),
                    float(waypoint_local[1]),
                )
                mission.append(
                    Position(
                        float(wp_lat),
                        float(wp_lon),
                        waypoint_id=f"{agent}-conflict-{waypoint_idx + 1}",
                    )
                )

            aircraft = FixedWingAircraft(
                id_tag=agent,
                initial_position=Position(float(start_lat), float(start_lon)),
                initial_heading=heading,
                cruise_speed=cruise_speed,
                turning_radius=turning_radius,
                mission=mission,
                turn_response_time_s=self.turn_response_time_s,
            )
            aircraft.waypoint_manager.arrival_threshold = self.waypoint_arrival_radius
            aircraft.path_history = [Position(float(start_lat), float(start_lon))]
            aircraft.last_waypoint_hit_pos = None
            aircraft.actual_turn_rate = 0.0
            aircraft.desired_turn_rate = 0.0
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
        aircraft.waypoint_manager.hit_waypoints = []
        aircraft.replace_waypoint_queue(
            [
                Position(
                    *self.transformer.local_to_geo(float(x), float(y))
                )
                for x, y in ordered_points
            ],
            replace_current=True,
        )
        self._align_generated_mission_heading(aircraft)
        aircraft.actual_turn_rate = 0.0
        aircraft.desired_turn_rate = 0.0
        if aircraft.waypoint_manager.has_waypoints():
            aircraft.flight_mode = FlightMode.NAVIGATING
            aircraft.loiter_center = None
