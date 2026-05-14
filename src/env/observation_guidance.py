from __future__ import annotations

from typing import List, Optional

import numpy as np

from flight_engine.helpers import FlightMode, clip_scalar, wrap_angle
from flight_engine.reference_guidance import (
    build_reference_route_local,
    compute_reference_action,
    dangerous_neighbor_turn_preview,
    guidance_turn_action_from_vector,
    time_to_boundary_ahead,
    turn_circle_feasibility_features,
)
from flight_engine.simulator import FixedWingAircraft


class ObservationGuidanceMixin:
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
            self._clear_circling_state(idx)
            self._circling_wp_signature[idx] = None
            return

        signature = (float(waypoint.latitude), float(waypoint.longitude))
        if self._circling_wp_signature[idx] == signature:
            return

        self._circling_wp_signature[idx] = signature
        self._clear_circling_state(idx)

    def _clear_circling_state(self, idx: int) -> None:
        self._circling_active[idx] = False
        self._circling_stagnation_steps[idx] = 0
        self._circling_angular_travel[idx] = 0.0
        self._circling_relief_progress[idx] = 0

    def _clear_waypoint_tracking(self, idx: int) -> None:
        self._closest_wp_signature[idx] = None
        self._closest_wp_distance[idx] = np.inf
        self._circling_wp_signature[idx] = None
        self._clear_circling_state(idx)
        self._clear_waypoint_reapproach(idx)

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
        return bool((constrained_capture and badly_misaligned) or overshot_waypoint)

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
        self._clear_circling_state(idx)
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
                (
                    aircraft.waypoint_manager.current_waypoint is None
                    and not self.disable_waypoint_navigation
                )
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

    def _refresh_avoidance_guidance_cache(self):
        self._last_avoidance_action_vector[:] = self._last_reference_action_vector
        self._last_neighbor_avoidance_pressure[:] = 0.0
        self._last_boundary_avoidance_pressure[:] = 0.0
        if not self.agents:
            return

        active_indices = (
            [
                self.agent_name_to_index[agent]
                for agent in self.agents
            ]
            if self.enable_inter_drone_awareness
            else []
        )
        center_local = np.asarray(
            [
                (self.local_min_x + self.local_max_x) * 0.5,
                (self.local_min_y + self.local_max_y) * 0.5,
            ],
            dtype=np.float32,
        )
        influence_distance = max(
            self.critical_dist * self.avoidance_neighbor_influence_scale,
            self.critical_dist,
            1.0,
        )
        boundary_buffer = max(
            min(self.box_width_m, self.box_height_m)
            * self.boundary_buffer_ratio
            * self.avoidance_boundary_influence_scale,
            self.caution_dist,
            1.0,
        )

        for agent in self.agents:
            idx = self.agent_name_to_index[agent]
            aircraft = self.aircraft_by_agent[agent]
            if (
                (
                    aircraft.waypoint_manager.current_waypoint is None
                    and not self.disable_waypoint_navigation
                )
                or aircraft.flight_mode == FlightMode.LOITERING
            ):
                continue

            pos_local = np.asarray(self._local_pos_cache[idx], dtype=np.float32)
            heading = float(aircraft.heading)
            left_axis = np.asarray(
                [-np.cos(heading), np.sin(heading)],
                dtype=np.float32,
            )

            avoidance_vec = np.zeros(2, dtype=np.float32)
            neighbor_pressure = 0.0
            preferred_turn_score = 0.0
            preferred_turn_weight = 0.0

            for other_idx in active_indices:
                if other_idx == idx:
                    continue
                rel_from_other = pos_local - self._local_pos_cache[other_idx]
                dist = float(np.linalg.norm(rel_from_other))
                if dist <= 1e-6:
                    continue
                weight = float(
                    np.clip(
                        (influence_distance - dist) / influence_distance,
                        0.0,
                        1.0,
                    )
                )

                rel_bearing = wrap_angle(
                    np.arctan2(-rel_from_other[0], -rel_from_other[1]) - heading
                )
                turn_pref = float(self._side_commitment[idx, other_idx])
                if turn_pref == 0.0:
                    if abs(rel_bearing) < (np.pi / 6.0):
                        turn_pref = 1.0
                    else:
                        turn_pref = -float(np.sign(rel_bearing))
                        if turn_pref == 0.0:
                            turn_pref = 1.0
                if weight > 0.0:
                    neighbor_pressure = max(neighbor_pressure, weight)
                    avoidance_vec += (rel_from_other / dist) * float(weight ** 2)
                    preferred_turn_score += weight * turn_pref
                    preferred_turn_weight += weight

                other_agent = self.possible_agents[other_idx]
                other = self.aircraft_by_agent.get(other_agent)
                if other is None:
                    continue
                own_speed = float(aircraft.dynamics.cruise_speed)
                other_heading = float(other.heading)
                other_speed = float(other.dynamics.cruise_speed)
                own_velocity = np.asarray(
                    [np.sin(heading), np.cos(heading)],
                    dtype=np.float32,
                ) * own_speed
                other_velocity = np.asarray(
                    [np.sin(other_heading), np.cos(other_heading)],
                    dtype=np.float32,
                ) * other_speed
                rel_position = self._local_pos_cache[other_idx] - pos_local
                rel_velocity = other_velocity - own_velocity
                speed_sq = float(np.dot(rel_velocity, rel_velocity))
                if speed_sq <= 1e-6:
                    continue
                t_cpa_raw = float(-np.dot(rel_position, rel_velocity) / speed_sq)
                if t_cpa_raw <= 0.0:
                    continue
                t_cpa = float(
                    np.clip(t_cpa_raw, 0.0, self.caution_lookahead_time_s)
                )
                projected_rel = rel_position + (rel_velocity * t_cpa)
                d_cpa = float(np.linalg.norm(projected_rel))
                predictive_influence = max(
                    self.caution_dist * self.avoidance_neighbor_influence_scale,
                    self.min_start_separation_m,
                    self.caution_dist,
                    1.0,
                )
                if d_cpa >= predictive_influence:
                    continue
                urgency = 1.0 - (
                    t_cpa / max(float(self.caution_lookahead_time_s), 1e-6)
                )
                predictive_weight = float(
                    np.clip(
                        urgency
                        * ((predictive_influence - d_cpa) / predictive_influence),
                        0.0,
                        1.0,
                    )
                )
                if predictive_weight <= 0.0:
                    continue
                neighbor_pressure = max(neighbor_pressure, predictive_weight)
                avoidance_vec += (rel_from_other / dist) * float(
                    predictive_weight ** 2
                )
                preferred_turn_score += predictive_weight * turn_pref
                preferred_turn_weight += predictive_weight

            x_pos = float(pos_local[0])
            y_pos = float(pos_local[1])
            boundary_vec = np.zeros(2, dtype=np.float32)
            boundary_pressure = 0.0
            boundary_side_specs = (
                (
                    y_pos - self.local_min_y,
                    np.asarray([0.0, 1.0], dtype=np.float32),
                ),
                (
                    self.local_max_y - y_pos,
                    np.asarray([0.0, -1.0], dtype=np.float32),
                ),
                (
                    x_pos - self.local_min_x,
                    np.asarray([1.0, 0.0], dtype=np.float32),
                ),
                (
                    self.local_max_x - x_pos,
                    np.asarray([-1.0, 0.0], dtype=np.float32),
                ),
            )
            for distance_to_wall, inward_normal in boundary_side_specs:
                wall_pressure = max(
                    (boundary_buffer - float(distance_to_wall)) / boundary_buffer,
                    0.0,
                )
                if wall_pressure <= 0.0:
                    continue
                boundary_pressure = max(boundary_pressure, float(wall_pressure))
                boundary_vec += inward_normal * float(wall_pressure ** 2)

            if x_pos < self.local_min_x:
                outside_pressure = 1.0 + ((self.local_min_x - x_pos) / boundary_buffer)
                boundary_pressure = max(boundary_pressure, float(outside_pressure))
                boundary_vec += np.asarray([outside_pressure ** 2, 0.0], dtype=np.float32)
            elif x_pos > self.local_max_x:
                outside_pressure = 1.0 + ((x_pos - self.local_max_x) / boundary_buffer)
                boundary_pressure = max(boundary_pressure, float(outside_pressure))
                boundary_vec += np.asarray([-(outside_pressure ** 2), 0.0], dtype=np.float32)
            if y_pos < self.local_min_y:
                outside_pressure = 1.0 + ((self.local_min_y - y_pos) / boundary_buffer)
                boundary_pressure = max(boundary_pressure, float(outside_pressure))
                boundary_vec += np.asarray([0.0, outside_pressure ** 2], dtype=np.float32)
            elif y_pos > self.local_max_y:
                outside_pressure = 1.0 + ((y_pos - self.local_max_y) / boundary_buffer)
                boundary_pressure = max(boundary_pressure, float(outside_pressure))
                boundary_vec += np.asarray([0.0, -(outside_pressure ** 2)], dtype=np.float32)

            boundary_pressure = float(
                np.clip(
                    max(boundary_pressure, float(np.linalg.norm(boundary_vec))),
                    0.0,
                    2.0,
                )
            )
            if preferred_turn_weight > 1e-6:
                dominant_turn = float(np.sign(preferred_turn_score))
                if dominant_turn == 0.0:
                    dominant_turn = 1.0
                avoidance_vec += (
                    left_axis
                    * dominant_turn
                    * self.avoidance_lateral_bias
                    * max(neighbor_pressure, 0.25)
                )

            avoidance_vec += boundary_vec.astype(np.float32, copy=False)
            center_vec = center_local - pos_local
            center_norm = float(np.linalg.norm(center_vec))
            if center_norm > 1e-6 and boundary_pressure > 0.0:
                avoidance_vec += (
                    (center_vec / center_norm)
                    * self.avoidance_center_pull
                    * boundary_pressure
                ).astype(np.float32, copy=False)

            self._last_neighbor_avoidance_pressure[idx] = float(neighbor_pressure)
            self._last_boundary_avoidance_pressure[idx] = float(boundary_pressure)
            if neighbor_pressure <= 1e-6 and boundary_pressure <= 1e-6:
                continue

            action = guidance_turn_action_from_vector(
                current_heading=heading,
                target_vec=avoidance_vec,
                cruise_speed=float(aircraft.dynamics.cruise_speed),
                turning_radius=float(aircraft.dynamics.turning_radius),
                max_turn_rate=float(aircraft.dynamics.max_turn_rate),
                arrival_threshold=float(aircraft.waypoint_manager.arrival_threshold),
                dt=self.dt,
                turn_gain=self.guidance_turn_gain,
                turn_lookahead_scale=self.guidance_turn_lookahead_scale,
                turn_radius_floor_scale=self.guidance_turn_radius_floor_scale,
            )
            self._last_avoidance_action_vector[idx] = float(action)

    def _refresh_skill_guidance_cache(self):
        self._refresh_route_guidance_cache()
        self._refresh_avoidance_guidance_cache()

    def _append_obs_history(self, *, fill_history: bool = False):
        if fill_history:
            self._obs_history[:] = self._base_obs_cache[:, None, :]
        else:
            if self.obs_stack_size > 1:
                self._obs_history[:, :-1] = self._obs_history[:, 1:]
            self._obs_history[:, -1] = self._base_obs_cache
        self._obs_cache[:] = self._obs_history.reshape(self.max_agents, self.obs_dim)
        self._active_mask_cache[:] = 0.0
        for agent in self.agents:
            self._active_mask_cache[self.agent_name_to_index[agent]] = 1.0
        self._state_cache[:] = np.concatenate(
            [
                self._obs_cache.reshape(-1),
                self._active_mask_cache,
                np.asarray(
                    [
                        self.box_width_m / max(self.map_size_scale, 1.0),
                        self.box_height_m / max(self.map_size_scale, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32, copy=False)

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
            has_navigation_waypoint = bool(
                aircraft.waypoint_manager.current_waypoint is not None
                and not self.disable_waypoint_navigation
            )
            if has_navigation_waypoint:
                wp_vec = wp_local_cache[idx] - local_pos_cache[idx]
                dist_wp = float(np.linalg.norm(wp_vec))
                bearing_wp = float(np.arctan2(wp_vec[0], wp_vec[1]))
                rel_bearing_wp = wrap_angle(bearing_wp - aircraft.heading)
            else:
                dist_wp = 0.0
                bearing_wp = float(aircraft.heading)
                rel_bearing_wp = 0.0
            next_wp_exists = bool(self._has_next_wp[idx] and has_navigation_waypoint)
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
                if has_navigation_waypoint
                else (1.0, 1.0)
            )
            episode_progress = float(
                clip_scalar(self.current_step / max(self.max_steps, 1), 0.0, 1.0)
            )
            has_active_waypoint = float(has_navigation_waypoint)
            loiter_mode = float(aircraft.flight_mode == FlightMode.LOITERING)
            _, remaining_waypoints, total_assigned_waypoints = self._mission_waypoint_counts(
                aircraft
            )
            remaining_waypoints_ratio = float(
                np.clip(
                    remaining_waypoints / max(total_assigned_waypoints, 1),
                    0.0,
                    1.0,
                )
                if has_navigation_waypoint
                else 0.0
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
                clip_scalar(float(self._last_reference_action_vector[idx]), -1.0, 1.0),
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
            neighbors = (
                [
                    other_idx
                    for other_idx in active_indices
                    if other_idx != idx
                ]
                if self.enable_inter_drone_awareness
                else []
            )
            neighbors.sort(key=lambda other_idx: dist_matrix[idx, other_idx])
            for slot in range(self.max_neighbors):
                if slot >= len(neighbors):
                    neighbor_features.extend(self.NO_NEIGHBOR_FEATURE_VECTOR)
                    continue

                other_idx = neighbors[slot]
                other_agent = self.possible_agents[other_idx]
                other = self.aircraft_by_agent[other_agent]
                other_max_turn_rate = max(float(other.dynamics.max_turn_rate), 1e-6)
                other_has_navigation_waypoint = bool(
                    other.waypoint_manager.current_waypoint is not None
                    and not self.disable_waypoint_navigation
                )
                rel_x = float(local_pos_cache[other_idx, 0] - local_pos_cache[idx, 0])
                rel_y = float(local_pos_cache[other_idx, 1] - local_pos_cache[idx, 1])
                dist = float(dist_matrix[idx, other_idx])
                rel_bearing = wrap_angle(
                    np.arctan2(rel_x, rel_y) - aircraft.heading
                )
                rel_heading = wrap_angle(other.heading - aircraft.heading)
                if other_has_navigation_waypoint:
                    other_wp_vec = wp_local_cache[other_idx] - local_pos_cache[other_idx]
                    other_dist_wp = float(np.linalg.norm(other_wp_vec))
                    other_bearing_wp = float(np.arctan2(other_wp_vec[0], other_wp_vec[1]))
                    other_rel_bearing_wp = wrap_angle(other_bearing_wp - other.heading)
                else:
                    other_dist_wp = 0.0
                    other_bearing_wp = float(other.heading)
                    other_rel_bearing_wp = 0.0
                other_next_wp_exists = bool(
                    self._has_next_wp[other_idx] and other_has_navigation_waypoint
                )
                if other_next_wp_exists:
                    other_next_leg_vec = (
                        next_wp_local_cache[other_idx] - wp_local_cache[other_idx]
                    )
                    other_next_leg_bearing = float(
                        np.arctan2(other_next_leg_vec[0], other_next_leg_vec[1])
                    )
                    other_lookahead_turn = wrap_angle(
                        other_next_leg_bearing - other_bearing_wp
                    )
                    other_next_leg_sin = float(np.sin(other_lookahead_turn))
                    other_next_leg_cos = float(np.cos(other_lookahead_turn))
                else:
                    other_next_leg_sin = 0.0
                    other_next_leg_cos = 1.0
                if other_has_navigation_waypoint:
                    own_to_other_wp_vec = wp_local_cache[other_idx] - local_pos_cache[idx]
                    own_to_other_wp_dist = float(np.linalg.norm(own_to_other_wp_vec))
                    own_to_other_wp_bearing = wrap_angle(
                        float(np.arctan2(own_to_other_wp_vec[0], own_to_other_wp_vec[1]))
                        - aircraft.heading
                    )
                else:
                    own_to_other_wp_dist = 0.0
                    own_to_other_wp_bearing = 0.0
                route_crossing_angle = (
                    wrap_angle(other_bearing_wp - bearing_wp)
                    if has_navigation_waypoint and other_has_navigation_waypoint
                    else 0.0
                )

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

                side_pass_cue = 0.0

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
                if abs(rel_bearing) < (np.pi / 6.0):
                    turn_preference = 1.0
                else:
                    turn_preference = -float(np.sign(rel_bearing))
                    if turn_preference == 0.0:
                        turn_preference = 1.0
                weighted_turn_preference += pair_conflict_pressure * turn_preference
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
                        0.0,
                        1.0 if self._deconfliction_active[other_idx] else 0.0,
                        0.0,
                        clip_scalar(other_dist_wp / map_diag_scale, 0.0, 2.0),
                        np.sin(other_rel_bearing_wp),
                        np.cos(other_rel_bearing_wp),
                        other_next_leg_sin,
                        other_next_leg_cos,
                        1.0 if other_next_wp_exists else 0.0,
                        clip_scalar(own_to_other_wp_dist / map_diag_scale, 0.0, 2.0),
                        np.sin(own_to_other_wp_bearing),
                        np.cos(own_to_other_wp_bearing),
                        np.sin(route_crossing_angle),
                        np.cos(route_crossing_angle),
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
            self_feature_vector = (
                self_features
                + [
                    left_sep_improvement,
                    right_sep_improvement,
                ]
            )
            if len(self_feature_vector) != self.self_feature_count:
                raise ValueError(
                    "Self observation feature length mismatch: "
                    f"expected {self.self_feature_count}, got "
                    f"{len(self_feature_vector)}"
                )

            feature_vector = self_feature_vector + neighbor_features
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
