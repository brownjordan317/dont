from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import numpy as np

from flight_engine.helpers import FlightMode, wrap_angle


class SafetyRewardMixin:
    def _update_deconfliction_state(
        self,
        active_agents: List[str],
        pair_metrics: dict,
        *,
        was_active: np.ndarray,
    ) -> None:
        if not self.enable_inter_drone_awareness:
            for agent in active_agents:
                idx = self.agent_name_to_index[agent]
                self._deconfliction_active[idx] = False
            return

        caution_pressure = np.asarray(
            pair_metrics["agent_caution_pressure"],
            dtype=np.float32,
        )
        critical_pressure = np.asarray(
            pair_metrics.get(
                "agent_critical_pressure",
                np.zeros_like(caution_pressure),
            ),
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
                continue

            within_caution = bool(float(caution_pressure[idx]) > 0.0)
            predicted_or_current_critical = bool(float(critical_pressure[idx]) > 0.0)

            if within_caution or predicted_or_current_critical:
                self._deconfliction_active[idx] = True
            else:
                self._deconfliction_active[idx] = False

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
        agent_caution_pressure = np.zeros(self.max_agents, dtype=np.float32)
        agent_conflict_pressure = np.zeros(self.max_agents, dtype=np.float32)
        agent_critical_pressure = np.zeros(self.max_agents, dtype=np.float32)
        parallel_conflict_pressure = np.zeros(self.max_agents, dtype=np.float32)
        preferred_turn_score = np.zeros(self.max_agents, dtype=np.float32)
        preferred_turn_weight = np.zeros(self.max_agents, dtype=np.float32)
        if not self.enable_inter_drone_awareness:
            return {
                "critical_pairs": critical_pairs,
                "caution_pairs": caution_pairs,
                "agent_caution_pressure": agent_caution_pressure,
                "agent_conflict_pressure": agent_conflict_pressure,
                "agent_critical_pressure": agent_critical_pressure,
                "parallel_conflict_pressure": parallel_conflict_pressure,
                "preferred_turn": np.zeros(self.max_agents, dtype=np.float32),
            }
        critical_buffer_dist = self.critical_dist
        critical_buffer_span = max(self.critical_dist, 1e-6)
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
        if critical_pairs and self.terminate_on_critical_violation:
            self._crashed = True

        preferred_turn = np.zeros(self.max_agents, dtype=np.float32)
        active_pref = preferred_turn_weight > 0.0
        preferred_turn[active_pref] = np.sign(preferred_turn_score[active_pref])
        neutral_pref = active_pref & (preferred_turn == 0.0)
        preferred_turn[neutral_pref] = 1.0

        return {
            "critical_pairs": critical_pairs,
            "caution_pairs": caution_pairs,
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
        boundary_soft_risks: List[float],
        outside_depths: List[float],
        pair_metrics: dict,
        completed: bool,
        truncated: bool,
    ) -> float:
        waypoint_term = self.reward_waypoint_hit * float(waypoint_hits / max(len(current_agents), 1))
        completion_term = self.reward_completion_bonus if completed else 0.0
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
        geofence_breach_term = -self.penalty_geofence_breach * float(
            np.mean(
                [
                    1.0 if float(depth) > 0.0 else 0.0
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
        crash_term = (
            -self.penalty_crash
            if self.terminate_on_critical_violation and pair_metrics["critical_pairs"]
            else 0.0
        )
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
            "geofence_penalty": geofence_term,
            "geofence_breach_penalty": geofence_breach_term,
            "boundary_soft_penalty": boundary_soft_term,
            "crash_penalty": crash_term,
            "harsh_turn_penalty": harsh_turn_term,
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

