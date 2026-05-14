from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np

from flight_engine.helpers import wrap_angle
from flight_engine.simulator import FixedWingAircraft


class EpisodeInfoMixin:
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
                or self._termination_reason
                in {
                    "completed",
                    "critical_violation",
                    "geofence_violation",
                }
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
            reached, remaining, assigned = self._mission_waypoint_counts(aircraft)
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
