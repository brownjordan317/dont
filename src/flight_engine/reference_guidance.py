from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from flight_engine.dubins import build_centered_dubins_route
from flight_engine.helpers import wrap_angle
from flight_engine.navigation_utils import (
    nearest_point_on_polyline,
    point_at_distance_on_polyline,
    waypoint_positions_local,
)
from flight_engine.simulator import FixedWingAircraft
from flight_engine.trans_coorders import CoordinateTransformer


@dataclass
class ReferenceRoute:
    points: np.ndarray
    current_wp_along: float
    total_length: float


def build_reference_route_local(
    *,
    cached_route: Optional[ReferenceRoute],
    aircraft: FixedWingAircraft,
    transformer: CoordinateTransformer,
    segment_start_local: np.ndarray,
    segment_start_heading: float,
    bounds_local: Tuple[float, float, float, float],
) -> Optional[ReferenceRoute]:
    if cached_route is not None:
        return cached_route

    waypoint_positions = waypoint_positions_local(aircraft, transformer)
    if not waypoint_positions:
        return None

    route_points = build_centered_dubins_route(
        start_position_local=(
            float(segment_start_local[0]),
            float(segment_start_local[1]),
        ),
        start_heading_rad=float(segment_start_heading),
        waypoint_positions_local=waypoint_positions,
        turn_radius_m=float(aircraft.dynamics.turning_radius),
        sample_step_m=max(
            1.5,
            min(8.0, float(aircraft.dynamics.turning_radius) * 0.2),
        ),
        bounds_local=tuple(float(value) for value in bounds_local),
    )
    points = np.asarray(route_points, dtype=np.float32)
    current_wp_local = np.asarray(waypoint_positions[0], dtype=np.float32)
    _, current_wp_along, _, total_length = nearest_point_on_polyline(
        points,
        current_wp_local,
    )
    return ReferenceRoute(
        points=points,
        current_wp_along=float(current_wp_along),
        total_length=float(total_length),
    )


def guidance_lookahead_distance(
    *,
    cruise_speed: float,
    lookahead_time_s: float,
    min_lookahead_m: float,
    max_lookahead_m: float,
) -> float:
    return float(
        np.clip(
            cruise_speed * lookahead_time_s,
            min_lookahead_m,
            max_lookahead_m,
        )
    )


def guidance_turn_action_from_vector(
    *,
    current_heading: float,
    target_vec: np.ndarray,
    cruise_speed: float,
    turning_radius: float,
    max_turn_rate: float,
    arrival_threshold: float,
    dt: float,
    turn_gain: float,
    turn_lookahead_scale: float,
    turn_radius_floor_scale: float,
) -> float:
    distance_to_target = float(np.linalg.norm(target_vec))
    if distance_to_target <= 1e-6:
        return 0.0

    target_bearing = float(np.arctan2(target_vec[0], target_vec[1]))
    heading_error = wrap_angle(target_bearing - current_heading)
    turn_sine = float(np.sin(heading_error))
    if float(np.cos(heading_error)) < -0.25 and abs(turn_sine) < 0.1:
        # A target almost exactly behind a fixed-wing aircraft is a singular
        # pure-pursuit case: either turn direction is valid, but zero turn is not.
        turn_sine = 1.0
    lookahead_distance = max(
        distance_to_target * turn_lookahead_scale,
        float(turning_radius) * turn_radius_floor_scale,
        float(cruise_speed) * dt * 2.0,
        float(arrival_threshold),
    )
    desired_turn_rate = float(
        turn_gain
        * 2.0
        * float(cruise_speed)
        * turn_sine
        / max(lookahead_distance, 1e-6)
    )
    return float(
        np.clip(
            desired_turn_rate / max(float(max_turn_rate), 1e-6),
            -1.0,
            1.0,
        )
    )


def compute_reference_action(
    *,
    pos_local: np.ndarray,
    wp_local: np.ndarray,
    route: Optional[ReferenceRoute],
    route_progress_anchor: float,
    arrival_threshold: float,
    current_heading: float,
    cruise_speed: float,
    turning_radius: float,
    max_turn_rate: float,
    dt: float,
    lookahead_time_s: float,
    min_lookahead_m: float,
    max_lookahead_m: float,
    route_commit_scale: float,
    turn_gain: float,
    turn_lookahead_scale: float,
    turn_radius_floor_scale: float,
) -> Tuple[float, float]:
    updated_anchor = float(route_progress_anchor)

    if route is not None and len(route.points) >= 2:
        current_wp_along = float(route.current_wp_along)
        total_length = float(route.total_length)
        backtrack_allowance = max(
            float(arrival_threshold) * 0.5,
            float(turning_radius) * 0.35,
            float(cruise_speed) * dt * 2.0,
        )
        min_along = max(0.0, float(route_progress_anchor) - backtrack_allowance)
        max_along = min(
            float(current_wp_along) + float(arrival_threshold),
            float(total_length),
        )
        _, nearest_along, _, _ = nearest_point_on_polyline(
            route.points,
            pos_local,
            min_along=min_along,
            max_along=max_along,
        )
        updated_anchor = min(
            max(float(route_progress_anchor), float(nearest_along)),
            float(current_wp_along),
        )
        lookahead = guidance_lookahead_distance(
            cruise_speed=cruise_speed,
            lookahead_time_s=lookahead_time_s,
            min_lookahead_m=min_lookahead_m,
            max_lookahead_m=max_lookahead_m,
        )
        target_along = min(
            max(float(nearest_along), updated_anchor)
            + (lookahead * route_commit_scale),
            float(current_wp_along),
        )
        nav_target = point_at_distance_on_polyline(route.points, target_along)
    else:
        nav_target = np.asarray(wp_local, dtype=np.float32)
        updated_anchor = 0.0

    if float(np.linalg.norm(wp_local - pos_local)) <= float(arrival_threshold):
        nav_target = np.asarray(wp_local, dtype=np.float32)

    target_vec = np.asarray(nav_target - pos_local, dtype=np.float32)
    if float(np.linalg.norm(target_vec)) <= 1e-6:
        target_vec = np.asarray(wp_local - pos_local, dtype=np.float32)
    if float(np.linalg.norm(target_vec)) <= 1e-6:
        return 0.0, updated_anchor

    action = guidance_turn_action_from_vector(
        current_heading=current_heading,
        target_vec=target_vec,
        cruise_speed=cruise_speed,
        turning_radius=turning_radius,
        max_turn_rate=max_turn_rate,
        arrival_threshold=arrival_threshold,
        dt=dt,
        turn_gain=turn_gain,
        turn_lookahead_scale=turn_lookahead_scale,
        turn_radius_floor_scale=turn_radius_floor_scale,
    )
    return action, updated_anchor


def time_to_boundary_ahead(
    *,
    pos_local: np.ndarray,
    heading: float,
    cruise_speed: float,
    local_min_x: float,
    local_max_x: float,
    local_min_y: float,
    local_max_y: float,
    lookahead_time_s: float,
) -> float:
    pos_x = float(pos_local[0])
    pos_y = float(pos_local[1])
    vel_x = float(cruise_speed * np.sin(heading))
    vel_y = float(cruise_speed * np.cos(heading))
    candidates = []
    if vel_x > 1e-6:
        candidates.append((local_max_x - pos_x) / vel_x)
    elif vel_x < -1e-6:
        candidates.append((local_min_x - pos_x) / vel_x)
    if vel_y > 1e-6:
        candidates.append((local_max_y - pos_y) / vel_y)
    elif vel_y < -1e-6:
        candidates.append((local_min_y - pos_y) / vel_y)

    positive_times = [time_s for time_s in candidates if time_s >= 0.0]
    if not positive_times:
        return 2.0
    return float(
        np.clip(
            min(positive_times) / max(lookahead_time_s, 1e-6),
            0.0,
            2.0,
        )
    )


def turn_circle_feasibility_features(
    *,
    pos_local: np.ndarray,
    wp_local: np.ndarray,
    heading: float,
    turning_radius: float,
) -> Tuple[float, float]:
    turn_radius = max(float(turning_radius), 1.0)
    left_axis = np.asarray(
        [-np.cos(heading), np.sin(heading)],
        dtype=np.float32,
    )
    right_axis = np.asarray(
        [np.cos(heading), -np.sin(heading)],
        dtype=np.float32,
    )
    left_center = pos_local + (left_axis * turn_radius)
    right_center = pos_local + (right_axis * turn_radius)
    left_margin = (float(np.linalg.norm(wp_local - left_center)) - turn_radius) / turn_radius
    right_margin = (float(np.linalg.norm(wp_local - right_center)) - turn_radius) / turn_radius
    return (
        float(np.clip(left_margin, -2.0, 4.0)),
        float(np.clip(right_margin, -2.0, 4.0)),
    )


def dangerous_neighbor_turn_preview(
    *,
    own_pos_local: np.ndarray,
    other_pos_local: np.ndarray,
    own_heading: float,
    other_heading: float,
    own_cruise_speed: float,
    other_cruise_speed: float,
    own_max_turn_rate: float,
    caution_dist: float,
    dt: float,
) -> Tuple[float, float]:
    preview_time_s = float(np.clip(dt * 6.0, 1.0, 3.0))

    def preview_cpa_distance(turn_sign: float) -> float:
        predicted_heading = wrap_angle(
            own_heading + (turn_sign * own_max_turn_rate * preview_time_s)
        )
        own_vx = own_cruise_speed * np.sin(predicted_heading)
        own_vy = own_cruise_speed * np.cos(predicted_heading)
        other_vx = other_cruise_speed * np.sin(other_heading)
        other_vy = other_cruise_speed * np.cos(other_heading)
        rel_position = np.asarray(
            [
                float(other_pos_local[0] - own_pos_local[0]),
                float(other_pos_local[1] - own_pos_local[1]),
            ],
            dtype=np.float32,
        )
        rel_velocity = np.asarray(
            [other_vx - own_vx, other_vy - own_vy],
            dtype=np.float32,
        )
        speed_sq = float(np.dot(rel_velocity, rel_velocity))
        if speed_sq <= 1e-6:
            return float(np.linalg.norm(rel_position))
        t_cpa = float(
            np.clip(
                -np.dot(rel_position, rel_velocity) / speed_sq,
                0.0,
                preview_time_s,
            )
        )
        return float(np.linalg.norm(rel_position + (rel_velocity * t_cpa)))

    baseline_cpa = preview_cpa_distance(0.0)
    left_cpa = preview_cpa_distance(1.0)
    right_cpa = preview_cpa_distance(-1.0)
    scale = max(caution_dist, 1.0)
    return (
        float(np.clip((left_cpa - baseline_cpa) / scale, -2.0, 2.0)),
        float(np.clip((right_cpa - baseline_cpa) / scale, -2.0, 2.0)),
    )
