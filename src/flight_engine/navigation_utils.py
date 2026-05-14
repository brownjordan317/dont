from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from flight_engine.trans_coorders import CoordinateTransformer


def heading_to_radians(value: float) -> float:
    heading = float(value)
    if abs(heading) > (2 * np.pi):
        return float(np.deg2rad(heading))
    return heading


def build_rect_bounds(
    origin: Tuple[float, float],
    width_m: float,
    height_m: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    lat_off = (float(height_m) / 2.0) / 111_320.0
    lon_off = (float(width_m) / 2.0) / (
        111_320.0 * max(np.cos(np.radians(origin[0])), 1e-6)
    )
    top_left = (origin[0] + lat_off, origin[1] - lon_off)
    bottom_right = (origin[0] - lat_off, origin[1] + lon_off)
    return top_left, bottom_right


def build_box_bounds(
    origin: Tuple[float, float],
    box_size_m: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return build_rect_bounds(origin, box_size_m, box_size_m)


def local_box_geometry(
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
) -> tuple[CoordinateTransformer, float, float]:
    min_lat, max_lat = sorted([top_left[0], bottom_right[0]])
    min_lon, max_lon = sorted([top_left[1], bottom_right[1]])
    transformer = CoordinateTransformer(min_lat, min_lon)
    local_max_x, _ = transformer.geo_to_local(min_lat, max_lon)
    _, local_max_y = transformer.geo_to_local(max_lat, min_lon)
    return transformer, float(local_max_x), float(local_max_y)


def ordered_waypoint_tuples(aircraft) -> List[Tuple[float, float]]:
    mission_waypoints: List[Tuple[float, float]] = []
    current_waypoint = aircraft.waypoint_manager.current_waypoint
    if current_waypoint is not None:
        mission_waypoints.append(current_waypoint.to_tuple())
    mission_waypoints.extend(
        waypoint.to_tuple()
        for waypoint in aircraft.waypoint_manager.waypoint_queue
    )
    return mission_waypoints


def waypoint_positions_local(
    aircraft,
    transformer: CoordinateTransformer,
) -> List[Tuple[float, float]]:
    waypoints = ordered_waypoint_tuples(aircraft)
    if not waypoints:
        return []

    latitudes = [latitude for latitude, _ in waypoints]
    longitudes = [longitude for _, longitude in waypoints]
    x_vals, y_vals = transformer.geo_to_local(latitudes, longitudes)
    return [
        (float(x_value), float(y_value))
        for x_value, y_value in zip(x_vals, y_vals)
    ]


def route_distance_local(points_local: List[Tuple[float, float]]) -> float:
    if len(points_local) < 2:
        return 0.0
    coords = np.asarray(points_local, dtype=np.float32)
    return float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())


def planned_route_distance_m(
    aircraft,
    transformer: CoordinateTransformer,
    *,
    start_position: Optional[Tuple[float, float]] = None,
) -> float:
    if start_position is None:
        if hasattr(aircraft, "initial_pos"):
            start_position = aircraft.initial_pos.to_tuple()
        else:
            start_position = aircraft.position.to_tuple()

    points = [start_position, *ordered_waypoint_tuples(aircraft)]
    if len(points) < 2:
        return 0.0

    latitudes = [latitude for latitude, _ in points]
    longitudes = [longitude for _, longitude in points]
    x_vals, y_vals = transformer.geo_to_local(latitudes, longitudes)
    return route_distance_local(
        [
            (float(x_value), float(y_value))
            for x_value, y_value in zip(x_vals, y_vals)
        ]
    )


def nearest_point_on_polyline(
    polyline: np.ndarray,
    point_local: np.ndarray,
    *,
    min_along: float | None = None,
    max_along: float | None = None,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    if len(polyline) <= 1:
        single_point = np.asarray(polyline[0], dtype=np.float32)
        return (
            single_point,
            0.0,
            np.asarray([0.0, 1.0], dtype=np.float32),
            0.0,
        )

    best_point = np.asarray(polyline[0], dtype=np.float32)
    best_tangent = np.asarray([0.0, 1.0], dtype=np.float32)
    best_along = 0.0
    best_distance = np.inf
    cumulative_length = 0.0
    total_length = 0.0
    min_along = None if min_along is None else float(max(min_along, 0.0))
    max_along = None if max_along is None else float(max(max_along, 0.0))

    for seg_start, seg_end in zip(polyline[:-1], polyline[1:]):
        segment = np.asarray(seg_end - seg_start, dtype=np.float32)
        seg_length = float(np.linalg.norm(segment))
        if seg_length <= 1e-6:
            continue

        seg_start_along = cumulative_length
        seg_end_along = cumulative_length + seg_length
        allowed_start = seg_start_along if min_along is None else max(seg_start_along, min_along)
        allowed_end = seg_end_along if max_along is None else min(seg_end_along, max_along)
        if allowed_end < allowed_start:
            cumulative_length += seg_length
            total_length += seg_length
            continue

        seg_unit = segment / seg_length
        rel_point = np.asarray(point_local - seg_start, dtype=np.float32)
        projection = float(np.dot(rel_point, seg_unit))
        projection = float(
            np.clip(
                projection,
                allowed_start - seg_start_along,
                allowed_end - seg_start_along,
            )
        )
        closest = np.asarray(seg_start + (seg_unit * projection), dtype=np.float32)
        distance = float(np.linalg.norm(point_local - closest))
        if distance < best_distance:
            best_distance = distance
            best_point = closest
            best_tangent = seg_unit
            best_along = cumulative_length + projection
        cumulative_length += seg_length
        total_length += seg_length

    return best_point, best_along, best_tangent, max(total_length, 1e-6)


def point_at_distance_on_polyline(
    polyline: np.ndarray,
    distance_along: float,
) -> np.ndarray:
    if len(polyline) <= 1:
        return np.asarray(polyline[0], dtype=np.float32)

    remaining = float(max(distance_along, 0.0))
    for seg_start, seg_end in zip(polyline[:-1], polyline[1:]):
        segment = np.asarray(seg_end - seg_start, dtype=np.float32)
        seg_length = float(np.linalg.norm(segment))
        if seg_length <= 1e-6:
            continue
        if remaining <= seg_length:
            seg_unit = segment / seg_length
            return np.asarray(seg_start + (seg_unit * remaining), dtype=np.float32)
        remaining -= seg_length
    return np.asarray(polyline[-1], dtype=np.float32)


def order_route_points(
    *,
    start_local: np.ndarray,
    start_heading: float,
    points: np.ndarray,
    turn_radius: float,
    arrival_threshold: float,
    caution_dist: float,
    box_width_m: float,
    box_height_m: float,
) -> np.ndarray:
    candidates = [
        np.asarray(point, dtype=np.float32)
        for point in np.asarray(points, dtype=np.float32)
    ]
    if len(candidates) <= 1:
        return np.asarray(candidates, dtype=np.float32)

    current_pos = np.asarray(start_local, dtype=np.float32)
    current_dir = np.asarray(
        [np.sin(start_heading), np.cos(start_heading)],
        dtype=np.float32,
    )
    current_dir_norm = float(np.linalg.norm(current_dir))
    if current_dir_norm <= 1e-6:
        current_dir = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        current_dir = current_dir / current_dir_norm

    min_leg_length = min(
        max(turn_radius * 2.0, arrival_threshold * 3.0, caution_dist),
        0.45 * min(box_width_m, box_height_m),
    )
    ordered: List[np.ndarray] = []
    remaining = list(candidates)

    while remaining:
        best_idx = 0
        best_cost = np.inf
        best_dir = current_dir
        for candidate_idx, candidate in enumerate(remaining):
            leg_vec = np.asarray(candidate - current_pos, dtype=np.float32)
            leg_dist = float(np.linalg.norm(leg_vec))
            if leg_dist <= 1e-6:
                continue
            leg_unit = leg_vec / leg_dist
            turn_angle = float(
                abs(
                    np.arctan2(
                        (current_dir[0] * leg_unit[1]) - (current_dir[1] * leg_unit[0]),
                        float(np.dot(current_dir, leg_unit)),
                    )
                )
            )
            reversal_pressure = float(np.clip(-np.dot(current_dir, leg_unit), 0.0, 1.0))
            shortfall = max(min_leg_length - leg_dist, 0.0)
            cost = (
                leg_dist
                + (turn_radius * 1.35 * turn_angle)
                + (turn_radius * 1.75 * reversal_pressure)
                + (2.25 * shortfall)
            )
            if len(remaining) > 1:
                follow_dists = [
                    float(np.linalg.norm(other - candidate))
                    for other_idx, other in enumerate(remaining)
                    if other_idx != candidate_idx
                ]
                if follow_dists:
                    nearest_follow = min(follow_dists)
                    follow_shortfall = max((0.8 * min_leg_length) - nearest_follow, 0.0)
                    cost += 1.5 * follow_shortfall
            if cost < best_cost:
                best_cost = cost
                best_idx = candidate_idx
                best_dir = leg_unit

        chosen = remaining.pop(best_idx)
        ordered.append(chosen)
        current_pos = chosen
        current_dir = best_dir

    return np.asarray(ordered, dtype=np.float32)
