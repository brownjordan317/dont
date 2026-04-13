from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


Pose2D = Tuple[float, float, float]


def heading_to_yaw(heading_rad: float) -> float:
    return _mod2pi((math.pi * 0.5) - float(heading_rad))


def build_centered_dubins_route(
    *,
    start_position_local: Tuple[float, float],
    start_heading_rad: float,
    waypoint_positions_local: Sequence[Tuple[float, float]],
    turn_radius_m: float,
    sample_step_m: Optional[float] = None,
    bounds_local: Optional[Tuple[float, float, float, float]] = None,
) -> List[Tuple[float, float]]:
    start_xy = np.asarray(start_position_local, dtype=np.float64)
    if len(waypoint_positions_local) == 0:
        return [(float(start_xy[0]), float(start_xy[1]))]

    turn_radius_m = max(float(turn_radius_m), 1.0)
    if sample_step_m is None:
        sample_step_m = max(1.5, min(5.0, turn_radius_m / 10.0))
    sample_step_m = max(float(sample_step_m), 0.5)

    waypoint_points = [
        np.asarray(point, dtype=np.float64)
        for point in waypoint_positions_local
    ]
    waypoint_yaws = _compute_waypoint_yaws(
        start_xy=start_xy,
        start_heading_rad=float(start_heading_rad),
        waypoint_points=waypoint_points,
    )

    route_points: List[Tuple[float, float]] = [
        (float(start_xy[0]), float(start_xy[1]))
    ]
    current_pose: Pose2D = (
        float(start_xy[0]),
        float(start_xy[1]),
        heading_to_yaw(float(start_heading_rad)),
    )

    for waypoint_xy, waypoint_yaw in zip(waypoint_points, waypoint_yaws):
        target_pose: Pose2D = (
            float(waypoint_xy[0]),
            float(waypoint_xy[1]),
            float(waypoint_yaw),
        )
        segment_points = _sample_variable_radius_dubins_path(
            start_pose=current_pose,
            end_pose=target_pose,
            min_turn_radius_m=turn_radius_m,
            sample_step_m=sample_step_m,
            bounds_local=bounds_local,
        )
        if not segment_points:
            segment_points = [(float(waypoint_xy[0]), float(waypoint_xy[1]))]
        else:
            segment_points[-1] = (float(waypoint_xy[0]), float(waypoint_xy[1]))

        route_points.extend(segment_points[1:])
        current_pose = target_pose

    return route_points


def _compute_waypoint_yaws(
    *,
    start_xy: np.ndarray,
    start_heading_rad: float,
    waypoint_points: Sequence[np.ndarray],
) -> List[float]:
    if not waypoint_points:
        return []

    waypoint_yaws: List[float] = []
    prev_point = np.asarray(start_xy, dtype=np.float64)
    prev_direction = np.asarray(
        [
            math.cos(heading_to_yaw(start_heading_rad)),
            math.sin(heading_to_yaw(start_heading_rad)),
        ],
        dtype=np.float64,
    )
    prev_direction = _normalize(prev_direction)
    if prev_direction is None:
        prev_direction = np.asarray([1.0, 0.0], dtype=np.float64)

    for waypoint_idx, waypoint_xy in enumerate(waypoint_points):
        inbound_direction = _normalize(waypoint_xy - prev_point)
        if inbound_direction is None:
            inbound_direction = prev_direction

        if waypoint_idx < (len(waypoint_points) - 1):
            outbound_direction = _normalize(
                waypoint_points[waypoint_idx + 1] - waypoint_xy
            )
            if outbound_direction is None:
                outbound_direction = inbound_direction

            bisector = _normalize(inbound_direction + outbound_direction)
            if bisector is None:
                waypoint_direction = outbound_direction
            else:
                waypoint_direction = bisector
                if float(np.dot(waypoint_direction, outbound_direction)) < 0.0:
                    waypoint_direction = -waypoint_direction
        else:
            waypoint_direction = inbound_direction

        waypoint_yaws.append(
            math.atan2(float(waypoint_direction[1]), float(waypoint_direction[0]))
        )
        prev_point = waypoint_xy
        prev_direction = waypoint_direction

    return waypoint_yaws


def _sample_variable_radius_dubins_path(
    *,
    start_pose: Pose2D,
    end_pose: Pose2D,
    min_turn_radius_m: float,
    sample_step_m: float,
    bounds_local: Optional[Tuple[float, float, float, float]] = None,
) -> List[Tuple[float, float]]:
    start_xy = np.asarray(start_pose[:2], dtype=np.float64)
    end_xy = np.asarray(end_pose[:2], dtype=np.float64)
    straight_distance = float(np.linalg.norm(end_xy - start_xy))
    max_turn_radius_m = max(
        float(min_turn_radius_m),
        0.45 * straight_distance,
    )

    radius_candidates = np.linspace(
        max_turn_radius_m,
        float(min_turn_radius_m),
        num=12,
        dtype=np.float64,
    )
    simple_candidate: Optional[Tuple[float, str, Tuple[float, float, float]]] = None
    shortest_candidate: Optional[Tuple[float, str, Tuple[float, float, float], float]] = None
    max_simple_length = max(
        straight_distance * 2.25,
        float(min_turn_radius_m) * math.pi * 3.0,
    )

    for candidate_radius in radius_candidates:
        radius_value = max(float(candidate_radius), float(min_turn_radius_m))
        candidate = _shortest_dubins_candidate(
            start_pose=start_pose,
            end_pose=end_pose,
            turn_radius_m=radius_value,
        )
        if candidate is None:
            continue
        modes, segment_lengths = candidate
        total_length = radius_value * float(sum(segment_lengths))
        sampled_points = _sample_dubins_candidate(
            start_pose=start_pose,
            modes=modes,
            segment_lengths=segment_lengths,
            turn_radius_m=radius_value,
            sample_step_m=sample_step_m,
        )
        if bounds_local is not None and not _points_within_bounds(
            sampled_points,
            bounds_local=bounds_local,
        ):
            continue
        if shortest_candidate is None or total_length < shortest_candidate[3]:
            shortest_candidate = (
                radius_value,
                modes,
                segment_lengths,
                total_length,
                sampled_points,
            )
        if "RLR" == modes or "LRL" == modes:
            continue
        if total_length <= max_simple_length:
            simple_candidate = (radius_value, modes, segment_lengths, sampled_points)
            break

    if simple_candidate is not None:
        _, _, _, sampled_points = simple_candidate
        return sampled_points

    if shortest_candidate is None:
        return _sample_straight_line(
            start_xy=(start_pose[0], start_pose[1]),
            end_xy=(end_pose[0], end_pose[1]),
            sample_step_m=sample_step_m,
        )

    _, _, _, _, sampled_points = shortest_candidate
    return sampled_points


def _points_within_bounds(
    points: Sequence[Tuple[float, float]],
    *,
    bounds_local: Tuple[float, float, float, float],
) -> bool:
    min_x, max_x, min_y, max_y = bounds_local
    for x_pos, y_pos in points:
        if x_pos < min_x or x_pos > max_x or y_pos < min_y or y_pos > max_y:
            return False
    return True


def _shortest_dubins_candidate(
    *,
    start_pose: Pose2D,
    end_pose: Pose2D,
    turn_radius_m: float,
) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    start_x, start_y, start_yaw = start_pose
    end_x, end_y, end_yaw = end_pose
    dx = float(end_x - start_x)
    dy = float(end_y - start_y)
    scaled_distance = float(math.hypot(dx, dy) / max(turn_radius_m, 1e-6))
    if scaled_distance <= 1e-9:
        delta_yaw = abs(_wrap_pi(end_yaw - start_yaw))
        if delta_yaw <= 1e-6:
            return ("LSL", (0.0, 0.0, 0.0))

    theta = _mod2pi(math.atan2(dy, dx))
    alpha = _mod2pi(start_yaw - theta)
    beta = _mod2pi(end_yaw - theta)

    candidates: List[Tuple[str, Tuple[float, float, float]]] = []
    for mode_name, builder in (
        ("LSL", _build_lsl),
        ("RSR", _build_rsr),
        ("LSR", _build_lsr),
        ("RSL", _build_rsl),
        ("RLR", _build_rlr),
        ("LRL", _build_lrl),
    ):
        lengths = builder(alpha, beta, scaled_distance)
        if lengths is not None:
            candidates.append((mode_name, lengths))

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: float(sum(item[1])),
    )


def _sample_dubins_candidate(
    *,
    start_pose: Pose2D,
    modes: str,
    segment_lengths: Tuple[float, float, float],
    turn_radius_m: float,
    sample_step_m: float,
) -> List[Tuple[float, float]]:
    x_pos, y_pos, yaw = start_pose
    points: List[Tuple[float, float]] = [(float(x_pos), float(y_pos))]

    for mode, normalized_length in zip(modes, segment_lengths):
        segment_length_m = float(normalized_length) * float(turn_radius_m)
        remaining = max(segment_length_m, 0.0)
        while remaining > 1e-9:
            step_m = min(sample_step_m, remaining)
            if mode == "S":
                x_pos += step_m * math.cos(yaw)
                y_pos += step_m * math.sin(yaw)
            else:
                turn_sign = 1.0 if mode == "L" else -1.0
                delta_yaw = turn_sign * step_m / max(turn_radius_m, 1e-6)
                mid_yaw = yaw + (0.5 * delta_yaw)
                x_pos += step_m * math.cos(mid_yaw)
                y_pos += step_m * math.sin(mid_yaw)
                yaw = _mod2pi(yaw + delta_yaw)
            remaining -= step_m
            points.append((float(x_pos), float(y_pos)))

    return points


def _sample_straight_line(
    *,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    sample_step_m: float,
) -> List[Tuple[float, float]]:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    total_distance = float(np.linalg.norm(end - start))
    if total_distance <= 1e-9:
        return [start_xy]

    step_count = max(int(math.ceil(total_distance / max(sample_step_m, 1e-6))), 1)
    points = []
    for step_idx in range(step_count + 1):
        t_value = float(step_idx / step_count)
        point = ((1.0 - t_value) * start) + (t_value * end)
        points.append((float(point[0]), float(point[1])))
    return points


def _build_lsl(alpha: float, beta: float, distance: float) -> Optional[Tuple[float, float, float]]:
    sin_alpha = math.sin(alpha)
    sin_beta = math.sin(beta)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_alpha_beta = math.cos(alpha - beta)
    p_sq = (
        2.0
        + (distance * distance)
        - (2.0 * cos_alpha_beta)
        + (2.0 * distance * (sin_alpha - sin_beta))
    )
    if p_sq < 0.0:
        return None
    p_val = math.sqrt(p_sq)
    tmp = math.atan2(cos_beta - cos_alpha, distance + sin_alpha - sin_beta)
    return (
        _mod2pi(-alpha + tmp),
        p_val,
        _mod2pi(beta - tmp),
    )


def _build_rsr(alpha: float, beta: float, distance: float) -> Optional[Tuple[float, float, float]]:
    sin_alpha = math.sin(alpha)
    sin_beta = math.sin(beta)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_alpha_beta = math.cos(alpha - beta)
    p_sq = (
        2.0
        + (distance * distance)
        - (2.0 * cos_alpha_beta)
        + (2.0 * distance * (sin_beta - sin_alpha))
    )
    if p_sq < 0.0:
        return None
    p_val = math.sqrt(p_sq)
    tmp = math.atan2(cos_alpha - cos_beta, distance - sin_alpha + sin_beta)
    return (
        _mod2pi(alpha - tmp),
        p_val,
        _mod2pi(-beta + tmp),
    )


def _build_lsr(alpha: float, beta: float, distance: float) -> Optional[Tuple[float, float, float]]:
    sin_alpha = math.sin(alpha)
    sin_beta = math.sin(beta)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_alpha_beta = math.cos(alpha - beta)
    p_sq = (
        -2.0
        + (distance * distance)
        + (2.0 * cos_alpha_beta)
        + (2.0 * distance * (sin_alpha + sin_beta))
    )
    if p_sq < 0.0:
        return None
    p_val = math.sqrt(p_sq)
    tmp = (
        math.atan2(-cos_alpha - cos_beta, distance + sin_alpha + sin_beta)
        - math.atan2(-2.0, p_val)
    )
    return (
        _mod2pi(-alpha + tmp),
        p_val,
        _mod2pi(-beta + tmp),
    )


def _build_rsl(alpha: float, beta: float, distance: float) -> Optional[Tuple[float, float, float]]:
    sin_alpha = math.sin(alpha)
    sin_beta = math.sin(beta)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_alpha_beta = math.cos(alpha - beta)
    p_sq = (
        -2.0
        + (distance * distance)
        + (2.0 * cos_alpha_beta)
        - (2.0 * distance * (sin_alpha + sin_beta))
    )
    if p_sq < 0.0:
        return None
    p_val = math.sqrt(p_sq)
    tmp = (
        math.atan2(cos_alpha + cos_beta, distance - sin_alpha - sin_beta)
        - math.atan2(2.0, p_val)
    )
    return (
        _mod2pi(alpha - tmp),
        p_val,
        _mod2pi(beta - tmp),
    )


def _build_rlr(alpha: float, beta: float, distance: float) -> Optional[Tuple[float, float, float]]:
    sin_alpha = math.sin(alpha)
    sin_beta = math.sin(beta)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_alpha_beta = math.cos(alpha - beta)
    tmp = (
        6.0
        - (distance * distance)
        + (2.0 * cos_alpha_beta)
        + (2.0 * distance * (sin_alpha - sin_beta))
    ) / 8.0
    if abs(tmp) > 1.0:
        return None
    p_val = _mod2pi((2.0 * math.pi) - math.acos(tmp))
    t_val = _mod2pi(
        alpha
        - math.atan2(cos_alpha - cos_beta, distance - sin_alpha + sin_beta)
        + (0.5 * p_val)
    )
    return (
        t_val,
        p_val,
        _mod2pi(alpha - beta - t_val + p_val),
    )


def _build_lrl(alpha: float, beta: float, distance: float) -> Optional[Tuple[float, float, float]]:
    sin_alpha = math.sin(alpha)
    sin_beta = math.sin(beta)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_alpha_beta = math.cos(alpha - beta)
    tmp = (
        6.0
        - (distance * distance)
        + (2.0 * cos_alpha_beta)
        + (2.0 * distance * (sin_beta - sin_alpha))
    ) / 8.0
    if abs(tmp) > 1.0:
        return None
    p_val = _mod2pi((2.0 * math.pi) - math.acos(tmp))
    t_val = _mod2pi(
        -alpha
        - math.atan2(cos_alpha - cos_beta, distance + sin_alpha - sin_beta)
        + (0.5 * p_val)
    )
    return (
        t_val,
        p_val,
        _mod2pi(beta - alpha - t_val + p_val),
    )


def _normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return np.asarray(vector / norm, dtype=np.float64)


def _wrap_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _mod2pi(angle_rad: float) -> float:
    return float(angle_rad % (2.0 * math.pi))
