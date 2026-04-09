import math

from src.heat_map_updates.heatmap_updates import DECAY_REFERENCE_SECONDS
from src.planner_folder.geometry import build_camera


HEADING_SUBDIVISIONS = 8
ANGLE_EPSILON = 1e-9


def normalize_heading_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def speed_to_step_distance(drone_speed_mps, dt_seconds=DECAY_REFERENCE_SECONDS):
    return max(0.0, float(drone_speed_mps)) * max(0.0, float(dt_seconds))


def turn_rate_to_step_delta(turn_rate_deg_per_second, dt_seconds=DECAY_REFERENCE_SECONDS):
    return max(0.0, float(turn_rate_deg_per_second)) * max(0.0, float(dt_seconds))


def _ordered_heading_steps(desired_step, max_turn_step_deg):
    if max_turn_step_deg <= 0.0:
        return [float(desired_step)]

    increment = float(max_turn_step_deg) / HEADING_SUBDIVISIONS
    grid_steps = [float(desired_step)]
    for grid_index in range(0, (2 * HEADING_SUBDIVISIONS) + 1):
        grid_steps.append(-float(max_turn_step_deg) + grid_index * increment)

    return sorted(
        {float(step) for step in grid_steps},
        key=lambda step: (abs(float(step) - float(desired_step)), abs(float(step))),
    )


def _heading_unit_vector(candidate_heading):
    heading_rad = math.radians(float(candidate_heading))
    return math.sin(heading_rad), math.cos(heading_rad)


def _max_in_bounds_step(state, candidate_heading, move_step, map_bounds):
    if map_bounds is None:
        return max(0.0, float(move_step))

    min_e, max_e, min_n, max_n = map_bounds
    step_limit = max(0.0, float(move_step))
    delta_e, delta_n = _heading_unit_vector(candidate_heading)
    start_e = float(state["e"])
    start_n = float(state["n"])

    if delta_e > ANGLE_EPSILON:
        step_limit = min(step_limit, (float(max_e) - start_e) / delta_e)
    elif delta_e < -ANGLE_EPSILON:
        step_limit = min(step_limit, (float(min_e) - start_e) / delta_e)

    if delta_n > ANGLE_EPSILON:
        step_limit = min(step_limit, (float(max_n) - start_n) / delta_n)
    elif delta_n < -ANGLE_EPSILON:
        step_limit = min(step_limit, (float(min_n) - start_n) / delta_n)

    return max(0.0, float(step_limit))


def _choose_lateral_sign(target_component, heading_component, positive_clearance, negative_clearance):
    if abs(float(target_component)) > ANGLE_EPSILON:
        return 1.0 if float(target_component) > 0.0 else -1.0
    if abs(float(heading_component)) > ANGLE_EPSILON:
        return 1.0 if float(heading_component) > 0.0 else -1.0
    return 1.0 if float(positive_clearance) >= float(negative_clearance) else -1.0


def boundary_turn_heading(state, desired_heading, map_bounds, trigger_distance):
    if map_bounds is None:
        return None

    min_e, max_e, min_n, max_n = map_bounds
    easting = float(state["e"])
    northing = float(state["n"])
    trigger_distance = max(float(trigger_distance), 0.0)
    if trigger_distance <= ANGLE_EPSILON:
        return None

    left_clearance = easting - float(min_e)
    right_clearance = float(max_e) - easting
    bottom_clearance = northing - float(min_n)
    top_clearance = float(max_n) - northing

    def edge_weight(clearance):
        if clearance > trigger_distance:
            return 0.0
        return max(0.0, 1.0 - (float(clearance) / trigger_distance))

    left_weight = edge_weight(left_clearance)
    right_weight = edge_weight(right_clearance)
    bottom_weight = edge_weight(bottom_clearance)
    top_weight = edge_weight(top_clearance)
    inward_e = left_weight - right_weight
    inward_n = bottom_weight - top_weight

    if abs(inward_e) <= ANGLE_EPSILON and abs(inward_n) <= ANGLE_EPSILON:
        return None

    desired_e, desired_n = _heading_unit_vector(desired_heading)
    current_e, current_n = _heading_unit_vector(state["heading"])

    def choose_sign(desired_component, current_component, positive_clearance, negative_clearance):
        return _choose_lateral_sign(
            desired_component,
            current_component,
            positive_clearance,
            negative_clearance,
        )

    escape_e = 3.0 * inward_e
    escape_n = 3.0 * inward_n

    lateral_weight_e = max(left_weight, right_weight)
    lateral_weight_n = max(bottom_weight, top_weight)

    if lateral_weight_e > ANGLE_EPSILON:
        tangent_sign = choose_sign(desired_n, current_n, top_clearance, bottom_clearance)
        escape_n += 1.5 * lateral_weight_e * tangent_sign

    if lateral_weight_n > ANGLE_EPSILON:
        tangent_sign = choose_sign(desired_e, current_e, right_clearance, left_clearance)
        escape_e += 1.5 * lateral_weight_n * tangent_sign

    if abs(escape_e) <= ANGLE_EPSILON and abs(escape_n) <= ANGLE_EPSILON:
        return None

    return math.degrees(math.atan2(escape_e, escape_n))


def _choose_bounded_motion(state, desired_heading, move_step, max_turn_step_deg, map_bounds):
    current_heading = float(state["heading"])
    desired_step = max(
        -max_turn_step_deg,
        min(max_turn_step_deg, normalize_heading_deg(desired_heading - current_heading)),
    )
    ordered_steps = _ordered_heading_steps(desired_step, max_turn_step_deg)
    best_partial_heading = None
    best_partial_key = None

    for heading_step in ordered_steps:
        candidate_heading = normalize_heading_deg(current_heading + float(heading_step))
        candidate_step = _max_in_bounds_step(
            state,
            candidate_heading,
            move_step,
            map_bounds,
        )
        if candidate_step >= float(move_step) - ANGLE_EPSILON:
            return candidate_heading, float(move_step)
        if candidate_step <= ANGLE_EPSILON:
            continue

        partial_key = (
            float(candidate_step),
            -abs(float(heading_step) - float(desired_step)),
            -abs(float(heading_step)),
        )
        if best_partial_key is None or partial_key > best_partial_key:
            best_partial_heading = candidate_heading
            best_partial_key = partial_key

    if best_partial_heading is not None and best_partial_key is not None:
        return best_partial_heading, float(best_partial_key[0])

    if ordered_steps:
        return normalize_heading_deg(current_heading + float(ordered_steps[0])), 0.0

    return current_heading, 0.0


def _advance_toward_target_resolved(
    state,
    target_utm,
    max_move_step,
    max_turn_step_deg,
    map_bounds,
):
    vec_e = float(target_utm[0]) - float(state["e"])
    vec_n = float(target_utm[1]) - float(state["n"])
    distance = float(math.hypot(vec_e, vec_n))
    if distance <= 1e-6:
        return state

    move_step = min(max_move_step, distance)
    desired_heading = math.degrees(math.atan2(vec_e, vec_n))
    if map_bounds is None:
        heading_delta = normalize_heading_deg(desired_heading - state["heading"])
        heading_step = max(-max_turn_step_deg, min(max_turn_step_deg, heading_delta))
        state["heading"] = normalize_heading_deg(state["heading"] + heading_step)
        bounded_move_step = move_step
    else:
        state["heading"], bounded_move_step = _choose_bounded_motion(
            state,
            desired_heading,
            move_step,
            max_turn_step_deg,
            map_bounds,
        )

    delta_e, delta_n = _heading_unit_vector(state["heading"])
    state["e"] += delta_e * bounded_move_step
    state["n"] += delta_n * bounded_move_step
    return state


def advance_toward_target(
    start_state,
    target_utm,
    drone_speed,
    max_turn_rate_deg,
    map_bounds=None,
    step_seconds=DECAY_REFERENCE_SECONDS,
):
    state = dict(start_state)
    return _advance_toward_target_resolved(
        state,
        target_utm,
        speed_to_step_distance(drone_speed, step_seconds),
        turn_rate_to_step_delta(max_turn_rate_deg, step_seconds),
        map_bounds,
    )


def simulate_motion_path(
    start_state,
    target_utm,
    drone_speed,
    max_turn_rate_deg,
    max_steps=64,
    map_bounds=None,
    step_seconds=DECAY_REFERENCE_SECONDS,
):
    state = dict(start_state)
    samples = []
    max_move_step = speed_to_step_distance(drone_speed, step_seconds)
    max_turn_step_deg = turn_rate_to_step_delta(max_turn_rate_deg, step_seconds)

    for _ in range(max_steps):
        next_state = _advance_toward_target_resolved(
            dict(state),
            target_utm,
            max_move_step,
            max_turn_step_deg,
            map_bounds,
        )
        if (
            abs(next_state["e"] - state["e"]) <= ANGLE_EPSILON
            and abs(next_state["n"] - state["n"]) <= ANGLE_EPSILON
            and abs(next_state["heading"] - state["heading"]) <= ANGLE_EPSILON
        ):
            break
        samples.append(dict(next_state))
        state = next_state

    return samples


def apply_state_transition_sweep(
    hmu,
    camera_config,
    start_state,
    end_state,
    step_distance,
    start_pitch_deg=None,
    end_pitch_deg=None,
    start_yaw_deg=None,
    end_yaw_deg=None,
    dt_seconds=DECAY_REFERENCE_SECONDS,
):
    vec_e = end_state["e"] - start_state["e"]
    vec_n = end_state["n"] - start_state["n"]
    distance = float(math.hypot(vec_e, vec_n))
    steps = max(1, int(math.ceil(distance / max(step_distance, 1e-6))))

    heading_delta = normalize_heading_deg(end_state["heading"] - start_state["heading"])
    if start_pitch_deg is None:
        start_pitch_deg = camera_config.pitch
    if end_pitch_deg is None:
        end_pitch_deg = camera_config.pitch
    if start_yaw_deg is None:
        start_yaw_deg = camera_config.yaw
    if end_yaw_deg is None:
        end_yaw_deg = camera_config.yaw
    start_agl = float(start_state.get("agl", camera_config.agl))
    end_agl = float(end_state.get("agl", start_agl))

    last_proj = None
    substep_dt_seconds = max(0.0, float(dt_seconds)) / steps
    for idx in range(1, steps + 1):
        frac = idx / steps
        heading = normalize_heading_deg(start_state["heading"] + heading_delta * frac)
        easting = start_state["e"] + vec_e * frac
        northing = start_state["n"] + vec_n * frac
        pitch = start_pitch_deg + (end_pitch_deg - start_pitch_deg) * frac
        yaw = start_yaw_deg + (end_yaw_deg - start_yaw_deg) * frac
        agl = start_agl + (end_agl - start_agl) * frac
        proj = build_camera(
            camera_config,
            (easting, northing),
            heading,
            pitch=pitch,
            yaw=yaw,
            agl=agl,
        ).project()
        hmu.change_to_zeroes(proj, dt_seconds=substep_dt_seconds)
        last_proj = proj

    if last_proj is None:
        proj = build_camera(
            camera_config,
            (end_state["e"], end_state["n"]),
            end_state["heading"],
            pitch=end_pitch_deg,
            yaw=end_yaw_deg,
            agl=end_agl,
        ).project()
        hmu.change_to_zeroes(proj, dt_seconds=dt_seconds)
        return proj

    return last_proj
