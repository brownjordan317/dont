import math

import numpy as np

from src.drone_controller.motion import (
    simulate_motion_path,
    speed_to_step_distance,
    turn_rate_to_step_delta,
)
from src.planner_folder.geometry import build_camera, projection_roi_mask, projection_span_meters


DEFAULT_BOUNDARY_MARGIN_FRACTION = 0.01
BOUNDARY_MARGIN_FRACTION = DEFAULT_BOUNDARY_MARGIN_FRACTION


def pick_cluster_target(data, cluster_bounds, resolution, height):
    row_min, col_min, row_max, col_max = cluster_bounds
    cluster = data[row_min:row_max, col_min:col_max]

    ys, xs = np.nonzero(cluster > 0)
    if xs.size == 0:
        return None

    xs = xs + col_min
    ys = ys + row_min

    weights = cluster[ys - row_min, xs - col_min].astype(np.float64, copy=False)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return None

    target_c = float((xs * weights).sum() / weight_sum)
    target_r = float((ys * weights).sum() / weight_sum)

    return {
        "target_c": target_c,
        "target_r": target_r,
        "target_e": target_c * resolution,
        "target_n": (height - target_r) * resolution,
    }


# ------------------------------------------------------------
# PATH RESERVATION SYSTEM
# ------------------------------------------------------------

def simulate_motion_prefixes(
    drone_state,
    target,
    planner,
    *,
    path_steps=15,
    predicted_steps=8,
    map_bounds=None,
):
    samples = simulate_motion_path(
        drone_state,
        (target["target_e"], target["target_n"]),
        planner.drone_speed,
        planner.max_turn_rate_deg,
        max_steps=max(path_steps, predicted_steps),
        map_bounds=map_bounds,
        step_seconds=planner.step_seconds,
    )
    return (
        [(state["e"], state["n"]) for state in samples[:path_steps]],
        samples[:predicted_steps],
    )


def path_conflicts_with_reserved(path, reserved_paths, planner):
    """Check spatial corridor conflict."""
    if not reserved_paths:
        return False

    corridor = planner.cluster_size * 1.5
    corridor_sq = corridor * corridor

    for e, n in path:
        for other_path in reserved_paths:
            for oe, on in other_path:
                dx = e - oe
                dy = n - on
                if dx * dx + dy * dy < corridor_sq:
                    return True

    return False


# ------------------------------------------------------------
# FAST PREDICTED COVERAGE
# ------------------------------------------------------------

def build_empty_reservation_heatmap(cluster_shape):
    return np.zeros(cluster_shape, dtype=np.float32)


def _coerce_cluster_heatmap(weights, cluster_shape):
    heatmap = build_empty_reservation_heatmap(cluster_shape)
    if weights is None:
        return heatmap

    if isinstance(weights, np.ndarray):
        rows = min(cluster_shape[0], weights.shape[0])
        cols = min(cluster_shape[1], weights.shape[1])
        if rows > 0 and cols > 0:
            heatmap[:rows, :cols] = weights[:rows, :cols]
        np.clip(heatmap, 0.0, 1.0, out=heatmap)
        return heatmap

    if isinstance(weights, dict):
        iterator = weights.items()
    else:
        iterator = ((block, 1.0) for block in weights)

    for block, value in iterator:
        if len(block) != 2:
            continue
        block_row = int(block[0])
        block_col = int(block[1])
        if 0 <= block_row < cluster_shape[0] and 0 <= block_col < cluster_shape[1]:
            heatmap[block_row, block_col] = max(
                heatmap[block_row, block_col],
                float(value),
            )

    np.clip(heatmap, 0.0, 1.0, out=heatmap)
    return heatmap


def merge_reservation_heatmap(reservation_heatmap, claim_heatmap):
    if reservation_heatmap is None:
        if claim_heatmap is None:
            return None
        return np.clip(np.asarray(claim_heatmap, dtype=np.float32), 0.0, 1.0)

    if claim_heatmap is None:
        return reservation_heatmap

    np.minimum(
        reservation_heatmap + np.asarray(claim_heatmap, dtype=np.float32),
        1.0,
        out=reservation_heatmap,
    )
    return reservation_heatmap


def claim_heatmap_to_block_set(claim_heatmap):
    if claim_heatmap is None:
        return set()
    rows, cols = np.nonzero(np.asarray(claim_heatmap, dtype=np.float32) > 0.0)
    return {
        (int(block_row), int(block_col))
        for block_row, block_col in zip(rows.tolist(), cols.tolist())
    }


def estimate_predicted_cluster_claim_heatmap(
    data_shape,
    camera_config,
    origin,
    resolution,
    cluster_size,
    predicted_states,
):
    block = max(1, int(cluster_size / max(resolution, 1e-6)))
    grid_rows = max(1, data_shape[0] // block)
    grid_cols = max(1, data_shape[1] // block)
    claim_heatmap = build_empty_reservation_heatmap((grid_rows, grid_cols))
    if not predicted_states:
        return claim_heatmap

    time_weights = np.arange(len(predicted_states), 0, -1, dtype=np.float32)
    time_weights /= float(time_weights.sum(dtype=np.float64))

    for state, step_weight in zip(predicted_states, time_weights.tolist()):
        proj = build_camera(
            camera_config,
            (state["e"], state["n"]),
            state["heading"],
            agl=state.get("agl", camera_config.agl),
        ).project()

        roi = projection_roi_mask(proj, origin, resolution, data_shape)
        if roi is None:
            continue

        row_min, row_max, col_min, col_max, _ = roi
        block_row_min = max(0, row_min // block)
        block_row_max = min(grid_rows - 1, row_max // block)
        block_col_min = max(0, col_min // block)
        block_col_max = min(grid_cols - 1, col_max // block)

        claim_heatmap[
            block_row_min:block_row_max + 1,
            block_col_min:block_col_max + 1,
        ] += float(step_weight)

    np.clip(claim_heatmap, 0.0, 1.0, out=claim_heatmap)
    return claim_heatmap


def estimate_coverage_scores(cluster_view, predicted_claim_heatmap, reservation_heatmap):
    if predicted_claim_heatmap is None:
        return 0.0, 0.0, 0.0

    claim_heatmap = _coerce_cluster_heatmap(
        predicted_claim_heatmap,
        cluster_view.shape,
    )
    if not np.any(claim_heatmap > 0.0):
        return 0.0, 0.0, 0.0

    reservation = _coerce_cluster_heatmap(
        reservation_heatmap,
        cluster_view.shape,
    )
    claim_weights = np.clip(claim_heatmap, 0.0, 1.0)
    overlap_weights = claim_weights * np.clip(reservation, 0.0, 1.0)
    unique_weights = claim_weights * np.clip(1.0 - reservation, 0.0, 1.0)

    values = np.asarray(cluster_view, dtype=np.float32)
    unique_gain = float((values * unique_weights).sum(dtype=np.float64))
    overlap_gain = float((values * overlap_weights).sum(dtype=np.float64))
    total_gain = float((values * claim_weights).sum(dtype=np.float64))
    return unique_gain, overlap_gain, total_gain


def target_has_clearance(target, reserved_targets, min_spacing):
    tx, ty = resolve_target_anchor_point(target)

    for reserved_target in reserved_targets:
        rx, ry = resolve_target_anchor_point(reserved_target)
        dx = tx - rx
        dy = ty - ry
        if math.hypot(dx, dy) < min_spacing:
            return False

    return True


def estimate_projection_hit_value(hmu, projection):
    roi = projection_roi_mask(
        projection,
        hmu.origin,
        hmu.resolution,
        hmu.data.shape,
    )
    if roi is None:
        return 0.0, 0.0

    row_min, row_max, col_min, col_max, mask = roi
    roi_values = hmu.data[row_min:row_max + 1, col_min:col_max + 1]
    hit_values = roi_values[mask]
    if hit_values.size == 0:
        return 0.0, 0.0

    return float(hit_values.sum()), float(hit_values.mean())


def estimate_target_travel_metrics(distance, footprint_span, planner):
    # The drone does not need to sit exactly on the hotspot center before its
    # footprint starts doing useful work. We treat a configurable fraction of
    # the camera footprint span as "usable" travel reduction so nearby targets
    # with similar value beat far-away targets more often.
    usable_radius = max(0.0, float(planner.target_radius_percent) * float(footprint_span))
    effective_distance = max(0.0, float(distance) - usable_radius)
    distance_scale = max(float(planner.cluster_size), 1.0)
    return usable_radius, effective_distance, distance_scale


def sample_heat_value_at_target(hmu, target_c, target_r):
    col = int(np.clip(np.floor(float(target_c)), 0, hmu.data.shape[1] - 1))
    row = int(np.clip(np.floor(float(target_r)), 0, hmu.data.shape[0] - 1))
    return float(hmu.data[row, col])


def build_target_ranking_key(
    unique_gain,
    overlap_gain,
    cluster_mean,
    effective_distance,
    distance_scale,
):
    net_gain = max(0.0, float(unique_gain) - 0.75 * float(overlap_gain))
    gain_efficiency = net_gain / (float(effective_distance) + float(distance_scale))
    return (
        gain_efficiency,
        net_gain,
        float(unique_gain),
        -float(overlap_gain),
        float(cluster_mean),
        -float(effective_distance),
    )


def score_cluster_target_candidate(cluster_view, candidate, reservation_heatmap):
    unique_gain, overlap_gain, route_gain = estimate_coverage_scores(
        cluster_view,
        candidate["claim_heatmap"],
        reservation_heatmap,
    )
    ranking_key = build_target_ranking_key(
        unique_gain,
        overlap_gain,
        candidate["cluster_mean"],
        candidate["effective_distance"],
        candidate["distance_scale"],
    )
    return ranking_key, unique_gain, overlap_gain, route_gain


def point_is_in_heatmap(easting, northing, resolution, data_shape):
    width_m = float(data_shape[1]) * float(resolution)
    height_m = float(data_shape[0]) * float(resolution)
    return 0.0 <= float(easting) <= width_m and 0.0 <= float(northing) <= height_m


def point_is_in_bounds(easting, northing, map_bounds):
    min_e, max_e, min_n, max_n = map_bounds
    return (
        float(min_e) <= float(easting) <= float(max_e)
        and float(min_n) <= float(northing) <= float(max_n)
    )


def build_target_fields(easting, northing, resolution, height):
    return {
        "target_e": float(easting),
        "target_n": float(northing),
        "target_c": float(easting) / float(resolution),
        "target_r": float(height) - (float(northing) / float(resolution)),
    }


def resolve_target_anchor_point(target):
    return (
        float(target.get("main_target_e", target["target_e"])),
        float(target.get("main_target_n", target["target_n"])),
    )


def distance_to_map_edge(easting, northing, map_bounds):
    min_e, max_e, min_n, max_n = map_bounds
    return float(
        min(
            float(easting) - float(min_e),
            float(max_e) - float(easting),
            float(northing) - float(min_n),
            float(max_n) - float(northing),
        )
    )


def _clip_point_to_bounds(easting, northing, map_bounds, *, margin=0.0):
    min_e, max_e, min_n, max_n = map_bounds
    margin = max(0.0, float(margin))
    max_margin_e = max(0.0, 0.5 * (float(max_e) - float(min_e)) - 1e-6)
    max_margin_n = max(0.0, 0.5 * (float(max_n) - float(min_n)) - 1e-6)
    margin_e = min(margin, max_margin_e)
    margin_n = min(margin, max_margin_n)
    return (
        float(np.clip(float(easting), float(min_e) + margin_e, float(max_e) - margin_e)),
        float(np.clip(float(northing), float(min_n) + margin_n, float(max_n) - margin_n)),
    )


def resolve_navigation_bounds(width_m, height_m, planner):
    width_m = float(width_m)
    height_m = float(height_m)
    step_distance = speed_to_step_distance(planner.drone_speed, planner.step_seconds)
    turn_rate_rad = math.radians(max(float(planner.max_turn_rate_deg), 0.0))
    if turn_rate_rad <= 1e-9 or float(planner.drone_speed) <= 1e-9:
        turn_radius = step_distance
    else:
        turn_radius = float(planner.drone_speed) / turn_rate_rad

    boundary_margin_fraction = float(
        getattr(planner, "boundary_margin_fraction", DEFAULT_BOUNDARY_MARGIN_FRACTION)
    )
    turn_radius_margin = boundary_margin_fraction * float(turn_radius)
    requested_margin = max(
        float(turn_radius) + float(turn_radius_margin),
        2.0 * float(step_distance),
    )
    max_margin = max(0.0, 0.5 * min(width_m, height_m) - 1e-6)
    margin = min(requested_margin, max_margin)
    return (
        margin,
        width_m - margin,
        margin,
        height_m - margin,
    )


def build_edge_approach_target(
    main_target,
    planner,
    resolution,
    height,
    map_bounds,
):
    main_e = float(main_target["main_target_e"])
    main_n = float(main_target["main_target_n"])
    edge_distance = distance_to_map_edge(main_e, main_n, map_bounds)
    if point_is_in_bounds(main_e, main_n, map_bounds):
        return None

    approach_e, approach_n = _clip_point_to_bounds(
        main_e,
        main_n,
        map_bounds,
    )
    min_e, max_e, min_n, max_n = map_bounds

    return {
        **build_target_fields(approach_e, approach_n, resolution, height),
        "edge_approach_target": True,
        "edge_margin_m": min(
            float(approach_e - min_e),
            float(max_e - approach_e),
            float(approach_n - min_n),
            float(max_n - approach_n),
        ),
        "main_target_edge_distance": edge_distance,
    }


def _projection_roi(projection, hmu):
    return projection_roi_mask(
        projection,
        hmu.origin,
        hmu.resolution,
        hmu.data.shape,
    )


def estimate_roi_overlap_ratio(candidate_roi, reserved_rois):
    if candidate_roi is None or not reserved_rois:
        return 0.0, 0.0

    row_min, row_max, col_min, col_max, mask = candidate_roi
    candidate_area = int(np.count_nonzero(mask))
    if candidate_area <= 0:
        return 0.0, 0.0

    total_overlap_ratio = 0.0
    max_overlap_ratio = 0.0

    for reserved_roi in reserved_rois:
        if reserved_roi is None:
            continue

        other_row_min, other_row_max, other_col_min, other_col_max, other_mask = reserved_roi
        overlap_row_min = max(row_min, other_row_min)
        overlap_row_max = min(row_max, other_row_max)
        overlap_col_min = max(col_min, other_col_min)
        overlap_col_max = min(col_max, other_col_max)
        if overlap_row_min > overlap_row_max or overlap_col_min > overlap_col_max:
            continue

        cand_row_slice = slice(
            overlap_row_min - row_min,
            overlap_row_max - row_min + 1,
        )
        cand_col_slice = slice(
            overlap_col_min - col_min,
            overlap_col_max - col_min + 1,
        )
        other_row_slice = slice(
            overlap_row_min - other_row_min,
            overlap_row_max - other_row_min + 1,
        )
        other_col_slice = slice(
            overlap_col_min - other_col_min,
            overlap_col_max - other_col_min + 1,
        )
        overlap_pixels = int(
            np.count_nonzero(
                mask[cand_row_slice, cand_col_slice]
                & other_mask[other_row_slice, other_col_slice]
            )
        )
        if overlap_pixels <= 0:
            continue

        overlap_ratio = overlap_pixels / float(candidate_area)
        total_overlap_ratio += overlap_ratio
        max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

    return min(total_overlap_ratio, 1.0), max_overlap_ratio


def _cluster_indices_from_point(easting, northing, resolution, height, block, cluster_shape):
    target_c = float(easting) / float(resolution)
    target_r = float(height) - (float(northing) / float(resolution))
    cluster_col = int(np.clip(np.floor(target_c / block), 0, cluster_shape[1] - 1))
    cluster_row = int(np.clip(np.floor(target_r / block), 0, cluster_shape[0] - 1))
    return cluster_row, cluster_col


def build_route_corridor_claim_heatmap(
    cluster_shape,
    resolution,
    height,
    cluster_size,
    start_e,
    start_n,
    end_e,
    end_n,
    *,
    width_cells=1,
):
    heatmap = build_empty_reservation_heatmap(cluster_shape)
    block = max(1, int(cluster_size / max(resolution, 1e-6)))
    start_row, start_col = _cluster_indices_from_point(
        start_e,
        start_n,
        resolution,
        height,
        block,
        cluster_shape,
    )
    end_row, end_col = _cluster_indices_from_point(
        end_e,
        end_n,
        resolution,
        height,
        block,
        cluster_shape,
    )
    steps = max(abs(end_row - start_row), abs(end_col - start_col), 1)
    for step in range(steps + 1):
        alpha = step / float(steps)
        row = int(round(start_row + alpha * (end_row - start_row)))
        col = int(round(start_col + alpha * (end_col - start_col)))
        for row_offset in range(-width_cells, width_cells + 1):
            for col_offset in range(-width_cells, width_cells + 1):
                rr = row + row_offset
                cc = col + col_offset
                if 0 <= rr < cluster_shape[0] and 0 <= cc < cluster_shape[1]:
                    distance = max(abs(row_offset), abs(col_offset))
                    weight = max(0.0, 1.0 - 0.35 * float(distance))
                    heatmap[rr, cc] = max(heatmap[rr, cc], weight)
    return heatmap


def build_cluster_halo_claim_heatmap(cluster_shape, cluster_bounds, resolution, cluster_size, *, radius_cells=1):
    heatmap = build_empty_reservation_heatmap(cluster_shape)
    block = max(1, int(cluster_size / max(resolution, 1e-6)))
    row_idx = int(np.clip(cluster_bounds[0] // block, 0, cluster_shape[0] - 1))
    col_idx = int(np.clip(cluster_bounds[1] // block, 0, cluster_shape[1] - 1))
    for row_offset in range(-radius_cells, radius_cells + 1):
        for col_offset in range(-radius_cells, radius_cells + 1):
            rr = row_idx + row_offset
            cc = col_idx + col_offset
            if 0 <= rr < cluster_shape[0] and 0 <= cc < cluster_shape[1]:
                distance = max(abs(row_offset), abs(col_offset))
                weight = max(0.0, 1.0 - 0.45 * float(distance))
                heatmap[rr, cc] = max(heatmap[rr, cc], weight)
    return heatmap


def combine_claim_heatmaps(*claim_heatmaps):
    combined = None
    for claim_heatmap in claim_heatmaps:
        if claim_heatmap is None:
            continue
        if combined is None:
            combined = np.zeros_like(np.asarray(claim_heatmap, dtype=np.float32))
        np.minimum(
            combined + np.asarray(claim_heatmap, dtype=np.float32),
            1.0,
            out=combined,
        )
    return combined


def _build_cluster_target_candidate(
    hmu,
    config,
    drone_state,
    height,
    cluster_bounds,
    cluster_mean,
    map_bounds,
    origin,
    reserved_targets,
    reserved_paths,
    *,
    enforce_clearance=True,
    enforce_path_conflicts=True,
):
    resolution = config.heatmap.resolution
    planner = config.planner
    camera = config.camera
    cluster_view = hmu.get_cluster_view(planner.cluster_size)
    if cluster_view is None:
        return None

    target = pick_cluster_target(hmu.data, cluster_bounds, resolution, height)
    if target is None:
        return None

    drone_e = float(drone_state["e"])
    drone_n = float(drone_state["n"])
    drone_agl = drone_state.get("agl", camera.agl)
    dx = target["target_e"] - drone_e
    dy = target["target_n"] - drone_n
    distance = math.hypot(dx, dy)
    target_heading = math.degrees(math.atan2(dx, dy))
    footprint_span = projection_span_meters(
        camera,
        (target["target_e"], target["target_n"]),
        target_heading,
        agl=drone_agl,
    )
    usable_radius, effective_distance, distance_scale = estimate_target_travel_metrics(
        distance,
        footprint_span,
        planner,
    )

    if enforce_clearance and not target_has_clearance(
        target,
        reserved_targets,
        footprint_span,
    ):
        return None

    path, predicted_states = simulate_motion_prefixes(
        drone_state,
        target,
        planner,
        path_steps=15,
        predicted_steps=8,
        map_bounds=map_bounds,
    )
    if enforce_path_conflicts and path_conflicts_with_reserved(
        path,
        reserved_paths,
        planner,
    ):
        return None

    claim_heatmap = estimate_predicted_cluster_claim_heatmap(
        hmu.data.shape,
        camera,
        origin,
        resolution,
        planner.cluster_size,
        predicted_states,
    )
    candidate_target = {
        "cluster_bounds": cluster_bounds,
        "cluster_mean": cluster_mean,
        "effective_distance": effective_distance,
        "target_standoff_radius": usable_radius,
        **target,
        "main_target_e": float(target["target_e"]),
        "main_target_n": float(target["target_n"]),
        "main_target_c": float(target["target_c"]),
        "main_target_r": float(target["target_r"]),
        "greedy_subtarget": False,
    }
    route_claim_heatmap = build_route_corridor_claim_heatmap(
        cluster_view.shape,
        resolution,
        height,
        planner.cluster_size,
        drone_e,
        drone_n,
        candidate_target["main_target_e"],
        candidate_target["main_target_n"],
        width_cells=1,
    )
    goal_claim_heatmap = build_cluster_halo_claim_heatmap(
        cluster_view.shape,
        cluster_bounds,
        resolution,
        planner.cluster_size,
        radius_cells=1,
    )
    claim_heatmap = combine_claim_heatmaps(
        claim_heatmap,
        0.35 * route_claim_heatmap,
        0.60 * goal_claim_heatmap,
    )

    edge_approach_target = build_edge_approach_target(
        candidate_target,
        planner,
        resolution,
        height,
        map_bounds,
    )
    if edge_approach_target is not None:
        candidate_target["edge_original_target_e"] = float(candidate_target["main_target_e"])
        candidate_target["edge_original_target_n"] = float(candidate_target["main_target_n"])
        candidate_target["edge_original_target_c"] = float(candidate_target["main_target_c"])
        candidate_target["edge_original_target_r"] = float(candidate_target["main_target_r"])
        candidate_target.update(edge_approach_target)
        candidate_target["main_target_e"] = float(candidate_target["target_e"])
        candidate_target["main_target_n"] = float(candidate_target["target_n"])
        candidate_target["main_target_c"] = float(candidate_target["target_c"])
        candidate_target["main_target_r"] = float(candidate_target["target_r"])
        path, predicted_states = simulate_motion_prefixes(
            drone_state,
            candidate_target,
            planner,
            path_steps=15,
            predicted_steps=8,
            map_bounds=map_bounds,
        )
        if enforce_path_conflicts and path_conflicts_with_reserved(
            path,
            reserved_paths,
            planner,
        ):
            return None
        claim_heatmap = estimate_predicted_cluster_claim_heatmap(
            hmu.data.shape,
            camera,
            origin,
            resolution,
            planner.cluster_size,
            predicted_states,
        )
        route_claim_heatmap = build_route_corridor_claim_heatmap(
            cluster_view.shape,
            resolution,
            height,
            planner.cluster_size,
            drone_e,
            drone_n,
            candidate_target["main_target_e"],
            candidate_target["main_target_n"],
            width_cells=1,
        )
        goal_claim_heatmap = build_cluster_halo_claim_heatmap(
            cluster_view.shape,
            cluster_bounds,
            resolution,
            planner.cluster_size,
            radius_cells=1,
        )
        claim_heatmap = combine_claim_heatmaps(
            claim_heatmap,
            0.35 * route_claim_heatmap,
            0.60 * goal_claim_heatmap,
        )

    route_gain = float(
        np.sum(
            np.asarray(cluster_view, dtype=np.float32)
            * np.clip(claim_heatmap, 0.0, 1.0),
            dtype=np.float64,
        )
    )
    base_ranking_key = build_target_ranking_key(
        route_gain,
        0.0,
        cluster_mean,
        effective_distance,
        distance_scale,
    )

    return {
        "target": candidate_target,
        "path": path,
        "predicted_states": predicted_states,
        "claim_heatmap": claim_heatmap,
        "cluster_bounds": cluster_bounds,
        "cluster_mean": cluster_mean,
        "footprint_span": footprint_span,
        "effective_distance": effective_distance,
        "distance_scale": distance_scale,
        "base_ranking_key": base_ranking_key,
    }


def collect_cluster_target_candidates(
    hmu,
    config,
    drone_state,
    height,
    reserved_clusters=None,
    reserved_targets=None,
    reserved_paths=None,
    candidate_limit=None,
):
    planner = config.planner
    reserved_clusters = set() if reserved_clusters is None else set(reserved_clusters)
    reserved_targets = [] if reserved_targets is None else list(reserved_targets)
    reserved_paths = [] if reserved_paths is None else list(reserved_paths)
    if candidate_limit is None:
        candidate_limit = max(2, min(6, int(planner.top_k_clusters) * 2))

    cluster_view = hmu.get_cluster_view(planner.cluster_size)
    if cluster_view is None:
        return []

    resolution = config.heatmap.resolution
    origin = (0.0, float(height) * resolution)
    map_bounds = resolve_navigation_bounds(
        float(hmu.data.shape[1]) * float(resolution),
        float(hmu.data.shape[0]) * float(resolution),
        planner,
    )
    candidate_count = max(
        int(candidate_limit) * 3,
        int(planner.top_k_clusters) * 4,
        int(planner.top_k_clusters) + 4,
    )
    top_clusters = hmu.get_top_clusters(
        planner.cluster_size,
        top_k=candidate_count,
    )
    if not top_clusters:
        return []

    built_candidates = []
    seen_clusters = set()

    for strict in (True, False):
        for candidate in top_clusters:
            cluster_bounds = candidate[:4]
            cluster_mean = candidate[4]
            if cluster_bounds in reserved_clusters or cluster_bounds in seen_clusters:
                continue

            built_candidate = _build_cluster_target_candidate(
                hmu,
                config,
                drone_state,
                height,
                cluster_bounds,
                cluster_mean,
                map_bounds,
                origin,
                reserved_targets,
                reserved_paths,
                enforce_clearance=strict,
                enforce_path_conflicts=strict,
            )
            if built_candidate is None:
                continue

            built_candidates.append(built_candidate)
            seen_clusters.add(cluster_bounds)
            if len(built_candidates) >= int(candidate_limit):
                break
        if len(built_candidates) >= int(candidate_limit):
            break

    built_candidates.sort(key=lambda item: item["base_ranking_key"], reverse=True)
    return built_candidates[: int(candidate_limit)]


def materialize_target_candidate(candidate, ranking_key, unique_gain, overlap_gain, route_gain):
    target = dict(candidate["target"])
    target["unique_route_gain"] = float(unique_gain)
    target["overlap_gain"] = float(overlap_gain)
    target["route_gain"] = float(route_gain)
    target["distance_efficiency"] = float(ranking_key[0])
    target["reserved_path"] = candidate["path"]
    target["reserved_claim_heatmap"] = candidate["claim_heatmap"]
    target["reserved_blocks"] = claim_heatmap_to_block_set(candidate["claim_heatmap"])
    return target


def select_joint_target_assignment(
    candidate_lists,
    cluster_view,
    planner,
    reservation_heatmap,
    reserved_targets=None,
    reserved_paths=None,
):
    reserved_targets = [] if reserved_targets is None else list(reserved_targets)
    reserved_paths = [] if reserved_paths is None else list(reserved_paths)
    reservation_heatmap = _coerce_cluster_heatmap(
        reservation_heatmap,
        cluster_view.shape,
    )

    searcher_ids = [idx for idx, candidates in candidate_lists.items() if candidates]
    searcher_ids.sort(
        key=lambda idx: candidate_lists[idx][0]["base_ranking_key"],
        reverse=True,
    )

    best_assignment = {}
    best_score = None

    def dfs(
        position,
        current_reservation,
        selected_targets,
        selected_paths,
        selected_clusters,
        current_assignment,
        assigned_count,
        total_net_gain,
        total_efficiency,
        total_unique_gain,
        total_overlap_gain,
    ):
        nonlocal best_assignment, best_score

        if position >= len(searcher_ids):
            score = (
                assigned_count,
                total_net_gain,
                total_efficiency,
                total_unique_gain,
                -total_overlap_gain,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = dict(current_assignment)
            return

        searcher_id = searcher_ids[position]
        candidates = candidate_lists[searcher_id]

        for candidate in candidates:
            cluster_bounds = candidate["cluster_bounds"]
            if cluster_bounds in selected_clusters:
                continue

            target = candidate["target"]
            if not target_has_clearance(
                target,
                reserved_targets + selected_targets,
                candidate["footprint_span"],
            ):
                continue
            if path_conflicts_with_reserved(
                candidate["path"],
                reserved_paths + selected_paths,
                planner,
            ):
                continue

            ranking_key, unique_gain, overlap_gain, route_gain = score_cluster_target_candidate(
                cluster_view,
                candidate,
                current_reservation,
            )
            net_gain = max(0.0, float(unique_gain) - 0.75 * float(overlap_gain))
            if net_gain <= 1e-9:
                continue

            next_reservation = np.array(current_reservation, copy=True)
            merge_reservation_heatmap(next_reservation, candidate["claim_heatmap"])
            current_assignment[searcher_id] = (
                candidate,
                ranking_key,
                unique_gain,
                overlap_gain,
                route_gain,
            )
            dfs(
                position + 1,
                next_reservation,
                selected_targets + [target],
                selected_paths + ([candidate["path"]] if candidate["path"] else []),
                selected_clusters | {cluster_bounds},
                current_assignment,
                assigned_count + 1,
                total_net_gain + net_gain,
                total_efficiency + float(ranking_key[0]),
                total_unique_gain + float(unique_gain),
                total_overlap_gain + float(overlap_gain),
            )
            current_assignment.pop(searcher_id, None)

        dfs(
            position + 1,
            current_reservation,
            selected_targets,
            selected_paths,
            selected_clusters,
            current_assignment,
            assigned_count,
            total_net_gain,
            total_efficiency,
            total_unique_gain,
            total_overlap_gain,
        )

    dfs(
        0,
        reservation_heatmap,
        [],
        [],
        set(),
        {},
        0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    return best_assignment


def choose_greedy_route_subtarget(
    hmu,
    config,
    drone_state,
    height,
    main_target,
    predicted_states,
    reservation_heatmap,
):
    if not predicted_states:
        return None, None, None

    resolution = config.heatmap.resolution
    planner = config.planner
    camera = config.camera
    cluster_view = hmu.get_cluster_view(planner.cluster_size)
    if cluster_view is None:
        return None, None, None

    origin = (0.0, float(height) * resolution)
    drone_e = float(drone_state["e"])
    drone_n = float(drone_state["n"])
    main_distance = float(
        math.hypot(
            float(main_target["main_target_e"]) - drone_e,
            float(main_target["main_target_n"]) - drone_n,
        )
    )

    best_choice = None
    best_key = None
    best_path = None
    best_blocks = None
    best_predicted_states = None

    # Greedy mode chooses an intermediate waypoint from the short predicted
    # route prefix when that prefix already earns useful value. This gives the
    # drone a local "next" waypoint while preserving the original hotspot as
    # the main target in the payload.
    for idx, state in enumerate(predicted_states):
        target_e = float(state["e"])
        target_n = float(state["n"])
        if not point_is_in_heatmap(target_e, target_n, resolution, hmu.data.shape):
            continue

        prefix_states = predicted_states[: idx + 1]
        prefix_claim_heatmap = estimate_predicted_cluster_claim_heatmap(
            hmu.data.shape,
            camera,
            origin,
            resolution,
            planner.cluster_size,
            prefix_states,
        )
        unique_gain, overlap_gain, route_gain = estimate_coverage_scores(
            cluster_view,
            prefix_claim_heatmap,
            reservation_heatmap,
        )
        net_gain = max(0.0, float(unique_gain) - 0.75 * float(overlap_gain))
        if net_gain <= 0.0:
            continue

        distance = float(math.hypot(target_e - drone_e, target_n - drone_n))
        if distance <= 1e-6:
            continue

        footprint_span = projection_span_meters(
            camera,
            (target_e, target_n),
            state["heading"],
            agl=state.get("agl", camera.agl),
        )
        usable_radius, effective_distance, distance_scale = estimate_target_travel_metrics(
            distance,
            footprint_span,
            planner,
        )
        ranking_key = build_target_ranking_key(
            unique_gain,
            overlap_gain,
            main_target["cluster_mean"],
            effective_distance,
            distance_scale,
        )
        progress_ratio = (
            1.0
            if main_distance <= 1e-6
            else min(1.0, distance / max(main_distance, 1e-6))
        )
        greedy_key = (
            ranking_key[0],
            progress_ratio,
            ranking_key[1],
            -effective_distance,
            -(idx + 1),
        )

        if best_key is None or greedy_key > best_key:
            best_key = greedy_key
            best_choice = {
                **build_target_fields(target_e, target_n, resolution, height),
                "unique_route_gain": unique_gain,
                "overlap_gain": overlap_gain,
                "route_gain": route_gain,
                "effective_distance": effective_distance,
                "target_standoff_radius": usable_radius,
                "distance_efficiency": ranking_key[0],
                "greedy_subtarget": True,
                "greedy_prefix_steps": idx + 1,
                "greedy_progress_ratio": progress_ratio,
            }
            best_path = [(float(path_state["e"]), float(path_state["n"])) for path_state in prefix_states]
            best_blocks = prefix_claim_heatmap

    return best_choice, best_path, best_blocks


def choose_camera_projection_target(
    hmu,
    config,
    drone_state,
    height,
    current_pitch_deg,
    current_yaw_deg,
    reserved_targets=None,
    reserved_rois=None,
    preferred_target=None,
    flight_target=None,
):
    resolution = config.heatmap.resolution
    planner = config.planner
    camera = config.camera
    reserved_targets = [] if reserved_targets is None else list(reserved_targets)
    reserved_rois = [] if reserved_rois is None else list(reserved_rois)
    cluster_view = hmu.get_cluster_view(planner.cluster_size)
    if cluster_view is None:
        return preferred_target

    block = max(1, int(planner.cluster_size / max(resolution, 1e-6)))
    reach_steps = max(0, int(planner.camera_reach_steps))
    preferred_cluster = None if preferred_target is None else preferred_target.get("cluster_bounds")
    width_m = float(hmu.data.shape[1]) * float(resolution)
    height_m = float(hmu.data.shape[0]) * float(resolution)
    pitch_step_deg = turn_rate_to_step_delta(camera.pitch_turn_rate_deg, planner.step_seconds)
    yaw_step_deg = turn_rate_to_step_delta(camera.yaw_turn_rate_deg, planner.step_seconds)

    if pitch_step_deg <= 0:
        pitch_candidates = [float(np.clip(current_pitch_deg, camera.min_pitch, camera.max_pitch))]
    else:
        pitch_candidates = sorted(
            {
                float(
                    np.clip(
                        current_pitch_deg + step * pitch_step_deg,
                        camera.min_pitch,
                        camera.max_pitch,
                    )
                )
                for step in range(-reach_steps, reach_steps + 1)
            }
        )

    if yaw_step_deg <= 0:
        yaw_candidates = [float(np.clip(current_yaw_deg, camera.min_yaw, camera.max_yaw))]
    else:
        yaw_candidates = sorted(
            {
                float(
                    np.clip(
                        current_yaw_deg + step * yaw_step_deg,
                        camera.min_yaw,
                        camera.max_yaw,
                    )
                )
                for step in range(-reach_steps, reach_steps + 1)
            }
        )

    best_choice = None
    best_key = None
    best_relaxed_choice = None
    best_relaxed_key = None
    route_unit = None
    route_length = 0.0

    if flight_target is not None:
        route_goal_e = float(flight_target.get("main_target_e", flight_target["target_e"]))
        route_goal_n = float(flight_target.get("main_target_n", flight_target["target_n"]))
        route_dx = route_goal_e - float(drone_state["e"])
        route_dy = route_goal_n - float(drone_state["n"])
        route_length = float(math.hypot(route_dx, route_dy))
        if route_length > 1e-6:
            route_unit = (
                route_dx / route_length,
                route_dy / route_length,
            )

    # The camera target is chosen from the pose envelope it can reach in the next
    # few turn-rate steps, rather than from the whole map. Within that small
    # reachable set we score the *actual projected footprint value* on the
    # heatmap, while also preferring value that sits along the drone's current
    # route instead of off to the side or behind it.
    for pitch_deg in pitch_candidates:
        for yaw_deg in yaw_candidates:
            try:
                camera_model = build_camera(
                    camera,
                    (drone_state["e"], drone_state["n"]),
                    drone_state["heading"],
                    pitch=pitch_deg,
                    yaw=yaw_deg,
                    agl=drone_state.get("agl", camera.agl),
                )
                projection = camera_model.project()
                projection_center = camera_model.project_center()
            except ValueError:
                continue

            projection_roi = _projection_roi(projection, hmu)
            if projection_roi is None:
                continue

            target_e = float(np.clip(projection_center[0], 0.0, width_m))
            target_n = float(np.clip(projection_center[1], 0.0, height_m))
            target_c = float(np.clip(target_e / resolution, 0.0, hmu.data.shape[1] - 1e-6))
            target_r = float(np.clip(height - (target_n / resolution), 0.0, hmu.data.shape[0] - 1e-6))

            cluster_col = int(np.floor(target_c / block))
            cluster_row = int(np.floor(target_r / block))
            if not (
                0 <= cluster_row < cluster_view.shape[0]
                and 0 <= cluster_col < cluster_view.shape[1]
            ):
                continue

            target = {
                "target_c": target_c,
                "target_r": target_r,
                "target_e": target_e,
                "target_n": target_n,
                "cluster_bounds": (
                    cluster_row * block,
                    cluster_col * block,
                    min((cluster_row + 1) * block, hmu.data.shape[0]),
                    min((cluster_col + 1) * block, hmu.data.shape[1]),
                ),
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
            }

            row_min = max(0, cluster_row - 1)
            row_max = min(cluster_view.shape[0] - 1, cluster_row + 1)
            col_min = max(0, cluster_col - 1)
            col_max = min(cluster_view.shape[1] - 1, cluster_col + 1)
            local_value = float(cluster_view[row_min:row_max + 1, col_min:col_max + 1].sum())
            footprint_value, footprint_mean = estimate_projection_hit_value(hmu, projection)
            if footprint_value <= 0.0:
                continue
            center_value = sample_heat_value_at_target(hmu, target_c, target_r)
            total_overlap_ratio, max_overlap_ratio = estimate_roi_overlap_ratio(
                projection_roi,
                reserved_rois,
            )
            overlap_too_high = (
                total_overlap_ratio > 0.45
                or max_overlap_ratio > 0.35
            )

            pitch_cost = abs(pitch_deg - current_pitch_deg)
            yaw_cost = abs(yaw_deg - current_yaw_deg)
            if route_unit is None:
                route_alignment = 0.0
                along_track = 0.0
                cross_track_penalty = 0.0
            else:
                candidate_dx = target_e - float(drone_state["e"])
                candidate_dy = target_n - float(drone_state["n"])
                candidate_distance = float(math.hypot(candidate_dx, candidate_dy))
                if candidate_distance <= 1e-6:
                    route_alignment = 0.0
                else:
                    route_alignment = (
                        (candidate_dx * route_unit[0]) + (candidate_dy * route_unit[1])
                    ) / candidate_distance
                along_track = max(
                    0.0,
                    min(
                        1.0,
                        (
                            (candidate_dx * route_unit[0]) + (candidate_dy * route_unit[1])
                        ) / max(route_length, 1e-6),
                    ),
                )
                cross_track_distance = abs(
                    (candidate_dx * route_unit[1]) - (candidate_dy * route_unit[0])
                )
                cross_track_penalty = cross_track_distance / max(float(planner.cluster_size), 1.0)

            preferred_bonus = 0.05 if target["cluster_bounds"] == preferred_cluster else 0.0
            route_bonus = (
                0.15 * max(route_alignment, 0.0)
                + 0.10 * along_track
                - 0.05 * cross_track_penalty
            )
            overlap_penalty = (
                0.40 * total_overlap_ratio
                + 0.20 * max_overlap_ratio
            )
            projection_score = (
                0.55 * center_value
                + 0.30 * footprint_mean
                + 0.08 * local_value
                + 0.00002 * footprint_value
                + preferred_bonus
                + route_bonus
                - overlap_penalty
            )
            ranking_key = (
                projection_score,
                center_value,
                footprint_mean,
                local_value,
                footprint_value,
                along_track,
                route_alignment,
                -cross_track_penalty,
                -total_overlap_ratio,
                -max_overlap_ratio,
                -(pitch_cost + yaw_cost),
            )

            candidate_choice = {
                **target,
                "projection_score": projection_score,
                "center_value": center_value,
                "footprint_value": footprint_value,
                "footprint_mean": footprint_mean,
                "local_value": local_value,
                "route_alignment": route_alignment,
                "along_track": along_track,
                "cross_track_penalty": cross_track_penalty,
                "projection_overlap_ratio": total_overlap_ratio,
                "projection_overlap_penalty": overlap_penalty,
                "projection_roi": projection_roi,
            }

            if best_relaxed_key is None or ranking_key > best_relaxed_key:
                best_relaxed_key = ranking_key
                best_relaxed_choice = candidate_choice

            if overlap_too_high:
                continue
            if not target_has_clearance(target, reserved_targets, planner.cluster_size):
                continue

            if best_key is None or ranking_key > best_key:
                best_key = ranking_key
                best_choice = candidate_choice

    # Never steer the camera toward a zero-hit pose. If reserved-target spacing
    # blocks every positive candidate, fall back to the best positive local pose.
    if best_choice is not None:
        return best_choice
    if best_relaxed_choice is not None:
        return best_relaxed_choice
    return None


# ------------------------------------------------------------
# MAIN TARGET SELECTION
# ------------------------------------------------------------

def choose_best_cluster_target(
    hmu,
    config,
    drone_state,
    height,
    reserved_clusters=None,
    reserved_targets=None,
    reserved_paths=None,
    reserved_blocks=None,
):
    planner = config.planner
    cluster_view = hmu.get_cluster_view(planner.cluster_size)
    if cluster_view is None:
        return None

    reservation_heatmap = _coerce_cluster_heatmap(
        reserved_blocks,
        cluster_view.shape,
    )
    candidate_choices = collect_cluster_target_candidates(
        hmu,
        config,
        drone_state,
        height,
        reserved_clusters=reserved_clusters,
        reserved_targets=reserved_targets,
        reserved_paths=reserved_paths,
    )
    if not candidate_choices:
        return None

    best_candidate = None
    best_key = None
    best_unique_gain = 0.0
    best_overlap_gain = 0.0
    best_route_gain = 0.0

    for candidate in candidate_choices:
        ranking_key, unique_gain, overlap_gain, route_gain = score_cluster_target_candidate(
            cluster_view,
            candidate,
            reservation_heatmap,
        )
        if best_key is None or ranking_key > best_key:
            best_candidate = candidate
            best_key = ranking_key
            best_unique_gain = unique_gain
            best_overlap_gain = overlap_gain
            best_route_gain = route_gain

    if best_candidate is None:
        return None

    best_choice = materialize_target_candidate(
        best_candidate,
        best_key,
        best_unique_gain,
        best_overlap_gain,
        best_route_gain,
    )
    best_path = best_candidate["path"]
    best_claim_heatmap = best_candidate["claim_heatmap"]

    if planner.greedy_paths_enabled:
        greedy_choice, greedy_path, greedy_claim_heatmap = choose_greedy_route_subtarget(
            hmu,
            config,
            drone_state,
            height,
            best_choice,
            best_candidate["predicted_states"],
            reservation_heatmap,
        )
        if greedy_choice is not None:
            best_choice.update(greedy_choice)
            best_path = greedy_path
            best_claim_heatmap = combine_claim_heatmaps(
                best_claim_heatmap,
                greedy_claim_heatmap,
            )

    best_choice["reserved_path"] = best_path
    best_choice["reserved_claim_heatmap"] = best_claim_heatmap
    best_choice["reserved_blocks"] = claim_heatmap_to_block_set(best_claim_heatmap)
    return best_choice


def cluster_has_remaining_heat(data, cluster_bounds):
    row_min, col_min, row_max, col_max = cluster_bounds
    return bool(np.any(data[row_min:row_max, col_min:col_max] > 0))
