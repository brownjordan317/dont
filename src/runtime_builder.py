from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from src.config import (
    InitialDronePositionConfig,
    SweeperConfig,
    coerce_drone_ids,
    coerce_initial_drone_positions,
    load_config,
)
from src.drone_controller.flight_controller import DroneState, SimulatedFlightController
from src.geo_reference import lon_lat_to_local_point, resolve_geo_corners
from src.heat_map_updates.heatmap_loader import make_heatmap_from_source
from src.heat_map_updates.heatmap_updates import HeatMapUpdates
from src.planner_folder.central_planner import CentralPlanner
from src.planner_folder.geometry import configure_camera


ImageSource = Union[str, Path, np.ndarray]


def _coerce_optional_bool(raw_value: Optional[object]) -> Optional[bool]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
        raise ValueError(f"Could not parse boolean value: {raw_value!r}")
    raise ValueError(f"Could not parse boolean value: {raw_value!r}")


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def load_default_config() -> SweeperConfig:
    return load_config(default_config_path())


def resolve_runtime_config(
    config: Optional[SweeperConfig] = None,
    *,
    config_path: Optional[Union[str, Path]] = None,
    greedy_paths_enabled: Optional[bool] = None,
    drone_ids: Optional[Sequence[str]] = None,
    initial_drone_positions: Optional[Sequence[Mapping[str, object]]] = None,
    initial_altitudes_agl: Optional[Sequence[float]] = None,
    search_decay_percent_per_100ms: Optional[float] = None,
    step_seconds: Optional[float] = None,
) -> SweeperConfig:
    runtime_config = config
    if runtime_config is None:
        runtime_config = (
            load_config(config_path)
            if config_path is not None
            else load_default_config()
        )

    return apply_runtime_overrides(
        runtime_config,
        greedy_paths_enabled=greedy_paths_enabled,
        drone_ids=drone_ids,
        initial_drone_positions=initial_drone_positions,
        initial_altitudes_agl=initial_altitudes_agl,
        search_decay_percent_per_100ms=search_decay_percent_per_100ms,
        step_seconds=step_seconds,
    )


def apply_runtime_overrides(
    config: SweeperConfig,
    *,
    greedy_paths_enabled: Optional[bool] = None,
    drone_ids: Optional[Sequence[str]] = None,
    initial_drone_positions: Optional[Sequence[Mapping[str, object]]] = None,
    initial_altitudes_agl: Optional[Sequence[float]] = None,
    search_decay_percent_per_100ms: Optional[float] = None,
    step_seconds: Optional[float] = None,
):
    if (
        greedy_paths_enabled is None
        and drone_ids is None
        and initial_drone_positions is None
        and initial_altitudes_agl is None
        and search_decay_percent_per_100ms is None
        and step_seconds is None
    ):
        return config

    if initial_altitudes_agl is None:
        resolved_initial_altitudes_agl = config.planner.initial_altitudes_agl
    elif isinstance(initial_altitudes_agl, (list, tuple)):
        resolved_initial_altitudes_agl = tuple(float(value) for value in initial_altitudes_agl)
    else:
        resolved_initial_altitudes_agl = (float(initial_altitudes_agl),)

    resolved_greedy_paths_enabled = _coerce_optional_bool(greedy_paths_enabled)
    resolved_drone_ids = (
        config.planner.drone_ids
        if drone_ids is None
        else coerce_drone_ids(drone_ids)
    )
    resolved_initial_drone_positions = (
        config.planner.initial_drone_positions
        if initial_drone_positions is None
        else coerce_initial_drone_positions(initial_drone_positions)
    )

    return replace(
        config,
        heatmap=replace(
            config.heatmap,
            search_decay_percent_per_100ms=(
                config.heatmap.search_decay_percent_per_100ms
                if search_decay_percent_per_100ms is None
                else float(search_decay_percent_per_100ms)
            ),
        ),
        planner=replace(
            config.planner,
            greedy_paths_enabled=(
                config.planner.greedy_paths_enabled
                if resolved_greedy_paths_enabled is None
                else resolved_greedy_paths_enabled
            ),
            drone_ids=resolved_drone_ids,
            initial_drone_positions=resolved_initial_drone_positions,
            initial_altitudes_agl=resolved_initial_altitudes_agl,
            step_seconds=(
                config.planner.step_seconds
                if step_seconds is None
                else float(step_seconds)
            ),
        ),
    )


def resolve_initial_altitudes_agl(config: SweeperConfig, num_searchers: int):
    raw_values = tuple(float(value) for value in config.planner.initial_altitudes_agl)
    if not raw_values:
        raw_values = (float(config.camera.agl),)

    if len(raw_values) == 1:
        return raw_values * max(1, int(num_searchers))

    if len(raw_values) != int(num_searchers):
        raise ValueError(
            "planner.initial_altitudes_agl must provide either one shared altitude "
            f"or exactly {int(num_searchers)} values."
        )

    return raw_values


def resolve_drone_ids(config: SweeperConfig, num_searchers: int):
    raw_ids = tuple(str(value) for value in config.planner.drone_ids)
    if not raw_ids:
        return tuple(f"drone_{idx}" for idx in range(max(1, int(num_searchers))))

    if len(raw_ids) != int(num_searchers):
        raise ValueError(
            "planner.drone_ids must provide exactly "
            f"{int(num_searchers)} IDs when set."
        )

    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("planner.drone_ids must be unique.")

    return raw_ids


def _build_default_initial_drone_states(
    *,
    num_searchers: int,
    width_m: float,
    height_m: float,
    initial_altitudes_agl,
):
    center_e = width_m / 2.0
    center_n = height_m / 2.0
    start_radius = (
        0.08 * min(width_m, height_m)
        if num_searchers > 1
        else 0.0
    )

    states = []
    for idx in range(num_searchers):
        angle = 2.0 * np.pi * idx / num_searchers if num_searchers > 1 else 0.0
        drone_e = center_e + start_radius * np.cos(angle)
        drone_n = center_n + start_radius * np.sin(angle)
        states.append(
            DroneState(
                float(drone_e),
                float(drone_n),
                0.0,
                float(initial_altitudes_agl[idx]),
            )
        )

    return tuple(states)


def _resolve_local_start_position(
    position: InitialDronePositionConfig,
    *,
    width_m: float,
    height_m: float,
    geo_corners,
):
    if position.e is not None and position.n is not None:
        return float(position.e), float(position.n)

    if position.lat is not None and position.lon is not None:
        return lon_lat_to_local_point(
            float(position.lon),
            float(position.lat),
            width_m,
            height_m,
            geo_corners,
        )

    raise ValueError("Each initial drone position must provide either e/n or lat/lon.")


def resolve_initial_drone_states(
    config: SweeperConfig,
    *,
    num_searchers: int,
    width_m: float,
    height_m: float,
    initial_altitudes_agl,
):
    raw_positions = tuple(config.planner.initial_drone_positions)
    if not raw_positions:
        return _build_default_initial_drone_states(
            num_searchers=num_searchers,
            width_m=width_m,
            height_m=height_m,
            initial_altitudes_agl=initial_altitudes_agl,
        )

    if len(raw_positions) != int(num_searchers):
        raise ValueError(
            "planner.initial_drone_positions must provide exactly "
            f"{int(num_searchers)} entries when set."
        )

    geo_corners = None
    if any(position.lat is not None or position.lon is not None for position in raw_positions):
        geo_corners = resolve_geo_corners(
            config.geo_reference,
            width_m,
            height_m,
        )

    states = []
    for idx, position in enumerate(raw_positions):
        easting, northing = _resolve_local_start_position(
            position,
            width_m=width_m,
            height_m=height_m,
            geo_corners=geo_corners,
        )
        states.append(
            DroneState(
                float(easting),
                float(northing),
                0.0 if position.heading is None else float(position.heading),
                float(initial_altitudes_agl[idx]),
            )
        )

    return tuple(states)


def build_planner(
    config: SweeperConfig,
    image_source: Optional[ImageSource] = None,
    source_image_ppm: Optional[float] = None,
):
    configure_camera(config.camera)

    raw = make_heatmap_from_source(
        config.heatmap.image_path if image_source is None else image_source,
        config.heatmap.source_image_ppm if source_image_ppm is None else float(source_image_ppm),
        config.heatmap.resolution,
    )
    height, width = raw.shape
    origin = (0.0, float(height) * config.heatmap.resolution)

    hmu = HeatMapUpdates(
        {
            "data": raw.copy(),
            "origin": origin,
            "resolution": config.heatmap.resolution,
        },
        search_decay_percent_per_100ms=config.heatmap.search_decay_percent_per_100ms,
    )

    num_searchers = max(1, int(config.planner.num_searchers))
    drone_ids = resolve_drone_ids(config, num_searchers)
    initial_altitudes_agl = resolve_initial_altitudes_agl(config, num_searchers)
    width_m = width * config.heatmap.resolution
    height_m = height * config.heatmap.resolution
    initial_drone_states = resolve_initial_drone_states(
        config,
        num_searchers=num_searchers,
        width_m=width_m,
        height_m=height_m,
        initial_altitudes_agl=initial_altitudes_agl,
    )
    map_bounds = (
        0.0,
        width_m,
        0.0,
        height_m,
    )

    controllers = []
    for idx in range(num_searchers):
        initial_state = initial_drone_states[idx]
        controllers.append(
            SimulatedFlightController(
                initial_state,
                config.planner.drone_speed,
                config.planner.step_seconds,
                config.planner.max_turn_rate_deg,
                map_bounds=map_bounds,
            )
        )

    planner = CentralPlanner(
        config,
        hmu,
        controllers,
        height,
        origin,
        drone_ids=drone_ids,
    )
    return planner, {"height": height, "width": width, "origin": origin}
