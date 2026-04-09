from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import yaml


DEFAULT_STEP_SECONDS = 0.1
DEFAULT_BOUNDARY_MARGIN_FRACTION = 0.01
DEFAULT_NAVIGATION_TOLERANCE_M = 1e-4


@dataclass(frozen=True)
class HeatmapConfig:
    image_path: str
    source_image_ppm: float
    resolution: float
    search_decay_percent_per_100ms: float


@dataclass(frozen=True)
class GeoReferenceConfig:
    tl_lat: Optional[float]
    tl_lon: Optional[float]
    tr_lat: Optional[float]
    tr_lon: Optional[float]
    br_lat: Optional[float]
    br_lon: Optional[float]
    bl_lat: Optional[float]
    bl_lon: Optional[float]
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]


@dataclass(frozen=True)
class CameraConfig:
    matrix: Tuple[Tuple[float, float, float], ...]
    agl: float
    pitch: float
    yaw: float
    min_pitch: float
    max_pitch: float
    min_yaw: float
    max_yaw: float
    pitch_turn_rate_deg: float
    yaw_turn_rate_deg: float


@dataclass(frozen=True)
class InitialDronePositionConfig:
    e: Optional[float]
    n: Optional[float]
    lat: Optional[float]
    lon: Optional[float]
    heading: Optional[float]


@dataclass(frozen=True)
class PlannerConfig:
    num_searchers: int
    cluster_size: int
    top_k_clusters: int
    camera_reach_steps: int
    greedy_paths_enabled: bool
    drone_ids: Tuple[str, ...]
    initial_drone_positions: Tuple[InitialDronePositionConfig, ...]
    initial_altitudes_agl: Tuple[float, ...]
    target_radius_percent: float
    drone_speed: float
    step_seconds: float
    max_turn_rate_deg: float
    boundary_margin_fraction: float
    navigation_tolerance_m: float


@dataclass(frozen=True)
class DisplayConfig:
    figure_size: Tuple[float, float]
    interval_ms: int
    sync_to_runtime: bool
    footprint_color: str
    footprint_alpha: float
    target_color: str
    drone_color: str
    background_color: str
    colormap: Tuple[str, ...]


@dataclass(frozen=True)
class ExportConfig:
    save_video: bool


@dataclass(frozen=True)
class SweeperConfig:
    heatmap: HeatmapConfig
    geo_reference: GeoReferenceConfig
    camera: CameraConfig
    planner: PlannerConfig
    display: DisplayConfig
    export: ExportConfig


def _parse_float_tuple(raw_value, default_value=()):
    if raw_value is None:
        return tuple(float(value) for value in default_value)

    if isinstance(raw_value, (list, tuple)):
        return tuple(float(value) for value in raw_value)

    return (float(raw_value),)


def coerce_drone_ids(raw_value, default_value=()):
    if raw_value is None:
        raw_values = default_value
    elif isinstance(raw_value, (list, tuple)):
        raw_values = raw_value
    else:
        raise ValueError("drone_ids must be provided as a full list of IDs.")

    drone_ids = []
    for raw_id in raw_values:
        drone_id = coerce_drone_id(raw_id)
        drone_ids.append(drone_id)

    return tuple(drone_ids)


def coerce_drone_id(raw_value) -> str:
    drone_id = str(raw_value).strip()
    if not drone_id:
        raise ValueError("Drone IDs must be non-empty strings.")
    if any(char in drone_id for char in ("/", "\\")):
        raise ValueError(
            f"Drone ID {drone_id!r} cannot contain path separators."
        )
    return drone_id


def _parse_initial_drone_position(raw_value) -> InitialDronePositionConfig:
    if isinstance(raw_value, InitialDronePositionConfig):
        return raw_value

    if not isinstance(raw_value, Mapping):
        raise ValueError("Each initial drone position must be a mapping.")

    position = raw_value.get("position", raw_value)
    if not isinstance(position, Mapping):
        raise ValueError("initial_drone_positions entries must use a mapping for position data.")

    has_local = "e" in position or "n" in position
    has_geo = "lat" in position or "lon" in position
    if has_local and has_geo:
        raise ValueError(
            "Each initial drone position must provide either e/n or lat/lon, not both."
        )
    if has_local:
        if "e" not in position or "n" not in position:
            raise ValueError("Local initial drone positions must provide both e and n.")
    elif has_geo:
        if "lat" not in position or "lon" not in position:
            raise ValueError("Geo initial drone positions must provide both lat and lon.")
    else:
        raise ValueError("Each initial drone position must provide either e/n or lat/lon.")

    heading = raw_value.get("heading", position.get("heading"))
    return InitialDronePositionConfig(
        e=float(position["e"]) if "e" in position else None,
        n=float(position["n"]) if "n" in position else None,
        lat=float(position["lat"]) if "lat" in position else None,
        lon=float(position["lon"]) if "lon" in position else None,
        heading=None if heading is None else float(heading),
    )


def coerce_initial_drone_positions(raw_value):
    if raw_value is None:
        return ()
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError(
            "initial_drone_positions must be provided as a full list of mappings."
        )
    return tuple(_parse_initial_drone_position(value) for value in raw_value)


def _parse_bool(raw_value, *, default_value: bool):
    if raw_value is None:
        return bool(default_value)

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


def load_config(path: Union[str, Path]) -> SweeperConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    image_path = Path(raw["heatmap"]["image_path"])
    if not image_path.is_absolute():
        image_path = config_path.parent / image_path

    geo_raw = raw.get("geo_reference", {})
    planner_step_seconds = float(raw["planner"].get("step_seconds", DEFAULT_STEP_SECONDS))
    default_display_interval_ms = max(
        1,
        min(16, int(round(planner_step_seconds * 1000.0))),
    )
    display_raw = raw["display"]
    export_raw = raw.get("export", {})

    planner_raw = raw["planner"]

    return SweeperConfig(
        heatmap=HeatmapConfig(
            image_path=str(image_path),
            source_image_ppm=float(raw["heatmap"]["source_image_ppm"]),
            resolution=float(raw["heatmap"]["resolution"]),
            search_decay_percent_per_100ms=float(
                raw["heatmap"].get("search_decay_percent_per_100ms", 100.0)
            ),
        ),
        geo_reference=GeoReferenceConfig(
            tl_lat=float(geo_raw["tl"]["lat"]) if "tl" in geo_raw else None,
            tl_lon=float(geo_raw["tl"]["lon"]) if "tl" in geo_raw else None,
            tr_lat=float(geo_raw["tr"]["lat"]) if "tr" in geo_raw else None,
            tr_lon=float(geo_raw["tr"]["lon"]) if "tr" in geo_raw else None,
            br_lat=float(geo_raw["br"]["lat"]) if "br" in geo_raw else None,
            br_lon=float(geo_raw["br"]["lon"]) if "br" in geo_raw else None,
            bl_lat=float(geo_raw["bl"]["lat"]) if "bl" in geo_raw else None,
            bl_lon=float(geo_raw["bl"]["lon"]) if "bl" in geo_raw else None,
            centroid_lat=float(geo_raw["centroid_lat"]) if "centroid_lat" in geo_raw else None,
            centroid_lon=float(geo_raw["centroid_lon"]) if "centroid_lon" in geo_raw else None,
        ),
        camera=CameraConfig(
            matrix=tuple(tuple(row) for row in raw["camera"]["matrix"]),
            agl=float(raw["camera"]["agl"]),
            pitch=float(raw["camera"]["pitch"]),
            yaw=float(raw["camera"].get("yaw", raw["camera"].get("roll", 0.0))),
            min_pitch=float(raw["camera"].get("min_pitch", raw["camera"]["pitch"])),
            max_pitch=float(raw["camera"].get("max_pitch", raw["camera"]["pitch"])),
            min_yaw=float(raw["camera"].get("min_yaw", raw["camera"].get("min_roll", 0.0))),
            max_yaw=float(raw["camera"].get("max_yaw", raw["camera"].get("max_roll", 0.0))),
            pitch_turn_rate_deg=float(raw["camera"].get("pitch_turn_rate_deg", 0.0)),
            yaw_turn_rate_deg=float(raw["camera"].get("yaw_turn_rate_deg", raw["camera"].get("roll_turn_rate_deg", 0.0))),
        ),
        planner=PlannerConfig(
            num_searchers=int(planner_raw["num_searchers"]),
            cluster_size=int(planner_raw["cluster_size"]),
            top_k_clusters=int(planner_raw["top_k_clusters"]),
            camera_reach_steps=int(planner_raw.get("camera_reach_steps", planner_raw.get("camera_target_top_k", 2))),
            greedy_paths_enabled=_parse_bool(
                planner_raw.get("greedy_paths_enabled", False),
                default_value=False,
            ),
            drone_ids=coerce_drone_ids(planner_raw.get("drone_ids")),
            initial_drone_positions=coerce_initial_drone_positions(
                planner_raw.get("initial_drone_positions")
            ),
            initial_altitudes_agl=_parse_float_tuple(
                planner_raw.get("initial_altitudes_agl"),
                default_value=(float(raw["camera"]["agl"]),),
            ),
            target_radius_percent=float(planner_raw["target_radius_percent"]),
            drone_speed=float(planner_raw["drone_speed"]),
            step_seconds=planner_step_seconds,
            max_turn_rate_deg=float(planner_raw["max_turn_rate_deg"]),
            boundary_margin_fraction=float(
                planner_raw.get(
                    "boundary_margin_fraction",
                    DEFAULT_BOUNDARY_MARGIN_FRACTION,
                )
            ),
            navigation_tolerance_m=float(
                planner_raw.get(
                    "navigation_tolerance_m",
                    DEFAULT_NAVIGATION_TOLERANCE_M,
                )
            ),
        ),
        display=DisplayConfig(
            figure_size=tuple(display_raw["figure_size"]),
            interval_ms=int(display_raw.get("interval_ms", default_display_interval_ms)),
            sync_to_runtime=_parse_bool(
                display_raw.get("sync_to_runtime", True),
                default_value=True,
            ),
            footprint_color=display_raw["footprint_color"],
            footprint_alpha=float(display_raw["footprint_alpha"]),
            target_color=display_raw["target_color"],
            drone_color=display_raw["drone_color"],
            background_color=display_raw["background_color"],
            colormap=tuple(display_raw["colormap"]),
        ),
        export=ExportConfig(
            save_video=_parse_bool(
                export_raw.get("save_video", True),
                default_value=True,
            ),
        ),
    )
