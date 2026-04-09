from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from src.config import SweeperConfig
from src.drone_controller.flight_controller import DroneState
from src.geo_reference import CornerMap, local_point_to_lon_lat, lon_lat_to_local_point, resolve_geo_corners
from src.runtime_builder import ImageSource, build_planner, resolve_runtime_config


ObservationInput = Union[Sequence[Mapping[str, object]], Mapping[str, Mapping[str, object]]]


class IncrementalSweeperSession:
    """Stateful planner session for external control loops.

    The caller owns drone motion. On each cycle, the caller submits observed
    drone positions plus the current camera projection point for each drone.
    The session updates the searched map from those observations and returns the
    next recommended flight targets.
    """

    def __init__(self, config, planner, width_m, height_m, geo_corners):
        self.config = config
        self.planner = planner
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.geo_corners = dict(geo_corners)
        self.drone_ids = list(
            getattr(
                self.planner,
                "drone_ids",
                [f"drone_{idx}" for idx in range(len(self.planner.searchers))],
            )
        )

    @classmethod
    def _build_session(
        cls,
        runtime_config: SweeperConfig,
        *,
        image_source: Optional[ImageSource] = None,
        source_image_ppm: Optional[float] = None,
        corners: Optional[CornerMap] = None,
    ):
        planner, metadata = build_planner(
            runtime_config,
            image_source=image_source,
            source_image_ppm=source_image_ppm,
        )
        width_m = float(metadata["width"] * runtime_config.heatmap.resolution)
        height_m = float(metadata["height"] * runtime_config.heatmap.resolution)
        geo_corners = resolve_geo_corners(
            runtime_config.geo_reference,
            width_m,
            height_m,
            corners=corners,
        )
        return cls(runtime_config, planner, width_m, height_m, geo_corners)

    @classmethod
    def from_image(
        cls,
        image_source: Optional[ImageSource] = None,
        corners: Optional[CornerMap] = None,
        *,
        source_image_ppm: Optional[float] = None,
        config: Optional[SweeperConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        greedy_paths_enabled: Optional[bool] = None,
        drone_ids: Optional[Sequence[str]] = None,
        initial_drone_positions: Optional[Sequence[Mapping[str, object]]] = None,
        initial_altitudes_agl: Optional[Sequence[float]] = None,
        search_decay_percent_per_100ms: Optional[float] = None,
        step_seconds: Optional[float] = None,
    ):
        runtime_config = resolve_runtime_config(
            config,
            config_path=config_path,
            greedy_paths_enabled=greedy_paths_enabled,
            drone_ids=drone_ids,
            initial_drone_positions=initial_drone_positions,
            initial_altitudes_agl=initial_altitudes_agl,
            search_decay_percent_per_100ms=search_decay_percent_per_100ms,
            step_seconds=step_seconds,
        )
        return cls._build_session(
            runtime_config,
            image_source=image_source,
            source_image_ppm=source_image_ppm,
            corners=corners,
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: Optional[SweeperConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        image_source: Optional[ImageSource] = None,
        source_image_ppm: Optional[float] = None,
        corners: Optional[CornerMap] = None,
        greedy_paths_enabled: Optional[bool] = None,
        drone_ids: Optional[Sequence[str]] = None,
        initial_drone_positions: Optional[Sequence[Mapping[str, object]]] = None,
        initial_altitudes_agl: Optional[Sequence[float]] = None,
        search_decay_percent_per_100ms: Optional[float] = None,
        step_seconds: Optional[float] = None,
    ):
        runtime_config = resolve_runtime_config(
            config,
            config_path=config_path,
            greedy_paths_enabled=greedy_paths_enabled,
            drone_ids=drone_ids,
            initial_drone_positions=initial_drone_positions,
            initial_altitudes_agl=initial_altitudes_agl,
            search_decay_percent_per_100ms=search_decay_percent_per_100ms,
            step_seconds=step_seconds,
        )
        return cls._build_session(
            runtime_config,
            image_source=image_source,
            source_image_ppm=source_image_ppm,
            corners=corners,
        )

    def get_plan(self):
        return self._build_response(self.planner.get_render_state())

    def observe_and_plan(self, observations: ObservationInput, dt_seconds: Optional[float] = None):
        """Update planner state from external observations and return new targets.

        Each drone observation may include `position.agl` to update that
        drone's altitude above ground in meters for this cycle. If AGL is
        omitted, the session keeps the previously known value for that drone.
        `dt_seconds` scales search decay for this observation batch; when it is
        omitted, the session uses the configured planner `step_seconds`.
        """
        ordered = self._order_observations(observations)
        normalized = [
            self._normalize_observation(idx, observation)
            for idx, observation in enumerate(ordered)
        ]
        render_state = self.planner.observe_external(
            normalized,
            dt_seconds=(
                self.config.planner.step_seconds
                if dt_seconds is None
                else float(dt_seconds)
            ),
        )
        return self._build_response(render_state)

    def _order_observations(self, observations: ObservationInput):
        if isinstance(observations, Mapping):
            missing = [drone_id for drone_id in self.drone_ids if drone_id not in observations]
            if missing:
                raise ValueError(
                    "Missing observations for: " + ", ".join(missing)
                )
            return [observations[drone_id] for drone_id in self.drone_ids]

        ordered = list(observations)
        if len(ordered) != len(self.drone_ids):
            raise ValueError(
                f"Expected {len(self.drone_ids)} observations, got {len(ordered)}."
            )

        if all(isinstance(obs, Mapping) and "drone_id" in obs for obs in ordered):
            by_id = {str(obs["drone_id"]): obs for obs in ordered}
            missing = [drone_id for drone_id in self.drone_ids if drone_id not in by_id]
            if missing:
                raise ValueError(
                    "Missing observations for: " + ", ".join(missing)
                )
            return [by_id[drone_id] for drone_id in self.drone_ids]

        return ordered

    def _normalize_observation(self, idx, observation):
        if not isinstance(observation, Mapping):
            raise ValueError("Each observation must be a mapping.")

        position = observation.get("position", observation)
        if not isinstance(position, Mapping):
            raise ValueError("Observation position must be a mapping.")

        easting, northing = self._resolve_local_point(position, label="position")
        heading = position.get("heading", observation.get("heading"))
        if heading is None:
            heading = self._infer_heading(idx, easting, northing)
        agl = self._resolve_agl(idx, observation, position)

        projection_point = self._resolve_projection_point(observation)
        return {
            "state": DroneState(float(easting), float(northing), float(heading), float(agl)),
            "projection_point": projection_point,
            "camera_pitch_deg": observation.get("camera_pitch_deg"),
            "camera_yaw_deg": observation.get("camera_yaw_deg", observation.get("camera_roll_deg")),
        }

    def _resolve_agl(self, idx, observation, position):
        for source in (position, observation):
            for key in ("agl", "altitude_agl", "altitude_m", "altitude"):
                if key in source and source[key] is not None:
                    return float(source[key])

        last_state = self.planner.searchers[idx]["last_state"]
        if last_state is not None:
            return float(last_state.agl)
        return float(self.config.camera.agl)

    def _resolve_projection_point(self, observation):
        if "projection_point" in observation:
            projection_point = observation["projection_point"]
            if projection_point is None:
                return None
            if not isinstance(projection_point, Mapping):
                raise ValueError("projection_point must be a mapping with e/n or lat/lon.")
            return self._resolve_local_point(projection_point, label="projection_point")

        if "projection_e" in observation and "projection_n" in observation:
            return float(observation["projection_e"]), float(observation["projection_n"])

        if "projection_lat" in observation and "projection_lon" in observation:
            return lon_lat_to_local_point(
                float(observation["projection_lon"]),
                float(observation["projection_lat"]),
                self.width_m,
                self.height_m,
                self.geo_corners,
            )

        return None

    def _resolve_local_point(self, payload, *, label):
        if "e" in payload and "n" in payload:
            return float(payload["e"]), float(payload["n"])

        if "lat" in payload and "lon" in payload:
            return lon_lat_to_local_point(
                float(payload["lon"]),
                float(payload["lat"]),
                self.width_m,
                self.height_m,
                self.geo_corners,
            )

        raise ValueError(
            f"{label} must provide either e/n or lat/lon coordinates."
        )

    def _infer_heading(self, idx, easting, northing):
        searcher = self.planner.searchers[idx]
        last_state = searcher["last_state"]

        if last_state is not None:
            delta_e = float(easting) - float(last_state.e)
            delta_n = float(northing) - float(last_state.n)
            if float(np.hypot(delta_e, delta_n)) > 1e-6:
                return float(np.degrees(np.arctan2(delta_e, delta_n)))
            return float(last_state.heading)

        current_target = searcher["current_target"]
        if current_target is not None:
            delta_e = float(current_target["target_e"]) - float(easting)
            delta_n = float(current_target["target_n"]) - float(northing)
            if float(np.hypot(delta_e, delta_n)) > 1e-6:
                return float(np.degrees(np.arctan2(delta_e, delta_n)))

        return 0.0

    def _point_payload(self, easting, northing):
        lon, lat = local_point_to_lon_lat(
            (easting, northing),
            self.width_m,
            self.height_m,
            self.geo_corners,
        )
        return {
            "e": float(easting),
            "n": float(northing),
            "lat": float(lat),
            "lon": float(lon),
        }

    def _state_payload(self, state):
        payload = self._point_payload(state.e, state.n)
        payload["heading"] = float(state.heading)
        payload["agl"] = float(state.agl)
        return payload

    def _target_payload(self, target):
        if target is None:
            return None

        payload = self._point_payload(target["target_e"], target["target_n"])
        payload["target_c"] = float(target["target_c"])
        payload["target_r"] = float(target["target_r"])

        if "cluster_mean" in target:
            payload["cluster_mean"] = float(target["cluster_mean"])
        if "unique_route_gain" in target:
            payload["unique_route_gain"] = float(target["unique_route_gain"])
        if "overlap_gain" in target:
            payload["overlap_gain"] = float(target["overlap_gain"])
        if "route_gain" in target:
            payload["route_gain"] = float(target["route_gain"])
        if "effective_distance" in target:
            payload["effective_distance"] = float(target["effective_distance"])
        if "target_standoff_radius" in target:
            payload["target_standoff_radius"] = float(target["target_standoff_radius"])
        if "distance_efficiency" in target:
            payload["distance_efficiency"] = float(target["distance_efficiency"])
        if "greedy_subtarget" in target:
            payload["greedy_subtarget"] = bool(target["greedy_subtarget"])
        if "greedy_prefix_steps" in target:
            payload["greedy_prefix_steps"] = int(target["greedy_prefix_steps"])
        if "greedy_progress_ratio" in target:
            payload["greedy_progress_ratio"] = float(target["greedy_progress_ratio"])
        if "recovery_target" in target:
            payload["recovery_target"] = bool(target["recovery_target"])
        if "edge_approach_target" in target:
            payload["edge_approach_target"] = bool(target["edge_approach_target"])
        if "edge_margin_m" in target:
            payload["edge_margin_m"] = float(target["edge_margin_m"])
        if "main_target_edge_distance" in target:
            payload["main_target_edge_distance"] = float(target["main_target_edge_distance"])
        if "pitch_deg" in target:
            payload["pitch_deg"] = float(target["pitch_deg"])
        if "yaw_deg" in target:
            payload["yaw_deg"] = float(target["yaw_deg"])
        if "footprint_value" in target:
            payload["footprint_value"] = float(target["footprint_value"])
        if "footprint_mean" in target:
            payload["footprint_mean"] = float(target["footprint_mean"])
        if "local_value" in target:
            payload["local_value"] = float(target["local_value"])
        if "projection_score" in target:
            payload["projection_score"] = float(target["projection_score"])
        if "center_value" in target:
            payload["center_value"] = float(target["center_value"])
        if "route_alignment" in target:
            payload["route_alignment"] = float(target["route_alignment"])
        if "along_track" in target:
            payload["along_track"] = float(target["along_track"])
        if "cross_track_penalty" in target:
            payload["cross_track_penalty"] = float(target["cross_track_penalty"])
        if "projection_overlap_ratio" in target:
            payload["projection_overlap_ratio"] = float(target["projection_overlap_ratio"])
        if "projection_overlap_penalty" in target:
            payload["projection_overlap_penalty"] = float(target["projection_overlap_penalty"])
        if "main_target_e" in target and "main_target_n" in target:
            payload["main_target"] = self._point_payload(
                target["main_target_e"],
                target["main_target_n"],
            )
            if "main_target_c" in target:
                payload["main_target"]["target_c"] = float(target["main_target_c"])
            if "main_target_r" in target:
                payload["main_target"]["target_r"] = float(target["main_target_r"])
        return payload

    def _build_response(self, render_state):
        drones = {}
        for drone_id, render_item in zip(self.drone_ids, render_state):
            drones[drone_id] = {
                "state": self._state_payload(render_item["state"]),
                "next_target": self._target_payload(render_item["target"]),
                "camera_target": self._target_payload(render_item["camera_target"]),
                "camera_projection_point": (
                    None
                    if render_item["camera_projection_point"] is None
                    else self._point_payload(
                        render_item["camera_projection_point"][0],
                        render_item["camera_projection_point"][1],
                    )
                ),
            }

        return {
            "finished": bool(self.planner.finished),
            "planner": {
                "drone_ids": list(self.drone_ids),
                "greedy_paths_enabled": bool(self.config.planner.greedy_paths_enabled),
                "initial_altitudes_agl": [
                    float(value) for value in self.config.planner.initial_altitudes_agl
                ],
                "search_decay_percent_per_100ms": float(
                    self.config.heatmap.search_decay_percent_per_100ms
                ),
                "step_seconds": float(self.config.planner.step_seconds),
            },
            "drones": drones,
        }


def create_incremental_session_from_image(
    image_source: Optional[ImageSource] = None,
    corners: Optional[CornerMap] = None,
    *,
    source_image_ppm: Optional[float] = None,
    config: Optional[SweeperConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    greedy_paths_enabled: Optional[bool] = None,
    drone_ids: Optional[Sequence[str]] = None,
    initial_drone_positions: Optional[Sequence[Mapping[str, object]]] = None,
    initial_altitudes_agl: Optional[Sequence[float]] = None,
    search_decay_percent_per_100ms: Optional[float] = None,
    step_seconds: Optional[float] = None,
):
    """Create a stateful observe-and-plan session from an image.

    `initial_altitudes_agl` sets each drone's starting altitude above ground in
    meters. Pass one value to apply it to every drone, or pass one value per
    drone. After creation, `observe_and_plan(...)` can update AGL per drone on
    each call through `position.agl`. `search_decay_percent_per_100ms`
    controls how quickly observed heat fades: `100` clears immediately and `1`
    takes about 10 seconds of continuous observation to clear a max-value cell.
    `image_source`, `source_image_ppm`, and `corners` may be omitted when the
    selected config already provides them. If neither `config` nor
    `config_path` is supplied, the repo's default `config.yaml` is used.
    `drone_ids` and `initial_drone_positions` let callers replace the
    configured drone names and start locations for the session with full lists.
    `step_seconds` controls the default timestep used by the session whenever
    `observe_and_plan(..., dt_seconds=...)` does not provide an explicit value.
    """
    return IncrementalSweeperSession.from_image(
        image_source,
        corners,
        source_image_ppm=source_image_ppm,
        config=config,
        config_path=config_path,
        greedy_paths_enabled=greedy_paths_enabled,
        drone_ids=drone_ids,
        initial_drone_positions=initial_drone_positions,
        initial_altitudes_agl=initial_altitudes_agl,
        search_decay_percent_per_100ms=search_decay_percent_per_100ms,
        step_seconds=step_seconds,
    )
