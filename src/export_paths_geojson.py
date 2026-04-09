#!/usr/bin/env python3
"""Export one per-drone JSON timeseries, either from config or a direct API call.

This module is intentionally importable from `src` so external code can call
the sweeper with an image and georeferenced corners without shelling out to a
separate helper script.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from src.config import SweeperConfig, load_config
from src.geo_reference import CornerMap, local_point_to_lon_lat, resolve_geo_corners
from src.planner_folder.geometry import build_camera
from src.runtime_builder import (
    ImageSource,
    apply_runtime_overrides,
    build_planner,
    default_config_path,
    resolve_runtime_config,
)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def _to_lat_lon(
    point_xy,
    *,
    width_m: float,
    height_m: float,
    corners: Dict[str, Tuple[float, float]],
):
    lon, lat = local_point_to_lon_lat(point_xy, width_m, height_m, corners)
    return float(lat), float(lon)


def _to_bearing(heading_deg: float) -> float:
    return float(heading_deg) % 360.0


def _hex_to_bgr(color_value: str):
    raw = color_value.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {color_value!r}")
    return np.array(
        [
            int(raw[4:6], 16),
            int(raw[2:4], 16),
            int(raw[0:2], 16),
        ],
        dtype=np.uint8,
    )


def _build_bgr_lut(color_stops):
    if not color_stops:
        color_stops = ("#000000", "#ffffff")
    stop_positions = np.linspace(0.0, 255.0, len(color_stops), dtype=np.float32)
    stop_colors = np.array([_hex_to_bgr(color) for color in color_stops], dtype=np.float32)
    sample_positions = np.arange(256, dtype=np.float32)
    return np.stack(
        [
            np.interp(sample_positions, stop_positions, stop_colors[:, channel_idx])
            for channel_idx in range(3)
        ],
        axis=1,
    ).astype(np.uint8)


def _resolve_video_output_path(output_arg: str, explicit_video_output=None):
    if explicit_video_output:
        video_path = Path(explicit_video_output)
        if video_path.parent != Path(""):
            video_path.parent.mkdir(parents=True, exist_ok=True)
        return video_path

    output_path = Path(output_arg)
    if output_path.suffix:
        video_path = output_path.with_suffix(".mp4")
        if video_path.parent != Path(""):
            video_path.parent.mkdir(parents=True, exist_ok=True)
        return video_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / "replay.mp4"


def _resolve_save_video_setting(config: SweeperConfig, save_video: Optional[bool]) -> bool:
    if save_video is None:
        return bool(config.export.save_video)
    if isinstance(save_video, bool):
        return save_video
    if isinstance(save_video, (int, float)):
        return bool(save_video)
    if isinstance(save_video, str):
        normalized = save_video.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"Could not parse boolean value: {save_video!r}")


def _resolve_covered_percent(initial_total_heat: float, remaining_total_heat: float) -> float:
    initial_total_heat = max(0.0, float(initial_total_heat))
    if initial_total_heat <= 1e-9:
        return 100.0

    remaining_total_heat = min(max(float(remaining_total_heat), 0.0), initial_total_heat)
    return ((initial_total_heat - remaining_total_heat) / initial_total_heat) * 100.0


def _coverage_stop_reached(
    initial_total_heat: float,
    remaining_total_heat: float,
    stop_when_covered_percent: Optional[float],
) -> bool:
    if stop_when_covered_percent is None:
        return False

    return _resolve_covered_percent(initial_total_heat, remaining_total_heat) >= (
        float(stop_when_covered_percent) - 1e-9
    )


class ReplayVideoWriter:
    def __init__(
        self,
        output_path,
        config: SweeperConfig,
        hmu,
        *,
        width_m: float,
        height_m: float,
    ):
        requested_output_path = Path(output_path)
        if requested_output_path.parent != Path(""):
            requested_output_path.parent.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.hmu = hmu
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.cluster_size = config.planner.cluster_size
        self.frame_height = max(240, int(round(config.display.figure_size[1] * 100.0)))
        self.frame_width = max(
            240,
            int(round(self.frame_height * (self.width_m / max(self.height_m, 1e-6)))),
        )
        self.heat_lut = _build_bgr_lut(config.display.colormap)
        self.footprint_color = tuple(int(value) for value in _hex_to_bgr(config.display.footprint_color))
        self.target_color = tuple(int(value) for value in _hex_to_bgr(config.display.target_color))
        self.drone_color = tuple(int(value) for value in _hex_to_bgr(config.display.drone_color))
        self.background_color = tuple(int(value) for value in _hex_to_bgr(config.display.background_color))
        self.camera_color = tuple(int(value) for value in _hex_to_bgr("#ffd166"))
        self.footprint_color_rgb = tuple(reversed(self.footprint_color))
        self.target_color_rgb = tuple(reversed(self.target_color))
        self.drone_color_rgb = tuple(reversed(self.drone_color))
        self.background_color_rgb = tuple(reversed(self.background_color))
        self.camera_color_rgb = tuple(reversed(self.camera_color))
        self.footprint_alpha = float(np.clip(config.display.footprint_alpha, 0.0, 1.0))
        self.fps = 1.0 / max(float(config.planner.step_seconds), 1e-6)
        self.output_path = requested_output_path
        self._backend = None
        self._writer = None

        if cv2 is not None:
            cv2_writer = cv2.VideoWriter(
                str(requested_output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.frame_width, self.frame_height),
            )
            if cv2_writer.isOpened():
                self._backend = "cv2"
                self._writer = cv2_writer
            else:
                cv2_writer.release()

        if self._writer is None and imageio is not None:
            try:
                imageio_writer = imageio.get_writer(
                    requested_output_path,
                    format="FFMPEG",
                    fps=self.fps,
                )
            except Exception:
                imageio_writer = None
            if imageio_writer is not None:
                self._backend = "imageio_ffmpeg"
                self._writer = imageio_writer

        if self._writer is None:
            raise RuntimeError(
                "Could not create an MP4 replay writer for "
                f"{requested_output_path}. Install OpenCV (`cv2`) or "
                "imageio ffmpeg support (`imageio[ffmpeg]` or `imageio-ffmpeg`)."
            )

    def close(self):
        if self._backend == "cv2":
            self._writer.release()
        else:
            self._writer.close()

    def _map_point_to_frame(self, easting: float, northing: float):
        x = int(round((float(easting) / max(self.width_m, 1e-6)) * (self.frame_width - 1)))
        y = int(round(((self.height_m - float(northing)) / max(self.height_m, 1e-6)) * (self.frame_height - 1)))
        x = max(0, min(self.frame_width - 1, x))
        y = max(0, min(self.frame_height - 1, y))
        return x, y

    def _projection_polygon(self, projection):
        return np.array(
            [
                self._map_point_to_frame(*projection[key])
                for key in ("tl", "tr", "br", "bl")
            ],
            dtype=np.int32,
        )

    def _base_heatmap_frame(self):
        cluster_view = self.hmu.get_cluster_view(self.cluster_size)
        if cluster_view is None:
            background_color = (
                self.background_color
                if cv2 is not None
                else self.background_color_rgb
            )
            return np.full(
                (self.frame_height, self.frame_width, 3),
                background_color,
                dtype=np.uint8,
            )

        heat_indices = np.clip(
            np.rint(cluster_view * 255.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        heat_image = self.heat_lut[heat_indices]
        if cv2 is not None:
            return cv2.resize(
                heat_image,
                (self.frame_width, self.frame_height),
                interpolation=cv2.INTER_NEAREST,
            )

        rgb_image = heat_image[:, :, ::-1]
        return np.asarray(
            Image.fromarray(rgb_image, mode="RGB").resize(
                (self.frame_width, self.frame_height),
                resample=Image.NEAREST,
            )
        )

    def _write_frame_pil(self, render_state, *, timestamp: float):
        frame = Image.fromarray(self._base_heatmap_frame(), mode="RGB")
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")

        for item in render_state:
            projection = item.get("projection")
            if projection is not None:
                polygon = [
                    self._map_point_to_frame(*projection[key])
                    for key in ("tl", "tr", "br", "bl")
                ]
                overlay_draw.polygon(
                    polygon,
                    fill=self.footprint_color_rgb + (
                        int(round(self.footprint_alpha * 255.0)),
                    ),
                )
                overlay_draw.line(
                    polygon + [polygon[0]],
                    fill=self.footprint_color_rgb + (255,),
                    width=1,
                )

        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(frame)

        for idx, item in enumerate(render_state):
            target = item.get("target")
            if target is not None:
                center_x, center_y = self._map_point_to_frame(target["target_e"], target["target_n"])
                draw.ellipse((center_x - 7, center_y - 7, center_x + 7, center_y + 7), fill=(0, 0, 0))
                draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill=self.target_color_rgb)

            camera_projection_point = item.get("camera_projection_point")
            state = item["state"]
            drone_center = self._map_point_to_frame(state.e, state.n)

            if camera_projection_point is not None:
                camera_center = self._map_point_to_frame(
                    camera_projection_point[0],
                    camera_projection_point[1],
                )
                draw.line((drone_center, camera_center), fill=self.camera_color_rgb, width=1)
                draw.ellipse((camera_center[0] - 5, camera_center[1] - 5, camera_center[0] + 5, camera_center[1] + 5), fill=(0, 0, 0))
                draw.ellipse((camera_center[0] - 3, camera_center[1] - 3, camera_center[0] + 3, camera_center[1] + 3), fill=self.camera_color_rgb)

            draw.ellipse((drone_center[0] - 9, drone_center[1] - 9, drone_center[0] + 9, drone_center[1] + 9), fill=(0, 0, 0))
            draw.ellipse((drone_center[0] - 6, drone_center[1] - 6, drone_center[0] + 6, drone_center[1] + 6), fill=self.drone_color_rgb)
            draw.text((drone_center[0] + 8, drone_center[1] - 8), str(idx), fill=(255, 255, 255))

        draw.text((12, 12), f"t = {timestamp:0.2f}s", fill=(255, 255, 255))
        self._writer.append_data(np.asarray(frame))

    def write_frame(self, render_state, *, timestamp: float):
        if cv2 is None:
            self._write_frame_pil(render_state, timestamp=timestamp)
            return

        frame = self._base_heatmap_frame()

        for idx, item in enumerate(render_state):
            projection = item.get("projection")
            if projection is not None:
                polygon = self._projection_polygon(projection)
                overlay = frame.copy()
                cv2.fillConvexPoly(overlay, polygon, self.footprint_color)
                frame = cv2.addWeighted(
                    overlay,
                    self.footprint_alpha,
                    frame,
                    1.0 - self.footprint_alpha,
                    0.0,
                )
                cv2.polylines(frame, [polygon], isClosed=True, color=self.footprint_color, thickness=1)

            target = item.get("target")
            if target is not None:
                center = self._map_point_to_frame(target["target_e"], target["target_n"])
                cv2.circle(frame, center, 7, (0, 0, 0), thickness=-1)
                cv2.circle(frame, center, 5, self.target_color, thickness=-1)

            camera_projection_point = item.get("camera_projection_point")
            state = item["state"]
            drone_center = self._map_point_to_frame(state.e, state.n)

            if camera_projection_point is not None:
                camera_center = self._map_point_to_frame(
                    camera_projection_point[0],
                    camera_projection_point[1],
                )
                cv2.line(frame, drone_center, camera_center, self.camera_color, thickness=1)
                cv2.circle(frame, camera_center, 5, (0, 0, 0), thickness=-1)
                cv2.circle(frame, camera_center, 3, self.camera_color, thickness=-1)

            cv2.circle(frame, drone_center, 9, (0, 0, 0), thickness=-1)
            cv2.circle(frame, drone_center, 6, self.drone_color, thickness=-1)
            cv2.putText(
                frame,
                str(idx),
                (drone_center[0] + 8, drone_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"t = {timestamp:0.2f}s",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if self._backend == "cv2":
            self._writer.write(frame)
            return

        # OpenCV draws in BGR; imageio expects RGB.
        self._writer.append_data(frame[:, :, ::-1])


def build_timestep_record(
    state,
    camera_projection_point,
    camera_orientation,
    *,
    timestamp: float,
    width_m: float,
    height_m: float,
    corners: Dict[str, Tuple[float, float]],
):
    drone_lat, drone_lon = _to_lat_lon(
        (state.e, state.n),
        width_m=width_m,
        height_m=height_m,
        corners=corners,
    )

    if camera_projection_point is None:
        cam_proj_lat = None
        cam_proj_lon = None
    else:
        cam_proj_lat, cam_proj_lon = _to_lat_lon(
            camera_projection_point,
            width_m=width_m,
            height_m=height_m,
            corners=corners,
        )

    return {
        "timestamp": float(timestamp),
        "drone_lat": drone_lat,
        "drone_lon": drone_lon,
        "agl": float(state.agl),
        "bearing": _to_bearing(state.heading),
        "cam_proj_lat": cam_proj_lat,
        "cam_proj_lon": cam_proj_lon,
        "cam_x": float(camera_orientation[0]),
        "cam_y": float(camera_orientation[1]),
        "cam_z": float(camera_orientation[2]),
    }


def record_render_state(
    render_state,
    drone_records,
    camera_config,
    *,
    width_m: float,
    height_m: float,
    corners: Dict[str, Tuple[float, float]],
    timestamp: float,
):
    for idx, item in enumerate(render_state):
        state = item["state"]
        camera_model = build_camera(
            camera_config,
            (state.e, state.n),
            state.heading,
            pitch=item["camera_pitch_deg"],
            yaw=item["camera_yaw_deg"],
            agl=state.agl,
        )
        drone_records[idx].append(
            build_timestep_record(
                state,
                item.get("camera_projection_point"),
                camera_model.center_ray_world(),
                timestamp=timestamp,
                width_m=width_m,
                height_m=height_m,
                corners=corners,
            )
        )


def export_drone_geojsons(
    config: SweeperConfig,
    max_steps: Optional[int],
    image_source: Optional[ImageSource] = None,
    source_image_ppm: Optional[float] = None,
    corners: Optional[CornerMap] = None,
    greedy_paths_enabled: Optional[bool] = None,
    drone_ids: Optional[Sequence[str]] = None,
    initial_drone_positions: Optional[Sequence[Mapping[str, object]]] = None,
    initial_altitudes_agl: Optional[Sequence[float]] = None,
    search_decay_percent_per_100ms: Optional[float] = None,
    step_seconds: Optional[float] = None,
    stop_when_covered_percent: Optional[float] = None,
    video_output: Optional[Union[str, Path]] = None,
    save_video: Optional[bool] = None,
):
    if stop_when_covered_percent is not None:
        stop_when_covered_percent = float(stop_when_covered_percent)
        if not 0.0 <= stop_when_covered_percent <= 100.0:
            raise ValueError("stop_when_covered_percent must be between 0 and 100.")

    runtime_config = apply_runtime_overrides(
        config,
        greedy_paths_enabled=greedy_paths_enabled,
        drone_ids=drone_ids,
        initial_drone_positions=initial_drone_positions,
        initial_altitudes_agl=initial_altitudes_agl,
        search_decay_percent_per_100ms=search_decay_percent_per_100ms,
        step_seconds=step_seconds,
    )
    save_video_enabled = _resolve_save_video_setting(runtime_config, save_video)
    planner, metadata = build_planner(
        runtime_config,
        image_source=image_source,
        source_image_ppm=source_image_ppm,
    )
    initial_total_heat = float(planner.hmu.total_heat)

    width_m = float(metadata["width"] * runtime_config.heatmap.resolution)
    height_m = float(metadata["height"] * runtime_config.heatmap.resolution)
    geo_corners = resolve_geo_corners(
        runtime_config.geo_reference,
        width_m,
        height_m,
        corners=corners,
    )

    drone_records = [[] for _ in planner.searchers]
    initial_render_state = planner.get_render_state()
    video_writer = None
    if save_video_enabled and video_output:
        video_writer = ReplayVideoWriter(
            video_output,
            runtime_config,
            planner.hmu,
            width_m=width_m,
            height_m=height_m,
        )

    record_render_state(
        initial_render_state,
        drone_records,
        runtime_config.camera,
        width_m=width_m,
        height_m=height_m,
        corners=geo_corners,
        timestamp=0.0,
    )
    if video_writer is not None:
        video_writer.write_frame(initial_render_state, timestamp=0.0)

    try:
        step_index = 0
        while (
            not planner.finished
            and (max_steps is None or step_index < max_steps)
            and not _coverage_stop_reached(
                initial_total_heat,
                planner.hmu.total_heat,
                stop_when_covered_percent,
            )
        ):
            render_state = planner.step()
            step_index += 1
            timestamp = round(step_index * runtime_config.planner.step_seconds, 10)
            record_render_state(
                render_state,
                drone_records,
                runtime_config.camera,
                width_m=width_m,
                height_m=height_m,
                corners=geo_corners,
                timestamp=timestamp,
            )
            if video_writer is not None:
                video_writer.write_frame(render_state, timestamp=timestamp)
    finally:
        if video_writer is not None:
            video_writer.close()

    return {
        drone_id: records
        for drone_id, records in zip(planner.drone_ids, drone_records)
    }


def export_drone_geojsons_from_image(
    image_source: Optional[ImageSource] = None,
    corners: Optional[CornerMap] = None,
    *,
    source_image_ppm: Optional[float] = None,
    max_steps: Optional[int] = 2000,
    config: Optional[SweeperConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    greedy_paths_enabled: Optional[bool] = None,
    drone_ids: Optional[Sequence[str]] = None,
    initial_drone_positions: Optional[Sequence[Mapping[str, object]]] = None,
    initial_altitudes_agl: Optional[Sequence[float]] = None,
    search_decay_percent_per_100ms: Optional[float] = None,
    step_seconds: Optional[float] = None,
    stop_when_covered_percent: Optional[float] = None,
    video_output: Optional[Union[str, Path]] = None,
    save_video: Optional[bool] = None,
):
    """Run the internal simulator and return one per-step JSON timeseries per drone.

    `initial_altitudes_agl` controls each drone's starting altitude above
    ground in meters. Pass one value to apply it to every drone, or pass one
    value per drone. `search_decay_percent_per_100ms` controls how quickly
    observed heat fades: `100` clears immediately and `1` takes about 10
    seconds of continuous observation to clear a max-value cell.
    `image_source`, `source_image_ppm`, and `corners` may be omitted when the
    selected config already provides them. If neither `config` nor
    `config_path` is supplied, the repo's default `config.yaml` is used.
    `drone_ids` and `initial_drone_positions` let callers replace the
    configured drone names and start locations for the export with full lists.
    `step_seconds` controls the simulator timestep duration used for motion,
    decay, and exported timestamps.
    `stop_when_covered_percent` stops the export early once that percentage of
    the initial heatmap value has been cleared by observation.
    `save_video` defaults to `config.export.save_video`. The API only writes a
    replay video when video saving is enabled and `video_output` is provided.
    """
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

    return export_drone_geojsons(
        runtime_config,
        max_steps=max_steps,
        image_source=image_source,
        source_image_ppm=source_image_ppm,
        corners=corners,
        stop_when_covered_percent=stop_when_covered_percent,
        video_output=video_output,
        save_video=save_video,
    )


def resolve_output_paths(output_arg: str, drone_ids) -> Dict[str, Path]:
    output_path = Path(output_arg)
    if output_path.suffix:
        stem = output_path.stem
        suffix = ".json" if output_path.suffix.lower() == ".geojson" else output_path.suffix
        return {
            drone_id: output_path.with_name(f"{stem}_{drone_id}{suffix}")
            for drone_id in drone_ids
        }

    output_path.mkdir(parents=True, exist_ok=True)
    return {drone_id: output_path / f"{drone_id}.json" for drone_id in drone_ids}


def write_drone_geojsons(exports, output_arg: str):
    output_paths = resolve_output_paths(output_arg, exports.keys())
    for drone_id, records in exports.items():
        output_path = output_paths[drone_id]
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_paths


def parse_max_steps_arg(raw_value: str) -> Optional[int]:
    normalized = str(raw_value).strip().lower()
    if normalized in {"none", "null", "unbounded", "until-done"}:
        return None

    parsed = int(raw_value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--max-steps must be non-negative or 'none'.")
    return parsed


def parse_coverage_percent_arg(raw_value: str) -> Optional[float]:
    normalized = str(raw_value).strip().lower()
    if normalized in {"none", "null", "off", "disabled"}:
        return None

    parsed = float(raw_value)
    if not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError(
            "--stop-when-covered-percent must be between 0 and 100 or 'none'."
        )
    return parsed


def parse_cli_corners(args) -> Optional[Dict[str, Tuple[float, float]]]:
    raw_values = (
        args.tl_lat,
        args.tl_lon,
        args.tr_lat,
        args.tr_lon,
        args.br_lat,
        args.br_lon,
        args.bl_lat,
        args.bl_lon,
    )
    if all(value is None for value in raw_values):
        return None

    if any(value is None for value in raw_values):
        raise ValueError("All tl/tr/br/bl corner lat/lon values must be provided together.")

    return {
        "tl": (args.tl_lat, args.tl_lon),
        "tr": (args.tr_lat, args.tr_lon),
        "br": (args.br_lat, args.br_lon),
        "bl": (args.bl_lat, args.bl_lon),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export one per-drone JSON timeseries with drone and camera data."
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to the sweeper config YAML.",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Optional heatmap image path override. Defaults to config.",
    )
    parser.add_argument(
        "--source-image-ppm",
        type=float,
        default=None,
        help="Optional source image pixels-per-meter override. Defaults to config.",
    )
    parser.add_argument(
        "--max-steps",
        type=parse_max_steps_arg,
        default=2000,
        help="Maximum number of planner steps to simulate. Use `none` to run until the planner finishes or another stop condition is reached.",
    )
    parser.add_argument(
        "--stop-when-covered-percent",
        type=parse_coverage_percent_arg,
        default=None,
        help="Stop early once this percentage of the initial heatmap value has been covered. Use `none` to disable.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path or directory. Defaults to stdout when omitted.",
    )
    parser.add_argument(
        "--video-output",
        default="",
        help="Optional replay video output path. Defaults to a sibling `.mp4` or `<dir>/replay.mp4` when `--output` is set.",
    )
    parser.set_defaults(save_video=None)
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        help="Override config and enable replay video generation.",
    )
    video_group.add_argument(
        "--no-video",
        dest="save_video",
        action="store_false",
        help="Override config and disable replay video generation.",
    )
    parser.set_defaults(greedy_paths_enabled=None)
    greedy_group = parser.add_mutually_exclusive_group()
    greedy_group.add_argument(
        "--greedy-paths-enabled",
        dest="greedy_paths_enabled",
        action="store_true",
        help="Override config and enable greedy intermediate waypoints.",
    )
    greedy_group.add_argument(
        "--no-greedy-paths",
        dest="greedy_paths_enabled",
        action="store_false",
        help="Override config and disable greedy intermediate waypoints.",
    )
    parser.add_argument(
        "--initial-altitudes-agl",
        type=float,
        nargs="*",
        default=None,
        help="Optional one-value or per-drone AGL override in meters.",
    )
    parser.add_argument(
        "--search-decay-percent-per-100ms",
        type=float,
        default=None,
        help="Optional search decay override. 100 clears immediately; 1 takes about 10 seconds of continuous observation to clear a max-value cell.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=None,
        help="Optional planner timestep override in seconds.",
    )
    parser.add_argument("--tl-lat", type=float, default=None, help="Top-left latitude.")
    parser.add_argument("--tl-lon", type=float, default=None, help="Top-left longitude.")
    parser.add_argument("--tr-lat", type=float, default=None, help="Top-right latitude.")
    parser.add_argument("--tr-lon", type=float, default=None, help="Top-right longitude.")
    parser.add_argument("--br-lat", type=float, default=None, help="Bottom-right latitude.")
    parser.add_argument("--br-lon", type=float, default=None, help="Bottom-right longitude.")
    parser.add_argument("--bl-lat", type=float, default=None, help="Bottom-left latitude.")
    parser.add_argument("--bl-lon", type=float, default=None, help="Bottom-left longitude.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    corners = parse_cli_corners(args)
    save_video = _resolve_save_video_setting(config, args.save_video)
    video_output = None
    if save_video and (args.video_output or args.output):
        video_output = _resolve_video_output_path(args.output or args.video_output, args.video_output or None)
    exports = export_drone_geojsons(
        config,
        max_steps=args.max_steps,
        image_source=args.image or None,
        source_image_ppm=args.source_image_ppm,
        corners=corners,
        greedy_paths_enabled=args.greedy_paths_enabled,
        initial_altitudes_agl=args.initial_altitudes_agl,
        search_decay_percent_per_100ms=args.search_decay_percent_per_100ms,
        step_seconds=args.step_seconds,
        stop_when_covered_percent=args.stop_when_covered_percent,
        video_output=video_output,
        save_video=save_video,
    )

    if args.output:
        output_paths = write_drone_geojsons(exports, args.output)
        for output_path in output_paths.values():
            print(output_path)
        if video_output is not None:
            print(Path(video_output))
    else:
        print(json.dumps(exports, indent=2))


if __name__ == "__main__":
    main()
