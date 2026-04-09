import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = None

from src import create_incremental_session_from_image, export_drone_geojsons_from_image
from src.config import SweeperConfig, load_config


DRONE_COLORS = ["#ff4d4d", "#00b3ff", "#2fbf71", "#ffb347", "#8b5cf6", "#14b8a6"]
DEFAULT_EXPORT_STEPS = 100
DEFAULT_EXPORT_COVERAGE_STOP_PERCENT = 2.0
DEFAULT_SESSION_STEP_SECONDS = 0.2
DEFAULT_REPLAY_STEPS = 24
DEFAULT_REPLAY_INTERVAL_MS = 700
DEFAULT_REPLAY_FPS = 2


@dataclass(frozen=True)
class ShowcaseContext:
    repo_root: Path
    config_path: Path
    config: SweeperConfig
    corners: dict[str, dict[str, float]]
    image_path: Path


@dataclass(frozen=True)
class IncrementalReplay:
    step_seconds: float
    max_steps: int
    frames: list[dict[str, Any]]


def build_showcase_context(
    repo_root: Path,
    *,
    config_path: str | Path = "config.yaml",
) -> ShowcaseContext:
    resolved_root = Path(repo_root).resolve()
    resolved_config_path = (resolved_root / config_path).resolve()
    config = load_config(resolved_config_path)
    return ShowcaseContext(
        repo_root=resolved_root,
        config_path=resolved_config_path,
        config=config,
        corners=build_corners(config),
        image_path=Path(config.heatmap.image_path),
    )


def build_corners(config: SweeperConfig) -> dict[str, dict[str, float]]:
    return {
        "tl": {"lat": config.geo_reference.tl_lat, "lon": config.geo_reference.tl_lon},
        "tr": {"lat": config.geo_reference.tr_lat, "lon": config.geo_reference.tr_lon},
        "br": {"lat": config.geo_reference.br_lat, "lon": config.geo_reference.br_lon},
        "bl": {"lat": config.geo_reference.bl_lat, "lon": config.geo_reference.bl_lon},
    }


def build_config_summary(context: ShowcaseContext) -> dict[str, Any]:
    config = context.config
    return {
        "config_path": str(context.config_path),
        "image_path": str(context.image_path),
        "source_image_ppm": config.heatmap.source_image_ppm,
        "resolution_m_per_px": config.heatmap.resolution,
        "search_decay_percent_per_100ms": config.heatmap.search_decay_percent_per_100ms,
        "drone_speed_mps": config.planner.drone_speed,
        "step_seconds": config.planner.step_seconds,
        "drone_ids": list(config.planner.drone_ids),
        "initial_drone_positions": [
            {
                key: value
                for key, value in {
                    "e": position.e,
                    "n": position.n,
                    "lat": position.lat,
                    "lon": position.lon,
                    "heading": position.heading,
                }.items()
                if value is not None
            }
            for position in config.planner.initial_drone_positions
        ],
        "initial_altitudes_agl_m": list(config.planner.initial_altitudes_agl),
        "export_save_video_default": config.export.save_video,
        "corners": context.corners,
    }


def build_api_call_examples(context: ShowcaseContext) -> dict[str, str]:
    try:
        config_path = str(context.config_path.relative_to(context.repo_root))
    except ValueError:
        config_path = str(context.config_path)

    return {
        "api_v1_config_object": (
            "export_drone_geojsons_from_image("
            "config=context.config, max_steps=100)"
        ),
        "api_v1_custom_ids_and_starts": (
            "export_drone_geojsons_from_image("
            "config=context.config, "
            "drone_ids=['alpha', 'beta', 'charlie', 'delta'], "
            "initial_drone_positions=["
            "{'e': 400.0, 'n': 1600.0, 'heading': 0.0}, "
            "{'e': 600.0, 'n': 1600.0, 'heading': 90.0}, "
            "{'lat': 47.9188, 'lon': -97.0898, 'heading': 180.0}, "
            "{'lat': 47.9186, 'lon': -97.0906, 'heading': 270.0}"
            "])"
        ),
        "api_v1_config_path": (
            f'export_drone_geojsons_from_image(config_path="{config_path}", max_steps=100)'
        ),
        "api_v2_config_object": (
            "create_incremental_session_from_image(config=context.config)"
        ),
        "api_v2_custom_ids_and_starts": (
            "create_incremental_session_from_image("
            "config=context.config, "
            "drone_ids=['alpha', 'beta', 'charlie', 'delta'], "
            "initial_drone_positions=["
            "{'e': 400.0, 'n': 1600.0, 'heading': 0.0}, "
            "{'e': 600.0, 'n': 1600.0, 'heading': 90.0}, "
            "{'lat': 47.9188, 'lon': -97.0898, 'heading': 180.0}, "
            "{'lat': 47.9186, 'lon': -97.0906, 'heading': 270.0}"
            "])"
        ),
        "api_v2_config_path": (
            f'create_incremental_session_from_image(config_path="{config_path}")'
        ),
    }


def show_payload(title: str, payload: Any, limit: int | None = None) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    if limit is not None and isinstance(payload, list):
        payload = payload[:limit]
    print(json.dumps(payload, indent=2))


def first_drone_id(drone_payloads: Mapping[str, Any]) -> str:
    return next(iter(drone_payloads))


def run_export_demo(
    context: ShowcaseContext,
    *,
    max_steps: Optional[int] = DEFAULT_EXPORT_STEPS,
    stop_when_covered_percent: Optional[float] = None,
    save_video: Optional[bool] = None,
    video_output: Optional[str | Path] = None,
) -> dict[str, list[dict[str, Any]]]:
    return export_drone_geojsons_from_image(
        config=context.config,
        max_steps=max_steps,
        stop_when_covered_percent=stop_when_covered_percent,
        save_video=save_video,
        video_output=video_output,
    )


def summarize_export_window(exports: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    if not exports:
        return {
            "drone_count": 0,
            "records_per_drone": {},
            "simulated_steps": 0,
            "last_timestamp": 0.0,
            "timestamp_spacing": None,
        }

    first_records = next(iter(exports.values()))
    timestamp_spacing = None
    if len(first_records) > 1:
        timestamp_spacing = first_records[1]["timestamp"] - first_records[0]["timestamp"]

    return {
        "drone_count": len(exports),
        "records_per_drone": {
            drone_id: len(records)
            for drone_id, records in exports.items()
        },
        "simulated_steps": max(0, len(first_records) - 1),
        "last_timestamp": first_records[-1]["timestamp"] if first_records else 0.0,
        "timestamp_spacing": timestamp_spacing,
    }


def summarize_exports(exports: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    export_summary: dict[str, dict[str, Any]] = {}
    for drone_id, records in exports.items():
        export_summary[drone_id] = {
            "records": len(records),
            "first_timestamp": records[0]["timestamp"],
            "last_timestamp": records[-1]["timestamp"],
            "final_drone_lat": records[-1]["drone_lat"],
            "final_drone_lon": records[-1]["drone_lon"],
            "final_cam_proj_lat": records[-1]["cam_proj_lat"],
            "final_cam_proj_lon": records[-1]["cam_proj_lon"],
        }
    return export_summary


def create_demo_session(
    context: ShowcaseContext,
    *,
    step_seconds: float = DEFAULT_SESSION_STEP_SECONDS,
):
    return create_incremental_session_from_image(
        config=context.config,
        step_seconds=step_seconds,
    )


def observation_from_plan(drone_payload: Mapping[str, Any]) -> dict[str, Any]:
    state = drone_payload["state"]
    observation = {
        "position": {
            "lat": state["lat"],
            "lon": state["lon"],
            "heading": state["heading"],
            "agl": state["agl"],
        }
    }

    projection_point = drone_payload["camera_projection_point"]
    if projection_point is not None:
        observation["projection_point"] = {
            "lat": projection_point["lat"],
            "lon": projection_point["lon"],
        }

    return observation


def build_observations_from_plan(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        drone_id: observation_from_plan(drone_payload)
        for drone_id, drone_payload in plan["drones"].items()
    }


def replay_summary_text(replay: IncrementalReplay) -> str:
    return (
        f"Built {len(replay.frames)} replay frames "
        f"({max(0, len(replay.frames) - 1)} observe_and_plan updates "
        f"at {replay.step_seconds:.2f}s each)."
    )


def build_incremental_replay(
    context: ShowcaseContext,
    *,
    step_seconds: float = DEFAULT_SESSION_STEP_SECONDS,
    max_steps: int = DEFAULT_REPLAY_STEPS,
) -> IncrementalReplay:
    replay_session = create_demo_session(context, step_seconds=step_seconds)
    replay_exports = export_drone_geojsons_from_image(
        config=context.config,
        max_steps=max_steps,
        step_seconds=step_seconds,
    )

    record_count = min(len(records) for records in replay_exports.values())
    frames = [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "observations": None,
            "result": replay_session.get_plan(),
        }
    ]

    for record_index in range(1, record_count):
        observations = {
            drone_id: observation_from_export_record(records[record_index])
            for drone_id, records in replay_exports.items()
        }
        result = replay_session.observe_and_plan(
            observations,
            dt_seconds=step_seconds,
        )
        frames.append(
            {
                "frame_index": record_index,
                "timestamp": replay_exports[first_drone_id(replay_exports)][record_index]["timestamp"],
                "observations": observations,
                "result": result,
            }
        )

    return IncrementalReplay(
        step_seconds=float(step_seconds),
        max_steps=int(max_steps),
        frames=frames,
    )


def plot_search_map(context: ShowcaseContext):
    fig, ax = plt.subplots(figsize=(10, 8))
    image_artist = add_geo_image(ax, context.image_path, context.corners, alpha=0.95)
    draw_geo_frame(ax, context.corners)

    for label, point in context.corners.items():
        ax.scatter(point["lon"], point["lat"], s=130, color="white", edgecolor="black", zorder=4)
        ax.text(
            point["lon"],
            point["lat"],
            f" {label.upper()}",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            zorder=5,
        )

    fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04, label="Relative heat intensity")
    finalize_geo_axes(ax, "Search Input Heatmap (Geo-Referenced)")
    plt.show()
    return fig, ax


def plot_export_paths(
    context: ShowcaseContext,
    exports: Mapping[str, list[Mapping[str, Any]]],
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    path_ax, camera_ax = axes

    for ax in axes:
        add_geo_image(ax, context.image_path, context.corners, alpha=0.22)
        draw_geo_frame(ax, context.corners)
        ax.set_facecolor("#faf7f2")
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    for idx, (drone_id, records) in enumerate(exports.items()):
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]
        drone_lons = [record["drone_lon"] for record in records]
        drone_lats = [record["drone_lat"] for record in records]
        camera_points = [
            (record["cam_proj_lon"], record["cam_proj_lat"])
            for record in records
            if record["cam_proj_lon"] is not None and record["cam_proj_lat"] is not None
        ]

        path_ax.plot(drone_lons, drone_lats, color=color, linewidth=3.5, marker="o", markersize=6, label=drone_id, zorder=4)
        path_ax.scatter(drone_lons[0], drone_lats[0], color=color, marker="s", s=140, edgecolor="black", zorder=5)
        path_ax.scatter(drone_lons[-1], drone_lats[-1], color=color, marker="X", s=190, edgecolor="black", linewidth=1.2, zorder=6)
        path_ax.annotate(
            drone_id,
            (drone_lons[-1], drone_lats[-1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            zorder=7,
        )

        if camera_points:
            cam_lons, cam_lats = zip(*camera_points)
            camera_ax.plot(cam_lons, cam_lats, color=color, linewidth=2.8, marker="o", markersize=5, label=drone_id, zorder=4)
            camera_ax.scatter(cam_lons[0], cam_lats[0], color=color, marker="s", s=140, edgecolor="black", zorder=5)
            camera_ax.scatter(cam_lons[-1], cam_lats[-1], color=color, marker="X", s=190, edgecolor="black", linewidth=1.2, zorder=6)

            for drone_lon, drone_lat, cam_lon, cam_lat in zip(drone_lons, drone_lats, cam_lons, cam_lats):
                camera_ax.plot([drone_lon, cam_lon], [drone_lat, cam_lat], color=color, linewidth=1.1, alpha=0.18, zorder=2)

    path_ax.set_title("Exported Drone Paths", fontsize=16, fontweight="bold")
    camera_ax.set_title("Camera Projection Tracks", fontsize=16, fontweight="bold")
    path_ax.text(0.02, 0.98, "Square = start\nX = final position", transform=path_ax.transAxes, va="top", ha="left", fontsize=11, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9})
    camera_ax.text(0.02, 0.98, "Thin connectors link each drone\nto its projected ground point", transform=camera_ax.transAxes, va="top", ha="left", fontsize=11, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9})
    path_ax.legend(title="Drone ID", loc="best")
    camera_ax.legend(title="Drone ID", loc="best")
    plt.show()
    return fig, axes


def plot_session_recommendations(
    context: ShowcaseContext,
    result: Mapping[str, Any],
    *,
    title: str = "Incremental Session Recommendations",
):
    fig, ax = plt.subplots(figsize=(12, 9))
    _draw_session_state(
        ax,
        context,
        result,
        title=title,
        subtitle="Large markers and arrows are intentional here so the planner output is easy to scan.",
    )
    plt.show()
    return fig, ax


def render_incremental_replay(
    context: ShowcaseContext,
    replay: IncrementalReplay,
    *,
    interval_ms: int = DEFAULT_REPLAY_INTERVAL_MS,
    fps: int = DEFAULT_REPLAY_FPS,
):
    fig, ax = plt.subplots(figsize=(13, 9))

    if HTML is None or display is None:
        _draw_replay_frame(ax, context, replay, len(replay.frames) - 1)
        print("Rich playback controls require an IPython notebook frontend. Re-run this cell inside Jupyter to get play/pause/rewind/step controls.")
        plt.show()
        return None

    def update(frame_index: int):
        _draw_replay_frame(ax, context, replay, frame_index)
        return ()

    replay_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(replay.frames),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    display(HTML(replay_animation.to_jshtml(fps=fps, default_mode="once")))
    plt.close(fig)
    return replay_animation


def observation_from_export_record(record: Mapping[str, Any]) -> dict[str, Any]:
    observation = {
        "position": {
            "lat": record["drone_lat"],
            "lon": record["drone_lon"],
            "heading": record["bearing"],
            "agl": record["agl"],
        }
    }

    if record["cam_proj_lat"] is not None and record["cam_proj_lon"] is not None:
        observation["projection_point"] = {
            "lat": record["cam_proj_lat"],
            "lon": record["cam_proj_lon"],
        }

    return observation


def geo_bounds(corners: Mapping[str, Mapping[str, float]]) -> tuple[float, float, float, float]:
    lons = [point["lon"] for point in corners.values()]
    lats = [point["lat"] for point in corners.values()]
    return min(lons), max(lons), min(lats), max(lats)


def draw_geo_frame(ax, corners: Mapping[str, Mapping[str, float]]) -> None:
    order = ["tl", "tr", "br", "bl", "tl"]
    ax.plot(
        [corners[key]["lon"] for key in order],
        [corners[key]["lat"] for key in order],
        color="black",
        linewidth=2.0,
        alpha=0.75,
        zorder=3,
    )


def add_geo_image(ax, image_path: Path, corners: Mapping[str, Mapping[str, float]], alpha: float = 0.45):
    preview = plt.imread(image_path)
    if preview.ndim == 3:
        preview = preview[..., :3].mean(axis=2)

    lon_min, lon_max, lat_min, lat_max = geo_bounds(corners)
    return ax.imshow(
        preview,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        cmap="inferno",
        alpha=alpha,
        aspect="auto",
        zorder=1,
    )


def finalize_geo_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)


def _draw_replay_frame(ax, context: ShowcaseContext, replay: IncrementalReplay, frame_index: int) -> None:
    frame = replay.frames[frame_index]
    trails = {
        drone_id: [
            replay.frames[past_index]["result"]["drones"][drone_id]["state"]
            for past_index in range(frame_index + 1)
        ]
        for drone_id in frame["result"]["drones"]
    }
    subtitle = (
        "Use the controls below to play, pause, rewind, or step frame by frame.\n"
        f"Observation timestamp: {frame['timestamp']:.2f}s"
    )
    _draw_session_state(
        ax,
        context,
        frame["result"],
        title=f"Incremental Session Playback | Frame {frame_index}/{len(replay.frames) - 1}",
        subtitle=subtitle,
        trails=trails,
        observations=frame["observations"],
    )


def _draw_session_state(
    ax,
    context: ShowcaseContext,
    result: Mapping[str, Any],
    *,
    title: str,
    subtitle: Optional[str] = None,
    trails: Optional[Mapping[str, list[Mapping[str, Any]]]] = None,
    observations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    ax.clear()
    add_geo_image(ax, context.image_path, context.corners, alpha=0.22)
    draw_geo_frame(ax, context.corners)
    ax.set_facecolor("#faf7f2")
    finalize_geo_axes(ax, title)

    if subtitle is not None:
        ax.text(
            0.02,
            0.98,
            subtitle,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
            zorder=20,
        )

    for idx, (drone_id, drone_payload) in enumerate(result["drones"].items()):
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]
        state = drone_payload["state"]
        next_target = drone_payload["next_target"]
        camera_target = drone_payload["camera_target"]
        projection_point = drone_payload["camera_projection_point"]
        trail = None if trails is None else trails.get(drone_id)
        observed_position = None
        if observations is not None and drone_id in observations:
            observed_position = observations[drone_id].get("position")

        if trail:
            ax.plot(
                [item["lon"] for item in trail],
                [item["lat"] for item in trail],
                color=color,
                linewidth=2.8,
                alpha=0.55,
                zorder=3,
            )

        ax.scatter(state["lon"], state["lat"], s=230, color=color, edgecolor="black", zorder=6)
        ax.text(
            state["lon"],
            state["lat"],
            f" {drone_id}",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            zorder=7,
        )

        if observed_position is not None:
            ax.scatter(
                observed_position["lon"],
                observed_position["lat"],
                s=95,
                marker="s",
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=7,
            )

        if next_target is not None:
            ax.annotate(
                "",
                xy=(next_target["lon"], next_target["lat"]),
                xytext=(state["lon"], state["lat"]),
                arrowprops={"arrowstyle": "-|>", "linewidth": 3.0, "color": color, "alpha": 0.95},
                zorder=5,
            )
            ax.scatter(next_target["lon"], next_target["lat"], s=230, marker="^", color=color, edgecolor="black", zorder=7)

        if camera_target is not None:
            ax.annotate(
                "",
                xy=(camera_target["lon"], camera_target["lat"]),
                xytext=(state["lon"], state["lat"]),
                arrowprops={"arrowstyle": "-|>", "linewidth": 2.0, "linestyle": "--", "color": color, "alpha": 0.8},
                zorder=4,
            )
            ax.scatter(camera_target["lon"], camera_target["lat"], s=150, marker="D", color=color, edgecolor="black", alpha=0.95, zorder=7)

        if projection_point is not None:
            ax.scatter(
                projection_point["lon"],
                projection_point["lat"],
                s=190,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=7,
            )
            ax.plot(
                [state["lon"], projection_point["lon"]],
                [state["lat"], projection_point["lat"]],
                color=color,
                linewidth=1.5,
                alpha=0.35,
                zorder=2,
            )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Session state", markerfacecolor="#444", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="^", color="w", label="Next flight target", markerfacecolor="#444", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="D", color="w", label="Camera target", markerfacecolor="#444", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="X", color="w", label="Current projection point", markerfacecolor="#444", markeredgecolor="black", markersize=10),
        Line2D([0], [0], color="#444", linewidth=3, label="Flight-target arrow"),
        Line2D([0], [0], color="#444", linewidth=2, linestyle="--", label="Camera-target arrow"),
    ]
    if observations is not None:
        legend_handles.insert(
            1,
            Line2D([0], [0], marker="s", color="w", label="Observed external position", markerfacecolor="white", markeredgecolor="#444", markersize=9),
        )
    if trails is not None:
        legend_handles.append(Line2D([0], [0], color="#444", linewidth=3, alpha=0.55, label="Cumulative trail"))

    ax.legend(handles=legend_handles, loc="upper right")


__all__ = [
    "DEFAULT_EXPORT_COVERAGE_STOP_PERCENT",
    "DEFAULT_EXPORT_STEPS",
    "DEFAULT_REPLAY_STEPS",
    "DEFAULT_SESSION_STEP_SECONDS",
    "IncrementalReplay",
    "ShowcaseContext",
    "build_api_call_examples",
    "build_config_summary",
    "build_corners",
    "build_incremental_replay",
    "build_observations_from_plan",
    "build_showcase_context",
    "create_demo_session",
    "first_drone_id",
    "observation_from_plan",
    "plot_export_paths",
    "plot_search_map",
    "plot_session_recommendations",
    "render_incremental_replay",
    "replay_summary_text",
    "run_export_demo",
    "show_payload",
    "summarize_export_window",
    "summarize_exports",
]
