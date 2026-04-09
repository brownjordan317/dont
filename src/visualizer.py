import numpy as np

from src.drone_controller.flight_controller import DroneState
from src.drone_controller.motion import normalize_heading_deg
from src.config import SweeperConfig
from src.planner_folder.geometry import proj_to_patch, proj_to_vertices
from src.runtime_builder import build_planner


_ACTIVE_ANIMATION = None


def _lerp(start_value, end_value, alpha):
    return float(start_value) + (float(end_value) - float(start_value)) * float(alpha)


def _lerp_heading_deg(start_heading_deg, end_heading_deg, alpha):
    delta = normalize_heading_deg(float(end_heading_deg) - float(start_heading_deg))
    return normalize_heading_deg(float(start_heading_deg) + delta * float(alpha))


def _lerp_point(start_point, end_point, alpha):
    if start_point is None or end_point is None:
        return end_point if alpha >= 1.0 else start_point

    return (
        _lerp(start_point[0], end_point[0], alpha),
        _lerp(start_point[1], end_point[1], alpha),
    )


def _lerp_projection(start_projection, end_projection, alpha):
    if start_projection is None or end_projection is None:
        return end_projection if alpha >= 1.0 else start_projection

    return {
        key: _lerp_point(start_projection[key], end_projection[key], alpha)
        for key in ("tl", "tr", "br", "bl")
    }


def _interpolate_render_state(start_render_state, end_render_state, alpha):
    if start_render_state is None:
        return end_render_state
    if end_render_state is None:
        return start_render_state

    interpolated_state = []
    for start_item, end_item in zip(start_render_state, end_render_state):
        start_state = start_item["state"]
        end_state = end_item["state"]
        interpolated_state.append(
            {
                "state": DroneState(
                    e=_lerp(start_state.e, end_state.e, alpha),
                    n=_lerp(start_state.n, end_state.n, alpha),
                    heading=_lerp_heading_deg(start_state.heading, end_state.heading, alpha),
                    agl=_lerp(start_state.agl, end_state.agl, alpha),
                ),
                "target": end_item["target"],
                "camera_target": end_item["camera_target"],
                "camera_pitch_deg": _lerp(
                    start_item["camera_pitch_deg"],
                    end_item["camera_pitch_deg"],
                    alpha,
                ),
                "camera_yaw_deg": _lerp(
                    start_item["camera_yaw_deg"],
                    end_item["camera_yaw_deg"],
                    alpha,
                ),
                "camera_projection_point": _lerp_point(
                    start_item["camera_projection_point"],
                    end_item["camera_projection_point"],
                    alpha,
                ),
                "projection": _lerp_projection(
                    start_item["projection"],
                    end_item["projection"],
                    alpha,
                ),
            }
        )

    return interpolated_state


def run_visualiser(config: SweeperConfig):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    planner, metadata = build_planner(config)
    hmu = planner.hmu
    height = metadata["height"]
    width = metadata["width"]
    origin = metadata["origin"]
    cluster_view = hmu.get_cluster_view(config.planner.cluster_size)

    hot_cmap = LinearSegmentedColormap.from_list(
        "heatmap",
        list(config.display.colormap),
        N=256,
    )

    fig = plt.figure(figsize=config.display.figure_size, facecolor=config.display.background_color)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.82])
    ax.set_facecolor(config.display.background_color)
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.imshow(
        cluster_view,
        cmap=hot_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(0, width, height, 0),
        aspect="equal",
    )

    artists_by_searcher = []
    for idx, searcher in enumerate(planner.searchers):
        best_dot = ax.scatter([], [], s=80, c=config.display.target_color, edgecolors="black", zorder=6)
        camera_dot = ax.scatter([], [], s=45, c="#ffd166", edgecolors="black", zorder=6)
        drone_dot = ax.scatter([], [], s=80, c=config.display.drone_color, edgecolors="black", zorder=7)
        fp_patch = proj_to_patch(
            searcher["last_projection"],
            color=config.display.footprint_color,
            alpha=config.display.footprint_alpha,
            utm_origin=origin,
            res=config.heatmap.resolution,
        )
        ax.add_patch(fp_patch)
        artists_by_searcher.append(
            {
                "best_dot": best_dot,
                "camera_dot": camera_dot,
                "drone_dot": drone_dot,
                "fp_patch": fp_patch,
            }
        )
    runtime_interval_ms = max(1, int(round(config.planner.step_seconds * 1000.0)))
    configured_interval_ms = max(1, int(config.display.interval_ms))
    animation_interval_ms = (
        min(configured_interval_ms, runtime_interval_ms)
        if config.display.sync_to_runtime
        else configured_interval_ms
    )
    animation_interval_seconds = animation_interval_ms / 1000.0
    render_step_seconds = max(float(config.planner.step_seconds), 1e-9)
    previous_render_state = planner.get_render_state()
    next_render_state = None
    interpolation_elapsed_seconds = 0.0

    def collect_artists():
        artists = [im]
        for artists_state in artists_by_searcher:
            artists.extend(
                [
                    artists_state["fp_patch"],
                    artists_state["best_dot"],
                    artists_state["camera_dot"],
                    artists_state["drone_dot"],
                ]
            )
        return tuple(artists)

    def apply_render_state(render_state):
        cluster_view = hmu.get_cluster_view(config.planner.cluster_size)
        if cluster_view is not None:
            im.set_data(cluster_view)

        artists = [im]
        for render_item, artists_state in zip(render_state, artists_by_searcher):
            state = render_item["state"]
            current_target = render_item["target"]
            camera_projection_point = render_item["camera_projection_point"]
            proj = render_item["projection"]

            artists_state["fp_patch"].set_xy(proj_to_vertices(proj, origin, config.heatmap.resolution))
            if current_target is None:
                artists_state["best_dot"].set_offsets(np.empty((0, 2)))
            else:
                artists_state["best_dot"].set_offsets([[current_target["target_c"], current_target["target_r"]]])

            # Draw the actual current projection center, not the desired camera
            # target, so the UI reflects the rate-limited camera motion.
            if camera_projection_point is None:
                artists_state["camera_dot"].set_offsets(np.empty((0, 2)))
            else:
                artists_state["camera_dot"].set_offsets(
                    [[
                        camera_projection_point[0] / config.heatmap.resolution,
                        height - camera_projection_point[1] / config.heatmap.resolution,
                    ]]
                )

            artists_state["drone_dot"].set_offsets(
                [[state.e / config.heatmap.resolution, height - state.n / config.heatmap.resolution]]
            )
            artists.extend(
                [
                    artists_state["fp_patch"],
                    artists_state["best_dot"],
                    artists_state["camera_dot"],
                    artists_state["drone_dot"],
                ]
            )

        return tuple(artists)

    def update(_):
        nonlocal previous_render_state, next_render_state, interpolation_elapsed_seconds

        if hmu.get_cluster_view(config.planner.cluster_size) is None:
            ani.event_source.stop()
            return (im,)

        if not config.display.sync_to_runtime:
            if planner.finished:
                ani.event_source.stop()
                return collect_artists()

            render_state = planner.step()
            return apply_render_state(render_state)

        if next_render_state is None and not planner.finished:
            next_render_state = planner.step()
            interpolation_elapsed_seconds = 0.0

        interpolation_elapsed_seconds += animation_interval_seconds

        while (
            next_render_state is not None
            and interpolation_elapsed_seconds >= render_step_seconds
            and not planner.finished
        ):
            previous_render_state = next_render_state
            next_render_state = planner.step()
            interpolation_elapsed_seconds -= render_step_seconds

        if next_render_state is None:
            if planner.finished:
                ani.event_source.stop()
            return apply_render_state(previous_render_state)

        if planner.finished and interpolation_elapsed_seconds >= render_step_seconds:
            ani.event_source.stop()
            return apply_render_state(next_render_state)

        alpha = min(1.0, interpolation_elapsed_seconds / render_step_seconds)
        render_state = _interpolate_render_state(
            previous_render_state,
            next_render_state,
            alpha,
        )
        return apply_render_state(render_state)

    def init():
        return apply_render_state(planner.get_render_state())

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=animation_interval_ms,
        # A full redraw is more reliable across IDE and GUI backends for this
        # mixed image/scatter/polygon scene than blitting.
        blit=False,
    )

    global _ACTIVE_ANIMATION
    _ACTIVE_ANIMATION = ani
    plt.show()
    return ani
