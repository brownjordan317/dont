import numpy as np

from src.heat_map_updates.camera_projection import CameraProjection
from src.heat_map_updates.heatmap_updates import HeatMapUpdates
from src.config import CameraConfig


CAMERA_MATRIX = None
CAMERA_MATRIX_INV = None
CAMERA_CENTER_RAY = None
CAMERA_CORNER_RAYS = None


def configure_camera(config: CameraConfig) -> None:
    global CAMERA_MATRIX, CAMERA_MATRIX_INV, CAMERA_CENTER_RAY, CAMERA_CORNER_RAYS
    CAMERA_MATRIX = np.array(config.matrix, dtype=float)
    CAMERA_MATRIX_INV = np.linalg.inv(CAMERA_MATRIX)
    CAMERA_CENTER_RAY, CAMERA_CORNER_RAYS = CameraProjection.precompute_rays(
        CAMERA_MATRIX,
        camera_matrix_inv=CAMERA_MATRIX_INV,
    )


def build_camera(camera_config: CameraConfig, utm, heading, pitch=None, yaw=None, agl=None):
    if (
        CAMERA_MATRIX is None
        or CAMERA_MATRIX_INV is None
        or CAMERA_CENTER_RAY is None
        or CAMERA_CORNER_RAYS is None
    ):
        configure_camera(camera_config)

    return CameraProjection(
        CAMERA_MATRIX,
        utm,
        camera_config.agl if agl is None else agl,
        heading,
        camera_config.pitch if pitch is None else pitch,
        camera_config.yaw if yaw is None else yaw,
        camera_matrix_inv=CAMERA_MATRIX_INV,
        center_ray=CAMERA_CENTER_RAY,
        corner_rays=CAMERA_CORNER_RAYS,
    )


def projection_span_meters(camera_config: CameraConfig, utm, heading, pitch=None, yaw=None, agl=None):
    proj = build_camera(camera_config, utm, heading, pitch=pitch, yaw=yaw, agl=agl).project()
    corners = np.array([proj[key] for key in ("tl", "tr", "br", "bl")], dtype=float)
    deltas = corners[:, None, :] - corners[None, :, :]
    distances = np.hypot(deltas[..., 0], deltas[..., 1])
    return float(distances.max())


def proj_to_vertices(proj, utm_origin=(0, 0), res=1.0):
    def to_px(easting, northing):
        return (
            (easting - utm_origin[0]) / res,
            (utm_origin[1] - northing) / res,
        )

    corners = [to_px(*proj[key]) for key in ("tl", "tr", "br", "bl")]
    return np.array(corners, dtype=float)


def proj_to_patch(proj, color="cyan", alpha=0.25, utm_origin=(0, 0), res=1.0):
    import matplotlib.patches as mpatches

    return mpatches.Polygon(
        proj_to_vertices(proj, utm_origin, res),
        closed=True,
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
    )


def projection_roi_mask(proj, origin, resolution, data_shape):
    poly = proj_to_vertices(proj, origin, resolution)
    col_min = max(int(np.floor(poly[:, 0].min())), 0)
    col_max = min(int(np.ceil(poly[:, 0].max())), data_shape[1] - 1)
    row_min = max(int(np.floor(poly[:, 1].min())), 0)
    row_max = min(int(np.ceil(poly[:, 1].max())), data_shape[0] - 1)
    if row_min > row_max or col_min > col_max:
        return None

    mask = HeatMapUpdates._roi_polygon_mask(poly, row_min, row_max, col_min, col_max)
    if not np.any(mask):
        return None

    return row_min, row_max, col_min, col_max, mask
