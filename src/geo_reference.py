import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.config import GeoReferenceConfig


CornerValue = Union[Sequence[float], Mapping[str, float]]
CornerMap = Mapping[str, CornerValue]


def meters_per_degree_lat(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return (
        111132.92
        - 559.82 * math.cos(2.0 * lat_rad)
        + 1.175 * math.cos(4.0 * lat_rad)
        - 0.0023 * math.cos(6.0 * lat_rad)
    )


def meters_per_degree_lon(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return (
        111412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3.0 * lat_rad)
        + 0.118 * math.cos(5.0 * lat_rad)
    )


def normalize_corner_value(corner: CornerValue) -> Tuple[float, float]:
    if isinstance(corner, Mapping):
        return float(corner["lat"]), float(corner["lon"])

    if len(corner) != 2:
        raise ValueError("Each corner must provide [lat, lon].")
    return float(corner[0]), float(corner[1])


def normalize_geo_corners(corners: CornerMap) -> Dict[str, Tuple[float, float]]:
    required = ("tl", "tr", "br", "bl")
    missing = [name for name in required if name not in corners]
    if missing:
        raise ValueError(
            "Geo corners must include tl, tr, br, bl. "
            f"Missing: {', '.join(missing)}"
        )

    return {name: normalize_corner_value(corners[name]) for name in required}


def derive_axis_aligned_corners_from_centroid(
    centroid_lat: float,
    centroid_lon: float,
    width_m: float,
    height_m: float,
) -> Dict[str, Tuple[float, float]]:
    half_width_m = width_m / 2.0
    half_height_m = height_m / 2.0
    lat_scale = meters_per_degree_lat(centroid_lat)
    lon_scale = meters_per_degree_lon(centroid_lat)
    north_delta = half_height_m / lat_scale
    east_delta = half_width_m / lon_scale

    return {
        "tl": (centroid_lat + north_delta, centroid_lon - east_delta),
        "tr": (centroid_lat + north_delta, centroid_lon + east_delta),
        "br": (centroid_lat - north_delta, centroid_lon + east_delta),
        "bl": (centroid_lat - north_delta, centroid_lon - east_delta),
    }


def geo_corners_from_config(
    geo_reference: GeoReferenceConfig,
    width_m: float,
    height_m: float,
) -> Dict[str, Tuple[float, float]]:
    if all(
        value is not None
        for value in (
            geo_reference.tl_lat,
            geo_reference.tl_lon,
            geo_reference.tr_lat,
            geo_reference.tr_lon,
            geo_reference.br_lat,
            geo_reference.br_lon,
            geo_reference.bl_lat,
            geo_reference.bl_lon,
        )
    ):
        return {
            "tl": (float(geo_reference.tl_lat), float(geo_reference.tl_lon)),
            "tr": (float(geo_reference.tr_lat), float(geo_reference.tr_lon)),
            "br": (float(geo_reference.br_lat), float(geo_reference.br_lon)),
            "bl": (float(geo_reference.bl_lat), float(geo_reference.bl_lon)),
        }

    if geo_reference.centroid_lat is not None and geo_reference.centroid_lon is not None:
        return derive_axis_aligned_corners_from_centroid(
            float(geo_reference.centroid_lat),
            float(geo_reference.centroid_lon),
            width_m,
            height_m,
        )

    raise ValueError(
        "Geo reference is missing. Provide tl/tr/br/bl corners or a centroid."
    )


def resolve_geo_corners(
    geo_reference: GeoReferenceConfig,
    width_m: float,
    height_m: float,
    corners: Optional[CornerMap] = None,
) -> Dict[str, Tuple[float, float]]:
    if corners is not None:
        return normalize_geo_corners(corners)
    return geo_corners_from_config(geo_reference, width_m, height_m)


def local_point_to_lon_lat(
    point_xy: Sequence[float],
    width_m: float,
    height_m: float,
    corners: Dict[str, Tuple[float, float]],
):
    east_ratio = float(point_xy[0]) / max(width_m, 1e-6)
    south_ratio = (height_m - float(point_xy[1])) / max(height_m, 1e-6)

    east_ratio = min(max(east_ratio, 0.0), 1.0)
    south_ratio = min(max(south_ratio, 0.0), 1.0)

    tl_lat, tl_lon = corners["tl"]
    tr_lat, tr_lon = corners["tr"]
    br_lat, br_lon = corners["br"]
    bl_lat, bl_lon = corners["bl"]

    top_lat = (1.0 - east_ratio) * tl_lat + east_ratio * tr_lat
    top_lon = (1.0 - east_ratio) * tl_lon + east_ratio * tr_lon
    bottom_lat = (1.0 - east_ratio) * bl_lat + east_ratio * br_lat
    bottom_lon = (1.0 - east_ratio) * bl_lon + east_ratio * br_lon

    lat = (1.0 - south_ratio) * top_lat + south_ratio * bottom_lat
    lon = (1.0 - south_ratio) * top_lon + south_ratio * bottom_lon
    return [lon, lat]


def _bilinear_coefficients(corners: Dict[str, Tuple[float, float]]):
    tl_lat, tl_lon = corners["tl"]
    tr_lat, tr_lon = corners["tr"]
    br_lat, br_lon = corners["br"]
    bl_lat, bl_lon = corners["bl"]

    a = np.array([tl_lon, tl_lat], dtype=float)
    b = np.array([tr_lon - tl_lon, tr_lat - tl_lat], dtype=float)
    c = np.array([bl_lon - tl_lon, bl_lat - tl_lat], dtype=float)
    d = np.array(
        [tl_lon - tr_lon - bl_lon + br_lon, tl_lat - tr_lat - bl_lat + br_lat],
        dtype=float,
    )
    return a, b, c, d


def lon_lat_to_local_point(
    lon: float,
    lat: float,
    width_m: float,
    height_m: float,
    corners: Dict[str, Tuple[float, float]],
    max_iterations: int = 8,
):
    target = np.array([float(lon), float(lat)], dtype=float)
    a, b, c, d = _bilinear_coefficients(corners)

    jacobian_affine = np.column_stack((b, c))
    try:
        initial = np.linalg.solve(jacobian_affine, target - a)
    except np.linalg.LinAlgError:
        initial, _, _, _ = np.linalg.lstsq(jacobian_affine, target - a, rcond=None)

    u = float(initial[0])
    v = float(initial[1])

    for _ in range(max_iterations):
        estimate = a + b * u + c * v + d * u * v
        residual = estimate - target

        if float(np.hypot(residual[0], residual[1])) < 1e-12:
            break

        jacobian = np.array(
            [
                [b[0] + d[0] * v, c[0] + d[0] * u],
                [b[1] + d[1] * v, c[1] + d[1] * u],
            ],
            dtype=float,
        )

        try:
            delta = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(jacobian, residual, rcond=None)

        u -= float(delta[0])
        v -= float(delta[1])
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)

    return (
        u * width_m,
        height_m * (1.0 - v),
    )
