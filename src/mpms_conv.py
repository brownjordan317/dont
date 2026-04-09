"""
convert_drone_json.py

Reads drone telemetry JSON files and produces two GeoJSON files per input:
  1. <stem>_drone_path.geojson   — RDP-simplified LineString of the drone path
  2. <stem>_camera_path.geojson  — RDP-simplified LineString of camera ground
                                   projections, with summarised gimbal angles
                                   as LineString properties

Usage:
    python convert_drone_json.py --in-dir <dir> [--out-dir <dir>] [--tolerance <float>]

Arguments:
    --in-dir      Directory containing drone_N.json files
    --out-dir     Directory for output files (default: same as --in-dir)
    --tolerance   RDP epsilon in degrees (default: 1e-6 ≈ ~0.1 m)
"""

import json
import math
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# RDP simplification
# ---------------------------------------------------------------------------

def _perp_distance(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x0 - x1, y0 - y1)
    t = max(0.0, min(1.0, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x0 - (x1 + t * dx), y0 - (y1 + t * dy))


def rdp(coords, tolerance):
    """Ramer-Douglas-Peucker — returns sorted list of indices to keep."""
    if len(coords) < 3:
        return list(range(len(coords)))
    stack = [(0, len(coords) - 1)]
    keep = {0, len(coords) - 1}
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        max_dist, max_idx = 0.0, start
        for i in range(start + 1, end):
            d = _perp_distance(coords[i][:2], coords[start][:2], coords[end][:2])
            if d > max_dist:
                max_dist, max_idx = d, i
        if max_dist > tolerance:
            keep.add(max_idx)
            stack.append((start, max_idx))
            stack.append((max_idx, end))
    return sorted(keep)


# ---------------------------------------------------------------------------
# Gimbal angles
# ---------------------------------------------------------------------------

def vector_to_gimbal_angles(cam_x, cam_y, cam_z):
    """Camera unit vector (body frame) → (tilt_from_nadir, pan_from_forward, elevation) degrees."""
    cam_z = max(-1.0, min(1.0, cam_z))
    tilt = math.degrees(math.acos(abs(cam_z)))
    elevation = math.degrees(math.asin(-cam_z))
    h = math.sqrt(cam_x ** 2 + cam_y ** 2)
    pan = math.degrees(math.atan2(cam_x, cam_y)) % 360 if h > 1e-9 else float("nan")
    return tilt, pan, elevation


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------

def linestring_feature(coords, properties):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": properties,
    }


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def simplify_telemetry_records(records, tolerance: float, resolution: int = 1):
    """Downsample records, simplify both paths, and build GeoJSON payloads."""
    if resolution < 1:
        raise ValueError("resolution must be >= 1")

    records = list(records)
    original_record_count = len(records)
    simplified_records = records[::resolution] if resolution > 1 else list(records)

    drone_coords = []
    cam_coords = []
    cam_props = []

    for rec in simplified_records:
        drone_coords.append([rec["drone_lon"], rec["drone_lat"], rec["agl"]])

        tilt, pan, elev = vector_to_gimbal_angles(rec["cam_x"], rec["cam_y"], rec["cam_z"])
        cam_coords.append([rec["cam_proj_lon"], rec["cam_proj_lat"]])
        cam_props.append({
            "tilt_from_nadir_deg":    round(tilt, 6),
            "pan_from_forward_deg":   round(pan, 6) if not math.isnan(pan) else None,
            "elevation_deg":          round(elev, 6),
            "absolute_azimuth_deg":   round((rec["bearing"] + (pan if not math.isnan(pan) else 0)) % 360, 6),
        })

    drone_idx = rdp(drone_coords, tolerance)
    cam_idx   = rdp(cam_coords, tolerance)

    def mean_of(key):
        vals = [p[key] for p in cam_props if p[key] is not None]
        return round(sum(vals) / len(vals), 6) if vals else None

    return {
        "original_record_count": original_record_count,
        "sampled_record_count": len(simplified_records),
        "drone_original_point_count": len(drone_coords),
        "drone_simplified_point_count": len(drone_idx),
        "camera_original_point_count": len(cam_coords),
        "camera_simplified_point_count": len(cam_idx),
        "drone_feature_collection": feature_collection([
            linestring_feature(
                [drone_coords[i] for i in drone_idx],
                {
                    "original_point_count": len(drone_coords),
                    "simplified_point_count": len(drone_idx),
                },
            )
        ]),
        "camera_feature_collection": feature_collection([
            linestring_feature(
                [cam_coords[i] for i in cam_idx],
                {
                    "tilt_from_nadir_deg_mean": mean_of("tilt_from_nadir_deg"),
                    "elevation_deg_mean": mean_of("elevation_deg"),
                    "absolute_azimuth_deg_mean": mean_of("absolute_azimuth_deg"),
                    "original_point_count": len(cam_coords),
                    "simplified_point_count": len(cam_idx),
                },
            )
        ]),
    }


def write_simplified_geojsons(simplified, output_dir: Path, stem: str):
    """Write simplified drone and camera GeoJSON payloads for one telemetry stem."""
    output_dir.mkdir(parents=True, exist_ok=True)

    drone_path = output_dir / f"{stem}_drone_path.geojson"
    drone_path.write_text(
        json.dumps(simplified["drone_feature_collection"], indent=2),
        encoding="utf-8",
    )

    cam_path = output_dir / f"{stem}_camera_path.geojson"
    cam_path.write_text(
        json.dumps(simplified["camera_feature_collection"], indent=2),
        encoding="utf-8",
    )

    return drone_path, cam_path


def convert(input_path: Path, output_dir: Path, tolerance: float, resolution: int = 1):
    print(f"Reading: {input_path}")
    records = json.loads(input_path.read_text())
    print(f"  Records loaded: {len(records)}")

    simplified = simplify_telemetry_records(
        records,
        tolerance=tolerance,
        resolution=resolution,
    )

    if resolution > 1:
        print(f"  After resolution={resolution}: {simplified['sampled_record_count']} records")

    print(
        "  Drone:  "
        f"{simplified['drone_original_point_count']} → "
        f"{simplified['drone_simplified_point_count']} pts after RDP"
    )
    print(
        "  Camera: "
        f"{simplified['camera_original_point_count']} → "
        f"{simplified['camera_simplified_point_count']} pts after RDP"
    )

    drone_path, cam_path = write_simplified_geojsons(
        simplified,
        output_dir=output_dir,
        stem=input_path.stem,
    )

    print(f"  Written: {drone_path}")
    print(f"  Written: {cam_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert drone telemetry JSON → RDP-simplified GeoJSON LineStrings"
    )
    parser.add_argument("--in-dir",     type=Path, required=True)
    parser.add_argument("--out-dir",    type=Path, default=None)
    parser.add_argument("--tolerance",  type=float, default=1e-4,
                        help="RDP epsilon in degrees (default: 1e-6 ≈ 0.1 m)")
    parser.add_argument("--resolution", type=int, default=1,
                        help="Keep every Nth point before RDP (default: 1 = all points)")
    args = parser.parse_args()

    if not args.in_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.in_dir}")

    json_files = sorted(args.in_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in: {args.in_dir}")

    out_base = (args.out_dir or args.in_dir).resolve()

    for path in json_files:
        convert(path.resolve(), out_base / path.stem, args.tolerance, args.resolution)


if __name__ == "__main__":
    main()
