"""Helpers used by the quick satellite notebook."""

from __future__ import annotations

import concurrent.futures
import io
import math
import os
from typing import Any

import numpy as np
from PIL import Image


_XYZ_TILES = {
    "OPENSTREETMAP": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "ROADMAP": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
    "SATELLITE": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "TERRAIN": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
    "HYBRID": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
}


def _check_bbox(bbox: list[float]) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(
            "bbox must be a list of 4 coordinates in the format "
            "[min_lon, min_lat, max_lon, max_lat]"
        )


def _resolution_to_zoom_level(resolution: float) -> int:
    initial_resolution = 156543.03392804097
    return int(math.log2(initial_resolution / resolution))


def _deg2num(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat_r = math.radians(lat)
    n = 2**zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return xtile, ytile


def _from4326_to3857(lat: float, lon: float) -> tuple[float, float]:
    radius = 6378137.0
    x = math.radians(lon) * radius
    y = math.log(math.tan(math.radians(45.0 + lat / 2.0))) * radius
    return x, y


def _normalize_mask_geometries(mask_geometry: Any) -> list[dict[str, Any]]:
    if mask_geometry is None:
        return []

    if isinstance(mask_geometry, dict):
        geometry_type = mask_geometry.get("type")
        if geometry_type == "FeatureCollection":
            return [
                feature["geometry"]
                for feature in mask_geometry.get("features", [])
                if feature.get("geometry") is not None
            ]
        if geometry_type == "Feature":
            geometry = mask_geometry.get("geometry")
            return [] if geometry is None else [geometry]
        return [mask_geometry]

    raise TypeError(
        "mask_geometry must be a GeoJSON geometry, Feature, or FeatureCollection."
    )


def _resolve_tile_source(source: str) -> str:
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if source.upper() in _XYZ_TILES:
        return _XYZ_TILES[source.upper()]
    if source.startswith("http"):
        return source
    raise ValueError(
        'source must be one of "OpenStreetMap", "ROADMAP", "SATELLITE", '
        '"TERRAIN", "HYBRID", or a full tile URL'
    )


def _get_http_session():
    try:
        import httpx

        return httpx.Client()
    except ImportError:
        import requests

        return requests.Session()


def _ensure_output_path(file_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(file_path))
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    return path


def bbox_to_polygon(bbox: list[float]) -> dict[str, Any]:
    """Convert a bbox to a GeoJSON polygon in EPSG:4326."""

    _check_bbox(bbox)
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }


def square_bbox(
    bbox: list[float],
    crs: str = "EPSG:4326",
    working_crs: str = "EPSG:3857",
) -> list[float]:
    """Expand a bbox to a square while keeping its center fixed."""

    from pyproj import Transformer

    _check_bbox(bbox)
    west, south, east, north = bbox

    forward = Transformer.from_crs(crs, working_crs, always_xy=True)
    inverse = Transformer.from_crs(working_crs, crs, always_xy=True)

    minx, miny = forward.transform(west, south)
    maxx, maxy = forward.transform(east, north)

    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    side = max(abs(maxx - minx), abs(maxy - miny))

    square_minx = center_x - side / 2.0
    square_miny = center_y - side / 2.0
    square_maxx = center_x + side / 2.0
    square_maxy = center_y + side / 2.0

    square_west, square_south = inverse.transform(square_minx, square_miny)
    square_east, square_north = inverse.transform(square_maxx, square_maxy)
    return [square_west, square_south, square_east, square_north]


def estimate_tms_request(
    bbox: list[float],
    zoom: int | None = None,
    resolution: float | None = None,
) -> dict[str, int]:
    """Estimate zoom level, tile count, and output pixel size for a request."""

    _check_bbox(bbox)

    if zoom is None and resolution is None:
        raise ValueError("Either zoom or resolution must be provided")
    if zoom is not None and resolution is not None:
        raise ValueError("Only one of zoom or resolution can be provided")

    if resolution is not None:
        zoom = _resolution_to_zoom_level(resolution)

    west, south, east, north = bbox
    x0, y0 = _deg2num(south, west, zoom)
    x1, y1 = _deg2num(north, east, zoom)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    xtiles = int(math.ceil(x1) - math.floor(x0))
    ytiles = int(math.ceil(y1) - math.floor(y0))
    pixel_width = int(round(256 * (x1 - x0)))
    pixel_height = int(round(256 * (y1 - y0)))

    return {
        "zoom": int(zoom),
        "tile_count": xtiles * ytiles,
        "xtiles": xtiles,
        "ytiles": ytiles,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
    }


def apply_mask_to_geotiff(
    image_path: str,
    mask_geometry: Any,
    geometry_crs: str = "EPSG:4326",
    fill_value: int = 0,
) -> None:
    """Fill everything outside the provided geometry with a solid value."""

    import rasterio as rio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom

    geometries = _normalize_mask_geometries(mask_geometry)
    if not geometries:
        return

    with rio.open(image_path, "r+") as dataset:
        dataset_crs = dataset.crs.to_string() if dataset.crs is not None else geometry_crs
        transformed = [
            transform_geom(geometry_crs, dataset_crs, geometry) for geometry in geometries
        ]
        inside_mask = geometry_mask(
            transformed,
            out_shape=(dataset.height, dataset.width),
            transform=dataset.transform,
            invert=True,
        )

        data = dataset.read()
        data[:, ~inside_mask] = fill_value
        dataset.write(data)


def tms_to_geotiff(
    output: str,
    bbox: list[float],
    zoom: int | None = None,
    resolution: float | None = None,
    source: str = "Satellite",
    overwrite: bool = False,
    quiet: bool = False,
    num_workers: int = 8,
    progress_interval: int = 25,
    max_tiles: int | None = None,
    print_plan: bool = True,
    mask_geometry: Any = None,
    mask_geometry_crs: str = "EPSG:4326",
    mask_fill_value: int = 0,
    return_image: bool = False,
) -> Image.Image | None:
    """Download tiles into a GeoTIFF and optionally black out pixels outside a mask."""

    import rasterio as rio
    from osgeo import gdal, osr

    _check_bbox(bbox)
    if zoom is None and resolution is None:
        raise ValueError("Either zoom or resolution must be provided")
    if zoom is not None and resolution is not None:
        raise ValueError("Only one of zoom or resolution can be provided")

    output = _ensure_output_path(output)
    if os.path.exists(output) and not overwrite:
        raise FileExistsError(f"{output} already exists. Use overwrite=True to replace it.")

    source_url = _resolve_tile_source(source)
    request_stats = estimate_tms_request(bbox=bbox, zoom=zoom, resolution=resolution)
    zoom = request_stats["zoom"]

    if max_tiles is not None and request_stats["tile_count"] > max_tiles:
        raise ValueError(
            f"Request would download {request_stats['tile_count']} tiles at zoom {zoom}, "
            f"which exceeds max_tiles={max_tiles}."
        )

    if print_plan and not quiet:
        print(
            "Planned request:",
            {
                "zoom": zoom,
                "tile_count": request_stats["tile_count"],
                "xtiles": request_stats["xtiles"],
                "ytiles": request_stats["ytiles"],
                "pixel_width": request_stats["pixel_width"],
                "pixel_height": request_stats["pixel_height"],
            },
        )

    west, south, east, north = bbox
    session = _get_http_session()
    Image.MAX_IMAGE_PIXELS = None
    gdal.UseExceptions()

    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)
    projection_wkt = web_mercator.ExportToWkt()

    def get_tile(url: str) -> bytes | None:
        retries = 3
        while retries > 0:
            try:
                response = session.get(url, timeout=60)
                break
            except Exception:
                retries -= 1
                if retries == 0:
                    raise
        if response.status_code == 404 or not response.content:
            return None
        response.raise_for_status()
        return response.content

    def paste_tile(
        canvas: Image.Image | None,
        base_size: list[int],
        tile_bytes: bytes | None,
        corner_xy: tuple[int, int],
        tile_bbox: tuple[int, int, int, int],
    ) -> Image.Image | None:
        if tile_bytes is None:
            return canvas

        tile_image = Image.open(io.BytesIO(tile_bytes))
        mode = "RGB" if tile_image.mode == "RGB" else "RGBA"
        size = tile_image.size

        if canvas is None:
            base_size[0], base_size[1] = size
            canvas = Image.new(
                mode,
                (size[0] * (tile_bbox[2] - tile_bbox[0]), size[1] * (tile_bbox[3] - tile_bbox[1])),
            )

        dx = abs(corner_xy[0] - tile_bbox[0])
        dy = abs(corner_xy[1] - tile_bbox[1])
        location = (size[0] * dx, size[1] * dy)

        if mode == "RGB":
            canvas.paste(tile_image, location)
        else:
            if tile_image.mode != mode:
                tile_image = tile_image.convert(mode)
            canvas.paste(tile_image, location)

        tile_image.close()
        return canvas

    def finish_picture(
        canvas: Image.Image,
        base_size: list[int],
        tile_bbox: tuple[int, int, int, int],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Image.Image:
        xfrac = x0 - tile_bbox[0]
        yfrac = y0 - tile_bbox[1]
        left = round(base_size[0] * xfrac)
        upper = round(base_size[1] * yfrac)
        width = round(base_size[0] * (x1 - x0))
        height = round(base_size[1] * (y1 - y0))
        image = canvas.crop((left, upper, left + width, upper + height))
        if image.mode == "RGBA" and image.getextrema()[3] == (255, 255):
            image = image.convert("RGB")
        canvas.close()
        return image

    x0, y0 = _deg2num(south, west, zoom)
    x1, y1 = _deg2num(north, east, zoom)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    corners = tuple(
        (x, y)
        for x in range(math.floor(x0), math.ceil(x1))
        for y in range(math.floor(y0), math.ceil(y1))
    )
    total_tiles = len(corners)
    tile_bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))

    futures: list[concurrent.futures.Future] = []
    base_size = [256, 256]
    canvas = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        for x, y in corners:
            futures.append(executor.submit(get_tile, source_url.format(z=zoom, x=x, y=y)))

        for index, (future, corner_xy) in enumerate(zip(futures, corners), start=1):
            canvas = paste_tile(canvas, base_size, future.result(), corner_xy, tile_bbox)
            if not quiet and (
                index == 1 or index == total_tiles or index % max(1, progress_interval) == 0
            ):
                print(f"Downloaded tile {index}/{total_tiles}")

    if canvas is None:
        raise RuntimeError("No tiles were downloaded for the requested area.")

    if not quiet:
        print("Saving GeoTIFF. Please wait...")

    image = finish_picture(canvas, base_size, tile_bbox, x0, y0, x1, y1)
    band_count = len(image.getbands())
    driver = gdal.GetDriverByName("GTiff")

    dataset = driver.Create(
        output,
        image.size[0],
        image.size[1],
        band_count,
        gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=9", "TILED=YES"],
    )

    xp0, yp0 = _from4326_to3857(south, west)
    xp1, yp1 = _from4326_to3857(north, east)
    pixel_width = abs(xp1 - xp0) / image.size[0]
    pixel_height = abs(yp1 - yp0) / image.size[1]
    dataset.SetGeoTransform((min(xp0, xp1), pixel_width, 0, max(yp0, yp1), 0, -pixel_height))
    dataset.SetProjection(projection_wkt)

    for band_index in range(band_count):
        array = np.array(image.getdata(band_index), dtype="u8").reshape((image.size[1], image.size[0]))
        dataset.GetRasterBand(band_index + 1).WriteArray(array)

    dataset.FlushCache()
    dataset = None

    if mask_geometry is not None:
        apply_mask_to_geotiff(
            output,
            mask_geometry=mask_geometry,
            geometry_crs=mask_geometry_crs,
            fill_value=mask_fill_value,
        )

    if not quiet:
        with rio.open(output) as src:
            print(
                "Saved GeoTIFF:",
                {
                    "path": output,
                    "width": src.width,
                    "height": src.height,
                    "crs": src.crs.to_string() if src.crs is not None else None,
                },
            )

    return image if return_image else None
