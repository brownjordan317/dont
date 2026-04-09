"""Sweeper application package."""


def export_drone_geojsons_from_image(*args, **kwargs):
    from src.export_paths_geojson import export_drone_geojsons_from_image as _export

    return _export(*args, **kwargs)


def create_incremental_session_from_image(*args, **kwargs):
    from src.session_api import create_incremental_session_from_image as _create

    return _create(*args, **kwargs)


__all__ = [
    "export_drone_geojsons_from_image",
    "create_incremental_session_from_image",
]
