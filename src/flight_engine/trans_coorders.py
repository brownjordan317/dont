from pyproj import CRS, Transformer
from typing import Tuple

# ============================================================================
# COORDINATE TRANSFORMATION
# ============================================================================

class CoordinateTransformer:
    def __init__(self, origin_lat: float, origin_lon: float):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self._setup_transformers()
    
    def _setup_transformers(self):
        """Setup transformers between geographic (lat, lon) and local (x, y) coordinates"""
        proj_local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.origin_lat} +lon_0={self.origin_lon} "
            f"+units=m +ellps=WGS84"
        )
        proj_geo = CRS.from_epsg(4326)
        self.to_xy = Transformer.from_crs(proj_geo, proj_local, always_xy=True)
        self.to_ll = Transformer.from_crs(proj_local, proj_geo, always_xy=True)
    
    def geo_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert geographic coordinates (lat, lon) to local coordinates (x, y) in meters"""
        return self.to_xy.transform(lon, lat)
    
    def local_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local coordinates (x, y) in meters to geographic coordinates (lat, lon)"""
        lon, lat = self.to_ll.transform(x, y)
        return lat, lon