"""Coordinate transformation utilities."""
from typing import Tuple
from pyproj import Transformer
import math


def wgs84_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
    """Convert WGS84 coordinates to UTM.

    Returns easting, northing, zone number, and hemisphere.
    """
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    epsg = 32600 + zone if hemisphere == 'north' else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing, zone, hemisphere


def local_xy(lat: float, lon: float, center_lat: float, center_lon: float) -> Tuple[float, float]:
    """Convert lat/lon to local Cartesian coordinates relative to a center."""
    e, n, zone, hemi = wgs84_to_utm(lat, lon)
    ce, cn, _, _ = wgs84_to_utm(center_lat, center_lon)
    return e - ce, n - cn


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great circle distance between two lat/lon pairs in meters."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

