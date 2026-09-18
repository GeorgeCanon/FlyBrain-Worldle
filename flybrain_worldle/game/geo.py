import math

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

EARTH_RADIUS_KM = 6371.0
MAX_DISTANCE_KM = 20_000.0

COMPASS_POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, clockwise from north in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return math.degrees(math.atan2(x, y)) % 360.0


def compass_point(bearing: float) -> str:
    return COMPASS_POINTS[int(((bearing + 22.5) % 360) // 45)]


def destination_point(lat: float, lon: float, bearing: float, distance_km: float) -> tuple[float, float]:
    """Point reached by travelling `distance_km` along `bearing` from (lat, lon)."""
    d = distance_km / EARTH_RADIUS_KM
    b = math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    raise TypeError(f"unsupported geometry {geometry.geom_type}")


def render_silhouette(geometry: BaseGeometry, size: int = 64, margin: float = 0.06) -> np.ndarray:
    """Rasterize a country outline into a square float32 array in [0, 1], centered and scale-normalized.

    Longitude is scaled by cos(mean latitude) so the shape's proportions match what a viewer sees on a map.
    """
    minx, miny, maxx, maxy = geometry.bounds
    lat_scale = math.cos(math.radians((miny + maxy) / 2))
    width = (maxx - minx) * lat_scale
    height = maxy - miny
    extent = max(width, height, 1e-9)
    usable = size * (1 - 2 * margin)
    scale = usable / extent

    x_off = (size - width * scale) / 2
    y_off = (size - height * scale) / 2

    def to_px(x: float, y: float) -> tuple[float, float]:
        return ((x - minx) * lat_scale * scale + x_off, (maxy - y) * scale + y_off)

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for poly in _polygons(geometry):
        draw.polygon([to_px(*pt) for pt in poly.exterior.coords], fill=255)
        for hole in poly.interiors:
            draw.polygon([to_px(*pt) for pt in hole.coords], fill=0)
    return np.asarray(img, dtype=np.float32) / 255.0
