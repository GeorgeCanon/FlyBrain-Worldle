import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from shapely.geometry import MultiPolygon, box, shape
from shapely.geometry.base import BaseGeometry

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GEOJSON_PATH = DATA_DIR / "ne_110m_admin_0_countries.geojson"

PLAYABLE_TYPES = {"Sovereign country", "Country", "Sovereignty"}
EXCLUDED_NAMES = {"Antarctica"}
MAX_TERRITORY_GAP_DEG = 12.0


def main_territory(geometry: BaseGeometry) -> BaseGeometry:
    """Drop overseas territories (e.g. French Guiana) so the silhouette is the country people recognize."""
    if not isinstance(geometry, MultiPolygon):
        return geometry
    largest = max(geometry.geoms, key=lambda p: p.area)
    keep = [p for p in geometry.geoms if box(*largest.bounds).distance(box(*p.bounds)) <= MAX_TERRITORY_GAP_DEG]
    return keep[0] if len(keep) == 1 else MultiPolygon(keep)


@dataclass(frozen=True)
class Country:
    name: str
    code: str
    continent: str
    lat: float
    lon: float
    geometry: BaseGeometry

    def __repr__(self) -> str:
        return f"Country({self.name!r}, {self.code}, {self.lat:.1f}, {self.lon:.1f})"


@lru_cache(maxsize=1)
def load_countries(path: Path = GEOJSON_PATH) -> tuple[Country, ...]:
    with open(path, encoding="utf-8") as f:
        collection = json.load(f)

    countries = []
    for feature in collection["features"]:
        props = feature["properties"]
        if props["TYPE"] not in PLAYABLE_TYPES or props["NAME"] in EXCLUDED_NAMES:
            continue
        code = props["ISO_A3"] if props["ISO_A3"] != "-99" else props["ADM0_A3"]
        countries.append(
            Country(
                name=props["NAME"],
                code=code,
                continent=props["CONTINENT"],
                lat=float(props["LABEL_Y"]),
                lon=float(props["LABEL_X"]),
                geometry=main_territory(shape(feature["geometry"])),
            )
        )
    countries.sort(key=lambda c: c.name)
    return tuple(countries)
