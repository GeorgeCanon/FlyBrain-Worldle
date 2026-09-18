import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from flybrain_worldle.game.geo import (
    bearing_deg,
    compass_point,
    destination_point,
    haversine_km,
    render_silhouette,
)

LONDON = (51.5074, -0.1278)
PARIS = (48.8566, 2.3522)
NEW_YORK = (40.7128, -74.0060)
SYDNEY = (-33.8688, 151.2093)


def test_haversine_known_distances():
    assert haversine_km(*LONDON, *PARIS) == pytest.approx(344, abs=5)
    assert haversine_km(*LONDON, *NEW_YORK) == pytest.approx(5570, abs=30)
    assert haversine_km(*LONDON, *SYDNEY) == pytest.approx(16994, abs=50)


def test_haversine_is_symmetric_and_zero_on_self():
    assert haversine_km(*LONDON, *LONDON) == 0.0
    assert haversine_km(*LONDON, *SYDNEY) == pytest.approx(haversine_km(*SYDNEY, *LONDON))


def test_bearing_cardinal_directions():
    assert bearing_deg(0, 0, 10, 0) == pytest.approx(0.0)
    assert bearing_deg(0, 0, 0, 10) == pytest.approx(90.0)
    assert bearing_deg(0, 0, -10, 0) == pytest.approx(180.0)
    assert bearing_deg(0, 0, 0, -10) == pytest.approx(270.0)


def test_bearing_london_to_new_york_is_roughly_west_northwest():
    assert bearing_deg(*LONDON, *NEW_YORK) == pytest.approx(288, abs=2)


def test_compass_points():
    assert compass_point(0) == "N"
    assert compass_point(22) == "N"
    assert compass_point(23) == "NE"
    assert compass_point(45) == "NE"
    assert compass_point(359) == "N"
    assert compass_point(180) == "S"
    assert compass_point(270) == "W"


def test_destination_point_round_trips_with_bearing_and_distance():
    dist = haversine_km(*LONDON, *SYDNEY)
    brg = bearing_deg(*LONDON, *SYDNEY)
    lat, lon = destination_point(*LONDON, brg, dist)
    assert lat == pytest.approx(SYDNEY[0], abs=0.01)
    assert lon == pytest.approx(SYDNEY[1], abs=0.01)


def test_destination_point_wraps_longitude():
    _, lon = destination_point(0, 179, 90, 500)
    assert -180 <= lon <= 180
    assert lon < 0


def test_render_silhouette_square_fills_and_is_normalized():
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    img = render_silhouette(square, size=32)
    assert img.shape == (32, 32)
    assert img.dtype == np.float32
    assert img.min() == 0.0 and img.max() == 1.0
    assert img[16, 16] == 1.0
    assert img[0, 0] == 0.0
    assert 0.7 < img.mean() < 0.9


def test_render_silhouette_hole_is_empty():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inner = [(4, 4), (6, 4), (6, 6), (4, 6)]
    img = render_silhouette(Polygon(outer, [inner]), size=64)
    assert img[32, 32] == 0.0
    assert img[8, 32] == 1.0


def test_render_silhouette_supersample_is_antialiased_and_same_coverage():
    tri = Polygon([(0, 0), (10, 0), (5, 9)])
    hard = render_silhouette(tri, size=64)
    soft = render_silhouette(tri, size=64, supersample=4)
    assert soft.shape == (64, 64)
    assert ((soft > 0.05) & (soft < 0.95)).sum() > 20
    assert abs(soft.mean() - hard.mean()) < 0.03


def test_render_silhouette_is_scale_invariant():
    small = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    large = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    assert math.isclose(render_silhouette(small, 64).mean(), render_silhouette(large, 64).mean(), abs_tol=0.03)
