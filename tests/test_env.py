import numpy as np
import pytest

from flybrain_worldle.game import WorldleEnv, load_countries


@pytest.fixture(scope="module")
def countries():
    return load_countries()


@pytest.fixture
def env():
    return WorldleEnv(image_size=32, seed=0)


def by_name(countries, name):
    return next(c for c in countries if c.name == name)


def test_country_list_is_un_members_plus_vatican_and_palestine(countries):
    from shapely.geometry import Point

    names = {c.name for c in countries}
    assert len(countries) == 195
    assert {"France", "Norway", "Brazil", "Japan", "Kenya", "Singapore", "Vatican", "Palestine", "Malta", "Tuvalu"} <= names
    assert not ({"Antarctica", "N. Cyprus", "Taiwan", "Kosovo", "W. Sahara", "Somaliland"} & names)
    assert len({c.code for c in countries}) == len(countries)
    assert all(-90 <= c.lat <= 90 and -180 <= c.lon <= 180 for c in countries)
    assert by_name(countries, "Cyprus").geometry.contains(Point(33.9, 35.3))  # the north is part of Cyprus


def test_reset_returns_normalized_silhouette(env):
    img = env.reset()
    assert img.shape == (32, 32)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() == 1.0
    assert env.target is not None
    assert env.history == []
    assert not env.done


def test_step_before_reset_raises(env, countries):
    with pytest.raises(RuntimeError):
        env.step(countries[0])


def test_correct_guess_ends_episode_with_bonus(env, countries):
    target = by_name(countries, "Brazil")
    env.reset(target)
    fb, reward, done = env.step(target)
    assert fb.correct and done
    assert fb.distance_km == 0.0 and fb.proximity == 1.0
    assert reward > 1.0
    with pytest.raises(RuntimeError):
        env.step(target)


def test_wrong_guess_feedback_points_toward_target(env, countries):
    env.reset(by_name(countries, "Norway"))
    fb, reward, done = env.step(by_name(countries, "South Africa"))
    assert not fb.correct and not done
    assert fb.direction == "N"
    assert 8000 < fb.distance_km < 11000
    assert 0.4 < fb.proximity < 0.6
    assert -0.6 < reward < -0.4
    assert len(env.history) == 1


def test_episode_ends_after_max_guesses(countries):
    env = WorldleEnv(image_size=32, max_guesses=3, seed=1)
    env.reset(by_name(countries, "Japan"))
    wrong = by_name(countries, "Chile")
    for i in range(3):
        _, _, done = env.step(wrong)
        assert done == (i == 2)
    assert env.done


def test_nearest_country_lookup(env, countries):
    assert env.nearest_country(48.8, 2.3).name == "France"
    assert env.nearest_country(-34.0, 151.0).name == "Australia"


def test_silhouettes_are_cached_and_distinct(env, countries):
    a = env.silhouette(by_name(countries, "Italy"))
    b = env.silhouette(by_name(countries, "Italy"))
    c = env.silhouette(by_name(countries, "Chile"))
    assert a is b
    assert not np.array_equal(a, c)
