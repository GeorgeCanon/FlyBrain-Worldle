import random
from dataclasses import dataclass

import numpy as np

from .countries import Country, load_countries
from .geo import MAX_DISTANCE_KM, bearing_deg, compass_point, haversine_km, render_silhouette

MAX_GUESSES = 6


@dataclass(frozen=True)
class Feedback:
    guess: Country
    distance_km: float
    bearing: float
    direction: str
    proximity: float
    correct: bool


class WorldleEnv:
    """Local re-implementation of Worldle's rules.

    Each episode picks a hidden target country and shows its silhouette. A guess is answered with the
    great-circle distance, the initial bearing from the guessed country to the target, and a proximity
    score in [0, 1]. The episode ends on a correct guess or after MAX_GUESSES.
    """

    def __init__(self, image_size: int = 64, max_guesses: int = MAX_GUESSES, seed: int | None = None):
        self.countries = load_countries()
        self.by_code = {c.code: c for c in self.countries}
        self.image_size = image_size
        self.max_guesses = max_guesses
        self.rng = random.Random(seed)
        self._silhouettes: dict[str, np.ndarray] = {}
        self.target: Country | None = None
        self.history: list[Feedback] = []

    def silhouette(self, country: Country) -> np.ndarray:
        if country.code not in self._silhouettes:
            self._silhouettes[country.code] = render_silhouette(country.geometry, self.image_size)
        return self._silhouettes[country.code]

    def reset(self, target: Country | None = None) -> np.ndarray:
        self.target = target or self.rng.choice(self.countries)
        self.history = []
        return self.silhouette(self.target)

    @property
    def done(self) -> bool:
        return bool(self.history) and (self.history[-1].correct or len(self.history) >= self.max_guesses)

    def step(self, guess: Country) -> tuple[Feedback, float, bool]:
        if self.target is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError("episode is over; call reset()")

        correct = guess.code == self.target.code
        distance = 0.0 if correct else haversine_km(guess.lat, guess.lon, self.target.lat, self.target.lon)
        bearing = 0.0 if correct else bearing_deg(guess.lat, guess.lon, self.target.lat, self.target.lon)
        fb = Feedback(
            guess=guess,
            distance_km=distance,
            bearing=bearing,
            direction="✓" if correct else compass_point(bearing),
            proximity=max(0.0, 1.0 - distance / MAX_DISTANCE_KM),
            correct=correct,
        )
        self.history.append(fb)

        reward = fb.proximity - 1.0
        if correct:
            reward += 1.0 + (self.max_guesses - len(self.history)) / self.max_guesses
        return fb, reward, self.done

    def nearest_country(self, lat: float, lon: float) -> Country:
        return min(self.countries, key=lambda c: haversine_km(lat, lon, c.lat, c.lon))
