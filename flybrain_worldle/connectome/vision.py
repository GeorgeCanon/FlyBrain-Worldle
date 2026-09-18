"""Visual front end: a frozen, pretrained flyvis optic-lobe network turned into a per-country feature extractor.

flyvis (Lappalainen et al., Nature 2024) models the fly optic lobe (R1-R8 -> lamina -> medulla -> T4/T5). It is
motion-sensitive, so a silhouette is shown as a short movie (onset, then drifting) and the responses of the
motion-output cell types are averaged over time and kept retinotopic. The network is deterministic and frozen,
so features are computed once per country and cached.
"""

from pathlib import Path

import numpy as np
import torch

from ..game.countries import Country
from ..game.geo import render_silhouette

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _patch_datamate_for_windows() -> None:
    """datamate's _write_h5 unlinks a file while its h5py handle is still open, which Windows refuses."""
    import h5py
    from datamate import directory, io

    def _write_h5(path: Path, val) -> None:
        val = np.asarray(val)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            path.unlink()
        with h5py.File(path, libver="latest", mode="w") as f:
            f["data"] = val
            f.swmr_mode = True

    io._write_h5 = _write_h5
    directory._write_h5 = _write_h5

OUTPUT_CELL_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
DT = 1 / 100
IMAGE_SIZE = 128
BACKGROUND = 0.2


def silhouette_movie(image: np.ndarray, hold: int = 10, drift: int = 12) -> np.ndarray:
    """(frames, H, W) luminance movie: hold the shape, drift it right, then drift it down."""
    frame = BACKGROUND + (1.0 - BACKGROUND) * image
    frames = [frame] * hold
    for i in range(1, drift + 1):
        frames.append(np.roll(frame, i, axis=1))
    for i in range(1, drift + 1):
        frames.append(np.roll(np.roll(frame, drift, axis=1), i, axis=0))
    return np.stack(frames).astype(np.float32)


class PixelVision:
    """Control front end with no fly biology: a downsampled silhouette as the feature vector."""

    name = "pixels"

    def __init__(self, size: int = 16):
        self.size = size
        self.dim = size * size

    def features(self, country: Country) -> np.ndarray:
        return render_silhouette(country.geometry, self.size).reshape(-1)


class FlyVision:
    name = "flyvis"

    def __init__(self, model: str = "flow/0000/000", cell_types: tuple[str, ...] = OUTPUT_CELL_TYPES):
        _patch_datamate_for_windows()
        import flyvis
        from flyvis.datasets.rendering import BoxEye

        self.model = model
        self.cell_types = cell_types
        self.receptors = BoxEye()
        self.network = flyvis.NetworkView(flyvis.results_dir / model).init_network()
        self.network.eval()
        self.dim = len(cell_types) * self.receptors.hexals

    @torch.no_grad()
    def features(self, country: Country) -> np.ndarray:
        from flyvis.utils.activity_utils import LayerActivity

        movie = silhouette_movie(render_silhouette(country.geometry, IMAGE_SIZE))
        movie = torch.as_tensor(movie)[None]  # (1, frames, H, W)
        rendered = self.receptors(movie)[0]  # (frames, 1, hexals)
        state = self.network.fade_in_state(1.0, DT, rendered[[0]])
        responses = self.network.simulate(rendered[None], DT, initial_state=state)  # (1, frames, cells)
        activity = LayerActivity(responses, self.network.connectome, keepref=True)
        return np.concatenate([activity[t][0].mean(0).cpu().numpy() for t in self.cell_types]).astype(np.float32)


CACHE_NAMES = {"flyvis": "features_flyvis_flow-0000-000_5768.npz", "pixels": "features_pixels_256.npz"}


def country_features(vision, countries: tuple[Country, ...], cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """[n_countries, dim] standardized features, cached on disk per vision front end.

    `vision` may be a front-end instance or just its name ("flyvis"/"pixels"); a name only works from the cache,
    which lets the demo run without the pretrained optic-lobe download.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    codes = [c.code for c in countries]
    if isinstance(vision, str):
        path = cache_dir / CACHE_NAMES[vision]
        if not path.exists():
            vision = FlyVision() if vision == "flyvis" else PixelVision()
    if not isinstance(vision, str):
        tag = getattr(vision, "model", "").replace("/", "-")
        path = cache_dir / f"features_{vision.name}{'_' + tag if tag else ''}_{vision.dim}.npz"
    if path.exists():
        cached = np.load(path)
        if list(cached["codes"]) == codes:
            return cached["features"]
    feats = np.stack([vision.features(c) for c in countries]).astype(np.float32)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    np.savez(path, codes=np.array(codes), features=feats)
    return feats
