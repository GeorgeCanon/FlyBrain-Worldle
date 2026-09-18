"""One-time export of neuropil meshes and one representative skeleton per CX cell type to data/brain/brain.json.gz.

Requires NEUPRINT_TOKEN (env or .env). Takes ~5 minutes: one skeleton fetch per cell type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain_worldle.connectome.anatomy import BRAIN_PATH, fetch_brain, load_types, save_brain


def main() -> None:
    brain = fetch_brain(load_types())
    save_brain(brain)
    print(f"{len(brain['skeletons'])}/{len(brain['types'])} skeletons, {len(brain['meshes'])} meshes")
    print(f"wrote {BRAIN_PATH} ({BRAIN_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
