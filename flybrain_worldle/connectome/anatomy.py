"""Brain geometry for the demo: neuropil meshes and one representative skeleton per central-complex cell type.

Everything comes from neuPrint (MaleCNS v1.0, CC-BY) in dataset voxel coordinates and is exported once to
data/brain/brain.json.gz so the web demo needs no API access.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .central_complex import DATA_DIR, NEUPRINT_DATASET, NEUPRINT_SERVER, TYPES_PATH, _token_from_dotenv

BRAIN_PATH = DATA_DIR / "brain" / "brain.json.gz"
MESH_ROIS = ("EB", "PB", "FB", "NO", "CentralBrain")
MAX_FACES = {"CentralBrain": 40_000}


def parse_obj(text: str) -> tuple[np.ndarray, np.ndarray]:
    """OBJ 'v'/'f' lines -> (vertices [V,3] float32, faces [F,3] int32, zero-based)."""
    verts, faces = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
            faces.append(idx)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def subsample_faces(vertices: np.ndarray, faces: np.ndarray, max_faces: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Keep a uniform random subset of faces and drop unused vertices; a rough but cheap decimation."""
    if len(faces) <= max_faces:
        return vertices, faces
    keep = np.sort(np.random.default_rng(seed).choice(len(faces), max_faces, replace=False))
    faces = faces[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(faces.shape).astype(np.int32)


def downsample_skeleton(nodes: np.ndarray, parents: np.ndarray, every: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Keep roots, branch points, leaves and every `every`-th node along unbranched runs.

    `parents[i]` is the index of node i's parent or -1. Returns compacted (nodes, parents).
    """
    n = len(nodes)
    n_children = np.bincount(parents[parents >= 0], minlength=n)
    keep = (parents < 0) | (n_children != 1)

    order = np.argsort(parents, kind="stable")
    run_len = np.zeros(n, dtype=np.int64)
    for i in order:
        p = parents[i]
        if p < 0 or keep[p]:
            run_len[i] = 1
        else:
            run_len[i] = run_len[p] + 1
        if not keep[i] and run_len[i] % every == 0:
            keep[i] = True

    new_index = np.full(n, -1, dtype=np.int64)
    new_index[keep] = np.arange(int(keep.sum()))
    new_parents = np.empty(int(keep.sum()), dtype=np.int32)
    for i in np.flatnonzero(keep):
        p = parents[i]
        while p >= 0 and not keep[p]:
            p = parents[p]
        new_parents[new_index[i]] = -1 if p < 0 else new_index[p]
    return nodes[keep].astype(np.float32), new_parents


def prune_twigs(nodes: np.ndarray, parents: np.ndarray, max_len: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Remove terminal branches of at most `max_len` nodes; these dominate node counts and add no visible shape."""
    n = len(nodes)
    n_children = np.bincount(parents[parents >= 0], minlength=n)
    drop = np.zeros(n, dtype=bool)
    for leaf in np.flatnonzero(n_children == 0):
        path, i = [], leaf
        while i >= 0 and n_children[i] <= 1 and len(path) <= max_len and parents[i] >= 0:
            path.append(i)
            i = parents[i]
        if len(path) <= max_len:
            drop[path] = True
    keep = ~drop
    new_index = np.full(n, -1, dtype=np.int64)
    new_index[keep] = np.arange(int(keep.sum()))
    new_parents = np.where(parents[keep] >= 0, new_index[np.maximum(parents[keep], 0)], -1).astype(np.int32)
    return nodes[keep], new_parents


def simplify_skeleton(nodes: np.ndarray, parents: np.ndarray, every: int = 6, prune_rounds: int = 2) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(prune_rounds):
        nodes, parents = prune_twigs(nodes, parents.astype(np.int64))
    return downsample_skeleton(nodes, parents.astype(np.int64), every=every)


def swc_to_arrays(swc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = swc["rowId"].to_numpy()
    index = {int(r): i for i, r in enumerate(ids)}
    parents = np.array([index.get(int(l), -1) for l in swc["link"].to_numpy()], dtype=np.int64)
    return swc[["x", "y", "z"]].to_numpy(dtype=np.float32), parents


def fetch_brain(types: list[str], token: str | None = None, log=print) -> dict:
    from neuprint import Client, NeuronCriteria, fetch_neurons, fetch_skeleton

    client = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=token or _token_from_dotenv())

    meshes = {}
    for roi in MESH_ROIS:
        raw = client.fetch_roi_mesh(roi)
        v, f = parse_obj(raw.decode() if isinstance(raw, bytes) else raw)
        v, f = subsample_faces(v, f, MAX_FACES.get(roi, len(f)))
        meshes[roi] = {"vertices": v.round(0).tolist(), "faces": f.tolist()}
        log(f"mesh {roi}: {len(v)} vertices, {len(f)} faces")

    skeletons = {}
    for i, t in enumerate(types):
        neurons, _ = fetch_neurons(NeuronCriteria(type=t, client=client), client=client)
        if neurons.empty:
            continue
        body = int(neurons.sort_values("size", ascending=False).iloc[0]["bodyId"])
        try:
            swc = fetch_skeleton(body, format="pandas", client=client)
        except Exception as e:  # a few bodies have no skeleton
            log(f"skeleton {t} ({body}): skipped ({str(e)[:60]})")
            continue
        nodes, parents = simplify_skeleton(*swc_to_arrays(swc))
        skeletons[t] = {"body": body, "nodes": nodes.round(0).tolist(), "parents": parents.tolist()}
        if i % 50 == 0:
            log(f"skeleton {i + 1}/{len(types)} {t}: {len(swc)} -> {len(nodes)} nodes")

    all_nodes = np.concatenate([np.asarray(s["nodes"]) for s in skeletons.values()])
    return {
        "dataset": NEUPRINT_DATASET,
        "types": types,
        "bbox": [all_nodes.min(0).tolist(), all_nodes.max(0).tolist()],
        "meshes": meshes,
        "skeletons": skeletons,
    }


def save_brain(brain: dict, path: Path = BRAIN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(brain, f, separators=(",", ":"))


def load_brain(path: Path = BRAIN_PATH) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def compact_brain(brain: dict, every: int = 6, prune_rounds: int = 2) -> dict:
    """Re-simplify already exported skeletons in place (idempotent enough to run on an existing export)."""
    for s in brain["skeletons"].values():
        nodes, parents = simplify_skeleton(np.asarray(s["nodes"], dtype=np.float32), np.asarray(s["parents"]), every, prune_rounds)
        s["nodes"], s["parents"] = nodes.round(0).tolist(), parents.tolist()
    return brain


def load_types(path: Path = TYPES_PATH) -> list[str]:
    return pd.read_csv(path)["type"].tolist()
