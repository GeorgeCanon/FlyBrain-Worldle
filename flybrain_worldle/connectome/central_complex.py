"""Cell-type-level central-complex (CX) connectivity from the MaleCNS connectome, and an RNN built on it.

The CX (protocerebral bridge, ellipsoid body, fan-shaped body, noduli) is the fly's heading-integration
circuit, which is why it is the substrate for the guess/feedback loop. Data: MaleCNS v1.0 (Janelia FlyEM),
CC-BY, https://male-cns.janelia.org/
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TYPES_PATH = DATA_DIR / "cx_types.csv"
EDGES_PATH = DATA_DIR / "cx_edges.csv"

NEUPRINT_SERVER = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"
CX_ROI_PREFIXES = ("PB", "EB", "FB", "NO", "AB")

# Predicted neurotransmitter -> synaptic sign (Drosophila: ACh excitatory, GABA/Glu inhibitory).
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}


@dataclass
class CXGraph:
    types: list[str]
    n_cells: np.ndarray  # cells per type
    weights: np.ndarray  # [N, N] mean synapses per (pre-cell, post-cell) pair, signed by pre type's NT
    signs: np.ndarray  # [N] +1 / -1 / 0 per type

    @property
    def n_types(self) -> int:
        return len(self.types)

    def save(self, types_path: Path = TYPES_PATH, edges_path: Path = EDGES_PATH) -> None:
        pd.DataFrame({"type": self.types, "n_cells": self.n_cells, "sign": self.signs}).to_csv(types_path, index=False)
        pre, post = np.nonzero(self.weights)
        pd.DataFrame(
            {"pre": [self.types[i] for i in pre], "post": [self.types[j] for j in post], "weight": self.weights[pre, post]}
        ).to_csv(edges_path, index=False)

    @classmethod
    def load(cls, types_path: Path = TYPES_PATH, edges_path: Path = EDGES_PATH) -> "CXGraph":
        types_df = pd.read_csv(types_path)
        edges = pd.read_csv(edges_path)
        types = types_df["type"].tolist()
        index = {t: i for i, t in enumerate(types)}
        weights = np.zeros((len(types), len(types)), dtype=np.float32)
        weights[edges["pre"].map(index).to_numpy(), edges["post"].map(index).to_numpy()] = edges["weight"].to_numpy()
        return cls(types, types_df["n_cells"].to_numpy(), weights, types_df["sign"].to_numpy(dtype=np.float32))

    def shuffled(self, seed: int = 0) -> "CXGraph":
        """Control graph: same edge weights and count, randomly rewired. Tests whether the real wiring matters."""
        rng = np.random.default_rng(seed)
        n = self.n_types
        values = self.weights[self.weights != 0]
        flat = np.zeros(n * n, dtype=np.float32)
        flat[rng.choice(n * n, size=len(values), replace=False)] = rng.permutation(values)
        return CXGraph(self.types, self.n_cells, flat.reshape(n, n), self.signs)

    @classmethod
    def stub(cls, n_types: int = 32, density: float = 0.2, seed: int = 0) -> "CXGraph":
        """Random graph with CX-like statistics, for tests and offline development."""
        rng = np.random.default_rng(seed)
        mask = rng.random((n_types, n_types)) < density
        signs = np.where(rng.random(n_types) < 0.6, 1.0, -1.0).astype(np.float32)
        weights = (mask * rng.lognormal(1.0, 0.8, (n_types, n_types)) * signs[:, None]).astype(np.float32)
        return cls([f"stub{i}" for i in range(n_types)], rng.integers(2, 40, n_types), weights, signs)


def _token_from_dotenv(path: Path = DATA_DIR.parent / ".env") -> str | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "NEUPRINT_TOKEN":
            return value.strip().strip("\"'")
    return None


def fetch_cx_graph(token: str | None = None) -> CXGraph:
    """Pull CX neurons and their connectivity from neuPrint and aggregate to cell types."""
    from neuprint import Client, NeuronCriteria, fetch_adjacencies, fetch_all_rois, fetch_neurons

    token = token or os.environ.get("NEUPRINT_TOKEN") or _token_from_dotenv()
    if not token:
        raise RuntimeError("set NEUPRINT_TOKEN in the environment or in .env (create one at https://neuprint.janelia.org/account)")
    client = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=token)

    rois = [r for r in fetch_all_rois(client=client) if r.split("(")[0].rstrip("0123456789") in CX_ROI_PREFIXES]
    neurons, _ = fetch_neurons(NeuronCriteria(rois=rois, roi_req="any", client=client), client=client)
    neurons = neurons.dropna(subset=["type"])
    neurons = neurons[neurons["type"].str.len() > 0]

    body_type = dict(zip(neurons["bodyId"], neurons["type"]))
    types = sorted(set(body_type.values()))
    index = {t: i for i, t in enumerate(types)}
    n_cells = neurons.groupby("type").size().reindex(types).to_numpy()

    signs = np.zeros(len(types), dtype=np.float32)
    if "predictedNt" in neurons.columns:
        nt_per_type = neurons.groupby("type")["predictedNt"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
        signs = np.array([NT_SIGN.get(str(nt_per_type.get(t)).lower(), 0.0) for t in types], dtype=np.float32)

    _, conn = fetch_adjacencies(neurons["bodyId"].tolist(), neurons["bodyId"].tolist(), client=client)
    conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    conn["pre"] = conn["bodyId_pre"].map(body_type)
    conn["post"] = conn["bodyId_post"].map(body_type)
    type_conn = conn.groupby(["pre", "post"])["weight"].sum()

    weights = np.zeros((len(types), len(types)), dtype=np.float32)
    for (pre, post), w in type_conn.items():
        i, j = index[pre], index[post]
        weights[i, j] = w / (n_cells[i] * n_cells[j])
    weights *= np.where(signs == 0, 1.0, signs)[:, None]
    return CXGraph(types, n_cells, weights, signs)


class CentralComplexRNN(nn.Module):
    """Leaky rate RNN whose recurrent topology and synaptic signs are fixed by the connectome.

    Only per-synapse gains (on existing edges), per-type biases and time constants are trainable; the
    graph itself is not. Inputs enter through a trainable projection.
    """

    def __init__(self, graph: CXGraph, input_dim: int, dt: float = 0.2):
        super().__init__()
        w0 = torch.as_tensor(graph.weights)
        strength = torch.log1p(w0.abs())
        strength = strength / (strength.sum(dim=0, keepdim=True).clamp_min(1e-6))
        self.register_buffer("base", strength * torch.sign(w0))
        self.register_buffer("mask", (w0 != 0).float())
        self.gain = nn.Parameter(torch.ones_like(w0))
        self.bias = nn.Parameter(torch.zeros(graph.n_types))
        self.log_tau = nn.Parameter(torch.zeros(graph.n_types))
        self.w_in = nn.Linear(input_dim, graph.n_types)
        self.dt = dt
        self.n_types = graph.n_types

    @property
    def recurrent_weight(self) -> torch.Tensor:
        return self.base * self.gain.abs() * self.mask

    def forward(self, x: torch.Tensor, h: torch.Tensor, steps: int = 5) -> torch.Tensor:
        alpha = (self.dt / self.log_tau.exp().clamp_min(self.dt)).unsqueeze(0)
        drive = self.w_in(x) + self.bias
        for _ in range(steps):
            h = (1 - alpha) * h + alpha * torch.tanh(h @ self.recurrent_weight + drive)
        return h

    def init_state(self, batch: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch, self.n_types, device=device or self.bias.device)
