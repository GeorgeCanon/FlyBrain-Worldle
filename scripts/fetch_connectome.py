"""One-time export of MaleCNS central-complex cell-type connectivity to data/cx_types.csv and data/cx_edges.csv.

Requires a free neuPrint account: export NEUPRINT_TOKEN from https://neuprint.janelia.org/account
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain_worldle.connectome.central_complex import EDGES_PATH, TYPES_PATH, fetch_cx_graph


def main() -> None:
    graph = fetch_cx_graph()
    graph.save()
    n_edges = int((graph.weights != 0).sum())
    signed = int((graph.signs != 0).sum())
    print(f"{graph.n_types} cell types, {int(graph.n_cells.sum())} cells, {n_edges} type->type edges, {signed} types with NT sign")
    print(f"wrote {TYPES_PATH} and {EDGES_PATH}")


if __name__ == "__main__":
    main()
