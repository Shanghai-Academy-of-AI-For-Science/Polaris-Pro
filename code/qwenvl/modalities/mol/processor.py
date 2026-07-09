"""
Molecular processor: SMILES string -> PyG Data object for GNN encoding.

Wraps mol/graph_NN.build_graph() and mol/org_mol2d.from_smiles().
"""

import logging
from typing import Optional

import torch
from torch_geometric.data import Data

from .graph_NN import build_graph, ALLOWED_ELEMENTS

logger = logging.getLogger(__name__)


def smiles_to_graph(smiles: str, max_atoms: int = 500) -> Optional[Data]:
    """Convert a SMILES string to a PyG Data object.

    Returns None if the SMILES is invalid, contains unsupported elements,
    or exceeds max_atoms (to prevent OOM from large fully-connected graphs).

    The returned Data object has fields:
        x:              [num_atoms, 5]  (long) — node features
        edge_index:     [2, num_edges]  (long) — local bond graph
        edge_attr:      [num_edges, 3]  (long) — edge features
        edge_index_all: [2, num_fc_edges] (long) — full-connect graph
        smiles:         str — original SMILES string
    """
    data, err = build_graph(smiles)
    if err is not None:
        logger.warning(f"Invalid SMILES '{smiles[:80]}': {err}")
        return None
    if data.x.shape[0] > max_atoms:
        logger.warning(
            f"Molecule too large ({data.x.shape[0]} atoms > {max_atoms}): '{smiles[:40]}'"
        )
        return None
    return data
