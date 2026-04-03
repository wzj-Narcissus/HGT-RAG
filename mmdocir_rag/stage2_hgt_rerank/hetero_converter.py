"""Convert MMDocIR Stage 2 graphs into PyG HeteroData.

The conversion logic is shared with the MultimodalQA pipeline via
``src.data.hetero_utils.graph_dict_to_heterodata``; this module binds the
MMDocIR-specific schema (node types, edge types, scored node types).

Node types:   query, page, layout
Scored types: page, layout
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from src.data.hetero_utils import graph_dict_to_heterodata
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MMDocIR graph schema constants
# ---------------------------------------------------------------------------

# Node types for which the model outputs relevance scores
SCORED_NODE_TYPES: List[str] = ["page", "layout"]

# All edge definitions: (src_type, rel_name, dst_type)
EDGE_SCHEMA: List[Tuple[str, str, str]] = [
    # Query ↔ page (bidirectional)
    ("query",  "query_to_page",        "page"),
    ("page",   "page_to_query",        "query"),
    # Query ↔ layout (bidirectional)
    ("query",  "query_to_layout",      "layout"),
    ("layout", "layout_to_query",      "query"),
    # Page ↔ layout membership (bidirectional)
    ("page",   "page_contains_layout", "layout"),
    ("layout", "layout_in_page",       "page"),
    # Spatial adjacency between layouts on the same page
    ("layout", "layout_adjacent",      "layout"),
]

_NODE_TYPES: List[str] = ["query", "page", "layout"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_to_heterodata(
    graph: Dict,
    features: Dict[str, torch.Tensor],
    labels: Optional[Dict[str, int]] = None,
) -> HeteroData:
    """Convert an MMDocIR Stage 2 graph dict to a PyG HeteroData object.

    Args:
        graph:    Output of ``build_query_graph``.
        features: Mapping from node_id to embedding tensor.
        labels:   Optional mapping from node_id to 0/1 label.

    Returns:
        PyG HeteroData with ``.x``, ``.y``, ``.node_ids`` for each node type
        and ``.edge_index`` for each edge type in ``EDGE_SCHEMA``.
    """
    return graph_dict_to_heterodata(
        graph=graph,
        features=features,
        edge_schema=EDGE_SCHEMA,
        node_types=_NODE_TYPES,
        scored_node_types=SCORED_NODE_TYPES,
        labels=labels,
    )


def build_labels(graph: Dict) -> Dict[str, Dict[str, int]]:
    """Build binary labels from the ``is_positive`` flags set during graph construction.

    Args:
        graph: Graph dict as returned by ``build_query_graph``.

    Returns:
        Dict with two keys:

        - ``"labels"``: flat mapping node_id → 0/1 (used by
          ``convert_to_heterodata``).
        - ``"breakdown"``: per-node-type mapping for diagnostics.
    """
    labels: Dict[str, int] = {}
    breakdown: Dict[str, Dict[str, int]] = {"page": {}, "layout": {}}

    for ntype in ("page", "layout"):
        for node in graph.get("nodes", {}).get(ntype, []):
            nid = node["node_id"]
            label = 1 if node.get("is_positive", False) else 0
            labels[nid] = label
            breakdown[ntype][nid] = label

    return {"labels": labels, "breakdown": breakdown}
