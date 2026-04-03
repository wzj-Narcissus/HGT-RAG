"""Convert a MultimodalQA graph dict + features + labels into a PyG HeteroData object.

Core function: ``convert_to_heterodata(graph, features, labels)``

The conversion logic is shared with the MMDocIR pipeline via
``src.data.hetero_utils.graph_dict_to_heterodata``; this module binds the
MultimodalQA-specific schema (node types, edge types, scored node types).
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from src.data.hetero_utils import graph_dict_to_heterodata
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MultimodalQA graph schema constants
# ---------------------------------------------------------------------------

# Node types that carry binary relevance labels
SCORED_NODE_TYPES: List[str] = ["textblock", "cell", "image", "caption", "table"]

# All edge definitions: (src_type, rel_name, dst_type)
# Kept here as the single source of truth for the MMQA schema.
# The model-side EDGE_TYPES in hgt_model.py is derived from this list.
EDGE_SCHEMA: List[Tuple[str, str, str]] = [
    # Query ↔ evidence nodes (bidirectional)
    ("query",     "query_to_text",        "textblock"),
    ("query",     "query_to_table",       "table"),
    ("query",     "query_to_cell",        "cell"),
    ("query",     "query_to_image",       "image"),
    ("textblock", "text_to_query",        "query"),
    ("table",     "table_to_query",       "query"),
    ("cell",      "cell_to_query",        "query"),
    ("image",     "image_to_query",       "query"),
    # Cross-modal co-reference edges (bidirectional)
    ("textblock", "text_refers_table",    "table"),
    ("table",     "table_refers_text",    "textblock"),
    ("textblock", "text_refers_image",    "image"),
    ("image",     "image_refers_text",    "textblock"),
    # Table ↔ cell membership (bidirectional)
    ("table",     "table_contains_cell",  "cell"),
    ("cell",      "cell_in_table",        "table"),
    # Within-table cell adjacency (row and column)
    ("cell",      "cell_to_cell_row",     "cell"),
    ("cell",      "cell_to_cell_col",     "cell"),
    # Image ↔ caption (bidirectional)
    ("image",     "image_has_caption",    "caption"),
    ("caption",   "caption_to_image",     "image"),
]

_NODE_TYPES: List[str] = ["query", "textblock", "table", "cell", "image", "caption"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_to_heterodata(
    graph: Dict,
    features: Dict[str, torch.Tensor],
    labels: Optional[Dict[str, int]] = None,
    graph_cfg: Optional[Dict] = None,  # kept for backward compatibility, unused
) -> HeteroData:
    """Convert a MultimodalQA graph dict to a PyG HeteroData object.

    Args:
        graph:     Output of ``build_question_graph``.  Must contain
                   ``{"nodes": {ntype: [node_dict, ...]},
                   "edges": {rel_name: [(src_id, dst_id), ...]}}``.
        features:  Output of ``build_node_features`` — mapping from node_id
                   to embedding tensor.
        labels:    Output of ``build_labels["labels"]`` — mapping from node_id
                   to 0/1 integer label.  Pass ``None`` for inference.
        graph_cfg: Ignored (kept for backward compatibility).

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


def build_similarity_augmented_edges(
    graph: Dict,
    features: Dict[str, torch.Tensor],
    graph_cfg: Optional[Dict] = None,
) -> Dict[str, list]:
    """Return the edge lists from the graph dict (no augmentation).

    This stub exists for backward compatibility; cosine-similarity-based
    edge augmentation was removed in favour of the bidirectional edge schema.
    """
    return {
        rel: [list(edge) for edge in edge_list]
        for rel, edge_list in graph.get("edges", {}).items()
    }
