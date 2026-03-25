"""Convert a graph dict + features + labels into a PyG HeteroData object.

Core function: convert_to_heterodata(graph, features, labels)
"""
from typing import Dict, Optional

import torch
from torch_geometric.data import HeteroData

from src.utils.logging import get_logger

logger = get_logger(__name__)

# node types that carry scoring labels
SCORED_NODE_TYPES = ["textblock", "cell", "image", "caption", "table"]

# all edge definitions: (src_type, rel, dst_type)
EDGE_SCHEMA = [
    ("query",     "query_to_text",        "textblock"),
    ("query",     "query_to_table",       "table"),
    ("query",     "query_to_cell",        "cell"),
    ("query",     "query_to_image",       "image"),
    ("table",     "table_contains_cell",  "cell"),
    ("textblock", "text_refers_table",    "table"),
    ("textblock", "text_refers_image",    "image"),
    ("cell",      "cell_to_cell_row",     "cell"),
    ("cell",      "cell_to_cell_col",     "cell"),
    ("image",     "image_has_caption",    "caption"),
]


def _infer_dim(features: Dict[str, torch.Tensor]) -> int:
    for v in features.values():
        return v.shape[-1]
    return 384


def build_similarity_augmented_edges(
    graph: Dict,
    features: Dict[str, torch.Tensor],
    graph_cfg: Optional[Dict] = None,
) -> Dict[str, list]:
    return {
        rel: [list(edge) for edge in edge_list]
        for rel, edge_list in graph.get("edges", {}).items()
    }


def convert_to_heterodata(
    graph: Dict,
    features: Dict[str, torch.Tensor],
    labels: Optional[Dict[str, int]] = None,
    graph_cfg: Optional[Dict] = None,
) -> HeteroData:
    """Convert graph dict to PyG HeteroData.

    Args:
        graph:    output of build_question_graph
        features: output of build_node_features  (node_id -> tensor)
        labels:   output of build_labels['labels'] (node_id -> 0/1), optional

    Returns:
        PyG HeteroData object with .x, .y, .node_ids for each node type.
    """
    data = HeteroData()
    nodes = graph.get("nodes", {})
    edges = build_similarity_augmented_edges(graph, features, graph_cfg)

    # ---- Build per-type node index mapping --------------------------------
    # node_id_to_local_idx[node_type][node_id] = local index
    node_id_to_local_idx: Dict[str, Dict[str, int]] = {}

    node_types = ["query", "textblock", "table", "cell", "image", "caption"]
    for ntype in node_types:
        node_list = nodes.get(ntype, [])
        if not node_list:
            # Add empty tensors so HGT doesn't choke
            data[ntype].x = torch.zeros(0, _infer_dim(features))
            data[ntype].node_ids = []
            # Also add empty y for scored node types to ensure consistent structure
            if ntype in SCORED_NODE_TYPES:
                data[ntype].y = torch.zeros(0, dtype=torch.float32)
            node_id_to_local_idx[ntype] = {}
            continue

        id_to_idx = {n["node_id"]: i for i, n in enumerate(node_list)}
        node_id_to_local_idx[ntype] = id_to_idx

        # Stack features
        dim = _infer_dim(features)
        feat_list = []
        for n in node_list:
            nid = n["node_id"]
            if nid in features:
                feat_list.append(features[nid])
            else:
                logger.warning(f"Missing feature for {nid}, using zeros")
                feat_list.append(torch.zeros(dim))
        data[ntype].x = torch.stack(feat_list, dim=0)  # (N, D)
        data[ntype].node_ids = [n["node_id"] for n in node_list]

        # Labels - always add y for scored node types to ensure consistent structure for batching
        if ntype in SCORED_NODE_TYPES:
            if labels:
                y = torch.tensor(
                    [labels.get(n["node_id"], 0) for n in node_list],
                    dtype=torch.float32,
                )
            else:
                # No labels provided, use zeros (e.g., for prediction)
                y = torch.zeros(len(node_list), dtype=torch.float32)
            data[ntype].y = y

    # ---- Build edges -------------------------------------------------------
    for src_type, rel, dst_type in EDGE_SCHEMA:
        edge_list = edges.get(rel, [])
        src_idx_map = node_id_to_local_idx.get(src_type, {})
        dst_idx_map = node_id_to_local_idx.get(dst_type, {})

        src_indices = []
        dst_indices = []
        for (src_id, dst_id) in edge_list:
            if src_id in src_idx_map and dst_id in dst_idx_map:
                src_indices.append(src_idx_map[src_id])
                dst_indices.append(dst_idx_map[dst_id])

        if src_indices:
            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data[(src_type, rel, dst_type)].edge_index = edge_index

    return data
