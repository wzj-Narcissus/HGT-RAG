"""Shared utilities for converting graph dicts into PyG HeteroData objects.

This module provides the schema-agnostic core of the heterogeneous graph
conversion pipeline, used by both:

- ``src/data/hetero_converter.py``  (MultimodalQA schema)
- ``mmdocir_rag/stage2_hgt_rerank/hetero_converter.py``  (MMDocIR schema)

Exported symbols
----------------
infer_feature_dim      — infer embedding dimension from a features dict
graph_dict_to_heterodata — schema-parameterised graph → HeteroData conversion
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from src.utils.logging import get_logger

logger = get_logger(__name__)


def infer_feature_dim(features: Dict[str, torch.Tensor]) -> int:
    """Return the embedding dimension of the first tensor in ``features``.

    Falls back to 384 (all-MiniLM-L6-v2 default) when the dict is empty.

    Args:
        features: Mapping from node_id to embedding tensor.

    Returns:
        Integer feature dimension.
    """
    for v in features.values():
        return v.shape[-1]
    return 384


def graph_dict_to_heterodata(
    graph: Dict,
    features: Dict[str, torch.Tensor],
    edge_schema: List[Tuple[str, str, str]],
    node_types: List[str],
    scored_node_types: List[str],
    labels: Optional[Dict[str, int]] = None,
) -> HeteroData:
    """Convert a graph dict and feature map into a PyG HeteroData object.

    This function is schema-agnostic — the graph schema (node types, edge
    types, and which node types carry labels) is passed as arguments, so the
    same logic works for both the MMQA and MMDocIR graph schemas.

    Args:
        graph:             Graph dict as produced by the relevant graph builder.
                           Must contain ``{"nodes": {ntype: [node_dict, ...]},
                           "edges": {rel_name: [(src_id, dst_id), ...]}}``.
        features:          Mapping from node_id (str) → embedding tensor.
        edge_schema:       List of ``(src_type, rel_name, dst_type)`` triples
                           that define the edge types present in this schema.
        node_types:        Ordered list of all node type names.
        scored_node_types: Subset of ``node_types`` that carry binary labels
                           (``data[ntype].y``).  These also get ``y = zeros``
                           when no labels are provided, ensuring a consistent
                           structure during batching.
        labels:            Optional mapping from node_id → 0/1 label.  When
                           ``None``, scored node types receive ``y = zeros``.

    Returns:
        PyG HeteroData with ``.x``, ``.node_ids``, and (for scored types)
        ``.y`` attributes on each node type, plus ``.edge_index`` for each
        edge type in ``edge_schema``.
    """
    data = HeteroData()
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})

    dim = infer_feature_dim(features)

    # Build node feature matrices and local-index maps
    node_id_to_local_idx: Dict[str, Dict[str, int]] = {}

    for ntype in node_types:
        node_list = nodes.get(ntype, [])

        if not node_list:
            data[ntype].x = torch.zeros(0, dim)
            data[ntype].node_ids = []
            if ntype in scored_node_types:
                data[ntype].y = torch.zeros(0, dtype=torch.float32)
            node_id_to_local_idx[ntype] = {}
            continue

        id_to_idx = {n["node_id"]: i for i, n in enumerate(node_list)}
        node_id_to_local_idx[ntype] = id_to_idx

        feat_list = []
        for n in node_list:
            nid = n["node_id"]
            if nid in features:
                feat_list.append(features[nid])
            else:
                logger.warning(f"Missing feature for {nid!r}, substituting zeros")
                feat_list.append(torch.zeros(dim))

        data[ntype].x = torch.stack(feat_list, dim=0)       # (N, D)
        data[ntype].node_ids = [n["node_id"] for n in node_list]

        if ntype in scored_node_types:
            if labels:
                y = torch.tensor(
                    [labels.get(n["node_id"], 0) for n in node_list],
                    dtype=torch.float32,
                )
            else:
                y = torch.zeros(len(node_list), dtype=torch.float32)
            data[ntype].y = y

    # Build edge index tensors
    for src_type, rel, dst_type in edge_schema:
        edge_list = edges.get(rel, [])
        src_idx_map = node_id_to_local_idx.get(src_type, {})
        dst_idx_map = node_id_to_local_idx.get(dst_type, {})

        src_indices, dst_indices = [], []
        for src_id, dst_id in edge_list:
            if src_id in src_idx_map and dst_id in dst_idx_map:
                src_indices.append(src_idx_map[src_id])
                dst_indices.append(dst_idx_map[dst_id])

        if src_indices:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data[(src_type, rel, dst_type)].edge_index = edge_index

    return data
