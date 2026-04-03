"""HGT model for MultimodalQA multi-modal evidence selection.

Architecture
------------
1. Per-type input projection: in_dim → hidden_dim
2. N layers of HeteroGraphConv (edge-type-aware message passing with learnable weights)
3. Query-aware scoring heads per evidence node type: MLP(concat(node_h, query_h)) → 1

Graph schema (MultimodalQA)
---------------------------
Node types:   query, textblock, table, cell, image, caption
Edge types:   see EDGE_TYPES below — all edges are bidirectional (explicit reverse edges)
Scored types: textblock, table, cell, image, caption  (all except query)
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.models.base_hgt import HeteroGraphConv, QueryAwareScoringHead
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MultimodalQA graph schema
# ---------------------------------------------------------------------------

NODE_TYPES: List[str] = [
    "query", "textblock", "table", "cell", "image", "caption",
]

EDGE_TYPES: List[Tuple[str, str, str]] = [
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

# Node types for which the model outputs relevance scores
SCORED_TYPES: List[str] = ["textblock", "cell", "image", "caption", "table"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HGTEvidenceModel(nn.Module):
    """Heterogeneous Graph Transformer for MultimodalQA evidence selection.

    Args:
        in_dim:        Dimensionality of input node features (e.g. 4096 for
                       Qwen3-VL-Embedding-8B).
        hidden_dim:    HGT hidden state dimension (default 256).
        num_heads:     Number of attention heads (default 4).
        num_layers:    Number of HGT message-passing layers (default 2).
        dropout:       Dropout probability (default 0.1).
        scoring_head:  Head type — ``"query_aware"`` (default), ``"bilinear"``,
                       or ``"mlp"``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        scoring_head: str = "query_aware",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scoring_head_type = scoring_head

        # Project all node types from in_dim to hidden_dim
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype in NODE_TYPES
        })

        # Stacked HGT layers (schema-agnostic, schema injected here)
        self.hgt_layers = nn.ModuleList([
            HeteroGraphConv(EDGE_TYPES, NODE_TYPES, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # One scoring head per evidence node type
        self.heads = nn.ModuleDict()
        for ntype in SCORED_TYPES:
            if scoring_head == "query_aware":
                self.heads[ntype] = QueryAwareScoringHead(hidden_dim, dropout)
            elif scoring_head == "bilinear":
                self.heads[ntype] = BilinearHead(hidden_dim)
            else:  # plain MLP
                self.heads[ntype] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute per-node relevance logits.

        Args:
            data: PyG HeteroData with ``.x`` tensors for each node type and
                  ``.edge_index`` for each edge type.

        Returns:
            Dict mapping scored node type → logit tensor of shape ``(N,)``.
        """
        # 1. Project input features to hidden_dim
        x_dict: Dict[str, torch.Tensor] = {}
        for ntype in NODE_TYPES:
            if data[ntype].x.shape[0] == 0:
                x_dict[ntype] = torch.zeros(0, self.hidden_dim, device=data[ntype].x.device)
            else:
                x_dict[ntype] = self.dropout(torch.relu(self.input_proj[ntype](data[ntype].x)))

        # 2. Collect non-empty edge types
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in EDGE_TYPES:
            if et not in data.edge_types:
                continue
            ei = data[et].edge_index
            if ei.shape[1] > 0:
                edge_index_dict[et] = ei

        # 3. HGT message passing
        for layer in self.hgt_layers:
            x_dict = layer(x_dict, edge_index_dict)

        # 4. Score evidence nodes
        logits: Dict[str, torch.Tensor] = {}
        query_h = x_dict.get("query")
        device = next(iter(x_dict.values())).device if x_dict else torch.device("cpu")

        for ntype in SCORED_TYPES:
            h = x_dict.get(ntype)
            if h is None or h.shape[0] == 0:
                logits[ntype] = torch.zeros(0, device=device)
                continue
            if self.scoring_head_type == "query_aware":
                logits[ntype] = self.heads[ntype](h, query_h)
            elif self.scoring_head_type == "bilinear" and query_h is not None and query_h.shape[0] > 0:
                logits[ntype] = self.heads[ntype](query_h[0:1], h)
            else:
                logits[ntype] = self.heads[ntype](h).squeeze(-1)

        return logits


class BilinearHead(nn.Module):
    """Bilinear scoring: score(q, n) = (W h_q)ᵀ h_n.

    Args:
        hidden_dim: Dimensionality of hidden states.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Score all nodes against the query.

        Args:
            q: Query tensor of shape ``(1, hidden_dim)``.
            h: Node tensor of shape ``(N, hidden_dim)``.

        Returns:
            Score tensor of shape ``(N,)``.
        """
        return (self.W(q) @ h.T).squeeze(0)
