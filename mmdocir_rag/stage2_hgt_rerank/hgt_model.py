"""HGT model for MMDocIR Stage 2 layout reranking.

Architecture
------------
1. Per-type input projection: in_dim → hidden_dim
2. N layers of HeteroGraphConv (from src.models.base_hgt)
3. Query-aware scoring heads for 'page' and 'layout' node types

Graph schema (MMDocIR)
----------------------
Node types:   query, page, layout
Edge types:   see EDGE_TYPES below — all edges are bidirectional
Scored types: page, layout

Difference from MultimodalQA HGT (src/models/hgt_model.py)
------------------------------------------------------------
- Node granularity: MMDocIR uses coarse page + layout nodes; MMQA uses
  fine-grained textblock/table/cell/image/caption nodes.
- Edge connectivity: MMDocIR adds spatial adjacency edges between layouts
  on the same page.
- Both models share HeteroGraphConv and QueryAwareScoringHead from
  src.models.base_hgt.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.models.base_hgt import HeteroGraphConv, QueryAwareScoringHead
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MMDocIR graph schema
# ---------------------------------------------------------------------------

NODE_TYPES: List[str] = ["query", "page", "layout"]

EDGE_TYPES: List[Tuple[str, str, str]] = [
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

# Node types for which the model outputs relevance scores
SCORED_TYPES: List[str] = ["page", "layout"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MMDocIRHGT(nn.Module):
    """3-layer HGT for MMDocIR layout reranking.

    Args:
        in_dim:     Encoder output dimension (e.g. 4096 for Qwen3-VL-Embedding-8B).
        hidden_dim: HGT hidden dimension (default 256).
        num_heads:  Number of attention heads (default 4).
        num_layers: Number of HGT message-passing layers (default 3).
        dropout:    Dropout probability (default 0.1).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project all node types from in_dim to hidden_dim
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype in NODE_TYPES
        })

        # Stacked HGT layers (shared implementation, MMDocIR schema injected)
        self.hgt_layers = nn.ModuleList([
            HeteroGraphConv(EDGE_TYPES, NODE_TYPES, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Query-aware scoring head for each scored node type
        self.heads = nn.ModuleDict({
            ntype: QueryAwareScoringHead(hidden_dim, dropout)
            for ntype in SCORED_TYPES
        })

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute per-node relevance logits.

        Args:
            data: PyG HeteroData with ``.x`` tensors for each node type and
                  ``.edge_index`` for each edge type.

        Returns:
            Dict mapping scored node type (``"page"``, ``"layout"``) → logit
            tensor of shape ``(N,)``.
        """
        # 1. Project input features to hidden_dim
        x_dict: Dict[str, torch.Tensor] = {}
        for ntype in NODE_TYPES:
            x = data[ntype].x
            if x.shape[0] == 0:
                x_dict[ntype] = torch.zeros(0, self.hidden_dim, device=x.device)
            else:
                x_dict[ntype] = self.dropout(torch.relu(self.input_proj[ntype](x)))

        # 2. Collect non-empty edge types
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in EDGE_TYPES:
            if et in data.edge_types:
                ei = data[et].edge_index
                if ei.shape[1] > 0:
                    edge_index_dict[et] = ei

        # 3. HGT message passing
        for layer in self.hgt_layers:
            x_dict = layer(x_dict, edge_index_dict)

        # 4. Score page and layout nodes
        logits: Dict[str, torch.Tensor] = {}
        query_h = x_dict.get("query")
        device = next(iter(x_dict.values())).device if x_dict else torch.device("cpu")

        for ntype in SCORED_TYPES:
            h = x_dict.get(ntype)
            if h is None or h.shape[0] == 0:
                logits[ntype] = torch.zeros(0, device=device)
            else:
                logits[ntype] = self.heads[ntype](h, query_h)

        return logits
