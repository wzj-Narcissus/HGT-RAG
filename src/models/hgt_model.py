"""HGT model for multi-modal evidence selection.

Architecture:
  - Linear input projections per node type (in_dim -> hidden_dim)
  - N layers of HeteroGraphConv (edge-type-aware message passing with learnable weights)
  - Query-aware scoring heads per node type: MLP(concat(node_h, query_h)) -> 1
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.utils.logging import get_logger

logger = get_logger(__name__)

NODE_TYPES = ["query", "textblock", "table", "cell", "image", "caption"]
EDGE_TYPES = [
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
SCORED_TYPES = ["textblock", "cell", "image", "caption", "table"]


class HeteroGraphConv(nn.Module):
    """Heterogeneous graph convolution with multi-head attention and learnable edge-type weights.

    For each edge type (src, rel, dst):
      1. Linear projections: src features -> query, key, value per head
      2. Attention: alpha_{i,j} = softmax_j(q_i^T k_j / sqrt(d))
      3. Weighted aggregation: weighted sum of value vectors by attention
    After per-edge-type aggregation, learnable scalar weights control each edge type's
    contribution to the destination node update. Output is combined with input via
    residual connection + LayerNorm.
    """

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0

        # Per-edge-type projections: Q, K, V from source features
        self.q_proj = nn.ModuleDict()
        self.k_proj = nn.ModuleDict()
        self.v_proj = nn.ModuleDict()
        self.edge_attn = nn.ParameterDict()

        for src, rel, dst in EDGE_TYPES:
            self.q_proj[rel] = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj[rel] = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj[rel] = nn.Linear(hidden_dim, hidden_dim)
            # Per-head attention scaling parameter
            self.edge_attn[rel] = nn.Parameter(torch.zeros(heads))

        # Output projection per destination node type
        self.out_proj = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, hidden_dim) for ntype in NODE_TYPES
        })

        # Learnable edge-type weights: per (dst_node_type, edge_type) scalar
        self.edge_type_weights = nn.ParameterDict()
        for ntype in NODE_TYPES:
            incoming = [
                rel for src, rel, dst in EDGE_TYPES
                if dst == ntype
            ]
            if incoming:
                key = f"w_{ntype}"
                self.edge_type_weights[key] = nn.Parameter(torch.zeros(len(incoming)))
                # Store which edge types map to this dst, for forward()
                setattr(self, f"_incoming_{ntype}", incoming)

        self.layer_norm = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_dim) for ntype in NODE_TYPES
        })
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Compute per-edge-type aggregated features
        per_edge_agg: Dict[str, List[torch.Tensor]] = {nt: [] for nt in NODE_TYPES}
        per_edge_rel: Dict[str, List[str]] = {nt: [] for nt in NODE_TYPES}

        for (src, rel, dst) in EDGE_TYPES:
            if (src, rel, dst) not in edge_index_dict:
                continue
            edge_index = edge_index_dict[(src, rel, dst)]
            if edge_index.shape[1] == 0:
                continue
            if x_dict.get(src) is None or x_dict[src].shape[0] == 0:
                continue
            if x_dict.get(dst) is None or x_dict[dst].shape[0] == 0:
                continue

            src_feat = x_dict[src]
            n_dst = x_dict[dst].shape[0]
            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            # Project source features
            q = self.q_proj[rel](src_feat[src_idx])  # (E, H)
            k = self.k_proj[rel](src_feat[src_idx])  # (E, H)
            v = self.v_proj[rel](src_feat[src_idx])  # (E, H)

            # Reshape to multi-head
            E = q.shape[0]
            q = q.view(E, self.heads, self.head_dim)
            k = k.view(E, self.heads, self.head_dim)
            v = v.view(E, self.heads, self.head_dim)

            # Attention scores (per-edge, unnormalized)
            attn = (q * k).sum(-1) / (self.head_dim ** 0.5)  # (E, H)
            attn = attn + self.edge_attn[rel].unsqueeze(0)  # (E, H) learned scaling

            # Normalize attention per destination node
            attn_out = torch.zeros(n_dst, self.heads, device=q.device)
            v_out = torch.zeros(n_dst, self.heads, self.head_dim, device=q.device)
            attn_out.index_add_(0, dst_idx, torch.ones_like(attn))
            attn_out = attn_out.clamp(min=1e-6)
            attn_norm = attn / attn_out[dst_idx]  # (E, H)
            attn_weights = F.relu(attn_norm)  # non-negative weights

            # Weighted value aggregation
            weighted_v = v * attn_weights.unsqueeze(-1)  # (E, H, d)
            for h in range(self.heads):
                v_out[:, h, :].index_add_(0, dst_idx, weighted_v[:, h, :])

            # Merge heads
            agg = v_out.reshape(n_dst, self.hidden_dim)  # (N_dst, H)
            per_edge_agg[dst].append(agg)
            per_edge_rel[dst].append(rel)

        # Weighted combination of per-edge-type aggregations + residual
        out_dict = {}
        for ntype in NODE_TYPES:
            if x_dict.get(ntype) is None or x_dict[ntype].shape[0] == 0:
                out_dict[ntype] = x_dict.get(ntype)
                continue

            if not per_edge_agg[ntype]:
                # No incoming edges: pass through
                out_dict[ntype] = x_dict[ntype]
                continue

            # Apply learnable edge-type weights (softmax-normalized)
            key = f"w_{ntype}"
            if key in self.edge_type_weights:
                w = F.softmax(self.edge_type_weights[key], dim=0)
                # Map stored incoming edge types to their indices
                incoming = getattr(self, f"_incoming_{ntype}")
                rel_to_weight = dict(zip(incoming, w))
                weighted_parts = []
                for agg, rel in zip(per_edge_agg[ntype], per_edge_rel[ntype]):
                    weighted_parts.append(rel_to_weight[rel] * agg)
                combined = torch.stack(weighted_parts).sum(0)
            else:
                combined = torch.stack(per_edge_agg[ntype]).sum(0)

            # Output projection + residual + layer norm
            combined = self.out_proj[ntype](combined)
            out_dict[ntype] = self.layer_norm[ntype](
                x_dict[ntype] + self.dropout(combined)
            )

        return out_dict


class QueryAwareScoringHead(nn.Module):
    """Scoring head that conditions on query representation.

    score(node, query) = MLP(concat(node_h, query_h)) -> scalar
    Falls back to MLP(node_h) when no query is available.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Fallback when query is unavailable
        self.mlp_no_query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor, query_h: Optional[torch.Tensor]) -> torch.Tensor:
        if query_h is not None and query_h.shape[0] > 0:
            q = query_h[0]  # (H,)
            q_expanded = q.unsqueeze(0).expand(h.shape[0], -1)  # (N, H)
            combined = torch.cat([h, q_expanded], dim=-1)  # (N, 2H)
            return self.mlp(combined).squeeze(-1)  # (N,)
        return self.mlp_no_query(h).squeeze(-1)


class HGTEvidenceModel(nn.Module):
    """Heterogeneous Graph Transformer for evidence selection.

    Architecture:
      1. Per-type input projection: in_dim -> hidden_dim
      2. N layers of HeteroGraphConv with learnable edge-type weights
      3. Query-aware scoring heads per node type
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

        # Per-node-type input projections (align all types to hidden_dim)
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype in NODE_TYPES
        })

        # HeteroGraphConv layers with learnable edge-type weights
        self.hgt_layers = nn.ModuleList([
            HeteroGraphConv(hidden_dim, heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Scoring heads: query-aware or standard MLP
        self.heads = nn.ModuleDict()
        for ntype in SCORED_TYPES:
            if scoring_head == "query_aware":
                self.heads[ntype] = QueryAwareScoringHead(hidden_dim, dropout)
            elif scoring_head == "bilinear":
                self.heads[ntype] = BilinearHead(hidden_dim)
            else:  # standard MLP
                self.heads[ntype] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Forward pass: encode graph and compute evidence scores.

        Returns:
            logits: dict of ntype -> (N,) tensor of raw scores (before sigmoid)
        """
        # Step 1: Project input features to hidden_dim
        x_dict = {}
        for ntype in NODE_TYPES:
            if data[ntype].x.shape[0] == 0:
                x_dict[ntype] = torch.zeros(
                    0, self.hidden_dim, device=data[ntype].x.device
                )
            else:
                x_dict[ntype] = self.dropout(
                    torch.relu(self.input_proj[ntype](data[ntype].x))
                )

        # Step 2: Build edge_index_dict (skip empty/missing edge types)
        edge_index_dict = {}
        for et in EDGE_TYPES:
            if et not in data.edge_types:
                continue
            ei = data[et].edge_index
            if ei.shape[1] > 0:
                edge_index_dict[et] = ei

        # Step 3: HeteroGraphConv message passing layers
        for layer in self.hgt_layers:
            x_dict = layer(x_dict, edge_index_dict)

        # Step 4: Compute evidence scores via scoring heads
        logits = {}
        query_h = x_dict.get("query")

        for ntype in SCORED_TYPES:
            h = x_dict.get(ntype)
            if h is None or h.shape[0] == 0:
                logits[ntype] = torch.zeros(0, device=list(x_dict.values())[0].device)
                continue
            if self.scoring_head_type == "query_aware":
                logits[ntype] = self.heads[ntype](h, query_h)
            elif self.scoring_head_type == "bilinear" and query_h is not None and query_h.shape[0] > 0:
                q = query_h[0:1]
                logits[ntype] = self.heads[ntype](q, h)
            else:
                logits[ntype] = self.heads[ntype](h).squeeze(-1)

        return logits


class BilinearHead(nn.Module):
    """Bilinear scoring head: score(q, n) = h_q^T W h_n"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # q: (1, H), h: (N, H)
        q_proj = self.W(q)  # (1, H)
        return (q_proj @ h.T).squeeze(0)  # (N,)
