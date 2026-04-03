"""Schema-agnostic HGT building blocks shared by all downstream HGT models.

This module decouples the HGT message-passing logic from the graph schema
(node types / edge types) so both the MultimodalQA pipeline (src/models/hgt_model.py)
and the MMDocIR RAG pipeline (mmdocir_rag/stage2_hgt_rerank/hgt_model.py) can
reuse the same implementation.

Exported symbols
----------------
HeteroGraphConv       — one HGT layer, schema injected at construction time
QueryAwareScoringHead — MLP head that conditions on the query node representation
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGraphConv(nn.Module):
    """One layer of Heterogeneous Graph Transformer (HGT) message passing.

    The schema (node types and edge types) is injected at construction time, so
    the same class works for any heterogeneous graph.

    For each edge type ``(src, rel, dst)``:

    1. Project source node features into Q/K/V spaces per attention head.
    2. Compute scaled dot-product attention scores, normalized per destination
       node with a numerically-stable softmax (exp-sum trick).
    3. Aggregate weighted value vectors into destination nodes.

    Per-edge-type contributions to each destination node type are combined via
    learnable softmax-normalized scalar weights.  A residual connection and
    LayerNorm are applied after the projection.

    Args:
        edge_types:  List of ``(src_type, rel_name, dst_type)`` triples that
                     define the graph schema.  Each unique ``rel_name`` gets its
                     own set of Q/K/V projections.
        node_types:  List of all node type names.  Used for output projections
                     and LayerNorm modules.
        hidden_dim:  Dimensionality of node hidden states (must be divisible by
                     ``heads``).
        heads:       Number of attention heads.
        dropout:     Dropout probability applied after the output projection.
    """

    def __init__(
        self,
        edge_types: List[Tuple[str, str, str]],
        node_types: List[str],
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.edge_types = edge_types
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        # Per-relation Q/K/V projections.
        # Q is projected from the *destination* node (what dst wants),
        # K and V from the *source* node (what src provides).
        self.q_proj = nn.ModuleDict()
        self.k_proj = nn.ModuleDict()
        self.v_proj = nn.ModuleDict()
        # Per-relation, per-head learnable attention bias
        self.edge_attn = nn.ParameterDict()

        for _, rel, _ in edge_types:
            if rel not in self.q_proj:
                self.q_proj[rel] = nn.Linear(hidden_dim, hidden_dim)
                self.k_proj[rel] = nn.Linear(hidden_dim, hidden_dim)
                self.v_proj[rel] = nn.Linear(hidden_dim, hidden_dim)
                self.edge_attn[rel] = nn.Parameter(torch.zeros(heads))

        # Output projection per destination node type
        self.out_proj = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, hidden_dim) for ntype in node_types
        })

        # Learnable softmax-normalized scalars that weight each incoming
        # edge-type's contribution to a destination node type.
        self.edge_type_weights = nn.ParameterDict()
        for ntype in node_types:
            incoming = [rel for _, rel, dst in edge_types if dst == ntype]
            if incoming:
                self.edge_type_weights[f"w_{ntype}"] = nn.Parameter(
                    torch.zeros(len(incoming))
                )
                # Keep the ordered list of relations for forward()
                setattr(self, f"_incoming_{ntype}", incoming)

        self.layer_norm = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_dim) for ntype in node_types
        })
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run one HGT message-passing step.

        Args:
            x_dict:          Dict mapping node type -> feature matrix (N, hidden_dim).
            edge_index_dict: Dict mapping ``(src, rel, dst)`` -> edge_index tensor
                             of shape ``(2, E)``.

        Returns:
            Updated ``x_dict`` with the same keys.
        """
        # Accumulate per-edge-type aggregations for each destination node type
        per_edge_agg: Dict[str, List[torch.Tensor]] = {nt: [] for nt in self.node_types}
        per_edge_rel: Dict[str, List[str]] = {nt: [] for nt in self.node_types}

        for src, rel, dst in self.edge_types:
            if (src, rel, dst) not in edge_index_dict:
                continue
            edge_index = edge_index_dict[(src, rel, dst)]
            if edge_index.shape[1] == 0:
                continue

            src_feat = x_dict.get(src)
            dst_feat = x_dict.get(dst)
            if src_feat is None or src_feat.shape[0] == 0:
                continue
            if dst_feat is None or dst_feat.shape[0] == 0:
                continue

            n_dst = dst_feat.shape[0]
            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            # Q from dst, K/V from src (standard HGT formulation)
            q = self.q_proj[rel](dst_feat[dst_idx])   # (E, hidden_dim)
            k = self.k_proj[rel](src_feat[src_idx])   # (E, hidden_dim)
            v = self.v_proj[rel](src_feat[src_idx])   # (E, hidden_dim)

            E = q.shape[0]
            q = q.view(E, self.heads, self.head_dim)
            k = k.view(E, self.heads, self.head_dim)
            v = v.view(E, self.heads, self.head_dim)

            # Scaled dot-product attention + per-relation bias
            attn = (q * k).sum(-1) / (self.head_dim ** 0.5)   # (E, heads)
            attn = attn + self.edge_attn[rel].unsqueeze(0)

            # Numerically-stable softmax per destination node (exp-sum trick)
            attn_max = torch.zeros(n_dst, self.heads, device=q.device)
            attn_max.index_reduce_(0, dst_idx, attn, reduce="amax", include_self=True)
            attn_exp = torch.exp(attn - attn_max[dst_idx])
            attn_sum = torch.zeros(n_dst, self.heads, device=q.device)
            attn_sum.index_add_(0, dst_idx, attn_exp)
            attn_norm = attn_exp / attn_sum[dst_idx].clamp(min=1e-9)  # (E, heads)

            # Aggregate values weighted by attention
            v_out = torch.zeros(n_dst, self.heads, self.head_dim, device=q.device)
            weighted_v = v * attn_norm.unsqueeze(-1)
            for h in range(self.heads):
                v_out[:, h, :].index_add_(0, dst_idx, weighted_v[:, h, :])

            agg = v_out.reshape(n_dst, self.hidden_dim)
            per_edge_agg[dst].append(agg)
            per_edge_rel[dst].append(rel)

        # Combine per-edge-type aggregations and apply residual + LayerNorm
        out_dict: Dict[str, torch.Tensor] = {}
        for ntype in self.node_types:
            x = x_dict.get(ntype)
            if x is None or x.shape[0] == 0:
                out_dict[ntype] = x
                continue
            if not per_edge_agg[ntype]:
                # No incoming edges: pass through unchanged
                out_dict[ntype] = x
                continue

            key = f"w_{ntype}"
            if key in self.edge_type_weights:
                # Softmax-normalize edge-type weights then compute weighted sum
                w = F.softmax(self.edge_type_weights[key], dim=0)
                incoming = getattr(self, f"_incoming_{ntype}")
                rel_to_w = dict(zip(incoming, w))
                parts = [rel_to_w[r] * a for a, r in zip(per_edge_agg[ntype], per_edge_rel[ntype])]
                combined = torch.stack(parts).sum(0)
            else:
                combined = torch.stack(per_edge_agg[ntype]).sum(0)

            combined = self.out_proj[ntype](combined)
            out_dict[ntype] = self.layer_norm[ntype](x + self.dropout(combined))

        return out_dict


class QueryAwareScoringHead(nn.Module):
    """Scoring head that conditions a node's score on the query representation.

    score(node_i) = MLP(concat(h_i, h_query))

    When no query representation is available, falls back to
    ``MLP(concat(h_i, h_i))`` (self-concatenation) to keep the input
    dimension consistent.

    Args:
        hidden_dim: Dimensionality of node hidden states.
        dropout:    Dropout probability inside the MLP.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, query_h: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute relevance logits for a batch of nodes.

        Args:
            h:       Node feature matrix of shape ``(N, hidden_dim)``.
            query_h: Query node feature matrix ``(Q, hidden_dim)`` or None.
                     Only the first row ``query_h[0]`` is used.

        Returns:
            Logit tensor of shape ``(N,)``.
        """
        if query_h is not None and query_h.shape[0] > 0:
            q = query_h[0].unsqueeze(0).expand(h.shape[0], -1)  # (N, hidden_dim)
        else:
            # Fallback: treat node itself as the query context
            q = h
        return self.mlp(torch.cat([h, q], dim=-1)).squeeze(-1)
