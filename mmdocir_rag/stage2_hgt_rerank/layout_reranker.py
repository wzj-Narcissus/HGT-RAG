"""Stage 2 HGT Layout Reranker.

Wraps MMDocIRHGT for training and inference on MMDocIR layout reranking.

Usage (inference):
    reranker = LayoutReranker(encoder, cfg)
    reranker.load("checkpoints/stage2_best.pt")

    results = reranker.rerank(
        qid="q001",
        query_text="What is the GDP of China?",
        retrieved_pages=[page_row1, page_row2, ...],
        layouts_df=loader.layouts_df,
        top_k_layouts=5,
    )

Usage (training):
    reranker.train_step(data)   # data: HeteroData from hetero_converter
    reranker.save("checkpoints/stage2_best.pt")
"""
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from mmdocir_rag.stage2_hgt_rerank.graph_builder import build_query_graph
from mmdocir_rag.stage2_hgt_rerank.hetero_converter import (
    convert_to_heterodata,
    build_labels,
    SCORED_NODE_TYPES,
)
from mmdocir_rag.stage2_hgt_rerank.hgt_model import MMDocIRHGT
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LayoutRerankLoss(nn.Module):
    """Focal + margin ranking loss for page and layout scoring."""

    def __init__(
        self,
        lambda_page: float = 0.5,
        lambda_layout: float = 1.0,
        lambda_rank: float = 0.5,
        margin: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.lambda_page = lambda_page
        self.lambda_layout = lambda_layout
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.gamma = focal_gamma
        self.alpha = focal_alpha

    def _focal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.gamma > 0:
            p_t = torch.exp(-bce)
            bce = ((1 - p_t) ** self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce = alpha_t * bce
        return bce.mean()

    def _ranking(self, logits: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return None
        pos = logits[pos_mask].unsqueeze(1)
        neg = logits[neg_mask].unsqueeze(0)
        return F.relu(self.margin - (pos - neg)).mean()

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        total = None

        for ntype, lam in [("page", self.lambda_page), ("layout", self.lambda_layout)]:
            if lam == 0:
                continue
            lg = logits.get(ntype)
            if lg is None or lg.shape[0] == 0:
                continue
            if not hasattr(data[ntype], "y") or data[ntype].y is None:
                continue
            y = data[ntype].y
            if y.shape[0] == 0:
                continue

            fl = self._focal(lg, y)
            loss_dict[f"focal_{ntype}"] = fl
            total = lam * fl if total is None else total + lam * fl

        # Ranking loss on layout scores (primary target)
        lg = logits.get("layout")
        if lg is not None and lg.shape[0] > 0 and self.lambda_rank > 0:
            if hasattr(data["layout"], "y") and data["layout"].y is not None:
                rl = self._ranking(lg, data["layout"].y)
                if rl is not None:
                    loss_dict["rank"] = rl
                    total = self.lambda_rank * rl if total is None else total + self.lambda_rank * rl

        if total is None:
            device = next(iter(logits.values())).device if logits else torch.device("cpu")
            total = torch.tensor(0.0, device=device)

        loss_dict["total"] = total
        return loss_dict


# ---------------------------------------------------------------------------
# Node feature encoder (text-only for Stage 2)
# ---------------------------------------------------------------------------

def encode_graph_features(
    graph: Dict,
    encoder,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """Encode all graph nodes as text embeddings.

    For MMDocIR Stage 2 we always encode via text (ocr_text / vlm_text already
    resolved in the graph nodes). Images and tables carry descriptive text from
    the loader, which is sufficient for reranking.
    """
    nodes = graph.get("nodes", {})
    features: Dict[str, torch.Tensor] = {}

    for ntype in ["query", "page", "layout"]:
        node_list = nodes.get(ntype, [])
        if not node_list:
            continue
        ids   = [n["node_id"] for n in node_list]
        texts = [n.get("text", "") or "" for n in node_list]
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embs = encoder.encode_texts(batch)  # (B, D)
            all_embs.append(embs.cpu())
        combined = torch.cat(all_embs, dim=0)
        for nid, emb in zip(ids, combined):
            features[nid] = emb.float()

    return features


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class LayoutReranker:
    """Stage 2 HGT layout reranker for MMDocIR.

    Args:
        encoder:     QwenVLFeatureEncoder (or compatible) for node feature encoding.
        in_dim:      Encoder output dimension (default 4096 for Qwen).
        hidden_dim:  HGT hidden dimension (default 256).
        num_heads:   Number of attention heads (default 4).
        num_layers:  Number of HGT layers (default 3).
        dropout:     Dropout rate (default 0.1).
        device:      Torch device string.
    """

    def __init__(
        self,
        encoder,
        in_dim: int = 4096,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cuda",
        text_mode: str = "ocr_text",
        spatial_proximity_frac: float = 0.05,
        max_layouts_per_page: Optional[int] = None,
    ):
        self.encoder = encoder
        self.device = torch.device(device)
        self.text_mode = text_mode
        self.spatial_proximity_frac = spatial_proximity_frac
        self.max_layouts_per_page = max_layouts_per_page

        self.model = MMDocIRHGT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.loss_fn = LayoutRerankLoss()
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ----------------------------------------------------------------
    # Training helpers
    # ----------------------------------------------------------------

    def setup_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_step(self, data: HeteroData) -> Dict[str, float]:
        """Single training step. Returns scalar loss values."""
        assert self.optimizer is not None, "Call setup_optimizer() first."
        self.model.train()
        data = data.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(data)
        loss_dict = self.loss_fn(logits, data)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

    # ----------------------------------------------------------------
    # Build HeteroData from raw inputs (used in both train and eval)
    # ----------------------------------------------------------------

    def build_data(
        self,
        qid: str,
        query_text: str,
        retrieved_pages: List[pd.Series],
        layouts_df: pd.DataFrame,
        positive_page_ids: Optional[List] = None,
        positive_layout_ids: Optional[List] = None,
    ) -> HeteroData:
        graph = build_query_graph(
            qid=qid,
            query_text=query_text,
            retrieved_pages=retrieved_pages,
            layouts_df=layouts_df,
            text_mode=self.text_mode,
            spatial_proximity_frac=self.spatial_proximity_frac,
            max_layouts_per_page=self.max_layouts_per_page,
            positive_page_ids=positive_page_ids,
            positive_layout_ids=positive_layout_ids,
        )
        features = encode_graph_features(graph, self.encoder)
        label_result = build_labels(graph)
        labels = label_result["labels"]
        data = convert_to_heterodata(graph, features, labels)
        return data

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    @torch.no_grad()
    def rerank(
        self,
        qid: str,
        query_text: str,
        retrieved_pages: List[pd.Series],
        layouts_df: pd.DataFrame,
        top_k_layouts: int = 5,
        top_k_pages: int = 3,
    ) -> Dict:
        """Rerank retrieved pages and their layouts.

        Returns:
            {
                "page_scores":   [(passage_id, score), ...],  sorted desc
                "layout_scores": [(layout_id, page_id, score, layout_type), ...],  sorted desc
            }
        """
        self.model.eval()

        graph = build_query_graph(
            qid=qid,
            query_text=query_text,
            retrieved_pages=retrieved_pages,
            layouts_df=layouts_df,
            text_mode=self.text_mode,
            spatial_proximity_frac=self.spatial_proximity_frac,
            max_layouts_per_page=self.max_layouts_per_page,
        )
        features = encode_graph_features(graph, self.encoder)
        data = convert_to_heterodata(graph, features)
        data = data.to(self.device)

        logits = self.model(data)

        # Collect page scores
        page_nodes  = graph["nodes"]["page"]
        page_logits = logits.get("page")
        page_scores: List[Tuple] = []
        if page_logits is not None and page_logits.shape[0] > 0:
            page_probs = torch.sigmoid(page_logits).cpu().tolist()
            for node, score in zip(page_nodes, page_probs):
                page_scores.append((
                    node["passage_id"],
                    node["doc_name"],
                    score,
                ))
        page_scores.sort(key=lambda x: x[2], reverse=True)

        # Collect layout scores
        layout_nodes  = graph["nodes"]["layout"]
        layout_logits = logits.get("layout")
        layout_scores: List[Tuple] = []
        if layout_logits is not None and layout_logits.shape[0] > 0:
            layout_probs = torch.sigmoid(layout_logits).cpu().tolist()
            for node, score in zip(layout_nodes, layout_probs):
                layout_scores.append((
                    node["layout_id"],
                    node["page_id"],
                    node["doc_name"],
                    score,
                    node.get("layout_type", ""),
                    node.get("bbox", []),
                ))
        layout_scores.sort(key=lambda x: x[3], reverse=True)

        return {
            "qid":           qid,
            "page_scores":   page_scores[:top_k_pages],
            "layout_scores": layout_scores[:top_k_layouts],
        }

    # ----------------------------------------------------------------
    # Checkpoint helpers
    # ----------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
        }, path)
        logger.info(f"Saved Stage 2 checkpoint to {path}")

    def load(self, path: str, load_optimizer: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if load_optimizer and self.optimizer and ckpt.get("optimizer_state"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info(f"Loaded Stage 2 checkpoint from {path}")
