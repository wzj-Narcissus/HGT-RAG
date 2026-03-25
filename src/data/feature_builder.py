"""Build node features for a heterogeneous graph.

Core function: build_node_features(graph, encoder, image_dir)
Returns a dict: node_id -> torch.Tensor (float32, CPU)
"""
import os
from typing import Dict, Optional

import torch

from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_node_features(
    graph: Dict,
    encoder: QwenVLFeatureEncoder,
    image_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Encode every node in the graph with the given encoder.

    For image nodes:
      - If the image file exists locally -> visual encoding
      - If not -> fall back to encoding image.title as text (much better than zero vector)

    Args:
        graph:     graph dict from build_question_graph.
        encoder:   QwenVLFeatureEncoder instance.
        image_dir: directory where image files live (optional).

    Returns:
        features: dict of node_id -> 1-D float32 tensor.
    """
    nodes = graph.get("nodes", {})
    features: Dict[str, torch.Tensor] = {}

    # ---- text-based nodes ------------------------------------------------

    def _encode_text_nodes(node_list, text_key="text"):
        if not node_list:
            return
        ids = [n["node_id"] for n in node_list]
        texts = [n.get(text_key, "") or "" for n in node_list]
        embs = encoder.encode_texts(texts)  # (N, D)
        for nid, emb in zip(ids, embs):
            features[nid] = emb.float()

    _encode_text_nodes(nodes.get("query", []))
    _encode_text_nodes(nodes.get("textblock", []))
    _encode_text_nodes(nodes.get("table", []))
    _encode_text_nodes(nodes.get("cell", []))
    _encode_text_nodes(nodes.get("caption", []))

    # ---- image nodes (visual encoding with text fallback) ----------------

    image_nodes = nodes.get("image", [])
    if not image_nodes:
        return features

    visual_nodes = []   # (node, full_path) where file exists
    text_fallback_nodes = []  # (node, title_text) where file missing

    for n in image_nodes:
        raw_path = n.get("path", "")
        full_path = ""
        if image_dir and raw_path:
            full_path = os.path.join(image_dir, raw_path)

        if full_path and os.path.exists(full_path):
            visual_nodes.append((n, full_path))
        else:
            # Use image title as text fallback
            title = n.get("title", "") or ""
            if not title:
                title = "[image]"
            text_fallback_nodes.append((n, title))

    # Encode visual images
    if visual_nodes:
        ids = [n["node_id"] for n, _ in visual_nodes]
        paths = [p for _, p in visual_nodes]
        embs = encoder.encode_images(paths)
        for nid, emb in zip(ids, embs):
            features[nid] = emb.float()

    # Encode missing images via title text
    if text_fallback_nodes:
        n_missing = len(text_fallback_nodes)
        logger.warning(
            f"[feature_builder] {n_missing} image(s) not found locally. "
            f"Falling back to title-text encoding instead of zero vector. "
            f"To use visual encoding, download images and set data.image_dir in config."
        )
        ids = [n["node_id"] for n, _ in text_fallback_nodes]
        titles = [t for _, t in text_fallback_nodes]
        embs = encoder.encode_texts(titles)
        for nid, emb in zip(ids, embs):
            features[nid] = emb.float()

    return features
