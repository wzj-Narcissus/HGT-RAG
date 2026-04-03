"""MMDocIR two-stage RAG pipeline.

Stage 1 — Page retrieval via dense embedding similarity (Qwen3-VL-Embedding-8B).
Stage 2 — Layout reranking via Heterogeneous Graph Transformer.
"""

from mmdocir_rag.data.mmdocir_loader import MMDocIRLoader
from mmdocir_rag.stage2_hgt_rerank.layout_reranker import LayoutReranker

__all__ = [
    "MMDocIRLoader",
    "LayoutReranker",
]
