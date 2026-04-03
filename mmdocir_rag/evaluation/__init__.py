"""Evaluation metrics and scripts for MMDocIR Stage 1 and Stage 2."""

from mmdocir_rag.evaluation.metrics import (
    calculate_overlap_score,
    recall_page_at_k,
    mrr_page,
    recall_layout_at_k,
    mrr_layout,
    recall_layout_exact_at_k,
    aggregate_metrics,
)

__all__ = [
    "calculate_overlap_score",
    "recall_page_at_k",
    "mrr_page",
    "recall_layout_at_k",
    "mrr_layout",
    "recall_layout_exact_at_k",
    "aggregate_metrics",
]
