"""Evaluation utilities for MMDocIR Stage 2.

Aligns with the official metric_eval.py from the MMDocIR benchmark:

  Page retrieval:
    recall_page@k  — binary: 1 if any top-k predicted page_id is in GT page_ids
    mrr_page       — 1/rank of first correct page

  Layout retrieval:
    recall_layout@k  — soft overlap score, accumulated over all (pred, gt) pairs
                       within top-k, clipped to [0, 1].  Mirrors:
                         recall_area += calculate_overlap_score(pred_bbox, gt_bbox)
                         return min(1.0, recall_area)
    mrr_layout     — 1/rank of first pred whose overlap with any GT layout > 0

  calculate_overlap_score  (official formula):
    overlap = intersection_area / min(area(box1), area(box2))
    This is a "coverage" score biased toward the smaller bbox, i.e. a small
    predicted region that is fully inside the GT counts as full credit.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Bbox primitives
# ---------------------------------------------------------------------------

def _area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def calculate_overlap_score(
    box1: Sequence[float],
    box2: Sequence[float],
) -> float:
    """Official MMDocIR overlap score: intersection / min(area1, area2).

    Returns 0 if either box has zero area or there is no intersection.
    """
    ix1 = max(float(box1[0]), float(box2[0]))
    iy1 = max(float(box1[1]), float(box2[1]))
    ix2 = min(float(box1[2]), float(box2[2]))
    iy2 = min(float(box1[3]), float(box2[3]))
    inter = _area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    min_area = min(_area(box1), _area(box2))
    if min_area <= 0:
        return 0.0
    return inter / min_area


# ---------------------------------------------------------------------------
# Page-level metrics
# ---------------------------------------------------------------------------

def recall_page_at_k(
    pred_page_ids: Sequence,
    gt_page_ids: set,
    k: int,
) -> float:
    """Binary recall: 1 if any top-k predicted page is in GT, else 0."""
    if not gt_page_ids:
        return 0.0
    for pid in pred_page_ids[:k]:
        if str(pid) in gt_page_ids:
            return 1.0
    return 0.0


def mrr_page(pred_page_ids: Sequence, gt_page_ids: set) -> float:
    """Mean reciprocal rank for page retrieval."""
    for rank, pid in enumerate(pred_page_ids, start=1):
        if str(pid) in gt_page_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Layout-level metrics  (official soft-overlap semantics)
# ---------------------------------------------------------------------------
#
# pred_layouts: ordered list of tuples from LayoutReranker.rerank():
#   (layout_id, page_id, doc_name, score, layout_type, bbox)
#
# gt_layout_mapping: list of dicts from MMDocIR annotations:
#   {"page": <page_id>, "bbox": [x1,y1,x2,y2], ...}
#
# ---------------------------------------------------------------------------

def _pred_overlaps_any_gt(
    pred_bbox: Sequence[float],
    pred_page_id,
    gt_layout_mapping: List[Dict],
) -> bool:
    """Return True if pred bbox overlaps any GT layout on the same page."""
    pid_str = str(pred_page_id)
    for gt in gt_layout_mapping:
        if str(gt.get("page")) != pid_str:
            continue
        gt_bbox = gt.get("bbox")
        if gt_bbox and calculate_overlap_score(pred_bbox, gt_bbox) > 0:
            return True
    return False


def recall_layout_at_k(
    pred_layouts: List[Tuple],
    gt_layout_mapping: List[Dict],
    k: int,
) -> float:
    """Soft layout recall score (official formula).

    For each of the top-k predicted layouts, for each GT layout on the same page,
    accumulate calculate_overlap_score(pred_bbox, gt_bbox).  Clip to [0, 1].
    """
    recall_area = 0.0
    for entry in pred_layouts[:k]:
        # entry = (layout_id, page_id, doc_name, score, layout_type, bbox)
        _, pred_page_id, _, _, _, pred_bbox = entry
        pid_str = str(pred_page_id)
        for gt in gt_layout_mapping:
            if str(gt.get("page")) != pid_str:
                continue
            gt_bbox = gt.get("bbox")
            if gt_bbox and len(pred_bbox) == 4:
                recall_area += calculate_overlap_score(pred_bbox, gt_bbox)
    return min(1.0, recall_area)


def mrr_layout(
    pred_layouts: List[Tuple],
    gt_layout_mapping: List[Dict],
) -> float:
    """MRR for layout retrieval: 1/rank of first pred that overlaps any GT layout."""
    for rank, entry in enumerate(pred_layouts, start=1):
        _, pred_page_id, _, _, _, pred_bbox = entry
        if _pred_overlaps_any_gt(pred_bbox, pred_page_id, gt_layout_mapping):
            return 1.0 / rank
    return 0.0


def recall_layout_exact_at_k(
    pred_layouts: List[Tuple],
    gt_layout_ids: set,
    k: int,
) -> float:
    """Binary recall by exact layout_id match (supplementary metric)."""
    if not gt_layout_ids:
        return 0.0
    for entry in pred_layouts[:k]:
        layout_id = str(entry[0])
        if layout_id in gt_layout_ids:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(
    rows: Iterable[Dict],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict:
    """Aggregate per-question metric dicts into corpus-level averages."""
    rows = list(rows)
    total = len(rows)
    out: Dict = {"total_questions": total}

    metric_keys = (
        [f"page_recall@{k}" for k in ks]
        + [f"layout_recall@{k}" for k in ks]
        + [f"layout_exact_recall@{k}" for k in ks]
        + ["page_mrr", "layout_mrr"]
    )
    if total == 0:
        for key in metric_keys:
            out[key] = 0.0
        return out

    for key in metric_keys:
        vals = [r.get(key, 0.0) for r in rows]
        out[key] = sum(vals) / total

    return out
