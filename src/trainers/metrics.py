"""Evaluation metrics for evidence selection.

Computes per-type and overall:
  - Recall@1, Recall@3, Recall@5
  - MRR
  - Evidence Precision, Recall, F1

Overall metrics are macro-averaged over the fixed scored node types
instead of ranking nodes across different scoring heads. Missing types
contribute 0.0 so the denominator stays consistent across evaluations.

Per-type thresholds are dynamically optimized on the validation set
to maximize F1, replacing the previous hardcoded 0.5 threshold.
"""
from typing import Dict, List, Tuple
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

SCORED_TYPES = ["textblock", "cell", "image", "caption", "table"]


def _recall_at_k(scores: List[float], labels: List[int], k: int) -> float:
    """Compute Recall@K: fraction of positive items in top-K ranked by score."""
    if not labels or sum(labels) == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_k_labels = [l for _, l in ranked[:k]]
    n_pos_total = sum(labels)
    n_pos_in_topk = sum(top_k_labels)
    return n_pos_in_topk / n_pos_total


def _mrr(scores: List[float], labels: List[int]) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first positive item."""
    if not labels or sum(labels) == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def _precision_recall_f1(
    pred_positive: List[int], gold_positive: List[int]
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 from binary predictions and labels."""
    tp = sum(p == 1 and g == 1 for p, g in zip(pred_positive, gold_positive))
    pred_pos = sum(pred_positive)
    gold_pos = sum(gold_positive)
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / gold_pos if gold_pos > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


class EvidenceMetrics:
    """Accumulates per-example results and computes aggregate metrics.

    Computes per-type and overall metrics:
      - Recall@1, Recall@3, Recall@5, MRR (ranking metrics)
      - Precision, Recall, F1 (threshold-based metrics)

    Overall metrics are macro-averaged over all scored node types.

    Per-type thresholds are dynamically optimized to maximize F1
    by scanning thresholds from 0.05 to 0.95 in steps of 0.05.
    """

    def __init__(self, threshold: float = 0.5):
        self.default_threshold = threshold
        self._reset()

    def _reset(self):
        # Per type: list of (scores_list, labels_list) for each graph
        self._per_type: Dict[str, List[Tuple[List[float], List[int]]]] = {
            t: [] for t in SCORED_TYPES
        }
        # Per-type optimal thresholds (found by search)
        self.per_type_thresholds: Dict[str, float] = {
            t: self.default_threshold for t in SCORED_TYPES
        }

    def update(
        self,
        logits: Dict[str, torch.Tensor],
        data,
    ):
        """Add one graph's predictions to accumulator."""
        for ntype in SCORED_TYPES:
            if ntype not in logits or logits[ntype].shape[0] == 0:
                continue
            if not hasattr(data[ntype], "y") or data[ntype].y is None:
                continue
            scores = torch.sigmoid(logits[ntype]).detach().cpu().tolist()
            labels = data[ntype].y.cpu().int().tolist()
            self._per_type[ntype].append((scores, labels))

    def find_optimal_thresholds(self) -> Dict[str, float]:
        """Search for per-type thresholds that maximize F1 on accumulated data.

        Scans thresholds from 0.05 to 0.95 in steps of 0.05.
        Returns the optimal threshold per node type.
        """
        candidates = [i * 0.05 for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95
        optimal: Dict[str, float] = {}

        for ntype in SCORED_TYPES:
            examples = self._per_type.get(ntype, [])
            if not examples:
                optimal[ntype] = self.default_threshold
                continue

            all_scores: List[float] = []
            all_labels: List[int] = []
            for scores, labels in examples:
                all_scores.extend(scores)
                all_labels.extend(labels)

            if sum(all_labels) == 0:
                optimal[ntype] = self.default_threshold
                continue

            best_f1 = -1.0
            best_t = self.default_threshold
            for t in candidates:
                preds = [1 if s >= t else 0 for s in all_scores]
                _, _, f1 = _precision_recall_f1(preds, all_labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            optimal[ntype] = best_t

        self.per_type_thresholds = optimal
        return optimal

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated predictions.

        Returns:
            Dict with per-type metrics (e.g., "textblock/f1") and
            overall metrics (e.g., "overall/mrr") macro-averaged across types.
        """
        result = {}
        per_type_metrics = {}

        # Find optimal per-type thresholds before computing metrics
        self.find_optimal_thresholds()
        threshold_str = ", ".join(f"{t}={v:.2f}" for t, v in self.per_type_thresholds.items())
        logger.info(f"[Metrics] Optimal per-type thresholds: {threshold_str}")

        # Compute per-type metrics
        for ntype, examples in self._per_type.items():
            if not examples:
                continue
            r1_vals, r3_vals, r5_vals, mrr_vals = [], [], [], []
            all_preds, all_golds = [], []
            threshold = self.per_type_thresholds.get(ntype, self.default_threshold)
            for scores, labels in examples:
                # Ranking metrics: only for examples with positive labels
                if sum(labels) > 0:
                    r1_vals.append(_recall_at_k(scores, labels, 1))
                    r3_vals.append(_recall_at_k(scores, labels, 3))
                    r5_vals.append(_recall_at_k(scores, labels, 5))
                    mrr_vals.append(_mrr(scores, labels))
                # Threshold metrics: for all examples
                preds = [1 if s >= threshold else 0 for s in scores]
                all_preds.extend(preds)
                all_golds.extend(labels)

            p, r, f1 = _precision_recall_f1(all_preds, all_golds)
            metrics = {
                "precision": p,
                "recall": r,
                "f1": f1,
            }
            if r1_vals:
                metrics.update({
                    "recall@1": sum(r1_vals) / len(r1_vals),
                    "recall@3": sum(r3_vals) / len(r3_vals),
                    "recall@5": sum(r5_vals) / len(r5_vals),
                    "mrr": sum(mrr_vals) / len(mrr_vals),
                })
            per_type_metrics[ntype] = metrics
            for metric_name, metric_value in metrics.items():
                result[f"{ntype}/{metric_name}"] = metric_value

        # Compute overall metrics: macro-average over all scored types
        if per_type_metrics:
            for metric_name in ["recall@1", "recall@3", "recall@5", "mrr", "precision", "recall", "f1"]:
                values = [
                    per_type_metrics.get(ntype, {}).get(metric_name, 0.0)
                    for ntype in SCORED_TYPES
                ]
                result[f"overall/{metric_name}"] = sum(values) / len(values)

        return result

    def reset(self):
        self._reset()
