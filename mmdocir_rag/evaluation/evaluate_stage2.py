"""Evaluate Stage 2 HGT reranker on MMDocIR.

Metrics align with the official MMDocIR metric_eval.py:
  - page_recall@1/5/10  (binary)
  - layout_recall@1/5/10  (soft overlap, intersection/min-area, clipped to 1)
  - layout_exact_recall@1/5/10  (binary by layout_id, supplementary)
  - page_mrr, layout_mrr

Usage:
    python -m mmdocir_rag.evaluation.evaluate_stage2 \
        --config mmdocir_rag/configs/stage2.yaml \
        --checkpoint outputs/stage2_checkpoints/best.pt \
        --top-k 10 \
        --output outputs/stage2_eval/metrics.json \
        [--save-predictions]
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mmdocir_rag.data.mmdocir_loader import MMDocIRLoader
from mmdocir_rag.evaluation.metrics import (
    aggregate_metrics,
    mrr_layout,
    mrr_page,
    recall_layout_at_k,
    recall_layout_exact_at_k,
    recall_page_at_k,
)
from mmdocir_rag.stage2_hgt_rerank.layout_reranker import LayoutReranker
from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_paths(cfg: Dict, config_path: str) -> Dict:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(os.path.dirname(config_dir))  # mmdocir_rag/configs/ -> mmdocir_rag/ -> project_root

    def _r(p: str) -> str:
        if not p or os.path.isabs(p):
            return p
        root_path = os.path.join(project_root, p)
        if os.path.exists(root_path):
            return root_path
        return os.path.join(config_dir, p)

    data = dict(cfg.get("data", {}))
    train = dict(cfg.get("train", {}))
    for key in ("pages_parquet", "layouts_parquet", "annotations_jsonl"):
        if data.get(key):
            data[key] = _r(data[key])
    if train.get("save_dir"):
        train["save_dir"] = _r(train["save_dir"])
    cfg["data"] = data
    cfg["train"] = train
    return cfg


def evaluate_question(
    reranker: LayoutReranker,
    loader: MMDocIRLoader,
    qa: Dict,
    top_k: int,
) -> Dict:
    doc_name = qa["doc_name"]
    pages_df = loader.get_document_pages(doc_name)
    retrieved_pages = [row for _, row in pages_df.iterrows()]

    result = reranker.rerank(
        qid=qa["qid"],
        query_text=qa["question"],
        retrieved_pages=retrieved_pages,
        layouts_df=loader.layouts_df,
        top_k_layouts=top_k,
        top_k_pages=top_k,
    )

    # pred_page_ids: str list ordered by score descending
    pred_page_ids = [str(r[0]) for r in result["page_scores"]]
    pred_layouts = result["layout_scores"]   # list of (layout_id, page_id, doc_name, score, type, bbox)

    gt_page_ids = {str(x) for x in qa.get("page_id", [])}
    gt_layout_ids = {
        str(m.get("layout_id"))
        for m in qa.get("layout_mapping", [])
        if m.get("layout_id") is not None
    }
    gt_layout_mapping = qa.get("layout_mapping", [])

    row: Dict = {
        "qid": qa["qid"],
        "doc_name": doc_name,
        "question": qa["question"],
        "question_type": qa.get("type", ""),
        "page_mrr": mrr_page(pred_page_ids, gt_page_ids),
        "layout_mrr": mrr_layout(pred_layouts, gt_layout_mapping),
    }
    for k in (1, 5, 10):
        row[f"page_recall@{k}"] = recall_page_at_k(pred_page_ids, gt_page_ids, k)
        row[f"layout_recall@{k}"] = recall_layout_at_k(pred_layouts, gt_layout_mapping, k)
        row[f"layout_exact_recall@{k}"] = recall_layout_exact_at_k(pred_layouts, gt_layout_ids, k)

    row["predictions"] = {
        "page_scores": [(r[0], r[1], r[2]) for r in result["page_scores"]],
        "layout_scores": [(r[0], r[1], r[2]) for r in result["layout_scores"]],
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, args.config)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    enc_cfg = cfg.get("encoder", {})

    loader = MMDocIRLoader(
        pages_parquet_path=data_cfg["pages_parquet"],
        layouts_parquet_path=data_cfg["layouts_parquet"],
        annotations_jsonl_path=data_cfg.get("annotations_jsonl"),
        text_mode=data_cfg.get("text_mode", "ocr_text"),
    )

    encoder = QwenVLFeatureEncoder.from_config(dict(enc_cfg))
    reranker = LayoutReranker(
        encoder=encoder,
        in_dim=model_cfg.get("in_dim", encoder.hidden_dim),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
        device=enc_cfg.get("device", "cuda"),
        text_mode=data_cfg.get("text_mode", "ocr_text"),
        max_layouts_per_page=data_cfg.get("max_layouts_per_page"),
    )
    reranker.load(args.checkpoint, load_optimizer=False)

    per_question: List[Dict] = []
    for qa in loader.iter_questions():
        try:
            row = evaluate_question(reranker, loader, qa, args.top_k)
            if not args.save_predictions:
                row.pop("predictions", None)
            per_question.append(row)
            if len(per_question) % 50 == 0:
                logger.info(f"Evaluated {len(per_question)} questions")
        except Exception as e:
            logger.warning(f"Skip {qa['qid']}: {e}")

    metrics = aggregate_metrics(per_question, ks=(1, 5, 10))

    # Per-type breakdown (if type field present)
    type_rows: Dict[str, List[Dict]] = {}
    for r in per_question:
        qtype = r.get("question_type", "unknown") or "unknown"
        type_rows.setdefault(qtype, []).append(r)
    by_type = {
        qtype: aggregate_metrics(rows, ks=(1, 5, 10))
        for qtype, rows in type_rows.items()
        if rows
    }

    output = {
        "metrics": metrics,
        "by_type": by_type,
        "num_questions": len(per_question),
    }
    if args.save_predictions:
        output["per_question"] = per_question

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Stage 2 Evaluation Results ===")
    for key in sorted(metrics):
        v = metrics[key]
        if isinstance(v, float):
            print(f"  {key}: {v:.4f}")
        else:
            print(f"  {key}: {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
