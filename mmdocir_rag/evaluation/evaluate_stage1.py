"""Evaluate Stage 1 page retrieval (embedding-only baseline) on MMDocIR.

Metrics align with the official MMDocIR metric_eval.py:
  - page_recall@1/5/10  (binary)
  - page_mrr

Stage 1 does not rank individual layouts, so layout metrics are 0 here and
exist only for compatibility with aggregate_metrics().

Usage:
    python -m mmdocir_rag.evaluation.evaluate_stage1 \
        --config mmdocir_rag/configs/stage2.yaml \
        --top-k 10 \
        --output outputs/stage1_eval/metrics.json
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mmdocir_rag.data.mmdocir_loader import MMDocIRLoader
from mmdocir_rag.evaluation.metrics import aggregate_metrics, mrr_page, recall_page_at_k
from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_paths(cfg: Dict, config_path: str) -> Dict:
    # 从项目根目录（configs/ 的父目录的父目录）解析相对路径
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(os.path.dirname(config_dir))  # mmdocir_rag/configs/ -> mmdocir_rag/ -> project_root

    def _r(p: str) -> str:
        if not p or os.path.isabs(p):
            return p
        # 先尝试相对于项目根目录
        root_path = os.path.join(project_root, p)
        if os.path.exists(root_path):
            return root_path
        # 回退到相对于配置文件目录
        return os.path.join(config_dir, p)

    data = dict(cfg.get("data", {}))
    for key in ("pages_parquet", "layouts_parquet", "annotations_jsonl"):
        if data.get(key):
            data[key] = _r(data[key])
    cfg["data"] = data
    return cfg


def encode_all_pages(
    loader: MMDocIRLoader,
    encoder,
    batch_size: int,
    text_mode: str,
) -> tuple:
    texts, meta = [], []
    for _, row in loader.pages_df.iterrows():
        txt = row.get(text_mode) or row.get("ocr_text") or ""
        texts.append(str(txt))
        meta.append({
            "doc_name": str(row.get("doc_name", "")),
            "passage_id": str(row.get("passage_id", "")),
        })

    embs = []
    for i in range(0, len(texts), batch_size):
        e = encoder.encode_texts(texts[i: i + batch_size])
        e = F.normalize(e, dim=-1).cpu().float().numpy()
        embs.append(e)
        if (i // batch_size) % 20 == 0:
            logger.info(f"  Encoded pages {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(embs), meta


def retrieve_pages_for_doc(
    query_emb: np.ndarray,
    page_matrix: np.ndarray,
    page_meta: List[Dict],
    doc_name: str,
    top_k: int,
) -> List[str]:
    doc_indices = [i for i, m in enumerate(page_meta) if m["doc_name"] == doc_name]
    if not doc_indices:
        return []
    sub = page_matrix[doc_indices]
    scores = sub @ query_emb
    order = np.argsort(scores)[::-1][:top_k]
    return [page_meta[doc_indices[i]]["passage_id"] for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, args.config)

    data_cfg = cfg.get("data", {})
    enc_cfg = cfg.get("encoder", {})
    batch_size = enc_cfg.get("batch_size", 32)
    text_mode = data_cfg.get("text_mode", "ocr_text")

    loader = MMDocIRLoader(
        pages_parquet_path=data_cfg["pages_parquet"],
        layouts_parquet_path=data_cfg["layouts_parquet"],
        annotations_jsonl_path=data_cfg.get("annotations_jsonl"),
        text_mode=text_mode,
    )

    encoder = QwenVLFeatureEncoder.from_config(dict(enc_cfg))

    logger.info("Encoding all pages...")
    page_matrix, page_meta = encode_all_pages(loader, encoder, batch_size, text_mode)
    logger.info(f"Page matrix shape: {page_matrix.shape}")

    per_question: List[Dict] = []
    for qa in loader.iter_questions():
        doc_name = qa["doc_name"]
        gt_page_ids = {str(x) for x in qa.get("page_id", [])}

        q_emb = encoder.encode_texts([qa["question"]])
        q_emb = F.normalize(q_emb, dim=-1).cpu().float().numpy().squeeze(0)

        pred_page_ids = retrieve_pages_for_doc(
            q_emb, page_matrix, page_meta, doc_name, top_k=args.top_k
        )

        row: Dict = {
            "qid": qa["qid"],
            "doc_name": doc_name,
            "question": qa["question"],
            "question_type": qa.get("type", ""),
            "page_mrr": mrr_page(pred_page_ids, gt_page_ids),
            "layout_mrr": 0.0,
        }
        for k in (1, 5, 10):
            row[f"page_recall@{k}"] = recall_page_at_k(pred_page_ids, gt_page_ids, k)
            row[f"layout_recall@{k}"] = 0.0
            row[f"layout_exact_recall@{k}"] = 0.0

        per_question.append(row)
        if len(per_question) % 50 == 0:
            logger.info(f"Evaluated {len(per_question)} questions")

    metrics = aggregate_metrics(per_question, ks=(1, 5, 10))

    type_rows: Dict[str, List[Dict]] = {}
    for r in per_question:
        qtype = r.get("question_type", "unknown") or "unknown"
        type_rows.setdefault(qtype, []).append(r)
    by_type = {
        qtype: aggregate_metrics(rows, ks=(1, 5, 10))
        for qtype, rows in type_rows.items()
        if rows
    }

    output = {"metrics": metrics, "by_type": by_type, "num_questions": len(per_question)}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Stage 1 Evaluation Results ===")
    for key in sorted(metrics):
        v = metrics[key]
        if isinstance(v, float):
            print(f"  {key}: {v:.4f}")
        else:
            print(f"  {key}: {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
