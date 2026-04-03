"""Train Stage 2 HGT reranker on MMDocIR.

This script:
1. Loads MMDocIR pages + layouts
2. For each question in the training set:
   - Retrieves top-K pages (using Stage 1 cached embeddings or oracle page_ids)
   - Builds a heterogeneous graph
   - Runs one gradient step on the HGT reranker
3. Evaluates on eval set periodically and saves best checkpoint

Usage:
    python -m mmdocir_rag.stage2_hgt_rerank.train_stage2 \\
        --config mmdocir_rag/configs/stage2.yaml \\
        [--checkpoint path/to/stage2_init.pt]

Config keys (see mmdocir_rag/configs/stage2.yaml):
    data:
        pages_parquet:     path to MMDocIR_pages.parquet
        layouts_parquet:   path to MMDocIR_layouts.parquet
        annotations_jsonl: path to MMDocIR_annotations.jsonl
        text_mode:         ocr_text | vlm_text
        top_k_pages:       how many pages Stage 1 retrieves (1 / 5 / 10)
    model:
        in_dim:        4096
        hidden_dim:    256
        num_heads:     4
        num_layers:    3
        dropout:       0.1
    train:
        lr:            1e-4
        weight_decay:  1e-5
        epochs:        10
        log_every:     50
        eval_every:    500
        save_dir:      outputs/stage2_checkpoints
    encoder:
        model_path:    /model/ModelScope/Qwen/Qwen3-VL-Embedding-8B
        device:        cuda
        dtype:         bf16
"""
import argparse
import os
import random
import sys
from typing import Dict, List

import yaml

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mmdocir_rag.data.mmdocir_loader import MMDocIRLoader
from mmdocir_rag.stage2_hgt_rerank.layout_reranker import LayoutReranker
from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate(reranker: LayoutReranker, loader: MMDocIRLoader, top_k: int = 5) -> Dict:
    """Layout recall@K and page recall@K evaluation.

    Uses the same top-K sampling strategy as training: all positive pages +
    randomly sampled negatives to fill up to top_k total. This mirrors the
    Stage 1 retrieval scenario and keeps evaluation fast.
    """
    hits_layout = 0
    hits_page = 0
    total = 0

    for qa in loader.iter_questions():
        doc_name = qa["doc_name"]
        positive_page_ids = qa["page_id"]
        positive_layout_ids = [
            m.get("layout_id") for m in qa.get("layout_mapping", [])
            if m.get("layout_id") is not None
        ]

        try:
            pages_df = loader.get_document_pages(doc_name)
        except ValueError:
            continue

        all_page_rows = [row for _, row in pages_df.iterrows()]
        pos_set = set(str(p) for p in positive_page_ids)
        pos_rows = [r for r in all_page_rows if str(r.get("passage_id", "")) in pos_set]
        neg_rows = [r for r in all_page_rows if str(r.get("passage_id", "")) not in pos_set]
        n_neg = max(top_k - len(pos_rows), 0)
        sampled_negs = random.sample(neg_rows, min(n_neg, len(neg_rows)))
        retrieved_pages = pos_rows + sampled_negs

        result = reranker.rerank(
            qid=qa["qid"],
            query_text=qa["question"],
            retrieved_pages=retrieved_pages,
            layouts_df=loader.layouts_df,
            top_k_layouts=top_k,
            top_k_pages=top_k,
        )

        pred_layout_ids = {r[0] for r in result["layout_scores"]}
        pred_page_ids   = {r[0] for r in result["page_scores"]}

        if any(lid in pred_layout_ids for lid in positive_layout_ids):
            hits_layout += 1
        if any(pid in pred_page_ids for pid in positive_page_ids):
            hits_page += 1
        total += 1

    return {
        f"page_recall@{top_k}":   hits_page / total if total else 0.0,
        f"layout_recall@{top_k}": hits_layout / total if total else 0.0,
        "total_questions": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 HGT reranker")
    parser.add_argument("--config", required=True, help="Path to stage2.yaml config")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths in config relative to the project root
    config_dir = os.path.dirname(os.path.abspath(args.config))
    project_root = os.path.dirname(os.path.dirname(config_dir))  # mmdocir_rag/configs/ -> mmdocir_rag/ -> project_root

    def _resolve(path: str) -> str:
        if path and not os.path.isabs(path):
            root_path = os.path.join(project_root, path)
            if os.path.exists(root_path):
                return root_path
            return os.path.join(config_dir, path)
        return path

    data_cfg  = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    enc_cfg   = cfg.get("encoder", {})

    # Load data (paths resolved relative to config file)
    loader = MMDocIRLoader(
        pages_parquet_path=_resolve(data_cfg["pages_parquet"]),
        layouts_parquet_path=_resolve(data_cfg["layouts_parquet"]),
        annotations_jsonl_path=_resolve(data_cfg["annotations_jsonl"]) if data_cfg.get("annotations_jsonl") else None,
        text_mode=data_cfg.get("text_mode", "ocr_text"),
    )

    # Load encoder
    enc_cfg_copy = dict(enc_cfg)
    encoder = QwenVLFeatureEncoder.from_config(enc_cfg_copy)

    # Build reranker
    top_k_pages = data_cfg.get("top_k_pages", 5)
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

    if args.checkpoint:
        reranker.load(args.checkpoint, load_optimizer=False)

    reranker.setup_optimizer(
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    save_dir    = _resolve(train_cfg.get("save_dir", "outputs/stage2_checkpoints"))
    log_every   = train_cfg.get("log_every", 50)
    eval_every  = train_cfg.get("eval_every", 500)
    num_epochs  = train_cfg.get("epochs", 10)

    os.makedirs(save_dir, exist_ok=True)

    best_recall = 0.0
    step = 0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{num_epochs} ===")
        for qa in loader.iter_questions():
            doc_name = qa["doc_name"]
            positive_page_ids   = qa["page_id"]
            positive_layout_ids = [
                m.get("layout_id") for m in qa.get("layout_mapping", [])
                if m.get("layout_id") is not None
            ]

            try:
                pages_df = loader.get_document_pages(doc_name)
            except ValueError:
                continue

            all_page_rows = [row for _, row in pages_df.iterrows()]
            pos_set = set(str(p) for p in positive_page_ids)

            # Separate positives and negatives by passage_id
            pos_rows = [r for r in all_page_rows if str(r.get("passage_id", "")) in pos_set]
            neg_rows = [r for r in all_page_rows if str(r.get("passage_id", "")) not in pos_set]

            # Sample negatives to fill up to top_k_pages total
            n_neg = max(top_k_pages - len(pos_rows), 0)
            sampled_negs = random.sample(neg_rows, min(n_neg, len(neg_rows)))
            retrieved_pages = pos_rows + sampled_negs

            try:
                data = reranker.build_data(
                    qid=qa["qid"],
                    query_text=qa["question"],
                    retrieved_pages=retrieved_pages,
                    layouts_df=loader.layouts_df,
                    positive_page_ids=positive_page_ids,
                    positive_layout_ids=positive_layout_ids,
                )
            except Exception as e:
                logger.warning(f"Failed to build data for {qa['qid']}: {e}")
                continue

            loss_dict = reranker.train_step(data)
            step += 1

            if step % log_every == 0:
                loss_str = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                logger.info(f"[step {step}] {loss_str}")

            if step % eval_every == 0:
                logger.info(f"[step {step}] Evaluating...")
                metrics = evaluate(reranker, loader, top_k=top_k_pages)
                for k, v in metrics.items():
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.4f}")
                    else:
                        logger.info(f"  {k}: {v}")

                recall = metrics.get(f"layout_recall@{top_k_pages}", 0.0)
                if recall > best_recall:
                    best_recall = recall
                    ckpt_path = os.path.join(save_dir, "best.pt")
                    reranker.save(ckpt_path)
                    logger.info(f"  New best layout_recall={recall:.4f}, saved to {ckpt_path}")

        # Save epoch checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
        reranker.save(ckpt_path)

    logger.info(f"Training complete. Best layout_recall={best_recall:.4f}")


if __name__ == "__main__":
    main()
