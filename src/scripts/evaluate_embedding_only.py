#!/usr/bin/env python3
"""
评估仅使用Qwen3-VL-Embedding的检索性能（不使用HGT）

这是一个消融实验，用于测试：
1. Qwen3-VL编码器的原始检索能力
2. HGT图神经网络带来的增益
"""

import os
import sys
import argparse
import logging
import hashlib
import json
from typing import Dict, List
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trainers.metrics import EvidenceMetrics
from src.scripts.train import GRAPH_EDGE_SCHEMA_VERSION

SCORED_TYPES = ["textblock", "cell", "image", "caption", "table"]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_default_hetero_dir(split: str) -> str:
    graph_cache_key = {
        "candidate_mode": "oracle",
        "max_cells_per_table": 200,
        "edge_schema_version": GRAPH_EDGE_SCHEMA_VERSION,
    }
    graph_cache_tag = hashlib.sha1(
        json.dumps(graph_cache_key, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"outputs/hetero/{split}_qwen3_vl_embedding_8b_h4096_d256_{graph_cache_tag}"


def load_hetero_data(hetero_dir: str, split: str):
    """加载HeteroData格式的数据"""
    import torch_geometric
    from torch_geometric.data import HeteroData

    # Load all .pt files in the directory
    hetero_files = sorted(Path(hetero_dir).glob("*.pt"))
    logger.info(f"Loading {len(hetero_files)} hetero data files from {hetero_dir}")

    dataset = []
    for file_path in tqdm(hetero_files, desc="Loading data"):
        data = torch.load(file_path, weights_only=False)
        dataset.append(data)

    return dataset


def compute_cosine_similarity(query_emb: torch.Tensor, candidate_embs: torch.Tensor) -> torch.Tensor:
    """
    计算query与所有候选证据的余弦相似度

    Args:
        query_emb: (D,) query embedding
        candidate_embs: (N, D) candidate embeddings

    Returns:
        scores: (N,) cosine similarity scores
    """
    # Normalize
    query_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)  # (1, D)
    candidate_norm = F.normalize(candidate_embs, p=2, dim=1)  # (N, D)

    # Cosine similarity
    scores = torch.mm(query_norm, candidate_norm.t()).squeeze(0)  # (N,)

    return scores


@torch.no_grad()
def evaluate_embedding_only(dataset: List, device: str = "cuda"):
    """
    仅使用embedding进行检索评估

    策略：
    1. 提取query节点的embedding
    2. 提取所有候选证据节点的embedding
    3. 计算余弦相似度作为检索分数
    4. 按分数排序并评估
    """
    metrics = EvidenceMetrics()
    metrics.reset()

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    for data in tqdm(dataset, desc="Evaluating"):
        data = data.to(device)

        # 1. 获取query embedding
        query_emb = data["query"].x[0]  # (D,) - 假设每个图只有一个query节点

        # 2. 对每种节点类型计算相似度
        # 注意：余弦相似度范围是[-1, 1]，需要转换为logits
        # 使用 logit = log(sim / (1 - sim)) 的反函数
        # 或者简单地缩放到合理范围
        all_logits = {}

        for ntype in SCORED_TYPES:
            if ntype not in data.node_types:
                continue

            # 检查是否有标签
            if not hasattr(data[ntype], "y") or data[ntype].y is None:
                continue

            # 获取该类型的所有节点embedding
            node_embs = data[ntype].x  # (N, D)

            # 计算余弦相似度 [-1, 1]
            cos_sim = compute_cosine_similarity(query_emb, node_embs)  # (N,)

            # 转换为logits: 将[-1, 1]映射到合理的logit范围
            # 使用简单的线性缩放: logit = 5 * cos_sim
            # 这样cos_sim=1时logit=5, sigmoid(5)≈0.993
            # cos_sim=0时logit=0, sigmoid(0)=0.5
            # cos_sim=-1时logit=-5, sigmoid(-5)≈0.007
            logits = 5.0 * cos_sim

            all_logits[ntype] = logits

        # 3. 更新metrics (传入data对象)
        metrics.update(all_logits, data)

    # 4. 计算最终指标
    result = metrics.compute()

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding-only retrieval")
    parser.add_argument(
        "--hetero_dir",
        type=str,
        default=None,
        help="Directory containing HeteroData files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/predictions/embedding_only_metrics.json",
        help="Output file for metrics"
    )

    args = parser.parse_args()

    hetero_dir = args.hetero_dir or build_default_hetero_dir(args.split)

    # Load dataset
    logger.info(f"Loading {args.split} dataset from {hetero_dir}")
    dataset = load_hetero_data(hetero_dir, args.split)
    logger.info(f"Loaded {len(dataset)} graphs")

    # Evaluate
    logger.info("Evaluating embedding-only retrieval...")
    metrics = evaluate_embedding_only(dataset, device=args.device)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("Embedding-Only Retrieval Results")
    logger.info("="*50)

    # Group by category
    categories = {
        "Overall": ["overall/recall@1", "overall/recall@3", "overall/recall@5",
                   "overall/mrr", "overall/precision", "overall/recall", "overall/f1"],
        "TextBlock": ["textblock/recall@1", "textblock/recall@5", "textblock/mrr", "textblock/f1"],
        "Cell": ["cell/recall@1", "cell/recall@5", "cell/mrr", "cell/f1"],
        "Image": ["image/recall@1", "image/recall@5", "image/mrr", "image/f1"],
        "Caption": ["caption/recall@1", "caption/recall@5", "caption/mrr", "caption/f1"],
        "Table": ["table/recall@1", "table/recall@5", "table/mrr", "table/f1"],
    }

    for category, metric_keys in categories.items():
        logger.info(f"\n{category}:")
        for key in metric_keys:
            if key in metrics:
                logger.info(f"  {key}: {metrics[key]:.4f}")

    # Save to file
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to {args.output_file}")

    # Print comparison hint
    logger.info("\n" + "="*50)
    logger.info("To compare with HGT results, run:")
    logger.info("  python -m src.scripts.evaluate --config configs/default.yaml --split dev")
    logger.info("="*50)


if __name__ == "__main__":
    main()
