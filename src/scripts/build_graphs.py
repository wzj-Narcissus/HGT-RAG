"""Build and save heterogeneous graphs for all splits.

Usage:
    python -m src.scripts.build_graphs --config configs/default.yaml [--split train]
"""
import argparse
import os
import sys

import yaml

from src.data.mmqa_loader import (
    load_mmqa_questions,
    load_mmqa_texts,
    load_mmqa_tables,
    load_mmqa_images,
)
from src.data.graph_builder import build_question_graph
from src.data.label_builder import build_labels
from src.utils.logging import get_logger
from src.scripts.train import GRAPH_EDGE_SCHEMA_VERSION
from src.utils.serialization import save_json

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override max_samples for quick debug")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    graph_cfg = cfg.get("graph", {})
    output_cfg = cfg.get("output", {})

    dataset_dir = data_cfg["dataset_dir"]
    graphs_dir = output_cfg.get("graphs_dir", "outputs/graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    max_samples = args.max_samples or data_cfg.get(
        f"max_{'train' if args.split == 'train' else 'dev'}_samples"
    )

    logger.info(f"Loading {args.split} questions ...")
    questions = load_mmqa_questions(dataset_dir, split=args.split,
                                    max_samples=max_samples)
    logger.info("Loading supporting documents ...")
    texts = load_mmqa_texts(dataset_dir)
    tables = load_mmqa_tables(dataset_dir)
    images = load_mmqa_images(dataset_dir)

    logger.info(f"Building graphs for {len(questions)} questions ...")
    n_nodes_total, n_edges_total = 0, 0
    for i, q in enumerate(questions):
        qid = q.get("qid", str(i))
        graph = build_question_graph(
            q, texts, tables, images,
            max_cells_per_table=graph_cfg.get("max_cells_per_table", 200),
            candidate_mode=graph_cfg.get("candidate_mode", "oracle"),
        )
        labels = build_labels(q, graph)
        # Attach labels to graph for easy loading
        graph["labels"] = labels["labels"]
        graph["label_breakdown"] = labels["breakdown"]
        graph["qid"] = qid
        graph["edge_schema_version"] = GRAPH_EDGE_SCHEMA_VERSION

        out_path = os.path.join(graphs_dir, f"{args.split}_{qid}.json")
        save_json(graph, out_path)

        stats = graph["stats"]
        n_nodes_total += sum(stats[k] for k in stats if k.startswith("n_") and k != "n_edges")
        n_edges_total += stats.get("n_edges", 0)

        if (i + 1) % 500 == 0 or i == 0:
            logger.info(
                f"  [{i+1}/{len(questions)}] qid={qid} "
                f"nodes={sum(v for k,v in stats.items() if k.startswith('n_') and k!='n_edges')} "
                f"edges={stats['n_edges']}"
            )

    logger.info(
        f"Done. {len(questions)} graphs saved to {graphs_dir}. "
        f"Total nodes={n_nodes_total}, edges={n_edges_total}"
    )


if __name__ == "__main__":
    main()
