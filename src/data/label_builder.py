"""Build node-level binary labels for evidence selection.

Core function: build_labels(question, graph)
"""
from typing import Dict, Set
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_labels(question: Dict, graph: Dict) -> Dict[str, Dict[str, int]]:
    """Assign binary positive/negative labels to each node.

    Returns:
        labels: dict of node_id -> 0/1
        Also returns a breakdown by node type.
    """
    qid = question.get("qid", "")
    answers = question.get("answers", [])
    supporting_context = question.get("supporting_context", [])

    # --- supporting_context positive doc_ids ---
    supporting_text_ids: Set[str] = set()
    supporting_table_ids: Set[str] = set()
    supporting_image_ids: Set[str] = set()
    for ctx in supporting_context:
        doc_id = ctx.get("doc_id", "")
        part = ctx.get("doc_part", "")
        if part == "text":
            supporting_text_ids.add(doc_id)
        elif part == "table":
            supporting_table_ids.add(doc_id)
        elif part == "image":
            supporting_image_ids.add(doc_id)

    # --- answer-level positive ids ---
    answer_text_ids: Set[str] = set()
    answer_table_ids: Set[str] = set()
    answer_image_ids: Set[str] = set()
    positive_cells: Set[tuple] = set()  # (table_id, row, col)

    for ans in answers:
        # text_instances: list of {doc_id, ...}
        for ti in ans.get("text_instances", []):
            if isinstance(ti, dict):
                answer_text_ids.add(ti.get("doc_id", ""))
        # image_instances
        for ii in ans.get("image_instances", []):
            if isinstance(ii, dict):
                answer_image_ids.add(ii.get("doc_id", ""))
            elif isinstance(ii, str):
                answer_image_ids.add(ii)
        # table_indices: list of [row, col]
        for idx in ans.get("table_indices", []):
            if isinstance(idx, (list, tuple)) and len(idx) == 2:
                positive_cells.add((idx[0], idx[1]))

    pos_text_ids = supporting_text_ids | answer_text_ids
    pos_table_ids = supporting_table_ids | answer_table_ids
    pos_image_ids = supporting_image_ids | answer_image_ids

    labels: Dict[str, int] = {}
    breakdown: Dict[str, Dict[str, int]] = {
        "textblock": {}, "table": {}, "cell": {}, "image": {}, "caption": {}
    }

    nodes = graph.get("nodes", {})

    # TextBlock labels
    for n in nodes.get("textblock", []):
        node_id = n["node_id"]
        doc_id = n.get("doc_id", "")
        label = 1 if doc_id in pos_text_ids else 0
        labels[node_id] = label
        breakdown["textblock"][node_id] = label

    # Table labels
    for n in nodes.get("table", []):
        node_id = n["node_id"]
        tid = n.get("table_id", "")
        label = 1 if tid in pos_table_ids else 0
        labels[node_id] = label
        breakdown["table"][node_id] = label

    # Cell labels
    # Need table_id from question metadata
    meta = question.get("metadata", {})
    q_table_id = meta.get("table_id", "")
    for n in nodes.get("cell", []):
        node_id = n["node_id"]
        row = n.get("row", -1)
        col = n.get("col", -1)
        label = 1 if (row, col) in positive_cells else 0
        labels[node_id] = label
        breakdown["cell"][node_id] = label

    # Image labels
    for n in nodes.get("image", []):
        node_id = n["node_id"]
        img_id = n.get("image_id", "")
        label = 1 if img_id in pos_image_ids else 0
        labels[node_id] = label
        breakdown["image"][node_id] = label

    # Caption labels: same as corresponding image
    for n in nodes.get("caption", []):
        node_id = n["node_id"]
        img_id = n.get("image_id", "")
        img_node_id = f"image::{img_id}"
        label = labels.get(img_node_id, 0)
        labels[node_id] = label
        breakdown["caption"][node_id] = label

    # log stats
    for ntype, lmap in breakdown.items():
        if lmap:
            pos = sum(v for v in lmap.values())
            total = len(lmap)
            logger.debug(f"[{qid}] {ntype}: {pos}/{total} positive")

    return {"labels": labels, "breakdown": breakdown}
