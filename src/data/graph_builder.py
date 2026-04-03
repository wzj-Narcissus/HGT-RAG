"""Build question-specific heterogeneous graphs from MMQA data.

Core function: build_question_graph(question, texts, tables, images, cfg)
Returns a graph dict (JSON-serializable) with nodes and edges.
"""
from typing import Dict, List, Optional, Set, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _get_positive_cell_indices(question: Dict) -> Set[Tuple[int, int]]:
    positive_cells: Set[Tuple[int, int]] = set()
    for ans in question.get("answers", []):
        for idx in ans.get("table_indices", []):
            if isinstance(idx, (list, tuple)) and len(idx) == 2:
                try:
                    positive_cells.add((int(idx[0]), int(idx[1])))
                except (TypeError, ValueError):
                    continue
    return positive_cells


# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------

def _qid(qid: str) -> str:
    return f"q::{qid}"

def _text_id(doc_id: str) -> str:
    return f"text::{doc_id}"

def _table_id(table_id: str) -> str:
    return f"table::{table_id}"

def _image_id(image_id: str) -> str:
    return f"image::{image_id}"

def _caption_id(image_id: str) -> str:
    return f"caption::{image_id}"


def _cell_id(table_id: str, row: int, col: int) -> str:
    return f"cell::{table_id}::{row}::{col}"


def build_question_graph(
    question: Dict,
    texts: Dict[str, Dict],
    tables: Dict[str, Dict],
    images: Dict[str, Dict],
    max_cells_per_table: int = 200,
    candidate_mode: str = "oracle",
) -> Dict:
    """Build a question-specific heterogeneous graph.

    Args:
        question:   A MMQA question record.
        texts:      id -> text record (from load_mmqa_texts).
        tables:     id -> table record (from load_mmqa_tables).
        images:     id -> image record (from load_mmqa_images).
        max_cells_per_table: Cap on non-positive cell nodes per table to avoid memory issues.
            Gold positive cells from MMQA annotations are always preserved, even beyond this cap.
        candidate_mode: 'oracle' uses gold supporting_context, 'retrieval_lite' uses metadata.

    Returns:
        Graph dict with 'nodes' and 'edges' (JSON-serializable).
    """
    if candidate_mode != "oracle":
        raise ValueError(
            f"Unsupported candidate_mode={candidate_mode!r}. Only 'oracle' is currently supported."
        )

    qid = question.get("qid", "")
    meta = question.get("metadata", {})

    # Extract candidate document IDs from question metadata (oracle mode)
    text_doc_ids: List[str] = meta.get("text_doc_ids", [])
    table_id: Optional[str] = meta.get("table_id", None)
    image_doc_ids: List[str] = meta.get("image_doc_ids", [])

    nodes: Dict[str, List[Dict]] = {
        "query": [],
        "textblock": [],
        "table": [],
        "cell": [],
        "image": [],
        "caption": [],
    }
    edges: Dict[str, List[Tuple[str, str]]] = {
        "query_to_text": [],
        "query_to_table": [],
        "query_to_image": [],
        "query_to_cell": [],
        "text_to_query": [],
        "table_to_query": [],
        "image_to_query": [],
        "cell_to_query": [],
        "text_refers_table": [],
        "table_refers_text": [],
        "text_refers_image": [],
        "image_refers_text": [],
        "image_has_caption": [],
        "caption_to_image": [],
        "table_contains_cell": [],
        "cell_in_table": [],
        "cell_to_cell_row": [],
        "cell_to_cell_col": [],
    }

    # --- Query node ---
    q_node_id = _qid(qid)
    nodes["query"].append({
        "node_id": q_node_id,
        "text": question.get("question", ""),
    })

    # --- TextBlock nodes ---
    text_node_ids = []
    for doc_id in text_doc_ids:
        if doc_id not in texts:
            logger.warning(f"[{qid}] text doc_id {doc_id} not found, skipping")
            continue
        rec = texts[doc_id]
        title = rec.get("title", "")
        body = rec.get("text", "")
        if not body and not title:
            logger.warning(f"[{qid}] text doc_id {doc_id} empty, skipping")
            continue
        node_id = _text_id(doc_id)
        nodes["textblock"].append({
            "node_id": node_id,
            "text": f"{title} [SEP] {body}" if title else body,
            "doc_id": doc_id,
        })
        text_node_ids.append(node_id)
        edges["query_to_text"].append((q_node_id, node_id))
        edges["text_to_query"].append((node_id, q_node_id))

    # --- Table node (as image) + Cell nodes ---
    table_node_id = None
    if table_id is not None:
        if table_id not in tables:
            logger.warning(f"[{qid}] table_id {table_id} not found, skipping table")
        else:
            rec = tables[table_id]
            table_node_id = _table_id(table_id)
            table_data = rec.get("table", {})
            nodes["table"].append({
                "node_id": table_node_id,
                "table_id": table_id,
                "title": rec.get("title", ""),
                "table_data": table_data,
            })
            edges["query_to_table"].append((q_node_id, table_node_id))
            edges["table_to_query"].append((table_node_id, q_node_id))

            table_rows = table_data.get("table_rows", []) if isinstance(table_data, dict) else []
            max_cells = max_cells_per_table if max_cells_per_table and max_cells_per_table > 0 else None
            positive_cells = _get_positive_cell_indices(question)
            cell_node_ids: List[List[Optional[str]]] = []
            negative_cell_budget = 0

            for row_idx, row in enumerate(table_rows):
                row_cell_ids: List[Optional[str]] = []
                for col_idx, cell in enumerate(row):
                    is_positive_cell = (row_idx, col_idx) in positive_cells
                    if max_cells is not None and not is_positive_cell and negative_cell_budget >= max_cells:
                        row_cell_ids.append(None)
                        continue
                    cell_text = cell.get("text", "") if isinstance(cell, dict) else str(cell)
                    cell_node_id = _cell_id(table_id, row_idx, col_idx)
                    nodes["cell"].append({
                        "node_id": cell_node_id,
                        "table_id": table_id,
                        "row": row_idx,
                        "col": col_idx,
                        "text": cell_text,
                        "is_positive_candidate": is_positive_cell,
                    })
                    row_cell_ids.append(cell_node_id)
                    edges["table_contains_cell"].append((table_node_id, cell_node_id))
                    edges["cell_in_table"].append((cell_node_id, table_node_id))
                    edges["query_to_cell"].append((q_node_id, cell_node_id))
                    edges["cell_to_query"].append((cell_node_id, q_node_id))
                    if not is_positive_cell:
                        negative_cell_budget += 1
                if any(cell_id is not None for cell_id in row_cell_ids):
                    cell_node_ids.append(row_cell_ids)

            for row_cell_ids in cell_node_ids:
                for left_idx in range(len(row_cell_ids) - 1):
                    src = row_cell_ids[left_idx]
                    dst = row_cell_ids[left_idx + 1]
                    if src and dst:
                        edges["cell_to_cell_row"].append((src, dst))
                        edges["cell_to_cell_row"].append((dst, src))

            max_cols = max((len(row) for row in cell_node_ids), default=0)
            for col_idx in range(max_cols):
                prev_cell_id = None
                for row_cell_ids in cell_node_ids:
                    if col_idx >= len(row_cell_ids):
                        prev_cell_id = None
                        continue
                    cell_id = row_cell_ids[col_idx]
                    if prev_cell_id and cell_id:
                        edges["cell_to_cell_col"].append((prev_cell_id, cell_id))
                        edges["cell_to_cell_col"].append((cell_id, prev_cell_id))
                    prev_cell_id = cell_id

    # --- Image + Caption nodes ---
    image_node_ids = []
    for img_id in image_doc_ids:
        if img_id not in images:
            logger.warning(f"[{qid}] image_id {img_id} not found, skipping")
            continue
        rec = images[img_id]
        img_node_id = _image_id(img_id)
        cap_node_id = _caption_id(img_id)
        caption_text = rec.get("title", "")  # use image.title as caption text

        nodes["image"].append({
            "node_id": img_node_id,
            "image_id": img_id,
            "path": rec.get("path", ""),
            "title": rec.get("title", ""),
        })
        nodes["caption"].append({
            "node_id": cap_node_id,
            "text": caption_text,
            "image_id": img_id,
        })

        image_node_ids.append(img_node_id)
        edges["query_to_image"].append((q_node_id, img_node_id))
        edges["image_to_query"].append((img_node_id, q_node_id))
        edges["image_has_caption"].append((img_node_id, cap_node_id))
        edges["caption_to_image"].append((cap_node_id, img_node_id))

    # Cross-modal edges: connect text/table and text/image bidirectionally
    if table_node_id is not None:
        for text_node_id in text_node_ids:
            edges["text_refers_table"].append((text_node_id, table_node_id))
            edges["table_refers_text"].append((table_node_id, text_node_id))

    if image_node_ids:
        for text_node_id in text_node_ids:
            for image_node_id in image_node_ids:
                edges["text_refers_image"].append((text_node_id, image_node_id))
                edges["image_refers_text"].append((image_node_id, text_node_id))

    stats = {
        "n_text": len(nodes["textblock"]),
        "n_table": len(nodes["table"]),
        "n_cell": len(nodes["cell"]),
        "n_image": len(nodes["image"]),
        "n_caption": len(nodes["caption"]),
        "n_edges": sum(len(v) for v in edges.values()),
    }

    return {
        "qid": qid,
        "nodes": nodes,
        "edges": {k: [list(e) for e in v] for k, v in edges.items()},
        "stats": stats,
    }
