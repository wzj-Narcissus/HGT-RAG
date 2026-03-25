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

def _cell_id(table_id: str, row: int, col: int) -> str:
    return f"cell::{table_id}::{row}::{col}"

def _image_id(image_id: str) -> str:
    return f"image::{image_id}"

def _caption_id(image_id: str) -> str:
    return f"caption::{image_id}"


# ---------------------------------------------------------------------------
# Table flattening helpers
# ---------------------------------------------------------------------------

def _get_table_rows(table_record: Dict) -> List[List[Dict]]:
    """Extract table_rows safely."""
    table_data = table_record.get("table", {})
    if isinstance(table_data, dict):
        rows = table_data.get("table_rows", [])
        return rows if isinstance(rows, list) else []
    return []


def _flatten_table_text(table_record: Dict, max_rows: int = 10) -> str:
    title = table_record.get("title", "")
    rows = _get_table_rows(table_record)
    if not rows:
        return title
    header = " | ".join(c.get("text", "") for c in rows[0])
    body_rows = []
    for row in rows[1:max_rows + 1]:
        body_rows.append(" ; ".join(c.get("text", "") for c in row))
    body = " [ROW] ".join(body_rows)
    return f"{title} [SEP] {header} [SEP] {body}"


def _cell_text_with_context(table_record: Dict, row_idx: int, col_idx: int) -> str:
    rows = _get_table_rows(table_record)
    if not rows:
        return ""
    # header (row 0)
    if len(rows) > 0 and col_idx < len(rows[0]):
        header_text = rows[0][col_idx].get("text", "")
    else:
        header_text = ""
    # row context: all cells in that row
    if row_idx < len(rows):
        row_context = " ; ".join(c.get("text", "") for c in rows[row_idx])
    else:
        row_context = ""
    # cell itself
    cell_text = ""
    if row_idx < len(rows) and col_idx < len(rows[row_idx]):
        cell_text = rows[row_idx][col_idx].get("text", "")
    return f"{header_text} [SEP] {row_context} [SEP] {cell_text}"


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

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
        max_cells_per_table: Cap on cells per table to avoid memory issues.
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
        "query_to_cell": [],
        "query_to_image": [],
        "table_contains_cell": [],
        "text_refers_table": [],
        "text_refers_image": [],
        "cell_to_cell_row": [],
        "cell_to_cell_col": [],
        "image_has_caption": [],
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

    # --- Table + Cell nodes ---
    table_node_id = None
    positive_cell_indices = _get_positive_cell_indices(question)
    cell_node_ids_by_row: Dict[int, List[str]] = {}  # row_idx -> [cell_ids]
    cell_node_ids_by_col: Dict[int, List[str]] = {}  # col_idx -> [cell_ids]
    all_cell_node_ids: List[str] = []

    if table_id is not None:
        if table_id not in tables:
            logger.warning(f"[{qid}] table_id {table_id} not found, skipping table")
        else:
            rec = tables[table_id]
            table_node_id = _table_id(table_id)
            nodes["table"].append({
                "node_id": table_node_id,
                "text": _flatten_table_text(rec),
                "table_id": table_id,
            })
            edges["query_to_table"].append((q_node_id, table_node_id))

            # Create cell nodes with context (header + row context + cell text)
            rows = _get_table_rows(rec)
            cell_count = 0
            for row_idx, row in enumerate(rows):
                for col_idx, cell in enumerate(row):
                    should_keep = cell_count < max_cells_per_table or (row_idx, col_idx) in positive_cell_indices
                    if not should_keep:
                        continue
                    cell_text = _cell_text_with_context(rec, row_idx, col_idx)
                    c_node_id = _cell_id(table_id, row_idx, col_idx)
                    nodes["cell"].append({
                        "node_id": c_node_id,
                        "text": cell_text,
                        "table_id": table_id,
                        "row": row_idx,
                        "col": col_idx,
                    })
                    all_cell_node_ids.append(c_node_id)
                    edges["query_to_cell"].append((q_node_id, c_node_id))
                    edges["table_contains_cell"].append((table_node_id, c_node_id))

                    cell_node_ids_by_row.setdefault(row_idx, []).append(c_node_id)
                    cell_node_ids_by_col.setdefault(col_idx, []).append(c_node_id)
                    if cell_count < max_cells_per_table:
                        cell_count += 1

            # Connect adjacent cells in same row (bidirectional)
            for row_cells in cell_node_ids_by_row.values():
                for i in range(len(row_cells) - 1):
                    edges["cell_to_cell_row"].append((row_cells[i], row_cells[i + 1]))
                    edges["cell_to_cell_row"].append((row_cells[i + 1], row_cells[i]))

            # Connect adjacent cells in same column (bidirectional)
            for col_cells in cell_node_ids_by_col.values():
                for i in range(len(col_cells) - 1):
                    edges["cell_to_cell_col"].append((col_cells[i], col_cells[i + 1]))
                    edges["cell_to_cell_col"].append((col_cells[i + 1], col_cells[i]))

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
        edges["image_has_caption"].append((img_node_id, cap_node_id))

    # Conservative first-pass heuristic: if text/table or text/image coexist in the same
    # question graph, connect them.
    if table_node_id is not None:
        for text_node_id in text_node_ids:
            edges["text_refers_table"].append((text_node_id, table_node_id))

    if image_node_ids:
        for text_node_id in text_node_ids:
            for image_node_id in image_node_ids:
                edges["text_refers_image"].append((text_node_id, image_node_id))

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
