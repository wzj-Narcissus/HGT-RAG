"""Build query-page-layout heterogeneous graphs for MMDocIR Stage 2 HGT reranking.

Graph schema (inspired by MultiModalQA graph_builder.py):
  Node types:
    - query:      1 node per query
    - page:       top-K retrieved pages (from Stage 1)
    - layout:     all layout elements on retrieved pages

  Edge types (bidirectional):
    - query <-> page:     query connected to each retrieved page
    - query <-> layout:   query connected to each layout on retrieved pages
    - page <-> layout:    page connected to its child layouts
    - layout <-> layout:  spatially adjacent layouts on the same page (by bbox)

Layout types in MMDocIR: text, title, equation, table, image
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------

def _qid(qid: str) -> str:
    return f"q::{qid}"

def _page_nid(doc_name: str, passage_id) -> str:
    return f"page::{doc_name}::{passage_id}"

def _layout_nid(doc_name: str, layout_id) -> str:
    return f"layout::{doc_name}::{layout_id}"


# ---------------------------------------------------------------------------
# Spatial adjacency helpers
# ---------------------------------------------------------------------------

def _bbox_to_xyxy(bbox) -> Tuple[float, float, float, float]:
    """Normalize bbox to (x1, y1, x2, y2). Returns zeros on failure."""
    if bbox is None:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        b = list(bbox)
        if len(b) == 4:
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except (TypeError, ValueError):
        pass
    return (0.0, 0.0, 0.0, 0.0)


def _are_adjacent(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
    proximity_frac: float = 0.05,
    page_h: float = 1.0,
) -> bool:
    """Return True if two bboxes overlap or are within proximity_frac * page_h vertically."""
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    gap = proximity_frac * (page_h if page_h > 0 else 1.0)
    x_overlap = x1a < x2b and x2a > x1b
    y_near = y1a < y2b + gap and y2a > y1b - gap
    return x_overlap and y_near


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_query_graph(
    qid: str,
    query_text: str,
    retrieved_pages: List[pd.Series],
    layouts_df: pd.DataFrame,
    text_mode: str = "ocr_text",
    spatial_proximity_frac: float = 0.05,
    max_layouts_per_page: Optional[int] = None,
    positive_page_ids: Optional[List] = None,
    positive_layout_ids: Optional[List] = None,
) -> Dict:
    """Build a query-specific heterogeneous graph for Stage 2 reranking.

    Args:
        qid:                     Unique question identifier.
        query_text:              Natural language question string.
        retrieved_pages:         List of page rows (pd.Series) from Stage 1 top-K retrieval.
        layouts_df:              Full layouts DataFrame; filtered by doc_name & page_id.
        text_mode:               "ocr_text" or "vlm_text".
        spatial_proximity_frac:  Fraction of page height for spatial adjacency threshold.
        max_layouts_per_page:    Cap on layouts per page (None = no cap).
        positive_page_ids:       GT positive page passage_ids (for labels, training/eval).
        positive_layout_ids:     GT positive layout_ids (for labels, training/eval).

    Returns:
        Graph dict with 'qid', 'nodes', 'edges', 'stats'.
    """
    q_node_id = _qid(qid)

    nodes: Dict[str, List[Dict]] = {
        "query":  [],
        "page":   [],
        "layout": [],
    }
    edges: Dict[str, List[Tuple[str, str]]] = {
        "query_to_page":        [],
        "page_to_query":        [],
        "query_to_layout":      [],
        "layout_to_query":      [],
        "page_contains_layout": [],
        "layout_in_page":       [],
        "layout_adjacent":      [],  # bidirectional spatial edges
    }

    nodes["query"].append({"node_id": q_node_id, "text": query_text})

    pos_page_set   = {str(p) for p in positive_page_ids}   if positive_page_ids   else set()
    pos_layout_set = {str(l) for l in positive_layout_ids} if positive_layout_ids else set()

    for page_row in retrieved_pages:
        doc_name   = str(page_row.get("doc_name", ""))
        passage_id = page_row.get("passage_id", "")
        passage_id_str = str(passage_id)  # normalize for comparisons

        if text_mode == "vlm_text":
            page_text = str(page_row.get("vlm_text", "") or "")
        else:
            page_text = str(page_row.get("ocr_text", "") or "")

        p_node_id = _page_nid(doc_name, passage_id)
        nodes["page"].append({
            "node_id":    p_node_id,
            "doc_name":   doc_name,
            "passage_id": passage_id,
            "text":       page_text,
            "is_positive": passage_id_str in pos_page_set,
        })
        edges["query_to_page"].append((q_node_id, p_node_id))
        edges["page_to_query"].append((p_node_id, q_node_id))

        # Layouts on this page — match on str(page_id) to handle dtype mismatch
        page_layouts = layouts_df[
            (layouts_df["doc_name"] == doc_name) &
            (layouts_df["page_id"].astype(str) == passage_id_str)
        ]
        if max_layouts_per_page is not None:
            page_layouts = page_layouts.iloc[:max_layouts_per_page]

        layout_nids:   List[str]                              = []
        layout_bboxes: List[Tuple[float, float, float, float]] = []
        page_h_list:   List[float]                            = []

        for _, lr in page_layouts.iterrows():
            layout_id   = lr.get("layout_id", "")
            layout_type = str(lr.get("type", "text"))

            if layout_type in ("table", "image"):
                if text_mode == "vlm_text":
                    ltext = str(lr.get("vlm_text", "") or "")
                else:
                    ltext = str(lr.get("ocr_text", "") or "")
            else:
                ltext = str(lr.get("text", "") or "")

            bbox = _bbox_to_xyxy(lr.get("bbox"))

            raw_ps = lr.get("page_size")
            try:
                ps = list(raw_ps)
                page_h = float(ps[1]) if len(ps) > 1 else 1.0
            except (TypeError, ValueError, IndexError):
                page_h = 1.0

            l_node_id = _layout_nid(doc_name, layout_id)
            nodes["layout"].append({
                "node_id":     l_node_id,
                "doc_name":    doc_name,
                "layout_id":   layout_id,
                "page_id":     passage_id,
                "layout_type": layout_type,
                "text":        ltext,
                "bbox":        list(bbox),
                "is_positive": str(layout_id) in pos_layout_set,
            })

            layout_nids.append(l_node_id)
            layout_bboxes.append(bbox)
            page_h_list.append(page_h)

            edges["page_contains_layout"].append((p_node_id, l_node_id))
            edges["layout_in_page"].append((l_node_id, p_node_id))
            edges["query_to_layout"].append((q_node_id, l_node_id))
            edges["layout_to_query"].append((l_node_id, q_node_id))

        # Spatial adjacency edges (bidirectional) — O(n^2) per page, fine for ~tens of layouts
        n = len(layout_nids)
        for i in range(n):
            for j in range(i + 1, n):
                if _are_adjacent(
                    layout_bboxes[i],
                    layout_bboxes[j],
                    proximity_frac=spatial_proximity_frac,
                    page_h=page_h_list[i],
                ):
                    edges["layout_adjacent"].append((layout_nids[i], layout_nids[j]))
                    edges["layout_adjacent"].append((layout_nids[j], layout_nids[i]))

    stats = {
        "n_pages":   len(nodes["page"]),
        "n_layouts": len(nodes["layout"]),
        "n_edges":   sum(len(v) for v in edges.values()),
    }

    return {
        "qid":   qid,
        "nodes": nodes,
        "edges": {k: [list(e) for e in v] for k, v in edges.items()},
        "stats": stats,
    }
