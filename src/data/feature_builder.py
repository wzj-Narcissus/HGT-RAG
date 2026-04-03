"""Build node features for a heterogeneous graph.

Core function: build_node_features(graph, encoder, image_dir)
Returns a dict: node_id -> torch.Tensor (float32, CPU)
"""
import os
import tempfile
from typing import Dict, List, Optional

import torch

from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _table_to_image(table_data: Dict, title: str = "", max_rows: Optional[int] = None) -> Optional[str]:
    """Convert table data to an image file, return the temp file path.

    Args:
        table_data: Dict with 'table_rows' key containing list of rows
        title: Table title to display
        max_rows: Maximum rows to render. None means render all rows.

    Returns:
        Path to temporary image file (caller should delete after use)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.error("PIL not available for table rendering")
        return None

    rows = table_data.get("table_rows", []) if isinstance(table_data, dict) else []
    if not rows:
        return None

    # Limit rows when requested
    if max_rows is not None:
        rows = rows[:max_rows]

    # Calculate cell dimensions
    cell_padding = 8
    cell_height = 30

    # Calculate column widths based on content
    num_cols = max(len(row) for row in rows) if rows else 0
    col_widths = [0] * num_cols

    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    # Calculate column widths
    for row in rows:
        for col_idx, cell in enumerate(row):
            if col_idx >= num_cols:
                break
            text = cell.get("text", "") if isinstance(cell, dict) else str(cell)
            bbox = font.getbbox(text) if hasattr(font, 'getbbox') else (0, 0, len(text) * 7, 14)
            text_width = bbox[2] - bbox[0] if bbox else len(text) * 7
            col_widths[col_idx] = max(col_widths[col_idx], text_width + 2 * cell_padding)

    # Ensure minimum width
    col_widths = [max(w, 60) for w in col_widths]

    # Calculate image dimensions
    table_width = sum(col_widths)
    table_height = len(rows) * cell_height + 40  # Extra for title

    # Create image with white background
    img = Image.new('RGB', (table_width + 20, table_height + 20), color='white')
    draw = ImageDraw.Draw(img)

    # Draw title
    y_offset = 10
    if title:
        draw.text((10, y_offset), title, fill='black', font=font)
        y_offset += 25

    # Draw table cells
    for row_idx, row in enumerate(rows):
        x_offset = 10
        for col_idx, cell in enumerate(row):
            if col_idx >= num_cols:
                break
            text = cell.get("text", "") if isinstance(cell, dict) else str(cell)

            # Draw cell border
            cell_x1 = x_offset
            cell_y1 = y_offset + row_idx * cell_height
            cell_x2 = x_offset + col_widths[col_idx]
            cell_y2 = cell_y1 + cell_height

            # Header row styling
            if row_idx == 0:
                draw.rectangle([cell_x1, cell_y1, cell_x2, cell_y2], fill='#E0E0E0', outline='black')
            else:
                draw.rectangle([cell_x1, cell_y1, cell_x2, cell_y2], outline='black')

            # Draw text
            text_x = cell_x1 + cell_padding
            text_y = cell_y1 + (cell_height - 14) // 2
            draw.text((text_x, text_y), text, fill='black', font=font)

            x_offset += col_widths[col_idx]

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(temp_file.name)
    temp_file.close()

    return temp_file.name


def build_node_features(
    graph: Dict,
    encoder: QwenVLFeatureEncoder,
    image_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Encode every node in the graph with the given encoder.

    For image nodes:
      - If the image file exists locally -> visual encoding
      - If not -> fall back to encoding image.title as text (much better than zero vector)

    For table nodes:
      - Convert the full table to an image -> visual encoding

    Args:
        graph:     graph dict from build_question_graph.
        encoder:   QwenVLFeatureEncoder instance.
        image_dir: directory where image files live (optional).

    Returns:
        features: dict of node_id -> 1-D float32 tensor.
    """
    nodes = graph.get("nodes", {})
    features: Dict[str, torch.Tensor] = {}
    temp_files: List[str] = []  # Track temp files for cleanup

    # ---- text-based nodes ------------------------------------------------

    def _encode_text_nodes(node_list, text_key="text"):
        if not node_list:
            return
        ids = [n["node_id"] for n in node_list]
        texts = [n.get(text_key, "") or "" for n in node_list]
        embs = encoder.encode_texts(texts)  # (N, D)
        for nid, emb in zip(ids, embs):
            features[nid] = emb.float()

    _encode_text_nodes(nodes.get("query", []))
    _encode_text_nodes(nodes.get("textblock", []))
    _encode_text_nodes(nodes.get("cell", []))
    _encode_text_nodes(nodes.get("caption", []))

    # ---- table nodes (convert to image then encode) ----------------------

    table_nodes = nodes.get("table", [])
    if table_nodes:
        table_image_paths = []
        table_node_ids = []

        for n in table_nodes:
            table_data = n.get("table_data", {})
            title = n.get("title", "")

            # Convert table to image
            img_path = _table_to_image(table_data, title)
            if img_path:
                table_image_paths.append(img_path)
                table_node_ids.append(n["node_id"])
                temp_files.append(img_path)
            else:
                # Fallback: encode title as text
                logger.warning(f"Failed to render table {n.get('table_id', '')} to image, using title fallback")
                emb = encoder.encode_texts([title] if title else ["[table]"])
                features[n["node_id"]] = emb[0].float()

        # Encode table images
        if table_image_paths:
            embs = encoder.encode_images(table_image_paths)
            for nid, emb in zip(table_node_ids, embs):
                features[nid] = emb.float()

    # ---- image nodes (visual encoding with text fallback) ----------------

    image_nodes = nodes.get("image", [])
    if image_nodes:
        visual_nodes = []   # (node, full_path) where file exists
        text_fallback_nodes = []  # (node, title_text) where file missing

        for n in image_nodes:
            raw_path = n.get("path", "")
            full_path = ""
            if image_dir and raw_path:
                full_path = os.path.join(image_dir, raw_path)

            if full_path and os.path.exists(full_path):
                visual_nodes.append((n, full_path))
            else:
                # Use image title as text fallback
                title = n.get("title", "") or ""
                if not title:
                    title = "[image]"
                text_fallback_nodes.append((n, title))

        # Encode visual images
        if visual_nodes:
            ids = [n["node_id"] for n, _ in visual_nodes]
            paths = [p for _, p in visual_nodes]
            embs = encoder.encode_images(paths)
            for nid, emb in zip(ids, embs):
                features[nid] = emb.float()

        # Encode missing images via title text
        if text_fallback_nodes:
            n_missing = len(text_fallback_nodes)
            logger.warning(
                f"[feature_builder] {n_missing} image(s) not found locally. "
                f"Falling back to title-text encoding instead of zero vector. "
                f"To use visual encoding, download images and set data.image_dir in config."
            )
            ids = [n["node_id"] for n, _ in text_fallback_nodes]
            titles = [t for _, t in text_fallback_nodes]
            embs = encoder.encode_texts(titles)
            for nid, emb in zip(ids, embs):
                features[nid] = emb.float()

    # Cleanup temp files
    for temp_path in temp_files:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return features
