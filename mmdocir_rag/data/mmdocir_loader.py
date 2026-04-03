"""MMDocIR dataset loader.

Loads parquet files and JSONL annotations for the MMDocIR benchmark.
"""
import hashlib
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
import io

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MMDocIRLoader:
    """Loader for MMDocIR evaluation and training datasets.

    Evaluation set: 313 documents, ~65 pages each, 1,685 questions
    Training set: 6,878 documents, ~32 pages each, 173,843 questions
    """

    def __init__(
        self,
        pages_parquet_path: str,
        layouts_parquet_path: str,
        annotations_jsonl_path: Optional[str] = None,
        text_mode: str = "ocr_text",  # "ocr_text" or "vlm_text"
    ):
        """Initialize MMDocIR loader.

        Args:
            pages_parquet_path: Path to MMDocIR_pages.parquet
            layouts_parquet_path: Path to MMDocIR_layouts.parquet
            annotations_jsonl_path: Path to MMDocIR_annotations.jsonl (eval set)
            text_mode: Which text field to use ("ocr_text" or "vlm_text")
        """
        self.text_mode = text_mode
        logger.info(f"Loading MMDocIR dataset with text_mode='{text_mode}'...")

        # Load pages
        logger.info(f"Loading pages from {pages_parquet_path}...")
        self.pages_df = pd.read_parquet(pages_parquet_path)
        logger.info(f"Loaded {len(self.pages_df)} pages")

        # Load layouts
        logger.info(f"Loading layouts from {layouts_parquet_path}...")
        self.layouts_df = pd.read_parquet(layouts_parquet_path)
        logger.info(f"Loaded {len(self.layouts_df)} layouts")

        # Load annotations (eval set only)
        self.annotations = []
        if annotations_jsonl_path:
            logger.info(f"Loading annotations from {annotations_jsonl_path}...")
            with open(annotations_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.annotations.append(json.loads(line.strip()))
            logger.info(f"Loaded {len(self.annotations)} document annotations")

        # Build doc_name -> page indices mapping
        self._build_doc_index()

    def _build_doc_index(self):
        """Build index for fast document lookup."""
        # Pages: group by doc_name, track start/end row indices
        self.doc_pages_index: Dict[str, Tuple[int, int]] = {}
        doc_page_groups = self.pages_df.groupby('doc_name', sort=False)
        for doc_name, group in doc_page_groups:
            indices = group.index.tolist()
            self.doc_pages_index[doc_name] = (min(indices), max(indices) + 1)

        # Layouts: group by doc_name, track start/end row indices
        self.doc_layouts_index: Dict[str, Tuple[int, int]] = {}
        doc_layout_groups = self.layouts_df.groupby('doc_name', sort=False)
        for doc_name, group in doc_layout_groups:
            indices = group.index.tolist()
            self.doc_layouts_index[doc_name] = (min(indices), max(indices) + 1)

        logger.info(f"Indexed {len(self.doc_pages_index)} documents")

    def get_document_pages(self, doc_name: str) -> pd.DataFrame:
        """Get all pages for a document."""
        if doc_name not in self.doc_pages_index:
            raise ValueError(f"Document {doc_name} not found")
        start, end = self.doc_pages_index[doc_name]
        return self.pages_df.iloc[start:end]

    def get_document_layouts(self, doc_name: str) -> pd.DataFrame:
        """Get all layouts for a document."""
        if doc_name not in self.doc_layouts_index:
            raise ValueError(f"Document {doc_name} not found")
        start, end = self.doc_layouts_index[doc_name]
        return self.layouts_df.iloc[start:end]

    def get_page_text(self, page_row: pd.Series) -> str:
        """Get text for a page based on text_mode."""
        if self.text_mode == "vlm_text":
            return page_row.get("vlm_text", "") or ""
        return page_row.get("ocr_text", "") or ""

    def get_layout_text(self, layout_row: pd.Series) -> str:
        """Get text for a layout based on type and text_mode."""
        layout_type = layout_row.get("type", "")

        if layout_type in ["table", "image"]:
            # For tables and images, use ocr_text or vlm_text
            if self.text_mode == "vlm_text":
                return layout_row.get("vlm_text", "") or ""
            return layout_row.get("ocr_text", "") or ""
        else:
            # For text, title, equation: use text field
            return layout_row.get("text", "") or ""

    def get_page_image(self, page_row: pd.Series) -> Optional[Image.Image]:
        """Get page image from binary data."""
        img_binary = page_row.get("image_binary")
        if img_binary is None:
            return None
        try:
            return Image.open(io.BytesIO(img_binary))
        except Exception as e:
            logger.warning(f"Failed to load page image: {e}")
            return None

    def get_layout_image(self, layout_row: pd.Series) -> Optional[Image.Image]:
        """Get layout image from binary data."""
        img_binary = layout_row.get("image_binary")
        if img_binary is None:
            return None
        try:
            return Image.open(io.BytesIO(img_binary))
        except Exception as e:
            logger.warning(f"Failed to load layout image: {e}")
            return None

    def iter_questions(self):
        """Iterate over all questions in evaluation set.

        Yields tuples of (question_dict, doc_annotation).
        """
        for doc_anno in self.annotations:
            # Annotation doc_name may include ".pdf" extension; parquet drops it.
            doc_name = doc_anno["doc_name"].removesuffix(".pdf")
            page_start, page_end = doc_anno["page_indices"]
            layout_start, layout_end = doc_anno["layout_indices"]

            for qa_idx, qa in enumerate(doc_anno.get("questions", [])):
                q_text = qa.get("Q", "")
                q_hash = hashlib.md5(f"{doc_name}::{qa_idx}::{q_text}".encode("utf-8")).hexdigest()[:12]
                yield {
                    "qid": f"{doc_name}::{qa_idx}::{q_hash}",
                    "question": q_text,
                    "answer": qa.get("A", ""),
                    "type": qa.get("type", ""),
                    "page_id": qa.get("page_id", []),
                    "layout_mapping": qa.get("layout_mapping", []),
                    "doc_name": doc_name,
                    "page_indices": (page_start, page_end),
                    "layout_indices": (layout_start, layout_end),
                }

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        layout_type_counts = self.layouts_df["type"].value_counts().to_dict()

        return {
            "n_documents": len(self.doc_pages_index),
            "n_pages": len(self.pages_df),
            "n_layouts": len(self.layouts_df),
            "layout_types": layout_type_counts,
            "n_questions": sum(
                len(doc.get("questions", [])) for doc in self.annotations
            ) if self.annotations else 0,
            "text_mode": self.text_mode,
        }


def load_mmdocir_eval(
    dataset_dir: str = "mmdocir_data",
    text_mode: str = "ocr_text",
) -> MMDocIRLoader:
    """Convenience function to load MMDocIR evaluation dataset.

    Args:
        dataset_dir: Directory containing MMDocIR data files
        text_mode: "ocr_text" or "vlm_text"

    Returns:
        MMDocIRLoader instance
    """
    import os
    return MMDocIRLoader(
        pages_parquet_path=os.path.join(dataset_dir, "MMDocIR_pages.parquet"),
        layouts_parquet_path=os.path.join(dataset_dir, "MMDocIR_layouts.parquet"),
        annotations_jsonl_path=os.path.join(dataset_dir, "MMDocIR_annotations.jsonl"),
        text_mode=text_mode,
    )
