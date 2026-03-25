"""MMQA dataset loader.

Exposes the four core load functions required by the pipeline:
  - load_mmqa_questions(...)
  - load_mmqa_texts(...)
  - load_mmqa_tables(...)
  - load_mmqa_images(...)
"""
import os
from typing import Dict, List, Optional
from src.data.io import read_jsonl, resolve_path, index_by_id
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_mmqa_questions(
    dataset_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    filename: Optional[str] = None,
) -> List[Dict]:
    """Load MMQA question records for a given split.

    Args:
        dataset_dir: directory containing the .jsonl.gz files.
        split: one of train / dev / test.
        max_samples: truncate to this many samples (useful for debugging).
        filename: override filename (auto-detected from split if None).

    Returns:
        List of question dicts.
    """
    if filename is None:
        filename = f"MMQA_{split}.jsonl.gz"
    path = resolve_path(dataset_dir, filename)
    records = read_jsonl(path)
    if max_samples is not None:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records)} questions from {path} (split={split})")
    return records


def load_mmqa_texts(
    dataset_dir: str,
    filename: Optional[str] = None,
) -> Dict[str, Dict]:
    """Load MMQA text documents, indexed by id."""
    if filename is None:
        filename = "MMQA_texts.jsonl.gz"
    path = resolve_path(dataset_dir, filename)
    records = read_jsonl(path)
    idx = index_by_id(records, id_field="id")
    logger.info(f"Loaded {len(idx)} text documents from {path}")
    return idx


def load_mmqa_tables(
    dataset_dir: str,
    filename: Optional[str] = None,
) -> Dict[str, Dict]:
    """Load MMQA table documents, indexed by id."""
    if filename is None:
        filename = "MMQA_tables.jsonl.gz"
    path = resolve_path(dataset_dir, filename)
    records = read_jsonl(path)
    idx = index_by_id(records, id_field="id")
    logger.info(f"Loaded {len(idx)} table documents from {path}")
    return idx


def load_mmqa_images(
    dataset_dir: str,
    filename: Optional[str] = None,
) -> Dict[str, Dict]:
    """Load MMQA image documents, indexed by id."""
    if filename is None:
        filename = "MMQA_images.jsonl.gz"
    path = resolve_path(dataset_dir, filename)
    records = read_jsonl(path)
    idx = index_by_id(records, id_field="id")
    logger.info(f"Loaded {len(idx)} image documents from {path}")
    return idx


def get_oracle_candidates(question: Dict) -> Dict:
    """Extract oracle candidate ids from question metadata.

    Returns a dict with keys: text_doc_ids, table_id, image_doc_ids.
    """
    meta = question.get("metadata", {})
    return {
        "text_doc_ids": meta.get("text_doc_ids", []),
        "table_id": meta.get("table_id", None),
        "image_doc_ids": meta.get("image_doc_ids", []),
    }
