"""Train the HGT evidence selection model.

Usage:
    python -m src.scripts.train --config configs/default.yaml
"""
import argparse
import gc
import hashlib
import os
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from typing import Optional

import torch
import yaml
from torch_geometric.loader import DataLoader

from src.data.qwen_vl_encoder import QwenVLFeatureEncoder
from src.data.feature_builder import _table_to_image
from src.data.hetero_converter import convert_to_heterodata
from src.models.hgt_model import HGTEvidenceModel
from src.trainers.losses import EvidenceSelectionLoss
from src.trainers.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

GRAPH_EDGE_SCHEMA_VERSION = "v5_table_image_with_cell_bidirectional"


# ---------------------------------------------------------------------------
# Phase 1: Parallel JSON loading (I/O bound → ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _load_one_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graphs_parallel(graphs_dir: str, split: str, n_workers: int = 16):
    pattern = os.path.join(graphs_dir, f"{split}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"No graph files found: {pattern}. Returning empty list.")
        return []
    logger.info(f"[Phase1] Loading {len(files)} {split} JSON files ({n_workers} threads)...")
    t0 = time.time()
    graphs = [None] * len(files)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        fut_to_idx = {ex.submit(_load_one_json, fp): i for i, fp in enumerate(files)}
        done = 0
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            graphs[idx] = fut.result()
            done += 1
            if done % 2000 == 0:
                logger.info(f"  {done}/{len(files)} JSONs loaded ...")
    stale_graphs = [
        graph.get("qid", f"index_{i}")
        for i, graph in enumerate(graphs)
        if graph.get("edge_schema_version") != GRAPH_EDGE_SCHEMA_VERSION
    ]
    if stale_graphs:
        preview = ", ".join(stale_graphs[:5])
        raise ValueError(
            "Detected stale graph JSONs without the restored edge schema "
            f"({GRAPH_EDGE_SCHEMA_VERSION}). Rebuild graphs first via "
            f"`python -m src.scripts.build_graphs --config configs/default.yaml --split {split}`. "
            f"Example qids: {preview}"
        )
    logger.info(f"[Phase1] Done in {time.time()-t0:.1f}s")
    return graphs


# ---------------------------------------------------------------------------
# Phase 2: Global feature cache (GPU batch encoding with deduplication)
# ---------------------------------------------------------------------------

def _collect_unique_nodes(graphs, image_dir):
    """Collect all unique text nodes and image-like nodes across all graphs."""
    text_nodes = {}   # node_id -> text  (query/textblock/cell/caption)
    image_nodes = {}  # node_id -> (path, title)
    table_nodes = {}  # node_id -> (table_data, title)

    for graph in graphs:
        nodes = graph.get("nodes", {})
        for ntype in ["query", "textblock", "cell", "caption"]:
            for n in nodes.get(ntype, []):
                nid = n["node_id"]
                if nid not in text_nodes:
                    text_nodes[nid] = n.get("text", "") or ""

        for n in nodes.get("image", []):
            nid = n["node_id"]
            if nid not in image_nodes:
                raw_path = n.get("path", "")
                full_path = os.path.join(image_dir, raw_path) if image_dir and raw_path else raw_path
                image_nodes[nid] = (full_path, n.get("title", "") or "")

        for n in nodes.get("table", []):
            nid = n["node_id"]
            if nid not in table_nodes:
                table_nodes[nid] = (n.get("table_data", {}), n.get("title", "") or "")

    return text_nodes, image_nodes, table_nodes


def _feature_dim_from_cache(cache) -> Optional[int]:
    if not cache:
        return None
    first = next(iter(cache.values()))
    if not isinstance(first, torch.Tensor):
        return None
    return int(first.shape[-1]) if first.ndim > 0 else 1


def _load_cache_if_compatible(cache_path: str, expected_dim: int, label: str):
    logger.info(f"[Phase2] {label} found, loading from {cache_path} ...")
    t0 = time.time()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_dim = _feature_dim_from_cache(cache)
    if cache_dim is not None and cache_dim != expected_dim:
        logger.warning(
            f"[Phase2] {label} dim mismatch at {cache_path}: "
            f"found {cache_dim}, expected {expected_dim}. Rebuilding."
        )
        return None
    dim_msg = f", dim={cache_dim}" if cache_dim is not None else ""
    logger.info(
        f"[Phase2] {label} loaded ({len(cache)} nodes{dim_msg}) in {time.time()-t0:.1f}s"
    )
    return cache


def build_or_load_feature_cache(
    graphs,
    encoder: QwenVLFeatureEncoder,
    image_dir: str,
    cache_path: str,
    text_batch_size: int = 32,
    image_batch_size: int = 8,
    projection_dim=None,
):
    """Build or incrementally update a global {node_id -> tensor} feature cache."""
    expected_cache_dim = (
        projection_dim if projection_dim and projection_dim < encoder.hidden_dim else encoder.hidden_dim
    )
    ckpt_path = cache_path + ".ckpt"
    use_projection = bool(projection_dim and projection_dim < encoder.hidden_dim)

    def _build_projection_matrix():
        proj_generator = torch.Generator(device="cpu")
        proj_generator.manual_seed(42)
        return torch.randn(
            encoder.hidden_dim,
            projection_dim,
            generator=proj_generator,
        ) / (encoder.hidden_dim ** 0.5)

    logger.info("[Phase2] Building feature cache (dedup + GPU batch encoding)...")
    t0 = time.time()

    save_every = 1000  # save checkpoint every N batches

    text_nodes, image_nodes, table_nodes = _collect_unique_nodes(graphs, image_dir)
    logger.info(
        f"[Phase2] Unique nodes: {len(text_nodes)} text, {len(image_nodes)} image, {len(table_nodes)} table"
    )

    cache = None
    if os.path.exists(cache_path):
        cache = _load_cache_if_compatible(cache_path, expected_cache_dim, "Feature cache")
        if cache is not None:
            logger.info("[Phase2] Reusing existing feature cache and filling missing nodes only.")
    if cache is None and os.path.exists(ckpt_path):
        cache = _load_cache_if_compatible(
            ckpt_path, expected_cache_dim, "Feature cache checkpoint"
        )
        if cache is None:
            os.remove(ckpt_path)
            logger.info(f"[Phase2] Removed stale checkpoint: {ckpt_path}")
        else:
            logger.info("[Phase2] Resuming from feature cache checkpoint and filling missing nodes only.")
    if cache is None:
        cache = {}

    proj_matrix = _build_projection_matrix() if use_projection else None

    # ---- Encode text nodes in large batches --------------------------------
    text_nids = [nid for nid in text_nodes.keys() if nid not in cache]
    texts = [text_nodes[nid] for nid in text_nids]
    n_text = len(texts)
    logger.info(f"[Phase2] Encoding {n_text} text nodes (batch={text_batch_size}, {len(text_nodes)-n_text} already cached)...")
    t1 = time.time()
    for i in range(0, n_text, text_batch_size):
        batch_ids = text_nids[i: i + text_batch_size]
        batch_texts = texts[i: i + text_batch_size]
        embs = encoder.encode_texts(batch_texts)  # (B, D) CPU float32
        if use_projection:
            embs = torch.nn.functional.normalize(embs @ proj_matrix, dim=-1)
        for nid, emb in zip(batch_ids, embs):
            cache[nid] = emb
        batch_num = i // text_batch_size + 1
        if batch_num % save_every == 0:
            torch.save(cache, ckpt_path)
            logger.info(f"  [ckpt] saved {len(cache)} nodes at batch {batch_num}")
        if batch_num % 500 == 0:
            pct = min(i + text_batch_size, n_text) / n_text * 100
            elapsed = time.time() - t1
            eta = elapsed / max(i + text_batch_size, 1) * (n_text - i - text_batch_size)
            logger.info(f"  text {pct:.1f}% | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")
    logger.info(f"[Phase2] Text encoding done in {time.time()-t1:.1f}s")

    # ---- Encode image nodes (visual or title-text fallback) ----------------
    img_nids = list(image_nodes.keys())
    n_img = len(img_nids)
    logger.info(f"[Phase2] Encoding {n_img} image nodes (batch={image_batch_size})...")
    t2 = time.time()

    # Split into visual (file exists) and text-fallback, skip already cached
    visual_ids, visual_paths = [], []
    fallback_ids, fallback_texts = [], []
    for nid in img_nids:
        if nid in cache:
            continue
        path, title = image_nodes[nid]
        if path and os.path.exists(path):
            visual_ids.append(nid)
            visual_paths.append(path)
        else:
            fallback_ids.append(nid)
            fallback_texts.append(title if title else "[image]")

    if visual_ids:
        for i in range(0, len(visual_ids), image_batch_size):
            batch_ids = visual_ids[i: i + image_batch_size]
            batch_paths = visual_paths[i: i + image_batch_size]
            embs = encoder.encode_images(batch_paths)
            if use_projection:
                embs = torch.nn.functional.normalize(embs @ proj_matrix, dim=-1)
            for nid, emb in zip(batch_ids, embs):
                cache[nid] = emb
            if (i // image_batch_size + 1) % save_every == 0:
                torch.save(cache, ckpt_path)

    if fallback_ids:
        logger.warning(
            f"[Phase2] {len(fallback_ids)} images missing locally, "
            f"using title-text fallback encoding."
        )
        for i in range(0, len(fallback_ids), text_batch_size):
            batch_ids = fallback_ids[i: i + text_batch_size]
            batch_texts = fallback_texts[i: i + text_batch_size]
            embs = encoder.encode_texts(batch_texts)
            if use_projection:
                embs = torch.nn.functional.normalize(embs @ proj_matrix, dim=-1)
            for nid, emb in zip(batch_ids, embs):
                cache[nid] = emb
            if (i // text_batch_size + 1) % save_every == 0:
                torch.save(cache, ckpt_path)

    logger.info(f"[Phase2] Image encoding done in {time.time()-t2:.1f}s")

    # ---- Encode table nodes (render to image then encode) ------------------
    table_nids = [nid for nid in table_nodes.keys() if nid not in cache]
    logger.info(f"[Phase2] Encoding {len(table_nids)} table nodes as images (batch={image_batch_size}, {len(table_nodes)-len(table_nids)} already cached)...")
    t3 = time.time()
    temp_table_files = []
    try:
        table_image_paths = []
        table_ids_for_paths = []
        table_fallback_ids = []
        table_fallback_titles = []
        for nid in table_nids:
            table_data, title = table_nodes[nid]
            img_path = _table_to_image(table_data, title, max_rows=None)
            if img_path:
                table_image_paths.append(img_path)
                table_ids_for_paths.append(nid)
                temp_table_files.append(img_path)
            else:
                table_fallback_ids.append(nid)
                table_fallback_titles.append(title if title else "[table]")
        if table_image_paths:
            for i in range(0, len(table_image_paths), image_batch_size):
                batch_ids = table_ids_for_paths[i: i + image_batch_size]
                batch_paths = table_image_paths[i: i + image_batch_size]
                embs = encoder.encode_images(batch_paths)
                if use_projection:
                    embs = torch.nn.functional.normalize(embs @ proj_matrix, dim=-1)
                for nid, emb in zip(batch_ids, embs):
                    cache[nid] = emb
                if (i // image_batch_size + 1) % save_every == 0:
                    torch.save(cache, ckpt_path)
        if table_fallback_ids:
            for i in range(0, len(table_fallback_ids), text_batch_size):
                batch_ids = table_fallback_ids[i: i + text_batch_size]
                batch_texts = table_fallback_titles[i: i + text_batch_size]
                embs = encoder.encode_texts(batch_texts)
                if use_projection:
                    embs = torch.nn.functional.normalize(embs @ proj_matrix, dim=-1)
                for nid, emb in zip(batch_ids, embs):
                    cache[nid] = emb
                if (i // text_batch_size + 1) % save_every == 0:
                    torch.save(cache, ckpt_path)
    finally:
        for p in temp_table_files:
            try:
                os.unlink(p)
            except Exception:
                pass
    logger.info(f"[Phase2] Table encoding done in {time.time()-t3:.1f}s")

    # ---- Save cache to disk (and clean up checkpoint) ----------------------
    if os.path.dirname(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info(f"[Phase2] Saving {len(cache)} node features to {cache_path} ...")
    torch.save(cache, cache_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info(f"[Phase2] Checkpoint removed: {ckpt_path}")
    logger.info(f"[Phase2] Feature cache built in {time.time()-t0:.1f}s total")
    return cache


# ---------------------------------------------------------------------------
# Phase 3: Build HeteroData objects (CPU, parallel)
# ---------------------------------------------------------------------------

def _build_one_heterodata(graph):
    """Worker fn: build HeteroData from graph + pre-built features (passed via closure)."""
    labels = graph.get("labels", {})
    # features are looked up from global cache injected by initializer
    features = {
        nid: _GLOBAL_CACHE[nid]
        for nodes in graph["nodes"].values()
        for n in nodes
        for nid in [n["node_id"]]
        if nid in _GLOBAL_CACHE
    }
    return convert_to_heterodata(graph, features, labels, graph_cfg=_GLOBAL_GRAPH_CFG)


# Module-level cache for multiprocessing workers (set via initializer)
_GLOBAL_CACHE = {}
_GLOBAL_GRAPH_CFG = {}


def _worker_init(cache, graph_cfg):
    global _GLOBAL_CACHE, _GLOBAL_GRAPH_CFG
    _GLOBAL_CACHE = cache
    _GLOBAL_GRAPH_CFG = graph_cfg


def build_heterodata_parallel(graphs, cache, graph_cfg, n_workers: int = 8):
    """Convert all graphs to HeteroData in single-process mode with GPU acceleration.

    Uses single-process mode to enable GPU-accelerated similarity computation in
    convert_to_heterodata, avoiding multiprocessing + CUDA conflicts.

    Args:
        graphs: List of graph dicts
        cache: Feature cache dict
        graph_cfg: Graph configuration
        n_workers: Unused, kept for API compatibility
    """
    cuda_available = torch.cuda.is_available()
    mode_desc = "single process with GPU" if cuda_available else "single process (CPU)"
    logger.info(
        f"[Phase3] Building {len(graphs)} HeteroData objects ({mode_desc})..."
    )
    t0 = time.time()

    dataset = []
    for i, graph in enumerate(graphs):
        labels = graph.get("labels", {})
        features = {
            nid: cache[nid]
            for nodes in graph["nodes"].values()
            for n in nodes
            for nid in [n["node_id"]]
            if nid in cache
        }
        data = convert_to_heterodata(graph, features, labels, graph_cfg=graph_cfg)
        dataset.append(data)

        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(graphs)} graphs processed...")

    logger.info(f"[Phase3] Done in {time.time()-t0:.1f}s")
    return dataset


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_hetero_dataset(
    graphs_dir: str,
    split: str,
    encoder: QwenVLFeatureEncoder,
    image_dir: str,
    cfg: dict,
    cache_dir: str = "outputs/features",
    n_io_workers: int = 16,
    n_build_workers: int = 8,
    chunk_size: int = None,
):
    """Three-phase pipeline:
      1. Load graph JSONs in parallel (I/O bound, ThreadPool)
      2. Build/load global feature cache (GPU batch encoding, dedup)
      3. Convert to HeteroData in parallel (CPU, ProcessPool) with optional chunking
    """
    enc_cfg = cfg.get("encoder", {})
    text_batch_size = enc_cfg.get("batch_size", 32)
    image_batch_size = enc_cfg.get("image_batch_size", 8)
    output_cfg = cfg.get("output", {})
    graph_cfg = cfg.get("graph", {})
    graph_cache_key = {
        "candidate_mode": graph_cfg.get("candidate_mode", "oracle"),
        "max_cells_per_table": graph_cfg.get("max_cells_per_table", 200),
        "edge_schema_version": GRAPH_EDGE_SCHEMA_VERSION,
    }
    graph_cache_tag = hashlib.sha1(
        json.dumps(graph_cache_key, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    hetero_dir = output_cfg.get("hetero_dir", "outputs/hetero")
    proj_dim = enc_cfg.get("projection_dim", None)
    effective_feature_dim = (
        proj_dim if proj_dim and proj_dim < encoder.hidden_dim else encoder.hidden_dim
    )
    encoder_tag = enc_cfg.get("name", "encoder").replace("/", "_")
    cache_path = os.path.join(
        cache_dir,
        f"{split}_{encoder_tag}_h{encoder.hidden_dim}_d{effective_feature_dim}_features.pt",
    )
    split_hetero_dir = os.path.join(
        hetero_dir,
        f"{split}_{encoder_tag}_h{encoder.hidden_dim}_d{effective_feature_dim}_{graph_cache_tag}",
    )

    # Lazy list that loads from disk on demand
    class LazyHeteroDataList:
        """Sequence that loads HeteroData objects from disk on-demand.

        Explicitly implements __iter__ so that `for x in dataset` works
        correctly. Inheriting from list and leaving the underlying list empty
        caused iteration to silently yield zero items while __len__ returned
        the correct count.
        """
        def __init__(self, file_paths):
            self.file_paths = list(file_paths)

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            path = self.file_paths[idx]
            return torch.load(path, map_location='cpu', weights_only=False)

        def __iter__(self):
            for i in range(len(self.file_paths)):
                yield self[i]

        def clear_cache(self):
            return None

    t_total = time.time()

    # Phase 1
    graphs = load_graphs_parallel(graphs_dir, split, n_workers=n_io_workers)

    # Early return if no graphs
    if not graphs:
        logger.warning(f"No graphs found for split '{split}'. Returning empty dataset.")
        if chunk_size is not None:
            # Return empty LazyHeteroDataList
            os.makedirs(split_hetero_dir, exist_ok=True)
            return LazyHeteroDataList([])
        else:
            return []

    # Phase 2
    cache = build_or_load_feature_cache(
        graphs, encoder, image_dir, cache_path,
        text_batch_size=text_batch_size,
        image_batch_size=image_batch_size,
        projection_dim=proj_dim,
    )

    # Phase 3: Convert to HeteroData, with optional chunking
    if chunk_size is None:
        # Original behavior: build all in memory
        dataset = build_heterodata_parallel(graphs, cache, graph_cfg, n_workers=n_build_workers)
        logger.info(
            f"[Load] {split} dataset ready: {len(dataset)} graphs | "
            f"total {time.time()-t_total:.1f}s"
        )
        return dataset

    # Chunked mode: process in chunks, save to disk, return lazy list
    logger.info(f"[Phase3] Processing {len(graphs)} graphs in chunks of {chunk_size}")

    # Create directory for HeteroData files
    os.makedirs(split_hetero_dir, exist_ok=True)

    file_paths = []

    # Process chunks
    n_chunks = (len(graphs) + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(graphs))
        chunk_graphs = graphs[start:end]

        # Check how many in this chunk already exist on disk
        chunk_file_paths = []
        pending_graphs, pending_indices = [], []
        for i, graph in enumerate(chunk_graphs):
            idx = start + i
            qid = graph.get('qid', f'{split}_{idx:06d}')
            fp = os.path.join(split_hetero_dir, f'{qid}.pt')
            chunk_file_paths.append(fp)
            if not os.path.exists(fp):
                pending_graphs.append(graph)
                pending_indices.append(i)

        if not pending_graphs:
            logger.info(f"[Phase3] Chunk {chunk_idx+1}/{n_chunks} already done, skipping.")
            file_paths.extend(chunk_file_paths)
            continue

        logger.info(f"[Phase3] Chunk {chunk_idx+1}/{n_chunks} ({len(pending_graphs)}/{len(chunk_graphs)} graphs to build)")

        # Build HeteroData for pending graphs only
        chunk_data = build_heterodata_parallel(
            pending_graphs, cache, graph_cfg, n_workers=n_build_workers
        )

        # Save each HeteroData object to disk
        for i, hetero_data in enumerate(chunk_data):
            file_path = chunk_file_paths[pending_indices[i]]
            torch.save(hetero_data, file_path)

        file_paths.extend(chunk_file_paths)

        # Clear chunk data to free memory
        del chunk_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        f"[Load] {split} dataset ready: {len(file_paths)} graphs saved to {split_hetero_dir} | "
        f"total {time.time()-t_total:.1f}s"
    )

    return LazyHeteroDataList(file_paths)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--io_workers", type=int, default=16, help="JSON loading threads")
    parser.add_argument("--build_workers", type=int, default=8, help="HeteroData build processes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("training", {}).get("seed", 42))

    data_cfg = cfg["data"]
    enc_cfg = cfg["encoder"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    chunk_size = train_cfg.get("chunk_size")
    output_cfg = cfg["output"]

    device = enc_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    logger.info(f"Using device: {device}")

    graphs_dir = output_cfg["graphs_dir"]
    image_dir = data_cfg.get("image_dir", "dataset/images")
    features_dir = output_cfg.get("features_dir", "outputs/features")

    # Build encoder on GPU
    enc_cfg_copy = dict(enc_cfg)
    enc_cfg_copy["device"] = device
    encoder = QwenVLFeatureEncoder.from_config(enc_cfg_copy)
    proj_dim = enc_cfg.get("projection_dim", None)
    hidden_dim_in = proj_dim if proj_dim and proj_dim < encoder.hidden_dim else encoder.hidden_dim

    # Load datasets (3-phase pipeline)
    n_io = min(args.io_workers, cpu_count())
    n_build = min(args.build_workers, cpu_count())

    train_dataset = load_hetero_dataset(
        graphs_dir, "train", encoder, image_dir, cfg,
        cache_dir=features_dir,
        n_io_workers=n_io,
        n_build_workers=n_build,
        chunk_size=chunk_size,
    )
    dev_graphs_pattern = os.path.join(graphs_dir, "dev_*.json")
    if glob.glob(dev_graphs_pattern):
        dev_dataset = load_hetero_dataset(
            graphs_dir, "dev", encoder, image_dir, cfg,
            cache_dir=features_dir,
            n_io_workers=n_io,
            n_build_workers=n_build,
            chunk_size=chunk_size,
        )
    else:
        logger.warning(f"No dev graph files found at {dev_graphs_pattern}, skipping dev dataset.")
        dev_dataset = []

    # Free encoder memory before training
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
        # Enable cuDNN auto-tuner for better performance with fixed-size inputs
        torch.backends.cudnn.benchmark = True

    # Create DataLoaders for efficient batching and parallel data loading
    batch_size = train_cfg.get("batch_size", 1)
    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = train_cfg.get("pin_memory", False) and device == "cuda"
    prefetch_factor = train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None

    logger.info(f"Creating DataLoaders: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, prefetch_factor={prefetch_factor}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )

    if dev_dataset:
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor,
        )
    else:
        dev_loader = None

    # Build model
    model = HGTEvidenceModel(
        in_dim=hidden_dim_in,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        scoring_head=model_cfg.get("scoring_head", "mlp"),
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = EvidenceSelectionLoss(
        lambda_text=train_cfg.get("lambda_text", 1.0),
        lambda_cell=train_cfg.get("lambda_cell", 1.0),
        lambda_image=train_cfg.get("lambda_image", 1.0),
        lambda_caption=train_cfg.get("lambda_caption", 0.5),
        lambda_table=train_cfg.get("lambda_table", 0.5),
        lambda_rank=train_cfg.get("lambda_rank", 0.5),
        margin=train_cfg.get("margin", 1.0),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha"),
    )

    trainer = Trainer(model, loss_fn, cfg, device=device)

    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    trainer.run(train_loader, dev_loader, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
