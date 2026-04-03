"""Stage 1: Page-Level Retrieval using Embedding-Only (4096-dim).

This module handles:
1. Encoding all pages in the corpus using Qwen3-VL-Embedding-8B
2. Building FAISS index for fast retrieval
3. Top-K page retrieval given a query
"""
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PageRetriever:
    """Page-level retriever using embedding similarity.

    Uses Qwen3-VL-Embedding-8B (4096-dim) for both query and page encoding.
    Supports FAISS for fast approximate nearest neighbor search.
    """

    def __init__(
        self,
        encoder,
        index_path: Optional[str] = None,
        use_faiss: bool = True,
        faiss_nlist: int = 100,
    ):
        """Initialize page retriever.

        Args:
            encoder: QwenVLFeatureEncoder instance
            index_path: Path to load/save pre-built index
            use_faiss: Whether to use FAISS for indexing (fallback to numpy if unavailable)
            faiss_nlist: Number of clusters for FAISS IVF index
        """
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_dim  # Should be 4096
        self.index_path = index_path
        self.use_faiss = use_faiss
        self.faiss_nlist = faiss_nlist

        # Storage
        self.page_embeddings: Optional[np.ndarray] = None  # (N, 4096)
        self.page_metadata: List[Dict] = []  # [{doc_name, page_id, row_idx}, ...]
        self._faiss_index = None

        # Try to load FAISS
        self._faiss_available = False
        if use_faiss:
            try:
                import faiss
                self._faiss = faiss
                self._faiss_available = True
                logger.info("FAISS loaded successfully")
            except ImportError:
                logger.warning("FAISS not available, falling back to numpy brute-force search")

    def encode_pages(
        self,
        pages_df,
        text_mode: str = "ocr_text",
        batch_size: int = 32,
        cache_path: Optional[str] = None,
    ) -> np.ndarray:
        """Encode all pages using the encoder.

        Args:
            pages_df: DataFrame with pages (from MMDocIRLoader.pages_df)
            text_mode: "ocr_text" or "vlm_text"
            batch_size: Batch size for encoding
            cache_path: Path to cache encoded embeddings

        Returns:
            Array of page embeddings (N, hidden_dim)
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached page embeddings from {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            self.page_embeddings = cache["embeddings"]
            self.page_metadata = cache["metadata"]
            logger.info(f"Loaded {len(self.page_embeddings)} cached page embeddings")
            return self.page_embeddings

        logger.info(f"Encoding {len(pages_df)} pages with text_mode='{text_mode}'...")

        texts = []
        self.page_metadata = []

        for idx, row in pages_df.iterrows():
            # Get text based on mode
            if text_mode == "vlm_text":
                text = row.get("vlm_text", "") or ""
            else:
                text = row.get("ocr_text", "") or ""

            texts.append(text)
            self.page_metadata.append({
                "row_idx": idx,
                "doc_name": row.get("doc_name", ""),
                "passage_id": row.get("passage_id", ""),
                "domain": row.get("domain", ""),
            })

        # Encode in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding pages"):
            batch_texts = texts[i:i + batch_size]
            embs = self.encoder.encode_texts(batch_texts)  # (B, hidden_dim)
            all_embeddings.append(embs.cpu().numpy())

        self.page_embeddings = np.vstack(all_embeddings)

        # Normalize embeddings
        self.page_embeddings = F.normalize(
            torch.from_numpy(self.page_embeddings), dim=-1
        ).numpy()

        logger.info(f"Encoded {len(self.page_embeddings)} pages, dim={self.hidden_dim}")

        # Cache if requested
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save({
                "embeddings": self.page_embeddings,
                "metadata": self.page_metadata,
                "text_mode": text_mode,
            }, cache_path)
            logger.info(f"Saved page embeddings to {cache_path}")

        return self.page_embeddings

    def build_index(self):
        """Build search index from encoded pages."""
        if self.page_embeddings is None:
            raise ValueError("Must call encode_pages() before build_index()")

        n_pages = len(self.page_embeddings)
        logger.info(f"Building index for {n_pages} pages...")

        if self._faiss_available and self.use_faiss:
            # Use FAISS IVF index for large datasets
            dim = self.hidden_dim

            if n_pages < 1000:
                # Small dataset: use Flat index
                logger.info("Using FAISS Flat index (small dataset)")
                self._faiss_index = self._faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
                self._faiss_index.add(self.page_embeddings.astype(np.float32))
            else:
                # Large dataset: use IVF index
                nlist = min(self.faiss_nlist, n_pages // 10)
                nlist = max(nlist, 1)
                logger.info(f"Using FAISS IVF index with nlist={nlist}")

                quantizer = self._faiss.IndexFlatIP(dim)
                self._faiss_index = self._faiss.IndexIVFFlat(quantizer, dim, nlist)
                self._faiss_index.train(self.page_embeddings.astype(np.float32))
                self._faiss_index.add(self.page_embeddings.astype(np.float32))
        else:
            # Fallback: numpy brute force
            logger.info("Using numpy brute-force search (no FAISS)")
            self._faiss_index = None

        logger.info("Index built successfully")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float, Dict]]:
        """Retrieve top-K pages for a query.

        Args:
            query: Query string
            top_k: Number of pages to retrieve

        Returns:
            List of (row_idx, score, metadata) tuples
        """
        # Encode query
        query_emb = self.encoder.encode_texts([query])  # (1, hidden_dim)
        query_emb = F.normalize(query_emb, dim=-1).cpu().numpy()

        if self._faiss_available and self._faiss_index is not None:
            # FAISS search
            scores, indices = self._faiss_index.search(query_emb.astype(np.float32), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.page_metadata):
                    results.append((idx, float(score), self.page_metadata[idx]))
            return results
        else:
            # Numpy brute force
            similarities = np.dot(self.page_embeddings, query_emb.T).squeeze()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append((idx, float(similarities[idx]), self.page_metadata[idx]))
            return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[Tuple[int, float, Dict]]]:
        """Retrieve top-K pages for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of pages to retrieve per query

        Returns:
            List of results per query
        """
        # Encode all queries
        query_embs = self.encoder.encode_texts(queries)  # (B, hidden_dim)
        query_embs = F.normalize(query_embs, dim=-1).cpu().numpy()

        results = []
        for query_emb in query_embs:
            query_emb = query_emb.reshape(1, -1)

            if self._faiss_available and self._faiss_index is not None:
                scores, indices = self._faiss_index.search(query_emb.astype(np.float32), top_k)
                query_results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.page_metadata):
                        query_results.append((idx, float(score), self.page_metadata[idx]))
                results.append(query_results)
            else:
                similarities = np.dot(self.page_embeddings, query_emb.T).squeeze()
                top_indices = np.argsort(similarities)[::-1][:top_k]
                query_results = []
                for idx in top_indices:
                    query_results.append((idx, float(similarities[idx]), self.page_metadata[idx]))
                results.append(query_results)

        return results

    def save(self, path: str):
        """Save retriever state."""
        state = {
            "page_embeddings": self.page_embeddings,
            "page_metadata": self.page_metadata,
            "hidden_dim": self.hidden_dim,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved retriever state to {path}")

    def load(self, path: str):
        """Load retriever state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.page_embeddings = state["page_embeddings"]
        self.page_metadata = state["page_metadata"]
        self.hidden_dim = state["hidden_dim"]
        logger.info(f"Loaded retriever state from {path}: {len(self.page_metadata)} pages")
        self.build_index()
