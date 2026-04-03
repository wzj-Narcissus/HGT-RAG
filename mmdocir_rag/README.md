# mmdocir_rag — MMDocIR Two-Stage RAG Pipeline

This sub-package implements a two-stage Retrieval-Augmented Generation (RAG)
pipeline for the [MMDocIR benchmark](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset),
which evaluates fine-grained layout retrieval from multi-page PDF documents.

## Architecture

```
Query
  │
  ▼
Stage 1: Dense Page Retrieval          (stage1_page_retrieval/)
  Qwen3-VL-Embedding-8B text encoder
  Cosine similarity over all pages
  → top-K pages (default K=10)
  │
  ▼
Stage 2: HGT Layout Reranking          (stage2_hgt_rerank/)
  Build heterogeneous graph:
    query ↔ pages ↔ layouts (bbox)
    + spatial adjacency edges
  3-layer HGT message passing
  Query-aware scoring heads
  → ranked layouts (page_id + bbox)
```

## Directory structure

```
mmdocir_rag/
├── data/
│   └── mmdocir_loader.py       # Load pages.parquet, layouts.parquet, annotations.jsonl
├── stage1_page_retrieval/
│   └── page_retriever.py       # Top-K page retrieval by embedding similarity
├── stage2_hgt_rerank/
│   ├── graph_builder.py        # Build query-page-layout heterogeneous graph
│   ├── hetero_converter.py     # Graph dict → PyG HeteroData (MMDocIR schema)
│   ├── hgt_model.py            # MMDocIRHGT (3-layer, uses src.models.base_hgt)
│   ├── layout_reranker.py      # LayoutReranker: encode + graph + infer
│   └── train_stage2.py         # Stage 2 training loop
├── evaluation/
│   ├── metrics.py              # Official MMDocIR overlap / recall / MRR
│   ├── evaluate_stage1.py      # Stage 1 evaluation script
│   └── evaluate_stage2.py      # Stage 2 evaluation script
├── scripts/
│   └── download_data.py        # Download evaluation dataset from HuggingFace
└── configs/
    └── stage2.yaml             # Model + data + training configuration
```

## Shared components from `src/`

| This package uses | From `src/` |
|---|---|
| `HeteroGraphConv`, `QueryAwareScoringHead` | `src.models.base_hgt` |
| `graph_dict_to_heterodata`, `infer_feature_dim` | `src.data.hetero_utils` |
| `QwenVLFeatureEncoder` | `src.data.qwen_vl_encoder` |
| `get_logger` | `src.utils.logging` |

## Data preparation

```bash
# From the repo root
python -m mmdocir_rag.scripts.download_data --out-dir mmdocir_data
```

Downloads three files from `MMDocIR/MMDocIR_Evaluation_Dataset` on HuggingFace:

| File | Size | Description |
|---|---|---|
| `MMDocIR_pages.parquet` | ~130 MB | 20 395 document pages with OCR text |
| `MMDocIR_layouts.parquet` | ~240 MB | 170 338 layout elements with bbox + type |
| `MMDocIR_annotations.jsonl` | <10 MB | 313 documents, 1 658 QA pairs |

Set `HF_TOKEN` environment variable for authenticated (higher-rate-limit) downloads.

### Annotation format

Each line in `MMDocIR_annotations.jsonl`:

```json
{
  "doc_name": "example.pdf",
  "page_indices": [0, 22],
  "layout_indices": [0, 153],
  "questions": [
    {
      "Q": "What is the GDP growth rate shown in the chart?",
      "A": "3.2%",
      "page_id": [4],
      "type": "['Chart']",
      "layout_mapping": [
        {"page": 4, "page_size": [612.0, 792.0], "bbox": [366, 229, 514, 383]}
      ]
    }
  ]
}
```

> **Note**: `doc_name` in annotations includes a `.pdf` extension that is
> stripped in the parquet files.  `MMDocIRLoader.iter_questions()` handles
> this normalisation automatically.

## Quick-start

```python
from mmdocir_rag import MMDocIRLoader, LayoutReranker

loader = MMDocIRLoader(
    pages_parquet_path="mmdocir_data/MMDocIR_pages.parquet",
    layouts_parquet_path="mmdocir_data/MMDocIR_layouts.parquet",
    annotations_jsonl_path="mmdocir_data/MMDocIR_annotations.jsonl",
)

print(loader.get_stats())

for qa in loader.iter_questions():
    pages = list(loader.get_document_pages(qa["doc_name"]).iterrows())
    print(qa["question"], "→ GT pages:", qa["page_id"])
    break
```

## Metrics (aligned with official MMDocIR `metric_eval.py`)

| Metric | Description |
|---|---|
| `page_recall@K` | Binary: 1 if any top-K predicted page is in GT |
| `page_mrr` | 1/rank of first correct page |
| `layout_recall@K` | Soft overlap score (intersection / min-area), clipped to [0,1] |
| `layout_exact_recall@K` | Binary: 1 if any top-K layout_id is in GT |
| `layout_mrr` | 1/rank of first predicted layout that overlaps any GT layout |
