# MultiModalQA — HGT Evidence Selection & MMDocIR RAG Pipeline

This repository extends the [MultimodalQA dataset repo](https://github.com/allenai/multimodalqa)
with two new pipelines for document understanding and evidence retrieval:

1. **HGT Evidence Selection** (`src/`) — Heterogeneous Graph Transformer over
   the MultimodalQA dataset for multi-modal evidence retrieval.
2. **MMDocIR Two-Stage RAG** (`mmdocir_rag/`) — Stage 1 dense page retrieval +
   Stage 2 HGT layout reranking for the MMDocIR benchmark.

The original MultimodalQA dataset files (`dataset/`) are preserved because the
HGT pipeline (v1) was trained and evaluated on that data.

---

## Repository layout

```
.
├── src/                        # HGT pipeline for MultimodalQA
│   ├── data/                   # Loaders, encoders, graph / label builders
│   │   ├── io.py               # JSONL / JSONL.GZ read/write
│   │   ├── mmqa_loader.py      # Load MMQA questions, texts, tables, images
│   │   ├── qwen_vl_encoder.py  # Qwen3-VL-Embedding-8B (+ fallback)
│   │   ├── graph_builder.py    # Build heterogeneous question graphs
│   │   ├── feature_builder.py  # Encode node features (text + image)
│   │   ├── label_builder.py    # Generate binary supervision labels
│   │   ├── hetero_converter.py # Graph dict → PyG HeteroData (MMQA schema)
│   │   └── hetero_utils.py     # Schema-agnostic HeteroData conversion core
│   ├── models/
│   │   ├── base_hgt.py         # Shared HeteroGraphConv + QueryAwareScoringHead
│   │   └── hgt_model.py        # HGTEvidenceModel (MMQA node/edge schema)
│   ├── trainers/
│   │   ├── losses.py           # Focal + Margin Ranking loss
│   │   ├── metrics.py          # Recall@K, MRR, P/R/F1, threshold optimisation
│   │   └── trainer.py          # Training loop, AMP, checkpoint management
│   ├── scripts/
│   │   ├── build_graphs.py     # Pre-build and cache graph objects
│   │   ├── train.py            # Training entry point
│   │   ├── evaluate.py         # Evaluation with threshold leakage prevention
│   │   └── evaluate_embedding_only.py  # Embedding-only baseline
│   └── utils/
│       └── logging.py          # Shared logger factory
├── mmdocir_rag/                # MMDocIR two-stage RAG
│   ├── data/
│   │   └── mmdocir_loader.py   # Load MMDocIR parquet + annotation JSONL
│   ├── stage1_page_retrieval/
│   │   └── page_retriever.py   # Dense page retrieval (embedding similarity)
│   ├── stage2_hgt_rerank/
│   │   ├── graph_builder.py    # Build query-page-layout graphs
│   │   ├── hetero_converter.py # Graph dict → HeteroData (MMDocIR schema)
│   │   ├── hgt_model.py        # MMDocIRHGT (uses src.models.base_hgt)
│   │   ├── layout_reranker.py  # LayoutReranker: encode + graph + score
│   │   └── train_stage2.py     # Stage 2 training entry point
│   ├── evaluation/
│   │   ├── metrics.py          # Official MMDocIR overlap / recall / MRR
│   │   ├── evaluate_stage1.py  # Stage 1 evaluation script
│   │   └── evaluate_stage2.py  # Stage 2 evaluation script
│   ├── scripts/
│   │   └── download_data.py    # Download MMDocIR evaluation dataset from HF
│   └── configs/
│       └── stage2.yaml         # Stage 2 model / training configuration
├── configs/
│   └── default.yaml            # MMQA HGT configuration
├── dataset/                    # Original MultimodalQA data files (JSONL.GZ)
└── mmdocir_data/               # Downloaded MMDocIR files (git-ignored)
```

---

## Part 1 — HGT Evidence Selection on MultimodalQA

### Installation

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements_hgt.txt
```

> `torch-geometric` must match your PyTorch version.
> See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### Encoder: Qwen3-VL-Embedding-8B

The default encoder is `Qwen/Qwen3-VL-Embedding-8B` (~16 GB VRAM in bf16).
Configure in `configs/default.yaml`:

```yaml
encoder:
  name: qwen3_vl_embedding_8b
  model_name_or_path: /path/to/Qwen3-VL-Embedding-8B
  dtype: bf16
  device: cuda
```

**Automatic fallback** — if the Qwen model cannot be loaded:
- Text encoding: `sentence-transformers/all-MiniLM-L6-v2`
- Image encoding: zero vectors (title text used instead)

### Graph schema

| Node type   | Description                              |
|-------------|------------------------------------------|
| `query`     | Question text                            |
| `textblock` | Wikipedia text paragraph                 |
| `table`     | Table (rendered to image for encoding)   |
| `cell`      | Individual table cell                    |
| `image`     | Image (encoded via vision encoder)       |
| `caption`   | Image title (pseudo-caption)             |

All edges are bidirectional (each direction is an explicit edge type).

### Running the pipeline

```bash
# 1. Pre-build graphs (cache to outputs/graphs/)
python -m src.scripts.build_graphs --config configs/default.yaml --split train
python -m src.scripts.build_graphs --config configs/default.yaml --split dev

# 2. Train
python -m src.scripts.train --config configs/default.yaml

# 3. Evaluate on dev (thresholds from checkpoint)
python -m src.scripts.evaluate --config configs/default.yaml --split dev

# 4. Evaluate on test (re-optimise thresholds on dev to prevent leakage)
python -m src.scripts.evaluate \
    --config configs/default.yaml \
    --split test \
    --checkpoint outputs/checkpoints/best.pt \
    --optimize-thresholds-on dev
```

Checkpoints are saved to `outputs/checkpoints/`.
Metrics: `Recall@1/3/5`, `MRR`, `Precision/Recall/F1` per node type.

---

## Part 2 — MMDocIR Two-Stage RAG

The MMDocIR pipeline retrieves fine-grained document layouts (bboxes) in
response to natural language questions over multi-page PDF documents.

### Stage 1: Dense Page Retrieval

Encodes all pages and the query with Qwen3-VL-Embedding-8B; retrieves the
top-K pages by cosine similarity.

### Stage 2: HGT Layout Reranking

Builds a heterogeneous graph per query (query ↔ pages ↔ layouts, with spatial
adjacency edges between layouts on the same page) and runs a 3-layer HGT to
score layouts.

### Setup

```bash
# Download evaluation dataset (MMDocIR/MMDocIR_Evaluation_Dataset on HuggingFace)
python -m mmdocir_rag.scripts.download_data --out-dir mmdocir_data

# (Optional) set HF_TOKEN env var for authenticated / faster downloads
HF_TOKEN=hf_... python -m mmdocir_rag.scripts.download_data
```

### Configuration

Edit `mmdocir_rag/configs/stage2.yaml`:

```yaml
data:
  pages_parquet:     mmdocir_data/MMDocIR_pages.parquet
  layouts_parquet:   mmdocir_data/MMDocIR_layouts.parquet
  annotations_jsonl: mmdocir_data/MMDocIR_annotations.jsonl
  text_mode:         ocr_text     # or "vlm_text"
  top_k_pages:       10

encoder:
  model_path: /path/to/Qwen3-VL-Embedding-8B
  device: cuda
  dtype: bf16
```

### Evaluation

```bash
# Stage 1 — embedding-only page retrieval
python -m mmdocir_rag.evaluation.evaluate_stage1 \
    --config mmdocir_rag/configs/stage2.yaml \
    --top-k 10 \
    --output outputs/stage1_eval/metrics.json

# Stage 2 — HGT layout reranking
python -m mmdocir_rag.evaluation.evaluate_stage2 \
    --config mmdocir_rag/configs/stage2.yaml \
    --checkpoint outputs/stage2_checkpoints/best.pt \
    --top-k 10 \
    --output outputs/stage2_eval/metrics.json

# Stage 2 — training
python -m mmdocir_rag.stage2_hgt_rerank.train_stage2 \
    --config mmdocir_rag/configs/stage2.yaml
```

Metrics (aligned with official MMDocIR `metric_eval.py`):
`page_recall@1/5/10`, `page_mrr`,
`layout_recall@1/5/10` (soft overlap), `layout_exact_recall@1/5/10`, `layout_mrr`.

### Shared components

The MMDocIR HGT reuses core building blocks from `src/`:

| MMDocIR file | Imports from src/ |
|---|---|
| `stage2_hgt_rerank/hgt_model.py` | `src.models.base_hgt.HeteroGraphConv`, `QueryAwareScoringHead` |
| `stage2_hgt_rerank/hetero_converter.py` | `src.data.hetero_utils.graph_dict_to_heterodata` |
| All modules | `src.utils.logging.get_logger` |

---

## Changelog

- `2021-04-23` Initial MultimodalQA release (Allen AI)
- `2025-XX` Added HGT evidence selection pipeline
- `2026-04` Added MMDocIR two-stage RAG; refactored shared HGT components into
  `src/models/base_hgt.py` and `src/data/hetero_utils.py`; removed legacy
  baselines

---

## MultiModalQA Dataset Format

*The sections below are the original format documentation from the Allen AI release.*

In the [`dataset/`](./dataset) folder you will find:

1. `MultiModalQA_train/dev/test.jsonl.gz` — questions, answers, and metadata
2. `tables.jsonl.gz` — table contexts
3. `texts.jsonl.gz` — text contexts
4. `images.jsonl.gz` — image metadata (images downloaded separately)

Images: [images.zip](https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip)

### QA file format

```json
{
  "qid": "5454c14ad01e722c2619b66778daa98b",
  "question": "who owns the rights to little shop of horrors?",
  "answers": [{"answer": "...", "type": "string", "modality": "text", ...}],
  "metadata": {"type": "Compose(TableQ,ImageListQ)", "modalities": ["image","table"], ...},
  "supporting_context": [
    {"doc_id": "46ae2a8e7928ed5a8e5f9c59323e5e49", "doc_part": "table"},
    {"doc_id": "d57e56eff064047af5a6ef074a570956", "doc_part": "image"}
  ]
}
```

`MultiModalQA_test.jsonl.gz` omits `answers` and `supporting_context`.

### Table format (`tables.jsonl.gz`)

```json
{
  "title": "Dutch Ruppersberger",
  "url": "https://en.wikipedia.org/wiki/Dutch_Ruppersberger",
  "id": "dcd7cb8f23737c6f38519c3770a6606f",
  "table": {
    "table_rows": [[{"text": "Baltimore County Executive", "links": [...]}]],
    "table_name": "Electoral history",
    "header": [{"column_name": "Year", "metadata": {...}}]
  }
}
```

### Image metadata format (`images.jsonl.gz`)

```json
{"title": "Taipei", "url": "...", "id": "632ea110be92836441adfb3167edf8ff", "path": "Taipei.jpg"}
```

### Text format (`texts.jsonl.gz`)

```json
{"title": "...", "url": "...", "id": "16c61fe756817f0b35df9717fae1000e", "text": "Over three years..."}
```
