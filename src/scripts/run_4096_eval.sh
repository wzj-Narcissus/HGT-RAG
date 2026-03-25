#!/bin/bash
# Run 4096-dim embedding-only evaluation
# Usage: bash src/scripts/run_4096_eval.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

GRAPH_CACHE_TAG="4aaabde7a8"
HETERO_DIR="outputs/hetero_4096"
SPLIT="dev"
OUTPUT_FILE="outputs/predictions/embedding_only_4096_metrics.json"

# Find the 4096-dim hetero dir
HETERO_PATH=$(ls -d ${HETERO_DIR}/${SPLIT}_*_d4096_${GRAPH_CACHE_TAG} 2>/dev/null | head -1)

if [ -z "$HETERO_PATH" ]; then
    echo "ERROR: 4096-dim hetero dir not found in ${HETERO_DIR}/"
    echo "Expected pattern: ${SPLIT}_*_d4096_${GRAPH_CACHE_TAG}"
    echo "Available dirs:"
    ls -d ${HETERO_DIR}/* 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "Using hetero dir: $HETERO_PATH"
echo "Output file: $OUTPUT_FILE"

python -m src.scripts.evaluate_embedding_only \
    --hetero_dir "$HETERO_PATH" \
    --split "$SPLIT" \
    --device cuda \
    --output_file "$OUTPUT_FILE"

echo ""
echo "=== Comparison with existing results ==="
echo ""
echo "--- Embedding-Only 256-dim ---"
python3 -c "
import json
with open('outputs/predictions/embedding_only_metrics.json') as f:
    m = json.load(f)
for k in ['overall/f1', 'overall/recall', 'overall/precision', 'overall/recall@1', 'overall/recall@3', 'overall/recall@5', 'overall/mrr']:
    print(f'  {k}: {m.get(k, \"N/A\"):.4f}')
"
echo ""
echo "--- Embedding-Only 4096-dim ---"
python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    m = json.load(f)
for k in ['overall/f1', 'overall/recall', 'overall/precision', 'overall/recall@1', 'overall/recall@3', 'overall/recall@5', 'overall/mrr']:
    print(f'  {k}: {m.get(k, \"N/A\"):.4f}')
"
echo ""
echo "--- HGT (Focal+3L+QW+EW) ---"
python3 -c "
import json
with open('outputs/predictions/metrics_dev.json') as f:
    m = json.load(f)
for k in ['overall/f1', 'overall/recall', 'overall/precision', 'overall/recall@1', 'overall/recall@3', 'overall/recall@5', 'overall/mrr']:
    print(f'  {k}: {m.get(k, \"N/A\"):.4f}')
"
