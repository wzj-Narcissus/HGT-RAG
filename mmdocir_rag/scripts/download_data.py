"""Download MMDocIR Evaluation Dataset from Hugging Face.

Usage:
    python -m mmdocir_rag.scripts.download_data [--out-dir mmdocir_data]

The script downloads three files from MMDocIR/MMDocIR_Evaluation_Dataset:
  - MMDocIR_pages.parquet     (eval pages, ~130MB)
  - MMDocIR_layouts.parquet   (eval layouts, ~240MB)
  - MMDocIR_annotations.jsonl (QA annotations, <10MB)

Set HF_TOKEN env var for authenticated (higher-rate) downloads.
"""
import argparse
import os
import sys


def download(out_dir: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    repo_id = "MMDocIR/MMDocIR_Evaluation_Dataset"
    files = [
        "MMDocIR_pages.parquet",
        "MMDocIR_layouts.parquet",
        "MMDocIR_annotations.jsonl",
    ]

    os.makedirs(out_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    for fname in files:
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            print(f"  [skip] {fname} already exists ({os.path.getsize(dest) // 1024 // 1024} MB)")
            continue
        print(f"  [download] {fname} ...")
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            local_dir=out_dir,
            token=token,
        )
        print(f"  [ok] {fname} -> {dest}")

    print(f"\nAll files ready in '{out_dir}'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="mmdocir_data")
    args = parser.parse_args()
    download(args.out_dir)


if __name__ == "__main__":
    main()
