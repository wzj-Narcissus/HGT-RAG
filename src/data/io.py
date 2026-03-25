"""Low-level I/O helpers for .jsonl and .jsonl.gz files."""
import gzip
import json
import os
from typing import List, Dict


def read_jsonl(path: str) -> List[Dict]:
    """Read a .jsonl or .jsonl.gz file and return list of dicts."""
    records = []
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(dataset_dir: str, filename: str) -> str:
    """Return the full path, trying both .jsonl.gz and .jsonl variants."""
    for name in [filename, filename + ".gz", filename.replace(".gz", "")]:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find {filename} (or .gz variant) in {dataset_dir}"
    )


def index_by_id(records: List[Dict], id_field: str = "id") -> Dict[str, Dict]:
    """Build a dict mapping id -> record."""
    return {r[id_field]: r for r in records if id_field in r}
