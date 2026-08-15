"""build_pools.py – Pool generation and deduplication (§3).

Loads the element master file (data/pools/elements.json) and writes four
deduplicated sub-pool JSON files:

    data/pools/pool_temporal.json
    data/pools/pool_location.json
    data/pools/pool_entity.json
    data/pools/pool_content.json

Each pool is a list of unique strings.  The script also verifies minimum
pool sizes required for the experiment (NUM_SETS samples per pattern).

Usage
-----
    python src/data_construction/build_pools.py \\
        --elements data/pools/elements.json \\
        --out-dir  data/pools/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Defaults (overridden by CLI / configs/data.yaml)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELEMENTS = REPO_ROOT / "data" / "pools" / "elements.json"
DEFAULT_OUT_DIR  = REPO_ROOT / "data" / "pools"

# Minimum pool sizes needed (= num_sets_per_pattern from configs/data.yaml)
MIN_POOL_SIZE = 10


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Pool extraction
# ---------------------------------------------------------------------------

def extract_pool(elements: Dict[str, Any], key: str) -> List[str]:
    """Return deduplicated list for *key* from the master elements dict."""
    raw: List[str] = elements.get(key, [])
    seen: set = set()
    deduped: List[str] = []
    for item in raw:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def validate_pool(pool: List[str], name: str, min_size: int) -> None:
    if len(pool) < min_size:
        raise ValueError(
            f"Pool '{name}' has only {len(pool)} items; "
            f"need at least {min_size} for sampling without replacement."
        )


def build_pools(elements_path: Path, out_dir: Path) -> Dict[str, List[str]]:
    """Load master elements, extract and deduplicate each pool, write to out_dir."""
    elements = load_json(elements_path)

    pools: Dict[str, List[str]] = {}
    for key in ("temporal", "location", "entity", "content"):
        pool = extract_pool(elements, key)
        validate_pool(pool, key, MIN_POOL_SIZE)
        pools[key] = pool

        out_path = out_dir / f"pool_{key}.json"
        save_json_atomic(out_path, {"pool": key, "count": len(pool), "items": pool})
        print(f"  {key:12s}: {len(pool):4d} items  →  {out_path}")

    return pools


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and deduplicate element pools.")
    parser.add_argument(
        "--elements", default=str(DEFAULT_ELEMENTS),
        help="Path to master elements.json",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="Directory to write pool_*.json files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    elements_path = Path(args.elements)
    out_dir = Path(args.out_dir)

    if not elements_path.exists():
        raise FileNotFoundError(f"elements.json not found at {elements_path}")

    print(f"Building pools from: {elements_path}")
    pools = build_pools(elements_path, out_dir)
    print(f"Done. {sum(len(p) for p in pools.values())} total items across 4 pools.")


if __name__ == "__main__":
    main()
