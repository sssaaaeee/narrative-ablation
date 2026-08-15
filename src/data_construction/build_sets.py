"""build_sets.py – Element-set construction for 4 narrative conditions (§3).

Constructs three set variants per pattern:

  original_sets   – 10 items sampled from 4 pools, sorted chronologically
  derived_sets    – chain structure: each set shares one field (location /
                    entity / content) with its predecessor → common-element
                    structure enabling multi-hop reasoning
  derived_sets2   – identical to derived_sets except temporal field is
                    shuffled (used for the w_o_time_series condition)

Output: data/sets/generated_sets.json

Usage
-----
    python src/data_construction/build_sets.py \\
        --elements  data/pools/elements.json \\
        --out       data/sets/generated_sets.json \\
        --num-patterns 300 \\
        --num-sets    10  \\
        --seed        42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELEMENTS = REPO_ROOT / "data" / "pools" / "elements.json"
DEFAULT_OUT      = REPO_ROOT / "data" / "sets" / "generated_sets.json"

NUM_PATTERNS_DEFAULT = 300
NUM_SETS_DEFAULT     = 10
SEED_DEFAULT         = 42


# ---------------------------------------------------------------------------
# I/O
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
# Set builders
# ---------------------------------------------------------------------------

def build_original_sets(elements: Dict[str, Any], num_sets: int) -> List[Dict[str, Any]]:
    """Sample num_sets items from each pool, sort chronologically by temporal."""
    selected_temporal = random.sample(elements["temporal"], num_sets)
    selected_location = random.sample(elements["location"], num_sets)
    selected_entity   = random.sample(elements["entity"],   num_sets)
    selected_content  = random.sample(elements["content"],  num_sets)

    # Sort indices by parsed date so original_sets are in chronological order
    paired = sorted(
        enumerate(selected_temporal),
        key=lambda kv: datetime.strptime(kv[1], "%Y-%m-%d"),
    )
    order = [idx for idx, _ in paired]

    return [
        {
            "set_id":   rank + 1,
            "temporal": selected_temporal[order[rank]],
            "location": selected_location[order[rank]],
            "entity":   selected_entity[order[rank]],
            "content":  selected_content[order[rank]],
        }
        for rank in range(num_sets)
    ]


def build_derived_sets(original_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Introduce chain-structure: each set randomly inherits one field value
    (location / entity / content) from the immediately preceding set.

    This creates common-element links between consecutive events, enabling
    multi-hop reasoning probes (§6 Common Element Decode probe).
    """
    derived: List[Dict[str, Any]] = []
    for i, item in enumerate(original_sets):
        new_item = item.copy()
        if i > 0:
            field = random.choice(["location", "entity", "content"])
            new_item[field] = derived[i - 1][field]
        derived.append(new_item)
    return derived


def build_temporal_shuffled_sets(derived_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shuffle the temporal field of derived_sets (all other fields unchanged).

    Used as input for the w_o_time_series condition.
    """
    temporals = [s["temporal"] for s in derived_sets]
    random.shuffle(temporals)
    return [
        {
            "set_id":   i + 1,
            "temporal": temporals[i],
            "location": s["location"],
            "entity":   s["entity"],
            "content":  s["content"],
        }
        for i, s in enumerate(derived_sets)
    ]


def build_pattern(elements: Dict[str, Any], pattern_id: int, num_sets: int) -> Dict[str, Any]:
    original_sets = build_original_sets(elements, num_sets)
    derived_sets  = build_derived_sets(original_sets)
    derived_sets2 = build_temporal_shuffled_sets(derived_sets)
    return {
        "pattern_id":   pattern_id,
        "original_sets": original_sets,
        "derived_sets":  derived_sets,
        "derived_sets2": derived_sets2,
    }


def validate_pattern(pattern: Dict[str, Any], num_sets: int) -> None:
    for key in ("original_sets", "derived_sets", "derived_sets2"):
        if len(pattern[key]) != num_sets:
            raise ValueError(
                f"pattern_id={pattern['pattern_id']} key={key} "
                f"has {len(pattern[key])} items, expected {num_sets}"
            )


# ---------------------------------------------------------------------------
# Incremental build (resume-safe)
# ---------------------------------------------------------------------------

def ensure_generated_sets(
    elements: Dict[str, Any],
    out_path: Path,
    num_patterns: int,
    num_sets: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Build or extend generated_sets.json to exactly num_patterns patterns."""
    random.seed(seed)

    if out_path.exists():
        data = load_json(out_path)
        patterns: List[Dict[str, Any]] = data.get("patterns", [])

        if len(patterns) == num_patterns:
            for p in patterns:
                validate_pattern(p, num_sets)
            print(f"generated_sets.json already complete ({num_patterns} patterns).")
            return patterns

        if len(patterns) < num_patterns:
            n_existing = len(patterns)
            for new_id in range(n_existing + 1, num_patterns + 1):
                patterns.append(build_pattern(elements, new_id, num_sets))
            meta = data.get("metadata", {})
            meta["num_patterns"] = num_patterns
            meta["extended_at"]  = datetime.now().isoformat(timespec="seconds")
            save_json_atomic(out_path, {"metadata": meta, "patterns": patterns})
            print(
                f"Extended: {n_existing} existing + "
                f"{num_patterns - n_existing} new = {num_patterns} total."
            )
            return patterns

        raise ValueError(
            f"{out_path} already has {len(patterns)} patterns > requested {num_patterns}."
        )

    # Fresh build
    patterns = [build_pattern(elements, pid, num_sets) for pid in range(1, num_patterns + 1)]
    output = {
        "metadata": {
            "created_at":          datetime.now().isoformat(timespec="seconds"),
            "num_patterns":        num_patterns,
            "num_sets_per_pattern": num_sets,
            "seed":                seed,
        },
        "patterns": patterns,
    }
    save_json_atomic(out_path, output)
    print(f"Saved {num_patterns} patterns → {out_path}")
    return patterns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build element sets for all patterns.")
    parser.add_argument("--elements",     default=str(DEFAULT_ELEMENTS))
    parser.add_argument("--out",          default=str(DEFAULT_OUT))
    parser.add_argument("--num-patterns", type=int, default=NUM_PATTERNS_DEFAULT)
    parser.add_argument("--num-sets",     type=int, default=NUM_SETS_DEFAULT)
    parser.add_argument("--seed",         type=int, default=SEED_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    elements = load_json(Path(args.elements))
    ensure_generated_sets(
        elements   = elements,
        out_path   = Path(args.out),
        num_patterns = args.num_patterns,
        num_sets     = args.num_sets,
        seed         = args.seed,
    )


if __name__ == "__main__":
    main()
