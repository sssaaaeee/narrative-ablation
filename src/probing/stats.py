"""stats.py – McNemar's test and χ² + BH FDR correction for probe results (§6).

Compares two probe JSON files (condition A vs condition B) layer by layer.

Tests applied
-------------
McNemar's test (paired, with Yates correction) when correct_mask lists are
available.  BH FDR correction is applied across all layers.

Supported probe types (auto-detected from JSON structure)
---------------------------------------------------------
  time             → subkey "time" → condition key → "correct_mask"
  entity_id        → subkey "entity_id" → condition key → "correct_mask"
  common_element   → subkeys "content_decode", "entity_decode", "location_decode"

Usage
-----
    # Time probe: baseline vs w_o_time_series
    python src/probing/stats.py \\
        --probe-dir results/probing/Meta-Llama-3.1-8B-Instruct/ \\
        --probe-type time \\
        --cond-a baseline \\
        --cond-b w_o_time_series

    # Common element probe: baseline vs w_o_causality
    python src/probing/stats.py \\
        --probe-dir results/probing/Meta-Llama-3.1-8B-Instruct/ \\
        --probe-type common_element \\
        --cond-a baseline \\
        --cond-b w_o_causality
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# McNemar's test (with Yates continuity correction)
# ---------------------------------------------------------------------------

def mcnemar_test(
    mask_a: List[bool],
    mask_b: List[bool],
    correction: bool = True,
) -> Tuple[float, float, int, int, int, int]:
    """Paired McNemar's test.

    Returns (statistic, p_value, a, b, c, d).
    """
    a = b = c = d = 0
    for ca, cb in zip(mask_a, mask_b):
        if ca and cb:   a += 1
        elif ca:        b += 1
        elif cb:        c += 1
        else:           d += 1
    n = b + c
    if n == 0:
        return float("nan"), float("nan"), a, b, c, d
    stat = (abs(b - c) - 1) ** 2 / n if correction else (b - c) ** 2 / n
    p    = float(1.0 - chi2.cdf(stat, df=1))
    return stat, p, a, b, c, d


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR correction
# ---------------------------------------------------------------------------

def bh_correction(p_values: List[float]) -> List[float]:
    n     = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj   = [1.0] * n
    for rank, idx in enumerate(order, start=1):
        adj[idx] = min(1.0, p_values[idx] * n / rank)
    # Enforce monotonicity
    for i in range(len(order) - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj


def sig_label(p: float) -> str:
    if math.isnan(p): return "nan"
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def compare_masks(
    mask_a: Optional[List[bool]],
    mask_b: Optional[List[bool]],
) -> Dict[str, Any]:
    if not mask_a or not mask_b:
        return {"error": "missing correct_mask"}
    if len(mask_a) != len(mask_b):
        return {"error": f"length mismatch: {len(mask_a)} vs {len(mask_b)}"}
    stat, p, a, b, c, d = mcnemar_test(mask_a, mask_b)
    return {
        "n": len(mask_a),
        "acc_a": sum(mask_a) / len(mask_a),
        "acc_b": sum(mask_b) / len(mask_b),
        "b": b, "c": c,
        "stat": stat,
        "p_raw": p,
    }


def run_comparison(
    probe_dir: Path,
    probe_type: str,
    cond_a: str,
    cond_b: str,
    layer_start: int,
    layer_end: int,
) -> None:
    layer_files = sorted(probe_dir.glob("probe_layer_*.json"))
    if not layer_files:
        print(f"No probe_layer_*.json files found in {probe_dir}")
        return

    rows: List[Dict] = []
    raw_p: List[float] = []

    for lf in layer_files:
        data  = load_json(lf)
        layer = data.get("layer", int(lf.stem.split("_")[-1]))
        if not (layer_start <= layer <= layer_end):
            continue

        subkeys = {"time": ["time"],
                   "entity_id": ["entity_id"],
                   "common_element": ["content_decode", "entity_decode", "location_decode"]}

        for subkey in subkeys.get(probe_type, [probe_type]):
            if probe_type == "common_element":
                da = (data.get("common_element") or {}).get(cond_a, {}).get(subkey)
                db = (data.get("common_element") or {}).get(cond_b, {}).get(subkey)
            else:
                da = (data.get(probe_type) or {}).get(cond_a)
                db = (data.get(probe_type) or {}).get(cond_b)

            if da is None or db is None:
                continue

            result = compare_masks(da.get("correct_mask"), db.get("correct_mask"))
            result.update({"layer": layer, "subkey": subkey})
            rows.append(result)
            raw_p.append(result.get("p_raw", 1.0) if not math.isnan(result.get("p_raw", float("nan"))) else 1.0)

    if not rows:
        print("No comparable rows found.")
        return

    adj_p = bh_correction(raw_p)

    print(f"\nProbe comparison: {cond_a} vs {cond_b}  [{probe_type}]  (BH-corrected)")
    print(f"{'Layer':>6} {'Key':<20} {'n':>6} {'acc_a':>7} {'acc_b':>7} {'b':>5} {'c':>5} "
          f"{'stat':>8} {'p_raw':>9} {'p_BH':>9} {'sig':>4}")
    print("-" * 90)
    for row, p_adj in zip(rows, adj_p):
        stat_s = f"{row['stat']:.3f}" if not math.isnan(row.get("stat", float("nan"))) else "NaN"
        p_s    = f"{row['p_raw']:.4f}" if not math.isnan(row.get("p_raw", float("nan"))) else "NaN"
        print(
            f"{row['layer']:>6} {row['subkey']:<20} {row.get('n', 0):>6} "
            f"{row.get('acc_a', float('nan')):>7.4f} {row.get('acc_b', float('nan')):>7.4f} "
            f"{row.get('b', 0):>5} {row.get('c', 0):>5} "
            f"{stat_s:>8} {p_s:>9} {p_adj:>9.4f} {sig_label(p_adj):>4}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="McNemar + BH tests for linear probe comparisons."
    )
    parser.add_argument("--probe-dir",   required=True,
                        help="Directory with probe_layer_*.json files")
    parser.add_argument("--probe-type",  default="time",
                        choices=["time", "entity_id", "common_element"])
    parser.add_argument("--cond-a",      default="baseline")
    parser.add_argument("--cond-b",      default="w_o_time_series")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end",   type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(
        probe_dir   = Path(args.probe_dir),
        probe_type  = args.probe_type,
        cond_a      = args.cond_a,
        cond_b      = args.cond_b,
        layer_start = args.layer_start,
        layer_end   = args.layer_end,
    )


if __name__ == "__main__":
    main()
