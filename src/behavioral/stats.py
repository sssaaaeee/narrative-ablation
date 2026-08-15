"""stats.py – McNemar's test + Benjamini–Hochberg correction for QA results (§5).

Binarises scores (1.0 → correct, <1.0 → incorrect) then applies McNemar's
paired test with Yates correction for each (condition vs baseline) comparison.
BH FDR correction is applied across the 3 comparisons per hop-type.

Usage
-----
    # Single model
    python src/behavioral/stats.py \\
        --eval-dirs results/:Llama-8B

    # Multiple models in one run
    python src/behavioral/stats.py \\
        --eval-dirs results/:Llama-8B  results_70B/:Llama-70B
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2


REPO_ROOT  = Path(__file__).resolve().parents[2]
VARIANTS   = ["w_o_causality", "w_o_time_series", "w_o_agency"]
THRESHOLD  = 1.0   # score >= THRESHOLD → correct


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
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

    Returns (statistic, p_value, a, b, c, d) where:
      a = both correct, b = A correct & B wrong,
      c = A wrong & B correct, d = both wrong
    """
    a = b = c = d = 0
    for ca, cb in zip(mask_a, mask_b):
        if ca and cb:
            a += 1
        elif ca and not cb:
            b += 1
        elif not ca and cb:
            c += 1
        else:
            d += 1
    n = b + c
    if n == 0:
        return float("nan"), float("nan"), a, b, c, d
    stat = (abs(b - c) - 1) ** 2 / n if correction else (b - c) ** 2 / n
    p    = float(1.0 - chi2.cdf(stat, df=1))
    return stat, p, a, b, c, d


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR correction
# ---------------------------------------------------------------------------

def bh_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    n     = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj   = [1.0] * n
    for rank, idx in enumerate(order, start=1):
        adj[idx] = min(1.0, p_values[idx] * n / rank)
    # Enforce monotonicity (BH-adjusted p must be non-decreasing)
    for i in range(len(order) - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj


def sig_label(p: float) -> str:
    if math.isnan(p):
        return "nan"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Single-hop analysis
# ---------------------------------------------------------------------------

def run_single_hop(eval_path: Path, model_label: str) -> None:
    data    = load_json(eval_path)
    results = data.get("results", [])

    # Key: (pattern_id, index, question_type) → {variant: score}
    by_key: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in results:
        key = (r["pattern_id"], r.get("index", r.get("question_index")), r["question_type"])
        by_key[key][r["variant"]] = r["score"]

    rows: List[Dict] = []
    raw_p: List[float] = []

    for variant in VARIANTS:
        base_mask: List[bool] = []
        var_mask:  List[bool] = []
        for scores in by_key.values():
            if "baseline" in scores and variant in scores:
                base_mask.append(scores["baseline"] >= THRESHOLD)
                var_mask.append(scores[variant]   >= THRESHOLD)

        n = len(base_mask)
        if n == 0:
            rows.append({"variant": variant, "n": 0, "acc_baseline": float("nan"),
                         "acc_variant": float("nan"), "diff": float("nan"),
                         "b": 0, "c": 0, "stat": float("nan"), "p_raw": float("nan")})
            raw_p.append(1.0)
            continue

        acc_a = sum(base_mask) / n
        acc_b = sum(var_mask)  / n
        stat, p, _, b_cnt, c_cnt, _ = mcnemar_test(base_mask, var_mask)
        rows.append({
            "variant": variant, "n": n,
            "acc_baseline": acc_a, "acc_variant": acc_b,
            "diff": acc_a - acc_b,
            "b": b_cnt, "c": c_cnt, "stat": stat, "p_raw": p,
        })
        raw_p.append(p if not math.isnan(p) else 1.0)

    adj_p = bh_correction(raw_p)

    print(f"\n[{model_label}] Single-hop QA  (McNemar, BH-corrected)")
    print(f"  {'Variant':<22} {'n':>6} {'Acc_base':>9} {'Acc_var':>8} {'diff':>6} "
          f"{'b':>5} {'c':>5} {'stat':>8} {'p_raw':>9} {'p_BH':>9} {'sig':>4}")
    print("  " + "-" * 98)
    for row, p_adj in zip(rows, adj_p):
        stat_str = f"{row['stat']:.3f}" if not math.isnan(row["stat"]) else "NaN"
        p_str    = f"{row['p_raw']:.4f}" if not math.isnan(row["p_raw"]) else "NaN"
        print(
            f"  {row['variant']:<22} {row['n']:>6} {row['acc_baseline']:>9.4f} "
            f"{row['acc_variant']:>8.4f} {row['diff']:>6.4f} "
            f"{row['b']:>5} {row['c']:>5} {stat_str:>8} {p_str:>9} "
            f"{p_adj:>9.4f} {sig_label(p_adj):>4}"
        )


# ---------------------------------------------------------------------------
# Multi-hop analysis
# ---------------------------------------------------------------------------

def run_multi_hop(eval_path: Path, model_label: str) -> None:
    data    = load_json(eval_path)
    results = data.get("results", [])

    by_key: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in results:
        key = (r["pattern_id"], r.get("index", r.get("question_index")), r["question_type"])
        by_key[key][r["variant"]] = r["score"]

    rows: List[Dict] = []
    raw_p: List[float] = []

    for variant in VARIANTS:
        base_mask: List[bool] = []
        var_mask:  List[bool] = []
        for scores in by_key.values():
            if "baseline" in scores and variant in scores:
                base_mask.append(scores["baseline"] >= THRESHOLD)
                var_mask.append(scores[variant]   >= THRESHOLD)

        n = len(base_mask)
        if n == 0:
            rows.append({"variant": variant, "n": 0, "acc_baseline": float("nan"),
                         "acc_variant": float("nan"), "stat": float("nan"), "p_raw": float("nan")})
            raw_p.append(1.0)
            continue

        acc_a = sum(base_mask) / n
        acc_b = sum(var_mask)  / n
        stat, p, _, b_cnt, c_cnt, _ = mcnemar_test(base_mask, var_mask)
        rows.append({
            "variant": variant, "n": n,
            "acc_baseline": acc_a, "acc_variant": acc_b,
            "diff": acc_a - acc_b,
            "b": b_cnt, "c": c_cnt, "stat": stat, "p_raw": p,
        })
        raw_p.append(p if not math.isnan(p) else 1.0)

    adj_p = bh_correction(raw_p)

    print(f"\n[{model_label}] Multi-hop QA  (McNemar, BH-corrected)")
    print(f"  {'Variant':<22} {'n':>6} {'Acc_base':>9} {'Acc_var':>8} {'diff':>6} "
          f"{'stat':>8} {'p_raw':>9} {'p_BH':>9} {'sig':>4}")
    print("  " + "-" * 88)
    for row, p_adj in zip(rows, adj_p):
        stat_str = f"{row['stat']:.3f}" if not math.isnan(row.get("stat", float("nan"))) else "NaN"
        p_str    = f"{row['p_raw']:.4f}" if not math.isnan(row.get("p_raw", float("nan"))) else "NaN"
        print(
            f"  {row['variant']:<22} {row['n']:>6} {row['acc_baseline']:>9.4f} "
            f"{row['acc_variant']:>8.4f} {row['diff']:>6.4f} "
            f"{stat_str:>8} {p_str:>9} {p_adj:>9.4f} {sig_label(p_adj):>4}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="McNemar + BH significance tests for QA results."
    )
    parser.add_argument(
        "--eval-dirs", nargs="+", required=True,
        help='Pairs of "results_dir:ModelLabel", e.g. results/:Llama-8B',
    )
    parser.add_argument("--mode", choices=["all", "single", "multi"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for spec in args.eval_dirs:
        if ":" in spec:
            dir_str, label = spec.rsplit(":", 1)
        else:
            dir_str, label = spec, Path(spec).name
        d = Path(dir_str)

        if args.mode in ("all", "single"):
            p = d / "eval_single.json"
            if p.exists():
                run_single_hop(p, label)
            else:
                print(f"[{label}] eval_single.json not found at {p}")

        if args.mode in ("all", "multi"):
            p = d / "eval_multi.json"
            if p.exists():
                run_multi_hop(p, label)
            else:
                print(f"[{label}] eval_multi.json not found at {p}")


if __name__ == "__main__":
    main()
