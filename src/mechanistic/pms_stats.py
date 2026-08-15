"""pms_stats.py – Statistical comparison of PMS across conditions (§7).

Compares per-head Prefix Matching Scores between baseline and ablation
conditions using paired Wilcoxon signed-rank tests across patterns.

Two selection criteria for "induction heads" (following Olsson et al.):
  undirected: top-K heads by mean baseline PMS
  directed:   heads where baseline PMS > threshold AND condition PMS drops
              significantly

Multiple comparisons: Bonferroni correction across all heads.

Usage
-----
    python src/mechanistic/pms_stats.py \\
        --pms-dir results/mechanistic/Meta-Llama-3.1-8B-Instruct/ \\
        --baseline-file pms_baseline.json \\
        --compare-files pms_w_o_causality.json pms_w_o_agency.json \\
        --top-k 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_pms_matrix(records: List[Dict]) -> Tuple[np.ndarray, int, int]:
    """Extract per-pattern PMS matrix.

    Returns (matrix, n_layers, n_heads) where matrix shape is
    (n_patterns, n_layers, n_heads).
    """
    first_layers = records[0]["layers"]
    n_layers     = max(int(k) for k in first_layers) + 1
    n_heads      = len(list(first_layers.values())[0])

    matrix = np.zeros((len(records), n_layers, n_heads), dtype=np.float32)
    for i, rec in enumerate(records):
        for layer_str, scores in rec["layers"].items():
            matrix[i, int(layer_str), :] = scores

    return matrix, n_layers, n_heads


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_paired(
    scores_a: np.ndarray,  # (n_patterns,)
    scores_b: np.ndarray,  # (n_patterns,)
    alternative: str = "greater",
) -> Tuple[Optional[float], float]:
    """One-sided Wilcoxon signed-rank test (H1: median(A) > median(B))."""
    diffs = scores_a - scores_b
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 5:
        return None, 1.0
    stat, p = wilcoxon(nonzero, alternative=alternative)
    return float(stat), float(p)


def bonferroni_correction(p_values: np.ndarray, n_comparisons: int) -> np.ndarray:
    return np.minimum(1.0, p_values * n_comparisons)


def sig_label(p: float) -> str:
    if math.isnan(p): return "nan"
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare_conditions(
    baseline_path: Path,
    compare_paths: List[Path],
    top_k:         int,
    threshold:     float,
) -> None:
    baseline_data    = load_json(baseline_path)
    baseline_records = baseline_data.get("records", [])
    base_mat, n_layers, n_heads = extract_pms_matrix(baseline_records)

    # Mean baseline PMS per (layer, head)
    base_mean = base_mat.mean(axis=0)  # (n_layers, n_heads)

    # Undirected top-K: highest mean baseline PMS globally
    flat_idx   = np.argsort(base_mean.ravel())[::-1][:top_k]
    top_k_set  = {(idx // n_heads, idx % n_heads) for idx in flat_idx}

    for compare_path in compare_paths:
        cmp_data    = load_json(compare_path)
        cmp_records = cmp_data.get("records", [])
        cmp_cond    = cmp_data.get("condition", compare_path.stem)

        # Align patterns by pattern_id
        base_pids = {r["pattern_id"]: i for i, r in enumerate(baseline_records)}
        cmp_pids  = {r["pattern_id"]: i for i, r in enumerate(cmp_records)}
        common    = sorted(set(base_pids) & set(cmp_pids))

        if not common:
            print(f"No common patterns between baseline and {cmp_cond}.")
            continue

        base_sub = np.stack([base_mat[base_pids[pid]] for pid in common])  # (N, L, H)
        cmp_sub  = np.stack([
            np.zeros((n_layers, n_heads)) if pid not in cmp_pids
            else extract_pms_matrix(cmp_records)[0][cmp_pids[pid]]
            for pid in common
        ])

        n_comparisons = n_layers * n_heads
        print(f"\n{'='*70}")
        print(f"Baseline vs {cmp_cond}  (N={len(common)} patterns, Bonferroni n={n_comparisons})")
        print(f"Top-{top_k} induction heads (undirected, by baseline PMS):")
        print(f"{'Layer':>6} {'Head':>5} {'Base mean':>11} {'Cmp mean':>10} {'stat':>8} {'p_raw':>9} {'p_Bonf':>9} {'sig':>4}")
        print("-" * 68)

        for layer, head in sorted(top_k_set):
            a = base_sub[:, layer, head]
            b = cmp_sub[:, layer, head]
            stat, p_raw = wilcoxon_paired(a, b)
            p_bonf      = min(1.0, p_raw * n_comparisons)
            stat_s      = f"{stat:.1f}" if stat is not None else "–"
            print(
                f"{layer:>6} {head:>5} {a.mean():>11.6f} {b.mean():>10.6f} "
                f"{stat_s:>8} {p_raw:>9.4f} {p_bonf:>9.4f} {sig_label(p_bonf):>4}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wilcoxon + Bonferroni tests for PMS comparison.")
    parser.add_argument("--pms-dir",       required=True,
                        help="Directory containing pms_*.json files")
    parser.add_argument("--baseline-file", default="pms_baseline.json")
    parser.add_argument("--compare-files", nargs="+",
                        default=["pms_w_o_causality.json", "pms_w_o_agency.json",
                                 "pms_w_o_time_series.json"])
    parser.add_argument("--top-k",         type=int, default=20)
    parser.add_argument("--threshold",     type=float, default=0.05,
                        help="Minimum baseline PMS for directed criterion")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    pms_dir = Path(args.pms_dir)

    baseline_path  = pms_dir / args.baseline_file
    compare_paths  = [pms_dir / f for f in args.compare_files if (pms_dir / f).exists()]

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    compare_conditions(baseline_path, compare_paths, args.top_k, args.threshold)


if __name__ == "__main__":
    main()
