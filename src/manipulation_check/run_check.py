"""run_check.py – Manipulation check: Wilcoxon + Holm correction (Appendix A.5).

Verifies that each ablation condition degrades only its intended narrative
dimension (diagonal pattern in the 4 × 3 metric matrix):

  Rows:    baseline | w_o_causality | w_o_time_series | w_o_agency
  Columns: causality_score | time_series_score | agency_score

Statistical method
------------------
Paired one-sided Wilcoxon signed-rank test (H1: baseline > condition) for
each (condition × metric) cell.  Holm family-wise correction is applied
across the 9 comparisons (3 conditions × 3 metrics), producing Table 7.

Output: results/manipulation_check_results.json

Usage
-----
    python src/manipulation_check/run_check.py \\
        --passages data/passages/generated_story.json \\
        --out      results/manipulation_check_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from metrics import compute_pattern_scores


REPO_ROOT      = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = REPO_ROOT / "data" / "passages" / "generated_story.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "manipulation_check_results.json"

CONDITIONS = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]
METRICS    = ["causality_score", "time_series_score", "agency_score"]


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
# Wilcoxon + Holm
# ---------------------------------------------------------------------------

def _wilcoxon_one_sided(diffs: np.ndarray) -> Tuple[Optional[float], float, str]:
    """One-sided Wilcoxon signed-rank test: H1 = median(diffs) > 0.

    diffs = baseline_scores − condition_scores
    Returns (statistic, p_value, note).
    """
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return None, 1.0, "trivially_equal_by_design"
    if len(nonzero) < 5:
        return None, 1.0, "insufficient_nonzero_diffs"
    stat, pval = wilcoxon(nonzero, alternative="greater")
    return float(stat), float(pval), "ok"


def run_wilcoxon_tests(per_pattern: List[Dict], metric_key: str) -> Dict[str, Dict]:
    """Wilcoxon tests for each ablation condition vs baseline on *metric_key*.

    Returns a dict keyed by "{condition}_vs_baseline" with statistical results.
    Holm correction is applied across the 3 conditions simultaneously.
    """
    test_conds     = [c for c in CONDITIONS if c != "baseline"]
    baseline_vals  = np.array([p["baseline"][metric_key] for p in per_pattern])

    raw: List[Tuple] = []
    for cond in test_conds:
        cond_vals = np.array([p[cond][metric_key] for p in per_pattern])
        diffs     = baseline_vals - cond_vals
        stat, pval, note = _wilcoxon_one_sided(diffs)
        raw.append((cond, stat, pval, note))

    _, pvals_holm, _, _ = multipletests([r[2] for r in raw], method="holm")

    out: Dict[str, Dict] = {}
    for (cond, stat, pval, note), p_holm in zip(raw, pvals_holm):
        out[f"{cond}_vs_baseline"] = {
            "statistic":       stat,
            "pvalue_raw":      pval,
            "pvalue_holm":     float(p_holm),
            "significant_p05": bool(p_holm < 0.05),
            "note":            note,
        }
    return out


def run_all_tests(per_pattern: List[Dict]) -> Dict:
    """Run Wilcoxon tests across all primary metrics + supplementary."""
    extra = ["temporal_connective_score"]
    return {m: run_wilcoxon_tests(per_pattern, m) for m in METRICS + extra}


# ---------------------------------------------------------------------------
# Summary statistics (mean ± SD / median per cell)
# ---------------------------------------------------------------------------

def compute_summary(per_pattern: List[Dict]) -> Dict:
    all_keys = METRICS + ["temporal_connective_score", "unique_entity_count"]
    summary: Dict[str, Dict] = {}
    for cond in CONDITIONS:
        summary[cond] = {}
        for key in all_keys:
            vals = np.array([p[cond][key] for p in per_pattern])
            summary[cond][key] = {
                "mean":   float(np.mean(vals)),
                "sd":     float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)),
            }
    return summary


# ---------------------------------------------------------------------------
# Console display  (Table 7 equivalent)
# ---------------------------------------------------------------------------

def _sig_marker(p_holm: float) -> str:
    if p_holm < 0.001:
        return "***"
    if p_holm < 0.01:
        return "**"
    if p_holm < 0.05:
        return "*"
    return "ns"


def print_table(summary: Dict, tests: Dict) -> None:
    col_labels = {
        "causality_score":   "Causal(↓woc)",
        "time_series_score": "TimeS-τ(↓wots)",
        "agency_score":      "Agency(↓woa)",
    }
    cond_labels = {
        "baseline":        "Baseline",
        "w_o_causality":   "w/o Causality",
        "w_o_time_series": "w/o TimeSeries",
        "w_o_agency":      "w/o Agency",
    }
    print()
    print("=" * 72)
    print("MANIPULATION CHECK  –  Diagonal Matrix  (mean ± SD)")
    print("Wilcoxon signed-rank, one-sided (baseline > condition), Holm correction")
    print("Markers: *** p<.001  ** p<.01  * p<.05  ns")
    print("=" * 72)
    print(f"{'Condition':<18}" + "".join(f"{col_labels[m]:>18}" for m in METRICS))
    print("-" * 72)

    for cond in CONDITIONS:
        row = f"{cond_labels[cond]:<18}"
        for metric in METRICS:
            s    = summary[cond][metric]
            cell = f"{s['mean']:.3f}±{s['sd']:.3f}"
            if cond != "baseline":
                res    = tests[metric][f"{cond}_vs_baseline"]
                cell  += _sig_marker(res["pvalue_holm"])
            row += f"{cell:>18}"
        print(row)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manipulation check.")
    parser.add_argument("--passages", default=str(DEFAULT_INPUT))
    parser.add_argument("--out",      default=str(DEFAULT_OUTPUT))
    parser.add_argument("--spacy-model", default="en_core_web_sm",
                        help="spaCy model name (must be installed)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)

    print(f"Loading passages: {args.passages}")
    data     = load_json(Path(args.passages))
    patterns = data.get("patterns", [])
    print(f"  {len(patterns)} patterns loaded.")

    print("Computing metrics for all patterns …")
    per_pattern: List[Dict] = []
    for i, pattern in enumerate(patterns):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(patterns)}")
        scores = compute_pattern_scores(pattern, nlp)
        per_pattern.append(scores)

    print("Running statistical tests …")
    summary = compute_summary(per_pattern)
    tests   = run_all_tests(per_pattern)

    print_table(summary, tests)

    output = {
        "n_patterns":  len(per_pattern),
        "summary":     summary,
        "tests":       tests,
        "per_pattern": per_pattern,
    }
    save_json_atomic(Path(args.out), output)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
