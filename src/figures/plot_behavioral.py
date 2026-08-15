"""plot_behavioral.py – Figures for single-hop and multi-hop QA results.

Reads eval_{single,multi}.json from results/ and produces:
  - Bar charts of mean accuracy per condition (Figure 1 equivalent)
  - Per-question-type breakdown (Figure 2 equivalent)
  - Multi-hop delta heatmap (Figure 3 equivalent)

Usage
-----
    python src/figures/plot_behavioral.py \\
        --eval-dirs results/:Llama-8B results_70B/:Llama-70B \\
        --out-dir   figures/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS  = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]
COND_LABELS = {
    "baseline":        "Baseline",
    "w_o_causality":   "w/o Causality",
    "w_o_time_series": "w/o TimeSeries",
    "w_o_agency":      "w/o Agency",
}
COLORS = {
    "baseline":        "#4C72B0",
    "w_o_causality":   "#DD8452",
    "w_o_time_series": "#55A868",
    "w_o_agency":      "#C44E52",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_mean_accuracy(results: List[Dict]) -> Dict[str, float]:
    by_cond: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_cond[r["variant"]].append(r["score"])
    return {cond: float(np.mean(scores)) for cond, scores in by_cond.items()}


def plot_accuracy_bars(
    model_label: str,
    mean_accs_single: Dict[str, float],
    mean_accs_multi:  Dict[str, float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, mean_accs, title in zip(
        axes,
        [mean_accs_single, mean_accs_multi],
        ["Single-hop QA", "Multi-hop QA"],
    ):
        conds  = [c for c in CONDITIONS if c in mean_accs]
        values = [mean_accs[c] for c in conds]
        labels = [COND_LABELS[c] for c in conds]
        colors = [COLORS[c] for c in conds]

        bars = ax.bar(range(len(conds)), values, color=colors, width=0.6)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean accuracy")
        ax.set_title(f"{model_label} – {title}")
        ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.8, label="Chance")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_qtype_breakdown(
    model_label: str,
    results:     List[Dict],
    hop_type:    str,
    out_path:    Path,
) -> None:
    """Per-question-type accuracy heatmap."""
    qtypes = sorted({r["question_type"] for r in results})
    by_ct: Dict[tuple, List[float]] = defaultdict(list)
    for r in results:
        by_ct[(r["variant"], r["question_type"])].append(r["score"])

    matrix = np.array([
        [float(np.mean(by_ct[(cond, qt)])) if by_ct[(cond, qt)] else float("nan")
         for qt in qtypes]
        for cond in CONDITIONS
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(qtypes) * 0.8), 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(qtypes)))
    ax.set_xticklabels(qtypes, fontsize=9)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_title(f"{model_label} – {hop_type} by question type")
    plt.colorbar(im, ax=ax, fraction=0.03)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot behavioral QA results.")
    parser.add_argument("--eval-dirs", nargs="+", required=True,
                        help='"results_dir:ModelLabel" pairs')
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)

    for spec in args.eval_dirs:
        if ":" in spec:
            dir_str, label = spec.rsplit(":", 1)
        else:
            dir_str, label = spec, Path(spec).name
        d = Path(dir_str)

        single_path = d / "eval_single.json"
        multi_path  = d / "eval_multi.json"

        mean_single = {}
        mean_multi  = {}
        results_single: List[Dict] = []
        results_multi:  List[Dict] = []

        if single_path.exists():
            data           = load_json(single_path)
            results_single = data.get("results", [])
            mean_single    = compute_mean_accuracy(results_single)

        if multi_path.exists():
            data          = load_json(multi_path)
            results_multi = data.get("results", [])
            mean_multi    = compute_mean_accuracy(results_multi)

        slug = label.replace(" ", "_")
        plot_accuracy_bars(label, mean_single, mean_multi,
                           out_dir / f"behavioral_accuracy_{slug}.pdf")

        if results_single:
            plot_qtype_breakdown(label, results_single, "single-hop",
                                 out_dir / f"behavioral_qtype_single_{slug}.pdf")
        if results_multi:
            plot_qtype_breakdown(label, results_multi, "multi-hop",
                                 out_dir / f"behavioral_qtype_multi_{slug}.pdf")


if __name__ == "__main__":
    main()
