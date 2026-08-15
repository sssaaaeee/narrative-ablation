"""aggregate.py – Aggregate human-evaluation annotations (Table 9).

Reads the completed annotation CSV produced by make_pairs.py (with
annotator_choice filled in) and outputs inter-annotator agreement and
condition-level preference summaries.

Usage
-----
    python src/manipulation_check/human_eval/aggregate.py \\
        --annotated results/human_eval_pairs_annotated.csv \\
        --out       results/human_eval_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT    = Path(__file__).resolve().parents[3]
DEFAULT_IN   = REPO_ROOT / "results" / "human_eval_pairs_annotated.csv"
DEFAULT_OUT  = REPO_ROOT / "results" / "human_eval_summary.json"

VALID_CHOICES = {"left", "right", "equal"}


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute per-condition baseline preference rates."""
    by_condition: Dict[str, Dict[str, int]] = defaultdict(lambda: {"baseline_preferred": 0, "condition_preferred": 0, "equal": 0})

    skipped = 0
    for row in rows:
        choice    = row.get("annotator_choice", "").strip().lower()
        condition = row.get("condition", "")
        left_lbl  = row.get("left_label", "")

        if choice not in VALID_CHOICES:
            skipped += 1
            continue

        if choice == "equal":
            by_condition[condition]["equal"] += 1
        elif choice == "left":
            winner = left_lbl
        else:
            winner = row.get("right_label", "")

        if choice != "equal":
            if winner == "baseline":
                by_condition[condition]["baseline_preferred"] += 1
            else:
                by_condition[condition]["condition_preferred"] += 1

    summary: Dict[str, Any] = {}
    for cond, counts in by_condition.items():
        total = sum(counts.values())
        summary[cond] = {
            **counts,
            "total": total,
            "baseline_pref_rate": counts["baseline_preferred"] / total if total else 0.0,
        }

    return {"n_rows": len(rows), "skipped": skipped, "by_condition": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", default=str(DEFAULT_IN))
    parser.add_argument("--out",       default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows    = load_csv(Path(args.annotated))
    result  = aggregate(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Aggregated {result['n_rows']} rows → {out_path}")
    for cond, stats in result["by_condition"].items():
        rate = stats["baseline_pref_rate"]
        print(f"  {cond:<20}: baseline preferred {rate:.1%}  (n={stats['total']})")


if __name__ == "__main__":
    main()
