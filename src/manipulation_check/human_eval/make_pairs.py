"""make_pairs.py – Create annotation pairs for human evaluation (Appendix A.5).

For each randomly sampled pattern, creates side-by-side passage pairs
(baseline vs. one ablation condition) in a randomised left/right order
to prevent position bias.  Outputs a CSV ready for crowdsourced annotation.

Usage
-----
    python src/manipulation_check/human_eval/make_pairs.py \\
        --passages data/passages/generated_story.json \\
        --n-samples 50 \\
        --seed 42 \\
        --out results/human_eval_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, List


REPO_ROOT     = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "passages" / "generated_story.json"
DEFAULT_OUT   = REPO_ROOT / "results" / "human_eval_pairs.csv"

COMPARE_CONDITIONS = ["w_o_causality", "w_o_time_series", "w_o_agency"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def passages_to_text(entries: List[dict]) -> str:
    return "\n\n".join(e.get("text", "") for e in entries)


def make_pairs(patterns: List[dict], n_samples: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    sampled = rng.sample(patterns, min(n_samples, len(patterns)))
    rows: List[dict] = []

    for pattern in sampled:
        pid       = pattern["pattern_id"]
        base_text = passages_to_text(pattern["baseline"])

        for condition in COMPARE_CONDITIONS:
            cond_text = passages_to_text(pattern[condition])

            # Randomise which side baseline appears on
            if rng.random() < 0.5:
                left_text, right_text = base_text, cond_text
                left_label, right_label = "baseline", condition
            else:
                left_text, right_text = cond_text, base_text
                left_label, right_label = condition, "baseline"

            rows.append({
                "pair_id":     f"{pid}_{condition}",
                "pattern_id":  pid,
                "condition":   condition,
                "left_label":  left_label,
                "right_label": right_label,
                "left_text":   left_text,
                "right_text":  right_text,
                "annotator_choice": "",  # filled in by annotator: left / right / equal
                "annotator_notes":  "",
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages",  default=str(DEFAULT_INPUT))
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       default=str(DEFAULT_OUT))
    args = parser.parse_args()

    data     = load_json(Path(args.passages))
    patterns = data.get("patterns", [])
    rows     = make_pairs(patterns, args.n_samples, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pairs → {out_path}")


if __name__ == "__main__":
    main()
