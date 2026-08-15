"""plot_probing.py – Figures for linear probe accuracy across layers.

Reads probe_layer_*.json files from results/probing/<model>/ and produces:
  - Line plots of balanced accuracy / accuracy vs. layer (one per probe type)
  - Cross-condition transfer plots for the time probe

Usage
-----
    python src/figures/plot_probing.py \\
        --probe-dir results/probing/Meta-Llama-3.1-8B-Instruct/ \\
        --out-dir   figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]
COND_COLORS = {
    "baseline":        "#4C72B0",
    "w_o_causality":   "#DD8452",
    "w_o_time_series": "#55A868",
    "w_o_agency":      "#C44E52",
}
COND_LABELS = {
    "baseline":        "Baseline",
    "w_o_causality":   "w/o Causality",
    "w_o_time_series": "w/o TimeSeries",
    "w_o_agency":      "w/o Agency",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_layer_metric(
    layer_files: List[Path],
    probe_type: str,
    cond: str,
    metric: str = "balanced_accuracy",
    subkey: Optional[str] = None,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for lf in layer_files:
        data  = load_json(lf)
        layer = data.get("layer", int(lf.stem.split("_")[-1]))
        probe_data = data.get(probe_type, {}).get(cond)
        if probe_data is None:
            continue
        if subkey:
            probe_data = probe_data.get(subkey)
        if probe_data is None:
            continue
        val = probe_data.get(metric)
        if val is not None and not (isinstance(val, float) and val != val):
            result[layer] = float(val)
    return result


def plot_probe_lines(
    probe_dir:  Path,
    probe_type: str,
    conditions: List[str],
    metric:     str,
    title:      str,
    out_path:   Path,
    subkey:     Optional[str] = None,
) -> None:
    layer_files = sorted(probe_dir.glob("probe_layer_*.json"))
    if not layer_files:
        print(f"No probe files in {probe_dir}")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for cond in conditions:
        data = collect_layer_metric(layer_files, probe_type, cond, metric, subkey)
        if not data:
            continue
        layers = sorted(data.keys())
        values = [data[l] for l in layers]
        ax.plot(layers, values, label=COND_LABELS.get(cond, cond),
                color=COND_COLORS.get(cond, "grey"), linewidth=1.5, marker="o", markersize=3)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (binary)")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot linear probe results.")
    parser.add_argument("--probe-dir", required=True,
                        help="Directory with probe_layer_*.json files")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--model-tag", default="model",
                        help="Label used in plot titles and filenames")
    return parser.parse_args()


# For type hints
from typing import Optional


def main() -> None:
    args      = parse_args()
    probe_dir = Path(args.probe_dir)
    out_dir   = Path(args.out_dir)
    tag       = args.model_tag

    # Time probe
    plot_probe_lines(
        probe_dir, "time", ["baseline", "w_o_time_series"],
        "balanced_accuracy", f"{tag} – Time probe (balanced accuracy)",
        out_dir / f"probe_time_{tag}.pdf",
    )

    # Entity ID Local probe
    plot_probe_lines(
        probe_dir, "entity_id", ["baseline", "w_o_agency"],
        "balanced_accuracy", f"{tag} – Entity ID Local probe",
        out_dir / f"probe_entity_id_{tag}.pdf",
    )

    # Common Element Decode probe (three sub-probes)
    for sub in ("content_decode", "entity_decode", "location_decode"):
        plot_probe_lines(
            probe_dir, "common_element", ["baseline", "w_o_causality"],
            "accuracy", f"{tag} – Common Element ({sub})",
            out_dir / f"probe_common_{sub}_{tag}.pdf",
            subkey=sub,
        )


if __name__ == "__main__":
    main()
