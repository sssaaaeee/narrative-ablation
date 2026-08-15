"""plot_mechanistic.py – Figures for PMS and attention analysis (§7).

Reads pms_<condition>.json and attention_<condition>.json from
results/mechanistic/<model>/ and produces:
  - PMS heatmap (layers × heads) for each condition
  - PMS difference map (baseline − condition)
  - Top-K head bar chart comparing conditions
  - Attention entropy line plot across layers

Usage
-----
    python src/figures/plot_mechanistic.py \\
        --mech-dir results/mechanistic/Meta-Llama-3.1-8B-Instruct/ \\
        --out-dir  figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT   = Path(__file__).resolve().parents[2]
CONDITIONS  = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]
COND_LABELS = {
    "baseline":        "Baseline",
    "w_o_causality":   "w/o Causality",
    "w_o_time_series": "w/o TimeSeries",
    "w_o_agency":      "w/o Agency",
}
COND_COLORS = {
    "baseline":        "#4C72B0",
    "w_o_causality":   "#DD8452",
    "w_o_time_series": "#55A868",
    "w_o_agency":      "#C44E52",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PMS matrix extraction
# ---------------------------------------------------------------------------

def extract_mean_pms(records: List[Dict]) -> Optional[np.ndarray]:
    """Compute mean PMS matrix across patterns.  Returns (n_layers, n_heads)."""
    if not records:
        return None
    first = records[0]["layers"]
    n_layers = max(int(k) for k in first) + 1
    n_heads  = len(list(first.values())[0])

    mat = np.zeros((len(records), n_layers, n_heads), dtype=np.float32)
    for i, rec in enumerate(records):
        for layer_str, scores in rec["layers"].items():
            mat[i, int(layer_str), :] = scores
    return mat.mean(axis=0)


# ---------------------------------------------------------------------------
# PMS heatmap
# ---------------------------------------------------------------------------

def plot_pms_heatmap(
    mean_pms: np.ndarray,
    title:    str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(mean_pms.T, aspect="auto", cmap="hot_r", origin="lower")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.03, label="PMS")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_pms_diff(
    base_pms: np.ndarray,
    cmp_pms:  np.ndarray,
    title:    str,
    out_path: Path,
) -> None:
    diff = base_pms - cmp_pms
    vmax = float(np.abs(diff).max())
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(diff.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.03, label="Δ PMS (baseline − cond)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def extract_mean_entropy(records: List[Dict]) -> Dict[int, float]:
    by_layer: Dict[int, List[float]] = {}
    for rec in records:
        for layer_str, val in rec.get("entropy", {}).items():
            layer = int(layer_str)
            by_layer.setdefault(layer, []).append(val)
    return {layer: float(np.mean(vals)) for layer, vals in sorted(by_layer.items())}


def plot_entropy_lines(
    cond_entropies: Dict[str, Dict[int, float]],
    title:          str,
    out_path:       Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for cond, data in cond_entropies.items():
        if not data:
            continue
        layers = sorted(data.keys())
        values = [data[l] for l in layers]
        ax.plot(layers, values, label=COND_LABELS.get(cond, cond),
                color=COND_COLORS.get(cond, "grey"), linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean attention entropy (nats)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mechanistic analysis figures.")
    parser.add_argument("--mech-dir", required=True,
                        help="Directory with pms_*.json and attention_*.json")
    parser.add_argument("--out-dir",   default=str(REPO_ROOT / "figures"))
    parser.add_argument("--model-tag", default="model")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    mech_dir = Path(args.mech_dir)
    out_dir  = Path(args.out_dir)
    tag      = args.model_tag

    # --- PMS plots ---
    base_pms: Optional[np.ndarray] = None
    base_path = mech_dir / "pms_baseline.json"
    if base_path.exists():
        records  = load_json(base_path).get("records", [])
        base_pms = extract_mean_pms(records)
        if base_pms is not None:
            plot_pms_heatmap(base_pms, f"{tag} – PMS (Baseline)",
                             out_dir / f"pms_heatmap_baseline_{tag}.pdf")

    for cond in ["w_o_causality", "w_o_time_series", "w_o_agency"]:
        cond_path = mech_dir / f"pms_{cond}.json"
        if not cond_path.exists():
            continue
        records  = load_json(cond_path).get("records", [])
        cmp_pms  = extract_mean_pms(records)
        if cmp_pms is None:
            continue
        plot_pms_heatmap(cmp_pms, f"{tag} – PMS ({COND_LABELS[cond]})",
                         out_dir / f"pms_heatmap_{cond}_{tag}.pdf")
        if base_pms is not None and cmp_pms.shape == base_pms.shape:
            plot_pms_diff(base_pms, cmp_pms,
                          f"{tag} – ΔPMS Baseline − {COND_LABELS[cond]}",
                          out_dir / f"pms_diff_{cond}_{tag}.pdf")

    # --- Attention entropy plots ---
    cond_entropies: Dict[str, Dict[int, float]] = {}
    for cond in CONDITIONS:
        att_path = mech_dir / f"attention_{cond}.json"
        if att_path.exists():
            records = load_json(att_path).get("records", [])
            cond_entropies[cond] = extract_mean_entropy(records)

    if cond_entropies:
        plot_entropy_lines(
            cond_entropies,
            f"{tag} – Attention entropy over event-end tokens",
            out_dir / f"attention_entropy_{tag}.pdf",
        )


if __name__ == "__main__":
    main()
