"""cross_condition.py – Cross-condition probe transfer (§6).

Trains the time probe on the baseline condition and evaluates it on
w_o_time_series (and vice versa).  A large accuracy drop in the
baseline→w/o-time-series transfer indicates that the model's internal
temporal encoding is disrupted by temporal shuffling.

This script reads .npz feature files produced by extract_hidden_states.py.

Usage
-----
    python src/probing/cross_condition.py \\
        --features-dir results/probing/Meta-Llama-3.1-8B-Instruct/ \\
        --train-condition baseline \\
        --test-condition  w_o_time_series \\
        --pca-dim 256 \\
        --out results/probing/Meta-Llama-3.1-8B-Instruct/cross_time.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from probes import (
    apply_pca,
    eval_binary,
    fit_logistic,
    load_npz_features,
    make_pairwise_samples,
    train_test_split_by_pattern,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def save_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_cross_condition_time_probe(
    features_dir: Path,
    train_cond: str,
    test_cond:  str,
    all_pids:   List[int],
    layer: int,
    pca_dim: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train on train_cond (full pattern set), evaluate on test_cond."""
    train_dir = features_dir / f"features_{train_cond}"
    test_dir  = features_dir / f"features_{test_cond}"

    X_train, meta_tr = load_npz_features(train_dir, all_pids, layer, "layer")
    X_test,  meta_te = load_npz_features(test_dir,  all_pids, layer, "layer")

    if X_train.size == 0 or X_test.size == 0:
        return {"error": "no data"}

    n_events  = 10
    n_tr_pats = X_train.shape[0] // n_events
    n_te_pats = X_test.shape[0]  // n_events

    ranks_tr = meta_tr.get("true_ranks", np.zeros(X_train.shape[0]))
    ranks_te = meta_te.get("true_ranks", np.zeros(X_test.shape[0]))

    Xp_tr, yp_tr = make_pairwise_samples(X_train, ranks_tr, n_tr_pats, seed=seed)
    Xp_te, yp_te = make_pairwise_samples(X_test,  ranks_te, n_te_pats, seed=seed)

    Xp_tr_pca, Xp_te_pca, var = apply_pca(Xp_tr, Xp_te, pca_dim)
    w, b = fit_logistic(Xp_tr_pca, yp_tr)
    y_pred = (Xp_te_pca @ w + b.squeeze() > 0).astype(np.int8)

    return {
        **eval_binary(yp_te, y_pred),
        "train_condition": train_cond,
        "test_condition":  test_cond,
        "pca_var_explained": var,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-condition probe transfer for time ordering.")
    parser.add_argument("--features-dir",    required=True)
    parser.add_argument("--train-condition", default="baseline")
    parser.add_argument("--test-condition",  default="w_o_time_series")
    parser.add_argument("--pca-dim",         type=int, default=256)
    parser.add_argument("--layer-start",     type=int, default=0)
    parser.add_argument("--layer-end",       type=int, default=31)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--out",             required=True)
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    feat_dir = Path(args.features_dir)

    # Collect all pattern IDs
    cond_dir = feat_dir / f"features_{args.train_condition}"
    all_pids = sorted(int(p.stem.split("_")[1]) for p in cond_dir.glob("pattern_*.npz"))
    print(f"Found {len(all_pids)} patterns in {cond_dir}")

    layer_results: List[Dict] = []
    for layer in range(args.layer_start, args.layer_end + 1):
        print(f"  Layer {layer} … ", end="", flush=True)
        result = run_cross_condition_time_probe(
            feat_dir, args.train_condition, args.test_condition,
            all_pids, layer, args.pca_dim, args.seed,
        )
        result["layer"] = layer
        layer_results.append(result)
        print(f"acc={result.get('accuracy', float('nan')):.3f}")

    output = {
        "train_condition": args.train_condition,
        "test_condition":  args.test_condition,
        "pca_dim":         args.pca_dim,
        "seed":            args.seed,
        "layers":          layer_results,
    }
    save_json_atomic(Path(args.out), output)
    print(f"Saved cross-condition results → {args.out}")


if __name__ == "__main__":
    main()
