"""probes.py – Linear probe training and evaluation (§6).

Three probes are implemented:

  1. Time probe  (pairwise binary classification)
     Feature: h_i - h_j (difference of event hidden states)
     Label:   1 if event i is chronologically earlier than event j, 0 otherwise
     Evaluated on: baseline, w_o_time_series

  2. Entity ID Local probe  (character recurrence)
     Feature: hidden state at entity= token in each event's header
     Label:   0 = entity first appearance, 1 = entity already seen earlier
     Evaluated on: baseline, w_o_agency

  3. Common Element Decode probe  (multi-class identification of shared field)
     Feature: hidden state at the shared-field token of event N+1
     Label:   content_id / entity_id / location_id of the previous event N
     Three separate probes per non-shared field; results are macro-averaged.
     Evaluated on: baseline, w_o_causality

All probes use PCA (256 components) for dimensionality reduction followed by
L2-regularised logistic regression (L-BFGS).  Pattern-level train/test split
(80/20) prevents data leakage.

Usage
-----
    python src/probing/probes.py \\
        --features-dir results/probing/Meta-Llama-3.1-8B-Instruct/ \\
        --probe-type all \\
        --conditions baseline w_o_causality w_o_agency w_o_time_series \\
        --pca-dim 256 \\
        --out-dir results/probing/Meta-Llama-3.1-8B-Instruct/
"""

from __future__ import annotations

import argparse
import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]

ALL_CONDITIONS       = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]
FIELD_TYPE_MAP       = {"location": 0, "entity": 1, "content": 2}
FIELD_TYPE_INV       = {0: "location", 1: "entity", 2: "content"}


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
# Pattern-level train/test split
# ---------------------------------------------------------------------------

def train_test_split_by_pattern(
    pattern_ids: List[int],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[set, set]:
    """Split pattern IDs: prevents leakage across train/test sets."""
    ids = list(pattern_ids)
    random.Random(seed).shuffle(ids)
    split = int(len(ids) * (1 - test_ratio))
    return set(ids[:split]), set(ids[split:])


# ---------------------------------------------------------------------------
# PCA (numpy SVD-based)
# ---------------------------------------------------------------------------

def apply_pca(
    X_train: np.ndarray, X_test: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit PCA on training data, transform both sets."""
    mean      = X_train.mean(axis=0)
    X_tr_c    = X_train - mean
    X_te_c    = X_test  - mean
    _, S, Vt  = np.linalg.svd(X_tr_c, full_matrices=False)
    n_comp    = min(n_components, len(S))
    comps     = Vt[:n_comp]
    var_expl  = float((S[:n_comp] ** 2).sum() / ((S ** 2).sum() + 1e-12))
    return X_tr_c @ comps.T, X_te_c @ comps.T, var_expl


# ---------------------------------------------------------------------------
# Logistic regression (L-BFGS, L2)
# ---------------------------------------------------------------------------

def fit_logistic(
    X_train: np.ndarray, y_train: np.ndarray,
    C: float = 1.0, max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = X_train.shape
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train.astype(np.float32))
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, tolerance_grad=1e-5)
    lam = 1.0 / (2.0 * C * n)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
        (loss + lam * (w ** 2).sum()).backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def fit_multiclass(
    X_train: np.ndarray, y_train: np.ndarray,
    n_classes: int, C: float = 1.0, max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = X_train.shape
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train.astype(np.int64))
    W = torch.zeros(n_classes, d, requires_grad=True)
    b = torch.zeros(n_classes,    requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, tolerance_grad=1e-5)
    lam = 1.0 / (2.0 * C * n)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W.T + b, y)
        (loss + lam * (W ** 2).sum()).backward()
        return loss

    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = float((y_true == y_pred).mean())
    tp  = int(((y_pred == 1) & (y_true == 1)).sum())
    tn  = int(((y_pred == 0) & (y_true == 0)).sum())
    fp  = int(((y_pred == 1) & (y_true == 0)).sum())
    fn  = int(((y_pred == 0) & (y_true == 1)).sum())
    p   = tp / (tp + fp + 1e-9)
    r   = tp / (tp + fn + 1e-9)
    f1  = 2 * p * r / (p + r + 1e-9)
    ba  = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
    return {
        "accuracy": acc, "balanced_accuracy": float(ba), "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "correct_mask": (y_true == y_pred).tolist(),
    }


def eval_multiclass(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, Any]:
    acc = float((y_true == y_pred).mean())
    f1s: List[float] = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    return {
        "accuracy": acc, "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "chance": 1.0 / max(n_classes, 1),
        "correct_mask": (y_true == y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Data loading from .npz files
# ---------------------------------------------------------------------------

def load_npz_features(
    cond_dir: Path,
    pattern_ids: List[int],
    layer: int,
    feature_key: str = "layer",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load hidden states + ancillary arrays for one layer and condition."""
    X_list:       List[np.ndarray] = []
    meta:         Dict[str, List]  = {}
    loaded_pids:  List[int] = []

    for pid in pattern_ids:
        npz_path = cond_dir / f"pattern_{pid}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        key  = f"{feature_key}_{layer}"
        if key not in data:
            continue
        X_list.append(data[key].astype(np.float32))
        loaded_pids.append(pid)
        for k in data.files:
            if k.startswith("layer_") or k.startswith("entity_layer_") or k.startswith("shared_layer_"):
                continue
            meta.setdefault(k, []).append(data[k])

    if not X_list:
        return np.empty((0,)), {}

    X    = np.concatenate(X_list, axis=0)
    meta = {k: np.concatenate(v, axis=0) for k, v in meta.items()}
    meta["_pattern_ids"] = np.array(loaded_pids)
    return X, meta


# ---------------------------------------------------------------------------
# Probe 1: Time probe
# ---------------------------------------------------------------------------

def make_pairwise_samples(
    X: np.ndarray,           # (N_patterns * 10, hidden_dim)
    ranks: np.ndarray,       # (N_patterns * 10,)
    n_patterns: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build pairwise difference features with randomised slot assignment."""
    X_pats     = X.reshape(n_patterns, 10, -1)
    ranks_pats = ranks.reshape(n_patterns, 10)
    pair_idxs  = list(combinations(range(10), 2))
    Xp, yp = [], []
    rng = np.random.default_rng(seed)
    for n in range(n_patterns):
        for i, j in pair_idxs:
            flip = bool(rng.integers(0, 2))
            if flip:
                feat  = X_pats[n, j] - X_pats[n, i]
                label = 1 if ranks_pats[n, j] < ranks_pats[n, i] else 0
            else:
                feat  = X_pats[n, i] - X_pats[n, j]
                label = 1 if ranks_pats[n, i] < ranks_pats[n, j] else 0
            Xp.append(feat)
            yp.append(label)
    return np.array(Xp, dtype=np.float32), np.array(yp, dtype=np.int8)


def run_time_probe(
    features_dir: Path,
    conditions: List[str],
    train_pids: set,
    test_pids:  set,
    layer: int,
    pca_dim: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for cond in conditions:
        cond_dir = features_dir / f"features_{cond}"
        all_pids = sorted(train_pids | test_pids)

        X_all, meta = load_npz_features(cond_dir, all_pids, layer, "layer")
        if X_all.size == 0:
            continue

        ranks = meta.get("true_ranks", np.zeros(len(X_all)))
        pids  = meta.get("_pattern_ids", np.array([]))
        n_events = 10

        tr_mask = np.isin(np.repeat(pids, n_events), list(train_pids))
        te_mask = ~tr_mask

        Xp_tr, yp_tr = make_pairwise_samples(X_all[tr_mask], ranks[tr_mask], tr_mask.sum() // n_events)
        Xp_te, yp_te = make_pairwise_samples(X_all[te_mask], ranks[te_mask], te_mask.sum() // n_events)

        Xp_tr_pca, Xp_te_pca, var = apply_pca(Xp_tr, Xp_te, pca_dim)
        w, b = fit_logistic(Xp_tr_pca, yp_tr)
        y_pred = (Xp_te_pca @ w + b.squeeze() > 0).astype(np.int8)
        results[cond] = {**eval_binary(yp_te, y_pred), "pca_var_explained": var}

    return results


# ---------------------------------------------------------------------------
# Probe 2: Entity ID Local (character recurrence)
# ---------------------------------------------------------------------------

def run_entity_id_probe(
    features_dir: Path,
    conditions: List[str],
    train_pids: set,
    test_pids:  set,
    layer: int,
    pca_dim: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for cond in conditions:
        cond_dir = features_dir / f"features_{cond}"
        all_pids = sorted(train_pids | test_pids)

        X_all, meta = load_npz_features(cond_dir, all_pids, layer, "entity_layer")
        if X_all.size == 0:
            continue

        labels = meta.get("recurrence_labels", np.zeros(len(X_all)))
        pids   = meta.get("_pattern_ids", np.array([]))
        n_events = 10

        tr_mask = np.isin(np.repeat(pids, n_events), list(train_pids))
        te_mask = ~tr_mask

        X_tr_pca, X_te_pca, var = apply_pca(X_all[tr_mask], X_all[te_mask], pca_dim)
        w, b = fit_logistic(X_tr_pca, labels[tr_mask])
        y_pred = (X_te_pca @ w + b.squeeze() > 0).astype(np.int8)
        results[cond] = {**eval_binary(labels[te_mask], y_pred), "pca_var_explained": var}

    return results


# ---------------------------------------------------------------------------
# Probe 3: Common Element Decode
# ---------------------------------------------------------------------------

def run_common_element_probe(
    features_dir: Path,
    conditions: List[str],
    train_pids: set,
    test_pids:  set,
    layer: int,
    pca_dim: int,
) -> Dict[str, Any]:
    TARGET_KEYS = {
        "content":  "shared_pair_prev_content_id",
        "entity":   "shared_pair_prev_entity_id",
        "location": "shared_pair_prev_location_id",
    }
    results: Dict[str, Any] = {}
    for cond in conditions:
        cond_dir = features_dir / f"features_{cond}"
        all_pids = sorted(train_pids | test_pids)

        X_all, meta = load_npz_features(cond_dir, all_pids, layer, "shared_layer")
        if X_all.size == 0:
            continue

        pids         = meta.get("_pattern_ids", np.array([]))
        field_types  = meta.get("shared_pair_field_type", np.full(len(X_all), -1))
        n_pairs = 9  # 10 events → 9 consecutive pairs

        tr_mask = np.isin(np.repeat(pids, n_pairs), list(train_pids))
        te_mask = ~tr_mask

        sub_results: Dict[str, Any] = {}
        for target_field, id_key in TARGET_KEYS.items():
            labels = meta.get(id_key, np.full(len(X_all), -1))
            # Only use pairs where shared field is NOT the target field
            target_ftype = FIELD_TYPE_MAP[target_field]
            valid = field_types != target_ftype

            X_tr_v  = X_all[tr_mask & valid]
            y_tr_v  = labels[tr_mask & valid]
            X_te_v  = X_all[te_mask & valid]
            y_te_v  = labels[te_mask & valid]

            valid_ids = np.unique(y_tr_v[y_tr_v >= 0])
            if len(valid_ids) < 2:
                continue

            id_to_cls = {v: i for i, v in enumerate(sorted(valid_ids))}
            tr_valid  = np.isin(y_tr_v, list(id_to_cls))
            te_valid  = np.isin(y_te_v, list(id_to_cls))

            if not tr_valid.any() or not te_valid.any():
                continue

            y_tr_cls = np.array([id_to_cls[v] for v in y_tr_v[tr_valid]], dtype=np.int32)
            y_te_cls = np.array([id_to_cls[v] for v in y_te_v[te_valid]], dtype=np.int32)

            X_tr_pca, X_te_pca, var = apply_pca(X_tr_v[tr_valid], X_te_v[te_valid], pca_dim)
            W, b = fit_multiclass(X_tr_pca, y_tr_cls, n_classes=len(id_to_cls))
            y_pred = np.argmax(X_te_pca @ W.T + b, axis=1).astype(np.int32)
            sub_results[f"{target_field}_decode"] = {
                **eval_multiclass(y_te_cls, y_pred, len(id_to_cls)),
                "pca_var_explained": var,
            }

        results[cond] = sub_results

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate linear probes.")
    parser.add_argument("--features-dir", required=True,
                        help="Directory containing features_<condition>/ subdirs")
    parser.add_argument("--probe-type", choices=["all", "time", "entity_id", "common_element"],
                        default="all")
    parser.add_argument("--conditions",  nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--pca-dim",     type=int, default=256)
    parser.add_argument("--test-ratio",  type=float, default=0.2)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end",   type=int, default=31)
    parser.add_argument("--out-dir",     required=True)
    return parser.parse_args()


def main() -> None:
    args        = parse_args()
    feat_dir    = Path(args.features_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all pattern IDs from existing .npz files
    sample_cond = args.conditions[0]
    cond_dir    = feat_dir / f"features_{sample_cond}"
    all_pids    = sorted(int(p.stem.split("_")[1]) for p in cond_dir.glob("pattern_*.npz"))
    train_pids, test_pids = train_test_split_by_pattern(all_pids, args.test_ratio, args.seed)

    for layer in range(args.layer_start, args.layer_end + 1):
        print(f"Layer {layer} …", end=" ", flush=True)
        row: Dict[str, Any] = {"layer": layer}

        if args.probe_type in ("all", "time"):
            row["time"] = run_time_probe(feat_dir, args.conditions, train_pids, test_pids, layer, args.pca_dim)

        if args.probe_type in ("all", "entity_id"):
            row["entity_id"] = run_entity_id_probe(feat_dir, args.conditions, train_pids, test_pids, layer, args.pca_dim)

        if args.probe_type in ("all", "common_element"):
            row["common_element"] = run_common_element_probe(feat_dir, args.conditions, train_pids, test_pids, layer, args.pca_dim)

        out_path = out_dir / f"probe_layer_{layer:03d}.json"
        save_json_atomic(out_path, row)
        print("done")

    print(f"Probe results saved to {out_dir}")


if __name__ == "__main__":
    main()
