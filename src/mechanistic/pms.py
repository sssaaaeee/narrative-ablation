"""pms.py – Prefix Matching Score (PMS) computation for induction heads (§7).

Computes the Prefix Matching Score (Olsson et al., 2022) for each attention
head across all layers.  A high PMS indicates that a head implements the
"induction" behaviour: attending to the token immediately following a
previous occurrence of the current token.

PMS definition
--------------
  PMS(head h, layer l) = (1/N) Σ_i  mean_{j≤i-2, token[j]==token[i]}  attn[h, i, j+1]

where N is the number of positions i with at least one valid predecessor.

The score is computed for each pattern × condition; per-head averages across
patterns are saved for downstream statistical analysis (pms_stats.py).

Output: results/mechanistic/<model>/pms_<condition>.json

Usage
-----
    python src/mechanistic/pms.py \\
        --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --load-in-4bit \\
        --n-patterns 20 \\
        --conditions baseline w_o_causality w_o_agency w_o_time_series \\
        --out-dir results/mechanistic/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT      = Path(__file__).resolve().parents[2]
DEFAULT_STORY  = REPO_ROOT / "data" / "passages" / "generated_story.json"
DEFAULT_OUT    = REPO_ROOT / "results" / "mechanistic"
ALL_CONDITIONS = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]


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
# Context formatting
# ---------------------------------------------------------------------------

def build_context_text(items: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in items:
        header = (
            f"Set {item.get('set_id')}: time={item.get('temporal')} | "
            f"location={item.get('location')} | entity={item.get('entity')} | "
            f"content={item.get('content')}"
        )
        text = item.get("text", "")
        parts.append(header + "\n" + text if text else header)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Model loading  (attn_implementation="eager" required for attention weights)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, int, int]:
    """Load model with eager attention (Flash Attention 2 does not expose
    attention weight tensors required for PMS computation)."""
    load_dotenv()
    model_name: str = args.hf_model or os.getenv("HF_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=args.trust_remote_code
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif args.load_in_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    device_map = {"": 0} if quant_cfg is not None else args.device

    print(f"Loading model: {model_name}  (attn_implementation=eager)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_cfg,
        attn_implementation="eager",
    )
    model.eval()

    config     = AutoConfig.from_pretrained(model_name, token=hf_token, trust_remote_code=args.trust_remote_code)
    num_layers = config.num_hidden_layers
    num_heads  = config.num_attention_heads
    print(f"  {num_layers} layers × {num_heads} heads")
    return model, tokenizer, num_layers, num_heads


# ---------------------------------------------------------------------------
# PMS computation
# ---------------------------------------------------------------------------

def compute_pms(
    attn_weights: np.ndarray,  # (num_heads, seq_len, seq_len)
    token_ids:    np.ndarray,  # (seq_len,)
) -> np.ndarray:               # (num_heads,)
    """Compute Prefix Matching Score per head (Olsson et al., 2022).

    For each position i (≥ 2), finds all predecessor positions j (j ≤ i-2)
    where token[j] == token[i], then averages the attention weight that head h
    places on position j+1.

    PMS[h] = (1/N) Σ_i  mean_{j: token[j]==token[i], j≤i-2}  attn[h, i, j+1]
    """
    seq_len   = len(token_ids)
    num_heads = attn_weights.shape[0]
    match_mat = (token_ids[:, None] == token_ids[None, :])  # (seq_len, seq_len)

    total = np.zeros(num_heads, dtype=np.float64)
    count = 0

    for i in range(2, seq_len):
        valid_js = np.where(match_mat[i, :i - 1])[0]
        if len(valid_js) == 0:
            continue
        succ_pos     = valid_js + 1                            # j+1
        succ_attn    = attn_weights[:, i, succ_pos]            # (num_heads, k)
        total       += succ_attn.mean(axis=1)
        count       += 1

    if count > 0:
        total /= count
    return total.astype(np.float32)


def _get_layer_container(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot locate transformer layer list.")


def compute_pms_from_forward(
    model:        AutoModelForCausalLM,
    input_ids:    torch.Tensor,   # (1, seq_len)
    token_ids:    np.ndarray,     # (seq_len,)
    target_layers: Optional[set] = None,
) -> Dict[int, np.ndarray]:
    """Single forward pass (output_attentions=True); returns {layer: pms_vector}."""
    layer_container = _get_layer_container(model)
    n_layers        = len(layer_container)
    if target_layers is None:
        target_layers = set(range(n_layers))

    captured: Dict[int, np.ndarray] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            if layer_idx not in target_layers:
                return None
            if not (isinstance(output, tuple) and len(output) >= 2 and output[1] is not None):
                return None
            attn = output[1]   # (1, num_heads, seq_len, seq_len)
            full = attn[0].detach().float().cpu().numpy()
            captured[layer_idx] = compute_pms(full, token_ids)
            return (output[0], None) + output[2:]
        return hook_fn

    hooks = []
    for i, layer in enumerate(layer_container):
        attn_mod = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(make_hook(i)))

    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids.to(model.device),
                attention_mask=torch.ones_like(input_ids).to(model.device),
                output_attentions=True,
            )
    finally:
        for h in hooks:
            h.remove()

    return captured


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_pms_scores(
    model:      AutoModelForCausalLM,
    tokenizer:  AutoTokenizer,
    patterns:   List[Dict],
    conditions: List[str],
    n_patterns: int,
    out_dir:    Path,
) -> None:
    """Compute and save per-pattern PMS for all conditions."""
    selected = patterns[:n_patterns]

    for cond in conditions:
        out_path = out_dir / f"pms_{cond}.json"
        # Resume
        done_pids: set = set()
        records:   List[Dict] = []
        if out_path.exists():
            existing = load_json(out_path)
            records  = existing.get("records", [])
            done_pids = {r["pattern_id"] for r in records}
            print(f"[{cond}] Resuming: {len(done_pids)} patterns done.")

        for pattern in tqdm(selected, desc=cond):
            pid = pattern["pattern_id"]
            if pid in done_pids:
                continue

            events = pattern.get(cond)
            if not events:
                continue

            context = build_context_text(events)
            enc     = tokenizer(context, return_tensors="pt")
            input_ids = enc["input_ids"]
            token_ids = input_ids[0].numpy()

            try:
                pms_dict = compute_pms_from_forward(model, input_ids, token_ids)
            except Exception as exc:
                print(f"  Error pattern {pid}: {exc}")
                continue

            # pms_dict: {layer_idx: (num_heads,) float32 array}
            records.append({
                "pattern_id": pid,
                "layers": {
                    str(li): scores.tolist()
                    for li, scores in pms_dict.items()
                },
            })
            done_pids.add(pid)
            save_json_atomic(out_path, {"condition": cond, "records": records})

        print(f"[{cond}] Done. {len(records)} patterns → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PMS for all attention heads.")
    parser.add_argument("--hf-model",     default=None)
    parser.add_argument("--story-path",   default=str(DEFAULT_STORY))
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT))
    parser.add_argument("--conditions",   nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--n-patterns",   type=int, default=20)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, num_layers, num_heads = load_model_and_tokenizer(args)

    model_name = args.hf_model or os.getenv("HF_MODEL") or "Meta-Llama-3.1-8B-Instruct"
    model_tag  = Path(model_name).name
    out_dir    = Path(args.out_dir) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = load_json(Path(args.story_path)).get("patterns", [])
    print(f"Loaded {len(patterns)} patterns; using first {args.n_patterns}.")

    collect_pms_scores(
        model, tokenizer, patterns, args.conditions, args.n_patterns, out_dir
    )


if __name__ == "__main__":
    main()
