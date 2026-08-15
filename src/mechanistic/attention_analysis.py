"""attention_analysis.py – Attention-based mechanistic analysis (§7, Appendix E–F).

Implements four complementary analyses of attention patterns across narrative
conditions:

  1. Distance profile    – mean attention weight from event-end tokens to all
                          other event-end tokens, as a function of token distance.
  2. Top-K hit rate      – fraction of times the highest-attended event-end token
                          is within the top-K events by relevance.
  3. Event Jaccard       – pairwise Jaccard similarity of the top-K attended
                          events between baseline and ablation conditions.
  4. Attention entropy   – entropy of the attention distribution over event-end
                          tokens; lower entropy → more focused retrieval.

All analyses use eager-attention models and forward hooks to extract attention
weight matrices without storing the full (layers × heads × seq × seq) tensor.

Output: results/mechanistic/<model>/attention_<condition>.json

Usage
-----
    python src/mechanistic/attention_analysis.py \\
        --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --load-in-4bit \\
        --analysis all \\
        --conditions baseline w_o_causality w_o_time_series w_o_agency \\
        --n-patterns 20
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
# Context construction (identical to run_qa.py format)
# ---------------------------------------------------------------------------

def build_context_text(items: List[Dict]) -> str:
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


def build_context_with_spans(items: List[Dict]) -> Tuple[str, List[Dict]]:
    """Return full context + character-span metadata for each event."""
    parts: List[str] = []
    for item in items:
        header = (
            f"Set {item.get('set_id')}: time={item.get('temporal')} | "
            f"location={item.get('location')} | entity={item.get('entity')} | "
            f"content={item.get('content')}"
        )
        text = item.get("text", "")
        parts.append(header + "\n" + text if text else header)

    segments: List[Dict] = []
    cursor = 0
    for i, part in enumerate(parts):
        start = cursor
        if i > 0:
            cursor += 2  # "\n\n" separator
            start   = cursor
        end    = start + len(part)
        cursor = end
        segments.append({"set_id": items[i].get("set_id"), "start": start, "end": end})

    return "\n\n".join(parts), segments


# ---------------------------------------------------------------------------
# Model loading (eager attention required)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, int, int]:
    load_dotenv()
    model_name: str = args.hf_model or os.getenv("HF_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_cfg,
        attn_implementation="eager",
    )
    model.eval()

    config     = AutoConfig.from_pretrained(model_name, token=hf_token)
    num_layers = config.num_hidden_layers
    num_heads  = config.num_attention_heads
    print(f"  {model_name}: {num_layers} layers × {num_heads} heads")
    return model, tokenizer, num_layers, num_heads


# ---------------------------------------------------------------------------
# Event-end token position mapping
# ---------------------------------------------------------------------------

def find_event_end_positions(
    tokenizer: AutoTokenizer,
    full_text: str,
    segments:  List[Dict],
) -> Tuple[torch.Tensor, List[int]]:
    """Return input_ids and last-token position per event."""
    try:
        enc     = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        positions = [
            max((k for k, (s, _) in enumerate(offsets) if s < seg["end"]), default=0)
            for seg in segments
        ]
        return enc["input_ids"], positions
    except Exception:
        all_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
        positions = [
            tokenizer(full_text[: seg["end"]], return_tensors="pt")["input_ids"].shape[1] - 1
            for seg in segments
        ]
        return all_ids, positions


# ---------------------------------------------------------------------------
# Attention extraction hook
# ---------------------------------------------------------------------------

def extract_event_attention_maps(
    model:          AutoModelForCausalLM,
    input_ids:      torch.Tensor,
    event_positions: List[int],
    target_layers:  set,
) -> Dict[int, np.ndarray]:
    """Extract 10×10 event-to-event attention maps for target layers.

    Returns {layer_idx: ndarray (num_heads, 10, 10)}.
    The (i, j) entry is the attention from event i's end-token to event j's
    end-token.
    """
    n_events  = len(event_positions)
    captured:  Dict[int, np.ndarray] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            if layer_idx not in target_layers:
                return None
            if not (isinstance(output, tuple) and len(output) >= 2 and output[1] is not None):
                return None
            # attn_weights: (1, num_heads, seq_len, seq_len)
            attn   = output[1][0].detach().float().cpu().numpy()
            # Extract sub-matrix at event-end positions
            rows   = np.array(event_positions)
            cols   = np.array(event_positions)
            matrix = attn[:, rows[:, None], cols]    # (num_heads, 10, 10)
            captured[layer_idx] = matrix
            return (output[0], None) + output[2:]
        return hook_fn

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_container = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_container = model.transformer.h
    else:
        raise RuntimeError("Cannot locate transformer layers.")

    hooks = []
    for i, layer in enumerate(layer_container):
        attn_mod = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn_mod and i in target_layers:
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
# Analysis 1 – Attention entropy over event-end tokens
# ---------------------------------------------------------------------------

def compute_entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability vector; ignores zeros."""
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_attention_entropy(attn_maps: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """Mean entropy of attention over event-end tokens, averaged over heads.

    attn_maps: {layer: (num_heads, 10, 10)}
    Returns per-layer mean entropy.
    """
    result: Dict[str, float] = {}
    for layer, maps in sorted(attn_maps.items()):
        # maps: (num_heads, 10, 10)
        # For each head and each query event, compute entropy of row (attention distribution)
        entropies = np.array([
            [compute_entropy(maps[h, i, :]) for i in range(maps.shape[1])]
            for h in range(maps.shape[0])
        ])  # (num_heads, 10)
        result[str(layer)] = float(entropies.mean())
    return result


# ---------------------------------------------------------------------------
# Analysis 2 – Event-to-event attention map (10 × 10 averaged)
# ---------------------------------------------------------------------------

def aggregate_attention_maps(attn_maps: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """Head-averaged 10×10 event attention matrix per layer."""
    return {
        str(layer): maps.mean(axis=0).tolist()   # (10, 10)
        for layer, maps in sorted(attn_maps.items())
    }


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_attention_analysis(
    model:       AutoModelForCausalLM,
    tokenizer:   AutoTokenizer,
    patterns:    List[Dict],
    conditions:  List[str],
    n_patterns:  int,
    out_dir:     Path,
    target_layers: set,
) -> None:
    selected = patterns[:n_patterns]

    for cond in conditions:
        out_path = out_dir / f"attention_{cond}.json"
        done_pids: set = set()
        records:   List[Dict] = []
        if out_path.exists():
            existing  = load_json(out_path)
            records   = existing.get("records", [])
            done_pids = {r["pattern_id"] for r in records}
            print(f"[{cond}] Resuming: {len(done_pids)} patterns done.")

        for pattern in tqdm(selected, desc=cond):
            pid = pattern["pattern_id"]
            if pid in done_pids:
                continue

            events = pattern.get(cond)
            if not events:
                continue

            full_text, segments = build_context_with_spans(events)
            input_ids, event_positions = find_event_end_positions(tokenizer, full_text, segments)

            try:
                attn_maps = extract_event_attention_maps(
                    model, input_ids, event_positions, target_layers
                )
            except Exception as exc:
                print(f"  Error pattern {pid}: {exc}")
                continue

            records.append({
                "pattern_id":     pid,
                "entropy":        compute_attention_entropy(attn_maps),
                "event_attn_map": aggregate_attention_maps(attn_maps),
            })
            done_pids.add(pid)
            save_json_atomic(out_path, {"condition": cond, "records": records})

        print(f"[{cond}] Done → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention-based mechanistic analysis.")
    parser.add_argument("--hf-model",     default=None)
    parser.add_argument("--story-path",   default=str(DEFAULT_STORY))
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT))
    parser.add_argument("--conditions",   nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--analysis",     choices=["all", "entropy", "map"], default="all")
    parser.add_argument("--n-patterns",   type=int, default=20)
    parser.add_argument("--layer-start",  type=int, default=0)
    parser.add_argument("--layer-end",    type=int, default=31)
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

    patterns      = load_json(Path(args.story_path)).get("patterns", [])
    target_layers = set(range(args.layer_start, min(args.layer_end + 1, num_layers)))

    collect_attention_analysis(
        model, tokenizer, patterns, args.conditions,
        args.n_patterns, out_dir, target_layers,
    )


if __name__ == "__main__":
    main()
