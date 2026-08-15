"""extract_hidden_states.py – Layer-wise hidden state extraction (§6).

For each pattern × condition, forward-passes the full 10-event context through
the LLM and captures the hidden state at the end-token of each event block using
forward hooks.  Target token positions are located via offset mapping (fast
tokenizers) or prefix-length fallback.

Also extracts:
  - entity token positions  (last token of entity= field in each set header)
  - shared-field token positions  (shared element between consecutive events)
  - recurrence labels  (has this entity appeared before in this pattern?)
  - stable content/entity/location IDs  (from elements.json)

Output per condition: results/probing/<model>/features_<condition>/pattern_<id>.npz

Usage
-----
    python src/probing/extract_hidden_states.py \\
        --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --load-in-4bit \\
        --conditions baseline w_o_causality w_o_agency w_o_time_series \\
        --out-dir results/probing/
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
DEFAULT_ELEM   = REPO_ROOT / "data" / "pools" / "elements.json"
DEFAULT_OUT    = REPO_ROOT / "results" / "probing"

ALL_CONDITIONS = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]
SHARED_FIELD_PRIORITY = ["location", "entity", "content"]
FIELD_TYPE_MAP = {"location": 0, "entity": 1, "content": 2}


# ---------------------------------------------------------------------------
# I/O helpers
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
# ID maps
# ---------------------------------------------------------------------------

def build_id_map(elements: Dict[str, Any], key: str) -> Dict[str, int]:
    return {v: i for i, v in enumerate(elements.get(key, []))}


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_date_to_rank(pattern: Dict[str, Any]) -> Dict[str, int]:
    """Chronological rank from the baseline condition (1-indexed)."""
    return {ev["temporal"]: (i + 1) for i, ev in enumerate(pattern["baseline"])}


def build_recurrence_labels(events: List[Dict]) -> np.ndarray:
    """0 = entity first appearance, 1 = entity seen before in this pattern."""
    seen: set = set()
    labels: List[int] = []
    for ev in events:
        entity = ev.get("entity", "")
        labels.append(1 if entity in seen else 0)
        seen.add(entity)
    return np.array(labels, dtype=np.int8)


def find_shared_field(ev_a: Dict, ev_b: Dict) -> Optional[str]:
    """Return the first field in SHARED_FIELD_PRIORITY with the same non-empty value."""
    for field in SHARED_FIELD_PRIORITY:
        va, vb = ev_a.get(field), ev_b.get(field)
        if va and va == vb:
            return field
    return None


# ---------------------------------------------------------------------------
# Tokenisation and position mapping
# ---------------------------------------------------------------------------

def build_event_end_char_positions(items: List[Dict]) -> Tuple[str, List[int]]:
    """Build full context string and end-char index per event block."""
    parts: List[str] = []
    for item in items:
        header = (
            f"Set {item.get('set_id')}: time={item.get('temporal')} | "
            f"location={item.get('location')} | entity={item.get('entity')} | "
            f"content={item.get('content')}"
        )
        text = item.get("text", "")
        parts.append(header + "\n" + text if text else header)

    cumulative = ""
    event_end_chars: List[int] = []
    for i, part in enumerate(parts):
        cumulative = part if i == 0 else cumulative + "\n\n" + part
        event_end_chars.append(len(cumulative))

    return "\n\n".join(parts), event_end_chars


def find_event_end_token_positions(
    tokenizer: AutoTokenizer,
    full_text: str,
    event_end_chars: List[int],
) -> Tuple[torch.Tensor, List[int], Optional[List[Tuple[int, int]]]]:
    """Tokenise and map event end-chars to token indices."""
    try:
        enc     = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        token_positions = [
            max((k for k, (start, _) in enumerate(offsets) if start < char_end), default=0)
            for char_end in event_end_chars
        ]
        return enc["input_ids"], token_positions, offsets
    except Exception:
        all_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
        token_positions = [
            tokenizer(full_text[:char_end], return_tensors="pt")["input_ids"].shape[1] - 1
            for char_end in event_end_chars
        ]
        return all_ids, token_positions, None


def _compute_part_starts(events: List[Dict]) -> Tuple[List[str], List[int]]:
    parts: List[str] = []
    for item in events:
        header = (
            f"Set {item.get('set_id')}: time={item.get('temporal')} | "
            f"location={item.get('location')} | entity={item.get('entity')} | "
            f"content={item.get('content')}"
        )
        text = item.get("text", "")
        parts.append(header + "\n" + text if text else header)
    starts = [0]
    for i in range(len(parts) - 1):
        starts.append(starts[-1] + len(parts[i]) + 2)
    return parts, starts


def _last_token_of_field(
    set_idx: int,
    field: str,
    events: List[Dict],
    part_starts: List[int],
    offsets: List[Tuple[int, int]],
) -> int:
    item       = events[set_idx]
    part_start = part_starts[set_idx]
    header     = (
        f"Set {item.get('set_id')}: time={item.get('temporal')} | "
        f"location={item.get('location')} | entity={item.get('entity')} | "
        f"content={item.get('content')}"
    )
    field_value  = str(item.get(field, ""))
    marker       = f"{field}={field_value}"
    marker_idx   = header.find(marker)
    if marker_idx < 0:
        return -1
    value_start  = part_start + marker_idx + len(f"{field}=")
    value_end    = value_start + len(field_value)
    best = -1
    for k, (tok_start, _) in enumerate(offsets):
        if value_start <= tok_start < value_end:
            best = k
    return best


def find_entity_token_positions(events: List[Dict], offsets: List[Tuple[int, int]]) -> List[int]:
    _, part_starts = _compute_part_starts(events)
    return [_last_token_of_field(i, "entity", events, part_starts, offsets)
            for i in range(len(events))]


def find_shared_pair_positions(
    events: List[Dict],
    offsets: List[Tuple[int, int]],
) -> List[Tuple[int, Optional[str]]]:
    _, part_starts = _compute_part_starts(events)
    result: List[Tuple[int, Optional[str]]] = []
    for k in range(len(events) - 1):
        shared = find_shared_field(events[k], events[k + 1])
        if shared is None:
            result.append((-1, None))
        else:
            pos = _last_token_of_field(k + 1, shared, events, part_starts, offsets)
            result.append((pos, shared))
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[AutoModelForCausalLM, AutoTokenizer, int]:
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

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=args.device,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_cfg,
    )
    model.eval()

    config     = AutoConfig.from_pretrained(model_name, token=hf_token, trust_remote_code=args.trust_remote_code)
    num_layers = config.num_hidden_layers
    print(f"  {num_layers} hidden layers")
    return model, tokenizer, num_layers


# ---------------------------------------------------------------------------
# Hidden state extraction via hooks
# ---------------------------------------------------------------------------

def extract_hidden_states_via_hooks(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    token_positions: List[int],
    target_layers: set,
) -> Dict[int, np.ndarray]:
    """Single forward pass; returns {layer_idx: ndarray (n_pos, hidden_dim)}."""
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0].detach().float().cpu()
        return hook_fn

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_container = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_container = model.transformer.h
    else:
        raise RuntimeError("Cannot locate transformer layers.")

    hooks = [
        layer_container[i].register_forward_hook(make_hook(i))
        for i in range(len(layer_container)) if i in target_layers
    ]
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids.to(model.device),
                attention_mask=torch.ones_like(input_ids).to(model.device),
            )
    finally:
        for h in hooks:
            h.remove()

    return {
        li: np.stack([hidden[pos].numpy() for pos in token_positions])
        for li, hidden in captured.items()
    }


# ---------------------------------------------------------------------------
# Extraction loop (resume-safe)
# ---------------------------------------------------------------------------

def run_extraction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    patterns: List[Dict],
    conditions: List[str],
    target_layers: set,
    out_dir: Path,
    save_every: int,
    content_id_map:  Dict[str, int],
    entity_id_map:   Dict[str, int],
    location_id_map: Dict[str, int],
) -> None:
    for condition in conditions:
        cond_dir      = out_dir / f"features_{condition}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        progress_file = cond_dir / "progress.json"

        processed: set = set()
        if progress_file.exists():
            processed = set(load_json(progress_file).get("processed", []))
            print(f"[{condition}] Resuming: {len(processed)}/{len(patterns)} done")

        for pattern in tqdm(patterns, desc=condition):
            pid    = pattern["pattern_id"]
            if pid in processed:
                continue

            events = pattern.get(condition)
            if not events:
                processed.add(pid)
                continue

            date_to_rank = build_date_to_rank(pattern)
            full_text, event_end_chars = build_event_end_char_positions(events)
            input_ids, token_positions, offsets = find_event_end_token_positions(
                tokenizer, full_text, event_end_chars
            )

            hidden = extract_hidden_states_via_hooks(
                model, input_ids, token_positions, target_layers
            )

            true_ranks = np.array(
                [date_to_rank.get(ev["temporal"], -1) for ev in events], dtype=np.int32
            )

            npz_data: Dict[str, np.ndarray] = {
                "token_positions": np.array(token_positions, dtype=np.int32),
                "true_ranks":      true_ranks,
                **{f"layer_{li}": vecs for li, vecs in hidden.items()},
            }

            # Entity / shared-field positions (requires fast tokenizer)
            if offsets is not None:
                entity_positions = find_entity_token_positions(events, offsets)
                shared_positions = find_shared_pair_positions(events, offsets)

                entity_vecs = {}
                for li, vecs_all in hidden.items():
                    rows = []
                    for pos in entity_positions:
                        if pos < 0 or pos >= input_ids.shape[1]:
                            rows.append(np.zeros(vecs_all.shape[1], dtype=np.float32))
                        else:
                            # Re-extract from full hidden state (requires separate hook capture)
                            rows.append(vecs_all[token_positions.index(pos)] if pos in token_positions else np.zeros(vecs_all.shape[1]))
                    entity_vecs[li] = np.stack(rows)

                recurrence_labels = build_recurrence_labels(events)
                shared_tok_pos = np.array([p for p, _ in shared_positions], dtype=np.int32)
                shared_field_type = np.array(
                    [FIELD_TYPE_MAP.get(f, -1) if f else -1 for _, f in shared_positions],
                    dtype=np.int8,
                )
                shared_content_ids = np.array(
                    [content_id_map.get(events[k].get("content", ""), -1) for k in range(len(events) - 1)],
                    dtype=np.int32,
                )
                shared_entity_ids = np.array(
                    [entity_id_map.get(events[k].get("entity", ""), -1)   for k in range(len(events) - 1)],
                    dtype=np.int32,
                )
                shared_location_ids = np.array(
                    [location_id_map.get(events[k].get("location", ""), -1) for k in range(len(events) - 1)],
                    dtype=np.int32,
                )

                npz_data.update({
                    "entity_token_positions":       np.array(entity_positions, dtype=np.int32),
                    "recurrence_labels":             recurrence_labels,
                    "shared_pair_token_pos":         shared_tok_pos,
                    "shared_pair_field_type":        shared_field_type,
                    "shared_pair_prev_content_id":   shared_content_ids,
                    "shared_pair_prev_entity_id":    shared_entity_ids,
                    "shared_pair_prev_location_id":  shared_location_ids,
                })

            np.savez_compressed(cond_dir / f"pattern_{pid}.npz", **npz_data)
            processed.add(pid)

            if len(processed) % save_every == 0:
                save_json_atomic(progress_file, {"processed": list(processed)})

        save_json_atomic(progress_file, {"processed": list(processed)})
        print(f"[{condition}] Done. {len(processed)} patterns saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract layer-wise hidden states.")
    parser.add_argument("--hf-model",     default=None)
    parser.add_argument("--story-path",   default=str(DEFAULT_STORY))
    parser.add_argument("--elements",     default=str(DEFAULT_ELEM))
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT))
    parser.add_argument("--conditions",   nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--layer-start",  type=int, default=0)
    parser.add_argument("--layer-end",    type=int, default=31)
    parser.add_argument("--save-every",   type=int, default=10)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    model, tokenizer, num_layers = load_model_and_tokenizer(args)

    model_name  = args.hf_model or os.getenv("HF_MODEL") or "Meta-Llama-3.1-8B-Instruct"
    model_tag   = Path(model_name).name
    out_dir     = Path(args.out_dir) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    elements       = load_json(Path(args.elements)) if Path(args.elements).exists() else {}
    content_id_map = build_id_map(elements, "content")
    entity_id_map  = build_id_map(elements, "entity")
    location_id_map= build_id_map(elements, "location")

    patterns = load_json(Path(args.story_path)).get("patterns", [])
    print(f"Loaded {len(patterns)} patterns.")

    target_layers = set(range(args.layer_start, min(args.layer_end + 1, num_layers)))

    run_extraction(
        model, tokenizer, patterns, args.conditions,
        target_layers, out_dir, args.save_every,
        content_id_map, entity_id_map, location_id_map,
    )


if __name__ == "__main__":
    main()
