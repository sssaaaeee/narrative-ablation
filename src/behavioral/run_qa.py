"""run_qa.py – Collect QA answers using a local LLM (§5).

Loads a HuggingFace model (default: Meta-Llama-3.1-8B-Instruct) with optional
4-bit / 8-bit quantisation via BitsAndBytes, then runs greedy decoding
(max_new_tokens=32) on each single-hop and multi-hop question in
data/questions/*.json.

Answers are written to results/answers_{single,multi}_all.json with a resume
mechanism: already-answered (pattern_id, variant, question_index) tuples are
skipped on restart.

Usage
-----
    python src/behavioral/run_qa.py \\
        --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --load-in-4bit \\
        --mode all \\
        --out-dir results/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT      = Path(__file__).resolve().parents[2]
DEFAULT_STORY  = REPO_ROOT / "data" / "passages" / "generated_story.json"
DEFAULT_SINGLE = REPO_ROOT / "data" / "questions" / "single_hop_questions.json"
DEFAULT_MULTI  = REPO_ROOT / "data" / "questions" / "multi_hop_questions.json"
DEFAULT_OUT    = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    os.makedirs(path.parent, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Context / prompt construction
# ---------------------------------------------------------------------------

def build_context_text(items: List[Dict[str, Any]]) -> str:
    """Format the 10-event set as a numbered context block."""
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


def build_model_prompt(tokenizer: AutoTokenizer, context: str, question: str) -> str:
    """Apply chat template if available, else fall back to raw formatting."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user",   "content": f"context:\n{context}\n\nquestion:\n{question}\n\nanswer:"},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        "System: You are a helpful assistant. Answer concisely.\n"
        f"User: context:\n{context}\n\nquestion:\n{question}\n\nanswer:\n"
        "Assistant:"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    """Greedy decoding, returns decoded new tokens only."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids      = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = generated_ids[0, input_ids.shape[-1]:]
    result  = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Collection loop (resume-safe)
# ---------------------------------------------------------------------------

def collect_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    patterns: List[Dict],
    questions: List[Dict],
    max_new_tokens: int,
    save_every: int,
    out_path: Path,
) -> None:
    # Resume: load existing answers
    answers: List[Dict] = []
    answered_keys: Set  = set()
    if out_path.exists():
        existing = load_json(out_path)
        answers  = existing.get("answers", [])
        for a in answers:
            answered_keys.add(
                (a.get("pattern_id"), a.get("variant"), a.get("question_index"))
            )
        print(f"Resuming: {len(answers)} answers already saved in {out_path}")

    pattern_map = {p["pattern_id"]: p for p in patterns}
    total       = len(questions)
    new_count   = 0

    for idx, q in enumerate(questions):
        try:
            pid   = q.get("pattern_id")
            var   = q.get("variant")
            q_idx = q.get("index", q.get("question_index", q.get("set_id")))

            if (pid, var, q_idx) in answered_keys:
                continue

            pattern = pattern_map.get(pid)
            if pattern is None:
                print(f"Warning: pattern_id {pid} not found; skipping")
                continue

            items    = pattern.get(var, [])
            context  = build_context_text(items)
            prompt   = build_model_prompt(tokenizer, context, q.get("question"))
            answer   = generate_answer(model, tokenizer, prompt, max_new_tokens)

            answers.append({
                "pattern_id":    pid,
                "variant":       var,
                "question_type": q.get("question_type"),
                "question_index": q_idx,
                "question":      q.get("question"),
                "answer":        answer,
                "gold_answer":   q.get("answer"),
            })
            new_count += 1

        except Exception as exc:
            print(f"Error at question idx {idx}: {exc}")

        if new_count > 0 and new_count % save_every == 0:
            save_json_atomic(out_path, {"count": len(answers), "answers": answers})
            print(f"Progress {idx + 1}/{total} | saved {len(answers)} answers")

    save_json_atomic(out_path, {"count": len(answers), "answers": answers})
    print(f"Finished: {len(answers)} answers → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect QA answers from a local LLM.")
    parser.add_argument("--hf-model",       default=None)
    parser.add_argument("--story-path",     default=str(DEFAULT_STORY))
    parser.add_argument("--single-q-path",  default=str(DEFAULT_SINGLE))
    parser.add_argument("--multi-q-path",   default=str(DEFAULT_MULTI))
    parser.add_argument("--out-dir",        default=str(DEFAULT_OUT))
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--save-every",     type=int, default=100)
    parser.add_argument("--load-in-4bit",   action="store_true")
    parser.add_argument("--load-in-8bit",   action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--mode", choices=["all", "single", "multi"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    model_name = args.hf_model or os.getenv("HF_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token   = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

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

    patterns  = load_json(Path(args.story_path)).get("patterns", [])
    single_qs = load_json(Path(args.single_q_path)).get("questions", [])
    multi_qs  = load_json(Path(args.multi_q_path)).get("questions", [])
    out_dir   = Path(args.out_dir)

    if args.mode in ("all", "single"):
        print(f"Collecting single-hop answers ({len(single_qs)} questions) …")
        collect_answers(
            model, tokenizer, patterns, single_qs,
            args.max_new_tokens, args.save_every,
            out_dir / "answers_single_all.json",
        )

    if args.mode in ("all", "multi"):
        print(f"Collecting multi-hop answers ({len(multi_qs)} questions) …")
        collect_answers(
            model, tokenizer, patterns, multi_qs,
            args.max_new_tokens, args.save_every,
            out_dir / "answers_multi_all.json",
        )


if __name__ == "__main__":
    main()
