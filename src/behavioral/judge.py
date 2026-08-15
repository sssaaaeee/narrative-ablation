"""judge.py – LLM-as-judge evaluation of collected answers (§5).

Evaluates model answers against gold answers using GPT-3.5-turbo as a judge.
Scoring:
  1.0 = correct (equivalent to gold; allows synonyms and paraphrases)
  0.5 = partially correct (incomplete or tangentially related)
  0.0 = incorrect or unrelated

Optimisation: exact-match shortcut (after whitespace normalisation) skips
the API call when the prediction literally equals the gold answer.

Output: results/eval_{single,multi}.json

Usage
-----
    python src/behavioral/judge.py \\
        --single-answers results/answers_single_all.json \\
        --multi-answers  results/answers_multi_all.json \\
        --out-dir        results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from dotenv import load_dotenv


REPO_ROOT      = Path(__file__).resolve().parents[2]
DEFAULT_SINGLE = REPO_ROOT / "results" / "answers_single_all.json"
DEFAULT_MULTI  = REPO_ROOT / "results" / "answers_multi_all.json"
DEFAULT_OUT    = REPO_ROOT / "results"

JUDGE_MODEL                      = "gpt-3.5-turbo"
MODEL_INPUT_PRICE_USD_PER_MTOK   = 0.500
MODEL_OUTPUT_PRICE_USD_PER_MTOK  = 1.500


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Judge logic
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * MODEL_INPUT_PRICE_USD_PER_MTOK
        + completion_tokens * MODEL_OUTPUT_PRICE_USD_PER_MTOK
    ) / 1_000_000.0


def judge_pair(
    client: openai.OpenAI,
    reference: str,
    prediction: str,
) -> Tuple[float, str, Dict[str, int], float]:
    """Call GPT-3.5-turbo to score prediction vs reference.

    Returns (score, reason, token_usage, cost_usd).
    """
    # Fast exact-match shortcut
    if _normalize(reference) == _normalize(prediction):
        return 1.0, "exact match (shortcut)", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

    system = (
        "You are an expert evaluation judge. Compare a model's predicted answer "
        "to a reference (ground-truth) answer. Return a numeric score and a brief reason.\n"
        "Scoring rules: 1.0 if the prediction is equivalent to the reference "
        "(allow synonyms/paraphrases), 0.5 if partially related or incomplete, "
        "0.0 if incorrect or unrelated.\n"
        "Output only JSON with keys: score (number), reason (short string)."
    )
    user = f"Reference answer: {reference}\nPrediction: {prediction}\nJudge:"

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    text     = resp.choices[0].message.content.strip()
    usage_obj = getattr(resp, "usage", None)
    usage    = {
        "prompt_tokens":     int(getattr(usage_obj, "prompt_tokens",     0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens":      int(getattr(usage_obj, "total_tokens",      0) or 0),
    }
    cost_usd = _cost_usd(usage["prompt_tokens"], usage["completion_tokens"])

    try:
        parsed = json.loads(text)
        return float(parsed.get("score", 0.0)), parsed.get("reason", ""), usage, cost_usd
    except Exception:
        first = text.splitlines()[0] if text.splitlines() else text
        m     = re.search(r"([01](?:\.5)?)", first)
        return (float(m.group(1)) if m else 0.0), text, usage, cost_usd


# ---------------------------------------------------------------------------
# Evaluation loop (resume-safe)
# ---------------------------------------------------------------------------

def evaluate(
    client: openai.OpenAI,
    answers_path: Path,
    out_path: Path,
) -> None:
    data    = load_json(answers_path)
    answers = data.get("answers", [])

    # Build key set from already-evaluated output
    evaluated_keys: set = set()
    results_list: List[Dict] = []
    if out_path.exists():
        existing = load_json(out_path)
        results_list = existing.get("results", [])
        for r in results_list:
            evaluated_keys.add(
                (r.get("pattern_id"), r.get("variant"), r.get("question_index"))
            )
        print(f"Resuming: {len(results_list)} evaluations already done.")

    total_cost = 0.0
    save_every = 100
    new_count  = 0

    for idx, ans in enumerate(answers):
        pid   = ans.get("pattern_id")
        var   = ans.get("variant")
        q_idx = ans.get("question_index")
        key   = (pid, var, q_idx)

        if key in evaluated_keys:
            continue

        reference  = str(ans.get("gold_answer") or ans.get("answer") or "")
        prediction = str(ans.get("answer", ""))
        # If gold_answer stored separately, use it
        if "gold_answer" in ans:
            reference = str(ans["gold_answer"])

        score, reason, usage, cost_usd = judge_pair(client, reference, prediction)
        total_cost += cost_usd

        results_list.append({
            "pattern_id":    pid,
            "variant":       var,
            "question_type": ans.get("question_type"),
            "index":         q_idx,
            "question_index": q_idx,
            "question":      ans.get("question"),
            "gold_answer":   reference,
            "model_answer":  prediction,
            "score":         score,
            "reason":        reason,
            "cost_usd":      cost_usd,
        })
        evaluated_keys.add(key)
        new_count += 1

        if new_count > 0 and new_count % save_every == 0:
            save_json(out_path, {"results": results_list, "total_cost_usd": total_cost})
            print(f"  {idx + 1}/{len(answers)} | cost so far: ${total_cost:.4f}")

    save_json(out_path, {"results": results_list, "total_cost_usd": total_cost})
    print(f"Done. {len(results_list)} evaluations | total API cost: ${total_cost:.4f} → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-judge evaluation of QA answers.")
    parser.add_argument("--single-answers", default=str(DEFAULT_SINGLE))
    parser.add_argument("--multi-answers",  default=str(DEFAULT_MULTI))
    parser.add_argument("--out-dir",        default=str(DEFAULT_OUT))
    parser.add_argument("--judge-model",    default=JUDGE_MODEL)
    parser.add_argument("--mode", choices=["all", "single", "multi"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY_1")
    if not api_key:
        raise RuntimeError("No API key found in OPENAI_API_KEY or OPENROUTER_API_KEY_1")

    client  = openai.OpenAI(api_key=api_key)
    out_dir = Path(args.out_dir)

    if args.mode in ("all", "single"):
        print(f"Evaluating single-hop answers: {args.single_answers}")
        evaluate(client, Path(args.single_answers), out_dir / "eval_single.json")

    if args.mode in ("all", "multi"):
        print(f"Evaluating multi-hop answers:  {args.multi_answers}")
        evaluate(client, Path(args.multi_answers), out_dir / "eval_multi.json")


if __name__ == "__main__":
    main()
