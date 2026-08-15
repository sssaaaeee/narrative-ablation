"""generate_passages.py – 4-condition passage generation via OpenRouter API (§3).

For each pattern in generated_sets.json, generates narrative passages under
four conditions using a chat-completion model (default: gpt-3.5-turbo).
Prompt templates are stored in src/data_construction/prompts/*.txt and
correspond to Appendix A.2 of the paper.

Conditions
----------
  baseline       – causality + temporal order + fixed protagonist
  w_o_causality  – no causal links; all other structure intact
  w_o_time_series– temporal order shuffled (uses derived_sets2); causality intact
  w_o_agency     – rotating viewpoint; no fixed protagonist; causality intact

Output: data/passages/generated_story.json

Usage
-----
    python src/data_construction/generate_passages.py \\
        --sets  data/sets/generated_sets.json \\
        --out   data/passages/generated_story.json \\
        --model openai/gpt-3.5-turbo
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
from openai import OpenAI


REPO_ROOT   = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SETS = REPO_ROOT / "data" / "sets" / "generated_sets.json"
DEFAULT_OUT  = REPO_ROOT / "data" / "passages" / "generated_story.json"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL_NARRATIVE", "openai/gpt-3.5-turbo")
TEMPERATURE   = 0.8
MAX_TOKENS    = 220


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
# Prompt construction
# ---------------------------------------------------------------------------

def _format_set_item(item: Dict[str, Any]) -> str:
    return (
        f"temporal: {item['temporal']}\n"
        f"location: {item['location']}\n"
        f"entity:   {item['entity']}\n"
        f"content:  {item['content']}"
    )


def _load_template(mode: str) -> str:
    """Load prompt template from prompts/<mode>.txt (Appendix A.2)."""
    path = PROMPTS_DIR / f"{mode}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""   # fallback: built-in templates below


def build_prompt(
    mode: str,
    item: Dict[str, Any],
    protagonist: Optional[str],
    previous_text: Optional[str],
) -> str:
    """Construct the generation prompt for *mode* and one element set.

    Prompt logic matches Appendix A.2:
      - baseline:        causal link required; fixed protagonist
      - w_o_causality:   no causal link; fixed protagonist
      - w_o_time_series: temporal order ignored; causal link required; fixed protagonist
      - w_o_agency:      rotating viewpoint; no fixed protagonist; causal link required
    """
    base = (
        "You are a writer of short English narratives. "
        "Generate exactly one short story passage that satisfies the following conditions.\n"
        "- Output story text only\n"
        "- Write in natural English, about 2 to 4 sentences\n"
        "- Do not use bullet points or explanations\n"
        "- Incorporate the provided elements as naturally as possible\n"
    )
    set_block = _format_set_item(item)
    prev_block = previous_text or "None"

    if mode == "baseline":
        return (
            base
            + f"Keep {protagonist} as the protagonist throughout the story.\n"
            + f"Previous story text:\n{prev_block}\n\n"
            + f"Next elements:\n{set_block}\n\n"
            + "Write the next story passage so that it naturally continues from the previous story text.\n"
            + "Important: Make this passage include a clear causal relation to the previous passage.\n"
            + "The events, actions, or state changes in this passage should be motivated by what happened before."
        )

    if mode == "w_o_causality":
        return (
            base
            + f"Keep {protagonist} as the protagonist throughout the story.\n"
            + f"Previous story text:\n{prev_block}\n\n"
            + f"Next elements:\n{set_block}\n\n"
            + "Write the next story passage so that it naturally continues from the previous story text.\n"
            + "Important: Do NOT create a causal relation with the previous passage.\n"
            + "The events, actions, and state changes in this passage should be independent of what happened before."
        )

    if mode == "w_o_time_series":
        return (
            base
            + f"Keep {protagonist} as the protagonist throughout the story.\n"
            + f"Previous story text:\n{prev_block}\n\n"
            + f"Next elements:\n{set_block}\n\n"
            + "Write the next story passage so that it naturally continues from the previous story text.\n"
            + "Important: Ignore temporal order and chronological consistency.\n"
            + "Write as if time is not a meaningful constraint.\n"
            + "Also: Make this passage include a clear causal relation to the previous passage.\n"
            + "The events, actions, or state changes in this passage should be motivated by what happened before."
        )

    if mode == "w_o_agency":
        return (
            base
            + f"Previous story text:\n{prev_block}\n\n"
            + f"Next elements:\n{set_block}\n\n"
            + "Write the next story passage so that it naturally continues from the previous story text.\n"
            + "Important: Do NOT keep a fixed protagonist. Change the viewpoint and perspective in each passage.\n"
            + "Each passage may have a different main character or narrator.\n"
            + "Also: Make this passage include a clear causal relation to the previous passage.\n"
            + "The events, actions, or state changes in this passage should be motivated by what happened before."
        )

    raise ValueError(f"Unknown generation mode: {mode!r}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def generate_condition(
    client: OpenAI,
    model: str,
    mode: str,
    sets: List[Dict[str, Any]],
    protagonist: Optional[str],
) -> List[Dict[str, Any]]:
    """Generate passages for all sets under *mode* sequentially (each passage
    conditions on the preceding text to maintain narrative continuity)."""
    results: List[Dict[str, Any]] = []
    previous_text: Optional[str] = None

    for item in sets:
        prot = None if mode == "w_o_agency" else protagonist
        prompt = build_prompt(mode, item, prot, previous_text)
        text   = generate_text(client, model, prompt)
        results.append(
            {
                "set_id":   item["set_id"],
                "temporal": item["temporal"],
                "location": item["location"],
                "entity":   item["entity"],
                "content":  item["content"],
                "prompt":   prompt,
                "text":     text,
            }
        )
        previous_text = text

    return results


def generate_pattern_narratives(
    client: OpenAI, model: str, pattern: Dict[str, Any]
) -> Dict[str, Any]:
    pid           = pattern["pattern_id"]
    derived_sets  = pattern["derived_sets"]
    derived_sets2 = pattern["derived_sets2"]

    protagonist_base  = derived_sets[0]["entity"]
    protagonist_ts    = derived_sets2[0]["entity"]

    print(f"  Generating pattern {pid} ...")
    return {
        "pattern_id":   pid,
        "original_sets":  pattern["original_sets"],
        "derived_sets":   derived_sets,
        "derived_sets2":  derived_sets2,
        "baseline":        generate_condition(client, model, "baseline",        derived_sets,  protagonist_base),
        "w_o_causality":   generate_condition(client, model, "w_o_causality",   derived_sets,  protagonist_base),
        "w_o_time_series": generate_condition(client, model, "w_o_time_series", derived_sets2, protagonist_ts),
        "w_o_agency":      generate_condition(client, model, "w_o_agency",      derived_sets,  None),
    }


# ---------------------------------------------------------------------------
# Main loop (resume-safe)
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dotenv.load_dotenv()

    api_key = (
        os.getenv("OPENROUTER_API_KEY_1")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("No API key found in OPENROUTER_API_KEY_1 / OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    sets_path = Path(args.sets)
    out_path  = Path(args.out)

    sets_data = load_json(sets_path)
    patterns  = sets_data.get("patterns", [])

    # Resume: load completed pattern IDs
    completed: Dict[int, Dict[str, Any]] = {}
    if out_path.exists():
        existing = load_json(out_path)
        for p in existing.get("patterns", []):
            completed[p["pattern_id"]] = p
        print(f"Resuming: {len(completed)}/{len(patterns)} patterns already done.")

    output: Dict[str, Any] = {
        "metadata": {
            "model":      args.model,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_sets": str(sets_path),
            "num_patterns": len(patterns),
        },
        "patterns": [],
    }

    for pattern in patterns:
        pid = pattern["pattern_id"]
        if pid in completed:
            print(f"  [{pid}/{len(patterns)}] already done, skipping.")
            continue

        generated = generate_pattern_narratives(client, args.model, pattern)
        completed[pid] = generated

        output["patterns"] = [completed[k] for k in sorted(completed)]
        save_json_atomic(out_path, output)
        print(f"  [{pid}/{len(patterns)}] saved.")

    output["patterns"] = [completed[k] for k in sorted(completed)]
    save_json_atomic(out_path, output)
    print(f"Done. {len(completed)} patterns → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 4-condition narrative passages.")
    parser.add_argument("--sets",  default=str(DEFAULT_SETS),  help="generated_sets.json path")
    parser.add_argument("--out",   default=str(DEFAULT_OUT),   help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL,      help="OpenRouter model string")
    return parser.parse_args()


if __name__ == "__main__":
    main()
