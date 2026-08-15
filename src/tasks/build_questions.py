"""build_questions.py – Single-hop (12 types) and multi-hop (10 types) QA generation (Appendix B).

Single-hop questions probe one attribute given another within the same event set:
  Types a–l  (12 types)  →  {temporal, location, entity, content} × {→ the other 3}
  Conditions w_o_agency exclude entity-querying types (h, i, l) that presuppose a
  fixed protagonist.

Multi-hop questions require bridging two adjacent events via a shared field:
  Types a–j  (10 types)  →  temporal-chain, location-chain, entity-chain, and
  protagonist-based chains (h, i, j)
  Conditions w_o_time_series exclude temporal-chain types (a, b, c, h).

Exclusion logic for both hop types:
  - Source and target fields must be unique among adjacent events (no ambiguity).
  - Common-field value must be shared between consecutive events (for multi-hop).

Output
------
  data/questions/single_hop_questions.json
  data/questions/multi_hop_questions.json

Usage
-----
    python src/tasks/build_questions.py \\
        --passages data/passages/generated_story.json \\
        --out-dir  data/questions/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT      = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = REPO_ROOT / "data" / "passages" / "generated_story.json"
DEFAULT_OUT    = REPO_ROOT / "data" / "questions"


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


def normalize_patterns(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "patterns" in data:
        return data["patterns"]
    if {"baseline", "w_o_causality"}.issubset(data.keys()):
        return [data]
    raise ValueError("Unrecognised story JSON schema.")


# ---------------------------------------------------------------------------
# ─── SINGLE-HOP (§B.1)  ────────────────────────────────────────────────────
#
# 12 question types (a–l): source_field → target_field
# ---------------------------------------------------------------------------

SOURCE_TARGET_BY_QTYPE: Dict[str, Tuple[str, str]] = {
    "a": ("temporal", "location"),
    "b": ("temporal", "entity"),
    "c": ("temporal", "content"),
    "d": ("location", "temporal"),
    "e": ("location", "entity"),
    "f": ("location", "content"),
    "g": ("entity",   "temporal"),
    "h": ("entity",   "location"),
    "i": ("entity",   "content"),
    "j": ("content",  "temporal"),
    "k": ("content",  "location"),
    "l": ("content",  "entity"),
}

# w_o_agency excludes entity-query types that presuppose a fixed protagonist
SINGLE_VARIANT_QTYPES: Dict[str, List[str]] = {
    "baseline":        list("abcdefghijkl"),
    "w_o_causality":   list("abcdefghijkl"),
    "w_o_time_series": list("abcdefghijkl"),
    "w_o_agency":      list("abcdefg"),
}

SINGLE_QUESTION_TEMPLATES: Dict[str, str] = {
    "a": "What is the location at time {source_value}?",
    "b": "Who appears at time {source_value}{person_hint}?",
    "c": "What event happens at time {source_value}?",
    "d": "At what time does the event at location {source_value} happen? Answer in YYYY-MM-DD format.",
    "e": "Who is at location {source_value}{person_hint}?",
    "f": "What event happens at location {source_value}?",
    "g": "At what time does the person {source_value} appear? Answer in YYYY-MM-DD format.",
    "h": "Where does the person {source_value} appear?",
    "i": "What event does the person {source_value} experience?",
    "j": "At what time does the event described by {source_value} happen? Answer in YYYY-MM-DD format.",
    "k": "Where does the event described by {source_value} happen?",
    "l": "Who experiences the event described by {source_value}{person_hint}?",
}


def _is_unique_among_neighbors(items: List[Dict], idx: int, field: str) -> bool:
    """Return True if items[idx][field] differs from its immediate neighbours."""
    value = items[idx].get(field)
    if value is None:
        return False
    for neighbor in (idx - 1, idx + 1):
        if 0 <= neighbor < len(items) and items[neighbor].get(field) == value:
            return False
    return True


def _build_single_question_text(qtype: str, source_value: Any, variant: str) -> str:
    person_hint    = ""
    full_name_sfx  = ""
    if qtype in ("b", "e", "l") and variant != "w_o_agency":
        person_hint   = " besides the protagonist"
        full_name_sfx = " Answer with the person's full name."
    return (
        SINGLE_QUESTION_TEMPLATES[qtype].format(
            source_value=source_value, person_hint=person_hint
        )
        + full_name_sfx
    )


def _get_items_for_variant(pattern: Dict, variant: str) -> List[Dict]:
    if variant == "w_o_time_series":
        return pattern["derived_sets2"]
    return pattern["derived_sets"]


def build_single_hop_questions(patterns: List[Dict]) -> List[Dict]:
    """Generate all single-hop questions for all patterns and conditions."""
    questions: List[Dict] = []
    for pattern in patterns:
        pid = pattern["pattern_id"]
        for variant in ("baseline", "w_o_causality", "w_o_time_series", "w_o_agency"):
            items        = _get_items_for_variant(pattern, variant)
            allowed      = SINGLE_VARIANT_QTYPES[variant]
            for idx, item in enumerate(items):
                set_id = item.get("set_id")
                # Skip first event for non-agency conditions (no preceding context)
                if variant != "w_o_agency" and set_id == 1:
                    continue
                for qtype in allowed:
                    src_field, tgt_field = SOURCE_TARGET_BY_QTYPE[qtype]
                    if not _is_unique_among_neighbors(items, idx, src_field):
                        continue
                    if not _is_unique_among_neighbors(items, idx, tgt_field):
                        continue
                    src_val = item.get(src_field)
                    tgt_val = item.get(tgt_field)
                    if src_val is None or tgt_val is None:
                        continue
                    questions.append({
                        "pattern_id":   pid,
                        "variant":      variant,
                        "set_id":       set_id,
                        "index":        idx,
                        "question_type": qtype,
                        "source_field":  src_field,
                        "target_field":  tgt_field,
                        "source_value":  src_val,
                        "answer":        tgt_val,
                        "question":      _build_single_question_text(qtype, src_val, variant),
                    })
    return questions


# ---------------------------------------------------------------------------
# ─── MULTI-HOP (§B.2)  ─────────────────────────────────────────────────────
#
# 10 question types (a–j): consecutive-event pairs sharing a common field
# ---------------------------------------------------------------------------

COMMON_FIELD_BY_QTYPE: Dict[str, Optional[str]] = {
    "a": "location",
    "b": "entity",
    "c": "content",
    "d": "entity",
    "e": "content",
    "f": "location",
    "g": "content",
    "h": "entity",
    "i": "entity",
    "j": "location",
}

# w_o_agency excludes protagonist-based types; w_o_time_series excludes temporal types
MULTI_VARIANT_QTYPES: Dict[str, List[str]] = {
    "baseline":        list("abcdefghij"),
    "w_o_causality":   list("abcdefghij"),
    "w_o_time_series": list("defgij"),
    "w_o_agency":      list("abcdefg"),
}


def _make_multi_question(
    qtype: str,
    item: Dict,
    next_item: Dict,
    protagonist: Optional[str],
) -> Tuple[str, str]:
    """Build (question_text, answer) for multi-hop type *qtype*."""
    builders: Dict[str, Any] = {
        "a": lambda i, n, p: (
            f"The location {i['location']} was the setting at time {i['temporal']}. "
            f"At what time did the next event occur at this location? Answer in YYYY-MM-DD format.",
            n["temporal"],
        ),
        "b": lambda i, n, p: (
            f"The person {i['entity']} appeared at time {i['temporal']}. "
            f"When did this person next appear? Answer in YYYY-MM-DD format.",
            n["temporal"],
        ),
        "c": lambda i, n, p: (
            f"An event described as '{i['content']}' occurred at time {i['temporal']}. "
            f"When did a similar event occur next? Answer in YYYY-MM-DD format.",
            n["temporal"],
        ),
        "d": lambda i, n, p: (
            f"The person {i['entity']} appeared at location {i['location']}. "
            f"At which other location did this person appear?",
            n["location"],
        ),
        "e": lambda i, n, p: (
            f"An event '{i['content']}' happened at location {i['location']}. "
            f"At which other location did this event (or a similar one) occur?",
            n["location"],
        ),
        "f": lambda i, n, p: (
            f"At the location where {i['entity']} appeared, who else appeared afterwards? "
            f"Answer with the person's full name.",
            n["entity"],
        ),
        "g": lambda i, n, p: (
            f"The person {i['entity']} experienced '{i['content']}'. "
            f"Which other person appeared in relation to this event? Answer with the person's full name.",
            n["entity"],
        ),
        "h": lambda i, n, p: (
            f"The protagonist {p} met {i['entity']} at time {i['temporal']}. "
            f"When did they meet this person again? Answer in YYYY-MM-DD format.",
            n["temporal"],
        ),
        "i": lambda i, n, p: (
            f"The protagonist {p} met {i['entity']} at location {i['location']}. "
            f"At which other location did the protagonist meet this person?",
            n["location"],
        ),
        "j": lambda i, n, p: (
            f"At the location {i['location']} where the protagonist {p} met {i['entity']}, "
            f"who else did the protagonist meet? Answer with the person's full name.",
            n["entity"],
        ),
    }
    return builders[qtype](item, next_item, protagonist)


def build_multi_hop_questions(patterns: List[Dict]) -> List[Dict]:
    """Generate all multi-hop questions for all patterns and conditions."""
    questions: List[Dict] = []
    for pattern in patterns:
        pid = pattern["pattern_id"]
        for variant in ("baseline", "w_o_causality", "w_o_time_series", "w_o_agency"):
            items       = _get_items_for_variant(pattern, variant)
            allowed     = MULTI_VARIANT_QTYPES[variant]
            protagonist = items[0].get("entity") if items else None

            for i in range(len(items) - 1):
                item      = items[i]
                next_item = items[i + 1]
                for qtype in allowed:
                    common_field = COMMON_FIELD_BY_QTYPE.get(qtype)
                    if common_field:
                        v1, v2 = item.get(common_field), next_item.get(common_field)
                        if v1 is None or v2 is None or v1 != v2:
                            continue
                    try:
                        q_text, answer = _make_multi_question(qtype, item, next_item, protagonist)
                    except Exception:
                        continue
                    questions.append({
                        "pattern_id":   pid,
                        "variant":      variant,
                        "index":        i,
                        "question_type": qtype,
                        "common_field":  common_field,
                        "common_value":  item.get(common_field) if common_field else None,
                        "question":      q_text,
                        "answer":        answer,
                    })
    return questions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-hop and multi-hop questions.")
    parser.add_argument("--passages", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir",  default=str(DEFAULT_OUT))
    parser.add_argument("--mode", choices=["all", "single", "multi"], default="all")
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    data     = load_json(Path(args.passages))
    patterns = normalize_patterns(data)
    print(f"Loaded {len(patterns)} patterns.")

    if args.mode in ("all", "single"):
        single_qs = build_single_hop_questions(patterns)
        out = {
            "metadata": {"source": args.passages, "variant": "single_hop",
                         "total": len(single_qs)},
            "questions": single_qs,
        }
        path = out_dir / "single_hop_questions.json"
        save_json_atomic(path, out)
        print(f"Single-hop: {len(single_qs)} questions → {path}")

    if args.mode in ("all", "multi"):
        multi_qs = build_multi_hop_questions(patterns)
        out = {
            "metadata": {"source": args.passages, "variant": "multi_hop",
                         "total": len(multi_qs)},
            "questions": multi_qs,
        }
        path = out_dir / "multi_hop_questions.json"
        save_json_atomic(path, out)
        print(f"Multi-hop:  {len(multi_qs)} questions → {path}")


if __name__ == "__main__":
    main()
