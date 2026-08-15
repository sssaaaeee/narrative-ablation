"""metrics.py – Narrative manipulation-check metrics (Appendix A.4–A.5).

Three lexical / structural metrics that operationalise the three narrative
dimensions ablated in the experiment:

  1. causality_score  – causal-marker density (Tier 1 + Tier 2 lexicon)
                        occurrences per 100 words
  2. time_series_tau  – Kendall's τ between set-presentation order and
                        chronological date order (from input metadata)
  3. agency_score     – protagonist dominance ratio: max PERSON-entity
                        frequency / total PERSON-entity mentions (spaCy)

Each metric is designed to be diagnostic for one condition only (diagonal
pattern in the 4 × 3 matrix); see manipulation_check.py run_check.py for
statistical evaluation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# Metric 1 – Causality score
#
# Two-tier causal-marker lexicon following Appendix A.4:
#   Tier 1: Explicit logical connectors (because, therefore, …)
#   Tier 2: Implicit causal participial/adjectival constructions used by
#            GPT-3.5-turbo to encode cross-passage causality
#            (inspired by, led her to, reminiscing about, …)
# ---------------------------------------------------------------------------

_CAUSAL_RE = re.compile(
    r"\b("
    # --- Tier 1: explicit logical connectors ---
    r"because|therefore|consequently|hence|thus|thereby|"
    r"as a result|led to|caused|due to|owing to|because of|"
    r"which is why|resulting in|that is why|"
    # --- Tier 2: causal participial / adjectival phrases ---
    r"inspired by|excited by|motivated by|driven by|fueled by|"
    r"sparked by|prompted by|urged by|compelled by|energized by|"
    r"emboldened by|stirred by|emboldened|"
    r"reminiscing about|reflecting on|drawing from|drawing inspiration|"
    r"memories of|recalling|haunted by|"
    r"led her|led him|led them|had sparked|had prompted|had inspired|"
    r"had encouraged|had driven|had motivated|had compelled|"
    r"still enamored|still inspired|still excited|still motivated|"
    r"feeling invigorated|feeling inspired|feeling energized"
    r")\b",
    re.IGNORECASE,
)


def causality_score(text: str) -> float:
    """Causal-marker occurrences per 100 words."""
    words = text.split()
    if not words:
        return 0.0
    return len(_CAUSAL_RE.findall(text)) / len(words) * 100.0


# ---------------------------------------------------------------------------
# Metric 2 – Time-series score  (primary: Kendall's τ from input metadata)
#
# τ is computed between set-presentation order [1…N] and the chronological
# ranks of the temporal dates.  For baseline/w_o_causality/w_o_agency,
# dates are always sorted → τ = 1.0 by design.  For w_o_time_series the
# dates are shuffled → τ < 1.
#
# Supplementary: temporal-connective frequency (text-based signal of
# how explicitly the author signals sequential flow).
# ---------------------------------------------------------------------------

_TEMPORAL_CONNECTIVE_RE = re.compile(
    r"\b("
    r"then|after|later|subsequently|meanwhile|afterwards|eventually|"
    r"next|finally|before|prior to|following|soon after|once"
    r")\b",
    re.IGNORECASE,
)


def time_series_tau(temporals: List[str]) -> float:
    """Kendall's τ between set-order positions and chronological date ranks."""
    dates = [datetime.strptime(t, "%Y-%m-%d") for t in temporals]
    positions = list(range(1, len(dates) + 1))
    tau, _ = kendalltau(positions, dates)
    return float(tau)


def temporal_connective_score(text: str) -> float:
    """Temporal connective occurrences per 100 words (supplementary)."""
    words = text.split()
    if not words:
        return 0.0
    return len(_TEMPORAL_CONNECTIVE_RE.findall(text)) / len(words) * 100.0


# ---------------------------------------------------------------------------
# Metric 3 – Agency score  (protagonist dominance via spaCy NER)
#
# For each pattern × condition we pool PERSON entities across all passages
# and compute:
#   dominance = count(most-frequent entity) / total PERSON mentions
# Possessive "'s" is stripped so "Victoria Carter's" and "Victoria Carter"
# are treated identically.
# ---------------------------------------------------------------------------

_POSSESSIVE_RE = re.compile(r"'s$|'s$", re.IGNORECASE)


def _normalize_entity(name: str) -> str:
    return _POSSESSIVE_RE.sub("", name).strip()


def agency_score(texts: List[str], nlp) -> Tuple[float, int]:
    """Compute protagonist dominance ratio using spaCy NER.

    Parameters
    ----------
    texts : list of passage strings
    nlp   : spaCy Language model (en_core_web_sm or larger)

    Returns
    -------
    dominance      : max entity frequency / total PERSON mentions
    unique_entities: number of distinct PERSON entity strings
    """
    ec: Dict[str, int] = {}
    total = 0
    for doc in nlp.pipe(texts, disable=["parser", "tagger", "lemmatizer"]):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = _normalize_entity(ent.text)
                if name:
                    ec[name] = ec.get(name, 0) + 1
                    total += 1
    if total == 0:
        return 0.0, 0
    return float(max(ec.values()) / total), len(ec)


# ---------------------------------------------------------------------------
# Convenience: compute all metrics for one pattern × condition entry
# ---------------------------------------------------------------------------

def compute_pattern_scores(pattern: dict, nlp) -> Dict[str, Dict]:
    """Compute all metrics for all 4 conditions in one pattern dict.

    Returns a nested dict keyed by condition name, each containing:
        causality_score, time_series_score, temporal_connective_score,
        agency_score, unique_entity_count
    """
    CONDITIONS = ["baseline", "w_o_causality", "w_o_time_series", "w_o_agency"]
    result: Dict[str, Dict] = {}
    for cond in CONDITIONS:
        entries = pattern[cond]
        texts     = [e["text"] for e in entries]
        temporals = [e["temporal"] for e in entries]

        c_scores  = [causality_score(t) for t in texts]
        tc_scores = [temporal_connective_score(t) for t in texts]
        tau       = time_series_tau(temporals)
        dom, n_unique = agency_score(texts, nlp)

        result[cond] = {
            "causality_score":            float(np.mean(c_scores)),
            "time_series_score":          tau,
            "temporal_connective_score":  float(np.mean(tc_scores)),
            "agency_score":               dom,
            "unique_entity_count":        n_unique,
        }
    return result
