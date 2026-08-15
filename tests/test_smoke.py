"""test_smoke.py – Smoke tests for the narrative-ablation pipeline.

Tests run without GPU or real API calls by using:
  - Minimal in-memory data (2 patterns, 3 events each)
  - Mocked OpenAI / HuggingFace calls
  - The full pipeline logic under controlled inputs

Run:
    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path so we can import modules directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data_construction"))
sys.path.insert(0, str(ROOT / "src" / "tasks"))
sys.path.insert(0, str(ROOT / "src" / "behavioral"))
sys.path.insert(0, str(ROOT / "src" / "probing"))
sys.path.insert(0, str(ROOT / "src" / "manipulation_check"))


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic data
# ---------------------------------------------------------------------------

EVENTS_BASELINE = [
    {"set_id": 1, "temporal": "2012-03-01", "location": "NYC",     "entity": "Alice", "content": "ran", "text": "Alice ran in NYC."},
    {"set_id": 2, "temporal": "2013-06-15", "location": "Boston",  "entity": "Alice", "content": "slept", "text": "Alice slept in Boston."},
    {"set_id": 3, "temporal": "2015-11-20", "location": "Chicago", "entity": "Bob",   "content": "cooked", "text": "Bob cooked in Chicago."},
]

EVENTS_WO_CAUSALITY = [
    {"set_id": 1, "temporal": "2012-03-01", "location": "NYC",     "entity": "Alice", "content": "ran",   "text": "Alice ran."},
    {"set_id": 2, "temporal": "2013-06-15", "location": "Boston",  "entity": "Alice", "content": "slept", "text": "Alice slept."},
    {"set_id": 3, "temporal": "2015-11-20", "location": "Chicago", "entity": "Bob",   "content": "cooked", "text": "Bob cooked."},
]

EVENTS_WO_TIME_SERIES = [
    {"set_id": 1, "temporal": "2015-11-20", "location": "NYC",     "entity": "Alice", "content": "ran",   "text": "Alice ran."},
    {"set_id": 2, "temporal": "2012-03-01", "location": "Boston",  "entity": "Alice", "content": "slept", "text": "Alice slept."},
    {"set_id": 3, "temporal": "2013-06-15", "location": "Chicago", "entity": "Bob",   "content": "cooked", "text": "Bob cooked."},
]

EVENTS_WO_AGENCY = [
    {"set_id": 1, "temporal": "2012-03-01", "location": "NYC",     "entity": "Alice", "content": "ran",   "text": "Someone ran."},
    {"set_id": 2, "temporal": "2013-06-15", "location": "Boston",  "entity": "Bob",   "content": "slept", "text": "Someone slept."},
    {"set_id": 3, "temporal": "2015-11-20", "location": "Chicago", "entity": "Carol", "content": "cooked","text": "Someone cooked."},
]


def make_pattern(pid: int) -> dict:
    return {
        "pattern_id":   pid,
        "original_sets":  EVENTS_BASELINE,
        "derived_sets":   EVENTS_BASELINE,
        "derived_sets2":  EVENTS_WO_TIME_SERIES,
        "baseline":        EVENTS_BASELINE,
        "w_o_causality":   EVENTS_WO_CAUSALITY,
        "w_o_time_series": EVENTS_WO_TIME_SERIES,
        "w_o_agency":      EVENTS_WO_AGENCY,
    }


PATTERNS = [make_pattern(1), make_pattern(2)]
ELEMENTS = {
    "temporal": [e["temporal"] for e in EVENTS_BASELINE] + ["2016-01-01"],
    "location": ["NYC", "Boston", "Chicago", "LA"],
    "entity":   ["Alice", "Bob", "Carol", "Dave"],
    "content":  ["ran", "slept", "cooked", "sang"],
}


# ---------------------------------------------------------------------------
# Test: build_sets
# ---------------------------------------------------------------------------

class TestBuildSets:
    def test_build_original_sets(self):
        import build_sets
        import random
        random.seed(42)
        sets = build_sets.build_original_sets(ELEMENTS, 3)
        assert len(sets) == 3
        # Should be sorted chronologically
        dates = [s["temporal"] for s in sets]
        assert dates == sorted(dates)

    def test_build_derived_sets(self):
        import build_sets
        orig = [
            {"set_id": i + 1, "temporal": f"201{i}-01-01",
             "location": f"L{i}", "entity": f"E{i}", "content": f"C{i}"}
            for i in range(3)
        ]
        derived = build_sets.build_derived_sets(orig)
        assert len(derived) == 3
        # First element unchanged
        assert derived[0]["location"] == orig[0]["location"]

    def test_build_temporal_shuffled(self):
        import build_sets
        import random
        random.seed(0)
        derived = [
            {"set_id": i + 1, "temporal": f"201{i}-01-01",
             "location": "L", "entity": "E", "content": "C"}
            for i in range(4)
        ]
        shuffled = build_sets.build_temporal_shuffled_sets(derived)
        assert len(shuffled) == 4
        orig_temps = [d["temporal"] for d in derived]
        shuf_temps = [s["temporal"] for s in shuffled]
        # Same multiset
        assert sorted(orig_temps) == sorted(shuf_temps)


# ---------------------------------------------------------------------------
# Test: build_questions (single-hop and multi-hop)
# ---------------------------------------------------------------------------

class TestBuildQuestions:
    def test_single_hop_generates_questions(self):
        from build_questions import build_single_hop_questions
        qs = build_single_hop_questions(PATTERNS)
        assert len(qs) > 0
        for q in qs:
            assert "question" in q
            assert "answer" in q
            assert q["variant"] in ("baseline", "w_o_causality", "w_o_time_series", "w_o_agency")

    def test_multi_hop_generates_questions(self):
        from build_questions import build_multi_hop_questions
        # Patterns need a shared field in consecutive events
        # Make events with shared location
        events = [
            {"set_id": 1, "temporal": "2012-01-01", "location": "NYC", "entity": "Alice", "content": "ran",  "text": "T1"},
            {"set_id": 2, "temporal": "2013-01-01", "location": "NYC", "entity": "Bob",   "content": "sang", "text": "T2"},
        ]
        pattern = {
            "pattern_id": 99,
            "derived_sets":  events,
            "derived_sets2": events,
            "baseline":        events,
            "w_o_causality":   events,
            "w_o_time_series": events,
            "w_o_agency":      events,
        }
        qs = build_multi_hop_questions([pattern])
        assert any(q["question_type"] == "f" for q in qs)  # location shared → type f


# ---------------------------------------------------------------------------
# Test: behavioral judge (mocked API)
# ---------------------------------------------------------------------------

class TestJudge:
    def test_exact_match_shortcut(self):
        from judge import judge_pair
        # Exact match skips API call entirely
        mock_client = MagicMock()
        score, reason, usage, cost = judge_pair(mock_client, "New York", "New York")
        assert score == 1.0
        assert "exact match" in reason.lower()
        mock_client.chat.completions.create.assert_not_called()

    def test_normalization(self):
        from judge import judge_pair
        mock_client = MagicMock()
        # Whitespace-normalised exact match
        score, reason, usage, cost = judge_pair(mock_client, "New   York", "New York")
        assert score == 1.0


# ---------------------------------------------------------------------------
# Test: manipulation check metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_causality_score_zero_for_neutral_text(self):
        from metrics import causality_score
        assert causality_score("She walked to the store.") == 0.0

    def test_causality_score_positive_for_causal_text(self):
        from metrics import causality_score
        assert causality_score("She went there because she was hungry.") > 0.0

    def test_time_series_tau_sorted(self):
        from metrics import time_series_tau
        temporals = ["2010-01-01", "2012-06-15", "2015-11-20"]
        tau = time_series_tau(temporals)
        assert tau == pytest.approx(1.0)

    def test_time_series_tau_reversed(self):
        from metrics import time_series_tau
        temporals = ["2015-11-20", "2012-06-15", "2010-01-01"]
        tau = time_series_tau(temporals)
        assert tau == pytest.approx(-1.0)

    def test_agency_score(self):
        from metrics import agency_score
        mock_nlp = MagicMock()
        # Simulate spaCy returning PERSON entities
        doc1 = MagicMock()
        doc1.ents = [MagicMock(label_="PERSON", text="Alice"), MagicMock(label_="PERSON", text="Alice")]
        doc2 = MagicMock()
        doc2.ents = [MagicMock(label_="PERSON", text="Alice")]
        mock_nlp.pipe.return_value = [doc1, doc2]
        dominance, n_unique = agency_score(["Alice ran.", "Alice slept."], mock_nlp)
        assert dominance == pytest.approx(1.0)
        assert n_unique == 1


# ---------------------------------------------------------------------------
# Test: probes (pairwise samples)
# ---------------------------------------------------------------------------

class TestProbes:
    def test_make_pairwise_samples_shape(self):
        from probes import make_pairwise_samples
        X = np.random.randn(2, 10, 32).reshape(20, 32).astype(np.float32)
        ranks = np.tile(np.arange(1, 11), 2).astype(np.float32)
        Xp, yp = make_pairwise_samples(X, ranks, n_patterns=2)
        # C(10, 2) = 45 pairs per pattern × 2 patterns = 90
        assert Xp.shape == (90, 32)
        assert yp.shape == (90,)
        assert set(yp.tolist()) <= {0, 1}

    def test_apply_pca_reduces_dims(self):
        from probes import apply_pca
        X_tr = np.random.randn(100, 256).astype(np.float32)
        X_te = np.random.randn(20,  256).astype(np.float32)
        X_tr_pca, X_te_pca, var = apply_pca(X_tr, X_te, n_components=64)
        assert X_tr_pca.shape == (100, 64)
        assert X_te_pca.shape == (20,  64)
        assert 0.0 <= var <= 1.0 + 1e-6

    def test_fit_and_predict_logistic(self):
        from probes import fit_logistic, eval_binary
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 8)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int8)
        w, b = fit_logistic(X[:160], y[:160])
        y_pred = (X[160:] @ w + b.squeeze() > 0).astype(np.int8)
        result = eval_binary(y[160:], y_pred)
        assert "balanced_accuracy" in result
        assert result["balanced_accuracy"] > 0.6  # should be learnable


# ---------------------------------------------------------------------------
# Test: PMS computation
# ---------------------------------------------------------------------------

class TestPMS:
    def test_pms_induction_pattern(self):
        """PMS should be 1.0 for a perfect induction-head attention pattern."""
        from pms import compute_pms
        # 5-token sequence: [A, B, A, B, ?]
        # token_ids = [0, 1, 0, 1, 2]
        # Induction head: position 4 should attend to position 2 (after prior A at pos 0)
        token_ids   = np.array([0, 1, 0, 1, 2])
        seq_len     = 5
        num_heads   = 1
        attn_weights = np.zeros((num_heads, seq_len, seq_len))
        # Perfect induction: pos 2 attends to pos 1 (j=0, j+1=1)
        # pos 3 attends to pos 1 (j=0, j+1=1) – wait, token[3]=1, prior pos with 1 is pos 1, j=1→j+1=2
        # Actually for pos i=2: token_ids[2]=0, prior same tokens at j=0 (j<=i-2=0); attend to j+1=1
        attn_weights[0, 2, 1] = 1.0
        # For pos i=3: token_ids[3]=1, prior same token at j=1 (j<=i-2=1); attend to j+1=2
        attn_weights[0, 3, 2] = 1.0
        scores = compute_pms(attn_weights, token_ids)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0)

    def test_pms_uniform_is_lower_than_induction(self):
        """Uniform attention should give lower PMS than a perfect induction head."""
        from pms import compute_pms
        token_ids = np.array([0, 1, 0, 1, 0])
        seq_len   = 5
        attn_unif = np.ones((1, seq_len, seq_len)) / seq_len
        attn_ind  = np.zeros((1, seq_len, seq_len))
        # Perfect induction
        attn_ind[0, 2, 1] = 1.0
        attn_ind[0, 3, 2] = 1.0
        attn_ind[0, 4, 1] = 1.0   # token[4]=0, prior at j=0, j=2; j+1 = 1 or 3
        pms_unif = compute_pms(attn_unif, token_ids)[0]
        pms_ind  = compute_pms(attn_ind, token_ids)[0]
        assert pms_ind >= pms_unif
