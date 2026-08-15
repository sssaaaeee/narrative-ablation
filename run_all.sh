#!/usr/bin/env bash
# run_all.sh – End-to-end pipeline for the narrativity ablation study.
#
# Phases:
#   Phase 1 – Data construction (§3)
#   Phase 2 – Manipulation check (Appendix A.4–A.5)
#   Phase 3 – Task (question) generation (Appendix B)
#   Phase 4 – Behavioral evaluation (§5)
#   Phase 5 – Linear probing (§6)
#   Phase 6 – Mechanistic analysis (§7)
#   Phase 7 – Figure generation
#
# Prerequisites:
#   pip install -r requirements.txt
#   python -m spacy download en_core_web_sm
#   .env file with OPENROUTER_API_KEY_1 and OPENAI_API_KEY (for judge)
#
# Estimated GPU memory: ≥ 24 GB for 4-bit quantised Llama-8B.
# Adjust --hf-model and --load-in-* flags for larger models.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
HF_MODEL="${HF_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
NUM_PATTERNS="${NUM_PATTERNS:-300}"
N_ANALYSIS_PATTERNS="${N_ANALYSIS_PATTERNS:-20}"
OUT_DIR="results"
FIG_DIR="figures"
MODEL_TAG=$(basename "$HF_MODEL")

echo "=================================================================="
echo "  Narrativity Ablation Study – Full Pipeline"
echo "  Model:    $HF_MODEL"
echo "  Patterns: $NUM_PATTERNS"
echo "=================================================================="

# ─── Phase 1: Data construction ───────────────────────────────────────────────
echo ""
echo "Phase 1 – Data construction"

echo "  1a. Building element pools …"
python src/data_construction/build_pools.py \
    --elements data/pools/elements.json \
    --out-dir  data/pools/

echo "  1b. Building element sets (${NUM_PATTERNS} patterns) …"
python src/data_construction/build_sets.py \
    --elements    data/pools/elements.json \
    --out         data/sets/generated_sets.json \
    --num-patterns "$NUM_PATTERNS" \
    --num-sets    10 \
    --seed        42

echo "  1c. Generating narrative passages (4 conditions) …"
python src/data_construction/generate_passages.py \
    --sets  data/sets/generated_sets.json \
    --out   data/passages/generated_story.json

# ─── Phase 2: Manipulation check ──────────────────────────────────────────────
echo ""
echo "Phase 2 – Manipulation check"

python src/manipulation_check/run_check.py \
    --passages data/passages/generated_story.json \
    --out      "$OUT_DIR/manipulation_check_results.json"

# ─── Phase 3: Question generation ─────────────────────────────────────────────
echo ""
echo "Phase 3 – Question generation"

python src/tasks/build_questions.py \
    --passages data/passages/generated_story.json \
    --out-dir  data/questions/ \
    --mode     all

# ─── Phase 4: Behavioral evaluation ───────────────────────────────────────────
echo ""
echo "Phase 4 – Behavioral QA"

echo "  4a. Collecting answers …"
python src/behavioral/run_qa.py \
    --hf-model      "$HF_MODEL" \
    --load-in-4bit \
    --story-path    data/passages/generated_story.json \
    --single-q-path data/questions/single_hop_questions.json \
    --multi-q-path  data/questions/multi_hop_questions.json \
    --out-dir       "$OUT_DIR" \
    --max-new-tokens 32 \
    --mode          all

echo "  4b. LLM-judge evaluation …"
python src/behavioral/judge.py \
    --single-answers "$OUT_DIR/answers_single_all.json" \
    --multi-answers  "$OUT_DIR/answers_multi_all.json" \
    --out-dir        "$OUT_DIR" \
    --mode           all

echo "  4c. Statistical tests …"
python src/behavioral/stats.py \
    --eval-dirs "$OUT_DIR:$MODEL_TAG" \
    --mode      all

# ─── Phase 5: Linear probing ──────────────────────────────────────────────────
echo ""
echo "Phase 5 – Linear probing"

PROBE_DIR="$OUT_DIR/probing/$MODEL_TAG"

echo "  5a. Extracting hidden states …"
python src/probing/extract_hidden_states.py \
    --hf-model    "$HF_MODEL" \
    --load-in-4bit \
    --story-path  data/passages/generated_story.json \
    --elements    data/pools/elements.json \
    --out-dir     "$OUT_DIR/probing" \
    --conditions  baseline w_o_causality w_o_time_series w_o_agency \
    --layer-start 0 \
    --layer-end   31

echo "  5b. Training and evaluating probes …"
python src/probing/probes.py \
    --features-dir "$PROBE_DIR" \
    --probe-type   all \
    --conditions   baseline w_o_causality w_o_time_series w_o_agency \
    --pca-dim      256 \
    --out-dir      "$PROBE_DIR"

echo "  5c. Cross-condition time probe …"
python src/probing/cross_condition.py \
    --features-dir    "$PROBE_DIR" \
    --train-condition baseline \
    --test-condition  w_o_time_series \
    --pca-dim         256 \
    --out             "$PROBE_DIR/cross_time_baseline_to_wots.json"

echo "  5d. Probe significance tests …"
for PROBE_TYPE in time entity_id common_element; do
    python src/probing/stats.py \
        --probe-dir  "$PROBE_DIR" \
        --probe-type "$PROBE_TYPE" \
        --cond-a     baseline \
        --cond-b     "w_o_time_series"
done

# ─── Phase 6: Mechanistic analysis ────────────────────────────────────────────
echo ""
echo "Phase 6 – Mechanistic analysis"

MECH_DIR="$OUT_DIR/mechanistic/$MODEL_TAG"

echo "  6a. Computing PMS (induction head scores) …"
python src/mechanistic/pms.py \
    --hf-model    "$HF_MODEL" \
    --load-in-4bit \
    --story-path  data/passages/generated_story.json \
    --out-dir     "$OUT_DIR/mechanistic" \
    --conditions  baseline w_o_causality w_o_time_series w_o_agency \
    --n-patterns  "$N_ANALYSIS_PATTERNS"

echo "  6b. PMS statistical comparison …"
python src/mechanistic/pms_stats.py \
    --pms-dir      "$MECH_DIR" \
    --baseline-file pms_baseline.json \
    --compare-files pms_w_o_causality.json pms_w_o_agency.json pms_w_o_time_series.json \
    --top-k        20

echo "  6c. Attention analysis …"
python src/mechanistic/attention_analysis.py \
    --hf-model    "$HF_MODEL" \
    --load-in-4bit \
    --story-path  data/passages/generated_story.json \
    --out-dir     "$OUT_DIR/mechanistic" \
    --conditions  baseline w_o_causality w_o_time_series w_o_agency \
    --n-patterns  "$N_ANALYSIS_PATTERNS"

# ─── Phase 7: Figures ─────────────────────────────────────────────────────────
echo ""
echo "Phase 7 – Generating figures"

python src/figures/plot_behavioral.py \
    --eval-dirs "$OUT_DIR:$MODEL_TAG" \
    --out-dir   "$FIG_DIR"

python src/figures/plot_probing.py \
    --probe-dir "$PROBE_DIR" \
    --out-dir   "$FIG_DIR" \
    --model-tag "$MODEL_TAG"

python src/figures/plot_mechanistic.py \
    --mech-dir  "$MECH_DIR" \
    --out-dir   "$FIG_DIR" \
    --model-tag "$MODEL_TAG"

echo ""
echo "=================================================================="
echo "  Pipeline complete."
echo "  Results in: $OUT_DIR/"
echo "  Figures in: $FIG_DIR/"
echo "=================================================================="
