# Narrative Ablation Study – Reproducibility Package

Code for the experiments in **"What Survives Narrative Ablation? Dissociating Temporal Structure from
Causal and Agency Information in LLM Representations"**.

We investigate whether large language models (LLMs) internally encode the three
defining dimensions of narrativity—**causal structure**, **temporal coherence**,
and **protagonist agency**—by ablating each dimension independently across
300 synthetic narrative patterns and four LLM sizes (7 B – 72 B parameters).

---

## Pipeline overview

```
Phase 1  Data construction    src/data_construction/   §3
Phase 2  Manipulation check   src/manipulation_check/  Appendix A.4–A.5
Phase 3  Task generation      src/tasks/               Appendix B
Phase 4  Behavioral QA        src/behavioral/          §5
Phase 5  Linear probing       src/probing/             §6
Phase 6  Mechanistic analysis src/mechanistic/         §7, Appendix E–F
Phase 7  Figures              src/figures/
```

**Run the complete pipeline in one command:**
```bash
HF_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct bash run_all.sh
```

---

## Directory structure

```
narrative-ablation/
├── README.md
├── requirements.txt
├── run_all.sh                       # End-to-end pipeline (Phases 1–7)
├── configs/
│   ├── data.yaml                    # #patterns, seed, generation model
│   ├── model.yaml                   # HF model, quantisation, max_new_tokens
│   ├── probe.yaml                   # PCA dim, split ratio, layer range
│   └── analysis.yaml                # n_patterns, top-K, α
├── src/
│   ├── data_construction/           # Phase 1
│   │   ├── build_pools.py           # 4-pool extraction + deduplication
│   │   ├── build_sets.py            # original / derived / temporal-shuffled sets
│   │   ├── generate_passages.py     # 4-condition passage generation (OpenRouter)
│   │   └── prompts/                 # Prompt templates (Appendix A.2)
│   ├── manipulation_check/          # Phase 2
│   │   ├── metrics.py               # Causal-marker lexicon, Kendall τ, agency share
│   │   ├── run_check.py             # Wilcoxon + Holm correction → Table 7
│   │   └── human_eval/              # Pair-creation + aggregation scripts (Table 9)
│   ├── tasks/                       # Phase 3
│   │   └── build_questions.py       # Single-hop 12 types / multi-hop 10 types
│   ├── behavioral/                  # Phase 4
│   │   ├── run_qa.py                # 4-bit greedy decoding, max_new_tokens=32
│   │   ├── judge.py                 # LLM-as-judge (GPT-3.5-turbo)
│   │   └── stats.py                 # McNemar + BH correction
│   ├── probing/                     # Phase 5
│   │   ├── extract_hidden_states.py # Layer-wise extraction via forward hooks
│   │   ├── probes.py                # Time / Entity ID Local / Common Element probes
│   │   ├── cross_condition.py       # Transfer: baseline → w/o time-series
│   │   └── stats.py                 # McNemar / χ² + BH (32 layers)
│   ├── mechanistic/                 # Phase 6
│   │   ├── pms.py                   # Prefix Matching Score (Olsson et al., 2022)
│   │   ├── pms_stats.py             # Wilcoxon + Bonferroni
│   │   └── attention_analysis.py    # Entropy, event-to-event map
│   └── figures/                     # Phase 7
│       ├── plot_behavioral.py
│       ├── plot_probing.py
│       └── plot_mechanistic.py
├── data/
│   ├── pools/                       # elements.json + pool_*.json
│   ├── sets/                        # generated_sets.json
│   ├── passages/                    # generated_story.json
│   └── questions/                   # single_hop_questions.json, multi_hop_questions.json
├── results/                         # CSV/JSON outputs (eval, probe accuracy, PMS)
├── figures/                         # Publication-ready PDF figures
└── tests/
    └── test_smoke.py                # Mock-based smoke tests (no GPU/API required)
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. API keys (.env file in repo root)
cp ../.env .env          # or create:
# OPENROUTER_API_KEY_1=sk-or-...
# OPENAI_API_KEY=sk-...
# HF_TOKEN=hf_...
```

---

## Running individual phases

### Phase 1 – Data construction (§3)

Place `elements.json` in `data/pools/`, then:

```bash
# Build pools
python src/data_construction/build_pools.py

# Build 300 element sets
python src/data_construction/build_sets.py --num-patterns 300

# Generate 4-condition passages (~$8 USD at GPT-3.5-turbo rates)
python src/data_construction/generate_passages.py
```

### Phase 2 – Manipulation check (Appendix A.5)

```bash
python src/manipulation_check/run_check.py \
    --passages data/passages/generated_story.json
```

Outputs `results/manipulation_check_results.json` with the 3 × 4 Wilcoxon
matrix and Holm-corrected p-values (Table 7).

### Phase 3 – Question generation (Appendix B)

```bash
python src/tasks/build_questions.py --mode all
```

### Phase 4 – Behavioral QA (§5)

```bash
# Collect answers (GPU required, ~2 h for 8B on A100)
python src/behavioral/run_qa.py --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --load-in-4bit --mode all

# Judge answers (~$15 USD at GPT-3.5-turbo rates)
python src/behavioral/judge.py --mode all

# Significance tests
python src/behavioral/stats.py --eval-dirs results/:Llama-8B
```

### Phase 5 – Linear probing (§6)

```bash
# Extract hidden states (GPU required, ~4 h for 8B)
python src/probing/extract_hidden_states.py \
    --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct --load-in-4bit

# Train and evaluate probes
python src/probing/probes.py \
    --features-dir results/probing/Meta-Llama-3.1-8B-Instruct/ \
    --out-dir results/probing/Meta-Llama-3.1-8B-Instruct/

# Cross-condition transfer
python src/probing/cross_condition.py \
    --features-dir results/probing/Meta-Llama-3.1-8B-Instruct/ \
    --out results/probing/Meta-Llama-3.1-8B-Instruct/cross_time.json
```

### Phase 6 – Mechanistic analysis (§7)

```bash
# Prefix Matching Score
python src/mechanistic/pms.py \
    --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct --load-in-4bit \
    --n-patterns 20

# PMS statistical comparison
python src/mechanistic/pms_stats.py \
    --pms-dir results/mechanistic/Meta-Llama-3.1-8B-Instruct/ --top-k 20

# Attention entropy + event maps
python src/mechanistic/attention_analysis.py \
    --hf-model meta-llama/Meta-Llama-3.1-8B-Instruct --load-in-4bit
```

---

## Smoke tests

```bash
pytest tests/test_smoke.py -v
```

Tests run entirely in-memory with mocked API and GPU dependencies.

---

## Estimated costs and GPU time

| Phase | Cost (USD) | GPU time (A100, 8B) |
|-------|-----------|---------------------|
| 1c – Passage generation (GPT-3.5) | ~8 | — |
| 4a – QA collection (8B, 4-bit) | — | ~2 h |
| 4b – Judge evaluation (GPT-3.5) | ~15 | — |
| 5a – Hidden state extraction | — | ~4 h |
| 6a – PMS (20 patterns) | — | ~30 min |
| 6c – Attention analysis | — | ~30 min |

Costs scale linearly with number of patterns and model size.

---

## Narrative conditions

| Condition | Causal links | Temporal order | Fixed protagonist |
|-----------|:---:|:---:|:---:|
| `baseline` | ✓ | ✓ | ✓ |
| `w_o_causality` | ✗ | ✓ | ✓ |
| `w_o_time_series` | ✓ | ✗ | ✓ |
| `w_o_agency` | ✓ | ✓ | ✗ |

---

## Reference

Olsson, C., Elhage, N., Nanda, N., et al. (2022).
*In-context Learning and Induction Heads.*
Transformer Circuits Thread.
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
