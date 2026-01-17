# Mechanistic Motif Discovery via Rate-Distortion Clustering

Unsupervised discovery of mechanistic motifs—clusters of LLM continuations that share both semantic meaning and internal computational patterns.

## Overview

When language models generate diverse continuations from the same prompt, different outputs may arise from different internal reasoning paths. This framework discovers these **mechanistic motifs** without requiring labels or pre-specified behaviors.

### Key Features

- **Two-View Representation**: Each continuation is represented by both semantic embeddings (meaning) and attribution embeddings (internal computation)
- **Rate-Distortion Clustering**: Automatically discovers the optimal number of motifs by balancing compression and reconstruction
- **Multi-Granularity Analysis**: Adjust β to explore motifs at different levels of abstraction
- **Steering Validation**: Verify discovered motifs are causally meaningful via activation interventions

## Method

### Pipeline
```
Prefix → Sample Continuations → Two-View Embedding → RD Clustering → Motifs
                                      ↓                    ↓
                              [Semantic, Attribution]  [Automatic K]
                                                           ↓
                                               Steering Validation
```

### Two-View Embedding

| View | Source | Captures |
|------|--------|----------|
| Semantic | Pretrained embedding model | What the continuation means |
| Attribution | Transcoder-based attribution | How the model computed it |

### Rate-Distortion Objective
```
L_RD = H(C) + β_e · D^(e) + β_a · D^(a)
       ───────   ─────────   ─────────
        Rate     Semantic    Attribution
                 Distortion  Distortion
```

- **Rate** H(C): Entropy of motif distribution (probability-weighted)
- **Distortion**: Reconstruction error (L2 for semantic, L1 for attribution)
- **β = β_e + β_a**: Controls number of motifs (higher → more motifs)
- **γ = β_e / β**: Balances semantic vs attribution (higher → more semantic-focused)

### Automatic K Selection

The number of motifs K is not pre-specified. Adaptive split/merge operations derived from the objective automatically determine K:

- **Split**: When distortion reduction > entropy cost
- **Merge**: When entropy savings > distortion increase

## Installation
```bash
git clone https://github.com/[username]/mechanistic-motif-discovery.git
cd mechanistic-motif-discovery
pip install -r requirements.txt
```

## Quick Start
```python
from motif_discovery import MotifDiscovery

# Initialize
md = MotifDiscovery(
    model="Qwen3-8B",
    embedding_model="gemma-embedding",
    transcoder="gemma-scope"
)

# Sample continuations
continuations = md.sample(prefix="The capital of France is", n=100)

# Discover motifs
motifs = md.cluster(continuations, beta=5.0, gamma=0.5)

# Inspect results
for motif in motifs:
    print(f"Motif {motif.id}: {motif.prior:.2%} of continuations")
    print(f"  Top features: {motif.top_features[:5]}")
    print(f"  Example: {motif.examples[0]}")

# Validate via steering
steering_results = md.validate_steering(motifs)
```

## Applications

### Safety Analysis
Discover different refusal mechanisms in safety-trained models:
- Are all refusals using the same internal pathway?
- Which refusal patterns are robust vs brittle?

### Hallucination Detection
Identify faithful vs shortcut reasoning:
- Correct answers from genuine knowledge retrieval
- Correct answers from pattern matching (brittle)


## Project Structure

```
latent_planning/
├── 0_preprocess/                    # Data preprocessing (optional)
│   ├── improve_clozes.py            # Rule-based cloze improvement
│   ├── llm_improve_clozes.py        # LLM-based cloze improvement
│   └── split_dataset.py             # Train/test splitting
├── 1_data_preparation/              # Stage 1: Select test prefixes
│   └── select_test_clozes.py
├── 2_branch_sampling/               # Stage 2: Sample continuations
│   └── sample_branches.py           # vLLM sampling with first-token discovery
├── 3_attribution_graphs/            # Stage 3: Compute attributions
│   ├── compute_continuation_attribution.py  # Prefix→continuation attribution
│   └── attribution_formula.tex
├── 4_feature_extraction/            # Stage 4: Extract embeddings
│   └── compute_embeddings.py        # Contextual continuation embeddings
├── 5_gaussian_clustering/           # Stage 5: R-D clustering
│   ├── cluster.py                   # Main orchestrator (single/sweep mode)
│   ├── rd_objective.py              # R-D objective computation
│   ├── em_loop.py                   # E-step + M-step
│   ├── adaptive_control.py          # Split operations
│   ├── initialize.py                # Single-component initialization
│   ├── sweep_utils.py               # Parameter sweep utilities
│   └── gpu_utils.py                 # GPU memory management
├── 6_semantic_graphs/               # Stage 6: Extract semantic graphs
│   ├── extract_graphs.py            # H_c, token scores extraction
│   └── visualize.py                 # Basic visualization
├── 7_validation/                    # Stage 7: Validation
│   ├── 7a_graph_validation.py       # Attribution graph quality
│   ├── 7c_hypotheses.py             # Steering validation entry point
│   ├── 7c_steering.py               # Steering hooks and forward passes
│   ├── 7c_graph.py                  # Graph loading utilities
│   ├── 7c_metrics.py                # Validation metrics
│   └── 7c_utils.py                  # Shared utilities
├── 8_visualization/                 # Stage 8: Visualization
│   ├── visualize.py                 # Main visualization orchestrator
│   ├── tsne_plots.py                # t-SNE clustering plots
│   ├── sankey_plots.py              # Sankey flow diagrams
│   ├── cluster_plots.py             # Clustering history plots
│   ├── semantic_graph_plots.py      # Semantic graph heatmaps
│   ├── parameter_sweep_plots.py     # Sweep analysis plots
│   └── html_explorer.py             # Interactive HTML explorers
├── hypotheses_experiment/           # Standalone hypothesis testing
│   ├── exp1_orthogonality.py        # H1: Cluster orthogonality
│   ├── exp2_hallucination_detection.py  # H2: Hallucination detection
│   ├── exp3_causal_validation.py    # H3: Causal steering validation
│   ├── generate_data.py             # Data generation for experiments
│   ├── run_all.py                   # Run all experiments
│   └── run_popqa.py                 # PopQA evaluation
├── circuit-tracer/                  # Attribution library (workspace member)
├── utils/                           # Shared utilities
│   ├── config.py                    # Configuration classes
│   ├── data_utils.py                # Data loading/saving
│   ├── logging_utils.py             # Centralized logging
│   ├── manifest.py                  # Stage manifest tracking
│   └── memory_utils.py              # GPU memory utilities
├── configs/                         # Configuration files
│   ├── default_config.json          # Main configuration
│   ├── Qwen3_8B_config.json         # Qwen3-8B settings
│   └── gemma3_4b_config.json        # Gemma3-4B settings
├── data/                            # Input data
├── run_pipeline.sh                  # Full pipeline runner
└── pyproject.toml                   # Project dependencies (uv)
```

## Rate-Distortion Objective

The algorithm optimizes:

```
L_RD = H(C) + β_e · D^(e) + β_a · D^(a)
```

Where:
- **H(C)** = -Σ_c P̄_c · log(P̄_c) — entropy (rate/compression term)
- **D^(e)** = Σ_n P_n · ||e_n - μ_{c(n)}^(e)||² / d_e — semantic distortion (L2 loss)
- **D^(a)** = Σ_n P_n · ||a_n - μ_{c(n)}^(a)|| / d_a — attribution distortion (L1 loss)
- **β = β_e + β_a** — total precision (inverse temperature)
- **γ = β_e / β** — view ratio (semantic vs attribution weighting)

### Key Parameters

| Parameter | Config Key | Description |
|-----------|------------|-------------|
| **β** | `clustering.beta` | Total precision. Higher = more clusters |
| **γ** | `clustering.gamma` | View ratio. 0.5 = equal weight; >0.5 favors semantic |
| **K_max** | `clustering.K_max` | Maximum number of components |
| **pooling** | `clustering.pooling` | Attribution pooling: `mean`, `max`, `sum` |


## Citation
TBD

## License

MIT License
