# Latent Planning: Rate-Distortion Clustering

Implementation of **Rate-Distortion two-view clustering** for analyzing semantic-attribution structure in language model continuations. This approach discovers latent semantic components by jointly optimizing over semantic embeddings and attribution features using rate-distortion theory.

## Overview

Given a prefix (e.g., "The capital of France is"), this pipeline:
1. Samples diverse continuations from the language model
2. Computes circuit-tracer attributions from prefix features to continuation tokens
3. Extracts semantic embeddings for each continuation
4. Clusters continuations using a Rate-Distortion objective over both views
5. Validates discovered clusters via steering interventions

**Key features:**
- **Rate-Distortion objective**: Information-theoretic clustering with principled trade-offs
- **Two-view clustering**: Joint optimization over semantic (embedding) and mechanistic (attribution) spaces
- **Probability weighting**: Path probabilities weight all component statistics
- **Steering validation**: Causal validation via feature steering interventions
- **Flexible attribution spans**: Full continuation, distinguishing token, or post-LCS attribution (Default: full)

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
| **span_mode** | `attribution.span_mode` | Attribution span: `full`, `lcs_plus_one`, `post_lcs` |