# Latent Planning: Rate-Distortion Semantic Clustering

Implementation of **Rate-Distortion two-view Gaussian clustering** for analyzing semantic structure in language model continuations. This approach discovers latent semantic components by jointly optimizing over semantic embeddings and attribution features using information-theoretic principles.

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
- **Flexible attribution spans**: Full continuation, distinguishing token, or post-LCS attribution

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
- **D^(e)** = Σ_n P_n · ||e_n - μ_{c(n)}^(e)||² / d_e — semantic distortion
- **D^(a)** = Σ_n P_n · ||a_n - μ_{c(n)}^(a)||² / d_a — attribution distortion
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

## Installation

```bash
# Clone with circuit-tracer submodule
git clone --recursive https://github.com/Merenova/RD_clustering.git
cd RD_clustering

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
pip install -e circuit-tracer/
```

### Requirements
- Python ≥ 3.11
- PyTorch ≥ 2.4.0
- CUDA-capable GPU (recommended: ≥24GB VRAM for full pipeline)

## Usage

### Quick Start with Pipeline

```bash
# Run full pipeline with default config
./run_pipeline.sh

# Run with custom config
CONFIG_FILE=configs/Qwen3_8B_config.json ./run_pipeline.sh

# Run specific stages
./run_pipeline.sh --stages 5,7c

# Resume from a stage
./run_pipeline.sh --resume 5

# Run only one stage
./run_pipeline.sh --only 7c

# Output to custom directory
./run_pipeline.sh --output_dir ./my_experiment
```

### Individual Stage Commands

#### Stage 1: Data Preparation
```bash
uv run python 1_data_preparation/select_test_clozes.py \
    --cloze-dir data/cloze_llm_improved_split_ratio_0.1 \
    --split train \
    --n-samples 50 \
    --mode cloze \
    --output results/test_clozes.json
```

#### Stage 2: Branch Sampling
```bash
uv run python 2_branch_sampling/sample_branches.py \
    --test-clozes results/test_clozes.json \
    --model Qwen/Qwen3-4B \
    --max-n-logits 151936 \
    --desired-logit-prob 0.9 \
    --max-total-continuations 1000 \
    --nucleus-p 0.9 \
    --temperature 1.0 \
    --max-tokens 30 \
    --output-dir results/2_branch_sampling/
```

#### Stage 3: Attribution
```bash
uv run python 3_attribution_graphs/compute_continuation_attribution.py \
    --branches-dir results/2_branch_sampling/ \
    --model Qwen/Qwen3-4B \
    --transcoder mwhanna/qwen3-4b-transcoders \
    --span-mode post_lcs \
    --store-all \
    --output-dir results/3_attribution_graphs/
```

#### Stage 4: Embeddings
```bash
uv run python 4_feature_extraction/compute_embeddings.py \
    --samples-dir results/2_branch_sampling/ \
    --embedding-model google/embeddinggemma-300m \
    --batch-size 32 \
    --output-dir results/4_feature_extraction/embeddings/
```

#### Stage 5: Clustering
```bash
# Single β-γ run
uv run python 5_gaussian_clustering/cluster.py \
    --embeddings-dir results/4_feature_extraction/embeddings/ \
    --attribution-graphs-dir results/3_attribution_graphs/ \
    --samples-dir results/2_branch_sampling/ \
    --beta 5.0 \
    --gamma 0.5 \
    --K-max 20 \
    --pooling mean \
    --output-dir results/5_clustering/

# With parameter sweep (configured in config file)
uv run python 5_gaussian_clustering/cluster.py \
    --config configs/default_config.json \
    --embeddings-dir results/4_feature_extraction/embeddings/ \
    --attribution-graphs-dir results/3_attribution_graphs/ \
    --samples-dir results/2_branch_sampling/ \
    --output-dir results/5_clustering/
```

#### Stage 7c: Steering Validation
```bash
uv run python 7_validation/7c_hypotheses.py \
    --attribution-graphs-dir results/3_attribution_graphs/ \
    --samples-dir results/2_branch_sampling/ \
    --clustering-dir results/5_clustering/ \
    --config configs/default_config.json \
    --output-dir results/7_validation/7c_steering/
```

#### Stage 8: Visualization
```bash
uv run python 8_visualization/visualize.py \
    --config configs/default_config.json \
    --sweep-results-dir results/5_clustering/ \
    --output-dir results/8_visualization/
```

## Configuration

Edit `configs/default_config.json`:

```json
{
  "experiment_name": "my_experiment",
  "random_seed": 42,

  "global": {
    "batch_size": 512,
    "max_seq_len": 64
  },

  "model": {
    "base_model": "Qwen/Qwen3-4B",
    "transcoder": "mwhanna/qwen3-4b-transcoders",
    "dtype": "bfloat16",
    "device": "cuda"
  },

  "sampling": {
    "max_total_continuations": 1000,
    "nucleus_p": 0.9,
    "temperature": 1.0,
    "max_tokens": 30
  },

  "attribution": {
    "span_mode": "post_lcs",
    "store_all": true,
    "max_feature_nodes": 8192,
    "pooling": "mean"
  },

  "clustering": {
    "beta": 5.0,
    "gamma": 0.5,
    "K_max": 20,
    "weight_mode": "perplexity",
    "pooling": "mean",
    "sweeps": {
      "beta_values": [1, 5, 10, 30],
      "gamma_values": [0.1, 0.3, 0.5, 0.7, 0.9],
      "selection_method": "harmonic"
    }
  },

  "stage_7c_steering": {
    "enabled": true,
    "steering_method": "sign",
    "h_c_strategy": "H_c_centered",
    "feature_selection": "distinct",
    "hypotheses": ["H4a", "H4c"],
    "epsilon_values": [-1, -0.5, -0.25, 0, 0.25, 0.5, 1],
    "top_B": 10,
    "sweeps": [
      {
        "name": "sign",
        "steering_method": "sign",
        "h_c_selections": ["full", "positive", "negative"],
        "top_B": [5, 10]
      }
    ]
  }
}
```

### Attribution Span Modes

| Mode | Description |
|------|-------------|
| `full` | Attribute entire continuation |
| `lcs_plus_one` | Attribute up to and including the distinguishing token |
| `post_lcs` | Attribute only tokens after the longest common prefix |

### Steering Methods

| Method | Description |
|--------|-------------|
| `sign` | Set feature activations to ±ε based on H_c sign |
| `multiplicative` | Scale activations: a' = a × (1 + ε × sign(H_c)) |
| `additive` | Add delta: a' = a + ε × H_c |
| `absolute` | Set to absolute value: a' = ε × |H_c| |
| `scaling` | Scale by magnitude: a' = a × (1 + ε × |H_c|) |

## Output Files

### Clustering Results (`results/5_clustering/`)

```
5_clustering/
├── cloze_0000_clustering.json      # Per-prefix clustering results
├── cloze_0000_sweep_results.json   # Sweep results (if enabled)
├── clustering_summary.json         # Aggregate summary
└── best_params.json                # Best β-γ per prefix
```

Each clustering result contains:
```json
{
  "prefix_id": "cloze_0000",
  "prefix": "The capital of France is",
  "n_components": 4,
  "converged": true,
  "components": {
    "0": {
      "mu_e": [...],      // Semantic centroid
      "mu_a": [...],      // Attribution centroid
      "W_c": 0.35         // Component probability mass
    }
  },
  "assignments": [0, 0, 1, 2, ...],
  "rd_objective": {
    "L_RD": 1.234,
    "H": 1.5,
    "D_e": 0.02,
    "D_a": 0.03
  }
}
```

### Steering Results (`results/7_validation/7c_steering/`)

```
7c_steering/
├── cloze_0000_sweep_results.json   # Per-prefix steering results
├── steering_summary.json           # Aggregate metrics
└── h4a_dose_response.json          # H4a: Dose-response curves
```

## Validation Hypotheses

| Hypothesis | Stage | Description |
|------------|-------|-------------|
| **H1** | exp1 | Cluster representations are orthogonal in activation space |
| **H2** | exp2 | Cluster structure predicts hallucination vs factual |
| **H3** | exp3 | Steering cluster features causally shifts generation |
| **H4a** | 7c | Dose-response: steering magnitude correlates with effect |
| **H4c** | 7c | Specificity: steering target cluster affects it most |

## GPU Requirements

| Stage | VRAM | Notes |
|-------|------|-------|
| 2 (Sampling) | ~12GB | vLLM inference (Qwen3-4B) |
| 3 (Attribution) | ~24GB | Circuit-tracer with CPU offload |
| 4 (Embeddings) | ~4GB | EmbeddingGemma-300m |
| 5 (Clustering) | ~2GB | CPU-heavy, minimal GPU |
| 7c (Steering) | ~24GB | On-the-fly steering hooks |

## Development

```bash
# Run tests
uv run pytest tests/

# Format code
uv run ruff format .

# Run linter
uv run ruff check .
```

## Citation

If you use this code, please cite:

```bibtex
@software{latent_planning,
  title = {Latent Planning: Rate-Distortion Semantic Clustering},
  author = {Merenova},
  year = {2026},
  url = {https://github.com/Merenova/RD_clustering}
}
```

## License

MIT License
