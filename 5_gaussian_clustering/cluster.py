#!/usr/bin/env -S uv run python
"""Main clustering orchestrator for rate-distortion Gaussian optimization.

Implements Algorithm 1 from rate_distortion.tex:
Rate-Distortion Two-View Clustering

Objective: L_RD = H(C) + β_e D^(e) + β_a D^(a)

Features:
- Single-component initialization (K emerges from optimization)
- Exact R-D criteria for split (no tunable thresholds)
"""

import sys

# Prevent version conflicts with global packages
sys.path = [p for p in sys.path if 'python3.12' not in p]

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

# Add circuit-tracer to path (relative to project root)
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuit_tracer.graph import Graph
from utils.config import PathConfig
from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger, get_log_path
from utils.manifest import filter_samples_by_manifest, update_manifest_with_results


# Span computation functions (for recomputing spans when span_mode changes)
def compute_lcs_length(cont1: List[int], cont2: List[int]) -> int:
    """Compute longest common prefix length between two token sequences."""
    for i, (a, b) in enumerate(zip(cont1, cont2)):
        if a != b:
            return i
    return min(len(cont1), len(cont2))


def get_min_distinguishing_index(target_idx: int, all_continuations: List[List[int]]) -> int:
    """Find earliest index where target differs from all others."""
    target = all_continuations[target_idx]
    max_lcs = 0
    for i, other in enumerate(all_continuations):
        if i != target_idx:
            lcs = compute_lcs_length(target, other)
            max_lcs = max(max_lcs, lcs)
    return max_lcs


def get_span_indices(cont_idx: int, all_continuations: List[List[int]], span_mode: str) -> tuple:
    """Get (start, end) indices for attribution span."""
    n = len(all_continuations[cont_idx])
    if span_mode == "full":
        return 0, n
    min_distinguishing_index = get_min_distinguishing_index(cont_idx, all_continuations)
    if span_mode == "lcs_plus_one":
        return 0, min(min_distinguishing_index + 1, n)
    elif span_mode == "post_lcs":
        return min(min_distinguishing_index, n - 1), n
    return 0, n


def compute_span_info(all_continuations: List[List[int]], span_mode: str) -> List[Dict]:
    """Compute span info for all continuations."""
    span_info = []
    for i in range(len(all_continuations)):
        start, end = get_span_indices(i, all_continuations, span_mode)
        span_info.append({
            "start": start,
            "end": end,
            "span_length": end - start,
            "continuation_length": len(all_continuations[i]),
        })
    return span_info

# Import R-D clustering modules
from initialize import initialize_single_component, compute_initial_statistics
from em_loop import run_em_iteration, check_convergence, GPU_AVAILABLE
from adaptive_control import apply_adaptive_control
from rd_objective import (
    compute_component_masses,
    compute_normalized_masses,
    compute_full_rd_statistics,
)

from sweep_utils import run_sweep_mode


def normalize_to_token_probability(path_probs, first_tokens, token_probs,
                                   include_first_token_prob=True):
    """
    Normalize continuation probabilities with optional first token weighting.

    Args:
        path_probs: Raw continuation probabilities (exp(logprob))
        first_tokens: Token ID for each continuation
        token_probs: Dict mapping token_id -> P(token) from vLLM
        include_first_token_prob: If True, weight by P(first_token) so that
            P(cont) = P(first_token) * P(cont|first_token) / Σ P(cont'|first_token)
            If False, use uniform weighting across first tokens:
            P(cont) = P(cont|first_token) / Σ P(cont'|first_token)

    Returns:
        Normalized probabilities
    """
    P_corrected = np.zeros_like(path_probs)

    for s in np.unique(first_tokens):
        mask = (first_tokens == s)

        # Sum of raw path probs for this token (empirical mass)
        empirical_mass = path_probs[mask].sum()

        if include_first_token_prob:
            # Weight by P(first_token) from vLLM logits
            true_prob = token_probs.get(int(s), 1e-10)
        else:
            # Equal weight for all first tokens (drop P(first_token))
            true_prob = 1.0

        # Rescale: preserve relative weights within token
        if empirical_mass > 0:
            P_corrected[mask] = path_probs[mask] * (true_prob / empirical_mass)

    # Do NOT normalize globally - P_bar_c = W_c / W_total handles that for entropy
    return P_corrected


def compute_global_mean(attributions_a: np.ndarray, path_probs: np.ndarray) -> np.ndarray:
    """Compute H_0 = probability-weighted global mean of attributions.

    This is the shared component that captures what is common across all
    continuations, weighted by their probabilities.

    Args:
        attributions_a: Attribution vectors, shape (n_samples, n_features)
        path_probs: Normalized path probabilities, shape (n_samples,)

    Returns:
        H_0: Global mean attribution vector, shape (n_features,)
    """
    W_total = path_probs.sum()
    if W_total == 0:
        return np.zeros(attributions_a.shape[1])
    return np.sum(path_probs[:, None] * attributions_a, axis=0) / W_total


def load_prefix_data(
    prefix_id: str,
    embeddings_dir: Path,
    attribution_graphs_dir: Path,
    samples_dir: Path,
    logger,
    weight_mode: str = "probability",
    include_first_token_prob: bool = True,
    use_continuation_attribution: bool = True,
    pooling: str = "mean",
    span_mode: str = None,
):
    """Load all data for a single prefix.

    Args:
        weight_mode: "probability" for raw P(continuation),
                     "perplexity" for exp(logprob/num_tokens) = 1/PPL_per_token
                     This normalizes by continuation length.
        include_first_token_prob: If True, weight by P(first_token) in normalization.
                     If False, drop P(first_token) term.
        use_continuation_attribution: If True, load pre-aggregated continuation attributions
                     from Stage 3 (latent_planning pipeline). If False, load from Graph.pt
                     (gaussian_optimization legacy format).
        pooling: Pooling method for aggregating token attributions ("mean", "max", "sum").
        span_mode: Span mode for slicing token attributions. If None, uses span_mode from metadata.

    Returns dict with embeddings_e, attributions_a, path_probs, first_tokens, token_list
    """
    logger.info(f"Loading data for prefix: {prefix_id}")

    # Load branch samples data
    branches_file = samples_dir / f"{prefix_id}_branches.json"
    branches_data = load_json(branches_file)
    prefix = branches_data["prefix"]

    # Extract path probabilities, first token IDs, and first token probabilities
    path_probs_original = []
    first_tokens = []
    token_probs = {}  # token_id -> P(token) from vLLM
    token_id_to_text = {}  # token_id -> token_text for visualization

    for ft_data in branches_data["first_tokens"]:
        token_id = ft_data["token_id"]
        # Get actual first token probability from vLLM (computed in Stage 3)
        token_probs[token_id] = ft_data.get("first_token_probability", 1e-10)
        # Store token text for visualization
        token_id_to_text[token_id] = ft_data.get("token_text", str(token_id))

        for cont in ft_data["continuations"]:
            # Compute weight based on weight_mode
            if weight_mode == "perplexity":
                # Per-token geometric mean probability = 1/PPL_per_token
                # This normalizes by continuation length
                logprob = cont["logprob"]
                num_tokens = cont.get("num_tokens", 1)
                if num_tokens == 0:
                    num_tokens = 1
                weight = np.exp(logprob / num_tokens)
            else:
                # Raw probability (original behavior)
                weight = cont["probability"]

            path_probs_original.append(weight)
            first_tokens.append(token_id)

    path_probs_original = np.array(path_probs_original)
    first_tokens = np.array(first_tokens)

    path_probs = path_probs_original

    # Log weight mode and statistics
    logger.info(f"Weight mode: {weight_mode}, include_first_token_prob: {include_first_token_prob} (IGNORED)")
    logger.info(f"Original weights: sum={path_probs_original.sum():.6f}, "
                f"min={path_probs_original.min():.2e}, max={path_probs_original.max():.2e}")
    logger.info(f"Using original weights for clustering (sum={path_probs.sum():.6f})")

    # Load embeddings
    embeddings_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    embeddings_e = np.load(embeddings_file)

    # Load attribution based on format
    if use_continuation_attribution:
        # New format: load from Stage 3
        logger.info("Loading continuation attribution from Stage 3...")
        prefix_context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
        context_data = torch.load(prefix_context_file, weights_only=False)

        # Load metadata from JSON
        attribution_meta_file = attribution_graphs_dir / f"{prefix_id}_attribution.json"
        attr_meta = load_json(attribution_meta_file)
        store_all = context_data.get("store_all", False)
        meta_span_mode = attr_meta.get("span_mode", "full")
        effective_span_mode = span_mode if span_mode is not None else meta_span_mode
        
        # Only log details if logger level is INFO or lower
        if logger.getEffectiveLevel() <= logging.INFO:
            logger.info(f"  Store all: {store_all}")
            logger.info(f"  Span mode (metadata): {meta_span_mode}")
            logger.info(f"  Span mode (effective): {effective_span_mode}")
            logger.info(f"  Pooling: {pooling}")
            logger.info(f"  N prefix sources: {attr_meta.get('n_prefix_sources', 'unknown')}")

        if store_all and "token_attributions" in context_data:
            # Process token-level attributions: slice + pool
            logger.info("Processing token-level attributions (slicing + pooling)...")
            token_attributions = context_data["token_attributions"]  # list of tensors

            # Determine span_info: recompute if span_mode differs and we have continuation tokens
            if span_mode is not None and span_mode != meta_span_mode:
                continuation_tokens = context_data.get("continuation_tokens", None)
                if continuation_tokens is not None:
                    logger.info(f"  Recomputing spans for span_mode='{effective_span_mode}'...")
                    span_info = compute_span_info(continuation_tokens, effective_span_mode)
                else:
                    logger.warning(f"Requested span_mode '{span_mode}' differs from Stage 3 '{meta_span_mode}', "
                                  f"but continuation_tokens not stored. Using Stage 3 spans.")
                    span_info = attr_meta.get("span_info", [])
            else:
                span_info = attr_meta.get("span_info", [])

            aggregated_list = []
            for i, token_attr in enumerate(token_attributions):
                # Get span for this continuation
                if i < len(span_info):
                    start = span_info[i]["start"]
                    end = span_info[i]["end"]
                else:
                    # Fallback to full span
                    start, end = 0, token_attr.shape[0]

                # Slice to span
                span_attr = token_attr[start:end]  # (span_len, n_prefix_sources)

                # Pool across positions
                if span_attr.shape[0] == 0:
                    # Empty span, use zeros
                    agg = torch.zeros(token_attr.shape[1])
                elif pooling == "mean":
                    agg = span_attr.mean(dim=0)
                elif pooling == "max":
                    agg = span_attr.max(dim=0).values
                elif pooling == "sum":
                    agg = span_attr.sum(dim=0)
                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")

                aggregated_list.append(agg)

            attributions_a = torch.stack(aggregated_list).float().numpy()
            logger.info(f"Processed {len(token_attributions)} continuations with {pooling} pooling")
        else:
            # Use pre-aggregated attributions (sum pooling from Stage 3)
            attributions_a = context_data["aggregated_attributions"].float().numpy()
            if span_mode is not None and span_mode != meta_span_mode:
                logger.warning(f"store_all=False: cannot change span_mode from '{meta_span_mode}' to '{span_mode}'. "
                              f"Re-run Stage 3 with --store-all to enable flexible span_mode.")
            if pooling != "sum":
                logger.warning(f"store_all=False: using pre-aggregated (sum) attributions. "
                              f"Requested pooling '{pooling}' not applied.")

        logger.info(f"Loaded continuation attributions: shape {attributions_a.shape}")

    else:
        # Legacy format: load from Graph.pt and extract per-first-token attribution
        logger.info("Loading attribution graph from Stage 2 (legacy format)...")
        graph_file = attribution_graphs_dir / f"{prefix_id}_graph.pt"
        graph = Graph.from_pt(graph_file)

        # Extract attribution embeddings
        attributions_list = []
        n_features = len(graph.selected_features)
        n_error = graph.cfg.n_layers * graph.n_pos
        n_token = graph.n_pos
        n_attribution_nodes = n_features + n_error + n_token

        for s_n in first_tokens:
            logit_tokens = graph.logit_tokens
            matches = (logit_tokens == s_n).nonzero(as_tuple=True)[0]

            if len(matches) == 0:
                logger.warning(f"Token {s_n} not found in attribution graph, using zero vector")
                attributions_list.append(np.zeros(n_attribution_nodes, dtype=np.float32))
                continue

            logit_idx = matches[0].item()
            logit_node_idx = n_features + n_error + n_token + logit_idx
            attribution_vector = graph.adjacency_matrix[logit_node_idx, :n_attribution_nodes]

            if isinstance(attribution_vector, torch.Tensor):
                attribution_vector = attribution_vector.cpu().float().numpy()

            attributions_list.append(attribution_vector.astype(np.float32))

        attributions_a = np.array(attributions_list)
        prefix_context_file = None

    logger.info(f"Attribution embeddings: shape {attributions_a.shape}")

    # Compute H_0 (global weighted mean) BEFORE clustering
    H_0 = compute_global_mean(attributions_a, path_probs)
    logger.info(f"Computed H_0: ||H_0|| = {np.linalg.norm(H_0):.4f}")

    # Center attributions for clustering (clustering operates on Delta_H)
    attributions_a_centered = attributions_a - H_0

    # Batch normalize attributions: a_i <- a_i / sqrt(mean(||a_i||^2))
    # This normalizes by the RMS of the L2 norms
    norms_sq = np.sum(attributions_a_centered ** 2, axis=1)  # ||a_i||^2
    rms_norm = np.sqrt(np.mean(norms_sq))  # sqrt(mean(||a_i||^2))
    if rms_norm > 1e-10:
        attributions_a_centered = attributions_a_centered / rms_norm
        logger.info(f"Batch normalized attributions: RMS norm = {rms_norm:.4f}")
    else:
        logger.warning("RMS norm near zero, skipping batch normalization")

    # Store original for reconstruction if needed
    attributions_a_original = attributions_a
    # Use centered and batch-normalized for clustering
    attributions_a = attributions_a_centered

    # Verify shapes
    n_samples = len(path_probs)
    assert embeddings_e.shape[0] == n_samples
    assert attributions_a.shape[0] == n_samples

    logger.info(f"Loaded {n_samples} samples")

    return {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "embeddings_e": embeddings_e,
        "attributions_a": attributions_a,  # Centered and batch-normalized for clustering
        "attributions_a_original": attributions_a_original,  # Original for reconstruction
        "H_0": H_0,  # Global mean (shared component)
        "attribution_rms_norm": rms_norm,  # RMS norm used for batch normalization
        "path_probs": path_probs,
        "first_tokens": first_tokens,
        "token_probs": token_probs,  # P(s) for each first token
        "token_id_to_text": token_id_to_text,  # token_id -> token_text for visualization
        "n_samples": n_samples,
        "prefix_context_file": prefix_context_file,  # Path to PrefixAttributionContext (for intervention)
    }


def run_clustering(
    data: dict,
    beta_e: float,
    beta_a: float,
    K_max: int,
    max_iterations: int,
    convergence_threshold: float,
    logger,
):
    """Run the full R-D clustering algorithm for one prefix.

    Implements Algorithm 1 from rate_distortion.tex.

    Args:
        data: Data dict with embeddings_e, attributions_a, path_probs, etc.
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight
        K_max: Maximum number of components
        max_iterations: Maximum EM iterations
        convergence_threshold: Convergence threshold for L_RD change
        logger: Logger instance

    Returns:
        Clustering result dict
    """
    prefix_id = data["prefix_id"]
    embeddings_e = data["embeddings_e"].copy()
    attributions_a = data["attributions_a"].copy()
    path_probs = data["path_probs"].copy()
    first_tokens = data["first_tokens"].copy()

    logger.info("=" * 60)
    logger.info(f"R-D CLUSTERING PREFIX: {prefix_id}")
    logger.info("=" * 60)
    logger.info(f"Parameters: β_e={beta_e}, β_a={beta_a}, K_max={K_max}")
    logger.info(f"N samples: {len(path_probs)}")

    # Initialization
    logger.info("Initializing with single component (K=1)...")
    components, assignments = initialize_single_component(
        embeddings_e, attributions_a, path_probs
    )

    # Initialize tracking
    next_component_id = max(components.keys()) + 1 if components else 2
    L_RD_prev = np.inf

    history = {
        "iterations": [],
        "n_components": [],
        "L_RD": [],
        "H": [],
        "D_e": [],
        "D_a": [],
    }

    # Compute initial statistics
    rd_stats = compute_full_rd_statistics(
        embeddings_e, attributions_a, assignments, path_probs,
        components, beta_e, beta_a
    )

    logger.info(f"Initial: {len(components)} component(s), L_RD={rd_stats['L_RD']:.4f}")

    # Step 2: Main loop
    logger.info("\nStarting R-D EM loop...")
    converged = False
    iteration = 0

    # Determine if we should show progress bar (quiet mode check via logger level)
    show_pbar = False
    if logger and logger.getEffectiveLevel() >= 30: # logging.WARNING
        show_pbar = True
        
    iterator = range(max_iterations)
    if show_pbar:
        iterator = tqdm(iterator, desc="  EM Iterations", leave=False)

    for iteration in iterator:
        logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # E-step + M-step
        assignments, components, rd_stats = run_em_iteration(
            embeddings_e,
            attributions_a,
            path_probs,
            components,
            beta_e,
            beta_a
        )

        L_RD_curr = rd_stats['L_RD']

        logger.info(f"After EM: K={len(components)}, L_RD={L_RD_curr:.4f}, "
                   f"H={rd_stats['H']:.4f}, D_e={rd_stats['D_e']:.4f}, D_a={rd_stats['D_a']:.4f}")

        # Diagnostic logging: per-component variance
        # Var_e[c], Var_a[c] are weighted total squared distances
        P_bar_diag = rd_stats['P_bar']
        Var_e_diag = rd_stats.get('Var_e', {})
        Var_a_diag = rd_stats.get('Var_a', {})
        for c, comp in components.items():
            indices = [i for i, a in enumerate(assignments) if a == c]
            p_c = P_bar_diag.get(c, 0)
            v_e = Var_e_diag.get(c, 0)
            v_a = Var_a_diag.get(c, 0)
            # Show: n, P̄_c, Var_e, Var_a, and contribution to D_e/D_a
            logger.info(f"  C{c}: n={len(indices)}, P̄={p_c:.4f}, Var_e={v_e:.4f}, Var_a={v_a:.4f}, "
                       f"contrib_e={p_c*v_e:.4f}, contrib_a={p_c*v_a:.4f}")

        # Adaptive control (Split → Junk)
        P_bar = rd_stats['P_bar']
        Var_e = rd_stats.get('Var_e', {})
        Var_a = rd_stats.get('Var_a', {})

        # Compute variances if not in rd_stats
        if not Var_e or not Var_a:
            W_c, _ = compute_component_masses(assignments, path_probs, list(components.keys()))
            for c, comp in components.items():
                indices = [i for i, a in enumerate(assignments) if a == c]
                W_c_val = W_c.get(c, 0)
                if not indices or W_c_val == 0:
                    Var_e[c] = 0.0
                    Var_a[c] = 0.0
                    continue
                diff_e = embeddings_e[indices] - comp['mu_e'][None, :]
                Var_e[c] = float(np.sum(path_probs[indices] * np.sum(diff_e ** 2, axis=1)) / W_c_val)
                diff_a = attributions_a[indices] - comp['mu_a'][None, :]
                Var_a[c] = float(np.sum(path_probs[indices] * np.sum(diff_a ** 2, axis=1)) / W_c_val)

        components, assignments, next_component_id = apply_adaptive_control(
            embeddings_e,
            attributions_a,
            path_probs,
            assignments,
            components,
            P_bar,
            Var_e,
            Var_a,
            beta_e,
            beta_a,
            K_max,
            next_component_id
        )

        logger.info(f"After adaptive: K={len(components)}")

        # Recompute R-D stats after adaptive control
        if len(components) > 0:
            rd_stats = compute_full_rd_statistics(
                embeddings_e, attributions_a, assignments, path_probs,
                components, beta_e, beta_a
            )
            L_RD_curr = rd_stats['L_RD']

        # Track history
        history["iterations"].append(iteration + 1)
        history["n_components"].append(len(components))
        history["L_RD"].append(L_RD_curr)
        history["H"].append(rd_stats['H'])
        history["D_e"].append(rd_stats['D_e'])
        history["D_a"].append(rd_stats['D_a'])

        # Check convergence
        converged = check_convergence(L_RD_prev, L_RD_curr, convergence_threshold)

        if converged:
            logger.info(f"\nConverged after {iteration + 1} iterations")
            break

        L_RD_prev = L_RD_curr

    # Final statistics
    logger.info("\n" + "=" * 60)
    logger.info("R-D CLUSTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final components: {len(components)}")
    logger.info(f"Final L_RD: {rd_stats['L_RD']:.4f}")
    logger.info(f"Iterations: {iteration + 1}")

    return {
        "prefix_id": prefix_id,
        "prefix": data["prefix"],
        "components": components,
        "rd_stats": rd_stats,
        "assignments": assignments,
        "history": history,
        "n_iterations": iteration + 1,
        "converged": converged,
        # Hierarchical decomposition
        "H_0": data.get("H_0"),  # Global mean (shared attribution)
    }


def serialize_clustering_result(result: dict) -> dict:
    """Convert clustering result to JSON-serializable format.

    Args:
        result: Raw clustering result dict

    Returns:
        JSON-serializable dict with all component info
    """
    # Get H_0 for computing full mu_a
    H_0 = result.get("H_0")

    def get_mu_a_full(comp):
        """Compute full mu_a = H_0 + Delta_H_c."""
        mu_a = comp["mu_a"]
        if hasattr(mu_a, "tolist"):
            mu_a = mu_a
        else:
            mu_a = np.array(mu_a)
        if H_0 is not None:
            return (mu_a + H_0).tolist() if hasattr(mu_a + H_0, "tolist") else (mu_a + H_0).tolist()
        return mu_a.tolist() if hasattr(mu_a, "tolist") else mu_a

    return {
        "prefix_id": result["prefix_id"],
        "prefix": result["prefix"],
        "method": "rate_distortion",
        "n_iterations": int(result["n_iterations"]),
        "converged": bool(result["converged"]),
        "n_components": len(result["components"]),
        # Hierarchical decomposition (shared components)
        "H_0": H_0.tolist() if H_0 is not None and hasattr(H_0, "tolist") else H_0,
        "rd_objective": {
            "L_RD": float(result["rd_stats"]["L_RD"]),
            "H": float(result["rd_stats"]["H"]),
            "D_e": float(result["rd_stats"]["D_e"]),
            "D_a": float(result["rd_stats"]["D_a"]),
            "beta_e": float(result["rd_stats"]["beta_e"]),
            "beta_a": float(result["rd_stats"]["beta_a"]),
        },
        "components": {
            str(c): {
                "mu_e": comp["mu_e"].tolist() if hasattr(comp["mu_e"], "tolist") else comp["mu_e"],
                "mu_a": comp["mu_a"].tolist() if hasattr(comp["mu_a"], "tolist") else comp["mu_a"],  # Delta_H_c (centered)
                "mu_a_full": get_mu_a_full(comp),  # H_0 + Delta_H_c
                "W_c": float(comp["W_c"]),
                "indices": [int(i) for i in comp.get("indices", [])],
            }
            for c, comp in result["components"].items()
        },
        "assignments": [int(a) for a in result["assignments"]],
        "statistics": {
            str(c): {
                "Var_e_w": float(result["rd_stats"]["Var_e"].get(c, 0.0)),
                "Var_a_w": float(result["rd_stats"]["Var_a"].get(c, 0.0)),
                "Err_c_w": 0.0,
                "n_samples": len(comp.get("indices", [])),
            }
            for c, comp in result["components"].items()
        },
    }


def process_prefix(meta_file, args, sweeps_config, n_sweep_workers, sweep_mode):
    """Process a single prefix (load data, run clustering/sweep, save results).

    Args:
        meta_file: Path to embeddings metadata file
        args: Argument namespace
        sweeps_config: Sweep configuration dict
        n_sweep_workers: Number of workers for sweep mode
        sweep_mode: Boolean indicating if sweep mode is enabled

    Returns:
        Dict with status and result summary
    """
    prefix_id = meta_file.stem.replace("_embeddings_meta", "")
    
    # Create worker-specific logger
    log_file = args.output_dir / "logs" / f"{prefix_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"clustering_{prefix_id}", log_file=log_file, level=logging.INFO)
    
    try:
        data = load_prefix_data(
            prefix_id,
            args.embeddings_dir,
            args.attribution_graphs_dir,
            args.samples_dir,
            logger,
            weight_mode=args.weight_mode,
            include_first_token_prob=args.include_first_token_prob,
            use_continuation_attribution=args.use_continuation_attribution,
            pooling=args.pooling,
            span_mode=args.span_mode,
        )

        result = {}
        if sweep_mode:
            # Sweep mode: run sweep over (beta, gamma) grid
            sweep_results = run_sweep_mode(
                data,
                sweeps_config,
                args.K_max,
                args.max_iterations,
                args.convergence_threshold,
                logger,
                n_workers=n_sweep_workers,
            )

            # Save sweep results
            sweep_file = args.output_dir / f"{prefix_id}_sweep_results.json"
            save_json(sweep_results, sweep_file)
            logger.info(f"Saved sweep results to: {sweep_file}")

            # Run clustering with best config for full result
            best = sweep_results.get("best", {})
            best_beta = best.get("beta", args.beta)
            best_gamma = best.get("gamma", args.gamma)
            best_beta_e = best_gamma * best_beta
            best_beta_a = (1 - best_gamma) * best_beta

            result = run_clustering(
                data,
                best_beta_e,
                best_beta_a,
                args.K_max,
                args.max_iterations,
                args.convergence_threshold,
                logger,
            )

            # Add sweep info to result
            result["sweep_best_config"] = {
                "beta": best_beta,
                "gamma": best_gamma,
                "selection_method": sweeps_config.get("selection_method", "harmonic"),
            }

        else:
            # Single mode: run with fixed beta, gamma
            result = run_clustering(
                data,
                args.beta_e,
                args.beta_a,
                args.K_max,
                args.max_iterations,
                args.convergence_threshold,
                logger,
            )

        # Save individual result
        output_file = args.output_dir / f"{prefix_id}_clustering.json"

        result_serializable = serialize_clustering_result(result)
        result_serializable["history"] = {
            k: [float(v) if isinstance(v, (int, float, np.number)) else v for v in vals]
            for k, vals in result["history"].items()
        }

        # Add sweep info if available
        if "sweep_best_config" in result:
            result_serializable["sweep_best_config"] = result["sweep_best_config"]

        save_json(result_serializable, output_file)
        logger.info(f"Saved result to: {output_file}\n")
        
        # Close handlers to avoid leaking
        for handler in logger.handlers:
            handler.close()
            
        return {
            "status": "success",
            "prefix_id": prefix_id,
            "result": result,
            "sweep_best_config": result.get("sweep_best_config", {})
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error processing {prefix_id}: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Close handlers
        for handler in logger.handlers:
            handler.close()
            
        return {
            "status": "failed",
            "prefix_id": prefix_id,
            "error": error_msg
        }


def main():
    parser = argparse.ArgumentParser(description="Run Rate-Distortion Gaussian clustering")
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)

    # R-D specific parameters
    parser.add_argument("--beta", type=float, default=2.0, help="Total β")
    parser.add_argument("--gamma", type=float, default=0.5, help="View ratio: β_e = γβ, β_a = (1-γ)β")
    parser.add_argument("--K-max", type=int, default=20, help="Maximum number of components")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--convergence-threshold", type=float, default=1e-6)
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for log files")
    parser.add_argument("--weight-mode", type=str, default="probability",
                        choices=["probability", "perplexity"],
                        help="Weight mode: 'probability' (raw P) or 'perplexity' (per-token, length-normalized)")
    parser.add_argument("--include-first-token-prob", type=lambda x: x.lower() == 'true',
                        default=True,
                        help="Include P(first_token) in continuation probability (default: True)")
    parser.add_argument("--use-continuation-attribution", type=lambda x: x.lower() == 'true',
                        default=True,
                        help="Use continuation attribution from Stage 3 (default: True). "
                             "If False, use legacy Graph.pt format from Stage 2.")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "sum"],
                        help="Pooling method for aggregating token attributions (default: mean)")
    parser.add_argument("--span-mode", type=str, default=None,
                        choices=["full", "lcs_plus_one", "post_lcs"],
                        help="Span mode for slicing token attributions (only used if store_all=True). "
                             "If None, uses span_mode from Stage 3 metadata.")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of workers for parallel prefix processing")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (only progress bars)")

    args = parser.parse_args()

    # Load config if provided
    sweep_mode = False
    sweeps_config = {}
    n_sweep_workers = 4

    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)

        # Clustering config
        clustering = config.get("clustering", {})
        args.beta = clustering.get("beta", args.beta)
        args.gamma = clustering.get("gamma", args.gamma)
        args.K_max = clustering.get("K_max", args.K_max)
        args.max_iterations = clustering.get("max_iterations", args.max_iterations)
        args.convergence_threshold = clustering.get("convergence_threshold", args.convergence_threshold)
        args.weight_mode = clustering.get("weight_mode", args.weight_mode)
        args.include_first_token_prob = clustering.get("include_first_token_prob", args.include_first_token_prob)
        args.use_continuation_attribution = clustering.get("use_continuation_attribution", args.use_continuation_attribution)
        args.pooling = clustering.get("pooling", args.pooling)
        if args.span_mode is None:
            args.span_mode = clustering.get("span_mode", None)
        
        # Worker config from config file (if provided)
        # Note: --n-workers cli arg overrides this for prefix parallelism
        
        # Detect sweep mode (like 7C pattern)
        sweeps = clustering.get("sweeps", {})
        if sweeps and sweeps.get("beta_values") and sweeps.get("gamma_values"):
            sweep_mode = True
            sweeps_config = sweeps
            n_sweep_workers = sweeps.get("n_workers", 4)

    # Compute beta_e, beta_a from beta and gamma (for single mode)
    args.beta_e = args.gamma * args.beta
    args.beta_a = (1 - args.gamma) * args.beta

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("test_results")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logger
    log_file = get_log_path("5_clustering", args.log_dir)
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("rd_clustering", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("RATE-DISTORTION GAUSSIAN CLUSTERING")
    logger.info("=" * 60)
    
    # Configure parallelism strategy
    # If parallelizing prefixes (n_workers > 1), force sequential sweeps (n_sweep_workers = 1)
    # If serial prefixes (n_workers = 1), allow parallel sweeps (n_sweep_workers from config)
    if args.n_workers > 1:
        logger.info(f"PARALLEL STRATEGY: Parallel Prefixes ({args.n_workers} workers) -> Serial Sweeps")
        n_sweep_workers = 1
    else:
        logger.info(f"PARALLEL STRATEGY: Serial Prefixes -> Parallel Sweeps ({n_sweep_workers} workers)")
    
    if sweep_mode:
        logger.info("MODE: SWEEP (hyperparameter search)")
        logger.info(f"Beta values: {sweeps_config.get('beta_values')}")
        logger.info(f"Gamma values: {sweeps_config.get('gamma_values')}")
        logger.info(f"Selection method: {sweeps_config.get('selection_method', 'harmonic')}")
    else:
        logger.info("MODE: SINGLE")
        logger.info(f"γ (view ratio): {args.gamma} → β_e = {args.gamma}β, β_a = {1-args.gamma:.2f}β")
        logger.info(f"β = {args.beta} → β_e = {args.beta_e:.4f}, β_a = {args.beta_a:.4f}")
    logger.info(f"K_max: {args.K_max}")
    logger.info(f"Weight mode: {args.weight_mode}")
    logger.info(f"GPU acceleration: {'enabled' if GPU_AVAILABLE else 'disabled'}")

    # Find all embedding metadata files
    embedding_meta_files = sorted(args.embeddings_dir.glob("*_embeddings_meta.json"))
    logger.info(f"\nFound {len(embedding_meta_files)} prefixes to process")

    # Filter based on Stage 4a manifest (embeddings extraction)
    paths = PathConfig()
    results_dir = paths.results
    all_prefix_ids = [f.stem.replace("_embeddings_meta", "") for f in embedding_meta_files]
    available_ids, skipped_ids = filter_samples_by_manifest(
        all_prefix_ids, results_dir, "stage4a", logger
    )
    # Filter embedding files to only available ones
    available_id_set = set(available_ids)
    embedding_meta_files = [f for f in embedding_meta_files if f.stem.replace("_embeddings_meta", "") in available_id_set]
    logger.info(f"Processing {len(embedding_meta_files)} available prefixes (skipped {len(skipped_ids)})")

    # Process prefixes
    results = []
    completed_ids = []
    failed_ids = []
    errors = {}
    
    if args.n_workers > 1:
        # Parallel execution
        logger.info(f"Starting parallel processing with {args.n_workers} workers...")
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(
                    process_prefix, 
                    meta_file, 
                    args, 
                    sweeps_config, 
                    n_sweep_workers, 
                    sweep_mode
                ): meta_file 
                for meta_file in embedding_meta_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Clustering prefixes"):
                res = future.result()
                if res["status"] == "success":
                    completed_ids.append(res["prefix_id"])
                    results.append(res["result"])
                else:
                    failed_ids.append(res["prefix_id"])
                    errors[res["prefix_id"]] = res.get("error", "Unknown error")
    else:
        # Sequential execution
        logger.info("Starting sequential processing...")
        for meta_file in tqdm(embedding_meta_files, desc="Clustering prefixes"):
            res = process_prefix(meta_file, args, sweeps_config, n_sweep_workers, sweep_mode)
            if res["status"] == "success":
                completed_ids.append(res["prefix_id"])
                results.append(res["result"])
            else:
                failed_ids.append(res["prefix_id"])
                errors[res["prefix_id"]] = res.get("error", "Unknown error")

    # Save summary
    summary = {
        "method": "rate_distortion",
        "mode": "sweep" if sweep_mode else "single",
        "n_prefixes": len(results),
        "parameters": {
            "K_max": args.K_max,
            "gamma": args.gamma,
            "beta": args.beta,
            "beta_e": args.beta_e,
            "beta_a": args.beta_a,
            "weight_mode": args.weight_mode,
        },
        "mean_n_components": float(np.mean([len(r["components"]) for r in results])) if results else 0.0,
        "mean_n_iterations": float(np.mean([r["n_iterations"] for r in results])) if results else 0.0,
        "mean_L_RD": float(np.mean([r["rd_stats"]["L_RD"] for r in results])) if results else 0.0,
        "results": [
            {
                "prefix_id": r["prefix_id"],
                "n_components": len(r["components"]),
                "n_iterations": int(r["n_iterations"]),
                "converged": bool(r["converged"]),
                "L_RD": float(r["rd_stats"]["L_RD"]),
                **({"sweep_best_config": r["sweep_best_config"]} if "sweep_best_config" in r else {}),
            }
            for r in results
        ]
    }

    # Add sweep config to summary if in sweep mode
    if sweep_mode:
        summary["sweep_config"] = sweeps_config

    summary_file = args.output_dir / "clustering_summary.json"
    save_json(summary, summary_file)

    # Write Stage 5 manifest
    update_manifest_with_results(
        results_dir=results_dir,
        stage_name="stage5",
        processed=completed_ids,
        failed=failed_ids,
        skipped=skipped_ids,
        logger=logger,
        errors=errors,
    )

    logger.info("=" * 60)
    logger.info("ALL PREFIXES COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed: {len(completed_ids)} prefixes")
    logger.info(f"Completed: {len(completed_ids)}, Failed: {len(failed_ids)}, Skipped: {len(skipped_ids)}")
    logger.info(f"Mean components: {summary['mean_n_components']:.2f}")
    logger.info(f"Mean L_RD: {summary['mean_L_RD']:.4f}")


if __name__ == "__main__":
    main()
