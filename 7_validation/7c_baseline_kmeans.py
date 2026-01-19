#!/usr/bin/env -S uv run python
"""7c_baseline_kmeans.py - K-means Clustering Baseline for Steering Validation.

This script implements a baseline for 7c steering that uses K-means clustering
on semantic embeddings only (vs. RD-clustering that uses both embeddings and attributions).

The baseline:
1. Loads embeddings from Stage 4a
2. For each (beta, gamma) config, reads K from RD sweep results
3. Runs K-means on embeddings to get cluster assignments
4. Computes weighted median of attributions for each cluster to get H_c
5. Reuses existing 7c steering infrastructure for evaluation

Usage:
    python 7c_baseline_kmeans.py --samples-dir ... --clustering-dir ... [options]
"""

import sys
import time
import gc
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from itertools import product

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger, get_log_path
from utils.memory_utils import clear_memory
from circuit_tracer import ReplacementModel

# Import from refactored modules
import importlib.util
from pathlib import Path as _Path

def _import_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_module_dir = _Path(__file__).parent
graph = _import_module("7c_graph", _module_dir / "7c_graph.py")
steering = _import_module("7c_steering", _module_dir / "7c_steering.py")
metrics = _import_module("7c_metrics", _module_dir / "7c_metrics.py")
utils = _import_module("7c_utils", _module_dir / "7c_utils.py")
hypotheses = _import_module("7c_hypotheses", _module_dir / "7c_hypotheses.py")


# =============================================================================
# K-means Clustering Functions
# =============================================================================

def run_kmeans_clustering(
    embeddings: np.ndarray,
    K: int,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Run K-means clustering on semantic embeddings.
    
    Args:
        embeddings: Semantic embeddings [n_samples, d_embedding]
        K: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (assignments, cluster_centers):
        - assignments: [n_samples] cluster assignment for each sample
        - cluster_centers: [K, d_embedding] cluster centers
    """
    kmeans = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    assignments = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    return assignments, centers


def compute_weighted_median_attribution(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int
) -> np.ndarray:
    """Compute probability-weighted median of attributions for a cluster.
    
    Uses weighted median along each dimension independently.
    
    Args:
        attributions: Attribution vectors [n_samples, d_attribution]
        assignments: Cluster assignments [n_samples]
        path_probs: Path probabilities [n_samples]
        cluster_id: Cluster ID to compute median for
        
    Returns:
        Weighted median attribution vector [d_attribution]
    """
    mask = assignments == cluster_id
    if not np.any(mask):
        return np.zeros(attributions.shape[1])
    
    cluster_attributions = attributions[mask]  # [n_cluster, d_attribution]
    cluster_weights = path_probs[mask]  # [n_cluster]
    
    # Normalize weights
    total_weight = cluster_weights.sum()
    if total_weight <= 0:
        return np.mean(cluster_attributions, axis=0)
    
    normalized_weights = cluster_weights / total_weight
    
    # Weighted median per dimension
    d_attribution = attributions.shape[1]
    result = np.zeros(d_attribution)
    
    for dim in range(d_attribution):
        dim_values = cluster_attributions[:, dim]
        
        # Sort by value
        sort_idx = np.argsort(dim_values)
        sorted_values = dim_values[sort_idx]
        sorted_weights = normalized_weights[sort_idx]
        
        # Find weighted median (cumulative weight >= 0.5)
        cumulative = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumulative, 0.5)
        
        # Clamp to valid range
        median_idx = min(median_idx, len(sorted_values) - 1)
        result[dim] = sorted_values[median_idx]
    
    return result


def compute_weighted_mean_attribution(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int
) -> np.ndarray:
    """Compute probability-weighted mean of attributions for a cluster.
    
    Fallback alternative to weighted median.
    
    Args:
        attributions: Attribution vectors [n_samples, d_attribution]
        assignments: Cluster assignments [n_samples]
        path_probs: Path probabilities [n_samples]
        cluster_id: Cluster ID to compute mean for
        
    Returns:
        Weighted mean attribution vector [d_attribution]
    """
    mask = assignments == cluster_id
    if not np.any(mask):
        return np.zeros(attributions.shape[1])
    
    cluster_attributions = attributions[mask]
    cluster_weights = path_probs[mask]
    
    total_weight = cluster_weights.sum()
    if total_weight <= 0:
        return np.mean(cluster_attributions, axis=0)
    
    # Weighted mean
    weighted_sum = np.sum(cluster_weights[:, None] * cluster_attributions, axis=0)
    return weighted_sum / total_weight


def load_rd_sweep_K_values(clustering_file: Path) -> Dict[str, int]:
    """Extract K values from RD sweep results for each (beta, gamma) config.
    
    Args:
        clustering_file: Path to {prefix}_sweep_results.json
        
    Returns:
        Dict mapping "beta{b}_gamma{g}" -> K
    """
    data = load_json(clustering_file)
    grid = data.get("grid", [])
    
    K_map = {}
    for entry in grid:
        beta = entry.get("beta")
        gamma = entry.get("gamma")
        K = entry.get("K", len(entry.get("components", {})))
        
        if beta is not None and gamma is not None:
            key = f"beta{beta}_gamma{gamma}"
            K_map[key] = K
    
    return K_map


def load_prefix_data_for_baseline(
    prefix_id: str,
    embeddings_dir: Path,
    attribution_graphs_dir: Path,
    samples_dir: Path,
    logger,
    pooling: str = "mean"
) -> Dict[str, Any]:
    """Load all data needed for K-means baseline.
    
    Similar to cluster.load_prefix_data but tailored for baseline use.
    """
    logger.info(f"Loading data for prefix: {prefix_id}")
    
    # Load branch samples data
    branches_file = samples_dir / f"{prefix_id}_branches.json"
    branches_data = load_json(branches_file)
    prefix = branches_data.get("prefix", "")
    
    # Extract path probabilities
    path_probs = []
    for cont in branches_data.get("continuations", []):
        path_probs.append(cont.get("probability", 0.0))
    path_probs = np.array(path_probs)
    
    # Load embeddings
    embeddings_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    embeddings = np.load(embeddings_file)
    
    # Load attribution context
    prefix_context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
    context_data = torch.load(prefix_context_file, weights_only=False)
    
    # Load metadata
    attribution_meta_file = attribution_graphs_dir / f"{prefix_id}_attribution.json"
    attr_meta = load_json(attribution_meta_file)
    store_all = context_data.get("store_all", False)
    
    # Process attributions
    if store_all and "token_attributions" in context_data:
        token_attributions = context_data["token_attributions"]
        span_info = attr_meta.get("span_info", [])
        
        aggregated_list = []
        for i, token_attr in enumerate(token_attributions):
            if i < len(span_info):
                start = span_info[i]["start"]
                end = span_info[i]["end"]
            else:
                start, end = 0, token_attr.shape[0]
            
            span_attr = token_attr[start:end]
            
            if span_attr.shape[0] == 0:
                agg = torch.zeros(token_attr.shape[1])
            elif pooling == "mean":
                agg = span_attr.mean(dim=0)
            elif pooling == "max":
                agg = span_attr.max(dim=0).values
            elif pooling == "sum":
                agg = span_attr.sum(dim=0)
            else:
                agg = span_attr.mean(dim=0)
            
            aggregated_list.append(agg)
        
        attributions = torch.stack(aggregated_list).float().numpy()
    else:
        attributions = context_data["aggregated_attributions"].float().numpy()
    
    # Compute H_0 (global weighted mean)
    W_total = path_probs.sum()
    if W_total > 0:
        H_0 = np.sum(path_probs[:, None] * attributions, axis=0) / W_total
    else:
        H_0 = np.zeros(attributions.shape[1])
    
    # Center attributions (clustering operates on Delta_H)
    attributions_centered = attributions - H_0
    
    # Batch normalize
    norms_sq = np.sum(attributions_centered ** 2, axis=1)
    rms_norm = np.sqrt(np.mean(norms_sq))
    if rms_norm > 1e-10:
        attributions_centered = attributions_centered / rms_norm
    
    n_samples = len(path_probs)
    logger.info(f"Loaded {n_samples} samples, emb shape: {embeddings.shape}, attr shape: {attributions.shape}")
    
    return {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "embeddings": embeddings,
        "attributions": attributions,  # Original (uncentered) for H_c computation
        "attributions_centered": attributions_centered,
        "H_0": H_0,
        "rms_norm": rms_norm,
        "path_probs": path_probs,
        "branches_data": branches_data,
        "n_samples": n_samples,
    }


def build_semantic_graphs_from_kmeans(
    assignments: np.ndarray,
    attributions: np.ndarray,
    path_probs: np.ndarray,
    H_0: np.ndarray,
    use_median: bool = True
) -> Dict[int, np.ndarray]:
    """Build semantic graphs (H_c) from K-means assignments.
    
    For each cluster, computes H_c = H_0 + weighted_center(attributions - H_0)
    
    Args:
        assignments: Cluster assignments [n_samples]
        attributions: Original (uncentered) attributions [n_samples, d]
        path_probs: Path probabilities [n_samples]
        H_0: Global mean attribution [d]
        use_median: If True, use weighted median; else weighted mean
        
    Returns:
        {cluster_id: H_c array}
    """
    # Convert to Python int to ensure JSON serializability
    unique_clusters = sorted([int(c) for c in set(assignments)])
    
    # Center attributions for computing Delta_H_c
    attributions_centered = attributions - H_0
    
    semantic_graphs = {}
    for c in unique_clusters:
        if use_median:
            Delta_H_c = compute_weighted_median_attribution(
                attributions_centered, assignments, path_probs, c
            )
        else:
            Delta_H_c = compute_weighted_mean_attribution(
                attributions_centered, assignments, path_probs, c
            )
        
        # H_c = H_0 + Delta_H_c
        H_c = H_0 + Delta_H_c
        semantic_graphs[c] = H_c
    
    return semantic_graphs


# =============================================================================
# Main Sweep Functions (reusing 7c infrastructure)
# =============================================================================

# run_steering_sweep is imported from 7c_hypotheses.py via the `hypotheses` module
# This avoids code duplication - the function handles both cross_prefix_batching
# and normal batching modes with all metric computation


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="K-means baseline for 7c steering validation")
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Directory with branch samples (2_branch_sampling)")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Directory with embeddings (4_feature_extraction/embeddings)")
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True,
                        help="Directory with attribution context (3_attribution_graphs)")
    parser.add_argument("--clustering-dir", type=Path, required=True,
                        help="Directory with RD clustering results (5_clustering)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for baseline results")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config JSON file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name or path")
    parser.add_argument("--transcoder", type=str, default="mwhanna/qwen3-8b-transcoders",
                        help="Transcoder name or path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of prefixes to process")
    parser.add_argument("--max-cluster-samples", type=int, default=20,
                        help="Maximum samples per cluster")
    parser.add_argument("--max-batch-size", type=int, default=512,
                        help="Maximum batch size for steering")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "sum"],
                        help="Pooling method for attributions")
    parser.add_argument("--use-mean", action="store_true",
                        help="Use weighted mean instead of median for H_c")
    parser.add_argument("--prefix-id", type=str, default=None,
                        help="Process only a specific prefix")
    parser.add_argument("--cross-prefix-batching", action="store_true",
                        help="Enable cross-prefix batching")
    parser.add_argument("--prefix-batch-size", type=int, default=16,
                        help="Number of prefixes to batch together")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    # Setup logging
    log_file = get_log_path("7c_baseline_kmeans", args.log_dir)
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("kmeans_baseline", log_file=log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("K-MEANS BASELINE FOR 7C STEERING")
    logger.info("=" * 60)
    
    # Load config
    config = {}
    sweeps = []
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
        
        # Parse steering sweep config
        steering_config = config.get("stage_7c_steering", {})
        sweeps = steering_config.get("sweeps", [])
        
        # Normalize sweeps
        defaults = {
            "h_c_selections": ["full"],
            "top_B": [10],
            "epsilon_values": [-1.0, 0.0, 1.0]
        }
        sweeps = [utils.validate_and_normalize_sweep_config(sw, defaults) for sw in sweeps]
        
        # Update args from config
        if args.max_cluster_samples is None:
            args.max_cluster_samples = steering_config.get("max_cluster_samples", 20)
        if args.max_batch_size is None:
            args.max_batch_size = steering_config.get("max_batch_size", 512)
        if not args.cross_prefix_batching:
            args.cross_prefix_batching = steering_config.get("cross_prefix_batching", False)
        if args.prefix_batch_size is None or args.prefix_batch_size <= 0:
            args.prefix_batch_size = steering_config.get("prefix_batch_size", 16)
    
    if not sweeps:
        # Default sweep
        sweeps = [{
            "name": "default",
            "steering_method": "sign",
            "h_c_selections": ["full", "positive", "negative"],
            "top_B": [5, 10],
            "epsilon_values": [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]
        }]
    
    logger.info(f"Sweep configurations: {len(sweeps)}")
    for i, sw in enumerate(sweeps):
        logger.info(f"  [{i+1}] {sw.get('name', 'unnamed')}: {sw.get('steering_method')} | "
                    f"hc={sw.get('h_c_selections')} | B={sw.get('top_B')} | "
                    f"eps={sw.get('epsilon_values')}")
    
    # Find prefixes
    embedding_files = sorted(args.embeddings_dir.glob("*_embeddings.npy"))
    prefix_ids = [f.stem.replace("_embeddings", "") for f in embedding_files]
    
    if args.prefix_id:
        prefix_ids = [args.prefix_id]
    
    if args.max_samples and len(prefix_ids) > args.max_samples:
        prefix_ids = prefix_ids[:args.max_samples]
    
    logger.info(f"Processing {len(prefix_ids)} prefixes")
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h4a_output_dir = args.output_dir / "H4a"
    h4a_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    global_config = config.get("global", {})
    max_seq_len = global_config.get("max_seq_len", 64)
    
    model_config = config.get("model", {})
    model_name = model_config.get("base_model", args.model)
    transcoder_name = model_config.get("transcoder", args.transcoder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReplacementModel.from_pretrained(
        model_name, transcoder_name, device=device, dtype=torch.bfloat16,
        lazy_encoder=True, lazy_decoder=False
    )
    device = model.cfg.device
    
    # Build max_top_B from sweeps
    max_top_B = max(max(sw.get("top_B", [10])) for sw in sweeps)
    
    # Split prefixes into batches
    prefix_batch_size = args.prefix_batch_size if args.cross_prefix_batching else 1
    prefix_batches = [prefix_ids[i:i + prefix_batch_size] for i in range(0, len(prefix_ids), prefix_batch_size)]
    logger.info(f"Prefix batching: {prefix_batch_size} prefixes per batch, {len(prefix_batches)} batches")
    
    # Process prefix batches
    for batch_idx, batch_prefix_ids in enumerate(tqdm(prefix_batches, desc="Processing batches")):
        logger.info(f"\n{'='*40}")
        logger.info(f"Batch {batch_idx + 1}/{len(prefix_batches)}: {len(batch_prefix_ids)} prefixes")
        logger.info(f"{'='*40}")
        
        # Load data for all prefixes in batch
        batch_prefix_state = {}  # {prefix_id: {...data...}}
        batch_contexts_by_key = {}  # {clustering_key: [ctx, ctx, ...]}
        
        for prefix_id in batch_prefix_ids:
            # Check required files
            clustering_file = args.clustering_dir / f"{prefix_id}_sweep_results.json"
            if not clustering_file.exists():
                logger.warning(f"Missing clustering file for {prefix_id}")
                continue
            
            # Load data
            try:
                prefix_data = load_prefix_data_for_baseline(
                    prefix_id,
                    args.embeddings_dir,
                    args.attribution_graphs_dir,
                    args.samples_dir,
                    logger,
                    pooling=args.pooling
                )
            except Exception as e:
                logger.error(f"Error loading data for {prefix_id}: {e}")
                continue
            
            # Load attribution context
            active_features, selected_features = graph.load_attribution_context(
                args.attribution_graphs_dir, prefix_id, use_continuation_attribution=True
            )
            n_features = len(selected_features)
            
            # Load RD sweep K values
            K_map = load_rd_sweep_K_values(clustering_file)
            rd_sweep_data = load_json(clustering_file)
            H_0_rd = rd_sweep_data.get("H_0")
            H_0 = np.array(H_0_rd) if H_0_rd is not None else prefix_data["H_0"]
            
            # Compute baseline log_P
            logger.info(f"  {prefix_id}: Computing baseline log_P...")
            first_K = list(K_map.values())[0] if K_map else 5
            temp_assignments = run_kmeans_clustering(prefix_data["embeddings"], first_K)[0]
            baseline_branches = utils.build_branches_from_data(
                prefix_data["branches_data"], temp_assignments.tolist()
            )
            
            baseline_branch_log_probs = steering.compute_branch_log_probs_batch(
                model, baseline_branches, logger,
                batch_size=args.max_batch_size, max_seq_len=max_seq_len
            )
            baseline_metadata = steering.compute_baseline_metadata(
                baseline_branches, baseline_branch_log_probs
            )
            
            # Initialize prefix results
            prefix_results = {
                "prefix_id": prefix_id,
                "baseline_method": "kmeans",
                "feature_selection": "magnitude",
                "clustering_runs": {}
            }
            
            batch_prefix_state[prefix_id] = {
                "prefix_data": prefix_data,
                "active_features": active_features,
                "selected_features": selected_features,
                "n_features": n_features,
                "K_map": K_map,
                "rd_sweep_data": rd_sweep_data,
                "H_0": H_0,
                "baseline_metadata": baseline_metadata,
                "prefix_results": prefix_results,
                "clustering_ctx": {}
            }
        
        if not batch_prefix_state:
            continue
        
        # Process each (beta, gamma) config - prepare contexts for all prefixes
        all_clustering_keys = set()
        for state in batch_prefix_state.values():
            all_clustering_keys.update(state["K_map"].keys())
        
        for clustering_key in sorted(all_clustering_keys):
            logger.info(f"  Processing clustering config: {clustering_key}")
            
            for prefix_id, state in batch_prefix_state.items():
                K_map = state["K_map"]
                if clustering_key not in K_map:
                    continue
                    
                K = K_map[clustering_key]
                rd_sweep_data = state["rd_sweep_data"]
                sweep_config = rd_sweep_data.get("sweep_config", {})
                K_max = sweep_config.get("K_max", 20)
                
                if K == 1 or K == K_max:
                    continue
                
                prefix_data = state["prefix_data"]
                H_0 = state["H_0"]
                active_features = state["active_features"]
                selected_features = state["selected_features"]
                n_features = state["n_features"]
                
                # Run K-means
                t0_kmeans = time.perf_counter()
                assignments, centers = run_kmeans_clustering(
                    prefix_data["embeddings"], K, random_state=42
                )
                t1_kmeans = time.perf_counter()
                
                # Build semantic graphs
                semantic_graphs = build_semantic_graphs_from_kmeans(
                    assignments, prefix_data["attributions"],
                    prefix_data["path_probs"], H_0,
                    use_median=not args.use_mean
                )
                
                if not semantic_graphs:
                    continue
                
                # Collect feature indices
                all_needed_indices = set()
                for cluster_id, H_c in semantic_graphs.items():
                    H_c_features = H_c[:n_features]
                    abs_vals = np.abs(H_c_features)
                    top_indices = np.argsort(abs_vals)[-(max_top_B * 2):]
                    for idx in top_indices:
                        if abs(H_c_features[idx]) >= utils.EPSILON_SMALL:
                            all_needed_indices.add(int(idx))
                
                # Build decoder cache
                t0_cache = time.perf_counter()
                global_decoder_cache = {}
                if all_needed_indices:
                    layer_to_indices = {}
                    for h_c_idx in all_needed_indices:
                        feat_idx = selected_features[h_c_idx].item()
                        layer, pos, feat_id = active_features[feat_idx].tolist()
                        layer = int(layer)
                        if layer not in layer_to_indices:
                            layer_to_indices[layer] = []
                        layer_to_indices[layer].append((h_c_idx, int(feat_id)))
                    
                    for layer, idx_list in layer_to_indices.items():
                        h_c_indices = [x[0] for x in idx_list]
                        feat_ids = [x[1] for x in idx_list]
                        feat_ids_t = torch.tensor(feat_ids, device=device, dtype=torch.long)
                        dec_vecs = model.transcoders._get_decoder_vectors(layer, feat_ids_t)
                        for i, h_c_idx in enumerate(h_c_indices):
                            global_decoder_cache[h_c_idx] = dec_vecs[i]
                t1_cache = time.perf_counter()
                
                decoder_cache = graph.build_cluster_decoder_cache(
                    semantic_graphs, global_decoder_cache, active_features, selected_features,
                    max_features=max_top_B * 2
                )
                
                features_by_cluster = {}
                for c, cache_data in decoder_cache.items():
                    tuples = [(
                        cache_data['layers'][i],
                        cache_data['positions'][i],
                        cache_data['feat_ids'][i],
                        cache_data['h_c_values'][i]
                    ) for i in range(len(cache_data['h_c_values']))]
                    features_by_cluster[c] = tuples
                
                t0_encoder = time.perf_counter()
                encoder_cache = graph.precompute_cluster_encoder_weights(
                    model, features_by_cluster, device
                )
                t1_encoder = time.perf_counter()
                
                # Build branches
                branches = utils.build_branches_from_data(
                    prefix_data["branches_data"], assignments.tolist()
                )
                
                # Initialize results for this clustering config
                state["prefix_results"]["clustering_runs"][clustering_key] = {
                    "beta": float(clustering_key.split("_")[0].replace("beta", "")),
                    "gamma": float(clustering_key.split("_")[1].replace("gamma", "")),
                    "K": K,
                    "n_clusters": len(semantic_graphs),
                    "n_branches": len(branches),
                    "results": {},
                    "timing": {
                        "kmeans_s": t1_kmeans - t0_kmeans,
                        "decoder_cache_s": t1_cache - t0_cache,
                        "encoder_cache_s": t1_encoder - t0_encoder,
                    }
                }
                
                # Build context for batching
                ctx = {
                    "prefix_id": prefix_id,
                    "branches": branches,
                    "decoder_cache": decoder_cache,
                    "encoder_cache": encoder_cache,
                    "baseline_metadata": state["baseline_metadata"],
                    "semantic_graphs": semantic_graphs,
                    "features_by_cluster": features_by_cluster,
                }
                batch_contexts_by_key.setdefault(clustering_key, []).append(ctx)
                state["clustering_ctx"][clustering_key] = ctx
        
        # Run sweeps for this batch
        for clustering_key, ctx_list in batch_contexts_by_key.items():
            t0_sweep = time.perf_counter()
            
            for sw in sweeps:
                method = sw.get("steering_method")
                hc_sels = sw.get("h_c_selections", ["full"])
                top_B_list = sw.get("top_B", [10])
                eps_list = sw.get("epsilon_values", [-1.0, 0.0, 1.0])
                
                for hc_sel, top_B in product(hc_sels, top_B_list):
                    key = metrics.generate_sweep_key(method, hc_sel, top_B)
                    logger.info(f"    Sweep: {key} ({clustering_key}, {len(ctx_list)} prefixes)")
                    
                    if args.cross_prefix_batching and len(ctx_list) > 1:
                        # Batch across prefixes
                        results_by_prefix = hypotheses.run_steering_sweep_prefix_batch(
                            model=model,
                            prefix_contexts=ctx_list,
                            epsilons=eps_list,
                            top_B=top_B,
                            steering_method=method,
                            hc_selection=hc_sel,
                            max_samples_per_cluster=args.max_cluster_samples,
                            log_details=False,
                            max_batch_size=args.max_batch_size,
                            max_seq_len=max_seq_len,
                            logger=logger,
                        )
                        for ctx in ctx_list:
                            p_id = ctx["prefix_id"]
                            result = results_by_prefix.get(p_id, {"error": "no_result"})
                            batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][clustering_key]["results"][key] = result
                    else:
                        # Process each prefix individually
                        for ctx in ctx_list:
                            p_id = ctx["prefix_id"]
                            result = hypotheses.run_steering_sweep(
                                model=model,
                                branches=ctx["branches"],
                                decoder_cache=ctx["decoder_cache"],
                                encoder_cache=ctx["encoder_cache"],
                                baseline_metadata=ctx["baseline_metadata"],
                                epsilons=eps_list,
                                top_B=top_B,
                                steering_method=method,
                                hc_selection=hc_sel,
                                max_samples_per_cluster=args.max_cluster_samples,
                                log_details=False,
                                max_batch_size=args.max_batch_size,
                                cross_prefix_batching=False,
                                max_seq_len=max_seq_len,
                                logger=logger,
                            )
                            batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][clustering_key]["results"][key] = result
                    
                    clear_memory()
            
            t1_sweep = time.perf_counter()
            for ctx in ctx_list:
                p_id = ctx["prefix_id"]
                if clustering_key in batch_prefix_state[p_id]["prefix_results"]["clustering_runs"]:
                    batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][clustering_key]["timing"]["sweep_s"] = t1_sweep - t0_sweep
        
        # Save results for all prefixes in batch
        for prefix_id, state in batch_prefix_state.items():
            output_file = h4a_output_dir / f"{prefix_id}_sweep_results.json"
            save_json(state["prefix_results"], output_file)
            logger.info(f"  Saved {prefix_id} results to {output_file}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 60)
    logger.info("K-MEANS BASELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

