#!/usr/bin/env python3
"""
Ablation study for adaptive control in Rate-Distortion Clustering.
Compares original adaptive clustering results with fixed-K clustering initialized
via K-Means on embeddings (e-only) or attributions (a-only).
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster import load_prefix_data
from em_loop import run_em_iteration, check_convergence
from rd_objective import compute_full_rd_statistics, compute_component_masses
from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger

def initialize_kmeans(
    data_source: np.ndarray,
    n_clusters: int,
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    seed: int = 42
) -> tuple:
    """
    Initialize components using K-Means on a specific data source.
    
    Args:
        data_source: Data to run K-Means on (e.g. embeddings_e or attributions_a)
        n_clusters: Number of clusters (K)
        embeddings_e, attributions_a: Full data for component stats
        path_probs: Path probabilities
        
    Returns:
        (components, assignments)
    """
    if n_clusters <= 1:
        # Fallback to single component init logic if K=1, but using K-Means
        # actually K-Means with K=1 is just the mean.
        # But let's use the explicit K-Means for consistency if K>1 logic is general.
        pass

    # Run K-Means
    # Note: K-Means in sklearn doesn't support weighted samples for clustering 
    # in the standard fit method in a way that affects centers exactly like our EM 
    # (standard k-means minimizes inertia). 
    # However, we can use sample_weight in fit() if available, or just unweighted.
    # Given path_probs are important, we should try to use them if possible, 
    # but for initialization, unweighted K-Means is a standard baseline "heuristic".
    # Let's use unweighted K-Means for robust initialization seeds.
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(data_source) # sample_weight=path_probs could be passed here
    
    # Recompute component statistics based on these assignments using OUR weighted logic
    components = {}
    
    # We need to map 0..K-1 to component IDs. Let's use 1..K
    unique_labels = np.unique(assignments)
    
    for label in unique_labels:
        comp_id = int(label) + 1
        indices = np.where(assignments == label)[0].tolist()
        
        if len(indices) == 0:
            continue
            
        # Compute statistics
        P_c = path_probs[indices]
        W_c = np.sum(P_c)
        
        if W_c == 0:
            continue
            
        mu_e = np.sum(P_c[:, None] * embeddings_e[indices], axis=0) / W_c
        mu_e_norm = np.linalg.norm(mu_e)
        if mu_e_norm > 1e-10:
            mu_e = mu_e / mu_e_norm
            
        mu_a = np.sum(P_c[:, None] * attributions_a[indices], axis=0) / W_c
        
        components[comp_id] = {
            'mu_e': mu_e,
            'mu_a': mu_a,
            'W_c': W_c,
            'indices': indices
        }
        
    # Remap assignments to match component IDs
    final_assignments = [int(x) + 1 for x in assignments]
    
    return components, final_assignments

def run_fixed_clustering(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    initial_components: dict,
    initial_assignments: list,
    beta_e: float,
    beta_a: float,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-6
) -> dict:
    """
    Run EM loop without adaptive control (fixed number of clusters).
    """
    components = initial_components.copy()
    assignments = list(initial_assignments)
    
    # Initial stats
    rd_stats = compute_full_rd_statistics(
        embeddings_e, attributions_a, assignments, path_probs,
        components, beta_e, beta_a
    )
    L_RD_prev = rd_stats['L_RD']
    
    iteration = 0
    converged = False
    
    for iteration in range(max_iterations):
        # EM Step
        assignments, components, rd_stats = run_em_iteration(
            embeddings_e,
            attributions_a,
            path_probs,
            components,
            beta_e,
            beta_a
        )
        
        L_RD_curr = rd_stats['L_RD']
        
        # Check convergence
        if check_convergence(L_RD_prev, L_RD_curr, convergence_threshold):
            converged = True
            break
            
        L_RD_prev = L_RD_curr
        
    return {
        "components": components,
        "assignments": assignments,
        "rd_stats": rd_stats,
        "converged": converged,
        "n_iterations": iteration + 1
    }

def main():
    # Setup paths
    base_results_dir = Path("/home/hyunjin/latent_planning/Qwen3_4B_results/sweep_span_full_pool_sum/results")
    clustering_dir = base_results_dir / "5_clustering"
    output_dir = base_results_dir / "compare_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Input directories
    embeddings_dir = base_results_dir / "4_feature_extraction/embeddings"
    attribution_graphs_dir = base_results_dir / "3_attribution_graphs" # Note: inferred from listing
    samples_dir = base_results_dir / "2_branch_sampling" # Note: inferred from listing
    
    # Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ablation")
    
    # Find all clustering sweep result files
    result_files = list(clustering_dir.glob("cloze_*_sweep_results.json"))
    logger.info(f"Found {len(result_files)} sweep results to process.")
    
    for result_file in tqdm(result_files, desc="Processing prefixes"):
        try:
            # Load sweep result
            res_data = load_json(result_file)
            prefix_id = res_data["prefix_id"]

            grid = res_data.get("grid", [])
            if not grid:
                logger.warning(f"No grid results for {prefix_id}, skipping")
                continue

            # Select best entry by harmonic score for comparison
            best_entry = max(grid, key=lambda x: x.get("harmonic", -1))
            beta = best_entry.get("beta")
            gamma = best_entry.get("gamma")

            beta_e = beta * gamma
            beta_a = beta * (1 - gamma)

            final_n_components = best_entry.get("K", 0)
            original_rd = best_entry.get("L_RD", None)
            
            # Prepare output structure
            comparison = {
                "prefix_id": prefix_id,
                "beta": beta,
                "gamma": gamma,
                "n_components": final_n_components,
                "original": {
                    "L_RD": original_rd,
                    "rd_stats": res_data["rd_objective"]
                },
                "ablations": {}
            }
            
            # If K=1, ablation is trivial (same as original single component), but let's run it 
            # or just skip. If K=1, K-means with K=1 is just the mean.
            # The user specifically asked to compare, so we should run it to see if adaptive control
            # changed anything (e.g. maybe it tried to split and failed, or junked something).
            # But "initiate with same number" implies we force K.
            
            # Load data (continuation attribution, full span)
            data = load_prefix_data(
                prefix_id,
                embeddings_dir,
                attribution_graphs_dir,
                samples_dir,
                logger,
                pooling="mean",
            )
            
            embeddings_e = data["embeddings_e"]
            attributions_a = data["attributions_a"]
            path_probs = data["path_probs"]
            
            # 1. Embeddings-only initialization (e-init)
            logger.debug(f"Running e-init for {prefix_id} K={final_n_components}")
            comps_e, assigns_e = initialize_kmeans(
                embeddings_e, final_n_components, 
                embeddings_e, attributions_a, path_probs, seed=42
            )
            res_e = run_fixed_clustering(
                embeddings_e, attributions_a, path_probs,
                comps_e, assigns_e, beta_e, beta_a
            )
            comparison["ablations"]["embeddings_only_init"] = {
                "L_RD": res_e["rd_stats"]["L_RD"],
                "rd_stats": res_e["rd_stats"]
            }
            
            # 2. Attributions-only initialization (a-init)
            logger.debug(f"Running a-init for {prefix_id} K={final_n_components}")
            comps_a, assigns_a = initialize_kmeans(
                attributions_a, final_n_components, 
                embeddings_e, attributions_a, path_probs, seed=42
            )
            res_a = run_fixed_clustering(
                embeddings_e, attributions_a, path_probs,
                comps_a, assigns_a, beta_e, beta_a
            )
            comparison["ablations"]["attributions_only_init"] = {
                "L_RD": res_a["rd_stats"]["L_RD"],
                "rd_stats": res_a["rd_stats"]
            }
            
            # Save individual comparison result
            save_json(comparison, output_dir / f"{prefix_id}_comparison.json")
            
        except Exception as e:
            logger.error(f"Failed to process {result_file.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

