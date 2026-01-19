"""Experiment 2: Binary vs RD Granularity (Core Experiment).

Shows that RD clustering finds meaningful substructure that binary labels miss.
Compares RD clustering against baselines (label-based, K-Means on single views).
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json

# Import from downstream
from downstream.config import OUTPUT_DIR
from downstream.clustering.rd_wrapper import run_rd_clustering, run_kmeans
from downstream.clustering.config_selection import (
    compute_cluster_metrics,
    compute_label_purity,
    compute_intra_cluster_label_variance,
    select_optimal_config,
)


def run_experiment_2(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    labels: np.ndarray,
    path_probs: np.ndarray = None,
    optimal_beta: float = None,
    optimal_gamma: float = None,
    output_dir: Path = None,
    logger=None,
) -> Dict[str, Any]:
    """Run Experiment 2: Binary vs RD Granularity.
    
    Goal: Show RD clustering finds meaningful substructure that binary misses.
    
    Baselines:
    - Label-based: 2 clusters by ground truth label
    - Semantic 2-means: K-Means on embeddings with K=2
    - Attribution 2-means: K-Means on attributions with K=2
    - Semantic K-means: K-Means on embeddings matching RD cluster count
    
    Args:
        embeddings_e: Semantic embeddings [n_samples, d_e]
        attributions_a: Attribution embeddings [n_samples, d_a]
        labels: Binary ground truth labels (1=positive, 0=negative)
        path_probs: Path probabilities
        optimal_beta: Pre-selected optimal beta (runs sweep if None)
        optimal_gamma: Pre-selected optimal gamma
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dict with experiment results
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "exp2_binary_vs_rd"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: BINARY VS RD GRANULARITY")
        logger.info("=" * 60)
        logger.info(f"N samples: {len(embeddings_e)}")
        logger.info(f"Label distribution: {labels.sum()}/{len(labels)} positive")
    
    # Run RD clustering
    if optimal_beta is None or optimal_gamma is None:
        if logger:
            logger.info("\nRunning RD sweep to find optimal config...")
        
        from downstream.clustering.rd_wrapper import run_rd_sweep
        
        sweep_results = run_rd_sweep(
            embeddings_e, attributions_a, path_probs
        )
        
        optimal_beta, optimal_gamma, rd_result, best_score = select_optimal_config(
            sweep_results, embeddings_e, attributions_a
        )
        
        if logger:
            logger.info(f"Optimal config: beta={optimal_beta}, gamma={optimal_gamma}")
    else:
        rd_result = run_rd_clustering(
            embeddings_e, attributions_a, path_probs,
            beta=optimal_beta, gamma=optimal_gamma,
        )
    
    rd_assignments = rd_result["assignments"]
    rd_n_clusters = len(np.unique(rd_assignments))
    
    if logger:
        logger.info(f"RD clustering found {rd_n_clusters} clusters")
    
    # Baselines
    baselines = {}
    
    # 1. Label-based (ground truth)
    baselines["label_based"] = labels.astype(int)
    
    # 2. Semantic 2-means
    baselines["semantic_2means"] = run_kmeans(embeddings_e, n_clusters=2)
    
    # 3. Attribution 2-means
    baselines["attribution_2means"] = run_kmeans(attributions_a, n_clusters=2)
    
    # 4. Semantic K-means (matching RD cluster count)
    if rd_n_clusters > 2:
        baselines["semantic_kmeans_k"] = run_kmeans(embeddings_e, n_clusters=rd_n_clusters)
    
    # 5. Attribution K-means (matching RD cluster count)
    if rd_n_clusters > 2:
        baselines["attribution_kmeans_k"] = run_kmeans(attributions_a, n_clusters=rd_n_clusters)
    
    # Evaluate all methods
    results_table = []
    
    all_methods = {
        **baselines,
        "rd_clustering": rd_assignments,
    }
    
    for method_name, assignments in all_methods.items():
        metrics = compute_cluster_metrics(
            embeddings_e, attributions_a, assignments, labels
        )
        
        # Add AUROC
        auroc = compute_auroc_from_clusters(assignments, labels)
        metrics["auroc"] = auroc
        
        results_table.append({
            "method": method_name,
            **metrics,
        })
    
    # Sort by harmonic silhouette
    results_table.sort(key=lambda x: x["harmonic_silhouette"], reverse=True)
    
    # Print table
    if logger:
        logger.info("\n" + "-" * 80)
        logger.info("RESULTS TABLE")
        logger.info("-" * 80)
        logger.info(f"{'Method':<25} {'K':<5} {'Sil':<8} {'Sem.Coh':<8} {'Attr.Con':<8} {'Purity':<8} {'LblVar':<8}")
        logger.info("-" * 80)
        
        for row in results_table:
            logger.info(
                f"{row['method']:<25} "
                f"{row['n_clusters']:<5} "
                f"{row['harmonic_silhouette']:.4f}  "
                f"{row['semantic_coherence']:.4f}  "
                f"{row['attribution_consistency']:.4f}  "
                f"{row.get('label_purity', 0):.4f}  "
                f"{row.get('label_variance', 0):.4f}"
            )
        
        logger.info("-" * 80)
    
    # Key insights
    rd_row = next(r for r in results_table if r["method"] == "rd_clustering")
    label_row = next(r for r in results_table if r["method"] == "label_based")
    
    insights = {
        "rd_better_silhouette": bool(rd_row["harmonic_silhouette"] > label_row["harmonic_silhouette"]),
        "rd_finds_substructure": bool(rd_row["n_clusters"] > 2 and rd_row["label_variance"] > 0.1),
        "rd_better_coherence": bool(
            rd_row["semantic_coherence"] > label_row["semantic_coherence"] and
            rd_row["attribution_consistency"] > label_row["attribution_consistency"]
        ),
    }
    
    if logger:
        logger.info("\nKey Insights:")
        logger.info(f"  RD better silhouette: {insights['rd_better_silhouette']}")
        logger.info(f"  RD finds substructure: {insights['rd_finds_substructure']}")
        logger.info(f"  RD better coherence: {insights['rd_better_coherence']}")
    
    # Compile results
    results = {
        "n_samples": len(embeddings_e),
        "optimal_beta": optimal_beta,
        "optimal_gamma": optimal_gamma,
        "rd_n_clusters": rd_n_clusters,
        "results_table": results_table,
        "insights": insights,
    }
    
    # Save results
    save_json(results, output_dir / "exp2_results.json")
    if logger:
        logger.info(f"\nResults saved to {output_dir / 'exp2_results.json'}")
    
    return results


def compute_auroc_from_clusters(
    assignments: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute AUROC treating cluster assignment as predictor.
    
    Assigns cluster labels based on majority vote and computes AUROC.
    
    Args:
        assignments: Cluster assignments
        labels: Binary ground truth labels
        
    Returns:
        AUROC score
    """
    unique_clusters = np.unique(assignments)
    
    if len(unique_clusters) < 2:
        return 0.5
    
    # For each cluster, compute fraction of positive labels
    cluster_positive_rate = {}
    for c in unique_clusters:
        mask = assignments == c
        if np.sum(mask) > 0:
            cluster_positive_rate[c] = np.mean(labels[mask])
        else:
            cluster_positive_rate[c] = 0.5
    
    # Use cluster positive rate as prediction score
    predictions = np.array([cluster_positive_rate[a] for a in assignments])
    
    try:
        return roc_auc_score(labels, predictions)
    except ValueError:
        return 0.5

