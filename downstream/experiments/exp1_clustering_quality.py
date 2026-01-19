"""Experiment 1: Clustering Quality (Sanity Check).

Verifies that the RD objective is minimized properly and examines
the trade-off between rate and distortion across (beta, gamma) configurations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json

# Import from downstream
from downstream.config import BETA_VALUES, GAMMA_VALUES, OUTPUT_DIR
from downstream.clustering.rd_wrapper import run_rd_sweep
from downstream.clustering.config_selection import (
    compute_cluster_metrics,
    summarize_sweep_results,
)


def run_experiment_1(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray = None,
    beta_values: List[float] = None,
    gamma_values: List[float] = None,
    labels: np.ndarray = None,
    output_dir: Path = None,
    logger=None,
) -> Dict[str, Any]:
    """Run Experiment 1: Clustering Quality.
    
    Goal: Verify RD objective is minimized properly.
    
    Expected outcomes:
    - Higher β → higher rate (more clusters), lower distortion
    - Higher γ → lower semantic distortion, higher attribution distortion
    
    Args:
        embeddings_e: Semantic embeddings [n_samples, d_e]
        attributions_a: Attribution embeddings [n_samples, d_a]
        path_probs: Path probabilities [n_samples]
        beta_values: List of beta values to sweep
        gamma_values: List of gamma values to sweep
        labels: Optional ground truth labels for purity metrics
        output_dir: Output directory for results
        logger: Logger instance
        
    Returns:
        Dict with experiment results
    """
    if beta_values is None:
        beta_values = BETA_VALUES
    if gamma_values is None:
        gamma_values = GAMMA_VALUES
    if output_dir is None:
        output_dir = OUTPUT_DIR / "exp1_clustering_quality"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: CLUSTERING QUALITY")
        logger.info("=" * 60)
        logger.info(f"N samples: {len(embeddings_e)}")
        logger.info(f"Embedding dim: {embeddings_e.shape[1]}")
        logger.info(f"Attribution dim: {attributions_a.shape[1]}")
        logger.info(f"Beta values: {beta_values}")
        logger.info(f"Gamma values: {gamma_values}")
    
    # Run sweep
    if logger:
        logger.info("\nRunning RD clustering sweep...")
    
    sweep_results = run_rd_sweep(
        embeddings_e,
        attributions_a,
        path_probs,
        beta_values=beta_values,
        gamma_values=gamma_values,
    )
    
    # Summarize results
    summaries = summarize_sweep_results(
        sweep_results,
        embeddings_e,
        attributions_a,
        labels,
    )
    
    # Extract trade-off curves
    rate_distortion_curve = []
    beta_distortion_curve = {}  # beta -> list of (rate, distortion)
    gamma_distortion_curve = {}  # gamma -> list of (sem_dist, attr_dist)
    
    for (beta, gamma), result in sweep_results.items():
        rd_stats = result.get("rd_stats", {})
        
        rate = rd_stats.get("H", 0)
        d_sem = rd_stats.get("D_e", 0)
        d_attr = rd_stats.get("D_a", 0)
        d_total = gamma * d_sem + (1 - gamma) * d_attr
        
        rate_distortion_curve.append({
            "beta": beta,
            "gamma": gamma,
            "rate": rate,
            "distortion": d_total,
            "distortion_semantic": d_sem,
            "distortion_attribution": d_attr,
            "n_clusters": result.get("n_components", 0),
            "L_RD": rd_stats.get("L_RD", 0),
        })
        
        # Group by beta
        if beta not in beta_distortion_curve:
            beta_distortion_curve[beta] = []
        beta_distortion_curve[beta].append({
            "gamma": gamma,
            "rate": rate,
            "distortion": d_total,
        })
        
        # Group by gamma
        if gamma not in gamma_distortion_curve:
            gamma_distortion_curve[gamma] = []
        gamma_distortion_curve[gamma].append({
            "beta": beta,
            "d_semantic": d_sem,
            "d_attribution": d_attr,
        })
    
    # Analyze trade-offs
    analysis = analyze_tradeoffs(rate_distortion_curve)
    
    # Find best configuration
    best_summary = summaries[0] if summaries else {}
    
    if logger:
        logger.info("\n" + "-" * 40)
        logger.info("RESULTS SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Best config: beta={best_summary.get('beta')}, gamma={best_summary.get('gamma')}")
        logger.info(f"  Harmonic silhouette: {best_summary.get('harmonic_silhouette', 0):.4f}")
        logger.info(f"  N clusters: {best_summary.get('n_clusters', 0)}")
        logger.info(f"\nTrade-off analysis:")
        logger.info(f"  Rate-distortion correlation: {analysis['rate_distortion_correlation']:.4f}")
        logger.info(f"  Beta increases rate: {analysis['beta_increases_rate']}")
        logger.info(f"  Gamma affects semantic: {analysis['gamma_affects_semantic']}")
    
    # Compile results
    results = {
        "n_samples": len(embeddings_e),
        "beta_values": beta_values,
        "gamma_values": gamma_values,
        "sweep_summaries": summaries,
        "rate_distortion_curve": rate_distortion_curve,
        "analysis": analysis,
        "best_config": {
            "beta": best_summary.get("beta"),
            "gamma": best_summary.get("gamma"),
            "harmonic_silhouette": best_summary.get("harmonic_silhouette", 0),
            "n_clusters": best_summary.get("n_clusters", 0),
        },
    }
    
    # Save results
    save_json(results, output_dir / "exp1_results.json")
    if logger:
        logger.info(f"\nResults saved to {output_dir / 'exp1_results.json'}")
    
    return results


def analyze_tradeoffs(
    rate_distortion_curve: List[Dict],
) -> Dict[str, Any]:
    """Analyze trade-off properties from sweep results.
    
    Args:
        rate_distortion_curve: List of (beta, gamma, rate, distortion) dicts
        
    Returns:
        Dict with analysis results
    """
    if not rate_distortion_curve:
        return {
            "rate_distortion_correlation": 0.0,
            "beta_increases_rate": False,
            "gamma_affects_semantic": False,
        }
    
    rates = [x["rate"] for x in rate_distortion_curve]
    distortions = [x["distortion"] for x in rate_distortion_curve]
    betas = [x["beta"] for x in rate_distortion_curve]
    gammas = [x["gamma"] for x in rate_distortion_curve]
    d_semantic = [x["distortion_semantic"] for x in rate_distortion_curve]
    d_attribution = [x["distortion_attribution"] for x in rate_distortion_curve]
    
    # Rate-distortion correlation (should be negative)
    if len(rates) > 1 and np.std(rates) > 0 and np.std(distortions) > 0:
        rd_corr = np.corrcoef(rates, distortions)[0, 1]
    else:
        rd_corr = 0.0
    
    # Higher beta -> higher rate? (should be positive)
    if len(betas) > 1 and np.std(betas) > 0 and np.std(rates) > 0:
        beta_rate_corr = np.corrcoef(betas, rates)[0, 1]
    else:
        beta_rate_corr = 0.0
    
    # Higher gamma -> lower semantic distortion? (should be negative)
    if len(gammas) > 1 and np.std(gammas) > 0 and np.std(d_semantic) > 0:
        gamma_sem_corr = np.corrcoef(gammas, d_semantic)[0, 1]
    else:
        gamma_sem_corr = 0.0
    
    # Higher gamma -> higher attribution distortion?
    if len(gammas) > 1 and np.std(gammas) > 0 and np.std(d_attribution) > 0:
        gamma_attr_corr = np.corrcoef(gammas, d_attribution)[0, 1]
    else:
        gamma_attr_corr = 0.0
    
    return {
        "rate_distortion_correlation": float(rd_corr),
        "beta_rate_correlation": float(beta_rate_corr),
        "gamma_semantic_correlation": float(gamma_sem_corr),
        "gamma_attribution_correlation": float(gamma_attr_corr),
        "beta_increases_rate": beta_rate_corr > 0.3,
        "gamma_affects_semantic": gamma_sem_corr < -0.3,
        "gamma_affects_attribution": gamma_attr_corr > 0.3,
    }

