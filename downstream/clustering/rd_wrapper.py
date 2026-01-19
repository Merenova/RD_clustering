"""Rate-Distortion clustering wrapper.

Thin wrapper around 5_gaussian_clustering/cluster.py for the downstream pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import config
from downstream.config import CLUSTERING_CONFIG, BETA_VALUES, GAMMA_VALUES

# Add 5_gaussian_clustering to path
CLUSTERING_MODULE_PATH = Path(__file__).resolve().parents[2] / "5_gaussian_clustering"
sys.path.insert(0, str(CLUSTERING_MODULE_PATH))

# Import the main clustering function
from cluster import run_clustering


def run_rd_clustering(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray = None,
    beta_e: float = None,
    beta_a: float = None,
    beta: float = None,
    gamma: float = None,
    K_max: int = None,
    max_iterations: int = None,
    convergence_threshold: float = None,
    metric_a: str = None,
    normalize_dims: bool = None,
    center_attributions: bool = True,
    normalize_attributions: bool = True,
    global_H_0: np.ndarray = None,
    global_rms_norm: float = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run Rate-Distortion clustering.

    Wraps cluster.run_clustering from 5_gaussian_clustering.
    All computations use probability-weighted distortion and centers.

    Args:
        embeddings_e: Semantic embeddings [n_samples, d_e]
        attributions_a: Attribution embeddings [n_samples, d_a]
        path_probs: Path probabilities [n_samples] (uniform if None)
        beta_e: Semantic distortion weight (computed from beta, gamma if None)
        beta_a: Attribution distortion weight (computed from beta, gamma if None)
        beta: Overall rate-distortion trade-off
        gamma: Semantic vs attribution trade-off (0-1)
        K_max: Maximum number of components
        max_iterations: Max EM iterations
        convergence_threshold: Convergence threshold
        metric_a: Metric for attribution distance ("l2" or "l1")
        normalize_dims: Whether to normalize beta by dimensions (beta_e /= sqrt(d_e), beta_a /= d_a)
        center_attributions: Whether to center attributions (pre-processing)
        normalize_attributions: Whether to normalize attributions (pre-processing)
        global_H_0: Pre-computed global mean for cross-prefix/cross-model comparison
        global_rms_norm: Pre-computed global RMS norm for cross-prefix/cross-model comparison
        logger: Optional logger

    Returns:
        Dict with assignments, components, rd_stats, etc.
    """
    # Apply defaults from config
    if K_max is None:
        K_max = CLUSTERING_CONFIG["K_max"]
    if max_iterations is None:
        max_iterations = CLUSTERING_CONFIG["max_iterations"]
    if convergence_threshold is None:
        convergence_threshold = CLUSTERING_CONFIG["convergence_threshold"]
    if metric_a is None:
        metric_a = CLUSTERING_CONFIG["metric_a"]
    if normalize_dims is None:
        normalize_dims = CLUSTERING_CONFIG.get("normalize_dims", False)

    # Get dimensions for normalization
    d_e = embeddings_e.shape[1]
    d_a = attributions_a.shape[1]

    # Compute beta_e, beta_a from beta, gamma if not provided
    if beta_e is None or beta_a is None:
        if beta is None or gamma is None:
            beta = 1.0
            gamma = 0.5

        # Apply dimension normalization if enabled
        # This accounts for L2 scaling as sqrt(d) and L1 scaling as d
        if normalize_dims:
            beta_e = gamma * beta / np.sqrt(d_e)
            beta_a = (1 - gamma) * beta / d_a
        else:
            beta_e = gamma * beta
            beta_a = (1 - gamma) * beta
    
    # Default uniform probabilities
    if path_probs is None:
        path_probs = np.ones(len(embeddings_e)) / len(embeddings_e)
    
    # Create null logger if none provided
    if logger is None:
        logger = logging.getLogger("rd_clustering")
        logger.setLevel(logging.WARNING)
    
    # Pre-process attributions (center + normalize)
    attributions_processed = attributions_a.copy()
    H_0 = None
    rms_norm = 1.0
    
    # Use global normalization if provided (for cross-prefix/cross-model comparisons)
    if global_H_0 is not None and global_rms_norm is not None:
        H_0 = global_H_0
        rms_norm = global_rms_norm
        attributions_processed = (attributions_a - H_0) / rms_norm
    else:
        # Compute per-call normalization (default behavior)
        if center_attributions:
            W_total = path_probs.sum()
            if W_total > 0:
                H_0 = np.sum(path_probs[:, None] * attributions_a, axis=0) / W_total
            else:
                H_0 = np.zeros(attributions_a.shape[1])
            attributions_processed = attributions_a - H_0
        
        if normalize_attributions:
            norms_sq = np.sum(attributions_processed ** 2, axis=1)
            rms_norm = np.sqrt(np.mean(norms_sq))
            if rms_norm > 1e-10:
                attributions_processed = attributions_processed / rms_norm
    
    # Prepare data dict for run_clustering
    data = {
        "prefix_id": "downstream",
        "prefix": "",
        "embeddings_e": embeddings_e,
        "attributions_a": attributions_processed,
        "attributions_a_original": attributions_a,
        "H_0": H_0,
        "attribution_rms_norm": rms_norm,
        "path_probs": path_probs,
        "n_samples": len(embeddings_e),
    }
    
    # Run clustering using main module (always probability-weighted)
    result = run_clustering(
        data=data,
        beta_e=beta_e,
        beta_a=beta_a,
        K_max=K_max,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        logger=logger,
        metric_a=metric_a,
    )
    
    # Reformat result for downstream use
    return {
        "assignments": np.array(result["assignments"]),
        "components": result["components"],
        "rd_stats": result["rd_stats"],
        "H_0": result.get("H_0"),
        "rms_norm": rms_norm,
        "n_iterations": result["n_iterations"],
        "n_components": len(result["components"]),
        "K_max": K_max,
        "beta_e": beta_e,
        "beta_a": beta_a,
        "converged": result.get("converged", False),
    }


def compute_global_normalization(
    attributions_list: List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Compute global H_0 and rms_norm across multiple attribution arrays.
    
    Use this before cross-prefix or cross-model clustering to ensure
    centroids are in comparable spaces.
    
    Args:
        attributions_list: List of attribution arrays to normalize together
        
    Returns:
        Tuple of (global_H_0, global_rms_norm)
    """
    # Concatenate all attributions
    all_attrs = np.vstack([a for a in attributions_list if len(a) > 0])
    
    # Compute global mean
    global_H_0 = np.mean(all_attrs, axis=0)
    
    # Center and compute RMS norm
    centered = all_attrs - global_H_0
    global_rms_norm = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    
    if global_rms_norm < 1e-10:
        global_rms_norm = 1.0
    
    return global_H_0, global_rms_norm


def run_rd_sweep(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray = None,
    beta_values: List[float] = None,
    gamma_values: List[float] = None,
    normalize_dims: bool = None,
    **kwargs,
) -> Dict[Tuple[float, float], Dict]:
    """Run RD clustering for all (beta, gamma) combinations.

    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        path_probs: Path probabilities
        beta_values: List of beta values to sweep
        gamma_values: List of gamma values to sweep
        normalize_dims: Whether to normalize beta by dimensions
        **kwargs: Additional arguments passed to run_rd_clustering

    Returns:
        Dict mapping (beta, gamma) -> clustering result dict
    """
    from tqdm import tqdm

    if beta_values is None:
        beta_values = BETA_VALUES
    if gamma_values is None:
        gamma_values = GAMMA_VALUES
    if normalize_dims is None:
        normalize_dims = CLUSTERING_CONFIG.get("normalize_dims", False)
    
    results = {}
    total = len(beta_values) * len(gamma_values)

    with tqdm(total=total, desc="RD Sweep", leave=False) as pbar:
        for beta in beta_values:
            for gamma in gamma_values:
                result = run_rd_clustering(
                    embeddings_e, attributions_a, path_probs,
                    beta=beta, gamma=gamma,
                    normalize_dims=normalize_dims,
                    **kwargs
                )

                result["beta"] = beta
                result["gamma"] = gamma
                result["normalize_dims"] = normalize_dims

                results[(beta, gamma)] = result
                pbar.update(1)

    return results


def run_kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Run K-Means clustering.
    
    Args:
        data: Data array [n_samples, n_features]
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Assignments array
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(data)


def run_kmeans_sweep(
    data: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[int, np.ndarray]:
    """Run K-Means for multiple K values.
    
    Args:
        data: Data array [n_samples, n_features]
        k_range: Range of K values to try
        random_state: Random seed
        
    Returns:
        Dict mapping K -> assignments array
    """
    from sklearn.cluster import KMeans
    
    results = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        results[k] = kmeans.fit_predict(data)
    return results
