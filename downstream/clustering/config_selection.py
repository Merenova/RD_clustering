"""Optimal configuration selection for RD clustering.

Selects the best (beta, gamma) based on silhouette scores and other metrics.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.metrics import silhouette_score

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compute_silhouette(
    data: np.ndarray,
    assignments: np.ndarray,
) -> float:
    """Compute silhouette score.
    
    Args:
        data: Data array [n_samples, n_features]
        assignments: Cluster assignments
        
    Returns:
        Silhouette score, or 0.0 if invalid
    """
    labels = np.array(assignments)
    unique_labels = np.unique(labels)
    
    # Need at least 2 clusters
    if len(unique_labels) < 2:
        return 0.0
    
    try:
        return silhouette_score(data, labels)
    except Exception:
        return 0.0


def compute_silhouette_both_spaces(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    assignments: np.ndarray,
) -> Tuple[float, float]:
    """Compute silhouette scores in both embedding and attribution spaces.
    
    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        assignments: Cluster assignments
        
    Returns:
        Tuple of (silhouette_e, silhouette_a)
    """
    sil_e = compute_silhouette(embeddings_e, assignments)
    sil_a = compute_silhouette(attributions_a, assignments)
    return sil_e, sil_a


def compute_harmonic_silhouette(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    assignments: np.ndarray,
) -> float:
    """Compute harmonic mean of silhouette scores.
    
    Silhouette scores are in [-1, 1], shifted to [0, 2] for valid harmonic mean.
    
    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        assignments: Cluster assignments
        
    Returns:
        Harmonic mean silhouette score (shifted back to [-1, 1])
    """
    sil_e, sil_a = compute_silhouette_both_spaces(embeddings_e, attributions_a, assignments)
    
    # Shift to [0, 2]
    sil_e_shifted = sil_e + 1.0
    sil_a_shifted = sil_a + 1.0
    
    if sil_e_shifted <= 0 or sil_a_shifted <= 0:
        return -1.0
    
    h_mean = 2 * sil_e_shifted * sil_a_shifted / (sil_e_shifted + sil_a_shifted)
    return h_mean - 1.0  # Shift back


def compute_intra_cluster_similarity(
    data: np.ndarray,
    assignments: np.ndarray,
) -> float:
    """Compute mean intra-cluster cosine similarity.
    
    Args:
        data: Data array [n_samples, n_features]
        assignments: Cluster assignments
        
    Returns:
        Mean intra-cluster similarity
    """
    unique_clusters = np.unique(assignments)
    
    if len(unique_clusters) < 2:
        return 0.0
    
    # Normalize data
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    data_normed = data / norms
    
    similarities = []
    for c in unique_clusters:
        mask = assignments == c
        cluster_data = data_normed[mask]
        
        if len(cluster_data) < 2:
            continue
        
        # Compute pairwise similarities within cluster
        sim_matrix = cluster_data @ cluster_data.T
        n = len(cluster_data)
        triu_indices = np.triu_indices(n, k=1)
        cluster_sims = sim_matrix[triu_indices]
        
        if len(cluster_sims) > 0:
            similarities.append(np.mean(cluster_sims))
    
    return np.mean(similarities) if similarities else 0.0


def compute_cluster_metrics(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    assignments: np.ndarray,
    labels: np.ndarray = None,
) -> Dict[str, float]:
    """Compute comprehensive clustering metrics.
    
    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        assignments: Cluster assignments
        labels: Optional ground truth labels for purity
        
    Returns:
        Dict of metrics
    """
    unique_clusters = np.unique(assignments)
    n_clusters = len(unique_clusters)
    
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_e": compute_silhouette(embeddings_e, assignments),
        "silhouette_a": compute_silhouette(attributions_a, assignments),
        "harmonic_silhouette": compute_harmonic_silhouette(embeddings_e, attributions_a, assignments),
        "semantic_coherence": compute_intra_cluster_similarity(embeddings_e, assignments),
        "attribution_consistency": compute_intra_cluster_similarity(attributions_a, assignments),
    }
    
    # Compute purity if labels provided
    if labels is not None:
        metrics["label_purity"] = compute_label_purity(assignments, labels)
        metrics["label_variance"] = compute_intra_cluster_label_variance(assignments, labels)
    
    return metrics


def compute_label_purity(
    assignments: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute cluster purity with respect to labels.
    
    Args:
        assignments: Cluster assignments
        labels: Ground truth labels
        
    Returns:
        Purity score (0-1)
    """
    unique_clusters = np.unique(assignments)
    total = len(labels)
    
    if total == 0:
        return 0.0
    
    correct = 0
    for c in unique_clusters:
        mask = assignments == c
        cluster_labels = labels[mask]
        
        if len(cluster_labels) == 0:
            continue
        
        # Count most common label
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        correct += counts.max()
    
    return correct / total


def compute_intra_cluster_label_variance(
    assignments: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute mean intra-cluster label variance.
    
    Higher variance = clusters contain more label diversity.
    
    Args:
        assignments: Cluster assignments
        labels: Ground truth labels (binary)
        
    Returns:
        Mean label variance
    """
    unique_clusters = np.unique(assignments)
    variances = []
    
    for c in unique_clusters:
        mask = assignments == c
        cluster_labels = labels[mask].astype(float)
        
        if len(cluster_labels) < 2:
            continue
        
        # For binary labels, variance = p(1-p)
        p = cluster_labels.mean()
        var = p * (1 - p)
        variances.append(var)
    
    return np.mean(variances) if variances else 0.0


def select_optimal_config(
    sweep_results: Dict[Tuple[float, float], Dict],
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    selection_metric: str = "harmonic_silhouette",
    min_clusters: int = 2,
    max_clusters: int = None,
) -> Tuple[float, float, Dict, float]:
    """Select optimal (beta, gamma) configuration.

    Args:
        sweep_results: Dict mapping (beta, gamma) -> clustering result
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        selection_metric: Metric to optimize ("harmonic_silhouette", "combined", etc.)
        min_clusters: Minimum acceptable number of clusters
        max_clusters: Maximum acceptable number of clusters (if None, uses K_max from results - 1)

    Returns:
        Tuple of (best_beta, best_gamma, best_result, best_score)
    """
    best_score = -float('inf')
    best_config = None
    best_result = None

    for (beta, gamma), result in sweep_results.items():
        assignments = result["assignments"]
        n_clusters = len(np.unique(assignments))

        # Get K_max from result if available, else use default
        K_max = result.get("K_max", 20)
        effective_max = max_clusters if max_clusters is not None else (K_max - 1)

        # Filter by cluster count (skip K=1 and K=K_max)
        if n_clusters < min_clusters or n_clusters >= K_max or n_clusters > effective_max:
            continue
        
        # Compute metrics
        metrics = compute_cluster_metrics(embeddings_e, attributions_a, assignments)
        
        # Select score based on metric
        if selection_metric == "harmonic_silhouette":
            score = metrics["harmonic_silhouette"]
        elif selection_metric == "combined":
            score = (
                0.4 * metrics["harmonic_silhouette"] +
                0.3 * metrics["semantic_coherence"] +
                0.3 * metrics["attribution_consistency"]
            )
        elif selection_metric == "silhouette_e":
            score = metrics["silhouette_e"]
        elif selection_metric == "silhouette_a":
            score = metrics["silhouette_a"]
        else:
            score = metrics.get(selection_metric, 0.0)
        
        if score > best_score:
            best_score = score
            best_config = (beta, gamma)
            best_result = result
            best_result["metrics"] = metrics
    
    if best_config is None:
        # Return first valid result if none meet criteria
        for (beta, gamma), result in sweep_results.items():
            return beta, gamma, result, 0.0
    
    return best_config[0], best_config[1], best_result, best_score


def summarize_sweep_results(
    sweep_results: Dict[Tuple[float, float], Dict],
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    labels: np.ndarray = None,
) -> List[Dict[str, Any]]:
    """Summarize sweep results for all configurations.
    
    Args:
        sweep_results: Dict mapping (beta, gamma) -> clustering result
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        labels: Optional ground truth labels
        
    Returns:
        List of summary dicts sorted by harmonic silhouette
    """
    summaries = []
    
    for (beta, gamma), result in sweep_results.items():
        assignments = result["assignments"]
        metrics = compute_cluster_metrics(embeddings_e, attributions_a, assignments, labels)
        
        summary = {
            "beta": beta,
            "gamma": gamma,
            "n_iterations": result.get("n_iterations", 0),
            **metrics,
        }
        summaries.append(summary)
    
    # Sort by harmonic silhouette
    summaries.sort(key=lambda x: x["harmonic_silhouette"], reverse=True)
    
    return summaries

