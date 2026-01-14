#!/usr/bin/env python3
"""Compare PCA vs K-means++ initialization for split operation.

Compares:
1. Final distortion after 2-means refinement
2. Convergence speed
3. Cluster quality (how well they separate)
"""

import numpy as np
from typing import Tuple, Dict, List
import time

from rd_objective import (
    compute_mse_distance_to_center,
    compute_component_variance,
)
from adaptive_control import pca_split_initialization


def kmeans_pp_initialization(
    e_c: np.ndarray,
    a_c: np.ndarray,
    P_c: np.ndarray,
    beta_e: float,
    beta_a: float,
    seed: int = 42,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Initialize 2-means split using k-means++ in weighted combined space.

    K-means++ selects centroids with probability proportional to squared
    distance from existing centroids, promoting well-separated initialization.

    Args:
        e_c: Semantic embeddings for component (n_c, d_e)
        a_c: Attribution embeddings for component (n_c, d_a)
        P_c: Probability weights (n_c,)
        beta_e: Semantic weight
        beta_a: Attribution weight
        seed: Random seed for reproducibility

    Returns:
        Tuple of (labels, (mu1_e, mu1_a, mu2_e, mu2_a))
        - labels: Binary array (n_c,) with 0/1 cluster assignments
        - centroids: Initial centroids for both clusters
    """
    rng = np.random.default_rng(seed)
    n_c = len(P_c)
    d_e = e_c.shape[1]
    d_a = a_c.shape[1]

    if n_c < 2:
        return np.zeros(n_c, dtype=int), (e_c[0], a_c[0], e_c[0], a_c[0])

    W_c = np.sum(P_c)
    if W_c == 0:
        return np.zeros(n_c, dtype=int), (e_c[0], a_c[0], e_c[0], a_c[0])

    # Normalize weights for sampling
    P_norm = P_c / W_c

    # Step 1: Select first centroid with probability proportional to P_c
    idx1 = rng.choice(n_c, p=P_norm)
    mu1_e = e_c[idx1].copy()
    mu1_a = a_c[idx1].copy()

    # Step 2: Compute squared distances to first centroid (MSE-normalized)
    dist_sq_e = np.sum((e_c - mu1_e) ** 2, axis=1) / d_e
    dist_sq_a = np.sum((a_c - mu1_a) ** 2, axis=1) / d_a
    dist_sq = beta_e * dist_sq_e + beta_a * dist_sq_a

    # Step 3: Select second centroid with probability ∝ P_c * dist_sq
    # (weighted k-means++ variant)
    weighted_dist_sq = P_c * dist_sq
    sum_weighted = np.sum(weighted_dist_sq)

    if sum_weighted == 0:
        # All points identical to first centroid, pick arbitrary second
        idx2 = (idx1 + 1) % n_c
    else:
        prob = weighted_dist_sq / sum_weighted
        idx2 = rng.choice(n_c, p=prob)

    mu2_e = e_c[idx2].copy()
    mu2_a = a_c[idx2].copy()

    # Step 4: Initial assignment based on R-D weighted distance
    dist_to_1_e = np.sum((e_c - mu1_e) ** 2, axis=1) / d_e
    dist_to_1_a = np.sum((a_c - mu1_a) ** 2, axis=1) / d_a
    dist_to_1 = beta_e * dist_to_1_e + beta_a * dist_to_1_a

    dist_to_2_e = np.sum((e_c - mu2_e) ** 2, axis=1) / d_e
    dist_to_2_a = np.sum((a_c - mu2_a) ** 2, axis=1) / d_a
    dist_to_2 = beta_e * dist_to_2_e + beta_a * dist_to_2_a

    labels = (dist_to_2 < dist_to_1).astype(np.int32)

    return labels, (mu1_e, mu1_a, mu2_e, mu2_a)


def refine_2means(
    e_c: np.ndarray,
    a_c: np.ndarray,
    P_c: np.ndarray,
    labels: np.ndarray,
    beta_e: float,
    beta_a: float,
    max_iters: int = 10,
) -> Tuple[np.ndarray, float, int]:
    """Refine 2-means clustering with Lloyd's algorithm.

    Returns:
        Tuple of (final_labels, final_distortion, n_iters)
    """
    labels = labels.copy()
    n_iters = 0

    for i in range(max_iters):
        c1_mask = labels == 0
        c2_mask = labels == 1

        if not np.any(c1_mask) or not np.any(c2_mask):
            break

        W1 = np.sum(P_c[c1_mask])
        W2 = np.sum(P_c[c2_mask])

        if W1 == 0 or W2 == 0:
            break

        # Compute centroids
        mu1_e = np.sum(P_c[c1_mask, None] * e_c[c1_mask], axis=0) / W1
        mu1_a = np.sum(P_c[c1_mask, None] * a_c[c1_mask], axis=0) / W1
        mu2_e = np.sum(P_c[c2_mask, None] * e_c[c2_mask], axis=0) / W2
        mu2_a = np.sum(P_c[c2_mask, None] * a_c[c2_mask], axis=0) / W2

        # Compute distances
        d1 = (beta_e * compute_mse_distance_to_center(e_c, mu1_e) +
              beta_a * compute_mse_distance_to_center(a_c, mu1_a))
        d2 = (beta_e * compute_mse_distance_to_center(e_c, mu2_e) +
              beta_a * compute_mse_distance_to_center(a_c, mu2_a))

        new_labels = (d2 < d1).astype(np.int32)
        n_iters = i + 1

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

    # Compute final distortion
    c1_mask = labels == 0
    c2_mask = labels == 1
    W1 = np.sum(P_c[c1_mask]) if np.any(c1_mask) else 0
    W2 = np.sum(P_c[c2_mask]) if np.any(c2_mask) else 0
    W_total = W1 + W2

    distortion = 0.0
    if W1 > 0 and np.any(c1_mask):
        mu1_e = np.sum(P_c[c1_mask, None] * e_c[c1_mask], axis=0) / W1
        mu1_a = np.sum(P_c[c1_mask, None] * a_c[c1_mask], axis=0) / W1
        var1_e = compute_component_variance(e_c[c1_mask], mu1_e, P_c[c1_mask], W1)
        var1_a = compute_component_variance(a_c[c1_mask], mu1_a, P_c[c1_mask], W1)
        distortion += (W1 / W_total) * (beta_e * var1_e + beta_a * var1_a)

    if W2 > 0 and np.any(c2_mask):
        mu2_e = np.sum(P_c[c2_mask, None] * e_c[c2_mask], axis=0) / W2
        mu2_a = np.sum(P_c[c2_mask, None] * a_c[c2_mask], axis=0) / W2
        var2_e = compute_component_variance(e_c[c2_mask], mu2_e, P_c[c2_mask], W2)
        var2_a = compute_component_variance(a_c[c2_mask], mu2_a, P_c[c2_mask], W2)
        distortion += (W2 / W_total) * (beta_e * var2_e + beta_a * var2_a)

    return labels, distortion, n_iters


def generate_test_data(
    n_samples: int,
    d_e: int,
    d_a: int,
    separation: float = 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data with two well-separated clusters.

    Returns:
        Tuple of (e_c, a_c, P_c, true_labels)
    """
    rng = np.random.default_rng(seed)

    n1 = n_samples // 2
    n2 = n_samples - n1

    # Cluster 1: centered at +separation
    e1 = rng.standard_normal((n1, d_e)) + separation
    a1 = rng.standard_normal((n1, d_a)) + separation

    # Cluster 2: centered at -separation
    e2 = rng.standard_normal((n2, d_e)) - separation
    a2 = rng.standard_normal((n2, d_a)) - separation

    e_c = np.vstack([e1, e2]).astype(np.float32)
    a_c = np.vstack([a1, a2]).astype(np.float32)

    # Random weights
    P_c = rng.random(n_samples).astype(np.float32)
    P_c = P_c / np.sum(P_c)

    true_labels = np.array([0] * n1 + [1] * n2, dtype=np.int32)

    return e_c, a_c, P_c, true_labels


def compute_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Compute clustering accuracy (handles label permutation)."""
    # Try both label assignments
    acc1 = np.mean(pred_labels == true_labels)
    acc2 = np.mean(pred_labels == (1 - true_labels))
    return max(acc1, acc2)


def compute_label_agreement(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Compute agreement between two label assignments (handles permutation)."""
    acc1 = np.mean(labels1 == labels2)
    acc2 = np.mean(labels1 == (1 - labels2))
    return max(acc1, acc2)


def compute_adjusted_rand_index(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Compute Adjusted Rand Index between two clusterings.

    ARI = 1.0 means perfect agreement, 0.0 means random, negative means worse than random.
    """
    n = len(labels1)

    # Contingency table
    n00 = np.sum((labels1 == 0) & (labels2 == 0))
    n01 = np.sum((labels1 == 0) & (labels2 == 1))
    n10 = np.sum((labels1 == 1) & (labels2 == 0))
    n11 = np.sum((labels1 == 1) & (labels2 == 1))

    # Row and column sums
    a0 = n00 + n01
    a1 = n10 + n11
    b0 = n00 + n10
    b1 = n01 + n11

    # Compute index
    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_nij = comb2(n00) + comb2(n01) + comb2(n10) + comb2(n11)
    sum_comb_ai = comb2(a0) + comb2(a1)
    sum_comb_bj = comb2(b0) + comb2(b1)
    comb_n = comb2(n)

    if comb_n == 0:
        return 1.0

    expected = sum_comb_ai * sum_comb_bj / comb_n
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)

    if max_index == expected:
        return 1.0

    ari = (sum_comb_nij - expected) / (max_index - expected)
    return ari


def run_comparison(
    n_samples: int = 200,
    d_e: int = 64,
    d_a: int = 128,
    beta_e: float = 1.0,
    beta_a: float = 1.0,
    separations: List[float] = [0.5, 1.0, 2.0, 3.0, 5.0],
    n_trials: int = 10,
):
    """Run comparison between PCA and K-means++ initialization."""

    print("=" * 80)
    print("Comparing PCA vs K-means++ Split Initialization")
    print("=" * 80)
    print(f"n_samples={n_samples}, d_e={d_e}, d_a={d_a}")
    print(f"beta_e={beta_e}, beta_a={beta_a}")
    print(f"n_trials={n_trials}")
    print()

    results = []

    for sep in separations:
        pca_distortions = []
        kpp_distortions = []
        pca_accuracies = []
        kpp_accuracies = []
        pca_iters = []
        kpp_iters = []
        pca_times = []
        kpp_times = []

        for trial in range(n_trials):
            seed = 42 + trial
            e_c, a_c, P_c, true_labels = generate_test_data(
                n_samples, d_e, d_a, separation=sep, seed=seed
            )

            # PCA initialization
            t0 = time.time()
            pca_labels, _ = pca_split_initialization(e_c, a_c, P_c, beta_e, beta_a)
            pca_labels, pca_dist, pca_iter = refine_2means(
                e_c, a_c, P_c, pca_labels, beta_e, beta_a
            )
            pca_time = time.time() - t0

            # K-means++ initialization
            t0 = time.time()
            kpp_labels, _ = kmeans_pp_initialization(e_c, a_c, P_c, beta_e, beta_a, seed=seed)
            kpp_labels, kpp_dist, kpp_iter = refine_2means(
                e_c, a_c, P_c, kpp_labels, beta_e, beta_a
            )
            kpp_time = time.time() - t0

            pca_distortions.append(pca_dist)
            kpp_distortions.append(kpp_dist)
            pca_accuracies.append(compute_accuracy(pca_labels, true_labels))
            kpp_accuracies.append(compute_accuracy(kpp_labels, true_labels))
            pca_iters.append(pca_iter)
            kpp_iters.append(kpp_iter)
            pca_times.append(pca_time)
            kpp_times.append(kpp_time)

        results.append({
            'separation': sep,
            'pca_dist_mean': np.mean(pca_distortions),
            'pca_dist_std': np.std(pca_distortions),
            'kpp_dist_mean': np.mean(kpp_distortions),
            'kpp_dist_std': np.std(kpp_distortions),
            'pca_acc_mean': np.mean(pca_accuracies),
            'kpp_acc_mean': np.mean(kpp_accuracies),
            'pca_iters_mean': np.mean(pca_iters),
            'kpp_iters_mean': np.mean(kpp_iters),
            'pca_time_mean': np.mean(pca_times),
            'kpp_time_mean': np.mean(kpp_times),
        })

    # Print results table
    print(f"{'Sep':<6} | {'PCA Dist':>12} | {'K++ Dist':>12} | {'PCA Acc':>8} | {'K++ Acc':>8} | {'PCA Iter':>8} | {'K++ Iter':>8} | {'Winner':>8}")
    print("-" * 95)

    for r in results:
        pca_better = r['pca_dist_mean'] < r['kpp_dist_mean']
        winner = "PCA" if pca_better else "K++"

        print(f"{r['separation']:<6.1f} | "
              f"{r['pca_dist_mean']:>8.4f}±{r['pca_dist_std']:.3f} | "
              f"{r['kpp_dist_mean']:>8.4f}±{r['kpp_dist_std']:.3f} | "
              f"{r['pca_acc_mean']:>8.2%} | "
              f"{r['kpp_acc_mean']:>8.2%} | "
              f"{r['pca_iters_mean']:>8.1f} | "
              f"{r['kpp_iters_mean']:>8.1f} | "
              f"{winner:>8}")

    print()
    print("Summary:")
    pca_wins = sum(1 for r in results if r['pca_dist_mean'] < r['kpp_dist_mean'])
    kpp_wins = len(results) - pca_wins
    print(f"  PCA wins: {pca_wins}/{len(results)}")
    print(f"  K++ wins: {kpp_wins}/{len(results)}")

    avg_pca_time = np.mean([r['pca_time_mean'] for r in results])
    avg_kpp_time = np.mean([r['kpp_time_mean'] for r in results])
    print(f"  Avg PCA time: {avg_pca_time*1000:.2f}ms")
    print(f"  Avg K++ time: {avg_kpp_time*1000:.2f}ms")

    return results


def run_elongated_cluster_test(
    n_samples: int = 200,
    d_e: int = 64,
    d_a: int = 128,
    beta_e: float = 1.0,
    beta_a: float = 1.0,
    n_trials: int = 10,
):
    """Test with elongated clusters where PCA should excel."""

    print("\n" + "=" * 80)
    print("Elongated Cluster Test (PCA should excel)")
    print("=" * 80)

    pca_distortions = []
    kpp_distortions = []

    for trial in range(n_trials):
        seed = 42 + trial
        rng = np.random.default_rng(seed)

        # Create elongated cluster along first dimension
        e_c = rng.standard_normal((n_samples, d_e)).astype(np.float32)
        a_c = rng.standard_normal((n_samples, d_a)).astype(np.float32)

        # Stretch along first dimension
        e_c[:, 0] *= 10
        a_c[:, 0] *= 10

        P_c = rng.random(n_samples).astype(np.float32)
        P_c = P_c / np.sum(P_c)

        # True split: along first dimension
        true_labels = (e_c[:, 0] > np.median(e_c[:, 0])).astype(np.int32)

        # PCA
        pca_labels, _ = pca_split_initialization(e_c, a_c, P_c, beta_e, beta_a)
        pca_labels, pca_dist, _ = refine_2means(e_c, a_c, P_c, pca_labels, beta_e, beta_a)

        # K-means++
        kpp_labels, _ = kmeans_pp_initialization(e_c, a_c, P_c, beta_e, beta_a, seed=seed)
        kpp_labels, kpp_dist, _ = refine_2means(e_c, a_c, P_c, kpp_labels, beta_e, beta_a)

        pca_distortions.append(pca_dist)
        kpp_distortions.append(kpp_dist)

    print(f"PCA distortion: {np.mean(pca_distortions):.4f} ± {np.std(pca_distortions):.4f}")
    print(f"K++ distortion: {np.mean(kpp_distortions):.4f} ± {np.std(kpp_distortions):.4f}")

    if np.mean(pca_distortions) < np.mean(kpp_distortions):
        print("Winner: PCA (as expected for elongated clusters)")
    else:
        print("Winner: K++ (unexpected)")


def run_spherical_cluster_test(
    n_samples: int = 200,
    d_e: int = 64,
    d_a: int = 128,
    beta_e: float = 1.0,
    beta_a: float = 1.0,
    n_trials: int = 10,
):
    """Test with spherical clusters where K-means++ might do well."""

    print("\n" + "=" * 80)
    print("Spherical Cluster Test")
    print("=" * 80)

    pca_distortions = []
    kpp_distortions = []

    for trial in range(n_trials):
        seed = 42 + trial
        rng = np.random.default_rng(seed)

        n1, n2 = n_samples // 2, n_samples - n_samples // 2

        # Two spherical clusters with random center directions
        dir_e = rng.standard_normal(d_e)
        dir_e = dir_e / np.linalg.norm(dir_e) * 3
        dir_a = rng.standard_normal(d_a)
        dir_a = dir_a / np.linalg.norm(dir_a) * 3

        e1 = rng.standard_normal((n1, d_e)).astype(np.float32) + dir_e
        e2 = rng.standard_normal((n2, d_e)).astype(np.float32) - dir_e
        a1 = rng.standard_normal((n1, d_a)).astype(np.float32) + dir_a
        a2 = rng.standard_normal((n2, d_a)).astype(np.float32) - dir_a

        e_c = np.vstack([e1, e2])
        a_c = np.vstack([a1, a2])

        P_c = rng.random(n_samples).astype(np.float32)
        P_c = P_c / np.sum(P_c)

        true_labels = np.array([0] * n1 + [1] * n2, dtype=np.int32)

        # PCA
        pca_labels, _ = pca_split_initialization(e_c, a_c, P_c, beta_e, beta_a)
        pca_labels, pca_dist, _ = refine_2means(e_c, a_c, P_c, pca_labels, beta_e, beta_a)

        # K-means++
        kpp_labels, _ = kmeans_pp_initialization(e_c, a_c, P_c, beta_e, beta_a, seed=seed)
        kpp_labels, kpp_dist, _ = refine_2means(e_c, a_c, P_c, kpp_labels, beta_e, beta_a)

        pca_distortions.append(pca_dist)
        kpp_distortions.append(kpp_dist)

    print(f"PCA distortion: {np.mean(pca_distortions):.4f} ± {np.std(pca_distortions):.4f}")
    print(f"K++ distortion: {np.mean(kpp_distortions):.4f} ± {np.std(kpp_distortions):.4f}")

    if np.mean(pca_distortions) < np.mean(kpp_distortions):
        print("Winner: PCA")
    else:
        print("Winner: K++")


def load_real_prefix_data(prefix_id: str):
    """Load actual data for a prefix from the pipeline results.

    Returns:
        Tuple of (e_c, a_c, P_c) or None if data not found
    """
    import sys
    from pathlib import Path
    import torch

    # Add parent directory to path
    parent_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(parent_dir))

    from utils.data_utils import load_json

    results_dir = parent_dir / "results"
    embeddings_dir = results_dir / "4_feature_extraction" / "embeddings"
    graphs_dir = results_dir / "3_attribution_graphs"
    samples_dir = results_dir / "2_branch_sampling"

    # Check files exist
    embeddings_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    graph_file = graphs_dir / f"{prefix_id}_graph.pt"
    branches_file = samples_dir / f"{prefix_id}_branches.json"

    if not all(f.exists() for f in [embeddings_file, graph_file, branches_file]):
        print(f"Missing files for {prefix_id}")
        return None

    # Load embeddings
    embeddings_e = np.load(embeddings_file)

    # Load branches data for probabilities
    branches_data = load_json(branches_file)
    path_probs = []
    first_tokens = []
    token_probs = {}

    for ft_data in branches_data["first_tokens"]:
        token_id = ft_data["token_id"]
        token_probs[token_id] = ft_data.get("first_token_probability", 1e-10)
        for cont in ft_data["continuations"]:
            path_probs.append(cont["probability"])
            first_tokens.append(token_id)

    path_probs = np.array(path_probs)
    first_tokens = np.array(first_tokens)

    # Use raw probabilities directly (consistent with cluster.py)
    P_c = path_probs.copy()

    # Load attribution graph
    circuit_tracer_path = parent_dir / "circuit-tracer"
    sys.path.insert(0, str(circuit_tracer_path))
    from circuit_tracer.graph import Graph

    graph = Graph.from_pt(graph_file)

    # Extract attribution embeddings
    n_features = len(graph.selected_features)
    n_error = graph.cfg.n_layers * graph.n_pos
    n_token = graph.n_pos
    n_attribution_nodes = n_features + n_error + n_token

    attributions_list = []
    for s_n in first_tokens:
        logit_tokens = graph.logit_tokens
        matches = (logit_tokens == s_n).nonzero(as_tuple=True)[0]

        if len(matches) == 0:
            attributions_list.append(np.zeros(n_attribution_nodes, dtype=np.float32))
            continue

        logit_idx = matches[0].item()
        logit_node_idx = n_features + n_error + n_token + logit_idx
        attribution_vector = graph.adjacency_matrix[logit_node_idx, :n_attribution_nodes]

        if isinstance(attribution_vector, torch.Tensor):
            attribution_vector = attribution_vector.cpu().numpy()

        attributions_list.append(attribution_vector.astype(np.float32))

    attributions_a = np.array(attributions_list)

    # Center attributions (as done in clustering)
    W_total = P_c.sum()
    H_0 = np.sum(P_c[:, None] * attributions_a, axis=0) / W_total
    attributions_a = attributions_a - H_0

    return embeddings_e.astype(np.float32), attributions_a.astype(np.float32), P_c.astype(np.float32)


def run_real_data_comparison(
    beta_e: float = 1.0,
    beta_a: float = 1.0,
    n_prefixes: int = 10,
    n_trials_per_prefix: int = 5,
):
    """Run comparison on actual data from the pipeline."""

    print("\n" + "=" * 80)
    print("Real Data Comparison (Pipeline Results)")
    print("=" * 80)

    from pathlib import Path

    results_dir = Path(__file__).resolve().parents[1] / "results" / "5_clustering"
    clustering_files = sorted(results_dir.glob("*_clustering.json"))[:n_prefixes]

    if not clustering_files:
        print("No clustering results found. Run the pipeline first.")
        return

    all_pca_distortions = []
    all_kpp_distortions = []
    all_pca_iters = []
    all_kpp_iters = []
    all_agreements = []
    all_aris = []
    pca_times = []
    kpp_times = []

    for clustering_file in clustering_files:
        prefix_id = clustering_file.stem.replace("_clustering", "")

        data = load_real_prefix_data(prefix_id)
        if data is None:
            continue

        e_c, a_c, P_c = data

        print(f"\n{prefix_id}: n={len(P_c)}, d_e={e_c.shape[1]}, d_a={a_c.shape[1]}")

        pca_distortions = []
        kpp_distortions = []
        pca_iters = []
        kpp_iters = []
        agreements = []
        aris = []

        for trial in range(n_trials_per_prefix):
            seed = 42 + trial

            # PCA initialization
            t0 = time.time()
            pca_labels, _ = pca_split_initialization(e_c, a_c, P_c, beta_e, beta_a)
            pca_labels, pca_dist, pca_iter = refine_2means(
                e_c, a_c, P_c, pca_labels, beta_e, beta_a
            )
            pca_time = time.time() - t0

            # K-means++ initialization
            t0 = time.time()
            kpp_labels, _ = kmeans_pp_initialization(e_c, a_c, P_c, beta_e, beta_a, seed=seed)
            kpp_labels, kpp_dist, kpp_iter = refine_2means(
                e_c, a_c, P_c, kpp_labels, beta_e, beta_a
            )
            kpp_time = time.time() - t0

            # Label similarity
            agreement = compute_label_agreement(pca_labels, kpp_labels)
            ari = compute_adjusted_rand_index(pca_labels, kpp_labels)

            pca_distortions.append(pca_dist)
            kpp_distortions.append(kpp_dist)
            pca_iters.append(pca_iter)
            kpp_iters.append(kpp_iter)
            agreements.append(agreement)
            aris.append(ari)
            pca_times.append(pca_time)
            kpp_times.append(kpp_time)

        pca_mean = np.mean(pca_distortions)
        kpp_mean = np.mean(kpp_distortions)
        winner = "PCA" if pca_mean < kpp_mean else "K++"
        diff_pct = 100 * (kpp_mean - pca_mean) / max(pca_mean, 1e-10)

        print(f"  PCA: dist={pca_mean:.4f}, iters={np.mean(pca_iters):.1f}")
        print(f"  K++: dist={kpp_mean:.4f}, iters={np.mean(kpp_iters):.1f}")
        print(f"  Label agreement: {np.mean(agreements):.1%}, ARI: {np.mean(aris):.3f}")
        print(f"  Winner: {winner} (diff={diff_pct:+.2f}%)")

        all_pca_distortions.extend(pca_distortions)
        all_kpp_distortions.extend(kpp_distortions)
        all_pca_iters.extend(pca_iters)
        all_kpp_iters.extend(kpp_iters)
        all_agreements.extend(agreements)
        all_aris.extend(aris)

    print("\n" + "-" * 80)
    print("Overall Summary (across all prefixes and trials):")
    print("-" * 80)
    print(f"PCA: dist={np.mean(all_pca_distortions):.4f} ± {np.std(all_pca_distortions):.4f}, "
          f"iters={np.mean(all_pca_iters):.1f}")
    print(f"K++: dist={np.mean(all_kpp_distortions):.4f} ± {np.std(all_kpp_distortions):.4f}, "
          f"iters={np.mean(all_kpp_iters):.1f}")

    print(f"\nLabel similarity (PCA vs K++):")
    print(f"  Agreement: {np.mean(all_agreements):.1%} ± {np.std(all_agreements):.1%}")
    print(f"  ARI: {np.mean(all_aris):.3f} ± {np.std(all_aris):.3f}")

    pca_wins = sum(1 for p, k in zip(all_pca_distortions, all_kpp_distortions) if p < k)
    kpp_wins = len(all_pca_distortions) - pca_wins
    print(f"\nTrial-level wins: PCA={pca_wins}, K++={kpp_wins}")

    print(f"Avg time: PCA={np.mean(pca_times)*1000:.2f}ms, K++={np.mean(kpp_times)*1000:.2f}ms")

    overall_winner = "PCA" if np.mean(all_pca_distortions) < np.mean(all_kpp_distortions) else "K++"
    print(f"\nOverall winner: {overall_winner}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare PCA vs K-means++ split initialization")
    parser.add_argument("--real-data", action="store_true", help="Use real pipeline data")
    parser.add_argument("--n-prefixes", type=int, default=10, help="Number of prefixes to test")
    parser.add_argument("--synthetic-only", action="store_true", help="Only run synthetic tests")
    args = parser.parse_args()

    if args.real_data:
        run_real_data_comparison(n_prefixes=args.n_prefixes)
    elif args.synthetic_only:
        run_comparison()
        run_elongated_cluster_test()
        run_spherical_cluster_test()
    else:
        # Run all comparisons
        run_comparison()
        run_elongated_cluster_test()
        run_spherical_cluster_test()
        run_real_data_comparison(n_prefixes=args.n_prefixes)
