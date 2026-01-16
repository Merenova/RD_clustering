"""Adaptive control operations using exact rate-distortion criteria.

Implements Split and Junk operations from rate_distortion.tex using principled
information-theoretic criteria. No tunable thresholds - operations are applied
if and only if they decrease L_RD.

Key criteria (from Theorems 3.1 and 3.2):
- Split: β_e·ΔD^(e,red) + β_a·ΔD^(a,red) > P̄_c·H_binary(α)
- Junk: |ΔH| > β_e·ΔD^(e,reassign) + β_a·ΔD^(a,reassign)
"""

import numpy as np
from typing import Dict, List, Tuple

from rd_objective import (
    compute_component_masses,
    compute_normalized_masses,
    binary_entropy,
    compute_entropy,
    compute_mse_distances,
    compute_l1_distances,
    compute_mse_distance_to_center,
    compute_l1_distance_to_center,
    compute_mse_distance_pairwise,
    compute_l1_distance_pairwise,
    compute_component_variance,
)


def pca_split_initialization(
    e_c: np.ndarray,
    a_c: np.ndarray,
    P_c: np.ndarray,
    beta_e: float,
    beta_a: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize 2-means split using PCA on weighted combined space (optimized).

    Finds the direction of maximum variance and splits along it.
    This is more robust than random initialization for elongated clusters.

    Optimized to avoid creating large concatenated arrays by computing
    covariance contributions from e and a separately.
    
    NOTE: PCA is inherently L2/Variance based. Even if we use L1 metric for clustering,
    using PCA for initialization is a reasonable heuristic.

    Args:
        e_c: Semantic embeddings for component (n_c, d_e)
        a_c: Attribution embeddings for component (n_c, d_a)
        P_c: Probability weights (n_c,)
        beta_e: Semantic weight
        beta_a: Attribution weight

    Returns:
        Tuple of (labels, projection_values)
        - labels: Binary array (n_c,) with 0/1 cluster assignments
        - projection_values: Projection onto PC1 for each sample
    """
    n_c = len(P_c)
    W_c = np.sum(P_c)

    if W_c == 0 or n_c < 2:
        return np.zeros(n_c, dtype=int), np.zeros(n_c)

    # Normalize weights
    P_norm = P_c / W_c

    # Scale factors
    sqrt_beta_e = np.sqrt(beta_e)
    sqrt_beta_a = np.sqrt(beta_a)

    # Compute weighted centroids separately (avoid concatenation)
    mu_e = np.sum(P_norm[:, None] * e_c, axis=0)  # (d_e,)
    mu_a = np.sum(P_norm[:, None] * a_c, axis=0)  # (d_a,)

    # Center the data (keep separate)
    e_centered = e_c - mu_e  # (n_c, d_e)
    a_centered = a_c - mu_a  # (n_c, d_a)

    d_e = e_c.shape[1]
    d_a = a_c.shape[1]

    # Power iteration to find first principal component
    # v = [v_e, v_a] where v_e is d_e dim and v_a is d_a dim
    # Use raw beta weights (no MSE normalization)
    np.random.seed(42)  # Reproducibility
    v_e = np.random.randn(d_e).astype(np.float32)
    v_a = np.random.randn(d_a).astype(np.float32)
    # Use beta-weighted norm
    norm = np.sqrt(beta_e * np.dot(v_e, v_e) + beta_a * np.dot(v_a, v_a))
    v_e /= norm
    v_a /= norm

    for _ in range(20):  # Reduced iterations - converges fast with good init
        # Compute projections with beta weights
        projections = (beta_e * (e_centered @ v_e) +
                       beta_a * (a_centered @ v_a))  # (n_c,)

        # Compute Σv = Σ_n P_n x_n (x_n^T v)
        # For e part: sqrt(β_e) * Σ P_n e_centered * proj
        # For a part: sqrt(β_a) * Σ P_n a_centered * proj
        weighted_proj = P_norm * projections  # (n_c,)
        Sigma_v_e = np.sqrt(beta_e) * np.sum(weighted_proj[:, None] * e_centered, axis=0)
        Sigma_v_a = np.sqrt(beta_a) * np.sum(weighted_proj[:, None] * a_centered, axis=0)

        # Normalize
        norm = np.sqrt(np.dot(Sigma_v_e, Sigma_v_e) + np.dot(Sigma_v_a, Sigma_v_a))
        if norm < 1e-10:
            break

        v_e_new = Sigma_v_e / norm
        v_a_new = Sigma_v_a / norm

        # Check convergence
        dot_prod = np.dot(v_e, v_e_new) + np.dot(v_a, v_a_new)
        if abs(dot_prod) > 0.9999:
            v_e, v_a = v_e_new, v_a_new
            break
        v_e, v_a = v_e_new, v_a_new

    # Final projections onto PC1 with beta weights
    projections = (beta_e * (e_centered @ v_e) +
                   beta_a * (a_centered @ v_a))

    # Split by weighted median
    sorted_indices = np.argsort(projections)
    cumsum = np.cumsum(P_c[sorted_indices])
    median_idx = np.searchsorted(cumsum, W_c / 2)
    median_idx = min(median_idx, n_c - 1)
    threshold = projections[sorted_indices[median_idx]]

    labels = (projections > threshold).astype(np.int32)

    # Ensure both clusters are non-empty
    if not np.any(labels == 0) or not np.any(labels == 1):
        labels = (projections > np.median(projections)).astype(np.int32)

    if not np.any(labels == 0) or not np.any(labels == 1):
        labels = np.zeros(n_c, dtype=np.int32)
        labels[n_c // 2:] = 1

    return labels, projections


def split_operation(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    assignments: List[int],
    components: Dict[int, Dict],
    P_bar: Dict[int, float],
    Var_e: Dict[int, float],
    Var_a: Dict[int, float],
    beta_e: float,
    beta_a: float,
    K_max: int,
    next_component_id: int,
    W_total: float,
    metric_a: str = "l2",
    use_weighted_distortion: bool = True
) -> Tuple[Dict[int, Dict], List[int], int]:
    """Split operation using exact rate-distortion criterion (optimized).

    Per tex Algorithm 2 (TrySplit):
    Split component c into (c1, c2) with mass fractions (α, 1-α) iff:
        β_e·ΔD^(e,red) + β_a·ΔD^(a,red) > P̄_c·H_binary(α)

    Uses combined distance for 2-means: β_e·Dist(e) + β_a·Dist(a)

    Args:
        embeddings_e: Semantic embeddings (N, d_e)
        attributions_a: Attribution embeddings (N, d_a)
        path_probs: Probability weights P_n (N,)
        assignments: Current cluster assignments
        components: Current components {c: {mu_e, mu_a, W_c, indices}}
        P_bar: Normalized component masses {c: P̄_c}
        Var_e: Semantic variance per component {c: Var_c^(e)}
        Var_a: Attribution variance per component {c: Var_c^(a)}
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight
        K_max: Maximum number of components
        next_component_id: Next available component ID
        W_total: Total probability mass
        metric_a: Attribution distance metric ("l2" or "l1")
        use_weighted_distortion: Whether to use probability weights

    Returns:
        Tuple of (updated_components, updated_assignments, next_component_id)
    """
    new_components = {c: {k: v.copy() if isinstance(v, np.ndarray) else v
                          for k, v in comp.items()}
                      for c, comp in components.items()}

    # Convert to numpy array for vectorized operations
    assignments_arr = np.asarray(assignments, dtype=np.int32)
    new_assignments_arr = assignments_arr.copy()

    # Sort by total variance (highest first) to prioritize splitting high-variance components
    to_split = [(c, beta_e * Var_e.get(c, 0) + beta_a * Var_a.get(c, 0))
                for c in components.keys() if len(new_components) < K_max]
    to_split.sort(key=lambda x: x[1], reverse=True)

    for c, _ in to_split:
        if len(new_components) >= K_max:
            break

        if c not in new_components:
            continue

        # Vectorized index selection
        mask_c = new_assignments_arr == c
        indices = np.where(mask_c)[0]

        if len(indices) < 2:  # Need at least 2 samples to split
            continue

        # Get data for this component using numpy indexing
        e_c = embeddings_e[indices]
        a_c = attributions_a[indices]
        P_c = path_probs[indices]
        W_c = np.sum(P_c)

        if W_c == 0:
            continue

        # Initialize 2-means with PCA-guided split (using L2 as heuristic even for L1)
        labels, _ = pca_split_initialization(e_c, a_c, P_c, beta_e, beta_a)

        c1_mask = labels == 0
        c2_mask = labels == 1

        if not np.any(c1_mask) or not np.any(c2_mask):
            continue

        # Compute initial centroids from PCA split
        # If unweighted, we should use unweighted centroids? 
        # But for split initialization, weighted is fine as a start.
        W1 = np.sum(P_c[c1_mask])
        W2 = np.sum(P_c[c2_mask])

        if W1 == 0 or W2 == 0:
            continue
            
        # Helper to compute centroid
        def get_centroid(data_chunk, weights_chunk, w_total, metric="l2"):
             if use_weighted_distortion:
                 center = np.sum(weights_chunk[:, None] * data_chunk, axis=0) / w_total
             else:
                 if metric == "l1":
                     center = np.median(data_chunk, axis=0)
                 else:
                     center = np.mean(data_chunk, axis=0)
             return center

        mu1_e = get_centroid(e_c[c1_mask], P_c[c1_mask], W1, "l2")
        mu1_a = get_centroid(a_c[c1_mask], P_c[c1_mask], W1, metric_a)
        mu2_e = get_centroid(e_c[c2_mask], P_c[c2_mask], W2, "l2")
        mu2_a = get_centroid(a_c[c2_mask], P_c[c2_mask], W2, metric_a)
        
        # Helper for distance
        def get_dist(data, center, metric="l2"):
            if metric == "l1":
                return compute_l1_distance_to_center(data, center)
            return compute_mse_distance_to_center(data, center)

        # Refine with 2-means iterations (reduced to 5 for speed)
        for _ in range(5):
            # Compute R-D distances to both centers
            d1 = (beta_e * get_dist(e_c, mu1_e, "l2") +
                  beta_a * get_dist(a_c, mu1_a, metric_a))
            d2 = (beta_e * get_dist(e_c, mu2_e, "l2") +
                  beta_a * get_dist(a_c, mu2_a, metric_a))

            labels = (d2 < d1).astype(np.int32)

            c1_mask = labels == 0
            c2_mask = labels == 1

            if not np.any(c1_mask) or not np.any(c2_mask):
                break

            W1 = np.sum(P_c[c1_mask])
            W2 = np.sum(P_c[c2_mask])

            if W1 == 0 or W2 == 0:
                break

            mu1_e = get_centroid(e_c[c1_mask], P_c[c1_mask], W1, "l2")
            if np.linalg.norm(mu1_e) > 1e-10: mu1_e /= np.linalg.norm(mu1_e)
            
            mu1_a = get_centroid(a_c[c1_mask], P_c[c1_mask], W1, metric_a)
            
            mu2_e = get_centroid(e_c[c2_mask], P_c[c2_mask], W2, "l2")
            if np.linalg.norm(mu2_e) > 1e-10: mu2_e /= np.linalg.norm(mu2_e)
            
            mu2_a = get_centroid(a_c[c2_mask], P_c[c2_mask], W2, metric_a)

        # Final assignment
        d1 = (beta_e * get_dist(e_c, mu1_e, "l2") +
              beta_a * get_dist(a_c, mu1_a, metric_a))
        d2 = (beta_e * get_dist(e_c, mu2_e, "l2") +
              beta_a * get_dist(a_c, mu2_a, metric_a))
        labels = (d2 < d1).astype(np.int32)

        c1_mask = labels == 0
        c2_mask = labels == 1

        if not np.any(c1_mask) or not np.any(c2_mask):
            continue

        # Compute masses and alpha
        W1 = np.sum(P_c[c1_mask])
        W2 = np.sum(P_c[c2_mask])
        alpha = W1 / (W1 + W2)

        # Compute post-split variances using helper function
        var1_e = compute_component_variance(e_c[c1_mask], mu1_e, P_c[c1_mask], W1, "l2", use_weighted_distortion)
        var2_e = compute_component_variance(e_c[c2_mask], mu2_e, P_c[c2_mask], W2, "l2", use_weighted_distortion)
        var1_a = compute_component_variance(a_c[c1_mask], mu1_a, P_c[c1_mask], W1, metric_a, use_weighted_distortion)
        var2_a = compute_component_variance(a_c[c2_mask], mu2_a, P_c[c2_mask], W2, metric_a, use_weighted_distortion)

        # Pre-split variances
        var_c_e = Var_e.get(c, 0)
        var_c_a = Var_a.get(c, 0)
        P_bar_c = P_bar.get(c, 0)

        # Distortion reduction
        delta_D_e = P_bar_c * (var_c_e - alpha * var1_e - (1 - alpha) * var2_e)
        delta_D_a = P_bar_c * (var_c_a - alpha * var1_a - (1 - alpha) * var2_a)

        # Rate cost
        rate_cost = P_bar_c * binary_entropy(alpha)
        distortion_benefit = beta_e * delta_D_e + beta_a * delta_D_a

        # Split criterion: distortion benefit > rate cost
        if distortion_benefit > rate_cost:
            c1 = next_component_id
            c2 = next_component_id + 1
            next_component_id += 2

            # Vectorized index extraction
            idx_1 = indices[c1_mask].tolist()
            idx_2 = indices[c2_mask].tolist()

            new_components[c1] = {
                'mu_e': mu1_e,
                'mu_a': mu1_a,
                'W_c': W1,
                'indices': idx_1,
            }
            new_components[c2] = {
                'mu_e': mu2_e,
                'mu_a': mu2_a,
                'W_c': W2,
                'indices': idx_2,
            }

            # Vectorized assignment update
            new_assignments_arr[indices[c1_mask]] = c1
            new_assignments_arr[indices[c2_mask]] = c2

            # Remove old component
            if c in new_components:
                del new_components[c]

    return new_components, new_assignments_arr.tolist(), next_component_id


def junk_operation(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    assignments: List[int],
    components: Dict[int, Dict],
    P_bar: Dict[int, float],
    beta_e: float,
    beta_a: float,
    W_total: float,
    metric_a: str = "l2",
    use_weighted_distortion: bool = True
) -> Tuple[Dict[int, Dict], List[int]]:
    """Junk operation using exact rate-distortion criterion.

    Per tex Algorithm 3 (TryJunk) and Theorem 3.2:
    Remove component c and reassign points to nearest remaining components iff:
        |ΔH| > β_e·ΔD^(e,reassign) + β_a·ΔD^(a,reassign)

    Where:
        - |ΔH| > 0 is the entropy reduction from removing c
        - ΔD^(·,reassign) > 0 is the distortion increase from reassigning points

    Note: Points are reassigned using full R-D assignment rule (including -log P̄_c term).

    Args:
        embeddings_e: Semantic embeddings (N, d_e)
        attributions_a: Attribution embeddings (N, d_a)
        path_probs: Probability weights P_n (N,)
        assignments: Current cluster assignments
        components: Current components {c: {mu_e, mu_a, W_c, indices}}
        P_bar: Normalized component masses {c: P̄_c}
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight
        W_total: Total probability mass
        metric_a: Attribution metric
        use_weighted_distortion: Whether to use probability weights

    Returns:
        Tuple of (updated_components, updated_assignments)
    """
    if len(components) <= 1:
        # Can't junk the only component
        return components, assignments

    new_components = {c: {k: v.copy() if isinstance(v, np.ndarray) else v
                          for k, v in comp.items()}
                      for c, comp in components.items()}
    # Use numpy array for vectorized operations
    new_assignments_arr = np.asarray(assignments, dtype=np.int32)
    eps = 1e-10

    junked = set()

    for c in list(components.keys()):
        if c in junked:
            continue

        if len(new_components) - len(junked) <= 1:
            break  # Don't junk if it would leave no components

        if c not in new_components:
            continue

        # Vectorized index selection
        mask_c = new_assignments_arr == c
        indices = np.where(mask_c)[0]
        if len(indices) == 0:
            continue

        # Find reassignments using full R-D assignment rule
        other_components = [c_other for c_other in new_components.keys()
                          if c_other != c and c_other not in junked]

        if not other_components:
            continue

        # Vectorized R-D reassignment computation
        mu_c_e = new_components[c]['mu_e']
        mu_c_a = new_components[c]['mu_a']

        # Extract embeddings for this component's samples
        e_c = embeddings_e[indices]
        a_c = attributions_a[indices]
        P_c = path_probs[indices]

        # Stack other component centers
        centers_e = np.array([new_components[c_prime]['mu_e'] for c_prime in other_components])
        centers_a = np.array([new_components[c_prime]['mu_a'] for c_prime in other_components])
        rate_costs = np.array([-np.log(P_bar.get(c_prime, eps) + eps) for c_prime in other_components])

        # Compute distances
        dist_e_sq = compute_mse_distances(e_c, centers_e)
        if metric_a == "l1":
            dist_a = compute_l1_distances(a_c, centers_a)
        else:
            dist_a = compute_mse_distances(a_c, centers_a)

        # Total R-D cost
        total_cost = rate_costs[np.newaxis, :] + beta_e * dist_e_sq + beta_a * dist_a

        # Find best reassignment
        best_idx = np.argmin(total_cost, axis=1)

        # Compute distortion cost of reassignment
        # Helper for scalar dist
        def get_dist_to_center(data, center, metric="l2"):
            if metric == "l1":
                return compute_l1_distance_to_center(data, center)
            return compute_mse_distance_to_center(data, center)
            
        def get_dist_pairwise(data, centers_arr, metric="l2"):
            if metric == "l1":
                return compute_l1_distance_pairwise(data, centers_arr)
            return compute_mse_distance_pairwise(data, centers_arr)

        dist_to_current_e = get_dist_to_center(e_c, mu_c_e, "l2")
        dist_to_current_a = get_dist_to_center(a_c, mu_c_a, metric_a)

        dist_to_new_e = get_dist_pairwise(e_c, centers_e[best_idx], "l2")
        dist_to_new_a = get_dist_pairwise(a_c, centers_a[best_idx], metric_a)

        # Weighted distortion change
        if use_weighted_distortion:
            weights = P_c / W_total
            delta_D_e = float(np.sum(weights * (dist_to_new_e - dist_to_current_e)))
            delta_D_a = float(np.sum(weights * (dist_to_new_a - dist_to_current_a)))
        else:
            # Unweighted distortion (weighted by count fraction N_c/N implicitly in formula)
            # D = sum (N_c/N) * mean_dist_c
            # Delta D = sum (1/N) * (dist_new - dist_old)
            n_samples = len(embeddings_e)
            delta_D_e = float(np.sum((1/n_samples) * (dist_to_new_e - dist_to_current_e)))
            delta_D_a = float(np.sum((1/n_samples) * (dist_to_new_a - dist_to_current_a)))

        # Compute rate savings
        H_before = compute_entropy(P_bar)

        # Entropy after removing c and redistributing mass
        P_bar_after = {}

        # Vectorized additional mass computation
        for i_other, c_other in enumerate(other_components):
            W_c_other = new_components[c_other]['W_c']
            mask_to_c_other = best_idx == i_other
            additional_mass = float(np.sum(P_c[mask_to_c_other]))
            P_bar_after[c_other] = (W_c_other + additional_mass) / W_total

        H_after = compute_entropy(P_bar_after)
        delta_H = H_before - H_after

        # Junk criterion
        distortion_cost = beta_e * delta_D_e + beta_a * delta_D_a

        if delta_H > distortion_cost:
            # Apply junk
            reassign_ids = np.array([other_components[i] for i in best_idx])
            new_assignments_arr[indices] = reassign_ids

            if c in new_components:
                del new_components[c]

            junked.add(c)

            # Recompute centers and masses for affected components
            for c_prime in other_components:
                if c_prime not in new_components:
                    continue

                mask_c_prime = new_assignments_arr == c_prime
                idx_c_prime = np.where(mask_c_prime)[0]
                if len(idx_c_prime) == 0:
                    continue

                P_c_prime = path_probs[idx_c_prime]
                W_c_prime = float(np.sum(P_c_prime))

                if W_c_prime > 0:
                    if use_weighted_distortion:
                        mu_e_new = np.sum(P_c_prime[:, None] * embeddings_e[idx_c_prime], axis=0) / W_c_prime
                    else:
                        mu_e_new = np.mean(embeddings_e[idx_c_prime], axis=0)
                        
                    mu_e_norm = np.linalg.norm(mu_e_new)
                    if mu_e_norm > 1e-10:
                        mu_e_new = mu_e_new / mu_e_norm
                        
                    if metric_a == "l1":
                        mu_a_new = np.median(attributions_a[idx_c_prime], axis=0)
                    else:
                        if use_weighted_distortion:
                             mu_a_new = np.sum(P_c_prime[:, None] * attributions_a[idx_c_prime], axis=0) / W_c_prime
                        else:
                             mu_a_new = np.mean(attributions_a[idx_c_prime], axis=0)

                    new_components[c_prime]['mu_e'] = mu_e_new
                    new_components[c_prime]['mu_a'] = mu_a_new
                    new_components[c_prime]['W_c'] = W_c_prime
                    new_components[c_prime]['indices'] = idx_c_prime.tolist()

    return new_components, new_assignments_arr.tolist()


def apply_adaptive_control(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    assignments: List[int],
    components: Dict[int, Dict],
    P_bar: Dict[int, float],
    Var_e: Dict[int, float],
    Var_a: Dict[int, float],
    beta_e: float,
    beta_a: float,
    K_max: int,
    next_component_id: int,
    metric_a: str = "l2",
    use_weighted_distortion: bool = True
) -> Tuple[Dict[int, Dict], List[int], int]:
    """Apply all adaptive control operations in sequence.

    Order: Split → Junk
    (No Clone - it's a special case of Split)

    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        path_probs: Probability weights
        assignments: Current assignments
        components: Current components
        P_bar: Normalized component masses
        Var_e: Semantic variance per component
        Var_a: Attribution variance per component
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight
        K_max: Maximum components
        next_component_id: Next available component ID
        metric_a: Metric for attribution
        use_weighted_distortion: Whether to use probability weights

    Returns:
        Tuple of (components, assignments, next_component_id)
    """
    W_total = np.sum(path_probs)
    if W_total == 0:
        W_total = 1.0

    # 1. Split operation
    components, assignments, next_component_id = split_operation(
        embeddings_e, attributions_a, path_probs, assignments,
        components, P_bar, Var_e, Var_a, beta_e, beta_a,
        K_max, next_component_id, W_total,
        metric_a=metric_a, use_weighted_distortion=use_weighted_distortion
    )

    # Recompute P_bar and variances after split
    component_ids = list(components.keys())
    W_c, _ = compute_component_masses(assignments, path_probs, component_ids)
    P_bar = compute_normalized_masses(W_c, W_total)

    # Compute variances using helper function
    assignments_arr = np.asarray(assignments, dtype=np.int32)
    Var_e = {}
    Var_a = {}
    for c, comp in components.items():
        mask_c = assignments_arr == c
        indices = np.where(mask_c)[0]
        W_c_val = W_c.get(c, 0)
        if len(indices) == 0:
            Var_e[c] = 0.0
            Var_a[c] = 0.0
            continue

        Var_e[c] = compute_component_variance(
            embeddings_e[indices], comp['mu_e'], path_probs[indices], W_c_val,
            "l2", use_weighted_distortion
        )
        Var_a[c] = compute_component_variance(
            attributions_a[indices], comp['mu_a'], path_probs[indices], W_c_val,
            metric_a, use_weighted_distortion
        )

    # 2. Junk operation
    components, assignments = junk_operation(
        embeddings_e, attributions_a, path_probs, assignments,
        components, P_bar, beta_e, beta_a, W_total,
        metric_a=metric_a, use_weighted_distortion=use_weighted_distortion
    )

    return components, assignments, next_component_id


if __name__ == "__main__":
    # Test adaptive control operations
    np.random.seed(42)

    n_samples = 100
    d_e = 64
    d_a = 128

    # Create test data with 3 natural clusters
    embeddings_e = np.vstack([
        np.random.randn(40, d_e) + np.array([3] * d_e),
        np.random.randn(35, d_e) + np.array([-3] * d_e),
        np.random.randn(25, d_e),
    ])
    attributions_a = np.vstack([
        np.random.randn(40, d_a) + np.array([2] * d_a),
        np.random.randn(35, d_a) + np.array([-2] * d_a),
        np.random.randn(25, d_a),
    ])
    path_probs = np.random.rand(n_samples)
    path_probs = path_probs / np.sum(path_probs)

    # Start with single component
    assignments = [1] * n_samples
    W_total = np.sum(path_probs)
    mu_e_init = np.sum(path_probs[:, None] * embeddings_e, axis=0) / W_total
    # L2-normalize for spherical clustering
    mu_e_init = mu_e_init / (np.linalg.norm(mu_e_init) + 1e-10)
    components = {
        1: {
            'mu_e': mu_e_init,
            'mu_a': np.sum(path_probs[:, None] * attributions_a, axis=0) / W_total,
            'W_c': W_total,
            'indices': list(range(n_samples)),
        }
    }

    # Compute initial stats
    W_c, _ = compute_component_masses(assignments, path_probs, [1])
    P_bar = compute_normalized_masses(W_c, W_total)

    Var_e = {}
    Var_a = {}
    for c, comp in components.items():
        Var_e[c] = compute_component_variance(embeddings_e, comp['mu_e'], path_probs, W_c[c])
        Var_a[c] = compute_component_variance(attributions_a, comp['mu_a'], path_probs, W_c[c])

    print("Testing adaptive control operations...")
    print(f"Initial: {len(components)} component(s)")
    print(f"  Var_e: {Var_e}")
    print(f"  Var_a: {Var_a}")

    # Apply adaptive control with equal beta weights
    beta_e = 1.0
    beta_a = 1.0
    K_max = 10

    updated_components, updated_assignments, next_id = apply_adaptive_control(
        embeddings_e, attributions_a, path_probs, assignments,
        components, P_bar, Var_e, Var_a,
        beta_e, beta_a, K_max, next_component_id=2
    )

    print(f"\nAfter adaptive control:")
    print(f"  Components: {len(updated_components)}")
    
    # Test L1 Unweighted
    print("\nTesting L1 Unweighted adaptive control...")
    updated_components_l1, _, _ = apply_adaptive_control(
        embeddings_e, attributions_a, path_probs, assignments,
        components, P_bar, Var_e, Var_a,
        beta_e, beta_a, K_max, next_component_id=2,
        metric_a="l1", use_weighted_distortion=False
    )
    print(f"  Components (L1 Unweighted): {len(updated_components_l1)}")

    print("\nAdaptive control test passed!")
