"""Sweep mode utilities for hyperparameter search in clustering.

Provides functions for:
- Running (beta, gamma) grid search with parallel processing
- Computing silhouette scores for clustering quality
- Finding Pareto-optimal configurations
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def compute_silhouette_scores(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    assignments: List[int],
) -> Tuple[float, float]:
    """Compute silhouette scores for clustering quality.

    Args:
        embeddings_e: Semantic embeddings (N, d_e)
        attributions_a: Attribution embeddings (N, d_a)
        assignments: Cluster assignments

    Returns:
        Tuple of (silhouette_e, silhouette_a)
    """
    labels = np.array(assignments)
    valid_mask = labels > 0
    unique_labels = np.unique(labels[valid_mask])

    # Need at least 2 clusters for silhouette score
    if len(unique_labels) < 2:
        return 0.0, 0.0

    n_valid = np.sum(valid_mask)
    if n_valid < 2:
        return 0.0, 0.0

    try:
        sil_e = silhouette_score(embeddings_e[valid_mask], labels[valid_mask])
        sil_a = silhouette_score(attributions_a[valid_mask], labels[valid_mask])
    except Exception:
        return 0.0, 0.0

    return float(sil_e), float(sil_a)


def compute_harmonic_mean(sil_e: float, sil_a: float) -> float:
    """Compute harmonic mean of silhouette scores.

    Silhouette scores are in [-1, 1], so we shift to [0, 2] for valid harmonic mean.

    Args:
        sil_e: Embedding silhouette score
        sil_a: Attribution silhouette score

    Returns:
        Harmonic mean shifted back to [-1, 1] range
    """
    sil_e_shifted = sil_e + 1.0
    sil_a_shifted = sil_a + 1.0

    if sil_e_shifted <= 0 or sil_a_shifted <= 0:
        return -1.0

    h_mean = 2 * sil_e_shifted * sil_a_shifted / (sil_e_shifted + sil_a_shifted)
    return h_mean - 1.0  # Shift back to [-1, 1]


def find_pareto_front(results: List[Dict], keys: List[str] = ["D_e", "D_a"]) -> List[Dict]:
    """Find Pareto-optimal points (minimizing given keys).

    Args:
        results: List of result dicts with metrics
        keys: Keys to minimize for Pareto optimality

    Returns:
        List of Pareto-optimal result dicts
    """
    if not results:
        return []

    pareto = []
    for r in results:
        is_dominated = False
        values = [r.get(k, float('inf')) for k in keys]

        for other in results:
            if other is r:
                continue
            other_values = [other.get(k, float('inf')) for k in keys]

            # Check if other dominates r (strictly better in at least one, not worse in any)
            better_in_at_least_one = any(ov < v for ov, v in zip(other_values, values))
            not_worse_in_any = all(ov <= v for ov, v in zip(other_values, values))

            if better_in_at_least_one and not_worse_in_any:
                is_dominated = True
                break

        if not is_dominated:
            pareto.append(r)

    return pareto


def _run_single_config(args_tuple):
    """Worker function for parallel sweep execution.

    Runs clustering for a single (beta, gamma) configuration.

    Args:
        args_tuple: Tuple of (data, beta, gamma, K_max, max_iterations, convergence_threshold, metric_a, use_weighted_distortion)

    Returns:
        Dict with beta, gamma, and clustering metrics
    """
    data, beta, gamma, K_max, max_iterations, convergence_threshold, metric_a, use_weighted_distortion = args_tuple

    # Compute beta_e, beta_a
    beta_e = gamma * beta
    beta_a = (1 - gamma) * beta

    # Create a minimal logger for worker
    import logging
    logger = logging.getLogger(f"sweep_worker_{beta}_{gamma}")
    logger.setLevel(logging.WARNING)  # Suppress detailed logs in workers

    try:
        # Import here to avoid issues with multiprocessing
        from initialize import initialize_single_component
        from em_loop import run_em_iteration, check_convergence
        from adaptive_control import apply_adaptive_control
        from rd_objective import compute_full_rd_statistics, compute_component_masses, compute_component_variance

        embeddings_e = data["embeddings_e"].copy()
        attributions_a = data["attributions_a"].copy()
        path_probs = data["path_probs"].copy()

        # Initialize
        components, assignments = initialize_single_component(
            embeddings_e, attributions_a, path_probs
        )

        next_component_id = max(components.keys()) + 1 if components else 2
        L_RD_prev = np.inf

        # Track history
        history = {
            "iterations": [],
            "n_components": [],
            "L_RD": [],
            "H": [],
            "D_e": [],
            "D_a": [],
        }

        # EM loop
        for iteration in range(max_iterations):
            assignments, components, rd_stats = run_em_iteration(
                embeddings_e, attributions_a, path_probs,
                components, beta_e, beta_a,
                metric_a=metric_a, use_weighted_distortion=use_weighted_distortion
            )

            L_RD_curr = rd_stats['L_RD']

            # Adaptive control
            P_bar = rd_stats['P_bar']
            Var_e = rd_stats.get('Var_e', {})
            Var_a = rd_stats.get('Var_a', {})

            if not Var_e or not Var_a:
                W_c, _ = compute_component_masses(assignments, path_probs, list(components.keys()))
                for c, comp in components.items():
                    indices = [i for i, a in enumerate(assignments) if a == c]
                    W_c_val = W_c.get(c, 0)
                    if not indices:
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

            components, assignments, next_component_id = apply_adaptive_control(
                embeddings_e, attributions_a, path_probs,
                assignments, components, P_bar, Var_e, Var_a,
                beta_e, beta_a, K_max, next_component_id,
                metric_a=metric_a, use_weighted_distortion=use_weighted_distortion
            )

            if len(components) > 0:
                rd_stats = compute_full_rd_statistics(
                    embeddings_e, attributions_a, assignments, path_probs,
                    components, beta_e, beta_a,
                    metric_a=metric_a, use_weighted_distortion=use_weighted_distortion
                )
                L_RD_curr = rd_stats['L_RD']

            # Track history
            history["iterations"].append(iteration + 1)
            history["n_components"].append(len(components))
            history["L_RD"].append(float(L_RD_curr))
            history["H"].append(float(rd_stats['H']))
            history["D_e"].append(float(rd_stats['D_e']))
            history["D_a"].append(float(rd_stats['D_a']))

            if check_convergence(L_RD_prev, L_RD_curr, convergence_threshold):
                break

            L_RD_prev = L_RD_curr

        # Compute silhouette scores
        sil_e, sil_a = compute_silhouette_scores(embeddings_e, attributions_a, assignments)
        harmonic = compute_harmonic_mean(sil_e, sil_a)

        # Prepare component data (centroids)
        # Note: Attribution vectors can be large, so we might want to be careful
        # But for 4B/8B models they are usually manageable (~4k-8k dims)
        components_data = {}
        for c_id, comp in components.items():
            components_data[str(c_id)] = {
                "mu_e": comp["mu_e"].tolist() if hasattr(comp["mu_e"], "tolist") else comp["mu_e"],
                # Save attribution centroid if available
                "mu_a": comp["mu_a"].tolist() if hasattr(comp["mu_a"], "tolist") else comp["mu_a"],
                "W_c": float(comp["W_c"]),
            }

        return {
            "beta": float(beta),
            "gamma": float(gamma),
            "beta_e": float(beta_e),
            "beta_a": float(beta_a),
            "K": len(components),
            "H": float(rd_stats['H']),
            "D_e": float(rd_stats['D_e']),
            "D_a": float(rd_stats['D_a']),
            "L_RD": float(rd_stats['L_RD']),
            "sil_e": float(sil_e),
            "sil_a": float(sil_a),
            "harmonic": float(harmonic),
            "converged": bool(check_convergence(L_RD_prev, L_RD_curr, convergence_threshold)),
            "n_iterations": iteration + 1,
            # Add detailed structure for retrospective analysis
            "assignments": [int(a) for a in assignments],
            "components": components_data,
            # Add iteration history
            "history": history
        }

    except Exception as e:
        return {
            "beta": float(beta),
            "gamma": float(gamma),
            "error": str(e),
            "harmonic": -1.0,
        }


def run_sweep_mode(
    data: Dict,
    sweeps_config: Dict,
    K_max: int,
    max_iterations: int,
    convergence_threshold: float,
    logger,
    n_workers: int = 4,
    metric_a: str = "l2",
    use_weighted_distortion: bool = True
) -> Dict:
    """Run sweep over (beta, gamma) grid with parallel processing.

    Args:
        data: Data dict with embeddings_e, attributions_a, path_probs
        sweeps_config: Sweep configuration with beta_values and gamma_values
        K_max: Maximum number of components
        max_iterations: Maximum EM iterations
        convergence_threshold: Convergence threshold
        logger: Logger instance
        n_workers: Number of parallel workers
        metric_a: Attribution distance metric
        use_weighted_distortion: Whether to use probability weights

    Returns:
        Dict with grid results
    """
    beta_values = sweeps_config.get("beta_values", [5.0])
    gamma_values = sweeps_config.get("gamma_values", [0.5])
    logger.info("=" * 60)
    logger.info("SWEEP MODE ENABLED")
    logger.info("=" * 60)
    logger.info(f"Beta values: {beta_values}")
    logger.info(f"Gamma values: {gamma_values}")
    logger.info(f"Total configurations: {len(beta_values) * len(gamma_values)}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Metric: {metric_a}, Weighted: {use_weighted_distortion}")

    # Build task list
    tasks = []
    for beta in beta_values:
        for gamma in gamma_values:
            tasks.append((data, beta, gamma, K_max, max_iterations, convergence_threshold, metric_a, use_weighted_distortion))

    # Run in parallel or sequential
    grid_results = []

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_config, task): task for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Sweep configs", disable=logger.getEffectiveLevel() > 20):
                result = future.result()
                grid_results.append(result)

                # Log progress
                if "error" in result:
                    logger.warning(f"Config β={result['beta']}, γ={result['gamma']}: {result['error']}")
                else:
                    logger.info(f"Config β={result['beta']}, γ={result['gamma']}: "
                               f"K={result['K']}, harmonic={result['harmonic']:.4f}")
    else:
        # Sequential execution
        for task in tqdm(tasks, desc="Sweep configs", disable=logger.getEffectiveLevel() > 20):
            result = _run_single_config(task)
            grid_results.append(result)
            
            # Log progress
            if "error" in result:
                logger.warning(f"Config β={result['beta']}, γ={result['gamma']}: {result['error']}")
            else:
                logger.info(f"Config β={result['beta']}, γ={result['gamma']}: "
                           f"K={result['K']}, harmonic={result['harmonic']:.4f}")

    # Sort by (beta, gamma) for consistent ordering
    grid_results.sort(key=lambda x: (x.get("beta", 0), x.get("gamma", 0)))

    # Build sweep results
    H_0 = data.get("H_0")
    sweep_results = {
        "prefix_id": data["prefix_id"],
        "prefix": data.get("prefix"),
        "H_0": H_0.tolist() if hasattr(H_0, "tolist") else H_0,
        "sweep_config": {
            "beta_values": beta_values,
            "gamma_values": gamma_values,
            "metric_a": metric_a,
            "use_weighted_distortion": use_weighted_distortion
        },
        "grid": grid_results,
    }

    logger.info("\n" + "=" * 60)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Completed {len(grid_results)} configurations")

    return sweep_results
