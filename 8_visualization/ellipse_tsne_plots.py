"""t-SNE visualization with cluster ellipses for parameter sweeps.

Extracted from gaussian_optimization/tests/clustering_visualization/visualize_beta_sweep.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def compute_cluster_ellipse(
    points: np.ndarray,
    weights: np.ndarray,
    n_std: float = 1.5
) -> Tuple[np.ndarray, float, float, float]:
    """Compute ellipse parameters for a cluster.

    Args:
        points: 2D points in cluster (n, 2)
        weights: Point weights (n,)
        n_std: Number of standard deviations for ellipse

    Returns:
        center: Ellipse center (2,)
        width: Ellipse width
        height: Ellipse height
        angle: Ellipse rotation angle in degrees
    """
    if len(points) < 2:
        center = points.mean(axis=0) if len(points) > 0 else np.zeros(2)
        return center, 0.1, 0.1, 0

    # Normalize weights
    weights = weights / weights.sum()

    # Weighted center
    center = np.sum(weights[:, None] * points, axis=0)

    # Weighted covariance
    diff = points - center
    cov = np.dot(weights * diff.T, diff)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-6)

    # Ellipse parameters
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    return center, width, height, angle


def plot_tsne_with_cluster_ellipses(
    tsne_coords: np.ndarray,
    cluster_results: Dict[float, Tuple[Dict, List[int], Dict]],
    path_probs: np.ndarray,
    title: str,
    output_path: Path,
    sweep_param: str = "beta",
    figsize: tuple = (14, 11),
    dpi: int = 150,
):
    """Plot t-SNE with cluster ellipses for different parameter values.

    Args:
        tsne_coords: 2D t-SNE coordinates (N, 2)
        cluster_results: Dict mapping param_value -> (components, assignments, stats)
        path_probs: Path probabilities for weighting
        title: Plot title
        output_path: Output file path
        sweep_param: Which parameter is being swept ("beta" or "gamma")
        figsize: Figure size
        dpi: Output DPI
    """
    fig, ax = plt.subplots(figsize=figsize)

    # High-contrast color palette
    sweep_colors = [
        '#1f77b4',  # blue
        '#d62728',  # red
        '#2ca02c',  # green
        '#9467bd',  # purple
        '#ff7f0e',  # orange
        '#17becf',  # cyan
    ]

    param_values = sorted(cluster_results.keys())
    legend_handles = []

    # Label format based on sweep parameter
    if sweep_param == "gamma":
        label_fmt = lambda v, k: f'gamma={v:.1f} (K={k})'
    else:
        label_fmt = lambda v, k: f'beta={v:.0f} (K={k})'

    for param_idx, param_val in enumerate(param_values):
        components, assignments, stats = cluster_results[param_val]
        base_color = sweep_colors[param_idx % len(sweep_colors)]

        # Convert hex to RGB tuple
        rgb = mcolors.to_rgb(base_color)

        # Get unique clusters (excluding junk)
        cluster_ids = sorted([c for c in set(assignments) if c > 0])

        # Plot sample points for this parameter value
        assignments_arr = np.array(assignments)
        for cluster_id in cluster_ids:
            mask = assignments_arr == cluster_id
            cluster_points = tsne_coords[mask]
            cluster_weights = path_probs[mask]

            if len(cluster_points) < 1:
                continue

            # Plot sample points with size proportional to probability
            max_weight = cluster_weights.max() if cluster_weights.max() > 0 else 1.0
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[base_color],
                s=30 + 100 * (cluster_weights / max_weight),
                alpha=0.6,
                edgecolors='white',
                linewidths=0.5,
                zorder=4 + param_idx * 0.1
            )

            if len(cluster_points) < 2:
                continue

            # Compute ellipse
            center, width, height, angle = compute_cluster_ellipse(
                cluster_points, cluster_weights, n_std=1.5
            )

            # Draw ellipse with transparent fill and solid outline
            ellipse_fill = Ellipse(
                center, width, height, angle=angle,
                facecolor=(*rgb, 0.12),  # Transparent fill
                edgecolor='none',
                zorder=2 + param_idx * 0.1
            )
            ax.add_patch(ellipse_fill)

            ellipse_outline = Ellipse(
                center, width, height, angle=angle,
                facecolor='none',
                edgecolor=(*rgb, 0.9),  # Solid outline
                linewidth=2.5,
                zorder=3 + param_idx * 0.1
            )
            ax.add_patch(ellipse_outline)

            # Mark cluster center with larger X
            ax.scatter(
                center[0], center[1],
                c=[base_color],
                s=150,
                marker='X',
                edgecolors='black',
                linewidths=1,
                zorder=10
            )

        # Add to legend
        n_clusters = len(cluster_ids)
        legend_handles.append(
            mpatches.Patch(
                facecolor=(*rgb, 0.4),
                edgecolor=base_color,
                linewidth=2,
                label=label_fmt(param_val, n_clusters)
            )
        )

    ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
