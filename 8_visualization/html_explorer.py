"""Interactive HTML cluster explorer.

Extracted from gaussian_optimization/tests/clustering_visualization/visualize_beta_sweep.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def generate_cluster_html(
    cluster_results: Dict[float, Tuple[Dict, List[int], Dict]],
    continuations: List[str],
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    output_path: Path,
    top_k: int = 10,
    sweep_param: str = "beta",
):
    """Generate interactive HTML for exploring clusters.

    Args:
        cluster_results: Dict mapping param_value -> (components, assignments, stats)
        continuations: List of continuation texts
        embeddings_e: Embedding vectors
        attributions_a: Attribution vectors
        path_probs: Path probabilities
        output_path: Output HTML path
        top_k: Number of top continuations to show per cluster
        sweep_param: Which parameter is being swept ("beta" or "gamma")
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cluster Explorer</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .beta-section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .beta-header {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .cluster-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .cluster {{
            background: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            min-width: 300px;
            max-width: 400px;
            flex: 1;
        }}
        .cluster-header {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #444;
        }}
        .cluster-stats {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 10px;
        }}
        .continuation {{
            padding: 8px 10px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #4a90d9;
            font-size: 0.95em;
        }}
        .continuation:hover {{
            background: #e8f4ff;
        }}
        .prob {{
            color: #888;
            font-size: 0.8em;
        }}
        .distance {{
            color: #999;
            font-size: 0.75em;
        }}
        .expand-btn {{
            color: #4a90d9;
            cursor: pointer;
            text-decoration: underline;
            margin-top: 10px;
            display: inline-block;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cluster Explorer</h1>
        <p>Interactive exploration of clustering results across different {sweep_param} values</p>
    </div>
"""

    # Label format based on sweep parameter
    if sweep_param == "gamma":
        header_fmt = lambda v: f'gamma = {v:.2f}'
    else:
        header_fmt = lambda v: f'beta = {v:.0f}'

    for param_val in sorted(cluster_results.keys()):
        components, assignments, stats = cluster_results[param_val]

        html_content += f"""
    <div class="beta-section">
        <div class="beta-header">{header_fmt(param_val)}</div>
        <div class="cluster-stats">
            K = {len([c for c in components.keys() if c > 0])} clusters |
            L_RD = {stats.get('L_RD', 0):.4f} |
            H(C) = {stats.get('H', 0):.4f} |
            D_e = {stats.get('D_e', 0):.4f} |
            D_a = {stats.get('D_a', 0):.4f}
        </div>
        <div class="cluster-container">
"""

        cluster_ids = sorted([c for c in components.keys() if c > 0])

        for cluster_id in cluster_ids:
            comp = components[cluster_id]
            indices = comp.get('indices', [])

            if not indices:
                continue

            # Compute distances to center
            mu_e = comp.get('mu_e')
            mu_a = comp.get('mu_a')

            if mu_e is None or mu_a is None:
                continue

            mu_e = np.array(mu_e)
            mu_a = np.array(mu_a)

            # Compute combined distance for ranking
            e_dists = np.linalg.norm(embeddings_e[indices] - mu_e, axis=1)
            a_dists = np.linalg.norm(attributions_a[indices] - mu_a, axis=1)
            combined_dists = e_dists + a_dists

            # Sort by distance (closest first)
            sorted_idx = np.argsort(combined_dists)

            # Get top-k
            top_indices = [indices[i] for i in sorted_idx[:top_k]]
            top_dists = combined_dists[sorted_idx[:top_k]]

            html_content += f"""
            <div class="cluster">
                <div class="cluster-header">Cluster {cluster_id}</div>
                <div class="cluster-stats">
                    {len(indices)} continuations | W_c = {comp.get('W_c', 0):.4f}
                </div>
"""

            for idx, dist in zip(top_indices, top_dists):
                cont_text = continuations[idx] if idx < len(continuations) else f"[{idx}]"
                cont_text = cont_text.replace('<', '&lt;').replace('>', '&gt;')
                cont_text = cont_text[:100] + '...' if len(cont_text) > 100 else cont_text
                prob = path_probs[idx] if idx < len(path_probs) else 0

                html_content += f"""
                <div class="continuation">
                    {cont_text}
                    <br><span class="prob">P = {prob:.2e}</span>
                    <span class="distance">| dist = {dist:.3f}</span>
                </div>
"""

            # Show expand button if more items
            if len(indices) > top_k:
                html_content += f"""
                <div class="expand-btn" onclick="alert('Full list: {len(indices)} items')">
                    +{len(indices) - top_k} more...
                </div>
"""

            html_content += """
            </div>
"""

        html_content += """
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_index_html(
    prefix_ids: List[str],
    output_dir: Path,
):
    """Generate index HTML linking all cluster explorers.

    Args:
        prefix_ids: List of prefix identifiers
        output_dir: Output directory containing HTML files
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Cluster Explorer Index</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .prefix-list {
            list-style: none;
            padding: 0;
        }
        .prefix-list li {
            margin: 10px 0;
        }
        .prefix-list a {
            display: inline-block;
            padding: 10px 20px;
            background: white;
            border-radius: 5px;
            text-decoration: none;
            color: #4a90d9;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: background 0.2s;
        }
        .prefix-list a:hover {
            background: #e8f4ff;
        }
    </style>
</head>
<body>
    <h1>Cluster Explorer Index</h1>
    <ul class="prefix-list">
"""

    for prefix_id in sorted(prefix_ids):
        html_content += f"""        <li><a href="{prefix_id}_cluster_explorer.html">{prefix_id}</a></li>
"""

    html_content += """    </ul>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)
