#!/usr/bin/env python3
"""Visualize cluster contents with nearest continuations and statistics.

For each prefix, generates an HTML file showing:
- Cluster statistics (silhouette, steering scores, mass, etc.)
- Top-K nearest continuations by semantic, attribution, and combined distance
- All assigned continuations

Usage:
    uv run python visualize_clusters.py --results-dir /path/to/results --output-dir ./output
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from html import escape


@dataclass
class ClusterVisualization:
    """Data for visualizing a single cluster."""
    cluster_id: str
    n_assigned: int
    total_mass: float
    mu_e: np.ndarray  # semantic centroid
    mu_a: np.ndarray  # attribution centroid

    # Statistics
    sil_e: Optional[float] = None
    sil_a: Optional[float] = None

    # Steering scores (from best config)
    steering_corr: Optional[float] = None
    steering_r2: Optional[float] = None
    steering_win_rate: Optional[float] = None

    # Assigned continuation indices
    assigned_indices: list = None

    # Top-K nearest by different metrics
    top_k_semantic: list = None  # [(idx, distance, text), ...]
    top_k_attribution: list = None
    top_k_combined: list = None


def load_prefix_data(prefix_id: str, results_dir: Path) -> dict:
    """Load all data for a prefix."""
    data = {
        'prefix_id': prefix_id,
        'prefix_text': '',
        'continuations': [],
        'embeddings': None,
        'attributions': None,
        'clustering': None,
        'steering': None,
    }

    # Load embeddings
    embed_dir = results_dir / "4_feature_extraction" / "embeddings"
    embed_file = embed_dir / f"{prefix_id}_embeddings.npy"
    meta_file = embed_dir / f"{prefix_id}_embeddings_meta.json"

    if embed_file.exists() and meta_file.exists():
        data['embeddings'] = np.load(embed_file)
        with open(meta_file) as f:
            meta = json.load(f)
        data['prefix_text'] = meta.get('prefix', '')
        data['continuations'] = meta.get('continuations', [])
        data['full_sequences'] = meta.get('full_sequences', [])

    # Load attributions
    attr_file = results_dir / "3_attribution_graphs" / f"{prefix_id}_prefix_context.pt"
    if attr_file.exists():
        attr_data = torch.load(attr_file, map_location='cpu', weights_only=False)
        data['attributions'] = attr_data['aggregated_attributions'].float().numpy()

    # Load clustering (best config from sweep)
    sweep_file = results_dir / "5_clustering" / f"{prefix_id}_sweep_results.json"
    if sweep_file.exists():
        with open(sweep_file) as f:
            sweep = json.load(f)
        data['sweep'] = sweep

        # Find best config in grid
        best = sweep.get('best', {})
        best_beta, best_gamma = best.get('beta'), best.get('gamma')

        for entry in sweep.get('grid', []):
            if entry['beta'] == best_beta and entry['gamma'] == best_gamma:
                data['clustering'] = entry
                break

    # Load steering results
    steering_file = results_dir / "7_validation" / "7c_steering" / f"{prefix_id}_sweep_results.json"
    if steering_file.exists():
        with open(steering_file) as f:
            data['steering'] = json.load(f)

    return data


def compute_distances(
    embeddings: np.ndarray,
    attributions: np.ndarray,
    mu_e: np.ndarray,
    mu_a: np.ndarray,
    gamma: float = 0.5
) -> tuple:
    """Compute distances from each continuation to cluster centroid."""
    # Semantic distance (cosine distance)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    mu_e_norm = mu_e / (np.linalg.norm(mu_e) + 1e-8)
    semantic_dist = 1 - emb_norm @ mu_e_norm  # cosine distance

    # Attribution distance (cosine distance)
    attr_norm = attributions / (np.linalg.norm(attributions, axis=1, keepdims=True) + 1e-8)
    mu_a_norm = mu_a / (np.linalg.norm(mu_a) + 1e-8)
    attribution_dist = 1 - attr_norm @ mu_a_norm

    # Combined distance (weighted by gamma)
    combined_dist = gamma * semantic_dist + (1 - gamma) * attribution_dist

    return semantic_dist, attribution_dist, combined_dist


def get_top_k_nearest(
    distances: np.ndarray,
    continuations: list,
    k: int = 10
) -> list:
    """Get top-K nearest continuations by distance."""
    indices = np.argsort(distances)[:k]
    return [(int(idx), float(distances[idx]), continuations[idx]) for idx in indices]


def build_cluster_visualization(
    cluster_id: str,
    data: dict,
    gamma: float,
    top_k: int = 10
) -> ClusterVisualization:
    """Build visualization data for a single cluster."""
    clustering = data['clustering']
    components = clustering.get('components', {})
    assignments = clustering.get('assignments', [])

    if cluster_id not in components:
        return None

    comp = components[cluster_id]
    mu_e = np.array(comp['mu_e'])
    mu_a = np.array(comp['mu_a'])
    W_c = comp.get('W_c', 0)

    # Get assigned indices
    assigned_indices = [i for i, a in enumerate(assignments) if str(a) == cluster_id]

    # Compute distances
    semantic_dist, attribution_dist, combined_dist = compute_distances(
        data['embeddings'],
        data['attributions'],
        mu_e,
        mu_a,
        gamma
    )

    # Get steering scores
    steering_corr = None
    steering_r2 = None
    steering_win_rate = None

    if data['steering']:
        # Use positive_B5 config as default
        results = data['steering'].get('results', {})
        config_key = 'sign_positive_B5_H_c'
        if config_key in results:
            per_cluster = results[config_key].get('per_cluster', {})
            if cluster_id in per_cluster:
                cluster_steering = per_cluster[cluster_id]
                steering_corr = cluster_steering.get('dose_response_corr')
                steering_r2 = cluster_steering.get('dose_response_r2')
                steering_win_rate = cluster_steering.get('win_rate_corr')  # Use correlation, not dict

    return ClusterVisualization(
        cluster_id=cluster_id,
        n_assigned=len(assigned_indices),
        total_mass=W_c,
        mu_e=mu_e,
        mu_a=mu_a,
        sil_e=clustering.get('sil_e'),
        sil_a=clustering.get('sil_a'),
        steering_corr=steering_corr,
        steering_r2=steering_r2,
        steering_win_rate=steering_win_rate,
        assigned_indices=assigned_indices,
        top_k_semantic=get_top_k_nearest(semantic_dist, data['continuations'], top_k),
        top_k_attribution=get_top_k_nearest(attribution_dist, data['continuations'], top_k),
        top_k_combined=get_top_k_nearest(combined_dist, data['continuations'], top_k),
    )


def generate_html(prefix_id: str, prefix_text: str, clusters: list, data: dict) -> str:
    """Generate HTML visualization for a prefix."""
    clustering = data.get('clustering', {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cluster Visualization: {escape(prefix_id)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        .prefix-text {{
            font-family: monospace;
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
        }}
        .summary {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-item .label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .summary-item .value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .cluster {{
            background: white;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .cluster-header {{
            background: #34495e;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .cluster-header h2 {{
            margin: 0;
            font-size: 18px;
        }}
        .cluster-stats {{
            display: flex;
            gap: 20px;
            font-size: 13px;
        }}
        .cluster-stats .stat {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .cluster-stats .stat-label {{
            font-size: 10px;
            opacity: 0.8;
        }}
        .cluster-stats .stat-value {{
            font-weight: bold;
        }}
        .cluster-stats .stat-value.positive {{
            color: #2ecc71;
        }}
        .cluster-stats .stat-value.negative {{
            color: #e74c3c;
        }}
        .cluster-body {{
            padding: 20px;
        }}
        .distance-section {{
            margin-bottom: 20px;
        }}
        .distance-section h3 {{
            font-size: 14px;
            color: #666;
            margin: 0 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }}
        .continuation-list {{
            font-family: monospace;
            font-size: 12px;
            line-height: 1.6;
        }}
        .continuation-item {{
            display: flex;
            padding: 5px 0;
            border-bottom: 1px solid #f5f5f5;
        }}
        .continuation-item:last-child {{
            border-bottom: none;
        }}
        .continuation-rank {{
            width: 30px;
            color: #999;
            flex-shrink: 0;
        }}
        .continuation-dist {{
            width: 80px;
            color: #3498db;
            flex-shrink: 0;
        }}
        .continuation-text {{
            flex: 1;
            word-break: break-word;
        }}
        .continuation-text.assigned {{
            background: #e8f5e9;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        .assigned-section {{
            background: #fafafa;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
        }}
        .assigned-section h4 {{
            margin: 0 0 10px 0;
            font-size: 13px;
            color: #666;
        }}
        .tabs {{
            display: flex;
            border-bottom: 2px solid #eee;
            margin-bottom: 15px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            font-size: 13px;
            color: #666;
        }}
        .tab:hover {{
            color: #2c3e50;
        }}
        .tab.active {{
            border-bottom-color: #3498db;
            color: #3498db;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{escape(prefix_id)}</h1>
        <div class="prefix-text">"{escape(prefix_text)}"</div>
    </div>

    <div class="summary">
        <div class="summary-grid">
            <div class="summary-item">
                <div class="label">Clusters</div>
                <div class="value">{clustering.get('K', 'N/A')}</div>
            </div>
            <div class="summary-item">
                <div class="label">β</div>
                <div class="value">{clustering.get('beta', 'N/A')}</div>
            </div>
            <div class="summary-item">
                <div class="label">γ</div>
                <div class="value">{clustering.get('gamma', 'N/A')}</div>
            </div>
            <div class="summary-item">
                <div class="label">Sil (semantic)</div>
                <div class="value">{clustering.get('sil_e', 0):.3f}</div>
            </div>
            <div class="summary-item">
                <div class="label">Sil (attribution)</div>
                <div class="value">{clustering.get('sil_a', 0):.3f}</div>
            </div>
            <div class="summary-item">
                <div class="label">Harmonic</div>
                <div class="value">{clustering.get('harmonic', 0):.3f}</div>
            </div>
        </div>
    </div>
"""

    for cluster in clusters:
        corr_class = 'positive' if cluster.steering_corr and cluster.steering_corr > 0 else 'negative'
        corr_str = f"{cluster.steering_corr:+.3f}" if cluster.steering_corr is not None else "N/A"
        r2_str = f"{cluster.steering_r2:.3f}" if cluster.steering_r2 is not None else "N/A"
        win_corr_class = 'positive' if cluster.steering_win_rate and cluster.steering_win_rate > 0 else 'negative'
        win_str = f"{cluster.steering_win_rate:+.3f}" if cluster.steering_win_rate is not None else "N/A"

        html += f"""
    <div class="cluster">
        <div class="cluster-header">
            <h2>Cluster {cluster.cluster_id}</h2>
            <div class="cluster-stats">
                <div class="stat">
                    <span class="stat-label">Assigned</span>
                    <span class="stat-value">{cluster.n_assigned}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Mass</span>
                    <span class="stat-value">{cluster.total_mass:.3f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Steering r</span>
                    <span class="stat-value {corr_class}">{corr_str}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">R²</span>
                    <span class="stat-value">{r2_str}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Win Corr</span>
                    <span class="stat-value {win_corr_class}">{win_str}</span>
                </div>
            </div>
        </div>
        <div class="cluster-body">
            <div class="tabs">
                <div class="tab active" onclick="switchTab(this, 'semantic-{cluster.cluster_id}')">Semantic</div>
                <div class="tab" onclick="switchTab(this, 'attribution-{cluster.cluster_id}')">Attribution</div>
                <div class="tab" onclick="switchTab(this, 'combined-{cluster.cluster_id}')">Combined</div>
            </div>

            <div id="semantic-{cluster.cluster_id}" class="tab-content active">
                <div class="continuation-list">
"""
        for rank, (idx, dist, text) in enumerate(cluster.top_k_semantic, 1):
            assigned_class = 'assigned' if idx in cluster.assigned_indices else ''
            html += f"""                    <div class="continuation-item">
                        <span class="continuation-rank">{rank}.</span>
                        <span class="continuation-dist">{dist:.4f}</span>
                        <span class="continuation-text {assigned_class}">{escape(text)}</span>
                    </div>
"""
        html += """                </div>
            </div>

"""
        html += f"""            <div id="attribution-{cluster.cluster_id}" class="tab-content">
                <div class="continuation-list">
"""
        for rank, (idx, dist, text) in enumerate(cluster.top_k_attribution, 1):
            assigned_class = 'assigned' if idx in cluster.assigned_indices else ''
            html += f"""                    <div class="continuation-item">
                        <span class="continuation-rank">{rank}.</span>
                        <span class="continuation-dist">{dist:.4f}</span>
                        <span class="continuation-text {assigned_class}">{escape(text)}</span>
                    </div>
"""
        html += """                </div>
            </div>

"""
        html += f"""            <div id="combined-{cluster.cluster_id}" class="tab-content">
                <div class="continuation-list">
"""
        for rank, (idx, dist, text) in enumerate(cluster.top_k_combined, 1):
            assigned_class = 'assigned' if idx in cluster.assigned_indices else ''
            html += f"""                    <div class="continuation-item">
                        <span class="continuation-rank">{rank}.</span>
                        <span class="continuation-dist">{dist:.4f}</span>
                        <span class="continuation-text {assigned_class}">{escape(text)}</span>
                    </div>
"""
        html += """                </div>
            </div>

"""
        # Assigned continuations section
        html += f"""            <div class="assigned-section">
                <h4>All Assigned Continuations ({cluster.n_assigned})</h4>
                <div class="continuation-list">
"""
        for idx in cluster.assigned_indices:
            if idx < len(data['continuations']):
                text = data['continuations'][idx]
                html += f"""                    <div class="continuation-item">
                        <span class="continuation-rank">{idx}</span>
                        <span class="continuation-text">{escape(text)}</span>
                    </div>
"""
        html += """                </div>
            </div>
        </div>
    </div>
"""

    html += """
    <script>
        function switchTab(tabElement, contentId) {
            // Find parent tabs container
            const tabs = tabElement.parentElement;
            const clusterBody = tabs.parentElement;

            // Deactivate all tabs and contents in this cluster
            tabs.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            clusterBody.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Activate selected tab and content
            tabElement.classList.add('active');
            document.getElementById(contentId).classList.add('active');
        }
    </script>
</body>
</html>
"""
    return html


def process_prefix(prefix_id: str, results_dir: Path, output_dir: Path, top_k: int = 10):
    """Process a single prefix and generate visualization."""
    print(f"Processing {prefix_id}...")

    data = load_prefix_data(prefix_id, results_dir)

    if data['embeddings'] is None or data['attributions'] is None:
        print(f"  Skipping: missing embeddings or attributions")
        return False

    if data['clustering'] is None:
        print(f"  Skipping: missing clustering data")
        return False

    gamma = data['clustering'].get('gamma', 0.5)
    components = data['clustering'].get('components', {})

    clusters = []
    for cluster_id in sorted(components.keys(), key=lambda x: int(x)):
        viz = build_cluster_visualization(cluster_id, data, gamma, top_k)
        if viz:
            clusters.append(viz)

    if not clusters:
        print(f"  Skipping: no clusters found")
        return False

    html = generate_html(prefix_id, data['prefix_text'], clusters, data)

    output_file = output_dir / f"{prefix_id}_clusters.html"
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"  Saved: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize cluster contents')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for HTML files')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Process only this prefix (e.g., cloze_0001)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top nearest continuations to show')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all prefixes
    embed_dir = results_dir / "4_feature_extraction" / "embeddings"
    if not embed_dir.exists():
        print(f"Error: Embeddings directory not found: {embed_dir}")
        return

    if args.prefix:
        prefixes = [args.prefix]
    else:
        prefixes = sorted(set(
            f.stem.replace('_embeddings', '').replace('_meta', '')
            for f in embed_dir.glob("*.npy")
        ))

    print(f"Found {len(prefixes)} prefixes to process")

    success_count = 0
    for prefix_id in prefixes:
        if process_prefix(prefix_id, results_dir, output_dir, args.top_k):
            success_count += 1

    print(f"\nDone! Generated {success_count}/{len(prefixes)} visualizations in {output_dir}")


if __name__ == '__main__':
    main()
