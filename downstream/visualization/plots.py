"""Visualization functions for experiment results."""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def plot_rd_tradeoff(
    sweep_summaries: List[Dict],
    output_file: Path = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """Plot rate-distortion trade-off curves.
    
    Args:
        sweep_summaries: List of summary dicts from Exp1
        output_file: Output file path (displays if None)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Group by beta
    by_beta = {}
    for s in sweep_summaries:
        beta = s["beta"]
        if beta not in by_beta:
            by_beta[beta] = []
        by_beta[beta].append(s)
    
    # Plot 1: Rate vs Distortion for different betas
    ax1 = axes[0]
    for beta, items in sorted(by_beta.items()):
        gammas = [s["gamma"] for s in items]
        sil_scores = [s["harmonic_silhouette"] for s in items]
        ax1.plot(gammas, sil_scores, 'o-', label=f'β={beta}', alpha=0.7)
    
    ax1.set_xlabel('γ (semantic weight)')
    ax1.set_ylabel('Harmonic Silhouette')
    ax1.set_title('Clustering Quality vs γ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of scores
    ax2 = axes[1]
    
    betas = sorted(set(s["beta"] for s in sweep_summaries))
    gammas = sorted(set(s["gamma"] for s in sweep_summaries))
    
    scores = np.zeros((len(betas), len(gammas)))
    for s in sweep_summaries:
        i = betas.index(s["beta"])
        j = gammas.index(s["gamma"])
        scores[i, j] = s["harmonic_silhouette"]
    
    im = ax2.imshow(scores, aspect='auto', cmap='viridis')
    ax2.set_xticks(range(len(gammas)))
    ax2.set_xticklabels([f'{g:.1f}' for g in gammas])
    ax2.set_yticks(range(len(betas)))
    ax2.set_yticklabels([f'{b:.1f}' for b in betas])
    ax2.set_xlabel('γ')
    ax2.set_ylabel('β')
    ax2.set_title('Harmonic Silhouette Heatmap')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tsne_clusters(
    embeddings: np.ndarray,
    assignments: np.ndarray,
    labels: np.ndarray = None,
    output_file: Path = None,
    figsize: Tuple[int, int] = (12, 5),
    perplexity: int = 30,
    title: str = "Cluster Visualization",
):
    """Plot t-SNE visualization of clusters.
    
    Args:
        embeddings: Data embeddings
        assignments: Cluster assignments
        labels: Optional ground truth labels
        output_file: Output file path
        figsize: Figure size
        perplexity: t-SNE perplexity
        title: Plot title
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)
    
    n_plots = 2 if labels is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: By cluster assignment
    scatter = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=assignments, cmap='tab20', alpha=0.7, s=20
    )
    axes[0].set_title(f'{title} - Clusters')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Plot 2: By label (if provided)
    if labels is not None:
        scatter2 = axes[1].scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap='RdYlGn', alpha=0.7, s=20
        )
        axes[1].set_title(f'{title} - Ground Truth')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_binary_vs_rd(
    results_table: List[Dict],
    output_file: Path = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot comparison of binary vs RD clustering.
    
    Args:
        results_table: Results from Exp2
        output_file: Output file path
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    methods = [r["method"] for r in results_table]
    sil_h = [r["harmonic_silhouette"] for r in results_table]
    sem_coh = [r["semantic_coherence"] for r in results_table]
    attr_con = [r["attribution_consistency"] for r in results_table]
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width, sil_h, width, label='Harmonic Silhouette', color='steelblue')
    bars2 = ax.bar(x, sem_coh, width, label='Semantic Coherence', color='forestgreen')
    bars3 = ax.bar(x + width, attr_con, width, label='Attribution Consistency', color='coral')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Binary vs RD Clustering Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cross_prefix_motifs(
    global_motifs: List[Dict],
    output_file: Path = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot cross-prefix motif analysis.
    
    Args:
        global_motifs: Results from Exp4
        output_file: Output file path
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    motif_ids = [m["motif_id"] for m in global_motifs]
    n_prompts = [m["n_prompts"] for m in global_motifs]
    attr_sim = [m["attribution_similarity"] for m in global_motifs]
    sem_sim = [m["semantic_similarity"] for m in global_motifs]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Number of prompts per motif
    ax1 = axes[0]
    bars = ax1.bar(motif_ids, n_prompts, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Motif ID')
    ax1.set_ylabel('Number of Prompts')
    ax1.set_title('Cross-Prefix Motif Generality')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Similarity comparison
    ax2 = axes[1]
    x = np.arange(len(motif_ids))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, attr_sim, width, label='Attribution', color='coral')
    bars2 = ax2.bar(x + width/2, sem_sim, width, label='Semantic', color='forestgreen')
    
    ax2.set_xlabel('Motif ID')
    ax2.set_ylabel('Intra-Motif Similarity')
    ax2.set_title('Motif Coherence')
    ax2.set_xticks(x)
    ax2.set_xticklabels(motif_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_safety_analysis(
    refusal_clusters: List[Dict],
    output_file: Path = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot safety experiment analysis.
    
    Args:
        refusal_clusters: Results from Exp5
        output_file: Output file path
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    with_motif = [c for c in refusal_clusters if c.get("has_safety_motif")]
    without_motif = [c for c in refusal_clusters if not c.get("has_safety_motif")]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Cluster distribution
    ax1 = axes[0]
    sizes = [len(with_motif), len(without_motif)]
    labels = ['With Safety Motif', 'Without Safety Motif']
    colors = ['forestgreen', 'coral']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Refusal Cluster Types')
    
    # Plot 2: Robustness comparison
    ax2 = axes[1]
    robustness_with = [c["robustness"] for c in with_motif if c.get("robustness") is not None]
    robustness_without = [c["robustness"] for c in without_motif if c.get("robustness") is not None]
    
    if robustness_with and robustness_without:
        ax2.boxplot([robustness_with, robustness_without], labels=labels)
        ax2.set_ylabel('Robustness Score')
        ax2.set_title('Robustness by Motif Presence')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No robustness data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_hallucination_analysis(
    cluster_analysis: List[Dict],
    output_file: Path = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """Plot hallucination experiment analysis.
    
    Args:
        cluster_analysis: Results from Exp6
        output_file: Output file path
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    correctness_rates = [c["correctness_rate"] for c in cluster_analysis]
    cluster_sizes = [c["cluster_size"] for c in cluster_analysis]
    
    shortcut = [c for c in cluster_analysis if c.get("reasoning_type") == "shortcut"]
    faithful = [c for c in cluster_analysis if c.get("reasoning_type") == "faithful"]
    other = [c for c in cluster_analysis if c.get("reasoning_type") not in ["shortcut", "faithful"]]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Correctness rate distribution
    ax1 = axes[0]
    ax1.scatter(
        [c["correctness_rate"] for c in shortcut],
        [c["cluster_size"] for c in shortcut],
        c='coral', label='Shortcut', alpha=0.7, s=100
    )
    ax1.scatter(
        [c["correctness_rate"] for c in faithful],
        [c["cluster_size"] for c in faithful],
        c='forestgreen', label='Faithful', alpha=0.7, s=100
    )
    ax1.scatter(
        [c["correctness_rate"] for c in other],
        [c["cluster_size"] for c in other],
        c='gray', label='Other', alpha=0.5, s=50
    )
    ax1.axvline(x=0.7, color='black', linestyle='--', alpha=0.5, label='Correct threshold')
    ax1.set_xlabel('Correctness Rate')
    ax1.set_ylabel('Cluster Size')
    ax1.set_title('Cluster Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reasoning type distribution
    ax2 = axes[1]
    counts = [len(shortcut), len(faithful), len(other)]
    labels = ['Shortcut', 'Faithful', 'Other']
    colors = ['coral', 'forestgreen', 'gray']
    ax2.bar(labels, counts, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Clusters')
    ax2.set_title('Reasoning Type Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (count, label) in enumerate(zip(counts, labels)):
        ax2.annotate(f'{count}', xy=(i, count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

