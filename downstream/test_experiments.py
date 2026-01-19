#!/usr/bin/env python3
"""End-to-end test of experiments with synthetic data.

Tests Exp 1, 2, 4, 6 with synthetic embeddings/attributions.
"""

import sys
from pathlib import Path
import tempfile

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def generate_synthetic_data(n_samples=100, n_clusters=3, seed=42):
    """Generate synthetic data with known cluster structure."""
    np.random.seed(seed)
    
    d_e = 32  # Embedding dim
    d_a = 64  # Attribution dim
    
    samples_per_cluster = n_samples // n_clusters
    
    embeddings = []
    attributions = []
    labels = []
    
    for c in range(n_clusters):
        # Create cluster centers
        e_center = np.random.randn(d_e) * 2
        a_center = np.random.randn(d_a) * 2
        
        # Generate samples around centers
        e_samples = np.random.randn(samples_per_cluster, d_e) * 0.5 + e_center
        a_samples = np.random.randn(samples_per_cluster, d_a) * 0.5 + a_center
        
        embeddings.append(e_samples)
        attributions.append(a_samples)
        
        # Labels: cluster 0 = mostly 0, cluster 1 = mixed, cluster 2 = mostly 1
        if c == 0:
            cluster_labels = np.random.binomial(1, 0.2, samples_per_cluster)
        elif c == 1:
            cluster_labels = np.random.binomial(1, 0.5, samples_per_cluster)
        else:
            cluster_labels = np.random.binomial(1, 0.8, samples_per_cluster)
        
        labels.append(cluster_labels)
    
    return (
        np.vstack(embeddings),
        np.vstack(attributions),
        np.concatenate(labels),
    )


def test_exp1():
    """Test Experiment 1: Clustering Quality."""
    print("\n[1/4] Testing Experiment 1: Clustering Quality...")
    
    from downstream.experiments.exp1_clustering_quality import run_experiment_1
    
    embeddings, attributions, labels = generate_synthetic_data(n_samples=60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_experiment_1(
            embeddings, attributions,
            labels=labels,
            beta_values=[0.5, 1.0, 2.0],
            gamma_values=[0.3, 0.5, 0.7],
            output_dir=Path(tmpdir),
        )
    
    print(f"  Sweep configs tested: {len(results['sweep_summaries'])}")
    print(f"  Best config: beta={results['best_config']['beta']}, gamma={results['best_config']['gamma']}")
    print(f"  Best harmonic silhouette: {results['best_config']['harmonic_silhouette']:.4f}")
    print("  ✓ Experiment 1 OK")
    
    return results


def test_exp2(optimal_beta, optimal_gamma):
    """Test Experiment 2: Binary vs RD."""
    print("\n[2/4] Testing Experiment 2: Binary vs RD...")
    
    from downstream.experiments.exp2_binary_vs_rd import run_experiment_2
    
    embeddings, attributions, labels = generate_synthetic_data(n_samples=60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_experiment_2(
            embeddings, attributions, labels,
            optimal_beta=optimal_beta,
            optimal_gamma=optimal_gamma,
            output_dir=Path(tmpdir),
        )
    
    print(f"  Methods compared: {len(results['results_table'])}")
    for row in results['results_table'][:3]:
        print(f"    {row['method']}: K={row['n_clusters']}, sil={row['harmonic_silhouette']:.3f}")
    
    print(f"  RD better silhouette: {results['insights']['rd_better_silhouette']}")
    print(f"  RD finds substructure: {results['insights']['rd_finds_substructure']}")
    print("  ✓ Experiment 2 OK")
    
    return results


def test_exp4(optimal_beta, optimal_gamma):
    """Test Experiment 4: Cross-Prefix Motif."""
    print("\n[3/4] Testing Experiment 4: Cross-Prefix Motif...")
    
    from downstream.experiments.exp4_cross_prefix_motif import run_experiment_4
    
    # Generate per-prompt data
    np.random.seed(42)
    per_prompt_data = {}
    
    for i in range(5):  # 5 prompts
        e, a, _ = generate_synthetic_data(n_samples=15, n_clusters=2, seed=42+i)
        per_prompt_data[f"prompt_{i}"] = {
            "embeddings": e,
            "attributions": a,
            "continuations": [{"text": f"cont_{j}"} for j in range(15)],
        }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_experiment_4(
            per_prompt_data,
            optimal_beta, optimal_gamma,
            n_global_motifs=3,
            output_dir=Path(tmpdir),
        )
    
    print(f"  Prompts clustered: {results['n_prompts_clustered']}")
    print(f"  Total centroids: {results['n_centroids']}")
    print(f"  Global motifs: {len(results['global_motifs'])}")
    
    for motif in results['global_motifs'][:2]:
        print(f"    Motif {motif['motif_id']}: {motif['n_prompts']} prompts, sim={motif['attribution_similarity']:.3f}")
    
    print("  ✓ Experiment 4 OK")
    
    return results


def test_exp6(optimal_beta, optimal_gamma):
    """Test Experiment 6: Hallucination Application."""
    print("\n[4/4] Testing Experiment 6: Hallucination Application...")
    
    from downstream.experiments.exp6_hallucination_application import run_experiment_6
    
    # Generate mock results
    np.random.seed(42)
    data = []
    
    for q_idx in range(5):  # 5 questions
        e, a, labels = generate_synthetic_data(n_samples=10, n_clusters=2, seed=42+q_idx)
        
        continuations = []
        for i in range(10):
            continuations.append({
                "text": f"answer_{i}",
                "is_correct": bool(labels[i]),
                "probability": np.random.rand(),
            })
        
        data.append({
            "question_idx": q_idx,
            "embeddings": e,
            "attributions": a,
            "continuations": continuations,
        })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_experiment_6(
            data,
            optimal_beta, optimal_gamma,
            output_dir=Path(tmpdir),
        )
    
    print(f"  Total clusters: {results['summary']['n_total_clusters']}")
    print(f"  Correct clusters: {results['summary']['n_correct_clusters']}")
    print(f"  Shortcut clusters: {results['summary']['n_shortcut_clusters']}")
    print(f"  Faithful clusters: {results['summary']['n_faithful_clusters']}")
    print("  ✓ Experiment 6 OK")
    
    return results


def main():
    print("=" * 60)
    print("DOWNSTREAM EXPERIMENTS END-TO-END TEST")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test Exp 1
    try:
        exp1_results = test_exp1()
        optimal_beta = exp1_results['best_config']['beta']
        optimal_gamma = exp1_results['best_config']['gamma']
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
        optimal_beta, optimal_gamma = 1.0, 0.5
    
    # Test Exp 2
    try:
        test_exp2(optimal_beta, optimal_gamma)
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # Test Exp 4
    try:
        test_exp4(optimal_beta, optimal_gamma)
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    # Test Exp 6
    try:
        test_exp6(optimal_beta, optimal_gamma)
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/4 experiments passed")
    if failed > 0:
        print(f"         {failed} experiments failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

