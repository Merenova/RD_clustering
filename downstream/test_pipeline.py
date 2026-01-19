#!/usr/bin/env python3
"""Minimal test script for the downstream pipeline.

Tests each component with tiny samples to verify imports and basic functionality.
"""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def test_config():
    """Test config imports."""
    print("\n[1/8] Testing config...")
    from downstream.config import (
        BASE_MODEL_NAME, INSTRUCT_MODEL_NAME, TRANSCODER_SET,
        ROLLOUT_CONFIG, SAFETY_CONFIG, HALLUCINATION_CONFIG,
    )
    print(f"  Base model: {BASE_MODEL_NAME}")
    print(f"  Instruct model: {INSTRUCT_MODEL_NAME}")
    print(f"  Transcoder: {TRANSCODER_SET}")
    print("  ✓ Config OK")


def test_data_loaders():
    """Test data loaders."""
    print("\n[2/8] Testing data loaders...")
    
    # HarmBench
    from downstream.data.load_harmbench import download_harmbench
    behaviors = download_harmbench()
    print(f"  HarmBench: {len(behaviors)} behaviors")
    
    # PopQA
    from downstream.data.load_popqa import load_popqa_questions
    try:
        questions = load_popqa_questions(n_samples=5)
        print(f"  PopQA: {len(questions)} questions")
    except FileNotFoundError:
        print("  PopQA: File not found (expected if data not downloaded)")
    
    print("  ✓ Data loaders OK")


def test_labeling():
    """Test labeling modules."""
    print("\n[3/8] Testing labeling...")
    
    from downstream.labeling.refusal_judge import is_refusal_keyword
    from downstream.labeling.correctness_judge import is_correct
    
    # Test refusal detection
    assert is_refusal_keyword("I cannot help with that") == True
    assert is_refusal_keyword("Sure, here's how to do it") == False
    print("  Refusal detection: OK")
    
    # Test correctness detection
    assert is_correct("The answer is Paris", ["Paris", "paris"]) == True
    assert is_correct("I think it's London", ["Paris"]) == False
    print("  Correctness detection: OK")
    
    print("  ✓ Labeling OK")


def test_filtering():
    """Test filtering modules."""
    print("\n[4/8] Testing filtering...")
    
    from downstream.filtering.hallucination_filter import compute_binary_entropy
    from downstream.filtering.safety_filter import compute_disagreement
    
    # Test entropy
    h = compute_binary_entropy(0.5, 0.5)
    assert 0.99 < h <= 1.0, f"Expected ~1.0, got {h}"
    print(f"  Binary entropy (0.5, 0.5): {h:.4f}")
    
    h = compute_binary_entropy(1.0, 0.0)
    assert h == 0.0, f"Expected 0.0, got {h}"
    print(f"  Binary entropy (1.0, 0.0): {h:.4f}")
    
    # Test disagreement
    d = compute_disagreement(0.2, 0.8)
    assert abs(d - 0.6) < 1e-9, f"Expected 0.6, got {d}"
    print(f"  Disagreement (0.2, 0.8): {d:.2f}")
    
    print("  ✓ Filtering OK")


def test_clustering():
    """Test clustering modules."""
    print("\n[5/8] Testing clustering...")
    
    from downstream.clustering.config_selection import (
        compute_silhouette,
        compute_harmonic_silhouette,
    )
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 50
    
    # Two clusters
    embeddings = np.vstack([
        np.random.randn(25, 10) + np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(25, 10) + np.array([-2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ])
    attributions = np.vstack([
        np.random.randn(25, 20) + np.array([1] * 10 + [0] * 10),
        np.random.randn(25, 20) + np.array([0] * 10 + [1] * 10),
    ])
    assignments = np.array([0] * 25 + [1] * 25)
    
    sil = compute_silhouette(embeddings, assignments)
    print(f"  Silhouette score: {sil:.4f}")
    
    h_sil = compute_harmonic_silhouette(embeddings, attributions, assignments)
    print(f"  Harmonic silhouette: {h_sil:.4f}")
    
    print("  ✓ Clustering metrics OK")


def test_rd_clustering():
    """Test RD clustering wrapper."""
    print("\n[6/8] Testing RD clustering...")
    
    from downstream.clustering.rd_wrapper import run_rd_clustering
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 30
    
    embeddings = np.vstack([
        np.random.randn(15, 8) + np.array([2, 0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(15, 8) + np.array([-2, 0, 0, 0, 0, 0, 0, 0]),
    ])
    attributions = np.vstack([
        np.random.randn(15, 16) + np.array([1] * 8 + [0] * 8),
        np.random.randn(15, 16) + np.array([0] * 8 + [1] * 8),
    ])
    
    result = run_rd_clustering(
        embeddings, attributions,
        beta=1.0, gamma=0.5,
        K_max=5,
        max_iterations=10,
    )
    
    print(f"  N clusters: {result['n_components']}")
    print(f"  N iterations: {result['n_iterations']}")
    print(f"  Assignments shape: {result['assignments'].shape}")
    
    print("  ✓ RD clustering OK")


def test_experiments():
    """Test experiment modules (import only)."""
    print("\n[7/8] Testing experiment imports...")
    
    from downstream.experiments.exp1_clustering_quality import run_experiment_1
    from downstream.experiments.exp2_binary_vs_rd import run_experiment_2
    from downstream.experiments.exp3_steering_validation import run_experiment_3
    from downstream.experiments.exp4_cross_prefix_motif import run_experiment_4
    from downstream.experiments.exp5_safety_application import run_experiment_5
    from downstream.experiments.exp6_hallucination_application import run_experiment_6
    
    print("  All 6 experiment modules imported")
    print("  ✓ Experiments OK")


def test_visualization():
    """Test visualization imports."""
    print("\n[8/8] Testing visualization imports...")
    
    from downstream.visualization.plots import (
        plot_rd_tradeoff,
        plot_tsne_clusters,
        plot_binary_vs_rd,
        plot_cross_prefix_motifs,
    )
    
    print("  All visualization functions imported")
    print("  ✓ Visualization OK")


def main():
    print("=" * 60)
    print("DOWNSTREAM PIPELINE TEST")
    print("=" * 60)
    
    tests = [
        test_config,
        test_data_loaders,
        test_labeling,
        test_filtering,
        test_clustering,
        test_rd_clustering,
        test_experiments,
        test_visualization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

