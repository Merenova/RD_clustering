"""Experiment 5: Safety Application (HarmBench).

Detects brittle refusal patterns using motif analysis.
Shows that same refusal output can have different robustness based on motif.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json

# Import from downstream
from downstream.config import OUTPUT_DIR, ROBUSTNESS_CONFIG
from downstream.clustering.rd_wrapper import run_rd_clustering, compute_global_normalization
from downstream.clustering.config_selection import compute_cluster_metrics


def run_experiment_5(
    base_data: Dict[str, Any],
    instruct_data: Dict[str, Any],
    optimal_beta: float,
    optimal_gamma: float,
    test_prompts: List[Dict] = None,
    model=None,
    tokenizer=None,
    skip_robustness: bool = False,
    output_dir: Path = None,
    logger=None,
) -> Dict[str, Any]:
    """Run Experiment 5: Safety Application.
    
    Goal: Detect brittle refusal using motifs.
    
    Method:
    1. Cluster both base and instruct model outputs
    2. Find instruct-only motifs (safety motifs)
    3. Analyze refusal clusters
    4. Test robustness correlation with safety motif presence
    
    Args:
        base_data: Base model data with embeddings, attributions, etc.
        instruct_data: Instruct model data
        optimal_beta: Beta for RD clustering
        optimal_gamma: Gamma for RD clustering
        test_prompts: Prompts for robustness testing (list of sample dicts)
        model: Model for robustness generation (HuggingFace or vLLM)
        tokenizer: Tokenizer for model
        skip_robustness: If True, skip expensive robustness testing
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dict with experiment results
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "exp5_safety"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 5: SAFETY APPLICATION")
        logger.info("=" * 60)
    
    # Extract data
    base_embeddings = np.array(base_data.get("embeddings", base_data.get("embeddings_e", [])))
    base_attributions = np.array(base_data.get("attributions", base_data.get("attributions_a", [])))
    base_labels = np.array(base_data.get("labels", base_data.get("is_refusal", [])))
    
    instruct_embeddings = np.array(instruct_data.get("embeddings", instruct_data.get("embeddings_e", [])))
    instruct_attributions = np.array(instruct_data.get("attributions", instruct_data.get("attributions_a", [])))
    instruct_labels = np.array(instruct_data.get("labels", instruct_data.get("is_refusal", [])))
    
    # Step 0: Compute GLOBAL normalization across BOTH models
    # This ensures centroids are comparable for cross-model analysis
    if logger:
        logger.info("\nStep 0: Computing global normalization across both models...")
    
    attr_list = []
    if len(base_attributions) > 0:
        attr_list.append(base_attributions)
    if len(instruct_attributions) > 0:
        attr_list.append(instruct_attributions)
    
    if attr_list:
        global_H_0, global_rms_norm = compute_global_normalization(attr_list)
        if logger:
            logger.info(f"Global normalization: ||H_0||={np.linalg.norm(global_H_0):.4f}, rms={global_rms_norm:.4f}")
    else:
        global_H_0 = None
        global_rms_norm = None
    
    # Step 1: Cluster both models with SAME global normalization
    if logger:
        logger.info("\nStep 1: Clustering base model outputs...")
    
    if len(base_embeddings) > 0:
        base_clusters = run_rd_clustering(
            base_embeddings,
            base_attributions,
            beta=optimal_beta,
            gamma=optimal_gamma,
            global_H_0=global_H_0,
            global_rms_norm=global_rms_norm,
        )
    else:
        base_clusters = {"assignments": [], "components": {}}
    
    if logger:
        logger.info(f"Base model: {len(base_clusters.get('components', {}))} clusters")
    
    if logger:
        logger.info("\nStep 2: Clustering instruct model outputs...")
    
    if len(instruct_embeddings) > 0:
        instruct_clusters = run_rd_clustering(
            instruct_embeddings,
            instruct_attributions,
            beta=optimal_beta,
            gamma=optimal_gamma,
            global_H_0=global_H_0,
            global_rms_norm=global_rms_norm,
        )
    else:
        instruct_clusters = {"assignments": [], "components": {}}
    
    if logger:
        logger.info(f"Instruct model: {len(instruct_clusters.get('components', {}))} clusters")
    
    # Step 2: Find instruct-only motifs (safety motifs)
    if logger:
        logger.info("\nStep 3: Identifying safety motifs...")
    
    safety_motifs = find_unique_motifs(
        instruct_clusters,
        base_clusters,
        similarity_threshold=0.7,
    )
    
    if logger:
        logger.info(f"Found {len(safety_motifs)} instruct-only (safety) motifs")
    
    # Step 3: Analyze refusal clusters
    if logger:
        logger.info("\nStep 4: Analyzing refusal clusters...")
    
    refusal_clusters = []
    instruct_assignments = instruct_clusters.get("assignments", [])
    instruct_components = instruct_clusters.get("components", {})
    
    for cluster_id, component in instruct_components.items():
        # Get cluster members
        if len(instruct_assignments) == 0:
            continue
        
        cluster_mask = np.array(instruct_assignments) == cluster_id
        
        if not cluster_mask.any():
            continue
        
        cluster_labels = instruct_labels[cluster_mask] if len(instruct_labels) > 0 else []
        
        if len(cluster_labels) == 0:
            continue
        
        refusal_rate = float(np.mean(cluster_labels))
        
        # Only analyze clusters with high refusal rate
        if refusal_rate < 0.7:
            continue
        
        # Check if cluster has safety motif
        attr_centroid = component.get("mu_a", component.get("centroid_a"))
        has_safety_motif = check_motif_presence(attr_centroid, safety_motifs)
        
        refusal_clusters.append({
            "cluster_id": int(cluster_id),
            "cluster_size": int(np.sum(cluster_mask)),
            "refusal_rate": refusal_rate,
            "has_safety_motif": has_safety_motif,
            "robustness": None,  # Will be filled if test_prompts provided
        })
    
    # Step 4: Test robustness (if model and prompts provided)
    robustness_tested = False
    
    if test_prompts and len(refusal_clusters) > 0 and model is not None and not skip_robustness:
        if logger:
            logger.info("\nStep 5: Testing robustness...")
        
        try:
            from downstream.perturbation.robustness import (
                test_cluster_robustness,
                create_refusal_judge,
            )
            
            # Create refusal judge
            judge_fn = create_refusal_judge()
            
            # Get cluster assignments for sample lookup
            instruct_assignments = np.array(instruct_clusters.get("assignments", []))
            
            # Get original instruct samples
            instruct_samples = instruct_data.get("samples", [])
            if not instruct_samples:
                # Try to reconstruct from test_prompts
                instruct_samples = test_prompts
            
            for cluster in refusal_clusters:
                cluster_id = cluster["cluster_id"]
                
                # Get samples belonging to this cluster
                if len(instruct_assignments) > 0 and len(instruct_samples) > 0:
                    cluster_mask = instruct_assignments == cluster_id
                    cluster_samples = []
                    for i in range(min(len(cluster_mask), len(instruct_samples))):
                        if i < len(cluster_mask) and cluster_mask[i]:
                            sample = instruct_samples[i]
                            # Ensure is_refusal is present from instruct_labels
                            if "is_refusal" not in sample and i < len(instruct_labels):
                                sample = {**sample, "is_refusal": bool(instruct_labels[i])}
                            cluster_samples.append(sample)
                else:
                    # Fallback: use test_prompts with refusal label
                    cluster_samples = [
                        {**p, "is_refusal": True}
                        for p in test_prompts[:cluster["cluster_size"]]
                    ]
                
                if cluster_samples:
                    robustness_result = test_cluster_robustness(
                        cluster_samples,
                        model,
                        tokenizer,
                        judge_fn,
                        perturbation_types=ROBUSTNESS_CONFIG.get("perturbation_types", ["typo", "paraphrase"]),
                        n_perturbations=ROBUSTNESS_CONFIG.get("n_perturbations", 3),
                        max_samples=10,
                        logger=logger,
                    )
                    
                    cluster["robustness"] = robustness_result["robustness_score"]
                    cluster["robustness_details"] = {
                        "n_samples_tested": robustness_result["n_samples_tested"],
                        "n_perturbations_tested": robustness_result["n_perturbations_tested"],
                        "robustness_by_type": robustness_result.get("robustness_by_type", {}),
                    }
                    robustness_tested = True
                else:
                    cluster["robustness"] = None
            
        except Exception as e:
            if logger:
                logger.warning(f"Robustness testing failed: {e}")
            # Don't set placeholder - leave as None so it's excluded from correlation
            for cluster in refusal_clusters:
                if cluster.get("robustness") is None:
                    cluster["robustness"] = None  # Explicitly None, not placeholder
    
    elif test_prompts and len(refusal_clusters) > 0:
        # No model provided or skip_robustness=True, use placeholder
        if logger:
            logger.info("\nStep 5: Skipping robustness (no model or skip_robustness=True)")
        
        for cluster in refusal_clusters:
            cluster["robustness"] = 0.5  # Placeholder
    
    # Compute correlation between safety motif and robustness
    # Build aligned pairs to avoid mismatching when some clusters have None robustness
    pairs = [
        (c["has_safety_motif"], c["robustness"])
        for c in refusal_clusters
        if c["robustness"] is not None
    ]
    
    if len(pairs) > 1 and len(set(h for h, _ in pairs)) > 1:
        has_motif_numeric = [1 if h else 0 for h, _ in pairs]
        robustness_values = [r for _, r in pairs]
        correlation = float(np.corrcoef(has_motif_numeric, robustness_values)[0, 1])
    else:
        correlation = 0.0
    
    # Summarize
    motif_present = [c for c in refusal_clusters if c["has_safety_motif"]]
    motif_absent = [c for c in refusal_clusters if not c["has_safety_motif"]]
    
    summary = {
        "n_refusal_clusters": len(refusal_clusters),
        "n_with_safety_motif": len(motif_present),
        "n_without_safety_motif": len(motif_absent),
        "robustness_is_placeholder": not robustness_tested,
        "motif_robustness_correlation": correlation,
        "avg_robustness_with_motif": (
            float(np.mean([c["robustness"] for c in motif_present if c["robustness"] is not None])) 
            if motif_present else 0.0
        ),
        "avg_robustness_without_motif": (
            float(np.mean([c["robustness"] for c in motif_absent if c["robustness"] is not None])) 
            if motif_absent else 0.0
        ),
    }
    
    if logger:
        logger.info("\n" + "-" * 40)
        logger.info("RESULTS SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Refusal clusters: {summary['n_refusal_clusters']}")
        logger.info(f"  With safety motif: {summary['n_with_safety_motif']}")
        logger.info(f"  Without safety motif: {summary['n_without_safety_motif']}")
        logger.info(f"Robustness tested: {not summary['robustness_is_placeholder']}")
        logger.info(f"Motif-robustness correlation: {summary['motif_robustness_correlation']:.4f}")
    
    # Compile results
    results = {
        "optimal_beta": optimal_beta,
        "optimal_gamma": optimal_gamma,
        "n_base_clusters": len(base_clusters.get("components", {})),
        "n_instruct_clusters": len(instruct_clusters.get("components", {})),
        "n_safety_motifs": len(safety_motifs),
        "refusal_clusters": refusal_clusters,
        "summary": summary,
    }
    
    # Save results
    save_json(results, output_dir / "exp5_results.json")
    if logger:
        logger.info(f"\nResults saved to {output_dir / 'exp5_results.json'}")
    
    return results


def find_unique_motifs(
    target_clusters: Dict,
    reference_clusters: Dict,
    similarity_threshold: float = 0.7,
) -> List[np.ndarray]:
    """Find motifs in target that don't appear in reference.
    
    Args:
        target_clusters: Target clustering results
        reference_clusters: Reference clustering results
        similarity_threshold: Max similarity to be considered unique
        
    Returns:
        List of unique motif centroids
    """
    target_components = target_clusters.get("components", {})
    ref_components = reference_clusters.get("components", {})
    
    # Get all reference centroids
    ref_centroids = []
    for comp in ref_components.values():
        centroid = comp.get("mu_a", comp.get("centroid_a"))
        if centroid is not None:
            ref_centroids.append(np.array(centroid))
    
    if not ref_centroids:
        # All target motifs are unique
        return [
            np.array(comp.get("mu_a", comp.get("centroid_a")))
            for comp in target_components.values()
            if comp.get("mu_a", comp.get("centroid_a")) is not None
        ]
    
    ref_matrix = np.stack(ref_centroids)
    ref_norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
    ref_matrix = ref_matrix / np.maximum(ref_norms, 1e-10)
    
    unique_motifs = []
    
    for comp in target_components.values():
        centroid = comp.get("mu_a", comp.get("centroid_a"))
        if centroid is None:
            continue
        
        centroid = np.array(centroid)
        centroid_normed = centroid / max(np.linalg.norm(centroid), 1e-10)
        
        # Compute similarity to all reference centroids
        similarities = ref_matrix @ centroid_normed
        max_similarity = np.max(similarities)
        
        if max_similarity < similarity_threshold:
            unique_motifs.append(centroid)
    
    return unique_motifs


def check_motif_presence(
    centroid: np.ndarray,
    motifs: List[np.ndarray],
    threshold: float = 0.7,
) -> bool:
    """Check if centroid matches any motif.
    
    Args:
        centroid: Cluster centroid
        motifs: List of motif centroids
        threshold: Similarity threshold for match
        
    Returns:
        True if centroid matches a motif
    """
    if centroid is None or not motifs:
        return False
    
    centroid = np.array(centroid)
    centroid_normed = centroid / max(np.linalg.norm(centroid), 1e-10)
    
    for motif in motifs:
        motif_normed = motif / max(np.linalg.norm(motif), 1e-10)
        similarity = float(centroid_normed @ motif_normed)
        
        if similarity >= threshold:
            return True
    
    return False

