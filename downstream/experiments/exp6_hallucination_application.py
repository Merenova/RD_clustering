"""Experiment 6: Hallucination Application (PopQA).

Detects shortcut reasoning vs faithful reasoning in correct answers.
Shows that output-level correctness misses this distinction.
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
from downstream.config import OUTPUT_DIR
from downstream.clustering.rd_wrapper import run_rd_clustering
from downstream.clustering.config_selection import compute_cluster_metrics


def run_experiment_6(
    data: List[Dict[str, Any]],
    optimal_beta: float,
    optimal_gamma: float,
    paraphrase_fn=None,
    model=None,
    tokenizer=None,
    correctness_fn=None,
    skip_robustness: bool = False,
    output_dir: Path = None,
    logger=None,
) -> Dict[str, Any]:
    """Run Experiment 6: Hallucination Application.
    
    Goal: Detect shortcut reasoning vs faithful reasoning.
    
    Method:
    1. Cluster all continuations
    2. Analyze "correct" clusters (high correctness rate)
    3. Test paraphrase robustness for each cluster
    4. Compare attribution patterns between faithful and shortcut clusters
    
    Args:
        data: List of result dicts with embeddings, attributions, correctness labels
        optimal_beta: Beta for RD clustering
        optimal_gamma: Gamma for RD clustering
        paraphrase_fn: Function to generate paraphrases (deprecated, use model instead)
        model: Model for paraphrase testing (HuggingFace or vLLM)
        tokenizer: Tokenizer for model
        correctness_fn: Function(question, answer) -> is_correct for paraphrase testing
        skip_robustness: If True, skip expensive paraphrase robustness testing
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dict with experiment results
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "exp6_hallucination"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 6: HALLUCINATION APPLICATION")
        logger.info("=" * 60)
        logger.info(f"N samples: {sum(len(d.get('continuations', [])) for d in data)}")
        logger.info(f"N questions: {len(data)}")
    
    # Flatten data for clustering
    # Use pooled attributions for cross-question clustering (consistent shape)
    # Position-specific attributions have different shapes per question (different prefix lengths)
    all_embeddings = []
    all_attributions_pooled = []
    all_labels = []  # Correctness labels
    all_probs = []
    sample_info = []  # Track which question each sample belongs to
    
    for q_idx, q_data in enumerate(data):
        embeddings = q_data.get("embeddings", [])
        # Use pooled attributions for cross-question operations
        attributions_pooled = q_data.get("attributions_pooled", q_data.get("attributions", []))
        continuations = q_data.get("continuations", [])
        
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        if isinstance(attributions_pooled, np.ndarray):
            attributions_pooled = attributions_pooled.tolist()
        
        for i, (emb, attr) in enumerate(zip(embeddings, attributions_pooled)):
            if emb is None or attr is None:
                continue
            
            all_embeddings.append(emb)
            all_attributions_pooled.append(attr)
            
            # Get label
            if i < len(continuations):
                cont = continuations[i]
                label = cont.get("is_correct", False) if isinstance(cont, dict) else False
                prob = cont.get("probability", 1.0) if isinstance(cont, dict) else 1.0
            else:
                label = False
                prob = 1.0
            
            all_labels.append(label)
            all_probs.append(prob)
            
            sample_info.append({
                "question_idx": q_idx,
                "continuation_idx": i,
            })
    
    if len(all_embeddings) == 0:
        if logger:
            logger.warning("No valid samples found")
        return {"error": "No valid samples"}
    
    embeddings_e = np.array(all_embeddings)
    attributions_a = np.array(all_attributions_pooled)
    labels = np.array(all_labels, dtype=int)
    # Note: path_probs not used for weighting - use uniform weights
    # Keeping all_probs for potential future use but not passing to clustering
    
    if logger:
        logger.info(f"Total samples for clustering: {len(embeddings_e)}")
        logger.info(f"Correctness rate: {labels.mean():.2%}")
    
    # Step 1: Cluster all continuations (uniform weights)
    if logger:
        logger.info("\nStep 1: Clustering all continuations...")
    
    clusters = run_rd_clustering(
        embeddings_e,
        attributions_a,
        path_probs=None,  # Use uniform weights
        beta=optimal_beta,
        gamma=optimal_gamma,
    )
    
    assignments = clusters["assignments"]
    components = clusters["components"]
    n_clusters = len(components)
    
    if logger:
        logger.info(f"Found {n_clusters} clusters")
    
    # Step 2: Analyze correct-answer clusters
    if logger:
        logger.info("\nStep 2: Analyzing correct-answer clusters...")
    
    cluster_analysis = []
    
    for cluster_id, component in components.items():
        cluster_mask = assignments == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size == 0:
            continue
        
        cluster_labels = labels[cluster_mask]
        correctness_rate = float(np.mean(cluster_labels))
        
        # Get cluster members
        cluster_indices = np.where(cluster_mask)[0]
        cluster_questions = set(sample_info[i]["question_idx"] for i in cluster_indices)
        
        cluster_analysis.append({
            "cluster_id": int(cluster_id),
            "cluster_size": int(cluster_size),
            "correctness_rate": correctness_rate,
            "n_questions": len(cluster_questions),
            "paraphrase_robustness": None,  # Will be computed if paraphrase_fn provided
            "is_correct_cluster": correctness_rate > 0.7,
        })
    
    # Identify correct clusters
    correct_clusters = [c for c in cluster_analysis if c["is_correct_cluster"]]
    
    if logger:
        logger.info(f"Found {len(correct_clusters)} correct-answer clusters (correctness > 70%)")
    
    # Step 3: Test paraphrase robustness
    robustness_tested = False
    
    if model is not None and correctness_fn is not None and not skip_robustness:
        if logger:
            logger.info("\nStep 3: Testing paraphrase robustness...")
        
        try:
            from downstream.perturbation.paraphraser import (
                test_cluster_paraphrase_robustness,
                classify_reasoning_type,
            )
            
            for cluster in correct_clusters:
                cluster_id = cluster["cluster_id"]
                
                # Get samples belonging to this cluster
                cluster_mask = assignments == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                # Build sample list with question info
                # Only include originally CORRECT samples for paraphrase robustness
                # (we want to test if correct answers remain correct, not if wrong stays wrong)
                cluster_samples = []
                for idx in cluster_indices[:20]:  # Check more to find enough correct ones
                    info = sample_info[idx]
                    q_idx = info["question_idx"]
                    c_idx = info["continuation_idx"]
                    
                    if q_idx < len(data):
                        q_data = data[q_idx]
                        question = q_data.get("question", q_data.get("prompt", ""))
                        
                        continuations = q_data.get("continuations", [])
                        if c_idx < len(continuations):
                            cont = continuations[c_idx]
                            answer = cont.get("text", cont.get("continuation", ""))
                            is_correct = cont.get("is_correct", False)
                        else:
                            answer = ""
                            is_correct = False
                        
                        # Only include correct samples for robustness testing
                        if is_correct:
                            cluster_samples.append({
                                "question": question,
                                "answer": answer,
                                "is_correct": is_correct,
                            })
                    
                    # Stop once we have enough correct samples
                    if len(cluster_samples) >= 10:
                        break
                
                if cluster_samples:
                    result = test_cluster_paraphrase_robustness(
                        cluster_samples,
                        model,
                        tokenizer,
                        correctness_fn,
                        n_paraphrases=3,
                        max_samples=5,
                        logger=logger,
                    )
                    
                    cluster["paraphrase_robustness"] = result["avg_robustness_score"]
                    cluster["paraphrase_details"] = {
                        "n_samples_tested": result["n_samples_tested"],
                        "all_correct_rate": result["all_correct_rate"],
                    }
                    robustness_tested = True
                else:
                    cluster["paraphrase_robustness"] = None
                    
        except Exception as e:
            if logger:
                logger.warning(f"Paraphrase robustness testing failed: {e}")
            # Don't set placeholder - leave as None so it's classified as "unknown"
            for cluster in correct_clusters:
                if cluster.get("paraphrase_robustness") is None:
                    cluster["paraphrase_robustness"] = None  # Explicitly None, not placeholder
    
    elif not skip_robustness:
        if logger:
            logger.info("\nStep 3: Skipping paraphrase robustness (no model/correctness_fn)")
    
    # Step 4: Classify clusters as shortcut vs faithful
    if logger:
        logger.info("\nStep 4: Classifying shortcut vs faithful clusters...")
    
    shortcut_clusters = []
    faithful_clusters = []
    
    unknown_clusters = []
    
    for cluster in correct_clusters:
        robustness = cluster.get("paraphrase_robustness")
        
        if robustness is None:
            # No robustness data - don't assume faithful/shortcut
            cluster["reasoning_type"] = "unknown"
            unknown_clusters.append(cluster)
        elif robustness < 0.5:
            cluster["reasoning_type"] = "shortcut"
            shortcut_clusters.append(cluster)
        else:
            cluster["reasoning_type"] = "faithful"
            faithful_clusters.append(cluster)
    
    # Step 5: Compare attribution patterns
    if logger:
        logger.info("\nStep 5: Comparing attribution patterns...")
    
    attribution_analysis = {}
    
    if shortcut_clusters and faithful_clusters:
        # Get attribution centroids
        shortcut_centroids = []
        faithful_centroids = []
        
        for cluster in shortcut_clusters:
            comp = components.get(cluster["cluster_id"])
            if comp:
                centroid = comp.get("mu_a", comp.get("centroid_a"))
                if centroid is not None:
                    shortcut_centroids.append(np.array(centroid))
        
        for cluster in faithful_clusters:
            comp = components.get(cluster["cluster_id"])
            if comp:
                centroid = comp.get("mu_a", comp.get("centroid_a"))
                if centroid is not None:
                    faithful_centroids.append(np.array(centroid))
        
        if shortcut_centroids and faithful_centroids:
            shortcut_mean = np.mean(shortcut_centroids, axis=0)
            faithful_mean = np.mean(faithful_centroids, axis=0)
            
            # Compute difference
            diff = faithful_mean - shortcut_mean
            top_diff_indices = np.argsort(np.abs(diff))[-20:]  # Top 20 differing features
            
            attribution_analysis = {
                "shortcut_mean_norm": float(np.linalg.norm(shortcut_mean)),
                "faithful_mean_norm": float(np.linalg.norm(faithful_mean)),
                "diff_norm": float(np.linalg.norm(diff)),
                "top_diff_features": top_diff_indices.tolist(),
            }
    
    # Summary
    # Only count classified clusters for brittle percentage
    classified_correct = len(shortcut_clusters) + len(faithful_clusters)
    summary = {
        "n_total_clusters": n_clusters,
        "n_correct_clusters": len(correct_clusters),
        "n_shortcut_clusters": len(shortcut_clusters),
        "n_faithful_clusters": len(faithful_clusters),
        "n_unknown_clusters": len(unknown_clusters),
        "pct_correct_but_brittle": (
            len(shortcut_clusters) / classified_correct * 100 
            if classified_correct > 0 else 0
        ),
        "robustness_is_placeholder": not robustness_tested,
    }
    
    if logger:
        logger.info("\n" + "-" * 40)
        logger.info("RESULTS SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Total clusters: {summary['n_total_clusters']}")
        logger.info(f"Correct clusters: {summary['n_correct_clusters']}")
        logger.info(f"  - Shortcut: {summary['n_shortcut_clusters']}")
        logger.info(f"  - Faithful: {summary['n_faithful_clusters']}")
        logger.info(f"  - Unknown: {summary['n_unknown_clusters']}")
        logger.info(f"% Correct but brittle: {summary['pct_correct_but_brittle']:.1f}%")
        logger.info(f"Robustness tested: {not summary['robustness_is_placeholder']}")
    
    # Compile results
    results = {
        "n_samples": len(embeddings_e),
        "n_questions": len(data),
        "optimal_beta": optimal_beta,
        "optimal_gamma": optimal_gamma,
        "cluster_analysis": cluster_analysis,
        "shortcut_clusters": shortcut_clusters,
        "faithful_clusters": faithful_clusters,
        "attribution_analysis": attribution_analysis,
        "summary": summary,
    }
    
    # Save results
    save_json(results, output_dir / "exp6_results.json")
    if logger:
        logger.info(f"\nResults saved to {output_dir / 'exp6_results.json'}")
    
    return results

