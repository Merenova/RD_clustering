"""Experiment 4: Cross-Prefix Motif (Generality).

Shows that the same mechanistic motifs appear across different prompts,
demonstrating the generality of discovered patterns.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json

# Import from downstream
from downstream.config import OUTPUT_DIR, CROSS_PREFIX_CONFIG
from downstream.clustering.rd_wrapper import run_rd_clustering, compute_global_normalization
from downstream.clustering.config_selection import compute_silhouette


def run_experiment_4(
    per_prompt_data: Dict[str, Dict[str, Any]],
    optimal_beta: float,
    optimal_gamma: float,
    n_global_motifs: int = None,
    output_dir: Path = None,
    logger=None,
) -> Dict[str, Any]:
    """Run Experiment 4: Cross-Prefix Motif Discovery.
    
    Goal: Show same motifs appear across different prompts.
    
    Method:
    1. Cluster each prompt separately using RD clustering
    2. Collect all cluster centroids (attribution space)
    3. Meta-cluster centroids using K-means to find global motifs
    4. Analyze global motifs for cross-prompt prevalence
    
    Args:
        per_prompt_data: Dict mapping prompt_id -> data dict with
            embeddings, attributions, continuations, etc.
        optimal_beta: Beta for RD clustering
        optimal_gamma: Gamma for RD clustering
        n_global_motifs: Number of global motifs to find via K-means meta-clustering
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dict with experiment results
    """
    if n_global_motifs is None:
        n_global_motifs = CROSS_PREFIX_CONFIG["n_global_motifs"]
    if output_dir is None:
        output_dir = OUTPUT_DIR / "exp4_cross_prefix"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 4: CROSS-PREFIX MOTIF DISCOVERY")
        logger.info("=" * 60)
        logger.info(f"N prompts: {len(per_prompt_data)}")
        logger.info(f"N global motifs: {n_global_motifs}")
        logger.info(f"Beta: {optimal_beta}, Gamma: {optimal_gamma}")
    
    # Step 0: Compute GLOBAL normalization across all prompts
    # This ensures centroids are comparable for meta-clustering
    if logger:
        logger.info("\nStep 0: Computing global normalization across all prompts...")
    
    all_attributions = []
    for prompt_id, data in per_prompt_data.items():
        attrs = data.get("attributions", data.get("attributions_a"))
        if attrs is not None:
            if not isinstance(attrs, np.ndarray):
                attrs = np.array(attrs)
            if len(attrs) > 0:
                all_attributions.append(attrs)
    
    if all_attributions:
        global_H_0, global_rms_norm = compute_global_normalization(all_attributions)
        if logger:
            logger.info(f"Global normalization: ||H_0||={np.linalg.norm(global_H_0):.4f}, rms={global_rms_norm:.4f}")
    else:
        global_H_0 = None
        global_rms_norm = None
        if logger:
            logger.warning("No attributions found, skipping global normalization")
    
    # Step 1: Cluster each prompt with global normalization
    if logger:
        logger.info("\nStep 1: Clustering each prompt (with global normalization)...")
    
    per_prompt_clusters = {}
    
    for prompt_id, data in tqdm(per_prompt_data.items(), desc="Per-prompt clustering"):
        embeddings_e = data.get("embeddings", data.get("embeddings_e"))
        attributions_a = data.get("attributions", data.get("attributions_a"))
        path_probs = data.get("path_probs", data.get("probabilities"))
        
        if embeddings_e is None or attributions_a is None:
            continue
        
        if len(embeddings_e) < 3:
            continue
        
        # Convert to numpy if needed
        if not isinstance(embeddings_e, np.ndarray):
            embeddings_e = np.array(embeddings_e)
        if not isinstance(attributions_a, np.ndarray):
            attributions_a = np.array(attributions_a)
        
        try:
            result = run_rd_clustering(
                embeddings_e,
                attributions_a,
                path_probs,
                beta=optimal_beta,
                gamma=optimal_gamma,
                global_H_0=global_H_0,
                global_rms_norm=global_rms_norm,
            )
            per_prompt_clusters[prompt_id] = result
        except Exception as e:
            if logger:
                logger.warning(f"Clustering failed for {prompt_id}: {e}")
    
    # Step 2: Collect all cluster centroids
    if logger:
        logger.info(f"\nStep 2: Collecting centroids from {len(per_prompt_clusters)} prompts...")
    
    all_centroids = []
    
    for prompt_id, result in per_prompt_clusters.items():
        components = result.get("components", {})
        assignments = result.get("assignments", [])
        
        for cluster_id, component in components.items():
            attr_centroid = component.get("mu_a", component.get("centroid_a"))
            sem_centroid = component.get("mu_e", component.get("centroid_e"))
            
            if attr_centroid is None:
                continue
            
            # Get sample continuations from this cluster
            cluster_mask = np.array(assignments) == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Get sample texts if available
            sample_texts = []
            data = per_prompt_data.get(prompt_id, {})
            continuations = data.get("continuations", [])
            for idx in cluster_indices[:3]:  # Up to 3 samples
                if idx < len(continuations):
                    cont = continuations[idx]
                    text = cont.get("text", cont) if isinstance(cont, dict) else cont
                    sample_texts.append(text)
            
            all_centroids.append({
                "prompt_id": prompt_id,
                "cluster_id": int(cluster_id),
                "attribution_centroid": attr_centroid,
                "semantic_centroid": sem_centroid,
                "cluster_size": int(np.sum(cluster_mask)),
                "sample_continuations": sample_texts,
            })
    
    if logger:
        logger.info(f"Collected {len(all_centroids)} centroids")
    
    if len(all_centroids) < 2:
        if logger:
            logger.warning(f"Not enough centroids ({len(all_centroids)}) for meta-clustering, need at least 2")
        # Return early with empty results
        results = {
            "n_prompts": len(per_prompt_data),
            "n_centroids": len(all_centroids),
            "error": "Not enough centroids for meta-clustering",
            "global_motifs": [],
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(results, output_dir / "exp4_results.json")
        return results
    
    if len(all_centroids) < n_global_motifs:
        if logger:
            logger.warning(f"Not enough centroids ({len(all_centroids)}) for {n_global_motifs} global motifs")
        n_global_motifs = max(2, min(len(all_centroids), len(all_centroids) // 2 + 1))
    
    # Step 3: Meta-clustering on attribution centroids
    if logger:
        logger.info(f"\nStep 3: Meta-clustering into {n_global_motifs} global motifs...")
    
    attr_centroid_matrix = np.stack([c["attribution_centroid"] for c in all_centroids])
    
    kmeans = KMeans(n_clusters=n_global_motifs, random_state=42, n_init=10)
    meta_assignments = kmeans.fit_predict(attr_centroid_matrix)
    
    # Step 4: Analyze global motifs
    if logger:
        logger.info("\nStep 4: Analyzing global motifs...")
    
    global_motifs = []
    
    for motif_id in range(n_global_motifs):
        motif_mask = meta_assignments == motif_id
        members = [c for c, m in zip(all_centroids, motif_mask) if m]
        
        if not members:
            continue
        
        # Count unique prompts
        unique_prompts = set(m["prompt_id"] for m in members)
        
        # Compute attribution similarity within motif
        member_centroids = np.stack([m["attribution_centroid"] for m in members])
        attr_similarity = compute_mean_pairwise_similarity(member_centroids)
        
        # Compute semantic similarity
        sem_centroids = [m["semantic_centroid"] for m in members if m["semantic_centroid"] is not None]
        if len(sem_centroids) > 1:
            sem_similarity = compute_mean_pairwise_similarity(np.stack(sem_centroids))
        else:
            sem_similarity = 0.0
        
        # Collect sample continuations
        sample_texts = []
        for m in members[:5]:
            if m["sample_continuations"]:
                sample_texts.append({
                    "prompt_id": m["prompt_id"],
                    "text": m["sample_continuations"][0],
                })
        
        global_motifs.append({
            "motif_id": motif_id,
            "n_prompts": len(unique_prompts),
            "n_clusters": len(members),
            "attribution_similarity": float(attr_similarity),
            "semantic_similarity": float(sem_similarity),
            "sample_continuations": sample_texts,
        })
    
    # Sort by number of prompts
    global_motifs.sort(key=lambda x: x["n_prompts"], reverse=True)
    
    # Print summary
    if logger:
        logger.info("\n" + "-" * 60)
        logger.info("GLOBAL MOTIFS SUMMARY")
        logger.info("-" * 60)
        
        for motif in global_motifs:
            logger.info(f"\nMotif {motif['motif_id']}:")
            logger.info(f"  Appears in {motif['n_prompts']}/{len(per_prompt_data)} prompts")
            logger.info(f"  N clusters: {motif['n_clusters']}")
            logger.info(f"  Attribution similarity: {motif['attribution_similarity']:.4f}")
            logger.info(f"  Semantic similarity: {motif['semantic_similarity']:.4f}")
            
            if motif["sample_continuations"]:
                logger.info("  Examples:")
                for sample in motif["sample_continuations"][:2]:
                    text = sample["text"][:80] + "..." if len(sample["text"]) > 80 else sample["text"]
                    logger.info(f"    [{sample['prompt_id']}] {text}")
    
    # Compile results
    results = {
        "n_prompts": len(per_prompt_data),
        "n_prompts_clustered": len(per_prompt_clusters),
        "n_centroids": len(all_centroids),
        "n_global_motifs": n_global_motifs,
        "optimal_beta": optimal_beta,
        "optimal_gamma": optimal_gamma,
        "global_motifs": global_motifs,
        "per_prompt_summary": {
            prompt_id: {
                "n_clusters": len(result.get("components", {})),
                "n_samples": len(result.get("assignments", [])),
            }
            for prompt_id, result in per_prompt_clusters.items()
        },
    }
    
    # Save results
    save_json(results, output_dir / "exp4_results.json")
    if logger:
        logger.info(f"\nResults saved to {output_dir / 'exp4_results.json'}")
    
    return results


def compute_mean_pairwise_similarity(
    vectors: np.ndarray,
) -> float:
    """Compute mean pairwise cosine similarity.
    
    Args:
        vectors: Matrix of vectors [n, d]
        
    Returns:
        Mean pairwise similarity
    """
    if len(vectors) < 2:
        return 1.0
    
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vectors_normed = vectors / norms
    
    # Compute similarity matrix
    sim_matrix = vectors_normed @ vectors_normed.T
    
    # Extract upper triangle
    n = len(vectors)
    triu_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[triu_indices]
    
    return float(np.mean(similarities)) if len(similarities) > 0 else 1.0

