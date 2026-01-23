"""Experiment 3: Steering Validation (Causality).

Verifies that discovered motifs are causally meaningful by testing
whether steering with cluster centroids affects model outputs.

Uses the steering adapter to apply pooled or non-pooled attribution
centroids as steering directions.
"""

# Flag to indicate this experiment requires ReplacementModel
EXP3_REQUIRES_REPLACEMENT_MODEL = True

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json

# Import from downstream
from downstream.config import OUTPUT_DIR, STEERING_CONFIG


def run_experiment_3(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    assignments: np.ndarray,
    components: Dict,
    test_prompts: List[Dict],
    model,
    tokenizer,
    epsilon_values: List[float] = None,
    steering_method: str = None,
    top_B: int = None,
    pooled: bool = False,
    prefix_length: int = None,
    n_layers: int = None,
    d_transcoder: int = None,
    prefix_context=None,
    embedding_model=None,
    output_dir: Path = None,
    logger=None,
    active_feature_indices: np.ndarray = None,
) -> Dict[str, Any]:
    """Run Experiment 3: Steering Validation.
    
    Goal: Verify discovered motifs are causally meaningful.
    
    Method: Steer model with cluster centroids and measure:
    - Dose-response: Does steering strength correlate with effect?
    - Specificity: Does steering affect target cluster more than others?
    
    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        assignments: Cluster assignments from RD clustering
        components: Cluster components with centroids
        test_prompts: Test prompts for steering
        model: ReplacementModel for generation with steering
        tokenizer: Tokenizer
        epsilon_values: Steering strengths to test
        steering_method: Steering method ("sign", "additive", etc.)
        top_B: Number of top features to use
        pooled: Whether attributions are position-pooled
        prefix_length: Prefix length (for pooled centroids)
        n_layers: Number of layers (for pooled centroids)
        d_transcoder: Transcoder size (for pooled centroids)
        prefix_context: PrefixAttributionContext (for non-pooled centroids)
        embedding_model: Semantic embedding model (for similarity computation)
        output_dir: Output directory
        logger: Logger instance
        active_feature_indices: For sparse storage, maps reduced indices to original
        
    Returns:
        Dict with experiment results
    """
    if epsilon_values is None:
        epsilon_values = STEERING_CONFIG["epsilon_values"]
    if steering_method is None:
        steering_method = STEERING_CONFIG["steering_method"]
    if top_B is None:
        top_B = STEERING_CONFIG["top_B"]
    if output_dir is None:
        output_dir = OUTPUT_DIR / "exp3_steering_validation"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 3: STEERING VALIDATION")
        logger.info("=" * 60)
        logger.info(f"N clusters: {len(components)}")
        logger.info(f"N test prompts: {len(test_prompts)}")
        logger.info(f"Epsilon values: {epsilon_values}")
        logger.info(f"Steering method: {steering_method}")
        logger.info(f"Top B features: {top_B}")
        logger.info(f"Pooled attributions: {pooled}")
    
    # Check if we have a valid model for steering
    has_steering = _check_steering_available(model)
    if not has_steering:
        if logger:
            logger.warning(
                "Model does not support steering (requires ReplacementModel). "
                "Running with placeholder values."
            )
    
    results_by_cluster = []
    unique_clusters = np.unique(assignments)
    
    for cluster_id in tqdm(unique_clusters, desc="Clusters"):
        if cluster_id not in components:
            continue
        
        component = components[cluster_id]
        
        # Get cluster centroid in attribution space
        attr_centroid = component.get("mu_a", component.get("centroid_a"))
        if attr_centroid is None:
            continue
        
        attr_centroid = np.array(attr_centroid)
        
        # Get semantic centroid
        sem_centroid = component.get("mu_e", component.get("centroid_e"))
        
        # Cluster members
        cluster_mask = assignments == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_embeddings = embeddings_e[cluster_mask]
        cluster_attributions = attributions_a[cluster_mask]
        
        # Run steering for each epsilon
        steering_results = []
        
        for epsilon in epsilon_values:
            if epsilon == 0:
                # Baseline (no steering)
                continue
            
            # Generate steered outputs for test prompts
            steered_outputs = generate_steered_outputs(
                test_prompts,
                model,
                tokenizer,
                attr_centroid,
                epsilon,
                steering_method,
                top_B,
                pooled=pooled,
                prefix_length=prefix_length,
                n_layers=n_layers,
                d_transcoder=d_transcoder,
                prefix_context=prefix_context,
                active_feature_indices=active_feature_indices,
            )
            
            # Compute similarity to cluster
            similarities = compute_similarity_to_cluster(
                steered_outputs,
                cluster_embeddings,
                cluster_attributions,
                embedding_model=embedding_model,
                attribution_model=model if has_steering else None,
                tokenizer=tokenizer,
                pooled=pooled,  # Match attribution format (pooled vs non-pooled)
            )
            
            # Handle None values in similarity results
            sem_vals = [v for v in similarities["semantic"] if v is not None]
            attr_vals = [v for v in similarities["attribution"] if v is not None]
            
            mean_sem = float(np.mean(sem_vals)) if sem_vals else None
            mean_attr = float(np.mean(attr_vals)) if attr_vals else None
            
            steering_results.append({
                "epsilon": epsilon,
                "mean_sem_similarity": mean_sem,
                "mean_attr_similarity": mean_attr,
                "n_outputs": len([o for o in steered_outputs if o.get("steered", False)]),
                "similarity_available": not similarities.get("unavailable", False),
            })
        
        # Compute dose-response correlation
        # Filter out results where similarity couldn't be computed
        valid_results = [r for r in steering_results if r.get("mean_attr_similarity") is not None]
        
        if len(valid_results) > 2:
            epsilons = [r["epsilon"] for r in valid_results]
            attr_sims = [r["mean_attr_similarity"] for r in valid_results]
            
            try:
                pearson_r, _ = pearsonr(epsilons, attr_sims)
                spearman_rho, _ = spearmanr(epsilons, attr_sims)
            except Exception:
                pearson_r = 0.0
                spearman_rho = 0.0
        else:
            pearson_r = 0.0
            spearman_rho = 0.0
        
        # Check if any similarity computation was available
        similarity_available = any(r.get("similarity_available", False) for r in steering_results)
        
        results_by_cluster.append({
            "cluster_id": int(cluster_id),
            "cluster_size": int(cluster_size),
            "steering_results": steering_results,
            "dose_response": {
                "pearson_r": float(pearson_r),
                "spearman_rho": float(spearman_rho),
            },
            "similarity_available": similarity_available,
        })
    
    # Aggregate results
    all_pearson = [r["dose_response"]["pearson_r"] for r in results_by_cluster]
    all_spearman = [r["dose_response"]["spearman_rho"] for r in results_by_cluster]
    
    # Check if similarity was available for any cluster
    any_similarity_available = any(r.get("similarity_available", False) for r in results_by_cluster)
    
    summary = {
        "mean_pearson_r": float(np.mean(all_pearson)) if all_pearson else 0.0,
        "mean_spearman_rho": float(np.mean(all_spearman)) if all_spearman else 0.0,
        "n_clusters_tested": len(results_by_cluster),
        "steering_available": has_steering,
        "similarity_available": any_similarity_available,
    }
    
    if logger:
        logger.info("\n" + "-" * 40)
        logger.info("RESULTS SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Mean Pearson r: {summary['mean_pearson_r']:.4f}")
        logger.info(f"Mean Spearman rho: {summary['mean_spearman_rho']:.4f}")
        logger.info(f"Steering available: {has_steering}")
        logger.info(f"Similarity available: {any_similarity_available}")
        if not any_similarity_available:
            logger.warning(
                "Similarity computation unavailable. Provide embedding_model or "
                "attribution_model+tokenizer for meaningful dose-response analysis."
            )
    
    # Compile results
    results = {
        "n_test_prompts": len(test_prompts),
        "epsilon_values": epsilon_values,
        "steering_method": steering_method,
        "top_B": top_B,
        "pooled": pooled,
        "results_by_cluster": results_by_cluster,
        "summary": summary,
    }
    
    # Save results
    save_json(results, output_dir / "exp3_results.json")
    if logger:
        logger.info(f"\nResults saved to {output_dir / 'exp3_results.json'}")
    
    return results


def _check_steering_available(model) -> bool:
    """Check if model supports steering (is a ReplacementModel)."""
    if model is None:
        return False
    
    # Check for ReplacementModel characteristics
    has_transcoders = hasattr(model, "transcoders")
    has_hooks = hasattr(model, "hooks")
    has_cfg = hasattr(model, "cfg")
    
    return has_transcoders and has_hooks and has_cfg


def generate_steered_outputs(
    test_prompts: List[Dict],
    model,
    tokenizer,
    steering_vector: np.ndarray,
    epsilon: float,
    steering_method: str,
    top_B: int,
    pooled: bool = False,
    prefix_length: int = None,
    n_layers: int = None,
    d_transcoder: int = None,
    prefix_context=None,
    active_feature_indices: np.ndarray = None,
) -> List[Dict]:
    """Generate outputs with steering applied.
    
    Uses the steering adapter for actual activation patching when a
    ReplacementModel is available. Falls back to placeholder for other models.
    
    Args:
        test_prompts: Test prompts
        model: Model for generation (ReplacementModel for steering)
        tokenizer: Tokenizer
        steering_vector: Steering direction in attribution space
        epsilon: Steering strength
        steering_method: Method for applying steering
        top_B: Number of top features
        pooled: Whether steering_vector is position-pooled
        prefix_length: Prefix length (for pooled)
        n_layers: Number of layers (for pooled)
        d_transcoder: Transcoder size (for pooled)
        prefix_context: PrefixAttributionContext (for non-pooled)
        active_feature_indices: For sparse storage, maps reduced indices to original
        
    Returns:
        List of output dicts with generated text
    """
    # Check if we can do real steering
    if _check_steering_available(model):
        try:
            from downstream.steering.steering_adapter import generate_steered_outputs_real
            
            return generate_steered_outputs_real(
                test_prompts,
                model,
                tokenizer,
                steering_vector,
                epsilon,
                prefix_context=prefix_context,
                prefix_length=prefix_length,
                n_layers=n_layers,
                d_transcoder=d_transcoder,
                steering_method=steering_method,
                top_B=top_B,
                pooled=pooled,
                active_feature_indices=active_feature_indices,
            )
        except Exception as e:
            # Fall back to placeholder on error
            import warnings
            warnings.warn(f"Steering failed, using placeholder: {e}")
    
    # Fallback: placeholder outputs
    outputs = []
    for prompt_data in test_prompts:
        prompt = prompt_data.get("prompt", prompt_data.get("prefix", ""))
        output = {
            "prompt": prompt,
            "output": "",
            "epsilon": epsilon,
            "steered": False,
            "placeholder": True,
        }
        outputs.append(output)
    
    return outputs


def compute_similarity_to_cluster(
    outputs: List[Dict],
    cluster_embeddings: np.ndarray,
    cluster_attributions: np.ndarray,
    embedding_model=None,
    attribution_model=None,
    tokenizer=None,
    pooled: bool = False,
) -> Dict[str, List[float]]:
    """Compute similarity of outputs to cluster.
    
    Uses actual embedding computation when models are available.
    Falls back to placeholder values otherwise.
    
    Args:
        outputs: Generated outputs
        cluster_embeddings: Cluster member embeddings
        cluster_attributions: Cluster member attributions (pooled or non-pooled)
        embedding_model: Semantic embedding model (optional)
        attribution_model: ReplacementModel for attributions (optional)
        tokenizer: Tokenizer (optional)
        pooled: Whether cluster_attributions are pooled (position-aggregated)
        
    Returns:
        Dict with semantic and attribution similarities
    """
    # Try real computation if model available
    if embedding_model is not None or attribution_model is not None:
        try:
            from downstream.steering.steering_adapter import compute_similarity_to_cluster_real
            
            return compute_similarity_to_cluster_real(
                outputs,
                cluster_embeddings,
                cluster_attributions,
                embedding_model=embedding_model,
                attribution_model=attribution_model,
                tokenizer=tokenizer,
                pooled=pooled,
            )
        except Exception:
            pass
    
    # Fallback: placeholder values
    return {
        "semantic": [0.0] * len(outputs),
        "attribution": [0.0] * len(outputs),
    }
