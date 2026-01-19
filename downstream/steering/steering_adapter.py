"""Steering adapter for downstream experiments.

Adapts the 7_validation/7c_steering module for use with downstream experiments,
particularly handling pooled attribution centroids by expanding them to all
prefix positions.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "7_validation"))

from downstream.config import STEERING_CONFIG


# =============================================================================
# Pooled Attribution Expansion
# =============================================================================

def expand_pooled_to_positions(
    pooled_centroid: np.ndarray,
    prefix_length: int,
    n_layers: int,
    d_transcoder: int,
    top_B: int = 10,
) -> List[Tuple[int, int, int, float]]:
    """Expand pooled centroid to position-specific features.
    
    Pooled centroids have shape (n_layers * d_transcoder,) with no position
    dimension. To steer, we "unsqueeze" to all prefix positions.
    
    For each top-B feature in (layer, feat_idx) space, create steering
    entries for ALL prefix positions [1, prefix_length).
    
    Args:
        pooled_centroid: (n_layers * d_transcoder,) pooled attribution centroid
        prefix_length: Number of tokens in prefix (including BOS)
        n_layers: Number of transformer layers
        d_transcoder: Transcoder dictionary size
        top_B: Number of top features to use for steering
        
    Returns:
        List of (layer, pos, feat_idx, h_c_val) tuples for steering
    """
    # Get top-B features by attribution magnitude
    top_indices = np.argsort(np.abs(pooled_centroid))[-top_B:]
    
    features = []
    for flat_idx in top_indices:
        layer = int(flat_idx // d_transcoder)
        feat_idx = int(flat_idx % d_transcoder)
        h_c_val = float(pooled_centroid[flat_idx])
        
        # Skip near-zero features
        if abs(h_c_val) < 1e-8:
            continue
        
        # Apply to all positions (excluding BOS at position 0)
        for pos in range(1, prefix_length):
            features.append((layer, pos, feat_idx, h_c_val))
    
    return features


def expand_nonpooled_to_features(
    attr_centroid: np.ndarray,
    prefix_context,
    top_B: int = 10,
) -> List[Tuple[int, int, int, float]]:
    """Convert non-pooled attribution centroid to steering features.
    
    Non-pooled centroids already have position information encoded via
    prefix_context.get_feature_info().
    
    Args:
        attr_centroid: (n_features,) attribution centroid
        prefix_context: PrefixAttributionContext with feature info
        top_B: Number of top features to use
        
    Returns:
        List of (layer, pos, feat_idx, h_c_val) tuples for steering
    """
    # Get top-B features by attribution magnitude
    top_indices = np.argsort(np.abs(attr_centroid))[-top_B:]
    
    features = []
    for idx in top_indices:
        h_c_val = float(attr_centroid[idx])
        
        # Skip near-zero features
        if abs(h_c_val) < 1e-8:
            continue
        
        # Get feature info from prefix context
        info = prefix_context.get_feature_info(int(idx))
        if info is None:
            continue
        
        layer, pos, feat_idx, _ = info
        
        # Skip BOS position
        if pos <= 0:
            continue
        
        features.append((int(layer), int(pos), int(feat_idx), h_c_val))
    
    return features


# =============================================================================
# Steering Preparation
# =============================================================================

def prepare_steering_from_centroid(
    model,
    attr_centroid: np.ndarray,
    prefix_context=None,
    prefix_length: int = None,
    n_layers: int = None,
    d_transcoder: int = None,
    top_B: int = None,
    pooled: bool = False,
) -> Tuple[List[Tuple], Dict, Dict]:
    """Convert attribution centroid to steering interventions.
    
    Args:
        model: ReplacementModel with transcoders
        attr_centroid: Cluster centroid in attribution space
        prefix_context: PrefixAttributionContext for non-pooled centroids
        prefix_length: Prefix length for pooled centroids
        n_layers: Number of layers (for pooled)
        d_transcoder: Transcoder size (for pooled)
        top_B: Number of top features to steer
        pooled: Whether centroid is position-pooled
        
    Returns:
        Tuple of (features, encoder_cache, decoder_cache)
    """
    if top_B is None:
        top_B = STEERING_CONFIG.get("top_B", 10)
    
    # Convert centroid to feature list
    if pooled:
        if prefix_length is None:
            raise ValueError("prefix_length required for pooled centroids")
        if n_layers is None:
            n_layers = model.cfg.n_layers
        if d_transcoder is None:
            d_transcoder = model.cfg.d_mlp * 8
        
        features = expand_pooled_to_positions(
            attr_centroid, prefix_length, n_layers, d_transcoder, top_B
        )
    else:
        if prefix_context is None:
            raise ValueError("prefix_context required for non-pooled centroids")
        
        features = expand_nonpooled_to_features(attr_centroid, prefix_context, top_B)
    
    if not features:
        return [], {}, {}
    
    # Import steering utilities via importlib (module names start with digit)
    import importlib.util
    graph_path = Path(__file__).resolve().parents[2] / "7_validation" / "7c_graph.py"
    spec = importlib.util.spec_from_file_location("graph_7c", graph_path)
    graph_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_module)
    preload_encoder_weights_for_cluster = graph_module.preload_encoder_weights_for_cluster
    precompute_cluster_decoder_vectors = graph_module.precompute_cluster_decoder_vectors
    
    device = model.cfg.device
    
    # Precompute caches
    encoder_cache = preload_encoder_weights_for_cluster(model, features, device)
    
    # precompute_cluster_decoder_vectors expects {cluster_id: features}
    features_by_cluster = {0: features}
    decoder_cache_all = precompute_cluster_decoder_vectors(model, features_by_cluster, device)
    decoder_cache = decoder_cache_all.get(0, {})
    
    return features, encoder_cache, decoder_cache


# =============================================================================
# Steered Generation
# =============================================================================

def generate_steered_outputs_real(
    test_prompts: List[Dict],
    model,
    tokenizer,
    steering_vector: np.ndarray,
    epsilon: float,
    prefix_context=None,
    prefix_length: int = None,
    n_layers: int = None,
    d_transcoder: int = None,
    steering_method: str = None,
    top_B: int = None,
    max_new_tokens: int = 50,
    pooled: bool = False,
) -> List[Dict]:
    """Generate outputs with actual steering applied.
    
    Args:
        test_prompts: Test prompts (list of dicts with 'prompt' or 'prefix' key)
        model: ReplacementModel (from circuit-tracer)
        tokenizer: Tokenizer
        steering_vector: Steering direction (attribution centroid)
        epsilon: Steering strength
        prefix_context: PrefixAttributionContext for non-pooled centroids
        prefix_length: Prefix length for pooled centroids
        n_layers: Number of layers
        d_transcoder: Transcoder size
        steering_method: Steering method (from STEERING_CONFIG)
        top_B: Number of top features
        max_new_tokens: Max tokens to generate
        pooled: Whether centroid is pooled
        
    Returns:
        List of output dicts with generated text
    """
    if steering_method is None:
        steering_method = STEERING_CONFIG.get("steering_method", "sign")
    if top_B is None:
        top_B = STEERING_CONFIG.get("top_B", 10)
    
    # Import steering function via importlib (module names start with digit)
    import importlib.util
    steering_path = Path(__file__).resolve().parents[2] / "7_validation" / "7c_steering.py"
    spec = importlib.util.spec_from_file_location("steering_7c", steering_path)
    steering_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(steering_module)
    generate_steered_sequences = steering_module.generate_steered_sequences
    
    outputs = []
    
    for prompt_data in test_prompts:
        prompt = prompt_data.get("prompt", prompt_data.get("prefix", ""))
        
        # Tokenize prefix
        prefix_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        current_prefix_length = len(prefix_tokens)
        
        # For pooled centroids, use current prefix length to avoid steering non-existent positions
        # If prefix_length is provided, use the minimum to cap (never steer beyond actual prompt)
        if prefix_length is not None:
            actual_prefix_length = min(prefix_length, current_prefix_length)
        else:
            actual_prefix_length = current_prefix_length
        
        # Prepare steering features for this prompt
        features, encoder_cache, decoder_cache = prepare_steering_from_centroid(
            model,
            steering_vector,
            prefix_context=prefix_context,
            prefix_length=actual_prefix_length,
            n_layers=n_layers,
            d_transcoder=d_transcoder,
            top_B=top_B,
            pooled=pooled,
        )
        
        if not features:
            # No valid features to steer with
            output = {
                "prompt": prompt,
                "output": "",
                "output_tokens": [],
                "epsilon": epsilon,
                "steered": False,
                "n_features": 0,
            }
            outputs.append(output)
            continue
        
        try:
            # Generate with steering
            results = generate_steered_sequences(
                model,
                prefix_tokens,
                features,
                encoder_cache,
                decoder_cache,
                steering_method,
                epsilon,
                max_new_tokens=max_new_tokens,
                num_samples=1,
            )
            
            if results:
                output = {
                    "prompt": prompt,
                    "output": results[0].get("generated_text", ""),
                    "output_tokens": results[0].get("generated_tokens", []),
                    "epsilon": epsilon,
                    "steered": True,
                    "n_features": len(features),
                }
            else:
                output = {
                    "prompt": prompt,
                    "output": "",
                    "output_tokens": [],
                    "epsilon": epsilon,
                    "steered": False,
                    "n_features": len(features),
                }
        except Exception as e:
            output = {
                "prompt": prompt,
                "output": "",
                "output_tokens": [],
                "epsilon": epsilon,
                "steered": False,
                "error": str(e),
                "n_features": len(features),
            }
        
        outputs.append(output)
    
    return outputs


# =============================================================================
# Similarity Computation
# =============================================================================

def compute_similarity_to_cluster_real(
    outputs: List[Dict],
    cluster_embeddings: np.ndarray,
    cluster_attributions: np.ndarray,
    embedding_model=None,
    attribution_model=None,
    tokenizer=None,
    prefix_tokens: List[int] = None,
    pooled: bool = False,
) -> Dict[str, List[float]]:
    """Compute similarity of steered outputs to cluster.
    
    At least one of embedding_model or attribution_model must be provided.
    If neither is provided, returns None values to indicate unavailable data.
    
    Args:
        outputs: Generated outputs (list of dicts with 'output' key)
        cluster_embeddings: Cluster member semantic embeddings
        cluster_attributions: Cluster member attribution embeddings (pooled or non-pooled)
        embedding_model: Semantic embedding model (optional)
        attribution_model: ReplacementModel for attributions (optional)
        tokenizer: Tokenizer (required if attribution_model is provided)
        prefix_tokens: Prefix token IDs for attribution computation (optional)
        pooled: Whether cluster_attributions are pooled (position-aggregated).
                If True, computes pooled attributions for outputs to match.
        
    Returns:
        Dict with 'semantic' and 'attribution' similarity lists.
        Values are None when computation is not possible.
    """
    # Compute cluster centroids
    cluster_sem_centroid = cluster_embeddings.mean(axis=0) if len(cluster_embeddings) > 0 else None
    cluster_attr_centroid = cluster_attributions.mean(axis=0) if len(cluster_attributions) > 0 else None
    
    semantic_sims = []
    attr_sims = []
    
    # Check if we can compute anything
    can_compute_semantic = embedding_model is not None and cluster_sem_centroid is not None
    can_compute_attribution = (
        attribution_model is not None 
        and tokenizer is not None 
        and cluster_attr_centroid is not None
    )
    
    if not can_compute_semantic and not can_compute_attribution:
        # Return None to indicate unavailable data (not 0 which implies dissimilarity)
        return {
            "semantic": [None] * len(outputs),
            "attribution": [None] * len(outputs),
            "unavailable": True,
        }
    
    for output in outputs:
        text = output.get("output", "")
        prompt = output.get("prompt", "")
        
        if not text:
            semantic_sims.append(None if can_compute_semantic else None)
            attr_sims.append(None if can_compute_attribution else None)
            continue
        
        # Compute semantic similarity if model available
        if can_compute_semantic:
            try:
                from downstream.embedding.semantic import compute_semantic_embeddings
                sem_emb = compute_semantic_embeddings([text], embedding_model)[0]
                
                # Cosine similarity
                sem_norm = np.linalg.norm(sem_emb)
                cluster_norm = np.linalg.norm(cluster_sem_centroid)
                if sem_norm > 0 and cluster_norm > 0:
                    sem_sim = np.dot(sem_emb, cluster_sem_centroid) / (sem_norm * cluster_norm)
                else:
                    sem_sim = 0.0
                
                semantic_sims.append(float(sem_sim))
            except Exception:
                semantic_sims.append(None)
        else:
            semantic_sims.append(None)
        
        # Compute attribution similarity if model available
        if can_compute_attribution:
            try:
                from downstream.embedding.attribution import compute_attribution_embeddings
                
                # Tokenize the output
                output_tokens = tokenizer.encode(text, add_special_tokens=False)
                
                # Use prompt as prefix if prefix_tokens not provided
                if prefix_tokens is None:
                    current_prefix = tokenizer.encode(prompt, add_special_tokens=True)
                else:
                    current_prefix = prefix_tokens
                
                # Compute attribution for this output
                # Match pooled/non-pooled format to cluster attributions
                attr_emb, attr_emb_pooled, _ = compute_attribution_embeddings(
                    current_prefix,
                    [output_tokens],
                    attribution_model,
                    compute_pooled=pooled,  # Match cluster attribution format
                )
                
                # Use pooled if cluster attributions are pooled
                if pooled and attr_emb_pooled is not None and len(attr_emb_pooled) > 0:
                    attr_emb_to_use = attr_emb_pooled[0]
                elif attr_emb is not None and len(attr_emb) > 0:
                    attr_emb_to_use = attr_emb[0]
                else:
                    attr_emb_to_use = None
                
                if attr_emb_to_use is not None:
                    # Cosine similarity
                    attr_norm = np.linalg.norm(attr_emb_to_use)
                    cluster_norm = np.linalg.norm(cluster_attr_centroid)
                    if attr_norm > 0 and cluster_norm > 0:
                        attr_sim = np.dot(attr_emb_to_use, cluster_attr_centroid) / (attr_norm * cluster_norm)
                    else:
                        attr_sim = 0.0
                    
                    attr_sims.append(float(attr_sim))
                else:
                    attr_sims.append(None)
            except Exception:
                attr_sims.append(None)
        else:
            # Use semantic similarity as proxy if available, else None
            if can_compute_semantic and semantic_sims[-1] is not None:
                attr_sims.append(semantic_sims[-1])
            else:
                attr_sims.append(None)
    
    return {
        "semantic": semantic_sims,
        "attribution": attr_sims,
        "unavailable": False,
    }

