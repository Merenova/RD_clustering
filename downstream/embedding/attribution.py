"""Attribution embedding computation using circuit-tracer.

Computes Input x Gradient attributions for the mechanistic view.
Supports both position-specific and position-pooled (cross-prefix) formats.
"""

import gc
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import config
from downstream.config import (
    DEFAULT_ATTRIBUTION_MODEL,
    TRANSCODER_SET,
    ATTRIBUTION_CONFIG,
    CIRCUIT_TRACER_PATH,
)

# Add circuit-tracer to path
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))


# =============================================================================
# Position Pooling for Cross-Prefix Clustering
# =============================================================================

def pool_attribution_over_positions(
    source_attribution: np.ndarray,
    prefix_context,
    n_layers: int,
    d_transcoder: int,
    pooling: str = "sum",
) -> np.ndarray:
    """Pool attribution from (layer, position, feature_idx) to (layer, feature_idx).
    
    This creates a position-invariant representation that can be compared
    across different prefixes (which have different lengths).
    
    Args:
        source_attribution: (n_prefix_sources,) attribution scores
        prefix_context: PrefixAttributionContext with feature info
        n_layers: Number of transformer layers
        d_transcoder: Transcoder dictionary size
        pooling: Aggregation method - "sum", "mean", or "max"
        
    Returns:
        (n_layers * d_transcoder,) pooled attribution vector
    """
    pooled = np.zeros(n_layers * d_transcoder, dtype=np.float32)
    counts = np.zeros(n_layers * d_transcoder, dtype=np.float32)
    
    n_features = min(len(source_attribution), prefix_context.n_prefix_features)
    
    for idx in range(n_features):
        info = prefix_context.get_feature_info(idx)
        if info is None:
            continue  # Skip error/token nodes
        
        layer, position, feature_idx, _ = info
        flat_idx = layer * d_transcoder + feature_idx
        
        # Bounds check
        if flat_idx >= len(pooled):
            continue
        
        attr_val = float(source_attribution[idx])
        
        if pooling == "sum":
            pooled[flat_idx] += attr_val
        elif pooling == "max":
            pooled[flat_idx] = max(pooled[flat_idx], attr_val)
        elif pooling == "mean":
            pooled[flat_idx] += attr_val
            counts[flat_idx] += 1
    
    if pooling == "mean":
        mask = counts > 0
        pooled[mask] /= counts[mask]
    
    return pooled


def pool_attributions_batch(
    attributions: np.ndarray,
    prefix_context,
    n_layers: int,
    d_transcoder: int,
    pooling: str = "sum",
) -> np.ndarray:
    """Pool attributions for a batch of continuations.
    
    Args:
        attributions: (n_continuations, n_features) attribution scores
        prefix_context: PrefixAttributionContext with feature info
        n_layers: Number of transformer layers
        d_transcoder: Transcoder dictionary size
        pooling: Aggregation method
        
    Returns:
        (n_continuations, n_layers * d_transcoder) pooled attributions
    """
    n_continuations = attributions.shape[0]
    pooled = np.zeros((n_continuations, n_layers * d_transcoder), dtype=np.float32)
    
    for i in range(n_continuations):
        pooled[i] = pool_attribution_over_positions(
            attributions[i],
            prefix_context,
            n_layers,
            d_transcoder,
            pooling,
        )
    
    return pooled


def load_replacement_model(
    model_name: str = None,
    transcoder_set: str = None,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """Load ReplacementModel for attribution computation.
    
    Args:
        model_name: HuggingFace model name (defaults to instruct model)
        transcoder_set: Transcoder set name
        dtype: Model dtype
        device: Device for computation
        
    Returns:
        ReplacementModel instance
    """
    from circuit_tracer import ReplacementModel
    
    if model_name is None:
        model_name = DEFAULT_ATTRIBUTION_MODEL
    if transcoder_set is None:
        transcoder_set = TRANSCODER_SET
    
    return ReplacementModel.from_pretrained(
        model_name,
        transcoder_set,
        dtype=dtype,
    )


def compute_attribution_embeddings(
    prefix_tokens: List[int],
    continuation_tokens: List[List[int]],
    model,
    batch_size: int = None,
    max_feature_nodes: int = None,
    add_bos: bool = None,
    compute_pooled: bool = True,
    pooling_method: str = "sum",
    logger=None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Compute Input x Gradient attributions for continuations.
    
    Returns both position-specific and position-pooled attributions to support
    different experiment needs:
    - Position-specific: For single-prefix clustering (Exp6)
    - Position-pooled: For cross-prefix clustering (Exp4, Exp5)
    
    Args:
        prefix_tokens: Token IDs for prefix (with BOS if needed)
        continuation_tokens: List of token ID lists for each continuation
        model: ReplacementModel instance
        batch_size: Batch size for attribution computation
        max_feature_nodes: Max number of feature nodes
        add_bos: Whether to add BOS token
        compute_pooled: Whether to compute position-pooled attributions
        pooling_method: Pooling method for position aggregation ("sum", "mean", "max")
        logger: Optional logger
        
    Returns:
        Tuple of:
        - attributions: (n_continuations, n_features) - position-specific
        - attributions_pooled: (n_continuations, n_layers * d_transcoder) or None
        - metadata: dict with prefix_context, model config, etc.
    """
    from circuit_tracer.attribution import attribute_prefix_to_continuations
    
    if batch_size is None:
        batch_size = ATTRIBUTION_CONFIG["batch_size"]
    if max_feature_nodes is None:
        max_feature_nodes = ATTRIBUTION_CONFIG["max_feature_nodes"]
    if add_bos is None:
        add_bos = ATTRIBUTION_CONFIG["add_bos"]
    
    # Run attribution
    result = attribute_prefix_to_continuations(
        prefix=prefix_tokens,
        continuations=continuation_tokens,
        model=model,
        batch_size=batch_size,
        add_bos=add_bos,
        max_feature_nodes=max_feature_nodes,
        verbose=logger is not None,
    )
    
    # Extract aggregated attributions (position-specific)
    attributions_list = []
    for cont_attrs in result.continuation_attributions:
        # Mean pool across continuation tokens
        if cont_attrs:
            token_attrs = torch.stack([ta.source_attribution for ta in cont_attrs])
            agg = token_attrs.mean(dim=0).float().cpu().numpy()
        else:
            agg = np.zeros(result.prefix_context.n_prefix_sources)
        attributions_list.append(agg)
    
    attributions = np.stack(attributions_list)
    
    # Get model config for pooled dimensions
    n_layers = model.cfg.n_layers
    d_transcoder = model.cfg.d_mlp * 8  # Typical transcoder expansion
    
    # Compute position-pooled attributions if requested
    attributions_pooled = None
    if compute_pooled:
        attributions_pooled = pool_attributions_batch(
            attributions,
            result.prefix_context,
            n_layers,
            d_transcoder,
            pooling=pooling_method,
        )
    
    metadata = {
        "n_continuations": len(continuation_tokens),
        "n_features": attributions.shape[1],
        "n_features_pooled": n_layers * d_transcoder if compute_pooled else None,
        "n_layers": n_layers,
        "d_transcoder": d_transcoder,
        "prefix_tokens": prefix_tokens,
        "pooling_method": pooling_method if compute_pooled else None,
        "prefix_context": result.prefix_context,  # For steering
    }
    
    return attributions, attributions_pooled, metadata


def compute_attributions_batch(
    data: List[Dict[str, Any]],
    tokenizer,
    model=None,
    batch_size: int = None,
    compute_pooled: bool = True,
    pooling_method: str = "sum",
    logger=None,
) -> List[Dict[str, Any]]:
    """Compute attributions for a batch of data.
    
    Groups data by prefix for efficient computation. Computes both
    position-specific and position-pooled attributions.
    
    Args:
        data: List of dicts with 'prefix' and 'continuation' keys
        tokenizer: HuggingFace tokenizer
        model: Pre-loaded ReplacementModel (loads new one if None)
        batch_size: Batch size
        compute_pooled: Whether to compute position-pooled attributions
        pooling_method: Pooling method for position aggregation
        logger: Logger instance
        
    Returns:
        Updated data with 'attribution' and 'attribution_pooled' fields added
    """
    if model is None:
        model = load_replacement_model()
        cleanup_model = True
    else:
        cleanup_model = False
    
    # Group by prefix
    by_prefix = {}
    for item in data:
        prefix = item.get("prefix", "")
        if prefix not in by_prefix:
            by_prefix[prefix] = []
        by_prefix[prefix].append(item)
    
    if logger:
        logger.info(f"Computing attributions for {len(data)} samples across {len(by_prefix)} prefixes")
    
    # Process each prefix group
    for prefix, items in tqdm(by_prefix.items(), desc="Attribution"):
        # Tokenize prefix
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
        
        # Tokenize continuations
        continuation_tokens = []
        for item in items:
            cont_text = item.get("continuation", "")
            # Use pre-computed token IDs if available
            if "continuation_token_ids" in item and item["continuation_token_ids"]:
                cont_tokens = item["continuation_token_ids"]
            else:
                cont_tokens = tokenizer.encode(cont_text, add_special_tokens=False)
            continuation_tokens.append(cont_tokens)
        
        try:
            attributions, attributions_pooled, metadata = compute_attribution_embeddings(
                prefix_tokens,
                continuation_tokens,
                model,
                batch_size=batch_size,
                compute_pooled=compute_pooled,
                pooling_method=pooling_method,
            )
            
            for i, item in enumerate(items):
                item["attribution"] = attributions[i]
                if attributions_pooled is not None:
                    item["attribution_pooled"] = attributions_pooled[i]
                else:
                    item["attribution_pooled"] = None
                
        except Exception as e:
            if logger:
                logger.warning(f"Attribution failed for prefix: {e}")
            for item in items:
                item["attribution"] = None
                item["attribution_pooled"] = None
    
    if cleanup_model:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return data


def compute_attributions_for_results(
    results: List[Dict[str, Any]],
    tokenizer,
    model=None,
    compute_pooled: bool = True,
    pooling_method: str = "sum",
    logger=None,
) -> List[Dict[str, Any]]:
    """Compute attributions for rollout results.
    
    Computes both position-specific and position-pooled attributions.
    
    Args:
        results: List of result dicts (from rollout)
        tokenizer: HuggingFace tokenizer
        model: Pre-loaded ReplacementModel
        compute_pooled: Whether to compute position-pooled attributions
        pooling_method: Pooling method for position aggregation
        logger: Logger instance
        
    Returns:
        Updated results with 'attributions' and 'attributions_pooled' fields
    """
    if model is None:
        model = load_replacement_model()
        cleanup_model = True
    else:
        cleanup_model = False
    
    for result in tqdm(results, desc="Computing attributions"):
        prefix = result.get("prompt", "")
        continuations = result.get("continuations", [])
        
        if not continuations:
            result["attributions"] = None
            result["attributions_pooled"] = None
            continue
        
        # Tokenize
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
        continuation_tokens = [
            c.get("token_ids", tokenizer.encode(c["text"], add_special_tokens=False))
            for c in continuations
        ]
        
        try:
            attributions, attributions_pooled, metadata = compute_attribution_embeddings(
                prefix_tokens,
                continuation_tokens,
                model,
                compute_pooled=compute_pooled,
                pooling_method=pooling_method,
            )
            result["attributions"] = attributions
            result["attributions_pooled"] = attributions_pooled
            result["attribution_metadata"] = metadata
            
        except Exception as e:
            if logger:
                logger.warning(f"Attribution failed for q{result.get('question_idx', '?')}: {e}")
            result["attributions"] = None
            result["attributions_pooled"] = None
            result["attribution_metadata"] = None
    
    if cleanup_model:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def cleanup_replacement_model(model):
    """Clean up ReplacementModel and free GPU memory.
    
    Args:
        model: ReplacementModel instance
    """
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_tokenizer(model_name: str = None):
    """Load HuggingFace tokenizer.
    
    Args:
        model_name: Model name (defaults to instruct model)
        
    Returns:
        Tokenizer instance
    """
    from transformers import AutoTokenizer
    
    if model_name is None:
        model_name = DEFAULT_ATTRIBUTION_MODEL
    
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

