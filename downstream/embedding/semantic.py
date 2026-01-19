"""Semantic embedding computation using SentenceTransformer.

Computes contextual continuation embeddings for the semantic view.
"""

import gc
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import config
from downstream.config import EMBEDDING_MODEL, EMBEDDING_CONFIG


def load_embedding_model(
    model_name: str = None,
    device: str = None,
):
    """Load SentenceTransformer model for embeddings.
    
    Args:
        model_name: Model name (uses config default if None)
        device: Device for computation
        
    Returns:
        SentenceTransformer instance
    """
    from sentence_transformers import SentenceTransformer
    
    if model_name is None:
        model_name = EMBEDDING_MODEL
    if device is None:
        device = EMBEDDING_CONFIG["device"]
    
    return SentenceTransformer(model_name, device=device)


def compute_semantic_embeddings(
    prefix: str,
    continuations: List[str],
    embedding_model,
    batch_size: int = None,
    device: str = None,
    normalize: bool = None,
    logger=None,
) -> np.ndarray:
    """Compute contextual continuation embeddings.
    
    Embeds full sequences (prefix + continuation) and extracts
    embeddings for the continuation tokens.
    
    Args:
        prefix: Prefix text
        continuations: List of continuation texts
        embedding_model: SentenceTransformer model
        batch_size: Batch size for computation
        device: Device for computation
        normalize: Whether to L2-normalize embeddings
        logger: Optional logger
        
    Returns:
        Embeddings array [n_continuations, embedding_dim]
    """
    import torch.nn.functional as F
    
    if batch_size is None:
        batch_size = EMBEDDING_CONFIG["batch_size"]
    if device is None:
        device = EMBEDDING_CONFIG["device"]
    if normalize is None:
        normalize = EMBEDDING_CONFIG["normalize"]
    
    tokenizer = embedding_model.tokenizer
    
    # Tokenize prefix to find boundary
    prefix_encoding = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_len = len(prefix_encoding)
    
    # Build full sequences
    full_sequences = [prefix + cont for cont in continuations]
    
    embeddings = []
    n_batches = (len(full_sequences) + batch_size - 1) // batch_size
    
    iterator = range(n_batches)
    if logger:
        iterator = tqdm(iterator, desc="Computing embeddings", leave=False)
    
    for batch_idx in iterator:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(full_sequences))
        batch_seqs = full_sequences[batch_start:batch_end]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=EMBEDDING_CONFIG.get("max_length", 2048)
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            transformer = embedding_model[0].auto_model
            outputs = transformer(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Extract continuation tokens and pool
        for i in range(len(batch_seqs)):
            seq_encoding = tokenizer.encode(batch_seqs[i], add_special_tokens=False)
            seq_len = len(seq_encoding)
            
            # Get continuation hidden states
            cont_hidden = hidden_states[i, prefix_len:seq_len, :]
            
            if cont_hidden.shape[0] == 0:
                emb = torch.zeros(hidden_states.shape[-1], device=device)
            else:
                emb = cont_hidden.mean(dim=0)  # Mean pooling
            
            if normalize:
                emb = F.normalize(emb, p=2, dim=0)
            
            embeddings.append(emb.cpu().numpy())
    
    return np.stack(embeddings)


def compute_embeddings_batch(
    data: List[Dict[str, Any]],
    embedding_model=None,
    model_name: str = None,
    batch_size: int = None,
    device: str = None,
    normalize: bool = None,
    logger=None,
) -> List[Dict[str, Any]]:
    """Compute embeddings for a batch of data.
    
    Groups data by prefix for efficient computation.
    
    Args:
        data: List of dicts with 'prefix' and 'continuation' keys
        embedding_model: Pre-loaded model (loads new one if None)
        model_name: Model name if loading new model
        batch_size: Batch size
        device: Device
        normalize: Whether to normalize
        logger: Logger instance
        
    Returns:
        Updated data with 'embedding' field added
    """
    if embedding_model is None:
        embedding_model = load_embedding_model(model_name, device)
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
        logger.info(f"Computing embeddings for {len(data)} samples across {len(by_prefix)} prefixes")
    
    # Process each prefix group
    for prefix, items in by_prefix.items():
        continuations = [item["continuation"] for item in items]
        
        embeddings = compute_semantic_embeddings(
            prefix,
            continuations,
            embedding_model,
            batch_size=batch_size,
            device=device,
            normalize=normalize,
        )
        
        for item, emb in zip(items, embeddings):
            item["embedding"] = emb
    
    if cleanup_model:
        del embedding_model
        gc.collect()
        torch.cuda.empty_cache()
    
    return data


def compute_embeddings_for_results(
    results: List[Dict[str, Any]],
    embedding_model=None,
    logger=None,
) -> List[Dict[str, Any]]:
    """Compute embeddings for rollout results.
    
    Args:
        results: List of result dicts (from rollout)
        embedding_model: Pre-loaded model
        logger: Logger instance
        
    Returns:
        Updated results with embeddings
    """
    if embedding_model is None:
        embedding_model = load_embedding_model()
        cleanup_model = True
    else:
        cleanup_model = False
    
    for result in tqdm(results, desc="Computing embeddings"):
        prefix = result.get("prompt", "")
        continuations = [c["text"] for c in result.get("continuations", [])]
        
        if continuations:
            embeddings = compute_semantic_embeddings(
                prefix,
                continuations,
                embedding_model,
            )
            result["embeddings"] = embeddings
        else:
            result["embeddings"] = np.array([])
    
    if cleanup_model:
        del embedding_model
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def cleanup_embedding_model(embedding_model):
    """Clean up embedding model and free GPU memory.
    
    Args:
        embedding_model: SentenceTransformer instance
    """
    del embedding_model
    gc.collect()
    torch.cuda.empty_cache()

