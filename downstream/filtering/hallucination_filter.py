"""Filter hallucination task data by entropy criteria.

First filter: Answer entropy (probability-weighted correct/wrong distribution)
Second filter: Embedding entropy (pairwise cosine similarity distribution)
Adapted from hypotheses_experiment/sample_and_filter.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json

# Import config
from downstream.config import HALLUCINATION_CONFIG


def compute_binary_entropy(
    correct_prob_sum: float,
    wrong_prob_sum: float,
) -> float:
    """Compute binary entropy from probability sums.
    
    Entropy is maximized when correct_prob_sum ≈ wrong_prob_sum (≈0.5 each).
    
    Args:
        correct_prob_sum: Sum of probabilities for correct answers
        wrong_prob_sum: Sum of probabilities for wrong answers
    
    Returns:
        Binary entropy in [0, 1] (1 = max uncertainty)
    """
    total = correct_prob_sum + wrong_prob_sum
    if total < 1e-10:
        return 0.0
    
    p_correct = correct_prob_sum / total
    p_wrong = wrong_prob_sum / total
    
    # Avoid log(0)
    if p_correct < 1e-10 or p_wrong < 1e-10:
        return 0.0
    
    # Binary entropy: -p*log2(p) - (1-p)*log2(1-p)
    h = -p_correct * np.log2(p_correct) - p_wrong * np.log2(p_wrong)
    return float(h)


def compute_pairwise_cosine_entropy(
    embeddings: np.ndarray,
    normalize: bool = True,
    n_bins: int = 50,
) -> float:
    """Compute entropy of pairwise cosine similarity distribution.
    
    High entropy = diverse embeddings.
    
    Args:
        embeddings: [n_samples, embedding_dim]
        normalize: Whether to L2-normalize embeddings first
        n_bins: Number of histogram bins
    
    Returns:
        Entropy of the similarity distribution
    """
    if len(embeddings) < 2:
        return 0.0
    
    # Normalize if needed
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings = embeddings / norms
    
    # Compute pairwise cosine similarities
    sim_matrix = embeddings @ embeddings.T
    
    # Extract upper triangle (excluding diagonal)
    n = len(embeddings)
    triu_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[triu_indices]
    
    if len(similarities) == 0:
        return 0.0
    
    # Convert similarities to [0, 1] range
    similarities_shifted = (similarities + 1) / 2
    
    # Bin the similarities
    actual_bins = min(n_bins, len(similarities) // 5 + 1)
    actual_bins = max(actual_bins, 2)
    
    hist, _ = np.histogram(similarities_shifted, bins=actual_bins, range=(0, 1), density=True)
    
    # Normalize to get probability distribution
    hist = hist / (hist.sum() + 1e-10)
    
    # Compute entropy
    h = scipy_entropy(hist + 1e-10)
    
    return float(h)


def compute_answer_entropy(result: Dict[str, Any]) -> float:
    """Compute answer entropy for a result dict.
    
    Args:
        result: Result dict with continuations and correctness info
        
    Returns:
        Answer entropy value
    """
    correct_prob_sum = result.get("correct_prob_sum", 0.0)
    wrong_prob_sum = result.get("wrong_prob_sum", 0.0)
    
    # If not pre-computed, compute from continuations
    if correct_prob_sum == 0.0 and wrong_prob_sum == 0.0:
        continuations = result.get("continuations", [])
        for cont in continuations:
            prob = cont.get("probability", 1.0 / len(continuations))
            if cont.get("is_correct", False):
                correct_prob_sum += prob
            else:
                wrong_prob_sum += prob
    
    return compute_binary_entropy(correct_prob_sum, wrong_prob_sum)


def filter_by_answer_entropy(
    results: List[Dict[str, Any]],
    top_k: int = None,
    min_entropy: float = None,
    logger=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """First filter: Select top-K by answer entropy.
    
    High answer entropy = ~50/50 probability split between correct/wrong.
    
    Args:
        results: List of rollout results with correctness info
        top_k: Number of top questions to keep
        min_entropy: Minimum entropy threshold
        logger: Logger instance
    
    Returns:
        Tuple of (filtered results, filter statistics)
    """
    if top_k is None:
        top_k = HALLUCINATION_CONFIG["top_k_answer_entropy"]
    if min_entropy is None:
        min_entropy = HALLUCINATION_CONFIG["min_answer_entropy"]
    
    # Compute entropy for each question
    entropies = []
    for result in results:
        h = compute_answer_entropy(result)
        result["answer_entropy"] = h
        entropies.append({
            "question_idx": result.get("question_idx", len(entropies)),
            "entropy": h,
            "correct_prob_sum": result.get("correct_prob_sum", 0.0),
            "wrong_prob_sum": result.get("wrong_prob_sum", 0.0),
            "n_correct": result.get("n_correct", 0),
            "n_wrong": result.get("n_wrong", 0),
        })
    
    # Filter out questions with entropy below threshold
    valid_indices = [i for i, e in enumerate(entropies) if e["entropy"] >= min_entropy]
    
    if logger:
        n_excluded = len(entropies) - len(valid_indices)
        logger.info(f"Excluded {n_excluded} questions with entropy < {min_entropy}")
    
    # Sort valid indices by entropy (descending)
    sorted_valid_indices = sorted(valid_indices, key=lambda i: entropies[i]["entropy"], reverse=True)
    
    # Select top-K
    selected_indices = sorted_valid_indices[:top_k]
    filtered_results = [results[i] for i in selected_indices]
    
    # Statistics
    stats = {
        "total_questions": len(results),
        "valid_questions": len(valid_indices),
        "excluded_low_entropy": len(entropies) - len(valid_indices),
        "selected_questions": len(filtered_results),
        "min_entropy_threshold": min_entropy,
        "max_entropy": entropies[selected_indices[0]]["entropy"] if selected_indices else 0,
        "min_entropy_in_selection": entropies[selected_indices[-1]]["entropy"] if selected_indices else 0,
    }
    
    if logger:
        logger.info(f"Answer entropy filter: {len(results)} -> {len(valid_indices)} -> {len(filtered_results)}")
        logger.info(f"  Entropy range: [{stats['min_entropy_in_selection']:.4f}, {stats['max_entropy']:.4f}]")
    
    return filtered_results, stats


def filter_by_embedding_entropy(
    results: List[Dict[str, Any]],
    top_k: int = None,
    logger=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Second filter: Select top-K by embedding entropy.
    
    High embedding entropy = diverse responses in semantic space.
    Requires embeddings to be pre-computed in results.
    
    Args:
        results: Results with computed embeddings
        top_k: Number of top questions to keep
        logger: Logger instance
    
    Returns:
        Tuple of (filtered results, filter statistics)
    """
    if top_k is None:
        top_k = HALLUCINATION_CONFIG["top_k_embedding_entropy"]
    
    # Compute embedding entropy if not already done
    for result in results:
        if "embedding_entropy" not in result:
            embeddings = result.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                result["embedding_entropy"] = compute_pairwise_cosine_entropy(embeddings)
            else:
                result["embedding_entropy"] = 0.0
    
    # Get entropy for each question
    entropies = [result.get("embedding_entropy", 0.0) for result in results]
    
    # Sort by embedding entropy (descending)
    sorted_indices = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)
    
    # Select top-K
    selected_indices = sorted_indices[:top_k]
    filtered_results = [results[i] for i in selected_indices]
    
    # Statistics
    stats = {
        "total_questions": len(results),
        "selected_questions": len(filtered_results),
        "max_entropy": entropies[sorted_indices[0]] if sorted_indices else 0,
        "min_entropy_in_selection": entropies[sorted_indices[min(top_k-1, len(sorted_indices)-1)]] if sorted_indices else 0,
    }
    
    if logger:
        logger.info(f"Embedding entropy filter: {len(results)} -> {len(filtered_results)}")
        logger.info(f"  Entropy range: [{stats['min_entropy_in_selection']:.4f}, {stats['max_entropy']:.4f}]")
    
    return filtered_results, stats


def filter_by_correctness_rate(
    results: List[Dict[str, Any]],
    min_rate: float = 0.1,
    max_rate: float = 0.9,
    logger=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter questions by correctness rate range.
    
    Exclude questions that are too easy (>max_rate) or too hard (<min_rate).
    
    Args:
        results: List of rollout results
        min_rate: Minimum correctness rate
        max_rate: Maximum correctness rate
        logger: Logger instance
        
    Returns:
        Tuple of (filtered results, filter statistics)
    """
    filtered = []
    stats = {
        "total": len(results),
        "too_easy": 0,
        "too_hard": 0,
        "passed": 0,
    }
    
    for result in results:
        rate = result.get("correctness_rate", 0.5)
        
        if rate > max_rate:
            stats["too_easy"] += 1
        elif rate < min_rate:
            stats["too_hard"] += 1
        else:
            stats["passed"] += 1
            filtered.append(result)
    
    if logger:
        logger.info(f"Correctness rate filter: {stats['total']} -> {stats['passed']}")
        logger.info(f"  Too easy (>{max_rate}): {stats['too_easy']}")
        logger.info(f"  Too hard (<{min_rate}): {stats['too_hard']}")
    
    return filtered, stats


def prepare_for_clustering(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepare filtered results for clustering pipeline.
    
    Args:
        results: Filtered results with continuations
        
    Returns:
        List of dicts ready for embedding/attribution computation
    """
    prepared = []
    
    for result in results:
        question_idx = result.get("question_idx", 0)
        prefix = result.get("prompt", "")
        continuations = result.get("continuations", [])
        correct_answers = result.get("correct_answers", [])
        
        for i, cont in enumerate(continuations):
            prepared.append({
                "sample_id": f"q{question_idx}_{i:02d}",
                "question_idx": question_idx,
                "prefix": prefix,
                "continuation": cont["text"],
                "continuation_token_ids": cont.get("token_ids", []),
                "probability": cont.get("probability", 1.0),
                "is_correct": cont.get("is_correct", False),
                "correct_answers": correct_answers,
                "metadata": result.get("metadata", {}),
            })
    
    return prepared


def save_filtered_results(
    results: List[Dict[str, Any]],
    output_file: Path,
    metadata: Dict = None,
    logger=None,
):
    """Save filtered results to JSON.
    
    Args:
        results: Filtered results
        output_file: Output file path
        metadata: Optional metadata dict
        logger: Logger instance
    """
    # Convert numpy arrays to lists for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        # Avoid serializing large non-JSON objects like PrefixAttributionContext
        elif getattr(obj, "__class__", None) is not None and obj.__class__.__name__ == "PrefixAttributionContext":
            return {
                "prefix_length": getattr(obj, "prefix_length", None),
                "n_layers": getattr(obj, "n_layers", None),
                "n_prefix_features": getattr(obj, "n_prefix_features", None),
                "n_prefix_sources": getattr(obj, "n_prefix_sources", None),
            }
        return obj

    def serialize_prefix_context(prefix_context, question_idx: int):
        prefix_ctx_dir = output_file.parent / "prefix_contexts"
        prefix_ctx_dir.mkdir(parents=True, exist_ok=True)
        prefix_ctx_path = prefix_ctx_dir / f"q{question_idx}_prefix_context.pt"
        torch.save(prefix_context, prefix_ctx_path)
        return {
            "prefix_length": getattr(prefix_context, "prefix_length", None),
            "n_layers": getattr(prefix_context, "n_layers", None),
            "n_prefix_features": getattr(prefix_context, "n_prefix_features", None),
            "n_prefix_sources": getattr(prefix_context, "n_prefix_sources", None),
            "prefix_context_path": str(prefix_ctx_path),
        }

    serializable_results = []
    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            serializable_results.append(convert_for_json(result))
            continue
        question_idx = result.get("question_idx", idx)
        result_copy = {}
        for key, value in result.items():
            if key == "attribution_metadata" and isinstance(value, dict):
                meta_copy = {}
                for meta_key, meta_value in value.items():
                    if (
                        getattr(meta_value, "__class__", None) is not None
                        and meta_value.__class__.__name__ == "PrefixAttributionContext"
                    ):
                        meta_copy[meta_key] = serialize_prefix_context(meta_value, question_idx)
                    else:
                        meta_copy[meta_key] = convert_for_json(meta_value)
                result_copy[key] = meta_copy
            else:
                result_copy[key] = convert_for_json(value)
        serializable_results.append(result_copy)
    
    output_data = {
        "metadata": {
            "n_questions": len(results),
            **(metadata or {}),
        },
        "results": serializable_results,
    }
    
    save_json(output_data, output_file)
    
    if logger:
        logger.info(f"Saved {len(results)} filtered results to {output_file}")

