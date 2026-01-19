"""Filter safety task data by disagreement between Base and Instruct models.

Computes disagreement = inst_refusal_rate - base_refusal_rate.
Adapted from Jailbreak_experiment/filter_by_disagreement.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json

# Import config
from downstream.config import SAFETY_CONFIG


@dataclass
class FilteredPrompt:
    """A prompt that passed filtering with computed metrics."""
    prompt_id: str
    behavior: str
    category: str
    base_prompt: str
    instruct_prompt: str
    
    # Refusal metrics
    base_refusal_rate: float
    instruct_refusal_rate: float
    disagreement: float  # inst - base
    
    # Response quality metrics
    avg_base_length: float
    avg_instruct_length: float
    
    # Data for downstream pipeline
    base_continuations: List[Dict[str, Any]]
    instruct_continuations: List[Dict[str, Any]]
    base_judgments: List[Dict[str, Any]]
    instruct_judgments: List[Dict[str, Any]]


def compute_disagreement(
    base_refusal_rate: float,
    instruct_refusal_rate: float,
) -> float:
    """Compute disagreement between base and instruct models.
    
    Args:
        base_refusal_rate: Refusal rate of base model
        instruct_refusal_rate: Refusal rate of instruct model
        
    Returns:
        Disagreement score (instruct - base)
    """
    return instruct_refusal_rate - base_refusal_rate


def compute_avg_length(continuations: List[Dict[str, Any]]) -> float:
    """Compute average response length in tokens.
    
    Args:
        continuations: List of continuation dicts
        
    Returns:
        Average length in tokens
    """
    if not continuations:
        return 0.0
    return np.mean([
        cont.get("n_tokens", len(cont.get("text", ""))) 
        for cont in continuations
    ])


def filter_by_disagreement(
    rollout_results: Dict[str, Dict[str, Any]],
    top_k: int = None,
    min_response_length: int = None,
    min_disagreement: float = None,
    exclude_negative_disagreement: bool = None,
    logger=None,
) -> Tuple[List[FilteredPrompt], Dict[str, Any]]:
    """Filter and rank prompts by disagreement.
    
    Args:
        rollout_results: Results from rollout_safety_task with judgments
        top_k: Number of top prompts to select
        min_response_length: Minimum average response length
        min_disagreement: Minimum disagreement threshold
        exclude_negative_disagreement: Exclude cases where base refuses more
        logger: Logger instance
        
    Returns:
        Tuple of (filtered prompts, filter statistics)
    """
    if top_k is None:
        top_k = SAFETY_CONFIG["n_filtered_prompts"]
    if min_response_length is None:
        min_response_length = SAFETY_CONFIG["min_response_length"]
    if min_disagreement is None:
        min_disagreement = SAFETY_CONFIG["min_disagreement"]
    if exclude_negative_disagreement is None:
        exclude_negative_disagreement = SAFETY_CONFIG["exclude_negative_disagreement"]
    
    filtered = []
    stats = {
        "total": 0,
        "missing_judgments": 0,
        "too_short": 0,
        "negative_disagreement": 0,
        "below_threshold": 0,
        "passed": 0,
    }
    
    for prompt_id, data in rollout_results.items():
        stats["total"] += 1
        
        # Check for judgments
        base_judgments = data.get("base_judgments", [])
        instruct_judgments = data.get("instruct_judgments", [])
        
        if not base_judgments or not instruct_judgments:
            stats["missing_judgments"] += 1
            if logger:
                logger.warning(f"No judgments for {prompt_id}")
            continue
        
        # Get continuations
        base_conts = data.get("base_continuations", [])
        instruct_conts = data.get("instruct_continuations", [])
        
        # Compute metrics
        base_refusal_rate = data.get("base_refusal_rate", 
            sum(1 for j in base_judgments if j.get("is_refusal", False)) / len(base_judgments))
        instruct_refusal_rate = data.get("instruct_refusal_rate",
            sum(1 for j in instruct_judgments if j.get("is_refusal", False)) / len(instruct_judgments))
        disagreement = compute_disagreement(base_refusal_rate, instruct_refusal_rate)
        
        avg_base_len = compute_avg_length(base_conts)
        avg_instruct_len = compute_avg_length(instruct_conts)
        
        # Filter: Response length
        if avg_instruct_len < min_response_length:
            stats["too_short"] += 1
            continue
        
        # Filter: Negative disagreement
        if exclude_negative_disagreement and disagreement < 0:
            stats["negative_disagreement"] += 1
            continue
        
        # Filter: Minimum disagreement threshold
        if disagreement < min_disagreement:
            stats["below_threshold"] += 1
            continue
        
        # Passed all filters
        stats["passed"] += 1
        
        filtered.append(FilteredPrompt(
            prompt_id=prompt_id,
            behavior=data.get("behavior", ""),
            category=data.get("category", "unknown"),
            base_prompt=data.get("base_prompt", ""),
            instruct_prompt=data.get("instruct_prompt", ""),
            base_refusal_rate=base_refusal_rate,
            instruct_refusal_rate=instruct_refusal_rate,
            disagreement=disagreement,
            avg_base_length=avg_base_len,
            avg_instruct_length=avg_instruct_len,
            base_continuations=base_conts,
            instruct_continuations=instruct_conts,
            base_judgments=base_judgments,
            instruct_judgments=instruct_judgments,
        ))
    
    # Sort by disagreement (highest first)
    filtered.sort(key=lambda x: x.disagreement, reverse=True)
    
    # Select top-k
    selected = filtered[:top_k]
    
    if logger:
        logger.info(f"\nFiltering statistics:")
        logger.info(f"  Total prompts: {stats['total']}")
        logger.info(f"  Missing judgments: {stats['missing_judgments']}")
        logger.info(f"  Too short: {stats['too_short']}")
        logger.info(f"  Negative disagreement: {stats['negative_disagreement']}")
        logger.info(f"  Below threshold: {stats['below_threshold']}")
        logger.info(f"  Passed: {stats['passed']}")
        logger.info(f"  Selected (top-{top_k}): {len(selected)}")
    
    stats["selected"] = len(selected)
    
    return selected, stats


def categorize_by_disagreement(
    prompts: List[FilteredPrompt],
    high_threshold: float = None,
    medium_threshold: float = None,
) -> Dict[str, List[FilteredPrompt]]:
    """Categorize prompts by disagreement level.
    
    Args:
        prompts: Filtered and sorted prompts
        high_threshold: Threshold for "high" disagreement
        medium_threshold: Threshold for "medium" disagreement
        
    Returns:
        Dict with 'high', 'medium', 'low' keys
    """
    if high_threshold is None:
        high_threshold = SAFETY_CONFIG["high_disagreement_threshold"]
    if medium_threshold is None:
        medium_threshold = SAFETY_CONFIG["medium_disagreement_threshold"]
    
    categories = {
        "high": [],
        "medium": [],
        "low": [],
    }
    
    for prompt in prompts:
        if prompt.disagreement >= high_threshold:
            categories["high"].append(prompt)
        elif prompt.disagreement >= medium_threshold:
            categories["medium"].append(prompt)
        else:
            categories["low"].append(prompt)
    
    return categories


def prepare_for_clustering(
    prompts: List[FilteredPrompt],
    model_type: str = "instruct",
) -> List[Dict[str, Any]]:
    """Prepare filtered prompts for clustering pipeline.
    
    Args:
        prompts: Filtered prompts
        model_type: Which model's continuations to use ("base" or "instruct")
        
    Returns:
        List of dicts ready for embedding/attribution computation
    """
    prepared = []
    
    for prompt in prompts:
        if model_type == "base":
            continuations = prompt.base_continuations
            judgments = prompt.base_judgments
            prefix = prompt.base_prompt
        else:
            continuations = prompt.instruct_continuations
            judgments = prompt.instruct_judgments
            prefix = prompt.instruct_prompt
        
        for i, (cont, judgment) in enumerate(zip(continuations, judgments)):
            prepared.append({
                "sample_id": f"{prompt.prompt_id}_{model_type}_{i:02d}",
                "prompt_id": prompt.prompt_id,
                "model_type": model_type,
                "prefix": prefix,
                "continuation": cont["text"],
                "continuation_token_ids": cont.get("token_ids", []),
                "probability": cont.get("probability", 1.0),
                "is_refusal": judgment.get("is_refusal", False),
                "category": prompt.category,
                "behavior": prompt.behavior,
                "disagreement": prompt.disagreement,
            })
    
    return prepared


def save_filtered_prompts(
    prompts: List[FilteredPrompt],
    output_file: Path,
    metadata: Dict = None,
    logger=None,
):
    """Save filtered prompts to JSON.
    
    Args:
        prompts: Filtered prompts
        output_file: Output file path
        metadata: Optional metadata dict
        logger: Logger instance
    """
    categories = categorize_by_disagreement(prompts)
    
    output_data = {
        "metadata": {
            "n_prompts": len(prompts),
            **(metadata or {}),
        },
        "category_counts": {
            "high": len(categories["high"]),
            "medium": len(categories["medium"]),
            "low": len(categories["low"]),
        },
        "prompts": [asdict(p) for p in prompts],
    }
    
    save_json(output_data, output_file)
    
    if logger:
        logger.info(f"Saved {len(prompts)} filtered prompts to {output_file}")
        logger.info(f"  High disagreement: {len(categories['high'])}")
        logger.info(f"  Medium disagreement: {len(categories['medium'])}")
        logger.info(f"  Low disagreement: {len(categories['low'])}")

