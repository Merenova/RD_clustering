"""Robustness testing for safety motifs.

Tests whether model behavior is consistent under perturbations.
Used primarily for Exp5 (Safety Application) to measure refusal robustness.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.perturbation.generator import generate_perturbations


def _extract_vllm_text(outputs: Any) -> str:
    """Extract text from vLLM RequestOutput objects.
    
    vLLM returns list of RequestOutput, each with .outputs list of CompletionOutput.
    We take the first completion from the first request.
    
    Args:
        outputs: vLLM generate() return value
        
    Returns:
        Generated text string
    """
    if outputs is None:
        return ""
    
    # Handle list of RequestOutput
    if isinstance(outputs, list):
        if len(outputs) == 0:
            return ""
        first_output = outputs[0]
    else:
        first_output = outputs
    
    # RequestOutput has .outputs attribute (list of CompletionOutput)
    if hasattr(first_output, "outputs"):
        completions = first_output.outputs
        if completions and len(completions) > 0:
            # CompletionOutput has .text attribute
            if hasattr(completions[0], "text"):
                return completions[0].text
    
    # Fallback: try direct .text attribute
    if hasattr(first_output, "text"):
        return first_output.text
    
    # Last resort: convert to string
    return str(first_output)


def test_sample_robustness(
    sample: Dict[str, Any],
    model,
    tokenizer,
    judge_fn: Callable[[str, str], bool],
    perturbation_types: List[str] = None,
    n_perturbations: int = 3,
    max_new_tokens: int = 100,
) -> Dict[str, Any]:
    """Test robustness of a single sample under perturbations.
    
    Args:
        sample: Sample dict with 'prefix' or 'prompt' and 'is_refusal'
        model: Model for generation (HuggingFace or vLLM)
        tokenizer: Tokenizer
        judge_fn: Function(prompt, response) -> is_refusal
        perturbation_types: Types of perturbations to test
        n_perturbations: Number of perturbations per type
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dict with:
        - robustness_score: Consistency rate [0, 1]
        - n_perturbations_tested: Number tested
        - consistent_count: Number consistent
        - results: Detailed per-perturbation results
    """
    if perturbation_types is None:
        perturbation_types = ["typo", "paraphrase", "rephrase"]
    
    prompt = sample.get("prefix", sample.get("prompt", ""))
    original_refusal = sample.get("is_refusal", False)
    
    # Generate perturbations
    perturbations = generate_perturbations(
        prompt,
        perturbation_types=perturbation_types,
        n_perturbations=n_perturbations,
    )
    
    results = []
    consistent_count = 0
    
    for pert in perturbations:
        perturbed_prompt = pert["text"]
        
        try:
            # Generate response
            if hasattr(model, "generate"):
                # Check if it's a HuggingFace model (has device attribute)
                if hasattr(model, "device"):
                    # HuggingFace model
                    inputs = tokenizer(
                        perturbed_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                    response = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                else:
                    # vLLM LLM.generate() returns list of RequestOutput
                    outputs = model.generate([perturbed_prompt], max_tokens=max_new_tokens)
                    response = _extract_vllm_text(outputs)
            else:
                # Try calling as vLLM
                outputs = model.generate([perturbed_prompt], max_tokens=max_new_tokens)
                response = _extract_vllm_text(outputs)
            
            # Judge refusal status
            perturbed_refusal = judge_fn(perturbed_prompt, response)
            is_consistent = perturbed_refusal == original_refusal
            
            if is_consistent:
                consistent_count += 1
            
            results.append({
                "perturbation_type": pert["type"],
                "variant": pert["variant"],
                "perturbed_prompt": perturbed_prompt,
                "response": response[:200],  # Truncate for storage
                "perturbed_refusal": perturbed_refusal,
                "original_refusal": original_refusal,
                "consistent": is_consistent,
            })
            
        except Exception as e:
            results.append({
                "perturbation_type": pert["type"],
                "variant": pert["variant"],
                "perturbed_prompt": perturbed_prompt,
                "error": str(e),
                "consistent": None,
            })
    
    n_valid = sum(1 for r in results if r.get("consistent") is not None)
    robustness_score = consistent_count / n_valid if n_valid > 0 else 0.0
    
    return {
        "robustness_score": float(robustness_score),
        "n_perturbations_tested": n_valid,
        "consistent_count": consistent_count,
        "results": results,
    }


def test_cluster_robustness(
    cluster_samples: List[Dict[str, Any]],
    model,
    tokenizer,
    judge_fn: Callable[[str, str], bool],
    perturbation_types: List[str] = None,
    n_perturbations: int = 3,
    max_samples: int = 10,
    max_new_tokens: int = 100,
    logger=None,
) -> Dict[str, Any]:
    """Test robustness of a cluster under perturbations.
    
    For each sample in the cluster, tests whether the model's behavior
    (refusal/compliance) is consistent when the prompt is perturbed.
    
    Args:
        cluster_samples: Samples belonging to cluster
        model: Model for generation
        tokenizer: Tokenizer
        judge_fn: Function(prompt, response) -> is_refusal
        perturbation_types: Types of perturbations
        n_perturbations: Perturbations per type per sample
        max_samples: Maximum samples to test from cluster
        max_new_tokens: Max tokens to generate
        logger: Optional logger
        
    Returns:
        Dict with:
        - robustness_score: Overall consistency rate
        - n_samples_tested: Number of samples tested
        - n_perturbations_tested: Total perturbations tested
        - robustness_by_type: Per-type breakdown
        - sample_results: Per-sample details
    """
    if perturbation_types is None:
        perturbation_types = ["typo", "paraphrase"]
    
    # Sample from cluster
    samples = cluster_samples[:max_samples]
    
    if logger:
        logger.info(f"Testing robustness on {len(samples)} samples...")
    
    all_results = []
    sample_results = []
    
    for sample in tqdm(samples, desc="Robustness testing", disable=logger is None):
        result = test_sample_robustness(
            sample,
            model,
            tokenizer,
            judge_fn,
            perturbation_types=perturbation_types,
            n_perturbations=n_perturbations,
            max_new_tokens=max_new_tokens,
        )
        
        sample_results.append({
            "robustness_score": result["robustness_score"],
            "n_tested": result["n_perturbations_tested"],
        })
        
        all_results.extend(result["results"])
    
    # Compute overall robustness
    valid_results = [r for r in all_results if r.get("consistent") is not None]
    consistent_count = sum(1 for r in valid_results if r["consistent"])
    overall_robustness = consistent_count / len(valid_results) if valid_results else 0.0
    
    # Per-type breakdown
    robustness_by_type = {}
    for ptype in perturbation_types:
        type_results = [r for r in valid_results if r.get("perturbation_type") == ptype]
        if type_results:
            type_consistent = sum(1 for r in type_results if r["consistent"])
            robustness_by_type[ptype] = type_consistent / len(type_results)
        else:
            robustness_by_type[ptype] = 0.0
    
    return {
        "robustness_score": float(overall_robustness),
        "n_samples_tested": len(samples),
        "n_perturbations_tested": len(valid_results),
        "consistent_count": consistent_count,
        "robustness_by_type": robustness_by_type,
        "sample_results": sample_results,
    }


def create_refusal_judge(keywords: List[str] = None) -> Callable[[str, str], bool]:
    """Create a simple keyword-based refusal judge.
    
    Args:
        keywords: List of refusal keywords (uses defaults if None)
        
    Returns:
        Function(prompt, response) -> is_refusal
    """
    if keywords is None:
        from downstream.config import REFUSAL_KEYWORDS
        keywords = REFUSAL_KEYWORDS
    
    def judge_fn(prompt: str, response: str) -> bool:
        response_lower = response.lower()
        return any(kw in response_lower for kw in keywords)
    
    return judge_fn

