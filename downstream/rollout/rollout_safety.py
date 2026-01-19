"""Rollout continuations from both Base and Instruct models for safety task.

Samples continuations from both models for HarmBench prompts.
Adapted from Jailbreak_experiment/rollout_both_models.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json, load_json

# Import from downstream
from downstream.config import (
    BASE_MODEL_NAME,
    INSTRUCT_MODEL_NAME,
    ROLLOUT_CONFIG,
    OUTPUT_DIR,
)
from downstream.rollout.sample_continuations import (
    SamplingConfig,
    load_vllm_model,
    sample_continuations,
    deduplicate_continuations,
    cleanup_vllm_model,
)


def rollout_single_model(
    model_name: str,
    prompts: List[Dict[str, Any]],
    prompt_key: str,
    config: SamplingConfig = None,
    logger=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run rollout for a single model.
    
    Args:
        model_name: HuggingFace model name
        prompts: List of prompt dicts
        prompt_key: Key to extract prompt text ("base_prompt" or "instruct_prompt")
        config: Sampling configuration
        logger: Logger instance
        
    Returns:
        Dict mapping prompt_id to list of continuations
    """
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info(f"Rollout: {model_name}")
        logger.info(f"{'='*60}")
    
    llm = load_vllm_model(model_name, logger=logger)
    
    # Extract prompts
    prompt_texts = [p[prompt_key] for p in prompts]
    prompt_ids = [p["prompt_id"] for p in prompts]
    
    # Sample
    all_continuations = sample_continuations(
        llm,
        prompt_texts,
        config=config,
        logger=logger,
    )
    
    # Map to prompt IDs
    results = {}
    for prompt_id, conts in zip(prompt_ids, all_continuations):
        results[prompt_id] = conts
    
    # Cleanup
    cleanup_vllm_model(llm)
    
    if logger:
        logger.info(f"Completed rollout for {len(results)} prompts")
    
    return results


def rollout_safety_task(
    prompts: List[Dict[str, Any]],
    n_continuations: int = None,
    max_tokens: int = None,
    temperature: float = None,
    run_base: bool = True,
    run_instruct: bool = True,
    output_dir: Path = None,
    save_results: bool = True,
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """Run rollout for safety task with both models.
    
    Args:
        prompts: List of prompt dicts with base_prompt and instruct_prompt
        n_continuations: Number of continuations per prompt
        max_tokens: Maximum tokens per continuation
        temperature: Sampling temperature
        run_base: Whether to run base model
        run_instruct: Whether to run instruct model
        output_dir: Output directory for results
        save_results: Whether to save results to disk
        logger: Logger instance
        
    Returns:
        Dict mapping prompt_id to result dict with base/instruct continuations
    """
    # Create sampling config
    config = SamplingConfig(
        n_continuations=n_continuations or ROLLOUT_CONFIG["n_continuations"],
        max_tokens=max_tokens or ROLLOUT_CONFIG["max_tokens"],
        temperature=temperature or ROLLOUT_CONFIG["temperature"],
        top_p=ROLLOUT_CONFIG["top_p"],
    )
    
    if logger:
        logger.info("=" * 60)
        logger.info("SAFETY TASK ROLLOUT")
        logger.info("=" * 60)
        logger.info(f"N prompts: {len(prompts)}")
        logger.info(f"N continuations: {config.n_continuations}")
        logger.info(f"Max tokens: {config.max_tokens}")
        logger.info(f"Temperature: {config.temperature}")
        logger.info(f"Run base: {run_base}, Run instruct: {run_instruct}")
    
    base_results = {}
    instruct_results = {}
    
    # Run base model
    if run_base:
        base_results = rollout_single_model(
            BASE_MODEL_NAME,
            prompts,
            "base_prompt",
            config,
            logger,
        )
    
    # Run instruct model
    if run_instruct:
        instruct_results = rollout_single_model(
            INSTRUCT_MODEL_NAME,
            prompts,
            "instruct_prompt",
            config,
            logger,
        )
    
    # Combine results
    combined_results = {}
    for prompt in prompts:
        prompt_id = prompt["prompt_id"]
        combined_results[prompt_id] = {
            "prompt_id": prompt_id,
            "behavior": prompt["behavior"],
            "category": prompt["category"],
            "base_prompt": prompt["base_prompt"],
            "instruct_prompt": prompt["instruct_prompt"],
            "base_continuations": base_results.get(prompt_id, []),
            "instruct_continuations": instruct_results.get(prompt_id, []),
        }
    
    # Save results
    if save_results and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "safety_rollout_results.json"
        output_data = {
            "metadata": {
                "base_model": BASE_MODEL_NAME,
                "instruct_model": INSTRUCT_MODEL_NAME,
                "n_continuations": config.n_continuations,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "n_prompts": len(prompts),
            },
            "results": combined_results,
        }
        
        save_json(output_data, output_file)
        if logger:
            logger.info(f"Saved rollout results to {output_file}")
    
    return combined_results


def load_safety_rollout_results(
    results_file: Path,
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """Load saved rollout results.
    
    Args:
        results_file: Path to results JSON file
        logger: Logger instance
        
    Returns:
        Dict mapping prompt_id to result dict
    """
    data = load_json(results_file)
    
    if logger:
        metadata = data.get("metadata", {})
        logger.info(f"Loaded rollout results: {metadata.get('n_prompts', '?')} prompts")
    
    return data.get("results", {})

