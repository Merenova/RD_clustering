"""Shared vLLM sampling utilities for continuation rollout.

Provides common sampling functionality used by both safety and hallucination tasks.
"""

import gc
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import config
from downstream.config import ROLLOUT_CONFIG, VLLM_CONFIG


@dataclass
class SamplingConfig:
    """Configuration for continuation sampling."""
    n_continuations: int = ROLLOUT_CONFIG["n_continuations"]
    max_tokens: int = ROLLOUT_CONFIG["max_tokens"]
    temperature: float = ROLLOUT_CONFIG["temperature"]
    top_p: float = ROLLOUT_CONFIG["top_p"]
    stop_tokens: List[str] = field(default_factory=list)
    logprobs: int = 1  # Number of logprobs to return


def create_sampling_params(config: SamplingConfig = None):
    """Create vLLM SamplingParams from config.
    
    Args:
        config: SamplingConfig instance, uses defaults if None
        
    Returns:
        vLLM SamplingParams instance
    """
    from vllm import SamplingParams
    
    if config is None:
        config = SamplingConfig()
    
    return SamplingParams(
        n=config.n_continuations,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stop=config.stop_tokens if config.stop_tokens else None,
        logprobs=config.logprobs,
    )


def load_vllm_model(
    model_name: str,
    gpu_memory_utilization: float = None,
    tensor_parallel_size: int = None,
    max_model_len: int = None,
    dtype: str = "bfloat16",
    logger=None,
):
    """Load vLLM model for sampling.
    
    Args:
        model_name: HuggingFace model name
        gpu_memory_utilization: GPU memory fraction to use
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        dtype: Model dtype
        logger: Optional logger
        
    Returns:
        vLLM LLM instance
    """
    from vllm import LLM
    
    if gpu_memory_utilization is None:
        gpu_memory_utilization = VLLM_CONFIG["gpu_memory_utilization"]
    if tensor_parallel_size is None:
        tensor_parallel_size = VLLM_CONFIG["tensor_parallel_size"]
    if max_model_len is None:
        max_model_len = VLLM_CONFIG["max_model_len"]
    
    if logger:
        logger.info(f"Loading vLLM model: {model_name}")
        logger.info(f"  GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"  Max model length: {max_model_len}")
    
    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    
    return llm


def sample_continuations(
    llm,
    prompts: List[str],
    config: SamplingConfig = None,
    batch_size: int = 32,
    show_progress: bool = True,
    logger=None,
) -> List[List[Dict[str, Any]]]:
    """Sample continuations for a batch of prompts.
    
    Args:
        llm: vLLM LLM instance
        prompts: List of prompt strings
        config: Sampling configuration
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
        logger: Optional logger
        
    Returns:
        List of continuation lists, one per prompt
        Each continuation is a dict with: text, token_ids, logprob, probability
    """
    import numpy as np
    
    sampling_params = create_sampling_params(config)
    
    if logger:
        logger.info(f"Sampling {len(prompts)} prompts with {sampling_params.n} continuations each")
    
    all_continuations = []
    
    # Process in batches
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    iterator = range(0, len(prompts), batch_size)
    
    if show_progress:
        iterator = tqdm(iterator, desc="Sampling", total=n_batches)
    
    for batch_start in iterator:
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        
        # Generate
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        
        # Extract continuations
        for request_output in outputs:
            continuations = []
            for completion in request_output.outputs:
                token_ids = list(completion.token_ids) if completion.token_ids else []
                
                # Extract logprobs
                total_logprob = -10.0  # Default if not available
                if completion.logprobs:
                    try:
                        logprob_sum = 0.0
                        for pos_dict in completion.logprobs:
                            if isinstance(pos_dict, dict):
                                for token_id, lp_obj in pos_dict.items():
                                    if hasattr(lp_obj, 'logprob'):
                                        logprob_sum += lp_obj.logprob
                                    elif isinstance(lp_obj, (int, float)):
                                        logprob_sum += float(lp_obj)
                                    break
                            elif hasattr(pos_dict, 'logprob'):
                                logprob_sum += pos_dict.logprob
                        total_logprob = logprob_sum
                    except Exception:
                        total_logprob = -10.0
                
                n_tokens = len(token_ids) if token_ids else 1
                logprob_per_token = total_logprob / n_tokens
                
                continuations.append({
                    "text": completion.text.strip(),
                    "token_ids": token_ids,
                    "n_tokens": n_tokens,
                    "logprob": total_logprob,
                    "logprob_per_token": logprob_per_token,
                    "probability": float(np.exp(logprob_per_token)),
                })
            
            all_continuations.append(continuations)
    
    return all_continuations


def deduplicate_continuations(
    continuations: List[Dict[str, Any]],
    sort_by_prob: bool = True,
) -> List[Dict[str, Any]]:
    """Remove duplicate continuations by text.

    Args:
        continuations: List of continuation dicts
        sort_by_prob: If True, sort by probability descending

    Returns:
        Deduplicated list
    """
    seen = set()
    unique = []
    for cont in continuations:
        text = cont.get("text", "").strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(cont)

    if sort_by_prob:
        unique.sort(key=lambda x: x.get("probability", 0), reverse=True)
    
    return unique


def cleanup_vllm_model(llm):
    """Clean up vLLM model and free GPU memory.
    
    Args:
        llm: vLLM LLM instance
    """
    del llm
    gc.collect()
    torch.cuda.empty_cache()

