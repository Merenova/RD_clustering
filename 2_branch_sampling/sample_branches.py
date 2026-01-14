#!/usr/bin/env -S uv run python
"""Sample continuations for prefixes using vLLM.

This script implements the branch sampling algorithm:
1. Discover first tokens via forward pass (cumulative top-p selection)
2. For each first token, sample N distinct continuations
3. Deduplicate continuations and record path probabilities

Note: In the latent_planning pipeline, this is Stage 2 (runs before attribution).
First token discovery is done here via forward pass, not loaded from attribution.
"""

import argparse
import sys
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import PathConfig, SamplingConfig
from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger
from utils.manifest import filter_samples_by_manifest, update_manifest_with_results


class SkipPrefixError(Exception):
    """Raised when a prefix should be skipped (not a failure, just filtered out)."""
    pass


@dataclass
class Continuation:
    """Represents a unique continuation."""
    text: str
    token_ids: List[int]
    logprob: float
    probability: float

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

@torch.no_grad()
def discover_first_tokens_via_forward_pass(
    prefix: str,
    model,  # HuggingFace model
    tokenizer,
    max_n_logits: int,
    desired_logit_prob: float,
    logger=None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Discover first tokens via forward pass using cumulative top-p selection.

    This replaces loading from Stage 2 attribution. Uses the same logic as
    compute_salient_logits: select the smallest logit set whose cumulative
    probability >= desired_logit_prob.

    Args:
        prefix: The prefix text string
        model: HuggingFace model for forward pass
        tokenizer: Tokenizer instance
        max_n_logits: Hard cap on number of first tokens
        desired_logit_prob: Cumulative probability threshold (e.g., 0.95)
        logger: Optional logger instance

    Returns:
        Tuple of (first_tokens_list, prefix_tokens_with_bos)
        - first_tokens_list: List of dicts with rank, token_id, token_text, probability
        - prefix_tokens_with_bos: List of token IDs including BOS at position 0
    """
    if logger:
        logger.info(f"Computing first tokens via forward pass (max={max_n_logits}, p={desired_logit_prob})")

    # Tokenize prefix
    tokens = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)

    # Forward pass
    outputs = model(tokens)
    logits = outputs.logits[0, -1, :]  # Last position logits

    # Compute probabilities and select by cumulative top-p
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, max_n_logits)

    # Find cutoff using cumulative probability threshold
    cumsum = torch.cumsum(top_probs, dim=0)
    cutoff_idx = int(torch.searchsorted(cumsum, desired_logit_prob)) + 1
    cutoff = min(cutoff_idx, max_n_logits)

    # Build first tokens list
    first_tokens = []
    for rank in range(cutoff):
        token_id = top_indices[rank].item()
        prob = top_probs[rank].item()
        first_tokens.append({
            "rank": rank,
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id]),
            "probability": prob,
        })

    # Build prefix tokens with BOS
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    if bos_id is None:
        bos_id = 0  # Fallback

    token_ids_list = tokens[0].tolist()
    
    # Check if BOS is already present to avoid duplication
    if token_ids_list and token_ids_list[0] == bos_id:
        prefix_tokens_with_bos = token_ids_list
    else:
        prefix_tokens_with_bos = [bos_id] + token_ids_list

    if logger:
        total_prob = sum(ft["probability"] for ft in first_tokens)
        logger.info(f"Selected {len(first_tokens)} first tokens with cumulative prob: {total_prob:.4f}")
        logger.info(f"Prefix tokens: {len(prefix_tokens_with_bos)} tokens (including BOS)")

    return first_tokens, prefix_tokens_with_bos


def sample_continuations_natural(
    llm: LLM,
    prefix: str,
    sampling_config: SamplingConfig,
    num_continuations: int,
    max_batches: int,
    logger
) -> Tuple[List[Continuation], int]:
    """Sample continuations naturally from the prefix (without forcing first token).

    Args:
        llm: vLLM LLM instance
        prefix: Prefix text
        sampling_config: Sampling configuration
        num_continuations: Number of distinct continuations to sample
        max_batches: Maximum batches to sample
        logger: Logger instance

    Returns:
        Tuple of (unique continuations, total samples drawn)
    """
    all_samples: List[Dict[str, Any]] = []
    total_samples = 0

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=sampling_config.temperature,
        top_p=sampling_config.nucleus_p,
        max_tokens=sampling_config.max_tokens,
        n=sampling_config.batch_size,  # Sample batch_size at a time
        stop=sampling_config.stop_tokens,
        logprobs=1,  # Get log probabilities (needed for first token)
    )

    for batch_idx in range(max_batches):
        # Generate batch
        outputs = llm.generate([prefix], sampling_params, use_tqdm=False)
        if not outputs:
            break

        request_output = outputs[0]
        batch_samples = []

        for completion in request_output.outputs:
            # Get cumulative logprob
            logprob = completion.cumulative_logprob
            if logprob is None and completion.logprobs is not None:
                # Fallback: sum token logprobs
                token_logprobs = [
                    tl.logprob for tl in completion.logprobs if tl is not None
                ]
                logprob = float(sum(token_logprobs))

            if completion.text.strip():  # Skip empty
                # We need token_ids to identify the first token
                batch_samples.append({
                    "text": completion.text,
                    "token_ids": list(completion.token_ids),
                    "logprob": float(logprob) if logprob is not None else float("-inf"),
                })

        all_samples.extend(batch_samples)
        total_samples += len(batch_samples)

        # Deduplicate and check if we have enough distinct continuations
        continuations = deduplicate_continuations(all_samples)

        # Check stopping condition: enough distinct continuations
        if len(continuations) >= num_continuations:
            # Truncate to exactly num_continuations (keep top by probability)
            continuations = continuations[:num_continuations]
            logger.info(f"  Reached target: {len(continuations)} distinct continuations")
            return continuations, total_samples

    # Return what we have even if we didn't reach the target count
    continuations = deduplicate_continuations(all_samples)
    # Also truncate here in case we have more than target
    continuations = continuations[:num_continuations]
    logger.info(f"  Max batches reached: {len(continuations)} distinct continuations (target: {num_continuations})")
    return continuations, total_samples


def continuations_to_payload(
    continuations: List[Continuation],
    prefix_tokens_with_bos: List[int],
    first_token_id: int
) -> List[Dict[str, Any]]:
    """Convert Continuation objects to JSON-serializable payload.

    Args:
        continuations: List of Continuation objects (sorted by probability)
        prefix_tokens_with_bos: Prefix token IDs including BOS at position 0
        first_token_id: The first token ID for this continuation group

    Returns:
        List of continuation dicts with full_token_ids
    """
    return [
        {
            "text": cont.text,
            "token_ids": cont.token_ids,
            "full_token_ids": prefix_tokens_with_bos + [first_token_id] + cont.token_ids,
            "num_tokens": cont.num_tokens,
            "logprob": cont.logprob,
            "probability": cont.probability,
        }
        for cont in continuations
    ]


def process_prefix(
    prefix: str,
    prefix_id: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    hf_model,  # HuggingFace model for first-token discovery
    sampling_config: SamplingConfig,
    num_continuations: int,
    max_total_continuations: int,
    max_batches: int,
    max_n_logits: int,
    desired_logit_prob: float,
    output_dir: Path,
    logger
) -> Path:
    """Process a single prefix: discover first tokens via forward pass, then sample continuations.

    Args:
        prefix: Prefix text
        prefix_id: Unique identifier
        llm: vLLM instance for continuation sampling
        tokenizer: Tokenizer
        hf_model: HuggingFace model for first-token discovery
        sampling_config: Sampling configuration
        max_total_continuations: Maximum total continuations across all first tokens
        max_batches: Max batches per first token
        max_n_logits: Maximum number of first tokens to consider
        desired_logit_prob: Cumulative probability threshold for first-token selection
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Path to saved output file
    """
    logger.info(f"Processing prefix: {prefix_id}")
    logger.info(f"Prefix text: {prefix[:100]}...")

    # Step 1: Discover first tokens via forward pass (cumulative top-p selection)
    logger.info(f"Discovering first tokens via forward pass (for verification/metadata)...")
    first_tokens_data, prefix_tokens_with_bos = discover_first_tokens_via_forward_pass(
        prefix, hf_model, tokenizer, max_n_logits, desired_logit_prob, logger
    )

    # Convert to the format expected by the rest of the function
    first_token_ids = [ft["token_id"] for ft in first_tokens_data]
    first_token_probs = {ft["token_id"]: ft["probability"] for ft in first_tokens_data}

    # Step 2: Natural Sampling from Prefix
    
    logger.info(f"\nSampling continuations naturally from prefix (max_total={max_total_continuations})...")
    
    continuations, num_samples = sample_continuations_natural(
        llm, prefix, sampling_config, max_total_continuations, max_batches * 10, logger
    )
    
    logger.info(f"Sampled {len(continuations)} natural continuations")
    
    # Step 3: Group by first token
    
    grouped_continuations = defaultdict(list)
    token_metadata = {ft["token_id"]: ft for ft in first_tokens_data}
    
    # Group sampled continuations
    for cont in continuations:
        if not cont.token_ids:
            continue
            
        first_token = cont.token_ids[0]

        rest_token_ids = cont.token_ids[1:]
        
        rest_text = tokenizer.decode(rest_token_ids, clean_up_tokenization_spaces=False)
        
        
        cont_entry = {
            "text": rest_text,
            "token_ids": rest_token_ids,
            "full_token_ids": prefix_tokens_with_bos + cont.token_ids, # BOS + first + rest
            "num_tokens": len(rest_token_ids),
            "logprob": cont.logprob,
            "probability": cont.probability,
        }
        
        grouped_continuations[first_token].append(cont_entry)
        
        # If this is a new first token not in discovery (unlikely with high p, but possible), add metadata
        if first_token not in token_metadata:
            token_text = tokenizer.decode([first_token])
            token_metadata[first_token] = {
                "rank": -1, # Unknown rank
                "token_id": first_token,
                "token_text": token_text,
                "first_token_probability": 0.0, # Unknown prob
            }

    # Step 4: Construct Output Format
    first_token_results = []
    
    # Sort groups by: 1. Rank in discovery (if exists), 2. Total probability mass in samples
    sorted_token_ids = sorted(
        grouped_continuations.keys(),
        key=lambda tid: (
            token_metadata[tid].get("rank", 9999) if token_metadata[tid].get("rank", -1) != -1 else 9999,
            -sum(c["probability"] for c in grouped_continuations[tid])
        )
    )
    
    total_continuations_so_far = 0
    
    for token_id in sorted_token_ids:
        conts = grouped_continuations[token_id]
        meta = token_metadata[token_id]
        
        total_unique_mass = sum(c["probability"] for c in conts)
        
        first_token_results.append({
            "rank": meta.get("rank", -1),
            "token_id": int(token_id),
            "token_text": meta.get("token_text", ""),
            "first_token_probability": float(meta.get("first_token_probability", 0.0)),
            "prompt": prefix + meta.get("token_text", ""), # Approx prompt
            "num_raw_samples": -1, # Not tracked per token anymore
            "num_unique_continuations": len(conts),
            "unique_mass": total_unique_mass,
            "continuations": conts,
        })
        total_continuations_so_far += len(conts)

    # Create output data
    output_data = {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "prefix_tokens_with_bos": prefix_tokens_with_bos,  # Token IDs with BOS at position 0
        "first_tokens_source": "natural_sampling",  # CHANGED from forward_pass_discovery
        "first_tokens_config": {
            "max_n_logits": max_n_logits,
            "desired_logit_prob": desired_logit_prob,
        },
        "max_total_continuations": max_total_continuations,
        "num_first_tokens_available": len(first_token_ids),
        "num_first_tokens_processed": len(first_token_results),
        "total_continuations": total_continuations_so_far,
        "first_tokens": first_token_results,
    }

    # Save output
    output_file = output_dir / f"{prefix_id}_branches.json"
    save_json(output_data, output_file)
    logger.info(f"\nSaved branch samples to: {output_file}")

    # Print statistics
    logger.info(f"Statistics for {prefix_id}:")
    logger.info(f"  First tokens available: {len(first_token_ids)}")
    logger.info(f"  First tokens processed: {len(first_token_results)}")
    logger.info(f"  Total continuations: {total_continuations_so_far} (max: {max_total_continuations})")
    logger.info(f"  Target per token: {num_continuations} (effective: {effective_num_continuations})")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Sample branch continuations for prefixes")
    parser.add_argument(
        "--test-clozes",
        type=Path,
        required=True,
        help="Path to test clozes JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path for both first-token discovery and vLLM continuation sampling"
    )
    # First-token discovery arguments
    parser.add_argument(
        "--max-n-logits",
        type=int,
        default=10,
        help="Maximum number of first tokens to consider (hard cap)"
    )
    parser.add_argument(
        "--desired-logit-prob",
        type=float,
        default=0.95,
        help="Cumulative probability threshold for first-token selection"
    )
    # Continuation sampling arguments
    parser.add_argument(
        "--max-total-continuations",
        type=int,
        default=10000,
        help="Maximum total continuations per prefix. If num_first_tokens * num_continuations exceeds this, num_continuations is adaptively reduced."
    )
    parser.add_argument(
        "--nucleus-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter for continuations"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per continuation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for sampling continuations"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum batches per first token"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: 2_branch_sampling/samples/)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0-1.0)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code for model loading"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (only progress bars)"
    )
    args = parser.parse_args()

    # Setup paths
    paths = PathConfig()
    paths.ensure_dirs()

    if args.output_dir is None:
        args.output_dir = paths.branch_sampling / "samples"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "branch_sampling",
        log_file=paths.branch_sampling / "sample_branches.log",
        level=log_level
    )

    logger.info("=" * 60)
    logger.info("BRANCH SAMPLING WITH FIRST-TOKEN DISCOVERY")
    logger.info("=" * 60)
    logger.info(f"Test clozes: {args.test_clozes}")
    logger.info(f"Model: {args.model}")
    logger.info(f"First-token discovery:")
    logger.info(f"  Max N logits: {args.max_n_logits}")
    logger.info(f"  Desired logit prob: {args.desired_logit_prob}")
    logger.info(f"Continuation sampling:")
    logger.info(f"  Num continuations per token: {args.num_continuations}")
    logger.info(f"  Max total continuations: {args.max_total_continuations}")
    logger.info(f"  Nucleus p: {args.nucleus_p}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load test clozes
    logger.info("\nLoading test clozes...")
    test_data = load_json(args.test_clozes)
    clozes = test_data["clozes"]
    logger.info(f"Loaded {len(clozes)} test clozes")

    # Filter samples based on Stage 1 manifest (data preparation)
    results_dir = paths.results
    all_cloze_ids = [c.get("cloze_id") or c.get("id") for c in clozes]
    available_ids, skipped_ids = filter_samples_by_manifest(
        all_cloze_ids, results_dir, "stage1", logger
    )
    # Filter clozes to only available ones
    available_id_set = set(available_ids)
    clozes = [c for c in clozes if (c.get("cloze_id") or c.get("id")) in available_id_set]
    logger.info(f"Processing {len(clozes)} available clozes (skipped {len(skipped_ids)})")

    # Setup sampling configuration
    sampling_config = SamplingConfig(
        n_samples=args.batch_size,  # Batch size
        nucleus_p=args.nucleus_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )

    # Initialize tokenizer (for decoding tokens in continuations)
    logger.info("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer initialized")

    # Initialize HuggingFace model for first-token discovery
    # This is done BEFORE vLLM because they may conflict on GPU memory
    logger.info("\nInitializing HuggingFace model for first-token discovery...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    hf_model.eval()
    logger.info("HuggingFace model initialized")

    # Initialize vLLM for continuation sampling
    logger.info("\nInitializing vLLM for continuation sampling...")
    logger.info(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"  Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"  Max model length: {args.max_model_len}")
    logger.info(f"  Trust remote code: {args.trust_remote_code}")

    # Delete HF model to free GPU memory for vLLM
    logger.info("  Freeing HuggingFace model to make room for vLLM...")
    del hf_model
    torch.cuda.empty_cache()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("vLLM initialized")

    # Reload HuggingFace model with lower memory usage for first-token discovery
    logger.info("\nReloading HuggingFace model for first-token discovery (after vLLM init)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    hf_model.eval()
    logger.info("HuggingFace model reloaded")

    # Process each prefix
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING PREFIXES")
    logger.info("=" * 60)

    output_files = []
    completed_ids = []
    failed_ids = []
    filtered_ids = []  # Skipped due to first-token count filtering
    errors = {}
    for idx, cloze in enumerate(tqdm(clozes, desc="Processing prefixes")):
        prefix_id = cloze.get("id") or cloze.get("cloze_id") or f"cloze_{idx:03d}"

        # Extract prefix from cloze
        if isinstance(cloze, dict):
            prefix = cloze.get('prefix', cloze.get('text', str(cloze)))
        else:
            prefix = str(cloze)

        # Process prefix
        try:
            output_file = process_prefix(
                prefix, prefix_id, llm, tokenizer, hf_model,
                sampling_config, args.max_total_continuations,
                args.max_batches, args.max_n_logits, args.desired_logit_prob,
                args.output_dir, logger
            )
            output_files.append(str(output_file))
            completed_ids.append(prefix_id)
        except SkipPrefixError as e:
            # Filtered out due to first-token count constraints
            logger.info(f"Skipping {prefix_id}: {str(e)}")
            filtered_ids.append(prefix_id)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to process {prefix_id}: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            failed_ids.append(prefix_id)
            errors[prefix_id] = error_msg

        logger.info("")

    # Save index of all output files
    index_data = {
        "model": args.model,
        "n_prefixes": len(clozes),
        "first_tokens_source": "forward_pass_discovery",
        "first_tokens_config": {
            "max_n_logits": args.max_n_logits,
            "desired_logit_prob": args.desired_logit_prob,
        },
        "max_total_continuations": args.max_total_continuations,
        "sampling_config": {
            "nucleus_p": args.nucleus_p,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
        },
        "output_files": output_files,
    }

    index_file = args.output_dir / "branches_index.json"
    save_json(index_data, index_file)

    all_skipped_ids = skipped_ids + filtered_ids
    update_manifest_with_results(
        results_dir=results_dir,
        stage_name="stage2",
        processed=completed_ids,
        failed=failed_ids,
        skipped=all_skipped_ids,
        logger=logger,
        errors=errors,
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {len(completed_ids)}/{len(clozes)} prefixes")
    logger.info(f"Completed: {len(completed_ids)}, Failed: {len(failed_ids)}, Skipped (stage1): {len(skipped_ids)}, Filtered (first-token count): {len(filtered_ids)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Index file: {index_file}")


if __name__ == "__main__":
    main()
