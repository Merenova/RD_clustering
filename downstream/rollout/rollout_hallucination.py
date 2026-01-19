"""Rollout continuations for hallucination task.

Samples continuations from instruct model for PopQA questions.
Adapted from hypotheses_experiment/sample_and_filter.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json, load_json

# Import from downstream
from downstream.config import (
    INSTRUCT_MODEL_NAME,
    ROLLOUT_CONFIG,
    HALLUCINATION_CONFIG,
    OUTPUT_DIR,
)
from downstream.data.load_popqa import prepare_prompt
from downstream.rollout.sample_continuations import (
    SamplingConfig,
    load_vllm_model,
    sample_continuations,
    deduplicate_continuations,
    cleanup_vllm_model,
)


def rollout_hallucination_task(
    questions: List[Dict[str, Any]],
    n_continuations: int = None,
    max_tokens: int = None,
    temperature: float = None,
    prompt_mode: str = "question",
    output_dir: Path = None,
    save_results: bool = True,
    logger=None,
) -> List[Dict[str, Any]]:
    """Run rollout for hallucination task.
    
    Args:
        questions: List of question dicts with 'question' and 'correct_answers'
        n_continuations: Number of continuations per question
        max_tokens: Maximum tokens per continuation
        temperature: Sampling temperature
        prompt_mode: "question" for chat template, "cloze" for prefix completion
        output_dir: Output directory for results
        save_results: Whether to save results to disk
        logger: Logger instance
        
    Returns:
        List of result dicts with continuations and metadata
    """
    from transformers import AutoTokenizer
    
    # Create sampling config
    config = SamplingConfig(
        n_continuations=n_continuations or ROLLOUT_CONFIG["n_continuations"],
        max_tokens=max_tokens or ROLLOUT_CONFIG["max_tokens"],
        temperature=temperature or ROLLOUT_CONFIG["temperature"],
        top_p=ROLLOUT_CONFIG["top_p"],
    )
    
    if logger:
        logger.info("=" * 60)
        logger.info("HALLUCINATION TASK ROLLOUT")
        logger.info("=" * 60)
        logger.info(f"N questions: {len(questions)}")
        logger.info(f"N continuations: {config.n_continuations}")
        logger.info(f"Max tokens: {config.max_tokens}")
        logger.info(f"Temperature: {config.temperature}")
        logger.info(f"Model: {INSTRUCT_MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_NAME, trust_remote_code=True)
    
    # Prepare prompts
    prepared_data = []
    prompt_texts = []
    
    for q_idx, q_data in enumerate(questions):
        question = q_data.get("question", q_data.get("query", ""))
        correct_answers = q_data.get("correct_answers", q_data.get("answers", []))
        
        prompt = prepare_prompt(question, tokenizer, mode=prompt_mode)
        
        prepared_data.append({
            "question_idx": q_idx,
            "question": question,
            "prompt": prompt,
            "correct_answers": correct_answers,
            "metadata": {k: v for k, v in q_data.items() 
                        if k not in ["question", "correct_answers", "query", "answers"]},
        })
        prompt_texts.append(prompt)
    
    if logger:
        logger.info(f"Prepared {len(prompt_texts)} prompts")
    
    # Load model and sample
    llm = load_vllm_model(INSTRUCT_MODEL_NAME, logger=logger)
    
    all_continuations = sample_continuations(
        llm,
        prompt_texts,
        config=config,
        logger=logger,
    )
    
    # Cleanup
    cleanup_vllm_model(llm)
    
    # Combine results
    results = []
    for prep_data, conts in zip(prepared_data, all_continuations):
        # Deduplicate
        conts = deduplicate_continuations(conts)
        
        results.append({
            "question_idx": prep_data["question_idx"],
            "question": prep_data["question"],
            "prompt": prep_data["prompt"],
            "correct_answers": prep_data["correct_answers"],
            "metadata": prep_data["metadata"],
            "continuations": conts,
        })
    
    # Save results
    if save_results and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "hallucination_rollout_results.json"
        output_data = {
            "metadata": {
                "model": INSTRUCT_MODEL_NAME,
                "n_continuations": config.n_continuations,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "n_questions": len(questions),
                "prompt_mode": prompt_mode,
            },
            "results": results,
        }
        
        save_json(output_data, output_file)
        if logger:
            logger.info(f"Saved rollout results to {output_file}")
    
    return results


def load_hallucination_rollout_results(
    results_file: Path,
    logger=None,
) -> List[Dict[str, Any]]:
    """Load saved rollout results.
    
    Args:
        results_file: Path to results JSON file
        logger: Logger instance
        
    Returns:
        List of result dicts
    """
    data = load_json(results_file)
    
    if logger:
        metadata = data.get("metadata", {})
        logger.info(f"Loaded rollout results: {metadata.get('n_questions', '?')} questions")
    
    return data.get("results", [])

