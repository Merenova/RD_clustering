"""Question paraphrasing for hallucination robustness testing.

Tests whether correct answers remain correct when questions are paraphrased.
Used for Exp6 (Hallucination Application) to distinguish shortcut vs faithful reasoning.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union

import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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


def paraphrase_question(
    question: str,
    n_paraphrases: int = 3,
) -> List[str]:
    """Generate paraphrases of a question.
    
    Uses template-based paraphrasing to create semantic variants
    of the original question.
    
    Args:
        question: Original question text
        n_paraphrases: Number of paraphrases to generate
        
    Returns:
        List of paraphrased questions
    """
    paraphrases = []
    
    # Clean question
    q_content = question.strip()
    if q_content.endswith("?"):
        q_content = q_content[:-1]
    
    # Template-based paraphrasing
    templates = [
        "Can you tell me {q}?",
        "What is {q}?",
        "I'd like to know {q}.",
        "Please answer: {q}?",
        "Tell me {q}.",
        "Could you explain {q}?",
        "Do you know {q}?",
        "I'm curious about {q}.",
        "What can you tell me about {q}?",
        "Help me understand {q}.",
    ]
    
    # Question word replacements
    q_words = {
        "what is": ["what's", "what would be", "tell me"],
        "who is": ["who's", "can you tell me who", "identify"],
        "where is": ["where's", "what's the location of", "can you tell me where"],
        "when was": ["what year was", "in what year was", "tell me when"],
        "how many": ["what's the number of", "count of", "how much"],
    }
    
    # First, try question word replacements
    q_lower = q_content.lower()
    for original, replacements in q_words.items():
        if q_lower.startswith(original):
            for i, repl in enumerate(replacements[:min(2, n_paraphrases)]):
                new_q = repl + q_content[len(original):]
                if not new_q.endswith("?"):
                    new_q += "?"
                paraphrases.append(new_q)
    
    # Then add template-based paraphrases
    for i in range(len(paraphrases), n_paraphrases):
        template = templates[i % len(templates)]
        paraphrased = template.format(q=q_content)
        paraphrases.append(paraphrased)
    
    return paraphrases[:n_paraphrases]


def test_paraphrase_robustness(
    question: str,
    original_answer: str,
    is_correct: bool,
    model,
    tokenizer,
    correctness_fn: Callable[[str, str], bool],
    n_paraphrases: int = 3,
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """Test if answer correctness holds under paraphrased questions.
    
    A correct answer is considered "faithful" if it remains correct
    when the question is paraphrased, and "shortcut" if correctness
    doesn't transfer.
    
    Args:
        question: Original question
        original_answer: Original model answer
        is_correct: Whether original was correct
        model: Model for generation
        tokenizer: Tokenizer
        correctness_fn: Function(question, answer) -> is_correct
        n_paraphrases: Number of paraphrases to test
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dict with:
        - robustness_score: Consistency rate [0, 1]
        - n_paraphrases: Number tested
        - all_correct: Whether all paraphrased answers were correct
        - details: Per-paraphrase results
    """
    paraphrases = paraphrase_question(question, n_paraphrases)
    
    results = []
    consistent_count = 0
    correct_count = 0
    
    for para in paraphrases:
        try:
            # Generate answer to paraphrased question
            if hasattr(model, "generate"):
                # Check if it's a HuggingFace model (has device attribute)
                if hasattr(model, "device"):
                    # HuggingFace model
                    inputs = tokenizer(
                        para,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                    answer = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                else:
                    # vLLM LLM.generate() returns list of RequestOutput
                    outputs = model.generate([para], max_tokens=max_new_tokens)
                    answer = _extract_vllm_text(outputs)
            else:
                # Try calling as vLLM
                outputs = model.generate([para], max_tokens=max_new_tokens)
                answer = _extract_vllm_text(outputs)
            
            # Check correctness
            para_correct = correctness_fn(para, answer)
            is_consistent = para_correct == is_correct
            
            if is_consistent:
                consistent_count += 1
            if para_correct:
                correct_count += 1
            
            results.append({
                "paraphrase": para,
                "answer": answer[:200],  # Truncate for storage
                "is_correct": para_correct,
                "consistent": is_consistent,
            })
            
        except Exception as e:
            results.append({
                "paraphrase": para,
                "error": str(e),
                "is_correct": None,
                "consistent": None,
            })
    
    valid_results = [r for r in results if r.get("consistent") is not None]
    n_valid = len(valid_results)
    
    if n_valid > 0:
        robustness_score = consistent_count / n_valid
        all_correct = correct_count == n_valid
    else:
        robustness_score = 0.0
        all_correct = False
    
    return {
        "robustness_score": float(robustness_score),
        "n_paraphrases": n_valid,
        "all_correct": all_correct,
        "correct_count": correct_count,
        "details": results,
    }


def test_cluster_paraphrase_robustness(
    cluster_samples: List[Dict[str, Any]],
    model,
    tokenizer,
    correctness_fn: Callable[[str, str], bool],
    n_paraphrases: int = 3,
    max_samples: int = 5,
    max_new_tokens: int = 50,
    logger=None,
) -> Dict[str, Any]:
    """Test paraphrase robustness for a cluster of samples.
    
    Args:
        cluster_samples: Samples in cluster (dicts with question, answer, is_correct)
        model: Model for generation
        tokenizer: Tokenizer
        correctness_fn: Function(question, answer) -> is_correct
        n_paraphrases: Paraphrases per sample
        max_samples: Max samples to test
        max_new_tokens: Max tokens to generate
        logger: Optional logger
        
    Returns:
        Dict with cluster-level robustness metrics
    """
    samples = cluster_samples[:max_samples]
    
    if logger:
        logger.info(f"Testing paraphrase robustness on {len(samples)} samples...")
    
    sample_results = []
    
    for sample in tqdm(samples, desc="Paraphrase testing", disable=logger is None):
        question = sample.get("question", sample.get("prompt", ""))
        answer = sample.get("answer", sample.get("continuation", ""))
        is_correct = sample.get("is_correct", False)
        
        result = test_paraphrase_robustness(
            question,
            answer,
            is_correct,
            model,
            tokenizer,
            correctness_fn,
            n_paraphrases=n_paraphrases,
            max_new_tokens=max_new_tokens,
        )
        
        sample_results.append(result)
    
    # Aggregate
    valid_results = [r for r in sample_results if r["n_paraphrases"] > 0]
    
    if valid_results:
        avg_robustness = np.mean([r["robustness_score"] for r in valid_results])
        all_correct_rate = np.mean([1 if r["all_correct"] else 0 for r in valid_results])
    else:
        avg_robustness = 0.0
        all_correct_rate = 0.0
    
    return {
        "avg_robustness_score": float(avg_robustness),
        "all_correct_rate": float(all_correct_rate),
        "n_samples_tested": len(valid_results),
        "sample_results": sample_results,
    }


def classify_reasoning_type(
    robustness_score: float,
    robustness_threshold: float = 0.5,
) -> str:
    """Classify reasoning type based on robustness.
    
    Args:
        robustness_score: Paraphrase robustness score [0, 1]
        robustness_threshold: Threshold for faithful classification
        
    Returns:
        "faithful" if robust, "shortcut" if not
    """
    if robustness_score >= robustness_threshold:
        return "faithful"
    else:
        return "shortcut"

