"""Correctness detection for hallucination task.

Evaluates whether model responses contain the correct answer.
Adapted from hypotheses_experiment/shared_utils.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import config
from downstream.config import HALLUCINATION_CONFIG


@dataclass
class CorrectnessResult:
    """Result of judging a single response."""
    is_correct: bool
    confidence: float
    method: str  # "exact", "fuzzy", or "llm"
    matched_answer: Optional[str] = None


def is_correct(
    response: str,
    correct_answers: List[str],
    fuzzy: bool = True,
    fuzzy_threshold: int = None,
) -> bool:
    """Check if response contains a correct answer.
    
    Args:
        response: Response text to check
        correct_answers: List of acceptable correct answers
        fuzzy: Whether to use fuzzy matching
        fuzzy_threshold: Minimum similarity score (0-100) for fuzzy match
        
    Returns:
        True if response contains a correct answer
    """
    if fuzzy_threshold is None:
        fuzzy_threshold = HALLUCINATION_CONFIG["fuzzy_threshold"]
    
    response_lower = response.lower().strip()
    
    for ans in correct_answers:
        ans_lower = ans.lower().strip()
        
        # Fast path: exact substring match
        if ans_lower in response_lower:
            return True
        
        # Fuzzy matching
        if fuzzy:
            try:
                from rapidfuzz import fuzz
                score = fuzz.token_set_ratio(ans_lower, response_lower)
            except ImportError:
                try:
                    from fuzzywuzzy import fuzz
                    score = fuzz.token_set_ratio(ans_lower, response_lower)
                except ImportError:
                    score = 0
            
            if score >= fuzzy_threshold:
                return True
    
    return False


class CorrectnessJudge:
    """Judge correctness of responses using exact/fuzzy matching."""
    
    def __init__(
        self,
        fuzzy: bool = True,
        fuzzy_threshold: int = None,
    ):
        """Initialize the correctness judge.
        
        Args:
            fuzzy: Whether to use fuzzy matching
            fuzzy_threshold: Minimum similarity score for fuzzy match
        """
        self.fuzzy = fuzzy
        self.fuzzy_threshold = fuzzy_threshold or HALLUCINATION_CONFIG["fuzzy_threshold"]
    
    def judge_single(
        self,
        response: str,
        correct_answers: List[str],
    ) -> CorrectnessResult:
        """Judge a single response.
        
        Args:
            response: Response text to judge
            correct_answers: List of acceptable correct answers
            
        Returns:
            CorrectnessResult with is_correct, confidence, method
        """
        response_lower = response.lower().strip()
        
        # Try exact match first
        for ans in correct_answers:
            ans_lower = ans.lower().strip()
            if ans_lower in response_lower:
                return CorrectnessResult(
                    is_correct=True,
                    confidence=1.0,
                    method="exact",
                    matched_answer=ans
                )
        
        # Try fuzzy match
        if self.fuzzy:
            try:
                from rapidfuzz import fuzz
            except ImportError:
                try:
                    from fuzzywuzzy import fuzz
                except ImportError:
                    fuzz = None
            
            if fuzz:
                for ans in correct_answers:
                    ans_lower = ans.lower().strip()
                    score = fuzz.token_set_ratio(ans_lower, response_lower)
                    if score >= self.fuzzy_threshold:
                        return CorrectnessResult(
                            is_correct=True,
                            confidence=score / 100.0,
                            method="fuzzy",
                            matched_answer=ans
                        )
        
        # No match
        return CorrectnessResult(
            is_correct=False,
            confidence=1.0,
            method="exact" if not self.fuzzy else "fuzzy",
            matched_answer=None
        )
    
    def judge_batch(
        self,
        responses: List[str],
        correct_answers: List[str],
    ) -> List[CorrectnessResult]:
        """Judge a batch of responses against the same correct answers.
        
        Args:
            responses: List of response texts
            correct_answers: List of acceptable correct answers
            
        Returns:
            List of CorrectnessResults
        """
        return [self.judge_single(r, correct_answers) for r in responses]


def label_correct_answers(
    continuations: List[str],
    correct_answers: List[str],
    fuzzy: bool = True,
    fuzzy_threshold: int = None,
) -> np.ndarray:
    """Label continuations as correct (1) or hallucination (0).

    Args:
        continuations: List of continuation texts
        correct_answers: List of correct answer strings
        fuzzy: If True, use token_set_ratio fuzzy matching
        fuzzy_threshold: Minimum similarity score (0-100) for fuzzy match

    Returns:
        Binary labels array
    """
    if fuzzy_threshold is None:
        fuzzy_threshold = HALLUCINATION_CONFIG["fuzzy_threshold"]
    
    labels = np.zeros(len(continuations), dtype=int)

    for i, cont in enumerate(continuations):
        if is_correct(cont, correct_answers, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold):
            labels[i] = 1

    return labels


def compute_correctness_rate(
    continuations: List[str],
    correct_answers: List[str],
    fuzzy: bool = True,
    fuzzy_threshold: int = None,
) -> float:
    """Compute correctness rate for a set of continuations.
    
    Args:
        continuations: List of continuation texts
        correct_answers: List of correct answer strings
        fuzzy: Whether to use fuzzy matching
        fuzzy_threshold: Minimum similarity score
        
    Returns:
        Fraction of continuations that are correct
    """
    if not continuations:
        return 0.0
    
    labels = label_correct_answers(continuations, correct_answers, fuzzy, fuzzy_threshold)
    return float(labels.mean())


def evaluate_responses_with_probs(
    responses: List[str],
    correct_answers: List[str],
    probabilities: List[float],
    fuzzy: bool = True,
    fuzzy_threshold: int = None,
) -> Dict[str, Any]:
    """Evaluate responses with probability-weighted metrics.
    
    Args:
        responses: List of response texts
        correct_answers: List of acceptable correct answers
        probabilities: List of probabilities for each response
        fuzzy: Whether to use fuzzy matching
        fuzzy_threshold: Minimum similarity score
        
    Returns:
        Dict with labels, probability sums, and metrics
    """
    labels = label_correct_answers(responses, correct_answers, fuzzy, fuzzy_threshold)
    probs = np.array(probabilities)
    
    # Normalize probabilities
    probs = probs / (probs.sum() + 1e-10)
    
    # Compute probability-weighted sums
    correct_prob_sum = float(probs[labels == 1].sum())
    wrong_prob_sum = float(probs[labels == 0].sum())
    
    return {
        "labels": labels.tolist(),
        "correct_prob_sum": correct_prob_sum,
        "wrong_prob_sum": wrong_prob_sum,
        "n_correct": int(labels.sum()),
        "n_wrong": int((1 - labels).sum()),
        "correctness_rate": float(labels.mean()),
    }


def judge_hallucination_rollout(
    rollout_results: List[Dict[str, Any]],
    judge: CorrectnessJudge = None,
    logger=None,
) -> List[Dict[str, Any]]:
    """Judge all responses in hallucination rollout results.
    
    Args:
        rollout_results: Results from rollout_hallucination_task
        judge: CorrectnessJudge instance (creates new one if None)
        logger: Logger instance
        
    Returns:
        Updated results with correctness labels
    """
    if judge is None:
        judge = CorrectnessJudge()
    
    total_correct = 0
    total_responses = 0
    
    for result in rollout_results:
        correct_answers = result.get("correct_answers", [])
        continuations = result.get("continuations", [])
        
        # Judge each continuation
        for cont in continuations:
            judgment = judge.judge_single(cont["text"], correct_answers)
            cont["is_correct"] = judgment.is_correct
            cont["correctness_confidence"] = judgment.confidence
            cont["correctness_method"] = judgment.method
            cont["matched_answer"] = judgment.matched_answer
            
            if judgment.is_correct:
                total_correct += 1
            total_responses += 1
        
        # Compute summary metrics
        labels = [c["is_correct"] for c in continuations]
        probs = [c.get("probability", 1.0) for c in continuations]
        
        result["correctness_rate"] = np.mean(labels) if labels else 0.0
        result["n_correct"] = sum(labels)
        result["n_wrong"] = len(labels) - sum(labels)
        
        # Probability-weighted
        probs = np.array(probs)
        probs = probs / (probs.sum() + 1e-10)
        labels = np.array(labels)
        
        result["correct_prob_sum"] = float(probs[labels == 1].sum()) if labels.any() else 0.0
        result["wrong_prob_sum"] = float(probs[labels == 0].sum()) if (~labels.astype(bool)).any() else 0.0
    
    if logger:
        logger.info(f"Correctness judgment: {total_correct}/{total_responses} correct "
                   f"({100*total_correct/total_responses:.1f}%)" if total_responses else "No responses")
    
    return rollout_results

