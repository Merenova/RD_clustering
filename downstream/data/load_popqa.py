"""Load and prepare PopQA questions for hallucination experiment.

Loads PopQA dataset with quadrant-based filtering and sampling.
Adapted from hypotheses_experiment/shared_utils.py
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json, load_json

# Import config
from downstream.config import (
    POPQA_FILE,
    HALLUCINATION_CONFIG,
    DEFAULT_SEED,
)


def load_popqa_questions(
    popqa_file: Path = None,
    n_samples: int = None,
    seed: int = None,
    require_single_answer: bool = True,
    quadrants: Optional[List[str]] = None,
    balanced_quadrants: bool = None,
    question_offset: int = 0,
    logger=None,
) -> List[Dict]:
    """Load PopQA questions with ground truth answers.

    Args:
        popqa_file: Path to PopQA JSON file
        n_samples: Number of questions to load
        seed: Random seed for sampling
        require_single_answer: If True, only include questions with single correct answer
        quadrants: Optional list of quadrants to filter by (e.g., ["Q1_low_low", "Q4_high_high"])
        balanced_quadrants: If True, sample equally from each quadrant
        question_offset: Skip first N questions (for batch splitting across GPUs)
        logger: Optional logger instance

    Returns:
        List of dicts with keys: question, correct_answers, subject, relation, etc.
    """
    if popqa_file is None:
        popqa_file = POPQA_FILE
    
    if n_samples is None:
        n_samples = HALLUCINATION_CONFIG["n_questions_initial"]
    
    if seed is None:
        seed = DEFAULT_SEED
    
    if balanced_quadrants is None:
        balanced_quadrants = HALLUCINATION_CONFIG["balanced_quadrants"]
    
    if quadrants is None:
        quadrants = HALLUCINATION_CONFIG.get("quadrants")
    
    popqa_file = Path(popqa_file)
    
    if logger:
        logger.info(f"Loading PopQA from {popqa_file}")
    
    data = load_json(popqa_file)

    # Handle different PopQA formats
    if isinstance(data, dict):
        questions = data.get("questions", data.get("data", []))
    else:
        questions = data

    # Parse correct_answers if it's a string (popqa_full.json format)
    for q in questions:
        answers = q.get("correct_answers", q.get("answers", q.get("possible_answers", [])))
        if isinstance(answers, str):
            try:
                answers = ast.literal_eval(answers)
            except (ValueError, SyntaxError):
                answers = [answers]
        q["correct_answers"] = answers if isinstance(answers, list) else [answers]

    # Compute quadrants if missing (based on s_pop and o_pop medians)
    if questions and "quadrant" not in questions[0]:
        s_pops = [q.get("s_pop", 0) for q in questions if q.get("s_pop", 0) > 0]
        o_pops = [q.get("o_pop", 0) for q in questions if q.get("o_pop", 0) > 0]
        
        if s_pops and o_pops:
            s_median = np.median(s_pops)
            o_median = np.median(o_pops)
            
            if logger:
                logger.info(f"Computing quadrants with medians: s_pop={s_median:.0f}, o_pop={o_median:.0f}")
            
            for q in questions:
                s_high = q.get("s_pop", 0) >= s_median
                o_high = q.get("o_pop", 0) >= o_median
                
                if not s_high and not o_high:
                    q["quadrant"] = "Q1_low_low"
                elif s_high and not o_high:
                    q["quadrant"] = "Q2_high_low"
                elif not s_high and o_high:
                    q["quadrant"] = "Q3_low_high"
                else:
                    q["quadrant"] = "Q4_high_high"

    # Filter if needed
    if require_single_answer:
        original_count = len(questions)
        questions = [q for q in questions if len(q.get("correct_answers", [])) == 1]
        if logger:
            logger.info(f"Filtered to single-answer questions: {original_count} -> {len(questions)}")

    # Filter by quadrants if specified
    if quadrants:
        original_count = len(questions)
        questions = [q for q in questions if q.get("quadrant") in quadrants]
        if logger:
            logger.info(f"Filtered by quadrants {quadrants}: {original_count} -> {len(questions)}")

    rng = np.random.RandomState(seed)

    # Balanced sampling from quadrants
    if balanced_quadrants and any(q.get("quadrant") for q in questions):
        by_quadrant = {}
        for q in questions:
            quad = q.get("quadrant", "unknown")
            if quad not in by_quadrant:
                by_quadrant[quad] = []
            by_quadrant[quad].append(q)

        n_quadrants = len(by_quadrant)
        # Request more questions to account for offset
        n_total_needed = n_samples + question_offset
        n_per_quadrant = max(1, n_total_needed // n_quadrants)  # At least 1 per quadrant

        sampled = []
        for quad, items in by_quadrant.items():
            if not items:
                continue
            if len(items) > n_per_quadrant:
                indices = rng.choice(len(items), size=n_per_quadrant, replace=False)
                sampled.extend([items[i] for i in indices])
            else:
                sampled.extend(items)

        rng.shuffle(sampled)
        
        if logger:
            logger.info(f"Balanced sampling from {n_quadrants} quadrants, ~{n_per_quadrant} each")
        
        # Apply offset then limit
        return sampled[question_offset:question_offset + n_samples]

    # Simple random sampling - get enough questions for offset + n_samples
    n_total_needed = n_samples + question_offset
    if len(questions) > n_total_needed:
        indices = rng.choice(len(questions), size=n_total_needed, replace=False)
        questions = [questions[i] for i in indices]

    # Apply offset then limit
    result = questions[question_offset:question_offset + n_samples]
    
    if logger:
        logger.info(f"Loaded {len(result)} questions")
    
    return result


def prepare_prompt(
    question: str,
    tokenizer,
    mode: str = "question",
) -> str:
    """Format question with appropriate template.

    Args:
        question: Question text
        tokenizer: HuggingFace tokenizer
        mode: "cloze" for prefix completion, "question" for chat template

    Returns:
        Formatted prompt string
    """
    if mode == "question":
        # Use chat template
        messages = [{"role": "user", "content": question}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    else:
        # Cloze format (prefix completion)
        prompt = question
        if not prompt.endswith(" "):
            prompt = prompt + " "

    return prompt


def filter_by_popularity(
    questions: List[Dict],
    top_k: int,
    ascending: bool = True,
    logger=None,
) -> List[Dict]:
    """Filter questions by popularity score (s_pop * o_pop).
    
    Args:
        questions: List of question dicts
        top_k: Number of questions to keep
        ascending: If True, keep lowest popularity (harder questions)
        logger: Optional logger
        
    Returns:
        Filtered list of questions
    """
    # Add popularity score to each question
    for q in questions:
        s_pop = q.get("s_pop", 0)
        o_pop = q.get("o_pop", 0)
        q["popularity_score"] = s_pop * o_pop
    
    # Sort by popularity score
    sorted_questions = sorted(questions, key=lambda x: x["popularity_score"], reverse=not ascending)
    
    # Select top K
    selected = sorted_questions[:top_k]
    
    if logger:
        logger.info(f"Popularity filter: Kept {len(selected)} {'lowest' if ascending else 'highest'} popularity questions")
        if selected:
            logger.info(f"  Score range: [{selected[0]['popularity_score']}, {selected[-1]['popularity_score']}]")
    
    return selected


def save_popqa_questions(
    questions: List[Dict],
    output_file: Path,
    metadata: Dict = None,
    logger=None,
):
    """Save prepared questions to JSON file.
    
    Args:
        questions: List of question dicts
        output_file: Output file path
        metadata: Optional metadata dict
        logger: Logger instance
    """
    output_data = {
        "metadata": {
            "source": "popqa",
            "n_questions": len(questions),
            **(metadata or {}),
        },
        "questions": questions
    }
    
    save_json(output_data, output_file)
    if logger:
        logger.info(f"Saved {len(questions)} questions to {output_file}")

