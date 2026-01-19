"""Perturbation module for robustness testing.

Provides perturbation generation and robustness testing utilities for
downstream experiments.
"""

from .generator import (
    generate_perturbations,
    add_typos,
    simple_paraphrase,
    rephrase_prompt,
)
from .robustness import (
    test_cluster_robustness,
    test_sample_robustness,
)
from .paraphraser import (
    paraphrase_question,
    test_paraphrase_robustness,
)

__all__ = [
    # Generator
    "generate_perturbations",
    "add_typos",
    "simple_paraphrase",
    "rephrase_prompt",
    # Robustness
    "test_cluster_robustness",
    "test_sample_robustness",
    # Paraphraser
    "paraphrase_question",
    "test_paraphrase_robustness",
]

