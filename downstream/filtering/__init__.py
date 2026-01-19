"""Filtering modules for downstream tasks."""

from .safety_filter import filter_by_disagreement, compute_disagreement
from .hallucination_filter import (
    filter_by_answer_entropy,
    filter_by_embedding_entropy,
    compute_binary_entropy,
)

__all__ = [
    "filter_by_disagreement",
    "compute_disagreement",
    "filter_by_answer_entropy",
    "filter_by_embedding_entropy",
    "compute_binary_entropy",
]

