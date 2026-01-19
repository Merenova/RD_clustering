"""Steering module for downstream experiments.

Provides adapters for using 7_validation steering with pooled attributions.
"""

from .steering_adapter import (
    expand_pooled_to_positions,
    prepare_steering_from_centroid,
    generate_steered_outputs_real,
    compute_similarity_to_cluster_real,
)

__all__ = [
    "expand_pooled_to_positions",
    "prepare_steering_from_centroid",
    "generate_steered_outputs_real",
    "compute_similarity_to_cluster_real",
]

