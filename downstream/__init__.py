"""Downstream Motif Discovery Pipeline.

This package implements unsupervised mechanistic motif discovery via
Rate-Distortion clustering for two downstream applications:
- Safety: Brittle refusal detection (HarmBench)
- Hallucination: Shortcut reasoning detection (PopQA)
"""

from . import config

__all__ = ["config"]

