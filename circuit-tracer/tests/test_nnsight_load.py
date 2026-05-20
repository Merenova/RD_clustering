"""Smoke test: load Gemma3-1B-it via the new circuit-tracer nnsight backend.

This test is marked 'slow' because it downloads the model + transcoders
(several GB) and runs on GPU. Skip in CI; run locally before any Gemma3
pipeline work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from circuit_tracer.attribution._nnsight_overlay import get_nnsight_replacement_model_cls


# Gemma3-1B-it transcoder from mwhanna/gemma-scope-2-1b-it collection
# Source: /home/hyunjin/circuit-tracer/tests/test_attributions_gemma3_nnsight.py:534
GEMMA3_1B_TRANSCODER_ID = "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine"


@pytest.mark.slow
def test_load_gemma3_1b_it():
    NN = get_nnsight_replacement_model_cls()
    model = NN.from_pretrained(
        "google/gemma-3-1b-it",
        GEMMA3_1B_TRANSCODER_ID,
        dtype=torch.bfloat16,
    )
    assert model.backend == "nnsight"
    assert model.cfg.n_layers > 0
    assert model.tokenizer is not None
