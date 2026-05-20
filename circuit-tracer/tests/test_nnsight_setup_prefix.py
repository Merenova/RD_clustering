"""Smoke test for nnsight setup_prefix_context on Gemma3-1B-it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from circuit_tracer.attribution._nnsight_overlay import get_nnsight_replacement_model_cls
from circuit_tracer.attribution.attribute_nnsight_prefix import setup_prefix_context


GEMMA3_1B_TRANSCODER_ID = "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine"


@pytest.mark.slow
def test_setup_prefix_context_shapes():
    NN = get_nnsight_replacement_model_cls()
    model = NN.from_pretrained(
        "google/gemma-3-1b-it",
        GEMMA3_1B_TRANSCODER_ID,
        dtype=torch.bfloat16,
    )

    bos = model.tokenizer.bos_token_id
    assert bos is not None
    prefix_ids = torch.tensor([bos, 100, 200, 300], device=model.device)

    ctx = setup_prefix_context(prefix_ids, model)

    n_prefix = prefix_ids.shape[0]
    n_layers = model.cfg.n_layers
    assert ctx.prefix_tokens.shape == (n_prefix,)
    assert ctx.token_vectors.shape[0] == n_prefix
    assert ctx.error_vectors.shape[0] == n_layers
    assert ctx.error_vectors.shape[1] == n_prefix
    assert ctx.n_layers == n_layers
    # Position 0 must be zeroed
    assert torch.all(ctx.error_vectors[:, 0] == 0)
