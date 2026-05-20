"""Smoke test: end-to-end nnsight attribute_prefix_to_continuations on Gemma3-1B-it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from circuit_tracer.attribution._nnsight_overlay import get_nnsight_replacement_model_cls
from circuit_tracer.attribution.attribute_nnsight_prefix import (
    attribute_prefix_to_continuations,
)


GEMMA3_1B_TRANSCODER_ID = "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine"


@pytest.mark.slow
def test_e2e_prefix_to_continuation():
    NN = get_nnsight_replacement_model_cls()
    model = NN.from_pretrained(
        "google/gemma-3-1b-it",
        GEMMA3_1B_TRANSCODER_ID,
        dtype=torch.bfloat16,
    )

    prefix = model.tokenizer.encode("The capital of France is", add_special_tokens=False)
    cont = model.tokenizer.encode(" Paris.", add_special_tokens=False)
    assert len(cont) >= 2

    result = attribute_prefix_to_continuations(
        prefix=prefix,
        continuations=[cont],
        model=model,
        batch_size=64,
        add_bos=True,
        max_feature_nodes=256,
        verbose=True,
    )

    assert len(result.continuation_attributions) == 1
    per_cont = result.continuation_attributions[0]
    assert len(per_cont) == len(cont)
    first = per_cont[0]
    assert first.source_attribution.ndim == 1
    assert first.source_attribution.numel() > 0
    assert torch.isfinite(first.source_attribution).all()
