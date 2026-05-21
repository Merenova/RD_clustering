from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from circuit_tracer.attribution.context import PrefixAttributionContext
from circuit_tracer.attribution import attribute_nnsight_prefix as attr_nn


def _sparse_activation_matrix(n_layers: int, n_pos: int, d_transcoder: int) -> torch.Tensor:
    indices = torch.tensor(
        [
            [0, 0, 0, 1, 1],
            [0, 1, 2, 0, 2],
            [3, 4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_layers, n_pos, d_transcoder),
    ).coalesce()


def _selected_prefix_context() -> PrefixAttributionContext:
    activation_matrix = _sparse_activation_matrix(n_layers=2, n_pos=3, d_transcoder=16)
    selected_features = torch.tensor([1, 4], dtype=torch.long)
    return PrefixAttributionContext(
        prefix_tokens=torch.tensor([2, 105, 2364]),
        activation_matrix=activation_matrix,
        error_vectors=torch.ones(2, 3, 4),
        token_vectors=torch.ones(3, 4),
        decoder_vecs=torch.ones(2, 4),
        encoder_vecs=torch.ones(2, 4),
        encoder_to_decoder_map=torch.arange(2),
        decoder_locations=activation_matrix.indices()[:2, selected_features],
        n_layers=2,
        selected_features=selected_features,
        total_active_features=activation_matrix._nnz(),
    )


class _RecordingNNSightAttributionContext:
    def __init__(
        self,
        activation_matrix,
        error_vectors,
        token_vectors,
        decoder_vecs,
        encoder_vecs,
        encoder_to_decoder_map,
        decoder_locations,
        logits,
    ):
        self.activation_matrix = activation_matrix
        self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        self.decoder_vecs = decoder_vecs
        self.encoder_vecs = encoder_vecs
        self.encoder_to_decoder_map = encoder_to_decoder_map
        self.decoder_locations = decoder_locations
        self.logits = logits
        self.calls = []
        self._row_size = (
            activation_matrix._nnz() + (activation_matrix.shape[0] + 1) * activation_matrix.shape[1]
        )

    def compute_score(self, grads, output_vecs, write_index, read_index=np.s_[:]):
        self.calls.append((write_index, read_index, tuple(output_vecs.shape)))


def _slice_bounds(s: slice) -> tuple[int, int]:
    return int(s.start), int(s.stop)


def test_selected_feature_offsets_write_inside_prefix_source_window(monkeypatch):
    prefix_ctx = _selected_prefix_context()

    import circuit_tracer.attribution._nnsight_overlay as overlay

    monkeypatch.setattr(
        overlay,
        "get_nnsight_attribution_context_cls",
        lambda: _RecordingNNSightAttributionContext,
    )

    cont_ctx = attr_nn._make_prefix_only_attribution_context(prefix_ctx, model=SimpleNamespace())
    grads = torch.ones(1, 5, 4)

    cont_ctx.compute_error_attributions(0, grads)
    cont_ctx.compute_error_attributions(1, grads)
    cont_ctx.compute_token_attributions(grads)

    assert len(cont_ctx.calls) == 3
    assert cont_ctx._row_size == prefix_ctx.n_prefix_sources
    assert _slice_bounds(cont_ctx.calls[0][0]) == (2, 5)
    assert _slice_bounds(cont_ctx.calls[1][0]) == (5, 8)
    assert _slice_bounds(cont_ctx.calls[2][0]) == (8, 11)
    assert cont_ctx.calls[0][1] == np.s_[:, :3]
    assert cont_ctx.calls[1][1] == np.s_[:, :3]
    assert cont_ctx.calls[2][1] == np.s_[:, :3]


class _NoOpTrace:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTranscoders:
    def __init__(self):
        self.seen_zero_positions = None

    def compute_attribution_components(self, mlp_inputs, zero_positions=slice(0, 1)):
        self.seen_zero_positions = zero_positions
        n_layers, n_pos, d_model = mlp_inputs.shape
        activation_matrix = torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.long),
            torch.empty((0,), dtype=mlp_inputs.dtype),
            size=(n_layers, n_pos, 8),
        ).coalesce()
        return {
            "activation_matrix": activation_matrix,
            "reconstruction": torch.zeros_like(mlp_inputs),
            "encoder_vecs": torch.empty((0, d_model), dtype=mlp_inputs.dtype),
            "decoder_vecs": torch.empty((0, d_model), dtype=mlp_inputs.dtype),
            "encoder_to_decoder_map": torch.empty((0,), dtype=torch.long),
            "decoder_locations": torch.empty((2, 0), dtype=torch.long),
        }


class _BoomingTranscoders:
    def compute_attribution_components(self, mlp_inputs, zero_positions):
        raise TypeError("internal boom")


class _FakeNNSightModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(device=torch.device("cpu"), n_layers=2, d_model=4)
        self.zero_positions = slice(0, 4)
        self.feature_input_locs = [
            SimpleNamespace(output=torch.ones(1, 5, 4)),
            SimpleNamespace(output=torch.ones(1, 5, 4) * 2),
        ]
        self.feature_output_locs = [
            SimpleNamespace(output=torch.ones(1, 5, 4) * 11),
            SimpleNamespace(output=torch.ones(1, 5, 4) * 13),
        ]
        self.embed_weight = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
        self.transcoders = _FakeTranscoders()

    def trace(self, prefix_ids):
        return _NoOpTrace()


def test_setup_prefix_context_uses_model_zero_positions(monkeypatch):
    monkeypatch.setattr(attr_nn, "_nnsight_save", lambda value: value)
    model = _FakeNNSightModel()

    ctx = attr_nn.setup_prefix_context(torch.tensor([0, 1, 2, 3, 4]), model)

    assert model.transcoders.seen_zero_positions == slice(0, 4)
    assert torch.all(ctx.error_vectors[:, :4] == 0)
    assert torch.all(ctx.error_vectors[:, 4] != 0)


def test_setup_prefix_context_does_not_swallow_internal_transcoder_typeerror(
    monkeypatch,
):
    monkeypatch.setattr(attr_nn, "_nnsight_save", lambda value: value)
    model = _FakeNNSightModel()
    model.transcoders = _BoomingTranscoders()

    with pytest.raises(TypeError, match="internal boom"):
        attr_nn.setup_prefix_context(torch.tensor([0, 1, 2, 3, 4]), model)
