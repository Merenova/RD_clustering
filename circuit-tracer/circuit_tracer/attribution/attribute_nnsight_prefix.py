"""nnsight-backed setup_prefix_context for use with Gemma3 transcoders.

Mirrors the TL version in attribute.py:358-407 but uses the new circuit-tracer's
nnsight backend for the forward pass. Returns the same PrefixAttributionContext
dataclass so the rest of attribute_prefix_to_continuations can reuse it.

Key APIs we lean on from NNSightReplacementModel (sibling repo at
/home/hyunjin/circuit-tracer):
    - model.trace(tokens) as a context manager for nnsight tracing
    - model.feature_input_locs / model.feature_output_locs (per-layer envoys)
    - model.transcoders.compute_attribution_components(mlp_in, zero_positions)
    - model.embed_weight (replacement for TL's W_E)
    - model.cfg.device / model.cfg.n_layers
"""
from __future__ import annotations

import logging

import torch
from nnsight import save as _nnsight_save  # type: ignore

from .context import PrefixAttributionContext

logger = logging.getLogger("attribution.nnsight")


@torch.no_grad()
def setup_prefix_context(prefix_ids: torch.Tensor, model) -> PrefixAttributionContext:
    """nnsight equivalent of attribute.setup_prefix_context.

    Args:
        prefix_ids: (n_prefix,) long tensor of token ids with BOS already
            prepended if the caller wanted one.
        model: NNSightReplacementModel from the sibling circuit-tracer repo.

    Returns:
        PrefixAttributionContext (the in-tree dataclass from attribution.context).
    """
    # Prefer model.device (always present on NNSightReplacementModel); fall
    # back to cfg.device for TL-style models if someone reuses this helper.
    device = getattr(model, "device", None) or model.cfg.device
    prefix_ids = prefix_ids.to(device)
    assert prefix_ids.ndim == 1, "prefix_ids must be 1D"

    # --- Forward pass: capture MLP-in / MLP-out via nnsight tracing ---------
    # Pattern mirrors NNSightReplacementModel.setup_attribution (sibling repo,
    # replacement_model_nnsight.py:481).
    with model.trace(prefix_ids):
        mlp_in_list = []
        mlp_out_list = []
        for feature_input_loc, feature_output_loc in zip(
            model.feature_input_locs, model.feature_output_locs
        ):
            mlp_in_list.append(feature_input_loc.output)
            y = feature_output_loc.output
            # Some architectures (e.g. GPT-OSS) drop the dummy batch dim.
            if y.ndim == 2:
                y = y.unsqueeze(0)
            mlp_out_list.append(y)

        mlp_in_cache = _nnsight_save(torch.cat(mlp_in_list, dim=0))
        mlp_out_cache = _nnsight_save(torch.cat(mlp_out_list, dim=0))

    # After exiting the trace context, the saved proxies resolve to concrete
    # tensors. In nnsight 0.7 `save(...)` returns a tensor-like proxy that
    # supports .value if it hasn't materialized yet.
    mlp_in_tensor = getattr(mlp_in_cache, "value", mlp_in_cache)
    mlp_out_tensor = getattr(mlp_out_cache, "value", mlp_out_cache)

    # --- Transcoder decomposition -------------------------------------------
    # Sibling repo's compute_attribution_components accepts an optional
    # `zero_positions` kwarg (default slice(0, 1)); match the TL call which
    # uses the default.
    attribution_data = model.transcoders.compute_attribution_components(mlp_in_tensor)

    # --- Error vectors -------------------------------------------------------
    error_vectors = mlp_out_tensor - attribution_data["reconstruction"]
    error_vectors[:, 0] = 0  # Zero first position (BOS artifact) — matches TL.

    # --- Token embeddings ----------------------------------------------------
    # NNSightReplacementModel exposes the embedding weight as `embed_weight`
    # (resolved at __init__ via _resolve_attr); TL uses W_E.
    if hasattr(model, "W_E"):
        embed_weight = model.W_E
    else:
        embed_weight = model.embed_weight
    token_vectors = embed_weight[prefix_ids].detach()

    return PrefixAttributionContext(
        prefix_tokens=prefix_ids,
        activation_matrix=attribution_data["activation_matrix"],
        error_vectors=error_vectors,
        token_vectors=token_vectors,
        decoder_vecs=attribution_data["decoder_vecs"],
        encoder_vecs=attribution_data["encoder_vecs"],
        encoder_to_decoder_map=attribution_data["encoder_to_decoder_map"],
        decoder_locations=attribution_data["decoder_locations"],
        n_layers=model.cfg.n_layers,
    )
