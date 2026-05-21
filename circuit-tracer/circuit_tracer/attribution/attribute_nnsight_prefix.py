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

import numpy as np
import torch
from nnsight import save as _nnsight_save  # type: ignore

from .context import PrefixAttributionContext

logger = logging.getLogger("attribution.nnsight")


def _compute_attribution_components(transcoders, mlp_in_tensor, zero_positions):
    """Call transcoders.compute_attribution_components with whichever signature
    is exposed.

    The in-tree TranscoderSet takes only ``mlp_inputs``; the sibling repo's
    version accepts an additional ``zero_positions`` slice. When the model is
    loaded via nnsight's envoy, ``inspect.signature`` reports a generic
    ``(*args, **kwargs)`` forwarder and cannot tell the two apart, so we try
    the richer calls first and fall back on signature-mismatch ``TypeError``s
    only. In-body ``TypeError``s (raised from within the function) propagate.
    """
    compute = transcoders.compute_attribution_components

    def _is_signature_error(err: TypeError) -> bool:
        msg = str(err)
        return (
            "unexpected keyword argument" in msg
            or "positional argument" in msg
            or "positional arguments" in msg
        )

    try:
        return compute(mlp_in_tensor, zero_positions=zero_positions)
    except TypeError as e:
        if not _is_signature_error(e):
            raise
    try:
        return compute(mlp_in_tensor, zero_positions)
    except TypeError as e:
        if not _is_signature_error(e):
            raise
    return compute(mlp_in_tensor)


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

    zero_positions = getattr(model, "zero_positions", slice(0, 1))
    attribution_data = _compute_attribution_components(
        model.transcoders,
        mlp_in_tensor,
        zero_positions,
    )

    # --- Error vectors -------------------------------------------------------
    error_vectors = mlp_out_tensor - attribution_data["reconstruction"]
    error_vectors[:, zero_positions] = 0

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


def _make_prefix_only_attribution_context(prefix_ctx: PrefixAttributionContext, model):
    """Build a subclass of the sibling nnsight ``AttributionContext`` that
    restricts gradient reads to the prefix positions only.

    The sibling repo's ``AttributionContext.compute_*_attributions`` methods
    contract gradients with prefix-shaped output vectors via einsum
    ``"batch position d_model, position d_model -> position batch"``. When the
    forward pass runs on ``prefix + continuation`` tokens, the gradient tensors
    have shape ``(batch, full_length, d_model)`` while the prefix's
    error/token vectors only span ``n_prefix`` positions. We therefore override
    the per-layer read to slice the first ``n_prefix`` positions, which mirrors
    ``ContinuationAttributionContext._make_prefix_only_attribution_hooks`` in
    the TransformerLens path (context.py:458-508).

    For feature nodes, ``compute_feature_attributions`` already uses
    ``nnz_positions[layer_mask]`` and the prefix's ``decoder_locations`` have
    positions in ``[0, n_prefix)`` — those indices remain valid on full-length
    grads because prefix positions occupy the same indices.

    Args:
        prefix_ctx: The cached prefix context (already optionally top-K trimmed).
        model: NNSightReplacementModel — used for ``n_layers``.

    Returns:
        An instance of an ``AttributionContext`` subclass, ready for
        ``cache_residual`` + ``compute_batch``.
    """
    from ._nnsight_overlay import get_nnsight_attribution_context_cls

    NNCtx = get_nnsight_attribution_context_cls()
    n_prefix = prefix_ctx.prefix_length
    feature_count = prefix_ctx.n_prefix_features

    class _PrefixOnlyAttributionContext(NNCtx):  # type: ignore[misc, valid-type]
        """nnsight AttributionContext that only reads prefix-position gradients
        for the error- and token-node hooks. Feature hooks already read the
        prefix-indexed positions via ``nnz_positions``.
        """

        def compute_error_attributions(self, layer, grads):  # type: ignore[override]
            def error_offset(lyr: int) -> int:
                return feature_count + lyr * n_prefix

            self.compute_score(
                grads,
                self.error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
                read_index=np.s_[:, :n_prefix],
            )

        def compute_token_attributions(self, grads):  # type: ignore[override]
            tok_start = feature_count + self.n_layers * n_prefix
            self.compute_score(
                grads,
                self.token_vectors,
                write_index=np.s_[tok_start : tok_start + n_prefix],
                read_index=np.s_[:, :n_prefix],
            )

    # The sibling AttributionContext does *not* accept ``prefix_tokens`` or
    # ``n_layers`` kwargs — it derives ``n_layers`` from the activation matrix.
    cont_ctx = _PrefixOnlyAttributionContext(
        activation_matrix=prefix_ctx.activation_matrix,
        error_vectors=prefix_ctx.error_vectors,
        token_vectors=prefix_ctx.token_vectors,
        decoder_vecs=prefix_ctx.decoder_vecs,
        encoder_vecs=prefix_ctx.encoder_vecs,
        encoder_to_decoder_map=prefix_ctx.encoder_to_decoder_map,
        decoder_locations=prefix_ctx.decoder_locations,
        logits=None,  # Not used for prefix-to-continuation attribution.
    )
    # Some test/overlay contexts do not initialize n_layers; keep the overridden
    # token offset tied to the prefix context rather than the activation shape.
    cont_ctx.n_layers = prefix_ctx.n_layers
    cont_ctx._row_size = prefix_ctx.n_prefix_sources
    return cont_ctx


def attribute_prefix_to_continuations(
    prefix,
    continuations,
    model,
    *,
    batch_size: int = 512,
    add_bos: bool = True,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
):
    """nnsight equivalent of attribute.attribute_prefix_to_continuations.

    For each continuation token, computes attribution scores from prefix
    components (features, errors, tokens) to that token's logit prediction,
    using the new circuit-tracer's nnsight backend.

    Mirrors the public TL signature at ``attribute.py:408-565``. Returns a
    backend-agnostic ``PrefixContinuationResult`` so the dispatch shim (A5)
    stays uniform across backends.

    Args:
        prefix: 1-D token ids (list[int] or LongTensor). BOS not included.
        continuations: List of continuation token id sequences.
        model: NNSightReplacementModel from the sibling repo.
        batch_size: How many continuation tokens to process per backward pass.
        add_bos: Whether to prepend BOS token to the prefix.
        max_feature_nodes: Optional cap on prefix feature-node count, picked by
            ``|activation|``.
        verbose: Emit info logs.

    Returns:
        PrefixContinuationResult.
    """
    # Local imports keep module import-time light (and break a circular import
    # with the public ``attribute`` module that re-exports our function).
    from .attribute import (
        ContinuationTokenAttribution,
        PrefixAttributionContext as _PAC,  # noqa: F401  (sanity check identity)
        PrefixContinuationResult,
    )

    log = logger if verbose else logging.getLogger("attribution.nnsight")
    device = getattr(model, "device", None) or model.cfg.device

    # ---- 1. Prefix → tensor on device --------------------------------------
    if isinstance(prefix, list):
        prefix_ids = torch.tensor(prefix, dtype=torch.long, device=device)
    else:
        prefix_ids = prefix.to(device=device, dtype=torch.long)

    # ---- 2. Optionally prepend BOS -----------------------------------------
    if add_bos:
        bos_token_id = getattr(model.tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            prefix_ids = torch.cat(
                [
                    torch.tensor([bos_token_id], dtype=torch.long, device=device),
                    prefix_ids,
                ]
            )
        else:
            log.warning("No BOS token found, No changes made to prefix")

    # ---- 3. Cache prefix components ----------------------------------------
    if verbose:
        log.info("Setting up prefix context (nnsight backend)...")
    prefix_ctx = setup_prefix_context(prefix_ids, model)

    # ---- 4. Optional top-K feature-node trimming ---------------------------
    total_active = prefix_ctx.activation_matrix._nnz()
    if max_feature_nodes is not None and max_feature_nodes < total_active:
        if verbose:
            log.info(
                f"Selecting top {max_feature_nodes} of {total_active} features by "
                "activation magnitude"
            )
        activation_values = prefix_ctx.activation_matrix.values()
        top_indices = torch.argsort(activation_values.abs(), descending=True)[:max_feature_nodes]
        top_indices = top_indices.sort().values  # keep original ordering

        prefix_ctx = PrefixAttributionContext(
            prefix_tokens=prefix_ctx.prefix_tokens,
            activation_matrix=prefix_ctx.activation_matrix,
            error_vectors=prefix_ctx.error_vectors,
            token_vectors=prefix_ctx.token_vectors,
            decoder_vecs=prefix_ctx.decoder_vecs[top_indices],
            encoder_vecs=prefix_ctx.encoder_vecs[top_indices],
            encoder_to_decoder_map=torch.arange(len(top_indices), device=device),
            decoder_locations=prefix_ctx.decoder_locations[:, top_indices],
            n_layers=prefix_ctx.n_layers,
            selected_features=top_indices,
            total_active_features=total_active,
        )

    n_prefix = prefix_ctx.prefix_length
    n_layers = model.cfg.n_layers

    # ---- 5. Unembed projection (with mean-centering) -----------------------
    # NNSightReplacementModel exposes ``unembed_weight`` rather than TL's
    # ``unembed.W_U``. We mean-center across the vocab axis so the inject
    # vectors are insensitive to a constant logit shift.
    unembed_weight = model.unembed_weight  # (d_model, vocab) or (vocab, d_model)
    # TL convention is (d_model, vocab); the sibling resolves the same matrix
    # via _resolve_attr — confirm shape by checking against d_model.
    d_model = model.cfg.d_model
    if unembed_weight.shape[0] != d_model:
        # (vocab, d_model) → transpose to (d_model, vocab).
        unembed_weight = unembed_weight.T
    unembed_mean = unembed_weight.mean(dim=-1, keepdim=True)

    # ---- 6. Per-continuation attribution loop ------------------------------
    all_continuation_attributions: list[list[ContinuationTokenAttribution]] = []

    for cont_idx, continuation in enumerate(continuations):
        if isinstance(continuation, list):
            cont_ids = torch.tensor(continuation, dtype=torch.long, device=device)
        else:
            cont_ids = continuation.to(device=device, dtype=torch.long)

        n_continuation = len(cont_ids)
        full_tokens = torch.cat([prefix_ids, cont_ids])
        full_length = len(full_tokens)

        if verbose:
            log.info(
                f"Processing continuation {cont_idx}: {n_continuation} tokens "
                f"(full_length={full_length})"
            )

        token_attributions: list[ContinuationTokenAttribution] = []

        # Process continuation tokens in chunks of `batch_size`. Each chunk
        # gets its OWN forward+backward trace because the batch dim must equal
        # the chunk size (the sibling's ``compute_batch`` runs one backward
        # over the cached activations and indexes by batch position).
        for batch_start in range(0, n_continuation, batch_size):
            batch_end = min(batch_start + batch_size, n_continuation)
            chunk = batch_end - batch_start

            batch_positions = torch.arange(
                n_prefix + batch_start,
                n_prefix + batch_end,
                device=device,
            )
            batch_token_ids = cont_ids[batch_start:batch_end]

            # Demeaned unembed columns: (d_model, chunk) → (chunk, d_model).
            inject_values = (unembed_weight[:, batch_token_ids] - unembed_mean).T

            cont_ctx = _make_prefix_only_attribution_context(prefix_ctx, model)

            with model.trace() as tracer:
                with tracer.invoke(full_tokens.expand(chunk, -1)):
                    pass
                detach_barrier = tracer.barrier(2)
                model.configure_gradient_flow(tracer)
                model.configure_skip_connection(tracer, barrier=detach_barrier)
                cont_ctx.cache_residual(model, tracer, barrier=detach_barrier)

            rows = cont_ctx.compute_batch(
                layers=torch.full((chunk,), n_layers, device=device),
                positions=batch_positions,
                inject_values=inject_values,
                retain_graph=False,
            )
            # `rows` shape: (chunk, row_size). Truncate to prefix-only sources.
            row_full = rows.detach().cpu()
            n_sources = prefix_ctx.n_prefix_sources
            row_prefix = row_full[:, :n_sources]

            for i in range(chunk):
                token_attributions.append(
                    ContinuationTokenAttribution(
                        token_id=int(batch_token_ids[i].item()),
                        position=n_prefix + batch_start + i,
                        source_attribution=row_prefix[i],
                    )
                )

            # Free graph memory before the next chunk.
            del cont_ctx, rows

        all_continuation_attributions.append(token_attributions)

    return PrefixContinuationResult(
        prefix_tokens=prefix_ids,
        prefix_context=prefix_ctx,
        continuation_attributions=all_continuation_attributions,
    )
