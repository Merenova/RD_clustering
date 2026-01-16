#!/usr/bin/env python3
"""7c_graph.py - Semantic graph and feature selection utilities.

This module contains functions for:
1. Loading attribution context
2. Computing semantic graphs with different strategies
3. Feature selection (by magnitude, distinctiveness)
4. Decoder/Encoder weight precomputation and caching
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from utils.data_utils import reconstruct_active_features

# Import shared utilities (using same pattern as other modules)
import importlib.util
from pathlib import Path as _Path

_utils_spec = importlib.util.spec_from_file_location("7c_utils", _Path(__file__).parent / "7c_utils.py")
_utils_module = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_module)
utils = _utils_module


# =============================================================================
# Attribution Context Loading
# =============================================================================

def load_attribution_context(
    attribution_graphs_dir: Path,
    prefix_id: str,
    use_continuation_attribution: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load attribution context (active_features, selected_features) from either format.

    Args:
        attribution_graphs_dir: Directory containing attribution files
        prefix_id: Prefix identifier
        use_continuation_attribution: If True, load from prefix_context.pt (new format).
                                      If False, load from graph.pt (legacy format).

    Returns:
        Tuple of (active_features, selected_features) tensors for feature selection.
        - active_features: (N, 3) tensor mapping feature index to [layer, pos, feat_id]
        - selected_features: (N,) tensor of indices (identity mapping for new format)
    """
    if use_continuation_attribution:
        # New format: load from prefix_context.pt
        context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
        context_data = torch.load(context_file, weights_only=False)

        # Use shared utility to reconstruct active_features (N, 3) from
        # decoder_locations (2, N) + selected_features (N,) + activation_matrix
        # NOTE: selected_features contains indices into the sparse tensor, not actual feature IDs.
        #       activation_matrix is needed to extract the actual feature IDs.
        active_features = reconstruct_active_features(
            context_data["decoder_locations"],
            context_data["selected_features"],
            activation_matrix=context_data.get("activation_matrix", None),
            return_numpy=False  # Return as torch tensor
        )

        # selected_features is identity mapping since active_features already contains
        # only the selected features
        n_features = active_features.shape[0]
        selected_features = torch.arange(n_features, dtype=torch.long)

        return active_features, selected_features
    else:
        raise ValueError("Only prefix context format is supported")


# =============================================================================
# Semantic Graph Strategies
# =============================================================================

def compute_semantic_graphs(
    components: Dict[str, Any],
    H_0: np.ndarray,
    logger=None
) -> Dict[int, np.ndarray]:
    """Compute semantic graphs using H_c = H_0 + Delta_H_c."""
    delta_H_c_dict = {}
    for c_str, comp in components.items():
        c = int(c_str)
        if "mu_a" in comp:
            delta_H_c_dict[c] = np.array(comp["mu_a"])

    if not delta_H_c_dict:
        return {}

    if H_0 is None:
        raise ValueError("H_0 is required for H_c computation but was not provided")

    semantic_graphs = {}
    for c, delta_H_c in delta_H_c_dict.items():
        semantic_graphs[c] = H_0 + delta_H_c

    if logger:
        logger.info(f"  Using H_c strategy: H_0 + Delta_H_c (||H_0||={np.linalg.norm(H_0):.4f})")

    return semantic_graphs


def build_semantic_graphs_from_clustering(clustering_data: Dict) -> Dict[int, np.ndarray]:
    """Build simple semantic graphs from clustering data (for backward compat).

    Args:
        clustering_data: Clustering result dict with "components" key

    Returns:
        {cluster_id: mu_a_array}
    """
    components = clustering_data.get("components", {})
    graphs = {}
    for c_str, comp in components.items():
        if "mu_a" in comp:
            graphs[int(c_str)] = np.array(comp["mu_a"])
    return graphs


# =============================================================================
# Feature Selection
# =============================================================================

def select_top_features_by_magnitude(
    H_c: np.ndarray,
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    top_B: int = 50
) -> List[Tuple[int, int, int, float]]:
    """Select top-B features by |H_c| magnitude.

    H_c has dimension n_attribution_nodes = n_features + n_error + n_token.
    Only the first n_features elements correspond to actual model features.
    selected_features provides the mapping from H_c index to active_features index.
    active_features[selected_features[i]] gives [layer, pos, feat_id] for H_c[i].
    """
    n_features = len(selected_features)
    H_c_features = H_c[:n_features]
    top_indices = np.argsort(np.abs(H_c_features))[-top_B:][::-1]

    features = []
    for idx in top_indices:
        # Map H_c index -> active_features index via selected_features
        feat_idx = selected_features[idx].item()
        layer, pos, feat_id = active_features[feat_idx].tolist()
        H_c_value = float(H_c_features[idx])
        features.append((int(layer), int(pos), int(feat_id), H_c_value))

    return features


def select_top_features_by_distinctiveness(
    cluster_id: int,
    all_semantic_graphs: Dict[int, np.ndarray],
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    top_B: int = 50
) -> List[Tuple[int, int, int, float]]:
    """Select top-B features that are DISTINCT to this cluster.

    Distinctiveness score: |H_c[i]| / (max(|H_c'[i]|) + epsilon) for c' != c
    High score means this cluster has high activation where others have low.

    Args:
        cluster_id: Current cluster ID
        all_semantic_graphs: Dict of {cluster_id: H_c} for all clusters
        active_features: Tensor mapping feat_idx -> (layer, pos, feat_id)
        selected_features: Tensor mapping H_c index -> active_features index
        top_B: Number of top features to select

    Returns:
        List of (layer, pos, feat_id, H_c_value) tuples
    """
    H_c = all_semantic_graphs[cluster_id]
    n_features = len(selected_features)
    H_c_features = np.abs(H_c[:n_features])

    # Compute max |H_c'[i]| for other clusters
    other_cluster_ids = [c for c in all_semantic_graphs.keys() if c != cluster_id]
    if not other_cluster_ids:
        # Fallback to magnitude if only one cluster
        return select_top_features_by_magnitude(
            H_c, active_features, selected_features, top_B
        )

    other_H_c_stack = np.stack([
        np.abs(all_semantic_graphs[c][:n_features]) for c in other_cluster_ids
    ], axis=0)  # [n_other_clusters, n_features]
    max_other = np.max(other_H_c_stack, axis=0)  # [n_features]

    # Distinctiveness score: |H_c[i]| / (max_other[i] + eps)
    epsilon = utils.EPSILON_TINY
    distinctiveness = H_c_features / (max_other + epsilon)

    # Select top-B by distinctiveness
    top_indices = np.argsort(distinctiveness)[-top_B:][::-1]

    features = []
    for idx in top_indices:
        feat_idx = selected_features[idx].item()
        layer, pos, feat_id = active_features[feat_idx].tolist()
        H_c_value = float(H_c[:n_features][idx])  # Original signed value
        features.append((int(layer), int(pos), int(feat_id), H_c_value))

    return features


def select_top_features_from_Hc(
    H_c: np.ndarray,
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    top_B: int = 50,
    selection_mode: str = "magnitude",
    cluster_id: int = None,
    all_semantic_graphs: Dict[int, np.ndarray] = None
) -> List[Tuple[int, int, int, float]]:
    """Select top-B features using specified selection mode.

    Args:
        H_c: Semantic graph for this cluster
        active_features: Tensor mapping feat_idx -> (layer, pos, feat_id)
        selected_features: Tensor mapping H_c index -> active_features index
        top_B: Number of top features to select
        selection_mode: "magnitude" (default) or "distinct"
        cluster_id: Required for "distinct" mode
        all_semantic_graphs: Required for "distinct" mode

    Returns:
        List of (layer, pos, feat_id, H_c_value) tuples
    """
    if selection_mode == "distinct":
        if cluster_id is None or all_semantic_graphs is None:
            raise ValueError("cluster_id and all_semantic_graphs required for 'distinct' mode")
        return select_top_features_by_distinctiveness(
            cluster_id, all_semantic_graphs, active_features, selected_features, top_B
        )
    else:
        # Default: magnitude
        return select_top_features_by_magnitude(
            H_c, active_features, selected_features, top_B
        )


def select_features_with_hc_selection(
    cache_data: Dict,
    top_B: int,
    hc_selection: str = 'full'
) -> Tuple[List[float], List[torch.Tensor], List[int], List[int], List[int]]:
    """Select features based on H_c selection mode and top_B.

    Args:
        cache_data: Dict with keys: h_c_values, decoder_vecs, layers, positions, feat_ids
        top_B: Number of top features to select
        hc_selection: 'full' (all), 'positive' (H_c > 0), 'negative' (H_c < 0)

    Returns:
        (h_c_vals, decoder_vecs, layers, positions, feat_ids) - all filtered
    """
    h_c_vals = cache_data['h_c_values']
    decoder_vecs = cache_data['decoder_vecs']
    layers = cache_data['layers']
    positions = cache_data['positions']
    feat_ids = cache_data['feat_ids']

    # Filter by sign
    if hc_selection == 'positive':
        indices = [i for i, v in enumerate(h_c_vals) if v > 0]
    elif hc_selection == 'negative':
        indices = [i for i, v in enumerate(h_c_vals) if v < 0]
    else:  # full
        indices = list(range(len(h_c_vals)))

    if not indices:
        return [], [], [], [], []

    # Sort by magnitude and take top_B
    sorted_indices = sorted(indices, key=lambda i: abs(h_c_vals[i]), reverse=True)
    if top_B > 0:
        sorted_indices = sorted_indices[:top_B]

    return (
        [h_c_vals[i] for i in sorted_indices],
        [decoder_vecs[i] for i in sorted_indices],
        [layers[i] for i in sorted_indices],
        [positions[i] for i in sorted_indices],
        [feat_ids[i] for i in sorted_indices]
    )


# =============================================================================
# Decoder Vector Precomputation
# =============================================================================

def precompute_decoder_vectors_global(
    model,
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    device: torch.device,
    max_features: int = 1000
) -> Dict[int, torch.Tensor]:
    """Precompute decoder vectors for top features ONCE (shared across h_c_strategies).

    This is called once per prefix and cached. The decoder vectors only depend on
    (layer, feat_id), not on cluster assignment.

    Args:
        model: ReplacementModel with transcoders
        active_features: Tensor mapping feat_idx -> (layer, pos, feat_id)
        selected_features: Tensor mapping H_c index -> active_features index
        device: Torch device
        max_features: Maximum features to precompute

    Returns:
        {h_c_idx: decoder_vector} - maps H_c index to its decoder vector
    """
    n_features = min(len(selected_features), max_features)

    # Group features by layer for batched fetching
    layer_to_indices = {}
    for h_c_idx in range(n_features):
        feat_idx = selected_features[h_c_idx].item()
        layer, pos, feat_id = active_features[feat_idx].tolist()
        layer = int(layer)
        if layer not in layer_to_indices:
            layer_to_indices[layer] = []
        layer_to_indices[layer].append((h_c_idx, int(feat_id)))

    # Batch fetch decoder vectors per layer
    decoder_cache = {}
    for layer, idx_list in layer_to_indices.items():
        h_c_indices = [x[0] for x in idx_list]
        feat_ids = [x[1] for x in idx_list]

        # Batch fetch all decoder vectors for this layer
        feat_ids_t = torch.tensor(feat_ids, device=device, dtype=torch.long)
        dec_vecs = model.transcoders._get_decoder_vectors(layer, feat_ids_t)

        for i, h_c_idx in enumerate(h_c_indices):
            decoder_cache[h_c_idx] = dec_vecs[i]

    return decoder_cache


def build_cluster_decoder_cache(
    semantic_graphs: Dict[int, np.ndarray],
    global_decoder_cache: Dict[int, torch.Tensor],
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    max_features: int = 1000
) -> Dict[int, Dict]:
    """Build per-cluster decoder cache from global cache (fast - no model calls).

    This uses the precomputed global decoder cache to build per-cluster caches
    based on H_c values. Called once per clustering config (but very fast).

    Args:
        semantic_graphs: {cluster_id: H_c array}
        global_decoder_cache: {h_c_idx: decoder_vector} from precompute_decoder_vectors_global
        active_features: Tensor mapping feat_idx -> (layer, pos, feat_id)
        selected_features: Tensor mapping H_c index -> active_features index
        max_features: Maximum features per cluster

    Returns:
        {cluster_id: {h_c_values, decoder_vecs, layers, positions, feat_ids, h_c_indices}}
    """
    n_features = len(selected_features)
    cache = {}

    for cluster_id, H_c in semantic_graphs.items():
        H_c_features = H_c[:n_features]

        # Get top features by magnitude
        abs_vals = np.abs(H_c_features)
        top_indices = np.argsort(abs_vals)[-max_features:][::-1]

        h_c_vals = []
        decoder_vecs = []
        layers = []
        positions = []
        feat_ids = []
        h_c_indices = []

        for h_c_idx in top_indices:
            h_c_val = H_c_features[h_c_idx]
            if abs(h_c_val) < utils.EPSILON_SMALL:
                continue

            # Skip if not in global cache
            if h_c_idx not in global_decoder_cache:
                continue

            feat_idx = selected_features[h_c_idx].item()
            layer, pos, feat_id = active_features[feat_idx].tolist()

            h_c_vals.append(float(h_c_val))
            decoder_vecs.append(global_decoder_cache[h_c_idx])
            layers.append(int(layer))
            positions.append(int(pos))
            feat_ids.append(int(feat_id))
            h_c_indices.append(int(h_c_idx))

        if h_c_vals:
            cache[cluster_id] = {
                'h_c_values': h_c_vals,
                'decoder_vecs': decoder_vecs,
                'layers': layers,
                'positions': positions,
                'feat_ids': feat_ids,
                'h_c_indices': h_c_indices
            }

    return cache


# =============================================================================
# Encoder Weight Precomputation
# =============================================================================

def precompute_cluster_encoder_weights(
    model,
    features_by_cluster: Dict[int, List[Tuple[int, int, int, float]]],
    device: torch.device
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """Pre-compute encoder weights for each cluster's features.

    This enables column-wise activation computation during on-the-fly steering.

    Args:
        model: ReplacementModel with transcoders
        features_by_cluster: {cluster_id: [(layer, pos, feat_id, h_c_val), ...]}
        device: torch device

    Returns:
        {cluster_id: {layer: {'W_enc': [n_feats, d_model], 'b_enc': [n_feats], ...}}}
    """
    cluster_encoders = {}

    for cluster_id, features in features_by_cluster.items():
        cluster_encoders[cluster_id] = preload_encoder_weights_for_cluster(
            model, features, device
        )

    return cluster_encoders


def preload_encoder_weights_for_cluster(
    model,
    features: List[Tuple[int, int, int, float]],
    device: torch.device
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Pre-load encoder weights for column-wise activation computation.

    Only loads the encoder columns for features we need to intervene on.
    This is much more memory efficient than loading the full encoder.

    Args:
        model: ReplacementModel with transcoders
        features: List of (layer, pos, feat_id, h_c_val) tuples
        device: torch device

    Returns:
        {layer: {'W_enc': [n_feats, d_model], 'b_enc': [n_feats], 'feat_ids': [feat_id, ...]}}
    """
    # Group features by layer
    by_layer = {}
    for layer, pos, feat_id, h_c_val in features:
        if layer not in by_layer:
            by_layer[layer] = set()
        by_layer[layer].add(feat_id)

    encoder_cache = {}
    for layer, feat_ids_set in by_layer.items():
        feat_ids_list = sorted(feat_ids_set)
        feat_ids_t = torch.tensor(feat_ids_list, device=device, dtype=torch.long)

        # Load only the needed encoder columns (column-wise) - using shared utility
        W_enc_subset, b_enc_subset = utils.get_encoder_weights(model, layer, feat_ids_t)

        encoder_cache[layer] = {
            'W_enc': W_enc_subset,
            'b_enc': b_enc_subset,
            'feat_ids': feat_ids_list,
            'feat_id_to_idx': {fid: i for i, fid in enumerate(feat_ids_list)}
        }

    return encoder_cache


def precompute_cluster_decoder_vectors(
    model,
    features_by_cluster: Dict[int, List[Tuple[int, int, int, float]]],
    device: torch.device
) -> Dict[int, Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]]]:
    """Pre-compute decoder vectors for each cluster's features.

    This avoids redundant calls to _get_decoder_vectors when processing multiple branches
    with the same cluster's steering features.

    Args:
        model: ReplacementModel
        features_by_cluster: {cluster_id: [(layer, pos, feat_id, h_c_val), ...]}
        device: torch device

    Returns:
        {cluster_id: {layer: (positions, feat_ids, decoder_vecs, h_c_values)}}
        where decoder_vecs is [n_features_in_layer, d_model] tensor
    """
    cluster_decoders = {}

    for cluster_id, features in features_by_cluster.items():
        # Group features by layer
        by_layer = {}
        for layer, pos, feat_id, h_c_val in features:
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append((pos, feat_id, h_c_val))

        cluster_decoders[cluster_id] = {}
        for layer, items in by_layer.items():
            positions = [x[0] for x in items]
            feat_ids_list = [x[1] for x in items]
            feat_ids_tensor = torch.tensor(feat_ids_list, device=device, dtype=torch.long)
            h_c_vals = torch.tensor([x[2] for x in items], device=device, dtype=model.cfg.dtype)

            # Get decoder vectors ONCE per cluster per layer
            decoder_vecs = model.transcoders._get_decoder_vectors(layer, feat_ids_tensor)

            cluster_decoders[cluster_id][layer] = (positions, feat_ids_list, decoder_vecs, h_c_vals)

    return cluster_decoders

