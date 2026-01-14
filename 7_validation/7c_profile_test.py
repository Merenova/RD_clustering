#!/usr/bin/env python3
"""Profile test for 7c steering validation.

Runs a single prefix with profiling to identify bottlenecks.

Usage:
    PROFILE_7C=1 python 7_validation/7c_profile_test.py \
        --sweep-dir Qwen3_4B_results/sweep_span_lcs_plus_one_pool_sum \
        --config configs/default_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

# Set profiling env var before imports
os.environ["PROFILE_7C"] = "1"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from utils.data_utils import load_json
from utils.memory_utils import clear_memory, reset_cuda_state
from circuit_tracer import ReplacementModel

# Import local modules
import importlib.util
_module_dir = Path(__file__).parent

def _import_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

graph = _import_module("7c_graph", _module_dir / "7c_graph.py")
steering = _import_module("7c_steering", _module_dir / "7c_steering.py")
metrics = _import_module("7c_metrics", _module_dir / "7c_metrics.py")
utils = _import_module("7c_utils", _module_dir / "7c_utils.py")

# Use shared profiler from sys.modules (set by steering.py)
if "7c_profiler" in sys.modules:
    profiler = sys.modules["7c_profiler"]
else:
    profiler = _import_module("7c_profiler", _module_dir / "profiler.py")
    sys.modules["7c_profiler"] = profiler


def run_single_prefix_profiled(
    model,
    prefix_id: str,
    branches_data: dict,
    clustering_data: dict,
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    config: dict,
    max_batch_size: int = 32,
    cross_prefix_batching: bool = True,
):
    """Run profiled steering on a single prefix."""
    stage_7c = config.get("stage_7c_steering", {})
    max_seq_len = config.get("global", {}).get("max_seq_len", 64)
    h_c_strategy = stage_7c.get("h_c_strategy", "H_c_centered")
    device = model.cfg.device

    print(f"\nConfig: h_c_strategy={h_c_strategy}")
    print(f"        max_batch_size={max_batch_size}, cross_prefix_batching={cross_prefix_batching}")

    # Get components and H_0 from clustering data
    components = clustering_data.get("components", {})
    H_0 = None
    H_0_raw = clustering_data.get("H_0")
    if H_0_raw is not None:
        H_0 = np.array(H_0_raw)

    # Build semantic graphs with strategy
    with profiler.timed("compute_semantic_graphs"):
        semantic_graphs = graph.compute_semantic_graphs_with_strategy(
            components, h_c_strategy, H_0, logger=None
        )
    print(f"  Built semantic graphs for {len(semantic_graphs)} clusters")

    # Use provided features
    n_features = len(selected_features)

    # Collect needed feature indices
    max_top_B = 10
    with profiler.timed("collect_feature_indices"):
        all_needed_indices = set()
        for cluster_id, H_c in semantic_graphs.items():
            H_c_features = H_c[:n_features]
            abs_vals = np.abs(H_c_features)
            top_indices = np.argsort(abs_vals)[-(max_top_B * 2):]
            for idx in top_indices:
                if abs(H_c_features[idx]) >= utils.EPSILON_SMALL:
                    all_needed_indices.add(int(idx))
        print(f"  Need {len(all_needed_indices)} decoder vectors")

    # Precompute decoder vectors
    with profiler.timed("precompute_decoder_vectors"):
        global_decoder_cache = {}
        if all_needed_indices:
            layer_to_indices = {}
            for h_c_idx in all_needed_indices:
                feat_idx = selected_features[h_c_idx].item()
                layer, pos, feat_id = active_features[feat_idx].tolist()
                layer = int(layer)
                if layer not in layer_to_indices:
                    layer_to_indices[layer] = []
                layer_to_indices[layer].append((h_c_idx, int(feat_id)))

            for layer, idx_list in layer_to_indices.items():
                h_c_indices = [x[0] for x in idx_list]
                feat_ids = [x[1] for x in idx_list]
                feat_ids_t = torch.tensor(feat_ids, device=device, dtype=torch.long)
                dec_vecs = model.transcoders._get_decoder_vectors(layer, feat_ids_t)
                for i, h_c_idx in enumerate(h_c_indices):
                    global_decoder_cache[h_c_idx] = dec_vecs[i]

    # Build cluster decoder cache
    with profiler.timed("build_cluster_decoder_cache"):
        decoder_cache = graph.build_cluster_decoder_cache(
            semantic_graphs, global_decoder_cache, active_features, selected_features,
            max_features=max_top_B * 2
        )

    # Build encoder cache
    with profiler.timed("build_encoder_cache"):
        feats_by_c = {}
        for c, cache_data in decoder_cache.items():
            tuples = []
            for i in range(len(cache_data['h_c_values'])):
                tuples.append((
                    cache_data['layers'][i],
                    cache_data['positions'][i],
                    cache_data['feat_ids'][i],
                    cache_data['h_c_values'][i]
                ))
            feats_by_c[c] = tuples

        encoder_cache = graph.precompute_cluster_encoder_weights(
            model, feats_by_c, device
        )

    # Build branches
    with profiler.timed("build_branches"):
        assignments = clustering_data.get("assignments", [])
        branches = utils.build_branches_from_data(branches_data, assignments)
        print(f"  Built {len(branches)} branches")

    # Limit branches for profiling
    branches = branches[:100]  # Limit to 100 branches for faster profiling
    print(f"  Using {len(branches)} branches for profiling")

    # Compute baselines
    with profiler.timed("compute_branch_log_probs"):
        branch_log_probs = steering.compute_branch_log_probs_batch(
            model, branches, logger=None, batch_size=max_batch_size, max_seq_len=max_seq_len
        )

    with profiler.timed("compute_baseline_metadata"):
        baseline_metadata = steering.compute_baseline_metadata(branches, branch_log_probs)

    # Run steering sweep (limited epsilons for profiling)
    epsilons = [-1.0, 0.0, 1.0]
    steering_method = stage_7c.get("steering_method", "sign")
    hc_selection = "full"
    top_B = 10

    print(f"\n--- Running steering sweep ---")
    print(f"  method={steering_method}, hc_selection={hc_selection}, top_B={top_B}")
    print(f"  epsilons={epsilons}")

    # Import run_steering_sweep
    hypotheses = _import_module("7c_hypotheses", _module_dir / "7c_hypotheses.py")

    with profiler.timed("run_steering_sweep_total"):
        result = hypotheses.run_steering_sweep(
            model=model,
            branches=branches,
            decoder_cache=decoder_cache,
            encoder_cache=encoder_cache,
            baseline_metadata=baseline_metadata,
            epsilons=epsilons,
            top_B=top_B,
            steering_method=steering_method,
            hc_selection=hc_selection,
            max_samples_per_cluster=30,
            log_details=False,
            max_batch_size=max_batch_size,
            cross_prefix_batching=cross_prefix_batching,
            max_seq_len=max_seq_len,
            logger=None,
        )

    print(f"\n  Result keys: {list(result.keys())[:5]}...")

    # Print profiling results
    profiler.print_timings()


def main():
    parser = argparse.ArgumentParser(description="Profile 7c steering validation")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Sweep results directory")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--cross-prefix-batching", action="store_true", default=True)
    parser.add_argument("--no-cross-prefix-batching", dest="cross_prefix_batching", action="store_false")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    config_path = Path(args.config)
    results_dir = sweep_dir / "results"

    with open(config_path) as f:
        config = json.load(f)

    # Load model
    model_cfg = config.get("model", {})
    print(f"\nLoading model: {model_cfg.get('base_model')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_cuda_state()
    model = ReplacementModel.from_pretrained(
        model_cfg.get("base_model"),
        model_cfg.get("transcoder"),
        device=device,
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=False,
    )

    # Find first prefix
    clustering_dir = results_dir / "5_clustering"
    samples_dir = results_dir / "2_branch_sampling"
    attr_dir = results_dir / "3_attribution_graphs"

    clustering_files = sorted(clustering_dir.glob("cloze_*_clustering.json"))[:1]
    if not clustering_files:
        raise ValueError(f"No clustering files found in {clustering_dir}")

    prefix_id = clustering_files[0].stem.replace("_clustering", "")
    print(f"Using prefix: {prefix_id}")

    # Load data
    clustering_data = load_json(clustering_files[0])
    branches_path = samples_dir / f"{prefix_id}_branches.json"
    branches_data = load_json(branches_path)

    # Load attribution data
    active_features, selected_features = graph.load_attribution_context(
        attr_dir, prefix_id, use_continuation_attribution=True
    )

    # Run profiled test
    run_single_prefix_profiled(
        model=model,
        prefix_id=prefix_id,
        branches_data=branches_data,
        clustering_data=clustering_data,
        active_features=active_features,
        selected_features=selected_features,
        config=config,
        max_batch_size=args.batch_size,
        cross_prefix_batching=args.cross_prefix_batching,
    )


if __name__ == "__main__":
    main()
