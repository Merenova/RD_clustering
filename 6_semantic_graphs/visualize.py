#!/usr/bin/env -S uv run python
"""Visualization tools for semantic graphs and clustering results.

NOTE: This module is now a thin wrapper. For comprehensive visualizations,
use Stage 8 (8_visualization/visualize.py) which consolidates all pipeline
visualizations and organizes outputs by source stage.

This file is kept for backward compatibility but delegates to Stage 8 functions.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import PathConfig
from utils.data_utils import load_json, load_torch
from utils.logging_utils import setup_logger, get_log_path

# Import visualization functions from Stage 8
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "8_visualization"))

from cluster_plots import plot_clustering_history
from semantic_graph_plots import plot_semantic_graph_heatmap, plot_token_scores


def main():
    parser = argparse.ArgumentParser(
        description="Visualize semantic graphs (delegates to Stage 8 functions)"
    )
    parser.add_argument(
        "--clustering-dir",
        type=Path,
        required=True,
        help="Directory with clustering results"
    )
    parser.add_argument(
        "--semantic-graphs-dir",
        type=Path,
        required=True,
        help="Directory with semantic graphs"
    )
    parser.add_argument(
        "--first-tokens-dir",
        type=Path,
        default=None,
        help="Directory with first tokens data (for token text labels)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/8_visualization/6_semantic_graphs/)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files"
    )
    args = parser.parse_args()

    # Setup paths
    paths = PathConfig()
    paths.ensure_dirs()

    # Default output to Stage 8 visualization directory
    if args.output_dir is None:
        args.output_dir = paths.results_visualization / "6_semantic_graphs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = get_log_path("6_semantic_graphs_visualize", args.log_dir)
    logger = setup_logger("visualization", log_file=log_file)

    logger.info("=" * 60)
    logger.info("SEMANTIC GRAPHS VISUALIZATION")
    logger.info("(Delegating to Stage 8 visualization functions)")
    logger.info("=" * 60)
    logger.info(f"Clustering dir: {args.clustering_dir}")
    logger.info(f"Semantic graphs dir: {args.semantic_graphs_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Find all results
    clustering_files = sorted(args.clustering_dir.glob("*_sweep_results.json"))
    logger.info(f"\nFound {len(clustering_files)} sweep results to visualize")

    # Process each result
    for clustering_file in clustering_files:
        prefix_id = clustering_file.stem.replace("_sweep_results", "")
        logger.info(f"\nVisualizing: {prefix_id}")

        # Create prefix output directory
        prefix_output_dir = args.output_dir / prefix_id
        prefix_output_dir.mkdir(parents=True, exist_ok=True)

        # Load clustering sweep results
        try:
            clustering_sweep = load_json(clustering_file)
        except Exception as e:
            logger.error(f"  Failed to load {clustering_file}: {e}")
            logger.error(f"  File may be corrupted - consider deleting and regenerating")
            continue

        grid_results = clustering_sweep.get("grid", [])
        valid_grid = [
            entry for entry in grid_results
            if entry.get("components") and entry.get("assignments") and "error" not in entry
        ]
        if not valid_grid:
            logger.warning(f"  No valid clustering results for {prefix_id}")
            continue

        for grid_entry in valid_grid:
            beta = grid_entry.get("beta")
            gamma = grid_entry.get("gamma")
            clustering_key = f"beta{beta}_gamma{gamma}"

            cluster_output_dir = prefix_output_dir / clustering_key
            cluster_output_dir.mkdir(parents=True, exist_ok=True)

            # Plot clustering history (using Stage 8 function)
            history = grid_entry.get("history", {})
            if history:
                history_plot = cluster_output_dir / f"{prefix_id}_{clustering_key}_clustering_history.png"
                plot_clustering_history(history, prefix_id, history_plot)
                logger.info(f"  Saved history plot: {history_plot}")

            # Load semantic graphs
            graphs_file_pt = args.semantic_graphs_dir / f"{prefix_id}_{clustering_key}_semantic_graphs.pt"

            if graphs_file_pt.exists():
                graphs_data = load_torch(graphs_file_pt)

                # Plot semantic graph heatmap (using Stage 8 function)
                semantic_graphs = graphs_data.get("semantic_graphs", {})
                if semantic_graphs:
                    heatmap_plot = cluster_output_dir / f"{prefix_id}_{clustering_key}_semantic_graph_heatmap.png"
                    plot_semantic_graph_heatmap(semantic_graphs, prefix_id, heatmap_plot)
                    logger.info(f"  Saved heatmap: {heatmap_plot}")

                # Load token_id to text mapping if first_tokens_dir is provided
                token_id_to_text = None
                if args.first_tokens_dir:
                    first_tokens_file = args.first_tokens_dir / f"{prefix_id}_first_tokens.json"
                    if first_tokens_file.exists():
                        first_tokens_data = load_json(first_tokens_file)
                        token_id_to_text = {
                            ft["token_id"]: ft["token_text"]
                            for ft in first_tokens_data.get("first_tokens", [])
                        }
                        logger.info(f"  Loaded token mapping: {len(token_id_to_text)} tokens")

                # Plot token scores (using Stage 8 function)
                token_scores = graphs_data.get("token_scores", {})
                if token_scores:
                    token_plot = cluster_output_dir / f"{prefix_id}_{clustering_key}_token_scores.png"
                    plot_token_scores(
                        token_scores, prefix_id, token_plot,
                        token_id_to_text=token_id_to_text
                    )
                    logger.info(f"  Saved token scores: {token_plot}")

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    logger.info("TIP: For comprehensive visualizations, use Stage 8:")
    logger.info("  uv run python 8_visualization/visualize.py --config configs/default_config.json")


if __name__ == "__main__":
    main()
