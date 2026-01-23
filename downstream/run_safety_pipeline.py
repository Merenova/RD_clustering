#!/usr/bin/env python3
"""Safety Pipeline Entry Point.

Full pipeline for HarmBench-based safety motif discovery:
1. Data Loading: Download and prepare HarmBench prompts
2. Rollout: Generate continuations from Base and Instruct models
3. Labeling: Classify refusal/compliance
4. Filtering: Select by disagreement
5. Embedding: Compute semantic + attribution embeddings
6. Clustering: Run RD clustering with sweep
7. Experiments: Run Exp 1, 2, 5 (core + safety application)
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import setup_logger
from utils.data_utils import save_json, load_json

# Import from downstream modules
from downstream.config import (
    OUTPUT_DIR,
    SAFETY_CONFIG,
    ATTRIBUTION_CONFIG,
    ROLLOUT_CONFIG,
    DEFAULT_SEED,
)


def main():
    parser = argparse.ArgumentParser(description="Run Safety Motif Discovery Pipeline")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "safety_pipeline")
    parser.add_argument("--n-prompts", type=int, default=SAFETY_CONFIG["n_prompts"])
    parser.add_argument("--n-continuations", type=int, default=ROLLOUT_CONFIG["n_continuations"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    
    # Stage flags
    parser.add_argument("--skip-rollout", action="store_true", help="Skip rollout if cached")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding if cached")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip clustering if cached")
    parser.add_argument("--only-stage", type=str, choices=["data", "rollout", "label", "filter", "embed", "cluster", "exp"], help="Run only specified stage")
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("safety_pipeline", args.output_dir / "pipeline.log")
    
    logger.info("=" * 60)
    logger.info("SAFETY MOTIF DISCOVERY PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"N prompts: {args.n_prompts}")
    logger.info(f"N continuations: {args.n_continuations}")
    
    # Stage 1: Data Loading
    prompts_file = args.output_dir / "prompts.json"
    if args.only_stage is None or args.only_stage == "data":
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 1: DATA LOADING")
        logger.info("=" * 40)
        
        from downstream.data.load_harmbench import download_harmbench, prepare_harmbench_prompts
        
        behaviors = download_harmbench(logger=logger)
        prompts = prepare_harmbench_prompts(
            behaviors,
            n_prompts=args.n_prompts,
            seed=args.seed,
            logger=logger,
        )
        
        save_json(prompts, prompts_file)
        logger.info(f"Prepared {len(prompts)} prompts")
        
        if args.only_stage == "data":
            logger.info("Stage complete (--only-stage data)")
            return
    else:
        prompts = load_json(prompts_file)
        logger.info(f"Loaded {len(prompts)} prompts from cache")
    
    # Stage 2: Rollout
    rollout_file = args.output_dir / "safety_rollout_results.json"
    
    if args.only_stage is None or args.only_stage == "rollout":
        if rollout_file.exists() and args.skip_rollout:
            logger.info("\nSkipping rollout (cached)")
            rollout_results = load_json(rollout_file)["results"]
        else:
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 2: ROLLOUT")
            logger.info("=" * 40)
            
            from downstream.rollout.rollout_safety import rollout_safety_task
            
            rollout_results = rollout_safety_task(
                prompts,
                n_continuations=args.n_continuations,
                output_dir=args.output_dir,
                save_results=True,
                logger=logger,
            )
        
        if args.only_stage == "rollout":
            logger.info("Stage complete (--only-stage rollout)")
            return
    else:
        rollout_results = load_json(rollout_file)["results"]
    
    # Stage 3: Labeling
    labeled_file = args.output_dir / "labeled_results.json"
    if args.only_stage is None or args.only_stage == "label":
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 3: LABELING")
        logger.info("=" * 40)
        
        from downstream.labeling.refusal_judge import RefusalJudge, judge_safety_rollout
        
        judge = RefusalJudge()
        rollout_results = judge_safety_rollout(rollout_results, judge, logger)
        judge.cleanup()
        
        save_json({"results": rollout_results}, labeled_file)
        
        if args.only_stage == "label":
            logger.info("Stage complete (--only-stage label)")
            return
    else:
        data = load_json(labeled_file)
        rollout_results = data["results"]
    
    # Stage 4: Filtering
    filtered_file = args.output_dir / "filtered_prompts.json"
    samples_file_instruct = args.output_dir / "instruct_samples.json"
    samples_file_base = args.output_dir / "base_samples.json"
    
    if args.only_stage is None or args.only_stage == "filter":
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 4: FILTERING")
        logger.info("=" * 40)
        
        from downstream.filtering.safety_filter import (
            filter_by_disagreement,
            prepare_for_clustering,
            save_filtered_prompts,
        )
        
        filtered_prompts, filter_stats = filter_by_disagreement(
            rollout_results,
            logger=logger,
        )
        
        save_filtered_prompts(filtered_prompts, filtered_file, logger=logger)
        
        # Prepare data for clustering
        instruct_data = prepare_for_clustering(filtered_prompts, model_type="instruct")
        base_data = prepare_for_clustering(filtered_prompts, model_type="base")
        
        save_json(instruct_data, samples_file_instruct)
        save_json(base_data, samples_file_base)
        
        if args.only_stage == "filter":
            logger.info("Stage complete (--only-stage filter)")
            return
    else:
        instruct_data = load_json(samples_file_instruct)
        base_data = load_json(samples_file_base)
    
    # Stage 5: Embedding (compute for BOTH base and instruct)
    embed_file = args.output_dir / "embeddings.npz"
    # Also save filtered sample metadata for label alignment
    instruct_meta_file = args.output_dir / "instruct_samples_valid.json"
    base_meta_file = args.output_dir / "base_samples_valid.json"
    
    if args.only_stage is None or args.only_stage == "embed":
        if embed_file.exists() and args.skip_embedding:
            logger.info("\nSkipping embedding (cached)")
            import numpy as np
            emb_data = np.load(embed_file)
            instruct_embeddings = emb_data["instruct_embeddings"]
            instruct_attributions = emb_data["instruct_attributions"]
            # Pooled attributions (fallback to non-pooled for backward compatibility)
            instruct_attributions_pooled = emb_data.get(
                "instruct_attributions_pooled",
                emb_data["instruct_attributions"]
            )
            base_embeddings = emb_data["base_embeddings"]
            base_attributions = emb_data["base_attributions"]
            base_attributions_pooled = emb_data.get(
                "base_attributions_pooled",
                emb_data["base_attributions"]
            )
            # Load filtered sample metadata for labels
            instruct_data = load_json(instruct_meta_file)
            base_data = load_json(base_meta_file)
        else:
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 5: EMBEDDING")
            logger.info("=" * 40)
            
            import numpy as np
            from downstream.embedding.semantic import compute_embeddings_batch, load_embedding_model
            from downstream.embedding.attribution import compute_attributions_batch, load_tokenizer
            
            embedding_model = load_embedding_model()
            tokenizer = load_tokenizer()
            
            # Compute embeddings for INSTRUCT data (both pooled and non-pooled)
            logger.info("Computing instruct embeddings...")
            instruct_data = compute_embeddings_batch(instruct_data, embedding_model, logger=logger)
            instruct_data = compute_attributions_batch(
                instruct_data,
                tokenizer,
                compute_pooled=ATTRIBUTION_CONFIG["compute_pooled"],
                pooling_method=ATTRIBUTION_CONFIG["pooling_method"],
                logger=logger,
            )
            
            # Compute embeddings for BASE data (both pooled and non-pooled)
            logger.info("Computing base embeddings...")
            base_data = compute_embeddings_batch(base_data, embedding_model, logger=logger)
            base_data = compute_attributions_batch(
                base_data,
                tokenizer,
                compute_pooled=ATTRIBUTION_CONFIG["compute_pooled"],
                pooling_method=ATTRIBUTION_CONFIG["pooling_method"],
                logger=logger,
            )
            
            # Extract arrays - filter out samples with missing embeddings
            # For cross-prefix experiments (safety), we need pooled attributions
            # Position-specific attributions have different shapes per prefix (due to error/token nodes)
            instruct_valid = [d for d in instruct_data 
                              if d.get("embedding") is not None and d.get("attribution_pooled") is not None]
            base_valid = [d for d in base_data 
                          if d.get("embedding") is not None and d.get("attribution_pooled") is not None]
            
            instruct_embeddings = np.stack([d["embedding"] for d in instruct_valid])
            # For cross-prefix clustering, use pooled attributions (consistent shape across prefixes)
            # Position-specific attributions can't be stacked across different prefixes
            instruct_attributions_pooled = np.stack([d["attribution_pooled"] for d in instruct_valid])
            instruct_attributions = instruct_attributions_pooled  # Use pooled for all cross-prefix experiments
            
            base_embeddings = np.stack([d["embedding"] for d in base_valid])
            base_attributions_pooled = np.stack([d["attribution_pooled"] for d in base_valid])
            base_attributions = base_attributions_pooled  # Use pooled for all cross-prefix experiments
            
            # === Sparse reduction: keep only non-zero features ===
            # This reduces storage from ~170GB to ~1GB and makes clustering feasible
            active_feature_indices = None
            original_attr_dim = instruct_attributions_pooled.shape[1]
            n_layers = None
            d_transcoder = None
            
            if ATTRIBUTION_CONFIG.get("enable_sparse", True):
                sparse_threshold = ATTRIBUTION_CONFIG.get("sparse_threshold", 1e-10)
                
                # Find features that are non-zero in ANY sample (across both models)
                all_pooled = np.vstack([instruct_attributions_pooled, base_attributions_pooled])
                nonzero_mask = np.any(np.abs(all_pooled) > sparse_threshold, axis=0)
                active_feature_indices = np.where(nonzero_mask)[0]
                
                n_active = len(active_feature_indices)
                logger.info(f"Sparse reduction: {n_active} / {original_attr_dim} features active "
                            f"({100*n_active/original_attr_dim:.2f}%)")
                
                # Reduce to active features only
                instruct_attributions_pooled = instruct_attributions_pooled[:, nonzero_mask]
                instruct_attributions = instruct_attributions_pooled
                base_attributions_pooled = base_attributions_pooled[:, nonzero_mask]
                base_attributions = base_attributions_pooled
                
                # Get model dimensions for later remapping (layer, feat_idx)
                # d_transcoder = d_mlp * 8, original_dim = n_layers * d_transcoder
                # We'll estimate from the first sample's metadata if available
                if instruct_valid and instruct_valid[0].get("n_layers"):
                    n_layers = instruct_valid[0]["n_layers"]
                    d_transcoder = original_attr_dim // n_layers
                else:
                    # Fallback: assume 36 layers (Qwen3-8B)
                    n_layers = 36
                    d_transcoder = original_attr_dim // n_layers
            # === End sparse reduction ===
            
            # Update data lists to only include valid samples (remove numpy arrays for JSON)
            instruct_data = [{k: v for k, v in d.items() if k not in ("embedding", "attribution", "attribution_pooled")} for d in instruct_valid]
            base_data = [{k: v for k, v in d.items() if k not in ("embedding", "attribution", "attribution_pooled")} for d in base_valid]
            
            # Save filtered sample metadata (for label alignment on reload)
            save_json(instruct_data, instruct_meta_file)
            save_json(base_data, base_meta_file)
            
            # Build save dict with sparse metadata
            save_dict = {
                "instruct_embeddings": instruct_embeddings,
                # For cross-prefix: both are pooled (position-specific can't be stacked across prefixes)
                "instruct_attributions": instruct_attributions_pooled,
                "instruct_attributions_pooled": instruct_attributions_pooled,
                "base_embeddings": base_embeddings,
                "base_attributions": base_attributions_pooled,
                "base_attributions_pooled": base_attributions_pooled,
            }
            
            # Add sparse metadata for steering remapping
            if active_feature_indices is not None:
                save_dict["active_feature_indices"] = active_feature_indices
                save_dict["original_attr_dim"] = np.array([original_attr_dim])
                if n_layers is not None:
                    save_dict["n_layers"] = np.array([n_layers])
                if d_transcoder is not None:
                    save_dict["d_transcoder"] = np.array([d_transcoder])
            
            np.savez(embed_file, **save_dict)
            logger.info(f"Saved embeddings to {embed_file}")
            logger.info(f"  Instruct samples: {len(instruct_embeddings)}")
            logger.info(f"  Base samples: {len(base_embeddings)}")
        
        if args.only_stage == "embed":
            logger.info("Stage complete (--only-stage embed)")
            return
    else:
        import numpy as np
        emb_data = np.load(embed_file)
        instruct_embeddings = emb_data["instruct_embeddings"]
        instruct_attributions = emb_data["instruct_attributions"]
        # Pooled attributions (fallback to non-pooled for backward compatibility)
        instruct_attributions_pooled = emb_data.get(
            "instruct_attributions_pooled",
            emb_data["instruct_attributions"]
        )
        base_embeddings = emb_data["base_embeddings"]
        base_attributions = emb_data["base_attributions"]
        base_attributions_pooled = emb_data.get(
            "base_attributions_pooled",
            emb_data["base_attributions"]
        )
        # Load sparse metadata for steering remapping (if present)
        active_feature_indices = emb_data.get("active_feature_indices", None)
        original_attr_dim = emb_data["original_attr_dim"][0] if "original_attr_dim" in emb_data else None
        n_layers = emb_data["n_layers"][0] if "n_layers" in emb_data else None
        d_transcoder = emb_data["d_transcoder"][0] if "d_transcoder" in emb_data else None
        # Load filtered sample metadata for labels
        instruct_data = load_json(instruct_meta_file)
        base_data = load_json(base_meta_file)
    
    # Get labels (now aligned with embeddings)
    import numpy as np
    instruct_labels = np.array([d.get("is_refusal", False) for d in instruct_data], dtype=int)
    base_labels = np.array([d.get("is_refusal", False) for d in base_data], dtype=int)
    
    # Stage 6: Clustering (on instruct data - used to find optimal hyperparameters)
    cluster_file = args.output_dir / "clustering_results.json"
    
    if args.only_stage is None or args.only_stage == "cluster":
        if cluster_file.exists() and args.skip_clustering:
            logger.info("\nSkipping clustering (cached)")
            cluster_results = load_json(cluster_file)
        else:
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 6: CLUSTERING")
            logger.info("=" * 40)
            
            from downstream.clustering.rd_wrapper import run_rd_sweep
            from downstream.clustering.config_selection import select_optimal_config, summarize_sweep_results
            
            sweep_results = run_rd_sweep(
                instruct_embeddings,
                instruct_attributions,
            )
            
            optimal_beta, optimal_gamma, best_result, best_score = select_optimal_config(
                sweep_results,
                instruct_embeddings,
                instruct_attributions,
            )
            
            logger.info(f"Optimal config: beta={optimal_beta}, gamma={optimal_gamma}")
            logger.info(f"Best harmonic silhouette: {best_score:.4f}")
            
            summaries = summarize_sweep_results(
                sweep_results, instruct_embeddings, instruct_attributions, instruct_labels
            )
            
            cluster_results = {
                "optimal_beta": optimal_beta,
                "optimal_gamma": optimal_gamma,
                "n_clusters": best_result["n_components"],
                "sweep_summaries": summaries,
            }
            save_json(cluster_results, cluster_file)
        
        if args.only_stage == "cluster":
            logger.info("Stage complete (--only-stage cluster)")
            return
    else:
        cluster_results = load_json(cluster_file)
    
    # Stage 7: Experiments
    if args.only_stage is None or args.only_stage == "exp":
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 7: EXPERIMENTS")
        logger.info("=" * 40)
        
        optimal_beta = cluster_results["optimal_beta"]
        optimal_gamma = cluster_results["optimal_gamma"]
        
        # Experiment 1: Clustering Quality
        from downstream.experiments.exp1_clustering_quality import run_experiment_1
        
        exp1_results = run_experiment_1(
            instruct_embeddings,
            instruct_attributions,
            labels=instruct_labels,
            output_dir=args.output_dir / "exp1",
            logger=logger,
        )
        
        # Experiment 2: Binary vs RD
        from downstream.experiments.exp2_binary_vs_rd import run_experiment_2
        
        exp2_results = run_experiment_2(
            instruct_embeddings,
            instruct_attributions,
            instruct_labels,
            optimal_beta=optimal_beta,
            optimal_gamma=optimal_gamma,
            output_dir=args.output_dir / "exp2",
            logger=logger,
        )
        
        # Experiment 5: Safety Application (compare BASE vs INSTRUCT embeddings)
        # Uses POOLED attributions for cross-model comparison
        from downstream.experiments.exp5_safety_application import run_experiment_5
        
        base_data_dict = {
            "embeddings": base_embeddings,
            "attributions": base_attributions_pooled,  # POOLED for cross-model Exp5
            "labels": base_labels,
        }
        
        instruct_data_dict = {
            "embeddings": instruct_embeddings,
            "attributions": instruct_attributions_pooled,  # POOLED for cross-model Exp5
            "labels": instruct_labels,
        }
        
        exp5_results = run_experiment_5(
            base_data_dict,
            instruct_data_dict,
            optimal_beta,
            optimal_gamma,
            output_dir=args.output_dir / "exp5",
            logger=logger,
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

