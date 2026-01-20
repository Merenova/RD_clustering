#!/usr/bin/env python3
"""Hallucination Pipeline Entry Point.

Full pipeline for PopQA-based hallucination motif discovery:
1. Data Loading: Load PopQA questions
2. Rollout: Generate continuations from Instruct model
3. Labeling: Classify correct/wrong
4. Filtering: Select by entropy
5. Embedding: Compute semantic + attribution embeddings
6. Clustering: Run RD clustering with sweep
7. Experiments: Run Exp 1, 2, 4, 6 (core + cross-prefix + hallucination)
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
    HALLUCINATION_CONFIG,
    ATTRIBUTION_CONFIG,
    ROLLOUT_CONFIG,
    DEFAULT_SEED,
)


def main():
    parser = argparse.ArgumentParser(description="Run Hallucination Motif Discovery Pipeline")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "hallucination_pipeline")
    parser.add_argument("--n-questions", type=int, default=HALLUCINATION_CONFIG["n_questions_initial"])
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
    logger = setup_logger("hallucination_pipeline", args.output_dir / "pipeline.log")
    
    logger.info("=" * 60)
    logger.info("HALLUCINATION MOTIF DISCOVERY PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"N questions: {args.n_questions}")
    logger.info(f"N continuations: {args.n_continuations}")
    
    # Stage 1: Data Loading
    questions_file = args.output_dir / "questions.json"
    if args.only_stage is None or args.only_stage == "data":
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 1: DATA LOADING")
        logger.info("=" * 40)
        
        from downstream.data.load_popqa import load_popqa_questions
        
        questions = load_popqa_questions(
            n_samples=args.n_questions,
            seed=args.seed,
            logger=logger,
        )
        
        save_json(questions, questions_file)
        logger.info(f"Loaded {len(questions)} questions")
        
        if args.only_stage == "data":
            logger.info("Stage complete (--only-stage data)")
            return
    else:
        questions = load_json(questions_file)
        logger.info(f"Loaded {len(questions)} questions from cache")
    
    # Stage 2: Rollout
    rollout_file = args.output_dir / "hallucination_rollout_results.json"
    
    if args.only_stage is None or args.only_stage == "rollout":
        if rollout_file.exists() and args.skip_rollout:
            logger.info("\nSkipping rollout (cached)")
            rollout_results = load_json(rollout_file)["results"]
        else:
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 2: ROLLOUT")
            logger.info("=" * 40)
            
            from downstream.rollout.rollout_hallucination import rollout_hallucination_task
            
            rollout_results = rollout_hallucination_task(
                questions,
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
        
        from downstream.labeling.correctness_judge import judge_hallucination_rollout
        
        rollout_results = judge_hallucination_rollout(rollout_results, logger=logger)
        
        save_json({"results": rollout_results}, labeled_file)
        
        if args.only_stage == "label":
            logger.info("Stage complete (--only-stage label)")
            return
    else:
        data = load_json(labeled_file)
        rollout_results = data["results"]
    
    # Stage 4: Filtering (answer entropy - first pass before embeddings)
    filtered_stage1_file = args.output_dir / "filtered_results_stage1.json"
    if args.only_stage is None or args.only_stage == "filter":
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 4: FILTERING (ANSWER ENTROPY)")
        logger.info("=" * 40)
        
        from downstream.filtering.hallucination_filter import (
            filter_by_answer_entropy,
            save_filtered_results,
        )
        
        filtered_results, filter_stats = filter_by_answer_entropy(
            rollout_results,
            logger=logger,
        )
        
        save_filtered_results(filtered_results, filtered_stage1_file, logger=logger)
        
        if args.only_stage == "filter":
            logger.info("Stage complete (--only-stage filter)")
            return
    else:
        data = load_json(filtered_stage1_file)
        filtered_results = data["results"]
    
    # Stage 5: Embedding + second-pass filter (embedding entropy)
    embed_file = args.output_dir / "embeddings.npz"
    
    if args.only_stage is None or args.only_stage == "embed":
        if embed_file.exists() and args.skip_embedding:
            logger.info("\nSkipping embedding (cached)")
            import numpy as np
            emb_data = np.load(embed_file, allow_pickle=True)
            filtered_results = emb_data["filtered_results"].tolist()
        else:
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 5: EMBEDDING")
            logger.info("=" * 40)
            
            import numpy as np
            from downstream.embedding.semantic import compute_embeddings_for_results
            from downstream.embedding.attribution import compute_attributions_for_results, load_tokenizer
            from downstream.filtering.hallucination_filter import (
                filter_by_embedding_entropy,
                save_filtered_results,
            )
            
            # Semantic embeddings
            logger.info("Computing semantic embeddings...")
            filtered_results = compute_embeddings_for_results(filtered_results, logger=logger)
            
            # Second-pass filter: embedding entropy (diverse responses)
            logger.info("Applying embedding entropy filter...")
            filtered_results, embed_filter_stats = filter_by_embedding_entropy(
                filtered_results,
                logger=logger,
            )
            
            # Attribution embeddings (both pooled and non-pooled)
            logger.info("Computing attribution embeddings...")
            tokenizer = load_tokenizer()
            filtered_results = compute_attributions_for_results(
                filtered_results,
                tokenizer,
                compute_pooled=ATTRIBUTION_CONFIG["compute_pooled"],
                pooling_method=ATTRIBUTION_CONFIG["pooling_method"],
                logger=logger,
            )
            
            # Save final filtered results
            save_filtered_results(filtered_results, args.output_dir / "filtered_results.json", logger=logger)
            
            # Save embeddings (both pooled and non-pooled stored in filtered_results)
            np.savez(
                embed_file,
                filtered_results=np.array(filtered_results, dtype=object),
            )
            logger.info(f"Saved embeddings to {embed_file}")
        
        if args.only_stage == "embed":
            logger.info("Stage complete (--only-stage embed)")
            return
    else:
        import numpy as np
        emb_data = np.load(embed_file, allow_pickle=True)
        filtered_results = emb_data["filtered_results"].tolist()
    
    # Flatten for clustering - track valid indices per question
    # For cross-question operations (clustering sweep, Exp1, Exp2, Exp4), we need pooled attributions
    # because different questions have different prefix lengths (different n_prefix_sources)
    # Position-specific attributions are kept per-question in filtered_results for potential per-question use
    import numpy as np
    
    all_embeddings = []
    all_attributions_pooled = []  # Pooled for all cross-question operations
    all_labels = []
    
    # Track valid sample indices per question for Exp4
    per_question_valid_indices = {}  # q_idx -> list of (global_idx, local_idx)
    global_idx = 0
    
    for result in filtered_results:
        q_idx = result.get("question_idx", len(per_question_valid_indices))
        embs = result.get("embeddings", [])
        attrs_pooled = result.get("attributions_pooled", [])
        conts = result.get("continuations", [])
        
        if isinstance(embs, np.ndarray):
            embs = embs.tolist()
        if isinstance(attrs_pooled, np.ndarray):
            attrs_pooled = attrs_pooled.tolist()
        
        valid_indices = []
        for i in range(len(embs)):
            e = embs[i] if i < len(embs) else None
            a_pooled = attrs_pooled[i] if i < len(attrs_pooled) else None
            
            # Filter by pooled attribution presence (consistent shape across questions)
            if e is not None and a_pooled is not None:
                all_embeddings.append(e)
                all_attributions_pooled.append(a_pooled)
                if i < len(conts):
                    label = conts[i].get("is_correct", False) if isinstance(conts[i], dict) else False
                else:
                    label = False
                all_labels.append(label)
                valid_indices.append((global_idx, i))  # (global flat index, local continuation index)
                global_idx += 1
        
        per_question_valid_indices[q_idx] = valid_indices
    
    embeddings_e = np.array(all_embeddings) if all_embeddings else np.array([])
    # Use pooled attributions for all cross-question operations (consistent shape)
    attributions_a = np.array(all_attributions_pooled) if all_attributions_pooled else np.array([])
    attributions_a_pooled = attributions_a  # Alias for clarity
    labels = np.array(all_labels, dtype=int)
    
    logger.info(f"Total samples for clustering: {len(embeddings_e)}")
    
    # Stage 6: Clustering
    cluster_file = args.output_dir / "clustering_results.json"
    
    if args.only_stage is None or args.only_stage == "cluster":
        if cluster_file.exists() and args.skip_clustering:
            logger.info("\nSkipping clustering (cached)")
            cluster_results = load_json(cluster_file)
        else:
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 6: CLUSTERING")
            logger.info("=" * 40)
            
            if len(embeddings_e) < 3:
                logger.warning("Not enough samples for clustering")
                cluster_results = {"error": "Not enough samples"}
            else:
                from downstream.clustering.rd_wrapper import run_rd_sweep
                from downstream.clustering.config_selection import select_optimal_config, summarize_sweep_results
                
                sweep_results = run_rd_sweep(
                    embeddings_e,
                    attributions_a,
                )
                
                optimal_beta, optimal_gamma, best_result, best_score = select_optimal_config(
                    sweep_results,
                    embeddings_e,
                    attributions_a,
                )
                
                logger.info(f"Optimal config: beta={optimal_beta}, gamma={optimal_gamma}")
                logger.info(f"Best harmonic silhouette: {best_score:.4f}")
                
                summaries = summarize_sweep_results(
                    sweep_results, embeddings_e, attributions_a, labels
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
        
        if "error" in cluster_results:
            logger.warning("Skipping experiments due to clustering error")
        else:
            optimal_beta = cluster_results["optimal_beta"]
            optimal_gamma = cluster_results["optimal_gamma"]
            
            # Experiment 1: Clustering Quality
            from downstream.experiments.exp1_clustering_quality import run_experiment_1
            
            exp1_results = run_experiment_1(
                embeddings_e,
                attributions_a,
                labels=labels,
                output_dir=args.output_dir / "exp1",
                logger=logger,
            )
            
            # Experiment 2: Binary vs RD
            from downstream.experiments.exp2_binary_vs_rd import run_experiment_2
            
            exp2_results = run_experiment_2(
                embeddings_e,
                attributions_a,
                labels,
                optimal_beta=optimal_beta,
                optimal_gamma=optimal_gamma,
                output_dir=args.output_dir / "exp2",
                logger=logger,
            )
            
            # Experiment 4: Cross-Prefix Motif (uses POOLED attributions)
            from downstream.experiments.exp4_cross_prefix_motif import run_experiment_4
            
            # Build per-prompt data using tracked valid indices
            # Exp4 needs pooled attributions for cross-prefix comparison
            per_prompt_data = {}
            for result in filtered_results:
                q_idx = result.get("question_idx", len(per_prompt_data))
                valid_indices = per_question_valid_indices.get(q_idx, [])
                
                if len(valid_indices) >= 2:  # Need at least 2 samples to cluster
                    global_indices = [gi for gi, li in valid_indices]
                    local_indices = [li for gi, li in valid_indices]
                    
                    # Get valid continuations
                    conts = result.get("continuations", [])
                    valid_conts = [conts[li] for li in local_indices if li < len(conts)]
                    
                    per_prompt_data[f"q{q_idx}"] = {
                        "embeddings": embeddings_e[global_indices],
                        "attributions": attributions_a_pooled[global_indices],  # POOLED for Exp4
                        "continuations": valid_conts,
                    }
            
            exp4_results = run_experiment_4(
                per_prompt_data,
                optimal_beta,
                optimal_gamma,
                output_dir=args.output_dir / "exp4",
                logger=logger,
            )
            
            # Experiment 6: Hallucination Application
            from downstream.experiments.exp6_hallucination_application import run_experiment_6
            
            exp6_results = run_experiment_6(
                filtered_results,
                optimal_beta,
                optimal_gamma,
                output_dir=args.output_dir / "exp6",
                logger=logger,
            )
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

