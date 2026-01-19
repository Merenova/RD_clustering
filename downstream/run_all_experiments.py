#!/usr/bin/env python3
"""Master Script to Run All Downstream Experiments.

Runs both Safety and Hallucination pipelines and generates summary report.

Usage:
    python run_all_experiments.py --mode all
    python run_all_experiments.py --mode safety
    python run_all_experiments.py --mode hallucination
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import setup_logger
from utils.data_utils import save_json, load_json

from downstream.config import OUTPUT_DIR


def run_safety_pipeline(output_dir: Path, logger, args):
    """Run the safety pipeline."""
    from downstream.run_safety_pipeline import main as safety_main
    
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING SAFETY PIPELINE")
    logger.info("=" * 60)
    
    # Modify sys.argv temporarily
    original_argv = sys.argv
    sys.argv = [
        "run_safety_pipeline.py",
        "--output-dir", str(output_dir / "safety"),
        "--n-prompts", str(args.n_prompts),
        "--n-continuations", str(args.n_continuations),
    ]
    
    if args.skip_rollout:
        sys.argv.append("--skip-rollout")
    if args.skip_embedding:
        sys.argv.append("--skip-embedding")
    if args.skip_clustering:
        sys.argv.append("--skip-clustering")
    
    try:
        safety_main()
        success = True
    except Exception as e:
        logger.error(f"Safety pipeline failed: {e}")
        success = False
    finally:
        sys.argv = original_argv
    
    return success


def run_hallucination_pipeline(output_dir: Path, logger, args):
    """Run the hallucination pipeline."""
    from downstream.run_hallucination_pipeline import main as hallu_main
    
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING HALLUCINATION PIPELINE")
    logger.info("=" * 60)
    
    # Modify sys.argv temporarily
    original_argv = sys.argv
    sys.argv = [
        "run_hallucination_pipeline.py",
        "--output-dir", str(output_dir / "hallucination"),
        "--n-questions", str(args.n_questions),
        "--n-continuations", str(args.n_continuations),
    ]
    
    if args.skip_rollout:
        sys.argv.append("--skip-rollout")
    if args.skip_embedding:
        sys.argv.append("--skip-embedding")
    if args.skip_clustering:
        sys.argv.append("--skip-clustering")
    
    try:
        hallu_main()
        success = True
    except Exception as e:
        logger.error(f"Hallucination pipeline failed: {e}")
        success = False
    finally:
        sys.argv = original_argv
    
    return success


def generate_summary_report(output_dir: Path, logger):
    """Generate summary report from all experiment results."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 60)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "pipelines": {},
    }
    
    # Safety results
    safety_dir = output_dir / "safety"
    if safety_dir.exists():
        safety_results = {}
        
        # Load available results
        for exp_name in ["exp1", "exp2", "exp5"]:
            results_file = safety_dir / exp_name / f"{exp_name}_results.json"
            if results_file.exists():
                safety_results[exp_name] = load_json(results_file)
        
        report["pipelines"]["safety"] = {
            "status": "completed" if safety_results else "no results",
            "experiments": list(safety_results.keys()),
            "summary": extract_safety_summary(safety_results),
        }
    
    # Hallucination results
    hallu_dir = output_dir / "hallucination"
    if hallu_dir.exists():
        hallu_results = {}
        
        for exp_name in ["exp1", "exp2", "exp4", "exp6"]:
            results_file = hallu_dir / exp_name / f"{exp_name}_results.json"
            if results_file.exists():
                hallu_results[exp_name] = load_json(results_file)
        
        report["pipelines"]["hallucination"] = {
            "status": "completed" if hallu_results else "no results",
            "experiments": list(hallu_results.keys()),
            "summary": extract_hallucination_summary(hallu_results),
        }
    
    # Save report
    save_json(report, output_dir / "summary_report.json")
    
    # Print summary
    logger.info("\n" + "-" * 40)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("-" * 40)
    
    for pipeline_name, pipeline_data in report["pipelines"].items():
        logger.info(f"\n{pipeline_name.upper()} PIPELINE:")
        logger.info(f"  Status: {pipeline_data['status']}")
        logger.info(f"  Experiments: {', '.join(pipeline_data['experiments'])}")
        
        if pipeline_data["summary"]:
            for key, value in pipeline_data["summary"].items():
                logger.info(f"  {key}: {value}")
    
    logger.info(f"\nFull report saved to {output_dir / 'summary_report.json'}")
    
    return report


def extract_safety_summary(results: dict) -> dict:
    """Extract key metrics from safety experiment results."""
    summary = {}
    
    if "exp1" in results:
        exp1 = results["exp1"]
        best = exp1.get("best_config", {})
        summary["optimal_beta"] = best.get("beta")
        summary["optimal_gamma"] = best.get("gamma")
        summary["n_clusters"] = best.get("n_clusters")
    
    if "exp2" in results:
        exp2 = results["exp2"]
        insights = exp2.get("insights", {})
        summary["rd_better_than_binary"] = insights.get("rd_better_silhouette", False)
        summary["finds_substructure"] = insights.get("rd_finds_substructure", False)
    
    if "exp5" in results:
        exp5 = results["exp5"]
        exp5_summary = exp5.get("summary", {})
        summary["n_safety_motifs"] = exp5.get("n_safety_motifs", 0)
        summary["motif_robustness_correlation"] = exp5_summary.get("motif_robustness_correlation", 0)
    
    return summary


def extract_hallucination_summary(results: dict) -> dict:
    """Extract key metrics from hallucination experiment results."""
    summary = {}
    
    if "exp1" in results:
        exp1 = results["exp1"]
        best = exp1.get("best_config", {})
        summary["optimal_beta"] = best.get("beta")
        summary["optimal_gamma"] = best.get("gamma")
        summary["n_clusters"] = best.get("n_clusters")
    
    if "exp2" in results:
        exp2 = results["exp2"]
        insights = exp2.get("insights", {})
        summary["rd_better_than_binary"] = insights.get("rd_better_silhouette", False)
    
    if "exp4" in results:
        exp4 = results["exp4"]
        motifs = exp4.get("global_motifs", [])
        if motifs:
            summary["n_global_motifs"] = len(motifs)
            summary["max_motif_prompts"] = max(m["n_prompts"] for m in motifs)
    
    if "exp6" in results:
        exp6 = results["exp6"]
        exp6_summary = exp6.get("summary", {})
        summary["pct_correct_but_brittle"] = exp6_summary.get("pct_correct_but_brittle", 0)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run All Downstream Experiments")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=["all", "safety", "hallucination"], default="all")
    parser.add_argument("--n-prompts", type=int, default=150, help="N prompts for safety")
    parser.add_argument("--n-questions", type=int, default=200, help="N questions for hallucination")
    parser.add_argument("--n-continuations", type=int, default=10)
    
    # Skip flags
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("all_experiments", args.output_dir / "run_all.log")
    
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("DOWNSTREAM MOTIF DISCOVERY: MASTER SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    
    results = {}
    
    # Run pipelines
    if args.mode in ["all", "safety"]:
        results["safety"] = run_safety_pipeline(args.output_dir, logger, args)
    
    if args.mode in ["all", "hallucination"]:
        results["hallucination"] = run_hallucination_pipeline(args.output_dir, logger, args)
    
    # Generate summary
    report = generate_summary_report(args.output_dir, logger)
    
    # Final summary
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    
    for pipeline, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {pipeline}: {status}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

