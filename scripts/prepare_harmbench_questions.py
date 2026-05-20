#!/usr/bin/env -S uv run python
"""Pull walledai/HarmBench and emit pipeline-ready prompts as test_clozes.json.

Mirrors scripts/prepare_mmlu_questions.py for shape compatibility with Stage 2.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import save_json
from utils.logging_utils import setup_logger


def apply_chat_template(question: str, tokenizer) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_samples(
    rows: List[dict],
    tokenizer,
    prompt_col: str,
    category_col: str | None,
    id_col: str | None,
    source_split: str,
) -> list[dict]:
    samples = []
    for idx, row in enumerate(rows):
        text = str(row[prompt_col])
        category = str(row[category_col]) if category_col and category_col in row else "unknown"
        if id_col and id_col in row:
            original_id = str(row[id_col])
        else:
            original_id = f"harmbench_{idx:04d}"
        samples.append({
            "cloze_id": f"cloze_{idx:04d}",
            "group_id": f"group_{idx:04d}",
            "prefix": apply_chat_template(text, tokenizer),
            "target": "",
            "category": category,
            "subject": category,
            "original_id": original_id,
            "question": text,
            "cloze": text,
            "cloze_type": "main",
            "mode": "question",
            "choices": [],
            "answer_index": -1,
            "answer_letter": "",
            "answer_text": "",
            "source_dataset": "walledai/HarmBench",
            "source_split": source_split,
        })
    return samples


def main():
    p = argparse.ArgumentParser(description="Prepare walledai/HarmBench prompts for latent_planning")
    p.add_argument("--dataset-id", default="walledai/HarmBench")
    p.add_argument("--config-name", default="standard")
    p.add_argument("--split", default="train")
    p.add_argument("--model", required=True, help="HF model id for tokenizer / chat template")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=None, help="Subsample to N (default: all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt-col",
        default=None,
        help="Column holding the prompt text (auto-detect if omitted)",
    )
    p.add_argument(
        "--category-col",
        default=None,
        help="Column holding the category (auto-detect if omitted)",
    )
    p.add_argument(
        "--id-col",
        default=None,
        help="Column holding the behavior id (auto-detect if omitted)",
    )
    args = p.parse_args()

    log = setup_logger("prepare_harmbench")
    log.info("Loading %s/%s split=%s", args.dataset_id, args.config_name, args.split)
    ds = load_dataset(args.dataset_id, args.config_name, split=args.split)
    log.info("Loaded %d rows; columns=%s", len(ds), ds.column_names)

    # Auto-detect columns
    # walledai/HarmBench standard has: prompt, category (no id column)
    prompt_col = args.prompt_col or next(
        (c for c in ["prompt", "behavior", "Behavior"] if c in ds.column_names), None
    )
    if prompt_col is None:
        raise SystemExit(f"Cannot find prompt column in {ds.column_names}")

    category_col = args.category_col or next(
        (
            c
            for c in [
                "category",
                "FunctionalCategory",
                "functional_category",
                "SemanticCategory",
            ]
            if c in ds.column_names
        ),
        None,
    )

    id_col = args.id_col or next(
        (c for c in ["BehaviorID", "id"] if c in ds.column_names), None
    )

    log.info("prompt_col=%s  category_col=%s  id_col=%s", prompt_col, category_col, id_col)

    rows = list(ds)
    if args.n_samples is not None and args.n_samples < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.n_samples)
        log.info("Subsampled to %d rows (seed=%d)", len(rows), args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    samples = build_samples(rows, tok, prompt_col, category_col, id_col, args.split)

    payload = {
        "metadata": {
            "dataset_id": args.dataset_id,
            "config_name": args.config_name,
            "source_dir": str(args.output.parent),
            "split": args.split,
            "subjects": [args.config_name],
            "mode": "question",
            "prompt_style": "question_only",
            "total_groups": len(samples),
            "selected_groups": len(samples),
            "total_samples": len(samples),
            "random_seed": args.seed,
            "model": args.model,
        },
        "clozes": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output)
    log.info("Wrote %d samples to %s", len(samples), args.output)


if __name__ == "__main__":
    main()
