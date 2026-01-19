"""Load and prepare HarmBench prompts for safety experiment.

Downloads HarmBench dataset and prepares prompts for both Base and Instruct models.
Adapted from Jailbreak_experiment/load_harmbench.py
"""

import json
import csv
import io
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.data_utils import save_json, load_json

# Import config
from downstream.config import (
    HARMBENCH_CACHE_DIR,
    BASE_MODEL_NAME,
    INSTRUCT_MODEL_NAME,
    SAFETY_CONFIG,
)

# HarmBench data URL (GitHub UI link; will be converted to raw when downloading)
HARMBENCH_URL = "https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"


def parse_csv_line(line: str) -> List[str]:
    """Parse a CSV line handling quoted fields with commas."""
    parts = []
    current = ""
    in_quotes = False
    
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            parts.append(current)
            current = ""
        else:
            current += char
    
    parts.append(current)  # Don't forget last field
    return parts


def download_harmbench(
    cache_dir: Path = None,
    force_download: bool = False,
    logger=None,
) -> List[Dict[str, str]]:
    """Download HarmBench dataset from GitHub.
    
    Args:
        cache_dir: Directory to cache downloaded data
        force_download: If True, re-download even if cached
        logger: Optional logger instance
        
    Returns:
        List of behavior dicts with 'behavior', 'category', 'id' keys
    """
    if cache_dir is None:
        cache_dir = HARMBENCH_CACHE_DIR
    
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / "harmbench_raw.json"
    
    if cache_file.exists() and not force_download:
        if logger:
            logger.info(f"Loading cached HarmBench data from {cache_file}")
        return load_json(cache_file)
    
    download_url = HARMBENCH_URL
    if "github.com" in download_url and "/blob/" in download_url:
        # Convert GitHub UI URL to raw content URL
        download_url = download_url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")

    if logger:
        logger.info(f"Downloading HarmBench from {download_url}")
    
    try:
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        csv_content = response.text
    except requests.RequestException as e:
        if logger:
            logger.error(f"Failed to download HarmBench: {e}")
        raise
    
    # Use Python's csv module for proper RFC 4180 parsing (handles multi-line quoted fields)
    # Expected columns: Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID
    # The CSV from GitHub might have spaces after commas in header, so we strip keys
    reader = csv.DictReader(io.StringIO(csv_content), skipinitialspace=True)
    
    behaviors = []
    for i, row in enumerate(reader, 1):
        # normalize keys (strip whitespace)
        row = {k.strip(): v for k, v in row.items() if k}
        
        behavior = row.get("Behavior", "").strip()
        if not behavior:
            continue
        
        functional_cat = row.get("FunctionalCategory", "").strip()
        semantic_cat = row.get("SemanticCategory", "").strip()
        behavior_id = row.get("BehaviorID", "").strip()
        tags = row.get("Tags", "").strip()
        context = row.get("ContextString", "").strip()
        
        behaviors.append({
            "id": behavior_id if behavior_id else f"harmbench_{i:04d}",
            "behavior": behavior,
            "functional_category": functional_cat,
            "semantic_category": semantic_cat,
            "tags": tags,
            "context": context,
            # Keep "category" as alias for functional_category for backward compat
            "category": functional_cat if functional_cat else semantic_cat,
        })
    
    if logger:
        logger.info(f"Parsed {len(behaviors)} behaviors from HarmBench")
    
    # Cache for future use
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_json(behaviors, cache_file)
    
    return behaviors


def apply_chat_template(
    behavior: str,
    tokenizer,
) -> str:
    """Apply chat template for instruct model.
    
    Args:
        behavior: The harmful behavior/prompt
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Formatted prompt with chat template
    """
    messages = [{"role": "user", "content": behavior}]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for Qwen
        )
    except TypeError:
        # Other models don't have enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    return text


def prepare_harmbench_prompts(
    behaviors: List[Dict[str, str]] = None,
    n_prompts: int = None,
    categories: Optional[List[str]] = None,
    seed: int = 42,
    logger=None,
) -> List[Dict[str, Any]]:
    """Prepare prompts for both Base and Instruct models.
    
    Args:
        behaviors: List of behavior dicts (will download if None)
        n_prompts: Number of prompts to sample
        categories: Optional list of categories to filter
        seed: Random seed
        logger: Logger instance
        
    Returns:
        List of prompt dicts with base_prompt, instruct_prompt, etc.
    """
    from transformers import AutoTokenizer
    
    if behaviors is None:
        behaviors = download_harmbench(logger=logger)
    
    if n_prompts is None:
        n_prompts = SAFETY_CONFIG["n_prompts"]
    
    if categories is None:
        categories = SAFETY_CONFIG.get("categories")
    
    random.seed(seed)
    
    # Filter by category if specified
    if categories and "all" not in categories:
        filtered = [b for b in behaviors if b["category"].lower() in [c.lower() for c in categories]]
        if logger:
            logger.info(f"Filtered to {len(filtered)} behaviors in categories: {categories}")
    else:
        filtered = behaviors
    
    # Sample
    if n_prompts >= len(filtered):
        selected = filtered
        if logger:
            logger.info(f"Using all {len(filtered)} available behaviors")
    else:
        selected = random.sample(filtered, n_prompts)
        if logger:
            logger.info(f"Sampled {n_prompts} behaviors")
    
    # Load tokenizer once for efficiency
    if logger:
        logger.info(f"Loading tokenizer for {INSTRUCT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_NAME, trust_remote_code=True)
    
    prompts = []
    for behavior in selected:
        # Base model: raw prompt (behavior text is the prompt)
        base_prompt = behavior["behavior"]
        
        # Instruct model: apply chat template
        instruct_prompt = apply_chat_template(behavior["behavior"], tokenizer)
        
        prompts.append({
            "prompt_id": behavior["id"],
            "behavior": behavior["behavior"],
            "functional_category": behavior.get("functional_category", ""),
            "semantic_category": behavior.get("semantic_category", ""),
            "category": behavior.get("category", ""),  # backward compat
            "tags": behavior.get("tags", ""),
            "context": behavior.get("context", ""),
            "base_prompt": base_prompt,
            "instruct_prompt": instruct_prompt,
        })
    
    # Log category distribution
    if logger:
        func_cat_count = {}
        sem_cat_count = {}
        for p in prompts:
            fc = p.get("functional_category", "") or "unknown"
            sc = p.get("semantic_category", "") or "unknown"
            func_cat_count[fc] = func_cat_count.get(fc, 0) + 1
            sem_cat_count[sc] = sem_cat_count.get(sc, 0) + 1
        
        logger.info("Functional category distribution:")
        for cat, count in sorted(func_cat_count.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {count}")
        
        logger.info("Semantic category distribution:")
        for cat, count in sorted(sem_cat_count.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {count}")
    
    return prompts


def save_harmbench_prompts(
    prompts: List[Dict[str, Any]],
    output_file: Path,
    logger=None,
):
    """Save prepared prompts to JSON file.
    
    Args:
        prompts: List of prompt dicts
        output_file: Output file path
        logger: Logger instance
    """
    output_data = {
        "metadata": {
            "source": "harmbench",
            "n_prompts": len(prompts),
            "base_model": BASE_MODEL_NAME,
            "instruct_model": INSTRUCT_MODEL_NAME,
        },
        "prompts": prompts
    }
    
    save_json(output_data, output_file)
    if logger:
        logger.info(f"Saved {len(prompts)} prompts to {output_file}")

