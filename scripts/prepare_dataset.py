#!/usr/bin/env python3
"""
SATQUERY AI - Dataset Preparation, Validation, and Statistics Generator
Cleans, normalizes, validates, and deterministically splits remote-sensing datasets
for Qwen2.5-VL LoRA fine-tuning without data leakage.
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SUPPORTED_CATEGORIES = {
    "scene_understanding",
    "land_cover",
    "objects",
    "infrastructure",
    "agriculture",
    "water",
    "vegetation",
    "urban_areas",
    "transportation",
    "disaster_assessment",
    "environmental_analysis",
    "spatial_relationships",
    "comparative_reasoning",
    "remote_sensing_terminology"
}

def resolve_image_path(img_ref: str, base_dir: Path) -> Path:
    """Resolve relative or absolute image path."""
    p = Path(img_ref)
    if p.is_absolute() and p.exists():
        return p
    # Check relative to base_dir
    candidate = base_dir / img_ref
    if candidate.exists():
        return candidate
    # Check relative to current working directory
    candidate2 = Path.cwd() / img_ref
    if candidate2.exists():
        return candidate2
    return candidate

def validate_image(image_path: Path) -> Tuple[bool, Tuple[int, int], str]:
    """Verify that an image exists and can be opened by PIL."""
    if not image_path.exists():
        return False, (0, 0), "File does not exist"
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            size = img.size # (width, height)
        return True, size, "OK"
    except Exception as e:
        return False, (0, 0), f"Corrupted image: {e}"

def clean_and_validate_dataset(
    jsonl_paths: List[Path],
    base_dir: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load, deduplicate, filter, and validate dataset records."""
    valid_samples = []
    seen_hashes = set()
    stats = {
        "total_read": 0,
        "valid_count": 0,
        "corrupted_images": 0,
        "missing_images": 0,
        "empty_fields": 0,
        "duplicates": 0,
        "image_sizes": [],
        "categories": {},
        "question_lengths": [],
        "answer_lengths": []
    }

    for jsonl_file in jsonl_paths:
        if not jsonl_file.exists():
            print(f"[WARN] Input file not found: {jsonl_file}")
            continue
            
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                stats["total_read"] += 1
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["empty_fields"] += 1
                    continue

                # Extract image, question, answer
                image_ref = record.get("image", "")
                question = record.get("question", "")
                answer = record.get("answer", "")
                category = record.get("category", "scene_understanding").lower()
                split_tag = record.get("split_tag", "")

                # Handle messages format if present
                if "messages" in record and (not question or not answer):
                    try:
                        for msg in record["messages"]:
                            if msg.get("role") == "user":
                                for c in msg.get("content", []):
                                    if c.get("type") == "text":
                                        question = c.get("text", "")
                            elif msg.get("role") == "assistant":
                                for c in msg.get("content", []):
                                    if c.get("type") == "text":
                                        answer = c.get("text", "")
                    except Exception:
                        pass

                # Check empty fields
                if not image_ref or not question or not answer:
                    stats["empty_fields"] += 1
                    continue

                # Validate image
                img_path = resolve_image_path(image_ref, base_dir)
                is_valid_img, (w, h), err = validate_image(img_path)
                if not is_valid_img:
                    if "does not exist" in err:
                        stats["missing_images"] += 1
                    else:
                        stats["corrupted_images"] += 1
                    continue

                # Normalize category
                if category not in SUPPORTED_CATEGORIES:
                    category = "scene_understanding"

                # Deduplication check
                hash_key = hashlib.md5(f"{img_path.name}|{question.strip().lower()}".encode("utf-8")).hexdigest()
                if hash_key in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(hash_key)

                # Normalized record
                normalized = {
                    "image": str(img_path.relative_to(base_dir)).replace("\\", "/") if img_path.is_relative_to(base_dir) else str(img_path).replace("\\", "/"),
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "category": category,
                    "split_tag": split_tag or img_path.stem
                }
                valid_samples.append(normalized)
                
                # Record metrics
                stats["image_sizes"].append(f"{w}x{h}")
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                stats["question_lengths"].append(len(question.split()))
                stats["answer_lengths"].append(len(answer.split()))

    stats["valid_count"] = len(valid_samples)
    return valid_samples, stats

def split_dataset(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministic group-based splitting to prevent spatial/image leakage across splits.
    """
    random.seed(seed)
    
    # Group samples by split_tag (e.g., scene/image identity)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        tag = s.get("split_tag", "default")
        groups.setdefault(tag, []).append(s)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    train_samples = []
    val_samples = []
    test_samples = []

    # If small number of groups, distribute samples proportionally
    total_samples = len(samples)
    target_train = int(total_samples * train_ratio)
    target_val = int(total_samples * val_ratio)

    for g_key in group_keys:
        g_samples = groups[g_key]
        if len(train_samples) + len(g_samples) <= max(target_train, 1) or (not train_samples):
            train_samples.extend(g_samples)
        elif len(val_samples) + len(g_samples) <= max(target_val, 1) or (not val_samples):
            val_samples.extend(g_samples)
        else:
            test_samples.extend(g_samples)

    # Ensure all splits have at least 1 sample if dataset is >= 3
    if len(samples) >= 3:
        if not val_samples and len(train_samples) > 1:
            val_samples.append(train_samples.pop())
        if not test_samples and len(train_samples) > 1:
            test_samples.append(train_samples.pop())

    return train_samples, val_samples, test_samples

def print_dataset_statistics(
    dataset_name: str,
    stats: Dict[str, Any],
    train_count: int,
    val_count: int,
    test_count: int
):
    """Print authentic calculated dataset statistics."""
    print("=" * 70)
    print(f" SATQUERY AI - DATASET STATISTICS & INTEGRITY REPORT: {dataset_name}")
    print("=" * 70)
    print(f"\n[1] Overall Sample Counts (Genuine Computed Totals):")
    print(f"  * Total Records Ingested:    {stats['total_read']:,}")
    print(f"  * Valid Usable Samples:      {stats['valid_count']:,}")
    print(f"  * Training Split:            {train_count:,} ({(train_count / max(stats['valid_count'], 1) * 100):.1f}%)")
    print(f"  * Validation Split:          {val_count:,} ({(val_count / max(stats['valid_count'], 1) * 100):.1f}%)")
    print(f"  * Held-out Test Split:       {test_count:,} ({(test_count / max(stats['valid_count'], 1) * 100):.1f}%)")

    print(f"\n[2] Quality Filtering & Data Cleansing:")
    print(f"  * Corrupted Images Removed:  {stats['corrupted_images']}")
    print(f"  * Missing Images Removed:    {stats['missing_images']}")
    print(f"  * Empty Fields / Malformed:  {stats['empty_fields']}")
    print(f"  * Duplicate Pairs Pruned:    {stats['duplicates']}")

    if stats["question_lengths"]:
        avg_q = sum(stats["question_lengths"]) / len(stats["question_lengths"])
        avg_a = sum(stats["answer_lengths"]) / len(stats["answer_lengths"])
        print(f"\n[3] Text & Token Metrics:")
        print(f"  * Average Question Length:   {avg_q:.1f} words (min: {min(stats['question_lengths'])}, max: {max(stats['question_lengths'])})")
        print(f"  * Average Answer Length:     {avg_a:.1f} words (min: {min(stats['answer_lengths'])}, max: {max(stats['answer_lengths'])})")

    print(f"\n[4] Question Category Distribution:")
    for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        pct = (count / max(stats['valid_count'], 1)) * 100.0
        print(f"  * {cat:<28}: {count:>4} samples ({pct:>5.1f}%)")

    print("=" * 70 + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery Dataset Preparation & Statistics Tool")
    parser.add_argument("--inputs", nargs="+", default=["data/finetuning/sample_train.jsonl", "data/finetuning/sample_validation.jsonl", "data/finetuning/sample_test.jsonl"], help="Input JSONL files")
    parser.add_argument("--output_dir", default="data/finetuning", help="Output directory for train/val/test jsonl")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--stats_only", action="store_true", help="Only compute and print statistics without writing")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    input_paths = [Path(p) if Path(p).is_absolute() else base_dir / p for p in args.inputs]
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, stats = clean_and_validate_dataset(input_paths, base_dir)

    if not samples:
        print("[ERROR] No valid samples found across input files.")
        sys.exit(1)

    train_data, val_data, test_data = split_dataset(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print_dataset_statistics("SatQuery Multimodal Remote Sensing Dataset", stats, len(train_data), len(val_data), len(test_data))

    if not args.stats_only:
        train_out = out_dir / "train.jsonl"
        val_out = out_dir / "validation.jsonl"
        test_out = out_dir / "test.jsonl"

        for p_out, data_split in [(train_out, train_data), (val_out, val_data), (test_out, test_data)]:
            with open(p_out, "w", encoding="utf-8") as f:
                for item in data_split:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[OK] Saved {len(data_split)} records to: {p_out}")

if __name__ == "__main__":
    main()
