#!/usr/bin/env python3
"""
SATQUERY AI - Interactive Dataset Sample Builder
Validates and safely appends a verified satellite VQA sample to a dataset split.
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SUPPORTED_CATEGORIES = [
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
]

def add_sample(image_path_str: str, question: str, answer: str, category: str, target_file: str, source_tag: str = "") -> bool:
    base_dir = Path(__file__).resolve().parent.parent
    img_path = Path(image_path_str)
    if not img_path.is_absolute():
        img_path = base_dir / image_path_str

    if not img_path.exists():
        print(f"[FAIL] Image file not found: {img_path}")
        return False

    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            w, h = img.size
        print(f"[OK] Image verified: {img_path.name} ({w}x{h} px)")
    except Exception as e:
        print(f"[FAIL] Corrupted or unreadable image: {e}")
        return False

    q = question.strip()
    a = answer.strip()
    if not q or not a:
        print("[FAIL] Question and answer must both be non-empty strings.")
        return False

    cat = category.strip().lower()
    if cat not in SUPPORTED_CATEGORIES:
        print(f"[WARN] Category '{cat}' is not in standard categories list. Defaulting to 'scene_understanding'.")
        cat = "scene_understanding"

    target_path = Path(target_file)
    if not target_path.is_absolute():
        target_path = base_dir / target_file
    target_path.parent.mkdir(parents=True, exist_ok=True)

    rel_img_path = str(img_path.relative_to(base_dir)).replace("\\", "/") if img_path.is_relative_to(base_dir) else str(img_path).replace("\\", "/")

    sample_dict = {
        "image": rel_img_path,
        "question": q,
        "answer": a,
        "category": cat,
        "split_tag": img_path.stem,
        "source": source_tag or "manual_annotation"
    }

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Appended verified sample to {target_path}:")
    print(f"  * Category: {cat}")
    print(f"  * Question: {q}")
    print(f"  * Answer:   {a[:80]}...")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add verified remote sensing sample to SatQuery dataset")
    parser.add_argument("--image", required=True, help="Path to satellite image file")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--answer", required=True, help="Verified answer text")
    parser.add_argument("--category", default="scene_understanding", choices=SUPPORTED_CATEGORIES, help="Question category")
    parser.add_argument("--target", default="data/finetuning/train.jsonl", help="Target JSONL file")
    parser.add_argument("--source", default="analyst_verified", help="Source metadata note")
    args = parser.parse_args()

    success = add_sample(args.image, args.question, args.answer, args.category, args.target, args.source)
    sys.exit(0 if success else 1)
