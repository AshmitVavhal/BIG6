#!/usr/bin/env python3
"""
SATQUERY AI - Standalone Dataset Validator
Verifies JSONL formatting, image references, schema consistency, and quality constraints.
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

def validate_dataset_file(filepath: Path, base_dir: Path) -> bool:
    print(f"Validating dataset file: {filepath}")
    if not filepath.exists():
        print(f"  [FAIL] File does not exist: {filepath}")
        return False

    valid_count = 0
    errors = []

    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {idx}: Invalid JSON ({e})")
                continue

            # Check image
            img_ref = record.get("image")
            if not img_ref:
                errors.append(f"Line {idx}: Missing 'image' field")
                continue

            img_path = Path(img_ref) if Path(img_ref).is_absolute() else base_dir / img_ref
            if not img_path.exists():
                errors.append(f"Line {idx}: Image file not found -> {img_path}")
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                errors.append(f"Line {idx}: Corrupted image {img_path.name} -> {e}")
                continue

            # Check question & answer
            q = record.get("question", "")
            a = record.get("answer", "")
            if not q and "messages" not in record:
                errors.append(f"Line {idx}: Missing 'question' text")
                continue
            if not a and "messages" not in record:
                errors.append(f"Line {idx}: Missing 'answer' text")
                continue

            valid_count += 1

    if errors:
        print(f"  [FAIL] Found {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"    * {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more.")
        return False

    print(f"  [OK] Validated {valid_count} samples successfully. Zero errors.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SatQuery dataset JSONL files")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    ds_path = Path(args.dataset) if Path(args.dataset).is_absolute() else base_dir / args.dataset
    is_valid = validate_dataset_file(ds_path, base_dir)
    sys.exit(0 if is_valid else 1)
