#!/usr/bin/env python3
"""
SATQUERY AI - GeoChat Vision-Language Inference CLI
Allows running interactive or one-off VQA queries using GeoChat or SatQuery LoRA adapters.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root and backend dir are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
for p in [str(BASE_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from app.models.geochat_model import GeoChatModelWrapper
from app.config import settings

def run_inference(
    image_path_str: str,
    question: str,
    model_id: str = "MBZUAI/geochat-7B",
    adapter_path_str: Optional[str] = None,
    max_new_tokens: int = 256
):
    base_dir = Path(__file__).resolve().parent.parent
    img_path = Path(image_path_str) if Path(image_path_str).is_absolute() else base_dir / image_path_str

    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    print("=" * 70)
    print(" SATQUERY AI - GEOCHAT SATELLITE INFERENCE ENGINE")
    print("=" * 70)

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[1/2] Initializing GeoChat Model Wrapper on {device}...")
    wrapper = GeoChatModelWrapper(device=device)

    # Run inference
    print("\n" + "-" * 70)
    print(f" Image:     {img_path.name}")
    print(f" Question:  {question}")
    print(f" Engine:    GeoChat (Remote-Sensing VLM)")
    print("-" * 70)

    pil_img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(pil_img)

    answer, caption, latency_ms = wrapper.generate_vqa_answer(
        image_rgb=img_rgb,
        question=question
    )

    print(f"\n ANSWER:\n{answer}\n")
    print("-" * 70)
    print(f" Inference Latency: {latency_ms:.1f} ms | Device: {device}")
    print("=" * 70 + "\n")
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery GeoChat Inference")
    parser.add_argument("--image", required=True, help="Path to satellite image")
    parser.add_argument("--question", required=True, help="VQA question text")
    parser.add_argument("--model", default="MBZUAI/geochat-7B", help="Base model identifier")
    parser.add_argument("--adapter", default="outputs/satquery-geochat-lora", help="Path to LoRA adapter")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens")
    args = parser.parse_args()

    run_inference(
        image_path_str=args.image,
        question=args.question,
        model_id=args.model,
        adapter_path_str=args.adapter,
        max_new_tokens=args.max_new_tokens
    )
