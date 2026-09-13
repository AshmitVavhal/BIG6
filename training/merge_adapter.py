#!/usr/bin/env python3
"""
SATQUERY AI - GeoChat LoRA Adapter Merger
Merges a trained PEFT LoRA adapter into base GeoChat weights for standalone deployment.
"""

import sys
import time
import argparse
from pathlib import Path

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

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel
from app.models.geochat_model import GeoChatConfig, GeoChatLlamaForCausalLM

def merge_lora_adapter(
    base_model_id: str,
    adapter_path_str: str,
    output_dir_str: str,
    torch_dtype: str = "float16"
):
    base_dir = Path(__file__).resolve().parent.parent
    adapter_path = Path(adapter_path_str) if Path(adapter_path_str).is_absolute() else base_dir / adapter_path_str
    output_dir = Path(output_dir_str) if Path(output_dir_str).is_absolute() else base_dir / output_dir_str

    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        print(f"[ERROR] LoRA adapter configuration not found at: {adapter_path}")
        sys.exit(1)

    print("=" * 70)
    print(" SATQUERY AI - GEOCHAT LoRA ADAPTER MERGE UTILITY")
    print("=" * 70)
    print(f"  * Base Model:     {base_model_id}")
    print(f"  * LoRA Adapter:   {adapter_path}")
    print(f"  * Output Merged:  {output_dir}")
    print("=" * 70 + "\n")

    t0 = time.time()
    dtype = torch.float16 if torch_dtype == "float16" and torch.cuda.is_available() else torch.float32

    # 1. Load Tokenizer
    print("[1/4] Loading Tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False)

    # 2. Load Base Model
    print(f"[2/4] Loading Base Model ({dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu",  # Merge in CPU memory safely
        trust_remote_code=True
    )

    # 3. Load LoRA Adapter & Merge
    print("[3/4] Attaching and Merging LoRA Weights into Base Model...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()
    print("  [OK] LoRA parameters merged into base architecture successfully.")

    # 4. Save Standalone Model
    print(f"[4/4] Saving Standalone Merged Model to: {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir), max_shard_size="4GB")
    tokenizer.save_pretrained(str(output_dir))

    duration = time.time() - t0
    print(f"\n[OK] Model successfully merged and saved in {duration:.1f}s.")
    print(f"Merged model path: {output_dir}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SatQuery LoRA adapter with Base GeoChat")
    parser.add_argument("--base_model", default="MBZUAI/geochat-7B", help="Base model identifier")
    parser.add_argument("--adapter", default="outputs/satquery-geochat-lora", help="Path to LoRA adapter")
    parser.add_argument("--output_dir", default="models/geochat_merged", help="Output directory for merged weights")
    parser.add_argument("--dtype", default="float16", help="Torch data type (float16 or float32)")
    args = parser.parse_args()

    merge_lora_adapter(
        base_model_id=args.base_model,
        adapter_path_str=args.adapter,
        output_dir_str=args.output_dir,
        torch_dtype=args.dtype
    )
