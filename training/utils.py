#!/usr/bin/env python3
"""
SATQUERY AI - GeoChat Training Utilities & Data Collator
Implements multimodal data collation, label masking for SFT, module inspection,
loss plotting, and experiment metadata tracking for GeoChat.
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def find_target_linear_modules(model) -> List[str]:
    """Inspect model architecture and identify linear module names for PEFT LoRA."""
    import torch.nn as nn
    linear_cls = (nn.Linear,)
    try:
        import bitsandbytes as bnb
        linear_cls = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        pass

    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            names = name.split(".")
            target_module_names.add(names[-1])

    # Filter out lm_head or embed_tokens if causal LM
    if "lm_head" in target_module_names:
        target_module_names.remove("lm_head")
    return sorted(list(target_module_names))

class GeoChatSFTDataCollator:
    """
    Multimodal Data Collator for GeoChat Supervised Fine-Tuning.
    Formats conversational satellite prompts into GeoChat conversation format
    and masks prompt tokens with -100 so loss is computed exclusively on target response tokens.
    """

    def __init__(self, tokenizer, image_processor, base_dir: Optional[Path] = None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.base_dir = base_dir or Path.cwd()

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_prompts = []
        user_prompts = []
        images = []

        for item in batch:
            img_ref = item.get("image", "")
            img_path = Path(img_ref) if Path(img_ref).is_absolute() else self.base_dir / img_ref
            
            if img_path.exists():
                pil_image = Image.open(img_path).convert("RGB")
            else:
                pil_image = Image.new("RGB", (336, 336), color=(30, 40, 50))

            question = item.get("question", "Describe this satellite image.")
            answer = item.get("answer", "")

            # GeoChat conversation format
            u_prompt = f"USER: <image>\n{question}\nASSISTANT:"
            f_prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}</s>"

            user_prompts.append(u_prompt)
            full_prompts.append(f_prompt)
            images.append(pil_image)

        # Image processing
        img_inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"]

        # Text tokenization
        text_inputs = self.tokenizer(
            full_prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # Construct labels: mask prompt tokens with -100
        labels = input_ids.clone()
        for i, u_prompt in enumerate(user_prompts):
            u_ids = self.tokenizer(u_prompt, add_special_tokens=False)["input_ids"]
            u_len = len(u_ids)
            if u_len < labels.shape[1]:
                labels[i, :u_len] = -100
            else:
                labels[i, :-1] = -100

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }

def plot_loss_curve(log_history: List[Dict[str, Any]], output_path: Path):
    """Generate loss vs steps/epochs plot from trainer log history."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_steps = []
        train_losses = []
        eval_steps = []
        eval_losses = []

        for entry in log_history:
            if "loss" in entry and "step" in entry:
                train_steps.append(entry["step"])
                train_losses.append(entry["loss"])
            if "eval_loss" in entry and "step" in entry:
                eval_steps.append(entry["step"])
                eval_losses.append(entry["eval_loss"])

        if not train_losses and not eval_losses:
            return

        plt.figure(figsize=(9, 5), dpi=150)
        plt.style.use("dark_background" if "dark_background" in plt.style.available else "default")
        
        if train_losses:
            plt.plot(train_steps, train_losses, label="Training Loss", color="#00ffcc", linewidth=2.0, marker="o", markersize=3)
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#ffaa00", linewidth=2.0, marker="s", markersize=4)

        plt.title("SatQuery GeoChat LoRA Fine-Tuning Loss Convergence", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Optimization Step", fontsize=10)
        plt.ylabel("Cross-Entropy Loss", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"[OK] Training loss curve saved to: {output_path}")
    except Exception as e:
        print(f"[WARN] Could not generate loss plot: {e}")

def save_training_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    trainable_params: int,
    total_params: int,
    train_loss: float,
    train_time_sec: float,
    device_name: str
):
    """Save experiment reproducibility metadata."""
    import transformers
    import peft
    import accelerate
    import datasets

    pct_trainable = (trainable_params / max(total_params, 1)) * 100.0

    metadata = {
        "model_name": config.get("model", {}).get("model_name_or_path", "MBZUAI/geochat-7B"),
        "peft_type": "LoRA",
        "lora_r": config.get("lora", {}).get("r", 16),
        "lora_alpha": config.get("lora", {}).get("lora_alpha", 32),
        "lora_dropout": config.get("lora", {}).get("lora_dropout", 0.05),
        "target_modules": config.get("lora", {}).get("target_modules", []),
        "learning_rate": config.get("training", {}).get("learning_rate", 2e-5),
        "epochs": config.get("training", {}).get("num_train_epochs", 3),
        "batch_size": config.get("training", {}).get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": config.get("training", {}).get("gradient_accumulation_steps", 4),
        "seed": config.get("training", {}).get("seed", 42),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": round(pct_trainable, 4),
        "final_train_loss": round(float(train_loss), 6) if train_loss is not None else None,
        "training_duration_seconds": round(float(train_time_sec), 2),
        "compute_device": device_name,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "accelerate_version": accelerate.__version__,
        "datasets_version": datasets.__version__
    }

    meta_file = output_dir / "training_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Training metadata saved to: {meta_file}")
    return metadata

def get_next_experiment_dir(base_dir: Path) -> Path:
    """Create next unique experiments directory: experiments/run_001, run_002, etc."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_nums = []
    for r in existing_runs:
        try:
            num = int(r.name.split("_")[1])
            run_nums.append(num)
        except (IndexError, ValueError):
            pass
    next_num = max(run_nums, default=0) + 1
    run_dir = base_dir / f"run_{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
