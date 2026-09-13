#!/usr/bin/env python3
"""
SATQUERY AI - Supervised Fine-Tuning Pipeline for GeoChat-7B with PEFT + LoRA
Finetunes LoRA adapters on frozen GeoChat base weights for satellite and remote sensing VQA.
"""

import os
import sys
import time
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

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
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    CLIPImageProcessor,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import load_dataset
from app.models.geochat_model import GeoChatConfig, GeoChatLlamaForCausalLM

try:
    from training.utils import (
        GeoChatSFTDataCollator,
        plot_loss_curve,
        save_training_metadata,
        get_next_experiment_dir,
        find_target_linear_modules
    )
except ImportError:
    from utils import (
        GeoChatSFTDataCollator,
        plot_loss_curve,
        save_training_metadata,
        get_next_experiment_dir,
        find_target_linear_modules
    )

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_preflight_checks(config: Dict[str, Any]) -> str:
    """Validate hardware capabilities and compute device before training."""
    print("=" * 70)
    print(" SATQUERY AI - PRE-FLIGHT HARDWARE VERIFICATION")
    print("=" * 70)

    device_name = "cpu"
    if torch.cuda.is_available():
        dev_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(dev_idx)
        total_vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
        device_name = f"cuda:{dev_idx} ({gpu_name}, {total_vram_gb:.2f} GB)"
        print(f"  * Target GPU: {gpu_name}")
        print(f"  * Total VRAM: {total_vram_gb:.2f} GB")
        print(f"  * CUDA Version: {torch.version.cuda}")
        print(f"  * BF16 Supported: {torch.cuda.is_bf16_supported()}")

        if total_vram_gb < 6.0:
            print("  [WARN] VRAM < 6GB. Low memory configurations enabled automatically.")
    else:
        print("  [WARN] CUDA is unavailable. Running on CPU.")
    
    print("=" * 70 + "\n")
    return device_name

def train(args):
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else base_dir / args.config
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_yaml_config(config_path)

    # CLI Overrides
    if args.model_name_or_path:
        cfg["model"]["model_name_or_path"] = args.model_name_or_path
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.num_epochs is not None:
        cfg["training"]["num_train_epochs"] = args.num_epochs
    if args.batch_size is not None:
        cfg["training"]["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
    if args.lora_alpha is not None:
        cfg["lora"]["lora_alpha"] = args.lora_alpha
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    device_name = run_preflight_checks(cfg)

    # 1. Output & Experiment directories
    output_dir = Path(cfg["training"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_base = Path(cfg["training"].get("experiments_dir", "experiments"))
    if not exp_base.is_absolute():
        exp_base = base_dir / exp_base
    exp_dir = get_next_experiment_dir(exp_base)
    print(f"[OK] Experiment Run Directory: {exp_dir}")

    # 2. Load Tokenizer & Vision Processor
    model_id = cfg["model"]["model_name_or_path"]
    
    print(f"\n[1/6] Loading Tokenizer and Image Processor for {model_id}...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_id,
            use_fast=False,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer/processor for '{model_id}': {e}")
        sys.exit(1)

    # 3. Load Base Model in Frozen State
    print(f"\n[2/6] Loading GeoChat Base Model weights ({cfg['model'].get('torch_dtype', 'float16')})...")
    dtype_str = cfg["model"].get("torch_dtype", "float16")
    if dtype_str == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif dtype_str == "float16" and torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device_map = "auto" if torch.cuda.is_available() else None

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to instantiate GeoChat model: {e}")
        sys.exit(1)

    # Freeze base model parameters explicitly
    for param in base_model.parameters():
        param.requires_grad = False

    # 4. Configure & Apply PEFT LoRA
    print(f"\n[3/6] Configuring PEFT LoRA (r={cfg['lora']['r']}, alpha={cfg['lora']['lora_alpha']}, dropout={cfg['lora']['lora_dropout']})...")
    target_modules = cfg["lora"].get("target_modules")
    if not target_modules:
        target_modules = find_target_linear_modules(base_model)
        print(f"  * Auto-detected Linear Target Modules: {target_modules}")
    else:
        print(f"  * Specified Target Modules: {target_modules}")

    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"].get("bias", "none"),
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, peft_config)

    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable_params, all_params = model.get_nb_trainable_parameters()
    pct_trainable = 100 * trainable_params / all_params
    print(f"\n[PEFT PARAMETER VERIFICATION]")
    print(f"  * Base Model Parameters:      {all_params:,} (FROZEN)")
    print(f"  * Trainable LoRA Parameters:  {trainable_params:,} ({pct_trainable:.4f}%)")
    print(f"  * Memory Saved vs Full SFT:   ~{100 - pct_trainable:.2f}%\n")

    # 5. Load and Collate Datasets
    print(f"[4/6] Loading Multimodal Remote Sensing Dataset...")
    train_file = Path(cfg["dataset"]["train_path"])
    val_file = Path(cfg["dataset"]["validation_path"])
    
    if not train_file.is_absolute():
        train_file = base_dir / train_file
    if not val_file.is_absolute():
        val_file = base_dir / val_file

    if not train_file.exists():
        print(f"[ERROR] Training dataset file not found: {train_file}")
        print("Run `python scripts/prepare_dataset.py` to build dataset splits first.")
        sys.exit(1)

    train_ds = load_dataset("json", data_files=str(train_file), split="train")
    val_ds = load_dataset("json", data_files=str(val_file), split="train") if val_file.exists() else None

    print(f"  * Loaded Training Samples:   {len(train_ds)}")
    if val_ds:
        print(f"  * Loaded Validation Samples: {len(val_ds)}")

    collator = GeoChatSFTDataCollator(tokenizer=tokenizer, image_processor=image_processor, base_dir=base_dir)

    # 6. Training Arguments & Trainer Setup
    print(f"\n[5/6] Initializing Trainer with Supervised Multimodal Collator...")
    use_bf16 = (torch_dtype == torch.bfloat16)
    use_fp16 = (torch_dtype == torch.float16)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"].get("max_steps", -1),
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"].get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.05)),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "cosine"),
        logging_steps=cfg["training"].get("logging_steps", 5),
        eval_strategy="steps" if val_ds else "no",
        eval_steps=cfg["training"].get("eval_steps", 25) if val_ds else None,
        save_strategy="steps",
        save_steps=cfg["training"].get("save_steps", 25),
        save_total_limit=cfg["training"].get("save_total_limit", 2),
        fp16=use_fp16,
        bf16=use_bf16,
        seed=seed,
        dataloader_num_workers=cfg["training"].get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator
    )

    # 7. Execute Training Loop
    print(f"\n[6/6] Executing Supervised Fine-Tuning...")
    train_result = trainer.train()
    total_time = time.time() - start_time
    final_loss = train_result.training_loss

    print(f"\n[OK] Training Finished Successfully!")
    print(f"  * Total Duration: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"  * Final Training Loss: {final_loss:.6f}")

    # 8. Save LoRA Adapter & Artifacts
    print(f"\n[SAVING ADAPTER ARTIFACTS]")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  [OK] Saved LoRA adapter weights to: {output_dir}")

    # Save to experiment run directory
    model.save_pretrained(str(exp_dir / "adapter"))
    tokenizer.save_pretrained(str(exp_dir / "adapter"))
    
    # Save config copy
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # Save metadata
    meta = save_training_metadata(
        output_dir=output_dir,
        config=cfg,
        trainable_params=trainable_params,
        total_params=all_params,
        train_loss=final_loss,
        train_time_sec=total_time,
        device_name=device_name
    )
    with open(exp_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Generate and save loss plot
    loss_plot_path = output_dir / "loss_plot.png"
    plot_loss_curve(trainer.state.log_history, loss_plot_path)
    if loss_plot_path.exists():
        shutil.copy(loss_plot_path, exp_dir / "loss_plot.png")

    print("\n" + "=" * 70)
    print(" SATQUERY AI - LoRA FINE-TUNING PIPELINE COMPLETED")
    print(f" Adapter Directory: {output_dir}")
    print(f" Experiment Log:   {exp_dir}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery GeoChat LoRA Fine-Tuning")
    parser.add_argument("--config", default="training/config.yaml", help="Path to YAML configuration")
    parser.add_argument("--model_name_or_path", default=None, help="Override base model name or path")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Override LoRA alpha")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    args = parser.parse_args()

    train(args)
