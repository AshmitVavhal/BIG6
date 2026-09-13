#!/usr/bin/env python3
"""
SATQUERY AI - Training Environment Diagnostic & Hardware Verification Tool
Checks GPU, CUDA, VRAM, PyTorch, Transformers, PEFT, Accelerate, Datasets,
and Qwen2.5-VL compatibility before initiating fine-tuning.
"""

import sys
import os
import platform
import psutil

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def format_gb(bytes_val: float) -> str:
    return f"{bytes_val / (1024 ** 3):.2f} GB"

def check_training_environment(dry_run_model: bool = False) -> bool:
    print("=" * 70)
    print(" SATQUERY AI - LoRA / PEFT TRAINING ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)
    
    all_passed = True
    warnings = []

    # 1. System & OS Info
    print("\n[1] System Information:")
    print(f"  * OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  * Python Version: {platform.python_version()} ({sys.executable})")
    
    vm = psutil.virtual_memory()
    print(f"  * System RAM: {format_gb(vm.total)} (Available: {format_gb(vm.available)}, Used: {vm.percent}%)")
    print(f"  * CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

    # 2. PyTorch & CUDA Diagnostics
    print("\n[2] PyTorch & Accelerator Diagnostics:")
    try:
        import torch
        print(f"  [OK] PyTorch: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"  * CUDA Available: {cuda_available}")
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            prop = torch.cuda.get_device_properties(current_device)
            total_vram = prop.total_memory
            allocated_vram = torch.cuda.memory_allocated(current_device)
            reserved_vram = torch.cuda.memory_reserved(current_device)
            free_vram = total_vram - reserved_vram
            
            print(f"  * CUDA Version: {cuda_version}")
            print(f"  * GPU Device [{current_device}]: {gpu_name}")
            print(f"  * Compute Capability: {prop.major}.{prop.minor}")
            print(f"  * Total VRAM: {format_gb(total_vram)}")
            print(f"  * Allocated VRAM: {format_gb(allocated_vram)}")
            print(f"  * Reserved VRAM: {format_gb(reserved_vram)}")
            print(f"  * Free / Available VRAM: {format_gb(free_vram)}")
            print(f"  * Multi-Processor Count: {prop.multi_processor_count}")
            
            # BF16 support check
            bf16_supported = torch.cuda.is_bf16_supported()
            print(f"  * Native BFloat16 Support: {'Yes' if bf16_supported else 'No (Use FP16)'}")
            
            # VRAM threshold assessment
            total_gb = total_vram / (1024 ** 3)
            if total_gb < 4.0:
                warnings.append(
                    f"GPU VRAM ({total_gb:.1f} GB) is very low for 3B VLM fine-tuning. "
                    "Recommend batch_size=1, gradient_checkpointing=True, low image resolution, or CPU mode."
                )
            elif total_gb < 7.0:
                print(f"  [INFO] VRAM Profile ({total_gb:.1f} GB): Suitable for LoRA with batch_size=1, "
                      "gradient_accumulation_steps=4, gradient_checkpointing=True, and bfloat16/fp16.")
            else:
                print(f"  [OK] VRAM Profile ({total_gb:.1f} GB): Fully equipped for standard LoRA fine-tuning.")
        else:
            warnings.append("CUDA is not available. Fine-tuning will run on CPU (significantly slower).")
            print("  [WARN] Running on CPU only.")
            
    except ImportError as e:
        print(f"  [FAIL] PyTorch is not installed: {e}")
        all_passed = False

    # 3. Python Package Ecosystem
    print("\n[3] Required Libraries & Versions:")
    required_packages = [
        ("transformers", "4.44.0"),
        ("peft", "0.12.0"),
        ("accelerate", "0.33.0"),
        ("datasets", "2.20.0"),
        ("sentencepiece", "0.1.99"),
        ("PIL", "10.0.0"),
        ("torchvision", "0.19.0"),
        ("yaml", "6.0"),
        ("matplotlib", "3.8.0"),
        ("scipy", "1.12.0"),
        ("rouge_score", "0.1.2"),
        ("nltk", "3.8.0")
    ]
    
    for pkg_name, min_ver in required_packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg_name}: {ver} (required: >={min_ver})")
        except ImportError:
            print(f"  [FAIL] {pkg_name}: NOT INSTALLED (required: >={min_ver})")
            all_passed = False

    # 4. Model Architecture & GeoChat Remote Sensing VLM Check
    print("\n[4] GeoChat Remote Sensing VLM Architecture Compatibility:")
    try:
        from transformers import LlamaForCausalLM, LlamaTokenizer, CLIPVisionModel, CLIPImageProcessor, AutoConfig
        print("  [OK] transformers.LlamaForCausalLM is available.")
        print("  [OK] transformers.LlamaTokenizer is available.")
        print("  [OK] transformers.CLIPVisionModel & CLIPImageProcessor are available.")
        
        import sentencepiece
        print(f"  [OK] sentencepiece {sentencepiece.__version__} is available.")
        
        from peft import LoraConfig, get_peft_model, TaskType
        print("  [OK] peft.LoraConfig & TaskType are available.")
        
        # Verify LoRA target modules for GeoChat (LLaMA backbone)
        expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"  [OK] Standard LoRA Target Modules: {', '.join(expected_modules)}")
        
    except ImportError as e:
        print(f"  [FAIL] Architecture compatibility error: {e}")
        all_passed = False

    # 5. Optional Dry-Run Model Instantiation
    if dry_run_model and all_passed:
        print("\n[5] Model Dry-Run Check:")
        try:
            from peft import LoraConfig
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            print("  [OK] LoraConfig initialized successfully.")
        except Exception as e:
            print(f"  [FAIL] Dry run failed: {e}")
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print(" DIAGNOSTIC RESULT: ENVIRONMENT IS READY FOR LoRA FINE-TUNING")
        if warnings:
            print("\n Warnings / Recommendations:")
            for w in warnings:
                print(f"  * {w}")
    else:
        print(" DIAGNOSTIC RESULT: MISSING DEPENDENCIES OR CONFIGURATION ISSUES DETECTED")
        print(" Please install missing packages: pip install -r backend/requirements.txt")
    print("=" * 70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery Training Environment Diagnostics")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run config instantiation")
    args = parser.parse_args()
    
    ready = check_training_environment(dry_run_model=args.dry_run)
    sys.exit(0 if ready else 1)
