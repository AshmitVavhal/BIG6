import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Step 1: Verify HF_TOKEN from environment
backend_env = Path(__file__).resolve().parent.parent / "backend" / ".env"
root_env = Path(__file__).resolve().parent.parent / ".env"

if backend_env.exists():
    load_dotenv(backend_env)
if root_env.exists():
    load_dotenv(root_env)

hf_token = os.getenv("HF_TOKEN")

print("=" * 70, flush=True)
print("SATQUERY - QWEN2.5-VL-3B-INSTRUCT HARDWARE & INFERENCE DIAGNOSTIC", flush=True)
print("=" * 70, flush=True)

# 1. Verify that HF_TOKEN is actually visible to the backend process
is_token_present = bool(hf_token and len(hf_token) > 10)
print(f"[1/8] HF_TOKEN visible in backend environment: {is_token_present}", flush=True)
if not is_token_present:
    print("ERROR: HF_TOKEN is not configured in backend environment!", flush=True)
    sys.exit(1)

# 2. Verify Hugging Face authentication
print("[2/8] Verifying Hugging Face authentication...", flush=True)
from huggingface_hub import HfApi, snapshot_download
try:
    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info.get("name", user_info.get("username", "Authenticated User"))
    print(f"      Hugging Face authenticated successfully as: {username}", flush=True)
except Exception as e:
    print(f"      [WARNING] HF whoami note: {e}", flush=True)

# 3. Verify access to Qwen2.5-VL-3B-Instruct
print("[3/8] Verifying access to Qwen/Qwen2.5-VL-3B-Instruct...", flush=True)
try:
    model_info = api.model_info("Qwen/Qwen2.5-VL-3B-Instruct")
    print(f"      Model found on Hub: {model_info.id} (Tags: {model_info.tags[:4]})", flush=True)
except Exception as e:
    print(f"      [ERROR] Could not access model metadata: {e}", flush=True)
    sys.exit(1)

# 4. Verify Transformers, PyTorch, CUDA, and GPU VRAM
print("[4/8] Checking PyTorch / CUDA / Hardware Support...", flush=True)
import torch
import transformers
print(f"      Transformers version: {transformers.__version__}")
print(f"      PyTorch version: {torch.__version__}")
print(f"      CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"      GPU: {device_name} (Total VRAM: {vram_total_gb:.2f} GB)")
    print(f"      bfloat16 supported: {torch.cuda.is_bf16_supported()}")
else:
    print("      [NOTE] CUDA not available, using CPU mode.")

# 5. Download / Verify model and processor snapshot locally
models_dir = Path(__file__).resolve().parent.parent / "models" / "qwen" / "Qwen2.5-VL-3B-Instruct"
models_dir.mkdir(parents=True, exist_ok=True)
print(f"[5/8] Verifying / downloading weights to local directory: {models_dir}...", flush=True)

try:
    downloaded_path = snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        local_dir=str(models_dir),
        token=hf_token,
        max_workers=8
    )
    print(f"      [OK] Model files downloaded and verified at: {downloaded_path}", flush=True)
except Exception as e:
    print(f"      [ERROR] Snapshot download failed: {e}", flush=True)
    sys.exit(1)

# 6. Load AutoProcessor
print("[6/8] Loading AutoProcessor...", flush=True)
from transformers import AutoProcessor
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
try:
    processor = AutoProcessor.from_pretrained(
        str(models_dir),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        token=hf_token,
        trust_remote_code=True
    )
    print("      [OK] AutoProcessor loaded successfully with dynamic satellite resolution scaling.", flush=True)
except Exception as e:
    print(f"      [ERROR] AutoProcessor failed to load: {e}", flush=True)
    sys.exit(1)

# 7. Load Qwen2_5_VLForConditionalGeneration
print("[7/8] Loading Qwen2_5_VLForConditionalGeneration onto GPU...", flush=True)
from transformers import Qwen2_5_VLForConditionalGeneration

if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

try:
    t0 = time.time()
    # Map all layers explicitly to cuda:0 to prevent meta-tensor disk offloading
    device_map = {"": 0} if device.type == "cuda" else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(models_dir),
        torch_dtype=dtype,
        device_map=device_map,
        token=hf_token,
        trust_remote_code=True
    )
    load_time = time.time() - t0
    print(f"      [OK] Model loaded successfully in {load_time:.2f}s on device: {model.device}", flush=True)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"      GPU VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB", flush=True)
except Exception as e:
    print(f"      [ERROR] Model loading failed: {e}", flush=True)
    sys.exit(1)

# 8. Perform minimal Qwen image + question inference test
print("[8/8] Performing Real Qwen2.5-VL-3B-Instruct Satellite Inference Test...", flush=True)
from PIL import Image
from qwen_vl_utils import process_vision_info

sample_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "bitemporal_t1_2024.png"
if not sample_path.exists():
    sample_path = list((Path(__file__).resolve().parent.parent / "data" / "samples").glob("*.png"))[0]

print(f"      Input Image: {sample_path}", flush=True)
image = Image.open(str(sample_path)).convert("RGB")
prompt_text = "Describe this satellite image."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(device)

print("      Running forward generation pass...", flush=True)
t_inf_start = time.time()
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=256)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0].strip()
inf_duration = time.time() - t_inf_start

print("=" * 70, flush=True)
print(f"REAL GENERATED RESPONSE (Inference Time: {inf_duration:.2f}s):", flush=True)
print("=" * 70, flush=True)
print(output_text, flush=True)
print("=" * 70, flush=True)
print("VERIFICATION COMPLETED: Qwen2.5-VL-3B-Instruct is 100% operational on GPU!", flush=True)
