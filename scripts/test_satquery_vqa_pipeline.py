import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure backend path is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.models.model_manager import model_manager
from app.services.semantic_reasoning_service import SemanticReasoningService
from app.pipelines.vqa_pipeline import VQAPipeline
from app.schemas.vqa import VQARequest

print("=" * 70, flush=True)
print("TESTING SATQUERY PIPELINE WITH QWEN2.5-VL-3B-INSTRUCT", flush=True)
print("=" * 70, flush=True)

# 1. Check System Status
status = model_manager.get_system_status()
print(f"System Device: {status.device} ({status.gpu_name})", flush=True)
print(f"GPU VRAM Total: {status.vram_total_gb} GB, Free: {status.vram_free_gb} GB", flush=True)

# 2. Initialize Qwen model wrapper
qwen = model_manager.get_qwen()
print(f"Qwen Model Loaded: {qwen.is_loaded}", flush=True)
print(f"Qwen Device: {qwen.device}", flush=True)

# 3. Test Direct VQA via SemanticReasoningService
img_path = BASE_DIR / "data" / "samples" / "bitemporal_t1_2024.png"
print(f"\nTesting VQA on sample image: {img_path.name}", flush=True)
img = Image.open(str(img_path)).convert("RGB")
img_np = np.array(img)

question = "Describe this satellite image."

service = SemanticReasoningService()
print(f"Running VQA with provider='qwen' and question: '{question}'...", flush=True)
res = service.analyze_vqa(
    image_rgb=img_np,
    question=question,
    provider="qwen"
)

print("\n" + "=" * 70, flush=True)
print(f"SATQUERY VQA PIPELINE RESULT ({res['semantic_model']}):", flush=True)
print("=" * 70, flush=True)
print(f"Answer: {res['answer']}")
print(f"Caption: {res.get('caption')}")
print(f"Inference Time: {res.get('inference_time_ms'):.1f} ms")
print(f"Device: {res.get('device')}")
print(f"Confidence: {res.get('confidence')}")
print("=" * 70, flush=True)
