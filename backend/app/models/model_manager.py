import gc
import os
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.config import settings
from app.schemas.system import SystemStatusResponse, ModelsStatusResponse, ModelStatusItem
from app.utils.logger import logger


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models: Dict[str, Any] = {}
            cls._instance.model_metadata: Dict[str, Dict[str, Any]] = {
                "geochat": {
                    "name": "GeoChat",
                    "task": "Remote-Sensing Visual-Language Reasoning",
                    "description": "Dedicated grounded remote-sensing Vision-Language Model for satellite scene interpretation, VQA, and domain visual observations",
                    "path": settings.GEOCHAT_MODEL_PATH
                },
                "lae_dino": {
                    "name": "LAE-DINO",
                    "task": "Open-Vocabulary Remote-Sensing Object Detection",
                    "description": "Locate Anything on Earth (LAE-DINO) open-vocabulary object detector trained on remote-sensing benchmark datasets",
                    "path": settings.LAE_DINO_MODEL_PATH
                },
                "mask2former": {
                    "name": "Mask2Former",
                    "task": "Universal Remote-Sensing Segmentation",
                    "description": "Masked-attention Mask Transformer for universal semantic segmentation and pixel-accurate land-cover masks",
                    "path": settings.MASK2FORMER_MODEL_PATH
                },
                "changemamba": {
                    "name": "ChangeMamba",
                    "task": "Bi-Temporal Change Detection",
                    "description": "Spatiotemporal Visual State Space (Mamba) architecture for pixel-level bi-temporal change probability mapping",
                    "path": settings.CHANGEMAMBA_MODEL_PATH
                }
            }
        return cls._instance

    def get_device(self) -> torch.device:
        """Return torch device (cuda if available, else cpu)"""
        if settings.FORCE_CPU_FALLBACK:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def get_device_name(self) -> str:
        dev = self.get_device()
        return "cuda" if dev.type == "cuda" else "cpu"

    def get_system_status(self) -> SystemStatusResponse:
        device_type = self.get_device_name()
        gpu_name = None
        vram_total = None
        vram_used = None
        vram_free = None

        if device_type == "cuda" and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                reserved_bytes = torch.cuda.memory_reserved(0)
                
                vram_total = round(total_bytes / (1024 ** 3), 2)
                vram_used = round(reserved_bytes / (1024 ** 3), 2)
                vram_free = round(max(0, (total_bytes - reserved_bytes) / (1024 ** 3)), 2)
            except Exception as e:
                logger.warning(f"Error querying CUDA properties: {e}")

        # CPU & System RAM
        vm = psutil.virtual_memory()
        cpu_p = psutil.cpu_percent(interval=None)

        loaded_keys = [k for k, v in self.models.items() if v is not None and getattr(v, "is_loaded", False)]

        # Check LoRA status
        lora_p = Path(settings.GEOCHAT_LORA_PATH)
        lora_available = lora_p.exists() and (lora_p / "adapter_config.json").exists()

        geochat_loaded = "geochat" in self.models and getattr(self.models["geochat"], "is_loaded", False)

        return SystemStatusResponse(
            online=True,
            device=device_type,
            gpu_name=gpu_name,
            vram_total_gb=vram_total,
            vram_used_gb=vram_used,
            vram_free_gb=vram_free,
            ram_percent=round(vm.percent, 1),
            cpu_percent=round(cpu_p, 1),
            active_models_count=len(loaded_keys),
            loaded_models=loaded_keys,
            geochat_model=settings.GEOCHAT_MODEL_PATH,
            geochat_status="ready" if geochat_loaded else "unloaded",
            lora_enabled=settings.GEOCHAT_USE_LORA,
            lora_path=settings.GEOCHAT_LORA_PATH,
            lora_available=lora_available,
            gemini_configured=bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip() and settings.GEMINI_API_KEY != "your_gemini_api_key_here"),
            gemini_model=settings.GEMINI_MODEL
        )

    def get_models_status(self) -> ModelsStatusResponse:
        items = []

        for key, meta in self.model_metadata.items():
            loaded = key in self.models and self.models[key] is not None and getattr(self.models[key], "is_loaded", False)
            vram_mb = 0.0
            if loaded and hasattr(self.models[key], "get_memory_mb"):
                vram_mb = self.models[key].get_memory_mb()

            items.append(ModelStatusItem(
                name=meta["name"],
                key=key,
                loaded=loaded,
                device=self.get_device_name() if loaded else "idle",
                vram_mb=round(vram_mb, 1),
                task=meta["task"],
                description=meta["description"],
                path=meta["path"],
                ready=True
            ))

        return ModelsStatusResponse(
            models=items,
            system_device=self.get_device_name(),
            cuda_available=torch.cuda.is_available()
        )

    def unload_model(self, model_key: str):
        """Unload a specific model to free VRAM"""
        if model_key in self.models and self.models[model_key] is not None:
            logger.info(f"Unloading model: {model_key}")
            if hasattr(self.models[model_key], "unload"):
                self.models[model_key].unload()
            del self.models[model_key]
            self.models[model_key] = None
            self.clear_vram_cache()

    def clear_vram_cache(self):
        """Run GC and empty PyTorch CUDA cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_geochat(self):
        from app.models.geochat_model import GeoChatModelWrapper
        if "geochat" not in self.models or self.models["geochat"] is None:
            self.models["geochat"] = GeoChatModelWrapper(self.get_device())
        return self.models["geochat"]

    def get_lae_dino(self):
        from app.models.lae_dino_model import LAEDINOModelWrapper
        if "lae_dino" not in self.models or self.models["lae_dino"] is None:
            self.models["lae_dino"] = LAEDINOModelWrapper(self.get_device())
        return self.models["lae_dino"]

    def get_mask2former(self):
        from app.models.mask2former_model import Mask2FormerModelWrapper
        if "mask2former" not in self.models or self.models["mask2former"] is None:
            self.models["mask2former"] = Mask2FormerModelWrapper(self.get_device())
        return self.models["mask2former"]

    def get_changemamba(self):
        from app.models.changemamba_model import ChangeMambaModelWrapper
        if "changemamba" not in self.models or self.models["changemamba"] is None:
            self.models["changemamba"] = ChangeMambaModelWrapper(self.get_device())
        return self.models["changemamba"]


model_manager = ModelManager()
