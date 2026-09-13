"""
GeoChat Dedicated Service Layer
Provides clean interface for GeoChat remote-sensing VLM operations.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from app.config import settings
from app.models.model_manager import model_manager
from app.utils.logger import logger

class GeoChatService:
    """Dedicated service for GeoChat VLM lifecycle, reasoning, and hardware management."""

    def __init__(self):
        self._manager = model_manager

    def load(self):
        """Explicitly load GeoChat model weights."""
        logger.info("GeoChatService: Loading GeoChat model...")
        return self._manager.get_geochat()

    def unload(self):
        """Unload GeoChat model from VRAM."""
        logger.info("GeoChatService: Unloading GeoChat model...")
        self._manager.unload_model("geochat")

    def is_loaded(self) -> bool:
        """Check if GeoChat model is currently loaded."""
        if "geochat" in self._manager.models and self._manager.models["geochat"] is not None:
            return getattr(self._manager.models["geochat"], "is_loaded", False)
        return False

    def get_status(self) -> Dict[str, Any]:
        """Query GeoChat status, device, memory, and diagnostics."""
        loaded = self.is_loaded()
        dev = self._manager.get_device_name()
        vram_mb = 0.0
        if "geochat" in self._manager.models and self._manager.models["geochat"] is not None:
            vram_mb = self._manager.models["geochat"].get_memory_mb()
            diag = getattr(self._manager.models["geochat"], "hardware_info", {})
        else:
            diag = {}

        return {
            "model": "GeoChat",
            "model_path": settings.GEOCHAT_MODEL_PATH,
            "type": "Remote-Sensing Vision-Language Model",
            "status": "ready" if loaded else "unloaded",
            "loaded": loaded,
            "device": dev,
            "vram_mb": vram_mb,
            "diagnostics": diag
        }

    def analyze_image(
        self,
        image_rgb: np.ndarray,
        question: str,
        geo_context: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Run single-image satellite VQA inference."""
        geochat = self._manager.get_geochat()
        answer, caption, inf_ms = geochat.generate_vqa_answer(
            image_rgb=image_rgb,
            question=question,
            geo_context=geo_context
        )
        return {
            "answer": answer,
            "caption": caption,
            "inference_time_ms": inf_ms,
            "model": "GeoChat",
            "device": self._manager.get_device_name()
        }

    def analyze_multiple_images(
        self,
        images_rgb: List[np.ndarray],
        question: str
    ) -> Dict[str, Any]:
        """Run multi-image satellite inference."""
        if not images_rgb:
            raise ValueError("No images provided for analysis.")
        return self.analyze_image(images_rgb[0], question)

    def explain_analysis(
        self,
        image_rgb: np.ndarray,
        analysis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate semantic explanation using specialized CV context."""
        geochat = self._manager.get_geochat()
        task_type = analysis_context.get("task_type", "vqa")

        if task_type == "bi_temporal_change":
            chg_pct = analysis_context.get("change_percentage", 0.0)
            n_reg = analysis_context.get("num_regions", 0)
            regs = analysis_context.get("regions", [])
            t1 = analysis_context.get("t1_rgb", image_rgb)
            t2 = analysis_context.get("t2_rgb", image_rgb)
            notes = analysis_context.get("coregistration_notes")
            explanation, inf_ms = geochat.explain_bi_temporal_changes(
                change_percentage=chg_pct,
                num_regions=n_reg,
                regions=regs,
                t1_rgb=t1,
                t2_rgb=t2,
                coregistration_notes=notes
            )
            return {
                "answer": explanation,
                "model": "GeoChat",
                "inference_time_ms": inf_ms,
                "context_used": ["change_percentage", "num_regions", "regions"]
            }
        elif task_type == "highlight":
            prompt = analysis_context.get("prompt", "target features")
            n_det = analysis_context.get("num_detections", 0)
            area_pct = analysis_context.get("total_area_pct", 0.0)
            dets = analysis_context.get("detections", [])
            explanation, inf_ms = geochat.explain_highlight(
                prompt=prompt,
                num_detections=n_det,
                total_area_pct=area_pct,
                detections=dets,
                image_rgb=image_rgb
            )
            return {
                "answer": explanation,
                "model": "GeoChat",
                "inference_time_ms": inf_ms,
                "context_used": ["prompt", "num_detections", "detections"]
            }
        elif task_type == "optical_sar":
            opt_stats = analysis_context.get("optical_stats")
            sar_stats = analysis_context.get("sar_stats")
            q = analysis_context.get("question")
            explanation, inf_ms = geochat.explain_optical_sar_fusion(
                optical_stats=opt_stats,
                sar_stats=sar_stats,
                question=q
            )
            return {
                "answer": explanation,
                "model": "GeoChat",
                "inference_time_ms": inf_ms,
                "context_used": ["optical_stats", "sar_stats"]
            }
        else:
            q = analysis_context.get("question", "What is visible in this satellite image?")
            return self.analyze_image(image_rgb, q)

geochat_service = GeoChatService()
