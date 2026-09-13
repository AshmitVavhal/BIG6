"""
SatQuery AI - ChangeMamba Dedicated Service
Provides standard interface for ChangeMamba bi-temporal remote sensing change detection.
"""

from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from app.models.model_manager import model_manager
from app.config import settings
from app.utils.logger import logger


class ChangeMambaService:
    """Service wrapper for ChangeMamba bi-temporal change detection."""

    def __init__(self):
        self.model_name = "ChangeMamba"

    def load(self):
        """Ensure ChangeMamba is loaded in memory."""
        model = model_manager.get_changemamba()
        if not model.is_loaded:
            model._load_model()

    def unload(self):
        """Unload ChangeMamba from VRAM."""
        model = model_manager.get_changemamba()
        model.unload()

    def is_loaded(self) -> bool:
        """Check if ChangeMamba is currently loaded."""
        model = model_manager.get_changemamba()
        return model.is_loaded

    def get_status(self) -> Dict[str, Any]:
        """Return model metadata and runtime status."""
        model = model_manager.get_changemamba()
        return {
            "model": "ChangeMamba",
            "type": "change_detection",
            "is_loaded": model.is_loaded,
            "device": str(model.device),
            "threshold": model.threshold,
            "architecture": "Spatiotemporal Visual State Space / Selective Scan Mamba",
            "load_error": model.load_error
        }

    def detect_change(
        self,
        t1: np.ndarray | Image.Image,
        t2: np.ndarray | Image.Image,
        threshold: Optional[float] = None,
        valid_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute ChangeMamba change detection on T1 and T2 imagery.
        
        Parameters:
            t1: Pre-change RGB image array or PIL Image
            t2: Post-change RGB image array or PIL Image
            threshold: Optional threshold for change probability (defaults to settings.CHANGE_THRESHOLD)
            valid_mask: Optional valid pixel mask (excluding nodata/padding)
            
        Returns:
            Structured results with change_mask, change_probability, change_percentage, regions, and timing.
        """
        if isinstance(t1, Image.Image):
            t1 = np.array(t1.convert("RGB"))
        if isinstance(t2, Image.Image):
            t2 = np.array(t2.convert("RGB"))

        model = model_manager.get_changemamba()
        return model.detect_changes(
            t1_rgb=t1,
            t2_rgb=t2,
            threshold=threshold,
            valid_mask=valid_mask
        )


changemamba_service = ChangeMambaService()
