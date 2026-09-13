"""
SatQuery AI - Mask2Former Dedicated Service
Provides standard interface for Mask2Former universal remote-sensing segmentation.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
from PIL import Image

from app.models.model_manager import model_manager
from app.config import settings
from app.utils.logger import logger


class Mask2FormerService:
    """Service wrapper for Mask2Former universal segmentation."""

    def __init__(self):
        self.model_name = "Mask2Former"

    def load(self):
        """Ensure Mask2Former is loaded in memory."""
        model = model_manager.get_mask2former()
        if not model.is_loaded:
            model._load_model()

    def unload(self):
        """Unload Mask2Former from VRAM."""
        model = model_manager.get_mask2former()
        model.unload()

    def is_loaded(self) -> bool:
        """Check if Mask2Former is currently loaded."""
        model = model_manager.get_mask2former()
        return model.is_loaded

    def get_status(self) -> Dict[str, Any]:
        """Return model metadata and runtime status."""
        model = model_manager.get_mask2former()
        return {
            "model": "Mask2Former",
            "full_name": "Masked-attention Mask Transformer (Mask2Former)",
            "type": "segmentation",
            "segmentation_mode": "semantic_and_instance",
            "is_loaded": model.is_loaded,
            "device": str(model.device),
            "num_classes": len(model.id2label),
            "target_checkpoint": model.checkpoint_target,
            "load_error": model.load_error
        }

    def segment(
        self,
        image: np.ndarray | Image.Image
    ) -> Dict[str, Any]:
        """
        Execute Mask2Former semantic segmentation on satellite imagery.
        
        Parameters:
            image: RGB image array or PIL Image
            
        Returns:
            Structured results with semantic_mask, class_distributions, classes_present, and timing.
        """
        model = model_manager.get_mask2former()
        return model.segment_image(image)


mask2former_service = Mask2FormerService()
