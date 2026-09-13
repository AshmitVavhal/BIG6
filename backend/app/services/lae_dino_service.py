"""
SatQuery AI - LAE-DINO Dedicated Service
Provides standard interface for LAE-DINO (Locate Anything on Earth) open-vocabulary remote-sensing object detection.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
from PIL import Image

from app.models.model_manager import model_manager
from app.config import settings
from app.utils.logger import logger


class LAEDINOService:
    """Service wrapper for LAE-DINO open-vocabulary object detection."""

    def __init__(self):
        self.model_name = "LAE-DINO"

    def load(self):
        """Ensure LAE-DINO is loaded in memory."""
        model = model_manager.get_lae_dino()
        if not model.is_loaded:
            model._load_model()

    def unload(self):
        """Unload LAE-DINO from VRAM."""
        model = model_manager.get_lae_dino()
        model.unload()

    def is_loaded(self) -> bool:
        """Check if LAE-DINO is currently loaded."""
        model = model_manager.get_lae_dino()
        return model.is_loaded

    def get_status(self) -> Dict[str, Any]:
        """Return model metadata and runtime status."""
        model = model_manager.get_lae_dino()
        return {
            "model": "LAE-DINO",
            "full_name": "Locate Anything on Earth (LAE-DINO)",
            "type": "object_detection",
            "is_loaded": model.is_loaded,
            "device": str(model.device),
            "target_checkpoint": model.checkpoint_target,
            "open_vocabulary": True,
            "load_error": model.load_error
        }

    def detect(
        self,
        image: np.ndarray | Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Execute LAE-DINO text-prompted detection on satellite imagery.
        
        Parameters:
            image: RGB image array or PIL Image
            text_prompt: Open-vocabulary text query (e.g. 'building', 'airplane', 'ship', 'solar panel')
            box_threshold: Minimum bounding box confidence threshold
            text_threshold: Minimum text-visual alignment threshold
            
        Returns:
            Structured results with detections, bounding boxes, labels, count, and inference time.
        """
        model = model_manager.get_lae_dino()
        return model.detect_objects(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )


lae_dino_service = LAEDINOService()
