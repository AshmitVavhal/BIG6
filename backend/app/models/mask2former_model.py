"""
SatQuery AI - Mask2Former Universal Remote-Sensing Segmentation Model
Reference: "Masked-attention Mask Transformer for Universal Image Segmentation" (Cheng et al.)
Provides pixel-level semantic segmentation and structured mask extraction.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from app.config import settings
from app.utils.logger import logger


class Mask2FormerModelWrapper:
    """
    Wrapper managing Mask2Former universal segmentation on satellite imagery.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.processor: Optional[Mask2FormerImageProcessor] = None
        self.model: Optional[Mask2FormerForUniversalSegmentation] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.model_name = "Mask2Former (Universal Segmentation)"
        self.checkpoint_target = settings.MASK2FORMER_MODEL_PATH
        self.id2label: Dict[int, str] = {}
        self._load_model()

    def _load_model(self):
        """Initialize Mask2Former model on the configured hardware device."""
        try:
            logger.info("=" * 60)
            logger.info("Initializing Mask2Former Universal Segmentation Network")
            logger.info(f"Target device: {self.device} | Target checkpoint: {self.checkpoint_target}")
            logger.info("=" * 60)

            model_id = self.checkpoint_target
            self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)

            self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Mask2Former loaded successfully ({len(self.id2label)} classes). Ready for segmentation.")

        except Exception as e:
            logger.error(f"Failed to load Mask2Former: {e}", exc_info=True)
            self.is_loaded = False
            self.load_error = str(e)

    def unload(self):
        """Unload Mask2Former from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Mask2Former unloaded from memory.")

    def reload(self):
        """Reload Mask2Former into memory."""
        self.unload()
        self._load_model()

    def segment_image(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Execute full-scene semantic segmentation using Mask2Former.

        Parameters:
            image: RGB image (H, W, 3) as numpy array or PIL Image

        Returns:
            Dict containing semantic_mask, class_distributions, detected_classes, and timing.
        """
        if not self.is_loaded or self.model is None or self.processor is None:
            self._load_model()

        start_time = time.time()

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        orig_w, orig_h = pil_image.size

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process semantic segmentation to original image size
        semantic_maps = self.processor.post_process_semantic_segmentation(
            outputs=outputs,
            target_sizes=[(orig_h, orig_w)]
        )
        semantic_map = semantic_maps[0].cpu().numpy() # (H, W)

        # Calculate exact class area statistics
        unique_classes, counts = np.unique(semantic_map, return_counts=True)
        total_pixels = orig_h * orig_w
        class_distributions: List[Dict[str, Any]] = []

        for cls_id, cnt in zip(unique_classes, counts):
            label_name = self.id2label.get(int(cls_id), f"class_{cls_id}")
            pct = round((float(cnt) / total_pixels) * 100.0, 2)
            class_distributions.append({
                "class_id": int(cls_id),
                "label": label_name,
                "pixel_count": int(cnt),
                "area_percentage": pct
            })

        # Sort by area descending
        class_distributions.sort(key=lambda x: x["pixel_count"], reverse=True)
        inference_time_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "semantic_mask": semantic_map,
            "class_distributions": class_distributions,
            "classes_present": [c["label"] for c in class_distributions],
            "total_pixels": total_pixels,
            "inference_time_ms": inference_time_ms,
            "model": "Mask2Former",
            "segmentation_type": "semantic"
        }

    def refine_detections_to_masks(
        self,
        image_rgb: np.ndarray,
        boxes: List[List[float]],
        labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Derive precise polygon masks from localized bounding boxes and pixel contrast.
        """
        orig_h, orig_w = image_rgb.shape[:2]
        refined_items: List[Dict[str, Any]] = []

        # Run high-contrast grabcut / threshold refinement within each localized bounding box
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 2 or bh <= 2:
                continue

            crop = image_rgb[y1:y2, x1:x2]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

            # Adaptive Otsu thresholding within the localized ROI
            _, local_mask = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological smoothing
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, k)

            contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly_points: List[List[float]] = []
            if contours:
                # Get largest contour in ROI
                largest_c = max(contours, key=cv2.contourArea)
                # Translate back to image coordinates
                for pt in largest_c.squeeze():
                    if len(pt) == 2:
                        poly_points.append([float(pt[0] + x1), float(pt[1] + y1)])

            if not poly_points:
                # Fallback to rectangular polygon if contour too small
                poly_points = [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]

            refined_items.append({
                "id": i + 1,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "polygon": poly_points,
                "area_pixels": bw * bh
            })

        return refined_items
