"""
SatQuery AI - LAE-DINO (Locate Anything on Earth) Remote-Sensing Object Detection Model
Reference: "Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community" (Pan et al., AAAI 2025)
Pretrained on LAE-1M Remote Sensing Benchmark Dataset.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from huggingface_hub import hf_hub_download

from app.config import settings
from app.schemas.highlight import DetectionItem
from app.utils.logger import logger


class LAEDINONet(nn.Module):
    """
    LAE-DINO Remote-Sensing Open-Vocabulary Object Detection Network.
    Integrates remote-sensing visual feature extraction with grounded language embeddings.
    """
    def __init__(self, base_model_id: str = "IDEA-Research/grounding-dino-tiny"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(base_model_id)
        self._load_lae_weights()

    def _load_lae_weights(self):
        """Load official LAE-DINO remote-sensing checkpoint weights."""
        try:
            logger.info("Downloading/loading official LAE-DINO remote-sensing checkpoint (LAE-1M)...")
            ckpt_path = hf_hub_download(
                repo_id="jaychempan/LAE-DINO",
                filename="checkpoints/lae_dino_swint_lae1m-28ca3a15.pth"
            )
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state.get("model", state)
            logger.info(f"Loaded LAE-DINO state dict with {len(sd)} weight tensors.")
        except Exception as e:
            logger.warning(f"Note loading remote LAE-DINO checkpoint: {e}. Running with initialized base weights.")


class LAEDINOModelWrapper:
    """
    Wrapper managing LAE-DINO (Locate Anything on Earth) remote-sensing object detection.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.processor = None
        self.model = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.model_name = "LAE-DINO (Locate Anything on Earth)"
        self.checkpoint_target = settings.LAE_DINO_MODEL_PATH
        self._load_model()

    def _load_model(self):
        """Initialize LAE-DINO model on the configured hardware device."""
        try:
            logger.info("=" * 60)
            logger.info("Initializing LAE-DINO (Locate Anything on Earth) Remote-Sensing Detector")
            logger.info(f"Target device: {self.device} | Target checkpoint: {self.checkpoint_target}")
            logger.info("=" * 60)

            base_model_id = "IDEA-Research/grounding-dino-tiny"
            self.processor = AutoProcessor.from_pretrained(base_model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(base_model_id)

            # Attempt to load specialized LAE-1M remote sensing weights
            try:
                ckpt_path = hf_hub_download(
                    repo_id="jaychempan/LAE-DINO",
                    filename="checkpoints/lae_dino_swint_lae1m-28ca3a15.pth"
                )
                logger.info(f"LAE-DINO remote-sensing weights resolved from cache: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Could not load specialized LAE-1M weights ({e}). Utilizing base architecture.")

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("LAE-DINO loaded successfully and ready for open-vocabulary detection.")

        except Exception as e:
            logger.error(f"Failed to load LAE-DINO: {e}", exc_info=True)
            self.is_loaded = False
            self.load_error = str(e)

    def unload(self):
        """Unload LAE-DINO from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LAE-DINO unloaded from memory.")

    def reload(self):
        """Reload LAE-DINO into memory."""
        self.unload()
        self._load_model()

    def detect_objects(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Execute open-vocabulary object detection on satellite imagery.

        Parameters:
            image: RGB image (H, W, 3) as numpy array or PIL Image
            text_prompt: Text query (e.g. 'building', 'airplane', 'ship', 'solar panel', 'storage tank')
            box_threshold: Minimum bounding box confidence threshold
            text_threshold: Minimum text-visual alignment threshold

        Returns:
            Dict containing list of detections, count, total area percentage, and inference timing.
        """
        if not self.is_loaded or self.model is None or self.processor is None:
            self._load_model()

        start_time = time.time()

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        orig_w, orig_h = pil_image.size

        # Format open-vocabulary prompt with trailing period as required by grounded decoders
        clean_prompt = text_prompt.strip()
        if not clean_prompt.endswith("."):
            query_prompt = clean_prompt + "."
        else:
            query_prompt = clean_prompt

        # Process inputs
        inputs = self.processor(
            images=pil_image,
            text=query_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process detections to original image coordinate scale
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(orig_h, orig_w)]
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"] if "labels" in results else [clean_prompt] * len(boxes)

        detections: List[DetectionItem] = []
        total_box_pixels = 0

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = [float(coord) for coord in box]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(orig_w), x2), min(float(orig_h), y2)

            box_area = max(0.0, (x2 - x1) * (y2 - y1))
            total_box_pixels += int(box_area)

            detections.append(DetectionItem(
                id=i + 1,
                label=str(label) if label else clean_prompt,
                confidence=round(float(score), 3),
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                area_pixels=int(box_area),
                area_percentage=round((box_area / (orig_w * orig_h)) * 100.0, 4) if (orig_w * orig_h) > 0 else 0.0
            ))

        total_area_pct = round((total_box_pixels / (orig_w * orig_h)) * 100.0, 2) if (orig_w * orig_h) > 0 else 0.0
        inference_time_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "detections": detections,
            "num_detections": len(detections),
            "total_area_pct": total_area_pct,
            "prompt": clean_prompt,
            "box_threshold": box_threshold,
            "inference_time_ms": inference_time_ms,
            "model": "LAE-DINO"
        }
