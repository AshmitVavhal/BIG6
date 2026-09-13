import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2
from PIL import Image

from app.config import settings
from app.models.model_manager import model_manager
from app.geospatial.raster_reader import raster_reader
from app.geospatial.geo_metadata import geo_extractor
from app.preprocessing.image_ops import image_ops
from app.schemas.highlight import HighlightRequest, HighlightResponse, DetectedRegion
from app.schemas.system import ExecutionStage
from app.services.semantic_reasoning_service import semantic_reasoning_service
from app.utils.benchmark_tracker import benchmark_tracker
from app.utils.file_manager import file_manager
from app.utils.logger import logger


class HighlightPipeline:
    def __init__(self):
        self.model_manager = model_manager
        self.semantic_service = semantic_reasoning_service

    def run(self, request: HighlightRequest) -> HighlightResponse:
        start_time = time.time()
        trace: List[ExecutionStage] = []

        # 1. Resolve image
        trace.append(ExecutionStage(
            stage="Upload & Resolution",
            message=f"Loading target satellite scene: {request.image_path}",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))
        image_path = file_manager.resolve_image_path(request.image_path)

        # 2. Extract metadata
        t0 = time.time()
        geo_meta = geo_extractor.extract(image_path)
        img_rgb, raw_meta = raster_reader.load_rgb_array(image_path)
        trace.append(ExecutionStage(
            stage="Geospatial Analysis",
            message=f"Loaded raster ({img_rgb.shape[1]}x{img_rgb.shape[0]} px, bands={raw_meta.get('count', 3)})",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        # 3. LAE-DINO Open-Vocabulary Remote-Sensing Object Detection (Authoritative Model)
        t0 = time.time()
        lae_dino = self.model_manager.get_lae_dino()
        trace.append(ExecutionStage(
            stage="Zero-Shot Detection",
            message=f"Executing LAE-DINO (Locate Anything on Earth) with text prompt: '{request.prompt}'",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        detection_res = lae_dino.detect_objects(
            img_rgb,
            request.prompt,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
        )
        raw_detections = detection_res["detections"]
        lae_ms = detection_res["inference_time_ms"]

        trace.append(ExecutionStage(
            stage="Bounding Box Extraction",
            message=f"LAE-DINO extracted {len(raw_detections)} candidate bounding boxes ({round(lae_ms, 1)} ms)",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(lae_ms, 1)
        ))

        # 4. Mask2Former Universal Segmentation Refinement (Authoritative Model)
        t0 = time.time()
        mask2former = self.model_manager.get_mask2former()
        trace.append(ExecutionStage(
            stage="Mask2Former Refinement",
            message="Refining localized detections into pixel-accurate segmentation polygon masks via Mask2Former",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        h, w = img_rgb.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        refined_detected_regions: List[DetectedRegion] = []
        m2f_ms = 0.0

        use_refinement = (
            request.use_mask2former_refinement 
            if request.use_mask2former_refinement is not None 
            else (request.use_sam2_refinement if request.use_sam2_refinement is not None else True)
        )

        if use_refinement and raw_detections:
            t_m2f = time.time()
            boxes_list = [d.bbox for d in raw_detections]
            labels_list = [d.label for d in raw_detections]
            refined_items = mask2former.refine_detections_to_masks(img_rgb, boxes_list, labels_list)
            m2f_ms = round((time.time() - t_m2f) * 1000.0, 1)

            for item, orig_d in zip(refined_items, raw_detections):
                poly = item.get("polygon")
                if poly:
                    cv2.fillPoly(full_mask, [np.array(poly, dtype=np.int32)], 255)
                else:
                    x1, y1, x2, y2 = [int(c) for c in item["bbox"]]
                    full_mask[y1:y2, x1:x2] = 255

                bx = item["bbox"]
                norm_box = [
                    round(bx[1] / h, 4) if h > 0 else 0.0,
                    round(bx[0] / w, 4) if w > 0 else 0.0,
                    round(bx[3] / h, 4) if h > 0 else 0.0,
                    round(bx[2] / w, 4) if w > 0 else 0.0
                ]

                refined_detected_regions.append(DetectedRegion(
                    id=item["id"],
                    label=item["label"],
                    confidence=orig_d.confidence,
                    confidence_type="model",
                    bounding_box=item["bbox"],
                    normalized_box=norm_box,
                    area_pixels=int(item.get("area_pixels", orig_d.area_pixels)),
                    mask_polygon=poly
                ))
        else:
            for d in raw_detections:
                x1, y1, x2, y2 = [int(c) for c in d.bbox]
                full_mask[y1:y2, x1:x2] = 255
                bx = d.bbox
                norm_box = [
                    round(bx[1] / h, 4) if h > 0 else 0.0,
                    round(bx[0] / w, 4) if w > 0 else 0.0,
                    round(bx[3] / h, 4) if h > 0 else 0.0,
                    round(bx[2] / w, 4) if w > 0 else 0.0
                ]
                refined_detected_regions.append(DetectedRegion(
                    id=d.id,
                    label=d.label,
                    confidence=d.confidence,
                    confidence_type="model",
                    bounding_box=d.bbox,
                    normalized_box=norm_box,
                    area_pixels=d.area_pixels,
                    mask_polygon=[[int(bx[0]), int(bx[1])], [int(bx[2]), int(bx[1])], [int(bx[2]), int(bx[3])], [int(bx[0]), int(bx[3])]]
                ))

        trace.append(ExecutionStage(
            stage="Mask Assembly",
            message=f"Mask2Former generated {len(refined_detected_regions)} boundary contour polygon masks",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(m2f_ms, 1)
        ))

        # 5. Render Output Images
        t0 = time.time()
        uid = uuid.uuid4().hex[:8]
        
        # Color overlay (emerald green alpha overlay)
        annotated_img = image_ops.apply_color_overlay(img_rgb, full_mask, color_rgb=(16, 185, 129), alpha=0.40)
        
        # Render bounding boxes and tags
        boxes = [d.bounding_box for d in refined_detected_regions]
        labels = [d.label for d in refined_detected_regions]
        confs = [d.confidence for d in refined_detected_regions]
        annotated_img = image_ops.draw_bounding_boxes(annotated_img, boxes, labels, confs, color=(16, 185, 129))

        annotated_filename = f"highlight_annotated_{uid}.png"
        mask_filename = f"highlight_mask_{uid}.png"
        orig_preview_filename = f"highlight_orig_{uid}.png"

        annotated_path = settings.OUTPUT_PATH / annotated_filename
        mask_path = settings.OUTPUT_PATH / mask_filename
        orig_path = settings.OUTPUT_PATH / orig_preview_filename

        Image.fromarray(annotated_img).save(annotated_path, format="PNG")
        Image.fromarray(full_mask).save(mask_path, format="PNG")
        Image.fromarray(img_rgb).save(orig_path, format="PNG")

        trace.append(ExecutionStage(
            stage="Rendering & Export",
            message="Exported annotated preview and binary mask layers",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        # Metrics
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        total_highlighted_pixels = int(np.sum(full_mask > 0))
        area_pct = round((total_highlighted_pixels / total_pixels) * 100.0, 2)

        # 6. Semantic Explanation via SemanticReasoningService (GeoChat + Gemini)
        trace.append(ExecutionStage(
            stage="GeoChat Observations",
            message="Generating remote-sensing spatial and geometric observations via GeoChat",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        sem_res = self.semantic_service.explain_highlight(
            prompt=request.prompt,
            num_detections=len(refined_detected_regions),
            total_area_pct=area_pct,
            detections=refined_detected_regions,
            image_rgb=img_rgb,
            provider="geochat"
        )

        semantic_summary = sem_res["summary"]
        sem_ms = sem_res["inference_time_ms"]
        geochat_ms = sem_res.get("geochat_time_ms", sem_ms)
        gemini_ms = sem_res.get("gemini_time_ms", 0.0)
        semantic_model_name = sem_res["semantic_model"]
        sem_device = sem_res["device"]

        if gemini_ms > 0:
            trace.append(ExecutionStage(
                stage="Gemini Reasoning",
                message=f"Gemini synthesized final explanation from LAE-DINO detections + Mask2Former masks + GeoChat ({round(gemini_ms, 1)} ms)",
                timestamp=datetime.now().strftime("%H:%M:%S"),
                duration_ms=round(gemini_ms, 1)
            ))
        else:
            trace.append(ExecutionStage(
                stage="Semantic Summary",
                message=f"Semantic explanation compiled by {semantic_model_name}",
                timestamp=datetime.now().strftime("%H:%M:%S"),
                duration_ms=round(sem_ms, 1)
            ))

        total_ms = (time.time() - start_time) * 1000.0
        specialized_time_ms = lae_ms + m2f_ms

        # Benchmark: Specialized model
        benchmark_tracker.record_run(
            model="LAE-DINO + Mask2Former",
            task="Object Highlighting",
            inference_time_ms=specialized_time_ms,
            device=self.model_manager.get_device_name(),
            image_size=f"{img_rgb.shape[1]}x{img_rgb.shape[0]}",
            vram_usage_mb=0.0
        )

        # Benchmark: Semantic Model
        benchmark_tracker.record_run(
            model=semantic_model_name,
            task="Highlight Semantic Explanation",
            inference_time_ms=sem_ms,
            device=sem_device,
            image_size=f"{img_rgb.shape[1]}x{img_rgb.shape[0]}",
            vram_usage_mb=0.0
        )

        return HighlightResponse(
            task="image_highlight",
            status="success",
            prompt=request.prompt,
            num_detections=len(refined_detected_regions),
            detections=refined_detected_regions,
            total_highlighted_area_pixels=total_highlighted_pixels,
            total_area_percentage=area_pct,
            annotated_image_url=f"/api/files/view/{annotated_filename}",
            mask_image_url=f"/api/files/view/{mask_filename}",
            original_image_url=f"/api/files/view/{orig_preview_filename}",
            model="LAE-DINO + Mask2Former",
            detection_model="LAE-DINO",
            segmentation_model="Mask2Former",
            semantic_model=semantic_model_name,
            device=self.model_manager.get_device_name(),
            processing_time_ms=round(total_ms, 1),
            specialized_time_ms=round(specialized_time_ms, 1),
            semantic_time_ms=round(sem_ms, 1),
            geo_metadata=geo_meta,
            execution_trace=trace,
            semantic_summary=semantic_summary
        )


highlight_pipeline = HighlightPipeline()
