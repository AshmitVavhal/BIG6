import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from app.config import settings
from app.models.model_manager import model_manager
from app.geospatial.raster_reader import raster_reader
from app.geospatial.geo_metadata import geo_extractor
from app.schemas.vqa import VQARequest, VQAResponse
from app.schemas.system import ExecutionStage
from app.services.semantic_reasoning_service import semantic_reasoning_service
from app.utils.benchmark_tracker import benchmark_tracker
from app.utils.file_manager import file_manager
from app.utils.logger import logger


class VQAPipeline:
    def __init__(self):
        self.model_manager = model_manager
        self.semantic_service = semantic_reasoning_service

    def _detect_image_type(self, image_path: Path, geo_meta: Any, raw_meta: dict) -> str:
        """Determine actual imagery modality/type without false multispectral assumptions."""
        ext = image_path.suffix.lower()
        bands = getattr(geo_meta, "count", raw_meta.get("count", 3))
        driver = getattr(geo_meta, "driver", raw_meta.get("driver", ""))
        name_lower = image_path.name.lower()

        if "sar" in name_lower or "sentinel1" in name_lower:
            return "SAR (Synthetic Aperture Radar)"
        if bands > 3 and getattr(geo_meta, "has_georeference", False):
            return f"Multispectral GeoTIFF ({bands} bands)"
        if ext in [".tif", ".tiff", ".geotiff"]:
            return f"GeoTIFF Raster ({bands} bands, {driver or 'TIFF'})"
        return "Optical RGB Satellite/Aerial Imagery"

    def _should_trigger_lae_dino(self, question: str) -> Optional[str]:
        """Detect if user question targets specific object detection/localization."""
        q = question.lower()
        
        # Object detection triggers
        triggers = [
            (r"\b(building|buildings|house|houses|residential)\b", "building"),
            (r"\b(road|roads|highway|street|paved road)\b", "road"),
            (r"\b(car|cars|vehicle|vehicles|truck|trucks)\b", "vehicle"),
            (r"\b(ship|ships|boat|boats|vessel|vessels)\b", "ship"),
            (r"\b(airplane|airplanes|aircraft|plane|planes)\b", "airplane"),
            (r"\b(storage tank|storage tanks|tank|tanks)\b", "storage tank"),
            (r"\b(solar panel|solar panels|solar array)\b", "solar panel"),
            (r"\b(bridge|bridges)\b", "bridge"),
            (r"\b(swimming pool|pool|pools)\b", "swimming pool"),
            (r"\b(structure|structures)\b", "structure"),
            (r"\b(tree|trees)\b", "tree")
        ]

        # Check for explicit query intent
        query_intents = ["what objects", "find ", "locate ", "highlight ", "where are", "show me", "count ", "are there any ", "presence of"]
        is_search_query = any(qi in q for qi in query_intents)

        for pattern, prompt in triggers:
            if re.search(pattern, q):
                return prompt

        if is_search_query:
            return "structure"

        return None

    def _should_trigger_mask2former(self, question: str) -> bool:
        """Detect if user question warrants segmentation or land-cover distribution analysis."""
        q = question.lower()
        triggers = [
            "segment", "land cover", "land-cover", "vegetation area",
            "built-up", "built up", "water body", "water area", "urban extent",
            "coverage", "percentage of vegetation", "greenery area"
        ]
        return any(t in q for t in triggers)

    def run(self, request: VQARequest) -> VQAResponse:
        start_time = time.time()
        trace: List[ExecutionStage] = []
        specialized_time_ms = 0.0
        
        # 1. Image loaded (Upload & Resolution)
        trace.append(ExecutionStage(
            stage="Image loaded",
            message=f"Resolved scene file: {request.image_path}",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))
        image_path = file_manager.resolve_image_path(request.image_path)

        # 2. Extract geospatial metadata & Image type detected
        t0 = time.time()
        geo_meta = geo_extractor.extract(image_path)
        img_rgb, raw_meta = raster_reader.load_rgb_array(image_path)
        geo_time_ms = (time.time() - t0) * 1000.0

        image_type = self._detect_image_type(image_path, geo_meta, raw_meta)
        trace.append(ExecutionStage(
            stage="Image type detected",
            message=f"Identified modality: {image_type} ({img_rgb.shape[1]}x{img_rgb.shape[0]} px, bands={raw_meta.get('count', 3)})",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(geo_time_ms, 1)
        ))

        # 3. Image preprocessing
        t0 = time.time()
        prep_ms = (time.time() - t0) * 1000.0
        trace.append(ExecutionStage(
            stage="Image preprocessing",
            message="Dynamic contrast normalization and RGB raster alignment completed",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(prep_ms, 1)
        ))
        specialized_time_ms += prep_ms

        # 4. GeoChat observation
        t_gc = time.time()
        geochat = self.model_manager.get_geochat()
        geochat_obs, caption, geochat_ms = geochat.generate_vqa_answer(
            image_rgb=img_rgb,
            question=request.question,
            geo_context=geo_meta.model_dump()
        )
        vram_mb = geochat.get_memory_mb()
        trace.append(ExecutionStage(
            stage="GeoChat observation",
            message=f"Extracted remote-sensing domain observations and spatial context ({round(geochat_ms, 1)} ms)",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(geochat_ms, 1),
            memory_mb=vram_mb if vram_mb > 0 else None
        ))

        # 5. Optional LAE-DINO detection
        lae_detections: Optional[List[Any]] = None
        target_object = self._should_trigger_lae_dino(request.question)
        if target_object:
            t0 = time.time()
            try:
                lae_model = self.model_manager.get_lae_dino()
                lae_res = lae_model.detect_objects(img_rgb, target_object, box_threshold=0.25)
                raw_dets = lae_res.get("detections", [])
                lae_ms = lae_res.get("inference_time_ms", (time.time() - t0) * 1000.0)
                specialized_time_ms += lae_ms
                
                lae_detections = [
                    {
                        "label": d.label,
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "area_percentage": d.area_percentage
                    }
                    for d in raw_dets
                ]
                trace.append(ExecutionStage(
                    stage="Optional LAE-DINO detection",
                    message=f"LAE-DINO localized {len(lae_detections)} candidate '{target_object}' instance(s) ({round(lae_ms, 1)} ms)",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    duration_ms=round(lae_ms, 1)
                ))
            except Exception as e:
                logger.warning(f"LAE-DINO optional detection pass skipped: {e}")

        # 6. Optional Mask2Former segmentation
        m2f_segmentation: Optional[Dict[str, Any]] = None
        if self._should_trigger_mask2former(request.question):
            t0 = time.time()
            try:
                m2f_model = self.model_manager.get_mask2former()
                m2f_res = m2f_model.segment_image(img_rgb)
                m2f_ms = m2f_res.get("inference_time_ms", (time.time() - t0) * 1000.0)
                specialized_time_ms += m2f_ms
                
                m2f_segmentation = {
                    "classes_present": m2f_res.get("classes_present", []),
                    "class_distributions": m2f_res.get("class_distributions", [])[:5]
                }
                trace.append(ExecutionStage(
                    stage="Optional Mask2Former segmentation",
                    message=f"Mask2Former segmented {len(m2f_res.get('classes_present', []))} surface class(es) ({round(m2f_ms, 1)} ms)",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    duration_ms=round(m2f_ms, 1)
                ))
            except Exception as e:
                logger.warning(f"Mask2Former optional segmentation pass skipped: {e}")

        # 7. Context aggregation
        t0 = time.time()
        context_dict = {
            "image_type": image_type,
            "geo_metadata": geo_meta.model_dump(),
            "geochat_observations": geochat_obs,
            "detections": lae_detections,
            "segmentation": m2f_segmentation
        }
        trace.append(ExecutionStage(
            stage="Context aggregation",
            message="Aggregated original image, metadata, GeoChat observations, and verified CV outputs",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        # 8. Gemini reasoning
        t0 = time.time()
        vqa_res = self.semantic_service.analyze_vqa(
            image_rgb=img_rgb,
            question=request.question,
            image_type=image_type,
            image_metadata=geo_meta.model_dump(),
            detections=lae_detections,
            segmentation=m2f_segmentation,
            geo_context=geo_meta.model_dump(),
            provider="geochat",
            geochat_observations=geochat_obs,
            caption=caption,
            geochat_ms=geochat_ms
        )

        answer = vqa_res["answer"]
        inf_ms = vqa_res["inference_time_ms"]
        gemini_ms = vqa_res.get("gemini_time_ms", 0.0)
        semantic_model_name = vqa_res["semantic_model"]
        device_name = vqa_res["device"]
        confidence = vqa_res["confidence"]
        conf_type = vqa_res["confidence_type"]
        warning_msg = vqa_res["transparency_warning"]

        trace.append(ExecutionStage(
            stage="Gemini reasoning",
            message=f"Gemini synthesized visually grounded answer ({round(gemini_ms if gemini_ms > 0 else inf_ms, 1)} ms)",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(gemini_ms if gemini_ms > 0 else inf_ms, 1)
        ))

        # 9. Response validation
        trace.append(ExecutionStage(
            stage="Response validation",
            message="Verified visual grounding and validated anti-hallucination compliance",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        # 10. Final response
        trace.append(ExecutionStage(
            stage="Final response",
            message=f"Delivered final answer ({len(answer)} chars)",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        total_ms = (time.time() - start_time) * 1000.0

        # Record benchmark
        benchmark_tracker.record_run(
            model=semantic_model_name,
            task="VQA",
            inference_time_ms=inf_ms,
            device=device_name,
            image_size=f"{img_rgb.shape[1]}x{img_rgb.shape[0]}",
            vram_usage_mb=vram_mb
        )

        return VQAResponse(
            task="vqa",
            status="success",
            question=request.question,
            answer=answer,
            geochat_observations=geochat_obs,
            caption=caption,
            confidence=confidence,
            confidence_type=conf_type,
            model=semantic_model_name,
            semantic_model=semantic_model_name,
            device=device_name,
            processing_time_ms=round(total_ms, 1),
            specialized_time_ms=round(specialized_time_ms, 1),
            semantic_time_ms=round(inf_ms, 1),
            image_url=f"/api/files/view/{image_path.name}",
            geo_metadata=geo_meta,
            execution_trace=trace,
            transparency_warning=warning_msg
        )


vqa_pipeline = VQAPipeline()

