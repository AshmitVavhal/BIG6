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
from app.preprocessing.coregistration import coregistration_engine
from app.preprocessing.image_ops import image_ops
from app.schemas.change import ChangeDetectionRequest, ChangeDetectionResponse, ChangedRegion
from app.schemas.system import ExecutionStage
from app.services.semantic_reasoning_service import semantic_reasoning_service
from app.utils.benchmark_tracker import benchmark_tracker
from app.utils.file_manager import file_manager
from app.utils.logger import logger


class ChangePipeline:
    def __init__(self):
        self.model_manager = model_manager
        self.semantic_service = semantic_reasoning_service

    def run(self, request: ChangeDetectionRequest) -> ChangeDetectionResponse:
        start_time = time.time()
        trace: List[ExecutionStage] = []

        # 1. Resolve T1 and T2 images
        trace.append(ExecutionStage(
            stage="Upload & Resolution",
            message=f"Resolving bi-temporal inputs: T1={request.t1_image_path}, T2={request.t2_image_path}",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))
        t1_path = file_manager.resolve_image_path(request.t1_image_path)
        t2_path = file_manager.resolve_image_path(request.t2_image_path)

        # 2. Extract Geospatial Metadata & Check Dimensions
        t0 = time.time()
        geo_meta_t1 = geo_extractor.extract(t1_path)
        geo_meta_t2 = geo_extractor.extract(t2_path)
        t1_rgb, meta1 = raster_reader.load_rgb_array(t1_path)
        t2_rgb, meta2 = raster_reader.load_rgb_array(t2_path)
        trace.append(ExecutionStage(
            stage="Geospatial Analysis",
            message=f"T1 ({t1_rgb.shape[1]}x{t1_rgb.shape[0]} px) | T2 ({t2_rgb.shape[1]}x{t2_rgb.shape[0]} px)",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        # 3. Spatial Co-Registration, Reprojection & Alignment
        t0 = time.time()
        t1_aligned, t2_aligned, is_aligned, coreg_notes = coregistration_engine.align_and_validate(
            t1_rgb, t2_rgb, meta1, meta2
        )
        trace.append(ExecutionStage(
            stage="Spatial Co-Registration",
            message=f"Alignment status: {'VALIDATED' if is_aligned else 'CORRECTED'} ({coreg_notes})",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        # 4. ChangeMamba Inference (Authoritative Spatiotemporal Change Detection Model)
        t0 = time.time()
        changemamba = self.model_manager.get_changemamba()
        trace.append(ExecutionStage(
            stage="Model Orchestration",
            message=f"Executing ChangeMamba Spatiotemporal State Space Model on {self.model_manager.get_device_name().upper()}",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        thresh = request.threshold if request.threshold is not None else settings.CHANGE_THRESHOLD
        cm_results = changemamba.detect_changes(
            t1_aligned,
            t2_aligned,
            threshold=thresh,
            min_region_area=request.min_region_area
        )
        mask = cm_results["change_mask"]
        change_pct = cm_results["change_percentage"]
        num_regions = cm_results["num_regions"]
        regions = cm_results["regions"]
        cm_inf_ms = cm_results["inference_time_ms"]

        trace.append(ExecutionStage(
            stage="Change Detection Inference",
            message=f"ChangeMamba complete: {change_pct}% changed area across {num_regions} connected regions (threshold={thresh})",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(cm_inf_ms, 1)
        ))

        # 5. Connected Component Analysis
        trace.append(ExecutionStage(
            stage="Connected Components Extraction",
            message=f"Extracted {num_regions} spatial bounding boxes and pixel cluster statistics",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        # 6. Semantic Reasoning via SemanticReasoningService (GeoChat + Gemini)
        semantic_text = ""
        structured_analysis = None
        sem_inf_ms = 0.0
        semantic_model_name = "GeoChat"
        sem_device = self.model_manager.get_device_name()
        warning_msg = None

        if request.enable_semantic_reasoning:
            trace.append(ExecutionStage(
                stage="GeoChat Observations",
                message="Generating remote-sensing bi-temporal visual observations via GeoChat",
                timestamp=datetime.now().strftime("%H:%M:%S")
            ))

            sem_res = self.semantic_service.explain_bi_temporal(
                change_percentage=change_pct,
                num_regions=num_regions,
                regions=regions,
                t1_rgb=t1_aligned,
                t2_rgb=t2_aligned,
                coregistration_notes=coreg_notes,
                provider="geochat"
            )

            semantic_text = sem_res["explanation"]
            structured_analysis = sem_res.get("structured_analysis")
            sem_inf_ms = sem_res["inference_time_ms"]
            geochat_ms = sem_res.get("geochat_time_ms", sem_inf_ms)
            gemini_ms = sem_res.get("gemini_time_ms", 0.0)
            semantic_model_name = sem_res["semantic_model"]
            sem_device = sem_res["device"]
            warning_msg = sem_res.get("transparency_warning")

            if gemini_ms > 0:
                trace.append(ExecutionStage(
                    stage="Gemini Multimodal Reasoning",
                    message=f"Gemini synthesized final explanation from T1/T2 imagery + ChangeMamba ({change_pct:.2f}%) + GeoChat ({round(gemini_ms, 1)} ms)",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    duration_ms=round(gemini_ms, 1)
                ))
            else:
                trace.append(ExecutionStage(
                    stage="Semantic Reasoning",
                    message=f"{semantic_model_name} generated natural-language change interpretation",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    duration_ms=round(sem_inf_ms, 1)
                ))

        # 7. Render & Save Visual Layers
        t0 = time.time()
        uid = uuid.uuid4().hex[:8]

        # Change Overlay (technical amber/orange overlay on T2)
        overlay_img = image_ops.apply_color_overlay(t2_aligned, mask, color_rgb=(245, 158, 11), alpha=0.50)
        
        # Side by side composite
        h, w = t1_aligned.shape[:2]
        sbs_img = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        sbs_img[:, :w] = t1_aligned
        sbs_img[:, w+10:] = overlay_img

        mask_filename = f"change_mask_{uid}.png"
        overlay_filename = f"change_overlay_{uid}.png"
        sbs_filename = f"change_sbs_{uid}.png"
        t1_filename = f"change_t1_{uid}.png"
        t2_filename = f"change_t2_{uid}.png"

        Image.fromarray(mask).save(settings.OUTPUT_PATH / mask_filename, format="PNG")
        Image.fromarray(overlay_img).save(settings.OUTPUT_PATH / overlay_filename, format="PNG")
        Image.fromarray(sbs_img).save(settings.OUTPUT_PATH / sbs_filename, format="PNG")
        Image.fromarray(t1_aligned).save(settings.OUTPUT_PATH / t1_filename, format="PNG")
        Image.fromarray(t2_aligned).save(settings.OUTPUT_PATH / t2_filename, format="PNG")

        trace.append(ExecutionStage(
            stage="Rendering & Export",
            message="Exported mask, colored overlay, and dual-date views",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        total_ms = (time.time() - start_time) * 1000.0

        # Benchmark: ChangeMamba
        benchmark_tracker.record_run(
            model="ChangeMamba",
            task="Bi-Temporal Change Detection",
            inference_time_ms=cm_inf_ms,
            device=self.model_manager.get_device_name(),
            image_size=f"{w}x{h}",
            vram_usage_mb=0.0
        )

        # Benchmark: Semantic Model
        if request.enable_semantic_reasoning:
            benchmark_tracker.record_run(
                model=semantic_model_name,
                task="Bi-Temporal Semantic Reasoning",
                inference_time_ms=sem_inf_ms,
                device=sem_device,
                image_size=f"{w}x{h}",
                vram_usage_mb=0.0
            )

        return ChangeDetectionResponse(
            task="bi_temporal_change",
            status="success",
            change_percentage=change_pct,
            num_regions=num_regions,
            total_changed_pixels=cm_results["changed_pixels"],
            total_pixels=cm_results["valid_pixels"],
            regions=regions,
            t1_image_url=f"/api/files/view/{t1_filename}",
            t2_image_url=f"/api/files/view/{t2_filename}",
            mask_url=f"/api/files/view/{mask_filename}",
            overlay_url=f"/api/files/view/{overlay_filename}",
            side_by_side_url=f"/api/files/view/{sbs_filename}",
            model="ChangeMamba",
            detection_model="ChangeMamba",
            semantic_model=semantic_model_name,
            device=self.model_manager.get_device_name(),
            processing_time_ms=round(total_ms, 1),
            specialized_time_ms=round(cm_inf_ms, 1),
            semantic_time_ms=round(sem_inf_ms, 1),
            is_coregistered=is_aligned,
            coregistration_notes=coreg_notes,
            geo_metadata_t1=geo_meta_t1,
            geo_metadata_t2=geo_meta_t2,
            semantic_analysis=semantic_text,
            structured_analysis=structured_analysis,
            execution_trace=trace,
            transparency_warning=warning_msg
        )



change_pipeline = ChangePipeline()
