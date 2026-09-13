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
from app.preprocessing.sar_ops import sar_processor
from app.schemas.optical_sar import OpticalSARRequest, OpticalSARResponse
from app.schemas.system import ExecutionStage
from app.services.semantic_reasoning_service import semantic_reasoning_service
from app.utils.benchmark_tracker import benchmark_tracker
from app.utils.file_manager import file_manager
from app.utils.logger import logger

class OpticalSARPipeline:
    def __init__(self):
        self.model_manager = model_manager
        self.semantic_service = semantic_reasoning_service

    def run(self, request: OpticalSARRequest) -> OpticalSARResponse:
        start_time = time.time()
        trace: List[ExecutionStage] = []

        # 1. Resolve inputs
        trace.append(ExecutionStage(
            stage="Upload & Resolution",
            message=f"Loading Optical ({request.optical_image_path}) & SAR ({request.sar_image_path})",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))
        opt_path = file_manager.resolve_image_path(request.optical_image_path)
        sar_path = file_manager.resolve_image_path(request.sar_image_path)

        # 2. Geospatial extraction
        t0 = time.time()
        geo_meta_opt = geo_extractor.extract(opt_path)
        geo_meta_sar = geo_extractor.extract(sar_path)
        opt_rgb, meta_opt = raster_reader.load_rgb_array(opt_path)
        sar_rgb, meta_sar = raster_reader.load_rgb_array(sar_path)
        trace.append(ExecutionStage(
            stage="Geospatial Analysis",
            message=f"Optical ({opt_rgb.shape[1]}x{opt_rgb.shape[0]} px) | SAR ({sar_rgb.shape[1]}x{sar_rgb.shape[0]} px)",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        # 3. Spatial Co-Registration & Alignment
        t0 = time.time()
        opt_aligned, sar_aligned, is_aligned, coreg_notes = coregistration_engine.align_and_validate(
            opt_rgb, sar_rgb, meta_opt, meta_sar
        )
        trace.append(ExecutionStage(
            stage="Spatial Alignment",
            message=f"Multi-sensor co-registration: {coreg_notes}",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        # 4. SAR Preprocessing (Lee Despeckle Filter)
        t0 = time.time()
        if request.despeckle_sar:
            sar_filtered = sar_processor.lee_filter(sar_aligned, size=7)
            trace.append(ExecutionStage(
                stage="SAR Signal Processing",
                message="Applied 7x7 adaptive Lee Speckle Filter (preserving radar edges & reducing speckle)",
                timestamp=datetime.now().strftime("%H:%M:%S"),
                duration_ms=round((time.time() - t0) * 1000, 1)
            ))
        else:
            sar_filtered = sar_aligned

        # 5. Extract Physical & Statistical Modality Parameters
        t0 = time.time()
        opt_stats = sar_processor.compute_stats(opt_aligned, modality="Optical RGB")
        sar_stats = sar_processor.compute_stats(sar_filtered, modality="SAR Sentinel-1")
        stat_ms = (time.time() - t0) * 1000.0
        trace.append(ExecutionStage(
            stage="Feature Extraction",
            message=f"Computed radar backscatter dynamic range ({sar_stats.dynamic_range_db} dB, high scatterers={sar_stats.high_backscatter_ratio*100:.1f}%)",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(stat_ms, 1)
        ))

        # 6. Generate Fusion & Difference Layers
        t0 = time.time()
        fused_img, heatmap_img = sar_processor.generate_fusion_layers(opt_aligned, sar_filtered)
        fusion_ms = (time.time() - t0) * 1000.0
        trace.append(ExecutionStage(
            stage="Fusion Generation",
            message="Synthesized false-color HSV structural fusion and cross-modal difference heatmap",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round(fusion_ms, 1)
        ))

        # 7. Semantic Reasoning via SemanticReasoningService (GeoChat + Gemini)
        trace.append(ExecutionStage(
            stage="GeoChat Observations",
            message="Generating remote-sensing cross-modal observations via GeoChat",
            timestamp=datetime.now().strftime("%H:%M:%S")
        ))

        sem_res = self.semantic_service.explain_optical_sar(
            optical_stats=opt_stats,
            sar_stats=sar_stats,
            opt_rgb=opt_aligned,
            sar_rgb=sar_filtered,
            question=request.question,
            provider="geochat"
        )

        semantic_explanation = sem_res["explanation"]
        sem_inf_ms = sem_res["inference_time_ms"]
        geochat_ms = sem_res.get("geochat_time_ms", sem_inf_ms)
        gemini_ms = sem_res.get("gemini_time_ms", 0.0)
        semantic_model_name = sem_res["semantic_model"]
        sem_device = sem_res["device"]
        warning_msg = sem_res.get("transparency_warning")

        if gemini_ms > 0:
            trace.append(ExecutionStage(
                stage="Gemini Multimodal Reasoning",
                message=f"Gemini synthesized cross-modal report from optical/SAR imagery + GeoChat observations ({round(gemini_ms, 1)} ms)",
                timestamp=datetime.now().strftime("%H:%M:%S"),
                duration_ms=round(gemini_ms, 1)
            ))
        else:
            trace.append(ExecutionStage(
                stage="Cross-Modal Summary",
                message=f"Multimodal report synthesized via {semantic_model_name}",
                timestamp=datetime.now().strftime("%H:%M:%S"),
                duration_ms=round(sem_inf_ms, 1)
            ))

        # 8. Render & Save Visual Layers
        t0 = time.time()
        uid = uuid.uuid4().hex[:8]
        opt_filename = f"optsar_optical_{uid}.png"
        sar_filename = f"optsar_sar_{uid}.png"
        fused_filename = f"optsar_fused_{uid}.png"
        sar_filt_filename = f"optsar_filtered_{uid}.png"
        heatmap_filename = f"optsar_heatmap_{uid}.png"

        Image.fromarray(opt_aligned).save(settings.OUTPUT_PATH / opt_filename, format="PNG")
        Image.fromarray(sar_aligned).save(settings.OUTPUT_PATH / sar_filename, format="PNG")
        Image.fromarray(sar_filtered).save(settings.OUTPUT_PATH / sar_filt_filename, format="PNG")
        Image.fromarray(fused_img).save(settings.OUTPUT_PATH / fused_filename, format="PNG")
        Image.fromarray(heatmap_img).save(settings.OUTPUT_PATH / heatmap_filename, format="PNG")

        trace.append(ExecutionStage(
            stage="Rendering & Export",
            message="Exported Optical, SAR, Fused, and Cross-Modal Heatmap layers",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            duration_ms=round((time.time() - t0) * 1000, 1)
        ))

        total_ms = (time.time() - start_time) * 1000.0
        specialized_ms = stat_ms + fusion_ms + 10.0

        detected_features = [
            f"Radar Dynamic Range: {sar_stats.dynamic_range_db:.1f} dB",
            f"High-Backscatter Structures (Buildings/Metals): {sar_stats.high_backscatter_ratio*100:.1f}%",
            f"Low-Backscatter Specular Zones (Water/Runways): {sar_stats.low_backscatter_ratio*100:.1f}%",
            f"Optical Mean Intensity: {opt_stats.mean_intensity:.1f}",
            "Lee Despeckling: Enabled" if request.despeckle_sar else "Lee Despeckling: Disabled"
        ]

        # Benchmark: Radar Signal Engine
        benchmark_tracker.record_run(
            model="SAR Radar Signal Engine",
            task="Optical + SAR Preprocessing & Fusion",
            inference_time_ms=specialized_ms,
            device=self.model_manager.get_device_name(),
            image_size=f"{opt_aligned.shape[1]}x{opt_aligned.shape[0]}",
            vram_usage_mb=0.0
        )

        # Benchmark: Semantic Model
        benchmark_tracker.record_run(
            model=semantic_model_name,
            task="Cross-Modal Semantic Reasoning",
            inference_time_ms=sem_inf_ms,
            device=sem_device,
            image_size=f"{opt_aligned.shape[1]}x{opt_aligned.shape[0]}",
            vram_usage_mb=0.0
        )

        return OpticalSARResponse(
            task="optical_sar_fusion",
            status="success",
            optical_image_url=f"/api/files/view/{opt_filename}",
            sar_image_url=f"/api/files/view/{sar_filename}",
            fused_image_url=f"/api/files/view/{fused_filename}",
            sar_filtered_url=f"/api/files/view/{sar_filt_filename}",
            difference_heatmap_url=f"/api/files/view/{heatmap_filename}",
            optical_stats=opt_stats,
            sar_stats=sar_stats,
            detected_features=detected_features,
            cross_modal_analysis=semantic_explanation,
            model="Multimodal Feature Pipeline + Semantic VLM",
            optical_processing="Visible Reflectance & Dynamic Contrast Normalization",
            sar_processing="7x7 Lee Speckle Filter & CFAR Signal Engine",
            semantic_model=semantic_model_name,
            device=self.model_manager.get_device_name(),
            processing_time_ms=round(total_ms, 1),
            specialized_time_ms=round(specialized_ms, 1),
            semantic_time_ms=round(sem_inf_ms, 1),
            geo_metadata_optical=geo_meta_opt,
            geo_metadata_sar=geo_meta_sar,
            execution_trace=trace,
            transparency_warning=warning_msg or (
                "Semantic interpretation provided by a general-purpose vision-language model and is not a substitute for a specialized SAR foundation model. "
                "Physical radar backscatter statistics (dB, speckle metrics) are computed deterministically via SAR signal algorithms."
            )
        )

optical_sar_pipeline = OpticalSARPipeline()
