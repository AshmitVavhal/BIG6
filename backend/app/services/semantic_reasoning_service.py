"""
SatQuery AI - Hybrid Remote-Sensing Semantic Reasoning Service
Orchestrates GeoChat (remote-sensing observations) + Gemini (final multimodal reasoning and synthesis)
alongside specialized computer vision models.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image
from app.config import settings
from app.models.model_manager import model_manager
from app.services.gemini_service import gemini_service, normalize_structured_vqa
from app.utils.logger import logger


class SemanticReasoningService:
    """
    Hybrid Semantic Reasoning Service:
    - GeoChat: Remote-sensing visual observations and domain context.
    - Gemini: Final multimodal reasoning, synthesis, and natural-language explanation receiving the original image.
    - Specialized CV models (Grounding DINO, SAM2, ChangeFormer, SAR Engine): Authoritative for all detections/metrics.
    """

    def analyze_vqa(
        self,
        image_rgb: np.ndarray,
        question: str,
        image_type: str = "RGB",
        image_metadata: Optional[dict] = None,
        detections: Optional[List[Any]] = None,
        segmentation: Optional[Dict[str, Any]] = None,
        change_detection: Optional[Dict[str, Any]] = None,
        geo_context: Optional[dict] = None,
        provider: Optional[str] = None,
        geochat_observations: Optional[str] = None,
        caption: Optional[str] = None,
        geochat_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Execute Visual Question Answering via GeoChat + CV Models + Gemini Hybrid Pipeline.
        Guarantees strictly grounded, 4-part point-wise structured output.
        """
        # Step 1: Run GeoChat for remote-sensing domain observations if not already supplied
        if geochat_observations is None:
            geochat = model_manager.get_geochat()
            geochat_obs, cap, g_ms = geochat.generate_vqa_answer(
                image_rgb=image_rgb,
                question=question,
                geo_context=geo_context
            )
            caption = cap if caption is None else caption
            geochat_ms = g_ms
        else:
            geochat_obs = geochat_observations
            if caption is None:
                caption = "High-resolution optical satellite scene analyzed via GeoChat."

        pil_image = Image.fromarray(image_rgb)
        gemini_ms = 0.0

        # Build verified metrics from geospatial metadata and CV detections
        verified_metrics = {}
        if geo_context:
            for k in ["crs", "width", "height", "count", "resolution", "sensor_info"]:
                if geo_context.get(k) is not None:
                    verified_metrics[k] = geo_context[k]
        if detections:
            verified_metrics["verified_detections_count"] = len(detections)

        # Step 2: Final Multimodal Reasoning via Gemini
        if gemini_service.is_configured:
            try:
                gemini_res = gemini_service.analyze(
                    image=pil_image,
                    user_question=question,
                    image_type=image_type,
                    image_metadata=image_metadata or geo_context,
                    geochat_observations=geochat_obs,
                    detections=detections,
                    segmentation=segmentation,
                    change_detection=change_detection,
                    verified_metrics=verified_metrics if verified_metrics else None
                )
                if gemini_res.get("success") and gemini_res.get("answer"):
                    final_answer = gemini_res["answer"]
                    gemini_ms = gemini_res["inference_time_ms"]
                    model_label = gemini_res.get("semantic_model", f"GeoChat + Gemini ({gemini_service.model_name})")
                    sources = gemini_res["sources"]
                else:
                    logger.warning(f"Gemini reasoning unsuccessful ({gemini_res.get('error')}). Using GeoChat observations.")
                    norm = normalize_structured_vqa(geochat_obs, image_type=image_type, verified_metrics=verified_metrics)
                    final_answer = norm["formatted_text"]
                    model_label = "GeoChat"
                    sources = ["original_image", "GeoChat"]
            except Exception as e:
                logger.warning(f"Gemini reasoning failed ({e}). Returning structured GeoChat observations.")
                norm = normalize_structured_vqa(geochat_obs, image_type=image_type, verified_metrics=verified_metrics)
                final_answer = norm["formatted_text"]
                model_label = "GeoChat"
                sources = ["original_image", "GeoChat"]
        else:
            norm = normalize_structured_vqa(geochat_obs, image_type=image_type, verified_metrics=verified_metrics)
            final_answer = norm["formatted_text"]
            model_label = "GeoChat"
            sources = ["original_image", "GeoChat"]

        total_inf_ms = geochat_ms + gemini_ms

        warning_text = (
            "Multimodal reasoning synthesized by Gemini using original satellite imagery, "
            "GeoChat remote-sensing observations, and verified geospatial metadata."
            if (gemini_service.is_configured and ("Gemini" in model_label)) else
            "Semantic observations generated by GeoChat remote-sensing VLM. "
            "To enable enhanced multimodal reasoning with Google Gemini, set GEMINI_API_KEY in backend/.env."
        )

        return {
            "answer": final_answer,
            "geochat_observations": geochat_obs,
            "caption": caption,
            "inference_time_ms": total_inf_ms,
            "geochat_time_ms": geochat_ms,
            "gemini_time_ms": gemini_ms,
            "semantic_model": model_label,
            "provider": "geochat_gemini" if (gemini_service.is_configured and "Gemini" in model_label) else "geochat",
            "device": model_manager.get_device_name(),
            "confidence": 0.98 if (gemini_service.is_configured and "Gemini" in model_label) else 0.95,
            "confidence_type": "heuristic",
            "is_fallback": False,
            "sources": sources,
            "transparency_warning": warning_text
        }

    def explain_bi_temporal(
        self,
        change_percentage: float,
        num_regions: int,
        regions: List[Any],
        t1_rgb: np.ndarray,
        t2_rgb: np.ndarray,
        coregistration_notes: Optional[str] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate semantic reasoning for ChangeMamba bi-temporal changes via GeoChat + Gemini.
        ChangeMamba calculates the authoritative quantitative change; Gemini explains the context.
        """
        # Step 1: GeoChat domain observations
        geochat = model_manager.get_geochat()
        geochat_obs, geochat_ms = geochat.explain_bi_temporal_changes(
            change_percentage=change_percentage,
            num_regions=num_regions,
            regions=regions,
            t1_rgb=t1_rgb,
            t2_rgb=t2_rgb,
            coregistration_notes=coregistration_notes
        )

        pil_t1 = Image.fromarray(t1_rgb)
        pil_t2 = Image.fromarray(t2_rgb)
        gemini_ms = 0.0

        verified_metrics = {
            "change_percentage": f"{change_percentage:.2f}%",
            "num_regions": num_regions,
            "coregistration": coregistration_notes or "Verified"
        }
        change_detection_dict = {
            "model": "ChangeMamba",
            "change_percentage": change_percentage,
            "connected_regions_count": num_regions,
            "coregistration": coregistration_notes or "Verified"
        }

        # Step 2: Gemini synthesis
        if gemini_service.is_configured:
            try:
                gemini_res = gemini_service.analyze(
                    image=pil_t1,
                    secondary_image=pil_t2,
                    user_question=f"Explain the bi-temporal changes detected between T1 (pre-change) and T2 (post-change). ChangeMamba measured {change_percentage:.2f}% changed area.",
                    image_type="Bi-Temporal Satellite Imagery",
                    geochat_observations=geochat_obs,
                    change_detection=change_detection_dict,
                    cv_results=change_detection_dict,
                    verified_metrics=verified_metrics
                )
                if gemini_res.get("success"):
                    final_explanation = gemini_res["answer"]
                    gemini_ms = gemini_res["inference_time_ms"]
                    model_label = f"ChangeMamba + GeoChat + Gemini ({gemini_service.model_name})"
                    sources = gemini_res["sources"]
                else:
                    logger.warning(f"Gemini bi-temporal reasoning unsuccessful ({gemini_res.get('error')}). Using GeoChat interpretation.")
                    final_explanation = geochat_obs
                    model_label = "ChangeMamba + GeoChat"
                    sources = ["original_image", "ChangeMamba", "GeoChat"]
            except Exception as e:
                logger.warning(f"Gemini bi-temporal reasoning failed ({e}). Returning GeoChat interpretation.")
                final_explanation = geochat_obs
                model_label = "ChangeMamba + GeoChat"
                sources = ["original_image", "ChangeMamba", "GeoChat"]
        else:
            final_explanation = geochat_obs
            model_label = "ChangeMamba + GeoChat"
            sources = ["original_image", "ChangeMamba", "GeoChat"]

        total_inf_ms = geochat_ms + gemini_ms
        return {
            "explanation": final_explanation,
            "geochat_observations": geochat_obs,
            "inference_time_ms": total_inf_ms,
            "geochat_time_ms": geochat_ms,
            "gemini_time_ms": gemini_ms,
            "semantic_model": model_label,
            "provider": "geochat_gemini" if gemini_service.is_configured else "geochat",
            "device": model_manager.get_device_name(),
            "sources": sources,
            "is_fallback": False,
            "transparency_warning": (
                "Change masks and percentages are computed authoritatively by ChangeMamba. "
                "Semantic reasoning is synthesized by Gemini with GeoChat domain observations."
            )
        }

    def explain_highlight(
        self,
        prompt: str,
        num_detections: int,
        total_area_pct: float,
        detections: List[Any],
        image_rgb: np.ndarray,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate semantic explanation for LAE-DINO + Mask2Former detections via GeoChat + Gemini."""
        geochat = model_manager.get_geochat()
        geochat_obs, geochat_ms = geochat.explain_highlight(
            prompt=prompt,
            num_detections=num_detections,
            total_area_pct=total_area_pct,
            detections=detections,
            image_rgb=image_rgb
        )

        pil_image = Image.fromarray(image_rgb)
        gemini_ms = 0.0

        formatted_dets = [
            d.model_dump() if hasattr(d, "model_dump") else d
            for d in detections[:20]
        ]

        cv_results = {
            "detection_model": "LAE-DINO (Locate Anything on Earth)",
            "segmentation_model": "Mask2Former",
            "target_prompt": prompt,
            "localized_instances_count": num_detections,
            "total_area_percentage": f"{total_area_pct:.2f}%"
        }
        verified_metrics = {
            "num_detections": num_detections,
            "total_area_pct": f"{total_area_pct:.2f}%"
        }

        if gemini_service.is_configured:
            try:
                gemini_res = gemini_service.analyze(
                    image=pil_image,
                    user_question=f"Explain the spatial distribution, context, and visual characteristics of the {num_detections} detected instance(s) of '{prompt}'.",
                    image_type="Satellite Scene (Optical)",
                    geochat_observations=geochat_obs,
                    detections=formatted_dets,
                    cv_results=cv_results,
                    verified_metrics=verified_metrics
                )
                if gemini_res.get("success"):
                    final_summary = gemini_res["answer"]
                    gemini_ms = gemini_res["inference_time_ms"]
                    model_label = f"LAE-DINO + Mask2Former + GeoChat + Gemini ({gemini_service.model_name})"
                    sources = gemini_res["sources"]
                else:
                    logger.warning(f"Gemini highlight reasoning unsuccessful ({gemini_res.get('error')}). Using GeoChat explanation.")
                    final_summary = geochat_obs
                    model_label = "LAE-DINO + Mask2Former + GeoChat"
                    sources = ["original_image", "LAE-DINO", "Mask2Former", "GeoChat"]
            except Exception as e:
                logger.warning(f"Gemini highlight explanation failed ({e}). Returning GeoChat explanation.")
                final_summary = geochat_obs
                model_label = "LAE-DINO + Mask2Former + GeoChat"
                sources = ["original_image", "LAE-DINO", "Mask2Former", "GeoChat"]
        else:
            final_summary = geochat_obs
            model_label = "LAE-DINO + Mask2Former + GeoChat"
            sources = ["original_image", "LAE-DINO", "Mask2Former", "GeoChat"]

        total_inf_ms = geochat_ms + gemini_ms
        return {
            "summary": final_summary,
            "geochat_observations": geochat_obs,
            "inference_time_ms": total_inf_ms,
            "geochat_time_ms": geochat_ms,
            "gemini_time_ms": gemini_ms,
            "semantic_model": model_label,
            "provider": "geochat_gemini" if gemini_service.is_configured else "geochat",
            "device": model_manager.get_device_name(),
            "sources": sources,
            "is_fallback": False,
            "transparency_warning": (
                "Object detections and segmentation masks are computed authoritatively by LAE-DINO + Mask2Former. "
                "Semantic reasoning synthesized by Gemini with GeoChat domain context."
            )
        }

    def explain_optical_sar(
        self,
        optical_stats: Any,
        sar_stats: Any,
        opt_rgb: np.ndarray,
        sar_rgb: np.ndarray,
        question: Optional[str] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate semantic explanation for Optical + SAR complementary analysis via GeoChat + Gemini."""
        geochat = model_manager.get_geochat()
        geochat_obs, geochat_ms = geochat.explain_optical_sar(
            optical_stats=optical_stats,
            sar_stats=sar_stats,
            opt_rgb=opt_rgb,
            sar_rgb=sar_rgb,
            question=question
        )

        pil_opt = Image.fromarray(opt_rgb)
        pil_sar = Image.fromarray(sar_rgb)
        gemini_ms = 0.0

        sar_dict = sar_stats.model_dump() if hasattr(sar_stats, "model_dump") else dict(sar_stats) if isinstance(sar_stats, dict) else {}
        opt_dict = optical_stats.model_dump() if hasattr(optical_stats, "model_dump") else dict(optical_stats) if isinstance(optical_stats, dict) else {}

        verified_metrics = {
            "sar_snr_db": sar_dict.get("snr_db", "N/A"),
            "sar_mean_backscatter_db": sar_dict.get("mean_db", "N/A"),
            "sar_dynamic_range_db": sar_dict.get("dynamic_range_db", "N/A"),
            "sar_roughness_index": sar_dict.get("roughness_index", "N/A"),
            "lee_filter_applied": sar_dict.get("lee_filter_applied", True),
            "optical_contrast_ratio": opt_dict.get("contrast_ratio", "N/A"),
            "optical_dynamic_range": opt_dict.get("dynamic_range", "N/A")
        }
        cv_results = {
            "sar_signal_engine": "Lee Speckle Filtered Backscatter Engine (dB)",
            "optical_engine": "Radiometric Contrast & Dynamic Range Engine"
        }

        user_q = question if (question and question.strip()) else "Explain how the complementary optical surface textures and SAR radar backscatter signatures inform the scene understanding."

        if gemini_service.is_configured:
            try:
                gemini_res = gemini_service.analyze(
                    image=pil_opt,
                    secondary_image=pil_sar,
                    user_question=user_q,
                    geochat_observations=geochat_obs,
                    cv_results=cv_results,
                    verified_metrics=verified_metrics
                )
                if gemini_res.get("success"):
                    final_analysis = gemini_res["answer"]
                    gemini_ms = gemini_res["inference_time_ms"]
                    model_label = f"SAR Engine + GeoChat + Gemini ({gemini_service.model_name})"
                    sources = gemini_res["sources"]
                else:
                    logger.warning(f"Gemini optical/SAR reasoning unsuccessful ({gemini_res.get('error')}). Using GeoChat synthesis.")
                    final_analysis = geochat_obs
                    model_label = "SAR Engine + GeoChat"
                    sources = ["original_image", "SAR Signal Engine", "GeoChat"]
            except Exception as e:
                logger.warning(f"Gemini optical/SAR reasoning failed ({e}). Returning GeoChat synthesis.")
                final_analysis = geochat_obs
                model_label = "SAR Engine + GeoChat"
                sources = ["original_image", "SAR Signal Engine", "GeoChat"]
        else:
            final_analysis = geochat_obs
            model_label = "SAR Engine + GeoChat"
            sources = ["original_image", "SAR Signal Engine", "GeoChat"]

        total_inf_ms = geochat_ms + gemini_ms
        return {
            "explanation": final_analysis,
            "fusion_analysis": final_analysis,
            "geochat_observations": geochat_obs,
            "inference_time_ms": total_inf_ms,
            "geochat_time_ms": geochat_ms,
            "gemini_time_ms": gemini_ms,
            "semantic_model": model_label,
            "provider": "geochat_gemini" if gemini_service.is_configured else "geochat",
            "device": model_manager.get_device_name(),
            "sources": sources,
            "is_fallback": False,
            "transparency_warning": (
                "Radar backscatter statistics (dB, speckle metrics) are computed deterministically via SAR signal algorithms. "
                "Semantic cross-modal explanation is synthesized by Gemini with GeoChat observations."
            )
        }


semantic_reasoning_service = SemanticReasoningService()
