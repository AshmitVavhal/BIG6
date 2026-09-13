"""
SatQuery AI - Gemini Multimodal Semantic Reasoning & Grounded Visual Analysis Service
Integrates Google Gemini as the authoritative final reasoning and natural-language generation layer,
receiving the original satellite imagery, GeoChat remote-sensing observations, and verified CV metrics.
"""

import time
import json
import re
from typing import Optional, Dict, Any, List, Union
from PIL import Image
import google.generativeai as genai
from app.config import settings
from app.utils.logger import logger

GEMINI_SYSTEM_INSTRUCTION = """You are the final visual analyst for SatQuery AI.

Analyze satellite or aerial imagery like a careful, professional human analyst.

Your job is to describe what is actually visible in the provided image and answer the user's question directly.

Be concrete, visually grounded, and specific.

Identify recognizable objects and scene elements such as:
- buildings
- houses
- roads
- vehicles
- trees
- vegetation
- grass
- agricultural fields
- bare soil
- water
- construction
- industrial structures
- parking areas
- bridges
- infrastructure
- coastline
- rivers
- urban blocks

Describe spatial relationships when clearly visible.

For example:
GOOD:
'The image shows a low-density residential neighborhood with detached houses distributed along several paved roads. Most properties are surrounded by mature trees and grassy yards.'

BAD:
'The image contains textured ground cover with variations in surface reflectance and natural boundaries.'

Avoid vague remote-sensing filler.
Do NOT use phrases such as:
- 'textured ground cover'
- 'surface reflectance variations'
- 'natural boundaries'
- 'contrasting terrain sectors'
- 'higher structural contrast'
- 'uniform terrain'
- 'radiometric characteristics'
- 'spectral profile'
- 'elevation variations'
unless the provided data actually supports those statements.

Do not infer elevation from RGB brightness.
Do not infer multispectral information from an RGB image.
Do not describe radiometric properties unless they were actually computed from the source raster.
Do not invent geographic location.
Do not invent sensor/platform information.
Do not invent percentages.
Do not invent exact object counts.
Do not invent coordinates.
Do not invent distances or dimensions.
Do not claim something is present if it is not visually supported.

When uncertain, use language such as: 'appears to be', 'visually consistent with', or 'not clearly visible'.
However, do not overuse uncertainty language when the object is obvious.

Write like a professional human imagery analyst, not like a generic image-captioning model.

Answer the user's question first.
Use structured sections only when they improve clarity.
Keep descriptions concise but informative.

For general scene questions (e.g., 'What is in this image?', 'Analyze this scene'), prefer this structure:
Overview:
One or two sentences describing the scene.

Visible Features:
Concrete objects and land-cover features.

Spatial Pattern:
How the objects/features are distributed.

Interpretation:
A cautious overall interpretation based only on visible evidence.

For specific questions (e.g., 'What buildings are visible?', 'Describe the roads', 'Is this urban or rural?', 'Are there industrial buildings?'), answer directly without forcing unnecessary section headers.
Do not add unnecessary technical terminology.
"""


class GeminiService:
    """Service for multimodal satellite imagery reasoning with Google Gemini."""

    FORBIDDEN_FILLER_PATTERNS = [
        r"textured ground cover",
        r"surface reflectance variations?",
        r"variations in surface reflectance",
        r"natural boundaries",
        r"contrasting (light and dark )?terrain sectors",
        r"higher structural contrast",
        r"uniform terrain",
        r"radiometric characteristics",
        r"spectral profile",
        r"elevation variations"
    ]

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self._custom_api_key = api_key
        self.model_name = model_name or settings.GEMINI_MODEL
        self._is_configured = False
        self._init_gemini()

    def _init_gemini(self):
        """Initialize Google Gemini API client securely."""
        api_key = self._custom_api_key if self._custom_api_key is not None else settings.GEMINI_API_KEY
        if api_key and api_key.strip() and api_key != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=api_key.strip())
                self._is_configured = True
                logger.info(f"Gemini API configured successfully with default model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to configure Gemini API: {e}")
                self._is_configured = False
        else:
            self._is_configured = False

    @property
    def is_configured(self) -> bool:
        """Check if Gemini API key is configured."""
        api_key = self._custom_api_key if self._custom_api_key is not None else settings.GEMINI_API_KEY
        return bool(api_key and api_key.strip() and api_key != "your_gemini_api_key_here")

    def _build_context_prompt(
        self,
        user_question: str,
        image_type: str = "RGB",
        image_metadata: Optional[Dict[str, Any]] = None,
        geochat_observations: Optional[Union[Dict[str, Any], str]] = None,
        detections: Optional[List[Any]] = None,
        segmentation: Optional[Dict[str, Any]] = None,
        change_detection: Optional[Dict[str, Any]] = None,
        verified_metrics: Optional[Dict[str, Any]] = None,
        cv_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construct structured multimodal context object and prompt for Gemini."""
        prompt_parts: List[str] = []

        context_obj: Dict[str, Any] = {
            "image_type": image_type,
            "user_question": user_question.strip()
        }

        if image_metadata:
            clean_meta = {}
            for k in ["width", "height", "count", "crs", "resolution", "sensor_info", "driver", "has_georeference"]:
                if k in image_metadata and image_metadata[k] is not None:
                    clean_meta[k] = image_metadata[k]
            if clean_meta:
                context_obj["image_metadata"] = clean_meta

        if geochat_observations:
            context_obj["geochat_observations"] = geochat_observations

        if detections:
            context_obj["detections"] = detections

        if segmentation:
            context_obj["segmentation"] = segmentation

        if change_detection:
            context_obj["change_detection"] = change_detection

        if verified_metrics:
            context_obj["verified_metrics"] = verified_metrics

        if cv_results:
            context_obj["cv_results"] = cv_results

        prompt_parts.append(f"User Question: {user_question.strip()}\n")
        prompt_parts.append("=== STRUCTURED CONTEXT & EVIDENCE ===")
        prompt_parts.append(json.dumps(context_obj, indent=2, default=str))
        prompt_parts.append("")

        prompt_parts.append(
            "Instructions for Response:\n"
            "1. Inspect the original image directly and answer the user's question with concrete, visually grounded evidence.\n"
            "2. Prioritize: (1) Original Image, (2) Verified CV model results, (3) Actual metadata, (4) GeoChat observations.\n"
            "3. If the user asks a specific or simple question (e.g., 'What buildings are visible?', 'Describe the roads', 'Is this urban or rural?'), answer directly without unnecessary section headers.\n"
            "4. For general scene questions (e.g., 'What is in this image?'), use Overview, Visible Features, Spatial Pattern, and Interpretation sections.\n"
            "5. Never use vague remote-sensing filler ('textured ground cover', 'surface reflectance variations', 'natural boundaries', 'contrasting terrain sectors'). Name real features like houses, roads, trees, grass, lawns, driveways, fields, water.\n"
            "6. Only report numbers, counts, or percentages that appear in verified CV metrics."
        )

        return "\n".join(prompt_parts)

    def validate_and_sanitize_response(
        self,
        response_text: str,
        image_type: str = "RGB",
        verified_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Post-generation validation to detect and clean generic remote-sensing filler
        or unsupported claims before returning to the user.
        """
        if not response_text:
            return ""

        cleaned = response_text.strip()

        replacements = [
            (r"(?i)\btextured ground cover\b", "ground surface and vegetation"),
            (r"(?i)\bvariations in surface reflectance\b", "visible land cover contrasts"),
            (r"(?i)\bsurface reflectance variations?\b", "visible surface features"),
            (r"(?i)\bnatural boundaries\b", "property and terrain boundaries"),
            (r"(?i)\bcontrasting light and dark terrain sectors\b", "vegetated and developed areas"),
            (r"(?i)\bcontrasting terrain sectors\b", "vegetation and open ground"),
            (r"(?i)\bhigher structural contrast\b", "higher building density"),
            (r"(?i)\buniform terrain\b", "open vegetated areas")
        ]

        for pattern, repl in replacements:
            cleaned = re.sub(pattern, repl, cleaned)

        # Ensure RGB imagery is not falsely labeled as multispectral
        if image_type.upper() in ("RGB", "OPTICAL RGB", "OPTICAL"):
            cleaned = re.sub(r"(?i)\bmultispectral bands?\b", "visible color bands", cleaned)
            cleaned = re.sub(r"(?i)\bmultispectral scene\b", "optical scene", cleaned)
            cleaned = re.sub(r"(?i)\bmultispectral imagery\b", "optical aerial imagery", cleaned)

        # If elevation variations are mentioned without DEM metadata, soften the claim
        has_dem = verified_metrics and ("elevation" in str(verified_metrics) or "dem" in str(verified_metrics))
        if not has_dem:
            cleaned = re.sub(r"(?i)\belevation variations?\b", "visible surface variations", cleaned)

        return cleaned

    def _get_api_key(self) -> Optional[str]:
        return self._custom_api_key if self._custom_api_key is not None else settings.GEMINI_API_KEY

    def analyze(
        self,
        image: Optional[Image.Image],
        user_question: str,
        image_type: str = "RGB",
        image_metadata: Optional[Dict[str, Any]] = None,
        geochat_observations: Optional[Union[Dict[str, Any], str]] = None,
        detections: Optional[List[Any]] = None,
        segmentation: Optional[Dict[str, Any]] = None,
        change_detection: Optional[Dict[str, Any]] = None,
        cv_results: Optional[Dict[str, Any]] = None,
        verified_metrics: Optional[Dict[str, Any]] = None,
        secondary_image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Synthesize satellite imagery observations using Google Gemini multimodal model.

        Parameters:
            image: Original PIL satellite image (T1 or single scene)
            user_question: User's original question or analytical goal
            image_type: Type of imagery (e.g. 'RGB', 'Optical', 'SAR', 'Multispectral')
            image_metadata: Metadata dictionary (dimensions, CRS, GSD resolution)
            geochat_observations: Semantic visual observations from GeoChat
            detections: Authoritative object detections from LAE-DINO
            segmentation: Authoritative segmentation info from Mask2Former
            change_detection: Authoritative change detection results from ChangeMamba
            cv_results: Additional specialized CV model results
            verified_metrics: Validated quantitative metrics
            secondary_image: Optional second PIL image (T2 or SAR)

        Returns:
            Dict containing answer, model name, inference time, sources, and verified metrics.
        """
        start_time = time.time()
        sources = ["original_image"] if image is not None else []
        if geochat_observations:
            sources.append("GeoChat")
        if detections or (cv_results and "LAE-DINO" in str(cv_results)):
            sources.append("LAE-DINO")
        if segmentation or (cv_results and "Mask2Former" in str(cv_results)):
            sources.append("Mask2Former")
        if change_detection or (cv_results and "ChangeMamba" in str(cv_results)):
            sources.append("ChangeMamba")
        if verified_metrics:
            sources.append("verified_metrics")
            if "lee_filter_applied" in str(verified_metrics) or "sar_snr_db" in str(verified_metrics):
                sources.append("SAR Signal Engine")

        api_key = self._get_api_key()
        if not self.is_configured or not api_key:
            error_msg = (
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your backend .env file "
                "to enable final multimodal reasoning."
            )
            logger.warning(error_msg)
            return {
                "answer": "",
                "model": "Gemini (unconfigured)",
                "semantic_source": "GeoChat (fallback)",
                "verified_metrics": verified_metrics or {},
                "inference_time_ms": 0.0,
                "sources": sources,
                "success": False,
                "error": error_msg
            }

        try:
            genai.configure(api_key=api_key.strip())

            # Build text prompt with structured context
            prompt_text = self._build_context_prompt(
                user_question=user_question,
                image_type=image_type,
                image_metadata=image_metadata,
                geochat_observations=geochat_observations,
                detections=detections,
                segmentation=segmentation,
                change_detection=change_detection,
                verified_metrics=verified_metrics,
                cv_results=cv_results
            )

            # Assemble multimodal content payload (Image + Text)
            contents: List[Any] = []
            if image is not None:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                contents.append(image)

            if secondary_image is not None:
                if secondary_image.mode != "RGB":
                    secondary_image = secondary_image.convert("RGB")
                contents.append(secondary_image)

            contents.append(prompt_text)

            candidate_models = [self.model_name]
            for fallback in [
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-flash-latest",
                "gemini-2.5-flash-lite",
                "gemini-pro-latest"
            ]:
                if fallback not in candidate_models:
                    candidate_models.append(fallback)

            response = None
            used_model = self.model_name
            last_err = None

            for m_name in candidate_models:
                try:
                    logger.info(f"Sending multimodal satellite reasoning request to Gemini ({m_name})...")
                    model = genai.GenerativeModel(
                        model_name=m_name,
                        system_instruction=GEMINI_SYSTEM_INSTRUCTION
                    )
                    response = model.generate_content(
                        contents=contents,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2,
                            max_output_tokens=1024
                        ),
                        request_options={"timeout": 15.0}
                    )
                    if response and response.text:
                        used_model = m_name
                        break
                except Exception as cand_err:
                    last_err = cand_err
                    logger.warning(f"Gemini call with '{m_name}' unsuccessful: {cand_err}. Trying next model candidate if available.")

            if response is None or not getattr(response, "text", None):
                if last_err:
                    raise last_err
                raise RuntimeError("No response generated by Gemini.")

            raw_answer = response.text.strip()
            sanitized_answer = self.validate_and_sanitize_response(
                raw_answer,
                image_type=image_type,
                verified_metrics=verified_metrics
            )

            inf_time = (time.time() - start_time) * 1000.0

            return {
                "answer": sanitized_answer,
                "model": f"GeoChat + Gemini ({used_model})",
                "semantic_source": "GeoChat + original image",
                "verified_metrics": verified_metrics or {},
                "inference_time_ms": round(inf_time, 2),
                "sources": sources,
                "success": True
            }

        except Exception as e:
            logger.error(f"Gemini API execution error: {e}", exc_info=True)
            inf_time = (time.time() - start_time) * 1000.0
            return {
                "answer": "",
                "model": f"Gemini ({self.model_name})",
                "semantic_source": "GeoChat (fallback)",
                "verified_metrics": verified_metrics or {},
                "inference_time_ms": round(inf_time, 2),
                "sources": sources,
                "success": False,
                "error": str(e)
            }


# Singleton instance
gemini_service = GeminiService()
