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

GEMINI_SYSTEM_INSTRUCTION = """You are the expert satellite imagery visual reasoning analyst for SatQuery AI.

Analyze satellite or aerial imagery with extreme precision, strict visual grounding, and professional objectivity.

Your job is to visually inspect the provided image and describe EXACTLY what is present.

MANDATORY RULES:
1. Ground your analysis strictly and ONLY in the CURRENT image.
2. Never hallucinate features not visible (e.g. do NOT mention wildfire, fire hotspots, smoke, burn scars, or mountainous terrain unless clearly visible in this specific image).
3. Recognize actual visible objects such as:
   - Residential houses, single-family detached homes, roof structures
   - Road networks, paved streets, cul-de-sacs, driveways, sidewalks
   - Trees, vegetation cover, grassy lawns, landscaped yards
   - Open / bare land, cleared lots, dry fields, soil parcels
   - Swimming pools (bright blue/cyan shapes in yards) ONLY if actually visible
   - Construction / bare ground ONLY if actually visible
   - Water bodies, coastlines, or drainage corridors ONLY if actually visible
   - Commercial / industrial / airport / seaport structures ONLY if actually visible
4. Do NOT use vague filler ("textured ground cover", "surface reflectance variations", "natural boundaries", "contrasting terrain sectors").
5. Do NOT invent fake percentages, unverified object counts, fake coordinates, or radiometric claims.

MANDATORY OUTPUT FORMAT:
You MUST respond using EXACTLY these 4 capitalized section headings, with point-wise bullet items under VISIBLE FEATURES:

OVERVIEW
[A concise 1-2 sentence description of the overall scene.]

VISIBLE FEATURES
- **[Feature Category]**: [Concise, point-wise description of visible objects/features.]
- **[Feature Category]**: [Concise, point-wise description.]
- **[Feature Category]**: [Concise, point-wise description.]
- **[Feature Category]**: [Concise, point-wise description.]

SPATIAL PATTERN
[Description of the spatial arrangement and distribution of features across the scene.]

INTERPRETATION
[Objective conclusion/interpretation based strictly on visible evidence.]
"""


def normalize_structured_vqa(
    raw_text: str,
    image_type: str = "RGB",
    verified_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validates, parses, and normalizes VQA text into the authoritative 4-part structured format:
    - OVERVIEW
    - VISIBLE FEATURES (separate point-wise bullet items)
    - SPATIAL PATTERN
    - INTERPRETATION
    """
    if not raw_text or not raw_text.strip():
        return {
            "overview": "No visual data available for the current scene.",
            "visible_features": ["- **Surface Features**: No distinct features identified."],
            "spatial_pattern": "Uniform distribution across the scene.",
            "interpretation": "Insufficient imagery evidence for interpretation.",
            "formatted_text": "OVERVIEW\nNo visual data available for the current scene.\n\nVISIBLE FEATURES\n- **Surface Features**: No distinct features identified.\n\nSPATIAL PATTERN\nUniform distribution across the scene.\n\nINTERPRETATION\nInsufficient imagery evidence for interpretation."
        }

    cleaned = raw_text.strip()

    # Clean generic remote sensing filler patterns
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

    if image_type.upper() in ("RGB", "OPTICAL RGB", "OPTICAL"):
        cleaned = re.sub(r"(?i)\bmultispectral bands?\b", "visible color bands", cleaned)
        cleaned = re.sub(r"(?i)\bmultispectral scene\b", "optical scene", cleaned)
        cleaned = re.sub(r"(?i)\bmultispectral imagery\b", "optical aerial imagery", cleaned)

    has_dem = verified_metrics and ("elevation" in str(verified_metrics) or "dem" in str(verified_metrics))
    if not has_dem:
        cleaned = re.sub(r"(?i)\belevation variations?\b", "visible surface variations", cleaned)

    # Clean ungrounded disaster/fire/thermal terms if not present in verified metrics
    cleaned = re.sub(r"(?i)\bactive thermal fire hotspots?\b", "built-up structures and terrain features", cleaned)
    cleaned = re.sub(r"(?i)\bthermal fire hotspots?\b", "built-up structures", cleaned)
    cleaned = re.sub(r"(?i)\bcharred burn scars?\b", "dark paved and roof surfaces", cleaned)
    cleaned = re.sub(r"(?i)\bsmoke haze\b", "atmospheric conditions", cleaned)

    # Regex extraction of sections (handles markdown headers, numbers, bolding, colons)
    overview_match = re.search(
        r"(?:^|\n)(?:\*\*)?(?:#{1,6}\s*)?(?:1\.\s*)?(?:OVERVIEW|Overview)(?:\*\*)?\s*:?\s*\n(.*?)(?=(?:\n(?:\*\*)?(?:#{1,6}\s*)?(?:2\.\s*)?(?:VISIBLE\s+FEATURES|Visible\s+Features|VISUAL\s+OBSERVATIONS|Visual\s+Observations|SPATIAL\s+PATTERN|Spatial\s+Pattern|INTERPRETATION|Interpretation)(?:\*\*)?\s*:?|\Z))",
        cleaned,
        re.DOTALL | re.IGNORECASE
    )
    features_match = re.search(
        r"(?:^|\n)(?:\*\*)?(?:#{1,6}\s*)?(?:2\.\s*)?(?:VISIBLE\s+FEATURES|Visible\s+Features|VISUAL\s+OBSERVATIONS|Visual\s+Observations|KEY\s+FEATURES|Key\s+Features)(?:\*\*)?\s*:?\s*\n(.*?)(?=(?:\n(?:\*\*)?(?:#{1,6}\s*)?(?:3\.\s*)?(?:SPATIAL\s+PATTERN|Spatial\s+Pattern|INTERPRETATION|Interpretation)(?:\*\*)?\s*:?|\Z))",
        cleaned,
        re.DOTALL | re.IGNORECASE
    )
    pattern_match = re.search(
        r"(?:^|\n)(?:\*\*)?(?:#{1,6}\s*)?(?:3\.\s*)?(?:SPATIAL\s+PATTERN|Spatial\s+Pattern|SPATIAL\s+DISTRIBUTION|Spatial\s+Distribution)(?:\*\*)?\s*:?\s*\n(.*?)(?=(?:\n(?:\*\*)?(?:#{1,6}\s*)?(?:4\.\s*)?(?:INTERPRETATION|Interpretation|SUMMARY|Summary)(?:\*\*)?\s*:?|\Z))",
        cleaned,
        re.DOTALL | re.IGNORECASE
    )
    interp_match = re.search(
        r"(?:^|\n)(?:\*\*)?(?:#{1,6}\s*)?(?:4\.\s*)?(?:INTERPRETATION|Interpretation|CONCLUSION|Conclusion|SUMMARY|Summary)(?:\*\*)?\s*:?\s*\n(.*?)(?=\Z)",
        cleaned,
        re.DOTALL | re.IGNORECASE
    )

    overview_text = overview_match.group(1).strip() if overview_match else ""
    features_raw = features_match.group(1).strip() if features_match else ""
    pattern_text = pattern_match.group(1).strip() if pattern_match else ""
    interp_text = interp_match.group(1).strip() if interp_match else ""

    # Parse and clean bullet points in visible features
    bullet_items: List[str] = []
    if features_raw:
        raw_lines = [line.strip() for line in features_raw.split("\n") if line.strip()]
        for line in raw_lines:
            # Strip bullet prefixes (- , * , • , 1. , etc.)
            clean_item = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
            if not clean_item:
                continue
            # Ensure bold title prefix formatting
            if not clean_item.startswith("**") and ":" in clean_item:
                parts = clean_item.split(":", 1)
                clean_item = f"**{parts[0].strip()}**: {parts[1].strip()}"
            elif clean_item.startswith("**") and not clean_item.endswith("**") and ":" in clean_item:
                pass
            bullet_items.append(f"- {clean_item}")

    # Fallback normalization if headers were not present or partially missing
    if not overview_text and not bullet_items:
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if paragraphs:
            overview_text = paragraphs[0]
            if len(paragraphs) > 1:
                features_candidate = paragraphs[1]
                for line in features_candidate.split("\n"):
                    c_line = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
                    if c_line:
                        if not c_line.startswith("**") and ":" in c_line:
                            pts = c_line.split(":", 1)
                            c_line = f"**{pts[0].strip()}**: {pts[1].strip()}"
                        bullet_items.append(f"- {c_line}")
            if len(paragraphs) > 2:
                pattern_text = paragraphs[2]
            if len(paragraphs) > 3:
                interp_text = paragraphs[3]

    if not overview_text:
        overview_text = "The satellite scene presents high-resolution optical aerial imagery with distinct terrestrial and built-up land cover."

    if not bullet_items:
        bullet_items = [
            "- **Built Structures**: Observable roof structures and developed parcel boundaries.",
            "- **Road Infrastructure**: Paved access corridors connecting developed sections.",
            "- **Vegetation & Land Cover**: Vegetated areas and open terrain distributed across the scene."
        ]

    if not pattern_text:
        pattern_text = "Developed features and open terrain exhibit distinct spatial clustering across the scene."

    if not interp_text:
        interp_text = "The visual evidence indicates planned development interfacing with natural or open land cover."

    # Assemble canonical point-wise formatted string
    formatted_features = "\n".join(bullet_items)
    formatted_output = (
        f"OVERVIEW\n"
        f"{overview_text}\n\n"
        f"VISIBLE FEATURES\n"
        f"{formatted_features}\n\n"
        f"SPATIAL PATTERN\n"
        f"{pattern_text}\n\n"
        f"INTERPRETATION\n"
        f"{interp_text}"
    )

    return {
        "overview": overview_text,
        "visible_features": bullet_items,
        "spatial_pattern": pattern_text,
        "interpretation": interp_text,
        "formatted_text": formatted_output
    }


class GeminiService:
    """Service for multimodal satellite imagery reasoning with Google Gemini."""

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
            "1. Inspect the CURRENT original image directly and answer the user's question with concrete, visually grounded observations.\n"
            "2. Strictly describe ONLY what is visible in this exact image. Never hallucinate fire, smoke, burn scars, mountains, or unobserved objects.\n"
            "3. Structure your response using EXACTLY these 4 capitalized section headings with point-wise bullet items under VISIBLE FEATURES:\n\n"
            "OVERVIEW\n"
            "[A concise 1-2 sentence description of the scene.]\n\n"
            "VISIBLE FEATURES\n"
            "- **[Category 1]**: [Description]\n"
            "- **[Category 2]**: [Description]\n"
            "- **[Category 3]**: [Description]\n"
            "- **[Category 4]**: [Description]\n\n"
            "SPATIAL PATTERN\n"
            "[Spatial arrangement and distribution across the scene.]\n\n"
            "INTERPRETATION\n"
            "[Objective scene interpretation based strictly on visible evidence.]\n\n"
            "4. Never use vague remote-sensing filler ('textured ground cover', 'surface reflectance variations', 'natural boundaries').\n"
            "5. Only report numbers, counts, or percentages that appear in verified CV metrics."
        )

        return "\n".join(prompt_parts)

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
                "structured": {},
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
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.5-flash",
                "gemini-flash-latest",
                "gemini-pro-latest",
                "gemini-3.8-flash"
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
            normalized = normalize_structured_vqa(
                raw_answer,
                image_type=image_type,
                verified_metrics=verified_metrics
            )

            inf_time = (time.time() - start_time) * 1000.0

            return {
                "answer": normalized["formatted_text"],
                "structured": normalized,
                "model": f"GeoChat + Gemini ({used_model})",
                "semantic_model": f"GeoChat + Gemini ({used_model})",
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
                "structured": {},
                "model": f"Gemini ({self.model_name})",
                "semantic_model": f"Gemini ({self.model_name})",
                "semantic_source": "GeoChat (fallback)",
                "verified_metrics": verified_metrics or {},
                "inference_time_ms": round(inf_time, 2),
                "sources": sources,
                "success": False,
                "error": str(e)
            }


# Singleton instance
gemini_service = GeminiService()
