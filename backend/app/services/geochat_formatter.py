"""
GeoChat Prompt and Conversation Formatter for Remote Sensing Tasks
Formats satellite imagery questions and specialized CV outputs into GeoChat-compliant prompts
following strict visual grounding, natural description, and anti-hallucination policies.
"""

from typing import Optional, List, Any, Dict

GEOCHAT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant specializing in satellite and Earth observation imagery. "
    "The assistant provides clear, natural, visually grounded descriptions of what is observable in the satellite scene without inventing numerical percentages or unverified radiometric statistics."
)

class GeoChatFormatter:
    """Formatter for GeoChat conversation templates and domain-specialized prompts."""

    @staticmethod
    def format_vqa_prompt(question: str, geo_context: Optional[dict] = None) -> str:
        """
        Format satellite VQA question for GeoChat with explicit grounding instructions:
        - Natural visual description
        - No invented percentages or radiometric statistics
        - Grounded probabilistic language ("appears to be", "visually consistent with", "likely represents")
        - Answer directly from the image content.
        """
        q_clean = question.strip()
        context_str = ""
        if geo_context and geo_context.get("has_georeference"):
            crs = geo_context.get("crs") or "WGS 84"
            w, h = geo_context.get("width"), geo_context.get("height")
            bands = geo_context.get("count", 3)
            band_desc = f"{bands}-band multispectral" if bands > 3 else "RGB optical"
            context_str = f" [Context: {w}x{h} px, {band_desc}, CRS: {crs}]"

        prompt = (
            f"USER: <image>\n"
            f"You are an Earth observation satellite imagery specialist.{context_str}\n"
            f"Instructions:\n"
            f"1. Provide a natural, visually grounded response based strictly on what is observable in the image.\n"
            f"2. Describe the overall scene, visible terrain, structures, spatial arrangement, and key visual patterns.\n"
            f"3. Do NOT invent land-cover percentages, numerical area estimates, or arbitrary statistics.\n"
            f"4. Do NOT claim unverified multispectral or radiometric properties for standard RGB imagery.\n"
            f"5. Use grounded terminology such as 'appears to be', 'visually consistent with', 'likely represents', or 'no obvious evidence of'.\n\n"
            f"Question: {q_clean}\n"
            f"ASSISTANT:"
        )
        return prompt

    @staticmethod
    def format_change_prompt(
        change_percentage: float,
        num_regions: int,
        regions: List[Any],
        coregistration_notes: Optional[str] = None
    ) -> str:
        """
        Format ChangeMamba quantitative change detection results for GeoChat semantic interpretation.
        GeoChat interprets what the changes visually represent without fabricating new metrics.
        """
        reg_summary = f"{num_regions} distinct connected clusters" if num_regions > 0 else "0 detected change zones"
        notes_str = f" Co-registration verification: {coregistration_notes}." if coregistration_notes else ""
        
        prompt = (
            f"USER: <image>\n"
            f"You are evaluating a satellite bi-temporal change detection pair.\n"
            f"The quantitative ChangeMamba pipeline measured that {change_percentage:.2f}% of the scene area changed across {reg_summary}.{notes_str}\n"
            f"Instructions:\n"
            f"- Explain what these detected changes appear to represent visually (e.g., new building construction, ground disturbance, road development, clearing).\n"
            f"- Describe where changes are spatially concentrated (e.g., central sector, perimeter, linear corridors).\n"
            f"- Reference the quantitative {change_percentage:.2f}% figure as the measured baseline, but do NOT invent other arbitrary percentages or pixel counts.\n"
            f"- Keep the explanation natural, concise, and useful to a human analyst.\n"
            f"ASSISTANT:"
        )
        return prompt

    @staticmethod
    def format_highlight_prompt(
        prompt_query: str,
        num_detections: int,
        total_area_pct: float,
        detections: List[Any]
    ) -> str:
        """
        Format LAE-DINO + Mask2Former detections for GeoChat semantic explanation.
        """
        labels = [getattr(d, "label", prompt_query) for d in detections]
        unique_labels = list(set(labels)) if labels else [prompt_query]
        
        prompt = (
            f"USER: <image>\n"
            f"Object detection (LAE-DINO) and segmentation (Mask2Former) localized {num_detections} instance(s) matching '{', '.join(unique_labels)}' "
            f"occupying approximately {total_area_pct:.2f}% of the scene area.\n"
            f"Instructions:\n"
            f"- Explain the spatial arrangement, visual shape, and operational context of these highlighted features.\n"
            f"- Describe how they relate to the surrounding terrain and infrastructure.\n"
            f"- Do NOT invent arbitrary coordinates or false numerical measurements.\n"
            f"ASSISTANT:"
        )
        return prompt

    @staticmethod
    def format_optical_sar_prompt(
        optical_stats: Any,
        sar_stats: Any,
        question: Optional[str] = None
    ) -> str:
        """
        Format Optical + SAR cross-modal features for GeoChat semantic interpretation.
        """
        high_sar_pct = getattr(sar_stats, "high_backscatter_ratio", 0.0) * 100.0
        low_sar_pct = getattr(sar_stats, "low_backscatter_ratio", 0.0) * 100.0
        sar_db = getattr(sar_stats, "dynamic_range_db", 0.0) or 0.0

        q_part = f" Focused question: {question}." if question else ""

        prompt = (
            f"USER: <image>\n"
            f"Cross-modal satellite analysis combining Optical reflectance and Sentinel-1 SAR microwave backscatter.\n"
            f"Validated SAR metrics: {sar_db:.1f} dB dynamic range, with {high_sar_pct:.1f}% high double-bounce backscatter and {low_sar_pct:.1f}% low specular backscatter.{q_part}\n"
            f"Instructions:\n"
            f"- Provide a clear, natural-language comparison explaining how optical features correspond with radar backscatter signatures.\n"
            f"- Describe bright backscatter areas (structures, vertical facades, metallic infrastructure) vs dark backscatter zones (water, smooth pavement, runways).\n"
            f"- Do NOT claim unverified physical constants or fake radiometric parameters.\n"
            f"ASSISTANT:"
        )
        return prompt

    @staticmethod
    def parse_output(raw_text: str) -> str:
        """Clean and parse the generated text from GeoChat."""
        text = raw_text.strip()
        # Remove any lingering prompt labels if present
        for prefix in ["ASSISTANT:", "Assistant:", "### Assistant:", "### Response:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Stop at termination tokens if present
        for stop_word in ["</s>", "###", "USER:", "User:"]:
            if stop_word in text:
                text = text.split(stop_word)[0].strip()
                
        return text

geochat_formatter = GeoChatFormatter()
