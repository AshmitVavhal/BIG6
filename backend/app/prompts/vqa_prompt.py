from typing import Optional, Dict, Any

VQA_SYSTEM_PROMPT = """You are SatQuery AI, an expert satellite imagery and remote sensing analyst workstation assistant.
Your role is to analyze high-resolution satellite imagery, multispectral scenes, and geospatial features with scientific precision.

Instructions:
1. Answer the user's question directly, concisely, and accurately based strictly on what is visible in the provided satellite scene.
2. Identify land-cover types, urban infrastructure, vegetation, water bodies, road networks, or terrain features with domain-specific remote-sensing terminology.
3. If geospatial metadata (such as CRS, pixel resolution, or band count) is provided, incorporate it when relevant.
4. Do NOT hallucinate fine details not discernable in the image. If an observation is ambiguous or uncertain due to resolution, explicitly state the limitation.
5. Provide a clear, natural-language response.
"""

def build_vqa_prompt(question: str, geo_context: Optional[Dict[str, Any]] = None) -> str:
    parts = []
    if geo_context:
        meta_summary = []
        if geo_context.get("crs"):
            meta_summary.append(f"CRS: {geo_context['crs']}")
        if geo_context.get("width") and geo_context.get("height"):
            meta_summary.append(f"Dimensions: {geo_context['width']}x{geo_context['height']} px")
        if geo_context.get("count"):
            meta_summary.append(f"Bands: {geo_context['count']}")
        if geo_context.get("resolution"):
            meta_summary.append(f"GSD Resolution: {geo_context['resolution']}")
        if geo_context.get("sensor_info"):
            meta_summary.append(f"Sensor: {geo_context['sensor_info']}")
        if meta_summary:
            parts.append(f"[Geospatial Metadata: {', '.join(meta_summary)}]")

    parts.append(f"Question: {question}")
    return "\n".join(parts)
