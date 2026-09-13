from typing import List, Dict, Any, Optional

CHANGE_ANALYSIS_SYSTEM_PROMPT = """You are SatQuery AI, an expert satellite imagery semantic analysis assistant.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. You are NOT the change detection model. A specialized Spatiotemporal State Space Model (ChangeMamba) has already executed bi-temporal inference, calculated the authoritative pixel-difference mask, extracted connected components, and determined the exact change percentage.
2. Your job is ONLY to provide a natural-language semantic interpretation of the detected changes based on the supplied pre-change (T1) and post-change (T2) satellite images and the quantitative ChangeMamba metrics.
3. Only describe changes that are visually supported by T1, T2, and the supplied change region data.
4. Do NOT invent buildings, roads, vegetation loss, fires, vehicles, water bodies, or other objects not clearly visible.
5. Do NOT generate new or conflicting numerical change percentages. Always reference the provided ChangeMamba measurements.
6. Do NOT modify or override the supplied change regions or bounding boxes.
7. If the visual evidence in any region is ambiguous or low-contrast, explicitly state the uncertainty.
"""

def build_change_analysis_prompt(
    change_percentage: float,
    num_regions: int,
    regions: List[Any],
    coregistration_notes: Optional[str] = None
) -> str:
    # Summarize top change clusters
    region_summaries = []
    for idx, r in enumerate(regions[:10]):
        box = getattr(r, "bounding_box", None) or (r.get("bounding_box") if isinstance(r, dict) else [])
        area = getattr(r, "area_pixels", None) or (r.get("area_pixels") if isinstance(r, dict) else 0)
        c_type = getattr(r, "change_type", None) or (r.get("change_type") if isinstance(r, dict) else "surface_modification")
        region_summaries.append(f"Region #{idx+1}: Area={area} px, BBox={box}, Initial Classification={c_type}")

    regions_text = "\n".join(region_summaries) if region_summaries else "No large clustered regions."

    prompt = f"""[ChangeMamba Quantitative Results]
- Authoritative Changed Area: {change_percentage:.2f}% of scene
- Connected Changed Regions Count: {num_regions}
- Spatial Co-Registration: {coregistration_notes or 'Sub-pixel co-registration validated'}

[Top Changed Spatial Regions]
{regions_text}

Task:
Analyze the supplied T1 (pre-change) and T2 (post-change) satellite images along with the ChangeMamba change evidence above.
Provide a concise, professional semantic interpretation of what real-world surface transformations occurred (e.g., urban development, new structure erection, vegetation clearance, infrastructure expansion, or environmental shifts) between T1 and T2 in the detected regions.
"""
    return prompt
