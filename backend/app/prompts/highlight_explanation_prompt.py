from typing import List, Any

HIGHLIGHT_EXPLANATION_SYSTEM_PROMPT = """You are SatQuery AI, an expert satellite imagery and object detection reasoning assistant.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. LAE-DINO (Locate Anything on Earth) zero-shot remote-sensing object detector and Mask2Former universal segmenter have already detected, bounded, and segmented the target objects.
2. All bounding boxes, pixel masks, and area percentages are authoritative and must NOT be altered.
3. Your role is to provide a concise semantic summary and spatial distribution explanation of the detected objects across the satellite scene.
4. Do NOT claim objects were missed or invent additional objects outside the detected bounding boxes.
"""

def build_highlight_explanation_prompt(
    prompt: str,
    num_detections: int,
    total_area_pct: float,
    detections: List[Any]
) -> str:
    det_summaries = []
    for d in detections[:8]:
        lbl = getattr(d, "label", "object")
        conf = getattr(d, "confidence", 0.0)
        area = getattr(d, "area_pixels", 0)
        det_summaries.append(f"- {lbl.upper()} (Confidence: {conf*100:.1f}%, Area: {area:,} px)")

    det_text = "\n".join(det_summaries) if det_summaries else "No detections above threshold."

    text = f"""[Specialized Detection Results (LAE-DINO + Mask2Former)]
- Target Prompt: "{prompt}"
- Total Instances Segmented: {num_detections}
- Scene Coverage Area: {total_area_pct:.2f}%
- Sample Detections:
{det_text}

Task:
Describe the spatial distribution, clustering, orientation, and characteristics of these detected "{prompt}" objects in the satellite scene.
"""
    return text
