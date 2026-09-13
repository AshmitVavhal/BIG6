from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.system import ExecutionStage, GeoMetadata

class ChangedRegion(BaseModel):
    region_id: int
    bounding_box: List[int]  # [x1, y1, x2, y2]
    area_pixels: int
    area_sq_meters: Optional[float] = None
    centroid: List[float]  # [cx, cy]
    confidence: float = 0.88
    confidence_type: str = "model"  # derived from ChangeMamba probability map
    change_type: Optional[str] = "urban_expansion"  # e.g. new_construction, vegetation_loss, water_gain

class ChangeDetectionRequest(BaseModel):
    t1_image_path: str = Field(..., description="Path to T1 (pre-change) satellite image")
    t2_image_path: str = Field(..., description="Path to T2 (post-change) satellite image")
    threshold: float = Field(0.5, ge=0.1, le=0.9, description="Change detection binary decision threshold")
    min_region_area: int = Field(50, description="Minimum pixel area for valid changed component")
    enable_semantic_reasoning: bool = Field(True, description="Enable natural-language semantic reasoning over changed regions")
    semantic_model: Optional[str] = Field("geochat", description="Semantic reasoning model: 'geochat'")

class ChangeDetectionResponse(BaseModel):
    task: str = "bi_temporal_change"
    status: str = "success"
    change_percentage: float = Field(..., description="Calculated pixel-level changed area percentage")
    num_regions: int = Field(..., description="Total count of connected changed components")
    total_changed_pixels: int
    total_pixels: int
    regions: List[ChangedRegion]
    t1_image_url: str
    t2_image_url: str
    mask_url: str
    overlay_url: str
    side_by_side_url: str
    model: str = "ChangeMamba"
    detection_model: str = "ChangeMamba"
    semantic_model: str = "GeoChat"
    device: str = "cuda"
    processing_time_ms: float
    specialized_time_ms: Optional[float] = None
    semantic_time_ms: Optional[float] = None
    is_coregistered: bool
    coregistration_notes: Optional[str] = None
    geo_metadata_t1: Optional[GeoMetadata] = None
    geo_metadata_t2: Optional[GeoMetadata] = None
    semantic_analysis: str
    execution_trace: List[ExecutionStage] = []
    transparency_warning: Optional[str] = (
        "Change masks and percentages are computed authoritatively by ChangeMamba spatiotemporal state space models. "
        "Semantic interpretation is synthesized by GeoChat + Gemini."
    )
