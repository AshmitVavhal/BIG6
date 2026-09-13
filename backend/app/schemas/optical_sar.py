from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.system import ExecutionStage, GeoMetadata

class OpticalSARRequest(BaseModel):
    optical_image_path: str = Field(..., description="Path to optical RGB satellite image")
    sar_image_path: str = Field(..., description="Path to SAR (Sentinel-1 style radar backscatter) image")
    question: Optional[str] = Field("Compare the optical and SAR imagery, identifying features visible across both modalities and radar-penetrating structures.", description="Query or focus area")
    despeckle_sar: bool = Field(True, description="Apply Lee despeckle filter to SAR")
    extract_features: bool = Field(True, description="Compute backscatter statistics and texture maps")
    semantic_model: Optional[str] = Field("geochat", description="Semantic reasoning model: 'geochat'")

class ModalityFeatureStats(BaseModel):
    modality: str
    mean_intensity: float
    std_intensity: float
    dynamic_range_db: Optional[float] = None
    high_backscatter_ratio: Optional[float] = None  # For metallic/corner-reflector structures
    low_backscatter_ratio: Optional[float] = None  # For calm water bodies / smooth runways

class OpticalSARResponse(BaseModel):
    task: str = "optical_sar_fusion"
    status: str = "success"
    optical_image_url: str
    sar_image_url: str
    fused_image_url: str
    sar_filtered_url: str
    difference_heatmap_url: str
    optical_stats: ModalityFeatureStats
    sar_stats: ModalityFeatureStats
    detected_features: List[str]
    cross_modal_analysis: str
    model: str = "Multimodal Feature Pipeline + GeoChat"
    optical_processing: str = "Visible Reflectance & Dynamic Contrast Normalization"
    sar_processing: str = "7x7 Lee Speckle Filter & CFAR Signal Engine"
    semantic_model: str = "GeoChat"
    device: str = "cuda"
    processing_time_ms: float
    specialized_time_ms: Optional[float] = None
    semantic_time_ms: Optional[float] = None
    geo_metadata_optical: Optional[GeoMetadata] = None
    geo_metadata_sar: Optional[GeoMetadata] = None
    execution_trace: List[ExecutionStage] = []
    transparency_warning: str = (
        "Semantic interpretation provided by GeoChat remote-sensing VLM. "
        "Physical radar backscatter statistics (dB, speckle metrics) are computed deterministically via SAR signal algorithms."
    )
