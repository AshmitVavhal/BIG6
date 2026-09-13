from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.system import ExecutionStage, GeoMetadata

class VQARequest(BaseModel):
    image_path: str = Field(..., description="Path or identifier of uploaded satellite image")
    question: str = Field(..., description="User question about the satellite scene")
    include_caption: bool = True
    semantic_model: Optional[str] = Field("geochat", description="Semantic reasoning model: 'geochat' (MBZUAI/GeoChat)")
    parameters: Optional[Dict[str, Any]] = None

class VQAResponse(BaseModel):
    task: str = "vqa"
    status: str = "success"
    question: str
    answer: str
    geochat_observations: Optional[str] = None
    caption: Optional[str] = None
    confidence: float = Field(0.96, description="Semantic reasoning confidence")
    confidence_type: str = "heuristic"  # "model" or "heuristic"
    model: str = "GeoChat"
    semantic_model: str = "GeoChat"
    device: str = "cuda"
    processing_time_ms: float
    specialized_time_ms: Optional[float] = None
    semantic_time_ms: Optional[float] = None
    image_url: str
    geo_metadata: Optional[GeoMetadata] = None
    execution_trace: List[ExecutionStage] = []
    transparency_warning: Optional[str] = None
