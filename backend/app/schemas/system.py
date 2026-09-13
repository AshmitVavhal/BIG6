from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SystemStatusResponse(BaseModel):
    online: bool = True
    device: str = "cpu"
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    ram_percent: float = 0.0
    cpu_percent: float = 0.0
    active_models_count: int = 0
    loaded_models: List[str] = []
    geochat_model: str = "MBZUAI/geochat-7B"
    geochat_status: str = "unloaded"  # "ready" | "unloaded" | "loading" | "error"
    lora_enabled: bool = False
    lora_path: Optional[str] = None
    lora_available: bool = False
    gemini_configured: bool = False
    gemini_model: str = "gemini-3.6-flash"

class ModelStatusItem(BaseModel):
    name: str
    key: str
    loaded: bool
    device: str
    vram_mb: float = 0.0
    task: str
    description: str
    path: str
    ready: bool

class ModelsStatusResponse(BaseModel):
    models: List[ModelStatusItem]
    system_device: str
    cuda_available: bool

class BenchmarkRecord(BaseModel):
    model: str
    task: str
    inference_time_ms: float
    device: str
    image_size: str
    vram_usage_mb: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class ExecutionStage(BaseModel):
    stage: str
    message: str
    timestamp: str
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None

class GeoMetadata(BaseModel):
    crs: Optional[str] = None
    bounds: Optional[List[float]] = None  # [minx, miny, maxx, maxy]
    transform: Optional[List[float]] = None
    width: int
    height: int
    count: int  # band count
    driver: Optional[str] = None
    nodata: Optional[float] = None
    resolution: Optional[List[float]] = None  # [res_x, res_y]
    sensor_info: Optional[str] = None
    acquisition_date: Optional[str] = None
    has_georeference: bool = False
