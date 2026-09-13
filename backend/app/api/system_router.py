from fastapi import APIRouter
from app.models.model_manager import model_manager
from app.schemas.system import SystemStatusResponse, ModelsStatusResponse
from app.utils.benchmark_tracker import benchmark_tracker

router = APIRouter(prefix="", tags=["System"])

@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    return model_manager.get_system_status()

@router.get("/models/status", response_model=ModelsStatusResponse)
async def get_models_status():
    return model_manager.get_models_status()

@router.post("/system/clear-cache")
async def clear_system_cache():
    model_manager.clear_vram_cache()
    return {"status": "success", "message": "VRAM cache cleared and garbage collected."}

@router.get("/benchmarks")
async def get_benchmarks():
    return {
        "summary": benchmark_tracker.get_summary(),
        "recent_runs": benchmark_tracker.get_records()[-20:]
    }
