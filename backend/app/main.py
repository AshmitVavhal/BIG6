import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.system_router import router as system_router
from app.api.vqa_router import router as vqa_router
from app.api.highlight_router import router as highlight_router
from app.api.change_router import router as change_router
from app.api.optical_sar_router import router as optical_sar_router
from app.api.files_router import router as files_router
from app.api.report_router import router as report_router
from app.models.model_manager import model_manager
from app.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Initializing SATQUERY AI Analysis Platform...")
    status = model_manager.get_system_status()
    logger.info(f"Compute Device: {status.device.upper()} (GPU: {status.gpu_name or 'None'})")
    logger.info(f"VRAM Total: {status.vram_total_gb} GB, Free: {status.vram_free_gb} GB")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down SATQUERY AI. Freeing GPU memory...")
    model_manager.clear_vram_cache()

app = FastAPI(
    title="SATQUERY AI API",
    version=settings.VERSION,
    description="AI-powered Satellite Imagery Analysis Platform for Remote-Sensing, Disaster-Management, and GIS Intelligence",
    lifespan=lifespan
)

# CORS configuration for Vite Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers under /api prefix
app.include_router(system_router, prefix=settings.API_PREFIX)
app.include_router(vqa_router, prefix=settings.API_PREFIX)
app.include_router(highlight_router, prefix=settings.API_PREFIX)
app.include_router(change_router, prefix=settings.API_PREFIX)
app.include_router(optical_sar_router, prefix=settings.API_PREFIX)
app.include_router(files_router, prefix=settings.API_PREFIX)
app.include_router(report_router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {
        "platform": "SATQUERY AI",
        "version": settings.VERSION,
        "status": "online",
        "compute_device": model_manager.get_device_name(),
        "endpoints": {
            "vqa": "/api/vqa/analyze",
            "highlight": "/api/highlight/analyze",
            "change": "/api/change/analyze",
            "optical_sar": "/api/optical-sar/analyze",
            "system_status": "/api/system/status",
            "reports_export": "/api/reports/export"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
