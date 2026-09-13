import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.config import settings
from app.utils.file_manager import file_manager
from app.geospatial.geo_metadata import geo_extractor
from app.geospatial.raster_reader import raster_reader
from app.utils.logger import logger

router = APIRouter(prefix="/files", tags=["Files & Imagery"])

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename, saved_path = await file_manager.save_upload(file)
        geo_meta = geo_extractor.extract(saved_path)
        
        # Pre-generate web preview if it is a GeoTIFF / TIFF
        if saved_path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
            cached_preview = settings.CACHE_PATH / f"{saved_path.stem}_preview.png"
            if not cached_preview.exists():
                arr, _ = raster_reader.load_rgb_array(saved_path)
                raster_reader.save_rgb_preview(arr, cached_preview)

        return {
            "status": "success",
            "filename": filename,
            "filepath": str(saved_path),
            "file_url": f"/api/files/view/{filename}",
            "geo_metadata": geo_meta
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/samples")
async def list_sample_datasets():
    """List pre-packaged satellite scenes available for immediate 1-click testing"""
    samples = []
    if settings.SAMPLES_PATH.exists():
        for f in os.listdir(settings.SAMPLES_PATH):
            p = settings.SAMPLES_PATH / f
            if p.is_file() and p.suffix.lower() in settings.ALLOWED_IMAGE_EXTENSIONS:
                meta = geo_extractor.extract(p)
                samples.append({
                    "filename": f,
                    "filepath": str(p),
                    "file_url": f"/api/files/view/{f}",
                    "size_bytes": p.stat().st_size,
                    "geo_metadata": meta
                })
    return {"samples": samples}

@router.get("/view/{filename}")
async def view_image(filename: str):
    try:
        path = file_manager.resolve_image_path(filename)
        ext = path.suffix.lower()

        # Web browsers cannot render TIFF/GeoTIFF natively.
        # Dynamically convert to contrast-stretched RGB PNG preview with caching.
        if ext in [".tif", ".tiff", ".geotiff"]:
            cached_preview = settings.CACHE_PATH / f"{path.stem}_preview.png"
            if not cached_preview.exists() or cached_preview.stat().st_mtime < path.stat().st_mtime:
                logger.info(f"Generating web PNG preview for GeoTIFF: {path.name}")
                arr, _ = raster_reader.load_rgb_array(path)
                raster_reader.save_rgb_preview(arr, cached_preview)
            return FileResponse(cached_preview, media_type="image/png")

        media_type = "image/png"
        if ext in [".jpg", ".jpeg"]:
            media_type = "image/jpeg"
        elif ext == ".png":
            media_type = "image/png"
            
        return FileResponse(path, media_type=media_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found.")
    except Exception as e:
        logger.error(f"Error serving image view '{filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Error rendering image: {str(e)}")

@router.get("/download/{filename}")
async def download_file(filename: str):
    try:
        path = file_manager.resolve_image_path(filename)
        return FileResponse(path, filename=filename, media_type="application/octet-stream")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
