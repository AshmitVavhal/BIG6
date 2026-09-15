import os
import uuid
import shutil
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException
from app.config import settings
from app.utils.logger import logger

class FileManager:
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        # Prevent directory traversal
        base = os.path.basename(filename)
        return "".join([c for c in base if c.isalnum() or c in "._-"])

    @staticmethod
    async def save_upload(file: UploadFile) -> Tuple[str, Path]:
        ext = Path(file.filename or "image.png").suffix.lower()
        if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
            )

        unique_id = uuid.uuid4().hex[:12]
        safe_name = f"{unique_id}_{FileManager.sanitize_filename(file.filename or 'upload')}"
        dest_path = settings.UPLOAD_PATH / safe_name
        
        # Stream file to disk while checking max size
        total_size = 0
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024): # 1MB chunks
                total_size += len(chunk)
                if total_size > settings.MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    if dest_path.exists():
                        dest_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f}MB"
                    )
                buffer.write(chunk)
                
        logger.info(f"Saved uploaded file: {dest_path.name} ({total_size / 1024:.1f} KB)")
        return safe_name, dest_path

    @staticmethod
    def resolve_image_path(path_or_name: str) -> Path:
        """Resolve a filename or path safely to an existing file"""
        # If absolute and exists
        p = Path(path_or_name)
        if p.is_file() and p.exists():
            return p

        # If relative to project base dir
        base_cand = settings.BASE_DIR / path_or_name
        if base_cand.is_file() and base_cand.exists():
            return base_cand
        
        # Check direct match in uploads
        upload_candidate = settings.UPLOAD_PATH / path_or_name
        if upload_candidate.exists():
            return upload_candidate
            
        # Check direct match in samples
        sample_candidate = settings.SAMPLES_PATH / path_or_name
        if sample_candidate.exists():
            return sample_candidate

        # Common sample aliases mapping
        sample_aliases = {
            "bitemporal_t1_2024.png": "Bi_Temporal T1.png",
            "bitemporal_t1.png": "Bi_Temporal T1.png",
            "bitemporal_t1": "Bi_Temporal T1.png",
            "bitemporal_t2_2026.png": "Bi_Temporal T2.png",
            "bitemporal_t2.png": "Bi_Temporal T2.png",
            "bitemporal_t2": "Bi_Temporal T2.png",
            "optical_multispectral.png": "optical.png",
            "optical_multispectral": "optical.png",
            "sar_sentinel1.png": "sar.png",
            "sar_sentinel1": "sar.png",
            "sample_geotiff_sac_scene.tif": "VQA1.png"
        }
        clean_key = path_or_name.lower().strip()
        for alias, real_file in sample_aliases.items():
            if clean_key == alias.lower() or clean_key == alias.lower().replace(".png", ""):
                real_cand = settings.SAMPLES_PATH / real_file
                if real_cand.exists():
                    return real_cand

        # Check by basename in samples
        basename = p.name
        if (settings.SAMPLES_PATH / basename).exists():
            return settings.SAMPLES_PATH / basename

        # Check by basename in uploads
        if (settings.UPLOAD_PATH / basename).exists():
            return settings.UPLOAD_PATH / basename

        # Check direct match in outputs
        output_candidate = settings.OUTPUT_PATH / path_or_name
        if output_candidate.exists():
            return output_candidate
        if (settings.OUTPUT_PATH / basename).exists():
            return settings.OUTPUT_PATH / basename

        # Check direct match in cache
        cache_candidate = settings.CACHE_PATH / path_or_name
        if cache_candidate.exists():
            return cache_candidate
        if (settings.CACHE_PATH / basename).exists():
            return settings.CACHE_PATH / basename

        # Search for partial/suffix match in uploads
        if settings.UPLOAD_PATH.exists():
            for f in os.listdir(settings.UPLOAD_PATH):
                if f.endswith(basename) or basename in f:
                    candidate = settings.UPLOAD_PATH / f
                    if candidate.is_file():
                        return candidate

        # Search for partial/suffix match in samples
        if settings.SAMPLES_PATH.exists():
            for root, _, files in os.walk(settings.SAMPLES_PATH):
                for f in files:
                    if f.endswith(basename) or basename in f:
                        return Path(root) / f

        raise FileNotFoundError(f"Image resource '{path_or_name}' not found in uploads, outputs, or samples.")

file_manager = FileManager()
