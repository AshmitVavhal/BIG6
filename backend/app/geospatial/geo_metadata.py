from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np
from app.schemas.system import GeoMetadata
from app.utils.logger import logger

try:
    import rasterio
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio is not available. Geospatial metadata will be extracted in basic mode.")

class GeoMetadataExtractor:
    @staticmethod
    def extract(file_path: Path) -> GeoMetadata:
        """Extract geospatial metadata from GeoTIFF / TIFF or fallback to standard raster image metadata"""
        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(file_path) as src:
                    crs_str = None
                    if src.crs:
                        crs_str = src.crs.to_string()
                    
                    bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                    transform = list(src.transform)[:6]
                    res = [abs(src.transform[0]), abs(src.transform[4])]
                    
                    # Sensor tags / metadata tags
                    tags = src.tags()
                    sensor_info = tags.get("SENSOR_ID") or tags.get("PLATFORM") or tags.get("MISSION")
                    acq_date = tags.get("ACQUISITION_DATE") or tags.get("DATETIME")
                    
                    return GeoMetadata(
                        crs=crs_str,
                        bounds=bounds if src.crs else None,
                        transform=transform if src.crs else None,
                        width=src.width,
                        height=src.height,
                        count=src.count,
                        driver=src.driver,
                        nodata=float(src.nodata) if src.nodata is not None else None,
                        resolution=res if src.crs else None,
                        sensor_info=sensor_info,
                        acquisition_date=acq_date,
                        has_georeference=bool(src.crs and src.transform)
                    )
            except Exception as e:
                logger.debug(f"Rasterio read failed for {file_path.name} ({e}), falling back to PIL metadata")
                
        # Standard PIL fallback for JPG / PNG
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                count = len(mode) if mode in ["RGB", "RGBA", "CMYK"] else 1
                return GeoMetadata(
                    crs=None,
                    bounds=None,
                    transform=None,
                    width=width,
                    height=height,
                    count=count,
                    driver="PIL/" + (img.format or "UNKNOWN"),
                    nodata=None,
                    resolution=None,
                    sensor_info=None,
                    acquisition_date=None,
                    has_georeference=False
                )
        except Exception as err:
            logger.error(f"Error reading basic image dimensions for {file_path}: {err}")
            return GeoMetadata(
                width=512,
                height=512,
                count=3,
                driver="UNKNOWN",
                has_georeference=False
            )

geo_extractor = GeoMetadataExtractor()
