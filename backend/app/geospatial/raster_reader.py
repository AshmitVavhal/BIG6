from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from app.utils.logger import logger

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

class RasterReader:
    @staticmethod
    def load_rgb_array(file_path: Path, max_dim: Optional[int] = 2048) -> Tuple[np.ndarray, dict]:
        """
        Load any satellite image format (GeoTIFF, TIFF, PNG, JPG) into a normalized uint8 RGB numpy array (H, W, 3).
        Preserves original dynamic range by applying 2%-98% percentile linear stretch for multi-band satellite data.
        """
        meta = {"original_format": file_path.suffix.lower(), "driver": "standard"}
        
        if RASTERIO_AVAILABLE and file_path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
            try:
                with rasterio.open(file_path) as src:
                    meta["crs"] = str(src.crs) if src.crs else None
                    meta["count"] = src.count
                    meta["transform"] = list(src.transform)
                    meta["driver"] = src.driver
                    
                    # Read bands
                    if src.count >= 3:
                        # Assume first 3 bands are RGB or Red, Green, Blue
                        r = src.read(1)
                        g = src.read(2)
                        b = src.read(3)
                        img_arr = np.dstack((r, g, b))
                    elif src.count == 1:
                        # Single band (SAR backscatter or panchromatic)
                        gray = src.read(1)
                        img_arr = np.dstack((gray, gray, gray))
                    else:
                        r = src.read(1)
                        g = src.read(2)
                        b = np.zeros_like(r)
                        img_arr = np.dstack((r, g, b))

                    # Normalize multi-spectral / 16-bit to 8-bit RGB
                    img_arr = RasterReader._normalize_to_uint8(img_arr)
                    return img_arr, meta
            except Exception as e:
                logger.warning(f"Rasterio read failed for {file_path}, falling back to PIL: {e}")

        # Standard PIL loading
        with Image.open(file_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            meta["count"] = 3
            return arr, meta

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        """Apply 2%-98% percentile contrast stretch and convert to uint8 RGB"""
        if arr.dtype == np.uint8:
            return arr

        arr = arr.astype(np.float32)
        out = np.zeros_like(arr, dtype=np.uint8)
        
        for c in range(arr.shape[2]):
            channel = arr[:, :, c]
            valid_mask = np.isfinite(channel) & (channel > 0)
            if np.any(valid_mask):
                p2 = np.percentile(channel[valid_mask], 2)
                p98 = np.percentile(channel[valid_mask], 98)
                if p98 > p2:
                    stretched = np.clip((channel - p2) / (p98 - p2), 0, 1) * 255.0
                    out[:, :, c] = stretched.astype(np.uint8)
                else:
                    out[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
            else:
                out[:, :, c] = 0
                
        return out

    @staticmethod
    def save_rgb_preview(arr: np.ndarray, output_path: Path) -> Path:
        """Save a uint8 RGB numpy array as a PNG preview"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(arr)
        img.save(output_path, format="PNG", optimize=True)
        return output_path

raster_reader = RasterReader()
