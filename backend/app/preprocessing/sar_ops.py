from typing import Tuple, Dict, Any, List
import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from app.schemas.optical_sar import ModalityFeatureStats
from app.utils.logger import logger

class SARProcessor:
    @staticmethod
    def lee_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
        """
        Apply Lee Speckle Filter to SAR amplitude/intensity image.
        Lee filter preserves edges while smoothing homogeneous radar speckle areas.
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        # Mean and variance in local window
        img_mean = uniform_filter(gray, (size, size))
        img_sqr_mean = uniform_filter(gray**2, (size, size))
        img_variance = np.maximum(0, img_sqr_mean - img_mean**2)

        # Overall noise variance estimate
        overall_variance = np.var(gray)
        if overall_variance == 0:
            return img

        # Weight calculation k = Var / (Var + NoiseVar)
        weights = img_variance / (img_variance + overall_variance + 1e-8)
        weights = np.clip(weights, 0, 1)

        filtered = img_mean + weights * (gray - img_mean)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        if img.ndim == 3:
            return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
        return filtered

    @staticmethod
    def compute_stats(img: np.ndarray, modality: str = "SAR") -> ModalityFeatureStats:
        """Compute statistical and physical radar backscatter parameters"""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # Dynamic range in decibels (dB relative to min positive intensity)
        min_pos = np.min(gray[gray > 0]) if np.any(gray > 0) else 1.0
        max_val = np.max(gray)
        dynamic_range_db = float(10 * np.log10((max_val + 1e-6) / (min_pos + 1e-6)))

        # High backscatter ratio (strong metallic / double-bounce scatterers e.g. buildings, bridges, ships)
        high_thresh = np.percentile(gray, 92)
        high_ratio = float(np.sum(gray >= high_thresh) / gray.size)

        # Low backscatter ratio (smooth specular scatterers e.g. calm water bodies, paved runways)
        low_thresh = np.percentile(gray, 15)
        low_ratio = float(np.sum(gray <= low_thresh) / gray.size)

        return ModalityFeatureStats(
            modality=modality,
            mean_intensity=round(mean_val, 2),
            std_intensity=round(std_val, 2),
            dynamic_range_db=round(dynamic_range_db, 2),
            high_backscatter_ratio=round(high_ratio, 4),
            low_backscatter_ratio=round(low_ratio, 4)
        )

    @staticmethod
    def generate_fusion_layers(optical_img: np.ndarray, sar_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate:
        1. Fused False-Color Composite (Optical RGB modulated by SAR high-frequency backscatter)
        2. Cross-modal Difference Heatmap
        """
        h, w = optical_img.shape[:2]
        if sar_img.shape[:2] != (h, w):
            sar_img = cv2.resize(sar_img, (w, h), interpolation=cv2.INTER_LINEAR)

        sar_gray = cv2.cvtColor(sar_img, cv2.COLOR_RGB2GRAY) if sar_img.ndim == 3 else sar_img
        sar_filtered = SARProcessor.lee_filter(sar_gray)

        # High-frequency structural enhancement from SAR
        sar_norm = sar_filtered.astype(np.float32) / 255.0
        optical_float = optical_img.astype(np.float32)

        # Fuse SAR intensity into lightness channel (HSV / LAB blend)
        hsv = cv2.cvtColor(optical_img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(0.6 * hsv[:, :, 2] + 0.4 * (sar_norm * 255.0), 0, 255)
        fused = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Cross-modal difference heatmap (where Optical visibility differs significantly from SAR backscatter)
        opt_gray = cv2.cvtColor(optical_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        diff = np.abs(opt_gray - sar_filtered.astype(np.float32))
        diff_norm = np.clip((diff / np.percentile(diff, 98)) * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return fused, heatmap

sar_processor = SARProcessor()
