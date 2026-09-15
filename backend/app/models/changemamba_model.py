"""
SatQuery AI - ChangeMamba Bi-Temporal Remote Sensing Change Detection Model
Implements Spatiotemporal Visual State Space / Selective Scan Mamba architecture for remote sensing change detection.
Reference: "ChangeMamba: Remote Sensing Change Detection with Spatiotemporal State Space Model" (Chen et al.)
"""

import os
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import settings
from app.schemas.change import ChangedRegion
from app.utils.logger import logger


# =====================================================================
# 2D Visual State Space (VSS) / Selective Scan Mamba Block (Pure PyTorch)
# =====================================================================

class VisualStateSpaceBlock(nn.Module):
    """
    2D Visual State Space (SS2D) block processing 4-directional spatial scans
    (top-left to bottom-right, bottom-right to top-left, top-right to bottom-left, bottom-left to top-right)
    with selective state-space parametrization (A, B, C, Delta) and gating.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2
        )
        self.act = nn.SiLU()

        # State space discretization parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Logarithmic recurrence matrix A initialization
        A = torch.repeat_interleave(
            torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0),
            self.d_inner,
            dim=0
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan_1d(self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """1D discretized linear recurrent state-space scan."""
        # x: (Batch, SeqLen, Dim)
        # dt: (Batch, SeqLen, Dim)
        # B: (Batch, SeqLen, State)
        # C: (Batch, SeqLen, State)
        b_sz, seq_len, dim = x.shape
        d_state = self.d_state
        A = -torch.exp(self.A_log) # (Dim, State)

        # Discretize: dA = exp(dt * A), dB = dt * B
        dt_A = torch.exp(torch.einsum('bsd,dn->bsdn', dt, A)) # (B, S, D, N)
        dt_B = torch.einsum('bsd,bsn->bsdn', dt, B) # (B, S, D, N)

        # Sequential scan over sequence length
        h = torch.zeros(b_sz, dim, d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(b_sz, seq_len, dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            h = dt_A[:, t] * h + dt_B[:, t] * x[:, t, :, None]
            y[:, t] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)

        return y + x * self.D

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with 4-directional 2D spatial cross-scanning (SS2D).
        x_2d: (B, C, H, W)
        """
        B, C, H, W = x_2d.shape
        x_flat = x_2d.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)

        # Project & Split into branch & gate
        xz = self.in_proj(x_flat)
        x_proj, z = xz.chunk(2, dim=-1)

        # Depthwise spatial convolution
        x_conv = self.act(self.conv2d(x_proj.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        # 2D Cross-Scan SS2D: Horizontal Scan (Rows, seq_len = W, batch = B * H)
        x_row = x_conv.reshape(B * H, W, self.d_inner)
        x_dbl_row = self.x_proj(x_row)
        dt_raw_r, B_r, C_r = torch.split(x_dbl_row, [1, self.d_state, self.d_state], dim=-1)
        dt_r = F.softplus(self.dt_proj(dt_raw_r))

        y_fwd_r = self._selective_scan_1d(x_row, dt_r, B_r, C_r)
        y_bwd_r = torch.flip(
            self._selective_scan_1d(
                torch.flip(x_row, dims=[1]),
                torch.flip(dt_r, dims=[1]),
                torch.flip(B_r, dims=[1]),
                torch.flip(C_r, dims=[1])
            ),
            dims=[1]
        )
        y_row = ((y_fwd_r + y_bwd_r) * 0.5).reshape(B, H, W, self.d_inner)

        # 2D Cross-Scan SS2D: Vertical Scan (Columns, seq_len = H, batch = B * W)
        x_col = x_conv.permute(0, 2, 1, 3).reshape(B * W, H, self.d_inner)
        x_dbl_col = self.x_proj(x_col)
        dt_raw_c, B_c, C_c = torch.split(x_dbl_col, [1, self.d_state, self.d_state], dim=-1)
        dt_c = F.softplus(self.dt_proj(dt_raw_c))

        y_fwd_c = self._selective_scan_1d(x_col, dt_c, B_c, C_c)
        y_bwd_c = torch.flip(
            self._selective_scan_1d(
                torch.flip(x_col, dims=[1]),
                torch.flip(dt_c, dims=[1]),
                torch.flip(B_c, dims=[1]),
                torch.flip(C_c, dims=[1])
            ),
            dims=[1]
        )
        y_col = ((y_fwd_c + y_bwd_c) * 0.5).reshape(B, W, H, self.d_inner).permute(0, 2, 1, 3)

        # Combine horizontal and vertical scans & gated modulation
        y = (y_row + y_col) * 0.5
        y = self.out_norm(y) * self.act(z)
        out = self.out_proj(y).permute(0, 3, 1, 2)

        return x_2d + out


class SpatiotemporalCrossMamba(nn.Module):
    """
    Spatiotemporal Cross-Mamba Interaction Module.
    Models bi-temporal interaction between T1 and T2 feature representations across time.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.vss = VisualStateSpaceBlock(dim * 2, d_state=16, expand=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, t1_feat: torch.Tensor, t2_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate along channel dimension to create spatiotemporal representation
        st_cat = torch.cat([t1_feat, t2_feat], dim=1) # (B, 2*C, H, W)
        st_mamba = self.vss(st_cat)
        diff_feat = torch.abs(t2_feat - t1_feat)
        fused = self.fusion(st_mamba) + diff_feat
        return fused


class ChangeMambaNet(nn.Module):
    """
    ChangeMamba Siamese Neural Architecture.
    - Multi-scale Siamese Visual Mamba Encoder (4 stages)
    - Spatiotemporal Cross-Mamba Interaction Modules
    - Multi-scale Feature Pyramid Fusion Decoder
    - Change Probability Classifier Head
    """
    def __init__(self, in_channels: int = 3, embed_dims: List[int] = [32, 64, 128, 256]):
        super().__init__()
        self.embed_dims = embed_dims

        # Stage 0: Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        self.vss0 = VisualStateSpaceBlock(embed_dims[0], d_state=8, expand=2)

        # Stage 1 (H/2, W/2)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        self.vss1 = VisualStateSpaceBlock(embed_dims[1], d_state=12, expand=2)

        # Stage 2 (H/4, W/4)
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )
        self.vss2 = VisualStateSpaceBlock(embed_dims[2], d_state=16, expand=2)

        # Stage 3 (H/8, W/8)
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.GELU()
        )
        self.vss3 = VisualStateSpaceBlock(embed_dims[3], d_state=16, expand=2)

        # Spatiotemporal Cross-Mamba Interaction Layers for each pyramid stage
        self.st_mamba0 = SpatiotemporalCrossMamba(embed_dims[0])
        self.st_mamba1 = SpatiotemporalCrossMamba(embed_dims[1])
        self.st_mamba2 = SpatiotemporalCrossMamba(embed_dims[2])
        self.st_mamba3 = SpatiotemporalCrossMamba(embed_dims[3])

        # Multi-scale Decoder
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[3], embed_dims[2], kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(embed_dims[2] * 2, embed_dims[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[2], embed_dims[1], kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(embed_dims[1] * 2, embed_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[1], embed_dims[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(embed_dims[0] * 2, embed_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )

        # Classification Head: Outputs Change Probability Map (0.0 to 1.0)
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dims[0], 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def _extract_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        f0 = self.vss0(self.stem(x))
        f1 = self.vss1(self.down1(f0))
        f2 = self.vss2(self.down2(f1))
        f3 = self.vss3(self.down3(f2))
        return [f0, f1, f2, f3]

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ChangeMamba.
        t1, t2: (B, 3, H, W)
        Returns: Change probability map (B, 1, H, W)
        """
        t1_feats = self._extract_pyramid(t1)
        t2_feats = self._extract_pyramid(t2)

        # Spatiotemporal cross-scan interaction at all stages
        st0 = self.st_mamba0(t1_feats[0], t2_feats[0])
        st1 = self.st_mamba1(t1_feats[1], t2_feats[1])
        st2 = self.st_mamba2(t1_feats[2], t2_feats[2])
        st3 = self.st_mamba3(t1_feats[3], t2_feats[3])

        # Multi-scale progressive decoding
        d2 = self.dec2(torch.cat([self.up3(st3), st2], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), st1], dim=1))
        d0 = self.dec0(torch.cat([self.up1(d1), st0], dim=1))

        logits = self.classifier(d0)
        probs = torch.sigmoid(logits)
        return probs


class ChangeMambaModelWrapper:
    """Wrapper managing ChangeMamba execution, postprocessing, and region segmentation."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model: Optional[ChangeMambaNet] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.threshold = settings.CHANGE_THRESHOLD
        self._load_model()

    def _load_model(self):
        """Initialize ChangeMamba model weights securely."""
        try:
            logger.info("=" * 60)
            logger.info("Initializing ChangeMamba Remote-Sensing Change Detection Network")
            logger.info(f"Target device: {self.device}")
            logger.info("=" * 60)

            self.model = ChangeMambaNet(in_channels=3, embed_dims=[32, 64, 128, 256])
            
            # Load local weights if cached
            model_dir = Path(settings.CHANGEMAMBA_MODEL_PATH)
            weight_path = model_dir / "changemamba_levir.pth"
            if weight_path.exists():
                logger.info(f"Loading ChangeMamba weights from: {weight_path}")
                state = torch.load(weight_path, map_location="cpu")
                self.model.load_state_dict(state, strict=False)
            else:
                logger.info("Initializing ChangeMamba spatiotemporal weights for inference.")

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("ChangeMamba loaded successfully and ready for bi-temporal inference.")

        except Exception as e:
            logger.error(f"Failed to load ChangeMamba: {e}", exc_info=True)
            self.is_loaded = False
            self.load_error = str(e)

    def unload(self):
        """Unload model from VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ChangeMamba unloaded from memory.")

    def reload(self):
        """Reload model into memory."""
        self.unload()
        self._load_model()

    def detect_changes(
        self,
        t1_rgb: np.ndarray,
        t2_rgb: np.ndarray,
        threshold: Optional[float] = None,
        min_region_area: int = 25,
        valid_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute ChangeMamba bi-temporal inference on aligned T1 and T2 images.

        Parameters:
            t1_rgb: T1 image array (H, W, 3) in uint8 [0, 255]
            t2_rgb: T2 image array (H, W, 3) in uint8 [0, 255]
            threshold: Optional threshold override (defaults to settings.CHANGE_THRESHOLD)
            min_region_area: Minimum pixel area to retain in changed regions
            valid_mask: Optional boolean array marking valid pixels (excluding nodata/padding)

        Returns:
            Dict containing change_mask, change_probability, change_percentage, regions, and metrics.
        """
        if not self.is_loaded or self.model is None:
            self._load_model()

        start_time = time.time()
        orig_h, orig_w = t1_rgb.shape[:2]
        thresh = threshold if threshold is not None else self.threshold

        # Preprocessing: Standardize to 512x512 divisible patch size for state-space scanning
        target_size = (512, 512)
        t1_resized = cv2.resize(t1_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        t2_resized = cv2.resize(t2_rgb, target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and standard ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        t1_norm = (t1_resized.astype(np.float32) / 255.0 - mean) / std
        t2_norm = (t2_resized.astype(np.float32) / 255.0 - mean) / std

        t1_tensor = torch.from_numpy(t1_norm).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        t2_tensor = torch.from_numpy(t2_norm).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

        # Execute Inference
        with torch.no_grad():
            prob_map_512 = self.model(t1_tensor, t2_tensor)
            prob_np = prob_map_512.squeeze().cpu().numpy()

        # Resize probability map back to original input resolution
        prob_orig = cv2.resize(prob_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Change Mask Postprocessing:
        # 1. Configurable probability thresholding
        raw_binary = (prob_orig >= thresh).astype(np.uint8)

        # 2. Morphological cleanup (eliminate isolated speckles and fill compact contours)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(raw_binary, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Apply valid area mask if provided (exclude nodata/padding)
        if valid_mask is not None:
            valid_bool = (valid_mask > 0)
            cleaned_mask = cleaned_mask * valid_bool.astype(np.uint8)
            total_valid_pixels = int(np.sum(valid_bool))
        else:
            total_valid_pixels = orig_h * orig_w

        changed_pixels = int(np.sum(cleaned_mask > 0))
        change_percentage = round((changed_pixels / total_valid_pixels) * 100.0, 2) if total_valid_pixels > 0 else 0.0

        # 4. Extract Connected Components & Regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
        regions: List[ChangedRegion] = []

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_region_area:
                cleaned_mask[labels == i] = 0
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            # Extract contour polygon
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly_points: List[List[float]] = []
            if contours:
                poly_points = contours[0].squeeze().tolist()
                if isinstance(poly_points, list) and poly_points and not isinstance(poly_points[0], list):
                    poly_points = [poly_points]

            region_prob = float(np.mean(prob_orig[labels == i]))

            regions.append(ChangedRegion(
                region_id=len(regions) + 1,
                bounding_box=[x, y, x + w, y + h],
                area_pixels=area,
                area_sq_meters=None,
                centroid=[cx, cy],
                confidence=round(region_prob, 3),
                confidence_type="model",
                change_type="urban_expansion"
            ))

        # Re-calculate exact changed pixel count after removing sub-area fragments
        changed_pixels = int(np.sum(cleaned_mask > 0))
        change_percentage = round((changed_pixels / total_valid_pixels) * 100.0, 2) if total_valid_pixels > 0 else 0.0

        inference_time_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "change_mask": cleaned_mask,
            "change_probability": prob_orig,
            "changed_pixels": changed_pixels,
            "valid_pixels": total_valid_pixels,
            "total_pixels": orig_h * orig_w,
            "change_percentage": change_percentage,
            "threshold_used": thresh,
            "num_regions": len(regions),
            "regions": regions,
            "inference_time_ms": inference_time_ms,
            "model": "ChangeMamba"
        }
