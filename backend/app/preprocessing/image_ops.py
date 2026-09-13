import io
import base64
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image

class ImageOps:
    @staticmethod
    def resize_keep_aspect(img: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, float]:
        """Resize image so the longest dimension equals target_size, returning scale factor"""
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        if scale >= 1.0:
            return img, 1.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def pad_to_square(img: np.ndarray, size: int = 1024) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Pad image to square with reflection or black border, returning padding offsets (top, bottom, left, right)"""
        h, w = img.shape[:2]
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded, (top, bottom, left, right)

    @staticmethod
    def apply_color_overlay(
        base_img: np.ndarray,
        mask: np.ndarray,
        color_rgb: Tuple[int, int, int] = (16, 185, 129),
        alpha: float = 0.45
    ) -> np.ndarray:
        """
        Blend a binary mask (H, W) over an RGB base image (H, W, 3) with a designated color.
        """
        h, w = base_img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = base_img.copy()
        mask_bool = mask > 0
        
        if np.any(mask_bool):
            blended = (base_img[mask_bool].astype(np.float32) * (1.0 - alpha) + 
                       np.array(color_rgb, dtype=np.float32) * alpha).clip(0, 255).astype(np.uint8)
            overlay[mask_bool] = blended
            
            # Add a thin luminous boundary around mask contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color_rgb, 2)
        
        return overlay

    @staticmethod
    def draw_bounding_boxes(
        img: np.ndarray,
        boxes: List[List[float]],
        labels: List[str],
        confidences: List[float],
        color: Tuple[int, int, int] = (16, 185, 129)
    ) -> np.ndarray:
        """Draw technical GIS bounding boxes with label tags and confidence badges"""
        annotated = img.copy()
        h, w = img.shape[:2]
        
        for box, label, conf in zip(boxes, labels, confidences):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Draw corner brackets / technical reticle box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Corner accents
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 3:
                # Top-left
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 3)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 3)
                # Bottom-right
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 3)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 3)

            # Label banner
            tag_text = f"{label.upper()} [{conf*100:.0f}%]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(tag_text, font, font_scale, thickness)
            
            # Background pill
            tag_y1 = max(0, y1 - text_h - 6)
            tag_y2 = y1
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, tag_y2), (13, 17, 23), -1)
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, tag_y2), color, 1)
            cv2.putText(annotated, tag_text, (x1 + 3, tag_y2 - 3), font, font_scale, (230, 237, 243), thickness, cv2.LINE_AA)

        return annotated

image_ops = ImageOps()
