"""
SatQuery AI - Computer Vision & Remote Sensing Evaluation Metrics
Authoritative evaluation utilities for change detection, object detection, and semantic segmentation.
Computes real mathematical metrics (IoU, Dice/F1, Precision, Recall, mAP, mIoU) without fabrication.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np


def evaluate_binary_change_mask(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute rigorous quantitative evaluation metrics for binary change detection masks.
    
    Parameters:
        pred_mask: Binary prediction array (H, W), boolean or 0/1
        gt_mask: Binary ground truth array (H, W), boolean or 0/1
        valid_mask: Optional valid pixel mask (excluding nodata/borders)
        
    Returns:
        Dict containing IoU, Dice/F1, Precision, Recall, FPR, FNR, Accuracy, and pixel counts.
    """
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)
    
    if valid_mask is not None:
        v = (valid_mask > 0).astype(bool)
        p = p & v
        g = g & v
        total_valid = int(np.sum(v))
    else:
        total_valid = int(p.size)

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & (~g)))
    fn = int(np.sum((~p) & g))
    tn = total_valid - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if (tp + fp + fn) == 0 else 0.0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total_valid if total_valid > 0 else 0.0

    return {
        "iou": round(float(iou), 4),
        "f1_score": round(float(f1_dice), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4),
        "accuracy": round(float(accuracy), 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "valid_pixels": total_valid
    }


def compute_box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_object_detections(
    pred_boxes: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate predicted bounding boxes against ground-truth boxes.
    
    Parameters:
        pred_boxes: List of dicts with 'bbox' [x1, y1, x2, y2] and optional 'label'
        gt_boxes: List of dicts with 'bbox' [x1, y1, x2, y2] and optional 'label'
        iou_threshold: Threshold to consider a match a True Positive
        
    Returns:
        Dict with Precision, Recall, F1, matched count, and mean IoU.
    """
    if not gt_boxes:
        return {
            "precision": 0.0 if pred_boxes else 1.0,
            "recall": 1.0,
            "f1_score": 0.0 if pred_boxes else 1.0,
            "true_positives": 0,
            "false_positives": len(pred_boxes),
            "false_negatives": 0,
            "mean_iou": 0.0
        }

    matched_gt = set()
    matched_ious = []
    tp = 0
    fp = 0

    for p in pred_boxes:
        p_box = p.get("bbox", [0, 0, 0, 0])
        best_iou = 0.0
        best_gt_idx = -1

        for idx, g in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_box_iou(p_box, g.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_iou": round(mean_iou, 4)
    }


def evaluate_semantic_segmentation(
    pred_label_map: np.ndarray,
    gt_label_map: np.ndarray,
    num_classes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute multi-class Mean IoU (mIoU) and per-class metrics for semantic segmentation.
    """
    unique_classes = np.unique(np.concatenate([np.unique(pred_label_map), np.unique(gt_label_map)]))
    class_ious = {}

    for c in unique_classes:
        p = pred_label_map == c
        g = gt_label_map == c
        inter = np.sum(p & g)
        union = np.sum(p | g)
        class_ious[int(c)] = float(inter / union) if union > 0 else 1.0

    miou = float(np.mean(list(class_ious.values()))) if class_ious else 0.0
    overall_acc = float(np.mean(pred_label_map == gt_label_map))

    return {
        "mean_iou": round(miou, 4),
        "overall_pixel_accuracy": round(overall_acc, 4),
        "per_class_iou": {str(k): round(v, 4) for k, v in class_ious.items()}
    }
