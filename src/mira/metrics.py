"""Accuracy and IoU computation for segmentation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImageMetrics:
    """Per-image evaluation metrics."""

    image_path: str
    pred_count: int
    gt_count: int
    mean_iou: float
    tp: int
    fp: int
    fn: int
    ious: list[float] = field(default_factory=list)


@dataclass
class DatasetMetrics:
    """Aggregated metrics across an entire dataset."""

    images: list[ImageMetrics] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def total_gt(self) -> int:
        return sum(m.gt_count for m in self.images)

    @property
    def total_pred(self) -> int:
        return sum(m.pred_count for m in self.images)

    @property
    def total_tp(self) -> int:
        return sum(m.tp for m in self.images)

    @property
    def total_fp(self) -> int:
        return sum(m.fp for m in self.images)

    @property
    def total_fn(self) -> int:
        return sum(m.fn for m in self.images)

    @property
    def mean_iou(self) -> float:
        if not self.images:
            return 0.0
        return float(np.mean([m.mean_iou for m in self.images]))

    @property
    def precision(self) -> float:
        denom = self.total_tp + self.total_fp
        return self.total_tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.total_tp + self.total_fn
        return self.total_tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Images: {self.n_images} | "
            f"Mean IoU: {self.mean_iou:.1%} | "
            f"Precision: {self.precision:.1%} | "
            f"Recall: {self.recall:.1%} | "
            f"F1: {self.f1:.1%} | "
            f"TP/FP/FN: {self.total_tp}/{self.total_fp}/{self.total_fn}"
        )


MASK_THRESH = 0.5


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection-over-Union between two binary masks."""
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_masks(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[float, int, int, int, list[float]]:
    """
    Greedy bipartite matching of predicted masks to ground-truth masks.

    Returns:
        mean_iou: Average IoU across matched + unmatched GT.
        tp: Matches with IoU >= *iou_threshold*.
        fp: Unmatched predictions.
        fn: Unmatched ground-truth.
        all_ious: Best IoU for each GT mask (0.0 if unmatched).
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0, 0, 0, 0, []
    if len(gt_masks) == 0:
        return 0.0, 0, len(pred_masks), 0, []
    if len(pred_masks) == 0:
        return 0.0, 0, 0, len(gt_masks), [0.0] * len(gt_masks)

    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            pred_bin = (pred > MASK_THRESH).astype(np.uint8) if pred.dtype != np.uint8 else pred
            iou_matrix[i, j] = compute_mask_iou(gt, pred_bin)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    all_ious: list[float] = []

    while len(matched_gt) < len(gt_masks) and len(matched_pred) < len(pred_masks):
        best_iou = -1.0
        best_i, best_j = -1, -1

        for i in range(len(gt_masks)):
            if i in matched_gt:
                continue
            for j in range(len(pred_masks)):
                if j in matched_pred:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_i, best_j = i, j

        if best_iou < 0:
            break

        matched_gt.add(best_i)
        matched_pred.add(best_j)
        all_ious.append(best_iou)

    for i in range(len(gt_masks)):
        if i not in matched_gt:
            all_ious.append(0.0)

    tp = sum(1 for iou in all_ious if iou >= iou_threshold)
    fp = len(pred_masks) - len(matched_pred)
    fn = len(gt_masks) - tp
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return mean_iou, tp, fp, fn, all_ious


def count_accuracy(gt: int, pred: int) -> float:
    """Simple count-based accuracy: ``1 - |pred - gt| / gt``."""
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, min(1.0, 1.0 - abs(pred - gt) / gt))
