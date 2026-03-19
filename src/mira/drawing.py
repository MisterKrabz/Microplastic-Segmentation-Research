"""Visualisation helpers for drawing boxes and masks on images."""

from __future__ import annotations

import cv2
import numpy as np


def draw_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray | None = None,
    class_names: list[str] | None = None,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    line_width: int = 1,
    font_scale: float = 0.30,
    font_thickness: int = 1,
    show_labels: bool = True,
    show_class_name: bool = False,
) -> np.ndarray:
    """Draw bounding boxes (and optional labels) on an image."""
    out = image_bgr.copy()
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return out

    for idx, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        if show_labels:
            conf = confidences[idx] if confidences is not None else None
            name = class_names[idx] if class_names is not None else None

            parts: list[str] = []
            if show_class_name and name is not None:
                parts.append(name)
            if conf is not None:
                parts.append(f"{conf:.2f}")
            label = " ".join(parts) if parts else ""

            if label:
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                text_x = x1 + 4
                text_y = y1 - 6
                bg_top = y1 - th - baseline - 8
                bg_bottom = y1

                if bg_top < 0:
                    bg_top = y1
                    bg_bottom = y1 + th + baseline + 8
                    text_y = y1 + th + 4

                cv2.rectangle(out, (x1, bg_top), (x1 + tw + 8, bg_bottom), color, -1)
                cv2.putText(
                    out, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thickness, cv2.LINE_AA,
                )
    return out


def draw_masks(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    *,
    alpha: float = 0.45,
    mask_thresh: float = 0.5,
) -> np.ndarray:
    """Draw filled segmentation masks with contours on an image."""
    out = image_bgr.copy()
    overlay = image_bgr.copy()

    for mask in masks:
        if mask is None:
            continue
        mask_bin = (mask > mask_thresh).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            continue

        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, -1)
        cv2.drawContours(out, contours, -1, color, 2)

    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def shrink_box(box_xyxy: np.ndarray, img_w: int, img_h: int, frac: float) -> np.ndarray:
    """Shrink a bounding box inward by *frac* of its width/height."""
    if frac <= 0:
        return box_xyxy

    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    dx, dy = bw * frac, bh * frac

    x1n = np.clip(x1 + dx, 0, img_w - 1)
    y1n = np.clip(y1 + dy, 0, img_h - 1)
    x2n = np.clip(x2 - dx, 0, img_w - 1)
    y2n = np.clip(y2 - dy, 0, img_h - 1)

    if x2n <= x1n + 1:
        x1n, x2n = x1, x2
    if y2n <= y1n + 1:
        y1n, y2n = y1, y2

    return np.array([x1n, y1n, x2n, y2n], dtype=np.float32)
