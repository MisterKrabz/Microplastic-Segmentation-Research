"""YOLO-only detection / segmentation runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import get_device, resolve_yolo_model, scan_images


@dataclass
class YOLOResult:
    """Container for a single YOLO prediction."""

    image_path: str
    image_rgb: np.ndarray
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    class_names: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.boxes_xyxy)


class MiraYOLO:
    """
    YOLO detection runner with model-version selection.

    Examples::

        # From an explicit path
        yolo = MiraYOLO(model_path="./models/hunter-yolo-v0.4.4.pt")

        # Auto-discover version from a models directory
        yolo = MiraYOLO(version="0.4.4", models_dir="./models")

        result = yolo.predict("image.jpg")
        results = yolo.predict_batch(["img1.jpg", "img2.jpg"])
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        version: str | None = None,
        models_dir: str | None = None,
        device: str | None = None,
        conf: float = 0.3,
        iou: float = 0.25,
        imgsz: int = 1280,
    ) -> None:
        self.device = get_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        resolved = resolve_yolo_model(model_path, version=version, models_dir=models_dir)
        self.model_path = resolved
        self.model = YOLO(resolved)

    def predict(
        self,
        image: str | np.ndarray,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> YOLOResult:
        """Run YOLO inference on a single image (path or ndarray)."""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        imgsz = imgsz if imgsz is not None else self.imgsz

        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"Could not read image: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            source = image
            path = image
        else:
            img_rgb = image
            source = image
            path = "<ndarray>"

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            agnostic_nms=True,
        )
        r0 = results[0]
        boxes = r0.boxes

        names = r0.names if hasattr(r0, "names") else {}

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            cls_names = [names.get(int(c), str(int(c))) for c in cls_ids]
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            cls_ids = np.empty((0,), dtype=int)
            cls_names = []

        return YOLOResult(
            image_path=path,
            image_rgb=img_rgb,
            boxes_xyxy=xyxy,
            confidences=confs,
            class_ids=cls_ids,
            class_names=cls_names,
        )

    def predict_batch(
        self,
        images: list[str],
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> list[YOLOResult]:
        """Run YOLO on a list of image paths."""
        return [self.predict(img, conf=conf, iou=iou, imgsz=imgsz) for img in images]

    def predict_directory(
        self,
        *paths: str,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> list[YOLOResult]:
        """Recursively scan directories for images and run inference."""
        files = scan_images(*paths)
        return self.predict_batch(files, conf=conf, iou=iou, imgsz=imgsz)

    @staticmethod
    def available_versions(models_dir: str) -> list[str]:
        """List available YOLO model versions in a directory."""
        from .utils import discover_yolo_versions
        return sorted(discover_yolo_versions(models_dir).keys())
