"""Sequential YOLO + SAM 2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .drawing import shrink_box
from .metrics import DatasetMetrics, ImageMetrics, match_masks
from .utils import (
    count_gt_instances,
    get_device,
    label_path_for_image,
    parse_yolo_seg_labels,
    resolve_yolo_model,
    scan_images,
)


@dataclass
class SequentialResult:
    """Output of the combined YOLO -> SAM 2 pipeline for a single image."""

    image_path: str
    image_rgb: np.ndarray
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    class_names: list[str] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)

    @property
    def pred_count(self) -> int:
        return len(self.boxes_xyxy)


class MiraSequential:
    """
    Combined YOLO detection + SAM 2 segmentation pipeline.

    Uses YOLO to detect bounding boxes, then feeds each box as a prompt
    into SAM 2 to produce per-object segmentation masks.

    Examples::

        pipeline = MiraSequential(
            yolo_path="./models/hunter-yolo-v0.4.4.pt",
            sam2_checkpoint="./models/sam2.1_hiera_large.pt",
        )
        result = pipeline.predict("image.jpg")
        results = pipeline.predict_batch(["img1.jpg", "img2.jpg"])

        # With YOLO version auto-discovery
        pipeline = MiraSequential(
            yolo_version="0.4.4",
            yolo_models_dir="./models",
            sam2_checkpoint="./models/sam2.1_hiera_large.pt",
        )
    """

    def __init__(
        self,
        yolo_path: str | None = None,
        sam2_checkpoint: str = "",
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        *,
        yolo_version: str | None = None,
        yolo_models_dir: str | None = None,
        device: str | None = None,
        conf: float = 0.3,
        iou: float = 0.25,
        imgsz: int = 1280,
        box_shrink: float = 0.10,
    ) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from ultralytics import YOLO

        self.device = get_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.box_shrink = box_shrink

        resolved_yolo = resolve_yolo_model(
            yolo_path, version=yolo_version, models_dir=yolo_models_dir
        )
        self.yolo_path = resolved_yolo
        self.yolo = YOLO(resolved_yolo)

        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def predict(
        self,
        image: str | np.ndarray,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        box_shrink: float | None = None,
    ) -> SequentialResult:
        """Run YOLO + SAM 2 on a single image."""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        imgsz = imgsz if imgsz is not None else self.imgsz
        shrink = box_shrink if box_shrink is not None else self.box_shrink

        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"Could not read image: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            path = image
        else:
            img_rgb = image
            path = "<ndarray>"

        h, w = img_rgb.shape[:2]

        results = self.yolo.predict(
            source=path if isinstance(image, str) else image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            agnostic_nms=True,
        )
        r0 = results[0]
        boxes_obj = r0.boxes
        names = r0.names if hasattr(r0, "names") else {}

        if boxes_obj is not None and len(boxes_obj) > 0:
            xyxy = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes_obj.conf.cpu().numpy()
            cls_ids = boxes_obj.cls.cpu().numpy().astype(int)
            cls_names = [names.get(int(c), str(int(c))) for c in cls_ids]
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            cls_ids = np.empty((0,), dtype=int)
            cls_names = []

        masks: list[np.ndarray] = []
        if len(xyxy) > 0:
            self.predictor.set_image(img_rgb)
            for b in xyxy:
                b_shrunk = shrink_box(b, w, h, shrink)
                m, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=b_shrunk[None, :],
                    multimask_output=False,
                )
                m0 = m[0]
                if m0.ndim == 3:
                    m0 = m0.squeeze(0)
                masks.append(m0)

        return SequentialResult(
            image_path=path,
            image_rgb=img_rgb,
            boxes_xyxy=xyxy,
            confidences=confs,
            class_ids=cls_ids,
            class_names=cls_names,
            masks=masks,
        )

    def predict_batch(
        self,
        images: list[str],
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        box_shrink: float | None = None,
    ) -> list[SequentialResult]:
        """Run the pipeline on a list of image paths."""
        return [
            self.predict(img, conf=conf, iou=iou, imgsz=imgsz, box_shrink=box_shrink)
            for img in images
        ]

    def predict_directory(
        self,
        *paths: str,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        box_shrink: float | None = None,
    ) -> list[SequentialResult]:
        """Recursively scan directories for images and run the pipeline."""
        files = scan_images(*paths)
        return self.predict_batch(
            files, conf=conf, iou=iou, imgsz=imgsz, box_shrink=box_shrink
        )

    def evaluate(
        self,
        *paths: str,
        iou_threshold: float = 0.5,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> DatasetMetrics:
        """
        Run the pipeline on a dataset and evaluate against ground-truth labels.

        Expects YOLO-format label files alongside the images.
        """
        files = scan_images(*paths)
        metrics = DatasetMetrics()

        for img_path in files:
            result = self.predict(img_path, conf=conf, iou=iou, imgsz=imgsz)
            h, w = result.image_rgb.shape[:2]

            lbl_path = label_path_for_image(img_path)
            gt_masks = parse_yolo_seg_labels(lbl_path, w, h)
            gt_count = count_gt_instances(lbl_path)

            mean_iou, tp, fp, fn, ious = match_masks(
                result.masks, gt_masks, iou_threshold=iou_threshold
            )

            metrics.images.append(
                ImageMetrics(
                    image_path=img_path,
                    pred_count=result.pred_count,
                    gt_count=gt_count,
                    mean_iou=mean_iou,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    ious=ious,
                )
            )

        return metrics
