"""
MIRA - Microplastic Intelligent Recognition Architecture
=========================================================

A framework for microplastic detection and segmentation using
YOLO object detection and SAM 2 instance segmentation.

Quick-start::

    from mira import MiraYOLO, MiraSAM2, MiraSequential

    # YOLO-only detection
    yolo = MiraYOLO(model_path="./models/hunter-yolo-v0.4.4.pt")
    result = yolo.predict("image.jpg")

    # SAM 2-only segmentation (requires bounding-box prompts)
    sam = MiraSAM2(checkpoint="./models/sam2.1_hiera_large.pt")
    result = sam.predict("image.jpg", boxes=result.boxes_xyxy)

    # Combined YOLO -> SAM 2 pipeline
    pipeline = MiraSequential(
        yolo_path="./models/hunter-yolo-v0.4.4.pt",
        sam2_checkpoint="./models/sam2.1_hiera_large.pt",
    )
    result = pipeline.predict("image.jpg")
"""

__version__ = "0.1.0"

from .yolo import MiraYOLO, YOLOResult
from .sam2 import MiraSAM2, SAM2Result
from .sequential import MiraSequential, SequentialResult
from .metrics import (
    DatasetMetrics,
    ImageMetrics,
    compute_mask_iou,
    count_accuracy,
    match_masks,
)
from .drawing import draw_boxes, draw_masks, shrink_box
from .utils import (
    discover_yolo_versions,
    get_device,
    scan_images,
    label_path_for_image,
)

__all__ = [
    # Runners
    "MiraYOLO",
    "MiraSAM2",
    "MiraSequential",
    # Result types
    "YOLOResult",
    "SAM2Result",
    "SequentialResult",
    # Metrics
    "DatasetMetrics",
    "ImageMetrics",
    "compute_mask_iou",
    "count_accuracy",
    "match_masks",
    # Drawing
    "draw_boxes",
    "draw_masks",
    "shrink_box",
    # Utilities
    "discover_yolo_versions",
    "get_device",
    "scan_images",
    "label_path_for_image",
]
