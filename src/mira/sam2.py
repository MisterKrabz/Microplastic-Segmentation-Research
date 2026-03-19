"""SAM 2 segmentation runner."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .drawing import shrink_box
from .utils import get_device


@dataclass
class SAM2Result:
    """Container for SAM 2 segmentation output on a single image."""

    image_path: str
    image_rgb: np.ndarray
    masks: list[np.ndarray] = field(default_factory=list)
    boxes_xyxy: np.ndarray | None = None

    @property
    def count(self) -> int:
        return len(self.masks)


class MiraSAM2:
    """
    SAM 2 segmentation runner.

    Given bounding-box prompts for an image, produces per-object masks.

    Examples::

        sam = MiraSAM2(
            checkpoint="./models/sam2.1_hiera_large.pt",
            config="configs/sam2.1/sam2.1_hiera_l.yaml",
        )
        result = sam.predict("image.jpg", boxes=[[x1, y1, x2, y2], ...])
    """

    def __init__(
        self,
        checkpoint: str,
        config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        *,
        device: str | None = None,
        box_shrink: float = 0.10,
    ) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = get_device(device)
        self.box_shrink = box_shrink
        self.checkpoint = checkpoint
        self.config = config

        self._sam2_model = build_sam2(config, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self._sam2_model)

    def predict(
        self,
        image: str | np.ndarray,
        boxes: np.ndarray | list[list[float]],
        *,
        box_shrink: float | None = None,
    ) -> SAM2Result:
        """
        Segment objects in *image* given bounding-box prompts.

        Args:
            image: File path or RGB numpy array.
            boxes: (N, 4) array of ``[x1, y1, x2, y2]`` bounding boxes.
            box_shrink: Override the default box-shrink fraction.
        """
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
        boxes_arr = np.asarray(boxes, dtype=np.float32)

        if boxes_arr.ndim == 1:
            boxes_arr = boxes_arr.reshape(1, -1)

        masks: list[np.ndarray] = []
        if len(boxes_arr) == 0:
            return SAM2Result(image_path=path, image_rgb=img_rgb, masks=[], boxes_xyxy=boxes_arr)

        self.predictor.set_image(img_rgb)

        for b in boxes_arr:
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

        return SAM2Result(
            image_path=path,
            image_rgb=img_rgb,
            masks=masks,
            boxes_xyxy=boxes_arr,
        )
