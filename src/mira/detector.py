import torch
import cv2
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class MicroplasticDetector:
    def __init__(self, yolo_path, sam2_checkpoint, sam2_config="sam2_hiera_l.yaml", device=None):
        """
        Initialize the detector.
        args:
            yolo_path (str): Path to .pt file
            sam2_checkpoint (str): Path to .pt file
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"
        
        print(f"🚀 Loading models on {self.device}...")
        
        # Load YOLO
        self.yolo = YOLO(yolo_path)
        
        # Load SAM 2
        self.sam2 = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2)
        print("✅ Models loaded.")

    def predict(self, image_path, conf=0.25):
        """
        Run inference on a single image.
        Returns:
            image_rgb: The original image
            masks: List of binary masks (numpy arrays)
        """
        # 1. Load Image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image at {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 2. YOLO Inference
        results = self.yolo(image_rgb, conf=conf, verbose=False)
        boxes = []
        for r in results:
            b = r.boxes.xyxy.cpu().numpy()
            if len(b) > 0: boxes.append(b)
        
        if not boxes:
            print("No particles detected.")
            return image_rgb, []

        boxes = np.vstack(boxes)
        
        # 3. SAM 2 Inference
        self.predictor.set_image(image_rgb)
        masks_logits, _, _ = self.predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )
        
        # 4. Process Masks
        valid_masks = []
        for logit in masks_logits:
            mask = (logit > 0.0).astype(np.uint8) 
            if mask.ndim > 2: mask = mask[0]
            valid_masks.append(cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0])))

        return image_rgb, valid_masks