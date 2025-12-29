import os
import random
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============================================================================
# 🎛️ TUNING DASHBOARD (Modify these to change mask appearance)
# ============================================================================

# 1. DETECTION
YOLO_CONFIDENCE_THRESHOLD = 0.1  # Moderate confidence
YOLO_IOU_THRESHOLD = 0.5          # Standard overlap removal

# 2. MASK "CLEANING" (New Section)
# ------------------------------------------------------------------
# SMOOTHING_ITERATIONS: Higher = Smoother/Rounder, Lower = More jagged.
# Recommended: 1 or 2. (0 disables smoothing)
SMOOTHING_ITERATIONS = 1          

# FILL_HOLES: True = Fills black spots inside the plastic.
FILL_HOLES = True                 

# KEEP_ONLY_LARGEST: True = Deletes floating pixel noise, keeps main blob.
KEEP_ONLY_LARGEST = True          
# ------------------------------------------------------------------

# 3. VISUALS
MASK_ALPHA = 0.60                 # 0.0 (Transparent) -> 1.0 (Solid)
BORDER_THICKNESS = 2              # Thickness of the outline
MIN_MASK_PIXELS = 50              # Delete masks smaller than this

# 4. SAFETY
MAX_OBJECTS_LIMIT = 100

# ============================================================================
# SETUP
# ============================================================================
TEST_DIR = "../data/Microplastics.v2-v2.coco-segmentation/test"
ANNOTATIONS_FILE = os.path.join(TEST_DIR, "_annotations.coco.json")
YOLO_PATH = "./models/yolo_hunter.pt"
SAM2_CHECKPOINT = "./models/sam2_final.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# CLEANING FUNCTIONS (THE FIX)
# ============================================================================

def clean_mask_prediction(binary_mask):
    """
    Applies computer vision morphology to smooth edges and remove noise.
    This turns a 'dirty' AI mask into a 'clean' vector-like shape.
    """
    # 1. Morphological Opening (Removes white noise/speckles outside)
    # 2. Morphological Closing (Fills black holes inside)
    kernel = np.ones((3, 3), np.uint8) # 3x3 smoothing brush
    
    cleaned = binary_mask.copy()
    
    if SMOOTHING_ITERATIONS > 0:
        # "Open" removes small noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=SMOOTHING_ITERATIONS)
        # "Close" fills holes
        if FILL_HOLES:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=SMOOTHING_ITERATIONS)
    
    # 3. Keep Only Largest Component (Deletes floating islands)
    if KEEP_ONLY_LARGEST:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            # stats[:, 4] is area. Index 0 is background, so we look at 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_label).astype(np.uint8)
            
    return cleaned

# ============================================================================
# UTILITIES
# ============================================================================

def load_coco_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    return {img['id']: img for img in coco_data['images']}

def generate_rainbow_colors(n):
    """Generate distinct colors using HSV spectrum"""
    colors = []
    for i in range(n):
        hue = int(180 * (i / n)) 
        saturation = 200
        value = 250
        hsv_color = np.uint8([[[hue, saturation, value]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(map(int, rgb_color)))
    import random
    random.shuffle(colors)
    return colors

def get_centroid(box):
    x1, y1, x2, y2 = box
    return np.array([[ (x1+x2)/2, (y1+y2)/2 ]])

def apply_mask_overlay(image, mask, color, alpha):
    overlay = image.copy()
    mask_bool = mask > 0
    if not mask_bool.any(): return overlay
    
    # Fill
    roi = overlay[mask_bool]
    color_layer = np.full_like(roi, color)
    blended = cv2.addWeighted(roi, 1.0 - alpha, color_layer, alpha, 0)
    overlay[mask_bool] = blended
    
    # Border
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, BORDER_THICKNESS)
    
    return overlay

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("INITIALIZING CLEAN SEGMENTATION PIPELINE")
    print("=" * 60)
    
    yolo_model = YOLO(YOLO_PATH)
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    
    images_dict = load_coco_annotations(ANNOTATIONS_FILE)
    
    for img_id, img_info in images_dict.items():
        filename = img_info['file_name']
        image_path = os.path.join(TEST_DIR, filename)
        
        if not os.path.exists(image_path): continue
        
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. YOLO
        results = yolo_model(image_rgb, conf=YOLO_CONFIDENCE_THRESHOLD, iou=YOLO_IOU_THRESHOLD, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        num_detected = len(boxes)
        print(f"\n[DEBUG] {filename}: Found {num_detected} bounding boxes.")
        
        if num_detected == 0: continue
        if num_detected > MAX_OBJECTS_LIMIT:
            print(f"   [WARNING] Capping objects at {MAX_OBJECTS_LIMIT}")
            boxes = boxes[:MAX_OBJECTS_LIMIT]
            num_detected = MAX_OBJECTS_LIMIT

        # 2. SAM 2 LOOP
        predictor.set_image(image_rgb)
        final_overlay = image_rgb.copy()
        instance_colors = generate_rainbow_colors(num_detected)
        valid_mask_count = 0
        
        for i in range(num_detected):
            # A. Predict
            point_coords = get_centroid(boxes[i])
            point_labels = np.array([1])
            
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False 
            )
            
            # B. Probability Threshold
            mask_raw = masks.squeeze()
            if mask_raw.max() > 1.0 or mask_raw.min() < 0.0:
                mask_prob = 1.0 / (1.0 + np.exp(-mask_raw))
            else:
                mask_prob = mask_raw
            
            binary_mask = (mask_prob > 0.5).astype(np.uint8)
            
            # C. ✨ THE CLEANING STEP ✨
            clean_mask = clean_mask_prediction(binary_mask)
            
            if np.sum(clean_mask) < MIN_MASK_PIXELS:
                continue
                
            # D. Overlay
            final_overlay = apply_mask_overlay(final_overlay, clean_mask, instance_colors[i], MASK_ALPHA)
            valid_mask_count += 1
            
        # 3. SHOW
        plt.figure(figsize=(12, 8))
        plt.imshow(final_overlay)
        plt.axis('off')
        plt.title(f"Cleaned Result: {valid_mask_count} Objects", fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()