#!/usr/bin/env python3
"""
YOLO Detection Only - Visualize Bounding Boxes
Runs YOLO detector on test images and shows bounding boxes (no segmentation)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test dataset path
TEST_DIR = "../data/Microplastics.v2-v2.coco-segmentation/test"

# YOLO model path
YOLO_PATH = "./models/yolo_hunter.pt"

# Detection settings
YOLO_CONFIDENCE = 0.05  # Detection confidence threshold (0.0-1.0)
YOLO_IOU = 0.5         # IoU threshold for NMS (Non-Maximum Suppression)

# Visualization settings
BOX_COLOR = (255, 0, 0)      # RGB color for bounding boxes (red)
BOX_THICKNESS = 3             # Thickness of bounding box lines
TEXT_COLOR = (255, 255, 255)  # RGB color for text (white)
TEXT_BG_COLOR = (255, 0, 0)   # RGB color for text background (red)
FONT_SCALE = 0.6              # Size of confidence text
SHOW_CONFIDENCE = True        # Show confidence scores on boxes

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run YOLO detection on all test images and display bounding boxes"""
    
    print("=" * 70)
    print("YOLO DETECTION ONLY - BOUNDING BOX VISUALIZATION")
    print("=" * 70)
    print(f"Test directory: {TEST_DIR}")
    print(f"YOLO model: {YOLO_PATH}")
    print(f"Confidence threshold: {YOLO_CONFIDENCE}")
    print(f"IoU threshold: {YOLO_IOU}")
    print()
    
    # Load YOLO model
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_PATH)
    print("✅ Model loaded successfully\n")
    
    # Get all test images
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory not found: {TEST_DIR}")
        return
    
    image_files = [f for f in os.listdir(TEST_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in {TEST_DIR}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    print("=" * 70)
    
    # Process each image
    for idx, filename in enumerate(sorted(image_files), 1):
        image_path = os.path.join(TEST_DIR, filename)
        
        print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
        
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"  ⚠️  Could not read image: {filename}")
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        print(f"  Image size: {w}×{h}")
        
        # Run YOLO detection
        results = yolo_model(
            image_rgb,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            verbose=False
        )
        
        # Extract detections
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        
        num_detections = len(boxes)
        print(f"  Detections: {num_detections} objects found")
        
        # Create visualization
        vis_image = image_rgb.copy()
        
        # Draw each bounding box
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (x1, y1),
                (x2, y2),
                BOX_COLOR,
                BOX_THICKNESS
            )
            
            # Draw confidence score
            if SHOW_CONFIDENCE:
                label = f"{conf:.2f}"
                
                # Get text size for background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    1
                )
                
                # Draw text background
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - text_h - baseline - 4),
                    (x1 + text_w + 4, y1),
                    TEXT_BG_COLOR,
                    -1  # Filled
                )
                
                # Draw text
                cv2.putText(
                    vis_image,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    TEXT_COLOR,
                    1,
                    cv2.LINE_AA
                )
            
            print(f"    Box {i+1}: [{x1}, {y1}, {x2}, {y2}], Confidence: {conf:.3f}")
        
        # Display result
        plt.figure(figsize=(14, 10))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(
            f"{filename}\n{num_detections} detections (Confidence ≥ {YOLO_CONFIDENCE})",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.tight_layout()
        
        print(f"  📊 Displaying results (close window to continue)...")
        plt.show()
    
    print("\n" + "=" * 70)
    print("DETECTION COMPLETE")
    print(f"Processed {len(image_files)} images")
    print("=" * 70)


if __name__ == "__main__":
    main()

