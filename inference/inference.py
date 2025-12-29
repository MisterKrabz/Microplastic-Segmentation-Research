import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 🛠️ MONKEY PATCH: Fix for PyTorch 2.6+ Security ---
_original_load = torch.load
def fixed_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = fixed_load
# ---------------------------------------------------

# --- CONFIGURATION ---
IMAGE_PATH = "../data/Microplastics.v2-v2.coco-segmentation/train/1_5umPS_0_4umPCTE-30sAu_X100_darkfield_1_jpg.rf.ed29c00f14aeed638dcdff6ad5adfb72.jpg"

# 1. Your Custom YOLO
YOLO_PATH = "./models/yolo_hunter.pt"

# 2. Official SAM2 Weights (Critical for avoiding grid)
SAM2_CHECKPOINT = "./models/sam2_hiera_l.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

YOLO_CONF = 0.05
DEVICE = "cpu"

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        return
    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"❌ Error: Official weights not found at {SAM2_CHECKPOINT}")
        return

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # STAGE 1: RUN YOLO (THE MASTER)
    # ---------------------------------------------------------
    print("--- Stage 1: Running YOLO Detector ---")
    yolo = YOLO(YOLO_PATH)
    results = yolo(image_rgb, conf=YOLO_CONF, verbose=False)
    
    valid_boxes = []
    for r in results:
        b = r.boxes.xyxy.cpu().numpy()
        if len(b) > 0: valid_boxes.append(b)
    
    if not valid_boxes:
        print("No particles detected.")
        return
    valid_boxes = np.vstack(valid_boxes)
    print(f"✓ YOLO found {len(valid_boxes)} bounding boxes.")

    # ---------------------------------------------------------
    # STAGE 2: RUN SAM 2 (STRICT BOX MODE)
    # ---------------------------------------------------------
    print("--- Stage 2: Running SAM 2 (Strict Box Mode) ---")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_rgb)
    
    final_masks = []
    
    print(f"Processing {len(valid_boxes)} boxes individually...")
    
    # We iterate 1-by-1 to guarantee 1 mask per box
    for i, box in enumerate(valid_boxes):
        # USE BOX PROMPT (x1, y1, x2, y2)
        # This forces SAM to look ONLY inside the box, preventing cluster merging.
        # Since we use Official Weights, this will NOT generate a grid.
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=True
        )
        
        # Select best mask by confidence score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # Binary conversion
        binary_mask = (best_mask > 0.0).astype(np.uint8)
        
        # We append it even if it's imperfect, to maintain the 1:1 count
        final_masks.append(binary_mask)

    print(f"✅ Generated {len(final_masks)} masks (Should equal {len(valid_boxes)})")

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    print("--- Generating Comparison Plot ---")
    
    # 1. Prepare YOLO Image
    img_yolo = image_rgb.copy()
    for box in valid_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    # 2. Prepare SAM Image
    img_sam = image_rgb.copy()
    
    # Generate distinct colors for each instance
    colors = [np.random.randint(50, 255, 3).tolist() for _ in final_masks]
    
    for i, mask in enumerate(final_masks):
        color = colors[i]
        
        # Draw Border
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_sam, contours, -1, color, 2)
        
        # Draw Semi-Transparent Fill
        mask_indices = mask == 1
        if np.any(mask_indices):
            roi = img_sam[mask_indices]
            color_layer = np.full_like(roi, color)
            # Blend: 60% Original + 40% Color
            blended = cv2.addWeighted(roi, 0.6, color_layer, 0.4, 0)
            img_sam[mask_indices] = blended

    # 3. Plot Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(img_yolo)
    axes[0].set_title(f"YOLO Detection\nCount: {len(valid_boxes)}", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_sam)
    axes[1].set_title(f"SAM 2 Segmentation (Box Prompt)\nCount: {len(final_masks)}", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    output_path = "final_comparison_strict.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved comparison to: {os.path.abspath(output_path)}")
    plt.show()

if __name__ == "__main__":
    main()