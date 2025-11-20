import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
# Paths
IMAGE_PATH = "../Data/images/test_image.jpg"  # Pick a random image to test
YOLO_PATH = "./models/yolo_hunter.pt"
SAM2_CHECKPOINT = "./models/sam2_final.pt"    # You will download this from CHTC later
SAM2_CONFIG = "sam2_hiera_l.yaml"             # Ensure this config is locally available

# Device (Mac uses 'mps' for acceleration if available, else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Running inference on: {DEVICE}")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def main():
    # 1. Load the Image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Error: Could not load image.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Load The Hunter (YOLO)
    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_PATH)

    # 3. Load The Artist (SAM2)
    # Note: We assume you haven't downloaded sam2_final.pt yet. 
    # If you want to test this script NOW, change SAM2_CHECKPOINT to the original "sam2.1_hiera_large.pt"
    if os.path.exists(SAM2_CHECKPOINT):
        print("Loading Fine-Tuned SAM2...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
    else:
        print("⚠️ SAM2 Checkpoint not found (Waiting for CHTC). Skipping SAM2 step.")
        predictor = None

    # 4. Step 1: YOLO Detection
    print("Running YOLO Detection...")
    results = yolo_model(image)
    
    # Extract boxes (xyxy format)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    # Filter for "Microplastic" class (assuming class 0)
    # If you trained YOLO on "plastics" and "bubbles", you would filter out bubbles here.
    valid_boxes = []
    for box, cls in zip(boxes, classes):
        # if cls == 0: # Uncomment if you have multiple classes
        valid_boxes.append(box)
    
    valid_boxes = np.array(valid_boxes)
    print(f"YOLO found {len(valid_boxes)} potential microplastics.")

    if len(valid_boxes) == 0:
        print("No plastics found.")
        return

    # 5. Step 2: SAM2 Segmentation
    if predictor:
        print("Running SAM2 Segmentation...")
        predictor.set_image(image)
        
        # SAM2 can process multiple boxes at once
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=valid_boxes,
            multimask_output=False,
        )
        
        # 6. Visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        for mask in masks:
            show_mask(mask, plt.gca())
            
        for box in valid_boxes:
            show_box(box, plt.gca())
            
        plt.axis('off')
        plt.title(f"Detected & Segmented: {len(masks)} Microplastics")
        plt.show()
        
    else:
        # Fallback visualization if SAM2 isn't ready yet
        results[0].show()

if __name__ == "__main__":
    main()