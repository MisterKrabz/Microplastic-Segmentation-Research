import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 🛠️ MONKEY PATCH FOR PYTORCH 2.6+ 🛠️ ---
_original_load = torch.load
def fixed_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = fixed_load
# ---------------------------------------------------

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
# We will test your CUSTOM model first since we know it exists
CHECKPOINT = "./models/sam2_final.pt" 
CONFIG = "sam2_hiera_l.yaml"
DEVICE = "cpu"

print(f"--- SAM2 MULTI-MASK DIAGNOSTIC ---")
try:
    # 1. Load Model
    sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    print("✅ Model loaded.")

    # 2. Create Dummy Image (White Square)
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img[300:700, 300:700, :] = 255
    predictor.set_image(img)
    
    # 3. Predict with multimask_output=TRUE (Show all options)
    print("running prediction with multimask_output=True...")
    masks, scores, _ = predictor.predict(
        box=np.array([[300, 300, 700, 700]]), 
        multimask_output=True  # <--- CRITICAL CHANGE
    )
    
    # 4. Visualize ALL 3 Options
    print(f"\n✅ SAM2 returned {len(masks)} candidate masks.")
    print(f"Scores: {scores}")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Show Input
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    
    # Show Mask 1
    axes[1].imshow(masks[0], cmap='gray')
    axes[1].set_title(f"Mask 0 (Score: {scores[0]:.2f})\n(Often the 'Inverted' one)")
    
    # Show Mask 2
    axes[2].imshow(masks[1], cmap='gray')
    axes[2].set_title(f"Mask 1 (Score: {scores[1]:.2f})\n(Is this the Clean Square?)")
    
    # Show Mask 3
    axes[3].imshow(masks[2], cmap='gray')
    axes[3].set_title(f"Mask 2 (Score: {scores[2]:.2f})\n(Alternative Option)")
    
    plt.show()
    print("Check the plot. Is 'Mask 1' or 'Mask 2' the clean white square you need?")

except Exception as e:
    print(f"❌ Error: {e}")