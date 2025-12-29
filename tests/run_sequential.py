import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- 1. DYNAMIC PATH SETUP ---
# Get the folder where THIS script is (inference/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (URS_PROJECT/)
project_root = os.path.dirname(script_dir)

# Add project root to Python path so we can import 'sam2'
if project_root not in sys.path:
    sys.path.append(project_root)

# Fallback: If sam2 is still in the old spot (sam2_training/sam2), add that too
old_sam2_path = os.path.join(project_root, "sam2_training", "sam2")
if os.path.exists(old_sam2_path) and old_sam2_path not in sys.path:
    sys.path.append(os.path.dirname(old_sam2_path))

# NOW we can import SAM2 safely
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("❌ CRITICAL ERROR: Could not find 'sam2' library.")
    print(f"   Please make sure the 'sam2' folder is in {project_root}")
    sys.exit(1)

# --- CONFIGURATION ---
# CHANGE THIS PATH to your real image
IMAGE_PATH = "../datasets/dataset/test/images/sample.jpg"

YOLO_MODEL_NAME = "yolo_hunter.pt"
SAM2_MODEL_NAME = "sam2_hiera_l.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

# Settings
YOLO_CONF = 0.25
SAM2_MASK_THRESHOLD = 0.5
FILL_ALPHA = 100    # Mask transparency (0-255)
BORDER_THICKNESS = 2

# --- DEVICE SETUP ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Using Apple Metal (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("🚀 Using CUDA")
else:
    DEVICE = "cpu"
    print("⚠️  Using CPU")

# --- HELPER FUNCTIONS ---

def get_binary_mask(mask_input, image_shape):
    """Converts SAM2 logit output to a clean binary mask."""
    if torch.is_tensor(mask_input):
        mask = mask_input.detach().cpu().numpy()
    else:
        mask = mask_input
    
    mask = mask.squeeze()
    
    # Sigmoid if logits
    if mask.min() < 0 or mask.max() > 1:
        mask = 1.0 / (1.0 + np.exp(-mask))
    
    # Resize
    target_h, target_w = image_shape[0], image_shape[1]
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    return (mask > SAM2_MASK_THRESHOLD).astype(np.uint8)

def generate_colors(num_colors):
    """Generates distinct bright colors."""
    colors = []
    for i in range(num_colors):
        hue = (i * 360.0 / num_colors) % 360.0
        h = hue / 60.0
        x = int(255 * (1 - abs(h % 2 - 1)))
        if 0 <= h < 1: r, g, b = 255, x, 0
        elif 1 <= h < 2: r, g, b = x, 255, 0
        elif 2 <= h < 3: r, g, b = 0, 255, x
        elif 3 <= h < 4: r, g, b = 0, x, 255
        elif 4 <= h < 5: r, g, b = x, 0, 255
        else: r, g, b = 255, 0, x
        colors.append((r, g, b))
    return colors

def apply_masks(image, masks):
    """Overlays colored masks on the image."""
    overlay = image.copy()
    colors = generate_colors(len(masks))
    
    for i, mask in enumerate(masks):
        color = colors[i]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask == 1] = color
        alpha = FILL_ALPHA / 255.0
        mask_bool = mask == 1
        overlay[mask_bool] = cv2.addWeighted(image[mask_bool], 1-alpha, colored_mask[mask_bool], alpha, 0)

        # Border
        cv2.drawContours(overlay, contours, -1, color, BORDER_THICKNESS)
        
    return overlay

# --- MAIN PIPELINE ---

def main():
    print("--- 🏁 STARTING SEQUENTIAL INFERENCE ---")

    # 1. LOCATE MODELS
    yolo_path = os.path.join(project_root, "models", YOLO_MODEL_NAME)
    sam2_checkpoint = os.path.join(project_root, "models", SAM2_MODEL_NAME)

    if not os.path.exists(yolo_path):
        sys.exit(f"❌ Error: YOLO model missing at {yolo_path}")
    if not os.path.exists(sam2_checkpoint):
        sys.exit(f"❌ Error: SAM2 checkpoint missing at {sam2_checkpoint}")

    # 2. LOAD IMAGE
    # Handle relative path from script location
    abs_image_path = os.path.abspath(os.path.join(script_dir, IMAGE_PATH))
    if not os.path.exists(abs_image_path):
         # Try direct path
        if os.path.exists(IMAGE_PATH):
            abs_image_path = IMAGE_PATH
        else:
            sys.exit(f"❌ Error: Image not found at {IMAGE_PATH}")

    print(f"📷 Processing: {os.path.basename(abs_image_path)}")
    image_bgr = cv2.imread(abs_image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. RUN YOLO
    print("🔍 Running YOLO...")
    yolo = YOLO(yolo_path)
    results = yolo(image_rgb, conf=YOLO_CONF, verbose=False)
    
    boxes = []
    for r in results:
        b = r.boxes.xyxy.cpu().numpy()
        if len(b) > 0: boxes.append(b)
    
    if not boxes:
        print("❌ No objects detected.")
        return
    
    boxes = np.vstack(boxes)
    print(f"✅ Found {len(boxes)} objects.")

    # 4. RUN SAM2
    print("🎨 Running SAM2...")
    try:
        sam2_model = build_sam2(SAM2_CONFIG, sam2_checkpoint, device=DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image_rgb)
        
        masks_logits, scores, _ = predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )
    except Exception as e:
        sys.exit(f"❌ SAM2 Error: {e}")

    # 5. PROCESS & SHOW
    valid_masks = [get_binary_mask(l, image_rgb.shape) for l in masks_logits if get_binary_mask(l, image_rgb.shape).sum() > 0]
    
    print(f"✅ Created {len(valid_masks)} masks. Displaying...")
    final_image = apply_masks(image_rgb, valid_masks)

    plt.figure(figsize=(10, 8))
    plt.imshow(final_image)
    plt.axis('off')
    plt.title(f"YOLO + SAM2 Results ({len(valid_masks)} objects)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()