import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
# CHANGE THIS to your image path
# Using a test image from the test dataset
IMAGE_PATH = "../data/Microplastics.v2-v2.coco-segmentation/train/1_5umPS_0_4umPCTE-30sAu_X100_darkfield_1_jpg.rf.ed29c00f14aeed638dcdff6ad5adfb72.jpg"
YOLO_PATH = "./models/yolo_hunter.pt"
SAM2_CHECKPOINT = "./models/sam2_final.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"
# Confidence threshold for YOLO to reduce noise
YOLO_CONF = 0.05
# Confidence threshold for SAM2 (Higher = cleaner masks, but might miss edges)
SAM2_MASK_THRESHOLD = 0.5

FILL_ALPHA = 128  # 50% opacity 
BORDER_ALPHA = 255  # 100% opacity 
BORDER_THICKNESS = 3  # in pixels 

if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Metal (MPS) acceleration")
else:
    DEVICE = "cpu"
    print("⚠️ Using CPU ⚠️")

# --- HELPER FUNCTIONS ---

def get_binary_mask(mask_input, image_shape):
    """
    Correctly converts SAM2 output to clean binary mask.
    
    SAM2 outputs LOW-RES LOGITS (e.g., 256x256).
    CRITICAL: Must convert logits→probabilities BEFORE upsampling,
    otherwise you get grid artifacts.
    """
    # Step 1: Convert torch tensor to numpy
    if torch.is_tensor(mask_input):
        mask = mask_input.detach().cpu().numpy()
        print(f"  Converted torch tensor to numpy, shape: {mask.shape}")
    else:
        mask = mask_input
    
    # Step 2: Squeeze extra dimensions (batch, channel dims)
    mask = mask.squeeze()
    print(f"  After squeeze, shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Value range: min={mask.min():.4f}, max={mask.max():.4f}")
    
    # Step 3: Convert logits to probabilities FIRST (CRITICAL!)
    # SAM2 outputs logits. MUST apply sigmoid BEFORE resizing.
    if mask.min() < 0 or mask.max() > 1:
        print(f"  Detected logits → applying sigmoid")
        mask = 1.0 / (1.0 + np.exp(-mask))
        print(f"  After sigmoid: min={mask.min():.4f}, max={mask.max():.4f}")
    
    # Step 4: Upsample probabilities to full resolution
    # Use INTER_LINEAR on probabilities (smooth), NOT on logits (artifacts)
    target_h, target_w = image_shape[0], image_shape[1]
    if mask.shape != (target_h, target_w):
        print(f"  Upsampling probabilities: {mask.shape} → ({target_h}, {target_w})")
        mask = cv2.resize(
            mask, 
            (target_w, target_h),  # cv2.resize expects (width, height)
            interpolation=cv2.INTER_LINEAR  # Smooth interpolation on probabilities
        )
        print(f"  After resize: shape={mask.shape}")
    
    # Step 5: Hard threshold to get clean binary mask
    # Threshold at 0.5, then convert to uint8 with values EXACTLY 0 or 1
    binary_mask = (mask > SAM2_MASK_THRESHOLD).astype(np.uint8)
    
    # CRITICAL: Ensure mask is EXACTLY 0 or 1 (no other values)
    # Some edge cases can leave non-binary values
    binary_mask = np.where(binary_mask > 0, 1, 0).astype(np.uint8)
    
    num_pixels = np.sum(binary_mask)
    unique_vals = np.unique(binary_mask)
    print(f"  Binary mask: {num_pixels} pixels (threshold={SAM2_MASK_THRESHOLD})")
    print(f"  Unique values in mask: {unique_vals} (should be [0, 1] only)")
    
    return binary_mask

def generate_distinct_colors(num_instances):
    """
    Manually generate distinct RGB colors using HSV color space.
    Returns colors in [0, 255] range as uint8.
    """
    colors = []
    for i in range(num_instances):
        # Spread hues evenly across the color wheel
        hue = (i * 360.0 / num_instances) % 360.0
        
        # Convert HSV (H, 1.0, 1.0) to RGB manually
        # H is in degrees [0, 360], S and V are 1.0 for bright saturated colors
        h_sector = hue / 60.0
        c = 1.0  # Chroma
        x = c * (1.0 - abs(h_sector % 2.0 - 1.0))
        
        if h_sector < 1:
            r, g, b = c, x, 0
        elif h_sector < 2:
            r, g, b = x, c, 0
        elif h_sector < 3:
            r, g, b = 0, c, x
        elif h_sector < 4:
            r, g, b = 0, x, c
        elif h_sector < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convert to [0, 255] range
        color_255 = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)
        colors.append(color_255)
    
    return colors

def create_instance_masks_manual(binary_masks, image_shape):
    """
    Manually create colored instance masks with borders from binary masks.
    
    Args:
        binary_masks: List of binary masks (H, W) with values 0 or 1
        image_shape: (height, width, channels) of original image
    
    Returns:
        instance_mask_rgba: RGBA image with colored masks and transparent background
    """
    h, w = image_shape[0], image_shape[1]
    num_instances = len(binary_masks)
    
    # Create RGBA image - initially all transparent (alpha = 0)
    instance_mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Generate distinct colors for each instance
    colors = generate_distinct_colors(num_instances)
    
    print(f"Manually creating {num_instances} colored instance masks with borders...")
    
    # Process each instance
    for idx, binary_mask in enumerate(binary_masks):
        color = colors[idx]
        
        # VERIFICATION: Ensure this mask is truly binary
        unique_vals = np.unique(binary_mask)
        if len(unique_vals) > 2 or not all(v in [0, 1] for v in unique_vals):
            print(f"  ⚠️ WARNING: Mask {idx+1} has non-binary values: {unique_vals}")
            # Force to binary
            binary_mask = np.where(binary_mask > 0, 1, 0).astype(np.uint8)
        
        # Step 1: Fill the mask region with SOLID semi-transparent color
        # Find all pixels where mask == 1 (EXACTLY 1, not probabilities)
        mask_pixels = binary_mask == 1
        num_mask_pixels = np.sum(mask_pixels)
        
        # Assign SOLID RGB + Alpha to these pixels (no gradients, no probabilities)
        instance_mask_rgba[mask_pixels, 0] = color[0]  # R - SOLID color
        instance_mask_rgba[mask_pixels, 1] = color[1]  # G - SOLID color
        instance_mask_rgba[mask_pixels, 2] = color[2]  # B - SOLID color
        instance_mask_rgba[mask_pixels, 3] = FILL_ALPHA  # A - uniform transparency
        
        # Step 2: Manually find and draw borders
        # Convert binary mask to uint8 for contour detection
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        
        # Find contours (edges) of the mask
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw borders by creating a border mask
        border_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(
            border_mask, 
            contours, 
            -1,  # Draw all contours
            255,  # White color
            BORDER_THICKNESS
        )
        
        # Apply solid border color (overwrite semi-transparent fill at borders)
        border_pixels = border_mask == 255
        instance_mask_rgba[border_pixels, 0] = color[0]  # R
        instance_mask_rgba[border_pixels, 1] = color[1]  # G
        instance_mask_rgba[border_pixels, 2] = color[2]  # B
        instance_mask_rgba[border_pixels, 3] = BORDER_ALPHA  # A (fully opaque)
        
        print(f"  Instance {idx + 1}/{num_instances}: Color RGB{tuple(color)}, Pixels: {num_mask_pixels}")
    
    return instance_mask_rgba

def show_box(box, ax):
    """Draws a red bounding box onto the plot."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    # Add a red rectangle patch
    rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# --- MAIN PIPELINE ---

def main():
    # Setup paths so SAM2 finds its configs
    os.environ["PYTHONPATH"] = os.path.abspath("../training/sam2")
    
    # 1. Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return
    # Read image in BGR (OpenCV default) then convert to RGB (Matplotlib wants this)
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # STAGE 1: THE HUNTER (YOLO)
    # ==========================================
    print("--- Stage 1: Running YOLO Detector ---")
    yolo = YOLO(YOLO_PATH)
    results = yolo(image_rgb, conf=YOLO_CONF, verbose=False)
    
    valid_boxes = []
    for r in results:
        b = r.boxes.xyxy.cpu().numpy()
        if len(b) > 0: valid_boxes.append(b)
    
    if not valid_boxes:
        print("YOLO found no particles. Stopping.")
        return
    
    valid_boxes = np.vstack(valid_boxes)
    print(f"YOLO found {len(valid_boxes)} candidate boxes.")

    # --- VISUALIZE STAGE 1 ---
    plt.figure(figsize=(12, 12))
    plt.imshow(image_rgb) # Display original image
    ax = plt.gca()
    for box in valid_boxes:
        show_box(box, ax)
    plt.title(f"Stage 1 Output: YOLO detected {len(valid_boxes)} boxes (Close window to continue)")
    plt.axis('off')
    print("Waiting for user to close Stage 1 window...")
    plt.show() # This pauses the script until you close the window

    # ==========================================
    # STAGE 2: THE ARTIST (SAM2)
    # ==========================================
    print("\n--- Stage 2: Running SAM2 Instance Segmentation ---")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_rgb)
    
    # Use YOLO boxes as prompts for SAM2
    masks_logits, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=valid_boxes,
        multimask_output=False,
    )

    # --- CREATE BINARY INSTANCE MASKS ---
    print("\n--- Creating Binary Instance Masks ---")
    print(f"SAM2 returned {len(masks_logits)} mask outputs")
    print(f"masks_logits type: {type(masks_logits)}")
    if len(masks_logits) > 0:
        print(f"First mask type: {type(masks_logits[0])}")
        print(f"First mask shape: {masks_logits[0].shape if hasattr(masks_logits[0], 'shape') else 'N/A'}")
    
    # Step 1: Convert all SAM2 outputs to clean binary masks
    valid_masks = []
    for i, mask_output in enumerate(masks_logits):
        print(f"\n--- Processing mask {i+1}/{len(masks_logits)} ---")
        # Manually threshold SAM2 output into binary mask
        binary_mask = get_binary_mask(mask_output, image_rgb.shape)
        
        # Only keep non-empty masks
        if binary_mask.sum() > 0:
            valid_masks.append(binary_mask)
            print(f"  ✓ Mask {i+1} is valid (added to valid_masks)")
        else:
            print(f"  ✗ Mask {i+1} is empty (skipped)")
    
    count = len(valid_masks)
    print(f"\n✅ SAM2 segmented {count} individual instances (out of {len(masks_logits)} outputs)")
    
    if count == 0:
        print("No valid masks found. Exiting.")
        return
    
    # Step 2: Manually create colored instance masks with borders
    instance_mask_rgba = create_instance_masks_manual(valid_masks, image_rgb.shape)
    
    # Step 3: Create overlay - ORIGINAL IMAGE + SOLID COLORED MASKS
    print("\nCreating overlay: original image + solid colored masks...")
    print(f"  Original image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")
    print(f"  Instance masks shape: {instance_mask_rgba.shape}, dtype: {instance_mask_rgba.dtype}")
    
    # Verify the RGBA mask only has solid colors (no probability gradients)
    non_zero_alpha = instance_mask_rgba[:, :, 3] > 0
    num_colored_pixels = np.sum(non_zero_alpha)
    unique_alphas = np.unique(instance_mask_rgba[:, :, 3])
    print(f"  Colored pixels: {num_colored_pixels}")
    print(f"  Unique alpha values: {unique_alphas} (should only be 0, {FILL_ALPHA}, {BORDER_ALPHA})")
    
    # Start with original image
    overlay = image_rgb.copy().astype(np.float32)
    
    # Extract alpha channel and normalize to [0, 1]
    alpha = instance_mask_rgba[:, :, 3].astype(np.float32) / 255.0
    
    # Manual alpha blending: result = (1 - alpha) * original_image + alpha * solid_mask_color
    # This creates: BACKGROUND IMAGE + COLORED MASKS (no logits, no probabilities)
    for channel in range(3):
        overlay[:, :, channel] = (
            (1.0 - alpha) * overlay[:, :, channel] + 
            alpha * instance_mask_rgba[:, :, channel].astype(np.float32)
        )
    
    # Convert back to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    print(f"  Overlay created: shape {overlay.shape}, dtype {overlay.dtype}")
    
    # Step 4: Create white background version - SHOWS ONLY SOLID COLORED MASKS
    print("\nCreating white background version (masks only, no image data)...")
    white_bg = np.ones((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8) * 255
    mask_with_white_bg = white_bg.copy().astype(np.float32)
    
    # Manually composite SOLID COLORED MASKS onto white background
    alpha_bg = instance_mask_rgba[:, :, 3].astype(np.float32) / 255.0
    for c in range(3):
        mask_with_white_bg[:, :, c] = (
            (1.0 - alpha_bg) * white_bg[:, :, c] + 
            alpha_bg * instance_mask_rgba[:, :, c].astype(np.float32)
        )
    
    mask_with_white_bg = np.clip(mask_with_white_bg, 0, 255).astype(np.uint8)
    print(f"  White background version created: shape {mask_with_white_bg.shape}")
    print(f"  This shows ONLY solid colored masks (no logits, no probabilities)")
    
    # Step 5: Save outputs
    output_dir = os.path.dirname(IMAGE_PATH) if os.path.dirname(IMAGE_PATH) else "."
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    
    instance_mask_rgba_path = os.path.join(output_dir, f"{base_name}_instance_masks_rgba.png")
    instance_mask_white_path = os.path.join(output_dir, f"{base_name}_instance_masks_white_bg.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    
    # Save instance masks as PNG (preserves alpha channel)
    cv2.imwrite(instance_mask_rgba_path, cv2.cvtColor(instance_mask_rgba, cv2.COLOR_RGBA2BGRA))
    # Save masks on white background
    cv2.imwrite(instance_mask_white_path, cv2.cvtColor(mask_with_white_bg, cv2.COLOR_RGB2BGR))
    # Save overlay as PNG
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"\n💾 Saved instance masks (RGBA with transparency) to: {instance_mask_rgba_path}")
    print(f"💾 Saved instance masks (white background) to: {instance_mask_white_path}")
    print(f"💾 Saved overlay to: {overlay_path}")
    
    # --- VISUALIZE STAGE 2 ---
    print("\n" + "="*60)
    print("FINAL OUTPUT VERIFICATION")
    print("="*60)
    print(f"✓ Binary masks created: {count} instances")
    print(f"✓ Non-transparent pixels: {np.sum(instance_mask_rgba[:,:,3] > 0)}")
    print(f"✓ Overlay contains: ORIGINAL IMAGE + SOLID COLORED MASKS")
    print(f"✓ NO logits, NO probability maps, ONLY binary masks")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original image (no masks)
    axes[0].imshow(image_rgb)
    axes[0].set_title("1. Original Image\n(No masks)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: ONLY solid colored masks (no original image)
    axes[1].imshow(mask_with_white_bg)
    axes[1].set_title(f"2. Binary Masks ONLY ({count} instances)\n(Solid colors + borders, no logits)", 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Final overlay = Original + Solid masks
    axes[2].imshow(overlay)
    axes[2].set_title("3. Final Overlay\n(Original image + solid colored masks)", 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    print("\n👁️  Displaying results. Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    main()