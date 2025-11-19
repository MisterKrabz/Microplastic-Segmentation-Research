"""
SAM 2 Image Segmentation Script
Segments images using Meta's Segment Anything Model 2 (SAM 2)
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Text prompt for segmentation (NOTE: SAM 2 doesn't use text prompts directly,
# but you can describe what you're looking for here as a reference)
TEXT_PROMPT = "microplastic particles"

# Path to SAM 2 checkpoint and config
# Adjust these paths based on where you installed SAM 2
SAM2_CHECKPOINT = "../sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

# Segmentation mode: "automatic" or "prompt"
SEGMENTATION_MODE = "automatic"

# Point prompts (only used if SEGMENTATION_MODE = "prompt")
# Format: [[x1, y1], [x2, y2], ...]
POINT_PROMPTS = [
    [100, 100],
    [200, 150],
]
POINT_LABELS = [1, 1]  # 1 = foreground, 0 = background

# Box prompt (only used if SEGMENTATION_MODE = "prompt")
# Format: [x_min, y_min, x_max, y_max]
BOX_PROMPT = None  # Example: [50, 50, 300, 300]

# Output settings
OUTPUT_FOLDER = "output_segmentations"
SAVE_RESULTS = True
SHOW_RESULTS = True

# Automatic segmentation parameters (adjust these to control detection)
AUTO_PARAMS = {
    "points_per_side": 256,          # Higher = more masks (try 16, 32, 64)
    "pred_iou_thresh": 0.7,         # Lower = more masks (try 0.7-0.95)
    "stability_score_thresh": 0.8,  # Lower = more masks (try 0.8-0.98)
    "min_mask_region_area": 100,     # Lower = smaller objects detected
    "crop_n_layers": 0,              # Higher = better for small objects
    "crop_n_points_downscale_factor": 1,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_paths(input_path):
    """Get list of image paths from file or folder"""
    path = Path(input_path)
    
    if path.is_file():
        return [path]
    elif path.is_dir():
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        images = []
        for ext in extensions:
            images.extend(path.glob(f'*{ext}'))
            images.extend(path.glob(f'*{ext.upper()}'))
        return sorted(images)
    else:
        raise ValueError(f"Path not found: {input_path}")

def load_sam2_model(checkpoint, config):
    """Load SAM 2 model"""
    print(f"Loading SAM 2 model...")
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {config}")
    
    # Check for available devices (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    sam2_model = build_sam2(config, checkpoint, device=device)
    return sam2_model, device

def show_mask(mask, ax, color=None, alpha=0.5):
    """Display a single mask"""
    if color is None:
        color = np.array([30/255, 144/255, 255/255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=alpha)

def show_points(coords, labels, ax, marker_size=200):
    """Display point prompts"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', 
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', 
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Display box prompt"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                               facecolor=(0,0,0,0), lw=2))

def segment_with_prompts(predictor, image, point_prompts=None, 
                        point_labels=None, box_prompt=None):
    """Segment image using point/box prompts"""
    predictor.set_image(image)
    
    # Prepare inputs
    points = np.array(point_prompts) if point_prompts else None
    labels = np.array(point_labels) if point_labels else None
    box = np.array(box_prompt) if box_prompt else None
    
    # Generate masks
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=True,
    )
    
    return masks, scores

def segment_automatic(mask_generator, image):
    """Automatically segment entire image"""
    masks = mask_generator.generate(image)
    return masks

def visualize_prompt_results(image, masks, scores, point_prompts=None,
                             point_labels=None, box_prompt=None):
    """Visualize segmentation results with prompts"""
    n_masks = len(masks)
    fig, axes = plt.subplots(1, n_masks + 1, figsize=(15, 5))
    
    if n_masks == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nPrompt: {TEXT_PROMPT}")
    axes[0].axis('off')
    
    # Each mask
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i+1].imshow(image)
        show_mask(mask, axes[i+1])
        
        if point_prompts is not None:
            show_points(np.array(point_prompts), np.array(point_labels), axes[i+1])
        if box_prompt is not None:
            show_box(box_prompt, axes[i+1])
        
        axes[i+1].set_title(f"Mask {i+1} (Score: {score:.3f})")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_automatic_results(image, masks):
    """Visualize automatic segmentation results with overlay on the original image."""
    if len(masks) == 0:
        print("No masks generated!")
        return None
    
    # Sort masks by area (largest first)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Create an overlay to combine masks with transparency
    overlay = image.copy().astype(np.float32)

    for mask_data in sorted_masks:
        mask = mask_data['segmentation']
        color = np.random.random(3)  # random RGB color for each mask
        color = np.array(color) * 255
        # Apply mask color with transparency
        overlay[mask] = overlay[mask] * 0.4 + color * 0.6

    # Convert overlay to uint8 for display
    overlay = overlay.astype(np.uint8)
    #
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nSearching for: {TEXT_PROMPT}")
    axes[0].axis('off')
    
    # Overlayed result
    axes[1].imshow(overlay)
    axes[1].set_title(f"Segmented Overlay ({len(masks)} masks)")
    axes[1].axis('off')

    plt.tight_layout()
    return fig


def save_results(image_path, fig, masks, mode, output_folder):
    """Save segmentation results"""
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Save visualization
    img_name = Path(image_path).stem
    fig_path = output_dir / f"{img_name}_segmented.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {fig_path}")
    
    # Save individual masks
    masks_dir = output_dir / f"{img_name}_masks"
    masks_dir.mkdir(exist_ok=True)
    
    if mode == "prompt":
        for i, mask in enumerate(masks):
            mask_path = masks_dir / f"mask_{i}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    else:  # automatic
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            mask_path = masks_dir / f"mask_{i}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    
    print(f"  Saved {len(masks)} masks: {masks_dir}")

def process_image(image_path, sam2_model, device, mode):
    """Process a single image"""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"Looking for: {TEXT_PROMPT}")
    
    # Segment based on mode
    if mode == "prompt":
        print(f"Mode: Prompt-based segmentation")
        predictor = SAM2ImagePredictor(sam2_model)
        
        # Use appropriate autocast based on device
        if device == "cuda":
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                masks, scores = segment_with_prompts(
                    predictor, image, POINT_PROMPTS, POINT_LABELS, BOX_PROMPT
                )
        else:
            with torch.inference_mode():
                masks, scores = segment_with_prompts(
                    predictor, image, POINT_PROMPTS, POINT_LABELS, BOX_PROMPT
                )
        
        print(f"Generated {len(masks)} masks")
        for i, score in enumerate(scores):
            print(f"  Mask {i+1}: score = {score:.3f}")
        
        fig = visualize_prompt_results(
            image, masks, scores, POINT_PROMPTS, POINT_LABELS, BOX_PROMPT
        )
    else:  # automatic
        print(f"Mode: Automatic segmentation")
        mask_generator = SAM2AutomaticMaskGenerator(sam2_model, **AUTO_PARAMS)
        
        # Use appropriate autocast based on device
        if device == "cuda":
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                masks = segment_automatic(mask_generator, image)
        else:
            with torch.inference_mode():
                masks = segment_automatic(mask_generator, image)
        
        print(f"Generated {len(masks)} masks")
        fig = visualize_automatic_results(image, masks)
    
    # Save and/or show results
    if SAVE_RESULTS and fig is not None:
        save_results(image_path, fig, masks, mode, OUTPUT_FOLDER)
    
    if SHOW_RESULTS and fig is not None:
        plt.show()
    else:
        plt.close(fig)

def main():
    """Main execution function"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python segment_sam2.py <image_path_or_folder>")
        print("\nExamples:")
        print("  python segment_sam2.py image.jpg")
        print("  python segment_sam2.py images/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print("=" * 70)
    print("SAM 2 IMAGE SEGMENTATION")
    print("=" * 70)
    print(f"Prompt: '{TEXT_PROMPT}'")
    print(f"Mode: {SEGMENTATION_MODE}")
    print(f"Input: {input_path}")
    print(f"Output: {OUTPUT_FOLDER}/")
    print("=" * 70)
    
    # Load SAM 2 model
    try:
        sam2_model, device = load_sam2_model(SAM2_CHECKPOINT, SAM2_CONFIG)
    except Exception as e:
        print(f"\nError loading SAM 2 model: {e}")
        print("\nMake sure:")
        print("1. SAM 2 is installed: pip install -e .")
        print("2. Checkpoint path is correct")
        print("3. Config path is correct")
        sys.exit(1)
    
    # Get image paths
    try:
        image_paths = get_image_paths(input_path)
        print(f"\nFound {len(image_paths)} image(s) to process")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Process each image
    for img_path in image_paths:
        try:
            process_image(img_path, sam2_model, device, SEGMENTATION_MODE)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    if SAVE_RESULTS:
        print(f"Results saved to: {OUTPUT_FOLDER}/")

if __name__ == "__main__":
    main()
