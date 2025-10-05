#
# image_preprocessor_debugger_v2.py
#
# Author: Gemini (Professional Coder)
# Date:   October 5, 2025
#
# Description:
#   A debugging tool using a more robust pipeline for low-contrast images.
#   It uses CLAHE to enhance contrast and Otsu's method for automatic thresholding.
#   It saves an intermediate image after each major operation.
#

import cv2
import numpy as np
import sys
import os

# --- PREPROCESSING CONFIGURATION ---
# These parameters are for the new CLAHE + Otsu pipeline.
CONFIG = {
    # CLAHE (Contrast Enhancement) Parameters
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),

    # Gaussian blur to apply AFTER contrast enhancement.
    "gaussian_blur_kernel": (5, 5),

    # Morphological opening to clean up the final mask.
    "morph_open_kernel": (3, 3),
}

def create_processing_steps(image_path):
    """
    Applies a robust preprocessing pipeline and returns a dictionary of all intermediate images.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        dict: A dictionary where keys are step names and values are the image data.
    """
    processing_steps = {}

    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None
    processing_steps['00_original'] = img

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processing_steps['01_grayscale'] = gray

    # 3. Enhance Contrast with CLAHE (NEW STEP)
    # This makes the dark particles "pop" from the bright background.
    clahe = cv2.createCLAHE(
        clipLimit=CONFIG["clahe_clip_limit"],
        tileGridSize=CONFIG["clahe_tile_grid_size"]
    )
    clahe_enhanced = clahe.apply(gray)
    processing_steps['02_clahe_enhanced'] = clahe_enhanced

    # 4. Apply Gaussian Blur
    # We blur AFTER enhancing contrast to smooth the image for thresholding.
    blurred = cv2.GaussianBlur(clahe_enhanced, CONFIG["gaussian_blur_kernel"], 0)
    processing_steps['03_blurred'] = blurred

    # 5. Apply Otsu's Thresholding (NEW METHOD)
    # Otsu's method automatically finds the best global threshold value.
    # We use THRESH_BINARY_INV because our objects (particles) are dark.
    _, otsu_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    processing_steps['04_otsu_mask_raw'] = otsu_mask

    # 6. Refine the Mask with Morphological Opening
    kernel = np.ones(CONFIG["morph_open_kernel"], np.uint8)
    clean_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    processing_steps['05_clean_mask_final'] = clean_mask

    return processing_steps

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python image_preprocessor_debugger_v2.py <path_to_image_file>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_dir = "processed_images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Generating debug images for {input_image_path}...")
    steps = create_processing_steps(input_image_path)

    if steps:
        base_filename = os.path.splitext(os.path.basename(input_image_path))[0]
        for step_name, image_data in steps.items():
            output_filename = f"{base_filename}_{step_name}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, image_data)
            print(f"  -> Saved step: {output_path}")
        print("\nDebug image generation complete.")