#
# microplastic_detector_v2.py
#
# Author: Gemini (Professional Coder)
# Date:   October 5, 2025
#
# Description:
#   Processes a microscope image to identify potential microplastic particles.
#   It converts the image to a high-contrast black and white mask, then draws
#   contours around detected particles based on size and circularity.
#   The output is a black and white image with green contours, saved to a
#   'processed_images' directory.
#

import cv2
import numpy as np
import sys
import os

# --- CONFIGURATION PARAMETERS ---
# Adjust these values to fine-tune the detection for your specific microscopy setup.
CONFIG = {
    # Gaussian blur kernel size. Must be odd numbers. (e.g., (5, 5))
    # Helps reduce noise.
    "gaussian_blur_kernel": (5, 5),

    # Adaptive thresholding parameters.
    # 'block_size': Size of the pixel neighborhood used to calculate the threshold. Must be odd.
    # 'C': A constant subtracted from the mean. Can be fine-tuned.
    "adaptive_thresh_block_size": 15,
    "adaptive_thresh_C": 2,

    # Morphological opening kernel size. Used to remove small noise specks.
    "morph_open_kernel": (3, 3),

    # Contour filtering parameters.
    # These are the most important values to tune.
    "min_contour_area": 20,         # Minimum pixel area to be considered a particle.
    "max_contour_area": 1500,       # Maximum pixel area.
    "max_aspect_ratio": 1.5,        # Maximum aspect ratio (width/height). 1.0 is a perfect circle.
}

def process_microplastic_image(image_path):
    """
    Processes a microscope image to highlight microplastics for AI training.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        A tuple containing:
        - numpy.ndarray: The black and white mask with contours drawn.
        - int: The number of particles detected.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None, 0

    # 2. Preprocessing
    # Convert to grayscale for color-independent processing.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise.
    blurred = cv2.GaussianBlur(gray, CONFIG["gaussian_blur_kernel"], 0)

    # 3. Segmentation
    # Use adaptive thresholding to create a binary (black and white) image.
    # This is robust to lighting changes. THRESH_BINARY makes bright spots white.
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        CONFIG["adaptive_thresh_block_size"], CONFIG["adaptive_thresh_C"]
    )

    # 4. Mask Refinement
    # Use a morphological opening to remove small noise specks from the binary mask.
    kernel = np.ones(CONFIG["morph_open_kernel"], np.uint8)
    opened_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Contour Detection and Filtering
    # Find all contours in the cleaned mask.
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a 3-channel BGR image from the black and white mask
    # so we can draw colored contours on it.
    output_image = cv2.cvtColor(opened_mask, cv2.COLOR_GRAY2BGR)

    detected_particles_count = 0
    for cnt in contours:
        # Filter by AREA
        area = cv2.contourArea(cnt)
        if CONFIG["min_contour_area"] < area < CONFIG["max_contour_area"]:
            # Filter by CIRCULARITY (using aspect ratio of a fitted ellipse)
            if len(cnt) >= 5:  # An ellipse needs at least 5 points to be fitted
                ellipse = cv2.fitEllipse(cnt)
                (xc, yc), (d1, d2), angle = ellipse
                aspect_ratio = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0

                # Check if the aspect ratio is within our circularity tolerance
                if aspect_ratio < CONFIG["max_aspect_ratio"]:
                    # If all checks pass, draw the contour in green
                    cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
                    detected_particles_count += 1

    return output_image, detected_particles_count

if __name__ == '__main__':
    # Check for command-line argument
    if len(sys.argv) != 2:
        print("Usage: python microplastic_detector_v2.py <path_to_image_file>")
        sys.exit(1)

    input_image_path = sys.argv[1]

    # Define the output directory
    output_dir = "processed_images"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Process the image
    print(f"Processing {input_image_path}...")
    processed_image, count = process_microplastic_image(input_image_path)

    if processed_image is not None:
        # Construct the full output path
        base_filename = os.path.basename(input_image_path)
        output_filename = f"mask_{base_filename}"
        output_path = os.path.join(output_dir, output_filename)

        # Save the resulting image
        cv2.imwrite(output_path, processed_image)
        print(f"  -> Successfully detected {count} particles.")
        print(f"  -> Saved processed mask to: {output_path}")