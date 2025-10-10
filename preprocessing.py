import cv2 
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

def basic_preprocess(image, blur_strength=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_strength, blur_strength), 0)
    return blurred

def adaptive_preprocess(image, min_ksize=3, max_ksize=11):
    """
    Returns (processed_image, kernel_size)
    kernel size chosen proportional to Laplacian variance (noise estimate)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
    # scale noise metric into [0,1] (tweak divisor if needed)
    noise_metric = np.clip(noise_metric / 300.0, 0, 1)
    ksize = int(min_ksize + noise_metric * (max_ksize - min_ksize))
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return blurred, ksize

def save_image(image, out_path):
    # OpenCV imwrite expects BGR for color; our image is grayscale here.
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    success = cv2.imwrite(out_path, image)
    if not success:
        raise IOError(f"Failed to write image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image and save result.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--out-dir", default=None, help="Directory to save processed image (default: 'processed_images' in script's directory)")
    parser.add_argument("--min-ksize", type=int, default=3, help="Minimum Gaussian kernel size (odd)")
    parser.add_argument("--max-ksize", type=int, default=11, help="Maximum Gaussian kernel size (odd)")
    parser.add_argument("--show", action="store_true", help="Show before/after with matplotlib")
    args = parser.parse_args()

    image_path = args.image_path
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from '{image_path}'")
        sys.exit(1)

    processed, used_ksize = adaptive_preprocess(img, min_ksize=args.min_ksize, max_ksize=args.max_ksize)

    # determine output path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # If --out-dir is not specified, save to a 'processed_images' subdirectory
    # in the same directory as this script.
    if args.out_dir:
        out_dir = args.out_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "processed_images")
    out_name = f"{base_name}_processed_ksize{used_ksize}.png"
    out_path = os.path.join(out_dir, out_name)

    save_image(processed, out_path)
    print(f"Chosen Gaussian kernel size: {used_ksize}")
    print(f"Processed image saved to: {out_path}")

    if args.show:
        # Convert BGR to RGB for matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Processed (ksize={used_ksize})")
        plt.imshow(processed, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()