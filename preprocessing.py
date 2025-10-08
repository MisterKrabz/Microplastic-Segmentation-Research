import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def basic_preprocess(image, blur_strength=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_strength, blur_strength), 0)
    return blurred

def adaptive_preprocess(image, min_ksize=3, max_ksize=11):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_metric = np.clip(noise_metric / 300.0, 0, 1)
    ksize = int(min_ksize + noise_metric * (max_ksize - min_ksize))
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return blurred, ksize

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from '{image_path}'")
        sys.exit(1)

    processed, used_ksize = adaptive_preprocess(img)
    print(f"Chosen Gaussian kernel size: {used_ksize}")

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
