import os
import matplotlib.pyplot as plt
from mira import MicroplasticDetector

# ==========================================
# 🎛️  USER CONFIGURATION (EDIT THESE)
# ==========================================
IMAGE_PATH = "datasets/dataset/test/images/sample.jpg"
YOLO_PATH  = "models/yolo_hunter.pt"
SAM_PATH   = "models/sam2_hiera_l.pt"

# Tuning Parameters
YOLO_CONF  = 0.25   # The most critical knob. Controls what gets sent to SAM.
# ==========================================

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        return

    print(f"🧪 Testing Full MIRA Pipeline")
    print(f"⚙️  YOLO Confidence: {YOLO_CONF}")

    # 1. Initialize Framework
    # We explicitly pass paths so you can test different model versions easily
    detector = MicroplasticDetector(yolo_path=YOLO_PATH, sam2_checkpoint=SAM_PATH)

    # 2. Run Prediction
    # We pass the tuning parameter 'conf' directly to the predict method
    original_img, masks = detector.predict(IMAGE_PATH, conf=YOLO_CONF)

    count = len(masks)
    print(f"✅ Result: Found {count} particles")

    # 3. Visualization
    plt.figure(figsize=(14, 7))

    # Left: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    # Right: Segmentation Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    
    # Overlay masks with distinct colors
    if count > 0:
        for mask in masks:
            plt.imshow(mask, alpha=0.5, cmap='spring')
    else:
        print("⚠️  No masks generated.")

    plt.title(f"MIRA Output (Conf: {YOLO_CONF}) | Count: {count}")
    plt.axis('off')
    
    print("👀 Displaying result... (Close window to exit)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()