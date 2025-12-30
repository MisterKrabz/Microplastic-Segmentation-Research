import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# TESTING CONFIGURATION 
# ==========================================
IMAGE_PATH = "./../datasets/raw_data/LMMP/1.5umPS/1umPCTE/1_5umPS_1umPCTE_X100_darkField_1.jpg"
MODEL_PATH = "./../models/hunter-yolo-v0.2.0.pt"

# Tuning Parameters
CONFIDENCE = 0.6   # Lower = Detect more (risk of noise). Higher = Strict.
IOU_THRESH = 0.45   # Intersection Over Union (removes duplicate boxes)
IMG_SIZE   = 640    # Inference size (640 is standard)
# ==========================================

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        return

    print(f"🔍 Running YOLO Only...")
    print(f"⚙️  Params: Conf={CONFIDENCE} | IoU={IOU_THRESH} | Size={IMG_SIZE}")

    # 1. Load Model
    model = YOLO(MODEL_PATH)

    # 2. Run Inference
    results = model.predict(
        source=IMAGE_PATH,
        conf=CONFIDENCE,
        iou=IOU_THRESH,
        imgsz=IMG_SIZE,
        save=False,
        verbose=False
    )

    # 3. Visualize
    # Ultralytics plotter returns BGR, convert to RGB for Matplotlib
    res_plot = results[0].plot(line_width=2, font_size=1)
    res_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(res_rgb)
    plt.title(f"YOLO Result | Conf: {CONFIDENCE} | Count: {len(results[0].boxes)}")
    plt.axis('off')
    plt.tight_layout()
    print("Displaying result... (Close window to exit)")
    plt.show()

if __name__ == "__main__":
    main()