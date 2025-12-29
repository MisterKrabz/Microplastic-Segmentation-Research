import matplotlib.pyplot as plt
# Look how clean this import is now!
from microplastics import MicroplasticDetector

# 1. Define Paths to YOUR local assets
# (These stay in your root folder, they are not part of the pip install)
YOLO_PATH = "models/yolo_hunter.pt"
SAM2_PATH = "models/sam2_hiera_l.pt"
IMAGE_PATH = "datasets/dataset/test/images/sample.jpg"

# 2. Initialize Framework
detector = MicroplasticDetector(YOLO_PATH, SAM2_PATH)

# 3. Run
print(f"Running on {IMAGE_PATH}...")
original_image, masks = detector.predict(IMAGE_PATH)

# 4. Visualize
print(f"Found {len(masks)} particles.")
plt.imshow(original_image)
for mask in masks:
    # Overlay masks with transparency
    plt.imshow(mask, alpha=0.5, cmap='spring') 
plt.show()