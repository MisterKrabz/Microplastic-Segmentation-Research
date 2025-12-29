import os
import sys
from ultralytics import YOLO

# --- CONFIGURATION ---
IMAGE_PATH = "/Users/patrickwang/Workspace/URS_Project/datasets/dataset/LMMP/1.5umPS/0.1umPCTE/1.5umPS_0.1umPCTE+30sAu_X100_darkfield_1.jpg"
# ---------------------

def main():
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "my_model.pt")

    # Validate files exist
    if not os.path.exists(model_path):
        sys.exit(f"Error: Model file not found at {model_path}")
    
    if not os.path.exists(IMAGE_PATH):
        sys.exit(f"Error: Image file not found at {IMAGE_PATH}")

    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        sys.exit(f"Error loading model: {e}")

    # Run inference
    print(f"Running inference on: {os.path.basename(IMAGE_PATH)}")
    results = model.predict(IMAGE_PATH, save=False, conf=0.03)

    # Display result
    results[0].show()

if __name__ == "__main__":
    main()